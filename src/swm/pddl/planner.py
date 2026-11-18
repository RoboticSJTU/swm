from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from swm.pddl.reorder import plan_reorder, evaluate, parse_sexpr

fast_downward_path = Path(__file__).parents[3] / "downward" / "fast-downward.py"
_FAST_DOWNWARD_TIME_LIMIT_SECONDS = 60
_SOLVER_WALL_TIMEOUT_SECONDS = 60


def summarize_solver_error(log: str) -> str:
    """Compress a Fast Downward failure log into one short message."""
    if "TimeoutExpired" in log:
        timeout = re.search(r"timed out after ([\d.]+) seconds", log)
        return f"Fast Downward timed out{f' after {timeout.group(1)} seconds' if timeout else ''}."

    exits = re.findall(r"(translate|search) exit code:\s*(-?\d+)", log, re.IGNORECASE)
    stage, code = exits[-1] if exits else ("planner", "unknown")
    prefix = f"Fast Downward {stage.lower()} failed (exit {code})"
    action = re.findall(r"Parsing action '([^']+)'", log)
    location = f" in action '{action[-1]}'" if action else ""
    match = re.search(
        r"(?:Undefined|Undeclared)\s+(predicate|object|variable)\s*\nGot:\s*([^\n]+)",
        log,
        re.IGNORECASE,
    )

    if match:
        detail = (
            f"undefined {match.group(1).lower()} '{match.group(2).strip()}'{location}"
        )
    elif match := re.search(
        r"Predicate '([^']+)' of arity (\d+) used\s+with (\d+) arguments",
        log,
        re.IGNORECASE,
    ):
        detail = f"predicate '{match.group(1)}' expects {match.group(2)} arguments but got {match.group(3)}{location}"
    elif re.search(r"Expected .*words:\s*:domain", log, re.IGNORECASE):
        detail = "the problem is missing the required (:domain NAME) block"
    elif match := re.search(r"(?:^|\n)([^\n]+)\nSyntax:", log):
        detail = match.group(1).strip()
    elif "No relaxed solution" in log or re.search(r"no solution", log, re.IGNORECASE):
        operators = re.search(r"Translator operators:\s*(\d+)", log)
        detail = "no plan was found" + (
            " and the translator produced no grounded operators"
            if operators and operators.group(1) == "0"
            else ""
        )
    elif code == "30":
        detail = "the translator crashed while compiling the PDDL task"
    elif code == "31":
        lines = re.split(r"translate exit code", log, flags=re.IGNORECASE)[
            0
        ].splitlines()
        detail = next(
            (
                line.strip()
                for line in reversed(lines)
                if line.strip() and not re.match(r"(?:INFO|Parsing|->)", line.strip())
            ),
            "invalid PDDL input",
        )
    elif code == "34":
        detail = "the requested search configuration is unsupported"
    else:
        diagnostics = re.findall(
            r"(?im)^\s*((?:\w*(?:Error|Exception):|Reason:|Critical error:|Expected |Missing |Unsupported |Invalid ).+)$",
            log,
        )
        detail = (
            diagnostics[-1].strip()
            if diagnostics
            else "the planner failed without a diagnostic message"
        )

    detail = re.sub(r"\s+", " ", detail).strip().rstrip(".")[:240]
    return f"{prefix}: {detail}."


def _output_text(value):
    """将 subprocess 的文本或字节输出统一为可写入日志的字符串。"""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _kill_process_tree(process: subprocess.Popen) -> None:
    """Terminate the solver driver and every child in its process group."""

    try:
        if os.name == "posix":
            # The group can outlive its leader, so do not return just because
            # the driver process itself has already exited.
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:  # pragma: no cover - Linux is the target
            process.kill()
    except ProcessLookupError:
        pass


def _run_fast_downward(cmd: list[object], cwd: Path, *, timeout: float | None = None) -> tuple[str, str]:
    """Run Fast Downward under both an internal and process-tree wall guard."""

    timeout = _SOLVER_WALL_TIMEOUT_SECONDS if timeout is None else timeout
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd,
            timeout,
            output=stdout,
            stderr=stderr,
        )
    except BaseException:
        _kill_process_tree(process)
        process.communicate()
        raise

    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            cmd,
            output=stdout,
            stderr=stderr,
        )
    return stdout, stderr


def solve_pddl(domain_file, problem_file, *, reorder_plan=True, capture: dict | None = None):
    """用 A*(LM-cut) 求最小代价计划；不支持的模型或求解失败返回 False。"""

    domain_file = Path(domain_file).resolve()
    problem_file = Path(problem_file).resolve()
    plan_file = Path(domain_file).parent / "plan.txt"
    error_file = Path(domain_file).parent / "error.log"

    cmd = [
        sys.executable,
        fast_downward_path,
        "--overall-time-limit",
        f"{_FAST_DOWNWARD_TIME_LIMIT_SECONDS}s",
        "--plan-file",
        plan_file,
        domain_file,
        problem_file,
        "--search",
        "astar(lmcut())",
    ]
    plan_file.unlink(missing_ok=True)
    error_file.unlink(missing_ok=True)
    try:
        stdout, stderr = _run_fast_downward(cmd, domain_file.parent)
        if reorder_plan:
            plan_reorder(domain_file, problem_file, plan_file, plan_file)
        if capture is not None:
            capture.update(
                {
                    "command": [str(part) for part in cmd],
                    "exit_status": 0,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
        return True

    except Exception as e:
        plan_file.unlink(missing_ok=True)
        stdout = _output_text(getattr(e, "stdout", None))
        stderr = _output_text(getattr(e, "stderr", None))
        if capture is not None:
            capture.update(
                {
                    "command": [str(part) for part in cmd],
                    "exit_status": getattr(e, "returncode", None),
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
        details = [f"{type(e).__name__}: {e}"]
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        error_file.write_text("\n\n".join(details), encoding="utf-8")
        return False


# PSV 等长搜索：原生状态图及原始事实的布尔SAS后备编码。
ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / '.build' / 'fd_psv'
# The generation pipeline has 20 workers. Native abstract tables are admitted
# separately so their measured multi-GiB peaks cannot multiply by that count.
_GRAPH_JOBS = threading.BoundedSemaphore(1)
DEFAULT_PDB_STATES = 1024000000


class FDGraphUnsupported(ValueError):
    """The adapter cannot prove this FD translation preserves original paths."""


@dataclass
class FDGraph:
    states: list[int]
    edges: list[list[tuple[int, int]]]
    layers: list[list[int]]
    suffix: list[int]
    proof: dict


def _strips_expression(node):
    if node == []:
        return True
    if not isinstance(node, list) or not node:
        return False
    if node[0] == 'and':
        return all(_strips_expression(child) for child in node[1:])
    if node[0] == 'not':
        return len(node) == 2 and isinstance(node[1], list) and all(isinstance(x, str) for x in node[1])
    if node[0] in {'or', 'imply', 'forall', 'exists', 'when'}:
        return False
    return all(isinstance(part, str) for part in node)


def _check_fragment(model):
    domain = parse_sexpr(model.domain_path.read_text())
    problem = parse_sexpr(model.problem_path.read_text())
    for section in domain[1:]:
        if section[:1] == [':action']:
            fields = dict(zip(section[2::2], section[3::2]))
            if not all(_strips_expression(fields[key]) for key in (':precondition', ':effect')):
                raise FDGraphUnsupported('non-STRIPS action expression')
        elif section[:1] == [':derived']:
            raise FDGraphUnsupported('derived predicates')
    for section in problem[1:]:
        if section[:1] == [':goal'] and not _strips_expression(section[1]):
            raise FDGraphUnsupported('non-STRIPS goal expression')


def _read_fd_graph(model, path, *, verify_edges=True):
    lines = iter(Path(path).read_text().splitlines())
    if next(lines) != 'psv_exact_length_v1':
        raise FDGraphUnsupported('unknown exact-length plugin artifact format')
    status, length = next(lines), int(next(lines))
    if status != 'complete' or length != model.length:
        raise FDGraphUnsupported('FD graph is not complete at the reference length: ' + status)
    variables = [[next(lines) for _ in range(int(next(lines)))] for _ in range(int(next(lines)))]
    names = [next(lines) for _ in range(int(next(lines)))]
    bindings = ['(' + ' '.join(name.strip('()').split()) + ')' for name in names]
    original = {action.line: i for i, action in enumerate(model.actions, 1)}
    if len(set(bindings)) != len(bindings) or set(bindings) != set(original):
        raise FDGraphUnsupported('FD changed the original reachable binding catalog')
    identifiers = [original[line] for line in bindings]
    written_add = written_delete = 0
    for action in model.actions:
        written_add |= action.possible_add
        written_delete |= action.possible_delete
    constants = model.initial & ~written_delete
    constant_false = ((1 << len(model.atoms)) - 1) & ~model.initial & ~written_add
    positive_names = {'Atom ' + fact[0] + '(' + ', '.join(fact[1:]) + ')': bit
                      for fact, bit in model.bits.items()}
    projection = [[positive_names.get(name, 0) for name in values] for values in variables]
    represented = 0
    for row in projection:
        for bit in row:
            represented |= bit
    if ((1 << len(model.atoms)) - 1) & ~(represented | constants | constant_false):
        raise FDGraphUnsupported('FD omitted an observable mutable original fact')
    states, edges, depths = [], [], []
    layers = [[] for _ in range(length + 1)]
    for index in range(int(next(lines))):
        depth, goal, count, *values = map(int, next(lines).split())
        if not 0 <= depth <= length or goal not in (0, 1) or count < 0 or len(values) != len(variables):
            raise FDGraphUnsupported('invalid projected FD state')
        state = constants
        for row, value in zip(projection, values):
            if not 0 <= value < len(row):
                raise FDGraphUnsupported('invalid FD variable value')
            state |= row[value]
        states.append(state)
        depths.append(depth)
        layers[depth].append(index)
        outgoing = []
        for _ in range(count):
            operator, child = map(int, next(lines).split())
            if not 0 <= operator < len(identifiers):
                raise FDGraphUnsupported('invalid FD operator identifier')
            outgoing.append((identifiers[operator], child))
        if len({identifier for identifier, _ in outgoing}) != len(outgoing):
            raise FDGraphUnsupported('duplicate original operator edge at one FD state')
        edges.append(outgoing)
        if bool(goal) != (depth == length and evaluate(model.goal, state)):
            raise FDGraphUnsupported('FD goal differs from original PDDL')
    raw_count = int(next(lines), 16)
    allocated, registered, generated, pruned = map(int, next(lines).split())
    if next(lines, None) is not None or not states or depths[0] != 0 or states[0] != model.initial:
        raise FDGraphUnsupported('FD graph did not retain the original initial state')
    suffix = [0] * len(states)
    for parent in layers[-1]:
        suffix[parent] = int(evaluate(model.goal, states[parent]))
    for depth in range(length - 1, -1, -1):
        for parent in layers[depth]:
            for identifier, child in edges[parent]:
                if not 0 <= child < len(states) or depths[child] != depth + 1:
                    raise FDGraphUnsupported('FD edge lost its exact depth')
                if verify_edges and model.apply(states[parent], identifier - 1) != states[child]:
                    raise FDGraphUnsupported('FD transition differs from original PDDL replay')
                suffix[parent] += suffix[child]
    if suffix[0] != raw_count:
        raise FDGraphUnsupported('FD path count disagrees with independent integer DP')
    if raw_count < 1:
        raise FDGraphUnsupported('FD lost the executable reference plan')
    reference_node = 0
    for identifier in model.reference_order:
        reference_node = next((child for action, child in edges[reference_node]
                               if action == identifier), -1)
        if reference_node < 0:
            raise FDGraphUnsupported('FD omitted an original bound reference occurrence')
    if depths[reference_node] != length or not evaluate(model.goal, states[reference_node]):
        raise FDGraphUnsupported('FD reference path does not end at the original goal')
    return FDGraph(states, edges, layers, suffix,
                   dict(coverage_complete=True, raw_orders=raw_count, raw_order_count_exact=True,
                        nodes=len(states), edges=sum(map(len, edges)), allocated_nodes=allocated,
                        registered_states=registered, generated=generated, hmax_pruned=pruned,
                        original_edges_verified=verify_edges, reference_path_verified=True))


def write_boolean_sas(model, stream):
    from translate import sas_tasks

    def mask(expression):
        if expression is True:
            return 0, 0
        if isinstance(expression, tuple) and expression[0] == 'mask':
            return expression[1:]
        raise FDGraphUnsupported('Boolean SAS requires conjunctive compiled guards/goals')

    goal_pos, goal_neg = mask(model.goal)
    values = []
    for fact in model.atoms:
        suffix = fact[0] + '(' + ', '.join(fact[1:]) + ')'
        values.append(['NegatedAtom ' + suffix, 'Atom ' + suffix])
    initial = [int(bool(model.initial & (1 << index))) for index in range(len(values))]
    goals = [(index, int(bool(goal_pos & (1 << index))))
             for index in range(len(values)) if (goal_pos | goal_neg) & (1 << index)]
    if not goals:
        # The legacy SAS input requires a nonempty goal. This immutable true
        # variable encodes a tautology and has no original atom projection.
        goals = [(len(values), 1)]
        values.append(['PSV constant false', 'PSV constant true'])
        initial.append(1)
    operators = []
    for action in model.actions:
        pos, neg = mask(action.guard)
        if any(guard is not True for guard, _, _ in action.effects):
            raise FDGraphUnsupported('Boolean SAS requires unconditional compiled effects')
        written = action.possible_add | action.possible_delete
        prevail, effects = [], []
        for index in range(len(model.atoms)):
            bit = 1 << index
            pre = 1 if pos & bit else 0 if neg & bit else -1
            if written & bit:
                # Original Boolean application gives adds priority over deletes.
                effects.append((index, pre, int(bool(action.possible_add & bit)), []))
            elif pre >= 0:
                prevail.append((index, pre))
        operators.append(sas_tasks.SASOperator(action.line, prevail, effects, 1))
    task = sas_tasks.SASTask(
        sas_tasks.SASVariables([2] * len(values), [-1] * len(values), values), [],
        sas_tasks.SASInit(initial), sas_tasks.SASGoal(goals), operators, [], False)
    task.variables.validate()
    task.init.validate(task.variables)
    task.goal.validate(task.variables)
    for operator in task.operators:
        # PSV's existing native parser extension permits genuine zero effects;
        # the legacy translator validator still asserts a nonempty effect list.
        if operator.pre_post:
            operator.validate(task.variables)
    task.output(stream)


def _build_boolean_graph(model, build, *, max_states=300000, timeout=120,
                         hmax=True, lmcut=True, pdb_states=4000000,
                         landmarks=True, pdb_portfolio=True, pdb_precheck=True):
    began = time.perf_counter()
    binary = Path(build).resolve() / 'bin/downward'
    if not binary.is_file():
        raise FDGraphUnsupported('Optional FD exact-length plugin has not been built')
    with TemporaryDirectory(prefix='swm-boolean-sas-') as temporary:
        directory = Path(temporary)
        sas_path, graph_path = directory / 'task.sas', directory / 'graph.txt'
        with sas_path.open('w') as stream:
            write_boolean_sas(model, stream)
        translation_seconds = time.perf_counter() - began
        remaining = timeout - translation_seconds
        if remaining <= 0:
            raise FDGraphUnsupported('Boolean SAS translation exhausted the search deadline')
        expression = (f'psv_exact_length(length={model.length},output={json.dumps(str(graph_path), ensure_ascii=False)},'
                      f'max_states={max_states},max_time={remaining},hmax={str(hmax).lower()},'
                      f'lmcut={str(lmcut).lower()},pdb_states={pdb_states},'
                      f'landmarks={str(landmarks).lower()},pdb_portfolio={str(pdb_portfolio).lower()},'
                      f'pdb_precheck={str(pdb_precheck).lower()})')
        command = [str(binary), '--search', expression,
                   '--internal-plan-file', str(directory / 'witness.txt')]
        with sas_path.open() as stream:
            result = subprocess.run(command, stdin=stream, cwd=directory, text=True,
                                    capture_output=True, timeout=remaining)
        if result.returncode not in (0, 12) or not graph_path.is_file():
            raise FDGraphUnsupported(f'Boolean SAS native search failed: {result.returncode}: {result.stderr[-500:]}')
        graph = _read_fd_graph(model, graph_path, verify_edges=True)
    graph.proof.update(translation='original_boolean_sas',
                       translation_seconds=translation_seconds,
                       total_seconds=time.perf_counter() - began,
                       backend_peak_rss_kib=max(map(int, re.findall(r'Peak memory: (\d+) KB', result.stdout)), default=0))
    metrics = re.findall(r'^PSV_METRICS (\{.*\})$', result.stdout, re.M)
    if metrics:
        graph.proof['pruning_metrics'] = json.loads(metrics[-1])
    return graph


def _build_fd_graph(model, *, max_states=300000, timeout=120, hmax=True, verify_edges=True,
                   translation='inprocess', landmarks=True, lmcut=True,
                   pdb_states=DEFAULT_PDB_STATES, pdb_portfolio=True,
                   pdb_precheck=True):
    """Measure translation, native graph construction and full projection together."""
    began = time.perf_counter()
    if translation not in ('inprocess', 'driver', 'boolean'):
        raise ValueError('Unknown FD translation interface')
    if translation == 'boolean':
        return _build_boolean_graph(model, BUILD, max_states=max_states, timeout=timeout,
                                    hmax=hmax, lmcut=lmcut, pdb_states=pdb_states,
                                    landmarks=landmarks, pdb_portfolio=pdb_portfolio,
                                    pdb_precheck=pdb_precheck)
    _check_fragment(model)
    if not (BUILD / 'bin' / 'downward').is_file():
        raise FDGraphUnsupported('optional FD exact-length plugin has not been built')
    if not math.isfinite(timeout):
        raise FDGraphUnsupported('native admission requires a finite search deadline')
    with TemporaryDirectory(prefix='swm-fd-exact-') as directory:
        directory = Path(directory)
        artifact = directory / 'graph.txt'
        sas_file = directory / 'output.sas'
        search = (f'psv_exact_length(length={model.length},output={json.dumps(str(artifact), ensure_ascii=False)},'
                  f'max_states={max_states},max_time={timeout},hmax={str(hmax).lower()},'
                  f'landmarks={str(landmarks).lower()},lmcut={str(lmcut).lower()},'
                  f'pdb_states={pdb_states},pdb_portfolio={str(pdb_portfolio).lower()},'
                  f'pdb_precheck={str(pdb_precheck).lower()})')
        command = [sys.executable, str(ROOT / 'downward' / 'fast-downward.py'),
                   '--build', str(BUILD), '--plan-file', str(directory / 'witness.txt'),
                   '--sas-file', str(sas_file), '--keep-sas-file',
                   str(model.domain_path.resolve()), str(model.problem_path.resolve()),
                   '--translate-options', '--keep-no-ops', '--keep-unimportant-variables',
                   '--keep-unreachable-facts', '--search-options', '--search', search]
        translation_seconds = None
        if translation == 'inprocess':
            from swm.pddl.reorder import _FD_LOCK
            from translate import main, normalize, options, pddl_parser
            with _FD_LOCK, redirect_stdout(io.StringIO()):
                options.set_options([str(model.domain_path), str(model.problem_path),
                                     '--keep-no-ops', '--keep-unimportant-variables',
                                     '--keep-unreachable-facts'])
                task = pddl_parser.open(str(model.domain_path), str(model.problem_path))
                normalize.normalize(task)
                sas = main.pddl_to_sas(task)
                with sas_file.open('w') as stream:
                    sas.output(stream)
            translation_seconds = time.perf_counter() - began
            remaining = timeout - translation_seconds
            if remaining <= 0:
                raise FDGraphUnsupported('native translation exhausted the shared search deadline')
            command = [str(BUILD / 'bin' / 'downward'), '--search', search,
                       '--internal-plan-file', str(directory / 'witness.txt')]
            with sas_file.open() as stream:
                result = subprocess.run(command, stdin=stream, cwd=directory, capture_output=True,
                                        text=True, timeout=remaining)
        else:
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=timeout + 1)
        subprocess_seconds = time.perf_counter() - began
        if result.returncode not in (0, 12) or not artifact.is_file():
            raise FDGraphUnsupported(f'FD graph subprocess failed with code {result.returncode}: {result.stderr[-500:]}')
        graph = _read_fd_graph(model, artifact, verify_edges=verify_edges)
    graph.proof.update(total_seconds=time.perf_counter() - began,
                       subprocess_seconds=subprocess_seconds,
                       translation=translation, translation_seconds=translation_seconds,
                       projection_seconds=time.perf_counter() - began - subprocess_seconds,
                       backend_peak_rss_kib=max(map(int, re.findall(r'Peak memory: (\d+) KB', result.stdout)), default=0))
    metrics = re.findall(r'^PSV_METRICS (\{.*\})$', result.stdout, re.M)
    if metrics:
        graph.proof['pruning_metrics'] = json.loads(metrics[-1])
    return graph


def build_fd_graph(model, *, max_states=300000, timeout=120, hmax=True, verify_edges=True,
                   translation='inprocess', landmarks=True, lmcut=True,
                   pdb_states=DEFAULT_PDB_STATES, pdb_portfolio=True,
                   pdb_precheck=True):
    """Return a verified native graph or one explicit capability failure.

    Translation, transport, budgets and malformed artifacts all fail closed;
    callers may then use the original-PDDL backend for the complete same scope.
    """
    began = time.perf_counter()
    if type(pdb_states) is not int or not 0 <= pdb_states <= 2147483646:
        # FD reserves the literal INT_MAX for infinity in its option parser.
        raise ValueError('PDB abstract-state budget must be an integer in [0, INT_MAX - 1]')
    if any(type(value) is not bool for value in (hmax, landmarks, lmcut, pdb_portfolio, pdb_precheck)):
        raise ValueError('Native pruning switches must be boolean')
    if not math.isfinite(timeout):
        raise FDGraphUnsupported('native admission requires a finite search deadline')
    if not (BUILD / 'bin' / 'downward').is_file():
        raise FDGraphUnsupported('optional FD exact-length plugin has not been built')
    if timeout <= 0 or not _GRAPH_JOBS.acquire(timeout=timeout):
        raise FDGraphUnsupported('native admission exhausted the shared search deadline')
    queue_seconds = time.perf_counter() - began
    attempts = []
    try:
        modes = (translation,) if translation != 'inprocess' else ('inprocess', 'boolean')
        for mode in modes:
            remaining = timeout - (time.perf_counter() - began)
            if remaining <= 0:
                raise FDGraphUnsupported('native attempts exhausted the shared search deadline')
            started = time.perf_counter()
            try:
                graph = _build_fd_graph(model, max_states=max_states, timeout=remaining, hmax=hmax,
                    verify_edges=verify_edges, translation=mode, landmarks=landmarks,
                    lmcut=lmcut, pdb_states=pdb_states, pdb_portfolio=pdb_portfolio,
                    pdb_precheck=pdb_precheck)
            except (OSError, subprocess.SubprocessError, StopIteration, IndexError,
                    KeyError, ValueError, AssertionError) as error:
                attempts.append({'translation': mode, 'status': 'unsupported',
                                 'seconds': time.perf_counter() - started,
                                 'reason': str(error)})
                continue
            attempts.append({'translation': mode, 'status': 'complete',
                             'seconds': time.perf_counter() - started})
            if time.perf_counter() - began > timeout:
                raise FDGraphUnsupported('native projection exhausted the shared search deadline')
            graph.proof.update(attempts=attempts, queue_seconds=queue_seconds,
                               total_seconds=time.perf_counter() - began)
            return graph
        raise FDGraphUnsupported('; '.join(row['translation'] + ': ' + row['reason']
                                          for row in attempts))
    finally:
        _GRAPH_JOBS.release()
