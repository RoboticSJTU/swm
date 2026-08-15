from pathlib import Path
import os
import signal
import subprocess
import sys
from swm.utils.pddl.plan_reorder import plan_reorder
import re
fast_downward_path = Path(__file__).parent.parent.parent.parent.parent / "downward" / "fast-downward.py"
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
    match = re.search(r"(?:Undefined|Undeclared)\s+(predicate|object|variable)\s*\nGot:\s*([^\n]+)", log, re.I)

    if match:
        detail = f"undefined {match.group(1).lower()} '{match.group(2).strip()}'{location}"
    elif match := re.search(r"Predicate '([^']+)' of arity (\d+) used\s+with (\d+) arguments", log, re.I):
        detail = f"predicate '{match.group(1)}' expects {match.group(2)} arguments but got {match.group(3)}{location}"
    elif re.search(r"Expected .*words:\s*:domain", log, re.IGNORECASE):
        detail = "the problem is missing the required (:domain NAME) block"
    elif match := re.search(r"(?:^|\n)([^\n]+)\nSyntax:", log):
        detail = match.group(1).strip()
    elif "No relaxed solution" in log or re.search(r"no solution", log, re.IGNORECASE):
        operators = re.search(r"Translator operators:\s*(\d+)", log)
        detail = "no plan was found" + (" and the translator produced no grounded operators" if operators and operators.group(1) == "0" else "")
    elif code == "30":
        detail = "the translator crashed while compiling the PDDL task"
    elif code == "31":
        lines = re.split(r"translate exit code", log, flags=re.I)[0].splitlines()
        detail = next((line.strip() for line in reversed(lines) if line.strip() and not re.match(r"(?:INFO|Parsing|->)", line.strip())), "invalid PDDL input")
    elif code == "34":
        detail = "the requested search configuration is unsupported"
    else:
        diagnostics = re.findall(r"(?im)^\s*((?:\w*(?:Error|Exception):|Reason:|Critical error:|Expected |Missing |Unsupported |Invalid ).+)$", log)
        detail = diagnostics[-1].strip() if diagnostics else "the planner failed without a diagnostic message"

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


def _run_fast_downward(cmd: list[object], cwd: Path) -> None:
    """Run Fast Downward under both an internal and process-tree wall guard."""

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = process.communicate(timeout=_SOLVER_WALL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd,
            _SOLVER_WALL_TIMEOUT_SECONDS,
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


def solve_pddl(domain_file, problem_file, *, reorder_plan=True):
    """求解PDDL，并返回True或False来判断是否求解成功，且支持并行化求解"""

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
        "astar(lmcut())"
    ]
    try:
        _run_fast_downward(cmd, domain_file.parent)
        if reorder_plan:
            plan_reorder(domain_file, problem_file, plan_file, plan_file)
        return True

    except Exception as e:
        stdout = _output_text(getattr(e, "stdout", None))
        stderr = _output_text(getattr(e, "stderr", None))
        details = [f"{type(e).__name__}: {e}"]
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        error_file.write_text("\n\n".join(details), encoding="utf-8")
        return False
