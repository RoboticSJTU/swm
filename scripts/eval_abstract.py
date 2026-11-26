#!/usr/bin/env python3
"""Evaluate plans from generated PDDL or natural language with fixed game rules."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event

from openai import APIStatusError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from swm.llm import call_gpt, call_gpt_json
from swm.pddl.generation import parse_pddl_output, strip_code_block
from swm.pddl.planner import _run_fast_downward, fast_downward_path as FD, _output_text
from swm.pddl.postprocess import repair_pddl
from swm.pddl.reorder import effect_predicates
from swm.abstract_planning import model as abstract_model
from swm.abstract_planning.contract import (
    DATASETS as ABSTRACT_SPLITS, execute, machine, names_only, read_plan, step, task_id, verify_plan,
)
from swm.abstract_planning.diagnostics import (
    AmbiguousActionMapping, align_actions, diagnose_failure, diagnose_reference_plan,
    diagnosis, initial_conflict, to_reference,
)
from swm.abstract_planning.eval_report import ERROR_TYPES, SPLIT_LABELS, diagnostic_group, render_abstract_report

# 基本配置
eval_model = 'gpt-5.6-sol'
eval_base_url = None # 'http://127.0.0.1:8001/v1'
mapping_model = 'deepseek/deepseek-flash'
datasets = ABSTRACT_SPLITS.copy()
max_workers = 200
prompt_file = 'training_input_nl.txt'  # 自然语言评测用 training_input_nl.txt
report_only = False
offline = False
check_data = False

MANIFEST = ROOT / 'tasks/meta/abstract_planning_test/manifest.jsonl'
PROMPT = ROOT / 'src/swm/prompt_templates' / prompt_file
try:
    output_format = {'training_input_abstract.txt': 'pddl', 'training_input_nl.txt': 'nl'}[PROMPT.name]
except KeyError:
    raise ValueError(f'Unknown prompt template: {PROMPT.name}') from None
VAL = ROOT / 'downloads/visual_pddl_v1/VAL/build/bin/Validate'
eval_root = ROOT / 'eval_results' / eval_model / (MANIFEST.parent.name + ('_nl' if output_format == 'nl' else ''))
UNRESOLVED = {'generation_error', 'mapping_error', 'mapping_incomplete', 'solver_timeout',
              'solver_memory_limit', 'search_incomplete', 'unsupported_pddl', 'infrastructure_error', 'action_mapping_ambiguous'}
MAPPING_STOP = Event()
PLAN_FILES = ('plan.txt', 'canonical_plan.txt', 'reference_val.log', 'environment_trace.json')
PDDL_FILES = ('domain.pddl', 'problem.pddl', 'candidate_val.log', 'solver.log', 'postprocess.json', 'probes.json')


class EvaluationIncomplete(RuntimeError):
    """A partial run must not be published as a final benchmark result."""


def file_revision(path):
    path = Path(path)
    stat = path.stat()
    return {'path':str(path), 'bytes':stat.st_size, 'modified_ns':stat.st_mtime_ns}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def solve_files(directory):
    started = time.monotonic()
    plan = directory / 'plan.txt'
    sas = directory / 'output.sas'
    inputs = [str(directory / 'domain.pddl'), str(directory / 'problem.pddl')]
    for path in [plan, *directory.glob('plan.txt.[0-9]*')]:
        path.unlink(missing_ok=True)

    def run(seconds, *args):
        command = [sys.executable, str(FD), '--overall-time-limit', f'{seconds}s',
                   '--overall-memory-limit', '2G', '--plan-file', str(plan),
                   '--sas-file', str(sas), *args]
        code = 0
        try:
            stdout, stderr = _run_fast_downward(command, directory, timeout=seconds + 5)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            code = error.returncode if isinstance(error, subprocess.CalledProcessError) else None
            stdout, stderr = error.stdout or '', error.stderr or ''
        return code, _output_text(stdout) + _output_text(stderr)

    code, log = run(60, '--alias', 'lama-first', *inputs)
    confirmation = None
    # LAMA's incomplete search cannot prove ADL tasks unsolvable. Confirm with
    # a complete search within the original task's remaining time budget.
    if code == 12 and not plan.exists() and not any(directory.glob('plan.txt.[0-9]*')):
        budget = max(1, int(60 - (time.monotonic() - started)))
        code, proof = run(budget, *inputs, '--search', 'astar(blind())')
        proven = code in {10,11} or (code == 12 and 'Completely explored state space -- no solution!' in proof)
        confirmation = {'budget_seconds':budget,'returncode':code,'proven_unsolvable':proven}
        log += '\nComplete-search confirmation:\n' + proof
        if proven: code = 11
    (directory / 'solver.log').write_text(log)
    plans = sorted(directory.glob('plan.txt.[0-9]*'), key=lambda p: int(p.suffix[1:]))
    if plans:
        plan.write_bytes(plans[-1].read_bytes())
    found = plan.is_file()
    for path in [sas, *plans]:
        path.unlink(missing_ok=True)
    status = {10:'candidate_unsolvable', 11:'candidate_unsolvable', 12:'search_incomplete',
              20:'solver_memory_limit', 22:'solver_memory_limit', 21:'solver_timeout',
              23:'solver_timeout', 24:'solver_timeout', 31:'invalid_pddl', 33:'invalid_pddl',
              34:'unsupported_pddl', 37:'unsupported_pddl'}.get(code, 'infrastructure_error')
    result = {'solved':found, 'returncode':code, 'status':'solved' if found else 'solver_timeout' if code is None else status}
    if confirmation is not None: result['confirmation'] = confirmation
    return result


def validate_files(domain, problem, plan, log_path):
    run = subprocess.run([str(VAL), '-v', str(domain), str(problem), str(plan)], capture_output=True, text=True, timeout=30)
    log_path.write_text(run.stdout + run.stderr)
    return run.returncode == 0 and 'Plan valid' in run.stdout


def local_paths(row, manifest):
    """Resolve installed artifacts, ignoring historical absolute build paths."""
    name = Path(manifest).parent.name
    return {
        'image': str(ROOT/'tasks/images'/name/row.get('image_file', row['task']+'.png')),
        'metadata': str(Path(manifest).parent/row['task']),
        'gt': str(ROOT/'eval_results/gt'/name/row['task']),
    }


def load_tasks(manifest=MANIFEST, selected_datasets=None):
    manifest = Path(manifest)
    source = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    name = manifest.parent.name
    inputs = json.loads((ROOT/'tasks/instructions'/f'instructions_{name}.json').read_text())
    expected_keys = defaultdict(set)
    for row in source:
        expected_keys[row['dataset']].add(row['task'])
    if set(expected_keys)-set(ABSTRACT_SPLITS):
        raise ValueError('Unknown abstract split in manifest')
    if set(inputs) != set(expected_keys):
        raise ValueError('Instruction split keys differ from manifest')
    for dataset, keys in expected_keys.items():
        if set(inputs[dataset]) != keys:
            raise ValueError('Instruction task keys differ from manifest: ' + dataset)
    if len({r['task'] for r in source}) != len(source):
        raise ValueError('Task IDs must be globally unique')
    selected = set(expected_keys) if selected_datasets is None else set(selected_datasets)
    if selected - expected_keys.keys():
        raise ValueError('Unknown abstract split')
    rows = [row.copy() for row in source if row['dataset'] in selected]
    if not rows:
        raise ValueError('No selected tasks')
    for row in rows:
        row.update(local_paths(row,manifest))
        row['id'] = task_id(row)
        row['split'] = SPLIT_LABELS[row['dataset']]
        prompt = inputs[row['dataset']][row['task']]
        if prompt != row['instruction'] or not Path(row['image']).is_file():
            raise ValueError('Input differs from manifest: ' + row['id'])
        gt = Path(row['gt'])
        scenario = json.loads((Path(row['metadata'])/'scenario.json').read_text())
        if any((gt/'round1'/f'{kind}.pddl').read_text().strip() != scenario[kind].strip()
               for kind in ('domain', 'problem')):
            raise ValueError('Reference differs from scenario: ' + row['id'])
        if scenario['instruction'] != prompt or scenario['game'] != row['game']:
            raise ValueError('Instruction differs from scenario: ' + row['id'])
        row['instruction'] = names_only(prompt,scenario['domain'])
        row['_scenario'] = scenario
        row['_oracle'] = machine(scenario['domain'], scenario['problem'])
    by_id = {task_id(r):r for r in source}
    for row in rows:
        if pair := row.get('paired_with'):
            paired = by_id[task_id(pair)]
            if paired['game'] != row['game'] or local_paths(paired,manifest)['image'] != row['image']:
                raise ValueError('Rule pair differs in game or image: ' + row['id'])
    return rows


def check_tasks(rows):
    """Audit reference steps and native rules offline, only when requested."""
    steps = json.loads((ROOT/'tasks/steps'/f'steps_{MANIFEST.parent.name}.json').read_text())
    for row in rows:
        plan = read_plan(Path(row['gt'])/'round1/plan.txt')
        reference_steps = steps[row['dataset']][row['task']]
        if (reference_steps != row['steps'] or len(plan) != row['plan_length']
                or len(reference_steps) != len(plan)
                or '\n'.join(reference_steps)+'\n' != (Path(row['gt'])/'kf_actions.txt').read_text()):
            raise ValueError('Steps differ from reference: ' + row['id'])
        for i, (action, text) in enumerate(zip(plan, reference_steps), 1):
            if not text.startswith(f'{i}. {action[0]}('+', '.join(action[1:])+'). Result: '):
                raise ValueError('Plan action differs from steps: ' + row['id'])
        verify_plan(row['_scenario'],plan)


def validate_mapping(value, source, targets, fixed):
    if not isinstance(value,dict) or set(value) != {'objects'} or not isinstance(value['objects'],dict):
        raise ValueError('Mapping must contain only an objects dictionary')
    aliases = value['objects']
    if aliases.keys() != set(source) - fixed.keys():
        raise ValueError('Incomplete/extra mapping keys')
    mapping = dict(fixed)
    for name,target in aliases.items():
        if target is None:
            continue
        if not isinstance(target, str) or target not in targets:
            raise ValueError('Unknown mapping target')
        if target in mapping.values():
            raise ValueError('Object mapping must be one-to-one')
        mapping[name] = target
    return mapping


def align_objects(row, candidate, directory, *, offline=False):
    source, target = set(candidate['objects']), set(row['_oracle']['objects'])
    fixed = {x:x for x in source & target}
    aliases, eligible = source - fixed.keys(), target - fixed.keys()
    identity = {'model':mapping_model, 'candidate':[(directory/n).read_text() for n in ['domain.pddl','problem.pddl']],
                'image':file_revision(row['image']), 'oracle':row['_scenario']['problem'], 'instruction':row['instruction']}
    evidence = {'source':'identity', 'objects':fixed}
    path = directory/'mapping.json'
    if aliases and eligible:
        saved = {}
        last_error = None
        try:
            if path.exists():
                saved = json.loads(path.read_text())
                if not isinstance(saved, dict):
                    raise ValueError('Mapping cache must be a JSON object')
        except (OSError,ValueError) as error:
            saved = {}
            last_error = f'{type(error).__name__}: {error}'
        use_cache = saved.get('identity') == identity and 'response' in saved and not saved.get('error')
        data = {'instruction':row['instruction'],'candidate_objects':sorted(aliases),
                'eligible_visible_ids':sorted(eligible), 'fixed_identity':fixed,
                'candidate_initial_facts':[list(a) for a in sorted(candidate['init'])]}
        prompt = ('Map candidate aliases to visible scene IDs or numeric symbols defined by the rules. '
                  'Use the image, labels and initial facts; preserve fixed identities. Each target may be used once. '
                  'Use null for unsupported or ambiguous identities. Return only JSON {"objects":{"candidate_alias":"visible_id_or_null"}}.\n'
                  + json.dumps(data,ensure_ascii=False))
        attempts = 0
        while True:
            if MAPPING_STOP.is_set():
                raise RuntimeError('Object mapping cancelled before validation')
            response, capture = None, {}
            try:
                if use_cache:
                    response, capture = saved['response'], saved.get('capture',{})
                else:
                    if offline:
                        raise RuntimeError('Offline evaluation requires a valid cached object mapping')
                    attempts += 1
                    feedback = (f'\nMapping attempt {attempts}: correct the previous JSON/schema error: {last_error[:500]}'
                                if last_error else '')
                    raw = call_gpt(mapping_model,prompt+feedback,[Path(row['image'])],
                                   response_format={'type':'json_object'},temperature=0,capture=capture)
                    capture['raw_output'] = raw
                    response = json.loads(raw)
                aligned = validate_mapping(response,source,target,fixed)
            except Exception as error:
                if offline and not use_cache:
                    raise
                last_error = f'{type(error).__name__}: {error}'
                write_json(path,{'source':'vlm','identity':identity,'objects':fixed,'response':response,
                                 'capture':capture,'attempts':attempts,'error':last_error})
                if use_cache:
                    use_cache = False
                    continue
                # 重试只依据解析与映射约束；不使用计划执行结果挑选对象身份。
                if MAPPING_STOP.wait(min(2**min(attempts-1,5),30)):
                    raise RuntimeError('Object mapping cancelled before validation') from error
            else:
                break
        evidence = {'source':'vlm','objects':aligned,'response':response,'capture':capture,
                    'attempts':attempts or saved.get('attempts',0)}
        if last_error: evidence['last_error'] = last_error
    evidence['identity'] = identity
    evidence['unmapped'] = sorted(source - evidence['objects'].keys())
    write_json(path, evidence)
    return evidence


def evaluate_candidate(row, raw_output, directory, *, offline=False):
    directory.mkdir(parents=True,exist_ok=True)
    for name in (*PLAN_FILES, *PDDL_FILES, 'action_mapping.json', 'result.json', 'result.md'):
        (directory/name).unlink(missing_ok=True)
    result = dict(result_identity(row), task_success=False, status='invalid_output', spl=0.0, init_comparable=False)
    domain_path, problem_path, plan_path = (directory / name for name in ('domain.pddl', 'problem.pddl', 'plan.txt'))
    try:
        domain,problem = parse_pddl_output(raw_output)
        # 保留修复前的输入；无法确定修复方式时仍能查看原始证据。
        domain_path.write_text(domain+'\n')
        problem_path.write_text(problem+'\n')
        domain,problem,changes = repair_pddl(domain,problem)
        if changes:
            domain_path.write_text(domain+'\n')
            problem_path.write_text(problem+'\n')
            write_json(directory/'postprocess.json',changes)
        candidate = machine(domain,problem)
    except (ValueError,KeyError,TypeError,IndexError,NotImplementedError) as error:
        (directory/'mapping.json').unlink(missing_ok=True)
        result['error'] = str(error)
        return result
    reference_model = row['_oracle']
    expected, actual = reference_model['ops'].keys(), candidate['ops'].keys()
    result['interface_ok'] = bool(actual) and actual <= expected
    result['missing_operations'] = sorted(expected - actual)
    if not result['interface_ok']:
        result.update(status='interface_mismatch', error='Unknown or missing action names')
        return result
    solver = result['solver'] = solve_files(directory)
    if not solver['solved']:
        result['status'] = solver['status']
        result['error'] = '\n'.join((directory/'solver.log').read_text().splitlines()[-12:])
        return result
    plan = read_plan(plan_path)
    result['plan_length'] = len(plan)
    result['candidate_val_pass'] = validate_files(domain_path, problem_path, plan_path, directory/'candidate_val.log')
    if not result['candidate_val_pass']:
        result['status'] = 'candidate_plan_invalid'
        return result
    try:
        alignment = align_objects(row,candidate,directory,offline=offline)
        mapping = alignment['objects']
        if missing := sorted({x for action in plan for x in action[1:]} - mapping.keys()):
            if set(mapping.values()) == set(reference_model['objects']):
                index,action = next((i,a) for i,a in enumerate(plan,1) if set(a[1:]) & set(missing))
                result.update(status='invalid_plan_object',unmapped_plan_objects=missing,
                    diagnosis=diagnosis('init_error','init_mismatch',
                        f'第 {index} 步 `{" ".join(action)}` 使用没有对应真实对象的 '+
                        '、'.join(f'`{x}`' for x in missing)+'；全部真实对象已有一一对应的身份。'))
                return result
            result.update(status='mapping_incomplete',error='Unmapped plan objects: '+', '.join(missing))
            return result
    except Exception as error:
        result.update(status='mapping_error', error=f'{type(error).__name__}: {error}')
        return result
    result['mapping_source'] = alignment['source']
    if (mapping.keys() == set(candidate['objects'])
            and set(mapping.values()) == set(reference_model['objects'])
            and candidate['arities'] == reference_model['arities']):
        result['init_comparable'] = True
        mapped = {(p, *(mapping[x] for x in args)) for p,*args in candidate['init']}
        result['init_correct'] = mapped == reference_model['init']
    try:
        adapters = align_actions(candidate,reference_model,mapping,{a[0] for a in plan})
        canonical = [to_reference(a,mapping,adapters) for a in plan]
        write_json(directory/'action_mapping.json',{'source':'predicate extensions and action structure; no parameter-name or goal matching','operations':adapters})
    except AmbiguousActionMapping as error:
        result.update(status='action_mapping_ambiguous', error=str(error))
        return result
    return replay_reference(row,canonical,directory,result)


def replay_reference(row,canonical,directory,result):
    plan_path = directory/'canonical_plan.txt'
    plan_path.write_text('\n'.join('('+' '.join(a)+')' for a in canonical)+'\n')
    gt = Path(row['gt'])/'round1'
    oracle = execute(row['_oracle'],canonical)
    reference = validate_files(gt/'domain.pddl', gt/'problem.pddl', plan_path, directory/'reference_val.log')
    result['reference_val_pass'] = reference
    result['environment'] = {k:v for k,v in oracle.items() if k != 'trace'}
    write_json(directory/'environment_trace.json',[[list(a) for a in sorted(s)] for s in oracle['trace']])
    oracle_pass = oracle['executable'] and oracle['goal_reached']
    independent = False
    if oracle_pass:
        try:
            verify_plan(row['_scenario'],canonical)
            independent = True
        except (ValueError, AssertionError):
            pass
    result['independent_game_pass'] = independent
    if independent != oracle_pass:
        result.update(status='infrastructure_error', error='Independent rules disagree with reference PDDL')
        return result
    if reference != oracle_pass:
        result.update(status='infrastructure_error', error='VAL and reference interpreter disagree')
        return result
    result['task_success'] = reference
    if reference and row.get('reference_optimal') and row.get('plan_length'):
        result['reference_plan_length'] = row['plan_length']
        result['spl'] = row['plan_length'] / max(row['plan_length'],len(canonical))
    result['status'] = ('success' if reference
                        else 'illegal_environment_action' if not oracle['executable'] else 'real_goal_not_reached')
    return result


def ground_nl_plan(row,steps,directory,*,offline=False):
    oracle = row['_oracle']
    operations = oracle['ops']
    signatures = {name:op[':parameters'] for name,op in operations.items()}
    rules = row['instruction'].partition('\n')[2]
    changing = set().union(*(effect_predicates(op[':effect']) for op in operations.values()))
    sexpr = abstract_model.sexpr
    initial = ' '.join(sexpr(list(fact)) for fact in sorted(oracle['init']) if fact[0] in changing)
    interface = '\n'.join(
        f"{name}({', '.join(signatures[name])})\n"
        f"  requires {sexpr(op[':precondition'])}\n  effects {sexpr(op[':effect'])}"
        for name,op in operations.items()
    )
    numbered = '\n'.join(f'{i}. {json.dumps(step,ensure_ascii=False)}' for i,step in enumerate(steps,1))
    prompt = (
        'Transcribe the numbered plan into the exact game interface below. A separate evaluator checks legality and goals.\n'
        '1. Output one entry per numbered step, in order, as {"actions":[["operation","argument"],null]}. '
        'Each action array starts with the operation name, followed by exactly its declared parameters in order. '
        'For a zero-parameter operation such as wait_for_controller, output ["wait_for_controller"]. '
        'Keep all arguments as strings. Use a bare JSON null only for an entire step whose single operation '
        'or binding cannot be determined.\n'
        '2. Read parameter roles from the operation definition. For example, a source in (at ?tile ?from) '
        'is the tile source, even when the text describes the blank moving in the opposite direction. '
        'Distinguish the actor location, object location, and destination. '
        'Copy explicit canonical IDs exactly and expand documented aliases to their canonical IDs. '
        'Preserve explicit operations and arguments even if they violate a rule. Never add, split, merge, '
        'reorder or substitute actions to make the plan work.\n'
        '3. Track the state needed for binding from the initial facts through earlier steps. Fill omitted arguments only when uniquely '
        'implied by the text, scene and rules. Bind counters and locations to their values before the current step, '
        'derive result values from that operation, then carry its effects forward. '
        'For example, two consecutive energy-consuming actions without recharge use q3 to q2 and then q2 to q1. '
        'Enable and wait operations update their own flags and preserve charge.\n'
        '4. Check each array against its numbered step and the declared parameter count. Return only the JSON object.\n'
        f'\nGame rules (the scene image is attached):\n{rules}\n'
        f"\nCanonical symbols:\n{' '.join(oracle['objects'])}\n"
        f'\nInitial values of changing facts:\n{initial}\n'
        f'\nOperation definitions:\n{interface}\n'
        f'\nNumbered plan:\n{numbered}'
    )
    path = directory/'grounding.json'
    identity = {'model':mapping_model,'prompt':prompt,'image':file_revision(row['image'])}

    def actions_from(response):
        actions = response.get('actions') if isinstance(response,dict) else None
        if not isinstance(actions,list):
            raise ValueError('Expected an actions array')
        if len(actions) != len(steps):
            raise ValueError(f'Expected {len(steps)} action entries, received {len(actions)}')
        for index,action in enumerate(actions,1):
            if action is None:
                continue
            if not isinstance(action,list) or not action or any(
                    not isinstance(token,str) or not re.fullmatch(r'[a-zA-Z0-9_-]+',token) for token in action):
                raise ValueError(f'Action entry {index} must be a nonempty array of symbol strings or null')
            name = action[0].lower()
            if name not in signatures:
                raise ValueError(f'Action entry {index} uses undeclared operation {name}')
            parameters = signatures[name]
            if len(action)-1 != len(parameters):
                raise ValueError(f"Action entry {index} must match {name}({', '.join(parameters)}) exactly")
        return [tuple(token.lower() for token in action) if action is not None else None for action in actions]

    try:
        cached = json.loads(path.read_text())
        if cached['identity'] == identity:
            return actions_from(cached['response'])
    except (OSError,ValueError,KeyError,TypeError):
        pass
    if offline:
        raise RuntimeError('Offline evaluation requires a valid cached NL grounding')

    action_schema = {'type':'array','minItems':1,'items':{'type':'string','pattern':'^[a-zA-Z0-9_-]+$'}}
    schema = {'type':'object','required':['actions'],'additionalProperties':False,'properties':{
        'actions':{'type':'array','minItems':len(steps),'maxItems':len(steps),
                   'items':{'anyOf':[action_schema,{'type':'null'}]}}}}
    response_format = {'type':'json_schema','json_schema':{'name':'grounded_plan','schema':schema}}
    request = base_request = f'{prompt}\nThe actions array must have exactly {len(steps)} entries.'
    for attempt in range(3):
        try:
            response = call_gpt_json(mapping_model,request,[Path(row['image'])],
                                     response_format=response_format,attempts=1,temperature=0)
            actions = actions_from(response)
        except (RuntimeError,ValueError) as error:
            if attempt == 2:
                raise
            cause = error.__cause__ if isinstance(error,RuntimeError) else error
            if (response_format['type'] == 'json_schema' and isinstance(cause,APIStatusError)
                    and cause.status_code in {400,422}):
                response_format = {'type':'json_object'}
            elif isinstance(cause,ValueError):
                request = f'{base_request}\nCorrect the response format. {cause}.'
            else:
                raise
        else:
            write_json(path,{'identity':identity,'response':response})
            return actions


def evaluate_nl_candidate(row,raw_output,directory,*,offline=False):
    for name in PLAN_FILES:
        (directory/name).unlink(missing_ok=True)
    result = dict(result_identity(row), task_success=False, status='invalid_output', spl=0.0, init_comparable=False)
    try:
        data = json.loads(strip_code_block(raw_output))
        steps = data.get('plan_sequence') if isinstance(data,dict) else None
        if not isinstance(steps,list) or not steps or any(not isinstance(s,str) or not s.strip() for s in steps):
            raise ValueError('plan_sequence must be a nonempty list of non-blank strings')
    except ValueError as error:
        result['error'] = str(error)
        return result
    steps = [s.strip() for s in steps]
    (directory/'plan.txt').write_text('\n'.join(steps)+'\n')
    try:
        canonical = ground_nl_plan(row,steps,directory,offline=offline)
    except Exception as error:
        result.update(status='mapping_error', error=f'{type(error).__name__}: {error}')
        return result
    if None in canonical:
        index = canonical.index(None)
        result.update(status='invalid_plan_step',error=f'第 {index+1} 步无法映射为单一游戏动作：{steps[index]}')
        return result
    result['plan_length'] = len(canonical)
    return replay_reference(row,canonical,directory,result)


def save_result(directory,result):
    write_json(directory/'result.json',result)
    lines = [f"任务：{result['id']}", f"游戏：{result['game']}", f"状态：{result['status']}",
             f"真实任务完成：{'是' if result['task_success'] else '否'}"]
    if proof := result.get('diagnosis'):
        lines.append('错误：'+proof['evidence'])
    elif result.get('error'):
        lines.append('错误：'+result['error'])
    links = [f'[{name}]({name})' for name in ['raw_output.txt','domain.pddl','problem.pddl','postprocess.json','plan.txt','mapping.json','grounding.json','canonical_plan.txt','result.json'] if (directory/name).exists()]
    (directory/'result.md').write_text('\n\n'.join(lines+[' · '.join(links)])+'\n')


def diagnose_result(result,row,directory):
    """Reuse saved models, mappings and this task's plan; never call a model or solver."""
    if result.get('output_format') == 'nl' or result['task_success'] or result['status'] in UNRESOLVED:
        return None
    if result['status'] == 'invalid_plan_object':
        return result.get('diagnosis')
    try:
        problem = (directory/'problem.pddl').read_text()
        if conflict := initial_conflict(problem):
            return conflict
        if result['status'] not in {'interface_mismatch','illegal_environment_action','real_goal_not_reached','candidate_unsolvable'}:
            return None
        domain = (directory/'domain.pddl').read_text()
        candidate = machine(domain,problem)
        reference = row.get('_oracle')
        if reference is None:
            gt = Path(row['gt'])/'round1'
            reference = machine((gt/'domain.pddl').read_text(),(gt/'problem.pddl').read_text())
        if result['status'] == 'interface_mismatch':
            return diagnose_failure(result,candidate,reference,{},[],[],execute)
        source,targets = set(candidate['objects']),set(reference['objects'])
        mapping = {name:name for name in source & targets}
        if source - mapping.keys() and targets - mapping.keys():
            saved = json.loads((directory/'mapping.json').read_text())
            if saved.get('error') or saved['identity']['candidate'] != [domain,problem]:
                return None
            mapping = validate_mapping(saved['response'],source,targets,mapping)
        if result['status'] == 'candidate_unsolvable':
            return diagnose_reference_plan(candidate,reference,mapping,
                read_plan(Path(row['gt'])/'round1/plan.txt'),execute)
        return diagnose_failure(result,candidate,reference,mapping,read_plan(directory/'plan.txt'),
                                read_plan(directory/'canonical_plan.txt'),execute)
    except (OSError,ValueError,KeyError,TypeError,IndexError,NotImplementedError):
        return None


def result_identity(row):
    return {key:row[key] for key in ['id','dataset','split','game','task','format_version','release_id','evaluation_kind'] if key in row}


def run_one(row,template,code_revision,*,offline=False):
    directory = eval_root/row['dataset']/row['task']
    directory.mkdir(parents=True, exist_ok=True)
    nl = output_format == 'nl'
    prompt = template.format(instruction=names_only(row['instruction'],row['_scenario']['domain']),
                             robot_configuration='single_arm')
    identity = {'model':eval_model,'prompt':prompt,'image':file_revision(row['image'])}
    identity.update({key:row[key] for key in ['format_version','release_id'] if key in row})
    start = time.monotonic()
    stage = 'generation_error'
    try:
        raw_path = directory/'raw_output.txt'
        generation_path = directory/'generation.json'
        raw = None
        try:
            saved = json.loads(generation_path.read_text())
            if isinstance(saved,dict) and saved.get('identity') == identity:
                cached = raw_path.read_text()
                if (nl or cached.strip()) and saved.get('output') == cached:
                    raw = cached
        except (OSError,ValueError):
            pass
        if raw is None:
            if offline:
                raise RuntimeError('Offline evaluation requires a valid cached generation')
            for name in (*PDDL_FILES, *PLAN_FILES, 'raw_output.txt', 'generation.json', 'mapping.json', 'grounding.json'):
                (directory/name).unlink(missing_ok=True)
            capture = {}
            raw = call_gpt(eval_model, prompt, [Path(row['image'])], base_url=eval_base_url, temperature=0, capture=capture)
            if nl and raw is None:
                raw = ''
            if not isinstance(raw, str) or (not nl and not raw.strip()):
                raise ValueError('Model returned empty output')
            raw_path.write_text(raw)
            write_json(generation_path, dict(identity=identity, output=raw, capture=capture))
        stage = 'infrastructure_error'
        evaluate = evaluate_nl_candidate if nl else evaluate_candidate
        result = evaluate(row,raw,directory,offline=offline)
    except Exception as error:
        result = dict(result_identity(row), task_success=False, status=stage, spl=0.0,
                      init_comparable=False, error=f'{type(error).__name__}: {error}')
    result.update(model=eval_model,elapsed_seconds=round(time.monotonic()-start,3),evaluator_revision=code_revision)
    if nl:
        result['output_format'] = output_format
    if not result['task_success']:
        result['diagnosis'] = diagnose_result(result, row, directory)
    save_result(directory, result)
    return result


def aggregate(results):
    evaluated = [r for r in results if r['status'] not in UNRESOLVED]
    successes = sum(r['task_success'] for r in evaluated)
    errors = Counter(diagnostic_group(r,UNRESOLVED) for r in evaluated if not r['task_success'])
    total = len(results)
    nl = output_format == 'nl'
    error_types = ('output_error','plan_error') if nl else ERROR_TYPES
    comparable = [r for r in results if r.get('init_comparable')]
    acc = successes/total if total else None
    return {'total':total,'evaluated':len(evaluated),'unresolved':total-len(evaluated),
            'unattributed':errors['unattributed'],
            'task_success':successes,'task_success_rate':acc,'acc':acc,
            'spl':sum(r.get('spl',0.0) for r in results if r['task_success'])/total if total else None,
            'solv':sum(bool(r.get('solver',{}).get('solved')) for r in results)/total if total and not nl else None,
            'init_comparable':len(comparable),'init_unassessed':total-len(comparable),
            'init_acc':sum(r.get('init_correct',False) for r in comparable)/len(comparable) if comparable else None,
            'error_counts':{key:errors[key] for key in error_types},
            'error_rates':{key:errors[key]/len(evaluated) if evaluated else None for key in error_types},
            'statuses':dict(Counter(r['status'] for r in results))}


def summarize(results,rows,*,require_complete=True):
    # Thread completion order must not change floating-point aggregates or reports.
    results = sorted(results,key=lambda result:result['id'])
    expected = {task_id(row) for row in rows}
    by_id = {r['id']:r for r in results}
    if len(expected) != len(rows) or len(by_id) != len(results) or by_id.keys() != expected:
        (eval_root/'summary_abs.json').unlink(missing_ok=True)
        (eval_root/'report_abs.md').unlink(missing_ok=True)
        raise EvaluationIncomplete('评测结果与 manifest 不一一对应，禁止生成最终报告')
    summary = {'model':eval_model,'mapping_model':mapping_model,'manifest':file_revision(MANIFEST),
               'overall':aggregate(results),'by_split':{},'by_game':{},'pairs':[], 'decision_source':'program; VLM supplies object identity aliases only'}
    if output_format == 'nl':
        summary.update(output_format=output_format,decision_source='program; VLM grounds natural-language plan steps')
    summary['test_version'] = MANIFEST.parent.name
    summary['release_ids'] = sorted({row['release_id'] for row in rows if 'release_id' in row})
    for field,label in [('split','by_split'),('game','by_game'),('evaluation_kind','by_evaluation_kind')]:
        groups = defaultdict(list)
        for result in results:
            if field in result: groups[result[field]].append(result)
        summary[label] = {key:aggregate(group) for key,group in groups.items()}
        if field == 'split':
            for key,group in groups.items():
                games = defaultdict(list)
                for result in group:
                    games[result['game']].append(result)
                by_game = {game: aggregate(games[game]) for game in sorted(games)}
                summary[label][key]['by_game'] = by_game
                rates = [stats['task_success_rate'] for stats in by_game.values()]
                summary[label][key]['macro_game_success_rate'] = sum(rates)/len(rates)
    macro = [stats['task_success_rate'] for stats in summary['by_game'].values()]
    summary['macro_game_success_rate'] = sum(macro)/len(macro) if macro else None
    for row in rows:
        if row['dataset'] != 'unseen_composition' or not (pair := row.get('paired_with')):
            continue
        base_id = task_id(pair)
        if base_id not in by_id:
            continue
        a,b = by_id[base_id],by_id[row['id']]
        summary['pairs'].append({'base':base_id,'variant':row['id'],
                                 'evaluated':a['status'] not in UNRESOLVED and b['status'] not in UNRESOLVED,
                                 'both_tasks_pass':a['task_success'] and b['task_success']})
    pairs = summary['pairs']
    summary['pairs_evaluated'] = sum(pair['evaluated'] for pair in pairs)
    summary['pairs_unresolved'] = len(pairs) - summary['pairs_evaluated']
    summary['pair_success_rate'] = sum(pair['both_tasks_pass'] for pair in pairs)/len(pairs) if pairs else None
    overall = summary['overall']
    summary['complete'] = bool(rows) and overall['unresolved'] == 0 and overall['unattributed'] == 0
    summary['unfinished_tasks'] = [r['id'] for r in results if diagnostic_group(r,UNRESOLVED) in {'pending','unattributed'}]
    suffix = '' if summary['complete'] else '.partial'
    write_json(eval_root/f'summary_abs{suffix}.json',summary)
    (eval_root/f'report_abs{suffix}.md').write_text(render_abstract_report(results,summary,eval_root,UNRESOLVED))
    if not summary['complete']:
        # An earlier final report must not masquerade as this run's result.
        (eval_root/'summary_abs.json').unlink(missing_ok=True)
        (eval_root/'report_abs.md').unlink(missing_ok=True)
        if require_complete:
            raise EvaluationIncomplete(f"已判定 {overall['evaluated']}/{overall['total']} 条，"
                                       f"未完成 {overall['unresolved']} 条，待归因 {overall['unattributed']} 条。"
                                       f"诊断报告 {eval_root / 'report_abs.partial.md'}")
    else:
        (eval_root/'summary_abs.partial.json').unlink(missing_ok=True)
        (eval_root/'report_abs.partial.md').unlink(missing_ok=True)
    return summary


def load_saved_results(rows):
    """Load decisions and derive local diagnoses without generation or solving."""
    results = []
    for row in rows:
        identity = result_identity({**row, 'id': task_id(row), 'split': SPLIT_LABELS[row['dataset']]})
        directory = eval_root/row['dataset']/row['task']
        try:
            result = json.loads((directory/'result.json').read_text())
            if not isinstance(result,dict) or any(result.get(key) != value for key,value in identity.items()):
                raise ValueError('Result identity differs from manifest')
            if not isinstance(result.get('status'),str) or type(result.get('task_success')) is not bool:
                raise ValueError('Result is missing a decision')
            if result.get('model',eval_model) != eval_model:
                raise ValueError('Result belongs to another model')
            if result.get('output_format','pddl') != output_format:
                raise ValueError('Result uses another output format')
            if result['status'] not in UNRESOLVED | {'success','invalid_output','invalid_pddl','interface_mismatch',
                    'candidate_unsolvable','candidate_plan_invalid','illegal_environment_action','real_goal_not_reached','invalid_plan_object','invalid_plan_step'}:
                raise ValueError('Unknown result status')
            if result['task_success'] != (result['status'] == 'success'):
                raise ValueError('Result status contradicts task success')
        except (OSError,ValueError,KeyError,TypeError) as error:
            result = {**identity,'task_success':False,'status':'infrastructure_error',
                      'error':f'结果文件无法读取：{type(error).__name__}: {error}'}
        if not result['task_success']:
            result['diagnosis'] = diagnose_result(result,row,directory)
        results.append(result)
    return results


def main(*,report_only=False,check_data=False,offline=False):
    if check_data:
        rows = load_tasks(MANIFEST,datasets)
        check_tasks(rows)
        print(f'数据检查通过：{len(rows)} 条；未调用模型。',flush=True)
        return
    eval_root.mkdir(parents=True,exist_ok=True)
    if report_only:
        rows = (json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip())
        rows = [row for row in rows if row['dataset'] in datasets]
        for row in rows:
            row['id'] = task_id(row)
            row.update(local_paths(row,MANIFEST))
        results = load_saved_results(rows)
    else:
        MAPPING_STOP.clear()
        rows = load_tasks(MANIFEST,datasets)
        template = PROMPT.read_text()
        code_revision = [file_revision(p) for p in [__file__, MANIFEST, abstract_model.__file__,
            *(ROOT/'src/swm/abstract_planning'/name for name in ['contract.py','diagnostics.py','eval_report.py'])]]
        mode = '离线重评，禁止模型调用' if offline else f'映射模型 {mapping_model}'
        print(f'开始评测：{eval_model}，{len(rows)} 条；{mode}。',flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            try:
                for row in rows:
                    futures.append(pool.submit(run_one, row, template, code_revision, offline=offline))
                results = [future.result() for future in as_completed(futures)]
            except BaseException:
                MAPPING_STOP.set()
                for future in futures:
                    future.cancel()
                raise
    summary = summarize(results,rows)
    overall = summary['overall']
    print(f"{eval_model}：任务成功 {overall['task_success']}/{overall['evaluated']}，未完成 {overall['unresolved']}；报告：   {eval_root / 'report_abs.md'}",flush=True)


if __name__ == '__main__':
    if check_data and report_only:
        raise SystemExit('check_data cannot be combined with report_only')
    try:
        main(report_only=report_only,check_data=check_data,offline=offline)
    except EvaluationIncomplete as error:
        raise SystemExit(f'评测失败：{error}') from error
