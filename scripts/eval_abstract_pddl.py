#!/usr/bin/env python3
"""Generate PDDL, align identities with a VLM, then evaluate with fixed rules."""
from __future__ import annotations

import json
import argparse
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from swm.llm import call_gpt
from swm.pddl.generation import parse_pddl_output
from swm.pddl.planner import _run_fast_downward, fast_downward_path as FD, _output_text
from swm.pddl.postprocess import repair_pddl
from swm.pddl.strips import validate_untyped_pddl
from swm.abstract_planning.model import parse_sexpr, section, fields, condition, apply_effect
from swm.abstract_planning import novel_games, visual_games
from swm.abstract_planning.instructions import names_only
from swm.abstract_planning.test_contract import DATASETS as ABSTRACT_SPLITS, task_id, verify_plan
from swm.abstract_planning.action_alignment import align_actions, to_reference, AmbiguousActionMapping
from swm.abstract_planning.eval_report import ERROR_TYPES, SPLIT_LABELS, diagnostic_group, render_abstract_report
from swm.abstract_planning.diagnostics import diagnose_failure, initial_conflict

# 固定路径；只保留模型、数据集、并发这些实验配置。
eval_model = '9B_human_abstact'
mapping_model = 'Qwen3.8-27B'
datasets = ['abs_a_id', 'abs_a_scale', 'abs_b_base', 'abs_b_rule']
max_workers = 200
MANIFEST = ROOT / 'tasks/meta/abstract_planning_test/manifest.jsonl'
PROMPT = ROOT / 'src/swm/prompt_templates/training_input_abstract.txt'
VAL = ROOT / 'downloads/visual_pddl_v1/VAL/build/bin/Validate'
eval_root = ROOT / 'eval_results' / eval_model / MANIFEST.parent.name
UNRESOLVED = {'generation_error', 'mapping_error', 'mapping_incomplete', 'solver_timeout',
              'solver_memory_limit', 'search_incomplete', 'unsupported_pddl', 'infrastructure_error', 'action_mapping_ambiguous'}
MAPPING_STOP = Event()


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


def read_plan(path):
    actions = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split(';', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'(?:\d+(?:\.\d+)?:\s*)?\(([^()]+)\)(?:\s*\[[\d.]+\])?', line)
        if not match:
            raise ValueError('Invalid plan line: ' + line)
        actions.append(tuple(match[1].lower().split()))
    return actions


def solve_files(directory, game):
    for path in [directory / 'plan.txt', *directory.glob('plan.txt.[0-9]*')]:
        path.unlink(missing_ok=True)
    cmd = [sys.executable, str(FD), '--overall-time-limit', '60s', '--overall-memory-limit', '2G',
           '--plan-file', str(directory / 'plan.txt'), '--sas-file', str(directory / 'output.sas')]
    inputs = [str(directory / 'domain.pddl'), str(directory / 'problem.pddl')]
    cmd += inputs + ['--search', 'astar(lmcut())'] if game in {'sliding_puzzle','numeric_sliding','rush_hour'} else ['--alias', 'lama-first', *inputs]
    code = 0
    try:
        stdout, stderr = _run_fast_downward(cmd, directory, timeout=65)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        code = getattr(error, 'returncode', None)
        stdout, stderr = error.stdout or '', error.stderr or ''
    (directory / 'solver.log').write_text(_output_text(stdout) + _output_text(stderr))
    plans = sorted(directory.glob('plan.txt.[0-9]*'), key=lambda p: int(p.suffix[1:]))
    if plans:
        (directory / 'plan.txt').write_bytes(plans[-1].read_bytes())
    found = (directory / 'plan.txt').is_file()
    for p in [directory / 'output.sas', *plans]: p.unlink(missing_ok=True)
    status = {10:'candidate_unsolvable', 11:'candidate_unsolvable', 12:'search_incomplete',
              20:'solver_memory_limit', 22:'solver_memory_limit', 21:'solver_timeout',
              23:'solver_timeout', 24:'solver_timeout', 31:'invalid_pddl', 33:'invalid_pddl',
              36:'invalid_pddl', 34:'unsupported_pddl', 37:'unsupported_pddl'}.get(code, 'solver_error')
    return {'solved':found, 'returncode':code, 'status':'solved' if found else 'solver_timeout' if code is None else status}


def validate_files(domain, problem, plan, log_path):
    run = subprocess.run([str(VAL), '-v', str(domain), str(problem), str(plan)], capture_output=True, text=True, timeout=30)
    log_path.write_text(run.stdout + run.stderr)
    return run.returncode == 0 and 'Plan valid' in run.stdout


def machine(domain, problem):
    validate_untyped_pddl(domain); validate_untyped_pddl(problem)
    d, p = parse_sexpr(domain), parse_sexpr(problem)
    operators = {}
    for a in d:
        if isinstance(a, list) and a[:1] == [':action']:
            if a[1] in operators: raise ValueError('Duplicate action: ' + a[1])
            operators[a[1]] = fields(a)
    initial = section(p, ':init')[1:]
    if any(not isinstance(a, list) or any(not isinstance(x, str) for x in a) for a in initial):
        raise ValueError('Initial state must contain positive Boolean ground facts')
    return {'objects':sorted(set(section(p, ':objects', [':objects'])[1:] + section(d, ':constants', [':constants'])[1:])),
            'init':{tuple(a) for a in initial}, 'goal':section(p, ':goal')[1], 'ops':operators}


def step(model, state, act):
    if act[0] not in model['ops']: raise ValueError('Unknown action: ' + act[0])
    op = model['ops'][act[0]]
    if len(act)-1 != len(op[':parameters']) or not set(act[1:]) <= set(model['objects']):
        raise ValueError('Invalid action arguments: ' + ' '.join(act))
    binding = dict(zip(op[':parameters'], act[1:]))
    if not condition(op[':precondition'], state, binding, model['objects']):
        raise ValueError('Precondition fails: ' + ' '.join(act))
    add, delete = set(), set()
    apply_effect(op[':effect'], state, binding, model['objects'], add, delete)
    return (state - delete) | add


def execute(model, actions):
    state = model['init']; trace = [state]
    for index, act in enumerate(actions, 1):
        try: state = step(model, state, act)
        except ValueError as error:
            return {'executable':False, 'goal_reached':False, 'failed_step':index,
                    'action':list(act), 'error':str(error), 'trace':trace}
        trace.append(state)
    return {'executable':True, 'goal_reached':condition(model['goal'], state, {}, model['objects']), 'trace':trace}


def load_tasks(manifest=MANIFEST, selected_datasets=None):
    manifest = Path(manifest)
    source = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    inputs = json.loads((ROOT/'tasks/instructions/instructions_abstract_planning_test.json').read_text())
    steps = json.loads((ROOT/'tasks/steps/steps_abstract_planning_test.json').read_text())
    expected_keys = {d:{r['task'] for r in source if r['dataset']==d} for d in ABSTRACT_SPLITS}
    if set(inputs) != set(steps) or set(inputs) != set(expected_keys):
        raise ValueError('Instruction/steps split keys differ from manifest')
    for dataset, keys in expected_keys.items():
        if set(inputs[dataset]) != keys or set(steps[dataset]) != keys:
            raise ValueError('Instruction/steps task keys differ from manifest: ' + dataset)
    if len({r['task'] for r in source}) != len(source):
        raise ValueError('Task IDs must be globally unique')
    audit_path = manifest.parent/'training_audit.json'
    trained_operations = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        trained_operations = audit.get('operations', {})
        records = json.loads(Path(audit['training_json']).read_text())
        trained = Counter(Path(r['images'][0]).parents[1].name for r in records)
        del records
        if trained != Counter(audit['counts']):
            raise ValueError('Training game counts changed; rebuild abstract planning tests')
        for dataset in ['abs_a_id','abs_a_scale']:
            covered = {r['game'] for r in source if r['dataset']==dataset}
            if covered != set(trained): raise ValueError('Incomplete training-game coverage: ' + dataset)
        if any(r['game'] in trained for r in source if r['dataset'] in {'abs_b_base','abs_b_rule'}):
            raise ValueError('B split contains a trained game')
    if selected_datasets is not None and set(selected_datasets)-set(expected_keys):
        raise ValueError('Unknown abstract split')
    rows = [r.copy() for r in source if selected_datasets is None or r['dataset'] in selected_datasets]
    if not rows: raise ValueError('No selected tasks')
    for row in rows:
        dataset = row['dataset']; row['id'] = task_id(row)
        row['split'] = SPLIT_LABELS[dataset]
        prompt = inputs[dataset][row['task']]
        image = ROOT/'tasks/images/abstract_planning_test'/(row['task']+'.png')
        if prompt != row['instruction'] or Path(row['image']).resolve() != image.resolve() or not image.is_file():
            raise ValueError('Input differs from manifest: ' + row['id'])
        reference_steps = steps[dataset][row['task']]; gt = Path(row['gt'])
        if reference_steps != row['steps'] or '\n'.join(reference_steps)+'\n' != (gt/'kf_actions.txt').read_text():
            raise ValueError('Steps differ from reference: ' + row['id'])
        scenario = json.loads((Path(row['metadata'])/'scenario.json').read_text())
        if any((gt/'round1'/(name+'.pddl')).read_text().strip() != scenario[name].strip() for name in ['domain','problem']):
            raise ValueError('Reference differs from scenario: ' + row['id'])
        if (Path(row['metadata'])/'instruction.txt').read_text() != prompt or scenario['instruction'] != prompt:
            raise ValueError('Instruction differs from scenario: ' + row['id'])
        row['instruction'] = names_only(prompt,scenario['domain'])
        row['_scenario'] = scenario; row['_oracle'] = machine(scenario['domain'],scenario['problem'])
        if dataset in {'abs_a_id','abs_a_scale'} and scenario['game'] in trained_operations:
            unseen = set(row['_oracle']['ops']) - set(trained_operations[scenario['game']])
            if unseen:
                raise ValueError('A split contains untrained operations: ' + row['id'] + ' ' + ', '.join(sorted(unseen)))
        row['_reference_plan'] = read_plan(gt/'round1/plan.txt')
        if len(row['_reference_plan']) != row['plan_length'] or len(reference_steps) != row['plan_length']:
            raise ValueError('Plan length differs from steps: ' + row['id'])
        for i,action in enumerate(row['_reference_plan'],1):
            if not reference_steps[i-1].startswith(f'{i}. {action[0]}('+', '.join(action[1:])+'). Result: '):
                raise ValueError('Plan action differs from steps: ' + row['id'])
        verify_plan(scenario,row['_reference_plan'])
    by_id = {task_id(r):r for r in source}
    for row in rows:
        if pair := row.get('paired_with'):
            paired = by_id[task_id(pair)]
            if paired['game'] != row['game'] or Path(paired['image']).read_bytes() != Path(row['image']).read_bytes():
                raise ValueError('Rule pair differs in game or image: ' + row['id'])
    return rows


def validate_mapping(value, source, targets, fixed):
    if not isinstance(value,dict) or set(value) != {'objects'} or not isinstance(value['objects'],dict):
        raise ValueError('Mapping must contain only an objects dictionary')
    aliases = value['objects']
    if set(aliases) != set(source)-set(fixed): raise ValueError('Incomplete/extra mapping keys')
    mapping = dict(fixed)
    for name,target in aliases.items():
        if target is None: continue
        if not isinstance(target,str) or target not in targets: raise ValueError('Unknown mapping target')
        if target in mapping.values(): raise ValueError('Object mapping must be one-to-one')
        mapping[name] = target
    return mapping


def align_objects(row, candidate, directory):
    source, target = set(candidate['objects']), set(row['_oracle']['objects'])
    fixed = {x:x for x in source & target}
    identity = {'model':mapping_model, 'candidate':[(directory/n).read_text() for n in ['domain.pddl','problem.pddl']],
                'image':file_revision(row['image']), 'oracle':row['_scenario']['problem'], 'instruction':row['instruction']}
    evidence = {'source':'identity', 'objects':fixed}
    if source-set(fixed):
        path = directory/'mapping.json'; saved = {}; last_error = None
        try:
            if path.exists():
                saved = json.loads(path.read_text())
                if not isinstance(saved,dict): raise ValueError('Mapping cache must be a JSON object')
        except (OSError,ValueError) as error:
            saved = {}; last_error = f'{type(error).__name__}: {error}'
        use_cache = saved.get('identity') == identity and 'response' in saved and not saved.get('error')
        data = {'instruction':row['instruction'],'candidate_objects':sorted(source-set(fixed)),
                'eligible_visible_ids':sorted(target-set(fixed.values())), 'fixed_identity':fixed,
                'candidate_initial_facts':[list(a) for a in sorted(candidate['init'])]}
        prompt = ('Map candidate aliases to visible scene IDs or numeric symbols defined by the rules. '
                  'Use the image, labels and initial facts; preserve fixed identities. Each target may be used once. '
                  'Use null for unsupported or ambiguous identities. Return only JSON {"objects":{"candidate_alias":"visible_id_or_null"}}.\n'
                  + json.dumps(data,ensure_ascii=False))
        attempts = 0
        while True:
            if MAPPING_STOP.is_set(): raise RuntimeError('Object mapping cancelled before validation')
            response, capture = None, {}
            try:
                if use_cache:
                    response, capture = saved['response'], saved.get('capture',{})
                else:
                    attempts += 1
                    feedback = (f'\nMapping attempt {attempts}: correct the previous JSON/schema error: {last_error[:500]}'
                                if last_error else '')
                    raw = call_gpt(mapping_model,prompt+feedback,[Path(row['image'])],
                                   response_format={'type':'json_object'},temperature=0,capture=capture)
                    capture['raw_output'] = raw
                    response = json.loads(raw)
                aligned = validate_mapping(response,source,target,fixed)
            except Exception as error:
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
    evidence['identity'] = identity; evidence['unmapped'] = sorted(source-set(evidence['objects']))
    write_json(directory/'mapping.json',evidence)
    return evidence


def evaluate_candidate(row, raw_output, directory):
    directory.mkdir(parents=True,exist_ok=True)
    for name in ['plan.txt','canonical_plan.txt','candidate_val.log','reference_val.log','environment_trace.json','probes.json','solver.log','domain.pddl','problem.pddl','action_mapping.json','postprocess.json','result.json','result.md']:
        (directory/name).unlink(missing_ok=True)
    result = {k:row[k] for k in ['id','dataset','split','game','task']}
    result.update(task_success=False,status='invalid_output')
    try:
        domain,problem = parse_pddl_output(raw_output)
        # 保留修复前的输入；无法确定修复方式时仍能查看原始证据。
        (directory/'domain.pddl').write_text(domain+'\n'); (directory/'problem.pddl').write_text(problem+'\n')
        domain,problem,changes = repair_pddl(domain,problem)
        (directory/'domain.pddl').write_text(domain+'\n'); (directory/'problem.pddl').write_text(problem+'\n')
        if changes:
            write_json(directory/'postprocess.json',changes)
        candidate = machine(domain,problem)
    except (ValueError,KeyError,TypeError,IndexError,NotImplementedError) as error:
        (directory/'mapping.json').unlink(missing_ok=True)
        result['error'] = str(error); return result
    expected = {name:len(op[':parameters']) for name,op in row['_oracle']['ops'].items()}
    actual = {name:len(op[':parameters']) for name,op in candidate['ops'].items()}
    result['interface_ok'] = bool(actual) and set(actual) <= set(expected)
    result['missing_operations'] = sorted(set(expected)-set(actual))
    if not result['interface_ok']:
        result.update(status='interface_mismatch',error='Unknown or missing action names');return result
    result['solver'] = solve_files(directory,row['game'])
    if not result['solver']['solved']:
        result['status'] = result['solver']['status']; result['error'] = '\n'.join((directory/'solver.log').read_text().splitlines()[-12:]); return result
    plan = read_plan(directory/'plan.txt'); result['plan_length'] = len(plan)
    own = validate_files(directory/'domain.pddl',directory/'problem.pddl',directory/'plan.txt',directory/'candidate_val.log')
    result['candidate_val_pass'] = own
    if not own: result['status'] = 'candidate_plan_invalid'; return result
    try:
        alignment = align_objects(row,candidate,directory); mapping = alignment['objects']
        required = {x for a in plan for x in a[1:]}
        if required-set(mapping):
            result.update(status='mapping_incomplete',error='Unmapped plan objects: '+', '.join(sorted(required-set(mapping)))); return result
    except Exception as error:
        result.update(status='mapping_error',error=f'{type(error).__name__}: {error}'); return result
    result['mapping_source'] = alignment['source']
    try:
        adapters = align_actions(candidate,row['_oracle'],mapping)
        canonical = [to_reference(a,mapping,adapters) for a in plan]
        write_json(directory/'action_mapping.json',{'source':'predicate extensions and action structure; no parameter-name or goal matching','operations':adapters})
    except AmbiguousActionMapping as error:
        result.update(status='action_mapping_ambiguous',error=str(error));return result
    (directory/'canonical_plan.txt').write_text('\n'.join('('+' '.join(a)+')' for a in canonical)+'\n')
    oracle = execute(row['_oracle'],canonical)
    reference = validate_files(Path(row['gt'])/'round1/domain.pddl',Path(row['gt'])/'round1/problem.pddl',directory/'canonical_plan.txt',directory/'reference_val.log')
    result['reference_val_pass'] = reference; result['environment'] = {k:v for k,v in oracle.items() if k != 'trace'}
    write_json(directory/'environment_trace.json',[[list(a) for a in sorted(s)] for s in oracle['trace']])
    independent = False
    if oracle['executable'] and oracle['goal_reached']:
        try: verify_plan(row['_scenario'],canonical); independent = True
        except (ValueError, AssertionError): pass
    result['independent_game_pass'] = independent
    if independent != (oracle['executable'] and oracle['goal_reached']):
        result.update(status='infrastructure_error',error='Independent rules disagree with reference PDDL'); return result
    if reference != (oracle['executable'] and oracle['goal_reached']):
        result.update(status='infrastructure_error',error='VAL and reference interpreter disagree'); return result
    result['task_success'] = result['interface_ok'] and own and reference
    result['status'] = ('success' if result['task_success'] else 'interface_mismatch' if not result['interface_ok']
                        else 'illegal_environment_action' if not oracle['executable'] else 'real_goal_not_reached')
    return result


def save_result(directory,result):
    write_json(directory/'result.json',result)
    lines = [f"任务：{result['id']}", f"游戏：{result['game']}", f"状态：{result['status']}",
             f"真实任务完成：{'是' if result['task_success'] else '否'}"]
    if proof := result.get('diagnosis'):
        lines.append('错误：'+proof['evidence'])
    elif result.get('error'):
        lines.append('错误：'+result['error'])
    links = [f'[{name}]({name})' for name in ['raw_output.txt','domain.pddl','problem.pddl','postprocess.json','plan.txt','mapping.json','canonical_plan.txt','result.json'] if (directory/name).exists()]
    (directory/'result.md').write_text('\n\n'.join(lines+[' · '.join(links)])+'\n')


def diagnose_result(result,row,directory):
    """Reuse saved models, mappings and this task's plan; never call a model or solver."""
    if result['task_success'] or result['status'] in UNRESOLVED:
        return None
    try:
        problem = (directory/'problem.pddl').read_text()
        if conflict := initial_conflict(problem):
            return conflict
        if result['status'] not in {'interface_mismatch','illegal_environment_action','real_goal_not_reached'}:
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
        if source-set(mapping):
            saved = json.loads((directory/'mapping.json').read_text())
            if saved.get('error') or saved.get('identity',{}).get('candidate') != [domain,problem]:
                return None
            mapping = validate_mapping(saved['response'],source,targets,mapping)
        return diagnose_failure(result,candidate,reference,mapping,read_plan(directory/'plan.txt'),
                                read_plan(directory/'canonical_plan.txt'),execute)
    except (OSError,ValueError,KeyError,TypeError,IndexError,NotImplementedError):
        return None


def run_one(row,template,code_revision):
    directory = eval_root/row['dataset']/row['task']; directory.mkdir(parents=True,exist_ok=True)
    prompt = template.format(instruction=names_only(row['instruction'],row['_scenario']['domain']),robot_configuration='single_arm')
    identity = {'model':eval_model,'prompt':prompt,'image':file_revision(row['image'])}
    start = time.monotonic(); stage = 'generation_error'
    try:
        saved = json.loads((directory/'generation.json').read_text()) if (directory/'generation.json').exists() else {}
        raw_path = directory/'raw_output.txt'
        if saved.get('identity') == identity and raw_path.exists() and saved.get('output') == raw_path.read_text(): raw = raw_path.read_text()
        else:
            for name in ['raw_output.txt','generation.json','domain.pddl','problem.pddl','plan.txt','canonical_plan.txt','mapping.json','candidate_val.log','reference_val.log','environment_trace.json','probes.json','solver.log','postprocess.json']:
                (directory/name).unlink(missing_ok=True)
            capture = {}; raw = call_gpt(eval_model,prompt,[Path(row['image'])],temperature=0,capture=capture)
            raw_path.write_text(raw); write_json(directory/'generation.json',{'identity':identity,'output':raw,'capture':capture})
        stage = 'infrastructure_error'; result = evaluate_candidate(row,raw,directory)
    except Exception as error:
        result = {k:row[k] for k in ['id','dataset','split','game','task']}
        result.update(task_success=False,status=stage,error=f'{type(error).__name__}: {error}')
    result.update(model=eval_model,elapsed_seconds=round(time.monotonic()-start,3),evaluator_revision=code_revision)
    if not result['task_success']: result['diagnosis'] = diagnose_result(result,row,directory)
    save_result(directory,result); return result


def aggregate(results):
    evaluated = [r for r in results if r['status'] not in UNRESOLVED]
    successes = sum(r['task_success'] for r in evaluated)
    errors = Counter(diagnostic_group(r,UNRESOLVED) for r in evaluated if not r['task_success'])
    return {'total':len(results),'evaluated':len(evaluated),'unresolved':len(results)-len(evaluated),
            'task_success':successes,'task_success_rate':successes/len(evaluated) if evaluated else None,
            'error_counts':{key:errors[key] for key in ERROR_TYPES},
            'error_rates':{key:errors[key]/len(evaluated) if evaluated else None for key in ERROR_TYPES},
            'statuses':dict(Counter(r['status'] for r in results))}


def summarize(results,rows):
    summary = {'model':eval_model,'mapping_model':mapping_model,'manifest':file_revision(MANIFEST),
               'overall':aggregate(results),'by_split':{},'by_game':{},'pairs':[], 'decision_source':'program; VLM supplies object identity aliases only'}
    audit_path = MANIFEST.parent/'training_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        summary['training'] = {key:audit[key] for key in ['training_json','records','counts','ranges','variant_of']}
    summary['test_version'] = MANIFEST.parent.name
    for field,label in [('split','by_split'),('game','by_game')]:
        groups = defaultdict(list)
        for result in results: groups[result[field]].append(result)
        summary[label] = {key:aggregate(group) for key,group in groups.items()}
        if field == 'split':
            for key,group in groups.items():
                games = {r['game'] for r in group}
                summary[label][key]['by_game'] = {g:aggregate([r for r in group if r['game']==g]) for g in sorted(games)}
                rates = [aggregate([r for r in group if r['game']==g])['task_success_rate'] for g in games]
                rates = [rate for rate in rates if rate is not None]
                summary[label][key]['macro_game_success_rate'] = sum(rates)/len(rates) if rates else None
    macro = [x['task_success_rate'] for x in summary['by_game'].values() if x['task_success_rate'] is not None]
    summary['macro_game_success_rate'] = sum(macro)/len(macro) if macro else None
    by_id = {r['id']:r for r in results}
    for row in rows:
        if row['dataset'] != 'abs_b_rule' or not row.get('paired_with'): continue
        pair = row['paired_with']; base_id = task_id(pair)
        if base_id not in by_id: continue
        a,b = by_id[base_id],by_id[row['id']]
        summary['pairs'].append({'base':base_id,'variant':row['id'],
                                 'evaluated':a['status'] not in UNRESOLVED and b['status'] not in UNRESOLVED,
                                 'both_tasks_pass':a['task_success'] and b['task_success']})
    evaluated_pairs = [p for p in summary['pairs'] if p['evaluated']]
    summary['pairs_evaluated'] = len(evaluated_pairs)
    summary['pairs_unresolved'] = len(summary['pairs'])-len(evaluated_pairs)
    summary['pair_success_rate'] = sum(p['both_tasks_pass'] for p in evaluated_pairs)/len(evaluated_pairs) if evaluated_pairs else None
    write_json(eval_root/'summary_abs.json',summary)
    (eval_root/'report_abs.md').write_text(render_abstract_report(results,summary,eval_root,UNRESOLVED))
    return summary


def load_saved_results(rows):
    """Load decisions and derive local diagnoses without generation or solving."""
    results = []
    for row in rows:
        identity = {key:row[key] for key in ['dataset','game','task']}
        identity.update(id=task_id(row),split=SPLIT_LABELS[row['dataset']])
        try:
            result = json.loads((eval_root/row['dataset']/row['task']/'result.json').read_text())
            if not isinstance(result,dict) or any(result.get(key) != value for key,value in identity.items()):
                raise ValueError('Result identity differs from manifest')
            if not isinstance(result.get('status'),str) or type(result.get('task_success')) is not bool:
                raise ValueError('Result is missing a decision')
        except (OSError,ValueError,KeyError,TypeError) as error:
            result = {**identity,'task_success':False,'status':'infrastructure_error',
                      'error':f'结果文件无法读取：{type(error).__name__}: {error}'}
        if not result['task_success']:
            result['diagnosis'] = diagnose_result(result,row,eval_root/row['dataset']/row['task'])
        results.append(result)
    return results


def main(*,report_only=False):
    eval_root.mkdir(parents=True,exist_ok=True)
    if report_only:
        rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
        rows = [row for row in rows if row['dataset'] in datasets]
        for row in rows: row['id'] = task_id(row)
        results = load_saved_results(rows)
    else:
        MAPPING_STOP.clear()
        rows = load_tasks(MANIFEST,datasets); template = PROMPT.read_text()
        code_revision = [file_revision(p) for p in [__file__, MANIFEST, novel_games.__file__]]
        print(f'开始评测：{eval_model}，{len(rows)} 条；映射模型 {mapping_model}。',flush=True)
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            try:
                for row in rows: futures.append(pool.submit(run_one,row,template,code_revision))
                for future in as_completed(futures): results.append(future.result())
            except BaseException:
                MAPPING_STOP.set()
                for future in futures: future.cancel()
                raise
    summary = summarize(results,rows)
    overall = summary['overall']
    print(f"{eval_model}：任务成功 {overall['task_success']}/{overall['evaluated']}，未完成 {overall['unresolved']}；报告：{eval_root / 'report_abs.md'}",flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate abstract PDDL or rebuild compact reports from cached results.')
    parser.add_argument('--report-only',action='store_true',help='rebuild reports and local diagnoses without model calls or solving')
    parser.add_argument('--report-root',type=Path,help='saved abstract results directory; requires --report-only')
    args = parser.parse_args()
    if args.report_root:
        if not args.report_only: parser.error('--report-root requires --report-only')
        eval_root = args.report_root.resolve()
        eval_model = eval_root.parent.name
    main(report_only=args.report_only)
