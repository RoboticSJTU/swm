#!/usr/bin/env python3
"""Generate PDDL, align identities with a VLM, then evaluate with fixed rules."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from swm.llm import call_gpt, call_gpt_json
from swm.pddl.generation import parse_pddl_output
from swm.pddl.planner import _run_fast_downward, fast_downward_path as FD, _output_text
from swm.pddl.strips import validate_untyped_pddl
from swm.abstract_planning.model import parse_sexpr, section, fields, condition, apply_effect
from swm.abstract_planning import novel_games, visual_games
from swm.abstract_planning.instructions import names_only
from swm.abstract_planning.test_contract import DATASETS as ABSTRACT_SPLITS, task_id, verify_plan
from swm.abstract_planning.action_alignment import align_actions, to_reference, to_candidate, AmbiguousActionMapping

# 固定路径；只保留模型、数据集、并发这些实验配置。
eval_model = '9B_full_106'
mapping_model = 'Qwen3.8-27B'
datasets = ['abs_a_id', 'abs_a_scale', 'abs_b_base', 'abs_b_rule']
max_workers = 200
MANIFEST = ROOT / 'tasks/meta/abstract_planning_test/manifest.jsonl'
PROMPT = ROOT / 'src/swm/prompt_templates/training_input_abstract.txt'
VAL = ROOT / 'downloads/visual_pddl_v1/VAL/build/bin/Validate'
eval_root = ROOT / 'eval_results' / eval_model / MANIFEST.parent.name
UNRESOLVED = {'generation_error', 'mapping_error', 'mapping_incomplete', 'solver_timeout',
              'solver_memory_limit', 'search_incomplete', 'unsupported_pddl', 'infrastructure_error', 'action_mapping_ambiguous'}


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


def distance(a, b):
    if re.fullmatch(r'r\d+c\d+', a) and re.fullmatch(r'r\d+c\d+', b):
        return (0, abs(int(a[1:3])-int(b[1:3])) + abs(int(a[4:6])-int(b[4:6])), b)
    prefix = lambda x: re.sub(r'\d+$', '', x)
    return (int(prefix(a) != prefix(b)), 0, b)


def prepare_probes(row):
    oracle, reference = row['_oracle'], row['_reference_plan']
    trace = execute(oracle, reference)['trace']
    probes, seen, failure_patterns = [], set(), set()
    def add(kind, actions):
        key = tuple(tuple(a) for a in actions)
        if (kind, key) in seen: return
        expected = execute(oracle, key)
        probes.append({'kind':kind, 'actions':[list(a) for a in key], **{k:v for k,v in expected.items() if k != 'trace'}})
        seen.add((kind,key))
    add('initial_goal', []); add('reference_plan', reference)
    indices = defaultdict(list)
    for i, act in enumerate(reference): indices[act[0]].append(i)
    for occurrences in indices.values():
        for i in occurrences:
            act, state = reference[i], trace[i]
            representative = i in {occurrences[0],occurrences[-1]}
            if representative: add('legal_action', reference[:i+1])
            op = oracle['ops'][act[0]]; pre = op[':precondition']
            parts = pre[1:] if pre[0] == 'and' else [pre]
            for position in range(1, len(act)):
                alternatives = []
                for other in oracle['objects']:
                    if other == act[position]: continue
                    changed = (*act[:position], other, *act[position+1:])
                    binding = dict(zip(op[':parameters'], changed[1:]))
                    violations = tuple(j for j,part in enumerate(parts) if not condition(part, state, binding, oracle['objects']))
                    alternatives.append((len(violations), distance(act[position],other), violations, changed))
                alternatives.sort(key=lambda item:(item[0],item[1]))
                legal = next((a[-1] for a in alternatives if a[0] == 0), None)
                if legal and representative: add('alternative_legal_action', [*reference[:i],legal])
                masks = set()
                for count, _, mask, changed in alternatives:
                    if not count or mask in masks: continue
                    pattern = (act[0],position,mask)
                    if pattern not in failure_patterns:
                        add('illegal_action', [*reference[:i],changed])
                        failure_patterns.add(pattern)
                    masks.add(mask)
                    if len(masks) == 2: break
    if row.get('_paired_reference'): add('rule_difference', row['_paired_reference'])
    return probes


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
        row['split'] = {'abs_a_id':'A-ID','abs_a_scale':'A-scale','abs_b_base':'B-base','abs_b_rule':'B-rule'}[dataset]
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
            row['_paired_reference'] = read_plan(Path(paired['gt'])/'round1/plan.txt')
        row['_probes'] = prepare_probes(row)
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
    capture, evidence = {}, {'source':'identity', 'objects':fixed}
    if source-set(fixed):
        path = directory/'mapping.json'; saved = json.loads(path.read_text()) if path.exists() else {}
        if saved.get('identity') == identity and 'response' in saved and not saved.get('error'):
            response = saved['response']; capture = saved.get('capture',{})
        else:
            data = {'instruction':row['instruction'],'candidate_objects':sorted(source-set(fixed)),
                    'eligible_visible_ids':sorted(target-set(fixed.values())), 'fixed_identity':fixed,
                    'candidate_initial_facts':[list(a) for a in sorted(candidate['init'])]}
            prompt = ('Map candidate aliases to visible scene IDs or numeric symbols defined by the rules. '
                      'Use the image, labels and initial facts; preserve fixed identities. Each target may be used once. '
                      'Use null for unsupported or ambiguous identities. Return only JSON {"objects":{"candidate_alias":"visible_id_or_null"}}.\n'
                      + json.dumps(data,ensure_ascii=False))
            try:
                response = call_gpt_json(mapping_model,prompt,[Path(row['image'])],response_format={'type':'json_object'},attempts=1,temperature=0,capture=capture)
            except Exception as error:
                write_json(path,{'source':'vlm','identity':identity,'objects':fixed,'capture':capture,'error':f'{type(error).__name__}: {error}'})
                raise
        try:
            aligned = validate_mapping(response,source,target,fixed)
        except ValueError as error:
            write_json(path,{'source':'vlm','identity':identity,'objects':fixed,'response':response,'capture':capture,'error':str(error)})
            raise
        evidence = {'source':'vlm','objects':aligned,'response':response,'capture':capture}
    evidence['identity'] = identity; evidence['unmapped'] = sorted(source-set(evidence['objects']))
    write_json(directory/'mapping.json',evidence)
    return evidence


def evaluate_probes(candidate, probes, mapping, adapters):
    inverse = {v:k for k,v in mapping.items()}; cache = {():{'executable':True,'state':candidate['init']}}
    results = []
    for probe in probes:
        prefix = (); actual = cache[prefix]
        for index,reference_act in enumerate(probe['actions'],1):
            try: act = to_candidate(reference_act,inverse,adapters)
            except KeyError:
                actual = {'executable':False,'failed_step':index,'error':'Unmapped probe object'}; break
            except ValueError as error:
                actual = {'executable':False,'failed_step':index,'error':str(error)}; break
            previous, prefix = cache[prefix], (*prefix,act)
            if prefix not in cache:
                if not previous['executable']: cache[prefix] = previous
                else:
                    try: cache[prefix] = {'executable':True,'state':step(candidate,previous['state'],act)}
                    except ValueError as error: cache[prefix] = {'executable':False,'failed_step':index,'error':str(error)}
            actual = cache[prefix]
            if not actual['executable']:break
        reached = actual['executable'] and condition(candidate['goal'],actual['state'],{},candidate['objects'])
        passed = actual['executable'] == probe['executable'] and reached == probe['goal_reached']
        if not probe['executable']: passed = passed and actual.get('failed_step') == probe['failed_step']
        results.append({'kind':probe['kind'],'pass':passed, 'expected':{k:probe[k] for k in ['executable','goal_reached']},
                        'actual':{'executable':actual['executable'],'goal_reached':reached,
                                  **{k:actual[k] for k in ['failed_step','error'] if k in actual}}})
    return {'pass':bool(results) and all(p['pass'] for p in results), 'passed':sum(p['pass'] for p in results),
            'total':len(results),'results':results,'scope':'Finite behavioral probes, not complete domain equivalence.'}


def evaluate_candidate(row, raw_output, directory):
    directory.mkdir(parents=True,exist_ok=True)
    for name in ['plan.txt','canonical_plan.txt','candidate_val.log','reference_val.log','environment_trace.json','probes.json','solver.log','domain.pddl','problem.pddl','action_mapping.json']:
        (directory/name).unlink(missing_ok=True)
    result = {k:row[k] for k in ['id','dataset','split','game','task']}
    result.update(task_success=False,behavior_success=False,status='invalid_output',probes={'pass':False,'passed':0,'total':len(row['_probes']),'results':[]})
    try:
        domain,problem = parse_pddl_output(raw_output)
        (directory/'domain.pddl').write_text(domain+'\n'); (directory/'problem.pddl').write_text(problem+'\n')
        candidate = machine(domain,problem)
    except (ValueError,KeyError,TypeError,IndexError) as error:
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
    result['probes'] = evaluate_probes(candidate,row['_probes'],mapping,adapters)
    write_json(directory/'probes.json',{'definitions':row['_probes'],**result['probes']})
    result['task_success'] = result['interface_ok'] and own and reference
    result['behavior_success'] = result['task_success'] and result['probes']['pass']
    result['status'] = ('success' if result['task_success'] else 'interface_mismatch' if not result['interface_ok']
                        else 'illegal_environment_action' if not oracle['executable'] else 'real_goal_not_reached')
    return result


def save_result(directory,result):
    write_json(directory/'result.json',result); probes = result['probes']
    lines = [f"任务：{result['id']}", f"游戏：{result['game']}", f"状态：{result['status']}",
             f"真实任务完成：{'是' if result['task_success'] else '否'}", f"规则探针：{probes['passed']}/{probes['total']}（有限覆盖）"]
    if result.get('error'): lines.append('错误：'+result['error'])
    links = [f'[{name}]({name})' for name in ['raw_output.txt','domain.pddl','problem.pddl','plan.txt','mapping.json','canonical_plan.txt','probes.json','result.json'] if (directory/name).exists()]
    (directory/'result.md').write_text('\n\n'.join(lines+[' · '.join(links)])+'\n')


def run_one(row,template,code_revision):
    directory = eval_root/row['dataset']/row['task']; directory.mkdir(parents=True,exist_ok=True)
    prompt = template.format(instruction=names_only(row['instruction'],row['_scenario']['domain']),robot_configuration='single-arm')
    identity = {'model':eval_model,'prompt':prompt,'image':file_revision(row['image'])}
    start = time.monotonic(); stage = 'generation_error'
    try:
        saved = json.loads((directory/'generation.json').read_text()) if (directory/'generation.json').exists() else {}
        raw_path = directory/'raw_output.txt'
        if saved.get('identity') == identity and raw_path.exists() and saved.get('output') == raw_path.read_text(): raw = raw_path.read_text()
        else:
            for name in ['raw_output.txt','generation.json','domain.pddl','problem.pddl','plan.txt','canonical_plan.txt','mapping.json','candidate_val.log','reference_val.log','environment_trace.json','probes.json','solver.log']:
                (directory/name).unlink(missing_ok=True)
            capture = {}; raw = call_gpt(eval_model,prompt,[Path(row['image'])],temperature=0,capture=capture)
            raw_path.write_text(raw); write_json(directory/'generation.json',{'identity':identity,'output':raw,'capture':capture})
        stage = 'infrastructure_error'; result = evaluate_candidate(row,raw,directory)
    except Exception as error:
        result = {k:row[k] for k in ['id','dataset','split','game','task']}
        result.update(task_success=False,behavior_success=False,status=stage,error=f'{type(error).__name__}: {error}',probes={'pass':False,'passed':0,'total':len(row['_probes']),'results':[]})
    result.update(model=eval_model,elapsed_seconds=round(time.monotonic()-start,3),evaluator_revision=code_revision)
    save_result(directory,result); return result


def aggregate(results):
    evaluated = [r for r in results if r['status'] not in UNRESOLVED]
    successes = sum(r['task_success'] for r in evaluated); behavior = sum(r['behavior_success'] for r in evaluated)
    return {'total':len(results),'evaluated':len(evaluated),'unresolved':len(results)-len(evaluated),
            'task_success':successes,'task_success_rate':successes/len(evaluated) if evaluated else None,
            'behavior_success':behavior,'behavior_success_rate':behavior/len(evaluated) if evaluated else None,
            'probe_passed':sum(r['probes']['passed'] for r in results),'probe_total':sum(r['probes']['total'] for r in results),
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
                                 'both_tasks_pass':a['task_success'] and b['task_success'], 'both_behaviors_pass':a['behavior_success'] and b['behavior_success']})
    evaluated_pairs = [p for p in summary['pairs'] if p['evaluated']]
    summary['pairs_evaluated'] = len(evaluated_pairs)
    summary['pairs_unresolved'] = len(summary['pairs'])-len(evaluated_pairs)
    summary['pair_success_rate'] = sum(p['both_tasks_pass'] for p in evaluated_pairs)/len(evaluated_pairs) if evaluated_pairs else None
    write_json(eval_root/'summary_abs.json',summary)
    lines = [f'# 抽象规划评测：{eval_model}', 'VLM 仅映射对象身份，程序判定任务与规则；API/映射/资源限制另列为未完成评测。',
             '| 分类 | 总数 | 已评测 | 未完成 | 真实任务成功 | 成功且规则探针全通过 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for split,s in summary['by_split'].items(): lines.append(f"| {split} | {s['total']} | {s['evaluated']} | {s['unresolved']} | {s['task_success']} | {s['behavior_success']} |")
    lines += ['', f"规则配对成功：{sum(p['both_tasks_pass'] for p in evaluated_pairs)}/{len(evaluated_pairs)}；未完成：{summary['pairs_unresolved']}", '', '| 任务 | 游戏 | 状态 | 任务完成 | 规则探针 |', '| --- | --- | --- | --- | --- |']
    for r in sorted(results,key=lambda r:r['id']):
        p=r['probes']; lines.append(f"| [{r['id']}]({r['id']}/result.md) | {r['game']} | {r['status']} | {'是' if r['task_success'] else '否'} | {p['passed']}/{p['total']} |")
    lines += ['', '规则探针只覆盖已检查行为；五个新增训练游戏和四个未训练游戏另以独立原生规则模拟器核验计划。']
    (eval_root/'report_abs.md').write_text('\n'.join(lines)+'\n'); return summary


def main():
    eval_root.mkdir(parents=True,exist_ok=True); rows = load_tasks(MANIFEST,datasets); template = PROMPT.read_text()
    code_revision = [file_revision(p) for p in [__file__, MANIFEST, novel_games.__file__]]
    print(f'开始评测：{eval_model}，{len(rows)} 条；映射模型 {mapping_model}。',flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_one,row,template,code_revision) for row in rows]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(f"{result['id']}: {result['status']} ({len(results)}/{len(rows)})",flush=True)
    summary = summarize(results,rows)
    print(json.dumps(summary['overall'],ensure_ascii=False,indent=2),flush=True)
    print(f'详细结果：  {eval_root / "report_abs.md"}',flush=True)


if __name__ == '__main__':
    main()
