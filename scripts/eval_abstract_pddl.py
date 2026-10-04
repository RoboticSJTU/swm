#!/usr/bin/env python3
"""Evaluate abstract-game PDDL with shared SWM model calls and planning utilities."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Use this checkout even when another SWM checkout is installed.
sys.path.insert(0, str(ROOT / 'src'))

from swm.llm import call_gpt
from swm.pddl.generation import parse_pddl_output
from swm.pddl.eval_report import ABSTRACT_STATUS_LABELS as STATUS_LABELS, render_abstract_report
from swm.pddl.planner import _run_fast_downward, fast_downward_path as FD
from swm.pddl.strips import parse_sexpr

DEFAULT_MANIFEST = ROOT / 'tasks/meta/abstract_planning_test_v1/manifest.jsonl'
PROMPT = ROOT / 'src/swm/prompt_templates/abstract_input.txt'
VAL = ROOT / 'downloads/visual_pddl_v1/VAL/build/bin/Validate'

# 只在这里修改关键配置，直接运行脚本即可。
eval_model = '9B_3e_full'
datasets = ['abs_a_id', 'abs_a_scale', 'abs_b_base', 'abs_b_rule']
generation_max_workers = 200
eval_root = ROOT / 'eval_results' / eval_model

DIRS = {'north': (-1, 0), 'east': (0, 1), 'south': (1, 0), 'west': (0, -1)}
OPPOSITE = {'north': 'south', 'east': 'west', 'south': 'north', 'west': 'east'}


def sha(value):
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


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


def solve_files(domain, problem, timeout=60):
    """Use one search policy for references and predictions; discard stale plans."""
    domain, problem = Path(domain).resolve(), Path(problem).resolve()
    directory = domain.parent
    for path in [directory / 'plan.txt', *directory.glob('plan.txt.[0-9]*')]:
        path.unlink(missing_ok=True)
    cmd = [sys.executable, str(FD), '--overall-time-limit', f'{timeout}s',
           '--overall-memory-limit', '2G', '--plan-file', str(directory / 'plan.txt'),
           '--sas-file', str(directory / 'output.sas'), '--alias', 'lama-first', str(domain), str(problem)]
    returncode, timed_out = 0, False
    try:
        stdout, stderr = _run_fast_downward(cmd, directory, timeout=timeout + 5)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        returncode = getattr(error, 'returncode', None)
        timed_out = isinstance(error, subprocess.TimeoutExpired)
        stdout, stderr = error.stdout or '', error.stderr or ''
    (directory / 'solver.log').write_text(stdout + stderr)
    plans = sorted(directory.glob('plan.txt.[0-9]*'), key=lambda p: int(p.name.rsplit('.', 1)[1]))
    if plans:
        (directory / 'plan.txt').write_bytes(plans[-1].read_bytes())
    found = (directory / 'plan.txt').is_file()
    return {'solved': found, 'timeout': timed_out or returncode in {21, 23, 24},
            'returncode': returncode, 'command': cmd}


def validate_files(domain, problem, plan):
    domain, problem, plan = (Path(path).resolve() for path in [domain, problem, plan])
    completed = subprocess.run([str(VAL), '-v', '-a', str(domain), str(problem), str(plan)],
                               cwd=plan.parent, capture_output=True, text=True, timeout=30)
    log = completed.stdout + completed.stderr
    return {'pass': 'Plan valid' in log and completed.returncode == 0,
            'returncode': completed.returncode, 'log': log}


def loc(r, c):
    return f'r{r + 1:02d}c{c + 1:02d}'


def rc(cell):
    match = re.fullmatch(r'r(\d+)c(\d+)', cell)
    if not match:
        raise ValueError('Invalid cell ID: ' + cell)
    return tuple(int(x) - 1 for x in match.groups())


def neighbor(cell, direction):
    r, c = rc(cell)
    dr, dc = DIRS[direction]
    return loc(r + dr, c + dc)


def legal_actions(spec, world):
    """Independent game rules: no PDDL operator or generated predicate is consulted."""
    game, actions = spec['domain'], []
    allowed = set(spec['allowed_actions'])
    positions = world.get('at', {})
    current = world.get('agent', positions.get('player'))
    edge_list = spec.get('edges', [])
    adjacent = [(b, d) for a, b, d in edge_list if a == current]
    if game == 'frozenlake_v1':
        actions = [(f'move_{d}', current, b) for b, d in adjacent if b in spec['safe']]
    elif game in {'maze_v1', 'package_v1', 'printer_v1'}:
        facing = world['facing']
        directions = list(DIRS)
        index = directions.index(facing)
        actions += [('turn_left', facing, directions[(index - 1) % 4]),
                    ('turn_right', facing, directions[(index + 1) % 4])]
        front = [(b, d) for b, d in adjacent if d == facing]
        occupied = set(positions.values()) if game != 'maze_v1' else set()
        actions += [('move_forward', current, b, d) for b, d in front if b not in occupied]
        if game == 'package_v1':
            actions += [('open_package', p, current, b, d) for b, d in front for p in spec['packages']
                        if positions[p] == b and p not in world['open']]
        if game == 'printer_v1':
            if world['holding'] is None:
                actions += [('pick_up_printer', p, current, b, d, 'hand1') for b, d in front
                            for p in spec['printers'] if positions.get(p) == b]
            for b, d in front:
                for table in spec['tables']:
                    if positions[table] != b:
                        continue
                    if world['holding'] is not None:
                        actions.append(('put_on_table', world['holding'], table, current, b, d, 'hand1'))
                    actions += [('power_on', p, table, current, b, d) for p in spec['printers']
                                if world['on'].get(p) == table and p not in world['powered']]
    elif game == 'sokoban_v1':
        occupied = set(positions.values())
        for b, direction in adjacent:
            d = {'north': 'dir-up', 'east': 'dir-right', 'south': 'dir-down', 'west': 'dir-left'}[direction]
            if b not in occupied:
                actions.append(('move', 'player', current, b, d))
            for stone in spec['stones']:
                if positions[stone] != b:
                    continue
                target = neighbor(b, direction)
                if [b, target, direction] in edge_list and target not in occupied:
                    name = 'push_to_goal' if target in spec['goal_cells'] else 'push_to_nongoal'
                    actions.append((name, 'player', stone, current, b, target, d))
    elif game == 'overcooked_v1':
        actions += [('move', current, b) for b, _ in adjacent]
        held = world['holding']
        if held is None:
            actions += [('pickup_ingredient', food, current, 'hand1') for food in ['tomato1', 'onion1']
                        if positions.get(food) == current]
            if world['ready'] and positions.get('plate1') == current:
                actions.append(('pickup_plate', 'plate1', current, 'hand1'))
        elif held in ['tomato1', 'onion1']:
            if current == spec['board'] and held not in world['chopped']:
                actions.append(('chop', held, current, 'hand1'))
            if held in world['chopped'] and positions.get('plate1') == current:
                actions.append(('put_on_plate', held, 'plate1', current, 'hand1'))
        elif held == 'plate1' and world['ready'] and current == spec['delivery']:
            actions.append(('deliver', 'plate1', current, 'hand1'))
        if set(world['on_plate']) == {'tomato1', 'onion1'}:
            actions.append(('combine_salad', 'tomato1', 'onion1', 'plate1'))
    elif game == 'pddlgym_blocks_medium':
        supports = world['supports']
        held = world['holding']
        clear = set(spec['blocks']) - set(supports.values()) - ({held} if held else set())
        if held is None:
            for block in sorted(clear):
                support = supports[block]
                actions.append(('pick_up', block, 'hand1') if support == 'table'
                               else ('unstack', block, support, 'hand1'))
        else:
            actions.append(('put_down', held, 'hand1'))
            actions += [('stack', held, block, 'hand1') for block in sorted(clear)]
    elif game == 'pddlgym_hanoi_operator_actions':
        supports = world['supports']
        clear = set(spec['disks'] + spec['pegs']) - set(supports.values())
        for disk in spec['disks']:
            if disk not in clear:
                continue
            for target in sorted(clear):
                if target in spec['pegs'] or int(target[1:]) > int(disk[1:]):
                    actions.append(('move', disk, supports[disk], target))
    elif game == 'pddlgym_slidetile':
        bx, by = world['blank']
        for tile, (x, y) in positions.items():
            if x == bx and abs(int(y[1:]) - int(by[1:])) == 1:
                name = 'move_down' if int(by[1:]) < int(y[1:]) else 'move_up'
                actions.append((name, tile, x, y, by))
            if y == by and abs(int(x[1:]) - int(bx[1:])) == 1:
                name = 'move_right' if int(bx[1:]) < int(x[1:]) else 'move_left'
                actions.append((name, tile, x, y, bx))
    elif game == 'pddlgym_tsp_operator_actions':
        actions += [('move', current, target) for target in spec['nodes']]
    elif game == 'phase_gates':
        phase = world['phase']
        red_phase = spec['rule_version']
        for b, _ in adjacent:
            needed = red_phase if b in spec['red'] else 1 - red_phase if b in spec['blue'] else None
            if needed is None or needed == phase:
                actions.append(('move', current, b))
        if current in spec['consoles']:
            actions.append(('toggle', current))
    elif game == 'energy_route':
        for b, _ in adjacent:
            cost = spec['rough_cost'] if b in spec['rough'] else 1
            if world['energy'] >= cost:
                actions.append(('move', current, b))
        if current in spec['chargers']:
            actions.append(('recharge', current))
    elif game == 'fragile_bridges':
        actions += [('move', current, b) for b, _ in adjacent if b not in world['collapsed']]
        actions += [('collect', marker, current) for marker, cell in spec['markers'].items()
                    if cell == current and marker not in world['collected']]
    elif game == 'coupled_tokens':
        edge_set = {tuple(e) for e in edge_list}
        for direction in DIRS:
            a, b = positions['a'], positions['b']
            aa, bb = neighbor(a, direction), neighbor(b, OPPOSITE[direction])
            if (a, aa, direction) in edge_set and (b, bb, OPPOSITE[direction]) in edge_set and aa != bb:
                actions.append((f'step_{direction}', a, aa, b, bb))
    else:
        raise ValueError('Unknown game: ' + game)
    return sorted({tuple(a) for a in actions if a[0] in allowed})


def transition(spec, world, action):
    action = tuple(action)
    if action not in legal_actions(spec, world):
        raise ValueError('Illegal environment action: ' + ' '.join(action))
    result = copy.deepcopy(world)
    game, name, args = spec['domain'], action[0], action[1:]
    if game == 'frozenlake_v1':
        result['at']['player'] = args[1]
    elif game in {'maze_v1', 'package_v1', 'printer_v1'}:
        if name.startswith('turn_'):
            result['facing'] = args[1]
        elif name == 'move_forward':
            result['agent'] = args[1]
            if game == 'maze_v1':
                result['at']['player'] = args[1]
        elif name == 'open_package':
            result['open'].append(args[0])
        elif name == 'pick_up_printer':
            result['holding'] = args[0]
            del result['at'][args[0]]
        elif name == 'put_on_table':
            result['on'][args[0]] = args[1]
            result['holding'] = None
        elif name == 'power_on':
            result['powered'].append(args[0])
    elif game == 'sokoban_v1':
        if name == 'move':
            result['at']['player'] = args[2]
        else:
            result['at']['player'], result['at'][args[1]] = args[3], args[4]
    elif game == 'overcooked_v1':
        if name == 'move':
            result['agent'] = args[1]
        elif name.startswith('pickup_'):
            result['holding'] = args[0]
            del result['at'][args[0]]
        elif name == 'chop':
            result['chopped'].append(args[0])
        elif name == 'put_on_plate':
            result['on_plate'].append(args[0])
            result['holding'] = None
        elif name == 'combine_salad':
            result['ready'] = True
        elif name == 'deliver':
            result['delivered'] = True
            result['holding'] = None
    elif game == 'pddlgym_blocks_medium':
        block = args[0]
        if name in {'pick_up', 'unstack'}:
            del result['supports'][block]
            result['holding'] = block
        else:
            result['supports'][block] = 'table' if name == 'put_down' else args[1]
            result['holding'] = None
    elif game == 'pddlgym_hanoi_operator_actions':
        result['supports'][args[0]] = args[2]
    elif game == 'pddlgym_slidetile':
        old = result['at'][args[0]]
        result['at'][args[0]], result['blank'] = result['blank'], old
    elif game == 'pddlgym_tsp_operator_actions':
        result['agent'] = args[1]
        result['visited'] = sorted(set(result['visited']) | {args[1]})
    elif game == 'phase_gates':
        if name == 'toggle':
            result['phase'] = 1 - result['phase']
        else:
            result['agent'] = args[1]
    elif game == 'energy_route':
        if name == 'recharge':
            result['energy'] = 4
        else:
            result['agent'] = args[1]
            result['energy'] -= spec['rough_cost'] if args[1] in spec['rough'] else 1
    elif game == 'fragile_bridges':
        if name == 'collect':
            result['collected'].append(args[0])
        else:
            if args[0] in spec['fragile']:
                result['collapsed'] = sorted(set(result['collapsed']) | {args[0]})
            result['agent'] = args[1]
    elif game == 'coupled_tokens':
        result['at']['a'], result['at']['b'] = args[1], args[3]
    return result


def goal_reached(spec, world):
    goal, game = spec['goal'], spec['domain']
    if game in {'frozenlake_v1', 'maze_v1'}:
        return world['at']['player'] == goal['agent']
    if game in {'phase_gates', 'energy_route'}:
        return world['agent'] == goal['agent']
    if game == 'sokoban_v1':
        return all(world['at'][stone] in spec['goal_cells'] for stone in spec['stones'])
    if game == 'package_v1':
        return set(goal['open']) <= set(world['open'])
    if game == 'printer_v1':
        return world['on'].get('printer1') == 'table1' and 'printer1' in world['powered']
    if game == 'overcooked_v1':
        return world['delivered']
    if game in {'pddlgym_blocks_medium', 'pddlgym_hanoi_operator_actions'}:
        return all(world['supports'].get(item) == support for item, support in goal['supports'].items())
    if game == 'pddlgym_slidetile':
        return all(world['at'].get(tile) == xy for tile, xy in goal['at'].items())
    if game == 'pddlgym_tsp_operator_actions':
        return set(goal['visited']) <= set(world['visited'])
    if game == 'fragile_bridges':
        return world['agent'] == goal['agent'] and set(spec['markers']) <= set(world['collected'])
    if game == 'coupled_tokens':
        return world['at'] == goal['at']
    raise ValueError(game)


def simulate(spec, actions):
    world = copy.deepcopy(spec['world'])
    trace = [copy.deepcopy(world)]
    for index, action in enumerate(actions):
        try:
            world = transition(spec, world, action)
        except ValueError as error:
            return {'pass': False, 'executable': False, 'goal_reached': False,
                    'failed_step': index + 1, 'action': list(action), 'error': str(error), 'trace': trace}
        trace.append(copy.deepcopy(world))
    reached = goal_reached(spec, world)
    return {'pass': reached, 'executable': True, 'goal_reached': reached, 'trace': trace}


def operation_signatures(domain):
    node = parse_sexpr(domain)
    if node[:1] != ['define']:
        raise ValueError('Expected a PDDL definition')
    signatures = {}
    for action in node:
        if not isinstance(action, list) or action[:1] != [':action']:
            continue
        name = action[1]
        if name in signatures:
            raise ValueError('Duplicate operation: ' + name)
        parameters = action[action.index(':parameters') + 1]
        signatures[name] = sum(isinstance(p, str) and p.startswith('?') for p in parameters)
    return signatures


def val_behavior(domain, problem, actions, directory):
    """VAL distinguishes a legal prefix with an unmet goal from an illegal plan."""
    plan = directory / 'probe_plan.txt'
    plan.write_text('\n'.join('(' + ' '.join(a) + ')' for a in actions) + '\n')
    result = validate_files(domain, problem, plan)
    log = result['log']
    if 'Plan executed successfully - checking goal' in log:
        return {'executable': True, 'goal_reached': result['pass'], 'log': log}
    if 'Plan failed to execute' in log:
        steps = re.findall(r'Checking next happening \(time (\d+)\)', log)
        return {'executable': False, 'goal_reached': False,
                'failed_step': int(steps[-1]) if steps else None, 'log': log}
    if 'Error: Bad operator in plan!' in log:
        return {'executable': False, 'goal_reached': False, 'bad_operator': True, 'log': log}
    # Parse/type errors must not be mistaken for correctly rejected illegal moves.
    return {'executable': None, 'goal_reached': None, 'error': 'VAL could not evaluate this trace', 'log': log}


def evaluate_probes(domain, problem, probes, directory):
    results, observed = [], set()
    for expected in probes:
        actual = val_behavior(domain, problem, expected['actions'], directory)
        passed = (actual['executable'] == expected['executable'] and
                  actual['goal_reached'] == expected['goal_reached'])
        if not expected['executable'] and actual.get('failed_step') is not None:
            passed = passed and actual['failed_step'] == expected['failed_step']
        if expected['executable']:
            observed.update(a[0] for a in expected['actions'])
        results.append({'kind': expected['kind'], 'mechanic': expected.get('mechanic'), 'pass': passed,
                        'expected': {k: expected[k] for k in ['executable', 'goal_reached']},
                        'actual': {k: v for k, v in actual.items() if k != 'log'}})
    return {'pass': bool(results) and all(r['pass'] for r in results),
            'passed': sum(r['pass'] for r in results), 'total': len(results),
            'observed_operations': sorted(observed), 'results': results}


def evaluate_candidate(row, raw_output, directory):
    """Always solve the submitted PDDL afresh, then execute in the real game."""
    directory.mkdir(parents=True, exist_ok=True)
    for old in [directory / name for name in ['plan.txt', 'domain.pddl', 'problem.pddl', 'solver.log',
                'candidate_val.log', 'reference_val.log', 'environment_trace.json', 'probe_plan.txt', 'output.sas']
                ] + list(directory.glob('plan.txt.[0-9]*')) + list(directory.glob('probe_*_failure.log')):
        old.unlink(missing_ok=True)
    result = {'id': row['id'], 'split': row['split'], 'domain': row['domain'],
              'group_id': row['group_id'], 'paired_with': row['paired_with'],
              'task_success': False, 'behavior_success': False, 'status': 'invalid_output'}
    probes = json.loads(Path(row['probes_path']).read_text())
    result['probes'] = {'pass': False, 'passed': 0, 'total': len(probes),
                        'observed_operations': [], 'results': []}
    try:
        dtext, ptext = parse_pddl_output(raw_output)
        if not isinstance(parse_sexpr(ptext), list):
            raise ValueError('Expected a PDDL expression')
        actual = operation_signatures(dtext)
        result['interface_ok'] = actual == row['operation_signatures']
        result['operation_signatures'] = actual
        domain, problem = directory / 'domain.pddl', directory / 'problem.pddl'
        domain.write_text(dtext + '\n')
        problem.write_text(ptext + '\n')
    except (ValueError, KeyError, TypeError, IndexError) as error:
        result['error'] = str(error)
        return result
    result['probes'] = evaluate_probes(domain, problem, probes, directory)
    result['unobserved_operations'] = sorted(set(row['operation_signatures']) - set(result['probes']['observed_operations']))
    solved = solve_files(domain, problem)
    result['solver'] = solved
    if not solved['solved']:
        # Fast Downward's documented exit codes distinguish malformed models,
        # proven unsolvability, incomplete search and resource limits.
        result['status'] = ('solver_timeout' if solved['timeout'] else {
            10: 'candidate_unsolvable', 11: 'candidate_unsolvable', 12: 'search_incomplete',
            20: 'solver_memory_limit', 22: 'solver_memory_limit',
            31: 'invalid_pddl', 33: 'invalid_pddl', 36: 'invalid_pddl',
            34: 'unsupported_pddl', 37: 'unsupported_pddl',
        }.get(solved['returncode'], 'solver_error'))
        result['error'] = '\n'.join((directory / 'solver.log').read_text().splitlines()[-12:])
        return result
    try:
        actions = read_plan(directory / 'plan.txt')
    except ValueError as error:
        result.update(status='invalid_plan', error=str(error))
        return result
    own = validate_files(domain, problem, directory / 'plan.txt')
    oracle = validate_files(row['gt_domain_path'], row['gt_problem_path'], directory / 'plan.txt')
    spec = json.loads(Path(row['state_path']).read_text())
    simulation = simulate(spec, actions)
    result.update(plan_length=len(actions), candidate_val_pass=own['pass'],
                  reference_val_pass=oracle['pass'],
                  environment={k: v for k, v in simulation.items() if k != 'trace'})
    if own['pass'] and oracle['pass'] != simulation['pass']:
        result['status'] = 'oracle_disagreement'
        return result
    result['task_success'] = result['interface_ok'] and own['pass'] and oracle['pass'] and simulation['pass']
    result['behavior_success'] = result['task_success'] and result['probes']['pass']
    result['status'] = ('success' if result['task_success'] else 'interface_mismatch' if not result['interface_ok']
                        else 'candidate_plan_invalid' if not own['pass']
                        else 'illegal_environment_action' if not simulation['executable'] else 'real_goal_not_reached')
    return result


def generation_identity(row, template):
    prompt = template.format(instruction=row['instruction'], robot_configuration='single-arm')
    return prompt, {'model': eval_model,
                    'prompt_sha256': sha(prompt), 'image_sha256': row['image_sha256'], 'temperature': 0}


def load_tasks(manifest, selected_datasets=None):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate manifest IDs')
    selected = [r for r in rows if selected_datasets is None or r['dataset'] in selected_datasets]
    if not selected:
        raise ValueError('No selected tasks')
    cache = {}
    for row in selected:
        # Read the same standard files as eval_planning.py; metadata is an oracle.
        tasks_root = Path(row['image_path']).parents[3]
        key = str(tasks_root), row['dataset']
        if key not in cache:
            cache[key] = tuple(json.loads((tasks_root / folder / f'{prefix}_{row["dataset"]}.json').read_text())
                               for folder, prefix in [('instructions', 'instructions'), ('steps', 'steps')])
        instructions, steps = cache[key]
        instruction = instructions[row['task']][row['episode']]
        reference_steps = steps[row['task']][row['episode']]
        if sha(instruction) != row['instruction_sha256'] or sha(Path(row['image_path']).read_bytes()) != row['image_sha256']:
            raise ValueError('Input hash mismatch: ' + row['id'])
        if '\n'.join(reference_steps).strip() != Path(row['kf_actions_path']).read_text().strip():
            raise ValueError('Standard steps / reference mismatch: ' + row['id'])
        for field, hash_field in [('state_path', 'state_sha256'), ('gt_domain_path', 'domain_sha256'),
                                  ('gt_problem_path', 'problem_sha256'), ('gt_plan_path', 'plan_sha256')]:
            value = json.loads(Path(row[field]).read_text()) if field == 'state_path' else Path(row[field]).read_bytes()
            if sha(value) != row[hash_field]:
                raise ValueError('Oracle hash mismatch: ' + row['id'] + ' ' + field)
        spec = json.loads(Path(row['state_path']).read_text())
        for probe in json.loads(Path(row['probes_path']).read_text()):
            expected = simulate(spec, probe['actions'])
            if any(probe[k] != expected[k] for k in ['executable', 'goal_reached']):
                raise ValueError('Probe oracle mismatch: ' + row['id'])
        row['instruction'] = instruction
    return selected


def save_result(directory, result, cache):
    """Keep a readable result and one cache; remove superseded runtime files."""
    stored = {k: v for k, v in result.items() if k != 'generation'}
    if 'solver' in stored:
        stored['solver'] = {k: v for k, v in stored['solver'].items() if k != 'command'}
    cache = dict(cache)
    cache.pop('fingerprint', None)
    cache['result'] = stored
    write_json(directory / '.cache.json', cache)

    probes = result['probes']
    reason = ('无' if result['behavior_success'] else
              '任务已完成，但部分规则检查未通过' if result['task_success'] else
              STATUS_LABELS.get(result['status'], result['status']))
    lines = [f'任务：{result["id"]}', f'游戏：{result["domain"]}',
             f'任务完成：{"成功" if result["task_success"] else "失败"}',
             f'全部规则检查通过：{"是" if probes["pass"] else "否"}（{probes["passed"]}/{probes["total"]}）',
             f'说明：{reason}']
    if 'plan_length' in result:
        lines.append(f'计划长度：{result["plan_length"]} 步')
    environment = result.get('environment', {})
    if environment.get('failed_step') is not None:
        lines.append(f'失败动作：第 {environment["failed_step"]} 步，`' +
                     ' '.join(environment.get('action', [])) + '`')
    if result.get('error'):
        lines.append('<details>\n<summary>错误详情（点击展开）</summary>\n\n```text\n' +
                     result['error'] + '\n```\n\n</details>')
    links = [f'[{label}]({name})' for name, label in
             [('domain.pddl', '动作模型'), ('problem.pddl', '初始状态与目标'), ('plan.txt', '求解计划')]
             if (directory / name).is_file()]
    if links:
        lines.append('相关文件：' + ' · '.join(links))
    (directory / 'result.md').write_text('\n\n'.join(lines) + '\n')
    obsolete = ['result.json', 'generation.json', 'output.txt', 'solver.log', 'error.log',
                'candidate_val.log', 'reference_val.log', 'environment_trace.json', 'probe_plan.txt', 'output.sas']
    for path in [directory / name for name in obsolete] + list(directory.glob('plan.txt.[0-9]*')) + list(directory.glob('probe_*_failure.log')):
        path.unlink(missing_ok=True)


def run_one(row, template, code_hash):
    start = time.monotonic()
    directory = eval_root / row['dataset'] / row['task'] / row['episode']
    directory.mkdir(parents=True, exist_ok=True)
    for name in ['domain.pddl', 'problem.pddl', 'plan.txt']:
        (directory / name).unlink(missing_ok=True)
    prompt, identity = generation_identity(row, template)
    cache = {}
    try:
        saved = json.loads((directory / '.cache.json').read_text()) if (directory / '.cache.json').is_file() else {}
        cache = saved
        matches = bool(saved) and all(saved.get('identity', {}).get(k) == v for k, v in identity.items())
        if matches and isinstance(saved.get('raw_output'), str) and saved.get('output_sha256') == sha(saved['raw_output']):
            raw = saved['raw_output']
            capture = saved['capture']
        else:
            cache = {'identity': identity}
            capture = {}
            raw = call_gpt(eval_model, prompt, [Path(row['image_path'])], temperature=0, capture=capture)
        cache = {'identity': identity, 'raw_output': raw, 'output_sha256': sha(raw), 'capture': capture}
        with tempfile.TemporaryDirectory(prefix='.eval-', dir=directory) as temporary:
            work = Path(temporary)
            result = evaluate_candidate(row, raw, work)
            for name in ['domain.pddl', 'problem.pddl', 'plan.txt']:
                if (work / name).is_file():
                    shutil.copy2(work / name, directory / name)
        result['generation'] = capture
    except Exception as error:
        if 'raw_output' not in cache:
            cache = {'identity': identity, 'error': str(error)[:1000]}
        result = {'id': row['id'], 'split': row['split'], 'domain': row['domain'],
                  'group_id': row['group_id'], 'paired_with': row['paired_with'],
                  'task_success': False, 'behavior_success': False,
                  'status': 'infrastructure_error', 'error': str(error)[:1000],
                  'probes': {'pass': False, 'passed': 0, 'total': len(json.loads(Path(row['probes_path']).read_text())), 'results': []}}
    result.update(elapsed_seconds=round(time.monotonic() - start, 3), evaluator_sha256=code_hash)
    save_result(directory, result, cache)
    return result


def aggregate(results):
    n = len(results)
    probe_total = sum(r.get('probes', {}).get('total', 0) for r in results)
    probe_pass = sum(r.get('probes', {}).get('passed', 0) for r in results)
    return {'total': n, 'task_success': sum(r['task_success'] for r in results),
            'task_success_rate': sum(r['task_success'] for r in results) / n,
            'behavior_success': sum(r['behavior_success'] for r in results),
            'behavior_success_rate': sum(r['behavior_success'] for r in results) / n,
            'probe_passed': probe_pass, 'probe_total': probe_total,
            'probe_agreement_rate': probe_pass / probe_total if probe_total else None,
            'statuses': dict(Counter(r['status'] for r in results))}


def summarize(results):
    summary = {'model': eval_model, 'manifest_sha256': sha(DEFAULT_MANIFEST.read_bytes()),
               'overall': aggregate(results), 'by_split': {}, 'by_domain': {}, 'pairs': []}
    for key in ['split', 'domain']:
        groups = defaultdict(list)
        for result in results:
            groups[result[key]].append(result)
        summary['by_' + key] = {k: aggregate(v) for k, v in groups.items()}
    by_id = {r['id']: r for r in results}
    for changed in results:
        base = by_id.get(changed['paired_with'])
        if base:
            summary['pairs'].append({'base': base['id'], 'variant': changed['id'],
                'both_tasks_pass': base['task_success'] and changed['task_success'],
                'both_behaviors_pass': base['behavior_success'] and changed['behavior_success'],
                'rule_difference_correct': all(p['pass'] for r in [base, changed]
                                               for p in r.get('probes', {}).get('results', []) if p['kind'] == 'rule_difference')
                    and all(any(p['kind'] == 'rule_difference' for p in r.get('probes', {}).get('results', [])) for r in [base, changed])})
    write_json(eval_root / 'summary_abs.json', summary)
    tasks = {row['id']: row for row in (
        json.loads(line) for line in DEFAULT_MANIFEST.read_text().splitlines() if line.strip())}
    report = render_abstract_report(summary, results, tasks, eval_root)
    (eval_root / 'report_abs.md').write_text(report)
    return summary


def main():
    eval_root.mkdir(parents=True, exist_ok=True)
    rows = load_tasks(DEFAULT_MANIFEST, datasets)
    template = PROMPT.read_text()
    code_hash = sha([sha(Path(path).read_bytes()) for path in [
        __file__, call_gpt.__code__.co_filename, parse_pddl_output.__code__.co_filename,
        parse_sexpr.__code__.co_filename, _run_fast_downward.__code__.co_filename, VAL, FD]])
    print(f'开始评测：{eval_model}，共 {len(rows)} 个任务。', flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=generation_max_workers) as pool:
        futures = [pool.submit(run_one, row, template, code_hash) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: r['id'])
    summary = summarize(results)
    stats = summary['overall']
    failures = '；'.join(
        f'{STATUS_LABELS.get(status, status)} {count} 个'
        for status, count in sorted(stats['statuses'].items(), key=lambda item: -item[1])
        if status != 'success'
    )
    print('\n评测完成：\n'
          f'任务成功：{stats["task_success"]}/{stats["total"]}（{stats["task_success_rate"]:.1%}）\n'
          f'任务成功且全部规则检查通过：{stats["behavior_success"]}/{stats["total"]}'
          f'（{stats["behavior_success_rate"]:.1%}）\n'
          f'任务失败原因：{failures or "无"}\n'
          f'详细结果：   {eval_root / "report_abs.md"}', flush=True)


if __name__ == '__main__':
    main()
