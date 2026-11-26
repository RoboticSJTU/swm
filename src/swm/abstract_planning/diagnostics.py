"""Align candidate actions and explain evaluation failures using reference evidence."""
from collections import Counter, defaultdict

from swm.pddl.reorder import effect_predicates
from swm.pddl.strips import parse_sexpr
from .model import section, sexpr


class AmbiguousActionMapping(ValueError):
    pass


def extensions(model, mapping):
    result = defaultdict(set)
    for predicate, *args in model['init']:
        if all(arg in mapping for arg in args):
            result[predicate].add(tuple(mapping[arg] for arg in args))
    return dict(result)


def atoms(node, env, predicates, context=()):
    """Retain polarity and conditional or quantified context when comparing roles."""
    head, *args = node
    if head == 'and':
        return [atom for child in args for atom in atoms(child, env, predicates, context)]
    if head in {'forall', 'exists'}:
        variables, body = args
        bound = {var: ('bound', len(context), i) for i, var in enumerate(variables)}
        return atoms(body, {**env, **bound}, predicates, (*context, head))
    if head in {'not', 'or', 'imply', 'when'}:
        return [atom for i, child in enumerate(args) for atom in atoms(child, env, predicates, (*context, head, i))]
    predicate = head if predicates is None else predicates.get(head, ('unknown', len(args)))
    return [(context, predicate, tuple(env.get(arg, ('constant', arg)) for arg in args))]


def align_actions(candidate, oracle, mapping, operations=None):
    reference_ext = extensions(oracle, {obj: obj for obj in oracle['objects']})
    candidate_ext = extensions(candidate, mapping)
    predicates = {}
    for name, extension in candidate_ext.items():
        matches = [ref for ref, facts in reference_ext.items() if facts == extension]
        if name in matches:
            predicates[name] = name
        elif len(matches) == 1:
            predicates[name] = matches[0]
    # Initial-state errors must preserve evidence from unchanged predicate names.
    predicates.update((name, name) for op in oracle['ops'].values()
                      for key in (':precondition', ':effect') for _, name, _ in atoms(op[key], {}, None))
    candidate_mutable, reference_mutable = (
        set().union(*(effect_predicates(op[':effect']) for op in model['ops'].values()))
        for model in (candidate, oracle)
    )

    def guards(node, extensions, mutable):
        roles = {}
        for context, predicate, args in atoms(node, {}, None):
            if context or len(args) != 1 or predicate not in extensions or predicate in mutable:
                continue
            parameter = args[0][1]
            if parameter.startswith('?'):
                allowed = {fact[0] for fact in extensions[predicate] if len(fact) == 1}
                if parameter in roles:
                    roles[parameter] &= allowed
                else:
                    roles[parameter] = allowed
        return roles

    aligned = {}
    constants = {obj: ('constant', mapping[obj]) for obj in candidate['objects'] if obj in mapping}
    for name, op in candidate['ops'].items():
        if name not in oracle['ops'] or (operations is not None and name not in operations):
            continue
        ref = oracle['ops'][name]
        cp, rp = op[':parameters'], ref[':parameters']
        cg = guards(op[':precondition'], candidate_ext, candidate_mutable)
        rg = guards(ref[':precondition'], reference_ext, reference_mutable)
        singletons = {v: next(iter(rg[v])) for v in rp if v in rg and len(rg[v]) == 1}
        ref_env = {v: ('constant', singletons[v]) if v in singletons else ('role', i) for i, v in enumerate(rp)}
        choices = [[i for i, cv in enumerate(cp) if rv not in rg or cv not in cg or rg[rv] & cg[cv]]
                   + ([singletons[rv]] if rv in singletons else []) for rv in rp]
        reference = Counter(atoms(ref[':precondition'], ref_env, None, ('pre',))
                            + atoms(ref[':effect'], ref_env, None, ('effect',)))
        candidate_positions = set(range(len(cp)))
        forced = {i: next(iter(cg[v])) for i, v in enumerate(cp) if v in cg and len(cg[v]) == 1}
        best_score, best, examined = -1, {}, 0

        def visit(index, route, used):
            nonlocal best_score, examined
            if examined > 100000:
                raise AmbiguousActionMapping('Too many possible roles for ' + name)
            if index < len(rp):
                for choice in choices[index]:
                    if isinstance(choice, int):
                        if choice not in used:
                            visit(index + 1, (*route, choice), used | {choice})
                    else:
                        visit(index + 1, (*route, choice), used)
                return
            remaining = candidate_positions - used
            if not remaining <= forced.keys():
                return
            examined += 1
            extras = {i: forced[i] for i in remaining}
            env = {**constants, **{cp[i]: ('constant', x) for i, x in extras.items()},
                   **{cp[i]: ref_env[v] for v, i in zip(rp, route) if isinstance(i, int)}}
            observed = Counter(atoms(op[':precondition'], env, predicates, ('pre',))
                               + atoms(op[':effect'], env, predicates, ('effect',)))
            score = sum((reference & observed).values())
            if score > best_score:
                best_score = score
                best.clear()
            if score == best_score:
                # Routes supplying the same singleton reference object are equivalent.
                best.setdefault(tuple(singletons.get(v, i) for v, i in zip(rp, route)), (route, extras))

        visit(0, (), set())
        if len(best) != 1:
            raise AmbiguousActionMapping(f'Cannot uniquely infer argument roles for {name}: {len(best)} candidates, score {best_score}')
        route, extras = next(iter(best.values()))
        aligned[name] = {'reference_arguments': list(route), 'candidate_singletons': extras, 'score': best_score,
                         'candidate_arity': len(cp), 'reference_arity': len(rp)}
    return aligned


def to_reference(action, mapping, adapters):
    if action[0] not in adapters:
        raise AmbiguousActionMapping('Unmapped operation ' + action[0])
    return (action[0], *(mapping[action[i + 1]] if isinstance(i, int) else i
                         for i in adapters[action[0]]['reference_arguments']))


def to_candidate(action, inverse, adapters):
    if action[0] not in adapters:
        raise ValueError('Missing operation ' + action[0])
    adapter = adapters[action[0]]
    args = [None] * adapter['candidate_arity']
    for value, position in zip(action[1:], adapter['reference_arguments']):
        if isinstance(position, int):
            args[position] = inverse[value]
        elif position != value:
            raise ValueError('Singleton role mismatch')
    for i, value in adapter['candidate_singletons'].items():
        args[int(i)] = inverse[value]
    if None in args:
        raise ValueError('Incomplete action mapping')
    return (action[0], *args)


def diagnosis(category, subcategory, evidence):
    return {'category': category, 'subcategory': subcategory, 'evidence': evidence}


def _normal(node, binding, depth=0):
    """Compare expressions modulo variable names and conjunction ordering."""
    if isinstance(node, str):
        return binding.get(node, node)
    if not node:
        return ()
    head, *args = node
    if head in {'forall', 'exists'}:
        variables = tuple(f'?bound{depth}_{i}' for i in range(len(args[0])))
        return (head, variables, _normal(args[1], {**binding, **dict(zip(args[0], variables))}, depth + 1))
    children = [_normal(child, binding, depth) for child in args]
    if head in {'and', 'or'}:
        children = sorted(set(children), key=repr)
        if len(children) == 1:
            return children[0]
    return (head, *children)


def initial_conflict(problem):
    positive, negative = set(), set()
    for fact in section(parse_sexpr(problem), ':init', [':init'])[1:]:
        if not isinstance(fact, list) or not fact:
            continue
        negated = fact[0] == 'not' and len(fact) == 2 and isinstance(fact[1], list)
        atom = fact[1] if negated else fact
        if all(isinstance(part, str) for part in atom):
            (negative if negated else positive).add(tuple(atom))
    if conflict := positive & negative:
        fact = sexpr(min(conflict))
        return diagnosis('init_error', 'init_inconsistent', f'init 同时包含 `{fact}` 和 `(not {fact})`。')


def _mapped_state(state, mapping):
    return {(predicate, *(mapping[arg] for arg in args)) for predicate, *args in state}


def _same_step(candidate, reference, action, canonical, mapping):
    op, ref = candidate['ops'][action[0]], reference['ops'][canonical[0]]
    if len(action) - 1 != len(op[':parameters']) or len(canonical) - 1 != len(ref[':parameters']):
        return False
    binding = {**mapping, **dict(zip(op[':parameters'], (mapping[arg] for arg in action[1:])))}
    expected = dict(zip(ref[':parameters'], canonical[1:]))
    return all(_normal(op[key], binding) == _normal(ref[key], expected) for key in (':precondition', ':effect'))


def _init_error(actual, expected, extra_objects):
    differences = []
    if extra_objects:
        differences.append('声明了没有对应真实对象的 ' + '、'.join(f'`{x}`' for x in sorted(extra_objects)[:3]))
    if added := actual - expected:
        differences.append(f'init 多写 `{sexpr(min(added))}`')
    if missing := expected - actual:
        differences.append(f'init 漏写 `{sexpr(min(missing))}`')
    return diagnosis('init_error', 'init_mismatch', '；'.join(differences) + '；截至失败处，动作规则与参考一致。')


def _goal_atoms(node):
    if node and node[0] == 'and':
        return set().union(*(_goal_atoms(child) for child in node[1:]))
    return {node} if node and node[0] not in {'or', 'forall', 'exists', 'imply'} else set()


def diagnose_failure(result, candidate, reference, mapping, plan, canonical, replay):
    """Attribute a failure when replay distinguishes the candidate and reference models."""
    status = result['status']
    if status == 'interface_mismatch':
        if not candidate['ops']:
            return diagnosis('operator_error', 'operator_missing', '未定义任务动作。')
        if unexpected := sorted(candidate['ops'].keys() - reference['ops'].keys()):
            return diagnosis('operator_error', 'operator_modeling',
                             '动作名不符合允许接口：' + '、'.join(f'`{name}`' for name in unexpected) + '。')
        return None
    if status not in {'illegal_environment_action', 'real_goal_not_reached'}:
        return None
    actual_objects, expected_objects = set(candidate['objects']), set(reference['objects'])
    if (not mapping.keys() <= actual_objects or set(mapping.values()) != expected_objects
            or len(mapping) != len(expected_objects) or len(plan) != len(canonical)
            or candidate.get('arities') != reference.get('arities')):
        return None
    # Keep additional objects in state comparisons and quantified rules.
    extra_objects = actual_objects - mapping.keys()
    mapping = {**mapping, **{x: 'candidate::' + x for x in extra_objects}}
    own, real = replay(candidate, plan), replay(reference, canonical)
    if not own['executable'] or not own['goal_reached']:
        return None
    actual_initial = _mapped_state(candidate['init'], mapping)
    same_initial = not extra_objects and actual_initial == reference['init']

    if status == 'illegal_environment_action':
        if real['executable']:
            return None
        step = real['failed_step']
        if same_initial:
            return diagnosis('operator_error', 'operator_modeling',
                             f'init 与参考一致；第 {step} 步 `{" ".join(canonical[step - 1])}` 不满足真实动作条件。')
        if all(_same_step(candidate, reference, a, b, mapping) for a, b in zip(plan[:step], canonical[:step])):
            return _init_error(actual_initial, reference['init'], extra_objects)
        return None

    if not real['executable'] or real['goal_reached']:
        return None
    actual_goal, expected_goal = _normal(candidate['goal'], mapping), _normal(reference['goal'], {})
    if not extra_objects and _mapped_state(own['trace'][-1], mapping) == real['trace'][-1]:
        missing = sorted(_goal_atoms(expected_goal) - _goal_atoms(actual_goal), key=repr)
        evidence = ('goal 未包含 ' + '、'.join(f'`{sexpr(atom)}`' for atom in missing[:3]) + '。'
                    if missing else '同一最终状态满足候选 goal，但不满足真实 goal。')
        return diagnosis('goal_error', 'goal_mismatch', evidence)
    if actual_goal == expected_goal:
        if same_initial:
            return diagnosis('operator_error', 'operator_modeling', 'init、goal 与参考一致；动作执行后的状态与真实结果不同。')
        if all(_same_step(candidate, reference, a, b, mapping) for a, b in zip(plan, canonical)):
            return _init_error(actual_initial, reference['init'], extra_objects)
    return None


def diagnose_reference_plan(candidate, reference, mapping, plan, replay):
    """Explain an unsolvable candidate by swapping init and operators one at a time."""
    expected_objects = set(reference['objects'])
    if (mapping.keys() != set(candidate['objects']) or set(mapping.values()) != expected_objects
            or len(mapping) != len(expected_objects) or candidate.get('arities') != reference.get('arities')):
        return None
    real = replay(reference, plan)
    if not real['executable'] or not real['goal_reached']:
        return None
    operations = {action[0] for action in plan}
    adapters = align_actions(candidate, reference, mapping, operations)
    inverse = {value: key for key, value in mapping.items()}
    try:
        translated = [to_candidate(action, inverse, adapters) for action in plan]
    except ValueError:
        if missing := sorted(operations - candidate['ops'].keys()):
            return diagnosis('operator_error', 'operator_missing',
                             '参考解所需动作整体缺失：' + '、'.join(f'`{name}`' for name in missing) + '。')
        return None
    own = replay(candidate, translated)
    actual_initial = _mapped_state(candidate['init'], mapping)
    if own['executable']:
        if not own['goal_reached'] and actual_initial == reference['init'] and all(
                _same_step(candidate, reference, a, b, mapping) for a, b in zip(translated, plan)):
            return diagnosis('goal_error', 'goal_mismatch', '参考解在候选模型中可执行并满足真实 goal，但不满足候选 goal。')
        return None
    if _normal(candidate['goal'], mapping) != _normal(reference['goal'], {}):
        return None
    fixed_init = replay({**candidate, 'init': _mapped_state(reference['init'], inverse)}, translated)
    fixed_ops = replay({**reference, 'init': actual_initial}, plan)
    init_restores = fixed_init['executable'] and fixed_init['goal_reached']
    ops_restore = fixed_ops['executable'] and fixed_ops['goal_reached']
    if init_restores == ops_restore:
        return None
    step = own['failed_step']
    witness = f'参考解第 {step} 步 `{" ".join(plan[step - 1])}` 被候选动作拒绝'
    if init_restores:
        proof = diagnosis('init_error', 'init_mismatch', witness + '；仅修正 init 后整条参考解成功，仅替换动作仍失败。')
    else:
        proof = diagnosis('operator_error', 'operator_modeling', witness + '；仅替换动作规则后整条参考解成功，仅修正 init 仍失败。')
    proof['witness'] = {'source': 'reference_plan', 'failed_step': step, 'action': list(plan[step - 1]),
                        'init_repair_pass': init_restores, 'operator_repair_pass': ops_restore}
    return proof
