"""Locate model errors using the existing plan, without extra searches or probes."""
from .model import parse_sexpr, section


def diagnosis(category, subcategory, evidence):
    return {'category':category,'subcategory':subcategory,'evidence':evidence}


def _normal(node, binding, depth=0):
    """Compare expressions modulo variable names and conjunction ordering."""
    if isinstance(node,str):
        return binding.get(node,node)
    if not node:
        return ()
    head = node[0]
    if head in {'forall','exists'}:
        variables = tuple(f'?bound{depth}_{i}' for i in range(len(node[1])))
        local = {**binding,**dict(zip(node[1],variables))}
        return (head,variables,_normal(node[2],local,depth+1))
    children = [_normal(child,binding,depth) for child in node[1:]]
    if head in {'and','or'}:
        children = sorted(set(children),key=repr)
        if len(children) == 1:
            return children[0]
    return (head,*children)


def _text(node):
    return '('+' '.join(map(_text,node))+')' if isinstance(node,(list,tuple)) else str(node)


def initial_conflict(problem):
    positive,negative = set(),set()
    for fact in section(parse_sexpr(problem),':init',[':init'])[1:]:
        if not isinstance(fact,list) or not fact:
            continue
        if fact[0] == 'not' and len(fact) == 2 and isinstance(fact[1],list):
            if all(isinstance(part,str) for part in fact[1]): negative.add(tuple(fact[1]))
        elif all(isinstance(part,str) for part in fact):
            positive.add(tuple(fact))
    if conflict := sorted(positive & negative):
        fact = _text(conflict[0])
        return diagnosis('init_error','init_inconsistent',f'init 同时包含 `{fact}` 和 `(not {fact})`。')


def _mapped_state(state,mapping):
    return {(predicate,*(mapping[arg] for arg in args)) for predicate,*args in state}


def _same_step(candidate,reference,action,canonical,mapping):
    op,ref = candidate['ops'][action[0]],reference['ops'][canonical[0]]
    if len(action)-1 != len(op[':parameters']) or len(canonical)-1 != len(ref[':parameters']):
        return False
    binding = {**mapping,**dict(zip(op[':parameters'],(mapping[arg] for arg in action[1:])))}
    expected = dict(zip(ref[':parameters'],canonical[1:]))
    return all(_normal(op[key],binding) == _normal(ref[key],expected)
               for key in [':precondition',':effect'])


def _init_error(actual,expected):
    differences = []
    if added := sorted(actual-expected): differences.append(f'init 多写 `{_text(added[0])}`')
    if missing := sorted(expected-actual): differences.append(f'init 漏写 `{_text(missing[0])}`')
    return diagnosis('init_error','init_mismatch',
                     '；'.join(differences)+'；截至失败处，动作规则与参考一致。')


def _goal_atoms(node):
    if node and node[0] == 'and':
        return {atom for child in node[1:] for atom in _goal_atoms(child)}
    if node and node[0] not in {'or','forall','exists','imply'}:
        return {node}
    return set()


def diagnose_failure(result,candidate,reference,mapping,plan,canonical,replay):
    """Require a concrete witness; execution failure alone never proves oper error."""
    if result['status'] == 'interface_mismatch':
        if not candidate['ops']:
            return diagnosis('operator_error','operator_missing','未定义任务动作。')
        if unexpected := sorted(set(candidate['ops'])-set(reference['ops'])):
            return diagnosis('operator_error','operator_modeling',
                             '动作名不符合允许接口：'+'、'.join(f'`{name}`' for name in unexpected)+'。')
        return None
    if result['status'] not in {'illegal_environment_action','real_goal_not_reached'}:
        return None
    # Equal object universes are needed for quantified conditions. Unknown
    # identities or alternative state encodings must not become false diagnoses.
    if set(mapping) != set(candidate['objects']) or set(mapping.values()) != set(reference['objects']):
        return None
    if len(set(mapping.values())) != len(mapping) or len(plan) != len(canonical):
        return None
    own,real = replay(candidate,plan),replay(reference,canonical)
    if not own['executable'] or not own['goal_reached']:
        return None
    actual_initial = _mapped_state(candidate['init'],mapping)
    same_initial = actual_initial == reference['init']
    same_goal = _normal(candidate['goal'],mapping) == _normal(reference['goal'],{})

    if result['status'] == 'illegal_environment_action':
        if real['executable']:
            return None
        step = real['failed_step']
        if same_initial:
            return diagnosis('operator_error','operator_modeling',
                             f'init 与参考一致；第 {step} 步 `{" ".join(canonical[step-1])}` 不满足真实动作条件。')
        if all(_same_step(candidate,reference,a,b,mapping) for a,b in zip(plan[:step],canonical[:step])):
            return _init_error(actual_initial,reference['init'])
        return None

    if not real['executable'] or real['goal_reached']:
        return None
    if _mapped_state(own['trace'][-1],mapping) == real['trace'][-1]:
        missing = sorted(_goal_atoms(_normal(reference['goal'],{}))-
                         _goal_atoms(_normal(candidate['goal'],mapping)),key=repr)
        evidence = ('goal 未包含 '+ '、'.join(f'`{_text(atom)}`' for atom in missing[:3])+'。'
                    if missing else '同一最终状态满足候选 goal，但不满足真实 goal。')
        return diagnosis('goal_error','goal_mismatch',evidence)
    if same_initial and same_goal:
        return diagnosis('operator_error','operator_modeling','init、goal 与参考一致；动作执行后的状态与真实结果不同。')
    if same_goal and all(_same_step(candidate,reference,a,b,mapping) for a,b in zip(plan,canonical)):
        return _init_error(actual_initial,reference['init'])
    return None
