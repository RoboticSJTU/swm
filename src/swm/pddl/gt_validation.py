"""Replay a semantically aligned plan in the official PDDL initial state and goal."""
from pathlib import Path
import re
import sys
from threading import Lock
from types import SimpleNamespace

from swm.pddl.attribution import map_objects

VALIDATOR_VERSION = 2
_LOCK = Lock()  # Unified Planning's environment is shared; release before VLM calls.


def _text(path):
    text = re.sub(r';[^\n]*', '', Path(path).read_text(encoding='utf-8')).lower()
    return re.sub(r'\(\s*:functions\s*\)', '', text)


def _expression(node, parameters=None, objects=None, variables=None):
    """An alpha-normalized expression; only logically safe ordering is normalized."""
    parameters, objects, variables = parameters or {}, objects or {}, variables or {}
    if node.is_parameter_exp():
        return ('parameter', parameters[node.parameter().name])
    if node.is_variable_exp():
        return ('variable', variables[node.variable()])
    if node.is_object_exp():
        name = node.object().name
        return ('object', objects[name]) if name in objects else ('unmapped_object', name)
    if node.is_constant():
        return ('constant', str(node.constant_value()))
    if node.is_exists() or node.is_forall():
        bound = dict(variables)
        start = max(bound.values(), default=-1) + 1
        for index, variable in enumerate(node.variables(), start):
            bound[variable] = index
        return (node.node_type.name, tuple(str(v.type) for v in node.variables()),
                _expression(node.arg(0), parameters, objects, bound))
    if node.is_fluent_exp():
        return ('fluent', node.fluent().name,
                tuple(_expression(arg, parameters, objects, variables) for arg in node.args))
    args = [_expression(arg, parameters, objects, variables) for arg in node.args]
    if node.is_and() or node.is_or():
        # Flatten conjunction/disjunction and remove duplicate operands.
        kind = node.node_type.name
        args = [part for arg in args for part in (arg[1] if arg[0] == kind else (arg,))]
        args = sorted(set(args), key=repr)
    elif node.is_equals():
        args = sorted(args, key=repr)
    return (node.node_type.name, tuple(args))


def _conjunction(nodes, parameters, objects):
    parts = set()
    for node in nodes:
        if node.is_true():
            continue
        if node.is_and():
            parts.update(_conjunction(node.args, parameters, objects))
        else:
            parts.add(_expression(node, parameters, objects))
    return tuple(sorted(parts, key=repr))


def _action_signature(action, objects):
    """Compare chosen actions, including negative/conditional/quantified effects."""
    parameters = {param.name: i for i, param in enumerate(action.parameters)}
    effects = []
    for effect in action.effects:
        variables = {var: i for i, var in enumerate(effect.forall)}
        effects.append((effect.kind.name, tuple(str(v.type) for v in effect.forall),
                        _expression(effect.fluent, parameters, objects, variables),
                        _expression(effect.value, parameters, objects, variables),
                        _expression(effect.condition, parameters, objects, variables)))
    # Parameter types are checked on each mapped official ActionInstance. The proof
    # concerns those grounded actions, not equality of the entire generated domain.
    return (len(action.parameters), _conjunction(action.preconditions, parameters, objects),
            tuple(sorted(effects, key=repr)))


def _objects_in(node):
    names = {node.object().name} if node.is_object_exp() else set()
    for arg in node.args:
        names.update(_objects_in(arg))
    return names


def _mapping_world(problem):
    """Expose UP objects/facts through the existing attribution mapper's data contract."""
    positive, negative = set(), set()
    changed = {effect.fluent.fluent().name for action in problem.actions for effect in action.effects}
    kinds = {obj.name: {str(obj.type)} for obj in problem.all_objects}
    for atom, value in problem.explicit_initial_values.items():
        if not atom.is_fluent_exp() or not value.is_bool_constant():
            continue
        if not all(arg.is_object_exp() for arg in atom.args):
            continue
        fact = (atom.fluent().name, *(arg.object().name for arg in atom.args))
        (positive if value.is_true() else negative).add(fact)
        if value.is_true() and len(fact) == 2 and fact[0] not in changed:
            kinds[fact[1]].add(fact[0])
    return SimpleNamespace(
        objects=tuple(SimpleNamespace(name=name, identity_kinds=tuple(sorted(values)))
                      for name, values in sorted(kinds.items())),
        facts=frozenset(positive), negative_facts=frozenset(negative),
    )


def validate_gt_plan(candidate: Path, gt: Path, *, instruction='', images=None,
                     model='Qwen3.8-27B') -> dict:
    """UNKNOWN: no proven action correspondence, incomplete identity, or unsupported PDDL.

    VLM only supplies object identity through attribution.map_objects. Its use is
    recorded separately; subsequent replay is deterministic conditional on that map.
    Candidate PDDL and plan files are never rewritten or repaired.
    """
    result = {'validator_version': VALIDATOR_VERSION, 'status': 'UNKNOWN', 'pass': None,
              'reason': '', 'executed_steps': 0, 'mapping_source': 'identity',
              'scope': 'Plan validity in the official symbolic initial state and goal.',
              'object_mapping': {}, 'action_mapping': {}}
    try:
        dependencies = str(Path(__file__).resolve().parents[3] / 'additional_experiments/tools/python')
        if dependencies not in sys.path:
            sys.path.append(dependencies)
        from unified_planning.io import PDDLReader
        from unified_planning.shortcuts import SequentialSimulator
        from unified_planning.environment import get_environment
        from unified_planning.plans import ActionInstance
        with _LOCK:
            get_environment().error_used_name = False
            reader = PDDLReader()
            official = reader.parse_problem_string(_text(gt / 'domain.pddl'), _text(gt / 'problem.pddl'))
            predicted = reader.parse_problem_string(_text(candidate / 'domain.pddl'), _text(candidate / 'problem.pddl'))
            lines = []
            for line in _text(candidate / 'plan.txt').splitlines():
                line = line.strip()
                if not line:
                    continue
                if not re.fullmatch(r'\([^()]+\)', line):
                    result.update(status='FAIL', **{'pass': False}, reason=f'Malformed plan line: {line}')
                    return result
                lines.append(line)
            try:
                plan = reader.parse_plan_string(predicted, '\n'.join(lines))
            except Exception as error:
                result.update(status='FAIL', **{'pass': False}, reason=f'Invalid candidate action/object: {error}')
                return result
            if any(not value.is_object_exp() for action in plan.actions for value in action.actual_parameters):
                raise NotImplementedError('Non-object action parameters are unsupported.')
            required = {value.object().name for action in plan.actions for value in action.actual_parameters}
            for action in plan.actions:
                for node in action.action.preconditions:
                    required.update(_objects_in(node))
                for effect in action.action.effects:
                    for node in (effect.fluent, effect.value, effect.condition):
                        required.update(_objects_in(node))
            names = {obj.name for obj in official.all_objects}
            mapping = {obj.name: obj.name for obj in predicted.all_objects if obj.name in names}
            candidate_world, gt_world = _mapping_world(predicted), _mapping_world(official)

        # Network access is outside the UP lock. Reuse the exact original mapper,
        # including its schema validation, one-to-one rule and two-attempt limit.
        if required - mapping.keys():
            evidence = {}
            result['object_mapping'] = evidence
            try:
                mapping = map_objects(candidate_world, gt_world, instruction, images, model, 'medium', evidence)
            finally:
                if evidence.get('attempts'):
                    result['mapping_source'] = 'vlm'
                elif evidence.get('fixed'):
                    result['mapping_source'] = 'deterministic_identity_rule'
        result['object_mapping'].update(mapping=mapping, unmapped=sorted(required - mapping.keys()))
        if required - mapping.keys():
            result['reason'] = 'Required plan objects have no confirmed correspondence; no actions were dropped.'
            return result

        with _LOCK:
            identity = {obj.name: obj.name for obj in official.all_objects}
            signatures = {action.name: _action_signature(action, identity) for action in official.actions}
            mapped_actions = {}
            for instance in plan.actions:
                action = instance.action
                if action.name in mapped_actions:
                    continue
                signature = _action_signature(action, mapping)
                matches = [name for name, key in signatures.items() if key == signature]
                if action.name in matches:
                    target = action.name
                elif len(matches) == 1:
                    target = matches[0]
                else:
                    result['reason'] = f'No unique official action with matching preconditions/effects for {action.name}.'
                    return result
                mapped_actions[action.name] = official.action(target)
                result['action_mapping'][action.name] = target
            translated = []
            for instance in plan.actions:
                action = mapped_actions[instance.action.name]
                values = [official.object(mapping[value.object().name]) for value in instance.actual_parameters]
                try:
                    translated.append(ActionInstance(action, values))
                except Exception as error:
                    result.update(status='FAIL', **{'pass': False}, reason=f'Invalid official action binding: {error}')
                    return result
            with SequentialSimulator(problem=official) as simulator:
                state = simulator.get_initial_state()
                for index, action in enumerate(translated, 1):
                    if not simulator.is_applicable(state, action):
                        result.update(status='FAIL', **{'pass': False}, reason=f'Official precondition fails at step {index}: {action}')
                        return result
                    state = simulator.apply(state, action)
                    if state is None:
                        raise RuntimeError(f'Simulator returned no state at step {index}')
                    result['executed_steps'] = index
                passed = simulator.is_goal(state)
                result.update(status='PASS' if passed else 'FAIL', **{'pass': passed},
                              reason='Official goal reached.' if passed else 'Official goal not reached.')
    except Exception as error:
        result['reason'] = f'Validation unavailable: {type(error).__name__}: {error}'
    return result
