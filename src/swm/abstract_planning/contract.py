"""Benchmark inputs, PDDL execution, and reference plan checks."""
from pathlib import Path
import re

from swm.pddl.strips import parse_sexpr, validate_untyped_pddl
from ._v7.model import (
    FREECELL_REVIEW, NEW, OLD, check_operation_list, names_only, normalize_scenario,
    operation_list, operation_names, reviewed_instruction,
)
from .model import apply_effect, condition, fields, replay, section
from .model import verify_plan as verify_source_plan

VERSION = "abstract_planning_test"
DATASETS = ["seen", "unseen_composition"]


def task_id(row):
    return row["dataset"] + "/" + row["task"]


def verify_plan(scenario, plan):
    """Check the public plan against the original game rules."""
    if "v7_task" not in scenario:
        return verify_source_plan(scenario, plan)

    from .v7_contract import verify_native

    trace = replay(scenario["domain"], scenario["problem"], plan)
    if trace != verify_native(scenario, plan)["trace"]:
        raise ValueError("Frozen v7 native trajectory differs from public PDDL")
    return trace


def check_parameters(domain):
    for action in parse_sexpr(domain):
        if isinstance(action, list) and action[:1] == [":action"]:
            params = fields(action)[":parameters"]
            assert len(params) == len(set(params)), "duplicate formal parameters"


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


def machine(domain, problem):
    validate_untyped_pddl(domain)
    validate_untyped_pddl(problem)
    d, p = parse_sexpr(domain), parse_sexpr(problem)
    operators = {}
    for action in d:
        if isinstance(action, list) and action[:1] == [':action']:
            name = action[1]
            if name in operators:
                raise ValueError('Duplicate action: ' + name)
            operators[name] = fields(action)
    objects = set(section(p, ':objects', [':objects'])[1:] + section(d, ':constants', [':constants'])[1:])
    arities = {a[0]:len(a)-1 for a in section(d, ':predicates', [':predicates'])[1:]}
    positive, negative = set(), set()
    for literal in section(p, ':init')[1:]:
        negated = isinstance(literal, list) and literal[:1] == ['not']
        atom = literal[1] if negated and len(literal)==2 else literal
        if (not isinstance(atom, list) or not atom or any(not isinstance(x, str) for x in atom)
                or atom[0] not in arities or len(atom)-1 != arities[atom[0]]
                or not set(atom[1:]) <= objects):
            raise ValueError('Initial state must contain declared Boolean ground literals with matching arity')
        (negative if negated else positive).add(tuple(atom))
    if positive & negative:
        raise ValueError('Initial state contains both positive and negative forms of the same fact')
    # Explicit negative facts agree with closed-world falsity; they must not
    # become positive state facts or influence predicate/action alignment.
    return dict(objects=sorted(objects), init=positive, goal=section(p, ':goal')[1],
                ops=operators, arities=arities)


def step(model, state, act):
    if act[0] not in model['ops']:
        raise ValueError('Unknown action: ' + act[0])
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
    trace = [model['init']]
    for index, act in enumerate(actions, 1):
        try:
            trace.append(step(model, trace[-1], act))
        except ValueError as error:
            return dict(executable=False, goal_reached=False, failed_step=index,
                        action=list(act), error=str(error), trace=trace)
    return dict(executable=True, goal_reached=condition(model['goal'], trace[-1], {}, model['objects']),
                trace=trace)
