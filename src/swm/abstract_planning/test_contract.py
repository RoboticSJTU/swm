"""Shared input layout and source-rule checks for the abstract game tests."""
from .model import fields, parse_sexpr, replay, verify_invariants

VERSION = "abstract_planning_test"
DATASETS = ["abs_a_id", "abs_a_scale", "abs_b_base", "abs_b_rule"]


def task_id(row):
    return row["dataset"] + "/" + row["task"]


def source_plan(scenario, plan):
    """Undo the public interface without changing the source game's argument roles."""
    interfaces = scenario.get("rules_migration", {}).get("action_mapping", {})
    if not interfaces:
        return list(plan)
    inverse = {info["name"]: (name, info) for name, info in interfaces.items()}
    result = []
    for action in plan:
        name, info = inverse[action[0]]
        binding = dict(zip(info["parameters"], action[1:]))
        result.append((name, *(binding[p] for p in info["old_parameters"])))
    return result


def project_state(scenario, state):
    proof = scenario.get("rules_migration", {})
    names = proof.get("predicate_mapping", {})
    inside = {a[1] for a in state if a[0] == "in"} if scenario["game"] == "briefcase" else set()
    result = {(names.get(a[0], a[0]), *a[1:]) for a in state
              if not (a[0] == "at" and a[1] in inside)}
    if scenario["game"] == "printer" and names:
        result |= {("is_off", a[1]) for a in state
                   if a[0] == "printer" and ("powered", a[1]) not in state}
    return result


def verify_plan(scenario, plan):
    """Check the public PDDL and replay the same moves in the original game."""
    from . import novel_games, visual_games

    trace = replay(scenario["domain"], scenario["problem"], plan)
    source = scenario.get("test_revision", {}).get("source_scenario", scenario)
    original = source_plan(scenario, plan) if source is not scenario else list(plan)
    original_trace = replay(source["domain"], source["problem"], original)
    if source is not scenario:
        assert trace == [project_state(scenario, state) for state in original_trace], "source trajectory differs"
    if source["game"] in novel_games.GAMES:
        novel_games.simulate(source, original)
    elif source["game"] in visual_games.GAMES:
        visual_games.simulate(source, original)
    else:
        verify_invariants(source, original, original_trace)
    return trace


def check_parameters(domain):
    for action in parse_sexpr(domain):
        if isinstance(action, list) and action[:1] == [":action"]:
            params = fields(action)[":parameters"]
            assert len(params) == len(set(params)), "duplicate formal parameters"
