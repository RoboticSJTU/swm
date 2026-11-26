"""Replay bundled game rules in a separate Python process."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().with_name("_v7")


def verify_native(scenario, plan, *, check_branches=False):
    """Return the independently generated, projected native trajectory."""
    payload = dict(scenario=scenario, plan=list(plan), check_branches=check_branches)
    run = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        input=json.dumps(payload), capture_output=True, text=True, timeout=90,
    )
    if run.returncode:
        raise ValueError("Bundled native contract failed: " + run.stderr.strip()[-3000:])
    result = json.loads(run.stdout)
    result["trace"] = [{tuple(fact) for fact in state} for state in result["trace"]]
    return result


def _worker(payload):
    scenario = payload["scenario"]
    if not (SOURCE / "rules_v2.py").is_file():
        raise ValueError("Packaged reference runtime is missing: " + str(SOURCE))
    from _v7.rules_v2 import catalog, construct
    from _v7.cleanup import project_native_state
    from _v7.rules_v2_native import successors, invariant, goal
    from _v7.model import reviewed_instruction

    # catalog() registers the additive physical policies used by construct().
    catalog()
    spec = scenario["v7_task"]
    task = construct(spec["cfg"], spec["seed"], spec["slot"], spec["alias"])
    task["instruction"] = reviewed_instruction(scenario, task["instruction"])
    if task["family"] != scenario["family"] or task["alias"] != scenario["game"]:
        raise ValueError("Bundled task identity differs from scenario")
    if task["cfg"]["rule_id"] != scenario["rule_id"]:
        raise ValueError("Bundled rule ID differs from scenario")
    for name in ("domain", "problem", "instruction"):
        if task[name] != scenario[name]:
            raise ValueError("Bundled reconstructed " + name + " differs from scenario")

    native_task = dict(task, initial=task["native_initial"])
    state = set(task["native_initial"])
    trace = [project_native_state(task, state)]
    branches = negatives = 0
    check_branches = payload.get("check_branches")
    if check_branches:
        # This PDDL interpreter is separate from the procedural game oracle.
        from _v7.model import Evaluator, parse_sexpr, section
        domain, problem = parse_sexpr(task["domain"]), parse_sexpr(task["problem"])
        objects = section(problem, ":objects")[1:] + section(domain, ":constants", [":constants"])[1:]

    plan = [tuple(action) for action in payload["plan"]]
    for index in range(len(plan) + 1):
        invariant(native_task, state)
        choices = successors(native_task, state)
        if check_branches:
            model_state = project_native_state(task, state)
            evaluator = Evaluator(model_state, objects)
            legal = evaluator.applicable(domain)
            if set(legal) != set(choices):
                raise ValueError("Native/PDDL legal alternatives differ at step " + str(index))
            for action, next_state in choices.items():
                invariant(native_task, next_state)
                fields, binding = legal[action]
                add, delete = set(), set()
                evaluator.effect(fields[":effect"], binding, add, delete)
                if (model_state - delete) | add != project_native_state(task, next_state):
                    raise ValueError("Native/PDDL branch effect differs: " + repr(action))
                branches += 1
            # Wrong-category/role substitutions are checked against the native
            # verdict, including variants that share all action names.
            for action in sorted(choices)[:3]:
                fields, _ = legal[action]
                for position, value in enumerate(action[1:]):
                    for other in objects[:12]:
                        if other == value:
                            continue
                        probe = (*action[:position + 1], other, *action[position + 2:])
                        if probe not in choices:
                            binding = dict(zip(fields[":parameters"], probe[1:]))
                            if evaluator.condition(fields[":precondition"], binding):
                                raise ValueError("Native/PDDL illegal argument differs: " + repr(probe))
                            negatives += 1
                            break
        if index == len(plan):
            break
        action = plan[index]
        if action not in choices:
            raise ValueError(f"Native illegal action at step {index + 1}: {action}")
        state = choices[action]
        trace.append(project_native_state(task, state))
    if not goal(native_task, state):
        raise ValueError("Independent native goal is not reached")
    return dict(trace=[[list(fact) for fact in sorted(state)] for state in trace],
                native_goal=True, checked_states=len(trace),
                legal_branches=branches, illegal_argument_probes=negatives)


if __name__ == "__main__":
    sys.path.insert(0, str(SOURCE.parent))
    try:
        print(json.dumps(_worker(json.load(sys.stdin))))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
