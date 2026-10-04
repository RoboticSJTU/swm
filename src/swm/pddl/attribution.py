"""Attribute one failure using deterministic rules; VLM only maps object names."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import re
import json
import subprocess
import sys

from swm.llm import call_gpt_json
from swm.pddl.planner import _run_fast_downward, fast_downward_path
from swm.simulator.alignment.objects import align_worlds

from swm.pddl.strips import parse_domain, parse_problem_model
from swm.simulator.alignment.predicates import compile_world


LABELS = ("pddl_invalid", "init_error", "goal_error", "operator_missing", "operator_semantics", "else")
SPATIAL = {"in", "on"}
OPPOSITES = dict(pair for a, b in (
    ("open", "closed"), ("power_on", "power_off"), ("locked", "unlocked"),
    ("upright", "upside_down"), ("flat", "vertical"), ("wet", "dry"),
    ("folded", "unfolded"), ("enabled", "disabled"), ("cut", "uncut"),
) for pair in ((a, b), (b, a)))


def load_world(directory):
    schemas = parse_domain(directory / "domain.pddl")
    problem = parse_problem_model(directory / "problem.pddl", schemas)
    return compile_world(schemas, problem)


def object_payload(world):
    return {
        "objects": [{"name": o.name, "kinds": list(o.identity_kinds)} for o in world.objects],
        "init": [list(f) for f in sorted(world.facts)],
        "negative_init": [list(f) for f in sorted(world.negative_facts)],
    }


def validate_mapping(value, query_names, gt_names, fixed):
    if not isinstance(value, dict) or set(value) != {"objects"} or not isinstance(value["objects"], dict):
        raise ValueError("Expected objects mapping")
    returned = value["objects"]
    if not set(query_names) <= returned.keys() or set(returned) - set(query_names) - fixed.keys():
        raise ValueError("Every queried object must occur exactly once")
    mapped, occupied = dict(fixed), set(fixed.values())
    for candidate, target in value["objects"].items():
        if target is not None and not isinstance(target, str):
            raise ValueError("Mapping values must be GT names or null, never judgments")
        if candidate in fixed:
            if target != fixed[candidate]:
                raise ValueError("A fixed identity cannot be changed")
            continue  # Some providers echo exact fixed identities; no new decision.
        if target is None:
            continue
        if not isinstance(target, str) or target not in gt_names or target in occupied:
            raise ValueError("Mapping must use existing, distinct GT objects")
        mapped[candidate] = target
        occupied.add(target)
    return mapped


def lid_owners(world):
    """Unique cover/container pairs, for open/closed comparisons ONLY."""
    facts = world.facts
    names = {f[i] for f in facts for i in range(1, len(f))}
    kinds = {name: {f[0][5:] for f in facts if len(f) == 2 and f[1] == name and f[0].startswith("kind:")}
             for name in names}
    cover = {name for name in names if any(k.split("_")[-1] in {"lid", "cap", "door"} for k in kinds[name])
             or name.endswith(("_lid", "_cap", "_door"))}
    openable = {"container", "pot", "kettle", "bottle", "box", "thermos", "bowl", "cup", "microwave",
                "ice_maker", "washing_machine", "fridge", "refrigerator", "cabinet", "drawer"}
    possible = {}
    for lid in cover:
        parents = {f[2] for f in facts if len(f) == 3 and f[1] == lid
                   and f[0].removeprefix("static:") in {"on", "part_of", "attached_to"}
                   and kinds.get(f[2], set()) & openable}
        for suffix in ("_lid", "_cap", "_door"):
            if lid.endswith(suffix) and lid[:-len(suffix)] in names:
                parents.add(lid[:-len(suffix)])
        possible[lid] = parents
    # A container with multiple possible covers is ambiguous even if one cover
    # itself has a unique parent. Do not select a convenient match.
    counts = Counter(parent for parents in possible.values() for parent in parents)
    return {lid: next(iter(parents)) for lid, parents in possible.items()
            if len(parents) == 1 and counts[next(iter(parents))] == 1}


def check_init(candidate, gt, mapping):
    """Compare shared, explicitly represented states/locations; absence is skipped."""
    inverse = {target: source for source, target in mapping.items()}
    candidate_owners, gt_owners = lid_owners(candidate), lid_owners(gt)
    owner_mapping = {}
    for source, target in mapping.items():
        owner_mapping.setdefault(gt_owners.get(target, target), set()).add(candidate_owners.get(source, source))
    checks = []
    unary = [(f, True) for f in gt.facts if len(f) == 2 and not f[0].startswith(("kind:", "static:"))]
    unary += [(f, False) for f in gt.negative_facts if len(f) == 2 and not f[0].startswith(("kind:", "static:"))]
    for fact, positive in sorted(unary):
        predicate, target = fact
        targets = {target}
        sources = {inverse[target]} if target in inverse else set()
        if predicate in {"open", "closed"}:
            owner = gt_owners.get(target, target)
            targets = {owner} | {lid for lid, parent in gt_owners.items() if parent == owner}
            owners = owner_mapping.get(owner, set())
            sources = set()
            if len(owners) == 1:
                candidate_owner = next(iter(owners))
                sources = {candidate_owner} | {lid for lid, parent in candidate_owners.items() if parent == candidate_owner}
        source = inverse.get(target) or (sorted(sources)[0] if sources else None)
        item = {"kind": "unary_state", "gt": list(fact), "positive": positive, "candidate_object": source}
        if predicate in {"open", "closed"}:
            item.update(gt_state_objects=sorted(targets), candidate_state_objects=sorted(sources))
        if not sources:
            item.update(status="unknown", reason="GT object has no confirmed candidate mapping")
        else:
            family = {predicate, OPPOSITES.get(predicate)}
            relevant = {f for f in candidate.facts | candidate.negative_facts
                        if len(f) == 2 and f[1] in sources and f[0] in family}
            item["candidate_states"] = [list(f) for f in sorted(relevant)]
            item["candidate_negative_states"] = [list(f) for f in sorted(relevant & candidate.negative_facts)]
            gt_conflict = positive and any((OPPOSITES.get(predicate), obj) in gt.facts or (predicate, obj) in gt.negative_facts for obj in targets)
            if gt_conflict:
                item.update(status="unknown", reason="GT has contradictory states on the container/cover")
            elif not relevant:
                item.update(status="skipped", reason="state family is not explicitly represented in candidate init")
            else:
                contradiction = any((predicate, obj) in candidate.negative_facts or (OPPOSITES.get(predicate), obj) in candidate.facts for obj in sources) if positive else any((predicate, obj) in candidate.facts for obj in sources)
                item.update(status="fail" if contradiction else "pass",
                            reason="explicit states contradict" if contradiction else "shared explicit states agree")
        checks.append(item)

    subjects = sorted({f[1] for f in gt.facts if len(f) == 3 and f[0] in SPATIAL})
    for target in subjects:
        expected = {f[2] for f in gt.facts if len(f) == 3 and f[0] in SPATIAL and f[1] == target}
        source = inverse.get(target)
        actual = {f[2] for f in candidate.facts if len(f) == 3 and f[0] in SPATIAL and f[1] == source}
        mapped_areas = {mapping[a] for a in actual if a in mapping}
        item = {"kind": "spatial_area", "gt_object": target, "candidate_object": source,
                "gt_areas": sorted(expected), "candidate_areas": sorted(actual), "mapped_areas": sorted(mapped_areas)}
        if source is None:
            item.update(status="unknown", reason="GT subject has no confirmed candidate mapping")
        elif mapped_areas - expected:
            item.update(status="fail", reason="candidate area maps to a different GT object/area")
        elif not actual:
            item.update(status="skipped", reason="location is not explicitly represented in candidate init")
        elif actual - mapping.keys() or expected - inverse.keys():
            item.update(status="unknown", reason="source/target area mapping is incomplete")
        else:
            item.update(status="pass", reason="shared areas agree (in/on ignored; GT-only areas not required)")
        checks.append(item)
    compared = sum(c["status"] in {"pass", "fail"} for c in checks)
    status = "fail" if any(c["status"] == "fail" for c in checks) else "unknown" if not compared or any(c["status"] == "unknown" for c in checks) else "pass"
    return {"status": status, "checks": checks, "compared": compared,
            "skipped": sum(c["status"] == "skipped" for c in checks),
            "scope": "shared explicit unary state families and in/on areas; absent facts skipped, unmapped identities unknown; pass is not full init validation"}


def check_goal_error(candidate, gt, mapping):
    """Explicit reversed arguments or a different mapped destination in in/on goals."""
    candidate_goals = {f for f in candidate.goal_positive if len(f) == 3 and f[0] in SPATIAL}
    gt_goals = {f for f in gt.goal_positive if len(f) == 3 and f[0] in SPATIAL}
    mapped = {tuple([f[0], *(mapping[a] for a in f[1:])]): f for f in candidate.goal_positive
              if len(f) == 3 and f[0] in SPATIAL and all(a in mapping for a in f[1:])}
    matches = []
    for fact in sorted(gt_goals):
        if fact[1] == fact[2]:
            continue
        reverse = (fact[0], fact[2], fact[1])
        if reverse in mapped and fact not in mapped and reverse not in gt_goals:
            matches.append({"kind": "reversed_arguments", "gt_goal": list(fact),
                            "candidate_goal": list(mapped[reverse]), "mapped_candidate_goal": list(reverse)})
    for subject in sorted({f[1] for f in gt_goals}):
        expected = {f[2] for f in gt_goals if f[1] == subject}
        actual = sorted(f for f in candidate_goals if mapping.get(f[1]) == subject)
        # Missing goals, unknown destinations, and goals that also include a
        # correct destination do not establish that the destination was replaced.
        if not actual or any(f[2] not in mapping for f in actual):
            continue
        if {mapping[f[2]] for f in actual} & expected:
            continue
        for fact in actual:
            matches.append({"kind": "wrong_destination", "gt_object": subject,
                            "gt_goals": [list(f) for f in sorted(gt_goals) if f[1] == subject],
                            "gt_areas": sorted(expected), "candidate_goal": list(fact),
                            "mapped_candidate_goal": [fact[0], subject, mapping[fact[2]]]})
    return {"matched": bool(matches), "matches": matches,
            "scope": "positive in/on goals: same-predicate reversed arguments, or shared subject with confirmed different target area (in/on ignored); no missing-goal or negation inference"}


def action_names(path):
    """Strictly read action names, preserving repetitions and ignoring arguments."""
    names = []
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split(";", 1)[0].strip().lower()
        if not line:
            continue
        match = re.fullmatch(r"\(\s*([a-z][a-z0-9_-]*)(?:\s+[^()]*)?\s*\)", line)
        if not match:
            raise ValueError(f"Malformed plan line {number}: {line}")
        names.append(match[1])
    return names


def compare_plans(candidate_path, gt_path):
    try:
        candidate, gt = action_names(candidate_path), action_names(gt_path)
    except (OSError, ValueError) as error:
        return {"available": False, "reason": str(error), "order_only_difference": False}
    same_count = len(candidate) == len(gt)
    same_multiset = Counter(candidate) == Counter(gt)
    different_order = candidate != gt
    return {"available": True, "candidate_names": candidate, "gt_names": gt,
            "same_count": same_count, "same_multiset": same_multiset, "different_order": different_order,
            "order_only_difference": same_count and same_multiset and different_order}


def fd_category(solver):
    code = solver.get("returncode")
    if code is None:
        return "else"  # No returned FD code: do not invent an input error.
    if code in {0, 1, 2, 3, 10, 11, 12}:
        return None
    if code in {20, 21, 22, 23, 24} or code < 0:
        return "else"
    return "pddl_invalid"


def classify(solver, init, counts, judge_failed, plans, goal=None):
    """First matching rule wins. Unknown init is never treated as correct."""
    if label := fd_category(solver):
        return label, "fast_downward_returncode"
    if init["status"] == "fail":
        return "init_error", "gt_init_mismatch"
    if goal and goal.get("matched"):
        return "goal_error", "mapped_goal_arguments_or_destination_wrong"
    if counts["candidate"] < counts["gt"]:
        return "operator_missing", "candidate_operator_count_less_than_gt"
    if init["status"] != "pass":
        return "else", "init_not_confirmed"
    if judge_failed and plans.get("order_only_difference"):
        return "operator_semantics", "same_action_multiset_different_order"
    return "else", "no_rule_matched"


def run_solver(source, dest):
    """Keep the original judged plan intact; run FD in an attribution directory."""
    source, dest = Path(source).resolve(), Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "plan.txt").unlink(missing_ok=True)
    command = [sys.executable, str(fast_downward_path), "--overall-time-limit", "55s",
               "--plan-file", str(dest / "plan.txt"), str(source / "domain.pddl"),
               str(source / "problem.pddl"), "--search", "astar(lmcut())"]
    result = {"command": command, "returncode": None}
    try:
        stdout, stderr = _run_fast_downward(command, dest)
        result.update(returncode=0, stdout=stdout, stderr=stderr)
    except subprocess.CalledProcessError as error:
        result.update(returncode=error.returncode, stdout=error.stdout, stderr=error.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        result.update(error=f"{type(error).__name__}: {error}",
                      stdout=getattr(error, "stdout", ""), stderr=getattr(error, "stderr", ""))
    for key in ("stdout", "stderr"):
        if isinstance(result.get(key), bytes):
            result[key] = result[key].decode("utf-8", errors="replace")
    (dest / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: value for key, value in result.items() if key not in {"stdout", "stderr"}}


def map_objects(candidate, gt, instruction, image_path, model, reasoning_effort, evidence):
    fixed = align_worlds(candidate, gt).mapping()
    queries = sorted({o.name for o in candidate.objects} - fixed.keys())
    targets = sorted({o.name for o in gt.objects} - set(fixed.values()))
    evidence.update(model=model, reasoning_effort=reasoning_effort, fixed=fixed, attempts=[])
    if not queries:
        evidence.update(mapping=fixed, unmapped=[])
        return fixed
    images = list(image_path) if isinstance(image_path, (list, tuple)) else [image_path] if image_path is not None else []
    if not images or any(not Path(path).is_file() for path in images):
        raise ValueError("Object mapping requires all initial images")
    data = {"instruction": instruction, "fixed_mapping": fixed, "query_objects": queries,
            "eligible_gt_objects": targets, "candidate": object_payload(candidate), "gt": object_payload(gt)}
    template = Path(__file__).parents[1] / "prompt_templates/object_mapping.txt"
    prompt = template.read_text().replace("{mapping_input}", json.dumps(data, ensure_ascii=False, indent=2))
    for attempt in range(1, 3):
        capture = {"attempt": attempt}
        evidence["attempts"].append(capture)
        try:
            value = call_gpt_json(model, prompt, [Path(path) for path in images], attempts=1,
                                  response_format={"type": "json_object"}, max_tokens=8192,
                                  reasoning_effort=reasoning_effort, temperature=0, capture=capture)
            mapping = validate_mapping(value, queries, set(targets), fixed)
            evidence.update(mapping=mapping, unmapped=sorted(set(queries) - mapping.keys()))
            return mapping
        except Exception as error:
            capture["error"] = f"{type(error).__name__}: {error}"
    raise ValueError("Object mapping failed after two API/schema attempts")


def attribute_failure(candidate_dir, gt_task_dir, instruction, image_path, *,
                      model="Qwen3.8-27B", reasoning_effort="medium"):
    """Save one attribution.json for a non-passing task, including tasks without plans."""
    candidate_dir, gt_task_dir = Path(candidate_dir), Path(gt_task_dir)
    checks = {}
    label, reason = "else", "input_unavailable"
    result = {"model": model, "reasoning_effort": reasoning_effort, "checks": checks}
    try:
        judge_path = candidate_dir / "judge.json"
        judge = json.loads(judge_path.read_text()) if judge_path.is_file() else {}
        if judge.get("pass") is True:
            return None
        for name in ("domain.pddl", "problem.pddl"):
            if not (candidate_dir / name).is_file():
                raise FileNotFoundError(f"Missing candidate {name}")
        solver = run_solver(candidate_dir, candidate_dir / "attribution_solver")
        checks["fast_downward"] = solver
        if immediate := fd_category(solver):
            label, reason = immediate, "fast_downward_returncode"
        else:
            rounds = [p for p in gt_task_dir.glob("round*") if p.is_dir() and p.name[5:].isdigit()]
            if not rounds:
                raise FileNotFoundError(f"No GT round directory in {gt_task_dir}")
            reference = max(rounds, key=lambda p: int(p.name[5:]))
            result["gt_directory"] = str(reference)
            candidate, gt = load_world(candidate_dir), load_world(reference)
            counts = {"candidate": len(candidate.schemas), "gt": len(gt.schemas)}
            checks["operator_counts"] = counts
            mapping = {}
            checks["object_mapping"] = {}
            try:
                mapping = map_objects(candidate, gt, instruction, image_path, model,
                                      reasoning_effort, checks["object_mapping"])
                init = check_init(candidate, gt, mapping)
            except Exception as error:
                checks["mapping_error"] = f"{type(error).__name__}: {error}"
                init = {"status": "unknown", "checks": []}
            checks["init"] = init
            goal = check_goal_error(candidate, gt, mapping)
            checks["goal_error"] = goal
            plans = compare_plans(candidate_dir / "plan.txt", reference / "plan.txt")
            checks["plans"] = plans
            checks["judge_failed"] = judge.get("pass") is False
            label, reason = classify(solver, init, counts, checks["judge_failed"], plans, goal)
    except Exception as error:
        checks["input_error"] = f"{type(error).__name__}: {error}"
    result.update(label=label, reason=reason)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "attribution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
