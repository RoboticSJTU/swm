from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from swm.llm import call_gpt_json
from swm.pddl.init_state_precheck import (
    _identity_compatible,
    compare_initial_states,
    implicit_running_device_start_conflicts,
    map_objects,
    parse_problem_text,
    reference_contract_conflicts,
    unfinished_started_process_conflicts,
)
from swm.pddl.strips import (
    ActionSchema,
    GroundAction,
    apply_action,
    goals_satisfied,
    ground_plan,
    parse_domain,
    parse_plan,
    parse_problem,
    parse_problem_model,
    rollout,
)


def latest_round_problem(task_dir: Path) -> Path | None:
    """Return the problem from the highest numeric round that contains one."""
    candidates = []
    if task_dir.is_dir():
        for round_dir in task_dir.glob("round*"):
            suffix = round_dir.name.removeprefix("round")
            problem_path = round_dir / "problem.pddl"
            if suffix.isdigit() and problem_path.is_file():
                candidates.append((int(suffix), problem_path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _validated_vlm_result(result: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Judge response is not a JSON object")  # noqa: TRY004
    required_keys = {"reasoning", "pass", "feedback"}
    if set(result) != required_keys:
        raise ValueError(
            "Judge response must contain exactly reasoning, pass, and feedback"
        )

    reasoning = result["reasoning"]
    passed = result["pass"]
    feedback = result["feedback"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Judge response has invalid reasoning")
    if type(passed) is not bool:
        raise ValueError("Judge response pass must be a JSON boolean")
    if not isinstance(feedback, str):
        raise ValueError("Judge response feedback must be a string")  # noqa: TRY004

    reasoning = reasoning.strip()
    feedback = feedback.strip()
    if passed and feedback:
        raise ValueError("Passing judge response must have empty feedback")
    if not passed and not feedback:
        raise ValueError("Failing judge response must have non-empty feedback")

    return {"reasoning": reasoning, "pass": passed, "feedback": feedback}


def _call_validated_judge(
    model: str,
    prompt: str,
    first_img: Path,
    capture: dict | None = None,
) -> dict:
    last_error: Exception | None = None
    attempts = []
    for attempt in range(1, 4):
        attempt_capture = {"attempt": attempt}
        try:
            call_kwargs = {"attempts": 1}
            if capture is not None:
                call_kwargs["capture"] = attempt_capture
            result = _validated_vlm_result(
                call_gpt_json(
                    model,
                    prompt,
                    [first_img],
                    **call_kwargs,
                )
            )
            attempts.append(attempt_capture)
            if capture is not None:
                capture["attempts"] = attempts
            return result
        except (RuntimeError, ValueError) as error:
            attempt_capture["error_type"] = type(error).__name__
            attempt_capture["error"] = str(error)
            attempts.append(attempt_capture)
            last_error = error
    if capture is not None:
        capture["attempts"] = attempts
    raise ValueError(
        "Judge did not return the required flat judge schema after 3 attempts: "
        f"{last_error}"
    )


def _evaluated_symbolic_trace(
    candidate_plan: str,
    predicted_domain: str | Path | None,
    pddl_plan: str | Path | None,
    predicted_problem: str | Path | None = None,
) -> str:
    if not isinstance(predicted_domain, Path) or not isinstance(pddl_plan, Path):
        return candidate_plan

    try:
        raw_plan, _ = parse_plan(pddl_plan)
        if not raw_plan:
            return "Candidate trace contains zero actions."
        schemas = parse_domain(predicted_domain)
        object_types = None
        if isinstance(predicted_problem, Path):
            object_types = parse_problem_model(
                predicted_problem,
                schemas,
            ).object_types
        actions = ground_plan(raw_plan, schemas, object_types)
    except (OSError, KeyError, ValueError, NotImplementedError) as error:
        return f"{candidate_plan}\n\nSymbolic details unavailable: {error}"

    dynamic_predicates = _dynamic_predicates(schemas)

    def format_literals(literals: set[tuple[str, ...]]) -> list[str]:
        return sorted(
            f"{literal[0]}({', '.join(literal[1:])})"
            for literal in literals
            if literal[0] in dynamic_predicates
        )

    lines = []
    for index, action in enumerate(actions, start=1):
        manipulators = [
            argument
            for argument in action.args
            if any(part in {"arm", "hand", "gripper"} for part in argument.split("_"))
        ]
        objects = [argument for argument in action.args if argument not in manipulators]
        before = format_literals(action.pre_pos)
        before.extend(f"not {literal}" for literal in format_literals(action.pre_neg))
        changes = [f"+{literal}" for literal in format_literals(action.add_eff)]
        changes.extend(f"-{literal}" for literal in format_literals(action.del_eff))

        action_text = f"{action.name}({', '.join(objects)})"
        if manipulators:
            action_text += f" with {' and '.join(manipulators)}"
        lines.append(f"{index}. {action_text}")
        if before:
            lines.append(f"   Before: {', '.join(before)}")
        if changes:
            lines.append(f"   State change: {', '.join(changes)}")
    return "\n".join(lines)


def _render_judge_prompt(
    instruction: str,
    kf_actions: str,
    candidate_plan: str,
    predicted_domain: str | Path | None,
    predicted_problem: str | Path | None,
    ground_truth_problem: str | Path | None,
    pddl_plan: str | Path | None,
) -> str:
    findings = []
    # Candidate initial-state facts and findings derived from them must never
    # enter the Judge VLM prompt.
    if isinstance(predicted_domain, Path) and isinstance(pddl_plan, Path):
        findings.extend(
            f"- Device: {conflict}"
            for conflict in implicit_running_device_start_conflicts(
                predicted_domain,
                pddl_plan,
                predicted_problem,
            )
        )
    if not findings:
        findings = [
            "GT PDDL was not supplied; only Candidate-only checks were available."
            if ground_truth_problem is None
            else "None."
        ]

    prompt_path = Path(__file__).parent.parent / "prompt_templates" / "pddl_judge.txt"
    return prompt_path.read_text(encoding="utf-8").format(
        instruction=instruction,
        kf_actions=kf_actions,
        programmatic_findings="\n".join(findings),
        evaluated_symbolic_trace=_evaluated_symbolic_trace(
            candidate_plan,
            predicted_domain,
            pddl_plan,
            predicted_problem,
        ),
    )


def _candidate_symbolic_failure(
    predicted_domain: Path,
    predicted_problem: Path,
    pddl_plan: Path,
) -> str | None:
    try:
        schemas = parse_domain(predicted_domain)
        problem = parse_problem_model(predicted_problem, schemas)
        raw_plan = parse_plan(pddl_plan)[0]
        if not raw_plan:
            return "Candidate PDDL plan contains zero actions."
        actions = ground_plan(raw_plan, schemas, problem.object_types)
        final_state = rollout(problem.init_state, actions)
    except (OSError, KeyError, NotImplementedError, TypeError, ValueError) as error:
        return f"Candidate PDDL plan is invalid: {error}"
    if not goals_satisfied(
        final_state,
        problem.goal_positive,
        problem.goal_negative,
    ):
        return "Candidate PDDL plan does not satisfy its own goal."
    return None


def _grounded_key(action) -> tuple:
    return (
        action.name,
        tuple(action.args),
        tuple(sorted(action.pre_pos)),
        tuple(sorted(action.pre_neg)),
        tuple(sorted(action.add_eff)),
        tuple(sorted(action.del_eff)),
        tuple(sorted(action.equality_preconditions)),
        tuple(sorted(action.inequality_preconditions)),
        action.parameter_types,
    )


@lru_cache(maxsize=1024)
def _reference_certificate(problem_path: Path) -> tuple:
    domain_path = problem_path.parent / "domain.pddl"
    plan_path = problem_path.parent / "plan.txt"
    schemas = parse_domain(domain_path)
    problem = parse_problem_model(problem_path, schemas)
    raw_plan = parse_plan(plan_path)[0]
    actions = ground_plan(raw_plan, schemas, problem.object_types)
    final = rollout(problem.init_state, actions)
    if not goals_satisfied(final, problem.goal_positive, problem.goal_negative):
        raise ValueError("reference plan does not satisfy the reference goal")
    return (
        schemas,
        problem.init_state,
        problem.goal_positive,
        problem.goal_negative,
        raw_plan,
        actions,
        problem.object_types,
    )


def symbolic_verdict(
    predicted_domain: Path,
    predicted_problem: Path,
    pddl_plan: Path,
    ground_truth_problem: Path,
) -> dict | None:
    try:
        (
            reference_schemas,
            reference_initial,
            reference_goal_positive,
            reference_goal_negative,
            reference_raw_plan,
            reference_actions,
            reference_object_types,
        ) = _reference_certificate(ground_truth_problem)

        candidate_schemas = parse_domain(predicted_domain)
        candidate_problem = parse_problem_model(predicted_problem, candidate_schemas)
        candidate_initial = candidate_problem.init_state
        candidate_goal_positive = candidate_problem.goal_positive
        candidate_goal_negative = candidate_problem.goal_negative
        candidate_raw_plan = parse_plan(pddl_plan)[0]
        candidate_actions = ground_plan(
            candidate_raw_plan,
            candidate_schemas,
            candidate_problem.object_types,
        )
        candidate_final = rollout(candidate_initial, candidate_actions)
        if not goals_satisfied(
            candidate_final, candidate_goal_positive, candidate_goal_negative
        ):
            return {
                "reasoning": "Candidate plan does not satisfy its own PDDL goal.",
                "pass": False,
                "feedback": "Regenerate a plan that satisfies the candidate PDDL goal.",
            }

        if list(map(_grounded_key, candidate_actions)) == list(
            map(_grounded_key, reference_actions)
        ):
            return {
                "reasoning": (
                    "Candidate grounded trace is identical to the verified "
                    "reference trace."
                ),
                "pass": True,
                "feedback": "",
            }

        if Counter((name, tuple(args)) for name, args in candidate_raw_plan) != Counter(
            (name, tuple(args)) for name, args in reference_raw_plan
        ):
            return None

        try:
            reordered_actions = ground_plan(
                candidate_raw_plan,
                reference_schemas,
                reference_object_types,
            )
            reordered_final = rollout(reference_initial, reordered_actions)
        except (KeyError, TypeError, ValueError):
            reordered_final = None
        if reordered_final is None or not goals_satisfied(
            reordered_final,
            reference_goal_positive,
            reference_goal_negative,
        ):
            return {
                "reasoning": (
                    "Candidate reorders the verified actions into a sequence that "
                    "is invalid under the verified reference model."
                ),
                "pass": False,
                "feedback": "Preserve the verified causal order of these actions.",
            }
    except (OSError, KeyError, NotImplementedError, TypeError, ValueError):
        return None
    return None


@dataclass(frozen=True)
class SymbolMapping:
    actions: dict[str, str]
    objects: dict[str, str]
    predicates: dict[str, str]


IDENTITY_TOKENS = {
    "black",
    "blue",
    "bottom",
    "brown",
    "cold",
    "front",
    "green",
    "gray",
    "hot",
    "left",
    "middle",
    "orange",
    "pink",
    "purple",
    "red",
    "right",
    "top",
    "white",
    "yellow",
}
NAME_ALIASES = {
    "adapter": "charger",
    "bottlecap": "bottle_cap",
    "carries": "holding",
    "countertop": "counter",
    "cupboard": "cabinet",
    "desktop": "desk",
    "disconnect": "unplug",
    "fridge": "refrigerator",
    "garbage": "trash",
    "garbage_bin": "trash_can",
    "grey": "gray",
    "grab": "pick",
    "grasp": "pick",
    "gripper": "hand",
    "inside": "in",
    "into": "in",
    "mug": "cup",
    "outlet": "socket",
    "rests_on": "on",
    "tap": "faucet",
    "trash_bin": "trash_can",
    "upon": "on",
    "within": "in",
    "worktop": "counter",
}
PHYSICAL_RELATIONS = {
    alias: meaning
    for meaning, aliases in {
        "above": "above over",
        "under": "beneath below under underneath",
        "holding": "carrying grasping holding",
        "closed": "closed shut",
        "near": "close_to near",
        "empty": "contains_nothing empty",
        "flat": "flat horizontal laid_flat lying_flat",
        "in": "in inside into within",
        "inserted": "inserted plugged_in seated_in_socket",
        "is_off": "is_off powered_off switched_off",
        "is_on": "is_on powered_on switched_on",
        "locked": "lock_engaged locked",
        "unlocked": "lock_released unlocked",
        "on": "on resting_on upon",
        "open": "open opened",
        "upright": "right_side_up standing_upright upright",
    }.items()
    for alias in aliases.split()
}
ACTION_METHODS = {
    alias: method
    for method, aliases in {
        "close": "close shut",
        "pour": "decant pour",
        "pick": "grab grasp pick",
        "place": "place put",
        "throw": "throw toss",
        "empty": "empty",
        "fill": "fill",
        "hold": "hold",
        "insert": "insert",
        "inspect": "inspect",
        "lift": "lift",
        "lock": "lock",
        "lower": "lower",
        "open": "open",
        "press": "press",
        "push": "push",
        "release": "release",
        "remove": "remove",
        "rotate": "rotate",
        "unlock": "unlock",
    }.items()
    for alias in aliases.split()
}


def _bind(
    forward: dict[str, str],
    reverse: dict[str, str],
    candidate: str,
    reference: str,
) -> bool:
    if candidate in forward and forward[candidate] != reference:
        return False
    if reference in reverse and reverse[reference] != candidate:
        return False
    forward[candidate] = reference
    reverse[reference] = candidate
    return True


def unique_structural_mapping(
    candidate: list[GroundAction],
    reference: list[GroundAction],
) -> SymbolMapping | None:
    """Find the unique rename under the trace's fixed step alignment."""
    if len(candidate) != len(reference):
        return None

    # action/object/predicate maps and their reverse maps
    initial = ({}, {}, {}, {}, {}, {})
    groups = []
    roles = (
        "pre_pos",
        "pre_neg",
        "add_eff",
        "del_eff",
        "equality_preconditions",
        "inequality_preconditions",
    )
    for left, right in zip(candidate, reference):
        if len(left.args) != len(right.args):
            return None
        if not _bind(initial[0], initial[1], left.name, right.name):
            return None
        for left_arg, right_arg in zip(left.args, right.args):
            if not _bind(initial[2], initial[3], left_arg, right_arg):
                return None
        for role in roles:
            left_literals = tuple(sorted(getattr(left, role)))
            right_literals = tuple(sorted(getattr(right, role)))
            if len(left_literals) != len(right_literals):
                return None
            if left_literals:
                groups.append((left_literals, right_literals))

    results = []

    def extend_literal(state, left, right):
        if len(left) != len(right):
            return None
        next_state = tuple(dict(values) for values in state)
        if not _bind(next_state[4], next_state[5], left[0], right[0]):
            return None
        for left_arg, right_arg in zip(left[1:], right[1:]):
            if not _bind(next_state[2], next_state[3], left_arg, right_arg):
                return None
        return next_state

    def search_group(group_index, state):
        if len(results) > 1:
            return
        if group_index == len(groups):
            mapping = SymbolMapping(
                actions=state[0],
                objects=state[2],
                predicates=state[4],
            )
            if mapping not in results:
                results.append(mapping)
            return

        left_literals, right_literals = groups[group_index]

        def match_literals(remaining_left, remaining_right, current):
            if len(results) > 1:
                return
            if not remaining_left:
                search_group(group_index + 1, current)
                return

            best_index = None
            best_options = None
            for left_index, left_literal in enumerate(remaining_left):
                options = []
                for right_index, right_literal in enumerate(remaining_right):
                    next_state = extend_literal(
                        current,
                        left_literal,
                        right_literal,
                    )
                    if next_state is not None:
                        options.append((right_index, next_state))
                if not options:
                    return
                if best_options is None or len(options) < len(best_options):
                    best_index = left_index
                    best_options = options

            next_left = remaining_left[:best_index] + remaining_left[best_index + 1 :]
            for right_index, next_state in best_options:
                next_right = (
                    remaining_right[:right_index] + remaining_right[right_index + 1 :]
                )
                match_literals(next_left, next_right, next_state)

        match_literals(left_literals, right_literals, state)

    search_group(0, initial)
    return results[0] if len(results) == 1 else None


def tokens(value: str) -> list[str]:
    value = re.sub(r"([a-z])([A-Z])", r"\1_\2", value)
    return re.findall(r"[a-z]+|[0-9]+", value.lower())


def _identity(value: str) -> frozenset[str]:
    return frozenset(
        "gray" if token == "grey" else token
        for token in tokens(value)
        if token in IDENTITY_TOKENS or token.isdigit()
    )


def _action_method(value: str) -> str | None:
    parts = tokens(value)
    if not parts:
        return None
    if (
        len(parts) > 1
        and parts[0] in {"power", "switch", "turn"}
        and parts[1] in {"on", "off"}
    ):
        return "turn_" + parts[1]
    if parts[0] == "turn":
        return "rotate"
    if parts[0] == "push" and any(
        token in {"button", "control"} for token in parts[1:]
    ):
        return "press"
    return ACTION_METHODS.get(parts[0])


def _physical_meanings(value: str) -> set[str]:
    parts = tokens(value)
    meanings = set()
    for alias, meaning in PHYSICAL_RELATIONS.items():
        alias_parts = tokens(alias)
        width = len(alias_parts)
        if any(
            parts[index : index + width] == alias_parts
            for index in range(len(parts) - width + 1)
        ):
            meanings.add(meaning)
    return meanings


def semantic_conflicts(mapping: SymbolMapping) -> list[str]:
    conflicts = []
    for candidate, reference in mapping.objects.items():
        candidate_identity = _identity(candidate)
        reference_identity = _identity(reference)
        if (
            candidate_identity
            and reference_identity
            and candidate_identity != reference_identity
        ):
            conflicts.append(f"{candidate} -> {reference}")
    for candidate, reference in mapping.predicates.items():
        candidate_relations = _physical_meanings(candidate)
        reference_relations = _physical_meanings(reference)
        if (
            candidate_relations
            and reference_relations
            and candidate_relations != reference_relations
        ):
            conflicts.append(f"{candidate} -> {reference}")
    for candidate, reference in mapping.actions.items():
        candidate_method = _action_method(candidate)
        reference_method = _action_method(reference)
        if (
            candidate_method is not None
            and reference_method is not None
            and candidate_method != reference_method
        ):
            conflicts.append(f"{candidate} -> {reference}")
        candidate_relations = _physical_meanings(candidate)
        reference_relations = _physical_meanings(reference)
        spatial = {"above", "below", "in", "on", "under"}
        candidate_spatial = candidate_relations & spatial
        reference_spatial = reference_relations & spatial
        if (
            candidate_spatial
            and reference_spatial
            and candidate_spatial != reference_spatial
        ):
            conflicts.append(f"{candidate} -> {reference}")
    return sorted(set(conflicts))


def _canonical_name(value: str) -> tuple[str, ...]:
    normalized = "_".join(tokens(value))
    for alias, canonical in sorted(
        NAME_ALIASES.items(),
        key=lambda item: len(tokens(item[0])),
        reverse=True,
    ):
        normalized = re.sub(
            rf"(^|_){re.escape(alias)}(?=_|$)",
            lambda match, replacement=canonical: match.group(1) + replacement,
            normalized,
        )
    return tuple(NAME_ALIASES.get(token, token) for token in tokens(normalized))


def _canonical_action_name(value: str) -> tuple[str, ...]:
    parts = tokens(value)
    method = _action_method(value)
    if method is None:
        return _canonical_name(value)
    consumed = (
        2
        if len(parts) > 1
        and parts[0] in {"power", "switch", "turn"}
        and parts[1] in {"on", "off"}
        else 1
    )
    return (method,) + tuple(
        NAME_ALIASES.get(token, token) for token in parts[consumed:]
    )


def _canonical_predicate_name(value: str) -> tuple[str, ...]:
    if value in PHYSICAL_RELATIONS:
        return tuple(tokens(PHYSICAL_RELATIONS[value]))
    return _canonical_name(value)


def obvious_renaming(mapping: SymbolMapping) -> bool:
    return (
        all(
            _canonical_action_name(candidate) == _canonical_action_name(reference)
            for candidate, reference in mapping.actions.items()
        )
        and all(
            _canonical_name(candidate) == _canonical_name(reference)
            for candidate, reference in mapping.objects.items()
        )
        and all(
            _canonical_predicate_name(candidate) == _canonical_predicate_name(reference)
            for candidate, reference in mapping.predicates.items()
        )
    )


def mapping_changes(mapping: SymbolMapping) -> dict[str, dict[str, str]]:
    def changed(values: dict[str, str]) -> dict[str, str]:
        return {
            candidate: reference
            for candidate, reference in values.items()
            if "".join(tokens(candidate)) != "".join(tokens(reference))
        }

    return {
        "actions": changed(mapping.actions),
        "objects": changed(mapping.objects),
        "predicates": changed(mapping.predicates),
    }


def local_mapping_verdict(mapping: SymbolMapping) -> bool | None:
    changes = mapping_changes(mapping)
    if not any(changes.values()) or obvious_renaming(mapping):
        return True
    if semantic_conflicts(mapping):
        return False
    return None


def verify_mapping_with_qwen(
    instruction: str,
    candidate_plan: str,
    reference_plan: str,
    mapping: SymbolMapping,
) -> bool:
    local_verdict = local_mapping_verdict(mapping)
    if local_verdict is not None:
        return local_verdict

    prompt = f"""Check whether all Candidate -> Reference symbol renamings preserve the same physical meaning.

Instruction: {instruction}

Candidate plan:
{candidate_plan.strip()}

Reference plan:
{reference_plan.strip()}

Renamings:
{json.dumps(mapping_changes(mapping), ensure_ascii=False, sort_keys=True)}

Allow spelling variants, abbreviations, and true synonyms. Judge changed names by their ordinary physical meaning, not by how the trace uses them or because the traces align. A wrong physical object kind is not a synonym or harmless misnomer. Reject changed object identity, relation, action, or method. If uncertain, reject.

Return exactly JSON: {{"equivalent": true or false}}"""
    result = call_gpt_json(
        "Qwen3.8-27B",
        prompt,
        attempts=1,
    )
    return (
        set(result) == {"equivalent"}
        and type(result["equivalent"]) is bool
        and result["equivalent"]
    )


@dataclass(frozen=True)
class ReplayCertificate:
    decision: str
    reason: str
    matched_actions: tuple[str, ...] = ()


def _object_mapping(candidate_parsed, reference_parsed) -> dict[str, str]:
    """Add only unique exact spelling variants to the conservative mapper."""
    mapping = map_objects(candidate_parsed, reference_parsed)
    available = set(reference_parsed.objects) - set(mapping.values())
    proposals = {}
    for candidate in candidate_parsed.objects - mapping.keys():
        matches = [
            reference
            for reference in available
            if tuple(sorted(tokens(candidate))) == tuple(sorted(tokens(reference)))
        ]
        if len(matches) == 1:
            proposals[candidate] = matches[0]
    claimed = Counter(proposals.values())
    for candidate, reference in proposals.items():
        if claimed[reference] == 1:
            mapping[candidate] = reference

    available = set(reference_parsed.objects) - set(mapping.values())
    proposals = {}
    for candidate in candidate_parsed.objects - mapping.keys():
        roles = candidate_parsed.unary_roles.get(candidate, frozenset())
        if not roles:
            continue
        matches = [
            reference
            for reference in available
            if reference_parsed.unary_roles.get(reference, frozenset()) == roles
        ]
        if len(matches) == 1:
            proposals[candidate] = matches[0]
    claimed = Counter(proposals.values())
    for candidate, reference in proposals.items():
        if claimed[reference] == 1:
            mapping[candidate] = reference
    return mapping


def _dynamic_predicates(schemas: dict[str, ActionSchema]) -> set[str]:
    return {
        literal[0]
        for schema in schemas.values()
        for literals in (schema.add_eff, schema.del_eff)
        for literal in literals
    }


def _canonical_predicate(name: str) -> str:
    return "_".join(_canonical_predicate_name(name))


def _relevant_effects(
    action: GroundAction,
    reference_predicates: set[str],
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    groups = []
    for literals in (action.add_eff, action.del_eff):
        relevant = []
        for literal in literals:
            predicate = _canonical_predicate(literal[0])
            if predicate not in reference_predicates:
                continue
            relevant.append((predicate, *literal[1:]))
        groups.append(tuple(sorted(relevant)))
    return groups[0], groups[1]


def _effect_mappings(
    candidate_effects: tuple[
        tuple[tuple[str, ...], ...],
        tuple[tuple[str, ...], ...],
    ],
    object_mapping: dict[str, str],
    reference_action: GroundAction,
) -> list[tuple[dict[str, str], set[tuple[str, ...]]]]:
    reference_effects = (
        tuple(sorted(reference_action.add_eff)),
        tuple(sorted(reference_action.del_eff)),
    )
    pending = [
        (group, literal)
        for group, literals in enumerate(candidate_effects)
        for literal in literals
    ]
    reverse_mapping = {
        reference: candidate for candidate, reference in object_mapping.items()
    }
    results = {}

    def search(
        index: int,
        mapping: dict[str, str],
        reverse: dict[str, str],
        used: tuple[set[tuple[str, ...]], set[tuple[str, ...]]],
    ) -> None:
        if index == len(pending):
            core_objects = {
                mapping[argument]
                for literals in candidate_effects
                for literal in literals
                for argument in literal[1:]
            }
            if not core_objects <= set(reference_action.args):
                return
            key = (
                tuple(sorted(mapping.items())),
                tuple(sorted(used[0])),
            )
            results[key] = (mapping, set(used[0]))
            return

        group, candidate = pending[index]
        for reference in reference_effects[group]:
            if reference in used[group]:
                continue
            if _canonical_predicate(reference[0]) != candidate[0] or len(
                reference
            ) != len(candidate):
                continue

            next_mapping = dict(mapping)
            next_reverse = dict(reverse)
            for candidate_object, reference_object in zip(candidate[1:], reference[1:]):
                if (
                    next_mapping.get(candidate_object, reference_object)
                    != reference_object
                ):
                    break
                if (
                    next_reverse.get(reference_object, candidate_object)
                    != candidate_object
                ):
                    break
                if candidate_object not in next_mapping and not _identity_compatible(
                    candidate_object, reference_object
                ):
                    break
                next_mapping[candidate_object] = reference_object
                next_reverse[reference_object] = candidate_object
            else:
                next_used = (set(used[0]), set(used[1]))
                next_used[group].add(reference)
                search(
                    index + 1,
                    next_mapping,
                    next_reverse,
                    next_used,
                )

    search(0, dict(object_mapping), reverse_mapping, (set(), set()))
    return list(results.values())


def _groundings(
    schema: ActionSchema,
    objects: set[str],
    object_types: dict[str, str],
    initial: set[tuple[str, ...]],
    dynamic_predicates: set[str],
):
    domains = []
    for parameter in schema.params:
        requirements = {
            literal[0]
            for literal in schema.pre_pos
            if len(literal) == 2
            and literal[1] == parameter
            and literal[0] not in dynamic_predicates
        }
        domains.append(
            sorted(
                obj
                for obj in objects
                if schema.type_hierarchy.is_subtype(
                    object_types.get(obj, "object"),
                    schema.param_types.get(parameter, "object"),
                )
                and all((predicate, obj) in initial for predicate in requirements)
            )
        )
    if any(not domain for domain in domains):
        return
    combinations = 1
    for domain in domains:
        combinations *= len(domain)
    if combinations > 100_000:
        return
    for arguments in itertools.product(*domains):
        yield ground_plan(
            [(schema.name, list(arguments))],
            {schema.name: schema},
            object_types,
        )[0]


def reference_replay_certificate(
    candidate_domain: Path,
    candidate_problem: Path,
    candidate_plan: Path,
    reference_problem: Path,
) -> ReplayCertificate:
    reference_dir = reference_problem.parent
    reference_domain = reference_dir / "domain.pddl"
    try:
        candidate_schemas = parse_domain(candidate_domain)
        reference_schemas = parse_domain(reference_domain)
        candidate_problem_model = parse_problem_model(
            candidate_problem,
            candidate_schemas,
        )
        reference_problem_model = parse_problem_model(
            reference_problem,
            reference_schemas,
        )
        candidate_actions = ground_plan(
            parse_plan(candidate_plan)[0],
            candidate_schemas,
            candidate_problem_model.object_types,
        )
        reference_actions = ground_plan(
            parse_plan(reference_dir / "plan.txt")[0],
            reference_schemas,
            reference_problem_model.object_types,
        )
        candidate_parsed = parse_problem_text(
            candidate_problem.read_text(encoding="utf-8"),
            candidate_domain.read_text(encoding="utf-8"),
        )
        reference_parsed = parse_problem_text(
            reference_problem.read_text(encoding="utf-8"),
            reference_domain.read_text(encoding="utf-8"),
        )
        object_mapping = _object_mapping(candidate_parsed, reference_parsed)
        reference_initial = reference_problem_model.init_state
        reference_goal_pos = reference_problem_model.goal_positive
        reference_goal_neg = reference_problem_model.goal_negative
    except (OSError, KeyError, NotImplementedError, TypeError, ValueError) as error:
        return ReplayCertificate("defer", f"parse_error:{type(error).__name__}")

    reference_objects = set(reference_parsed.objects)
    reference_dynamic = _dynamic_predicates(reference_schemas)
    reference_predicates = {
        _canonical_predicate(literal[0])
        for schema in reference_schemas.values()
        for literals in (
            schema.pre_pos,
            schema.pre_neg,
            schema.add_eff,
            schema.del_eff,
        )
        for literal in literals
    }
    reference_predicates.update(
        _canonical_predicate(literal[0])
        for literals in (
            reference_initial,
            reference_goal_pos,
            reference_goal_neg,
        )
        for literal in literals
    )
    state = set(reference_initial)
    matched = []
    matched_keys = []

    for index, candidate_action in enumerate(candidate_actions, start=1):
        candidate_effects = _relevant_effects(
            candidate_action,
            reference_predicates,
        )
        if not candidate_effects[0] and not candidate_effects[1]:
            return ReplayCertificate("defer", f"step_{index}:no_effect", tuple(matched))

        matches = {}
        for schema in reference_schemas.values():
            for reference_action in (
                _groundings(
                    schema,
                    reference_objects,
                    reference_problem_model.object_types,
                    reference_initial,
                    reference_dynamic,
                )
                or ()
            ):
                for next_mapping, projected_add in _effect_mappings(
                    candidate_effects,
                    object_mapping,
                    reference_action,
                ):
                    try:
                        apply_action(state, reference_action)
                    except ValueError:
                        continue
                    next_state = state - reference_action.del_eff
                    next_state.update(projected_add)
                    key = (
                        _grounded_key(reference_action),
                        tuple(sorted(next_mapping.items())),
                    )
                    matches[key] = (
                        reference_action,
                        next_state,
                        next_mapping,
                    )

        if len(matches) != 1:
            status = "no_grounding" if not matches else "ambiguous_grounding"
            return ReplayCertificate("defer", f"step_{index}:{status}", tuple(matched))
        reference_action, state, object_mapping = next(iter(matches.values()))
        matched.append(reference_action.to_line())
        matched_keys.append(_grounded_key(reference_action))

    reference_positions = {}
    for position, action in enumerate(reference_actions):
        reference_positions.setdefault(_grounded_key(action), []).append(position)
    last_position = -1
    for key in matched_keys:
        positions = [
            position
            for position in reference_positions.get(key, ())
            if position > last_position
        ]
        if key in reference_positions and not positions:
            return ReplayCertificate(
                "defer",
                "reference_order_conflict",
                tuple(matched),
            )
        if positions:
            last_position = positions[0]

    if not goals_satisfied(state, reference_goal_pos, reference_goal_neg):
        return ReplayCertificate("defer", "reference_goal_not_met", tuple(matched))
    return ReplayCertificate(
        "pass",
        "Every candidate step uniquely replays as an applicable reference action and the reference goal is satisfied.",
        tuple(matched),
    )


def _reference_verdict(
    instruction: str,
    candidate_plan: str,
    predicted_domain: Path,
    predicted_problem: Path,
    pddl_plan: Path,
    ground_truth_problem: Path,
) -> dict | None:
    verdict = symbolic_verdict(
        predicted_domain,
        predicted_problem,
        pddl_plan,
        ground_truth_problem,
    )
    if verdict is not None:
        return verdict

    reference_domain = ground_truth_problem.parent / "domain.pddl"
    reference_plan = ground_truth_problem.parent / "plan.txt"

    try:
        initial_state = compare_initial_states(
            predicted_problem.read_text(encoding="utf-8"),
            ground_truth_problem.read_text(encoding="utf-8"),
        )
        if initial_state.should_reject:
            conflict = initial_state.contradictions[0]
            return {
                "reasoning": "Reference-backed contradiction: " + conflict,
                "pass": False,
                "feedback": conflict,
            }
    except (OSError, TypeError, ValueError):
        pass

    try:
        conflicts = reference_contract_conflicts(
            predicted_domain,
            predicted_problem,
            pddl_plan,
            reference_domain,
            ground_truth_problem,
            reference_plan,
            instruction,
        )
        if conflicts:
            return {
                "reasoning": "Reference-backed contradiction: " + conflicts[0],
                "pass": False,
                "feedback": conflicts[0],
            }
    except (OSError, KeyError, NotImplementedError, TypeError, ValueError):
        pass

    try:
        reference_plan_text = reference_plan.read_text(encoding="utf-8")
        candidate_schemas = parse_domain(predicted_domain)
        reference_schemas = parse_domain(reference_domain)
        candidate_problem_model = parse_problem_model(
            predicted_problem,
            candidate_schemas,
        )
        reference_problem_model = parse_problem_model(
            ground_truth_problem,
            reference_schemas,
        )
        mapping = unique_structural_mapping(
            ground_plan(
                parse_plan(pddl_plan)[0],
                candidate_schemas,
                candidate_problem_model.object_types,
            ),
            ground_plan(
                parse_plan(reference_plan)[0],
                reference_schemas,
                reference_problem_model.object_types,
            ),
        )
    except (OSError, KeyError, NotImplementedError, TypeError, ValueError):
        mapping = None

    if mapping is not None:
        mapping_verdict = local_mapping_verdict(mapping)
        if mapping_verdict is True:
            return {
                "reasoning": "Candidate trace is identical to the verified reference trace after a structurally verified symbol renaming.",
                "pass": True,
                "feedback": "",
            }
        if mapping_verdict is None:
            try:
                accepted = verify_mapping_with_qwen(
                    instruction,
                    candidate_plan,
                    reference_plan_text,
                    mapping,
                )
            except Exception:  # noqa: BLE001
                accepted = False
            if accepted:
                return {
                    "reasoning": "Candidate trace is identical to the verified reference trace after a structurally and semantically verified symbol renaming.",
                    "pass": True,
                    "feedback": "",
                }
        return None

    replay = reference_replay_certificate(
        predicted_domain,
        predicted_problem,
        pddl_plan,
        ground_truth_problem,
    )
    if replay.decision == "pass":
        return {"reasoning": replay.reason, "pass": True, "feedback": ""}
    return None


def judge_pddl(
    model: str,
    first_img: Path,
    instruction: str,
    kf_actions: str,
    candidate_plan: str,
    predicted_problem: str | Path | None = None,
    ground_truth_problem: str | Path | None = None,
    predicted_domain: str | Path | None = None,
    pddl_plan: str | Path | None = None,
    capture: dict | None = None,
):
    candidate_plan = candidate_plan.strip()
    if not candidate_plan:
        raise ValueError("candidate_plan must be non-empty")

    candidate_paths = all(
        isinstance(source, Path)
        for source in (predicted_domain, predicted_problem, pddl_plan)
    )
    if candidate_paths:
        failure = _candidate_symbolic_failure(
            predicted_domain,
            predicted_problem,
            pddl_plan,
        )
        if failure is not None:
            if capture is not None:
                capture["decision_source"] = "candidate_symbolic_validation"
            return {
                "reasoning": failure,
                "pass": False,
                "feedback": failure,
            }
    if candidate_paths and isinstance(ground_truth_problem, Path):
        verdict = _reference_verdict(
            instruction,
            candidate_plan,
            predicted_domain,
            predicted_problem,
            pddl_plan,
            ground_truth_problem,
        )
        if verdict is not None:
            if capture is not None:
                capture["decision_source"] = "reference_symbolic_validation"
            return verdict

    if candidate_paths:
        conflicts = unfinished_started_process_conflicts(
            predicted_domain,
            predicted_problem,
            pddl_plan,
        )
        if conflicts:
            failure = "; ".join(conflicts)
            if capture is not None:
                capture["decision_source"] = "started_process_validation"
            return {
                "reasoning": "The Candidate leaves a task-started process active: "
                + failure,
                "pass": False,
                "feedback": "Stop the started process before completion. " + failure,
            }

    prompt = _render_judge_prompt(
        instruction=instruction,
        kf_actions=kf_actions,
        candidate_plan=candidate_plan,
        predicted_domain=predicted_domain,
        predicted_problem=predicted_problem,
        ground_truth_problem=ground_truth_problem,
        pddl_plan=pddl_plan,
    )
    if capture is not None:
        capture.update(
            {
                "decision_source": "vlm",
                "model": model,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    return _call_validated_judge(model, prompt, first_img, capture)
