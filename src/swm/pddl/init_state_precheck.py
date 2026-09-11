from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from swm.pddl.strips import (
    goals_satisfied,
    ground_plan,
    parse_domain,
    parse_plan,
    parse_problem_model,
)
from swm.pddl.strips import parse_problem as parse_strips_problem
from swm.pddl.typing import TypeHierarchy, typed_symbol_map

Literal = tuple[str, ...]

STATE_OPPOSITES = {
    "open": "closed",
    "closed": "open",
    "is_on": "is_off",
    "is_off": "is_on",
    "locked": "unlocked",
    "unlocked": "locked",
}
SPATIAL_PREDICATES = {"in", "on"}
IGNORED_UNARY_PREDICATES = {
    "clear",
    "hand_free",
    "holding",
    *STATE_OPPOSITES,
}
TOKEN_ALIASES = {
    "arm": "hand",
    "gripper": "hand",
    "countertop": "surface",
    "counter": "surface",
    "desk": "surface",
    "mug": "cup",
    "table": "surface",
    "worktop": "surface",
    "outlet": "socket",
    "tap": "faucet",
}
TYPE_ALIASES = {
    **TOKEN_ALIASES,
    "dish_rack": "rack",
    "water_bottle": "bottle",
    "detergent_bottle": "bottle",
    "laundry_detergent_bottle": "bottle",
    "bottle_cap": "cap",
    "detergent_cap": "cap",
    "washing_machine_door": "door",
}
IDENTITY_TOKENS = {
    "black",
    "blue",
    "bottom",
    "brown",
    "cold",
    "front",
    "green",
    "grey",
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


@dataclass(frozen=True)
class ParsedProblem:
    objects: frozenset[str]
    positive_init: frozenset[Literal]
    negative_init: frozenset[Literal]
    object_types: dict[str, str] = field(default_factory=dict)
    type_hierarchy: TypeHierarchy = field(default_factory=TypeHierarchy.object_only)

    @cached_property
    def unary_roles(self) -> dict[str, frozenset[str]]:
        roles: dict[str, set[str]] = {obj: set() for obj in self.objects}
        for obj, type_name in self.object_types.items():
            for ancestor in self.type_hierarchy.ancestors(type_name):
                if ancestor != "object":
                    roles.setdefault(obj, set()).add(_canonical_type(ancestor))
        for literal in self.positive_init:
            if len(literal) == 2 and literal[0] not in IGNORED_UNARY_PREDICATES:
                roles.setdefault(literal[1], set()).add(_canonical_type(literal[0]))
        return {obj: frozenset(values) for obj, values in roles.items()}


@dataclass
class PrecheckResult:
    decision: str
    reason: str
    contradictions: list[str] = field(default_factory=list)
    unmapped_predicted_objects: list[str] = field(default_factory=list)

    @property
    def should_reject(self) -> bool:
        return self.decision == "reject"


@dataclass
class ReferenceContext:
    predicted: ParsedProblem
    reference: ParsedProblem
    mapping: dict[str, str]
    candidate_schemas: dict
    reference_schemas: dict
    candidate_actions: list
    reference_actions: list
    candidate_raw_plan: list
    candidate_goal: set[Literal]
    reference_initial: set[Literal]
    reference_goal: set[Literal]
    reference_goal_negative: set[Literal]
    instruction: str


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def _canonical_type(value: str) -> str:
    normalized = "_".join(_tokenize(value))
    if normalized in TYPE_ALIASES:
        return TYPE_ALIASES[normalized]
    tokens = [TOKEN_ALIASES.get(token, token) for token in _tokenize(value)]
    return "_".join(tokens)


def _name_tokens(value: str) -> frozenset[str]:
    return frozenset(
        TOKEN_ALIASES.get(token, token) for token in _tokenize(value) if token != "of"
    )


def _identity_compatible(left: str, right: str) -> bool:
    left_identity = _name_tokens(left) & IDENTITY_TOKENS
    right_identity = _name_tokens(right) & IDENTITY_TOKENS
    return not left_identity or not right_identity or left_identity == right_identity


def _parse_sexpr(text: str) -> list[object]:
    text = re.sub(r";[^\n]*", "", text).lower()
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    stack: list[list[object]] = []
    roots: list[object] = []

    for token in tokens:
        if token == "(":
            expression: list[object] = []
            if stack:
                stack[-1].append(expression)
            else:
                roots.append(expression)
            stack.append(expression)
        elif token == ")":
            if not stack:
                raise ValueError("unexpected ')' in PDDL")
            stack.pop()
        elif stack:
            stack[-1].append(token)
        else:
            raise ValueError(f"atom outside an expression: {token}")

    if stack:
        raise ValueError("missing ')' in PDDL")
    if len(roots) != 1 or not isinstance(roots[0], list):
        raise ValueError("PDDL must contain one root expression")
    return roots[0]


def _literal(expression: object) -> Literal:
    if (
        not isinstance(expression, list)
        or not expression
        or any(not isinstance(item, str) for item in expression)
    ):
        raise ValueError(f"malformed :init literal: {expression!r}")
    return tuple(expression)


def parse_problem_text(text: str, domain_text: str | None = None) -> ParsedProblem:
    root = _parse_sexpr(text)
    if not root or root[0] != "define":
        raise ValueError("not a PDDL define expression")

    object_sections = [
        item
        for item in root[1:]
        if isinstance(item, list) and item and item[0] == ":objects"
    ]
    init_sections = [
        item
        for item in root[1:]
        if isinstance(item, list) and item and item[0] == ":init"
    ]
    if len(object_sections) > 1 or len(init_sections) != 1:
        raise ValueError(
            "problem must contain one :init and at most one :objects section"
        )

    hierarchy = TypeHierarchy.object_only()
    if domain_text is not None:
        domain = _parse_sexpr(domain_text)
        type_sections = [
            item
            for item in domain[1:]
            if isinstance(item, list) and item and item[0] == ":types"
        ]
        if len(type_sections) > 1:
            raise ValueError("domain contains duplicate :types sections")
        if type_sections:
            hierarchy = TypeHierarchy.from_declaration(type_sections[0][1:])

    if object_sections:
        order, object_types, _ = typed_symbol_map(
            object_sections[0][1:],
            context=":objects",
        )
        objects = set(order)
        if domain_text is None:
            hierarchy = TypeHierarchy(
                {
                    type_name: "object"
                    for type_name in set(object_types.values())
                    if type_name != "object"
                }
            )
        for type_name in object_types.values():
            hierarchy.require(type_name, ":objects")
    else:
        objects = set()
        object_types = {}
    positive: set[Literal] = set()
    negative: set[Literal] = set()
    for expression in init_sections[0][1:]:
        if isinstance(expression, list) and expression and expression[0] == "not":
            if len(expression) != 2:
                raise ValueError(f"malformed negative :init literal: {expression!r}")
            negative.add(_literal(expression[1]))
        else:
            positive.add(_literal(expression))

    referenced = {
        argument for literal in positive | negative for argument in literal[1:]
    }
    objects.update(referenced)
    for obj in referenced:
        object_types.setdefault(obj, "object")
    return ParsedProblem(
        frozenset(objects),
        frozenset(positive),
        frozenset(negative),
        object_types,
        hierarchy,
    )


def _relation_neighborhood(
    problem: ParsedProblem,
    obj: str,
    mapping: dict[str, str] | None = None,
) -> set[tuple[str, str, str]]:
    neighbors: set[tuple[str, str, str]] = set()
    for literal in problem.positive_init:
        if len(literal) != 3 or literal[0] not in SPATIAL_PREDICATES:
            continue
        _, subject, target = literal
        if subject == obj:
            neighbor = mapping.get(target, target) if mapping else target
            neighbors.add(("direct_area", "out", neighbor))
        if target == obj:
            neighbor = mapping.get(subject, subject) if mapping else subject
            neighbors.add(("direct_area", "in", neighbor))
    return neighbors


def _candidate_score(
    pred: str,
    gt: str,
    pred_problem: ParsedProblem,
    gt_problem: ParsedProblem,
    mapping: dict[str, str],
) -> float | None:
    if not _identity_compatible(pred, gt):
        return None

    pred_roles = pred_problem.unary_roles[pred]
    gt_roles = gt_problem.unary_roles[gt]
    role_overlap = pred_roles & gt_roles
    pred_tokens = _name_tokens(pred)
    gt_tokens = _name_tokens(gt)
    token_union = pred_tokens | gt_tokens
    token_score = (
        len(pred_tokens & gt_tokens) / len(token_union) if token_union else 0.0
    )

    mapped_neighbors = _relation_neighborhood(pred_problem, pred, mapping)
    gt_neighbors = _relation_neighborhood(gt_problem, gt)
    relation_overlap = len(mapped_neighbors & gt_neighbors)

    if pred_tokens == gt_tokens:
        return 1.0
    if pred_roles and gt_roles and pred_roles == gt_roles and len(pred_roles) == 1:
        return 0.90 + min(token_score, 0.09)
    if role_overlap and token_score >= 0.5:
        return 0.75 + min(token_score * 0.2, 0.2)
    if role_overlap and relation_overlap:
        return 0.75 + min(relation_overlap * 0.05, 0.15)
    if token_score >= 0.67:
        return 0.68 + min(token_score * 0.2, 0.2)
    return None


def map_objects(
    predicted: ParsedProblem,
    ground_truth: ParsedProblem,
) -> dict[str, str]:
    mapping = {obj: obj for obj in predicted.objects & ground_truth.objects}
    available_gt = set(ground_truth.objects) - set(mapping.values())

    while True:
        proposals: dict[str, list[tuple[float, str]]] = {}
        for pred in sorted(predicted.objects - mapping.keys()):
            candidates = []
            for gt in sorted(available_gt):
                scored = _candidate_score(pred, gt, predicted, ground_truth, mapping)
                if scored is not None:
                    candidates.append((scored, gt))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            proposals[pred] = candidates

        accepted: list[tuple[str, str]] = []
        for pred, candidates in proposals.items():
            if not candidates:
                continue
            best_score, best_gt = candidates[0]
            if len(candidates) > 1 and best_score - candidates[1][0] < 0.10:
                continue
            competitors = [
                other_candidates[0][0]
                for other_pred, other_candidates in proposals.items()
                if other_pred != pred
                and other_candidates
                and other_candidates[0][1] == best_gt
            ]
            if competitors and best_score - max(competitors) < 0.10:
                continue
            accepted.append((pred, best_gt))

        if not accepted:
            break
        for pred, gt in accepted:
            if pred in mapping or gt not in available_gt:
                continue
            mapping[pred] = gt
            available_gt.remove(gt)

    return mapping


def _reference_context(
    predicted_domain: Path,
    predicted_problem: Path,
    predicted_plan: Path,
    ground_truth_domain: Path,
    ground_truth_problem: Path,
    ground_truth_plan: Path,
    instruction: str,
) -> ReferenceContext | None:
    try:
        predicted_domain_text = predicted_domain.read_text(encoding="utf-8")
        reference_domain_text = ground_truth_domain.read_text(encoding="utf-8")
        predicted = parse_problem_text(
            predicted_problem.read_text(encoding="utf-8"),
            predicted_domain_text,
        )
        reference = parse_problem_text(
            ground_truth_problem.read_text(encoding="utf-8"),
            reference_domain_text,
        )
        mapping = map_objects(predicted, reference)
        candidate_schemas = parse_domain(predicted_domain)
        reference_schemas = parse_domain(ground_truth_domain)
        candidate_problem_model = parse_problem_model(
            predicted_problem,
            candidate_schemas,
        )
        reference_problem_model = parse_problem_model(
            ground_truth_problem,
            reference_schemas,
        )
        candidate_raw_plan = parse_plan(predicted_plan)[0]
        candidate_actions = ground_plan(
            candidate_raw_plan,
            candidate_schemas,
            candidate_problem_model.object_types,
        )
        reference_actions = ground_plan(
            parse_plan(ground_truth_plan)[0],
            reference_schemas,
            reference_problem_model.object_types,
        )
        candidate_goal = candidate_problem_model.goal_positive
        reference_initial = reference_problem_model.init_state
        reference_goal = reference_problem_model.goal_positive
        reference_goal_negative = reference_problem_model.goal_negative
    except (OSError, KeyError, NotImplementedError, ValueError):
        return None
    return ReferenceContext(
        predicted,
        reference,
        mapping,
        candidate_schemas,
        reference_schemas,
        candidate_actions,
        reference_actions,
        candidate_raw_plan,
        candidate_goal,
        reference_initial,
        reference_goal,
        reference_goal_negative,
        instruction,
    )


def _format_literal(literal: Literal) -> str:
    return f"({' '.join(literal)})"


def _mapped_literal(literal: Literal, mapping: dict[str, str]) -> Literal | None:
    mapped_arguments = []
    for argument in literal[1:]:
        if argument not in mapping:
            return None
        mapped_arguments.append(mapping[argument])
    return (literal[0], *mapped_arguments)


def _state_contradictions(
    predicted: ParsedProblem,
    ground_truth: ParsedProblem,
    mapping: dict[str, str],
) -> list[str]:
    contradictions: set[str] = set()
    gt_positive = set(ground_truth.positive_init)
    gt_negative = set(ground_truth.negative_init)

    checked_self_opposites: set[frozenset[Literal]] = set()
    for literal in predicted.positive_init:
        if literal[0] not in STATE_OPPOSITES or len(literal) != 2:
            continue
        opposite = (STATE_OPPOSITES[literal[0]], literal[1])
        if opposite in predicted.positive_init:
            pair = frozenset({literal, opposite})
            if pair not in checked_self_opposites:
                checked_self_opposites.add(pair)
                ordered = sorted(pair)
                contradictions.add(
                    "predicted init is self-contradictory: "
                    f"{_format_literal(ordered[0])} and {_format_literal(ordered[1])}"
                )
        mapped = _mapped_literal(literal, mapping)
        if mapped is None:
            continue
        gt_opposite = (STATE_OPPOSITES[literal[0]], *mapped[1:])
        if gt_opposite in gt_positive or mapped in gt_negative:
            contradictions.add(
                f"state mismatch: predicted {_format_literal(literal)} maps to "
                f"{_format_literal(mapped)}, but GT asserts {_format_literal(gt_opposite)}"
            )

    for literal in predicted.negative_init:
        if literal[0] not in STATE_OPPOSITES or len(literal) != 2:
            continue
        mapped = _mapped_literal(literal, mapping)
        if mapped is not None and mapped in gt_positive:
            contradictions.add(
                f"state mismatch: predicted (not {_format_literal(literal)}) maps to "
                f"(not {_format_literal(mapped)}), but GT asserts {_format_literal(mapped)}"
            )

    gt_holding = {
        (literal[1], literal[2])
        for literal in gt_positive
        if len(literal) == 3 and literal[0] == "holding"
    }
    gt_held_objects = {obj for _, obj in gt_holding}
    gt_located_objects = {
        literal[1]
        for literal in gt_positive
        if len(literal) == 3 and literal[0] in SPATIAL_PREDICATES
    }
    for literal in predicted.positive_init:
        if literal[0] == "hand_free" and len(literal) == 2:
            mapped_hand = mapping.get(literal[1])
            if mapped_hand is not None and any(
                hand == mapped_hand for hand, _ in gt_holding
            ):
                contradictions.add(
                    f"possession mismatch: predicted {_format_literal(literal)}, "
                    f"but GT asserts that {mapped_hand} is holding an object"
                )
        elif literal[0] == "holding" and len(literal) == 3:
            mapped_hand = mapping.get(literal[1])
            mapped_object = mapping.get(literal[2])
            if mapped_hand is not None and ("hand_free", mapped_hand) in gt_positive:
                contradictions.add(
                    f"possession mismatch: predicted {_format_literal(literal)}, "
                    f"but GT asserts (hand_free {mapped_hand})"
                )
            if mapped_object is not None and mapped_object in gt_located_objects:
                contradictions.add(
                    f"location mismatch: predicted {_format_literal(literal)}, "
                    f"but GT places {mapped_object} at a direct location"
                )
        elif literal[0] in SPATIAL_PREDICATES and len(literal) == 3:
            mapped_object = mapping.get(literal[1])
            if mapped_object is not None and mapped_object in gt_held_objects:
                contradictions.add(
                    f"location mismatch: predicted {_format_literal(literal)}, "
                    f"but GT asserts that {mapped_object} is held"
                )
    return sorted(contradictions)


def _spatial_locations(problem: ParsedProblem) -> dict[str, set[str]]:
    """Return immediate support/container targets, intentionally ignoring in/on."""
    locations: dict[str, set[str]] = {}
    for literal in problem.positive_init:
        if len(literal) == 3 and literal[0] in SPATIAL_PREDICATES:
            locations.setdefault(literal[1], set()).add(literal[2])
    return locations


def _spatial_contradictions(
    predicted: ParsedProblem,
    ground_truth: ParsedProblem,
    mapping: dict[str, str],
) -> list[str]:
    contradictions: set[str] = set()
    pred_locations = _spatial_locations(predicted)
    gt_locations = _spatial_locations(ground_truth)

    for pred_subject, pred_areas in pred_locations.items():
        gt_subject = mapping.get(pred_subject)
        if gt_subject is None or gt_subject not in gt_locations:
            continue
        mapped_pred_areas = {mapping[area] for area in pred_areas if area in mapping}
        if not mapped_pred_areas:
            continue
        gt_areas = gt_locations[gt_subject]
        if mapped_pred_areas.isdisjoint(gt_areas):
            contradictions.add(
                "direct-area mismatch for "
                f"'{pred_subject}' -> '{gt_subject}': predicted area(s) "
                f"{', '.join(sorted(pred_areas))} map to "
                f"{', '.join(sorted(mapped_pred_areas))}; GT direct area(s) "
                f"{', '.join(sorted(gt_areas))}"
            )
    return sorted(contradictions)


def compare_initial_states(
    predicted_problem: str,
    ground_truth_problem: str,
) -> PrecheckResult:
    try:
        predicted = parse_problem_text(predicted_problem)
        ground_truth = parse_problem_text(ground_truth_problem)
    except (TypeError, ValueError) as error:
        return PrecheckResult("defer", f"PDDL parsing failed: {error}")

    mapping = map_objects(predicted, ground_truth)
    contradictions = _state_contradictions(predicted, ground_truth, mapping)
    contradictions.extend(_spatial_contradictions(predicted, ground_truth, mapping))
    contradictions = sorted(set(contradictions))
    unmapped = sorted(predicted.objects - mapping.keys())

    if contradictions:
        return PrecheckResult(
            "reject",
            f"found {len(contradictions)} explicit initial-state contradiction(s)",
            contradictions,
            unmapped,
        )
    if not mapping:
        reason = "no objects could be mapped with sufficient confidence"
    elif unmapped:
        reason = "no proven contradiction; some predicted objects remain unmapped"
    else:
        reason = (
            "no proven contradiction; programmatic checks cannot establish equivalence"
        )
    return PrecheckResult("defer", reason, contradictions, unmapped)


def implicit_running_device_start_conflicts(
    predicted_domain: Path,
    predicted_plan: Path,
    predicted_problem: Path | None = None,
) -> list[str]:
    """Reject a running-device use that instead starts the device from off.

    An action explicitly named as using a running device cannot require that
    same device to be off and make it on.  Starting the device is an
    independently meaningful state transition, not an inherent effect of the
    already-running use.
    """
    try:
        schemas = parse_domain(predicted_domain)
        object_types = None
        if predicted_problem is not None:
            object_types = parse_problem_model(
                predicted_problem,
                schemas,
            ).object_types
        actions = ground_plan(
            parse_plan(predicted_plan)[0],
            schemas,
            object_types,
        )
    except (OSError, KeyError, NotImplementedError, ValueError):
        return []

    conflicts = []
    for action in actions:
        if "running" not in _tokenize(action.name):
            continue
        off_devices = {
            literal[1]
            for literal in action.pre_pos
            if len(literal) == 2 and literal[0] == "is_off"
        }
        on_devices = {
            literal[1]
            for literal in action.add_eff
            if len(literal) == 2 and literal[0] == "is_on"
        }
        for device in sorted(off_devices & on_devices):
            conflicts.append(
                f"action {action.to_line()} claims to use running {device} but "
                f"requires it off and starts it as an effect"
            )
    return conflicts


def unfinished_started_process_conflicts(
    predicted_domain: Path,
    predicted_problem: Path,
    predicted_plan: Path,
) -> list[str]:
    """Find a process started by the plan but left active outside the goal."""
    try:
        schemas = parse_domain(predicted_domain)
        problem = parse_problem_model(predicted_problem, schemas)
        actions = ground_plan(
            parse_plan(predicted_plan)[0],
            schemas,
            problem.object_types,
        )
        initial_state = problem.init_state
        goal = problem.goal_positive
    except (OSError, KeyError, NotImplementedError, ValueError):
        return []

    state = set(initial_state)
    started = set()
    for action in actions:
        started.update(
            literal
            for literal in action.add_eff
            if len(literal) == 2
            and literal[0] == "is_on"
            and literal not in state
        )
        state.difference_update(action.del_eff)
        state.update(action.add_eff)

    return [
        f"plan starts {device} but never stops it and the goal does not require it on"
        for _, device in sorted((started & state) - goal)
    ]


def _first_pickup_source_conflicts(context: ReferenceContext) -> list[str]:
    """Compare a fully mapped first pickup source with its reference binding.

    Direct pickup sources are not transitive.  The check defers when the
    candidate source is an explicit qualifier of the reference action (for
    example, a wall next to a table) or a named ``top`` whose reference source
    is modeled as a direct upper part of it.  All other fully mapped first-step
    source disagreements are causal scene conflicts.
    """
    predicted = context.predicted
    ground_truth = context.reference
    mapping = context.mapping
    candidate_actions = context.candidate_actions
    reference_actions = context.reference_actions
    if not candidate_actions or not reference_actions:
        return []

    candidate = candidate_actions[0]
    reference = reference_actions[0]
    pickup_tokens = {"pick", "take", "get", "remove"}
    if not (
        pickup_tokens & set(_tokenize(candidate.name))
        and pickup_tokens & set(_tokenize(reference.name))
    ):
        return []

    conflicts = []
    for literal in candidate.pre_pos:
        if (
            len(literal) != 3
            or literal[0] not in SPATIAL_PREDICATES
            or literal[1] not in mapping
        ):
            continue
        reference_sources = {
            requirement
            for requirement in reference.pre_pos
            if len(requirement) == 3
            and requirement[0] in SPATIAL_PREDICATES
            and requirement[1] == mapping[literal[1]]
        }
        if not reference_sources:
            continue
        reference_source_targets = {source[2] for source in reference_sources}
        mapped_source = mapping.get(literal[2])
        if mapped_source in reference_source_targets:
            continue

        if mapped_source is None:
            candidate_roles = predicted.unary_roles.get(literal[2], frozenset())
            reference_roles = [
                ground_truth.unary_roles.get(source, frozenset())
                for source in reference_source_targets
            ]
            if (
                candidate_roles
                and reference_roles
                and all(
                    roles and candidate_roles.isdisjoint(roles)
                    for roles in reference_roles
                )
            ):
                conflicts.append(
                    f"candidate first pickup {candidate.to_line()} uses direct "
                    f"source {_format_literal(literal)} of type "
                    f"{', '.join(sorted(candidate_roles))}, but reference first "
                    f"pickup {reference.to_line()} uses "
                    f"{', '.join(_format_literal(source) for source in sorted(reference_sources))}"
                )
            continue

        source_is_reference_qualifier = mapped_source in reference.args
        source_is_top_alias = "top" in _tokenize(candidate.name) and any(
            len(relation) == 3
            and relation[0] in SPATIAL_PREDICATES
            and relation[1] in reference_source_targets
            and relation[2] == mapped_source
            for relation in ground_truth.positive_init
        )
        if source_is_reference_qualifier or source_is_top_alias:
            continue
        conflicts.append(
            f"candidate first pickup {candidate.to_line()} requires "
            f"{_format_literal(literal)} mapping to direct source {mapped_source}, "
            f"but reference first pickup {reference.to_line()} requires "
            f"{', '.join(_format_literal(source) for source in sorted(reference_sources))}"
        )
    return conflicts


def _missing_required_state_transition_conflicts(
    context: ReferenceContext,
) -> list[str]:
    """Find an explicit reference task-state change absent from the candidate.

    A different action wording remains valid when it establishes any unary
    state for the same object.  This only rejects a plan that both omits the
    instruction's reference verb and leaves the object without any modeled
    non-spatial state transition at all.
    """
    ground_truth = context.reference
    mapping = context.mapping
    candidate_actions = context.candidate_actions
    reference_actions = context.reference_actions
    inverse_mapping = {
        reference: candidate for candidate, reference in mapping.items()
    }

    ignored_states = {"hand_free", "is_on", "is_off", "open", "closed"}
    instruction_tokens = set(_tokenize(context.instruction))
    candidate_action_tokens = {
        token for action in candidate_actions for token in _tokenize(action.name)
    }
    candidate_unary_changes = {
        literal[1]
        for action in candidate_actions
        for literal in action.add_eff
        if len(literal) == 2 and literal[0] not in ignored_states
    }

    conflicts = []
    for reference in reference_actions:
        action_tokens = _tokenize(reference.name)
        if (
            not action_tokens
            or action_tokens[0] not in instruction_tokens
            or action_tokens[0] in candidate_action_tokens
        ):
            continue
        for effect in reference.add_eff:
            if (
                len(effect) != 2
                or effect[0] in ignored_states
                or effect in ground_truth.positive_init
            ):
                continue
            candidate_object = inverse_mapping.get(effect[1])
            if candidate_object is None or candidate_object in candidate_unary_changes:
                continue
            conflicts.append(
                f"reference action {reference.to_line()} establishes "
                f"{_format_literal(effect)} for instruction verb "
                f"'{action_tokens[0]}', but the candidate neither uses that "
                f"verb nor establishes another unary state for {candidate_object}"
            )
    return sorted(set(conflicts))


def _timeline_conflicts(context: ReferenceContext) -> list[str]:
    """Find initial requirements that the reference establishes only later.

    A raw initial-state mismatch is only advisory because the reference PDDL can
    disagree with the image. This narrower check reports a causal contradiction
    only when the candidate requires a mapped fact before establishing it, while
    the executable reference plan creates that same fact later.
    """
    ground_truth = context.reference
    mapping = context.mapping
    candidate_actions = context.candidate_actions
    reference_actions = context.reference_actions

    established: set[Literal] = set()
    initial_requirements: set[Literal] = set()
    for action in candidate_actions:
        initial_requirements.update(action.pre_pos - established)
        established.difference_update(action.del_eff)
        established.update(action.add_eff)

    later_effects: dict[Literal, tuple[int, str]] = {}
    for index, action in enumerate(reference_actions, start=1):
        for literal in action.add_eff:
            later_effects.setdefault(literal, (index, action.to_line()))

    conflicts = []
    for literal in sorted(initial_requirements):
        mapped = _mapped_literal(literal, mapping)
        if mapped is None or mapped in ground_truth.positive_init:
            continue
        established_later = later_effects.get(mapped)
        if established_later is None:
            continue
        step, action = established_later
        conflicts.append(
            "candidate requires initial "
            f"{_format_literal(literal)} mapping to {_format_literal(mapped)}, "
            f"but GT does not assert it initially and establishes it only at "
            f"reference step {step}: {action}"
        )
    return conflicts


def _contract_tokens(text: str) -> frozenset[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return frozenset(
        TOKEN_ALIASES.get(token, token)
        for token in re.findall(r"[a-z]+|[0-9]+", text.lower())
        if token != "of"
    )


def _goal_specificity_conflicts(context: ReferenceContext) -> list[str]:
    """Detect an instruction-named destination collapsed to a generic target."""
    predicted_goal = context.candidate_goal
    ground_truth_goal = context.reference_goal
    mapping = context.mapping
    instruction_tokens = _contract_tokens(context.instruction)

    def same_object(candidate: str, reference: str) -> bool:
        return (
            candidate == reference
            or mapping.get(candidate) == reference
            or _contract_tokens(candidate) == _contract_tokens(reference)
        )

    conflicts = []
    for reference in sorted(ground_truth_goal):
        if len(reference) != 3 or reference[0] not in SPATIAL_PREDICATES:
            continue
        candidates = [
            literal
            for literal in predicted_goal
            if len(literal) == 3
            and literal[0] in SPATIAL_PREDICATES
            and same_object(literal[1], reference[1])
        ]
        if not candidates:
            continue
        required_tokens = _contract_tokens(reference[2]) & instruction_tokens
        if not required_tokens or any(
            required_tokens <= _contract_tokens(candidate[2])
            for candidate in candidates
        ):
            continue
        missing = min(
            (
                (
                    required_tokens - _contract_tokens(candidate[2]),
                    candidate,
                )
                for candidate in candidates
            ),
            key=lambda item: len(item[0]),
        )
        conflicts.append(
            f"reference goal {_format_literal(reference)} preserves the "
            f"instruction-named destination detail "
            f"{', '.join(sorted(missing[0]))}, but candidate goal "
            f"{_format_literal(missing[1])} omits it"
        )
    return conflicts


def _empty_candidate_trace_conflicts(context: ReferenceContext) -> list[str]:
    """Reject doing nothing when the verified initial state is not already a goal."""
    if context.candidate_raw_plan or goals_satisfied(
        context.reference_initial,
        context.reference_goal,
        context.reference_goal_negative,
    ):
        return []
    return [
        (
            "candidate trace has zero actions, but the verified reference initial "
            "state does not satisfy the task goal"
        )
    ]


def _quantity_conflicts(context: ReferenceContext) -> list[str]:
    """Detect a task quantity collapsed into fewer candidate objects."""
    predicted = context.predicted
    ground_truth = context.reference
    predicted_goal = context.candidate_goal
    ground_truth_goal = context.reference_goal

    def goal_objects_by_role(problem: ParsedProblem, goal: set[Literal]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for obj, roles in problem.unary_roles.items():
            if not any(obj in literal[1:] for literal in goal):
                continue
            for role in roles:
                result.setdefault(role, set()).add(obj)
        return result

    count_words = {
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    instruction_tokens = _tokenize(context.instruction)
    required_counts = {
        int(token) for token in instruction_tokens if token.isdigit() and int(token) > 1
    }
    required_counts.update(
        count_words[token] for token in instruction_tokens if token in count_words
    )
    if not required_counts:
        return []

    predicted_goal_roles = goal_objects_by_role(predicted, predicted_goal)
    ground_truth_goal_roles = goal_objects_by_role(ground_truth, ground_truth_goal)
    predicted_roles: dict[str, set[str]] = {}
    for obj, roles in predicted.unary_roles.items():
        for role in roles:
            predicted_roles.setdefault(role, set()).add(obj)

    conflicts = []
    for role, reference_objects in sorted(ground_truth_goal_roles.items()):
        candidate_objects = predicted_roles.get(role, set())
        candidate_goal_objects = predicted_goal_roles.get(role, set())
        if (
            len(reference_objects) in required_counts
            and candidate_goal_objects
            and 0 < len(candidate_objects) < len(reference_objects)
        ):
            conflicts.append(
                f"reference goal requires {len(reference_objects)} distinct {role} objects "
                f"({', '.join(sorted(reference_objects))}), but candidate represents only "
                f"{len(candidate_objects)} ({', '.join(sorted(candidate_objects))})"
            )
    return conflicts


def _bundled_transition_conflicts(context: ReferenceContext) -> list[str]:
    """Detect a shared action that invents a separate release and relocation."""
    candidate_schemas = context.candidate_schemas
    reference_schemas = context.reference_schemas
    candidate_action_names = {name for name, _ in context.candidate_raw_plan}
    conflicts = []
    for name in sorted(
        candidate_action_names & candidate_schemas.keys() & reference_schemas.keys()
    ):
        candidate = candidate_schemas[name]
        reference = reference_schemas[name]
        held_resources = {
            (literal[1], literal[2])
            for literal in candidate.pre_pos
            if literal[0] == "holding" and len(literal) == 3
        }
        if not held_resources or not any(
            literal[0] == "holding" and len(literal) == 3
            for literal in reference.pre_pos
        ):
            continue
        reference_releases = any(
            literal[0] == "holding" and len(literal) == 3
            for literal in reference.del_eff
        )
        reference_relocates = any(
            len(literal) == 3 and literal[0] in SPATIAL_PREDICATES
            for literal in reference.add_eff
        )
        if reference_releases or reference_relocates:
            continue
        for hand, resource in sorted(held_resources):
            releases = ("holding", hand, resource) in candidate.del_eff
            relocates = any(
                len(literal) == 3
                and literal[0] in SPATIAL_PREDICATES
                and literal[1] == resource
                for literal in candidate.add_eff
            )
            if releases and relocates:
                conflicts.append(
                    f"candidate action {name} releases and relocates held {resource}, "
                    f"but the same reference action preserves the held resource"
                )
    return conflicts


def _activation_support_conflicts(context: ReferenceContext) -> list[str]:
    """Compare required support relations for a shared device-start action."""
    mapping = context.mapping
    candidate_actions = context.candidate_actions
    reference_actions = context.reference_actions

    conflicts = []
    for candidate in candidate_actions:
        activated = {
            literal[1]
            for literal in candidate.add_eff
            if len(literal) == 2 and literal[0] == "is_on"
        }
        for device in activated:
            mapped_device = mapping.get(device)
            if mapped_device is None:
                continue
            candidate_supports = {
                mapped
                for literal in candidate.pre_pos
                if len(literal) == 3
                and literal[0] in SPATIAL_PREDICATES
                and literal[1] == device
                if (mapped := _mapped_literal(literal, mapping)) is not None
            }
            for reference in reference_actions:
                if (
                    reference.name != candidate.name
                    or ("is_on", mapped_device) not in reference.add_eff
                ):
                    continue
                reference_supports = {
                    literal
                    for literal in reference.pre_pos
                    if len(literal) == 3
                    and literal[0] in SPATIAL_PREDICATES
                    and literal[1] == mapped_device
                }
                if reference_supports and reference_supports.isdisjoint(
                    candidate_supports
                ):
                    conflicts.append(
                        f"candidate device-start action {candidate.to_line()} requires "
                        f"{', '.join(_format_literal(item) for item in sorted(candidate_supports)) or '<no direct support>'}, "
                        f"but matching reference action {reference.to_line()} requires "
                        f"{', '.join(_format_literal(item) for item in sorted(reference_supports))}"
                    )
    return conflicts


def _locked_goal_target_conflicts(context: ReferenceContext) -> list[str]:
    """Find a locked terminal state moved to a different mapped object.

    A lock is a direct state of its target.  When both PDDL problems map the
    intended target unambiguously, declaring the same ``locked`` state for a
    different object does not preserve the reference terminal condition.
    """
    mapping = context.mapping
    predicted_goal = context.candidate_goal
    ground_truth_goal = context.reference_goal
    inverse_mapping = {reference: candidate for candidate, reference in mapping.items()}
    candidate_targets = {
        literal[1]
        for literal in predicted_goal
        if len(literal) == 2 and literal[0] == "locked" and literal[1] in mapping
    }
    mapped_targets = {mapping[target] for target in candidate_targets}
    conflicts = []
    for reference_target in sorted(
        literal[1]
        for literal in ground_truth_goal
        if len(literal) == 2 and literal[0] == "locked"
    ):
        expected_candidate = inverse_mapping.get(reference_target)
        if expected_candidate is None or expected_candidate in candidate_targets:
            continue
        if mapped_targets - {reference_target}:
            conflicts.append(
                f"reference goal requires (locked {reference_target}), but candidate "
                f"locks {', '.join(sorted(candidate_targets))} instead"
            )
    return conflicts


def _collapsed_closed_locked_entity_conflicts(
    context: ReferenceContext,
) -> list[str]:
    """Find a reference close/lock pair collapsed onto one candidate entity."""
    predicted = context.predicted
    ground_truth = context.reference
    predicted_goal = context.candidate_goal
    ground_truth_goal = context.reference_goal

    reference_closed = {
        literal[1]
        for literal in ground_truth_goal
        if len(literal) == 2 and literal[0] == "closed"
    }
    reference_locked = {
        literal[1]
        for literal in ground_truth_goal
        if len(literal) == 2 and literal[0] == "locked"
    }
    candidate_closed = {
        literal[1]
        for literal in predicted_goal
        if len(literal) == 2 and literal[0] == "closed"
    }
    candidate_locked = {
        literal[1]
        for literal in predicted_goal
        if len(literal) == 2 and literal[0] == "locked"
    }
    if len(reference_closed) != 1 or len(reference_locked) != 1:
        return []
    reference_pair = next(iter(reference_closed)), next(iter(reference_locked))
    collapsed_targets = candidate_closed & candidate_locked
    if reference_pair[0] == reference_pair[1] or not collapsed_targets:
        return []

    common_roles = (
        ground_truth.unary_roles.get(reference_pair[0], frozenset())
        & ground_truth.unary_roles.get(reference_pair[1], frozenset())
    )
    for role in sorted(common_roles):
        reference_count = sum(
            role in roles for roles in ground_truth.unary_roles.values()
        )
        candidate_count = sum(
            role in roles for roles in predicted.unary_roles.values()
        )
        if candidate_count < reference_count:
            return [
                (
                    "reference separately closes "
                    f"{reference_pair[0]} and locks {reference_pair[1]}, but "
                    "candidate collapses both states onto "
                    f"{', '.join(sorted(collapsed_targets))}"
                )
            ]
    return []


def reference_contract_conflicts(
    predicted_domain: Path,
    predicted_problem: Path,
    predicted_plan: Path,
    ground_truth_domain: Path,
    ground_truth_problem: Path,
    ground_truth_plan: Path,
    instruction: str = "",
) -> list[str]:
    """Return narrow reference-backed semantic contradictions for a candidate."""
    conflicts = unfinished_started_process_conflicts(
        predicted_domain,
        predicted_problem,
        predicted_plan,
    )
    context = _reference_context(
        predicted_domain,
        predicted_problem,
        predicted_plan,
        ground_truth_domain,
        ground_truth_problem,
        ground_truth_plan,
        instruction,
    )
    if context is None:
        return sorted(set(conflicts))

    for check in (
        _empty_candidate_trace_conflicts,
        _goal_specificity_conflicts,
        _timeline_conflicts,
        _first_pickup_source_conflicts,
        _missing_required_state_transition_conflicts,
        _locked_goal_target_conflicts,
        _collapsed_closed_locked_entity_conflicts,
        _quantity_conflicts,
        _bundled_transition_conflicts,
        _activation_support_conflicts,
    ):
        conflicts.extend(check(context))
    return sorted(set(conflicts))
