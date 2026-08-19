from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

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

    @property
    def unary_roles(self) -> dict[str, frozenset[str]]:
        roles: dict[str, set[str]] = {obj: set() for obj in self.objects}
        for literal in self.positive_init:
            if len(literal) == 2 and literal[0] not in IGNORED_UNARY_PREDICATES:
                roles.setdefault(literal[1], set()).add(_canonical_type(literal[0]))
        return {obj: frozenset(values) for obj, values in roles.items()}


@dataclass(frozen=True)
class ObjectMatch:
    predicted: str
    ground_truth: str
    method: str
    score: float


@dataclass
class PrecheckResult:
    decision: str
    reason: str
    object_mapping: dict[str, str] = field(default_factory=dict)
    mapping_details: list[ObjectMatch] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    unmapped_predicted_objects: list[str] = field(default_factory=list)

    @property
    def should_reject(self) -> bool:
        return self.decision == "reject"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def _typed_symbols(items: Iterable[object]) -> set[str]:
    tokens = list(items)
    if any(not isinstance(item, str) for item in tokens):
        raise ValueError(":objects may contain only symbols and type annotations")

    objects: set[str] = set()
    pending: list[str] = []
    index = 0
    while index < len(tokens):
        token = str(tokens[index])
        if token != "-":
            pending.append(token)
            index += 1
            continue
        if not pending or index + 1 >= len(tokens):
            raise ValueError("malformed typed :objects section")
        objects.update(pending)
        pending = []
        index += 2
    objects.update(pending)
    return objects


def _literal(expression: object) -> Literal:
    if (
        not isinstance(expression, list)
        or not expression
        or any(not isinstance(item, str) for item in expression)
    ):
        raise ValueError(f"malformed :init literal: {expression!r}")
    return tuple(str(item) for item in expression)


def parse_problem_text(text: str) -> ParsedProblem:
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

    objects = _typed_symbols(object_sections[0][1:]) if object_sections else set()
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
    return ParsedProblem(frozenset(objects), frozenset(positive), frozenset(negative))


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
) -> tuple[float, str] | None:
    if not _identity_compatible(pred, gt):
        return None

    pred_roles = pred_problem.unary_roles.get(pred, frozenset())
    gt_roles = gt_problem.unary_roles.get(gt, frozenset())
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
        return 1.0, "normalized_name"
    if pred_roles and gt_roles and pred_roles == gt_roles and len(pred_roles) == 1:
        return 0.90 + min(token_score, 0.09), "unique_role"
    if role_overlap and token_score >= 0.5:
        return 0.75 + min(token_score * 0.2, 0.2), "role_and_name"
    if role_overlap and relation_overlap:
        return 0.75 + min(relation_overlap * 0.05, 0.15), "role_and_relation"
    if token_score >= 0.67:
        return 0.68 + min(token_score * 0.2, 0.2), "name_tokens"
    return None


def map_objects(
    predicted: ParsedProblem,
    ground_truth: ParsedProblem,
) -> tuple[dict[str, str], list[ObjectMatch]]:
    mapping: dict[str, str] = {}
    matches: list[ObjectMatch] = []
    available_gt = set(ground_truth.objects)

    while True:
        proposals: dict[str, list[tuple[float, str, str]]] = {}
        for pred in sorted(predicted.objects - mapping.keys()):
            candidates = []
            for gt in sorted(available_gt):
                scored = _candidate_score(pred, gt, predicted, ground_truth, mapping)
                if scored is not None:
                    score, method = scored
                    candidates.append((score, gt, method))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            proposals[pred] = candidates

        accepted: list[tuple[str, str, str, float]] = []
        for pred, candidates in proposals.items():
            if not candidates:
                continue
            best_score, best_gt, method = candidates[0]
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
            accepted.append((pred, best_gt, method, best_score))

        if not accepted:
            break
        for pred, gt, method, score in accepted:
            if pred in mapping or gt not in available_gt:
                continue
            mapping[pred] = gt
            available_gt.remove(gt)
            matches.append(ObjectMatch(pred, gt, method, round(score, 3)))

    return mapping, matches


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

    mapping, details = map_objects(predicted, ground_truth)
    contradictions = _state_contradictions(predicted, ground_truth, mapping)
    contradictions.extend(_spatial_contradictions(predicted, ground_truth, mapping))
    contradictions = sorted(set(contradictions))
    unmapped = sorted(predicted.objects - mapping.keys())

    if contradictions:
        reason = f"found {len(contradictions)} explicit initial-state contradiction(s)"
        decision = "reject"
    elif not mapping:
        reason = "no objects could be mapped with sufficient confidence"
        decision = "defer"
    elif unmapped:
        reason = "no proven contradiction; some predicted objects remain unmapped"
        decision = "defer"
    else:
        reason = (
            "no proven contradiction; programmatic checks cannot establish equivalence"
        )
        decision = "defer"

    return PrecheckResult(
        decision=decision,
        reason=reason,
        object_mapping=mapping,
        mapping_details=details,
        contradictions=contradictions,
        unmapped_predicted_objects=unmapped,
    )


def read_problem_source(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if "(" in source or "\n" in source:
        return source
    return Path(source).read_text(encoding="utf-8")
