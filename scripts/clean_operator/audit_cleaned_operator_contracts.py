#!/usr/bin/env python3
"""Audit cleaned PDDL operator names against their semantic contracts.

The audit is read-only.  It can inspect either every round or only the highest
numbered round in each episode, and emits enough canonical-contract detail to
locate same-name schema drift without relying on ``unified_domain.pddl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from clean_operator_data import (
    Action,
    EVAL_ROOT,
    Node,
    PROJECT_ROOT,
    ROUND_RE,
    canonical_action,
    latest_rounds,
    parse_sexp,
    sexp,
)
from validate_operator_cleanup import CandidateResolver


NUMBERED_SUFFIX_RE = re.compile(r"_\d+$")
LEGACY_PLACE_RELATION_RE = re.compile(r"(?:^|_)(into|onto)(?:_|$)")
NAME_RELATION_RE = re.compile(r"(?:^|_)(along|in_front_of|in|into|on|onto|under)(?:_|$)")
SPATIAL_RELATIONS = {
    "along", "in", "on", "under", "in_front_of", "in_front", "at_edge", "on_edge",
    "neatly_on", "at_position", "fifteen_cm_in_front_of", "at_15cm_in_front_of",
    "in_front_of_15cm", "in_front_15cm", "fifteen_cm_in_front",
    "fifteen_centimeters_in_front_of", "distance_15cm_in_front_of",
}
SPATIAL_IMPLICATIONS = {
    "at_15cm_in_front_of": {"in_front_of"},
    "at_edge": {"on"},
    "distance_15cm_in_front_of": {"in_front_of"},
    "fifteen_centimeters_in_front_of": {"in_front_of"},
    "fifteen_cm_in_front": {"in_front_of"},
    "fifteen_cm_in_front_of": {"in_front_of"},
    "in_front": {"in_front_of"},
    "in_front_15cm": {"in_front_of"},
    "in_front_of_15cm": {"in_front_of"},
    "neatly_on": {"on"},
    "on_edge": {"on"},
}
ACTION_VERBS = {
    "add", "close", "cut", "fold", "hang", "insert", "lift", "move",
    "open", "pass", "peel", "pick", "place", "pour", "press", "pull",
    "push", "put", "remove", "rinse", "scoop", "scrub", "shake", "stir",
    "take", "tie", "turn", "untie", "wash", "wipe",
}


@dataclass(frozen=True)
class SignedLiteral:
    negative: bool
    predicate: str
    arguments: tuple[str, ...]

    def render(self) -> str:
        atom = "(" + " ".join((self.predicate, *self.arguments)) + ")"
        return f"(not {atom})" if self.negative else atom


@dataclass
class LintIssue:
    code: str
    detail: str
    action: str | None = None
    domain: str | None = None
    dataset: str | None = None


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def natural_round_key(round_dir: Path) -> tuple[object, ...]:
    def numeric_suffix(value: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", value)
        return (int(match.group(1)), value) if match else (-1, value)

    return (
        *numeric_suffix(round_dir.parent.parent.name),
        *numeric_suffix(round_dir.parent.name),
        *numeric_suffix(round_dir.name),
        str(round_dir),
    )


def all_rounds(
    dataset_root: Path,
    task_ids: set[int] | None = None,
) -> list[Path]:
    rounds: list[Path] = []
    for domain_path in dataset_root.rglob("domain.pddl"):
        round_dir = domain_path.parent
        if (
            ROUND_RE.fullmatch(round_dir.name)
            and round_dir.parent.name.startswith("episode_")
            and round_dir.parent.parent.name.startswith("task_")
        ):
            task = int(round_dir.parent.parent.name.removeprefix("task_"))
            if task_ids is not None and task not in task_ids:
                continue
            rounds.append(round_dir)
    return sorted(rounds, key=natural_round_key)


def selected_rounds(
    dataset_root: Path,
    scope: str,
    task_ids: set[int] | None = None,
) -> list[Path]:
    if scope == "latest":
        return latest_rounds(dataset_root, task_ids)
    if scope == "all":
        return all_rounds(dataset_root, task_ids)
    raise ValueError(f"unsupported scope: {scope}")


def parse_domain_actions(path: Path) -> list[Action]:
    root = parse_sexp(path.read_text(encoding="utf-8"))
    if not isinstance(root, list) or not root or str(root[0]).lower() != "define":
        raise ValueError("expected a top-level (define ...) form")

    actions: list[Action] = []
    for item in root[1:]:
        if (
            isinstance(item, list)
            and item
            and isinstance(item[0], str)
            and item[0].lower() == ":action"
        ):
            actions.append(Action.parse(sexp(item), "; audit"))
    return actions


def signed_literals(expression: Node, negative: bool = False) -> list[SignedLiteral]:
    if isinstance(expression, str) or not expression:
        return []
    head = str(expression[0]).lower()
    if head == "and":
        output: list[SignedLiteral] = []
        for child in expression[1:]:
            output.extend(signed_literals(child, negative))
        return output
    if head == "not" and len(expression) == 2:
        return signed_literals(expression[1], not negative)
    if any(not isinstance(argument, str) for argument in expression[1:]):
        return []
    return [
        SignedLiteral(
            negative,
            head,
            tuple(str(argument).lower() for argument in expression[1:]),
        )
    ]


def positive_unary_predicates(literals: Iterable[SignedLiteral]) -> dict[str, set[str]]:
    predicates: dict[str, set[str]] = defaultdict(set)
    for literal in literals:
        if not literal.negative and len(literal.arguments) == 1:
            predicates[literal.arguments[0]].add(literal.predicate)
    return predicates


def normalize_relation(relation: str) -> str:
    return {"into": "in", "onto": "on"}.get(relation, relation)


def name_relations(name: str) -> set[str]:
    return {
        normalize_relation(match.group(1))
        for match in NAME_RELATION_RE.finditer(name.lower())
    }


def expanded_spatial_relations(relations: set[str]) -> set[str]:
    expanded = set(relations)
    for relation in tuple(relations):
        expanded.update(SPATIAL_IMPLICATIONS.get(relation, set()))
    return expanded


def placed_object(action: Action, pre: list[SignedLiteral], eff: list[SignedLiteral]) -> str | None:
    held = {
        literal.arguments[1]
        for literal in pre
        if (
            not literal.negative
            and literal.predicate == "holding"
            and len(literal.arguments) == 2
        )
    }
    released = {
        literal.arguments[1]
        for literal in eff
        if (
            literal.negative
            and literal.predicate == "holding"
            and len(literal.arguments) == 2
        )
    }
    # A hand can keep holding a tool while the tool moves another object.  In
    # that form, the spatial delete/add pair identifies the placed object more
    # reliably than the hand's resource relation.
    removed_sources = {
        literal.arguments[0]
        for literal in eff
        if (
            literal.negative
            and literal.predicate in SPATIAL_RELATIONS
            and len(literal.arguments) == 2
        )
    }
    added_targets = {
        literal.arguments[0]
        for literal in eff
        if (
            not literal.negative
            and literal.predicate in SPATIAL_RELATIONS
            and len(literal.arguments) == 2
        )
    }
    moved = removed_sources & added_targets
    if len(moved) == 1:
        return next(iter(moved))
    if len(added_targets) == 1:
        return next(iter(added_targets))
    candidates = held & released
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(released) == 1:
        return next(iter(released))
    if len(held) == 1:
        return next(iter(held))
    return None


def place_relation_issue(action: Action) -> str | None:
    if not action.name.lower().startswith("place_"):
        return None
    pre = signed_literals(action.pre)
    eff = signed_literals(action.eff)
    moved = placed_object(action, pre, eff)
    if moved is None:
        return None
    expected = name_relations(action.name)
    actual = expanded_spatial_relations({
        literal.predicate
        for literal in eff
        if (
            not literal.negative
            and literal.predicate in SPATIAL_RELATIONS
            and len(literal.arguments) == 2
            and literal.arguments[0] == moved
        )
    })
    already_established = expanded_spatial_relations({
        literal.predicate
        for literal in pre
        if (
            not literal.negative
            and literal.predicate in SPATIAL_RELATIONS
            and len(literal.arguments) == 2
            and literal.arguments[0] == moved
        )
    })
    deleted = expanded_spatial_relations({
        literal.predicate
        for literal in eff
        if (
            literal.negative
            and literal.predicate in SPATIAL_RELATIONS
            and len(literal.arguments) == 2
            and literal.arguments[0] == moved
        )
    })
    # A placement may establish several compatible relations at once, such as
    # a cup being on a drip tray and under a spout. The name need only expose
    # the primary relation, while later relation tokens may qualify a target
    # rather than the moved object (for example, ``on_package_in_compartment``).
    if expected.issubset(actual) and (expected or not actual):
        return None
    # Some tool-release actions start with the item already at its final
    # location.  Releasing it preserves that relation; requiring a duplicate
    # add effect would not make the resulting state more precise.
    if expected.issubset(actual | (already_established - deleted)):
        return None
    # Later prepositions can qualify the support rather than the placed object:
    # ``place_package_on_package_in_compartment`` adds (on moving support) and
    # requires (in support compartment).  Accept only a connected relation
    # chain, not an arbitrary scene fact that happens to use the same token.
    represented = set(actual)
    frontier = {
        literal.arguments[1]
        for literal in eff
        if (
            not literal.negative
            and literal.predicate in SPATIAL_RELATIONS
            and len(literal.arguments) == 2
            and literal.arguments[0] == moved
        )
    }
    visited: set[str] = set()
    remaining = list(pre)
    while frontier:
        frontier -= visited
        if not frontier:
            break
        visited.update(frontier)
        next_frontier: set[str] = set()
        for literal in remaining:
            if (
                not literal.negative
                and literal.predicate in SPATIAL_RELATIONS
                and len(literal.arguments) == 2
                and literal.arguments[0] in frontier
            ):
                represented.add(literal.predicate)
                next_frontier.add(literal.arguments[1])
        frontier = next_frontier
    if expected.issubset(represented) and actual:
        return None
    return (
        f"placed object {moved} has name relations {sorted(expected)} but "
        f"positive effects {sorted(actual)}"
    )


def bowl_bowl_in_literals(action: Action) -> list[str]:
    pre = signed_literals(action.pre)
    eff = signed_literals(action.eff)
    unary = positive_unary_predicates(pre)
    bowls = {variable for variable, predicates in unary.items() if "bowl" in predicates}
    matches: list[str] = []
    for location, literals in (("precondition", pre), ("effect", eff)):
        for literal in literals:
            if (
                literal.predicate == "in"
                and len(literal.arguments) == 2
                and literal.arguments[0] in bowls
                and literal.arguments[1] in bowls
            ):
                matches.append(f"{location} {literal.render()}")
    return matches


def has_compound_action_name(name: str) -> bool:
    """Distinguish two verbs from object names such as wash_and_care_set."""
    tokens = name.split("_")
    for index, token in enumerate(tokens):
        if token != "and":
            continue
        before = any(part in ACTION_VERBS for part in tokens[max(0, index - 2):index])
        after = any(part in ACTION_VERBS for part in tokens[index + 1:index + 3])
        if before and after:
            return True
    return False


MICROWAVE_RESULT_PREDICATES = {"boiled", "boiling", "cooked", "heated", "warm", "warmed"}
MICROWAVE_RESULT_NEGATIONS = {"cold", "unheated"}


def microwave_result_effects(action: Action) -> set[SignedLiteral]:
    return {
        literal
        for literal in signed_literals(action.eff)
        if not literal.negative and literal.predicate in MICROWAVE_RESULT_PREDICATES
    }


def is_microwave_result_transition(literal: SignedLiteral) -> bool:
    return (
        (not literal.negative and literal.predicate in MICROWAVE_RESULT_PREDICATES)
        or (
            literal.negative
            and (
                literal.predicate in MICROWAVE_RESULT_NEGATIONS
                or (
                    literal.predicate.startswith("not_")
                    and literal.predicate.removeprefix("not_") in MICROWAVE_RESULT_PREDICATES
                )
            )
        )
    )


def microwave_purity_problems(action: Action, actions: list[Action]) -> list[str]:
    if action.name.lower() != "turn_on_microwave":
        return []
    pre = signed_literals(action.pre)
    eff = signed_literals(action.eff)
    unary = positive_unary_predicates(pre)
    microwaves = {
        variable for variable, predicates in unary.items() if "microwave" in predicates
    }
    if len(microwaves) != 1:
        return [f"expected one microwave parameter, found {sorted(microwaves)}"]
    microwave = next(iter(microwaves))
    expected_effects = {
        SignedLiteral(True, "is_off", (microwave,)),
        SignedLiteral(False, "is_on", (microwave,)),
    }
    actual_effects = set(eff)
    problems: list[str] = []
    if actual_effects != expected_effects:
        missing = sorted(item.render() for item in expected_effects - actual_effects)
        extra = sorted(
            item.render()
            for item in actual_effects - expected_effects
            if not is_microwave_result_transition(item)
        )
        parts: list[str] = []
        if missing:
            parts.append(f"missing effects {missing}")
        if extra:
            parts.append(f"extra effects {extra}")
        if parts:
            problems.append("; ".join(parts))
    if SignedLiteral(False, "is_off", (microwave,)) not in pre:
        problems.append(f"missing precondition (is_off {microwave})")

    results = microwave_result_effects(action)
    if not results:
        return problems
    problems.append("start action commits a heating result; completion must produce it")
    return problems


def faucet_rinse_problems(action: Action) -> list[str]:
    if action.name.lower() not in {
        "turn_off_faucet_after_rinsing_bowl",
        "turn_off_faucet_after_rinsing",
    }:
        return []
    pre = signed_literals(action.pre)
    rinsed = {
        literal.arguments[0]
        for literal in pre
        if (
            not literal.negative
            and literal.predicate == "rinsed"
            and len(literal.arguments) == 1
        )
    }
    if rinsed:
        return []
    return ["faucet completion lacks a rinsed-object precondition"]


def drawer_target(pre: list[SignedLiteral], eff: list[SignedLiteral]) -> str | None:
    opened = {
        literal.arguments[0]
        for literal in eff
        if (
            not literal.negative
            and literal.predicate == "open"
            and len(literal.arguments) == 1
        )
    }
    closed_deleted = {
        literal.arguments[0]
        for literal in eff
        if (
            literal.negative
            and literal.predicate == "closed"
            and len(literal.arguments) == 1
        )
    }
    candidates = opened & closed_deleted
    if len(candidates) == 1:
        return next(iter(candidates))
    drawer_variables = {
        literal.arguments[0]
        for literal in pre
        if (
            not literal.negative
            and literal.predicate == "drawer"
            and len(literal.arguments) == 1
        )
    }
    candidates = opened & drawer_variables
    return next(iter(candidates)) if len(candidates) == 1 else None


def drawer_role(variable: str, unary: dict[str, set[str]]) -> str | None:
    role_order = ("top", "middle", "bottom", "upper", "lower", "left", "right")
    predicates = unary.get(variable, set())
    for role in role_order:
        if f"is_{role}" in predicates or f"{role}_drawer" in predicates:
            return role
    lowered = variable.lower().lstrip("?")
    for role in role_order:
        if role in lowered:
            return role
    return None


def drawer_guard_problems(action: Action) -> list[str]:
    name = action.name.lower()
    if "drawer" not in name or not (name.startswith("open_") or "_open_drawer" in name):
        return []
    pre = signed_literals(action.pre)
    eff = signed_literals(action.eff)
    target = drawer_target(pre, eff)
    if target is None:
        return []
    unary = positive_unary_predicates(pre)
    positive = set(pre)
    problems: list[str] = []

    if SignedLiteral(False, "unlocked", (target,)) in positive and "unlocked" not in name:
        problems.append(f"unlocked access variant on target {target} is absent from the name")
    if SignedLiteral(False, "unblocked", (target,)) in positive and not (
        "unblocked" in name
    ):
        problems.append(f"unblocked access variant on target {target} is absent from the name")
    if SignedLiteral(False, "clear_to_open", (target,)) in positive and "unblocked" not in name:
        problems.append(f"clear_to_open access variant on target {target} is absent from the name")

    drawer_variables = {
        literal.arguments[0]
        for literal in pre
        if (
            not literal.negative
            and literal.predicate == "drawer"
            and len(literal.arguments) == 1
        )
    }
    other_closed = sorted(
        literal.arguments[0]
        for literal in pre
        if (
            not literal.negative
            and literal.predicate == "closed"
            and len(literal.arguments) == 1
            and literal.arguments[0] in drawer_variables
            and literal.arguments[0] != target
        )
    )
    for other in other_closed:
        role = drawer_role(other, unary)
        if "interlocked" not in name:
            expected = "an interlocked access variant"
            problems.append(
                f"other-drawer guard (closed {other}) is not reflected as {expected}"
            )
    return problems


def multi_hand_distinctness_problems(action: Action) -> list[str]:
    """Detect independent hand roles that can be bound to one hand object."""
    parameters = set(action.params)
    hands: list[str] = []
    for literal in signed_literals(action.pre):
        if (
            not literal.negative
            and literal.predicate == "hand"
            and len(literal.arguments) == 1
            and literal.arguments[0] in parameters
            and literal.arguments[0] not in hands
        ):
            hands.append(literal.arguments[0])
    if len(hands) < 2:
        return []
    inequalities = {
        (literal.arguments[0], literal.arguments[1])
        for literal in signed_literals(action.pre)
        if (
            literal.negative
            and literal.predicate == "="
            and len(literal.arguments) == 2
        )
    }
    missing = [
        (first, second)
        for index, first in enumerate(hands)
        for second in hands[index + 1:]
        if (first, second) not in inequalities and (second, first) not in inequalities
    ]
    return [
        "missing distinctness for independently declared hand roles "
        + ", ".join(f"{first}/{second}" for first, second in missing)
    ] if missing else []


def signature_payload(signature: tuple[object, ...]) -> object:
    if isinstance(signature, tuple):
        return [signature_payload(item) for item in signature]
    return signature


def signature_id(signature: tuple[object, ...]) -> str:
    rendered = json.dumps(
        signature_payload(signature),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("ascii")).hexdigest()


def add_action_issues(
    action: Action,
    domain_actions: list[Action],
    domain_path: Path,
    dataset: str,
    issues: list[LintIssue],
) -> None:
    name = action.name.lower()
    location = relative_path(domain_path)

    def add(code: str, detail: str) -> None:
        issues.append(LintIssue(code, detail, name, location, dataset))

    if NUMBERED_SUFFIX_RE.search(name):
        add("numbered_action_suffix", "action name ends in a numeric conflict suffix")
    if "_when_" in name:
        add("legacy_when_clause", "action name encodes a precondition with a when clause")
    if "_and_" in name and not name.startswith("press_and_hold_") and has_compound_action_name(name):
        add("compound_action_name", "action name joins clauses with 'and'")
    legacy_relation = LEGACY_PLACE_RELATION_RE.search(name) if name.startswith("place_") else None
    if legacy_relation:
        replacement = normalize_relation(legacy_relation.group(1))
        add(
            "legacy_place_preposition",
            f"place name uses '{legacy_relation.group(1)}'; canonical spelling is '{replacement}'",
        )
    mismatch = place_relation_issue(action)
    if mismatch:
        add("place_relation_mismatch", mismatch)
    for detail in microwave_purity_problems(action, domain_actions):
        add("turn_on_microwave_not_pure", detail)
    for detail in faucet_rinse_problems(action):
        add("faucet_rinse_effect_missing", detail)
    for detail in drawer_guard_problems(action):
        add("drawer_guard_not_named", detail)
    for detail in multi_hand_distinctness_problems(action):
        add("multi_hand_distinctness_missing", detail)


def audit(
    datasets: list[str],
    scope: str,
    mapping_dir: Path | None = None,
    task_ids: set[int] | None = None,
    include_fingerprints: bool = True,
    task_min: int | None = None,
    task_max: int | None = None,
) -> dict[str, object]:
    issues: list[LintIssue] = []
    fingerprint_groups: dict[
        str,
        dict[str, dict[str, object]],
    ] = defaultdict(dict)
    action_declarations: Counter[str] = Counter()
    scanned_rounds = 0
    parsed_domains = 0
    resolver = CandidateResolver(mapping_dir)

    for dataset in datasets:
        root = EVAL_ROOT / dataset
        rounds = selected_rounds(root, scope, task_ids)
        if task_min is not None or task_max is not None:
            rounds = [
                round_dir
                for round_dir in rounds
                if (
                    (task_min is None or int(round_dir.parent.parent.name.removeprefix("task_")) >= task_min)
                    and (task_max is None or int(round_dir.parent.parent.name.removeprefix("task_")) <= task_max)
                )
            ]
        for round_dir in rounds:
            scanned_rounds += 1
            source_domain_path = round_dir / "domain.pddl"
            domain_path = resolver.resolve(round_dir, "domain.pddl")
            try:
                actions = parse_domain_actions(domain_path)
            except Exception as exc:
                candidate_detail = (
                    f" (candidate: {domain_path})"
                    if domain_path != source_domain_path
                    else ""
                )
                issues.append(
                    LintIssue(
                        "parse_error",
                        f"{type(exc).__name__}: {exc}{candidate_detail}",
                        domain=relative_path(source_domain_path),
                        dataset=dataset,
                    )
                )
                continue
            parsed_domains += 1
            for action in actions:
                action.name = action.name.lower()
                action_declarations[action.name] += 1
                add_action_issues(action, actions, source_domain_path, dataset, issues)
                if include_fingerprints:
                    signature = canonical_action(action)
                    fingerprint = signature_id(signature)
                    group = fingerprint_groups[action.name].setdefault(
                        fingerprint,
                        {
                            "count": 0,
                            "datasets": Counter(),
                            "examples": [],
                            "signature": signature_payload(signature),
                        },
                    )
                    group["count"] = int(group["count"]) + 1
                    dataset_counts = group["datasets"]
                    assert isinstance(dataset_counts, Counter)
                    dataset_counts[dataset] += 1
                    examples = group["examples"]
                    assert isinstance(examples, list)
                    location = relative_path(source_domain_path)
                    if len(examples) < 5 and location not in examples:
                        examples.append(location)

    canonical_report: dict[str, object] = {}
    conflict_count = 0
    for name in sorted(fingerprint_groups):
        groups = fingerprint_groups[name]
        rows: list[dict[str, object]] = []
        for fingerprint, group in sorted(groups.items()):
            rows.append(
                {
                    "fingerprint": fingerprint,
                    "count": group["count"],
                    "datasets": dict(sorted(group["datasets"].items())),
                    "examples": group["examples"],
                    "signature": group["signature"],
                }
            )
        canonical_report[name] = {
            "declarations": action_declarations[name],
            "fingerprint_count": len(rows),
            "fingerprints": rows,
        }
        if len(rows) > 1:
            conflict_count += 1

    issue_counts = Counter(issue.code for issue in issues)
    summary = {
        "scanned_rounds": scanned_rounds,
        "parsed_domains": parsed_domains,
        "parse_failed_domains": scanned_rounds - parsed_domains,
        "action_declarations": sum(action_declarations.values()),
        "unique_action_names": len(action_declarations),
        "action_names_with_multiple_fingerprints": conflict_count if include_fingerprints else None,
        "lint_issues": len(issues),
        "issue_counts": dict(sorted(issue_counts.items())),
    }
    return {
        "datasets": datasets,
        "scope": scope,
        "mapping_dir": str(resolver.mapping_dir) if resolver.mapping_dir else None,
        "tasks": sorted(task_ids) if task_ids is not None else None,
        "task_min": task_min,
        "task_max": task_max,
        "override_files": resolver.override_files,
        "summary": summary,
        "issues": [asdict(issue) for issue in issues],
        "canonical_action_fingerprints": canonical_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("agibot", "agibot_aug", "both"),
        default="both",
        help="dataset to audit (default: both)",
    )
    parser.add_argument(
        "--scope",
        choices=("latest", "all"),
        default="latest",
        help="inspect the latest round per episode or every round (default: latest)",
    )
    parser.add_argument(
        "--task",
        type=int,
        action="append",
        dest="tasks",
        help="task id to audit; repeat for multiple tasks",
    )
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        help=(
            "optional sparse candidate tree; domain.pddl overrides use the "
            "same layouts as validate_operator_cleanup.py"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional path for the complete JSON report",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="skip the large same-name fingerprint index and emit lint results only",
    )
    parser.add_argument("--task-min", type=int, help="include only task IDs at or above this value")
    parser.add_argument("--task-max", type=int, help="include only task IDs at or below this value")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = ["agibot", "agibot_aug"] if args.dataset == "both" else [args.dataset]
    missing = [str(EVAL_ROOT / dataset) for dataset in datasets if not (EVAL_ROOT / dataset).is_dir()]
    if missing:
        raise SystemExit("dataset roots do not exist: " + ", ".join(missing))
    if args.mapping_dir is not None and not args.mapping_dir.is_dir():
        raise SystemExit(f"mapping directory does not exist: {args.mapping_dir}")

    selected_tasks = set(args.tasks) if args.tasks else None
    report = audit(
        datasets,
        args.scope,
        args.mapping_dir,
        selected_tasks,
        include_fingerprints=not args.summary_only,
        task_min=args.task_min,
        task_max=args.task_max,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False))
        print(f"report: {args.report}")
    else:
        print(rendered, end="")
    summary = report["summary"]
    assert isinstance(summary, dict)
    return 1 if int(summary["lint_issues"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
