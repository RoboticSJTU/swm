"""Checks for newly generated manipulation PDDL, without changing shared parsers."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from swm.pddl.strips import (
    DomainSchemas,
    Literal,
    ProblemModel,
    apply_action,
    assert_goals,
    ground_plan,
    parse_domain,
    parse_plan,
    parse_problem_model,
    parse_sexpr,
    validate_untyped_pddl,
)
from swm.prompts import read_prompt

_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_REQUIREMENTS = {":strips", ":negative-preconditions", ":equality"}
_STATIC = {"hand", "part_of", "dispenses", "controls"}
_MUTEX = (
    ("open", "closed"),
    ("is_on", "is_off"),
    ("locked", "unlocked"),
    ("pressed", "unpressed"),
    ("folded", "unfolded"),
    ("empty", "filled"),
    ("upright", "upside_down"),
    ("flat", "vertical"),
)
_UNSUPPORTED = {
    "forall", "exists", "when", "or", "imply", "oneof", "probabilistic",
    "increase", "decrease", "assign", "scale-up", "scale-down",
}


@lru_cache(maxsize=1)
def _predicate_arities() -> dict[str, int]:
    text = read_prompt("PREDICATE_TEMPLATE.txt")
    declarations = parse_sexpr(text[text.index("(:predicates"):])
    return {item[0]: len(item) - 1 for item in declarations[1:]}


def _walk(node):
    if isinstance(node, list):
        if node and isinstance(node[0], str):
            yield node
        for child in node:
            yield from _walk(child)


def _profile(text: str, kind: str) -> list:
    validate_untyped_pddl(text)
    root = parse_sexpr(text)
    if (
        len(root) < 2 or not isinstance(root[1], list) or len(root[1]) != 2
        or root[1][0] != kind or not isinstance(root[1][1], str)
    ):
        raise ValueError(f"Expected (define ({kind} NAME) ...)")
    allowed = (
        {":requirements", ":predicates", ":action"}
        if kind == "domain" else {":domain", ":objects", ":init", ":goal"}
    )
    seen = set()
    for section in root[2:]:
        if not isinstance(section, list) or not section or not isinstance(section[0], str):
            raise ValueError(f"Malformed {kind} section")
        if section[0] not in allowed:
            raise ValueError(f"Unsupported {kind} section '{section[0]}'")
        if section[0] != ":action" and section[0] in seen:
            raise ValueError(f"Duplicate {kind} section '{section[0]}'")
        seen.add(section[0])
    for node in _walk(root):
        if node[0] in _UNSUPPORTED:
            raise ValueError(
                f"Forbidden construct '{node[0]}'; use conjunctive atomic conditions and add/delete effects"
            )
    # Check original spelling: the shared parser intentionally lowercases input.
    for token in re.findall(r";[^\n]*|[^\s();]+", text):
        if token.startswith((";", ":")) or token == "=":
            continue
        name = token[1:] if token.startswith("?") else token
        if not _NAME.fullmatch(name):
            raise ValueError(f"Identifier '{token}' must use lowercase snake_case")
    return root


def _consistent(positive: set[Literal], negative: set[Literal], context: str) -> None:
    if positive & negative:
        raise ValueError(f"{context}: contradictory literal {sorted(positive & negative)[0]}")
    for first, second in _MUTEX:
        for literal in sorted(positive):
            if literal[0] == first and (second, *literal[1:]) in positive:
                raise ValueError(f"{context}: mutually exclusive {first}/{second} for {literal[1:]}")
    for literal in positive:
        if literal[0] in {"holding", "pressing"} and ("hand_free", literal[1]) in positive:
            raise ValueError(f"{context}: occupied hand '{literal[1]}' is also hand_free")


def _state_consistent(state: set[Literal], context: str) -> None:
    _consistent(state, set(), context)
    occupied: dict[str, set[Literal]] = {}
    for literal in state:
        if literal[0] in {"holding", "pressing"}:
            occupied.setdefault(literal[1], set()).add(literal)
    for hand, loads in occupied.items():
        if len(loads) > 1:
            raise ValueError(f"{context}: hand '{hand}' has multiple loads or controls")


def validate_generated_pddl(domain_path: Path, problem_path: Path) -> tuple[DomainSchemas, ProblemModel]:
    """Validate the generation profile; allow new predicates and contextual guards."""
    domain_root = _profile(domain_path.read_text(encoding="utf-8"), "domain")
    problem_root = _profile(problem_path.read_text(encoding="utf-8"), "problem")
    declarations = [node for node in domain_root[2:] if node[0] == ":requirements"]
    if len(declarations) != 1 or any(not isinstance(item, str) for item in declarations[0][1:]):
        raise ValueError("Declare one :requirements section")
    requirements = set(declarations[0][1:])
    if requirements - _REQUIREMENTS or ":strips" not in requirements:
        raise ValueError("Requirements must contain :strips, with only optional :negative-preconditions and :equality")
    references = [node for node in problem_root[2:] if node[0] == ":domain"]
    if len(references) != 1 or references[0] != [":domain", domain_root[1][1]]:
        raise ValueError("Problem :domain must match the domain name")

    schemas = parse_domain(domain_path)
    problem = parse_problem_model(problem_path, schemas)
    known_arities = _predicate_arities()
    for name, arity in schemas.predicate_arities.items():
        if name in {"and", "not", "="}:
            raise ValueError(f"Reserved predicate name '{name}'")
        if name in known_arities and arity != known_arities[name]:
            raise ValueError(f"Predicate '{name}' requires {known_arities[name]} arguments, got {arity}")
    for action in schemas.values():
        if "_" not in action.name or re.search(r"_\d+$", action.name):
            raise ValueError(f"Action '{action.name}' must name its operation and object without numbering suffixes")
        _consistent(action.pre_pos, action.pre_neg, f"{action.name} precondition")
        _consistent(action.add_eff, action.del_eff, f"{action.name} effect")
        _consistent((action.pre_pos - action.del_eff) | action.add_eff, set(), f"{action.name} successor")
        for literal in action.add_eff | action.del_eff:
            if literal[0] in _STATIC or literal[0] == "=":
                raise ValueError(f"{action.name}: cannot modify static predicate '{literal[0]}'")
        if action.pre_neg and ":negative-preconditions" not in requirements:
            raise ValueError(f"{action.name}: negative preconditions require :negative-preconditions")
        if any(literal[0] == "=" for literal in action.pre_pos | action.pre_neg) and ":equality" not in requirements:
            raise ValueError(f"{action.name}: equality requires :equality")
    problem_literals = problem.init_state | problem.init_negative | problem.goal_positive | problem.goal_negative
    if any(literal[0] == "=" for literal in problem_literals):
        raise ValueError("Use equality only in action preconditions, not init or goal")
    if problem.goal_negative and ":negative-preconditions" not in requirements:
        raise ValueError("Negative goals require :negative-preconditions")
    _state_consistent(problem.init_state, "init")
    _consistent(problem.goal_positive, problem.goal_negative, "goal")
    return schemas, problem


def validate_generated_plan(plan_path: Path, schemas: DomainSchemas, problem: ProblemModel) -> None:
    """Replay the saved solver plan and verify known invariants and all goals."""
    for raw_line in plan_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if line and not re.fullmatch(r"\([^()]+\)", line):
            raise ValueError(f"Malformed plan line: {line}")
    raw_plan, _ = parse_plan(plan_path)
    plan = ground_plan(raw_plan, schemas, problem.objects)
    state = set(problem.init_state)
    for index, action in enumerate(plan, 1):
        state = apply_action(state, action)
        _state_consistent(state, f"plan step {index} ({action.name})")
    assert_goals(state, problem.goal_positive, problem.goal_negative, "Saved plan does not satisfy goal")
