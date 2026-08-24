from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

Literal = tuple[str, ...]


@dataclass
class ActionSchema:
    name: str
    params: list[str]
    pre_pos: set[Literal]
    pre_neg: set[Literal]
    add_eff: set[Literal]
    del_eff: set[Literal]


@dataclass
class GroundAction:
    name: str
    args: list[str]
    pre_pos: set[Literal]
    pre_neg: set[Literal]
    add_eff: set[Literal]
    del_eff: set[Literal]
    equality_preconditions: set[Literal] = field(default_factory=set)
    inequality_preconditions: set[Literal] = field(default_factory=set)

    def to_line(self) -> str:
        return f"({self.name} {' '.join(self.args)})"


def parse_sexpr_file(path: Path):
    text = re.sub(r";[^\n]*", "", path.read_text(encoding="utf-8")).lower()
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    index = 0

    def parse():
        nonlocal index
        if index >= len(tokens):
            raise ValueError(f"Unexpected EOF in {path}")

        token = tokens[index]
        index += 1
        if token != "(":
            if token == ")":
                raise ValueError(f"Unexpected ')' in {path}")
            if token in {"when", "forall", "or"}:
                raise NotImplementedError(
                    f"Unsupported PDDL construct '{token}' in {path}"
                )
            return token

        expression = []
        while index < len(tokens) and tokens[index] != ")":
            expression.append(parse())
        if index >= len(tokens):
            raise ValueError(f"Missing ')' in {path}")
        index += 1
        return expression

    root = parse()
    if index != len(tokens):
        raise ValueError(f"Unparsed tokens remain in {path}")
    return root


def strip_types(items: list[str]) -> list[str]:
    result = []
    skip_type = False
    for item in items:
        if item == "-":
            skip_type = True
        elif skip_type:
            skip_type = False
        else:
            result.append(item)
    return result


def read_literals(expression) -> tuple[set[Literal], set[Literal]]:
    if expression is None:
        return set(), set()
    if isinstance(expression, str):
        raise ValueError(f"Unexpected atom: {expression}")
    if expression[0] == "and":
        positive = set()
        negative = set()
        for child in expression[1:]:
            child_positive, child_negative = read_literals(child)
            positive.update(child_positive)
            negative.update(child_negative)
        return positive, negative
    if expression[0] == "not":
        return set(), {tuple(expression[1])}
    return {tuple(expression)}, set()


def parse_domain(path: Path) -> dict[str, ActionSchema]:
    root = parse_sexpr_file(path)
    if root[0] != "define":
        raise ValueError(f"{path} is not a valid domain file")

    schemas = {}
    for item in root[1:]:
        if not isinstance(item, list) or not item or item[0] != ":action":
            continue

        name = item[1]
        params = []
        precondition = None
        effect = None
        for index in range(2, len(item), 2):
            if item[index] == ":parameters":
                params = strip_types(item[index + 1])
            elif item[index] == ":precondition":
                precondition = item[index + 1]
            elif item[index] == ":effect":
                effect = item[index + 1]

        pre_pos, pre_neg = read_literals(precondition)
        add_eff, del_eff = read_literals(effect)
        schemas[name] = ActionSchema(name, params, pre_pos, pre_neg, add_eff, del_eff)
    return schemas


def parse_problem(path: Path) -> tuple[set[Literal], set[Literal], set[Literal]]:
    root = parse_sexpr_file(path)
    if root[0] != "define":
        raise ValueError(f"{path} is not a valid problem file")

    init_state = set()
    init_negative = set()
    goal_positive = set()
    goal_negative = set()
    for item in root[1:]:
        if not isinstance(item, list) or not item:
            continue
        if item[0] == ":init":
            for literal in item[1:]:
                if isinstance(literal, list) and literal and literal[0] == "not":
                    atom = tuple(literal[1])
                    if atom in init_state:
                        raise ValueError(f"{path}: contradictory init literal {atom}")
                    init_negative.add(atom)
                else:
                    atom = tuple(literal)
                    if atom in init_negative:
                        raise ValueError(f"{path}: contradictory init literal {atom}")
                    init_state.add(atom)
        elif item[0] == ":goal":
            goal_positive, goal_negative = read_literals(item[1])
    return init_state, goal_positive, goal_negative


def parse_plan(path: Path) -> tuple[list[tuple[str, list[str]]], list[str]]:
    actions = []
    comments = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(";"):
            comments.append(line)
            continue
        match = re.match(r"^\(([^()]*)\)$", line.lower())
        if match:
            parts = match.group(1).split()
            actions.append((parts[0], parts[1:]))
    return actions, comments


def ground_plan(
    raw_plan: list[tuple[str, list[str]]], schemas: dict[str, ActionSchema]
) -> list[GroundAction]:
    plan = []
    for name, args in raw_plan:
        if name not in schemas:
            raise KeyError(f"Action '{name}' not found in domain")
        schema = schemas[name]
        if len(args) != len(schema.params):
            raise ValueError(
                f"Arity mismatch for action {name}: "
                f"expected {len(schema.params)}, got {len(args)}"
            )

        mapping = dict(zip(schema.params, args))

        def substitute(
            literals: set[Literal], mapping: dict[str, str] = mapping
        ) -> set[Literal]:
            return {
                tuple(mapping.get(token, token) for token in literal)
                for literal in literals
            }

        pre_pos, equality_preconditions = split_equalities(substitute(schema.pre_pos))
        pre_neg, inequality_preconditions = split_equalities(substitute(schema.pre_neg))
        add_effects, equality_effects = split_equalities(substitute(schema.add_eff))
        del_effects, inequality_effects = split_equalities(substitute(schema.del_eff))
        if equality_effects or inequality_effects:
            raise ValueError(f"Equality cannot be used as an effect of {name}")

        plan.append(
            GroundAction(
                name,
                args,
                pre_pos,
                pre_neg,
                add_effects,
                del_effects - add_effects,
                equality_preconditions,
                inequality_preconditions,
            )
        )
    return plan


def split_equalities(literals: set[Literal]) -> tuple[set[Literal], set[Literal]]:
    equalities = {literal for literal in literals if literal[0] == "="}
    for literal in equalities:
        if len(literal) != 3:
            raise ValueError(f"Equality must have exactly two arguments: {literal}")
    return literals - equalities, equalities


def apply_action(state: set[Literal], action: GroundAction) -> set[Literal]:
    missing = action.pre_pos - state
    violated = action.pre_neg & state
    unequal = {
        literal
        for literal in action.equality_preconditions
        if literal[1] != literal[2]
    }
    equal = {
        literal
        for literal in action.inequality_preconditions
        if literal[1] == literal[2]
    }
    if missing or violated or unequal or equal:
        message = [f"Action not applicable: {action.to_line()}"]
        if missing:
            message.append(f"Missing positive preconditions: {sorted(missing)}")
        if violated:
            message.append(f"Violated negative preconditions: {sorted(violated)}")
        if unequal:
            message.append(f"Unsatisfied equality preconditions: {sorted(unequal)}")
        if equal:
            message.append(f"Violated inequality preconditions: {sorted(equal)}")
        raise ValueError("\n".join(message))

    next_state = state - action.del_eff
    next_state.update(action.add_eff)
    return next_state


def rollout(init_state: set[Literal], plan: list[GroundAction]) -> set[Literal]:
    state = set(init_state)
    for action in plan:
        state = apply_action(state, action)
    return state


def goals_satisfied(
    state: set[Literal], goal_pos: set[Literal], goal_neg: set[Literal]
) -> bool:
    return goal_pos <= state and not goal_neg & state


def assert_goals(
    state: set[Literal],
    goal_pos: set[Literal],
    goal_neg: set[Literal],
    title: str,
) -> None:
    if goals_satisfied(state, goal_pos, goal_neg):
        return
    message = [title]
    missing = sorted(goal_pos - state)
    violated = sorted(goal_neg & state)
    if missing:
        message.append(f"Missing positive goals: {missing}")
    if violated:
        message.append(f"Violated negative goals: {violated}")
    raise ValueError("\n".join(message))
