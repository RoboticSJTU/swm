from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from swm.pddl.typing import TypeHierarchy, typed_symbol_map

Literal = tuple[str, ...]


def normalize_domain_types(path: Path) -> bool:
    """Remove redundant root declarations and repair grouped self-inheritance."""
    text = path.read_text(encoding="utf-8")
    tokens: list[tuple[str, int, int]] = []
    for match in re.finditer(r";[^\n]*(?:\n|$)|[()]|[^()\s;]+", text):
        value = match.group()
        if not value.startswith(";"):
            tokens.append((value, match.start(), match.end()))

    depth = 0
    section_start: int | None = None
    type_sections: list[tuple[int, int]] = []
    for index, (value, _, _) in enumerate(tokens):
        if value == "(":
            if (
                depth == 1
                and index + 1 < len(tokens)
                and tokens[index + 1][0].lower() == ":types"
            ):
                section_start = index
            depth += 1
        elif value == ")":
            depth -= 1
            if section_start is not None and depth == 1:
                type_sections.append((section_start, index))
                section_start = None

    # Leave malformed or duplicate sections untouched for the normal parser to reject.
    if len(type_sections) != 1:
        return False

    start, end = type_sections[0]
    items = tokens[start + 2 : end]
    if any(value in {"(", ")"} for value, _, _ in items):
        return False

    groups: list[
        tuple[
            list[tuple[str, int, int]],
            tuple[str, int, int] | None,
            tuple[str, int, int] | None,
        ]
    ] = []
    cursor = 0
    while cursor < len(items):
        dash = next(
            (index for index in range(cursor, len(items)) if items[index][0] == "-"),
            None,
        )
        if dash is None:
            groups.append((items[cursor:], None, None))
            break
        if dash == cursor or dash + 1 >= len(items) or items[dash + 1][0] == "-":
            return False
        groups.append((items[cursor:dash], items[dash], items[dash + 1]))
        cursor = dash + 2

    # Redeclaring the built-in root below another type is a real error.
    for names, _, parent in groups:
        parent_name = parent[0].lower() if parent is not None else "object"
        if parent_name != "object" and any(
            name.lower() == "object" for name, _, _ in names
        ):
            return False

    edits: list[tuple[int, int, str]] = []
    for names, dash, parent in groups:
        parent_name = parent[0].lower() if parent is not None else "object"
        if parent_name == "object":
            redundant = [token for token in names if token[0].lower() == "object"]
            for _, token_start, token_end in redundant:
                edits.append((token_start, token_end, ""))
            if redundant and len(redundant) == len(names) and parent is not None:
                edits.append((dash[1], dash[2], ""))
                edits.append((parent[1], parent[2], ""))
            continue

        self_types = [
            (index, token)
            for index, token in enumerate(names)
            if token[0].lower() == parent_name
        ]
        if not self_types:
            continue
        if len(self_types) != 1:
            return False

        index, self_type = self_types[0]
        has_before = index > 0
        has_after = index + 1 < len(names)
        if has_before:
            edits.append((self_type[1], self_type[1], f"- {parent[0]} "))
        if has_after:
            edits.append((self_type[2], self_type[2], " - object"))
        else:
            edits.append((parent[1], parent[2], "object"))

    if not edits:
        return False
    for edit_start, edit_end, replacement in sorted(edits, reverse=True):
        text = text[:edit_start] + replacement + text[edit_end:]
    path.write_text(text, encoding="utf-8")
    return True


@dataclass
class ActionSchema:
    name: str
    params: list[str]
    pre_pos: set[Literal]
    pre_neg: set[Literal]
    add_eff: set[Literal]
    del_eff: set[Literal]
    param_types: dict[str, str] = field(default_factory=dict)
    explicit_param_types: set[str] = field(default_factory=set)
    type_hierarchy: TypeHierarchy = field(default_factory=TypeHierarchy.object_only)


class DomainSchemas(dict[str, ActionSchema]):
    def __init__(
        self,
        *args,
        type_hierarchy: TypeHierarchy | None = None,
        predicate_types: dict[str, tuple[str, ...]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.type_hierarchy = type_hierarchy or TypeHierarchy.object_only()
        self.predicate_types = predicate_types or {}


@dataclass(frozen=True)
class ProblemModel:
    object_types: dict[str, str]
    init_state: set[Literal]
    init_negative: set[Literal]
    goal_positive: set[Literal]
    goal_negative: set[Literal]


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
    parameter_types: tuple[str, ...] = ()

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
            if token in {"when", "forall", "or", "exists", "imply"}:
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
    """Compatibility helper for historical experiment readers."""
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
    if not expression:
        raise ValueError("Empty logical expression")
    if expression[0] == "and":
        positive = set()
        negative = set()
        for child in expression[1:]:
            child_positive, child_negative = read_literals(child)
            positive.update(child_positive)
            negative.update(child_negative)
        return positive, negative
    if expression[0] == "not":
        if len(expression) != 2 or not isinstance(expression[1], list):
            raise ValueError("Malformed negated literal")
        return set(), {tuple(expression[1])}
    if any(not isinstance(token, str) for token in expression):
        raise ValueError(f"Nested term in literal: {expression}")
    return {tuple(expression)}, set()


def _sections(root, name: str) -> list[list]:
    return [
        item
        for item in root[1:]
        if isinstance(item, list) and item and item[0] == name
    ]


def _validate_literal_types(
    literals: set[Literal],
    predicate_types: dict[str, tuple[str, ...]],
    argument_types: dict[str, str],
    hierarchy: TypeHierarchy,
    context: str,
) -> None:
    for literal in literals:
        if literal[0] == "=":
            if len(literal) != 3:
                raise ValueError(f"{context}: equality expects two arguments")
            continue
        if literal[0] not in predicate_types:
            raise ValueError(f"{context}: undeclared predicate '{literal[0]}'")
        expected = predicate_types[literal[0]]
        if len(literal) - 1 != len(expected):
            raise ValueError(
                f"{context}: predicate '{literal[0]}' expects "
                f"{len(expected)} arguments, got {len(literal) - 1}"
            )
        for argument, expected_type in zip(literal[1:], expected):
            if argument not in argument_types:
                raise ValueError(f"{context}: undeclared argument '{argument}'")
            actual_type = argument_types[argument]
            if not hierarchy.is_subtype(actual_type, expected_type):
                raise ValueError(
                    f"{context}: argument '{argument}' has type '{actual_type}', "
                    f"expected '{expected_type}'"
                )


def parse_domain(path: Path) -> DomainSchemas:
    normalize_domain_types(path)
    root = parse_sexpr_file(path)
    if not isinstance(root, list) or not root or root[0] != "define":
        raise ValueError(f"{path} is not a valid domain file")

    type_sections = _sections(root, ":types")
    if len(type_sections) > 1:
        raise ValueError(f"{path}: duplicate :types section")
    hierarchy = TypeHierarchy.from_declaration(type_sections[0][1:]) if type_sections else TypeHierarchy.object_only()

    predicate_sections = _sections(root, ":predicates")
    if len(predicate_sections) != 1:
        raise ValueError(f"{path}: expected exactly one :predicates section")
    predicate_types: dict[str, tuple[str, ...]] = {}
    for declaration in predicate_sections[0][1:]:
        if not isinstance(declaration, list) or not declaration or not isinstance(declaration[0], str):
            raise ValueError(f"{path}: invalid predicate declaration")
        params, param_types, _ = typed_symbol_map(
            declaration[1:],
            context=f"{path}: predicate {declaration[0]}",
        )
        if any(not param.startswith("?") for param in params):
            raise ValueError(f"{path}: predicate parameters must be variables")
        for type_name in param_types.values():
            hierarchy.require(type_name, f"{path}: predicate {declaration[0]}")
        signature = tuple(param_types[param] for param in params)
        if declaration[0] in predicate_types:
            if predicate_types[declaration[0]] != signature:
                raise ValueError(
                    f"{path}: conflicting predicate signature '{declaration[0]}'"
                )
            raise ValueError(f"{path}: duplicate predicate '{declaration[0]}'")
        predicate_types[declaration[0]] = signature

    schemas = DomainSchemas(
        type_hierarchy=hierarchy,
        predicate_types=predicate_types,
    )
    for item in root[1:]:
        if not isinstance(item, list) or not item or item[0] != ":action":
            continue
        if len(item) < 2 or not isinstance(item[1], str) or item[1] in schemas:
            raise ValueError(f"{path}: invalid or duplicate action")

        fields = {}
        for index in range(2, len(item), 2):
            if index + 1 >= len(item) or not isinstance(item[index], str):
                raise ValueError(f"{path}: malformed action '{item[1]}'")
            if item[index] in fields:
                raise ValueError(f"{path}: duplicate action field '{item[index]}'")
            fields[item[index]] = item[index + 1]
        if set(fields) != {":parameters", ":precondition", ":effect"}:
            raise ValueError(f"{path}: invalid fields in action '{item[1]}'")

        if not isinstance(fields[":parameters"], list):
            raise ValueError(f"{path}: invalid parameters in action '{item[1]}'")
        params, param_types, explicit_types = typed_symbol_map(
            fields[":parameters"],
            context=f"{path}: parameters of {item[1]}",
        )
        if any(not param.startswith("?") for param in params):
            raise ValueError(f"{path}: action parameters must be variables")
        for type_name in param_types.values():
            hierarchy.require(type_name, f"{path}: parameters of {item[1]}")

        pre_pos, pre_neg = read_literals(fields[":precondition"])
        add_eff, del_eff = read_literals(fields[":effect"])
        _validate_literal_types(
            pre_pos | pre_neg | add_eff | del_eff,
            predicate_types,
            param_types,
            hierarchy,
            f"{path}: action {item[1]}",
        )
        if (add_eff & del_eff):
            raise ValueError(f"{path}: action '{item[1]}' adds and deletes one literal")
        schemas[item[1]] = ActionSchema(
            item[1],
            params,
            pre_pos,
            pre_neg,
            add_eff,
            del_eff,
            param_types,
            explicit_types,
            hierarchy,
        )
    return schemas


def parse_problem_model(
    path: Path,
    schemas: DomainSchemas | None = None,
) -> ProblemModel:
    root = parse_sexpr_file(path)
    if not isinstance(root, list) or not root or root[0] != "define":
        raise ValueError(f"{path} is not a valid problem file")

    object_sections = _sections(root, ":objects")
    init_sections = _sections(root, ":init")
    goal_sections = _sections(root, ":goal")
    if len(object_sections) != 1 or len(init_sections) != 1 or len(goal_sections) != 1:
        raise ValueError(f"{path}: problem requires one :objects, :init, and :goal")
    objects, object_types, _ = typed_symbol_map(
        object_sections[0][1:],
        context=f"{path}: objects",
    )
    if any(name.startswith("?") for name in objects):
        raise ValueError(f"{path}: problem objects cannot be variables")

    hierarchy = schemas.type_hierarchy if schemas is not None else TypeHierarchy.object_only()
    for type_name in object_types.values():
        hierarchy.require(type_name, f"{path}: objects")

    init_state: set[Literal] = set()
    init_negative: set[Literal] = set()
    for expression in init_sections[0][1:]:
        positive, negative = read_literals(expression)
        if len(positive) + len(negative) != 1:
            raise ValueError(f"{path}: :init facts must be atomic")
        init_state.update(positive)
        init_negative.update(negative)
    contradiction = init_state & init_negative
    if contradiction:
        raise ValueError(f"{path}: contradictory init literal {sorted(contradiction)}")

    if len(goal_sections[0]) != 2:
        raise ValueError(f"{path}: invalid :goal")
    goal_positive, goal_negative = read_literals(goal_sections[0][1])

    if schemas is not None:
        _validate_literal_types(
            init_state | init_negative | goal_positive | goal_negative,
            schemas.predicate_types,
            object_types,
            hierarchy,
            str(path),
        )
    return ProblemModel(
        object_types,
        init_state,
        init_negative,
        goal_positive,
        goal_negative,
    )


def parse_problem(
    path: Path,
    schemas: DomainSchemas | None = None,
) -> tuple[set[Literal], set[Literal], set[Literal]]:
    problem = parse_problem_model(path, schemas)
    return problem.init_state, problem.goal_positive, problem.goal_negative


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
            if not parts:
                raise ValueError(f"Empty plan action in {path}")
            actions.append((parts[0], parts[1:]))
    return actions, comments


def ground_plan(
    raw_plan: list[tuple[str, list[str]]],
    schemas: dict[str, ActionSchema],
    object_types: dict[str, str] | None = None,
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
        if object_types is not None:
            for parameter, argument in zip(schema.params, args):
                if argument not in object_types:
                    raise ValueError(
                        f"Action {name} uses undeclared object '{argument}'"
                    )
                actual_type = object_types[argument]
                expected_type = schema.param_types.get(parameter, "object")
                if not schema.type_hierarchy.is_subtype(actual_type, expected_type):
                    raise ValueError(
                        f"Type mismatch for action {name}: object '{argument}' has "
                        f"type '{actual_type}', parameter '{parameter}' expects "
                        f"'{expected_type}'"
                    )

        mapping = dict(zip(schema.params, args))

        def substitute(literals: set[Literal]) -> set[Literal]:
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
                tuple(schema.param_types.get(parameter, "object") for parameter in schema.params),
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
