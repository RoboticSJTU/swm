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


class DomainSchemas(dict[str, ActionSchema]):
    def __init__(
        self,
        *args,
        predicate_arities: dict[str, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.predicate_arities = predicate_arities or {}


@dataclass(frozen=True)
class ProblemModel:
    objects: frozenset[str]
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

    def to_line(self) -> str:
        return f"({self.name} {' '.join(self.args)})"


def parse_sexpr_file(path: Path):
    return parse_sexpr(path.read_text(encoding="utf-8"), str(path))


def parse_sexpr(text: str, context: str = "PDDL"):
    text = re.sub(r";[^\n]*", "", text).lower()
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    index = 0

    def parse():
        nonlocal index
        if index >= len(tokens):
            raise ValueError(f"Unexpected EOF in {context}")

        token = tokens[index]
        index += 1
        if token != "(":
            if token == ")":
                raise ValueError(f"Unexpected ')' in {context}")
            if token in {"when", "or", "exists", "imply"}:
                raise NotImplementedError(
                    f"Unsupported PDDL construct '{token}' in {context}"
                )
            return token

        expression = []
        while index < len(tokens) and tokens[index] != ")":
            expression.append(parse())
        if index >= len(tokens):
            raise ValueError(f"Missing ')' in {context}")
        index += 1
        return expression

    root = parse()
    if index != len(tokens):
        raise ValueError(f"Unparsed tokens remain in {context}")
    return root


def validate_untyped_pddl(text: str) -> None:
    """Reject typed declarations without rewriting the source model."""
    def visit(node):
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if not isinstance(head, str):
            raise ValueError("PDDL expression must start with a symbol")
        if head == ":types" or (head == ":requirements" and ":typing" in node):
            raise ValueError("Use unary category predicates instead of :types/:typing")
        declarations = []
        if head in {":parameters", ":objects", ":constants"}:
            declarations = [node[1:]]
        elif head == "forall":
            if len(node) != 3 or not isinstance(node[1], list):
                raise ValueError("Malformed universal condition")
            declarations = [node[1]]
        elif head == ":predicates":
            declarations = [item[1:] for item in node[1:] if isinstance(item, list)]
        elif head == ":action" and ":parameters" in node:
            index = node.index(":parameters") + 1
            if index >= len(node) or not isinstance(node[index], list):
                raise ValueError("Action parameters must be a list")
            declarations = [node[index]]
        if any("-" in declaration for declaration in declarations):
            raise ValueError("Typed declarations are not allowed; use unary category predicates")
        for child in node:
            visit(child)

    root = parse_sexpr(text)
    if not isinstance(root, list) or root[:1] != ["define"]:
        raise ValueError("Expected a complete PDDL domain or problem")
    visit(root)


def format_action_conditions(text: str) -> str:
    """Put each precondition/effect on one line without changing literal order."""
    tokens = list(re.finditer(r";[^\n]*|[()]|[^()\s;]+", text))
    edits = []
    for index, token in enumerate(tokens):
        if token.group().lower() not in {":precondition", ":effect"}:
            continue
        depth = 0
        parts = []
        for item in tokens[index + 1:]:
            value = item.group()
            if value.startswith(";"):
                continue
            if not parts and value != "(":
                raise ValueError(f"{token.group()} must be a parenthesized expression")
            parts.append(value)
            depth += (value == "(") - (value == ")")
            if depth == 0:
                expression = " ".join(parts).replace("( ", "(").replace(" )", ")")
                edits.append((token.end(), item.end(), " " + expression))
                break
        else:
            raise ValueError(f"Unclosed {token.group()} expression")
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


def read_literals(expression) -> tuple[set[Literal], set[Literal]]:
    if expression is None:
        return set(), set()
    if isinstance(expression, str):
        raise ValueError(f"Unexpected atom: {expression}")
    if not expression:
        raise ValueError("Empty logical expression")
    if expression[0] == "forall":
        raise NotImplementedError("Universal conditions are not supported by STRIPS plan reordering")
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


def parse_symbols(items: list, *, variables: bool = False) -> list[str]:
    pattern = r"\?[a-z][a-z0-9_-]*" if variables else r"[a-z][a-z0-9_-]*"
    if any(not isinstance(item, str) or not re.fullmatch(pattern, item) for item in items):
        raise ValueError("Expected untyped variables" if variables else "Expected untyped object names")
    if len(set(items)) != len(items):
        raise ValueError("Duplicate symbol in declaration")
    return list(items)


def _validate_literals(
    literals: set[Literal],
    predicate_arities: dict[str, int],
    arguments: set[str] | frozenset[str],
    context: str,
) -> None:
    for literal in literals:
        if literal[0] == "=":
            expected = 2
        elif literal[0] not in predicate_arities:
            raise ValueError(f"{context}: undeclared predicate '{literal[0]}'")
        else:
            expected = predicate_arities[literal[0]]
        if len(literal) - 1 != expected:
            raise ValueError(
                f"{context}: predicate '{literal[0]}' expects "
                f"{expected} arguments, got {len(literal) - 1}"
            )
        for argument in literal[1:]:
            if argument not in arguments:
                raise ValueError(f"{context}: undeclared argument '{argument}'")


def parse_domain(path: Path) -> DomainSchemas:
    text = path.read_text(encoding="utf-8")
    validate_untyped_pddl(text)
    root = parse_sexpr(text, str(path))
    if not isinstance(root, list) or not root or root[0] != "define":
        raise ValueError(f"{path} is not a valid domain file")

    predicate_sections = _sections(root, ":predicates")
    if len(predicate_sections) != 1:
        raise ValueError(f"{path}: expected exactly one :predicates section")
    predicate_arities: dict[str, int] = {}
    for declaration in predicate_sections[0][1:]:
        if not isinstance(declaration, list) or not declaration or not isinstance(declaration[0], str):
            raise ValueError(f"{path}: invalid predicate declaration")
        params = parse_symbols(declaration[1:], variables=True)
        if declaration[0] in predicate_arities:
            raise ValueError(f"{path}: duplicate predicate '{declaration[0]}'")
        predicate_arities[declaration[0]] = len(params)

    schemas = DomainSchemas(predicate_arities=predicate_arities)
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
        params = parse_symbols(fields[":parameters"], variables=True)

        pre_pos, pre_neg = read_literals(fields[":precondition"])
        add_eff, del_eff = read_literals(fields[":effect"])
        _validate_literals(
            pre_pos | pre_neg | add_eff | del_eff,
            predicate_arities,
            set(params),
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
        )
    return schemas


def parse_problem_model(
    path: Path,
    schemas: DomainSchemas | None = None,
) -> ProblemModel:
    text = path.read_text(encoding="utf-8")
    validate_untyped_pddl(text)
    root = parse_sexpr(text, str(path))
    if not isinstance(root, list) or not root or root[0] != "define":
        raise ValueError(f"{path} is not a valid problem file")

    object_sections = _sections(root, ":objects")
    init_sections = _sections(root, ":init")
    goal_sections = _sections(root, ":goal")
    if len(object_sections) != 1 or len(init_sections) != 1 or len(goal_sections) != 1:
        raise ValueError(f"{path}: problem requires one :objects, :init, and :goal")
    objects = frozenset(parse_symbols(object_sections[0][1:]))

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
        _validate_literals(
            init_state | init_negative | goal_positive | goal_negative,
            schemas.predicate_arities,
            objects,
            str(path),
        )
    return ProblemModel(
        objects,
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
    objects: frozenset[str] | None = None,
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
        if objects is not None:
            for argument in args:
                if argument not in objects:
                    raise ValueError(f"Action {name} uses undeclared object '{argument}'")

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
