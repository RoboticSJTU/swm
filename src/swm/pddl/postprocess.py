from __future__ import annotations

import re
from typing import Union

PDDL = Union[str, list["PDDL"]]

CATEGORY_ORDER = {"type": 0, "state": 1, "relation": 2}


class PDDLPostprocessError(ValueError):
    pass


def _head(node: PDDL) -> str:
    return node[0].lower() if isinstance(node, list) and node and isinstance(node[0], str) else ""


def _predicate_name(node: PDDL) -> str:
    if _head(node) == "not" and isinstance(node, list) and len(node) == 2:
        node = node[1]
    name = _head(node)
    return "" if name in {"", "and", "or", "when", "forall", "exists", "imply"} else name


def _predicate_arity(declaration: PDDL) -> int:
    if not isinstance(declaration, list):
        raise ValueError("invalid predicate declaration")
    return sum(isinstance(value, str) and value.startswith("?") for value in declaration[1:])


def _action_field(action: PDDL, field: str) -> PDDL:
    if not isinstance(action, list) or _head(action) != ":action":
        raise ValueError("expected an action")
    for index in range(2, len(action) - 1, 2):
        if action[index] == field:
            return action[index + 1]
    raise ValueError(f"action is missing {field}")


def _effect_predicates(node: PDDL, signatures: dict[str, int]) -> set[str]:
    """Return predicate names added or deleted by a STRIPS effect."""
    if isinstance(node, str) or not node:
        return set()
    operator = _head(node)
    if operator == "not":
        return _effect_predicates(node[1], signatures) if len(node) == 2 else set()
    if operator in signatures:
        arguments = node[1:]
        return {
            operator
        } if len(arguments) == signatures[operator] and all(
            isinstance(value, str) for value in arguments
        ) else set()
    if operator == "and":
        return set().union(
            *(_effect_predicates(child, signatures) for child in node[1:])
        )
    return set()


def build_predicate_labels(domain: PDDL) -> dict[str, dict[str, object]]:
    """Classify declarations using only the domain's action syntax."""
    if not isinstance(domain, list) or _head(domain) != "define":
        raise ValueError("invalid domain")

    declarations = next(
        (
            section[1:]
            for section in domain[2:]
            if isinstance(section, list) and _head(section) == ":predicates"
        ),
        None,
    )
    if declarations is None:
        raise ValueError("domain has no :predicates")

    signatures: dict[str, int] = {}
    for declaration in declarations:
        name = _predicate_name(declaration)
        arity = _predicate_arity(declaration)
        if not name or (name in signatures and signatures[name] != arity):
            raise ValueError(f"invalid predicate declaration: {declaration!r}")
        signatures.setdefault(name, arity)

    changed: set[str] = set()
    for action in (
        section
        for section in domain[2:]
        if isinstance(section, list) and _head(section) == ":action"
    ):
        effect = _action_field(action, ":effect")
        changed.update(_effect_predicates(effect, signatures))

    return {
        name: {
            "arity": arity,
            # holding is the binary state of a hand, not a generic relation.
            "category": "type" if arity == 1 and name not in changed else "state" if arity <= 1 or name == "holding" else "relation",
            "rank": index,
        }
        for index, (name, arity) in enumerate(signatures.items())
    }


def sort_predicate_declarations(declarations: list[PDDL], labels: dict[str, dict[str, object]]) -> list[PDDL]:
    for declaration in declarations:
        name = _predicate_name(declaration)
        label = labels.get(name)
        if label is None or _predicate_arity(declaration) != label["arity"]:
            raise ValueError(f"unclassified predicate '{name}'")
    return sorted(
        declarations,
        key=lambda declaration: (
            CATEGORY_ORDER[labels[_predicate_name(declaration)]["category"]],
            labels[_predicate_name(declaration)]["rank"],
        ),
    )


def literal_sort_key(
    item: PDDL,
    labels: dict[str, dict[str, object]],
    declaration_order: dict[str, int],
    parameter_order: dict[str, int],
) -> tuple[int, bool, int, int]:
    name = _predicate_name(item)
    literal = item[1] if _head(item) == "not" and isinstance(item, list) and len(item) == 2 else item
    first_argument = (
        literal[1].lower()
        if isinstance(literal, list) and len(literal) > 1 and isinstance(literal[1], str)
        else ""
    )
    label = labels.get(name)
    return (
        CATEGORY_ORDER[label["category"]] if label else len(CATEGORY_ORDER),
        _head(item) != "not",
        parameter_order.get(first_argument, len(parameter_order)),
        declaration_order.get(name, len(declaration_order)),
    )


def sort_logic(
    node: PDDL,
    labels: dict[str, dict[str, object]],
    declaration_order: dict[str, int],
    parameter_order: dict[str, int],
) -> PDDL:
    if isinstance(node, str) or not node:
        return node
    operator = _head(node)
    children = [sort_logic(child, labels, declaration_order, parameter_order) for child in node[1:]]
    if operator == "and":
        children.sort(key=lambda child: literal_sort_key(child, labels, declaration_order, parameter_order))
    return [node[0], *children]


def sort_action(
    action: list[PDDL], labels: dict[str, dict[str, object]], declaration_order: dict[str, int]
) -> None:
    parameters = _action_field(action, ":parameters")
    parameter_order = {
        value.lower(): index
        for index, value in enumerate(parameters if isinstance(parameters, list) else [])
        if isinstance(value, str) and value.startswith("?")
    }
    for index in range(2, len(action) - 1, 2):
        if action[index] in {":precondition", ":effect"}:
            action[index + 1] = sort_logic(action[index + 1], labels, declaration_order, parameter_order)


def sort_facts(
    facts: list[PDDL], labels: dict[str, dict[str, object]], declaration_order: dict[str, int]
) -> list[PDDL]:
    return sorted(facts, key=lambda fact: literal_sort_key(fact, labels, declaration_order, {}))


def _name(token: str) -> str:
    if token == "-":
        return token
    if token.startswith(":"):
        return token.lower()

    prefix = "?" if token.startswith("?") else ""
    value = token[len(prefix) :].lower().replace("-", "_")
    if value == "into":
        value = "in"
    elif value == "onto":
        value = "on"
    elif value == "pick_up":
        value = "pick"

    if value.startswith("pick_up_"):
        value = "pick_" + value[len("pick_up_") :]
    value = value.replace("_into_", "_in_").replace("_onto_", "_on_")
    return prefix + value


def _parse(text: str) -> PDDL:
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    root: list[PDDL] = []
    stack = [root]
    spellings = {}

    for token in tokens:
        if token == "(":
            node: list[PDDL] = []
            stack[-1].append(node)
            stack.append(node)
        elif token == ")":
            if len(stack) == 1:
                raise PDDLPostprocessError("unexpected closing parenthesis")
            stack.pop()
        else:
            lower = token.lower()
            separator_form = lower.replace("-", "_")
            if not token.startswith(("?", ":")):
                if separator_form in spellings and spellings[separator_form] != lower:
                    raise PDDLPostprocessError(
                        f"identifier separator collision: {spellings[separator_form]} / {lower}"
                    )
                spellings[separator_form] = lower
            stack[-1].append(_name(token))

    if len(stack) != 1:
        raise PDDLPostprocessError("unclosed parenthesis")
    if len(root) != 1 or not isinstance(root[0], list):
        raise PDDLPostprocessError("expected exactly one PDDL expression")

    def normalize(node: PDDL) -> PDDL:
        if isinstance(node, str):
            return node
        values = [normalize(value) for value in node]
        if values and values[0] == "and":
            unique = {}
            for value in values[1:]:
                unique.setdefault(repr(value), value)
            return ["and", *unique.values()]
        return values

    return normalize(root[0])


def _facts(expression: PDDL) -> list[tuple[bool, list[PDDL]]]:
    if not isinstance(expression, list) or not expression:
        raise PDDLPostprocessError("expected a predicate expression")
    if expression[0] == "and":
        result = []
        for child in expression[1:]:
            result.extend(_facts(child))
        return result
    if expression[0] == "not":
        if len(expression) != 2 or not isinstance(expression[1], list):
            raise PDDLPostprocessError("invalid negated literal")
        return [(True, expression[1])]
    if expression[0] in {"or", "when", "forall", "exists", "imply"}:
        raise PDDLPostprocessError(f"unsupported construct: {expression[0]}")
    if not isinstance(expression[0], str) or any(
        not isinstance(value, str) for value in expression[1:]
    ):
        raise PDDLPostprocessError("invalid predicate expression")
    return [(False, expression)]


def _inline(expression: PDDL) -> str:
    if isinstance(expression, str):
        return expression
    return "(" + " ".join(_inline(value) for value in expression) + ")"


def _render_expression(expression: PDDL, indent: int) -> list[str]:
    if isinstance(expression, list) and expression and expression[0] == "and":
        lines = [" " * indent + "(and"]
        lines.extend(" " * (indent + 2) + _inline(value) for value in expression[1:])
        lines.append(" " * indent + ")")
        return lines
    return [" " * indent + _inline(expression)]


def _action_comments(text: str) -> dict[str, str]:
    comments = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.search(r"\(\s*:action\s+([^\s()]+)", line, re.IGNORECASE)
        if not match:
            continue
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous >= 0 and lines[previous].lstrip().startswith(";"):
            comment = re.sub(
                r"\?[A-Za-z0-9_-]+",
                lambda value: _name(value.group()),
                lines[previous].strip(),
            )
            comments[_name(match.group(1))] = comment
    return comments


def _unordered_signature(node: PDDL):
    if isinstance(node, str):
        return node
    values = [_unordered_signature(value) for value in node]
    if _head(node) in {"and", "or"}:
        values[1:] = sorted(values[1:], key=repr)
    return tuple(values)


def postprocess_pddl(domain_text: str, problem_text: str) -> tuple[str, str]:
    comments = _action_comments(domain_text)
    domain = _parse(domain_text)
    problem = _parse(problem_text)

    if not isinstance(domain, list) or len(domain) < 2 or domain[0] != "define":
        raise PDDLPostprocessError("invalid Domain root")
    if (
        not isinstance(domain[1], list)
        or len(domain[1]) != 2
        or domain[1][0] != "domain"
    ):
        raise PDDLPostprocessError("missing Domain name")

    requirements_section = None
    predicate_section = None
    actions = []
    for section in domain[2:]:
        if not isinstance(section, list) or not section:
            raise PDDLPostprocessError("invalid Domain section")
        if section[0] == ":requirements" and requirements_section is None:
            if len(section) == 1 or any(
                not isinstance(requirement, str) or not requirement.startswith(":")
                for requirement in section[1:]
            ):
                raise PDDLPostprocessError("invalid :requirements")
            requirements_section = section
        elif section[0] == ":predicates" and predicate_section is None:
            predicate_section = section
        elif section[0] == ":action":
            actions.append(section)
        else:
            raise PDDLPostprocessError(f"unsupported Domain section: {section[0]}")
    if predicate_section is None:
        raise PDDLPostprocessError("missing :predicates")

    declarations = {}
    predicates = []
    for predicate in predicate_section[1:]:
        if (
            not isinstance(predicate, list)
            or not predicate
            or not isinstance(predicate[0], str)
        ):
            raise PDDLPostprocessError("invalid predicate declaration")
        if any(
            not isinstance(value, str) or not value.startswith("?")
            for value in predicate[1:]
        ):
            raise PDDLPostprocessError(f"invalid declaration: {_inline(predicate)}")
        name, arity = predicate[0], len(predicate) - 1
        if name in declarations:
            if declarations[name] != arity:
                raise PDDLPostprocessError(f"predicate arity conflict: {name}")
            continue
        declarations[name] = arity
        predicates.append(predicate)

    clean_actions = []
    action_names = {}
    for action in actions:
        if len(action) < 4 or not isinstance(action[1], str) or len(action[2:]) % 2:
            raise PDDLPostprocessError("invalid action structure")
        fields = {}
        for index in range(2, len(action), 2):
            key = action[index]
            if not isinstance(key, str) or key in fields:
                raise PDDLPostprocessError(f"invalid action field in {action[1]}")
            fields[key] = action[index + 1]
        if set(fields) != {":parameters", ":precondition", ":effect"}:
            raise PDDLPostprocessError(f"invalid action fields: {action[1]}")

        parameters = fields[":parameters"]
        if not isinstance(parameters, list) or any(
            not isinstance(value, str) or not value.startswith("?")
            for value in parameters
        ):
            raise PDDLPostprocessError(f"invalid parameters: {action[1]}")
        if len(parameters) != len(set(parameters)):
            raise PDDLPostprocessError(f"duplicate parameter: {action[1]}")

        canonical = [
            ":action",
            action[1],
            ":parameters",
            parameters,
            ":precondition",
            fields[":precondition"],
            ":effect",
            fields[":effect"],
        ]
        if action[1] in action_names:
            if _unordered_signature(action_names[action[1]]) != _unordered_signature(canonical):
                raise PDDLPostprocessError(
                    f"different contracts share action name: {action[1]}"
                )
            continue
        action_names[action[1]] = canonical
        clean_actions.append(canonical)

    if not isinstance(problem, list) or len(problem) < 2 or problem[0] != "define":
        raise PDDLPostprocessError("invalid Problem root")
    if (
        not isinstance(problem[1], list)
        or len(problem[1]) != 2
        or problem[1][0] != "problem"
    ):
        raise PDDLPostprocessError("missing Problem name")

    sections = {}
    for section in problem[2:]:
        if (
            not isinstance(section, list)
            or not section
            or not isinstance(section[0], str)
        ):
            raise PDDLPostprocessError("invalid Problem section")
        if section[0] in sections:
            raise PDDLPostprocessError(f"duplicate Problem section: {section[0]}")
        sections[section[0]] = section
    if set(sections) != {":domain", ":objects", ":init", ":goal"}:
        raise PDDLPostprocessError(
            "Problem requires :domain, :objects, :init, and :goal"
        )
    if len(sections[":domain"]) != 2 or sections[":domain"][1] != domain[1][1]:
        raise PDDLPostprocessError("Domain/Problem name mismatch")
    if len(sections[":goal"]) != 2:
        raise PDDLPostprocessError("invalid :goal")

    objects = sections[":objects"][1:]
    if any(
        not isinstance(value, str) or value.startswith("?") or value == "-"
        for value in objects
    ):
        raise PDDLPostprocessError("only untyped Problem objects are supported")
    objects = sorted(set(objects))
    object_set = set(objects)
    used_predicates = set()
    uses_negative_preconditions = False
    uses_equality = False

    def validate(
        expression: PDDL,
        variables: set[str] | None,
        context: str,
        negative: bool,
    ) -> None:
        nonlocal uses_negative_preconditions, uses_equality
        for is_negative, fact in _facts(expression):
            name = fact[0]
            if name == "=":
                if not is_negative or not context.endswith(" precondition"):
                    raise PDDLPostprocessError(
                        "equality is only supported as a negated action precondition"
                    )
                if len(fact) != 3:
                    raise PDDLPostprocessError("equality expects exactly two arguments")
                uses_negative_preconditions = True
                uses_equality = True
            elif is_negative and not negative:
                raise PDDLPostprocessError(
                    f"negative literal is not supported in {context}"
                )
            elif name not in declarations:
                raise PDDLPostprocessError(f"undeclared predicate in {context}: {name}")
            elif declarations[name] != len(fact) - 1:
                raise PDDLPostprocessError(
                    f"predicate arity mismatch in {context}: {name}"
                )
            else:
                used_predicates.add(name)
            if is_negative and context.endswith(" precondition"):
                uses_negative_preconditions = True
            for argument in fact[1:]:
                if variables is not None:
                    if argument not in variables:
                        raise PDDLPostprocessError(
                            f"unbound variable in {context}: {argument}"
                        )
                elif argument.startswith("?") or argument not in object_set:
                    raise PDDLPostprocessError(
                        f"undeclared object in {context}: {argument}"
                    )

    for action in clean_actions:
        parameters = set(action[3])
        validate(action[5], parameters, action[1] + " precondition", True)
        validate(action[7], parameters, action[1] + " effect", True)
        effect_facts = _facts(action[7])
        if not effect_facts:
            raise PDDLPostprocessError(f"empty effect: {action[1]}")
        used = {
            argument
            for _, fact in _facts(action[5]) + effect_facts
            for argument in fact[1:]
        }
        unused = parameters - used
        if unused:
            raise PDDLPostprocessError(
                f"unused parameter in {action[1]}: {', '.join(sorted(unused))}"
            )
        positive = {repr(fact) for negated, fact in effect_facts if not negated}
        negative = {repr(fact) for negated, fact in effect_facts if negated}
        if positive & negative:
            raise PDDLPostprocessError(
                f"effect adds and deletes the same literal: {action[1]}"
            )

    init = ["and"] + sections[":init"][1:]
    goal = sections[":goal"][1]
    if not _facts(goal):
        raise PDDLPostprocessError("empty goal")
    validate(init, None, "init", False)
    validate(goal, None, "goal", False)
    labels = build_predicate_labels(domain)
    predicates = sort_predicate_declarations(
        [predicate for predicate in predicates if predicate[0] in used_predicates],
        labels,
    )
    declaration_order = {predicate[0]: index for index, predicate in enumerate(predicates)}
    clean_actions.sort(key=lambda value: value[1])
    for action in clean_actions:
        sort_action(action, labels, declaration_order)
    init_facts = [fact for _, fact in _facts(init)]
    unique_init = {}
    for fact in init_facts:
        unique_init.setdefault(repr(fact), fact)
    init_facts = sort_facts(list(unique_init.values()), labels, declaration_order)
    goal = sort_logic(goal, labels, declaration_order, {})

    requirements = list(requirements_section[1:]) if requirements_section else []
    if uses_negative_preconditions and ":negative-preconditions" not in requirements:
        requirements.append(":negative-preconditions")
    if uses_equality and ":equality" not in requirements:
        requirements.append(":equality")

    domain_lines = [f"(define (domain {domain[1][1]})"]
    if requirements:
        domain_lines.append(f"  (:requirements {' '.join(requirements)})")
    domain_lines.append("  (:predicates")
    domain_lines.extend("    " + _inline(predicate) for predicate in predicates)
    domain_lines.extend(["  )", ""])
    for action in clean_actions:
        if action[1] in comments:
            domain_lines.append("  " + comments[action[1]])
        else:
            phrase = action[1].replace("_", " ").capitalize()
            parameters = " ".join(action[3])
            suffix = f" with parameters {parameters}" if parameters else ""
            domain_lines.append(f"  ; {phrase}{suffix}.")
        domain_lines.append(f"  (:action {action[1]}")
        domain_lines.append(f"    :parameters {_inline(action[3])}")
        for label, expression in ((":precondition", action[5]), (":effect", action[7])):
            rendered = _render_expression(expression, 4)
            domain_lines.append(f"    {label} {rendered[0].strip()}")
            domain_lines.extend(rendered[1:])
        domain_lines.extend(["  )", ""])
    if domain_lines[-1] == "":
        domain_lines.pop()
    domain_lines.append(")")

    problem_lines = [
        f"(define (problem {problem[1][1]})",
        f"  (:domain {domain[1][1]})",
        f"  (:objects {' '.join(objects)})",
        "  (:init",
    ]
    problem_lines.extend("    " + _inline(fact) for fact in init_facts)
    problem_lines.append("  )")
    rendered_goal = _render_expression(goal, 4)
    problem_lines.append(f"  (:goal {rendered_goal[0].strip()}")
    problem_lines.extend(rendered_goal[1:])
    problem_lines.extend(["  )", ")"])

    return "\n".join(domain_lines) + "\n", "\n".join(problem_lines) + "\n"
