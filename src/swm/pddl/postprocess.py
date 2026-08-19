from __future__ import annotations

import re
from typing import Union

PDDL = Union[str, list["PDDL"]]


class PDDLPostprocessError(ValueError):
    pass


def _name(token: str) -> str:
    if token == "-":
        return token

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
            unique = {repr(value): value for value in values[1:]}
            return ["and"] + [unique[key] for key in sorted(unique)]
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
    if expression[0] in {"or", "when", "forall", "exists", "imply", "="}:
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
            if action_names[action[1]] != canonical:
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

    def validate(
        expression: PDDL,
        variables: set[str] | None,
        context: str,
        negative: bool,
    ) -> None:
        for is_negative, fact in _facts(expression):
            if is_negative and not negative:
                raise PDDLPostprocessError(
                    f"negative literal is not supported in {context}"
                )
            name = fact[0]
            if name not in declarations:
                raise PDDLPostprocessError(f"undeclared predicate in {context}: {name}")
            if declarations[name] != len(fact) - 1:
                raise PDDLPostprocessError(
                    f"predicate arity mismatch in {context}: {name}"
                )
            used_predicates.add(name)
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
        validate(action[5], parameters, action[1] + " precondition", False)
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
    predicates = sorted(
        (predicate for predicate in predicates if predicate[0] in used_predicates),
        key=lambda value: (value[0], len(value), repr(value)),
    )
    clean_actions.sort(key=lambda value: value[1])
    init_facts = [fact for _, fact in _facts(init)]
    init_facts = [
        value
        for _, value in sorted({repr(value): value for value in init_facts}.items())
    ]

    domain_lines = [f"(define (domain {domain[1][1]})"]
    if requirements_section is not None:
        domain_lines.append(f"  (:requirements {' '.join(requirements_section[1:])})")
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
