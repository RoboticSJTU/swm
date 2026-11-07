"""Repair declarations without changing facts or action logic."""
from __future__ import annotations

from collections import defaultdict

from swm.pddl.strips import parse_sexpr, parse_symbols


def _section(root, name):
    matches = [node for node in root[2:] if node[:1] == [name]]
    if len(matches) > 1:
        raise ValueError(f"Duplicate {name} section")
    return matches[0] if matches else [name]


def _atoms(node, bound=frozenset()):
    """Yield atomic expressions with their lexical variable bindings."""
    if not isinstance(node, list):
        raise ValueError("Expected a logical expression")
    if not node:
        return
    head = node[0]
    if not isinstance(head, str):
        raise ValueError("A logical expression must start with a symbol")
    if head in {"and", "or", "not", "imply", "when"}:
        for child in node[1:]:
            yield from _atoms(child, bound)
    elif head in {"forall", "exists"}:
        if len(node) != 3 or not isinstance(node[1], list):
            raise ValueError("Malformed quantified expression")
        variables = parse_symbols(node[1], variables=True)
        yield from _atoms(node[2], bound | set(variables))
    else:
        if head != "=":
            parse_symbols([head])
        for argument in node[1:]:
            if not isinstance(argument, str):
                raise NotImplementedError("Declaration repair requires atomic terms")
            parse_symbols([argument], variables=argument.startswith("?"))
        yield node, bound


def _explicit_empty_expressions(node, path=()):
    """Make logical true/no-op explicit, leaving terms and bound lists alone."""
    if not isinstance(node, list):
        return []
    if not node:
        node.append("and")
        return [list(path)]
    if not isinstance(node[0], str):
        return []
    if node[0] in {"and", "or", "not", "imply", "when"}:
        return [location for i, child in enumerate(node[1:], 1)
                for location in _explicit_empty_expressions(child, (*path, i))]
    if node[0] in {"forall", "exists"} and len(node) == 3:
        return _explicit_empty_expressions(node[2], (*path, 2))
    return []


def _unconditional_effects(node, path=()):
    """Drop true when guards for validators requiring nonempty conditions."""
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        return []
    if node[0] == "and":
        return [location for i, child in enumerate(node[1:], 1)
                for location in _unconditional_effects(child, (*path, i))]
    if node[0] == "forall" and len(node) == 3:
        return _unconditional_effects(node[2], (*path, 2))
    if node[0] == "when" and len(node) == 3:
        paths = _unconditional_effects(node[2], (*path, 2))
        if node[1] == ["and"] and isinstance(node[2], list):
            node[:] = node[2]
            paths.append(list(path))
        return paths
    return []


def _sexpr(node):
    return "(" + " ".join(map(_sexpr, node)) + ")" if isinstance(node, list) else node


def _format(root):
    lines = [f"(define {_sexpr(root[1])}"]
    for node in root[2:]:
        if node[0] == ":action":
            lines.append(f"  (:action {node[1]}")
            lines.extend(f"    {key} {_sexpr(value)}" for key, value in zip(node[2::2], node[3::2]))
            lines.append("  )")
        elif node[0] in {":predicates", ":init"}:
            lines.append(f"  ({node[0]}")
            lines.extend(f"    {_sexpr(child)}" for child in node[1:])
            lines.append("  )")
        else:
            lines.append(f"  {_sexpr(node)}")
    return "\n".join([*lines, ")", ""])


def repair_pddl(domain: str, problem: str) -> tuple[str, str, dict]:
    """Complete missing declarations and remove identical duplicate actions.

    Arity conflicts are left for the planner. Unsupported/malformed structures
    raise instead of guessing; an unchanged document is returned verbatim.
    """
    domain_root, problem_root = parse_sexpr(domain), parse_sexpr(problem)
    for root, kind in ((domain_root, "domain"), (problem_root, "problem")):
        if (not isinstance(root, list) or root[:1] != ["define"] or len(root) < 2
                or not isinstance(root[1], list) or root[1][:1] != [kind]
                or any(not isinstance(node, list) or not node or not isinstance(node[0], str)
                       for node in root[2:])):
            raise ValueError(f"Expected a complete PDDL {kind}")
        for node in root[2:]:
            if node[0] in {":types", ":functions", ":derived"} or (
                node[0] == ":requirements" and ":typing" in node
            ):
                raise NotImplementedError("Declaration repair requires untyped PDDL without functions/derived rules")

    actions = {}
    duplicate_actions = defaultdict(int)
    for action in domain_root[2:]:
        if action[:1] != [":action"] or len(action) < 2:
            continue
        name = action[1]
        if not isinstance(name, str):
            raise ValueError("Action name must be a symbol")
        if name in actions:
            if action != actions[name]:
                raise ValueError(f"Conflicting action definitions: {name}")
            duplicate_actions[name] += 1
        else:
            actions[name] = action
    if duplicate_actions:
        seen = set()
        nodes = []
        for node in domain_root[2:]:
            if node[:1] == [":action"]:
                if node[1] in seen:
                    continue
                seen.add(node[1])
            nodes.append(node)
        domain_root[2:] = nodes

    predicates = _section(domain_root, ":predicates")
    objects = _section(problem_root, ":objects")
    declared_predicates = set()
    for declaration in predicates[1:]:
        if not isinstance(declaration, list) or not declaration:
            raise ValueError("Malformed predicate declaration")
        parse_symbols([declaration[0]])
        parse_symbols(declaration[1:], variables=True)
        declared_predicates.add(declaration[0])
    declared_objects = set(parse_symbols(objects[1:]))
    declared_objects.update(parse_symbols(_section(domain_root, ":constants")[1:]))

    arities = defaultdict(set)
    used_objects = set()
    added_parameters = {}
    action_defaults = {}
    empty_expressions = []
    true_when_effects = []

    def record(atom):
        if atom[0] != "=":
            arities[atom[0]].add(len(atom) - 1)

    for action in domain_root[2:]:
        if action[0] != ":action":
            continue
        if len(action) < 2 or not isinstance(action[1], str) or len(action) % 2:
            raise ValueError("Malformed action declaration")
        if any(not isinstance(key, str) for key in action[2::2]):
            raise ValueError("Action fields must start with a symbol")
        fields = dict(zip(action[2::2], action[3::2]))
        if (len(fields) != (len(action) - 2) // 2
                or not set(fields) <= {":parameters", ":precondition", ":effect"}
                or ":effect" not in fields):
            raise ValueError(f"Malformed fields in action {action[1]}")
        # PDDL permits omitted empty parameters and an omitted true
        # precondition. Make these defaults explicit for the interpreter.
        defaults = {key: value for key, value in ((":parameters", []), (":precondition", ["and"]))
                    if key not in fields}
        if defaults:
            action_defaults[action[1]] = {key: value.copy() for key, value in defaults.items()}
            fields.update(defaults)
            action[2:] = [item for key in (":parameters", ":precondition", ":effect")
                          for item in (key, fields[key])]
        if not isinstance(fields[":parameters"], list):
            raise ValueError("Action parameters must be a list")
        parameters = parse_symbols(fields[":parameters"], variables=True)
        free = set()
        for key in (":precondition", ":effect"):
            paths = _explicit_empty_expressions(fields[key])
            if paths:
                empty_expressions.append({"action": action[1], "field": key, "paths": paths})
            if key == ":effect":
                paths = _unconditional_effects(fields[key])
                if paths:
                    true_when_effects.append({"action": action[1], "field": key, "paths": paths})
            for atom, bound in _atoms(fields[key], set(parameters)):
                record(atom)
                free.update(arg for arg in atom[1:] if arg.startswith("?") and arg not in bound)
        if free:
            added_parameters[action[1]] = sorted(free)
            fields[":parameters"].extend(sorted(free))

    goal = _section(problem_root, ":goal")
    if len(goal) == 2:
        paths = _explicit_empty_expressions(goal[1])
        if paths:
            empty_expressions.append({"field": ":goal", "paths": paths})
    for expression in _section(problem_root, ":init")[1:] + goal[1:]:
        for atom, _ in _atoms(expression):
            record(atom)
            used_objects.update(arg for arg in atom[1:] if not arg.startswith("?"))

    added_predicates = {name: next(iter(arities[name])) for name in sorted(arities)
                        if name not in declared_predicates and len(arities[name]) == 1}
    added_objects = sorted(used_objects - declared_objects)
    for name, arity in added_predicates.items():
        predicates.append([name, *(f"?x{i}" for i in range(1, arity + 1))])
    objects.extend(added_objects)
    if added_predicates and predicates not in domain_root:
        index = next((i for i, node in enumerate(domain_root)
                      if isinstance(node, list) and node[:1] == [":action"]), len(domain_root))
        domain_root.insert(index, predicates)
    if added_objects and objects not in problem_root:
        index = next((i for i, node in enumerate(problem_root)
                      if isinstance(node, list) and node[:1] in ([":init"], [":goal"])), len(problem_root))
        problem_root.insert(index, objects)

    changes = {key: value for key, value in {
        "predicates": added_predicates, "objects": added_objects, "parameters": added_parameters,
        "duplicate_actions": dict(duplicate_actions), "action_defaults": action_defaults,
        "empty_expressions": empty_expressions,
        "true_when_effects": true_when_effects,
    }.items() if value}
    return (
        _format(domain_root) if added_predicates or added_parameters or duplicate_actions or action_defaults
            or true_when_effects or any("action" in change for change in empty_expressions) else domain,
        _format(problem_root) if added_objects or any("action" not in change for change in empty_expressions) else problem,
        changes,
    )
