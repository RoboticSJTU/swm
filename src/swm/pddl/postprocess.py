"""Complete missing declarations without changing facts or action logic."""
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
    """Scan once; complete predicates, problem objects and free action parameters.

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
        if len(fields) != (len(action) - 2) // 2 or set(fields) != {":parameters", ":precondition", ":effect"}:
            raise ValueError(f"Malformed fields in action {action[1]}")
        if not isinstance(fields[":parameters"], list):
            raise ValueError("Action parameters must be a list")
        parameters = parse_symbols(fields[":parameters"], variables=True)
        free = set()
        for key in (":precondition", ":effect"):
            for atom, bound in _atoms(fields[key], set(parameters)):
                record(atom)
                free.update(arg for arg in atom[1:] if arg.startswith("?") and arg not in bound)
        if free:
            added_parameters[action[1]] = sorted(free)
            fields[":parameters"].extend(sorted(free))

    for expression in _section(problem_root, ":init")[1:] + _section(problem_root, ":goal")[1:]:
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
    }.items() if value}
    return (
        _format(domain_root) if added_predicates or added_parameters else domain,
        _format(problem_root) if added_objects else problem,
        changes,
    )
