"""Formatting shared by the PDDL training data exporters."""

from heapq import heappop, heappush

from .strips import parse_sexpr

ACTION_HEADS = {":action", ":durative-action"}


def parse_pddl(text):
    root = parse_sexpr(text, lowercase=False)
    if not isinstance(root, list):
        raise ValueError("PDDL 结构不完整")
    return root


def pddl_line(expression):
    if isinstance(expression, str):
        return expression
    return "(" + " ".join(pddl_line(item) for item in expression) + ")"


def expression_head(expression):
    if isinstance(expression, list) and expression and isinstance(expression[0], str):
        return expression[0].lower()
    return ""


def normalize_names(domain, problem):
    """Rename the domain, problem, and domain reference in the parsed export pair."""
    domain[1][1] = "domain_name"
    problem[1][1] = "problem_name"
    for section in problem[2:]:
        if expression_head(section) == ":domain":
            section[1] = domain[1][1]


def render_domain(domain):
    """Render all the sections of a parsed domain in their supplied order."""
    lines = ["(define " + pddl_line(domain[1])]
    for section in domain[2:]:
        head = expression_head(section)
        if head == ":predicates":
            lines.append("  (:predicates")
            lines.extend("    " + pddl_line(item) for item in section[1:])
            lines.append("  )")
        elif head in ACTION_HEADS:
            if len(section[2:]) % 2:
                raise ValueError(f"动作 {section[1]} 的字段不完整")
            lines.extend(["", f"  ({section[0]} {section[1]}"])
            for key, value in zip(section[2::2], section[3::2]):
                lines.append(f"    {key} {pddl_line(value)}")
            lines.append("  )")
        else:
            lines.append("  " + pddl_line(section))
    lines.append(")")
    return "\n".join(lines)


def _chain_order(atoms, positions):
    successors = {
        i: [j for j in positions if i != j and atoms[i][2].lower() == atoms[j][1].lower()]
        for i in positions
    }
    if not any(successors.values()):
        return []

    indegree = dict.fromkeys(positions, 0)
    for following in successors.values():
        for index in following:
            indegree[index] += 1

    ready = [index for index in positions if indegree[index] == 0]
    ordered = []
    while ready:
        index = heappop(ready)
        ordered.append(index)
        for following in successors[index]:
            indegree[following] -= 1
            if indegree[following] == 0:
                heappush(ready, following)

    return ordered if len(ordered) == len(positions) else []


def reorder_stacks(atoms, goal=False):
    """Order positive binary on and in chains from supported object to support or container."""
    result = list(atoms)
    front = []
    moved = set()
    for relation in ("on", "in"):
        positions = [
            index for index, atom in enumerate(atoms)
            if expression_head(atom) == relation and len(atom) == 3
            and all(isinstance(argument, str) for argument in atom[1:])
        ]
        order = _chain_order(atoms, positions)
        if not order:
            continue
        if goal:
            front.extend(atoms[index] for index in order)
            moved.update(positions)
        else:
            for position, index in zip(positions, order):
                result[position] = atoms[index]
    return front + [atom for index, atom in enumerate(result) if index not in moved]


def format_problem(problem):
    objects = None
    init = None
    goal = None
    domain_section = None
    extra_sections = []
    for section in problem[2:]:
        name = expression_head(section)
        if name == ":domain" and domain_section is None:
            domain_section = section
        elif name == ":objects" and objects is None:
            objects = section
        elif name == ":init":
            init = section[1:]
        elif name == ":goal":
            if len(section) != 2:
                raise ValueError(":goal 格式错误")
            goal = section[1]
        elif name not in {":domain", ":objects"}:
            extra_sections.append(section)

    if domain_section is None:
        raise ValueError("problem 缺少 :domain")
    lines = ["(define " + pddl_line(problem[1]), "  " + pddl_line(domain_section)]
    if objects is not None:
        lines.append("  " + pddl_line(objects))
    if init is not None:
        lines.append("  (:init")
        lines.extend("    " + pddl_line(atom) for atom in reorder_stacks(init))
        lines.append("  )")
    if goal is not None:
        atoms = goal[1:] if expression_head(goal) == "and" else [goal]
        lines.extend(["  (:goal", "    (and"])
        lines.extend("      " + pddl_line(atom) for atom in reorder_stacks(atoms, goal=True))
        lines.extend(["    )", "  )"])
    lines.extend("  " + pddl_line(section) for section in extra_sections)
    lines.append(")")
    return "\n".join(lines)
