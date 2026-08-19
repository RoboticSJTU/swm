#!/usr/bin/env python3
"""Classify and align predicates in an episode PDDL dataset."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regenerate_plan_nl import translate_plan_text


DATASET_ROOT = PROJECT_ROOT / "eval_results" / "gpt-5.6-sol" / "human_aug"
UNIFIED_DOMAIN = DATASET_ROOT / "unified_domain.pddl"
CLASSIFICATION_FILE = DATASET_ROOT / "predicate_classification.json"
MODEL = "gpt-5.6-sol"
WORKERS = 8
CLASSIFICATION_ATTEMPTS = 3
ROUND_RE = re.compile(r"round[_-]?(\d+)$", re.IGNORECASE)
TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
CATEGORY_ORDER = {"type": 0, "state": 1, "relation": 2}
SUBCATEGORY_ORDER = {
    "type": {"object": 0, "material": 1, "role": 2},
    "state": {"configuration": 0, "availability": 1, "result": 2},
    "relation": {"spatial": 0, "non-spatial": 1},
}

Node = str | list["Node"]
Labels = dict[str, dict[str, Any]]


def parse_sexp(text: str) -> Node:
    tokens = TOKEN_RE.findall(
        "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    )
    position = 0

    def parse() -> Node:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("unexpected end of PDDL expression")
        token = tokens[position]
        position += 1
        if token != "(":
            if token == ")":
                raise ValueError("unexpected closing parenthesis")
            return token
        result: list[Node] = []
        while position < len(tokens) and tokens[position] != ")":
            result.append(parse())
        if position == len(tokens):
            raise ValueError("missing closing parenthesis")
        position += 1
        return result

    if not tokens:
        raise ValueError("empty PDDL expression")
    result = parse()
    if position != len(tokens):
        raise ValueError("trailing PDDL tokens")
    return result


def sexp(node: Node) -> str:
    return node if isinstance(node, str) else "(" + " ".join(map(sexp, node)) + ")"


def head(node: Node) -> str:
    return (
        node[0].lower()
        if isinstance(node, list) and node and isinstance(node[0], str)
        else ""
    )


def predicate_name(node: Node) -> str:
    if head(node) == "not":
        if not isinstance(node, list) or len(node) != 2:
            raise ValueError(f"invalid negative literal: {sexp(node)}")
        node = node[1]
    name = head(node)
    return "" if name in {"", "and", "or", "forall", "exists", "when", "imply"} else name


def find_token(text: str, token: str, start: int = 0) -> int:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text, start):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if ";" not in text[line_start : match.start()]:
            return match.start()
    return -1


def matching_paren(text: str, start: int) -> int:
    depth = 0
    comment = False
    for index in range(start, len(text)):
        char = text[index]
        if comment:
            comment = char != "\n"
        elif char == ";":
            comment = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unmatched '(' at position {start}")


def form_span(text: str, keyword: str, start: int = 0) -> tuple[int, int]:
    keyword_pos = find_token(text, keyword, start)
    if keyword_pos < 0:
        raise ValueError(f"missing {keyword}")
    open_pos = text.rfind("(", start, keyword_pos + 1)
    if open_pos < 0:
        raise ValueError(f"missing '(' before {keyword}")
    return open_pos, matching_paren(text, open_pos) + 1


def expression_span(
    text: str, keyword: str, action_start: int, action_end: int
) -> tuple[int, int]:
    start = find_token(text, keyword, action_start)
    if start < 0 or start >= action_end:
        raise ValueError(f"missing {keyword} in action")
    start += len(keyword)
    while start < action_end:
        if text[start].isspace():
            start += 1
        elif text[start] == ";":
            newline = text.find("\n", start)
            start = action_end if newline < 0 else newline + 1
        else:
            break
    if start >= action_end or text[start] != "(":
        raise ValueError(f"{keyword} is not followed by an expression")
    end = matching_paren(text, start) + 1
    if end > action_end:
        raise ValueError(f"{keyword} expression escapes its action")
    return start, end


def action_spans(text: str):
    position = 0
    while (action_pos := find_token(text, ":action", position)) >= 0:
        start = text.rfind("(", position, action_pos + 1)
        if start < 0:
            raise ValueError("missing '(' before :action")
        end = matching_paren(text, start) + 1
        yield start, end
        position = end


def indent_at(text: str, position: int) -> int:
    line_start = text.rfind("\n", 0, position) + 1
    return len(text[line_start:position]) - len(text[line_start:position].lstrip())


def render_logic(node: Node, indent: int) -> str:
    if head(node) not in {"and", "or"} or not isinstance(node, list) or len(node) == 1:
        return sexp(node)
    lines = [f"({node[0]}"]
    for child in node[1:]:
        child_lines = render_logic(child, indent + 2).splitlines()
        lines.append(" " * (indent + 2) + child_lines[0])
        lines.extend(child_lines[1:])
    lines.append(" " * indent + ")")
    return "\n".join(lines)


def predicate_declarations(text: str) -> list[list[Node]]:
    start, end = form_span(text, ":predicates")
    form = parse_sexp(text[start:end])
    if not isinstance(form, list) or head(form) != ":predicates":
        raise ValueError("invalid :predicates form")
    declarations = []
    for declaration in form[1:]:
        if not isinstance(declaration, list) or not declaration or not predicate_name(declaration):
            raise ValueError(f"invalid predicate declaration: {sexp(declaration)}")
        declarations.append(declaration)
    return declarations


def predicate_arity(declaration: list[Node]) -> int:
    return sum(isinstance(item, str) and item.startswith("?") for item in declaration[1:])


def validate_labels(data: dict[str, Any], signatures: dict[str, int], digest: str) -> Labels:
    if data["unified_domain_sha256"] != digest:
        raise ValueError("classification was generated from a different unified domain")

    labels: Labels = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for raw in data["predicates"]:
        name = str(raw["name"]).strip().lower()
        category = str(raw["category"]).strip().lower()
        subcategory = str(raw["subcategory"]).strip().lower()
        mutex = raw["mutex_group"]
        if mutex is not None and not isinstance(mutex, str):
            raise ValueError(f"predicate '{name}' has invalid mutex_group")
        mutex = mutex.strip().lower() if mutex and mutex.strip().lower() != "null" else None
        arity = int(raw["arity"])

        if name in labels:
            raise ValueError(f"duplicate classification for predicate '{name}'")
        if name not in signatures:
            raise ValueError(f"classification contains unknown predicate '{name}'")
        if arity != signatures[name]:
            raise ValueError(f"predicate '{name}' has arity {arity}, expected {signatures[name]}")
        if category not in CATEGORY_ORDER or subcategory not in SUBCATEGORY_ORDER[category]:
            raise ValueError(f"predicate '{name}' has invalid category '{category}/{subcategory}'")
        if category in {"type", "state"} and arity != 1:
            raise ValueError(f"predicate '{name}' category does not match arity {arity}")
        if category == "relation" and arity < 2:
            raise ValueError(f"predicate '{name}' category does not match arity {arity}")
        if mutex and category != "state":
            raise ValueError(f"non-state predicate '{name}' cannot have a mutex_group")
        if not str(raw["reason"]).strip():
            raise ValueError(f"predicate '{name}' has an empty reason")

        labels[name] = {
            "arity": arity,
            "category": category,
            "subcategory": subcategory,
            "mutex": mutex,
        }
        if mutex:
            groups[mutex].append(name)

    missing = sorted(signatures.keys() - labels.keys())
    if missing:
        raise ValueError(f"classification is missing predicates: {', '.join(missing)}")
    singletons = [f"{group}={names[0]}" for group, names in groups.items() if len(names) == 1]
    if singletons:
        raise ValueError("mutex groups must contain at least two predicates: " + ", ".join(singletons))

    anchors = {group: min(names) for group, names in groups.items()}
    for name, label in labels.items():
        if label["category"] == "state":
            anchor = anchors[label["mutex"]] if label["mutex"] else name
            label["order"] = (1, anchor, name)
        else:
            label["order"] = (
                CATEGORY_ORDER[label["category"]],
                SUBCATEGORY_ORDER[label["category"]][label["subcategory"]],
                name,
            )
    return labels


def classification_prompt(unified_domain: str) -> str:
    return f"""
You are classifying every predicate in one PDDL domain. Return one JSON object
and nothing else.

Classification standard:
- Type: a unary predicate describing what something is.
  - object: a concrete entity, such as hand, plate, drawer, or kettle.
  - material: a substance or content, such as water, milk, or detergent.
  - role: a scene-specific role, such as bottom_drawer or hot_water_button.
- State: a unary predicate describing how something currently is.
  - configuration: physical, pose, device, or content state, such as open,
    closed, is_on, upright, empty, or full.
  - availability: whether something is usable or unobstructed, such as clear
    or hand_free.
  - result: an achieved processing result, such as heated, clean, rinsed,
    boiled, stirred, or wet.
- Relation: a predicate with two or more arguments.
  - spatial: a spatial or topological relation, such as on, in, under,
    beside, left_of, or inserted.
  - non-spatial: every other relation, such as holding, filled_with,
    handle_of, part_of, dispenses, or can_place_on.

Decision rule: first use arity. Unary predicates are Type or State: "what it
is" means Type and "how it currently is" means State. Predicates with two or
more arguments are Relation; use Spatial only when they directly describe
where entities are or how they are spatially connected.

Rules:
1. Include every predicate declared in :predicates exactly once. Copy its exact
   lowercase name and arity.
2. category is one of: type, state, relation.
3. subcategory must match the category: type -> object/material/role;
   state -> configuration/availability/result; relation -> spatial/non-spatial.
4. Type and State predicates must be unary. Relation predicates have arity >= 2.
5. For State predicates, mutex_group is a short semantic group name when two or
   more predicates are mutually exclusive values of the same state variable,
   for example open/closed, on/off, empty/full, locked/unlocked, or
   upright/inverted/sideways. Every member must use the identical group name.
   Use null when no mutually exclusive partner exists in this domain.
6. A mutex group is not merely a list of related states: its members must not be
   simultaneously true of the same object in a valid state.
7. reason is one short sentence based on predicate meaning and action usage.
8. Inspect action preconditions and effects to disambiguate names and identify
   state transitions. Do not invent predicates or labels.

Required JSON schema:
{{
  "predicates": [
    {{
      "name": "open",
      "arity": 1,
      "category": "state",
      "subcategory": "configuration",
      "mutex_group": "open_closed",
      "reason": "It describes the current configuration of an object."
    }}
  ]
}}

[Complete unified PDDL domain]
{unified_domain.strip()}
""".strip()


def load_labels() -> Labels:
    unified_domain = UNIFIED_DOMAIN.read_text(encoding="utf-8")
    signatures = {}
    for declaration in predicate_declarations(unified_domain):
        name = predicate_name(declaration)
        if name in signatures:
            raise ValueError(f"duplicate predicate '{name}' in unified domain")
        signatures[name] = predicate_arity(declaration)
    digest = hashlib.sha256(unified_domain.encode()).hexdigest()

    if CLASSIFICATION_FILE.is_file():
        data = json.loads(CLASSIFICATION_FILE.read_text(encoding="utf-8"))
        labels = validate_labels(data, signatures, digest)
        print(f"[classification] using validated cache: {CLASSIFICATION_FILE}")
        return labels

    from swm.llm import call_gpt_json

    prompt = classification_prompt(unified_domain)
    for attempt in range(CLASSIFICATION_ATTEMPTS):
        response = call_gpt_json(MODEL, prompt, [], temperature=0.0)
        data = {
            "schema_version": 1,
            "model": MODEL,
            "unified_domain": str(UNIFIED_DOMAIN),
            "unified_domain_sha256": digest,
            "predicates": response["predicates"],
        }
        try:
            labels = validate_labels(data, signatures, digest)
            break
        except ValueError as error:
            if attempt + 1 == CLASSIFICATION_ATTEMPTS:
                raise
            prompt += (
                f"\n\nPrevious response error: {error}\n"
                "Return the complete corrected JSON object, not a patch."
            )
    atomic_write(
        CLASSIFICATION_FILE, json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"[classification] wrote {len(labels)} predicates: {CLASSIFICATION_FILE}")
    return labels


def local_predicate_order(
    declarations: list[list[Node]], labels: Labels
) -> tuple[list[list[Node]], dict[str, int]]:
    names = [predicate_name(declaration) for declaration in declarations]
    if len(names) != len(set(names)):
        raise ValueError("local domain contains duplicate predicate declarations")
    for name, declaration in zip(names, declarations):
        if name not in labels:
            raise ValueError(f"predicate '{name}' is not classified")
        if predicate_arity(declaration) != labels[name]["arity"]:
            raise ValueError(f"predicate '{name}' has inconsistent arity")
    declarations.sort(key=lambda declaration: labels[predicate_name(declaration)]["order"])
    return declarations, {
        predicate_name(declaration): index for index, declaration in enumerate(declarations)
    }


def align_logic(node: Node, order: dict[str, int], labels: Labels, effect: bool) -> Node:
    if isinstance(node, str) or not node:
        return node
    operator = head(node)
    children = [align_logic(child, order, labels, effect) for child in node[1:]]
    if operator not in {"and", "or"}:
        return [node[0], *children]

    def sort_key(item: Node):
        name = predicate_name(item)
        rank = order[name] if name in order else len(order)
        return rank, name, head(item) == "not", sexp(item).lower()

    if not effect or operator == "or":
        children.sort(key=sort_key)
        return [node[0], *children]

    units: list[list[Node]] = []
    transitions: dict[tuple[str, tuple[str, ...]], list[Node]] = {}
    for child in children:
        name = predicate_name(child)
        literal = child[1] if head(child) == "not" and isinstance(child, list) else child
        args = (
            tuple(str(item).lower() for item in literal[1:])
            if isinstance(literal, list) and all(isinstance(item, str) for item in literal[1:])
            else ()
        )
        label = labels[name] if name in labels else None
        if label and label["category"] == "state" and label["mutex"] and args:
            key = label["mutex"], args
            if key not in transitions:
                transitions[key] = []
                units.append(transitions[key])
            transitions[key].append(child)
        else:
            units.append([child])

    units.sort(key=lambda unit: min(sort_key(item) for item in unit))
    result = []
    for unit in units:
        unit.sort(key=lambda item: (head(item) != "not", sort_key(item)))
        result.extend(unit)
    return [node[0], *result]


def align_domain(text: str, labels: Labels) -> tuple[str, dict[str, int]]:
    declarations, order = local_predicate_order(predicate_declarations(text), labels)
    start, end = form_span(text, ":predicates")
    indent = indent_at(text, start)
    predicate_block = "\n".join(
        ["(:predicates"]
        + [" " * (indent + 2) + sexp(item) for item in declarations]
        + [" " * indent + ")"]
    )
    replacements = [(start, end, predicate_block)]

    for action_start, action_end in action_spans(text):
        for keyword, effect in ((":precondition", False), (":effect", True)):
            start, end = expression_span(text, keyword, action_start, action_end)
            keyword_pos = find_token(text, keyword, action_start)
            aligned = align_logic(parse_sexp(text[start:end]), order, labels, effect)
            replacements.append(
                (start, end, render_logic(aligned, indent_at(text, keyword_pos)))
            )

    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text, order


def align_problem(text: str, order: dict[str, int]) -> str:
    start, end = form_span(text, ":init")
    form = parse_sexp(text[start:end])
    if not isinstance(form, list) or head(form) != ":init":
        raise ValueError("invalid :init form")

    def sort_key(fact: Node):
        name = predicate_name(fact)
        if name not in order:
            raise ValueError(f":init uses undeclared predicate '{name}'")
        return order[name], name, head(fact) == "not", sexp(fact).lower()

    facts = sorted(form[1:], key=sort_key)
    indent = indent_at(text, start)
    block = "\n".join(
        ["(:init"]
        + [" " * (indent + 2) + sexp(fact) for fact in facts]
        + [" " * indent + ")"]
    )
    return text[:start] + block + text[end:]


def canonical(node: Node):
    if isinstance(node, str):
        return node.lower()
    values = [canonical(child) for child in node]
    if head(node) in {"and", "or", ":predicates", ":init"}:
        values[1:] = sorted(values[1:], key=repr)
    return tuple(values)


def plan_actions(text: str) -> list[tuple[str, ...]]:
    actions = []
    for raw_line in text.splitlines():
        line = re.sub(r";.*$", "", raw_line).strip()
        line = re.sub(r"^\d+\s*:\s*", "", line)
        line = re.sub(r"\s*\[[\d.]+\]\s*$", "", line).strip()
        if not line:
            continue
        if not line.startswith("(") or not line.endswith(")"):
            raise ValueError(f"unrecognized plan line: {raw_line.strip()}")
        tokens = tuple(line[1:-1].lower().split())
        if tokens:
            actions.append(tokens)
    return actions


def atomic_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def process_round(round_dir: Path, labels: Labels) -> bool:
    domain_path = round_dir / "domain.pddl"
    problem_path = round_dir / "problem.pddl"
    old_domain = domain_path.read_text(encoding="utf-8")
    old_problem = problem_path.read_text(encoding="utf-8")
    new_domain, order = align_domain(old_domain, labels)
    new_problem = align_problem(old_problem, order)

    if canonical(parse_sexp(old_domain)) != canonical(parse_sexp(new_domain)):
        raise ValueError(f"{domain_path}: rewrite changed more than expression order")
    if canonical(parse_sexp(old_problem)) != canonical(parse_sexp(new_problem)):
        raise ValueError(f"{problem_path}: rewrite changed more than expression order")
    if align_domain(new_domain, labels)[0] != new_domain:
        raise ValueError(f"{domain_path}: rewrite is not idempotent")
    if align_problem(new_problem, order) != new_problem:
        raise ValueError(f"{problem_path}: rewrite is not idempotent")

    if new_domain == old_domain and new_problem == old_problem:
        return False

    judge = json.loads((round_dir / "judge.json").read_text(encoding="utf-8"))
    if judge["pass"] is not True:
        raise ValueError(f"{round_dir}: this round has no passing judge")
    old_plan = (round_dir / "plan.txt").read_text(encoding="utf-8")
    if not old_plan.strip():
        raise ValueError(f"{round_dir}: empty plan.txt")

    from swm.pddl.planner import solve_pddl, summarize_solver_error

    with tempfile.TemporaryDirectory(prefix="pddl-align-") as temp_name:
        temp_dir = Path(temp_name)
        temp_domain = temp_dir / "domain.pddl"
        temp_problem = temp_dir / "problem.pddl"
        temp_domain.write_text(new_domain, encoding="utf-8")
        temp_problem.write_text(new_problem, encoding="utf-8")
        if not solve_pddl(temp_domain, temp_problem):
            error = (temp_dir / "error.log").read_text(encoding="utf-8")
            raise ValueError(f"{round_dir}: {summarize_solver_error(error)}")
        new_plan = (temp_dir / "plan.txt").read_text(encoding="utf-8")
        if plan_actions(new_plan) != plan_actions(old_plan):
            raise ValueError(f"{round_dir}: regenerated plan differs from judged plan")
        new_plan_nl = translate_plan_text(new_domain, new_plan)

    atomic_write(domain_path, new_domain)
    atomic_write(problem_path, new_problem)
    atomic_write(round_dir / "plan.txt", new_plan)
    atomic_write(round_dir / "plan_nl.txt", new_plan_nl)
    return True


def select_rounds() -> list[Path]:
    def ids(episode: Path) -> tuple[int, int]:
        return int(episode.parent.name[5:]), int(episode.name[8:])

    selected = []
    for episode in sorted(DATASET_ROOT.glob("task_*/episode_*"), key=ids):
        rounds = []
        for child in episode.iterdir():
            match = ROUND_RE.fullmatch(child.name) if child.is_dir() else None
            if match and (child / "domain.pddl").is_file() and (child / "problem.pddl").is_file():
                rounds.append((int(match.group(1)), child))
        rounds.sort()
        if rounds:
            selected.append(rounds[-1][1])
    return selected


def run_alignment(rounds: list[Path], labels: Labels) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rounds": len(rounds),
        "unchanged": 0,
        "written": 0,
        "failed": 0,
        "errors": [],
    }
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_round, path, labels): path for path in rounds}
        for future in as_completed(futures):
            path = futures[future]
            try:
                if future.result():
                    summary["written"] += 1
                else:
                    summary["unchanged"] += 1
            except Exception as error:
                message = f"{path}: {error}"
                summary["failed"] += 1
                summary["errors"].append(message)
                print(f"[ERROR] {message}", file=sys.stderr)
    summary["errors"].sort()
    return summary


def main() -> None:
    labels = load_labels()
    summary = run_alignment(select_rounds(), labels)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
