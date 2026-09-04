#!/usr/bin/env python3
"""Canonicalize approved PDDL operator contracts in human and human_aug.

Only the highest numbered round in each episode is considered.  The script is a
dry run unless --apply is supplied.  It intentionally leaves kf_actions files and
all older rounds untouched.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import hashlib
import itertools
import json
import re
import shutil
import sys
import tempfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = PROJECT_ROOT / "eval_results" / "gpt-5.6-sol"
ROUND_RE = re.compile(r"roun(?:d)?[_-]?(\d+)$", re.IGNORECASE)
TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
ACTION_RE = re.compile(r"\(\s*:action\s+([^\s()]+)", re.IGNORECASE)


Node = str | list["Node"]


def strip_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def parse_sexp(text: str) -> Node:
    tokens = TOKEN_RE.findall(strip_comments(text))
    pos = 0

    def parse_one() -> Node:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("unexpected end of S-expression")
        token = tokens[pos]
        if token != "(":
            if token == ")":
                raise ValueError("unexpected closing parenthesis")
            pos += 1
            return token
        pos += 1
        result: list[Node] = []
        while pos < len(tokens) and tokens[pos] != ")":
            result.append(parse_one())
        if pos >= len(tokens):
            raise ValueError("missing closing parenthesis")
        pos += 1
        return result

    root = parse_one()
    if pos != len(tokens):
        raise ValueError("trailing S-expression tokens")
    return root


def sexp(node: Node) -> str:
    if isinstance(node, str):
        return node
    return "(" + " ".join(sexp(item) for item in node) + ")"


def find_matching_paren(text: str, start: int) -> int:
    depth = 0
    comment = False
    for index in range(start, len(text)):
        char = text[index]
        if comment:
            if char == "\n":
                comment = False
            continue
        if char == ";":
            comment = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unbalanced parentheses")


def latest_rounds(dataset_root: Path, task_ids: set[int] | None = None) -> list[Path]:
    rounds: list[Path] = []
    for episode in sorted(dataset_root.glob("task_*/episode_*")):
        task_match = re.fullmatch(r"task_(\d+)", episode.parent.name)
        if not task_match:
            continue
        if task_ids is not None and int(task_match.group(1)) not in task_ids:
            continue
        candidates: list[tuple[int, Path]] = []
        for child in episode.iterdir():
            match = ROUND_RE.fullmatch(child.name) if child.is_dir() else None
            if match and (child / "domain.pddl").is_file():
                candidates.append((int(match.group(1)), child))
        if candidates:
            rounds.append(max(candidates, key=lambda item: item[0])[1])
    return rounds


def task_id(round_dir: Path) -> int:
    return int(round_dir.parent.parent.name.removeprefix("task_"))


def replace_token(text: str, old: str, new: str) -> str:
    return re.sub(rf"(?<![A-Za-z0-9_?-]){re.escape(old)}(?![A-Za-z0-9_-])", new, text)


def replace_predicate_heads(text: str, aliases: dict[str, str]) -> str:
    for old, new in aliases.items():
        text = re.sub(
            rf"(\(\s*){re.escape(old)}(?=\s|\))",
            rf"\g<1>{new}",
            text,
            flags=re.IGNORECASE,
        )
    return text


def atom(node: Node, name: str, args: Iterable[str]) -> bool:
    return isinstance(node, list) and node == [name, *args]


def conjunction(expr: Node) -> list[Node]:
    if isinstance(expr, list) and expr and expr[0] == "and":
        return list(expr[1:])
    return [expr]


def make_and(items: Iterable[Node]) -> Node:
    return ["and", *items]


def has_literal(expr: Node, pred: str, *args: str, negative: bool = False) -> bool:
    target: Node = [pred, *args]
    if negative:
        target = ["not", target]
    return target in conjunction(expr)


def add_literal(expr: Node, literal: Node) -> Node:
    items = conjunction(expr)
    if literal not in items:
        items.append(literal)
    return make_and(items)


def remove_literal(expr: Node, literal: Node) -> Node:
    return make_and(item for item in conjunction(expr) if item != literal)


def unary_var(expr: Node, predicate_names: set[str]) -> str | None:
    for item in conjunction(expr):
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and item[0] in predicate_names
            and isinstance(item[1], str)
            and item[1].startswith("?")
        ):
            return item[1]
    return None


def unary_type(expr: Node, variable: str) -> str | None:
    ignored = {
        "hand_free", "open", "closed", "is_on", "is_off", "locked",
        "unlocked", "flat", "vertical", "upright", "clear", "wet",
        "rinsed", "poured", "heated", "boiled",
    }
    for item in conjunction(expr):
        if (
            isinstance(item, list)
            and len(item) == 2
            and item[1] == variable
            and isinstance(item[0], str)
            and item[0] not in ignored
        ):
            return item[0]
    return None


def fresh_variable(parameters: Iterable[str], preferred: str) -> str:
    used = set(parameters)
    if preferred not in used:
        return preferred
    index = 2
    while f"{preferred}{index}" in used:
        index += 1
    return f"{preferred}{index}"


def relation_with_variable(
    expr: Node,
    predicates: set[str],
    variable: str,
    *,
    negative: bool = False,
) -> list[str] | None:
    for item in conjunction(expr):
        candidate = item
        if negative:
            if not (
                isinstance(item, list)
                and len(item) == 2
                and item[0] == "not"
                and isinstance(item[1], list)
            ):
                continue
            candidate = item[1]
        if (
            isinstance(candidate, list)
            and len(candidate) == 3
            and candidate[0] in predicates
            and candidate[1] == variable
        ):
            return [str(token) for token in candidate]
    return None


@dataclass
class Action:
    name: str
    params: list[str]
    pre: Node
    eff: Node
    comment: str

    @classmethod
    def parse(cls, block: str, comment: str) -> "Action":
        root = parse_sexp(block)
        if not isinstance(root, list) or len(root) < 2 or root[0] != ":action":
            raise ValueError("invalid action block")
        fields: dict[str, Node] = {}
        index = 2
        while index + 1 < len(root):
            fields[str(root[index])] = root[index + 1]
            index += 2
        params = fields.get(":parameters")
        if not isinstance(params, list):
            raise ValueError(f"{root[1]} has no parameter list")
        return cls(
            name=str(root[1]),
            params=[str(item) for item in params],
            pre=fields[":precondition"],
            eff=fields[":effect"],
            comment=comment,
        )

    def render(self) -> str:
        comment = self.comment.strip()
        if not comment.startswith(";"):
            comment = f"; {comment}"
        return (
            f"  {comment}\n"
            f"  (:action {self.name}\n"
            f"    :parameters {sexp(self.params)}\n"
            f"    :precondition {sexp(self.pre)}\n"
            f"    :effect {sexp(self.eff)}\n"
            f"  )"
        )


ActionTransform = Callable[[Action], Action | None]


@dataclass(frozen=True)
class ActionEdit:
    old_parameters: tuple[str, ...]
    new_name: str | None
    new_parameters: tuple[str, ...]


@dataclass(frozen=True)
class PlanExpansion:
    old_parameters: tuple[str, ...]
    steps: tuple[tuple[str, tuple[str, ...]], ...]


def canonical_action(action: Action) -> tuple[object, ...]:
    usage: dict[str, list[str]] = {parameter: [] for parameter in action.params}

    def collect_usage(node: Node, prefix: str) -> None:
        if isinstance(node, str) or not node:
            return
        head = str(node[0]).lower()
        if head in {"and", "or"}:
            for child in node[1:]:
                collect_usage(child, prefix)
            return
        if head == "not" and len(node) == 2:
            collect_usage(node[1], prefix + ":not")
            return
        for index, argument in enumerate(node[1:]):
            if isinstance(argument, str) and argument.startswith("?"):
                usage.setdefault(argument, []).append(f"{prefix}:{head}:{index}")

    collect_usage(action.pre, "pre")
    collect_usage(action.eff, "eff")
    ordered_variables = sorted(
        usage,
        key=lambda variable: (tuple(sorted(usage[variable])), variable),
    )
    variable_map = {
        variable: f"?v{index}" for index, variable in enumerate(ordered_variables)
    }

    def canonical(node: Node) -> object:
        if isinstance(node, str):
            return variable_map.get(node, node.lower())
        if node and node[0] == "and":
            children = [canonical(child) for child in node[1:]]
            return ("and", *sorted(children, key=repr))
        return tuple(canonical(child) for child in node)

    return (
        len(action.params),
        canonical(action.pre),
        canonical(action.eff),
    )


def _device_transition_name(action: Action, before: str, after: str) -> str | None:
    for literal in conjunction(action.eff):
        if not (
            isinstance(literal, list)
            and len(literal) == 2
            and literal[0] == after
            and isinstance(literal[1], str)
        ):
            continue
        target = literal[1]
        if not has_literal(action.eff, before, target, negative=True):
            continue
        target_kind = unary_type(action.pre, target)
        if target_kind:
            return f"turn_{'on' if after == 'is_on' else 'off'}_{target_kind}"
    return None


def _open_close_contrast_name(name: str) -> str | None:
    match = re.fullmatch(r"(open|close)_(.+?)(?:_when_(.+))?", name)
    if not match:
        return None
    verb, affected, guard = match.groups()
    if not guard:
        return f"{verb}_{affected}"

    access: list[str] = []
    if "unlocked" in guard:
        access.append("unlocked")
    if any(token in guard for token in ("clear_to_open", "clear_to_close", "unblocked")):
        access.append("unblocked")
    if "cardboard_box_right_of_cabinet" in guard:
        access.append("unblocked")
    if "drawer" in guard and "closed" in guard and not (
        guard in {"drawer_closed", "unlocked"}
        or "drawer_unlocked" in guard
    ):
        access.append("interlocked")

    access = list(dict.fromkeys(access))
    return "_".join([verb, *access, affected]) if access else f"{verb}_{affected}"


def _process_phase_name(name: str) -> str | None:
    if name.startswith("turn_off_microwave_after_"):
        return "turn_off_microwave_after_heating"
    if name.startswith("turn_off_faucet_after_"):
        return "turn_off_faucet_after_rinsing"

    if "_after_" not in name:
        return None
    prefix, detail = name.split("_after_", 1)
    detail = detail.split("_when_", 1)[0]

    if prefix.startswith("turn_off_"):
        process = "mixing" if "mixing" in detail else "filling"
        finite_source = re.search(r"_from_(water_jug|water_bottle)$", detail)
        if finite_source and finite_source.group(1) not in prefix:
            prefix += f"_from_{finite_source.group(1)}"
        return f"{prefix}_after_{process}"

    if prefix.startswith("release_"):
        cold_control = any(token in prefix for token in ("cold", "blue"))
        mixed_result = any(token in detail for token in ("warm", "temper", "mix"))
        process = "mixing" if cold_control and mixed_result else "filling"
        return f"{prefix}_after_{process}"
    return None


def minimal_contrast_action_name(action: Action) -> str:
    """Project a schema name onto the compact dataset-level naming grammar."""
    original = action.name.lower().replace("_into_", "_in_").replace("_onto_", "_on_")

    process_name = _process_phase_name(original)
    if process_name:
        return process_name

    open_close_name = _open_close_contrast_name(original)
    if open_close_name:
        return open_close_name

    if unary_var(action.eff, {"cycle_selected"}) is not None or original.startswith(
        "select_wash_cycle"
    ):
        return "select_wash_cycle"
    if unary_var(action.eff, {"started"}) is not None and "washing_machine" in original:
        return "start_washing_machine"

    if any(token in original for token in ("turn_on_", "to_turn_on_")):
        transitioned = _device_transition_name(action, "is_off", "is_on")
        if transitioned:
            return transitioned
    if "turn_off_" in original:
        transitioned = _device_transition_name(action, "is_on", "is_off")
        if transitioned:
            return transitioned

    name = original.split("_when_", 1)[0]
    # Keep one finite verb in compound place-result names.  "Opening" denotes
    # the coupled state transition rather than a second sequential command.
    name = name.replace("_and_open_", "_opening_")
    if name.startswith("push_start_button_on_washing_machine"):
        return "start_washing_machine"
    if name.startswith("select_wash_cycle"):
        return "select_wash_cycle"

    if name.startswith("place_"):
        name = re.sub(r"_and_clear_.*$", "", name)
        name = "_".join(token for token in name.split("_") if token != "clear")

    if name == "wash_clothes_in_washing_machine_until_washed":
        return "wash_clothes_in_washing_machine"
    return name


def minimal_contrast_name_transform(action: Action) -> Action:
    action.name = minimal_contrast_action_name(action)
    return action


def rewrite_actions(domain: str, transform: ActionTransform) -> tuple[str, dict[str, ActionEdit]]:
    matches = list(ACTION_RE.finditer(domain))
    replacements: list[tuple[int, int, str]] = []
    edits: dict[str, ActionEdit] = {}
    canonical_by_name: dict[str, tuple[object, ...]] = {}
    for match in matches:
        action_start = match.start()
        action_end = find_matching_paren(domain, action_start) + 1
        line_start = domain.rfind("\n", 0, action_start) + 1
        prefix = domain[:line_start]
        previous_end = len(prefix.rstrip(" \t\r\n"))
        previous_start = prefix.rfind("\n", 0, previous_end) + 1
        previous_line = prefix[previous_start:previous_end].strip()
        span_start = previous_start if previous_line.startswith(";") else line_start
        comment = previous_line if previous_line.startswith(";") else f"; {match.group(1).replace('_', ' ').capitalize()}."
        original_block = domain[action_start:action_end]
        action = Action.parse(original_block, comment)
        original_action = copy.deepcopy(action)
        old_name = action.name
        new_action = transform(action)
        if new_action is None:
            edits[old_name] = ActionEdit(tuple(original_action.params), None, ())
            replacement = ""
        else:
            signature = canonical_action(new_action)
            previous_signature = canonical_by_name.get(new_action.name)
            if previous_signature is not None:
                if previous_signature != signature:
                    raise ValueError(
                        f"canonical action collision for {new_action.name}: "
                        "schemas are not equivalent"
                    )
                edits[old_name] = ActionEdit(
                    tuple(original_action.params),
                    new_action.name,
                    tuple(new_action.params),
                )
                replacement = ""
                replacements.append((span_start, action_end, replacement))
                continue
            canonical_by_name[new_action.name] = signature
            if new_action == original_action:
                continue
            edits[old_name] = ActionEdit(
                tuple(original_action.params),
                new_action.name,
                tuple(new_action.params),
            )
            replacement = new_action.render()
        replacements.append((span_start, action_end, replacement))
    if not replacements:
        return domain, edits
    for start, end, replacement in reversed(replacements):
        domain = domain[:start] + replacement + domain[end:]
    domain = re.sub(r"\n{3,}", "\n\n", domain)
    return domain, edits


def predicate_block(domain: str) -> tuple[int, int, list[Node]]:
    match = re.search(r"\(\s*:predicates\b", domain, re.IGNORECASE)
    if not match:
        raise ValueError("domain has no predicates block")
    end = find_matching_paren(domain, match.start()) + 1
    root = parse_sexp(domain[match.start():end])
    if not isinstance(root, list) or root[0] != ":predicates":
        raise ValueError("invalid predicates block")
    return match.start(), end, list(root[1:])


def rewrite_predicates(
    domain: str,
    *,
    ensure: Iterable[Node] = (),
    remove_names: set[str] | None = None,
) -> str:
    start, end, predicates = predicate_block(domain)
    original_predicates = list(predicates)
    remove_names = remove_names or set()
    predicates = [
        pred for pred in predicates
        if not (isinstance(pred, list) and pred and pred[0] in remove_names)
    ]
    for pred in ensure:
        if pred not in predicates:
            predicates.append(pred)
    if predicates == original_predicates:
        return domain
    rendered = "  (:predicates\n" + "\n".join(f"    {sexp(pred)}" for pred in predicates) + "\n  )"
    return domain[:start] + rendered + domain[end:]


def ensure_action_predicates_declared(domain: str) -> str:
    start, end, predicates = predicate_block(domain)
    declarations: dict[str, int] = {}
    unique: list[Node] = []
    changed = False
    for declaration in predicates:
        if not isinstance(declaration, list) or not declaration or not isinstance(declaration[0], str):
            unique.append(declaration)
            continue
        name = declaration[0]
        arity = len(declaration) - 1
        if name in declarations:
            if declarations[name] != arity:
                raise ValueError(f"predicate {name} has conflicting arities")
            changed = True
            continue
        declarations[name] = arity
        unique.append(declaration)

    used: dict[str, int] = {}

    def collect(node: Node) -> None:
        if isinstance(node, str) or not node:
            return
        head = node[0]
        if head in {"and", "or"}:
            for child in node[1:]:
                collect(child)
            return
        if head == "not":
            if len(node) == 2:
                collect(node[1])
            return
        if isinstance(head, str):
            arity = len(node) - 1
            if head in used and used[head] != arity:
                raise ValueError(f"predicate {head} is used with conflicting arities")
            used[head] = arity

    for match in ACTION_RE.finditer(domain):
        action_end = find_matching_paren(domain, match.start()) + 1
        action = Action.parse(domain[match.start():action_end], "; action")
        collect(action.pre)
        collect(action.eff)
    for name, arity in used.items():
        if name in declarations:
            if declarations[name] != arity:
                raise ValueError(
                    f"predicate {name} declared with arity {declarations[name]}, used with {arity}"
                )
            continue
        variables = [f"?x{index + 1}" for index in range(arity)]
        unique.append([name, *variables])
        declarations[name] = arity
        changed = True
    if not changed:
        return domain
    rendered = "  (:predicates\n" + "\n".join(f"    {sexp(pred)}" for pred in unique) + "\n  )"
    return domain[:start] + rendered + domain[end:]


def ensure_predicate_arities(domain: str, required: dict[str, int]) -> str:
    """Declare problem-only predicates without duplicating existing schemas."""
    start, end, predicates = predicate_block(domain)
    declarations = {
        declaration[0]: len(declaration) - 1
        for declaration in predicates
        if isinstance(declaration, list)
        and declaration
        and isinstance(declaration[0], str)
    }
    changed = False
    for name, arity in required.items():
        if name in declarations:
            if declarations[name] != arity:
                raise ValueError(
                    f"predicate {name} declared with arity {declarations[name]}, expected {arity}"
                )
            continue
        predicates.append([name, *(f"?x{index + 1}" for index in range(arity))])
        declarations[name] = arity
        changed = True
    if not changed:
        return domain
    rendered = "  (:predicates\n" + "\n".join(
        f"    {sexp(pred)}" for pred in predicates
    ) + "\n  )"
    return domain[:start] + rendered + domain[end:]


def remove_unused_named_predicates(
    domain: str,
    problem_text: str,
    names: set[str],
) -> str:
    """Remove selected declarations only when no action or problem uses them."""
    used: set[str] = set()

    def collect(node: Node) -> None:
        if isinstance(node, str) or not node:
            return
        head = node[0]
        if head in {"and", "or"}:
            for child in node[1:]:
                collect(child)
        elif head == "not" and len(node) == 2:
            collect(node[1])
        elif isinstance(head, str):
            used.add(head)

    for action in domain_actions(domain).values():
        collect(action.pre)
        collect(action.eff)
    problem = parse_problem(problem_text)
    for section in (":init", ":goal"):
        for fact in problem_facts(problem, section):
            collect(fact)
    removable = names - used
    return rewrite_predicates(domain, remove_names=removable) if removable else domain


def parse_problem(text: str) -> list[Node]:
    root = parse_sexp(text)
    if not isinstance(root, list) or not root or root[0] != "define":
        raise ValueError("invalid problem")
    return root


def problem_section(problem: list[Node], key: str) -> list[Node]:
    for section in problem:
        if isinstance(section, list) and section and section[0] == key:
            return section
    raise ValueError(f"problem has no {key} section")


def render_problem(problem: list[Node]) -> str:
    problem_decl = next(item for item in problem if isinstance(item, list) and item and item[0] == "problem")
    domain = problem_section(problem, ":domain")
    objects = problem_section(problem, ":objects")
    init = problem_section(problem, ":init")
    goal = problem_section(problem, ":goal")
    goal_expr = goal[1]
    lines = [
        f"(define (problem {problem_decl[1]})",
        f"  (:domain {domain[1]})",
        "  (:objects " + " ".join(str(item) for item in objects[1:]) + ")",
        "  (:init",
    ]
    lines.extend(f"    {sexp(item)}" for item in init[1:])
    lines.extend(["  )", "  (:goal"])
    if isinstance(goal_expr, list) and goal_expr and goal_expr[0] == "and":
        lines.append("    (and")
        lines.extend(f"      {sexp(item)}" for item in goal_expr[1:])
        lines.append("    )")
    else:
        lines.append(f"    {sexp(goal_expr)}")
    lines.extend(["  )", ")"])
    return "\n".join(lines) + "\n"


def problem_facts(problem: list[Node], section: str) -> list[Node]:
    node = problem_section(problem, section)
    if section == ":goal":
        return conjunction(node[1])
    return list(node[1:])


def set_problem_facts(problem: list[Node], section: str, facts: list[Node]) -> None:
    node = problem_section(problem, section)
    if section == ":goal":
        node[1] = make_and(facts)
    else:
        node[:] = [section, *facts]


def typed_objects(problem: list[Node], predicate: str) -> list[str]:
    return [
        str(fact[1]) for fact in problem_facts(problem, ":init")
        if isinstance(fact, list) and len(fact) == 2 and fact[0] == predicate
    ]


def add_object(problem: list[Node], name: str) -> None:
    objects = problem_section(problem, ":objects")
    if name not in objects[1:]:
        objects.append(name)


def remove_object(problem: list[Node], name: str) -> None:
    objects = problem_section(problem, ":objects")
    objects[:] = [objects[0], *(item for item in objects[1:] if item != name)]


def rewrite_plan(
    text: str,
    edits: dict[str, ActionEdit],
) -> str:
    if not edits:
        return text
    output: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("(") or stripped.startswith("(;"):
            output.append(line)
            continue
        tokens = stripped.strip("()").split()
        if not tokens:
            output.append(line)
            continue
        old_name = tokens[0]
        args = tokens[1:]
        edit = edits.get(old_name)
        if edit is None:
            output.append(line)
            continue
        if edit.new_name is None:
            continue
        # Some generated plans contain stale trailing scene arguments, while
        # others already omit a parameter that this cleanup projects away.
        # Ground every supplied source position and require only the parameters
        # that survive in the rewritten schema.
        grounding = dict(zip(edit.old_parameters, args[:len(edit.old_parameters)]))
        missing = [parameter for parameter in edit.new_parameters if parameter not in grounding]
        if missing:
            raise ValueError(
                f"cannot ground new parameters {missing} for plan action {old_name}"
            )
        name = edit.new_name
        args = [grounding[parameter] for parameter in edit.new_parameters]
        output.append("(" + " ".join([name, *args]) + ")")
    action_count = sum(1 for line in output if line.strip().startswith("(") and not line.strip().startswith("(;"))
    output = [line for line in output if not line.strip().startswith("; cost =")]
    output.append(f"; cost = {action_count} (unit cost)")
    return "\n".join(output) + "\n"


def expand_plan_actions(text: str, expansions: dict[str, PlanExpansion]) -> str:
    if not expansions:
        return text
    output: list[str] = []
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        tokens = stripped.strip("()").split() if stripped.startswith("(") else []
        expansion = expansions.get(tokens[0]) if tokens else None
        if expansion is None:
            if not stripped.startswith("; cost ="):
                output.append(line)
            continue
        arguments = tokens[1:]
        if len(arguments) != len(expansion.old_parameters):
            raise ValueError(
                f"plan action {tokens[0]} expects {len(expansion.old_parameters)} "
                f"source arguments, got {len(arguments)}"
            )
        grounding = dict(zip(expansion.old_parameters, arguments))
        for name, parameters in expansion.steps:
            missing = [parameter for parameter in parameters if parameter not in grounding]
            if missing:
                raise ValueError(
                    f"cannot ground expanded parameters {missing} for plan action {tokens[0]}"
                )
            output.append(
                "(" + " ".join([name, *(grounding[parameter] for parameter in parameters)]) + ")"
            )
        changed = True
    if not changed:
        return text
    count = sum(
        1 for line in output
        if line.strip().startswith("(") and not line.strip().startswith("(;" )
    )
    output.append(f"; cost = {count} (unit cost)")
    return "\n".join(output) + "\n"


def remove_plan_actions(text: str, names: set[str]) -> str:
    output: list[str] = []
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        tokens = stripped.strip("()").split() if stripped.startswith("(") else []
        if tokens and tokens[0] in names:
            changed = True
            continue
        if stripped.startswith("; cost ="):
            continue
        output.append(line)
    if not changed:
        return text
    count = sum(
        1 for line in output
        if line.strip().startswith("(") and not line.strip().startswith("(;" )
    )
    output.append(f"; cost = {count} (unit cost)")
    return "\n".join(output) + "\n"


def normalize_plan_schema_arity(
    domain: str,
    problem_text: str,
    plan_text: str,
) -> str:
    """Repair only unambiguous stale trailing plan arguments."""
    schemas = domain_actions(domain)
    problem = parse_problem(problem_text)
    typed: dict[str, list[str]] = {}
    for fact in problem_facts(problem, ":init"):
        if isinstance(fact, list) and len(fact) == 2:
            typed.setdefault(str(fact[0]), []).append(str(fact[1]))
    output: list[str] = []
    changed = False
    for line in plan_text.splitlines():
        stripped = line.split(";", 1)[0].strip()
        if not stripped.startswith("("):
            if not stripped.startswith("; cost ="):
                output.append(line)
            continue
        tokens = stripped.strip("()").split()
        schema = schemas.get(tokens[0]) if tokens else None
        if schema is None:
            output.append(line)
            continue
        args = tokens[1:]
        if len(args) > len(schema.params):
            args = args[:len(schema.params)]
            changed = True
        while len(args) < len(schema.params):
            parameter = schema.params[len(args)]
            kind = unary_type(schema.pre, parameter)
            candidates = [
                obj for obj in typed.get(kind or "", [])
                if obj not in args
            ]
            if len(candidates) != 1:
                break
            args.append(candidates[0])
            changed = True
        output.append("(" + " ".join([tokens[0], *args]) + ")")
    if not changed:
        return plan_text
    count = sum(line.strip().startswith("(") for line in output)
    output.append(f"; cost = {count} (unit cost)")
    return "\n".join(output) + "\n"


def replace_plan_action_with_grounding(
    text: str,
    classifiers: list[tuple[Callable[[str], bool], str, list[str]]],
) -> str:
    output: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("("):
            old = stripped.strip("()").split()[0]
            matched = next((entry for entry in classifiers if entry[0](old)), None)
            if matched:
                output.append("(" + " ".join([matched[1], *matched[2]]) + ")")
                continue
        if not stripped.startswith("; cost ="):
            output.append(line)
    count = sum(line.strip().startswith("(") for line in output)
    output.append(f"; cost = {count} (unit cost)")
    return "\n".join(output) + "\n"


ORIENTATION_TASKS = {36, 38, 39, 42, 43, 45, 80, 81, 95, 96, 97, 102, 103}
POSE_CLEANUP_TASKS = {95, 113, 228, 264, 265, 266, 267, 268, 274, 278, 291, 296}
POSE_PREDICATES = {"upright", "flat", "vertical", "sideways", "on_side", "upside_down", "inverted"}
OPPOSITE_POSES = {
    "upright": {"vertical", "sideways", "on_side", "upside_down", "inverted"},
    "flat": {"vertical", "upright", "sideways", "on_side"},
    "vertical": {"flat", "upright"},
}
ORIENTED_PLACE_RENAMES = {
    "place_box_on_table": "place_box_upright_on_table",
    "place_box_on_counter": "place_box_upright_on_counter",
    "place_bowl_on_table": "place_bowl_upright_on_table",
    "place_bowl_on_counter": "place_bowl_upright_on_counter",
    "place_cup_on_table": "place_cup_upright_on_table",
    "place_cup_on_counter": "place_cup_upright_on_counter",
    "place_mug_on_table": "place_mug_upright_on_table",
    "place_mug_on_counter": "place_mug_upright_on_counter",
    "place_plate_on_table": "place_plate_flat_on_table",
    "place_plate_on_counter": "place_plate_flat_on_counter",
    "place_cutting_board_on_table": "place_cutting_board_flat_on_table",
    "place_cutting_board_on_counter": "place_cutting_board_flat_on_counter",
}


def _unique_nodes(items: Iterable[Node]) -> list[Node]:
    output: list[Node] = []
    for item in items:
        if item not in output:
            output.append(item)
    return output


def node_variables(node: Node) -> set[str]:
    if isinstance(node, str):
        return {node} if node.startswith("?") else set()
    output: set[str] = set()
    for child in node:
        output.update(node_variables(child))
    return output


def replace_literal_predicate_heads(node: Node, aliases: dict[str, str]) -> Node:
    """Return a copy with only literal predicate heads canonicalized."""
    if isinstance(node, str):
        return node
    if not node:
        return []
    head = str(node[0])
    if head in {"and", "or"}:
        return [head, *(replace_literal_predicate_heads(child, aliases) for child in node[1:])]
    if head == "not" and len(node) == 2:
        return [head, replace_literal_predicate_heads(node[1], aliases)]
    return [aliases.get(head, head), *copy.deepcopy(node[1:])]


def ordered_used_parameters(action: Action, preferred: Iterable[str] = ()) -> list[str]:
    """Project parameters onto variables still used by the action contract."""
    used = node_variables(make_and([action.pre, action.eff]))
    output: list[str] = []
    for parameter in [*preferred, *action.params]:
        if parameter in used and parameter not in output:
            output.append(parameter)
    return output


def remove_literals_with_predicates(expr: Node, predicates: set[str]) -> Node:
    output: list[Node] = []
    for literal in conjunction(expr):
        candidate = literal
        if (
            isinstance(literal, list)
            and len(literal) == 2
            and literal[0] == "not"
            and isinstance(literal[1], list)
        ):
            candidate = literal[1]
        if isinstance(candidate, list) and candidate and candidate[0] in predicates:
            continue
        output.append(literal)
    return make_and(output)


def split_remove_lid_place_macro(action: Action) -> tuple[Action, Action] | None:
    """Split a legacy remove-lid action that also teleports the lid to a surface."""
    if not action.name.startswith("remove_lid_from_"):
        return None
    hand = unary_var(action.pre, {"hand"})
    if not hand or not has_literal(action.pre, "hand_free", hand):
        return None
    sources = [
        literal for literal in conjunction(action.pre)
        if (
            isinstance(literal, list)
            and len(literal) == 3
            and literal[0] == "on"
            and has_literal(
                action.eff, "on", str(literal[1]), str(literal[2]), negative=True
            )
            and has_literal(action.pre, "closed", str(literal[2]))
            and has_literal(action.eff, "open", str(literal[2]))
        )
    ]
    if len(sources) != 1:
        return None
    source = sources[0]
    lid, container = str(source[1]), str(source[2])
    targets = [
        literal for literal in conjunction(action.eff)
        if (
            isinstance(literal, list)
            and len(literal) == 3
            and literal[0] == "on"
            and literal[1] == lid
            and literal[2] != container
        )
    ]
    if len(targets) != 1:
        return None
    target = str(targets[0][2])
    lid_kind = unary_type(action.pre, lid)
    container_kind = unary_type(action.pre, container)
    target_kind = unary_type(action.pre, target)
    if not lid_kind or not container_kind or not target_kind:
        return None

    remove = Action(
        f"remove_lid_from_{container_kind}",
        [hand, lid, container],
        make_and([
            ["hand", hand], [lid_kind, lid], [container_kind, container],
            ["hand_free", hand], ["closed", container], ["on", lid, container],
        ]),
        make_and([
            ["not", ["hand_free", hand]], ["holding", hand, lid],
            ["not", ["closed", container]], ["open", container],
            ["not", ["on", lid, container]],
        ]),
        f"; Remove lid {lid} from {container_kind.replace('_', ' ')} "
        f"{container} with hand {hand}.",
    )
    place = Action(
        f"place_{lid_kind}_on_{target_kind}",
        [hand, lid, target],
        make_and([
            ["hand", hand], [lid_kind, lid], [target_kind, target],
            ["holding", hand, lid],
        ]),
        make_and([
            ["not", ["holding", hand, lid]], ["hand_free", hand],
            ["on", lid, target],
        ]),
        f"; Place {lid_kind.replace('_', ' ')} {lid} on "
        f"{target_kind.replace('_', ' ')} {target} with hand {hand}.",
    )
    return remove, place


def split_pick_place_macro(action: Action) -> tuple[Action, Action] | None:
    """Split one-step grasp-and-place macros with an unambiguous binding."""
    if not action.name.startswith("place_"):
        return None
    effects = conjunction(action.eff)
    added_holding = {
        (str(item[1]), str(item[2]))
        for item in effects
        if isinstance(item, list) and len(item) == 3 and item[0] == "holding"
    }
    deleted_holding = {
        (str(item[1][1]), str(item[1][2]))
        for item in effects
        if (
            isinstance(item, list)
            and len(item) == 2
            and item[0] == "not"
            and isinstance(item[1], list)
            and len(item[1]) == 3
            and item[1][0] == "holding"
        )
    }
    held_pairs = added_holding & deleted_holding
    if not held_pairs:
        # A second legacy macro form teleports an object directly from its
        # source relation to its target relation while the hand stays free.
        # Materialize the implicit grasp/release interface, then let the
        # normal splitter below construct the concrete pick and place schemas.
        hand = unary_var(action.pre, {"hand"})
        source_candidates = [
            literal for literal in conjunction(action.pre)
            if (
                isinstance(literal, list)
                and len(literal) == 3
                and literal[0] in {"on", "in"}
                and has_literal(
                    action.eff,
                    str(literal[0]),
                    str(literal[1]),
                    str(literal[2]),
                    negative=True,
                )
            )
        ]
        target_candidates = [
            literal for literal in effects
            if (
                isinstance(literal, list)
                and len(literal) == 3
                and literal[0] in {"on", "in"}
                and any(
                    source[1] == literal[1] and source[2] != literal[2]
                    for source in source_candidates
                )
            )
        ]
        moved_items = {
            str(source[1]) for source in source_candidates
            if any(target[1] == source[1] for target in target_candidates)
        }
        if (
            not hand
            or len(moved_items) != 1
            or not has_literal(action.pre, "hand_free", hand)
        ):
            return None
        moved_item = next(iter(moved_items))
        action = copy.deepcopy(action)
        action.eff = add_literal(action.eff, ["not", ["hand_free", hand]])
        action.eff = add_literal(action.eff, ["holding", hand, moved_item])
        action.eff = add_literal(action.eff, ["not", ["holding", hand, moved_item]])
        action.eff = add_literal(action.eff, ["hand_free", hand])
        effects = conjunction(action.eff)
        held_pairs = {(hand, moved_item)}
    if len(held_pairs) != 1:
        return None
    hand, item = next(iter(held_pairs))
    if not (
        has_literal(action.pre, "hand_free", hand)
        and has_literal(action.eff, "hand_free", hand, negative=True)
        and has_literal(action.eff, "hand_free", hand)
    ):
        return None

    source_relations = [
        literal for literal in conjunction(action.pre)
        if (
            isinstance(literal, list)
            and len(literal) == 3
            and literal[0] in {"on", "in"}
            and literal[1] == item
            and has_literal(action.eff, str(literal[0]), str(literal[1]), str(literal[2]), negative=True)
        )
    ]
    target_relations = [
        literal for literal in effects
        if (
            isinstance(literal, list)
            and len(literal) == 3
            and literal[0] in {"on", "in"}
            and literal[1] == item
        )
    ]
    if len(source_relations) != 1 or len(target_relations) != 1:
        return None
    source_relation = source_relations[0]
    target_relation = target_relations[0]
    source = str(source_relation[2])
    target = str(target_relation[2])

    positive_poses = [
        item_effect for item_effect in effects
        if (
            isinstance(item_effect, list)
            and len(item_effect) == 2
            and item_effect[0] in {"upright", "flat", "vertical"}
            and item_effect[1] == item
        )
    ]
    pose_effects = [
        item_effect for item_effect in effects
        if (
            isinstance(item_effect, list)
            and (
                (
                    len(item_effect) == 2
                    and item_effect[0] in POSE_PREDICATES
                    and item_effect[1] == item
                )
                or (
                    len(item_effect) == 2
                    and item_effect[0] == "not"
                    and isinstance(item_effect[1], list)
                    and len(item_effect[1]) == 2
                    and item_effect[1][0] in POSE_PREDICATES
                    and item_effect[1][1] == item
                )
            )
        )
    ]
    if len(positive_poses) > 1:
        return None
    pose = str(positive_poses[0][0]) if positive_poses else None

    item_kind = unary_type(action.pre, item)
    source_kind = unary_type(action.pre, source)
    target_kind = unary_type(action.pre, target)
    if not item_kind or not source_kind or not target_kind:
        return None
    pick_preconditions: list[Node] = [
        ["hand", hand], [item_kind, item], [source_kind, source],
        ["hand_free", hand], source_relation,
    ]
    place_preconditions: list[Node] = [
        ["hand", hand], [item_kind, item], [target_kind, target],
        ["holding", hand, item],
    ]
    pose_preconditions = [
        literal for literal in conjunction(action.pre)
        if (
            isinstance(literal, list)
            and (
                (
                    len(literal) == 2
                    and literal[0] in POSE_PREDICATES
                    and literal[1] == item
                )
                or (
                    len(literal) == 2
                    and literal[0] == "not"
                    and isinstance(literal[1], list)
                    and len(literal[1]) == 2
                    and literal[1][0] in POSE_PREDICATES
                    and literal[1][1] == item
                )
            )
        )
    ]
    place_preconditions.extend(pose_preconditions)

    core_preconditions = _unique_nodes([
        *pick_preconditions, *place_preconditions,
    ])
    for literal in conjunction(action.pre):
        if literal in core_preconditions:
            continue
        variables = node_variables(literal)
        # Unary category declarations are regenerated from the variables that
        # each split action actually uses.
        if (
            isinstance(literal, list)
            and len(literal) == 2
            and isinstance(literal[1], str)
            and unary_type(action.pre, str(literal[1])) == literal[0]
        ):
            continue
        if source in variables and target in variables and item not in variables:
            # This is collection context (for example, a tray happens to sit on
            # the destination appliance), not an executability condition of
            # either canonical manipulation interface.
            continue
        if target in variables and source not in variables:
            place_preconditions.append(literal)
        else:
            pick_preconditions.append(literal)

    core_effects: list[Node] = [
        ["not", ["hand_free", hand]], ["holding", hand, item],
        ["not", source_relation], ["not", ["holding", hand, item]],
        ["hand_free", hand], target_relation,
    ]
    allowed_effects = _unique_nodes([*core_effects, *pose_effects])
    extra_place_effects = [
        literal for literal in effects if literal not in allowed_effects
    ]

    source_label = source_kind
    if re.search(rf"_from_{re.escape(source_kind)}_top(?:_|$)", action.name):
        source_label += "_top"
    pick_name = f"pick_{item_kind}_from_{source_label}"
    target_preposition = "on" if target_relation[0] == "on" else "in"
    target_label = target_kind
    if re.search(
        rf"_(?:on|onto|in|into)_{re.escape(target_kind)}_top(?:_|$)",
        action.name,
    ):
        target_label += "_top"
    pose_modifier = f"_{pose}" if pose else ""
    place_name = f"place_{item_kind}{pose_modifier}_{target_preposition}_{target_label}"

    def parameters_for(preconditions: list[Node], effects: list[Node], base: list[str]) -> list[str]:
        used = node_variables(make_and([*preconditions, *effects]))
        ordered = list(base)
        ordered.extend(
            parameter for parameter in action.params
            if parameter in used and parameter not in ordered
        )
        return ordered

    pick_preconditions = _unique_nodes(pick_preconditions)
    place_preconditions = _unique_nodes(place_preconditions)
    pick_effects: list[Node] = [
        ["not", ["hand_free", hand]], ["holding", hand, item],
        ["not", source_relation],
    ]
    place_effects = _unique_nodes([
        ["not", ["holding", hand, item]], ["hand_free", hand],
        *pose_effects, *extra_place_effects, target_relation,
    ])
    pick = Action(
        pick_name,
        parameters_for(pick_preconditions, pick_effects, [hand, item, source]),
        make_and(pick_preconditions),
        make_and(pick_effects),
        f"; Pick {item_kind.replace('_', ' ')} {item} from "
        f"{source_label.replace('_', ' ')} {source} with hand {hand}.",
    )
    place = Action(
        place_name,
        parameters_for(place_preconditions, place_effects, [hand, item, target]),
        make_and(place_preconditions),
        make_and(place_effects),
        f"; Place {item_kind.replace('_', ' ')} {item} "
        f"{(pose + ' ') if pose else ''}{target_preposition} "
        f"{target_label.replace('_', ' ')} {target} with hand {hand}.",
    )
    return pick, place


def expand_pick_place_macros(
    domain: str,
) -> tuple[str, dict[str, PlanExpansion]]:
    matches = list(ACTION_RE.finditer(domain))
    existing: dict[str, Action] = {}
    for match in matches:
        end = find_matching_paren(domain, match.start()) + 1
        existing[match.group(1)] = Action.parse(domain[match.start():end], "; action")

    replacements: list[tuple[int, int, str]] = []
    expansions: dict[str, PlanExpansion] = {}
    for match in matches:
        action_start = match.start()
        action_end = find_matching_paren(domain, action_start) + 1
        line_start = domain.rfind("\n", 0, action_start) + 1
        prefix = domain[:line_start]
        previous_end = len(prefix.rstrip(" \t\r\n"))
        previous_start = prefix.rfind("\n", 0, previous_end) + 1
        previous_line = prefix[previous_start:previous_end].strip()
        span_start = previous_start if previous_line.startswith(";") else line_start
        action = Action.parse(domain[action_start:action_end], previous_line or "; action")
        split = split_remove_lid_place_macro(action) or split_pick_place_macro(action)
        if split is None:
            continue
        pick, place = split
        rendered: list[str] = []
        for generated in (pick, place):
            prior = existing.get(generated.name)
            if prior is not None and prior.name != action.name:
                if canonical_action(prior) != canonical_action(generated):
                    raise ValueError(
                        f"expanded action collision for {generated.name}: schemas are not equivalent"
                    )
                continue
            existing[generated.name] = generated
            rendered.append(generated.render())
        replacements.append((span_start, action_end, "\n\n".join(rendered)))
        expansions[action.name] = PlanExpansion(
            tuple(action.params),
            (
                (pick.name, tuple(pick.params)),
                (place.name, tuple(place.params)),
            ),
        )

    for start, end, replacement in reversed(replacements):
        domain = domain[:start] + replacement + domain[end:]
    if replacements:
        domain = re.sub(r"\n{3,}", "\n\n", domain)
    return domain, expansions


WASH_CYCLE_ACTION_RE = re.compile(
    r"(?:turn|rotate)(?:_wash_cycle)?_(dial|knob)_on_washing_machine$"
)


def split_washing_start_completion(
    domain: str,
) -> tuple[str, dict[str, PlanExpansion]]:
    """Split a generated start-button action that also claims washing is done."""
    matches = list(ACTION_RE.finditer(domain))
    replacements: list[tuple[int, int, str]] = []
    expansions: dict[str, PlanExpansion] = {}
    existing_names = {match.group(1) for match in matches}
    completion_name = "wash_clothes_in_washing_machine_until_washed"

    for match in matches:
        action_start = match.start()
        action_end = find_matching_paren(domain, action_start) + 1
        action = Action.parse(domain[action_start:action_end], "; Start washing machine.")
        if action.name != "push_start_button_on_washing_machine":
            continue
        washing_machine = unary_var(action.pre, {"washing_machine"})
        washed = [
            str(item[1])
            for item in conjunction(action.eff)
            if isinstance(item, list) and len(item) == 2 and item[0] == "washed"
        ]
        if washing_machine is None or len(set(washed)) != 1:
            continue
        clothes = washed[0]
        clothes_kind = unary_type(action.pre, clothes)
        if clothes_kind not in {"cloth", "clothes"}:
            raise ValueError(
                f"cannot infer washed clothes role in {action.name}: {clothes}"
            )
        if completion_name in existing_names:
            raise ValueError(
                f"cannot split {action.name}: {completion_name} already exists"
            )

        original_parameters = tuple(action.params)
        action.eff = remove_literal(action.eff, ["washed", clothes])
        completion = Action(
            completion_name,
            [clothes, washing_machine],
            make_and([
                [clothes_kind, clothes], ["washing_machine", washing_machine],
                ["started", washing_machine], ["in", clothes, washing_machine],
            ]),
            make_and([["washed", clothes]]),
            (
                f"; Wash {clothes_kind.replace('_', ' ')} {clothes} in washing "
                f"machine {washing_machine} until washed."
            ),
        )
        replacement = action.render() + "\n\n" + completion.render()
        replacements.append((action_start, action_end, replacement))
        expansions[action.name] = PlanExpansion(
            original_parameters,
            (
                (action.name, original_parameters),
                (completion.name, (clothes, washing_machine)),
            ),
        )
        existing_names.add(completion_name)

    for start, end, replacement in reversed(replacements):
        domain = domain[:start] + replacement + domain[end:]
    return domain, expansions


def _washing_guard_suffix(conditions: list[str]) -> str:
    return "_when_" + "_and_".join(conditions) if conditions else ""


def washing_machine_contract_transform(action: Action) -> Action:
    """Canonicalize cycle selection and start as separate process contracts."""
    cycle_match = WASH_CYCLE_ACTION_RE.fullmatch(action.name)
    is_start = action.name == "push_start_button_on_washing_machine"
    if cycle_match is None and not is_start:
        return action

    hand = unary_var(action.pre, {"hand"})
    washing_machine = unary_var(action.pre, {"washing_machine"})
    if hand is None or washing_machine is None:
        return action

    conditions: list[str] = []
    preconditions: list[Node] = [["hand", hand]]
    preferred = [hand]

    if cycle_match is not None:
        control_kind = cycle_match.group(1)
        control = unary_var(action.pre, {control_kind})
        if control is None:
            return action
        action.name = f"select_wash_cycle_with_{control_kind}"
        preconditions.extend([
            [control_kind, control], ["washing_machine", washing_machine],
            ["hand_free", hand],
        ])
        preferred.extend([control, washing_machine])
        action.eff = make_and([["cycle_selected", washing_machine]])
        action.comment = (
            f"; Select a wash cycle on washing machine {washing_machine} with "
            f"{control_kind} {control} using hand {hand}."
        )
    else:
        button = unary_var(action.pre, {"start_button"})
        if button is None:
            return action
        action.name = "push_start_button_on_washing_machine"
        preconditions.extend([
            ["start_button", button], ["washing_machine", washing_machine],
            ["hand_free", hand],
        ])
        preferred.extend([button, washing_machine])
        if has_literal(action.pre, "cycle_selected", washing_machine):
            preconditions.append(["cycle_selected", washing_machine])
            conditions.append("cycle_selected")
        action.eff = make_and([["started", washing_machine]])
        action.comment = (
            f"; Push start button {button} on washing machine {washing_machine} "
            f"with hand {hand}."
        )

    if has_literal(action.pre, "closed", washing_machine):
        preconditions.append(["closed", washing_machine])
        conditions.append("washing_machine_closed")

    door = unary_var(action.pre, {"washing_machine_door"})
    if door and has_literal(action.pre, "closed", door):
        preconditions.extend([["washing_machine_door", door], ["closed", door]])
        preferred.append(door)
        conditions.append("washing_machine_door_closed")

    drawer = unary_var(action.pre, {"detergent_drawer"})
    if drawer and has_literal(action.pre, "closed", drawer):
        preconditions.extend([["detergent_drawer", drawer], ["closed", drawer]])
        preferred.append(drawer)
        conditions.append("detergent_drawer_closed")

    off_predicate = next(
        (
            predicate for predicate in ("is_off", "off")
            if has_literal(action.pre, predicate, washing_machine)
        ),
        None,
    )
    if off_predicate:
        preconditions.append([off_predicate, washing_machine])
        conditions.append(f"washing_machine_{off_predicate}")

    action.name += _washing_guard_suffix(conditions)
    action.pre = make_and(_unique_nodes(preconditions))
    action.params = ordered_used_parameters(action, preferred)
    return action


def placed_object_var(action: Action) -> str | None:
    held_pairs = {
        (str(item[1]), str(item[2]))
        for item in conjunction(action.pre)
        if isinstance(item, list) and len(item) == 3 and item[0] == "holding"
    }
    released_pairs = {
        (str(item[1][1]), str(item[1][2]))
        for item in conjunction(action.eff)
        if (
            isinstance(item, list)
            and len(item) == 2
            and item[0] == "not"
            and isinstance(item[1], list)
            and len(item[1]) == 3
            and item[1][0] == "holding"
        )
    }
    candidates = held_pairs & released_pairs
    if len(candidates) == 1:
        return next(iter(candidates))[1]
    return None


def _pose_literal_for_object(literal: Node, object_var: str) -> tuple[bool, str] | None:
    if (
        isinstance(literal, list)
        and len(literal) == 2
        and literal[0] in POSE_PREDICATES
        and literal[1] == object_var
    ):
        return False, str(literal[0])
    if (
        isinstance(literal, list)
        and len(literal) == 2
        and literal[0] == "not"
        and isinstance(literal[1], list)
        and len(literal[1]) == 2
        and literal[1][0] in POSE_PREDICATES
        and literal[1][1] == object_var
    ):
        return True, str(literal[1][0])
    return None


def preserve_pick_pose(
    action: Action,
    tid: int,
    pending_pose_effects: dict[str, list[tuple[bool, str]]],
) -> Action:
    """Keep pickup pose-neutral and defer an explicitly named reorientation."""
    if tid not in POSE_CLEANUP_TASKS or not action.name.startswith("pick_"):
        return action
    held_objects = {
        str(item[2])
        for item in conjunction(action.eff)
        if isinstance(item, list) and len(item) == 3 and item[0] == "holding"
    }
    if len(held_objects) != 1:
        return action
    object_var = next(iter(held_objects))
    pose_effects = [
        parsed
        for literal in conjunction(action.eff)
        if (parsed := _pose_literal_for_object(literal, object_var)) is not None
    ]
    if not pose_effects:
        return action

    for negative, predicate in pose_effects:
        literal: Node = [predicate, object_var]
        if negative:
            literal = ["not", literal]
        action.eff = remove_literal(action.eff, literal)

    named_poses = [
        pose for pose in ("upright", "flat", "vertical")
        if re.search(rf"(?:^|_){pose}(?:_|$)", action.name)
    ]
    positive_named = [
        (negative, predicate)
        for negative, predicate in pose_effects
        if not negative and predicate in named_poses
    ]
    object_kind = unary_type(action.pre, object_var)
    if positive_named and object_kind:
        pending_pose_effects[object_kind] = positive_named
        for pose in named_poses:
            action.name = re.sub(rf"_{pose}(?=_|$)", "", action.name)
        action.comment = (
            f"; Pick {object_kind.replace('_', ' ')} {object_var} from its "
            "source without changing its pose."
        )
    return action


def canonical_oriented_place_name(name: str, pose: str) -> str:
    """Put a pose modifier immediately before the first spatial preposition."""
    for candidate in ("upright", "flat", "vertical"):
        name = re.sub(rf"_{candidate}(?=_|$)", "", name)
    relation = re.search(r"_(?:on|in|into|onto|under)_", name)
    if relation is None:
        return f"{name}_{pose}"
    return name[:relation.start()] + f"_{pose}" + name[relation.start():]


def orientation_transform(
    action: Action,
    tid: int,
    pending_pose_effects: dict[str, list[tuple[bool, str]]],
) -> Action | None:
    turn_match = re.fullmatch(r"turn_(box|bowl|cup|mug)_upright", action.name)
    if turn_match:
        object_kind = turn_match.group(1)
        object_var = unary_var(action.pre, {object_kind})
        if not object_var:
            raise ValueError(f"cannot infer turned object for {action.name}")
        effects: list[tuple[bool, str]] = []
        for item in conjunction(action.eff):
            if (
                isinstance(item, list)
                and len(item) == 2
                and item[0] in POSE_PREDICATES
                and item[1] == object_var
            ):
                effects.append((False, str(item[0])))
            elif (
                isinstance(item, list)
                and len(item) == 2
                and item[0] == "not"
                and isinstance(item[1], list)
                and len(item[1]) == 2
                and item[1][0] in POSE_PREDICATES
                and item[1][1] == object_var
            ):
                effects.append((True, str(item[1][0])))
        if (False, "upright") not in effects:
            effects.append((False, "upright"))
        pending_pose_effects[object_kind] = effects
        return None
    if not action.name.startswith("place_"):
        return action
    effect_items = conjunction(action.eff)
    object_var = placed_object_var(action)
    pose = None
    pose_var = None
    for item in effect_items:
        if (
            isinstance(item, list)
            and len(item) == 2
            and item[0] in {"upright", "flat", "vertical"}
            and item[1] == object_var
        ):
            pose, pose_var = str(item[0]), str(item[1])
            break
    object_kind = unary_type(action.pre, object_var) if object_var else None
    carried_effects = (
        pending_pose_effects.pop(object_kind, None)
        if object_kind
        else None
    )
    should_orient = tid in ORIENTATION_TASKS and action.name in ORIENTED_PLACE_RENAMES
    should_expose_existing = pose is not None
    # Guards live after ``_when_``.  A target guard such as
    # ``_when_plate_flat`` is not a result pose for the placed object.
    result_name = action.name.split("_when_", 1)[0]
    name_pose_match = re.search(
        r"(?:^|_)(upright|flat|vertical)(?:_|$)", result_name
    )
    named_pose = name_pose_match.group(1) if name_pose_match else None
    if not should_orient and not should_expose_existing and not carried_effects and not named_pose:
        return action
    if pose is None and carried_effects:
        positive_pose = next(
            (predicate for negative, predicate in carried_effects if not negative),
            None,
        )
        if positive_pose:
            pose, pose_var = positive_pose, object_var
        for negative, predicate in carried_effects:
            literal: Node = [predicate, object_var]
            if negative:
                literal = ["not", literal]
            action.eff = add_literal(action.eff, literal)
    if pose is None:
        pose = named_pose or "upright"
        pose_var = object_var
        if not pose_var:
            raise ValueError(f"cannot infer oriented object for {action.name}")
        action.eff = add_literal(action.eff, [pose, pose_var])
    action.pre = remove_literal(action.pre, [pose, pose_var])
    for opposite in sorted(OPPOSITE_POSES.get(pose, set())):
        if has_literal(action.pre, opposite, pose_var):
            action.eff = add_literal(action.eff, ["not", [opposite, pose_var]])
        action.pre = remove_literal(action.pre, [pose, pose_var])
    action.name = canonical_oriented_place_name(action.name, pose)
    action.comment = f"; Place {unary_type(action.pre, pose_var) or 'item'} {pose} on its target with hand ?h."
    return action


def normalize_pick_up(action: Action, *, preserve_container_state: bool = False) -> Action:
    if action.name.startswith("pick_up_"):
        action.name = "pick_" + action.name.removeprefix("pick_up_")
    if not action.name.startswith("pick_"):
        return action
    hand = unary_var(action.pre, {"hand"})
    if not hand:
        return action
    held = next(
        (
            str(item[2]) for item in conjunction(action.eff)
            if isinstance(item, list) and len(item) == 3 and item[:2] == ["holding", hand]
        ),
        None,
    )
    if not held:
        return action

    # Preserve task-specific conditions and effects.  Canonicalizing a pick is
    # an interface repair, not permission to rebuild an otherwise valid schema.
    action.pre = add_literal(action.pre, ["hand_free", hand])
    action.eff = add_literal(action.eff, ["not", ["hand_free", hand]])
    relation = relation_with_variable(action.pre, {"on", "in"}, held)
    if relation:
        action.eff = add_literal(action.eff, ["not", relation])
    return action


def normalize_place_hand_interface(action: Action) -> Action:
    """A placement consumes holding; it cannot require the same hand free."""
    if not action.name.startswith("place_"):
        return action
    held = [
        (str(item[1]), str(item[2]))
        for item in conjunction(action.pre)
        if isinstance(item, list) and len(item) == 3 and item[0] == "holding"
    ]
    if len(held) != 1:
        return action
    hand, item = held[0]
    action.pre = remove_literal(action.pre, ["hand_free", hand])
    action.eff = add_literal(action.eff, ["not", ["holding", hand, item]])
    action.eff = add_literal(action.eff, ["hand_free", hand])
    action.eff = remove_literal(action.eff, ["not", ["hand_free", hand]])
    return action


def restore_grounded_pick_source_types(
    domain: str,
    problem_text: str,
    plan_text: str,
) -> str:
    """Recover an omitted pick-source type from unambiguous plan grounding."""
    schemas = domain_actions(domain)
    init = problem_facts(parse_problem(problem_text), ":init")
    inferred: dict[str, dict[str, set[str]]] = {}
    source_kinds = SURFACE_SOURCE_TYPES | CONTAINER_SOURCE_TYPES | {
        "box", "cardboard_box", "red_box", "red_cardboard_box",
    }
    for line in plan_text.splitlines():
        tokens = line.split(";", 1)[0].strip().strip("()").split()
        if not tokens or not tokens[0].startswith("pick_"):
            continue
        schema = schemas.get(tokens[0])
        if schema is None or len(tokens[1:]) != len(schema.params):
            continue
        holding = _positive_holding_effect(schema)
        if holding is None:
            continue
        relation = _direct_source_relation(schema, holding[1])
        if relation is None or unary_type(schema.pre, relation[2]) is not None:
            continue
        grounding = dict(zip(schema.params, tokens[1:]))
        grounded_source = grounding.get(relation[2])
        if grounded_source is None:
            continue
        kinds = {
            str(fact[0]) for fact in init
            if isinstance(fact, list) and len(fact) == 2
            and fact[1] == grounded_source and fact[0] in source_kinds
        }
        if len(kinds) == 1:
            inferred.setdefault(schema.name, {}).setdefault(relation[2], set()).update(kinds)
    additions = {
        name: [(next(iter(kinds)), variable) for variable, kinds in variables.items() if len(kinds) == 1]
        for name, variables in inferred.items()
    }
    if not any(additions.values()):
        return domain

    def transform(action: Action) -> Action:
        for kind, variable in additions.get(action.name, []):
            action.pre = add_literal(action.pre, [kind, variable])
        return action

    return rewrite_actions(domain, transform)[0]


SPATIAL_PREDICATES = {"in", "on", "under", "beside", "left_of", "right_of", "in_front_of"}
SURFACE_SOURCE_TYPES = {
    "table", "counter", "desk", "floor", "rack", "dish_rack", "book_rack",
    "bookstand", "stand", "bookend", "tray", "drip_tray", "plate",
    "cutting_board", "paper_towel", "lid", "power_base", "kettle_base",
    "turntable", "microwave_turntable", "washing_machine_top", "cabinet_top",
    "box_top",
}
CLEAR_TOP_SURFACE_TYPES = {
    "table", "counter", "desk", "floor", "washing_machine_top",
    "cabinet_top", "box_top",
}
CONTAINER_SOURCE_TYPES = {
    "drawer", "box", "cardboard_box", "red_box", "red_cardboard_box",
    "cabinet", "compartment", "cabinet_compartment", "bottom_compartment",
    "bottom_cabinet_compartment", "laundry_basket", "basket", "microwave",
    "washing_machine", "kettle", "pot", "sink", "cup", "mug", "paper_cup",
}


def _positive_holding_effect(action: Action) -> tuple[str, str] | None:
    pairs = {
        (str(item[1]), str(item[2]))
        for item in conjunction(action.eff)
        if isinstance(item, list) and len(item) == 3 and item[0] == "holding"
    }
    return next(iter(pairs)) if len(pairs) == 1 else None


def _direct_source_relation(action: Action, moving: str) -> list[str] | None:
    candidates = [
        [str(item[0]), str(item[1]), str(item[2])]
        for item in conjunction(action.pre)
        if (
            isinstance(item, list)
            and len(item) == 3
            and item[0] in {"in", "on"}
            and item[1] == moving
        )
    ]
    deleted = [
        relation
        for relation in candidates
        if ["not", relation] in conjunction(action.eff)
    ]
    pool = deleted or candidates
    if len(pool) == 1:
        return pool[0]
    if not pool:
        return None
    source_phrase = action.name.split("_from_", 1)[1] if "_from_" in action.name else ""
    named = [
        relation
        for relation in pool
        if (unary_type(action.pre, relation[2]) or "") in source_phrase
    ]
    return named[0] if len(named) == 1 else None


def _canonical_pick_relation(
    moving_kind: str,
    source_kind: str,
    relation: str,
    *,
    top_surface: bool = False,
) -> str:
    if top_surface:
        return "on"
    if moving_kind == source_kind and source_kind in {"block", "book", "bowl"}:
        return "on"
    if source_kind in SURFACE_SOURCE_TYPES:
        return "on"
    if source_kind == "laundry_basket":
        return "in"
    if source_kind == "microwave" and moving_kind == "paper_cup":
        return "in"
    if source_kind in {"tray", "detergent_tray"} and moving_kind == "cap":
        return "on"
    return relation


def _replace_binary_literal(
    expr: Node,
    old: list[str],
    new: list[str],
) -> Node:
    return make_and(
        ["not", new]
        if item == ["not", old]
        else new if item == old else item
        for item in conjunction(expr)
    )


def _source_label(action_name: str, source_kind: str, relation: str) -> str:
    source_phrase = action_name.split("_from_", 1)[1] if "_from_" in action_name else ""
    named_top = "top" in source_phrase.split("_when_", 1)[0].split("_")
    implicit_top = relation == "on" and source_kind in CONTAINER_SOURCE_TYPES
    if (named_top or implicit_top) and not source_kind.endswith("_top"):
        return f"{source_kind}_top"
    return source_kind


def canonical_pick_contract(action: Action) -> Action:
    """Project an ordinary pick onto its direct grasp/source contract."""
    if not action.name.startswith("pick_"):
        return action
    if action.name == "pick_bowl_from_sink" and unary_var(action.pre, {"faucet"}):
        return action
    holding = _positive_holding_effect(action)
    if holding is None:
        return action
    hand, moving = holding
    moving_kind = unary_type(action.pre, moving)
    source_relation = _direct_source_relation(action, moving)
    if moving_kind is None or source_relation is None:
        return action
    source = source_relation[2]
    source_kind = unary_type(action.pre, source)
    if source_kind is None:
        return action

    canonical_relation = _canonical_pick_relation(
        moving_kind,
        source_kind,
        source_relation[0],
        top_surface="top" in action.name.split("_from_", 1)[-1].split("_when_", 1)[0].split("_"),
    )
    relation = [canonical_relation, moving, source]
    if relation != source_relation:
        action.pre = _replace_binary_literal(action.pre, source_relation, relation)
        action.eff = _replace_binary_literal(action.eff, source_relation, relation)

    preconditions: list[Node] = [
        ["hand", hand], [moving_kind, moving], [source_kind, source],
        ["hand_free", hand], relation,
    ]
    effects: list[Node] = [
        ["not", ["hand_free", hand]], ["holding", hand, moving],
        ["not", relation],
    ]
    source_label = _source_label(action.name, source_kind, canonical_relation)
    conditions: list[str] = []

    same_family_stack = (
        canonical_relation == "on"
        and moving_kind == source_kind
        and moving_kind in {"block", "book", "bowl"}
    )
    if same_family_stack:
        preconditions.append(["clear", moving])
        effects.extend([["not", ["clear", moving]], ["clear", source]])
    elif has_literal(action.pre, "clear", moving):
        preconditions.append(["clear", moving])
        effects.append(["not", ["clear", moving]])
        conditions.append("clear")
        if has_literal(action.eff, "clear", source):
            effects.append(["clear", source])

    for predicate, label in (
        ("open", f"{source_kind}_open"),
        ("unlocked", f"{source_kind}_unlocked"),
        ("is_off", f"{source_kind}_off"),
    ):
        if has_literal(action.pre, predicate, source):
            preconditions.append([predicate, source])
            conditions.append(label)

    # A turntable is a distinct support inside the microwave.  Its containment
    # and the microwave access state are part of that concrete source role.
    if source_kind in {"turntable", "microwave_turntable"}:
        microwave_relations = [
            item
            for item in conjunction(action.pre)
            if (
                isinstance(item, list)
                and len(item) == 3
                and item[0] == "in"
                and item[1] == source
                and unary_type(action.pre, str(item[2])) == "microwave"
            )
        ]
        if len(microwave_relations) == 1:
            microwave_relation = microwave_relations[0]
            microwave = str(microwave_relation[2])
            preconditions.extend([["microwave", microwave], microwave_relation])
            source_label = f"{source_kind}_in_microwave"
            for predicate, label in (
                ("open", "microwave_open"),
                ("is_off", "microwave_off"),
            ):
                if has_literal(action.pre, predicate, microwave):
                    preconditions.append([predicate, microwave])
                    conditions.append(label)

    action.name = f"pick_{moving_kind}_from_{source_label}"
    if conditions:
        action.name += "_when_" + "_and_".join(dict.fromkeys(conditions))
    action.pre = make_and(_unique_nodes(preconditions))
    action.eff = make_and(_unique_nodes(effects))
    action.params = ordered_used_parameters(
        action, [hand, moving, source]
    )
    action.comment = (
        f"; Pick {moving_kind.replace('_', ' ')} {moving} from "
        f"{source_label.replace('_', ' ')} {source} with hand {hand}."
    )
    return action


def _direct_place_relations(action: Action, moving: str) -> list[list[str]]:
    return [
        [str(item[0]), str(item[1]), str(item[2])]
        for item in conjunction(action.eff)
        if (
            isinstance(item, list)
            and len(item) == 3
            and item[0] in SPATIAL_PREDICATES
            and item[1] == moving
        )
    ]


def _target_label(action_name: str, target_kind: str, relation: str) -> str:
    target_phrase = re.split(r"_(?:in|on|under)_", action_name, maxsplit=1)
    suffix = target_phrase[1] if len(target_phrase) == 2 else ""
    named_top = "top" in suffix.split("_when_", 1)[0].split("_")
    implicit_top = relation == "on" and target_kind in CONTAINER_SOURCE_TYPES
    if target_kind == "counter_right_of_microwave":
        return "counter_right_of_microwave_surface"
    if (named_top or implicit_top) and not target_kind.endswith("_top"):
        return f"{target_kind}_top"
    return target_kind


def canonical_place_contract(action: Action) -> Action:
    """Project an ordinary place onto release, result pose, and target roles."""
    if not action.name.startswith("place_"):
        return action
    # A cover placement that also closes its target is a compound closure
    # transition, not an ordinary spatial release.  Detect the contract from
    # its effects so newly encountered container categories are preserved.
    closed_targets = {
        str(item[1])
        for item in conjunction(action.eff)
        if (
            isinstance(item, list)
            and len(item) == 2
            and item[0] == "closed"
            and has_literal(action.eff, "open", str(item[1]), negative=True)
        )
    }
    if closed_targets:
        return action
    opened_targets = {
        str(item[1])
        for item in conjunction(action.eff)
        if (
            isinstance(item, list)
            and len(item) == 2
            and item[0] == "open"
            and has_literal(action.eff, "closed", str(item[1]), negative=True)
        )
    }
    if len(opened_targets) == 1:
        opened = next(iter(opened_targets))
        opened_kind = unary_type(action.pre, opened)
        moving = placed_object_var(action)
        relations = _direct_place_relations(action, moving) if moving else []
        if opened_kind and moving and len(relations) == 1:
            target_kind = unary_type(action.pre, relations[0][2])
            moving_kind = unary_type(action.pre, moving)
            hand = unary_var(action.pre, {"hand"})
            if hand and moving_kind and target_kind:
                action.name = (
                    f"place_{moving_kind}_{relations[0][0]}_{target_kind}"
                    f"_and_open_{opened_kind}"
                )
                action.params = ordered_used_parameters(
                    action, [hand, moving, relations[0][2], opened]
                )
                return action
    moving = placed_object_var(action)
    if moving is None:
        return action
    held = [
        (str(item[1]), str(item[2]))
        for item in conjunction(action.pre)
        if isinstance(item, list) and len(item) == 3 and item[0] == "holding"
        and item[2] == moving
    ]
    if len(held) != 1:
        return action
    hand = held[0][0]
    moving_kind = unary_type(action.pre, moving)
    relations = _direct_place_relations(action, moving)
    if moving_kind is None or not relations:
        return action

    # Correct only relations whose target category has a stable physical
    # interpretation.  Relative predicates remain additional results.
    normalized_relations: list[list[str]] = []
    for relation in relations:
        target_kind = unary_type(action.pre, relation[2])
        replacement = relation
        if target_kind:
            canonical = _canonical_pick_relation(
                moving_kind,
                target_kind,
                relation[0],
                top_surface="top" in action.name.split("_when_", 1)[0].split("_"),
            ) if relation[0] in {"in", "on"} else relation[0]
            replacement = [canonical, relation[1], relation[2]]
        if replacement != relation:
            action.eff = _replace_binary_literal(action.eff, relation, replacement)
        normalized_relations.append(replacement)
    relations = _unique_nodes(normalized_relations)  # type: ignore[assignment]

    primary_candidates = [r for r in relations if r[0] in {"in", "on"}]
    primary = primary_candidates[0] if primary_candidates else relations[0]
    primary_target = primary[2]
    primary_kind = unary_type(action.pre, primary_target)
    if primary_kind is None:
        return action

    preconditions: list[Node] = [
        ["hand", hand], [moving_kind, moving], [primary_kind, primary_target],
        ["holding", hand, moving],
    ]
    effects: list[Node] = [
        ["not", ["holding", hand, moving]], ["hand_free", hand],
    ]
    conditions: list[str] = []
    result_modifiers: list[str] = []

    for relation in relations:
        target = relation[2]
        target_kind = unary_type(action.pre, target)
        if target_kind and [target_kind, target] not in preconditions:
            preconditions.append([target_kind, target])
        effects.append(relation)

    clearance_results: list[tuple[str, str]] = []
    for predicate, label in (
        ("clear_to_open", "clear_to_open"),
        ("clear_to_close", "clear_to_close"),
        ("unblocked", "clear_to_open"),
    ):
        for literal in conjunction(action.eff):
            if not (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[0] == predicate
            ):
                continue
            drawer = str(literal[1])
            if unary_type(action.pre, drawer) != "drawer":
                continue
            directly_unblocked = any(
                isinstance(effect, list)
                and len(effect) == 2
                and effect[0] == "not"
                and isinstance(effect[1], list)
                and effect[1] == ["blocking", moving, drawer]
                for effect in conjunction(action.eff)
            )
            if directly_unblocked:
                preconditions.extend([["drawer", drawer], ["blocking", moving, drawer]])
                effects.extend([
                    ["not", ["blocking", moving, drawer]], [predicate, drawer]
                ])
                clearance_results.append((drawer, label))

    positive_poses = [
        str(item[0])
        for item in conjunction(action.eff)
        if (
            isinstance(item, list)
            and len(item) == 2
            and item[0] in POSE_PREDICATES
            and item[1] == moving
        )
    ]
    pose = positive_poses[0] if len(set(positive_poses)) == 1 else None
    if pose:
        result_modifiers.append(pose)
        effects.append([pose, moving])
        for opposite in sorted(OPPOSITE_POSES.get(pose, set())):
            effects.append(["not", [opposite, moving]])

    same_family_stack = (
        primary[0] == "on"
        and moving_kind == primary_kind
        and moving_kind in {"block", "book", "bowl"}
    )
    if same_family_stack:
        preconditions.append(["clear", primary_target])
        effects.extend([
            ["not", ["clear", primary_target]], ["clear", moving],
        ])
    else:
        if has_literal(action.pre, "clear", primary_target):
            preconditions.append(["clear", primary_target])
            conditions.append(f"{primary_kind}_clear")
        if has_literal(action.eff, "clear", moving):
            effects.append(["clear", moving])
            result_modifiers.append("clear")

    if primary[0] == "in" and has_literal(action.pre, "open", primary_target):
        preconditions.append(["open", primary_target])
        conditions.append(f"{primary_kind}_open")
    for target in {relation[2] for relation in relations}:
        target_kind = unary_type(action.pre, target)
        for predicate in POSE_PREDICATES:
            if has_literal(action.pre, predicate, target):
                preconditions.append([predicate, target])
                conditions.append(f"{target_kind or 'target'}_{predicate}")

    target_label = _target_label(action.name, primary_kind, primary[0])
    result_label = "".join(f"_{modifier}" for modifier in result_modifiers)
    action.name = f"place_{moving_kind}{result_label}_{primary[0]}_{target_label}"
    secondary_relations = [relation for relation in relations if relation != primary]
    for relation in secondary_relations:
        target_kind = unary_type(action.pre, relation[2]) or "target"
        action.name += f"_{relation[0]}_{target_kind}"
    for _drawer, label in clearance_results:
        action.name += f"_and_{label.replace('_to_', '_drawer_to_')}"
    if conditions:
        action.name += "_when_" + "_and_".join(dict.fromkeys(conditions))
    action.pre = make_and(_unique_nodes(preconditions))
    action.eff = make_and(_unique_nodes(effects))
    action.params = ordered_used_parameters(
        action, [hand, moving, primary_target]
    )
    action.comment = (
        f"; Place {moving_kind.replace('_', ' ')} {moving} "
        f"{primary[0]} {target_label.replace('_', ' ')} {primary_target} "
        f"with hand {hand}."
    )
    return action


def canonical_open_close_contract(action: Action) -> Action:
    """Remove scene payload from a door/lid transition and name local guards."""
    if not action.name.startswith(("open_", "close_")):
        return action
    if "drawer" in action.name or "microwave" in action.name:
        return action
    opening = action.name.startswith("open_")
    before, after = ("closed", "open") if opening else ("open", "closed")
    transitioned = {
        str(item[1])
        for item in conjunction(action.eff)
        if isinstance(item, list) and len(item) == 2 and item[0] == after
        and has_literal(action.eff, before, str(item[1]), negative=True)
    }
    if len(transitioned) != 1:
        return action
    target = next(iter(transitioned))
    target_kind = unary_type(action.pre, target)
    hand = unary_var(action.pre, {"hand", "robot_hand"})
    if target_kind is None or hand is None:
        return action
    preconditions: list[Node] = [
        ["hand", hand], [target_kind, target], ["hand_free", hand],
        [before, target],
    ]
    conditions: list[str] = []
    for predicate, label in (
        ("unlocked", "unlocked"),
        ("clear_to_open", "clear_to_open"),
        ("unblocked", "clear_to_open"),
        ("is_off", "off"),
    ):
        if opening and has_literal(action.pre, predicate, target):
            preconditions.append([predicate, target])
            conditions.append(label)

    # Preserve an explicitly modeled object-clearance guard, but expose it in
    # the name instead of silently overloading the simple transition.
    for literal in conjunction(action.pre):
        if not (
            isinstance(literal, list)
            and len(literal) == 3
            and literal[0] in {"left_of", "right_of", "in_front_of", "beside"}
            and target in literal[1:]
        ):
            continue
        other = str(literal[1] if literal[2] == target else literal[2])
        other_kind = unary_type(action.pre, other)
        if other_kind:
            preconditions.extend([[other_kind, other], literal])
            conditions.append(f"{other_kind}_{literal[0]}_{target_kind}")

    action.name = f"{'open' if opening else 'close'}_{target_kind}"
    if conditions:
        action.name += "_when_" + "_and_".join(dict.fromkeys(conditions))
    action.pre = make_and(_unique_nodes(preconditions))
    action.eff = make_and([["not", [before, target]], [after, target]])
    action.params = ordered_used_parameters(action, [hand, target])
    action.comment = (
        f"; {'Open' if opening else 'Close'} {target_kind.replace('_', ' ')} "
        f"{target} with hand {hand}."
    )
    return action


def microwave_open_transform(action: Action) -> Action:
    opens_microwave = (
        action.name == "reopen_microwave"
        or action.name.startswith("open_microwave")
    )
    if not opens_microwave:
        return action
    hand = unary_var(action.pre, {"hand"})
    microwave = unary_var(action.pre, {"microwave"})
    if not hand or not microwave:
        return action
    pre: list[Node] = [
        ["hand", hand], ["microwave", microwave], ["hand_free", hand], ["closed", microwave]
    ]
    if has_literal(action.pre, "is_off", microwave):
        pre.append(["is_off", microwave])
    action.name = (
        "open_microwave_when_off"
        if has_literal(action.pre, "is_off", microwave)
        else "open_microwave"
    )
    action.params = [hand, microwave]
    action.pre = make_and(pre)
    action.eff = make_and([["not", ["closed", microwave]], ["open", microwave]])
    action.comment = "; Open microwave ?m with hand ?h."
    return action


def microwave_door_transform(action: Action) -> Action:
    """Keep microwave door actions independent of earlier/later task phases."""
    if not (
        action.name == "reopen_microwave"
        or action.name.startswith("open_microwave")
        or action.name.startswith("close_microwave")
        or action.name == "close_empty_microwave"
    ):
        return action
    hand = unary_var(action.pre, {"hand", "robot_hand"})
    microwave = unary_var(action.pre, {"microwave"})
    if not hand or not microwave:
        return action
    opening = action.name == "reopen_microwave" or action.name.startswith("open_microwave")
    before, after = ("closed", "open") if opening else ("open", "closed")
    preconditions: list[Node] = [
        ["hand", hand], ["microwave", microwave], ["hand_free", hand],
        [before, microwave],
    ]
    if opening and has_literal(action.pre, "is_off", microwave):
        preconditions.append(["is_off", microwave])
    action.name = (
        "open_microwave_when_off"
        if opening and has_literal(action.pre, "is_off", microwave)
        else "open_microwave" if opening else "close_microwave"
    )
    action.params = [hand, microwave]
    action.pre = make_and(preconditions)
    action.eff = make_and([["not", [before, microwave]], [after, microwave]])
    action.comment = (
        f"; {'Open' if opening else 'Close'} microwave ?m with hand ?h."
    )
    return action


def _typed_variable(action: Action, variable: str) -> str | None:
    return unary_type(action.pre, variable)


def _microwave_heating_path(
    action: Action,
    affected: str,
    microwave: str,
    direct_microwave_relation: str | None = None,
) -> list[list[str]] | None:
    """Find the shortest represented support/containment path into a microwave."""
    relations = [
        [
            direct_microwave_relation
            if str(item[2]) == microwave and direct_microwave_relation
            else str(item[0]),
            str(item[1]),
            str(item[2]),
        ]
        for item in conjunction(action.pre)
        if (
            isinstance(item, list)
            and len(item) == 3
            and item[0] in {"in", "on"}
            and isinstance(item[1], str)
            and isinstance(item[2], str)
        )
    ]
    queue: deque[tuple[str, list[list[str]]]] = deque([(affected, [])])
    visited = {affected}
    while queue:
        current, path = queue.popleft()
        if current == microwave:
            return path
        for relation in relations:
            if relation[1] != current or relation[2] in visited:
                continue
            visited.add(relation[2])
            queue.append((relation[2], [*path, relation]))
    return None


def microwave_contract_transform(
    action: Action,
    direct_microwave_relation: str | None = None,
) -> Action:
    """Canonicalize microwave controls as process begin/end contracts."""
    microwave_control = bool(
        re.fullmatch(r"turn_on_microwave(?:_with_[a-z0-9_]+)?", action.name)
        or action.name.startswith("turn_off_microwave")
        or action.name == "press_lower_button"
    )
    if not microwave_control:
        return action

    hand = unary_var(action.pre, {"hand", "robot_hand"})
    microwave = unary_var(action.pre, {"microwave"})
    if not microwave:
        return action

    if not hand:
        heated = [
            str(item[1]) for item in conjunction(action.eff)
            if isinstance(item, list) and len(item) == 2 and item[0] == "heated"
        ]
        if action.name.startswith("turn_off_microwave") and len(set(heated)) == 1:
            affected = heated[0]
            affected_kind = _typed_variable(action, affected) or "object"
            action.name = f"finish_microwave_heating_{affected_kind}"
            action.params = ordered_used_parameters(action, [microwave, affected])
        return action

    starts_microwave = (
        action.name.startswith("turn_on_microwave")
        or (
            has_literal(action.pre, "is_off", microwave)
            and has_literal(action.eff, "is_on", microwave)
        )
    )
    if starts_microwave:
        action.name = "turn_on_microwave"
        action.params = [hand, microwave]
        action.pre = make_and([
            ["hand", hand], ["microwave", microwave], ["hand_free", hand],
            ["closed", microwave], ["is_off", microwave],
        ])
        action.eff = make_and([
            ["not", ["is_off", microwave]], ["is_on", microwave],
        ])
        action.comment = "; Turn on microwave ?m with hand ?h."
        return action

    heated = [
        str(item[1])
        for item in conjunction(action.eff)
        if isinstance(item, list) and len(item) == 2 and item[0] == "heated"
    ]
    if not heated:
        action.name = "turn_off_microwave"
        action.params = [hand, microwave]
        action.pre = make_and([
            ["hand", hand], ["microwave", microwave], ["hand_free", hand],
            ["closed", microwave], ["is_on", microwave],
        ])
        action.eff = make_and([
            ["not", ["is_on", microwave]], ["is_off", microwave],
        ])
        action.comment = "; Turn off microwave ?m with hand ?h."
        return action
    if len(set(heated)) != 1:
        raise ValueError(f"{action.name} heats multiple affected variables")

    affected = heated[0]
    affected_kind = _typed_variable(action, affected)
    if affected_kind is None:
        # A few generated rounds omitted the water type from the schema while
        # retaining it in the problem.  The task family and relation still make
        # the affected role explicit, so use a conservative semantic fallback.
        affected_kind = "water" if affected != microwave else "object"
    path = _microwave_heating_path(
        action, affected, microwave, direct_microwave_relation
    )
    if path is None:
        # Some augmented examples omit the turntable-in-microwave relation from
        # this action even though the scene has it.  Preserve the shortest
        # represented affected-to-carrier relation instead of unrelated payload.
        candidate_paths = [
            [str(item[0]), str(item[1]), str(item[2])]
            for item in conjunction(action.pre)
            if (
                isinstance(item, list)
                and len(item) == 3
                and item[0] in {"in", "on"}
                and item[1] == affected
            )
        ]
        path = candidate_paths[:1]
        if path and path[0][2] == microwave and direct_microwave_relation:
            path[0][0] = direct_microwave_relation
        elif path and direct_microwave_relation:
            carrier = path[-1][2]
            path.append([direct_microwave_relation, carrier, microwave])

    variables = {hand, microwave, affected}
    for relation in path:
        variables.update(relation[1:])

    types: list[Node] = [["hand", hand], ["microwave", microwave]]
    role_order = [affected]
    for relation in path:
        role_order.extend(relation[1:])
    role_order.extend(action.params)
    for variable in dict.fromkeys(role_order):
        if variable not in variables or variable in {hand, microwave}:
            continue
        kind = _typed_variable(action, variable)
        if kind:
            types.append([kind, variable])
    if affected not in {hand, microwave} and not any(
        isinstance(item, list) and len(item) == 2 and item[1] == affected
        for item in types
    ):
        types.append([affected_kind, affected])
    affected_type = next(
        (
            item for item in types
            if isinstance(item, list) and len(item) == 2 and item[1] == affected
        ),
        None,
    )
    if affected_type is not None:
        types = [item for item in types if item != affected_type]
        types.insert(2, affected_type)

    suffix = affected_kind
    microwave_guards: list[str] = []
    for relation, source, target in path:
        if source == microwave:
            continue
        if target == microwave:
            if relation == "on":
                suffix += "_on_microwave_top"
            elif relation == "in" and source != affected:
                source_kind = _typed_variable(action, source) or "carrier"
                microwave_guards.append(f"{source_kind}_in_microwave")
            continue
        target_kind = _typed_variable(action, target)
        if target_kind:
            suffix += f"_{relation}_{target_kind}"
    action.name = f"turn_off_microwave_after_heating_{suffix}"
    if microwave_guards:
        action.name += "_when_" + "_and_".join(dict.fromkeys(microwave_guards))
    action.pre = make_and([
        *types, ["hand_free", hand], ["closed", microwave], ["is_on", microwave],
        *path,
    ])
    action.eff = make_and([
        ["not", ["is_on", microwave]], ["is_off", microwave], ["heated", affected],
    ])
    action.params = ordered_used_parameters(action, [hand, microwave, affected])
    action.comment = (
        f"; Turn off microwave ?{microwave.lstrip('?')} after heating "
        f"{affected_kind.replace('_', ' ')} ?{affected.lstrip('?')} with hand "
        f"?{hand.lstrip('?')}."
    )
    return action


def canonical_turntable_pick(action: Action) -> Action:
    if not re.fullmatch(
        r"pick_(?:heated_)?paper_cup_from_microwave_turntable.*", action.name
    ):
        return action
    hand = unary_var(action.pre, {"hand"})
    cup = unary_var(action.pre, {"paper_cup"})
    turntable = unary_var(action.pre, {"microwave_turntable", "turntable"})
    if hand is None or cup is None or turntable is None:
        return action
    action.name = "pick_paper_cup_from_microwave_turntable"
    action.params = [hand, cup, turntable]
    action.pre = make_and([
        ["hand", hand], ["paper_cup", cup],
        ["microwave_turntable", turntable], ["hand_free", hand],
        ["on", cup, turntable],
    ])
    action.eff = make_and([
        ["not", ["hand_free", hand]], ["holding", hand, cup],
        ["not", ["on", cup, turntable]],
    ])
    return action


def infer_direct_microwave_relation(domain: str) -> str | None:
    """Infer whether a simplified microwave target denotes its top or cavity."""
    candidates: set[str] = set()
    for action in domain_actions(domain).values():
        if not action.name.startswith("place_"):
            continue
        moving = placed_object_var(action)
        if moving is None or unary_type(action.pre, moving) != "paper_cup":
            continue
        direct = [
            item
            for item in conjunction(action.eff)
            if (
                isinstance(item, list)
                and len(item) == 3
                and item[0] in {"in", "on"}
                and item[1] == moving
                and unary_type(action.pre, str(item[2])) == "microwave"
            )
        ]
        if not direct:
            continue
        semantic_name = action.name.split("_when_", 1)[0]
        if "microwave_top" in semantic_name:
            candidates.add("on")
        elif "microwave_turntable" in semantic_name or "_in_microwave" in semantic_name:
            candidates.add("in")
        else:
            target = str(direct[0][2])
            candidates.add("in" if has_literal(action.pre, "open", target) else str(direct[0][0]))
    return next(iter(candidates)) if len(candidates) == 1 else None


def ensure_microwave_heating_type_facts(
    domain: str,
    problem_text: str,
    plan_text: str,
) -> str:
    """Ground inferred affected-object types used by microwave end actions."""
    schemas = domain_actions(domain)
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    original_init = copy.deepcopy(init)
    for line in plan_text.splitlines():
        stripped = line.split(";", 1)[0].strip()
        if not stripped.startswith("("):
            continue
        tokens = stripped.strip("()").split()
        if not tokens or not tokens[0].startswith("turn_off_microwave_after_heating_"):
            continue
        schema = schemas.get(tokens[0])
        if schema is None or len(tokens[1:]) != len(schema.params):
            continue
        affected = {
            str(item[1])
            for item in conjunction(schema.eff)
            if isinstance(item, list) and len(item) == 2 and item[0] == "heated"
        }
        if len(affected) != 1:
            continue
        affected_var = next(iter(affected))
        affected_kind = _typed_variable(schema, affected_var)
        if affected_kind is None:
            continue
        grounding = dict(zip(schema.params, tokens[1:]))
        affected_object = grounding.get(affected_var)
        if affected_object is None:
            continue
        type_fact = [affected_kind, affected_object]
        if type_fact not in init:
            init.append(type_fact)
    if init == original_init:
        return problem_text
    set_problem_facts(problem, ":init", init)
    return render_problem(problem)


def ensure_water_button_type_facts(
    domain: str,
    problem_text: str,
    plan_text: str,
) -> str:
    """Add canonical role types introduced by water-button projection."""
    schemas = domain_actions(domain)
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    original_init = copy.deepcopy(init)
    for line in plan_text.splitlines():
        stripped = line.split(";", 1)[0].strip()
        if not stripped.startswith("("):
            continue
        tokens = stripped.strip("()").split()
        if not tokens or not (
            re.fullmatch(r"turn_(?:on|off)_(?:hot|cold)_water_button.*", tokens[0])
            or tokens[0] in {"lock_child_lock", "unlock_child_lock"}
        ):
            continue
        schema = schemas.get(tokens[0])
        if schema is None or len(tokens[1:]) != len(schema.params):
            continue
        grounding = dict(zip(schema.params, tokens[1:]))
        for literal in conjunction(schema.pre):
            if (
                isinstance(literal, list)
                and len(literal) == 3
                and literal[0] == "dispenses"
            ):
                fact = [
                    "dispenses",
                    grounding.get(str(literal[1]), literal[1]),
                    grounding.get(str(literal[2]), literal[2]),
                ]
                if fact not in init:
                    init.append(fact)
                continue
            if not (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[0] in {
                    "hand", "hot_water_button", "cold_water_button",
                    "child_lock", "water", "cup", "water_dispenser",
                }
            ):
                continue
            fact: Node = [literal[0], grounding.get(str(literal[1]), literal[1])]
            if fact not in init:
                init.append(fact)
    if init == original_init:
        return problem_text
    set_problem_facts(problem, ":init", init)
    return render_problem(problem)


def ensure_power_connection_type_facts(
    domain: str,
    problem_text: str,
    plan_text: str,
) -> str:
    names = {
        "insert_plug_in_wall_outlet", "plug_plug_in_wall_outlet",
        "plug_power_base_cord_in_wall_outlet", "insert_plug_in_outlet",
        "plug_power_base_cord_in_outlet", "turn_on_kettle_when_plug_inserted",
    }
    schemas = domain_actions(domain)
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    original_init = copy.deepcopy(init)
    for line in plan_text.splitlines():
        tokens = line.split(";", 1)[0].strip().strip("()").split()
        if not tokens or tokens[0] not in names:
            continue
        schema = schemas.get(tokens[0])
        if schema is None or len(tokens[1:]) != len(schema.params):
            continue
        grounding = dict(zip(schema.params, tokens[1:]))
        for literal in conjunction(schema.pre):
            if not (
                isinstance(literal, list) and len(literal) == 2
                and literal[0] in {
                    "hand", "plug", "power_base_cord", "outlet", "wall_outlet",
                }
            ):
                continue
            fact: Node = [literal[0], grounding.get(str(literal[1]), literal[1])]
            if fact not in init:
                init.append(fact)
    if init == original_init:
        return problem_text
    set_problem_facts(problem, ":init", init)
    return render_problem(problem)


def normalize_power_connection_problem_predicates(problem_text: str) -> str:
    """Mirror the canonical inserted relation for typed power connections."""
    problem = parse_problem(problem_text)
    object_types: dict[str, set[str]] = {}
    for fact in problem_facts(problem, ":init"):
        if isinstance(fact, list) and len(fact) == 2:
            object_types.setdefault(str(fact[1]), set()).add(str(fact[0]))

    source_types = {"plug", "power_base_cord"}
    target_types = {"outlet", "wall_outlet"}
    aliases = {"in", "plugged", "plugged_in", "plugged_into", "connected"}
    changed = False
    for section in (":init", ":goal"):
        rewritten: list[Node] = []
        for fact in problem_facts(problem, section):
            if (
                isinstance(fact, list) and len(fact) == 3
                and fact[0] in aliases
                and object_types.get(str(fact[1]), set()) & source_types
                and object_types.get(str(fact[2]), set()) & target_types
            ):
                fact = ["inserted", fact[1], fact[2]]
                changed = True
            rewritten.append(fact)
        set_problem_facts(problem, section, _unique_nodes(rewritten))
    return render_problem(problem) if changed else problem_text


def ensure_turntable_type_facts(
    domain: str,
    problem_text: str,
    plan_text: str,
) -> str:
    name = "pick_paper_cup_from_microwave_turntable"
    schema = domain_actions(domain).get(name)
    if schema is None:
        return problem_text
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    original_init = copy.deepcopy(init)
    for line in plan_text.splitlines():
        tokens = line.split(";", 1)[0].strip().strip("()").split()
        if not tokens or tokens[0] != name or len(tokens[1:]) != len(schema.params):
            continue
        grounding = dict(zip(schema.params, tokens[1:]))
        for literal in conjunction(schema.pre):
            if not (
                isinstance(literal, list) and len(literal) == 2
                and literal[0] in {"hand", "paper_cup", "microwave_turntable"}
            ):
                continue
            fact: Node = [literal[0], grounding.get(str(literal[1]), literal[1])]
            if fact not in init:
                init.append(fact)
    if init == original_init:
        return problem_text
    set_problem_facts(problem, ":init", init)
    return render_problem(problem)


def normalize_initial_pressed_control_interface(
    domain: str,
    problem_text: str,
) -> str:
    """Use pressing for an initially held control consumed by release."""
    release_types = {
        str(item[0])
        for action in domain_actions(domain).values()
        if action.name.startswith("release_")
        for item in conjunction(action.pre)
        if isinstance(item, list) and len(item) == 2
        and isinstance(item[0], str) and "control" in item[0]
    }
    if not release_types:
        return problem_text
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    typed = {
        str(fact[1]): str(fact[0]) for fact in init
        if isinstance(fact, list) and len(fact) == 2 and fact[0] in release_types
    }
    pressed = {
        str(fact[1]) for fact in init
        if isinstance(fact, list) and len(fact) == 2 and fact[0] == "pressed"
    }
    changed = False
    rewritten: list[Node] = []
    for fact in init:
        if (
            isinstance(fact, list) and len(fact) == 3 and fact[0] == "holding"
            and str(fact[2]) in pressed and str(fact[2]) in typed
        ):
            fact = ["pressing", fact[1], fact[2]]
            changed = True
        rewritten.append(fact)
    if not changed:
        return problem_text
    set_problem_facts(problem, ":init", _unique_nodes(rewritten))
    return render_problem(problem)


def connected_lid_transform(action: Action, tid: int) -> Action:
    if tid in {19, 20} and action.name in {"open_kettle_lid", "open_kettle"}:
        hand = unary_var(action.pre, {"hand"})
        kettle = unary_var(action.pre, {"kettle"})
        if hand and kettle:
            action.name = "open_kettle"
            action.params = [hand, kettle]
            action.pre = make_and([["hand", hand], ["kettle", kettle], ["hand_free", hand], ["closed", kettle]])
            action.eff = make_and([["not", ["closed", kettle]], ["open", kettle]])
            action.comment = "; Open kettle ?k with hand ?h."
    if tid in {19, 20} and action.name in {"close_kettle_lid", "close_kettle"}:
        hand = unary_var(action.pre, {"hand"})
        kettle = unary_var(action.pre, {"kettle"})
        if kettle is None:
            open_variables = [
                str(item[1]) for item in conjunction(action.pre)
                if isinstance(item, list) and len(item) == 2 and item[0] == "open"
            ]
            closed_variables = [
                str(item[1]) for item in conjunction(action.eff)
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and item[0] == "closed"
                )
            ]
            candidates = set(open_variables) & set(closed_variables)
            if len(candidates) == 1:
                kettle = next(iter(candidates))
        if hand and kettle:
            action.name = "close_kettle"
            action.params = [hand, kettle]
            action.pre = make_and([
                ["hand", hand], ["kettle", kettle], ["hand_free", hand], ["open", kettle],
            ])
            action.eff = make_and([["not", ["open", kettle]], ["closed", kettle]])
            action.comment = "; Close kettle ?k with hand ?h."
    if action.name == "open_laptop_lid":
        action.name = "open_laptop"
        action.comment = "; Open laptop ?l with hand ?h."
    if action.name == "close_laptop_lid":
        action.name = "close_laptop"
        action.comment = "; Close laptop ?l with hand ?h."
    action.name = action.name.replace("_from_laptop_lid", "_from_laptop")
    return action


def closure_binding(action: Action) -> tuple[str, str, str, bool] | None:
    """Return hand, closure, container, and whether the closure is removed."""
    hand = unary_var(action.pre, {"hand"})
    if not hand:
        return None
    opening_candidates: set[tuple[str, str]] = set()
    for item in conjunction(action.eff):
        if not (
            isinstance(item, list)
            and len(item) == 3
            and item[0] == "holding"
            and item[1] == hand
        ):
            continue
        closure = str(item[2])
        for relation in conjunction(action.pre):
            if (
                isinstance(relation, list)
                and len(relation) == 3
                and relation[:2] == ["on", closure]
                and has_literal(
                    action.eff, "on", closure, str(relation[2]), negative=True
                )
                and has_literal(action.eff, "open", str(relation[2]))
            ):
                opening_candidates.add((closure, str(relation[2])))
    if len(opening_candidates) == 1:
        closure, container = next(iter(opening_candidates))
        return hand, closure, container, True

    closing_candidates: set[tuple[str, str]] = set()
    for item in conjunction(action.pre):
        if not (
            isinstance(item, list)
            and len(item) == 3
            and item[0] == "holding"
            and item[1] == hand
        ):
            continue
        closure = str(item[2])
        for relation in conjunction(action.eff):
            if (
                isinstance(relation, list)
                and len(relation) == 3
                and relation[:2] == ["on", closure]
                and has_literal(action.eff, "holding", hand, closure, negative=True)
                and has_literal(action.eff, "closed", str(relation[2]))
            ):
                closing_candidates.add((closure, str(relation[2])))
    if len(closing_candidates) == 1:
        closure, container = next(iter(closing_candidates))
        return hand, closure, container, False
    return None


def is_threaded_closure(
    action: Action,
    closure_kind: str,
    _container_kind: str,
) -> bool:
    return bool(
        action.name.startswith(("screw_", "unscrew_"))
        or "_screw_" in action.name
        or closure_kind.endswith("cap")
    )


def separable_lid_transform(action: Action, tid: int) -> Action:
    binding = closure_binding(action)
    if binding is not None:
        hand, lid, container, opening = binding
        lid_kind = unary_type(action.pre, lid) or "lid"
        container_kind = unary_type(action.pre, container) or "container"
        if is_threaded_closure(action, lid_kind, container_kind):
            return action
        if opening:
            action.name = f"remove_lid_from_{container_kind}"
            action.params = [hand, lid, container]
            action.pre = make_and([
                ["hand", hand], [lid_kind, lid], [container_kind, container],
                ["hand_free", hand], ["closed", container], ["on", lid, container],
            ])
            action.eff = make_and([
                ["not", ["hand_free", hand]], ["holding", hand, lid],
                ["not", ["closed", container]], ["open", container],
                ["not", ["on", lid, container]],
            ])
            action.comment = (
                f"; Remove lid ?l from {container_kind.replace('_', ' ')} "
                f"?{container.lstrip('?')} with hand ?h."
            )
        else:
            canonical_effects: list[Node] = [
                ["not", ["holding", hand, lid]], ["hand_free", hand],
                ["not", ["open", container]], ["closed", container],
                ["on", lid, container],
            ]
            if container_kind == "pot":
                canonical_effects.append(["not", ["upright", lid]])
            action.name = f"place_lid_on_{container_kind}"
            action.params = [hand, lid, container]
            action.pre = make_and([
                ["hand", hand], ["lid", lid], [container_kind, container],
                ["holding", hand, lid], ["open", container],
            ])
            action.eff = make_and(canonical_effects)
            action.comment = (
                f"; Place lid ?l on {container_kind.replace('_', ' ')} "
                f"?{container.lstrip('?')} with hand ?h."
            )
        return action

    pre = action.pre
    eff = action.eff
    hand = unary_var(pre, {"hand"})
    holding_effects = [
        (str(item[1]), str(item[2]))
        for item in conjunction(eff)
        if isinstance(item, list) and len(item) == 3 and item[0] == "holding"
    ]
    lid = next(
        (
            obj for held_hand, obj in holding_effects
            if held_hand == hand and (unary_type(pre, obj) or "").endswith(("lid", "cap"))
        ),
        None,
    )
    if lid is None and hand:
        lid = next(
            (
                str(item[2]) for item in conjunction(pre)
                if (
                    isinstance(item, list) and len(item) == 3
                    and item[:2] == ["holding", hand]
                    and (unary_type(pre, str(item[2])) or "").endswith(("lid", "cap"))
                )
            ),
            None,
        )
    container = None
    if lid:
        cover_relation = next(
            (
                item for item in conjunction(pre)
                if isinstance(item, list) and len(item) == 3 and item[:2] == ["on", lid]
            ),
            None,
        )
        if cover_relation:
            container = str(cover_relation[2])
        if container is None:
            placed_relation = next(
                (
                    item for item in conjunction(eff)
                    if isinstance(item, list) and len(item) == 3 and item[:2] == ["on", lid]
                ),
                None,
            )
            if placed_relation:
                container = str(placed_relation[2])
    removes_lid = bool(
        hand and lid and container
        and has_literal(eff, "holding", hand, lid)
        and has_literal(eff, "open", container)
        and has_literal(eff, "on", lid, container, negative=True)
    )
    if removes_lid:
        kind = unary_type(pre, container) or "container"
        lid_kind = unary_type(pre, lid) or "lid"
        action.name = f"remove_lid_from_{kind}"
        action.params = [hand, lid, container]
        action.pre = make_and([
            ["hand", hand], [lid_kind, lid], [kind, container], ["hand_free", hand],
            ["closed", container], ["on", lid, container],
        ])
        action.eff = make_and([
            ["not", ["hand_free", hand]], ["holding", hand, lid],
            ["not", ["closed", container]], ["open", container],
            ["not", ["on", lid, container]],
        ])
        action.comment = f"; Remove lid ?l from {kind} ?{container.lstrip('?')} with hand ?h."
        return action

    if tid in {114} and action.name == "cover_pot_with_lid":
        action.name = "place_lid_on_pot"
        action.comment = "; Place lid ?l on pot ?p with hand ?h."
    if tid in {270, 271, 272, 276, 277, 278}:
        if re.fullmatch(
            r"place_(?:kettle_)?lid_on_(?:(?:red|cardboard)_)?box",
            action.name,
        ):
            action.name = "place_lid_on_box_top"
            action.comment = "; Place lid ?l on box top ?b with hand ?h."
        if re.fullmatch(
            r"pick_(?:kettle_)?lid_from_(?:(?:red|cardboard)_)?box",
            action.name,
        ):
            action.name = "pick_lid_from_box_top"
            action.comment = "; Pick lid ?l from box top ?b with hand ?h."
    if tid in {270, 271, 272, 276, 277, 278, 291, 292, 293} and action.name in {
        "place_kettle_lid_on_kettle", "place_lid_on_kettle"
    }:
        action.name = "place_lid_on_kettle"
        action.comment = "; Place lid ?l on kettle ?k with hand ?h."
    return action


def faucet_transform(action: Action, tid: int) -> Action:
    if tid not in {104, 105, 106}:
        return action
    hand = unary_var(action.pre, {"hand"})
    faucet = unary_var(action.pre, {"faucet"})
    if action.name == "turn_on_faucet" and hand and faucet:
        action.params = [hand, faucet]
        action.pre = make_and([["hand", hand], ["faucet", faucet], ["is_off", faucet]])
        action.eff = make_and([["not", ["is_off", faucet]], ["is_on", faucet]])
        action.comment = "; Turn on faucet ?f with hand ?h."
    if action.name == "turn_off_faucet" and hand and faucet:
        bowl = unary_var(action.pre, {"bowl"})
        sink = unary_var(action.pre, {"sink"})
        if not bowl or not sink:
            raise ValueError("cannot infer bowl/sink for rinsing faucet action")
        action.name = "turn_off_faucet_after_rinsing_bowl"
        action.params = [hand, faucet, bowl, sink]
        action.pre = make_and([
            ["hand", hand], ["faucet", faucet], ["bowl", bowl], ["sink", sink],
            ["hand_free", hand], ["is_on", faucet], ["in", bowl, sink],
        ])
        action.eff = make_and([
            ["not", ["is_on", faucet]], ["is_off", faucet], ["rinsed", bowl], ["has_water", bowl],
        ])
        action.comment = "; Turn off faucet ?f after rinsing bowl ?b in sink ?s with hand ?h."
    return action


def canonical_faucet_toggle(action: Action) -> Action:
    """Keep ordinary faucet controls independent of processing context."""
    if action.name not in {"turn_on_faucet", "turn_off_faucet"}:
        return action
    hand = unary_var(action.pre, {"hand"})
    faucet = unary_var(action.pre, {"faucet"})
    if not hand or not faucet:
        return action
    turning_on = action.name == "turn_on_faucet"
    before = "is_off" if turning_on else "is_on"
    after = "is_on" if turning_on else "is_off"
    action.params = [hand, faucet]
    action.pre = make_and([
        ["hand", hand], ["faucet", faucet], [before, faucet],
    ])
    action.eff = make_and([
        ["not", [before, faucet]], [after, faucet],
    ])
    action.comment = (
        f"; Turn {'on' if turning_on else 'off'} faucet ?f with hand ?h."
    )
    return action


def canonical_water_button_contract(action: Action) -> Action:
    """Project hot/cold dispenser buttons onto toggle and fill contracts."""
    match = re.fullmatch(r"turn_(on|off)_(hot|cold)_water_button", action.name)
    if match is None:
        return action
    mode, temperature = match.groups()
    hand = unary_var(action.pre, {"hand"})
    button_binding = _variable_with_type(
        action, {f"{temperature}_water_button", "button"}
    )
    if hand is None or button_binding is None:
        return action
    button = button_binding[1]
    button_kind = f"{temperature}_water_button"
    before, after = ("is_off", "is_on") if mode == "on" else ("is_on", "is_off")

    if mode == "on":
        lock_binding = _variable_with_type(action, {"child_lock", "child_lock_button"})
        pre: list[Node] = [
            ["hand", hand], [button_kind, button], ["hand_free", hand],
            [before, button],
        ]
        preferred = [hand, button]
        if lock_binding is not None and has_literal(action.pre, "unlocked", lock_binding[1]):
            lock = lock_binding[1]
            pre.extend([["child_lock", lock], ["unlocked", lock]])
            preferred.append(lock)
            action.name += "_when_child_lock_unlocked"
        action.pre = make_and(pre)
        action.eff = make_and([["not", [before, button]], [after, button]])
        action.params = ordered_used_parameters(action, preferred)
        return action

    additions = [
        literal for literal in conjunction(action.eff)
        if isinstance(literal, list) and len(literal) == 3 and literal[0] == "in"
    ]
    if not additions:
        action.pre = make_and([
            ["hand", hand], [button_kind, button], ["hand_free", hand],
            [before, button],
        ])
        action.eff = make_and([["not", [before, button]], [after, button]])
        action.params = ordered_used_parameters(action, [hand, button])
        return action

    liquid, cup = str(additions[0][1]), str(additions[0][2])
    removed_contents = [
        str(literal[1][1]) for literal in conjunction(action.eff)
        if (
            isinstance(literal, list) and len(literal) == 2
            and literal[0] == "not" and isinstance(literal[1], list)
            and len(literal[1]) == 3 and literal[1][0] == "in"
            and literal[1][2] == cup
        )
    ]
    dispensed_liquids = [
        str(literal[2]) for literal in conjunction(action.pre)
        if isinstance(literal, list) and len(literal) == 3
        and literal[0] == "dispenses"
    ]
    supplied = dispensed_liquids[0] if dispensed_liquids else liquid
    source = next(
        (
            str(literal[2]) for literal in conjunction(action.pre)
            if (
                isinstance(literal, list)
                and len(literal) == 3
                and literal[0] == "under"
                and literal[1] == cup
            )
        ),
        None,
    )
    if source is None:
        return action
    pre = [
        ["hand", hand], [button_kind, button], ["water", supplied],
        ["cup", cup], ["water_dispenser", source], ["hand_free", hand],
        [before, button], ["under", cup, source], ["dispenses", source, supplied],
    ]
    effects: list[Node] = [
        ["not", [before, button]], [after, button], ["in", liquid, cup],
    ]
    preferred = [hand, button, supplied]
    if removed_contents:
        existing = removed_contents[0]
        pre.extend([["water", existing], ["water", liquid], ["in", existing, cup]])
        effects.insert(-1, ["not", ["in", existing, cup]])
        preferred.extend([existing, liquid])
        action.name += "_after_mixing_water_in_cup"
    else:
        preferred.append(liquid)
        action.name += "_after_filling_cup"
    preferred.extend([cup, source])
    lock_binding = _variable_with_type(action, {"child_lock", "child_lock_button"})
    if lock_binding is not None and has_literal(action.eff, "locked", lock_binding[1]):
        lock = lock_binding[1]
        pre.extend([["child_lock", lock], ["unlocked", lock]])
        effects.extend([["not", ["unlocked", lock]], ["locked", lock]])
        preferred.append(lock)
        action.name += "_and_lock_child_lock"
    action.pre = make_and(pre)
    action.eff = make_and(effects)
    action.params = ordered_used_parameters(action, preferred)
    return action


def canonical_child_lock_contract(action: Action) -> Action:
    if action.name not in {"lock_child_lock", "unlock_child_lock"}:
        return action
    hand = unary_var(action.pre, {"hand"})
    lock_binding = _variable_with_type(
        action, {"child_lock", "child_lock_button", "button"}
    )
    if hand is None or lock_binding is None:
        return action
    lock = lock_binding[1]
    unlocking = action.name == "unlock_child_lock"
    before, after = ("locked", "unlocked") if unlocking else ("unlocked", "locked")
    action.params = [hand, lock]
    action.pre = make_and([
        ["hand", hand], ["child_lock", lock], ["hand_free", hand],
        [before, lock],
    ])
    action.eff = make_and([["not", [before, lock]], [after, lock]])
    return action


def canonical_simple_power_toggle(action: Action) -> Action:
    match = re.fullmatch(r"turn_(on|off)_(water_pump|water_dispenser)", action.name)
    if match is None:
        return action
    mode, kind = match.groups()
    hand = unary_var(action.pre, {"hand"})
    control = unary_var(action.pre, {kind})
    if hand is None or control is None:
        return action
    before, after = ("is_off", "is_on") if mode == "on" else ("is_on", "is_off")
    # A resultful stop is normalized by the process-end contract instead.
    if mode == "off" and any(
        isinstance(item, list) and len(item) == 3 and item[0] == "in"
        for item in conjunction(action.eff)
    ):
        return action
    action.params = [hand, control]
    action.pre = make_and([["hand", hand], [kind, control], [before, control]])
    action.eff = make_and([["not", [before, control]], [after, control]])
    return action


def canonical_kettle_start_contract(action: Action) -> Action:
    if action.name != "turn_on_kettle":
        return action
    hand = unary_var(action.pre, {"hand"})
    kettle = unary_var(action.pre, {"kettle"})
    if hand is None or kettle is None:
        return action
    water_relation = next(
        (
            item for item in conjunction(action.pre)
            if isinstance(item, list) and len(item) == 3 and item[0] == "in"
            and item[2] == kettle and unary_type(action.pre, str(item[1])) == "water"
        ),
        None,
    )
    base_relation = next(
        (
            item for item in conjunction(action.pre)
            if isinstance(item, list) and len(item) == 3 and item[:2] == ["on", kettle]
            and unary_type(action.pre, str(item[2])) == "kettle_base"
        ),
        None,
    )
    if water_relation is None or base_relation is None:
        return action
    water, base = str(water_relation[1]), str(base_relation[2])
    pre: list[Node] = [
        ["hand", hand], ["kettle", kettle], ["water", water],
        ["kettle_base", base], ["hand_free", hand], ["closed", kettle],
        ["is_off", kettle], ["in", water, kettle], ["on", kettle, base],
    ]
    preferred = [hand, kettle, water, base]
    inserted = next(
        (
            item for item in conjunction(action.pre)
            if isinstance(item, list) and len(item) == 3 and item[0] == "inserted"
            and unary_type(action.pre, str(item[1])) == "plug"
        ),
        None,
    )
    if inserted is not None:
        plug, outlet = str(inserted[1]), str(inserted[2])
        pre.extend([["plug", plug], ["outlet", outlet], ["inserted", plug, outlet]])
        preferred.extend([plug, outlet])
        action.name += "_when_plug_inserted"
    action.pre = make_and(pre)
    action.eff = make_and([["not", ["is_off", kettle]], ["is_on", kettle]])
    action.params = ordered_used_parameters(action, preferred)
    return action


def canonical_detergent_drawer_contract(action: Action) -> Action:
    if "detergent_drawer" not in action.name:
        return action
    opening = _drawer_transition_target(action, opening=True)
    closing = _drawer_transition_target(action, opening=False)
    target = opening or closing
    if target is None:
        return action
    hand = unary_var(action.pre, {"hand"})
    if hand is None:
        return action
    before, after = ("closed", "open") if opening else ("open", "closed")
    action.name = f"{'open' if opening else 'close'}_detergent_drawer"
    action.params = [hand, target]
    action.pre = make_and([
        ["hand", hand], ["detergent_drawer", target], ["hand_free", hand],
        [before, target],
    ])
    action.eff = make_and([["not", [before, target]], [after, target]])
    return action


def restore_named_manipulation_type(action: Action) -> Action:
    """Restore an omitted moved-object type when the action name is explicit."""
    match = re.match(r"(?:pick|place)_(mug|cup|bowl|plate|book|box|plug)_", action.name)
    if match is None:
        return action
    moving = placed_object_var(action) if action.name.startswith("place_") else None
    if moving is None:
        held = {
            str(literal[2]) for literal in conjunction(action.eff)
            if isinstance(literal, list) and len(literal) == 3
            and literal[0] == "holding"
        }
        moving = next(iter(held)) if len(held) == 1 else None
    if moving is not None and unary_type(action.pre, moving) is None:
        action.pre = add_literal(action.pre, [match.group(1), moving])
    return action


def canonical_power_connection_contract(action: Action) -> Action:
    specs = {
        "insert_plug_in_wall_outlet": ("plug", "wall_outlet"),
        "plug_plug_in_wall_outlet": ("plug", "wall_outlet"),
        "plug_power_base_cord_in_wall_outlet": ("power_base_cord", "wall_outlet"),
        "insert_plug_in_outlet": ("plug", "outlet"),
        "plug_power_base_cord_in_outlet": ("power_base_cord", "outlet"),
    }
    spec = specs.get(action.name)
    if spec is None:
        return action
    hand = unary_var(action.pre, {"hand"})
    held = [
        str(literal[2]) for literal in conjunction(action.pre)
        if isinstance(literal, list) and len(literal) == 3
        and literal[0] == "holding" and literal[1] == hand
    ]
    if hand is None or len(set(held)) != 1:
        return action
    moving = held[0]
    relation = next(
        (
            candidate for literal in conjunction(action.eff)
            for candidate in (
                literal[1] if (
                    isinstance(literal, list) and len(literal) == 2
                    and literal[0] == "not" and isinstance(literal[1], list)
                ) else literal,
            )
            if isinstance(candidate, list) and len(candidate) == 3
            and candidate[0] in {"inserted", "plugged", "plugged_in", "plugged_into", "connected", "in"}
            and candidate[1] == moving
        ),
        None,
    )
    if relation is None:
        return action
    target = str(relation[2])
    moving_kind, target_kind = spec
    action.pre = make_and([
        ["hand", hand], [moving_kind, moving], [target_kind, target],
        ["holding", hand, moving],
    ])
    action.eff = make_and([
        ["not", ["holding", hand, moving]], ["hand_free", hand],
        ["inserted", moving, target],
    ])
    action.params = [hand, moving, target]
    return action


def canonical_rinsed_bowl_pick(
    domain: str,
    problem_text: str,
    plan_text: str,
    tid: int,
) -> tuple[str, str, str, bool]:
    """Bind post-rinse bowl pickup to the faucet that is already off."""
    if tid not in {104, 105, 106}:
        return domain, problem_text, plan_text, False

    steps: list[tuple[int, list[str]]] = []
    for line_index, line in enumerate(plan_text.splitlines()):
        stripped = line.split(";", 1)[0].strip()
        if stripped.startswith("("):
            steps.append((line_index, stripped.strip("()").split()))
    if not any(tokens and tokens[0] == "pick_bowl_from_sink" for _, tokens in steps):
        return domain, problem_text, plan_text, False

    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    objects = {str(item) for item in problem_section(problem, ":objects")[1:]}
    faucets = typed_objects(problem, "faucet")
    assignments: dict[int, str] = {}
    latest_close: dict[tuple[str, str, str], str] = {}
    problem_changed = False

    def require_init(fact: Node, context: str) -> None:
        if fact not in init:
            raise ValueError(f"post-rinse suffix missing {context}: {sexp(fact)}")

    def add_suffix_faucet(bowl: str) -> str:
        nonlocal problem_changed
        base = "faucet"
        faucet = base
        suffix = 2
        while faucet in objects:
            faucet = f"{base}_{suffix}"
            suffix += 1
        add_object(problem, faucet)
        objects.add(faucet)
        faucets.append(faucet)
        for fact in (["faucet", faucet], ["is_off", faucet], ["rinsed", bowl]):
            if fact not in init:
                init.append(fact)
        problem_changed = True
        return faucet

    for step_index, (line_index, tokens) in enumerate(steps):
        if not tokens:
            continue
        if tokens[0] in {
            "turn_off_faucet_after_rinsing_bowl",
            "turn_off_faucet_after_rinsing",
        }:
            if len(tokens) != 5:
                raise ValueError("contextual faucet close must have four arguments")
            hand, faucet, bowl, sink = tokens[1:]
            latest_close[(hand, bowl, sink)] = faucet
            continue
        if tokens[0] != "pick_bowl_from_sink":
            continue
        if len(tokens) not in {4, 5}:
            raise ValueError("pick_bowl_from_sink must have three or four arguments")
        hand, bowl, sink = tokens[1:4]
        faucet = latest_close.get((hand, bowl, sink))
        if faucet is not None:
            if len(tokens) == 5 and tokens[4] != faucet:
                raise ValueError("post-rinse bowl pick is bound to a different faucet")
        elif len(tokens) == 5:
            faucet = tokens[4]
            require_init(["faucet", faucet], "faucet type")
            require_init(["is_off", faucet], "closed rinsing interval")
            require_init(["rinsed", bowl], "completed rinsing result")
        else:
            if step_index != 0 or faucets:
                raise ValueError("pick_bowl_from_sink has no matching rinsing faucet")
            for fact, context in (
                (["hand", hand], "hand type"),
                (["bowl", bowl], "bowl type"),
                (["sink", sink], "sink type"),
                (["hand_free", hand], "free hand"),
                (["has_water", bowl], "post-rinse bowl content"),
                (["in", bowl, sink], "bowl in sink"),
            ):
                require_init(fact, context)
            faucet = add_suffix_faucet(bowl)
        assignments[line_index] = faucet

    def gate_pick(action: Action) -> Action:
        if action.name != "pick_bowl_from_sink":
            return action
        hand = unary_var(action.pre, {"hand"})
        bowl = unary_var(action.pre, {"bowl"})
        sink = unary_var(action.pre, {"sink"})
        if not hand or not bowl or not sink:
            raise ValueError("cannot infer pick-bowl variables")
        faucet = unary_var(action.pre, {"faucet"}) or fresh_variable(action.params, "?f")
        action.params = [hand, bowl, sink, faucet]
        action.pre = make_and([
            ["hand", hand], ["faucet", faucet], ["bowl", bowl], ["sink", sink],
            ["hand_free", hand], ["is_off", faucet], ["rinsed", bowl], ["in", bowl, sink],
        ])
        action.eff = make_and([
            ["not", ["hand_free", hand]], ["holding", hand, bowl],
            ["not", ["in", bowl, sink]],
        ])
        action.comment = "; Pick rinsed bowl ?b from sink ?s after faucet ?f is off with hand ?h."
        return action

    original_domain = domain
    domain, _ = rewrite_actions(domain, gate_pick)
    lines = plan_text.splitlines()
    plan_changed = False
    for line_index, faucet in assignments.items():
        tokens = lines[line_index].split(";", 1)[0].strip().strip("()").split()
        rewritten = "(" + " ".join([*tokens[:4], faucet]) + ")"
        if lines[line_index].strip() != rewritten:
            lines[line_index] = rewritten
            plan_changed = True
    if plan_changed:
        lines = [line for line in lines if not line.strip().startswith("; cost =")]
        lines.append(f"; cost = {len(steps)} (unit cost)")
        plan_text = "\n".join(lines) + "\n"
    if problem_changed:
        set_problem_facts(problem, ":init", init)
        problem_text = render_problem(problem)
    return domain, problem_text, plan_text, bool(
        domain != original_domain or problem_changed or plan_changed
    )


def running_water_transform(action: Action, tid: int) -> Action:
    if tid in {107, 108} and action.name == "rinse_bowl_under_faucet":
        hand = unary_var(action.pre, {"hand"})
        bowl = unary_var(action.pre, {"bowl"})
        faucet = unary_var(action.pre, {"faucet"})
        if hand and bowl and faucet:
            action.name = "hold_bowl_under_running_faucet_until_rinsed"
            action.params = [hand, bowl, faucet]
            action.pre = make_and([
                ["hand", hand], ["bowl", bowl], ["faucet", faucet],
                ["holding", hand, bowl], ["is_on", faucet],
            ])
            action.eff = make_and([["rinsed", bowl], ["has_water", bowl]])
            action.comment = "; Hold bowl ?b under running faucet ?f with hand ?h until rinsed."
    if tid == 241 and action.name == "wet_sponge_under_faucet":
        hand = unary_var(action.pre, {"hand"})
        sponge = unary_var(action.pre, {"sponge"})
        faucet = unary_var(action.pre, {"faucet"})
        if hand and sponge and faucet:
            action.name = "hold_sponge_under_running_faucet_until_wet"
            action.params = [hand, sponge, faucet]
            action.pre = make_and([
                ["hand", hand], ["sponge", sponge], ["faucet", faucet],
                ["holding", hand, sponge], ["is_on", faucet],
            ])
            action.eff = make_and([["wet", sponge]])
            action.comment = "; Hold sponge ?s under running faucet ?f with hand ?h until wet."
    return action


INFINITE_SOURCE_TYPES = {
    "faucet", "water_faucet", "water_pump", "dispenser", "hot_water_dispenser",
    "cold_water_dispenser", "water_dispenser", "spout", "water_spout",
    "hot_water_spout", "cold_water_spout", "countertop_water_dispenser",
    "hot_water_nozzle", "cold_water_nozzle", "hot_water_outlet", "cold_water_outlet",
}

FINITE_WATER_SOURCE_TYPES = {"water_jug", "water_bottle"}
WATER_CONTROL_TYPES = {
    "water_pump", "water_jug_pump", "water_bottle_pump",
    "water_dispenser", "hot_water_dispenser", "cold_water_dispenser",
    "countertop_water_dispenser",
}
WATER_BUTTON_TYPES = {
    "pump_button", "dispenser_button", "water_button",
    "hot_water_button", "cold_water_button",
}


def _variable_with_type(action: Action, kinds: set[str]) -> tuple[str, str] | None:
    for literal in conjunction(action.pre):
        if (
            isinstance(literal, list)
            and len(literal) == 2
            and literal[0] in kinds
            and isinstance(literal[1], str)
            and literal[1].startswith("?")
        ):
            return str(literal[0]), str(literal[1])
    return None


def _water_transfer(action: Action) -> tuple[str, str, str] | None:
    additions = [
        literal for literal in conjunction(action.eff)
        if isinstance(literal, list) and len(literal) == 3 and literal[0] == "in"
    ]
    for addition in additions:
        liquid = str(addition[1])
        target = str(addition[2])
        target_kind = unary_type(action.pre, target)
        if unary_type(action.pre, liquid) == "water" and target_kind in {
            "kettle", "glass_kettle", "electric_kettle",
        }:
            return liquid, target, target_kind
    return None


def _finite_water_source(action: Action, liquid: str) -> tuple[str, str] | None:
    for literal in conjunction(action.pre):
        if not (
            isinstance(literal, list)
            and len(literal) == 3
            and literal[0] == "in"
            and literal[1] == liquid
        ):
            continue
        source = str(literal[2])
        source_kind = unary_type(action.pre, source)
        if source_kind in FINITE_WATER_SOURCE_TYPES:
            return source_kind, source
    return None


def water_supply_action_contract(action: Action) -> Action:
    """Canonicalize an atomic, momentary water-control action."""
    transfer = _water_transfer(action)
    button_binding = _variable_with_type(action, WATER_BUTTON_TYPES)
    control_binding = _variable_with_type(action, WATER_CONTROL_TYPES)
    hand = unary_var(action.pre, {"hand"})
    if transfer is None or button_binding is None or control_binding is None or hand is None:
        return action
    if any(
        has_literal(action.eff, predicate, control_binding[1])
        for predicate in ("is_on", "is_off")
    ):
        return action

    liquid, kettle, kettle_kind = transfer
    button_kind, button = button_binding
    control_kind, control = control_binding
    action.name = f"press_{control_kind}_button_to_fill_kettle"
    action.pre = make_and([
        ["hand", hand], [button_kind, button], [control_kind, control],
        ["water", liquid], [kettle_kind, kettle], ["hand_free", hand],
        *([["open", kettle]] if has_literal(action.pre, "open", kettle) else []),
        ["dispenses", control, liquid],
    ])
    action.eff = make_and([["in", liquid, kettle]])
    action.params = ordered_used_parameters(
        action, [hand, button, control, liquid, kettle]
    )
    action.comment = "; Press the dispenser button to fill the kettle."
    return action


def held_press_control_kinds(domain: str) -> set[str]:
    """Return control categories with a hand-occupying press schema."""
    result: set[str] = set()
    for action in domain_actions(domain).values():
        hand = unary_var(action.pre, {"hand", "robot_hand"})
        if hand is None:
            continue
        controls = [
            str(item[1]) for item in conjunction(action.eff)
            if isinstance(item, list) and len(item) == 2 and item[0] == "pressed"
        ]
        if len(set(controls)) != 1:
            continue
        control = controls[0]
        if not (
            has_literal(action.eff, "holding", hand, control)
            or has_literal(action.eff, "pressing", hand, control)
        ):
            continue
        kind = unary_type(action.pre, control)
        if kind is not None:
            result.add(kind)
    return result


def canonical_press_release_contract(
    action: Action,
    paired_control_kinds: set[str] | None = None,
) -> Action:
    """Canonicalize controls that occupy the hand until a paired release."""
    hand = unary_var(action.pre, {"hand", "robot_hand"})
    if hand is None:
        return action

    pressing_effects = [
        str(item[2]) for item in conjunction(action.eff)
        if (
            isinstance(item, list) and len(item) == 3
            and item[0] == "pressing" and item[1] == hand
        )
    ]
    if len(set(pressing_effects)) == 1 and has_literal(
        action.eff, "hand_free", hand, negative=True
    ):
        control = pressing_effects[0]
        control_kind = unary_type(action.pre, control)
        if (
            control_kind in {"water_jug_pump", "water_bottle_pump"}
            and has_literal(action.pre, "hand_free", hand)
        ):
            action.name = f"press_and_hold_{control_kind}"
            action.params = [hand, control]
            action.pre = make_and([
                ["hand", hand], [control_kind, control], ["hand_free", hand],
            ])
            action.eff = make_and([
                ["not", ["hand_free", hand]], ["pressing", hand, control],
            ])
            action.comment = f"; Press and hold {control_kind.replace('_', ' ')}."
            return action

    pump_pressed_effects = [
        str(item[1]) for item in conjunction(action.eff)
        if isinstance(item, list) and len(item) == 2 and item[0] == "pump_pressed"
    ]
    if len(set(pump_pressed_effects)) == 1 and has_literal(
        action.eff, "hand_free", hand, negative=True
    ):
        control = pump_pressed_effects[0]
        control_kind = unary_type(action.pre, control)
        if control_kind is not None and has_literal(action.pre, "hand_free", hand):
            action.name = f"press_and_hold_{control_kind}"
            action.params = [hand, control]
            action.pre = make_and([
                ["hand", hand], [control_kind, control], ["hand_free", hand],
            ])
            action.eff = make_and([
                ["not", ["hand_free", hand]], ["pressing", hand, control],
            ])
            action.comment = f"; Press and hold {control_kind.replace('_', ' ')}."
            return action

    pressed_effects = [
        str(item[1]) for item in conjunction(action.eff)
        if isinstance(item, list) and len(item) == 2 and item[0] == "pressed"
    ]
    released_effects = [
        str(item[1][1]) for item in conjunction(action.eff)
        if (
            isinstance(item, list) and len(item) == 2 and item[0] == "not"
            and isinstance(item[1], list) and len(item[1]) == 2
            and item[1][0] == "pressed"
        )
    ]

    if len(set(pressed_effects)) == 1:
        control = pressed_effects[0]
        held = (
            has_literal(action.eff, "holding", hand, control)
            or has_literal(action.eff, "pressing", hand, control)
        )
        if not held or not has_literal(action.pre, "unpressed", control):
            return action
        control_kind = unary_type(action.pre, control)
        if control_kind is None:
            return action
        action.name = f"press_and_hold_{control_kind}"
        action.params = [hand, control]
        action.pre = make_and([
            ["hand", hand], [control_kind, control], ["hand_free", hand],
            ["unpressed", control],
        ])
        action.eff = make_and([
            ["not", ["hand_free", hand]], ["pressing", hand, control],
            ["not", ["unpressed", control]], ["pressed", control],
        ])
        action.comment = f"; Press and hold {control_kind.replace('_', ' ')}."
        return action

    if len(set(released_effects)) != 1:
        return action
    control = released_effects[0]
    held = (
        has_literal(action.pre, "holding", hand, control)
        or has_literal(action.pre, "pressing", hand, control)
    )
    if not held or not has_literal(action.pre, "pressed", control):
        return action
    control_kind = unary_type(action.pre, control)
    if control_kind is None:
        return action

    interface_effects = {
        sexp(["not", ["holding", hand, control]]),
        sexp(["not", ["pressing", hand, control]]),
        sexp(["hand_free", hand]),
        sexp(["not", ["pressed", control]]),
        sexp(["unpressed", control]),
    }
    outcomes = [
        item for item in conjunction(action.eff)
        if sexp(item) not in interface_effects
    ]
    outcome_vars = set().union(*(node_variables(item) for item in outcomes)) if outcomes else set()
    preferred = [hand, control]
    uses_pressing = True
    preconditions: list[Node] = [
        ["hand", hand], [control_kind, control],
        ["pressing" if uses_pressing else "holding", hand, control],
        ["pressed", control],
    ]
    type_literals: list[Node] = []
    for variable in action.params:
        if variable not in outcome_vars:
            continue
        kind = unary_type(action.pre, variable)
        if kind is not None:
            type_literals.append([kind, variable])
            preferred.append(variable)
    preconditions.extend(type_literals)

    type_heads = {
        str(item[0]) for item in type_literals
        if isinstance(item, list) and len(item) == 2
    } | {"hand", control_kind}
    guards: list[Node] = []
    guard_labels: list[str] = []
    allowed_vars = {hand, control, *outcome_vars}
    for literal in conjunction(action.pre):
        if not isinstance(literal, list) or not literal:
            continue
        if literal[0] in type_heads or literal[0] in {"holding", "pressing", "pressed"}:
            continue
        variables = node_variables(literal)
        if not variables or not variables <= allowed_vars:
            continue
        guards.append(literal)
        if len(literal) == 2:
            kind = unary_type(action.pre, str(literal[1])) or "target"
            guard_labels.append(f"{kind}_{literal[0]}")
        elif len(literal) == 3:
            guard_labels.append(str(literal[0]))

    result_labels: list[str] = []
    for outcome in outcomes:
        candidate = outcome[1] if (
            isinstance(outcome, list) and len(outcome) == 2
            and outcome[0] == "not" and isinstance(outcome[1], list)
        ) else outcome
        if not isinstance(candidate, list) or len(candidate) < 2 or outcome is not candidate:
            continue
        if len(candidate) == 2:
            kind = unary_type(action.pre, str(candidate[1])) or "target"
            result_labels.append(f"{candidate[0]}_{kind}")
        elif len(candidate) == 3:
            target_kind = unary_type(action.pre, str(candidate[2])) or "target"
            source_kind = unary_type(action.pre, str(candidate[1])) or "content"
            result_labels.append(f"{candidate[0]}_{source_kind}_in_{target_kind}")

    action.name = f"release_{control_kind}"
    if result_labels:
        action.name += "_after_" + "_and_".join(dict.fromkeys(result_labels))
    if guard_labels:
        action.name += "_when_" + "_and_".join(dict.fromkeys(guard_labels))
    action.pre = make_and(_unique_nodes([*preconditions, *guards]))
    action.eff = make_and(_unique_nodes([
        ["not", ["pressing" if uses_pressing else "holding", hand, control]],
        ["hand_free", hand],
        ["not", ["pressed", control]], ["unpressed", control], *outcomes,
    ]))
    action.params = ordered_used_parameters(action, preferred)
    action.comment = f"; Release {control_kind.replace('_', ' ')}."
    return action


def canonical_finite_water_end_contract(action: Action) -> Action:
    """Project a finite reservoir transfer onto its release/stop contract."""
    if not action.name.startswith(("stop_", "release_", "turn_off_")):
        return action
    transfer = _water_transfer(action)
    hand = unary_var(action.pre, {"hand", "robot_hand"})
    if transfer is None or hand is None:
        return action
    liquid, kettle, _legacy_kettle_kind = transfer
    finite = _finite_water_source(action, liquid)
    if finite is None:
        return action
    source_kind, source = finite

    control_binding = _variable_with_type(action, WATER_CONTROL_TYPES)
    control_kind, control = control_binding or (source_kind, source)
    pre: list[Node] = [
        ["hand", hand], [control_kind, control], ["water", liquid],
        ["kettle", kettle], [source_kind, source], ["in", liquid, source],
    ]
    effects: list[Node] = [
        ["not", ["in", liquid, source]], ["in", liquid, kettle],
        ["not", ["empty", kettle]], ["filled", kettle],
    ]
    releasing = False
    if has_literal(action.pre, "pressing", hand, control) or has_literal(
        action.pre, "holding", hand, control
    ):
        pre.append(["pressing", hand, control])
        effects[:0] = [["not", ["pressing", hand, control]], ["hand_free", hand]]
        releasing = True
    elif has_literal(action.pre, "pump_pressed", control):
        pre.append(["pressing", hand, control])
        effects[:0] = [["not", ["pressing", hand, control]], ["hand_free", hand]]
        releasing = True
    elif has_literal(action.pre, "is_on", control):
        pre.append(["is_on", control])
        effects[:0] = [["not", ["is_on", control]], ["is_off", control]]
    else:
        return action
    if has_literal(action.pre, "open", kettle):
        pre.append(["open", kettle])
    action.name = (
        f"{'release' if releasing else 'turn_off'}_{control_kind}"
        f"_after_filling_kettle_from_{source_kind}"
    )
    action.pre = make_and(pre)
    action.eff = make_and(effects)
    action.params = ordered_used_parameters(
        action, [hand, control, liquid, kettle, source]
    )
    action.comment = "; Stop the finite water source after filling the kettle."
    return action


def normalize_water_supply_contracts(domain: str) -> tuple[str, dict[str, ActionEdit]]:
    """Normalize paired kettle-filling controls using their state transitions."""
    schemas = domain_actions(domain)
    starts: dict[str, Action] = {}
    finite_start: dict[str, tuple[str, str, str, str]] = {}
    for schema in schemas.values():
        control_binding = _variable_with_type(schema, WATER_CONTROL_TYPES)
        if control_binding is None:
            continue
        control_kind, control = control_binding
        if has_literal(schema.pre, "is_off", control) and has_literal(schema.eff, "is_on", control):
            starts[control_kind] = schema
            transfer = _water_transfer(schema)
            if transfer is not None:
                liquid, kettle, _ = transfer
                finite = _finite_water_source(schema, liquid)
                if finite is not None:
                    finite_start[control_kind] = (liquid, kettle, finite[0], finite[1])

    def transform(action: Action) -> Action:
        original_name = action.name
        atomic = water_supply_action_contract(action)
        if atomic.name != original_name:
            return atomic

        control_binding = _variable_with_type(action, WATER_CONTROL_TYPES)
        hand = unary_var(action.pre, {"hand"})
        if control_binding is None or hand is None:
            return action
        control_kind, control = control_binding
        start = has_literal(action.pre, "is_off", control) and has_literal(
            action.eff, "is_on", control
        )
        stop = has_literal(action.pre, "is_on", control) and has_literal(
            action.eff, "is_off", control
        )
        if not start and not stop:
            return action

        button_binding = _variable_with_type(action, WATER_BUTTON_TYPES)
        pressing = has_literal(action.eff, "pressing", hand, control)
        held_start = pressing or any(
            isinstance(literal, list)
            and len(literal) == 3
            and literal[0] == "pressing"
            and literal[1] == hand
            for literal in conjunction(action.eff)
        )
        if start:
            pre: list[Node] = [
                ["hand", hand], [control_kind, control], ["is_off", control],
            ]
            preferred = [hand, control]
            if button_binding is not None:
                pre.insert(2, [button_binding[0], button_binding[1]])
                preferred.append(button_binding[1])
            if held_start:
                pre.insert(-1, ["hand_free", hand])
                action.name = f"press_and_hold_{control_kind}"
                action.eff = make_and([
                    ["not", ["hand_free", hand]], ["pressing", hand, control],
                    ["not", ["is_off", control]], ["is_on", control],
                ])
            else:
                action.name = f"turn_on_{control_kind}"
                action.eff = make_and([
                    ["not", ["is_off", control]], ["is_on", control],
                ])
            action.pre = make_and(pre)
            action.params = ordered_used_parameters(action, preferred)
            action.comment = f"; Start {control_kind.replace('_', ' ')}."
            return action

        transfer = _water_transfer(action)
        inherited_finite = finite_start.get(control_kind)
        held = any(
            isinstance(literal, list)
            and len(literal) == 3
            and literal[0] == "pressing"
            and literal[1] == hand
            and literal[2] == control
            for literal in conjunction(action.pre)
        )
        if transfer is None and inherited_finite is not None:
            liquid, kettle, source_kind, source = inherited_finite
            if source not in action.params:
                source = fresh_variable(action.params, f"?{source_kind.removeprefix('water_')}")
            action.pre = add_literal(action.pre, ["water", liquid])
            kettle_kind = unary_type(starts[control_kind].pre, kettle) or "kettle"
            action.pre = add_literal(action.pre, [kettle_kind, kettle])
            action.pre = add_literal(action.pre, [source_kind, source])
            action.pre = add_literal(action.pre, ["in", liquid, source])
            action.eff = add_literal(action.eff, ["not", ["in", liquid, source]])
            action.eff = add_literal(action.eff, ["in", liquid, kettle])
            transfer = liquid, kettle, "kettle"

        if transfer is None:
            return action
        liquid, kettle, kettle_kind = transfer
        finite = _finite_water_source(action, liquid)
        pre = [
            ["hand", hand], [control_kind, control], ["water", liquid],
            ["kettle", kettle], ["is_on", control],
        ]
        if held:
            pre.append(["pressing", hand, control])
        if has_literal(action.pre, "open", kettle):
            pre.append(["open", kettle])
        effects: list[Node] = [
            ["not", ["is_on", control]], ["is_off", control],
        ]
        if held:
            effects[:0] = [["not", ["pressing", hand, control]], ["hand_free", hand]]
        suffix = ""
        preferred = [hand, control, liquid, kettle]
        if finite is not None:
            source_kind, source = finite
            pre.extend([[source_kind, source], ["in", liquid, source]])
            effects.extend([["not", ["in", liquid, source]], ["in", liquid, kettle]])
            suffix = f"_from_{source_kind}"
            preferred.append(source)
        else:
            pre.append(["dispenses", control, liquid])
            effects.append(["in", liquid, kettle])
        effects.extend([["not", ["empty", kettle]], ["filled", kettle]])
        action.name = (
            f"release_{control_kind}_after_filling_kettle{suffix}"
            if held else f"turn_off_{control_kind}_after_filling_kettle{suffix}"
        )
        action.pre = make_and(pre)
        action.eff = make_and(effects)
        action.params = ordered_used_parameters(action, preferred)
        action.comment = "; Stop the water control after filling the kettle."
        return action

    return rewrite_actions(domain, transform)


def rewrite_water_supply_plan(
    old_domain: str,
    new_domain: str,
    plan_text: str,
    edits: dict[str, ActionEdit],
) -> str:
    """Rewrite water actions, carrying a finite source from press to release."""
    if not edits:
        return plan_text
    old_schemas = domain_actions(old_domain)
    new_schemas = domain_actions(new_domain)
    last_finite_source: tuple[str, str] | None = None
    output: list[str] = []
    changed = False
    for line in plan_text.splitlines():
        stripped = line.split(";", 1)[0].strip()
        tokens = stripped.strip("()").split() if stripped.startswith("(") else []
        edit = edits.get(tokens[0]) if tokens else None
        old_schema = old_schemas.get(tokens[0]) if tokens else None
        if edit is None or old_schema is None or len(tokens[1:]) != len(old_schema.params):
            if not stripped.startswith("; cost ="):
                output.append(line)
            continue
        grounding = dict(zip(old_schema.params, tokens[1:]))
        for literal in conjunction(old_schema.pre):
            if not (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[0] in FINITE_WATER_SOURCE_TYPES
            ):
                continue
            source = grounding.get(str(literal[1]))
            if source is not None:
                last_finite_source = str(literal[0]), source

        if edit.new_name is None:
            changed = True
            continue
        new_schema = new_schemas[edit.new_name]
        args: list[str] = []
        for parameter in new_schema.params:
            if parameter in grounding:
                args.append(grounding[parameter])
                continue
            source_kind = unary_type(new_schema.pre, parameter)
            if last_finite_source is None or source_kind != last_finite_source[0]:
                raise ValueError(
                    f"cannot ground new water-source parameter {parameter} "
                    f"for {edit.new_name}"
                )
            args.append(last_finite_source[1])
        output.append("(" + " ".join([edit.new_name, *args]) + ")")
        changed = changed or edit.new_name != tokens[0] or args != tokens[1:]
    if not changed:
        return plan_text
    count = sum(line.strip().startswith("(") for line in output)
    output.append(f"; cost = {count} (unit cost)")
    return "\n".join(output) + "\n"


def infinite_source_action(action: Action) -> tuple[Action, bool]:
    finite_liquids = {
        str(literal[1])
        for literal in conjunction(action.pre)
        if (
            isinstance(literal, list)
            and len(literal) == 3
            and literal[0] == "in"
            and unary_type(action.pre, str(literal[2])) in FINITE_WATER_SOURCE_TYPES
        )
    }
    source_vars = {
        str(item[1])
        for item in conjunction(action.pre)
        if isinstance(item, list) and len(item) == 2 and item[0] in INFINITE_SOURCE_TYPES
    }
    source_liquids: set[tuple[str, str]] = set()
    for expr in (action.pre, action.eff):
        for item in conjunction(expr):
            candidate = item[1] if (
                isinstance(item, list)
                and len(item) == 2
                and item[0] == "not"
                and isinstance(item[1], list)
            ) else item
            if (
                isinstance(candidate, list)
                and len(candidate) == 3
                and candidate[0] == "in"
                and candidate[2] in source_vars
                and candidate[1] not in finite_liquids
            ):
                source_liquids.add((str(candidate[2]), str(candidate[1])))

    changed = False
    for source, liquid in source_liquids:
        action.pre = add_literal(action.pre, ["dispenses", source, liquid])
        source_fact: Node = ["in", liquid, source]
        action.pre = remove_literal(action.pre, source_fact)
        action.eff = remove_literal(action.eff, source_fact)
        action.eff = remove_literal(action.eff, ["not", source_fact])

        # Some legacy pump models move a finite water token from a reservoir
        # into the pump and then from the pump into the destination.  The pump
        # is an unlimited provider, so neither staging location is physical
        # state.  Keep only the static provider relation and the final fill.
        predecessor_vars: set[str] = set()
        for effect in list(conjunction(action.eff)):
            if not (
                isinstance(effect, list)
                and len(effect) == 2
                and effect[0] == "not"
                and isinstance(effect[1], list)
                and len(effect[1]) == 3
                and effect[1][0] == "in"
                and effect[1][1] == liquid
            ):
                continue
            predecessor = str(effect[1][2])
            if predecessor == source:
                continue
            predecessor_vars.add(predecessor)
            action.eff = remove_literal(action.eff, effect)
            action.pre = remove_literal(action.pre, effect[1])

        for predecessor in predecessor_vars:
            remaining = node_variables(action.pre) | node_variables(action.eff)
            non_type_use = any(
                predecessor in node_variables(literal)
                and not (
                    isinstance(literal, list)
                    and len(literal) == 2
                    and literal[1] == predecessor
                )
                for literal in [*conjunction(action.pre), *conjunction(action.eff)]
            )
            if not non_type_use:
                action.pre = make_and(
                    literal for literal in conjunction(action.pre)
                    if not (
                        isinstance(literal, list)
                        and len(literal) == 2
                        and literal[1] == predecessor
                    )
                )
                action.params = [param for param in action.params if param != predecessor]
        changed = True
    return action, changed


def finite_transfer_action(action: Action) -> tuple[Action, bool]:
    if not re.match(
        r"(?:pour|scoop|transfer|fill|stop_pressing|turn_off)_", action.name
    ):
        return action, False
    pre_in = [
        item for item in conjunction(action.pre)
        if isinstance(item, list) and len(item) == 3 and item[0] == "in"
    ]
    add_in = [
        item for item in conjunction(action.eff)
        if isinstance(item, list) and len(item) == 3 and item[0] == "in"
    ]
    changed = False
    for source in pre_in:
        if any(target[1] == source[1] and target[2] != source[2] for target in add_in):
            deletion: Node = ["not", source]
            if deletion not in conjunction(action.eff):
                action.eff = add_literal(action.eff, deletion)
                changed = True
    return action, changed


LIQUID_CARRIER_PREDICATES = {"has_water", "has_liquid"}
LIQUID_GUARD_PREDICATES = {
    "open", "clear", "upright", "vertical", "flat",
}


def _liquid_guard_suffix(
    action: Action,
    source: str,
    source_kind: str,
    target: str,
    target_kind: str,
) -> tuple[list[Node], str]:
    guards: list[Node] = []
    labels: list[str] = []
    for variable, kind in ((source, source_kind), (target, target_kind)):
        for predicate in LIQUID_GUARD_PREDICATES:
            literal: Node = [predicate, variable]
            if literal in conjunction(action.pre):
                guards.append(literal)
                labels.append(f"{kind}_{predicate}")
    return guards, ("_when_" + "_and_".join(labels)) if labels else ""


def finite_liquid_contract_transform(action: Action) -> Action:
    """Project finite liquid transfers onto their local causal contract."""
    if action.name == "empty_bowl_in_sink" or re.fullmatch(
        r"(?:pour|empty|drain)_water_from_bowl_(?:in|into)_sink", action.name
    ):
        hand = unary_var(action.pre, {"hand"})
        bowl = unary_var(action.pre, {"bowl"})
        sink = unary_var(action.pre, {"sink"})
        if hand and bowl and sink and any(
            has_literal(action.pre, predicate, bowl)
            for predicate in LIQUID_CARRIER_PREDICATES
        ):
            action.name = "empty_bowl_in_sink"
            action.params = [hand, bowl, sink]
            action.pre = make_and([
                ["hand", hand], ["bowl", bowl], ["sink", sink],
                ["holding", hand, bowl], ["has_water", bowl],
            ])
            action.eff = make_and([
                ["not", ["has_water", bowl]], ["empty", bowl],
            ])
            action.comment = "; Empty bowl ?b in sink ?s with hand ?h."
        return action

    if re.match(r"(?:pour|transfer)_", action.name):
        hand = unary_var(action.pre, {"hand"})
        source_binding = _variable_with_type(action, {"spoon"})
        target_binding = _variable_with_type(action, {"paper_cup", "cup", "bowl"})
        if hand and source_binding and target_binding and any(
            has_literal(action.pre, predicate, source_binding[1])
            for predicate in LIQUID_CARRIER_PREDICATES
        ):
            source_kind, source = source_binding
            target_kind, target = target_binding
            guards, suffix = _liquid_guard_suffix(
                action, source, source_kind, target, target_kind
            )
            action.name = f"pour_spoonful_from_{source_kind}_in_{target_kind}{suffix}"
            action.pre = make_and([
                ["hand", hand], [source_kind, source], [target_kind, target],
                ["holding", hand, source], ["has_water", source], *guards,
            ])
            action.eff = make_and([
                ["not", ["has_water", source]], ["has_water", target],
            ])
            action.params = ordered_used_parameters(action, [hand, source, target])
            action.comment = "; Pour a spoonful into the target cup."
            return action

    if re.match(r"scoop_", action.name):
        hand = unary_var(action.pre, {"hand"})
        spoon = unary_var(action.pre, {"spoon"})
        source_binding = _variable_with_type(
            action, {"kettle", "bowl", "cup", "paper_cup"}
        )
        if hand and spoon and source_binding:
            source_kind, source = source_binding
            if any(
                has_literal(action.pre, predicate, source)
                for predicate in LIQUID_CARRIER_PREDICATES
            ):
                guards, suffix = _liquid_guard_suffix(
                    action, source, source_kind, spoon, "spoon"
                )
                action.name = f"scoop_spoonful_from_{source_kind}_with_spoon{suffix}"
                action.pre = make_and([
                    ["hand", hand], [source_kind, source], ["spoon", spoon],
                    ["holding", hand, spoon], ["has_water", source], *guards,
                ])
                action.eff = make_and([["has_water", spoon]])
                action.params = ordered_used_parameters(action, [hand, source, spoon])
                action.comment = "; Scoop a spoonful from the source."
                return action

    if not re.match(r"(?:pour|transfer)_", action.name):
        return action
    pre_in = [
        literal for literal in conjunction(action.pre)
        if isinstance(literal, list) and len(literal) == 3 and literal[0] == "in"
    ]
    add_in = [
        literal for literal in conjunction(action.eff)
        if isinstance(literal, list) and len(literal) == 3 and literal[0] == "in"
    ]
    transfers = [
        (str(before[1]), str(before[2]), str(after[2]))
        for before in pre_in for after in add_in
        if before[1] == after[1] and before[2] != after[2]
    ]
    if len(set(transfers)) != 1:
        return action
    liquid, source, target = transfers[0]
    hand = unary_var(action.pre, {"hand"})
    liquid_kind = unary_type(action.pre, liquid)
    source_kind = unary_type(action.pre, source)
    target_kind = unary_type(action.pre, target)
    if not all((hand, liquid_kind, source_kind, target_kind)):
        return action
    guards, suffix = _liquid_guard_suffix(
        action, source, source_kind, target, target_kind
    )
    action.name = f"pour_{liquid_kind}_from_{source_kind}_in_{target_kind}{suffix}"
    action.pre = make_and([
        ["hand", hand], [liquid_kind, liquid], [source_kind, source],
        [target_kind, target], ["holding", hand, source],
        ["in", liquid, source], *guards,
    ])
    action.eff = make_and([
        ["not", ["in", liquid, source]], ["in", liquid, target],
    ])
    action.params = ordered_used_parameters(action, [hand, liquid, source, target])
    action.comment = "; Pour finite liquid from the held source into the target."
    return action


def wipe_contract_transform(action: Action) -> Action:
    """Project wiping onto the held tool, affected object, and direct result."""
    if not action.name.startswith(("wipe_", "clean_")):
        return action
    hand = unary_var(action.pre, {"hand"})
    tool_binding = _variable_with_type(action, {"cloth", "sponge", "towel", "rag"})
    results = [
        (str(literal[0]), str(literal[1]))
        for literal in conjunction(action.eff)
        if (
            isinstance(literal, list)
            and len(literal) == 2
            and literal[0] in {"wiped", "cleaned"}
        )
    ]
    if hand is None or tool_binding is None or len(set(results)) != 1:
        return action
    result_predicate, affected = results[0]
    affected_kind = unary_type(action.pre, affected)
    if affected_kind is None:
        return action
    tool_kind, tool = tool_binding
    action.name = (
        f"wipe_{affected_kind}_with_{tool_kind}"
        if result_predicate == "wiped"
        else f"clean_{affected_kind}_with_{tool_kind}"
    )
    action.pre = make_and([
        ["hand", hand], [affected_kind, affected], [tool_kind, tool],
        ["holding", hand, tool],
    ])
    action.eff = make_and([[result_predicate, affected]])
    action.params = ordered_used_parameters(action, [hand, affected, tool])
    action.comment = f"; {action.name.replace('_', ' ').capitalize()}."
    return action


def normalize_liquid_problem_predicates(
    problem_text: str,
    *,
    normalize_spoonful: bool,
    normalize_empty: bool,
) -> str:
    aliases: dict[str, str] = {}
    if normalize_spoonful:
        aliases.update({"has_liquid": "has_water", "received_spoonful": "has_water"})
    if normalize_empty:
        aliases.update({"poured": "empty", "drained": "empty"})
    if not aliases:
        return problem_text
    problem = parse_problem(replace_predicate_heads(problem_text, aliases))
    for section in (":init", ":goal"):
        facts = _unique_nodes(problem_facts(problem, section))
        set_problem_facts(problem, section, facts)
    return render_problem(problem)


def source_dispense_bindings(domain: str) -> dict[str, set[tuple[str, str]]]:
    """Map action names to legacy infinite-source variable bindings."""
    bindings: dict[str, set[tuple[str, str]]] = {}
    for match in ACTION_RE.finditer(domain):
        end = find_matching_paren(domain, match.start()) + 1
        action = Action.parse(domain[match.start():end], "; action")
        source_vars = {
            str(item[1])
            for item in conjunction(action.pre)
            if (
                isinstance(item, list)
                and len(item) == 2
                and item[0] in INFINITE_SOURCE_TYPES
            )
        }
        for expr in (action.pre, action.eff):
            for item in conjunction(expr):
                candidate = item[1] if (
                    isinstance(item, list)
                    and len(item) == 2
                    and item[0] == "not"
                    and isinstance(item[1], list)
                ) else item
                if (
                    isinstance(candidate, list)
                    and len(candidate) == 3
                    and candidate[0] == "in"
                    and candidate[2] in source_vars
                ):
                    bindings.setdefault(action.name, set()).add(
                        (str(candidate[2]), str(candidate[1]))
                    )
    return bindings


def ground_source_dispense_pairs(
    domain: str,
    plan_text: str,
    bindings: dict[str, set[tuple[str, str]]],
) -> set[tuple[str, str]]:
    """Ground legacy source/liquid variables with the demonstrated plan."""
    if not bindings:
        return set()
    schemas = domain_actions(domain)
    pairs: set[tuple[str, str]] = set()
    for line in plan_text.splitlines():
        stripped = line.split(";", 1)[0].strip()
        if not stripped.startswith("("):
            continue
        tokens = stripped.strip("()").split()
        if not tokens or tokens[0] not in bindings:
            continue
        schema = schemas.get(tokens[0])
        if schema is None or len(tokens[1:]) != len(schema.params):
            continue
        grounding = dict(zip(schema.params, tokens[1:]))
        for source_var, liquid_var in bindings[tokens[0]]:
            source = grounding.get(source_var)
            liquid = grounding.get(liquid_var)
            if source and liquid:
                pairs.add((source, liquid))
    return pairs


def threaded_cap_transform(action: Action, tid: int) -> tuple[Action, bool]:
    del tid  # The closure transition, not a task whitelist, determines the contract.
    binding = closure_binding(action)
    if binding is None:
        return action, False
    hand, cap, container, opening = binding
    cap_kind = unary_type(action.pre, cap) or "cap"
    container_kind = unary_type(action.pre, container) or "bottle"
    if not is_threaded_closure(action, cap_kind, container_kind):
        return action, False

    action.params = [hand, cap, container]
    if opening:
        action.name = f"unscrew_cap_from_{container_kind}"
        action.pre = make_and([
            ["hand", hand], [cap_kind, cap], [container_kind, container],
            ["hand_free", hand], ["closed", container], ["on", cap, container],
        ])
        action.eff = make_and([
            ["not", ["hand_free", hand]], ["holding", hand, cap],
            ["not", ["closed", container]], ["open", container],
            ["not", ["on", cap, container]],
        ])
        action.comment = (
            f"; Unscrew cap ?cp from {container_kind.replace('_', ' ')} "
            f"?{container.lstrip('?')} with hand ?h."
        )
    else:
        action.name = f"screw_cap_onto_{container_kind}"
        action.pre = make_and([
            ["hand", hand], [cap_kind, cap], [container_kind, container],
            ["holding", hand, cap], ["open", container],
        ])
        action.eff = make_and([
            ["not", ["holding", hand, cap]], ["hand_free", hand],
            ["not", ["open", container]], ["closed", container],
            ["on", cap, container],
        ])
        action.comment = (
            f"; Screw cap ?cp onto {container_kind.replace('_', ' ')} "
            f"?{container.lstrip('?')} with hand ?h."
        )
    return action, True


def canonical_named_cap_action(action: Action) -> Action:
    """Rebuild an already named screw/unscrew action to one closure contract."""
    match = re.fullmatch(r"(unscrew_cap_from|screw_cap_onto)_(.+)", action.name)
    if match is None:
        return action
    opening = match.group(1) == "unscrew_cap_from"
    container_kind = match.group(2)
    hand = unary_var(action.pre, {"hand"})
    cap = unary_var(action.pre, {"cap"})
    container = unary_var(action.pre, {container_kind})
    if not hand or not cap or not container:
        return action
    action.params = [hand, cap, container]
    if opening:
        action.pre = make_and([
            ["hand", hand], ["cap", cap], [container_kind, container],
            ["hand_free", hand], ["closed", container], ["on", cap, container],
        ])
        action.eff = make_and([
            ["not", ["hand_free", hand]], ["holding", hand, cap],
            ["not", ["closed", container]], ["open", container],
            ["not", ["on", cap, container]],
        ])
        action.comment = (
            f"; Unscrew cap ?cp from {container_kind.replace('_', ' ')} "
            f"?{container.lstrip('?')} with hand ?h."
        )
    else:
        action.pre = make_and([
            ["hand", hand], ["cap", cap], [container_kind, container],
            ["holding", hand, cap], ["open", container],
        ])
        action.eff = make_and([
            ["not", ["holding", hand, cap]], ["hand_free", hand],
            ["not", ["open", container]], ["closed", container],
            ["on", cap, container],
        ])
        action.comment = (
            f"; Screw cap ?cp onto {container_kind.replace('_', ' ')} "
            f"?{container.lstrip('?')} with hand ?h."
        )
    return action


def misc_transform(action: Action, tid: int) -> Action:
    exact = {
        "remove_chair_part_from_drawer": "pick_chair_part_from_drawer",
        "remove_box_from_drawer": "pick_box_from_drawer",
        "remove_cardboard_box_from_drawer": "pick_cardboard_box_from_drawer",
        "place_upright_paper_cup_on_counter": "place_paper_cup_upright_on_counter",
        "place_block_on_block_final": "place_block_on_block",
        "pick_block_from_table_final": "pick_block_from_table",
        "pick_block_from_block_final": "pick_block_from_block",
    }
    if action.name == "place_upright_paper_cup_on_counter":
        action.name = "place_paper_cup_upright_on_counter"
    elif action.name in exact:
        action.name = exact[action.name]
    if tid in {6, 7, 16}:
        cap_surface_names = {
            "place_bottle_cap_on_desk": "place_cap_on_desk",
            "place_lid_on_desk": "place_cap_on_desk",
            "pick_bottle_cap_from_desk": "pick_cap_from_desk",
            "pick_lid_from_desk": "pick_cap_from_desk",
            "place_lid_on_table": "place_cap_on_table",
            "pick_lid_from_table": "pick_cap_from_table",
        }
        if action.name in cap_surface_names:
            action.name = cap_surface_names[action.name]
    if tid == 8 and action.name == "place_bowl_on_rack":
        bowl = unary_var(action.pre, {"bowl"})
        if bowl:
            action.eff = add_literal(action.eff, ["vertical", bowl])
            if has_literal(action.pre, "upright", bowl):
                action.eff = add_literal(action.eff, ["not", ["upright", bowl]])
        action.name = "place_bowl_vertical_on_rack"
        action.comment = "; Place bowl ?b vertical on rack ?r with hand ?h."
    if action.name == "place_book_on_book":
        moving_book = next(
            (
                str(item[2]) for item in conjunction(action.pre)
                if isinstance(item, list) and len(item) == 3 and item[0] == "holding"
            ),
            None,
        )
        if moving_book and has_literal(action.eff, "flat", moving_book):
            action.name = "place_book_flat_on_book"
            action.comment = "; Place book ?b1 flat on book ?b2 with hand ?h."
    if action.name == "place_paper_cup_upright_on_counter":
        cup = unary_var(action.pre, {"paper_cup"})
        if cup:
            action.eff = add_literal(action.eff, ["upright", cup])
    if action.name == "pull_detergent_drawer_from_washing_machine":
        hand = unary_var(action.pre, {"hand"})
        if hand:
            action.eff = remove_literal(action.eff, ["not", ["hand_free", hand]])
    if action.name == "turn_off_microwave":
        for item in list(conjunction(action.eff)):
            if (
                isinstance(item, list)
                and len(item) == 2
                and item[0] == "heated"
                and has_literal(action.eff, "heated", str(item[1]), negative=True)
            ):
                action.eff = remove_literal(action.eff, ["not", item])
    if action.name == "place_paper_cup_upright_on_turntable":
        cup = unary_var(action.pre, {"paper_cup"})
        turntable = unary_var(action.pre, {"turntable"})
        if cup and turntable and has_literal(action.eff, "on", cup, turntable):
            action.eff = remove_literal(action.eff, ["not", ["on", cup, turntable]])
    if action.name.startswith("place_") and "_flat_on_" in action.name:
        item_kind = action.name.removeprefix("place_").split("_flat_on_", 1)[0]
        item = unary_var(action.pre, {item_kind})
        if item:
            action.eff = add_literal(action.eff, ["flat", item])
            if has_literal(action.pre, "vertical", item):
                action.eff = add_literal(action.eff, ["not", ["vertical", item]])
    return action


def _canonical_bowl_stack_action(action: Action) -> Action:
    names = {
        "pick_bowl_from_bowl", "pick_bowl_from_bowl_interior",
        "pick_nested_bowl_from_bowl", "place_bowl_in_bowl",
        "place_bowl_into_bowl", "place_bowl_on_bowl", "place_bowl_onto_bowl",
    }
    if action.name not in names:
        return action

    relation_expr = action.pre if action.name.startswith("pick_") else action.eff
    relation = next(
        (
            item for item in conjunction(relation_expr)
            if (
                isinstance(item, list)
                and len(item) == 3
                and item[0] in {"in", "on"}
                and unary_type(action.pre, str(item[1])) == "bowl"
                and unary_type(action.pre, str(item[2])) == "bowl"
            )
        ),
        None,
    )
    if relation is None:
        return action
    moving, support = str(relation[1]), str(relation[2])
    old_relation = [str(relation[0]), moving, support]
    new_relation = ["on", moving, support]
    action.pre = make_and(
        new_relation if item == old_relation else item
        for item in conjunction(action.pre)
    )
    action.eff = make_and(
        ["not", new_relation]
        if item == ["not", old_relation]
        else new_relation if item == old_relation else item
        for item in conjunction(action.eff)
    )
    if action.name.startswith("pick_"):
        action.name = "pick_bowl_from_bowl"
        action.pre = remove_literal(action.pre, ["upright", moving])
        action.comment = "; Pick bowl ?b1 from bowl ?b2 with hand ?h."
    else:
        action.name = "place_bowl_on_bowl"
        action.comment = "; Place bowl ?b1 on bowl ?b2 with hand ?h."
    return action


def migrate_bowl_stack_contract(domain: str, problem_text: str) -> tuple[str, str]:
    """Migrate the dataset's overlapping-bowl scenes to one on-stack contract."""
    original_domain = domain
    domain, _edits = rewrite_actions(domain, _canonical_bowl_stack_action)
    if domain == original_domain:
        return domain, problem_text

    problem = parse_problem(problem_text)
    bowls = set(typed_objects(problem, "bowl"))
    for section in (":init", ":goal"):
        facts = problem_facts(problem, section)
        migrated: list[Node] = []
        for fact in facts:
            if (
                isinstance(fact, list)
                and len(fact) == 3
                and fact[0] == "in"
                and str(fact[1]) in bowls
                and str(fact[2]) in bowls
            ):
                fact = ["on", fact[1], fact[2]]
            migrated.append(fact)
        set_problem_facts(problem, section, migrated)
    domain = rewrite_predicates(domain, ensure=[["clear", "?o"]])
    problem_text = normalize_family_clear_init(render_problem(problem), "bowl")
    return domain, problem_text


def normalize_problem_spatial_contract(problem_text: str) -> str:
    """Mirror stable in/on category corrections in problem init and goals."""
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    object_types: dict[str, str] = {}
    for fact in init:
        if isinstance(fact, list) and len(fact) == 2:
            # Generated problems conventionally declare object categories
            # before unary state facts (open, closed, clear, and so on).
            object_types.setdefault(str(fact[1]), str(fact[0]))
    changed = False

    for section in (":init", ":goal"):
        rewritten: list[Node] = []
        for fact in problem_facts(problem, section):
            if (
                isinstance(fact, list)
                and len(fact) == 3
                and fact[0] in {"in", "on"}
            ):
                moving_kind = object_types.get(str(fact[1]))
                target_kind = object_types.get(str(fact[2]))
                if moving_kind and target_kind and target_kind != "microwave":
                    relation = _canonical_pick_relation(
                        moving_kind, target_kind, str(fact[0])
                    )
                    if relation != fact[0]:
                        fact = [relation, fact[1], fact[2]]
                        changed = True
            rewritten.append(fact)
        set_problem_facts(problem, section, _unique_nodes(rewritten))
    return render_problem(problem) if changed else problem_text


def normalize_relation_lexemes(action: Action) -> Action:
    """Use in/on consistently in action identifiers and relation predicates."""
    action.name = re.sub(r"(?<=_)into(?=_|$)", "in", action.name)
    action.name = re.sub(r"(?<=_)onto(?=_|$)", "on", action.name)
    aliases = {"into": "in", "onto": "on"}
    action.pre = replace_literal_predicate_heads(action.pre, aliases)
    action.eff = replace_literal_predicate_heads(action.eff, aliases)
    return action


def normalize_spatial_action_contract(action: Action) -> Action:
    """Align concrete place/pick names with their represented support relation."""
    moved = placed_object_var(action) if action.name.startswith("place_") else None
    if moved:
        target_relations = [
            item
            for item in conjunction(action.eff)
            if (
                isinstance(item, list)
                and len(item) == 3
                and item[0] in {"in", "on", "under"}
                and item[1] == moved
            )
        ]
        if action.name in {"place_ladle_in_rack"} and any(
            item[0] == "on" and unary_type(action.pre, str(item[2])) == "rack"
            for item in target_relations
        ):
            action.name = "place_ladle_on_rack"
        if re.fullmatch(r"place_(?:empty_)?bowl(?:_upright)?_in_dish_rack", action.name):
            if any(
                item[0] == "on" and unary_type(action.pre, str(item[2])) == "dish_rack"
                for item in target_relations
            ):
                action.name = action.name.replace("_in_dish_rack", "_on_dish_rack")
        relative_place_specs = {
            "place_block_beside_bowl": (
                "place_block_on_table_beside_bowl", "beside", "bowl",
            ),
            "place_block_left_of_block": (
                "place_block_on_table_left_of_block", "left_of", "block",
            ),
            "place_towel_beside_kettle": (
                "place_towel_on_counter_beside_kettle", "beside", "kettle",
            ),
        }
        spec = relative_place_specs.get(action.name)
        if spec and any(item[0] == "on" for item in target_relations):
            replacement, relative_predicate, reference_kind = spec
            references = {
                str(item[1])
                for item in conjunction(action.pre)
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and item[0] == reference_kind
                    and item[1] != moved
                )
            }
            reference = next(iter(references)) if len(references) == 1 else None
            if reference and has_literal(
                action.eff, relative_predicate, moved, reference
            ):
                action.name = replacement
            elif action.name == "place_block_beside_bowl" and reference:
                # One generated episode names a beside relation that is absent
                # from both the transition and goal.  It is an ordinary table
                # placement; the bowl is scene payload, not an action role.
                action.pre = make_and(
                    literal
                    for literal in conjunction(action.pre)
                    if reference not in node_variables(literal)
                )
                action.name = "place_block_on_table"
                action.params = ordered_used_parameters(action)

    if action.name == "slide_box_on_floor_right_of_cabinet":
        cabinet = unary_var(action.pre, {"cabinet"})
        if cabinet is not None and has_literal(action.pre, "blocked", cabinet):
            action.name += "_when_cabinet_blocked"
    
    # A turntable is a support; direct microwave containment is not a
    # turntable relation even when a legacy action name says otherwise.
    for expression_name in ("pre", "eff"):
        expression = getattr(action, expression_name)
        rewritten: list[Node] = []
        for literal in conjunction(expression):
            negative = (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[0] == "not"
                and isinstance(literal[1], list)
            )
            atom = literal[1] if negative else literal
            if (
                isinstance(atom, list)
                and len(atom) == 3
                and atom[0] == "in"
                and unary_type(action.pre, str(atom[1])) == "paper_cup"
                and unary_type(action.pre, str(atom[2])) in {"turntable", "microwave_turntable"}
            ):
                atom = ["on", atom[1], atom[2]]
                literal = ["not", atom] if negative else atom
            rewritten.append(literal)
        setattr(action, expression_name, make_and(rewritten))

    if "paper_cup" in action.name and "microwave_turntable" in action.name:
        relation_expr = action.pre if action.name.startswith("pick_") else action.eff
        paper_cup = unary_var(action.pre, {"paper_cup"})
        relation = (
            relation_with_variable(relation_expr, {"in", "on"}, paper_cup)
            if paper_cup
            else None
        )
        if relation and unary_type(action.pre, str(relation[2])) == "microwave":
            direct_relation = [str(relation[0]), str(relation[1]), str(relation[2])]
            if direct_relation[0] == "on":
                replacement_relation = ["in", direct_relation[1], direct_relation[2]]
                action.pre = make_and(
                    replacement_relation if item == direct_relation else item
                    for item in conjunction(action.pre)
                )
                action.eff = make_and(
                    ["not", replacement_relation]
                    if item == ["not", direct_relation]
                    else replacement_relation if item == direct_relation else item
                    for item in conjunction(action.eff)
                )
            if action.name.startswith("pick_"):
                action.name = re.sub(
                    r"pick_(?:heated_)?paper_cup_from_microwave_turntable",
                    "pick_paper_cup_from_microwave",
                    action.name,
                )
            elif action.name.startswith("place_"):
                action.name = re.sub(
                    r"place_(?:heated_)?paper_cup(?:_upright)?_on_microwave_turntable",
                    "place_paper_cup_in_microwave",
                    action.name,
                )
    return action


def _drawer_transition_target(action: Action, *, opening: bool) -> str | None:
    old_state, new_state = ("closed", "open") if opening else ("open", "closed")
    candidates = {
        str(item[1])
        for item in conjunction(action.pre)
        if (
            isinstance(item, list) and len(item) == 2 and item[0] == old_state
            and has_literal(action.eff, old_state, str(item[1]), negative=True)
            and has_literal(action.eff, new_state, str(item[1]))
            and unary_type(action.pre, str(item[1])) in {"drawer", "detergent_drawer"}
        )
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _drawer_role(action: Action, variable: str) -> str | None:
    for role in ("top", "middle", "bottom", "lowest"):
        if has_literal(action.pre, f"is_{role}", variable):
            return role
    return None


def drawer_contract_transform(action: Action) -> Action:
    """Expose only the drawer guards actually present in the task-local schema."""
    opening = action.name.startswith("open_") and "drawer" in action.name
    closing = action.name.startswith("close_") and "drawer" in action.name
    if not opening and not closing:
        return action
    target = _drawer_transition_target(action, opening=opening)
    if target is None:
        return action

    if closing:
        hand = unary_var(action.pre, {"hand", "robot_hand"})
        target_kind = unary_type(action.pre, target) or "drawer"
        target_role = _drawer_role(action, target)
        guard_predicate = next(
            (
                predicate for predicate in ("clear_to_close", "unblocked")
                if has_literal(action.pre, predicate, target)
            ),
            None,
        )
        other_closed = [
            str(item[1])
            for item in conjunction(action.pre)
            if (
                isinstance(item, list) and len(item) == 2 and item[0] == "closed"
                and item[1] != target
                and unary_type(action.pre, str(item[1])) == "drawer"
            )
        ]
        retained: list[Node] = []
        if hand:
            retained.extend([["hand", hand]])
        retained.extend([[target_kind, target]])
        if target_role:
            retained.append([f"is_{target_role}", target])
        for variable in other_closed:
            retained.append(["drawer", variable])
            role = _drawer_role(action, variable)
            if role:
                retained.append([f"is_{role}", variable])
        if hand:
            retained.append(["hand_free", hand])
        retained.append(["open", target])
        if guard_predicate:
            retained.append([guard_predicate, target])
        retained.extend(["closed", variable] for variable in other_closed)
        action.pre = make_and(_unique_nodes(retained))
        action.eff = make_and([["not", ["open", target]], ["closed", target]])
        base = (
            f"close_{target_role}_drawer" if target_role
            else f"close_{target_kind}" if target_kind != "drawer"
            else "close_drawer"
        )
        roles = {_drawer_role(action, variable) for variable in other_closed}
        roles.discard(None)
        if roles == {"top", "middle"}:
            suffix = "when_top_and_middle_drawers_closed"
        elif len(roles) == 1 and len(other_closed) == 1:
            suffix = f"when_{next(iter(roles))}_drawer_closed"
        elif other_closed:
            suffix = "when_other_drawer_closed"
        elif guard_predicate:
            suffix = "when_clear_to_close"
        else:
            suffix = ""
        action.name = f"{base}_{suffix}" if suffix else base
        action.params = ordered_used_parameters(action, [hand, target] if hand else [target])
        return action

    unlocked = has_literal(action.pre, "unlocked", target)
    clear_to_open = any(
        has_literal(action.pre, predicate, target)
        for predicate in {"unblocked", "clear_to_open"}
    )
    other_closed = [
        str(item[1])
        for item in conjunction(action.pre)
        if (
            isinstance(item, list) and len(item) == 2 and item[0] == "closed"
            and item[1] != target
            and unary_type(action.pre, str(item[1])) == "drawer"
        )
    ]
    target_role = _drawer_role(action, target)
    target_kind = unary_type(action.pre, target)
    if target_role:
        base = f"open_{target_role}_drawer"
    elif target_kind and target_kind != "drawer":
        base = f"open_{target_kind}"
    else:
        base = "open_drawer"
    roles = [_drawer_role(action, variable) for variable in other_closed]
    role_set = {role for role in roles if role}
    if role_set == {"top", "middle"}:
        suffix = "when_top_and_middle_drawers_closed"
    elif len(role_set) == 1 and len(other_closed) == 1:
        suffix = f"when_{next(iter(role_set))}_drawer_closed"
    elif other_closed:
        suffix = "when_other_drawer_closed"
    elif unlocked and clear_to_open:
        suffix = "when_unlocked_and_clear_to_open"
    elif unlocked:
        suffix = "when_unlocked"
    elif clear_to_open:
        suffix = "when_clear_to_open"
    else:
        suffix = ""
    action.name = f"{base}_{suffix}" if suffix else base
    hand = unary_var(action.pre, {"hand", "robot_hand"})
    retained: list[Node] = []
    if hand:
        retained.append(["hand", hand])
    retained.append([target_kind or "drawer", target])
    if target_role:
        retained.append([f"is_{target_role}", target])
    for variable in other_closed:
        retained.append(["drawer", variable])
        role = _drawer_role(action, variable)
        if role:
            retained.append([f"is_{role}", variable])
    if hand:
        retained.append(["hand_free", hand])
    retained.append(["closed", target])
    if unlocked:
        retained.append(["unlocked", target])
    if clear_to_open:
        clear_predicate = (
            "clear_to_open"
            if has_literal(action.pre, "clear_to_open", target)
            else "unblocked"
        )
        retained.append([clear_predicate, target])
    retained.extend(["closed", variable] for variable in other_closed)
    action.pre = make_and(_unique_nodes(retained))
    action.eff = make_and([["not", ["closed", target]], ["open", target]])
    action.params = ordered_used_parameters(action, [hand, target] if hand else [target])
    return action


def canonical_key_contract(action: Action) -> Action:
    """Canonicalize direct and separate-lock key mechanisms."""
    if "key" not in action.name and not action.name.startswith(("lock_", "unlock_")):
        return action
    hand = unary_var(action.pre, {"hand", "robot_hand"})
    key = unary_var(action.pre, {"key"})
    if hand is None or key is None:
        return action

    def lock_relation(lock: str) -> tuple[str, str, str] | None:
        for literal in conjunction(action.pre):
            if (
                isinstance(literal, list) and len(literal) == 3
                and literal[0] in {"lock_of", "part_of", "lock_for"}
                and literal[1] == lock
            ):
                target = str(literal[2])
                target_kind = unary_type(action.pre, target)
                if target_kind in {"drawer", "cabinet", "filing_cabinet"}:
                    return str(literal[0]), target, target_kind
        return None

    inserted_additions = [
        item for item in conjunction(action.eff)
        if isinstance(item, list) and len(item) == 3
        and item[0] == "inserted" and item[1] == key
    ]
    inserted_preconditions = [
        item for item in conjunction(action.pre)
        if isinstance(item, list) and len(item) == 3
        and item[0] == "inserted" and item[1] == key
    ]
    for relation in [*inserted_additions, *inserted_preconditions]:
        target = str(relation[2])
        if unary_type(action.pre, target) is None and "drawer" in action.name:
            action.pre = add_literal(action.pre, ["drawer", target])

    if len(inserted_additions) == 1:
        target = str(inserted_additions[0][2])
        target_kind = unary_type(action.pre, target)
        if target_kind not in {"drawer", "cabinet", "filing_cabinet", "drawer_lock", "lock"}:
            return action
        relation = lock_relation(target)
        pre: list[Node] = [
            ["hand", hand], ["key", key], [target_kind, target],
            ["holding", hand, key],
        ]
        preferred = [hand, key, target]
        target_label = target_kind
        guard_target = target
        guard_kind = target_kind
        if relation is not None:
            predicate, owner, owner_kind = relation
            pre.extend([[owner_kind, owner], [predicate, target, owner]])
            preferred.append(owner)
            target_label = f"{target_kind}_for_{owner_kind}"
            guard_target, guard_kind = owner, owner_kind
        guards: list[str] = []
        for predicate in ("closed", "locked", "unlocked"):
            if has_literal(action.pre, predicate, guard_target):
                pre.append([predicate, guard_target])
                guards.append(f"{guard_kind}_{predicate}")
        keeps_holding = not has_literal(action.eff, "holding", hand, key, negative=True)
        action.name = f"insert_{'held_' if keeps_holding else ''}key_in_{target_label}"
        if guards:
            action.name += "_when_" + "_and_".join(guards)
        action.pre = make_and(pre)
        action.eff = make_and(
            [["inserted", key, target]]
            if keeps_holding
            else [
                ["not", ["holding", hand, key]], ["hand_free", hand],
                ["inserted", key, target],
            ]
        )
        action.params = ordered_used_parameters(action, preferred)
        action.comment = "; Insert the held key into its lock target."
        return action

    state_results: list[tuple[str, str, str]] = []
    for before, after in (("unlocked", "locked"), ("locked", "unlocked")):
        for literal in conjunction(action.eff):
            if (
                isinstance(literal, list) and len(literal) == 2
                and literal[0] == after
                and has_literal(action.eff, before, str(literal[1]), negative=True)
            ):
                state_results.append((before, after, str(literal[1])))
    if len(state_results) == 1 and len(inserted_preconditions) == 1:
        before, after, owner = state_results[0]
        owner_kind = unary_type(action.pre, owner)
        inserted_target = str(inserted_preconditions[0][2])
        inserted_kind = unary_type(action.pre, inserted_target)
        if owner_kind not in {"drawer", "cabinet", "filing_cabinet"} or inserted_kind is None:
            return action
        keeps_holding = has_literal(action.pre, "holding", hand, key)
        pre = [
            ["hand", hand], ["key", key], [owner_kind, owner],
            ["holding" if keeps_holding else "hand_free", hand, key]
            if keeps_holding else ["hand_free", hand],
            [before, owner], ["inserted", key, inserted_target],
        ]
        preferred = [hand, owner, key, inserted_target]
        mechanism = ""
        if inserted_target == owner:
            pass
        elif inserted_kind in {"drawer_lock", "lock"}:
            relation = lock_relation(inserted_target)
            pre.append([inserted_kind, inserted_target])
            if relation is not None and relation[1] == owner:
                pre.append([relation[0], inserted_target, owner])
            mechanism = f"_in_{inserted_kind}"
            if relation is None:
                mechanism += "_without_owner_relation"
        else:
            return action
        guards: list[str] = []
        if has_literal(action.pre, "closed", owner):
            pre.append(["closed", owner])
            guards.append(f"{owner_kind}_closed")
        key_label = "held_key" if keeps_holding else "key"
        action.name = f"{'lock' if after == 'locked' else 'unlock'}_{owner_kind}_with_{key_label}{mechanism}"
        if guards:
            action.name += "_when_" + "_and_".join(guards)
        action.pre = make_and(_unique_nodes(pre))
        action.eff = make_and([["not", [before, owner]], [after, owner]])
        action.params = ordered_used_parameters(action, preferred)
        action.comment = f"; {after.capitalize()} the {owner_kind.replace('_', ' ')} with the key."
        return action

    inserted_deletions = [
        item[1] for item in conjunction(action.eff)
        if (
            isinstance(item, list) and len(item) == 2 and item[0] == "not"
            and isinstance(item[1], list) and len(item[1]) == 3
            and item[1][0] == "inserted" and item[1][1] == key
        )
    ]
    relation = inserted_preconditions[0] if len(inserted_preconditions) == 1 else None
    if relation is None and len(inserted_deletions) == 1:
        relation = inserted_deletions[0]
    if relation is None:
        return action
    target = str(relation[2])
    target_kind = unary_type(action.pre, target)
    if target_kind not in {"drawer", "cabinet", "filing_cabinet", "drawer_lock", "lock"}:
        return action
    structural = lock_relation(target)
    keeps_holding = has_literal(action.pre, "holding", hand, key)
    pre = [
        ["hand", hand], ["key", key], [target_kind, target],
        ["holding", hand, key] if keeps_holding else ["hand_free", hand],
        ["inserted", key, target],
    ]
    preferred = [hand, key, target]
    target_label = target_kind
    guard_target, guard_kind = target, target_kind
    if structural is not None:
        predicate, owner, owner_kind = structural
        pre.extend([[owner_kind, owner], [predicate, target, owner]])
        preferred.append(owner)
        target_label = f"{target_kind}_for_{owner_kind}"
        guard_target, guard_kind = owner, owner_kind
    guards: list[str] = []
    for predicate in ("closed", "locked", "unlocked"):
        if has_literal(action.pre, predicate, guard_target):
            pre.append([predicate, guard_target])
            guards.append(f"{guard_kind}_{predicate}")
    action.name = f"remove_{'held_' if keeps_holding else ''}key_from_{target_label}"
    if guards:
        action.name += "_when_" + "_and_".join(guards)
    action.pre = make_and(pre)
    action.eff = make_and(
        [["not", ["inserted", key, target]]]
        if keeps_holding
        else [
            ["not", ["hand_free", hand]], ["holding", hand, key],
            ["not", ["inserted", key, target]],
        ]
    )
    action.params = ordered_used_parameters(action, preferred)
    action.comment = "; Remove the key from its lock target."
    return action


def normalize_drawer_clearance_predicate(
    domain: str,
    problem_text: str,
) -> tuple[str, str]:
    """Name a task-local unblocked state by the transition that consumes it."""
    consumers: set[str] = set()
    for action in domain_actions(domain).values():
        if not any(
            isinstance(item, list) and len(item) == 2 and item[0] == "unblocked"
            for item in conjunction(action.pre)
        ):
            continue
        if action.name.startswith("open_") and "drawer" in action.name:
            consumers.add("clear_to_open")
        elif action.name.startswith("close_") and "drawer" in action.name:
            consumers.add("clear_to_close")
    if len(consumers) != 1:
        return domain, problem_text
    replacement = next(iter(consumers))
    aliases = {"unblocked": replacement, "unobstructed": replacement}
    return (
        replace_predicate_heads(domain, aliases),
        replace_predicate_heads(problem_text, aliases),
    )


def normalize_processed_manipulation(action: Action) -> Action:
    """Keep ordinary pick/place independent of an object's process history."""
    if not action.name.startswith(("pick_", "place_")):
        return action
    heated_variables = {
        str(item[1])
        for item in conjunction(action.pre)
        if isinstance(item, list) and len(item) == 2 and item[0] == "heated"
    }
    if not heated_variables and "heated" not in action.name:
        return action

    action.name = re.sub(r"(?<=_)heated(?=_|$)", "", action.name)
    action.name = re.sub(r"__+", "_", action.name)
    filtered_pre: list[Node] = []
    for literal in conjunction(action.pre):
        variables = node_variables(literal)
        if (
            isinstance(literal, list)
            and len(literal) == 2
            and literal[0] == "heated"
        ):
            continue
        if variables & heated_variables:
            # If the processed item is itself the manipulated object, retain
            # its ordinary type and spatial literals; only the state guard goes.
            manipulated = (
                next(iter({str(item[2]) for item in conjunction(action.eff)
                           if isinstance(item, list) and len(item) == 3 and item[0] == "holding"}), None)
                if action.name.startswith("pick_")
                else placed_object_var(action)
            )
            if manipulated not in heated_variables:
                continue
        filtered_pre.append(literal)
    action.pre = make_and(filtered_pre)
    action.eff = remove_literals_with_predicates(action.eff, {"heated"})
    action.params = ordered_used_parameters(action)
    return action


TASK_275_BOOKKEEPING_PREDICATES = {
    "phase",
    "storage_complete",
    "stack_complete",
    "assembly_complete",
    "stacking_ready",
    "stacking_allowed",
    "green_stored",
}


def normalize_task_275_bookkeeping(
    domain: str,
    problem_text: str,
    plan_text: str,
) -> tuple[str, str, str]:
    """Remove generated phase flags that merely serialize task 275's plan."""
    original_actions = domain_actions(domain)

    def transform(action: Action) -> Action:
        if action.name == "place_green_block_in_drawer":
            action.name = "place_block_in_drawer"
            action.comment = "; Place block ?b in drawer ?d with hand ?h."
        action.pre = remove_literals_with_predicates(
            action.pre, TASK_275_BOOKKEEPING_PREDICATES
        )
        action.eff = remove_literals_with_predicates(
            action.eff, TASK_275_BOOKKEEPING_PREDICATES
        )
        action.params = ordered_used_parameters(action)
        return action

    domain, edits = rewrite_actions(domain, transform)
    plan_text = rewrite_plan(plan_text, edits)
    domain = rewrite_predicates(
        domain, remove_names=TASK_275_BOOKKEEPING_PREDICATES
    )

    problem = parse_problem(problem_text)
    bookkeeping_objects = {
        str(argument)
        for section in (":init", ":goal")
        for fact in problem_facts(problem, section)
        if (
            isinstance(fact, list)
            and fact
            and fact[0] in TASK_275_BOOKKEEPING_PREDICATES
        )
        for argument in fact[1:]
    }
    for section in (":init", ":goal"):
        set_problem_facts(
            problem,
            section,
            [
                fact
                for fact in problem_facts(problem, section)
                if not (
                    isinstance(fact, list)
                    and fact
                    and fact[0] in TASK_275_BOOKKEEPING_PREDICATES
                )
            ],
        )

    still_referenced = {
        str(argument)
        for section in (":init", ":goal")
        for fact in problem_facts(problem, section)
        if isinstance(fact, list)
        for argument in fact[1:]
    }
    plan_objects = {
        token
        for line in plan_text.splitlines()
        if line.split(";", 1)[0].strip().startswith("(")
        for token in line.split(";", 1)[0].strip().strip("()").split()[1:]
    }
    for obj in bookkeeping_objects - still_referenced - plan_objects:
        remove_object(problem, obj)

    # Every old schema must either survive under its name or have an explicit
    # plan projection. This guards against silently leaving legacy arguments.
    for name in original_actions:
        if name not in domain_actions(domain) and name not in edits:
            raise ValueError(f"task 275 action {name} disappeared without a plan edit")
    return domain, render_problem(problem), plan_text


def stacking_clear_repair_families(domain: str) -> set[str]:
    """Return stack families whose on-stack schemas omit the clear contract."""
    families: set[str] = set()
    actions: list[Action] = []
    for match in ACTION_RE.finditer(domain):
        end = find_matching_paren(domain, match.start()) + 1
        actions.append(Action.parse(domain[match.start():end], "; action"))
    for action in actions:
        operation = (
            "pick" if action.name.startswith("pick_")
            else "place" if action.name.startswith("place_")
            else None
        )
        if operation is None:
            continue
        for family in ("block", "book", "bowl"):
            moving = (
                next(
                    (
                        str(item[2])
                        for item in conjunction(action.eff)
                        if (
                            isinstance(item, list)
                            and len(item) == 3
                            and item[0] == "holding"
                            and unary_type(action.pre, str(item[2])) == family
                        )
                    ),
                    None,
                )
                if operation == "pick"
                else placed_object_var(action)
            )
            if not moving or unary_type(action.pre, moving) != family:
                continue
            relation_expr = action.pre if operation == "pick" else action.eff
            relation = relation_with_variable(relation_expr, {"on"}, moving)
            if (
                relation is None
                or len(relation) != 3
                or unary_type(action.pre, relation[2]) != family
            ):
                continue
            support = relation[2]
            canonical = (
                has_literal(action.pre, "clear", moving)
                and has_literal(action.eff, "clear", moving, negative=True)
                and has_literal(action.eff, "clear", support)
                if operation == "pick"
                else has_literal(action.pre, "clear", support)
                and has_literal(action.eff, "clear", support, negative=True)
                and has_literal(action.eff, "clear", moving)
            )
            if not canonical:
                families.add(family)
    return families


def canonical_family_clear_action(action: Action, family: str) -> Action:
    """Repair the physical clear interface throughout one stacking domain."""
    if action.name.startswith(f"pick_{family}_"):
        hand = unary_var(action.pre, {"hand", "robot_hand"})
        moving = next(
            (
                str(item[2])
                for item in conjunction(action.eff)
                if isinstance(item, list)
                and len(item) == 3
                and item[0] == "holding"
                and (hand is None or item[1] == hand)
                and unary_type(action.pre, str(item[2])) == family
            ),
            None,
        )
        relation = relation_with_variable(action.pre, {"on"}, moving) if moving else None
        if (
            moving
            and relation is not None
            and len(relation) == 3
            and unary_type(action.pre, relation[2]) == family
        ):
            action.pre = add_literal(action.pre, ["clear", moving])
            action.eff = add_literal(action.eff, ["not", ["clear", moving]])
            action.eff = add_literal(action.eff, ["clear", relation[2]])
        return action

    if action.name.startswith(f"place_{family}_"):
        moving = placed_object_var(action)
        relation = relation_with_variable(action.eff, {"on"}, moving) if moving else None
        if (
            moving
            and unary_type(action.pre, moving) == family
            and relation is not None
            and len(relation) == 3
            and unary_type(action.pre, relation[2]) == family
        ):
            support = relation[2]
            action.eff = add_literal(action.eff, ["clear", moving])
            action.pre = add_literal(action.pre, ["clear", support])
            action.eff = add_literal(action.eff, ["not", ["clear", support]])
        elif moving and unary_type(action.pre, moving) == family:
            relation = relation_with_variable(action.eff, {"on"}, moving)
            if (
                relation is not None
                and unary_type(action.pre, relation[2]) in CLEAR_TOP_SURFACE_TYPES
            ):
                action.eff = add_literal(action.eff, ["clear", moving])
    return action


def normalize_family_clear_init(problem_text: str, family: str) -> str:
    """Derive initial clear facts from the represented on graph."""
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    objects = {
        str(item[1])
        for item in init
        if isinstance(item, list) and len(item) == 2 and item[0] == family
    }
    if not objects:
        return problem_text
    covered = {
        str(item[2])
        for item in init
        if isinstance(item, list) and len(item) == 3 and item[0] == "on"
    }
    held = {
        str(item[2])
        for item in init
        if isinstance(item, list) and len(item) == 3 and item[0] == "holding"
    }
    expected = objects - covered - held
    updated = [
        item
        for item in init
        if not (
            isinstance(item, list)
            and len(item) == 2
            and item[0] == "clear"
            and str(item[1]) in objects
        )
    ]
    updated.extend(["clear", item] for item in sorted(expected))
    if updated == init:
        return problem_text
    set_problem_facts(problem, ":init", updated)
    return render_problem(problem)


def canonical_task_275_episode_98_plan(domain: str, plan_text: str) -> str:
    """Follow the demonstrated unstack-store-restack order for this outlier."""
    expected_objects = {
        "hand_single", "yellow_block", "red_block", "green_block",
        "purple_block", "blue_block", "orange_block", "table_main",
        "black_box_drawer",
    }
    if not expected_objects <= set(plan_text.replace("(", " ").replace(")", " ").split()):
        raise ValueError("task 275 episode 98 plan objects do not match the audited scene")
    schemas = domain_actions(domain)

    def resolve(base: str, pattern: str) -> str:
        if base in schemas:
            return base
        candidates = sorted(name for name in schemas if re.fullmatch(pattern, name))
        if len(candidates) != 1:
            raise ValueError(
                f"cannot uniquely resolve {base} in task 275 episode 98: {candidates}"
            )
        return candidates[0]

    place_on_table = resolve(
        "place_block_on_table", r"place_block(?:_[a-z0-9]+)*_on_table(?:_when_[a-z0-9_]+)?"
    )
    place_in_drawer = resolve(
        "place_block_in_drawer", r"place_block(?:_[a-z0-9]+)*_in_drawer(?:_when_[a-z0-9_]+)?"
    )
    pick_from_table = resolve(
        "pick_block_from_table", r"pick_block_from_table(?:_when_[a-z0-9_]+)?"
    )
    steps = [
        f"({place_on_table} hand_single yellow_block table_main)",
        "(pick_block_from_block hand_single red_block green_block)",
        f"({place_on_table} hand_single red_block table_main)",
        "(open_drawer hand_single black_box_drawer)",
        "(pick_block_from_block hand_single green_block purple_block)",
        f"({place_in_drawer} hand_single green_block black_box_drawer)",
        f"({pick_from_table} hand_single purple_block table_main)",
        f"({place_in_drawer} hand_single purple_block black_box_drawer)",
        f"({pick_from_table} hand_single red_block table_main)",
        "(place_block_on_block hand_single red_block blue_block)",
        f"({pick_from_table} hand_single orange_block table_main)",
        "(place_block_on_block hand_single orange_block red_block)",
        f"({pick_from_table} hand_single yellow_block table_main)",
        "(place_block_on_block hand_single yellow_block orange_block)",
        "(close_drawer hand_single black_box_drawer)",
    ]
    return "\n".join([*steps, f"; cost = {len(steps)} (unit cost)"]) + "\n"


def normalize_task_60(domain: str, problem_text: str) -> tuple[str, str]:
    domain = rewrite_predicates(domain, ensure=[["open", "?b"], ["closed", "?b"]])

    def transform(action: Action) -> Action:
        box = unary_var(action.pre, {"box"})
        if action.name == "place_apple_in_box" and box:
            action.pre = add_literal(action.pre, ["open", box])
        if action.name == "place_lid_on_box" and box:
            lid = unary_var(action.pre, {"lid"})
            hand = unary_var(action.pre, {"hand"})
            if lid and hand:
                action.params = [hand, lid, box]
                action.pre = make_and([
                    ["hand", hand], ["lid", lid], ["box", box],
                    ["holding", hand, lid], ["open", box],
                ])
                action.eff = make_and([
                    ["not", ["holding", hand, lid]], ["hand_free", hand],
                    ["not", ["open", box]], ["closed", box], ["on", lid, box],
                ])
        return action

    domain, _ = rewrite_actions(domain, transform)
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    goals = problem_facts(problem, ":goal")
    boxes = typed_objects(problem, "box")
    lids = typed_objects(problem, "lid")
    for box in boxes:
        covered = any(isinstance(f, list) and f == ["on", lid, box] for f in init for lid in lids)
        state = ["closed", box] if covered else ["open", box]
        opposite = ["open", box] if covered else ["closed", box]
        init = [fact for fact in init if fact != opposite]
        if state not in init:
            init.append(state)
        if any(isinstance(f, list) and f == ["on", lid, box] for f in goals for lid in lids):
            if ["closed", box] not in goals:
                goals.append(["closed", box])
    set_problem_facts(problem, ":init", init)
    set_problem_facts(problem, ":goal", goals)
    return domain, render_problem(problem)


def normalize_early_pick_pose_contract(
    domain: str,
    problem_text: str,
    plan_text: str,
    tid: int,
) -> tuple[str, str, str, bool]:
    """Move four audited pickup pose changes to their physical source or place."""
    if tid not in {60, 114, 254, 257}:
        return domain, problem_text, plan_text, False

    vertical_pick_vars: dict[str, str] = {}

    def held_object(action: Action) -> str | None:
        held = {
            str(literal[2])
            for literal in conjunction(action.eff)
            if (
                isinstance(literal, list)
                and len(literal) == 3
                and literal[0] == "holding"
            )
        }
        return next(iter(held)) if len(held) == 1 else None

    def transform(action: Action) -> Action:
        if tid == 60 and action.name == "pick_box_from_table":
            box = held_object(action)
            if box:
                action.eff = remove_literal(action.eff, ["not", ["on_side", box]])
                action.comment = (
                    f"; Pick box {box} from table without changing its pose."
                )
            return action

        if tid == 60 and action.name in {
            "place_box_on_table", "place_box_upright_on_table",
        }:
            box = placed_object_var(action)
            if box:
                action.name = "place_box_upright_on_table"
                action.eff = add_literal(action.eff, ["upright", box])
                action.eff = add_literal(action.eff, ["not", ["on_side", box]])
                action.comment = f"; Place box {box} upright on table with hand ?h."
            return action

        if tid == 114 and action.name == "pick_lid_from_counter":
            lid = held_object(action)
            if lid:
                action.eff = remove_literal(action.eff, ["not", ["upright", lid]])
                action.comment = (
                    f"; Pick lid {lid} from counter without changing its pose."
                )
            return action

        if tid == 114 and action.name == "place_lid_on_pot":
            lid = placed_object_var(action)
            if lid:
                # Covering the pot consumes the temporary counter pose.  This
                # is part of the concrete lid placement, not a pickup effect.
                action.eff = add_literal(action.eff, ["not", ["upright", lid]])
            return action

        if tid in {254, 257} and action.name.startswith("pick_"):
            item = held_object(action)
            if item and has_literal(action.eff, "vertical", item):
                action.eff = remove_literal(action.eff, ["vertical", item])
                action.pre = add_literal(action.pre, ["vertical", item])
                vertical_pick_vars[action.name] = item
                action.comment = (
                    f"; Pick {unary_type(action.pre, item) or 'item'} {item} "
                    "from its source without changing its pose."
                )
        return action

    original_domain = domain
    domain, edits = rewrite_actions(domain, transform)
    plan_text = rewrite_plan(plan_text, edits)

    problem_changed = False
    if vertical_pick_vars:
        schemas = domain_actions(domain)
        problem = parse_problem(problem_text)
        init = problem_facts(problem, ":init")
        for line in plan_text.splitlines():
            stripped = line.split(";", 1)[0].strip()
            if not stripped.startswith("("):
                continue
            tokens = stripped.strip("()").split()
            if not tokens or tokens[0] not in vertical_pick_vars:
                continue
            schema = schemas[tokens[0]]
            if len(tokens[1:]) != len(schema.params):
                raise ValueError(
                    f"cannot ground audited pose source for {tokens[0]}"
                )
            grounding = dict(zip(schema.params, tokens[1:]))
            item = grounding[vertical_pick_vars[tokens[0]]]
            fact: Node = ["vertical", item]
            if fact not in init:
                init.append(fact)
                problem_changed = True
        if problem_changed:
            set_problem_facts(problem, ":init", init)
            problem_text = render_problem(problem)

    return domain, problem_text, plan_text, domain != original_domain or problem_changed


def normalize_laptop_task_88(domain: str, problem_text: str, plan_text: str) -> tuple[str, str, str]:
    domain = replace_token(domain, "laptop_lid", "laptop")
    problem_text = replace_token(problem_text, "laptop_lid", "laptop")
    plan_text = replace_token(plan_text, "laptop_lid", "laptop")
    return domain, problem_text, plan_text


def normalize_infinite_sources(
    domain: str,
    problem_text: str,
    plan_text: str,
    inferred_pairs: set[tuple[str, str]] | None = None,
) -> tuple[str, str, bool]:
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    source_objects = {
        str(fact[1])
        for fact in init
        if isinstance(fact, list) and len(fact) == 2 and fact[0] in INFINITE_SOURCE_TYPES
    }
    changed = False
    inferred_pairs = inferred_pairs or set()
    migrated_liquids = {liquid for _source, liquid in inferred_pairs}
    required_in_facts: set[tuple[str, ...]] = set()
    schemas = domain_actions(domain)
    for line in plan_text.splitlines():
        stripped = line.split(";", 1)[0].strip()
        tokens = stripped.strip("()").split() if stripped.startswith("(") else []
        schema = schemas.get(tokens[0]) if tokens else None
        if schema is None or len(tokens[1:]) != len(schema.params):
            continue
        grounded = ground_action(schema, tokens[1:])
        required_in_facts.update(
            literal for literal in grounded.positive_preconditions
            if literal and literal[0] == "in"
        )
    new_init: list[Node] = []
    for fact in init:
        if (
            isinstance(fact, list) and len(fact) == 3 and fact[0] == "in"
            and (
                fact[2] in source_objects
                or (
                    fact[1] in migrated_liquids
                    and tuple(str(token) for token in fact) not in required_in_facts
                )
            )
        ):
            if fact[2] in source_objects:
                replacement: Node = ["dispenses", fact[2], fact[1]]
                if replacement not in new_init:
                    new_init.append(replacement)
            changed = True
        else:
            new_init.append(fact)
    object_set = {str(item) for item in problem_section(problem, ":objects")[1:]}
    for source, liquid in inferred_pairs:
        if source not in object_set or liquid not in object_set:
            continue
        relation: Node = ["dispenses", source, liquid]
        if relation not in new_init:
            new_init.append(relation)
            changed = True
    if changed:
        set_problem_facts(problem, ":init", new_init)
        domain = rewrite_predicates(domain, ensure=[["dispenses", "?s", "?l"]])
    return domain, render_problem(problem) if changed else problem_text, changed


def split_refill_liquid_identity(
    domain: str,
    problem_text: str,
    plan_text: str,
) -> tuple[str, str, str]:
    """Split a reused refill token into old cup water and fresh source water."""
    schemas = domain_actions(domain)
    fill_actions = {
        name: action for name, action in schemas.items()
        if "fill_" in name and "_from_faucet" in name
    }
    if not fill_actions:
        return domain, problem_text, plan_text

    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    dispensed = {
        (str(fact[1]), str(fact[2]))
        for fact in init
        if isinstance(fact, list) and len(fact) == 3 and fact[0] == "dispenses"
    }
    occupied = {
        str(fact[1])
        for fact in init
        if isinstance(fact, list) and len(fact) == 3 and fact[0] == "in"
    }
    reused = sorted({liquid for _source, liquid in dispensed} & occupied)
    if not reused:
        return domain, problem_text, plan_text

    objects = {str(item) for item in problem_section(problem, ":objects")[1:]}
    replacements: dict[str, str] = {}
    for liquid in reused:
        candidate = f"fresh_{liquid}"
        suffix = 2
        while candidate in objects:
            candidate = f"fresh_{liquid}_{suffix}"
            suffix += 1
        replacements[liquid] = candidate
        add_object(problem, candidate)
        objects.add(candidate)

    new_init: list[Node] = []
    for fact in init:
        if (
            isinstance(fact, list)
            and len(fact) == 3
            and fact[0] == "dispenses"
            and str(fact[2]) in replacements
        ):
            new_init.append(["dispenses", fact[1], replacements[str(fact[2])]])
        else:
            new_init.append(fact)
    for old, new in replacements.items():
        type_facts = [
            fact for fact in init
            if isinstance(fact, list) and len(fact) == 2 and fact[1] == old
        ]
        for fact in type_facts:
            replacement: Node = [fact[0], new]
            if replacement not in new_init:
                new_init.append(replacement)
    set_problem_facts(problem, ":init", new_init)

    output: list[str] = []
    refill_destinations: dict[tuple[str, str], str] = {}
    for line in plan_text.splitlines():
        stripped = line.split(";", 1)[0].strip()
        tokens = stripped.strip("()").split() if stripped.startswith("(") else []
        schema = schemas.get(tokens[0]) if tokens else None
        fill_schema = fill_actions.get(tokens[0]) if tokens else None
        if fill_schema is not None and len(tokens[1:]) == len(fill_schema.params):
            schema = fill_schema
            grounding = dict(zip(schema.params, tokens[1:]))
            dispenses_literals = [
                literal for literal in conjunction(schema.pre)
                if isinstance(literal, list) and len(literal) == 3 and literal[0] == "dispenses"
            ]
            for literal in dispenses_literals:
                liquid_var = str(literal[2])
                old = grounding.get(liquid_var)
                if old not in replacements:
                    continue
                parameter_index = schema.params.index(liquid_var) + 1
                tokens[parameter_index] = replacements[old]
                add_effects, _delete_effects = split_literals(schema.eff)
                for effect in add_effects:
                    if len(effect) != 3 or effect[0] != "in" or effect[1] != liquid_var:
                        continue
                    target = grounding.get(effect[2], effect[2])
                    refill_destinations[(old, target)] = replacements[old]
            line = "(" + " ".join(tokens) + ")"
        elif schema is not None and len(tokens[1:]) == len(schema.params):
            grounding = dict(zip(schema.params, tokens[1:]))
            positive_pre, _negative_pre = split_literals(schema.pre)
            for literal in positive_pre:
                if len(literal) != 3 or literal[0] != "in":
                    continue
                old = grounding.get(literal[1], literal[1])
                target = grounding.get(literal[2], literal[2])
                fresh = refill_destinations.get((old, target))
                if fresh is None or literal[1] not in schema.params:
                    continue
                parameter_index = schema.params.index(literal[1]) + 1
                tokens[parameter_index] = fresh
                grounding[literal[1]] = fresh
            line = "(" + " ".join(tokens) + ")"
        if not line.strip().startswith("; cost ="):
            output.append(line)

    goals = problem_facts(problem, ":goal")
    rewritten_goals: list[Node] = []
    for fact in goals:
        if isinstance(fact, list) and len(fact) == 3 and fact[0] == "in":
            fresh = refill_destinations.get((str(fact[1]), str(fact[2])))
            if fresh is not None:
                fact = ["in", fresh, fact[2]]
        rewritten_goals.append(fact)
    set_problem_facts(problem, ":goal", rewritten_goals)
    count = sum(line.strip().startswith("(") for line in output)
    output.append(f"; cost = {count} (unit cost)")
    return domain, render_problem(problem), "\n".join(output) + "\n"


def remove_dead_microwave_buttons(domain: str, problem_text: str) -> tuple[str, str]:
    """Remove microwave control objects after controls become internal details."""
    _start, _end, declarations = predicate_block(domain)
    control_types = {
        str(item[0])
        for item in declarations
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and (
                "button" in item[0]
                or item[0] in {"knob", "timer_knob", "microwave_knob"}
            )
        )
    }
    if not control_types:
        return domain, problem_text

    used_predicates: set[str] = set()
    for action in domain_actions(domain).values():
        for literal in [*conjunction(action.pre), *conjunction(action.eff)]:
            candidate = literal[1] if (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[0] == "not"
                and isinstance(literal[1], list)
            ) else literal
            if isinstance(candidate, list) and candidate:
                used_predicates.add(str(candidate[0]))
    dead_types = control_types - used_predicates
    if not dead_types:
        return domain, problem_text

    problem = parse_problem(problem_text)
    dead_objects = {
        obj for predicate in dead_types for obj in typed_objects(problem, predicate)
    }
    goals = problem_facts(problem, ":goal")
    goal_references = {
        obj for obj in dead_objects
        if any(isinstance(fact, list) and obj in fact[1:] for fact in goals)
    }
    if goal_references:
        raise ValueError(
            f"dead microwave controls remain goal-relevant: {sorted(goal_references)}"
        )
    init = [
        fact for fact in problem_facts(problem, ":init")
        if not (
            isinstance(fact, list)
            and (
                (fact and fact[0] in dead_types)
                or any(obj in fact[1:] for obj in dead_objects)
            )
        )
    ]
    for obj in dead_objects:
        remove_object(problem, obj)
    set_problem_facts(problem, ":init", init)
    return rewrite_predicates(domain, remove_names=dead_types), render_problem(problem)


def canonical_key_set_round(
    domain: str,
    problem_text: str,
    plan_text: str,
    *,
    drop_leading_pick: bool = False,
) -> tuple[str, str, str]:
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    goals = problem_facts(problem, ":goal")
    hands = typed_objects(problem, "hand")
    sets = typed_objects(problem, "key_set")
    if not sets:
        key_candidates = typed_objects(problem, "key")
        inferred_set = next(
            (
                candidate for candidate in key_candidates
                if "set" in candidate
            ),
            None,
        )
        if inferred_set is None and len(key_candidates) == 1:
            candidate = key_candidates[0]
            if any(
                isinstance(fact, list)
                and len(fact) == 3
                and fact[0] == "holding"
                and fact[2] == candidate
                for fact in init
            ):
                inferred_set = candidate
        if inferred_set:
            sets = [inferred_set]
    drawers = typed_objects(problem, "drawer")
    cabinets = typed_objects(problem, "filing_cabinet") or typed_objects(problem, "cabinet")
    keys = [obj for obj in typed_objects(problem, "key") if obj not in sets]
    if not hands or not sets or not drawers or not cabinets:
        raise ValueError("cannot ground key-set task")
    hand, key_set, drawer, cabinet = hands[0], sets[0], drawers[0], cabinets[0]
    key = keys[0] if keys else (
        "individual_key"
    )
    if sets and sets[0] == "top_drawer_key_set":
        key = "top_drawer_key"
    add_object(problem, key)
    if sets:
        add_object(problem, sets[0])
    if cabinets:
        add_object(problem, cabinets[0])

    # Normalize cabinet category and rebuild key-related init facts.
    def remap_fact(fact: Node) -> Node:
        if not isinstance(fact, list):
            return fact
        mapped = ["filing_cabinet" if item == "cabinet" else item for item in fact]
        if len(mapped) == 2 and mapped[0] == "key" and mapped[1] == key_set:
            mapped[1] = key
        return mapped

    init = [remap_fact(fact) for fact in init]
    goals = [remap_fact(fact) for fact in goals]
    structural_facts = [
        fact for fact in init
        if isinstance(fact, list) and fact and fact[0] in {"lock", "lock_of", "lock_for"}
    ]
    init = [
        fact for fact in init
        if not (
            isinstance(fact, list)
            and fact
            and fact[0] in {"in", "contains", "inserted", "holding", "hand_free", "key", "key_set"}
        )
    ]
    init.extend([["key_set", key_set], ["key", key], ["contains", key_set, key]])
    init = [fact for fact in init if fact not in structural_facts]

    if drop_leading_pick:
        output_lines: list[str] = []
        dropped = False
        for line in plan_text.splitlines():
            stripped = line.strip()
            if not dropped and stripped.startswith("(pick_key"):
                dropped = True
                continue
            output_lines.append(line)
        plan_text = "\n".join(output_lines) + "\n"

    plan_actions = [
        line.strip().strip("()").split()[0]
        for line in plan_text.splitlines() if line.strip().startswith("(")
    ]
    first = plan_actions[0] if plan_actions else ""
    starts_pick = first.startswith("pick_key")
    starts_insert = "insert" in first
    starts_unlock = "unlock" in first or first.startswith("turn_key")
    starts_remove = first.startswith("remove_key")
    starts_place = first.startswith("place_key")
    if starts_pick:
        init.extend([["hand_free", hand], ["on", key_set, cabinet]])
    elif starts_insert or starts_unlock or starts_remove or starts_place:
        init.append(["holding", hand, key_set])
        init = [
            fact for fact in init
            if not (
                isinstance(fact, list)
                and len(fact) == 3
                and fact[0] == "on"
                and fact[1] == key_set
            )
        ]
    if starts_unlock or starts_remove:
        init.append(["inserted", key, drawer])
    if starts_remove or starts_place:
        init = [fact for fact in init if fact != ["locked", drawer]]
        if ["unlocked", drawer] not in init:
            init.append(["unlocked", drawer])

    # Remove stale references to a key-set object used as the individual key.
    goals = [
        [key if item == key_set and fact and fact[0] in {"inserted"} else item for item in fact]
        if isinstance(fact, list) else fact
        for fact in goals
    ]
    init = _unique_nodes(init)
    goals = _unique_nodes(goals)
    set_problem_facts(problem, ":init", init)
    set_problem_facts(problem, ":goal", goals)

    domain_name_match = re.search(r"\(\s*domain\s+([^\s()]+)", domain)
    domain_name = domain_name_match.group(1) if domain_name_match else "filing_cabinet_key_task"
    problem_section(problem, ":domain")[1] = domain_name

    stages = {
        "pick": any(name.startswith("pick_key") for name in plan_actions),
        "insert": any("insert" in name for name in plan_actions),
        "unlock": any(name.startswith("unlock") or name.startswith("turn_key") for name in plan_actions),
        "remove": any(name.startswith("remove_key") for name in plan_actions),
        "place": any(name.startswith("place_key") for name in plan_actions),
        "open": any(name.startswith("open") for name in plan_actions),
    }
    predicates = [
        ["hand", "?h"], ["key_set", "?s"], ["key", "?k"],
        ["filing_cabinet", "?c"], ["drawer", "?d"], ["hand_free", "?h"],
        ["holding", "?h", "?s"], ["closed", "?d"], ["open", "?d"],
        ["locked", "?d"], ["unlocked", "?d"], ["contains", "?s", "?k"],
        ["inserted", "?k", "?d"], ["on", "?o", "?s"],
    ]
    actions: list[Action] = []
    if stages["pick"]:
        actions.append(Action(
            "pick_key_set_from_filing_cabinet_top", ["?h", "?s", "?c"],
            make_and([["hand", "?h"], ["key_set", "?s"], ["filing_cabinet", "?c"], ["hand_free", "?h"], ["on", "?s", "?c"]]),
            make_and([["not", ["hand_free", "?h"]], ["holding", "?h", "?s"], ["not", ["on", "?s", "?c"]]]),
            "; Pick key set ?s from filing cabinet top ?c with hand ?h.",
        ))
    if stages["insert"]:
        actions.append(Action(
            "insert_key_from_key_set_into_drawer", ["?h", "?k", "?s", "?d"],
            make_and([["hand", "?h"], ["key", "?k"], ["key_set", "?s"], ["drawer", "?d"], ["holding", "?h", "?s"], ["contains", "?s", "?k"], ["closed", "?d"], ["locked", "?d"]]),
            make_and([["inserted", "?k", "?d"]]),
            "; Insert key ?k from key set ?s into drawer ?d with hand ?h.",
        ))
    if stages["unlock"]:
        actions.append(Action(
            "unlock_drawer_with_key_from_key_set", ["?h", "?d", "?k", "?s"],
            make_and([["hand", "?h"], ["drawer", "?d"], ["key", "?k"], ["key_set", "?s"], ["holding", "?h", "?s"], ["contains", "?s", "?k"], ["closed", "?d"], ["locked", "?d"], ["inserted", "?k", "?d"]]),
            make_and([["not", ["locked", "?d"]], ["unlocked", "?d"]]),
            "; Unlock drawer ?d with key ?k attached to key set ?s held by hand ?h.",
        ))
    if stages["remove"]:
        actions.append(Action(
            "remove_key_from_drawer_with_key_set", ["?h", "?k", "?s", "?d"],
            make_and([["hand", "?h"], ["key", "?k"], ["key_set", "?s"], ["drawer", "?d"], ["holding", "?h", "?s"], ["contains", "?s", "?k"], ["inserted", "?k", "?d"]]),
            make_and([["not", ["inserted", "?k", "?d"]]]),
            "; Remove key ?k from drawer ?d while holding attached key set ?s with hand ?h.",
        ))
    if stages["place"]:
        actions.append(Action(
            "place_key_set_on_filing_cabinet_top", ["?h", "?s", "?c"],
            make_and([["hand", "?h"], ["key_set", "?s"], ["filing_cabinet", "?c"], ["holding", "?h", "?s"]]),
            make_and([["not", ["holding", "?h", "?s"]], ["hand_free", "?h"], ["on", "?s", "?c"]]),
            "; Place key set ?s on filing cabinet top ?c with hand ?h.",
        ))
    if stages["open"]:
        actions.append(Action(
            "open_drawer_when_unlocked", ["?h", "?d"],
            make_and([["hand", "?h"], ["drawer", "?d"], ["hand_free", "?h"], ["closed", "?d"], ["unlocked", "?d"]]),
            make_and([["not", ["closed", "?d"]], ["open", "?d"]]),
            "; Open drawer ?d with hand ?h.",
        ))
    domain_lines = [f"(define (domain {domain_name})", "  (:predicates"]
    domain_lines.extend(f"    {sexp(pred)}" for pred in predicates)
    domain_lines.extend(["  )", ""])
    for action in actions:
        domain_lines.append(action.render())
        domain_lines.append("")
    if domain_lines[-1] == "":
        domain_lines.pop()
    domain_lines.append(")")
    domain = "\n".join(domain_lines) + "\n"

    classifiers = [
        (lambda n: n.startswith("pick_key"), "pick_key_set_from_filing_cabinet_top", [hand, key_set, cabinet]),
        (lambda n: "insert" in n, "insert_key_from_key_set_into_drawer", [hand, key, key_set, drawer]),
        (lambda n: n.startswith("unlock") or n.startswith("turn_key"), "unlock_drawer_with_key_from_key_set", [hand, drawer, key, key_set]),
        (lambda n: n.startswith("remove_key"), "remove_key_from_drawer_with_key_set", [hand, key, key_set, drawer]),
        (lambda n: n.startswith("place_key"), "place_key_set_on_filing_cabinet_top", [hand, key_set, cabinet]),
        (lambda n: n.startswith("open"), "open_drawer_when_unlocked", [hand, drawer]),
    ]
    plan_text = replace_plan_action_with_grounding(plan_text, classifiers)
    problem_text = render_problem(problem)
    plan_text = repair_plan_hand_interfaces(domain, problem_text, plan_text)
    return domain, problem_text, plan_text


KEY_SET_AUG_EPISODES = set(range(1, 15))


def dataset_name(round_dir: Path) -> str:
    return round_dir.parents[2].name


def normalize_declared_plan_objects(problem_text: str, plan_text: str) -> str:
    problem = parse_problem(problem_text)
    objects = {str(item) for item in problem_section(problem, ":objects")[1:]}
    type_facts = {
        str(fact[0]): str(fact[1])
        for fact in problem_facts(problem, ":init")
        if isinstance(fact, list) and len(fact) == 2
    }
    typed_by_base: dict[str, str] = {}
    for obj in objects:
        base = re.sub(r"\d+$", "", obj)
        typed_by_base.setdefault(base, obj)
    replacements: dict[str, str] = {}
    unary_objects_by_type: dict[str, list[str]] = {}
    for fact in problem_facts(problem, ":init"):
        if isinstance(fact, list) and len(fact) == 2:
            unary_objects_by_type.setdefault(str(fact[0]), []).append(str(fact[1]))
    for line in plan_text.splitlines():
        stripped = line.split(";", 1)[0].strip()
        if not stripped.startswith("("):
            continue
        for argument in stripped.strip("()").split()[1:]:
            if argument in objects:
                continue
            base = re.sub(r"\d+$", "", argument)
            if base.endswith("_"):
                base = base[:-1]
            candidate = typed_by_base.get(base)
            if candidate is None and base in objects:
                candidate = base
            if candidate is None and base in type_facts:
                candidate = type_facts[base]
            if candidate is not None:
                replacements[argument] = candidate
                continue
            lexical = [
                obj for obj in objects
                if obj in argument or argument in obj
            ]
            if len(lexical) == 1:
                replacements[argument] = lexical[0]
                continue
            if "hand" in argument and len(unary_objects_by_type.get("hand", [])) == 1:
                replacements[argument] = unary_objects_by_type["hand"][0]
    for old, new in replacements.items():
        plan_text = replace_token(plan_text, old, new)
    return plan_text


def normalize_closure_problem_state(problem_text: str) -> str:
    """Translate legacy uncapped/capped state into the open/closed contract."""
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    original_init = copy.deepcopy(init)
    uncapped = {
        str(fact[1]) for fact in init
        if isinstance(fact, list) and len(fact) == 2 and fact[0] == "uncapped"
    }
    if not uncapped:
        return problem_text
    init = [
        fact for fact in init
        if not (isinstance(fact, list) and fact and fact[0] == "uncapped")
    ]
    for container in sorted(uncapped):
        init = [fact for fact in init if fact != ["closed", container]]
        if ["open", container] not in init:
            init.append(["open", container])
    if init == original_init:
        return problem_text
    set_problem_facts(problem, ":init", init)
    return render_problem(problem)


def remove_redundant_lid_clear_goals(domain: str, problem_text: str) -> tuple[str, str]:
    """Drop clear(lid) when the goal already expresses the canonical cover state."""
    problem = parse_problem(problem_text)
    goals = problem_facts(problem, ":goal")
    lids = set(typed_objects(problem, "lid"))
    covered_lids = {
        str(fact[1]) for fact in goals
        if isinstance(fact, list) and len(fact) == 3 and fact[0] == "on" and fact[1] in lids
    }
    redundant = {
        lid for lid in covered_lids
        if any(
            isinstance(fact, list) and fact == ["clear", lid]
            for fact in goals
        )
    }
    if not redundant:
        return domain, problem_text
    goals = [
        fact for fact in goals
        if not (
            isinstance(fact, list)
            and len(fact) == 2
            and fact[0] == "clear"
            and fact[1] in redundant
        )
    ]
    set_problem_facts(problem, ":goal", goals)
    return domain, render_problem(problem)


def normalize_problem_state_for_plan(problem_text: str, plan_text: str) -> str:
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    actions = [
        line.strip().strip("()").split()
        for line in plan_text.splitlines()
        if line.strip().startswith("(")
    ]
    if not actions:
        return problem_text
    original_init = copy.deepcopy(init)
    hands = typed_objects(problem, "hand")
    for hand in hands:
        held = [
            fact for fact in init
            if isinstance(fact, list) and fact[:2] == ["holding", hand]
        ]
        if held:
            init = [fact for fact in init if fact != ["hand_free", hand]]
        else:
            hand_free = ["hand_free", hand]
            if hand_free not in init:
                init.append(hand_free)
    if init == original_init:
        return problem_text
    set_problem_facts(problem, ":init", init)
    return render_problem(problem)


def repair_audited_round_anomalies(
    problem_text: str,
    plan_text: str,
    *,
    tid: int,
    dataset: str,
    episode: int,
) -> tuple[str, str]:
    """Repair the seven source rounds rejected by the absolute replay gate."""
    key = (dataset, tid, episode)
    replacement_plans: dict[tuple[str, int, int], list[str]] = {
        ("human", 218, 218): [
            "(pick_block_from_block arm red block5)",
            "(place_block_on_table arm red table)",
            "(pick_block_from_block arm block5 green)",
            "(place_block_on_table arm block5 table)",
            "(pick_block_from_block arm green block1)",
            "(place_block_on_table arm green table)",
            "(pick_block_from_block arm block1 orange)",
            "(place_block_on_block arm block1 red)",
            "(pick_block_from_table arm green table)",
            "(place_block_on_block arm green orange)",
            "(pick_block_from_table arm block5 table)",
            "(place_block_on_block arm block5 block1)",
        ],
        ("human", 235, 235): [
            "(pick_block_from_block arm block_2 red)",
            "(place_block_on_table arm block_2 table)",
            "(pick_block_from_block arm red block_1)",
            "(place_block_on_table arm red table)",
            "(pick_block_from_block arm block_1 yellow)",
            "(place_block_on_table arm block_1 table)",
            "(pick_block_from_block arm yellow green)",
            "(place_block_on_block arm yellow block_2)",
            "(pick_block_from_table arm red table)",
            "(place_block_on_block arm red yellow)",
            "(pick_block_from_block arm green block_5)",
            "(place_block_on_table arm green table)",
            "(pick_block_from_table arm block_1 table)",
            "(place_block_on_block arm block_1 block_5)",
            "(pick_block_from_table arm green table)",
            "(place_block_on_block arm green block_1)",
        ],
        ("human_aug", 218, 127): [
            "(pick_block_from_block hand green_block block_1)",
            "(place_block_on_block hand green_block block_5)",
            "(pick_block_from_block hand block_1 orange_block)",
            "(place_block_on_block hand block_1 red_block)",
            "(pick_block_from_block hand green_block block_5)",
            "(place_block_on_block hand green_block orange_block)",
            "(pick_block_from_table hand block_5 table)",
            "(place_block_on_block hand block_5 block_1)",
        ],
    }
    steps = replacement_plans.get(key)
    if steps is not None:
        plan_text = "\n".join([*steps, f"; cost = {len(steps)} (unit cost)"]) + "\n"

    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    original_init = copy.deepcopy(init)

    def remove_fact(fact: Node) -> None:
        nonlocal init
        init = [item for item in init if item != fact]

    def add_fact(fact: Node) -> None:
        if fact not in init:
            init.append(fact)

    if key == ("human_aug", 21, 37):
        remove_fact(["is_off", "hot_water_button"])
    elif key == ("human_aug", 263, 83):
        remove_fact(["closed", "detergent_bottle_on_washing_machine"])
    elif key == ("human_aug", 264, 8):
        remove_fact(["open", "plastic_water_bottle"])
    elif key == ("human_aug", 264, 65):
        add_fact(["on", "capped_water_bottle", "counter"])
        lines = [
            line for line in plan_text.splitlines()
            if not line.strip().startswith("(place_bottle_on_counter ")
            and not line.strip().startswith("; cost =")
        ]
        count = sum(line.strip().startswith("(") for line in lines)
        plan_text = "\n".join([*lines, f"; cost = {count} (unit cost)"]) + "\n"

    if init != original_init:
        set_problem_facts(problem, ":init", init)
        problem_text = render_problem(problem)
    return problem_text, plan_text


@dataclass(frozen=True)
class GroundedAction:
    positive_preconditions: frozenset[tuple[str, ...]]
    negative_preconditions: frozenset[tuple[str, ...]]
    add_effects: frozenset[tuple[str, ...]]
    delete_effects: frozenset[tuple[str, ...]]


def domain_actions(domain: str) -> dict[str, Action]:
    actions: dict[str, Action] = {}
    for match in ACTION_RE.finditer(domain):
        end = find_matching_paren(domain, match.start()) + 1
        action = Action.parse(domain[match.start():end], "; action")
        if action.name in actions:
            raise ValueError(f"duplicate action {action.name}")
        actions[action.name] = action
    return actions


def split_literals(expr: Node) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    positive: set[tuple[str, ...]] = set()
    negative: set[tuple[str, ...]] = set()
    for item in conjunction(expr):
        if (
            isinstance(item, list)
            and len(item) == 2
            and item[0] == "not"
            and isinstance(item[1], list)
        ):
            negative.add(tuple(str(token) for token in item[1]))
        elif isinstance(item, list):
            positive.add(tuple(str(token) for token in item))
        else:
            raise ValueError(f"unsupported literal {item}")
    return positive, negative


def ground_action(action: Action, arguments: list[str]) -> GroundedAction:
    if len(arguments) != len(action.params):
        raise ValueError(
            f"plan action {action.name} expects {len(action.params)} arguments, "
            f"got {len(arguments)}"
        )
    bindings = dict(zip(action.params, arguments))

    def ground(items: set[tuple[str, ...]]) -> frozenset[tuple[str, ...]]:
        return frozenset(
            tuple(bindings.get(token, token) for token in item)
            for item in items
        )

    positive_pre, negative_pre = split_literals(action.pre)
    add_effects, delete_effects = split_literals(action.eff)
    return GroundedAction(
        ground(positive_pre), ground(negative_pre),
        ground(add_effects), ground(delete_effects),
    )


def schedule_blocking_release(domain: str, problem_text: str, plan_text: str) -> str:
    schemas = domain_actions(domain)
    problem = parse_problem(problem_text)
    state = {
        tuple(str(token) for token in fact)
        for fact in problem_facts(problem, ":init")
        if isinstance(fact, list)
    }
    lines = plan_text.splitlines()
    action_slots: list[int] = []
    entries: list[tuple[str, list[str], GroundedAction]] = []
    for index, line in enumerate(lines):
        stripped = line.split(";", 1)[0].strip()
        if not stripped.startswith("("):
            continue
        tokens = stripped.strip("()").split()
        if not tokens:
            continue
        schema = schemas.get(tokens[0])
        if schema is None:
            return plan_text
        try:
            grounded = ground_action(schema, tokens[1:])
        except ValueError:
            return plan_text
        action_slots.append(index)
        entries.append((line, tokens, grounded))

    moved = False
    index = 0
    while index < len(entries):
        current = entries[index][2]
        missing = current.positive_preconditions - state
        violated = current.negative_preconditions & state
        if not missing and not violated:
            state.difference_update(current.delete_effects)
            state.update(current.add_effects)
            index += 1
            continue

        missing_free = {
            item for item in missing if len(item) == 2 and item[0] == "hand_free"
        }
        if violated or not missing_free or missing != missing_free:
            return plan_text

        release_index: int | None = None
        for candidate_index in range(index + 1, len(entries)):
            candidate = entries[candidate_index][2]
            held_consumers = {
                item for item in candidate.positive_preconditions
                if len(item) == 3
                and item[0] == "holding"
                and ("hand_free", item[1]) in missing_free
                and item in state
            }
            if not held_consumers:
                continue
            releases = all(
                held in candidate.delete_effects
                and ("hand_free", held[1]) in candidate.add_effects
                for held in held_consumers
            )
            applicable = (
                candidate.positive_preconditions <= state
                and not (candidate.negative_preconditions & state)
            )
            if releases and applicable:
                release_index = candidate_index
            # Never move a release across an earlier consumer of that object.
            break
        if release_index is None:
            return plan_text
        entries.insert(index, entries.pop(release_index))
        moved = True

    if not moved:
        return plan_text
    for slot, entry in zip(action_slots, entries):
        lines[slot] = entry[0]
    lines = [line for line in lines if not line.strip().startswith("; cost =")]
    lines.append(f"; cost = {len(entries)} (unit cost)")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class PlanStep:
    line: str
    tokens: tuple[str, ...]
    grounded: GroundedAction


def apply_grounded_action(
    state: frozenset[tuple[str, ...]],
    grounded: GroundedAction,
) -> frozenset[tuple[str, ...]]:
    return frozenset((set(state) - set(grounded.delete_effects)) | set(grounded.add_effects))


def applicable(
    state: frozenset[tuple[str, ...]],
    grounded: GroundedAction,
) -> bool:
    return (
        grounded.positive_preconditions <= state
        and not (grounded.negative_preconditions & state)
    )


def bind_literal(
    schema_literal: tuple[str, ...],
    grounded_literal: tuple[str, ...],
) -> dict[str, str] | None:
    if len(schema_literal) != len(grounded_literal) or schema_literal[0] != grounded_literal[0]:
        return None
    bindings: dict[str, str] = {}
    for schema_token, grounded_token in zip(schema_literal[1:], grounded_literal[1:]):
        if schema_token.startswith("?"):
            previous = bindings.get(schema_token)
            if previous is not None and previous != grounded_token:
                return None
            bindings[schema_token] = grounded_token
        elif schema_token != grounded_token:
            return None
    return bindings


def grounded_schema_candidates(
    schema: Action,
    fixed: dict[str, str],
    state: frozenset[tuple[str, ...]],
    objects: set[str],
) -> list[PlanStep]:
    domains: list[list[str]] = []
    for parameter in schema.params:
        if parameter in fixed:
            domains.append([fixed[parameter]])
            continue
        candidates = set(objects)
        unary_constraints = [
            str(literal[0]) for literal in conjunction(schema.pre)
            if (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[1] == parameter
                and isinstance(literal[0], str)
            )
        ]
        for predicate in unary_constraints:
            candidates &= {
                fact[1] for fact in state
                if len(fact) == 2 and fact[0] == predicate
            }
        if not candidates:
            return []
        domains.append(sorted(candidates))

    candidates: list[PlanStep] = []
    for arguments in itertools.product(*domains):
        grounded = ground_action(schema, list(arguments))
        if not applicable(state, grounded):
            continue
        tokens = (schema.name, *arguments)
        candidates.append(PlanStep(
            "(" + " ".join(tokens) + ")",
            tokens,
            grounded,
        ))
    return candidates


def hand_interface_bridges(
    schemas: dict[str, Action],
    state: frozenset[tuple[str, ...]],
    objects: set[str],
    missing: frozenset[tuple[str, ...]],
) -> list[PlanStep]:
    """Find executable concrete pick/place actions that repair a hand interface."""
    requests: list[tuple[str, tuple[str, ...]]] = []
    for literal in sorted(missing):
        if len(literal) == 3 and literal[0] == "holding":
            requests.append(("pick", literal))
        elif len(literal) == 2 and literal[0] == "hand_free":
            held = sorted(
                fact for fact in state
                if len(fact) == 3 and fact[:2] == ("holding", literal[1])
            )
            requests.extend(("place", fact) for fact in held)

    output: list[PlanStep] = []
    for mode, holding_literal in requests:
        for schema in schemas.values():
            if not schema.name.startswith(f"{mode}_"):
                continue
            positive_pre, _negative_pre = split_literals(schema.pre)
            add_effects, delete_effects = split_literals(schema.eff)
            effect_literals = add_effects if mode == "pick" else delete_effects
            hand_effects = [
                literal for literal in effect_literals
                if len(literal) == 3 and literal[0] == "holding"
            ]
            for effect in hand_effects:
                fixed = bind_literal(effect, holding_literal)
                if fixed is None:
                    continue
                if mode == "pick" and not any(
                    len(literal) == 2
                    and literal[0] == "hand_free"
                    and literal in delete_effects
                    for literal in positive_pre
                ):
                    continue
                if mode == "place" and not any(
                    len(literal) == 2
                    and literal[0] == "hand_free"
                    for literal in add_effects
                ):
                    continue
                output.extend(grounded_schema_candidates(schema, fixed, state, objects))

    unique: dict[tuple[str, ...], PlanStep] = {}
    for candidate in output:
        unique.setdefault(candidate.tokens, candidate)
    return [unique[key] for key in sorted(unique)]


def repair_plan_hand_interfaces(domain: str, problem_text: str, plan_text: str) -> str:
    """Minimally reorder/bridge legacy plans after canonical hand contracts."""
    schemas = domain_actions(domain)
    problem = parse_problem(problem_text)
    initial_state = frozenset(
        tuple(str(token) for token in fact)
        for fact in problem_facts(problem, ":init")
        if isinstance(fact, list)
    )
    objects = {str(item) for item in problem_section(problem, ":objects")[1:]}
    positive_goals, negative_goals = split_literals(
        make_and(problem_facts(problem, ":goal"))
    )
    steps: list[PlanStep] = []
    for line in plan_text.splitlines():
        stripped = line.split(";", 1)[0].strip()
        if not stripped.startswith("("):
            continue
        tokens = tuple(stripped.strip("()").split())
        if not tokens or tokens[0] not in schemas:
            return plan_text
        try:
            grounded = ground_action(schemas[tokens[0]], list(tokens[1:]))
        except ValueError:
            return plan_text
        steps.append(PlanStep(line, tokens, grounded))

    state = initial_state
    first_failure: tuple[frozenset[tuple[str, ...]], frozenset[tuple[str, ...]]] | None = None
    for step in steps:
        missing = step.grounded.positive_preconditions - state
        violated = step.grounded.negative_preconditions & state
        if missing or violated:
            first_failure = (frozenset(missing), frozenset(violated))
            break
        state = apply_grounded_action(state, step.grounded)
    if first_failure is None:
        return plan_text
    initial_missing, initial_violated = first_failure
    if initial_violated or not any(
        literal and literal[0] in {"hand_free", "holding"}
        for literal in initial_missing
    ):
        return plan_text

    node_budget = 50_000
    visited = 0
    failed: set[tuple[object, ...]] = set()

    def solve(
        current_state: frozenset[tuple[str, ...]],
        remaining: tuple[PlanStep, ...],
        inserted: int,
    ) -> tuple[PlanStep, ...] | None:
        nonlocal visited
        visited += 1
        if visited > node_budget:
            return None
        if not remaining:
            if positive_goals <= current_state and not (negative_goals & current_state):
                return ()
            return None
        key = (
            tuple(step.tokens for step in remaining),
            current_state,
            inserted,
        )
        if key in failed:
            return None

        current = remaining[0]
        missing = current.grounded.positive_preconditions - current_state
        violated = current.grounded.negative_preconditions & current_state
        bridge_candidates: list[PlanStep] = []
        if inserted < 4 and not violated and any(
            literal and literal[0] in {"hand_free", "holding"}
            for literal in missing
        ):
            remaining_tokens = {step.tokens for step in remaining}
            bridge_candidates = [
                bridge for bridge in hand_interface_bridges(
                    schemas, current_state, objects, frozenset(missing)
                )
                if bridge.tokens not in remaining_tokens
            ]

        # If the current canonical action specifically needs an object in hand
        # and no demonstrated step can establish that fact, re-grasp it now.
        # Deferring the action across unrelated task phases is executable but
        # produces a poor demonstration (for example, recapping detergent only
        # after the washing machine has already been started).
        if any(literal and literal[0] == "holding" for literal in missing):
            for bridge in bridge_candidates:
                suffix = solve(
                    apply_grounded_action(current_state, bridge.grounded),
                    remaining,
                    inserted + 1,
                )
                if suffix is not None:
                    return (bridge, *suffix)

        order = list(range(len(remaining)))
        order.sort(key=lambda index: (index != 0, index))
        for index in order:
            candidate = remaining[index]
            if not applicable(current_state, candidate.grounded):
                continue
            next_remaining = remaining[:index] + remaining[index + 1:]
            suffix = solve(
                apply_grounded_action(current_state, candidate.grounded),
                next_remaining,
                inserted,
            )
            if suffix is not None:
                return (candidate, *suffix)

        if inserted < 4 and not violated:
            for bridge in bridge_candidates:
                suffix = solve(
                    apply_grounded_action(current_state, bridge.grounded),
                    remaining,
                    inserted + 1,
                )
                if suffix is not None:
                    return (bridge, *suffix)

        failed.add(key)
        return None

    repaired = solve(initial_state, tuple(steps), 0)
    if repaired is None or [step.tokens for step in repaired] == [step.tokens for step in steps]:
        return plan_text
    output = [step.line for step in repaired]
    output.append(f"; cost = {len(repaired)} (unit cost)")
    return "\n".join(output) + "\n"


def resolve_input_file(
    round_dir: Path,
    filename: str,
    input_mapping_dir: Path | None,
) -> Path:
    source = round_dir / filename
    if input_mapping_dir is None:
        return source
    for relative in (
        source.relative_to(PROJECT_ROOT),
        source.relative_to(EVAL_ROOT),
        source.relative_to(EVAL_ROOT / dataset_name(round_dir)),
    ):
        candidate = input_mapping_dir / relative
        if candidate.is_file():
            return candidate
    return source


REQUESTED_POSE_TASKS = {
    6, 7, 8, 9, 11, 12, 37, 38, 40, 52, 53, 54, 55, 56, 57,
    113, 114, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    294, 295, 296,
}
REQUESTED_MICROWAVE_TASKS = {15, 16, 30, 264, 265, 266}
REQUESTED_WASHING_TASKS = {14, 261, 262, 263}
REQUESTED_FAUCET_TASKS = {19, 20, 26, 27, 31, 104, 105, 106, 107, 108, 241}
REQUESTED_KETTLE_POUR_TASKS = {20}
FAUCET_HELD_KETTLE_EPISODES = {
    ("human_aug", 19, 132),
    ("human_aug", 19, 142),
    ("human_aug", 20, 206),
    ("human_aug", 20, 207),
    ("human_aug", 20, 355),
}


def _literal_is_pose_for(literal: Node, variable: str) -> bool:
    candidate = literal
    if (
        isinstance(literal, list)
        and len(literal) == 2
        and literal[0] == "not"
        and isinstance(literal[1], list)
    ):
        candidate = literal[1]
    return bool(
        isinstance(candidate, list)
        and len(candidate) == 2
        and candidate[0] in POSE_PREDICATES
        and candidate[1] == variable
    )


def _requested_bowl_source_pose(
    tid: int,
    dataset: str,
    problem_text: str,
) -> str:
    if tid == 8 or tid == 296:
        return "upright"
    if tid in {37, 40}:
        return "inverted"
    if tid == 295:
        return "upside_down"
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    bowls = set(typed_objects(problem, "bowl"))
    if tid == 7:
        if dataset == "human":
            return "inverted"
        if any(
            isinstance(fact, list)
            and len(fact) == 3
            and fact[0] == "on"
            and fact[1] in bowls
            and "desk" in str(fact[2])
            for fact in init
        ):
            return "inverted"
        return "vertical"
    if tid == 38:
        if dataset == "human_aug" and any(
            isinstance(fact, list)
            and len(fact) == 3
            and fact[0] == "holding"
            and fact[2] in bowls
            for fact in init
        ):
            return "upright"
        return "inverted"
    return "vertical"


def _requested_oriented_bowl_action(
    action: Action,
    source_pose: str,
) -> Action:
    if not (
        "bowl" in action.name
        and any(
            token in action.name
            for token in ("upright", "vertical", "inverted", "sideways", "on_side", "upside")
        )
    ):
        return action
    moving = placed_object_var(action)
    if moving is None or unary_type(action.pre, moving) != "bowl":
        return action
    target_pose = "vertical" if "vertical" in action.name else "upright"
    action.pre = make_and([
        *(
            literal for literal in conjunction(action.pre)
            if not _literal_is_pose_for(literal, moving)
        ),
        [source_pose, moving],
    ])
    effects = [
        literal for literal in conjunction(action.eff)
        if not _literal_is_pose_for(literal, moving)
    ]
    if source_pose != target_pose:
        effects.extend([
            ["not", [source_pose, moving]],
            [target_pose, moving],
        ])
    action.eff = make_and(_unique_nodes(effects))
    action.comment = (
        f"; Place bowl {moving} {target_pose} from {source_pose} on its target."
    )
    return action


def _requested_microwave_start(action: Action, tid: int) -> Action:
    if action.name != "turn_on_microwave" or tid not in REQUESTED_MICROWAVE_TASKS:
        return action
    hand = unary_var(action.pre, {"hand"})
    microwave = unary_var(action.pre, {"microwave"})
    if hand is None or microwave is None:
        return action
    if tid == 15:
        content, content_kind = "?cn", "corn"
        container, container_kind = "?p", "plate"
        content_relation: Node = ["on", content, container]
    elif tid == 16:
        content, content_kind = "?mk", "milk"
        container, container_kind = "?mg", "mug"
        content_relation = ["in", content, container]
    elif tid == 30:
        content, content_kind = "?mk", "milk"
        container, container_kind = "?c", "paper_cup"
        content_relation = ["in", content, container]
    else:
        content, content_kind = "?w", "water"
        container, container_kind = "?c", "paper_cup"
        content_relation = ["in", content, container]
    action.params = [hand, microwave, content, container]
    action.pre = make_and([
        ["hand", hand], ["microwave", microwave],
        [content_kind, content], [container_kind, container],
        ["hand_free", hand], ["closed", microwave], ["is_off", microwave],
        content_relation, ["in", container, microwave],
    ])
    action.eff = make_and([
        ["not", ["is_off", microwave]], ["is_on", microwave],
    ])
    action.comment = "; Turn on a loaded microwave with a free hand."
    return action


def _requested_washing_start(action: Action, tid: int) -> Action:
    if action.name != "start_washing_machine" or tid not in REQUESTED_WASHING_TASKS:
        return action
    machine = unary_var(action.pre, {"washing_machine"})
    if machine is None:
        return action
    drawer = unary_var(action.pre, {"detergent_drawer"}) or fresh_variable(
        action.params, "?d"
    )
    clothes_kind = "cloth" if tid == 14 else "clothes"
    clothes = unary_var(action.pre, {clothes_kind}) or fresh_variable(
        action.params, "?c"
    )
    detergent = unary_var(action.pre, {"detergent"}) or fresh_variable(
        [*action.params, clothes], "?dt"
    )
    action.pre = make_and(_unique_nodes([
        *conjunction(action.pre),
        ["detergent_drawer", drawer], [clothes_kind, clothes],
        ["detergent", detergent],
        ["in", clothes, machine], ["in", detergent, drawer],
    ]))
    action.params = ordered_used_parameters(
        action, [*action.params, drawer, clothes, detergent]
    )
    action.comment = "; Start the loaded washing machine after all preparation is complete."
    return action


def _requested_faucet_toggle(action: Action, tid: int) -> Action:
    if tid not in REQUESTED_FAUCET_TASKS or action.name not in {
        "turn_on_faucet", "turn_off_faucet"
    }:
        return action
    hand = unary_var(action.pre, {"hand"})
    if hand is not None and not has_literal(action.pre, "hand_free", hand):
        action.pre = add_literal(action.pre, ["hand_free", hand])
    return action


def _requested_boiled_kettle_pour(action: Action, tid: int) -> Action:
    if not (
        tid in REQUESTED_KETTLE_POUR_TASKS
        and "pour" in action.name
        and "water" in action.name
        and "kettle" in action.name
    ):
        return action
    water = unary_var(action.pre, {"water"})
    if water is not None and not has_literal(action.pre, "boiled", water):
        action.pre = add_literal(action.pre, ["boiled", water])
    return action


def _requested_direct_microwave_container(
    action: Action,
    *,
    force_top_inside: bool = False,
) -> Action:
    turntable = unary_var(action.pre, {"turntable", "microwave_turntable"})
    top_action = force_top_inside and "microwave_top" in action.name
    if turntable is None and not force_top_inside:
        return action
    hand = unary_var(action.pre, {"hand"})
    cup = unary_var(action.pre, {"paper_cup"})
    microwave = unary_var(action.pre, {"microwave"})
    converts_location_action = turntable is not None or top_action
    if microwave is None and converts_location_action:
        microwave = fresh_variable(action.params, "?m")
    if (
        converts_location_action
        and hand
        and cup
        and action.name.startswith("place_paper_cup")
    ):
        action.name = "place_paper_cup_in_microwave"
        action.params = [hand, cup, microwave]
        action.pre = make_and([
            ["hand", hand], ["paper_cup", cup], ["microwave", microwave],
            ["holding", hand, cup], ["open", microwave],
        ])
        action.eff = make_and([
            ["not", ["holding", hand, cup]], ["hand_free", hand],
            ["in", cup, microwave],
        ])
        action.comment = "; Place paper cup directly in microwave."
        return action
    if (
        converts_location_action
        and hand
        and cup
        and action.name.startswith("pick_paper_cup")
    ):
        action.name = "pick_paper_cup_from_microwave"
        action.params = [hand, cup, microwave]
        action.pre = make_and([
            ["hand", hand], ["paper_cup", cup], ["microwave", microwave],
            ["hand_free", hand], ["in", cup, microwave],
            ["open", microwave],
        ])
        action.eff = make_and([
            ["not", ["hand_free", hand]], ["holding", hand, cup],
            ["not", ["in", cup, microwave]],
        ])
        action.comment = "; Pick paper cup directly from microwave."
        return action
    if force_top_inside and cup is not None and microwave is not None:
        def direct_inside(literal: Node) -> Node:
            negated = (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[0] == "not"
                and isinstance(literal[1], list)
            )
            atom = literal[1] if negated else literal
            if atom == ["on", cup, microwave]:
                replacement: Node = ["in", cup, microwave]
                return ["not", replacement] if negated else replacement
            return literal

        action.pre = make_and([direct_inside(item) for item in conjunction(action.pre)])
        action.eff = make_and([direct_inside(item) for item in conjunction(action.eff)])
    if turntable is None:
        return action
    filtered_pre = [
        literal for literal in conjunction(action.pre)
        if turntable not in node_variables(literal)
    ]
    filtered_eff = [
        literal for literal in conjunction(action.eff)
        if turntable not in node_variables(literal)
    ]
    if cup is not None:
        filtered_pre.append(["in", cup, microwave])
    action.pre = make_and(_unique_nodes(filtered_pre))
    action.eff = make_and(_unique_nodes(filtered_eff))
    action.params = ordered_used_parameters(action, [*action.params, microwave])
    return action


def _rewrite_plan_with_inferred_parameters(
    text: str,
    edits: dict[str, ActionEdit],
    domain: str,
    problem_text: str,
) -> str:
    if not edits:
        return text
    schemas = domain_actions(domain)
    problem = parse_problem(problem_text)
    typed: dict[str, list[str]] = {}
    for fact in problem_facts(problem, ":init"):
        if isinstance(fact, list) and len(fact) == 2:
            typed.setdefault(str(fact[0]), []).append(str(fact[1]))
    init_atoms = {
        tuple(str(item) for item in fact)
        for fact in problem_facts(problem, ":init")
        if isinstance(fact, list) and all(isinstance(item, str) for item in fact)
    }
    output: list[str] = []
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        tokens = stripped.strip("()").split() if stripped.startswith("(") else []
        edit = edits.get(tokens[0]) if tokens else None
        if edit is None:
            if not stripped.startswith("; cost ="):
                output.append(line)
            continue
        if edit.new_name is None:
            changed = True
            continue
        grounding = dict(zip(edit.old_parameters, tokens[1:]))
        schema = schemas[edit.new_name]
        unresolved = [
            parameter for parameter in edit.new_parameters
            if parameter not in grounding
        ]
        while unresolved:
            progress = False
            for parameter in list(unresolved):
                kind = unary_type(schema.pre, parameter)
                candidates = list(dict.fromkeys(typed.get(kind or "", [])))
                if len(candidates) == 1:
                    grounding[parameter] = candidates[0]
                    unresolved.remove(parameter)
                    progress = True
                    continue
                supported: list[str] = []
                for candidate in candidates:
                    trial = {**grounding, parameter: candidate}
                    valid = True
                    for literal in conjunction(schema.pre):
                        if not isinstance(literal, list) or not literal or literal[0] == "not":
                            continue
                        args = literal[1:]
                        if parameter not in args:
                            continue
                        if not all(
                            isinstance(arg, str)
                            and (not arg.startswith("?") or arg in trial)
                            for arg in args
                        ):
                            continue
                        grounded = tuple(
                            trial.get(str(arg), str(arg)) for arg in literal
                        )
                        if grounded not in init_atoms:
                            valid = False
                            break
                    if valid:
                        supported.append(candidate)
                if len(supported) == 1:
                    grounding[parameter] = supported[0]
                    unresolved.remove(parameter)
                    progress = True
            if not progress:
                parameter = unresolved[0]
                kind = unary_type(schema.pre, parameter)
                raise ValueError(
                    f"cannot infer {parameter}:{kind} for plan action {tokens[0]}"
                )
        arguments = [grounding[parameter] for parameter in edit.new_parameters]
        output.append("(" + " ".join([edit.new_name, *arguments]) + ")")
        changed = True
    if not changed:
        return text
    count = sum(line.strip().startswith("(") for line in output)
    output.append(f"; cost = {count} (unit cost)")
    return "\n".join(output) + "\n"


def _remove_turntable_problem_model(problem_text: str) -> str:
    problem = parse_problem(problem_text)
    turntables = set(typed_objects(problem, "turntable")) | set(
        typed_objects(problem, "microwave_turntable")
    )
    turntables.update(
        str(item)
        for item in problem_section(problem, ":objects")[1:]
        if isinstance(item, str) and "turntable" in item.lower()
    )
    if not turntables:
        return problem_text
    microwaves = typed_objects(problem, "microwave")
    if len(microwaves) != 1:
        raise ValueError("turntable simplification requires one microwave")
    microwave = microwaves[0]

    def contains_turntable(node: Node) -> bool:
        if isinstance(node, str):
            return node in turntables
        return any(contains_turntable(child) for child in node)

    for section in (":init", ":goal"):
        facts = problem_facts(problem, section)
        replacements: list[Node] = []
        for fact in facts:
            if (
                isinstance(fact, list)
                and len(fact) == 3
                and fact[0] == "on"
                and fact[2] in turntables
            ):
                replacements.append(["in", fact[1], microwave])
            if not contains_turntable(fact):
                replacements.append(fact)
        set_problem_facts(problem, section, _unique_nodes(replacements))
    for turntable in turntables:
        remove_object(problem, turntable)
    return render_problem(problem)


def _simplify_turntable_model(
    problem_text: str, plan_text: str
) -> tuple[str, str]:
    """Remove real turntables and rename semantic objects with legacy names."""
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    objects = [
        str(item) for item in problem_section(problem, ":objects")[1:]
        if isinstance(item, str)
    ]
    occupied = set(objects)
    semantic_types = ("microwave", "paper_cup", "cup", "plate", "mug")
    for object_name in objects:
        if "turntable" not in object_name.lower():
            continue
        declared_types = {
            str(fact[0]) for fact in init
            if isinstance(fact, list)
            and len(fact) == 2
            and fact[1] == object_name
        }
        kind = next((item for item in semantic_types if item in declared_types), None)
        if kind is None:
            continue
        replacement = kind
        suffix = 2
        while replacement in occupied and replacement != object_name:
            replacement = f"{kind}_{suffix}"
            suffix += 1
        occupied.discard(object_name)
        occupied.add(replacement)
        problem_text = replace_token(problem_text, object_name, replacement)
        plan_text = replace_token(plan_text, object_name, replacement)
    return _remove_turntable_problem_model(problem_text), plan_text


def _ensure_requested_payload_objects(
    problem_text: str,
    plan_text: str,
    tid: int,
) -> str:
    if tid not in REQUESTED_MICROWAVE_TASKS | REQUESTED_WASHING_TASKS:
        return problem_text
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")

    def ensure_typed_object(kind: str, preferred: str) -> str:
        existing = typed_objects(problem, kind)
        if existing:
            return existing[0]
        objects = problem_section(problem, ":objects")
        name = preferred
        if name not in objects[1:]:
            add_object(problem, name)
        init.append([kind, name])
        return name

    if tid in REQUESTED_MICROWAVE_TASKS:
        if tid == 15:
            content_kind, content_name, container_kind, relation = (
                "corn", "corn", "plate", "on"
            )
        elif tid == 16:
            content_kind, content_name, container_kind, relation = (
                "milk", "milk", "mug", "in"
            )
        elif tid == 30:
            content_kind, content_name, container_kind, relation = (
                "milk", "milk", "paper_cup", "in"
            )
        else:
            content_kind, content_name, container_kind, relation = (
                "water", "water", "paper_cup", "in"
            )
        containers = typed_objects(problem, container_kind)
        if len(containers) == 1:
            had_content = bool(typed_objects(problem, content_kind))
            content = ensure_typed_object(content_kind, content_name)
            produced_before_start = any(
                line.strip().startswith("(pour_")
                for line in plan_text.splitlines()
            )
            if not had_content and not produced_before_start:
                init.append([relation, content, containers[0]])

    if tid in REQUESTED_WASHING_TASKS:
        machines = typed_objects(problem, "washing_machine")
        clothes_kind = "cloth" if tid == 14 else "clothes"
        if len(machines) == 1:
            drawer = ensure_typed_object("detergent_drawer", "detergent_drawer")
            had_clothes = bool(typed_objects(problem, clothes_kind))
            clothes = ensure_typed_object(
                clothes_kind, "cloth" if tid == 14 else "laundry_clothes"
            )
            placed_before_start = any(
                re.match(r"\(place_(?:cloth|clothes)_in_washing_machine\b", line.strip())
                for line in plan_text.splitlines()
            )
            already_loaded = ["in", clothes, machines[0]] in init
            if not placed_before_start and not already_loaded:
                init.append(["in", clothes, machines[0]])
            had_detergent = bool(typed_objects(problem, "detergent"))
            detergent = ensure_typed_object("detergent", "detergent")
            poured_before_start = any(
                "pour_detergent" in line
                for line in plan_text.splitlines()
            )
            already_loaded = ["in", detergent, drawer] in init
            if not poured_before_start and not already_loaded:
                init.append(["in", detergent, drawer])

    set_problem_facts(problem, ":init", _unique_nodes(init))
    return render_problem(problem)


def _set_requested_bowl_pose_facts(
    domain: str,
    problem_text: str,
    plan_text: str,
    source_pose: str,
) -> str:
    actions = domain_actions(domain)
    moved: set[str] = set()
    for line in plan_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("("):
            continue
        tokens = stripped.strip("()").split()
        action = actions.get(tokens[0]) if tokens else None
        if action is None or not (
            "bowl" in action.name
            and any(token in action.name for token in ("upright", "vertical"))
        ):
            continue
        grounding = dict(zip(action.params, tokens[1:]))
        moving = placed_object_var(action)
        if moving is not None and moving in grounding:
            moved.add(grounding[moving])
    if not moved:
        return problem_text
    problem = parse_problem(problem_text)
    init = [
        fact for fact in problem_facts(problem, ":init")
        if not (
            isinstance(fact, list)
            and len(fact) == 2
            and fact[0] in POSE_PREDICATES
            and fact[1] in moved
        )
    ]
    init.extend([source_pose, bowl] for bowl in sorted(moved))
    set_problem_facts(problem, ":init", _unique_nodes(init))
    return render_problem(problem)


def _append_action(domain: str, action: Action) -> str:
    if action.name in domain_actions(domain):
        return domain
    end = len(domain.rstrip()) - 1
    return domain[:end].rstrip() + "\n\n" + action.render() + "\n)\n"


def _repair_held_kettle_faucet_start(
    domain: str,
    problem_text: str,
    plan_text: str,
) -> tuple[str, str, str]:
    problem = parse_problem(problem_text)
    hands = typed_objects(problem, "hand")
    kettles = typed_objects(problem, "kettle")
    counters = typed_objects(problem, "counter")
    if not counters:
        add_object(problem, "counter")
        init = problem_facts(problem, ":init")
        init.append(["counter", "counter"])
        set_problem_facts(problem, ":init", _unique_nodes(init))
        counters = ["counter"]
    if len(hands) != 1 or len(kettles) != 1 or len(counters) != 1:
        raise ValueError("held-kettle faucet repair requires one hand, kettle, and counter")
    hand, kettle, counter = hands[0], kettles[0], counters[0]
    place = Action(
        "place_kettle_on_counter", ["?h", "?k", "?c"],
        make_and([["hand", "?h"], ["kettle", "?k"], ["counter", "?c"],
                  ["holding", "?h", "?k"]]),
        make_and([["not", ["holding", "?h", "?k"]], ["hand_free", "?h"],
                  ["on", "?k", "?c"]]),
        "; Place kettle on counter to free the hand.",
    )
    pick = Action(
        "pick_kettle_from_counter", ["?h", "?k", "?c"],
        make_and([["hand", "?h"], ["kettle", "?k"], ["counter", "?c"],
                  ["hand_free", "?h"], ["on", "?k", "?c"]]),
        make_and([["not", ["hand_free", "?h"]], ["holding", "?h", "?k"],
                  ["not", ["on", "?k", "?c"]]]),
        "; Pick kettle from counter after turning on the faucet.",
    )
    domain = _append_action(_append_action(domain, place), pick)
    if "(place_kettle_on_counter " not in plan_text:
        output: list[str] = []
        inserted = False
        for line in plan_text.splitlines():
            stripped = line.strip()
            if not inserted and stripped.startswith("(turn_on_faucet "):
                output.append(f"(place_kettle_on_counter {hand} {kettle} {counter})")
                output.append(line)
                output.append(f"(pick_kettle_from_counter {hand} {kettle} {counter})")
                inserted = True
            elif not stripped.startswith("; cost ="):
                output.append(line)
        if not inserted:
            raise ValueError("held-kettle faucet repair found no turn_on_faucet step")
        count = sum(line.strip().startswith("(") for line in output)
        output.append(f"; cost = {count} (unit cost)")
        plan_text = "\n".join(output) + "\n"
    return domain, render_problem(problem), plan_text


def _schedule_microwave_close_after_loading(plan_text: str) -> str:
    lines = [
        line for line in plan_text.splitlines()
        if not line.strip().startswith("; cost =")
    ]
    place_index = next(
        (
            index for index, line in enumerate(lines)
            if line.strip().startswith("(place_paper_cup_in_microwave ")
        ),
        None,
    )
    if place_index is None:
        return plan_text
    close_indices = [
        index for index, line in enumerate(lines[:place_index])
        if line.strip().startswith("(close_microwave ")
    ]
    if not close_indices:
        return plan_text
    close_index = close_indices[-1]
    if any(
        line.strip().startswith("(open_microwave ")
        for line in lines[close_index + 1:place_index]
    ):
        return plan_text
    close_line = lines.pop(close_index)
    place_index -= 1
    lines.insert(place_index + 1, close_line)
    count = sum(line.strip().startswith("(") for line in lines)
    lines.append(f"; cost = {count} (unit cost)")
    return "\n".join(lines) + "\n"


def apply_requested_contract_fixes(
    domain: str,
    problem_text: str,
    plan_text: str,
    *,
    tid: int,
    dataset: str,
    episode: int,
) -> tuple[str, str, str, list[str]]:
    notes: list[str] = []
    action_names = set(domain_actions(domain))
    if "turn_on_microwave" in action_names or "start_washing_machine" in action_names:
        problem_text = _ensure_requested_payload_objects(problem_text, plan_text, tid)
    source_pose = (
        _requested_bowl_source_pose(tid, dataset, problem_text)
        if tid in REQUESTED_POSE_TASKS else None
    )
    force_top_inside = any(
        name.startswith("place_paper_cup") and "microwave_top" in name
        for name in action_names
    )

    def transform(action: Action) -> Action:
        action = _requested_direct_microwave_container(
            action, force_top_inside=force_top_inside
        )
        if source_pose is not None:
            action = _requested_oriented_bowl_action(action, source_pose)
        action = _requested_microwave_start(action, tid)
        action = _requested_washing_start(action, tid)
        action = _requested_faucet_toggle(action, tid)
        action = _requested_boiled_kettle_pour(action, tid)
        return action

    domain, edits = rewrite_actions(domain, transform)
    if edits:
        if any(
            "turntable" in old or "turntable" in (edit.new_name or "")
            for old, edit in edits.items()
        ):
            problem_text, plan_text = _simplify_turntable_model(
                problem_text, plan_text
            )
            notes.append("remove_microwave_turntable")
        plan_text = _rewrite_plan_with_inferred_parameters(
            plan_text, edits, domain, problem_text
        )
        plan_text = _schedule_microwave_close_after_loading(plan_text)
        notes.append("requested_operator_contracts")

    if "turntable" in problem_text.lower():
        problem_text, plan_text = _simplify_turntable_model(
            problem_text, plan_text
        )
        notes.append("remove_microwave_turntable")

    if source_pose is not None:
        problem_text = _set_requested_bowl_pose_facts(
            domain, problem_text, plan_text, source_pose
        )

    if (dataset, tid, episode) in FAUCET_HELD_KETTLE_EPISODES:
        domain, problem_text, plan_text = _repair_held_kettle_faucet_start(
            domain, problem_text, plan_text
        )
        notes.append("held_kettle_faucet_start")

    domain = ensure_action_predicates_declared(domain)
    if "start_washing_machine" in domain_actions(domain):
        clothes_kind = "cloth" if tid == 14 else "clothes"
        domain = ensure_predicate_arities(
            domain, {clothes_kind: 1, "detergent": 1, "in": 2}
        )
    domain = remove_unused_named_predicates(
        domain, problem_text, {"turntable", "microwave_turntable"}
    )
    plan_text = normalize_plan_schema_arity(domain, problem_text, plan_text)
    return domain, problem_text, plan_text, notes


DRAWER_ROLES = ("top", "middle", "bottom", "lowest")


def simplify_drawer_role_predicates(
    domain: str, problem_text: str, plan_text: str
) -> tuple[str, str, str, list[str]]:
    """Fold drawer plus positional tags into concrete drawer predicates."""
    role_predicates = {f"is_{role}": role for role in DRAWER_ROLES}
    if not any(f"({name}" in domain or f"({name}" in problem_text for name in role_predicates):
        return domain, problem_text, plan_text, []

    problem = parse_problem(problem_text)
    role_by_object: dict[str, str] = {}
    for fact in problem_facts(problem, ":init"):
        if (
            isinstance(fact, list)
            and len(fact) == 2
            and fact[0] in role_predicates
        ):
            object_name = str(fact[1])
            role = role_predicates[str(fact[0])]
            previous = role_by_object.get(object_name)
            if previous is not None and previous != role:
                raise ValueError(f"drawer {object_name} has conflicting roles")
            role_by_object[object_name] = role

    roles_needed = set(role_by_object.values())
    for section in (":init", ":goal"):
        facts = problem_facts(problem, section)
        rewritten: list[Node] = []
        concrete_objects: set[str] = set()
        for fact in facts:
            if not isinstance(fact, list) or len(fact) != 2:
                rewritten.append(fact)
                continue
            predicate, object_name = str(fact[0]), str(fact[1])
            role = role_by_object.get(object_name)
            if predicate in role_predicates:
                role = role_predicates[predicate]
                roles_needed.add(role)
                concrete_objects.add(object_name)
                continue
            if predicate == "drawer" and role is not None:
                concrete_objects.add(object_name)
                continue
            rewritten.append(fact)
        rewritten.extend(
            [f"{role_by_object[object_name]}_drawer", object_name]
            for object_name in concrete_objects
        )
        set_problem_facts(problem, section, _unique_nodes(rewritten))
    problem_text = render_problem(problem)

    original_actions = domain_actions(domain)
    grounded_roles: dict[tuple[str, str], set[str]] = {}
    for line in plan_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("("):
            continue
        tokens = stripped.strip("()").split()
        action = original_actions.get(tokens[0]) if tokens else None
        if action is None:
            continue
        grounding = dict(zip(action.params, tokens[1:]))
        for literal in conjunction(action.pre):
            if not (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[0] == "drawer"
            ):
                continue
            variable = str(literal[1])
            if any(has_literal(action.pre, name, variable) for name in role_predicates):
                continue
            role = role_by_object.get(grounding.get(variable, ""))
            if role is not None:
                grounded_roles.setdefault((action.name, variable), set()).add(role)

    def assignments_for(action: Action) -> dict[str, str]:
        assignments: dict[str, str] = {}
        for literal in conjunction(action.pre):
            if (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[0] in role_predicates
            ):
                assignments[str(literal[1])] = role_predicates[str(literal[0])]
        for literal in conjunction(action.pre):
            if not (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[0] == "drawer"
            ):
                continue
            variable = str(literal[1])
            roles = grounded_roles.get((action.name, variable), set())
            if variable not in assignments and len(roles) == 1:
                assignments[variable] = next(iter(roles))
        return assignments

    def specialize(action: Action, assignments: dict[str, str]) -> Action:
        if not assignments:
            return action

        def keep(literal: Node) -> bool:
            return not (
                isinstance(literal, list)
                and len(literal) == 2
                and str(literal[1]) in assignments
                and (literal[0] == "drawer" or literal[0] in role_predicates)
            )

        action.pre = make_and([
            *(literal for literal in conjunction(action.pre) if keep(literal)),
            *(
                [f"{role}_drawer", variable]
                for variable, role in assignments.items()
            ),
        ])
        action.eff = make_and(
            literal for literal in conjunction(action.eff) if keep(literal)
        )
        roles_needed.update(assignments.values())
        return action

    replacements: list[tuple[int, int, str]] = []
    close_role_argument: dict[str, int] = {}
    for match in list(ACTION_RE.finditer(domain)):
        action_start = match.start()
        action_end = find_matching_paren(domain, action_start) + 1
        line_start = domain.rfind("\n", 0, action_start) + 1
        prefix = domain[:line_start]
        previous_end = len(prefix.rstrip(" \t\r\n"))
        previous_start = prefix.rfind("\n", 0, previous_end) + 1
        previous_line = prefix[previous_start:previous_end].strip()
        span_start = previous_start if previous_line.startswith(";") else line_start
        comment = previous_line if previous_line.startswith(";") else f"; {match.group(1)}."
        action = Action.parse(domain[action_start:action_end], comment)
        assignments = assignments_for(action)
        generic_variables = [
            str(literal[1])
            for literal in conjunction(action.pre)
            if isinstance(literal, list)
            and len(literal) == 2
            and literal[0] == "drawer"
            and str(literal[1]) not in assignments
        ]
        multi_role = [
            variable for variable in generic_variables
            if len(grounded_roles.get((action.name, variable), set())) > 1
        ]
        if action.name == "close_drawer" and len(multi_role) == 1:
            variable = multi_role[0]
            close_role_argument[action.name] = action.params.index(variable) + 1
            variants: list[str] = []
            used_roles = grounded_roles[(action.name, variable)]
            for role in DRAWER_ROLES:
                if role not in used_roles:
                    continue
                variant = specialize(copy.deepcopy(action), {**assignments, variable: role})
                variant.name = f"close_{role}_drawer"
                variant.comment = f"; Close {role} drawer {variable}."
                variants.append(variant.render())
            replacements.append((span_start, action_end, "\n".join(variants)))
            continue
        specialized = specialize(copy.deepcopy(action), assignments)
        if specialized != action:
            replacements.append((span_start, action_end, specialized.render()))

    for start, end, replacement in reversed(replacements):
        domain = domain[:start] + replacement + domain[end:]
    domain = re.sub(r"\n{3,}", "\n\n", domain)

    if close_role_argument:
        output: list[str] = []
        for line in plan_text.splitlines():
            stripped = line.strip()
            tokens = stripped.strip("()").split() if stripped.startswith("(") else []
            argument_index = close_role_argument.get(tokens[0]) if tokens else None
            if argument_index is not None and len(tokens) > argument_index:
                role = role_by_object.get(tokens[argument_index])
                if role is None:
                    raise ValueError(f"cannot infer drawer role for plan step {stripped}")
                tokens[0] = f"close_{role}_drawer"
                line = "(" + " ".join(tokens) + ")"
            output.append(line)
        plan_text = "\n".join(output) + ("\n" if plan_text.endswith("\n") else "")

    domain = rewrite_predicates(domain, remove_names=set(role_predicates))
    domain = ensure_predicate_arities(
        domain, {f"{role}_drawer": 1 for role in sorted(roles_needed)}
    )
    domain = remove_unused_named_predicates(
        domain, problem_text, {"drawer", *role_predicates}
    )
    plan_text = normalize_plan_schema_arity(domain, problem_text, plan_text)
    return domain, problem_text, plan_text, ["simplify_drawer_role_predicates"]


VISUAL_SOURCE_POSE_BY_TASK_KIND = {
    (5, "mug"): "sideways",
    (36, "box"): "upside_down",
    (39, "cup"): "upside_down",
    (41, "cup"): "upside_down",
    (42, "cup"): "on_side",
    (43, "cup"): "sideways",
    (44, "mug"): "on_side",
    (45, "mug"): "upside_down",
    (60, "box"): "on_side",
    (61, "box"): "upright",
    (75, "cup"): "upside_down",
    (77, "cup"): "sideways",
    (80, "cup"): "on_side",
    (81, "cup"): "on_side",
    (95, "cup"): "on_side",
    (96, "cup"): "sideways",
    (97, "cup"): "upside_down",
    (99, "cup"): "upside_down",
    (99, "paper_cup"): "upside_down",
    (100, "cup"): "upright",
    (102, "mug"): "upside_down",
    (103, "mug"): "on_side",
    (291, "paper_cup"): "upside_down",
}


def _non_bowl_source_pose(
    tid: int,
    object_kind: str,
    target_pose: str,
    acquire_action: str | None,
    existing_pose: str | None,
) -> str:
    if object_kind == "lid":
        return existing_pose or ("upright" if target_pose == "flat" else "flat")
    if acquire_action is None:
        return existing_pose or target_pose
    visual_pose = VISUAL_SOURCE_POSE_BY_TASK_KIND.get((tid, object_kind))
    if visual_pose is not None:
        return visual_pose
    if object_kind in {"book", "cutting_board"}:
        return "vertical"
    if object_kind == "plate":
        return "flat" if "microwave" in acquire_action else "vertical"
    return existing_pose or target_pose


def simplify_non_bowl_pose_contracts(
    domain: str,
    problem_text: str,
    plan_text: str,
    *,
    tid: int,
) -> tuple[str, str, str, list[str]]:
    """Make every non-bowl oriented placement a local source-to-target change."""
    actions = domain_actions(domain)
    oriented: dict[str, tuple[str, str, str]] = {}
    for action in actions.values():
        moving = placed_object_var(action)
        if moving is None:
            continue
        object_kind = unary_type(action.pre, moving)
        target_poses = {
            str(literal[0])
            for literal in conjunction(action.eff)
            if isinstance(literal, list)
            and len(literal) == 2
            and literal[0] in POSE_PREDICATES
            and literal[1] == moving
        }
        if (
            not target_poses
            and object_kind == "lid"
            and re.fullmatch(r"place_lid_on_(?:pot|kettle)", action.name)
            and any(
                has_literal(action.eff, pose, moving, negative=True)
                for pose in POSE_PREDICATES
            )
        ):
            target_poses = {"flat"}
        if object_kind == "bowl" or len(target_poses) != 1:
            continue
        oriented[action.name] = (moving, object_kind or "object", next(iter(target_poses)))
    if not oriented:
        return domain, problem_text, plan_text, []

    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    initial_poses: dict[str, set[str]] = {}
    for fact in init:
        if (
            isinstance(fact, list)
            and len(fact) == 2
            and fact[0] in POSE_PREDICATES
        ):
            initial_poses.setdefault(str(fact[1]), set()).add(str(fact[0]))

    plan_lines = plan_text.splitlines()
    last_acquire: dict[str, str] = {}
    first_context: dict[str, tuple[str, str, str | None]] = {}
    parsed_steps: list[tuple[int, list[str], Action, dict[str, str]]] = []
    for index, line in enumerate(plan_lines):
        stripped = line.strip()
        tokens = stripped.strip("()").split() if stripped.startswith("(") else []
        action = actions.get(tokens[0]) if tokens else None
        if action is None:
            continue
        grounding = dict(zip(action.params, tokens[1:]))
        parsed_steps.append((index, tokens, action, grounding))
        contract = oriented.get(action.name)
        if contract is not None:
            moving, object_kind, target_pose = contract
            object_name = grounding[moving]
            first_context.setdefault(
                object_name,
                (object_kind, target_pose, last_acquire.get(object_name)),
            )
        for literal in conjunction(action.eff):
            if (
                isinstance(literal, list)
                and len(literal) == 3
                and literal[0] == "holding"
            ):
                object_name = grounding.get(str(literal[2]))
                if object_name is not None:
                    last_acquire[object_name] = action.name

    inferred_initial: dict[str, str] = {}
    for object_name, (object_kind, target_pose, acquire_action) in first_context.items():
        poses = initial_poses.get(object_name, set())
        existing_pose = next(iter(poses)) if len(poses) == 1 else None
        inferred_initial[object_name] = _non_bowl_source_pose(
            tid, object_kind, target_pose, acquire_action, existing_pose
        )
    if inferred_initial:
        init = [
            fact for fact in init
            if not (
                isinstance(fact, list)
                and len(fact) == 2
                and fact[0] in POSE_PREDICATES
                and str(fact[1]) in inferred_initial
            )
        ]
        init.extend(
            [pose, object_name]
            for object_name, pose in sorted(inferred_initial.items())
        )
        set_problem_facts(problem, ":init", _unique_nodes(init))
        problem_text = render_problem(problem)

    pose_state: dict[str, set[str]] = {
        object_name: {pose} for object_name, pose in inferred_initial.items()
    }
    for object_name, poses in initial_poses.items():
        pose_state.setdefault(object_name, set(poses))
    step_sources: dict[int, str] = {}
    sources_by_action: dict[str, set[str]] = {}
    for index, _tokens, action, grounding in parsed_steps:
        contract = oriented.get(action.name)
        if contract is not None:
            moving, _object_kind, _target_pose = contract
            object_name = grounding[moving]
            poses = pose_state.get(object_name, set())
            if len(poses) != 1:
                raise ValueError(
                    f"cannot infer one source pose for {object_name} before {action.name}: {sorted(poses)}"
                )
            source_pose = next(iter(poses))
            step_sources[index] = source_pose
            sources_by_action.setdefault(action.name, set()).add(source_pose)
        for literal in conjunction(action.eff):
            negative = (
                isinstance(literal, list)
                and len(literal) == 2
                and literal[0] == "not"
                and isinstance(literal[1], list)
            )
            atom = literal[1] if negative else literal
            if not (
                isinstance(atom, list)
                and len(atom) == 2
                and atom[0] in POSE_PREDICATES
            ):
                continue
            object_name = grounding.get(str(atom[1]), str(atom[1]))
            pose_state.setdefault(object_name, set())
            if negative:
                pose_state[object_name].discard(str(atom[0]))
            else:
                pose_state[object_name].add(str(atom[0]))

    for action_name, (_moving, _kind, target_pose) in oriented.items():
        sources_by_action.setdefault(action_name, {target_pose})

    def local_pose_action(action: Action, source_pose: str) -> Action:
        moving, object_kind, target_pose = oriented[action.name]
        action.pre = make_and([
            *(
                literal for literal in conjunction(action.pre)
                if not _literal_is_pose_for(literal, moving)
            ),
            [source_pose, moving],
        ])
        effects = [
            literal for literal in conjunction(action.eff)
            if not _literal_is_pose_for(literal, moving)
        ]
        if source_pose != target_pose:
            effects.extend([
                ["not", [source_pose, moving]],
                [target_pose, moving],
            ])
        action.eff = make_and(_unique_nodes(effects))
        action.comment = (
            f"; Place {object_kind.replace('_', ' ')} {moving} "
            f"{target_pose} from {source_pose} on its target."
        )
        return action

    replacements: list[tuple[int, int, str]] = []
    variant_names: dict[tuple[str, str], str] = {}
    for match in list(ACTION_RE.finditer(domain)):
        action_start = match.start()
        action_end = find_matching_paren(domain, action_start) + 1
        action_name = match.group(1)
        if action_name not in oriented:
            continue
        line_start = domain.rfind("\n", 0, action_start) + 1
        prefix = domain[:line_start]
        previous_end = len(prefix.rstrip(" \t\r\n"))
        previous_start = prefix.rfind("\n", 0, previous_end) + 1
        previous_line = prefix[previous_start:previous_end].strip()
        span_start = previous_start if previous_line.startswith(";") else line_start
        comment = previous_line if previous_line.startswith(";") else f"; {action_name}."
        original = Action.parse(domain[action_start:action_end], comment)
        source_poses = sources_by_action[action_name]
        variants: list[str] = []
        for source_pose in sorted(source_poses):
            variant = local_pose_action(copy.deepcopy(original), source_pose)
            if len(source_poses) > 1:
                variant.name = f"{action_name}_from_{source_pose}_pose"
            variant_names[(action_name, source_pose)] = variant.name
            variants.append(variant.render())
        replacements.append((span_start, action_end, "\n".join(variants)))
    for start, end, replacement in reversed(replacements):
        domain = domain[:start] + replacement + domain[end:]
    domain = re.sub(r"\n{3,}", "\n\n", domain)

    for index, source_pose in step_sources.items():
        tokens = plan_lines[index].strip().strip("()").split()
        tokens[0] = variant_names[(tokens[0], source_pose)]
        plan_lines[index] = "(" + " ".join(tokens) + ")"
    plan_text = "\n".join(plan_lines) + ("\n" if plan_text.endswith("\n") else "")

    required_poses = {
        pose for poses in sources_by_action.values() for pose in poses
    } | {target for _moving, _kind, target in oriented.values()}
    domain = ensure_predicate_arities(
        domain, {pose: 1 for pose in sorted(required_poses)}
    )
    domain = remove_unused_named_predicates(domain, problem_text, POSE_PREDICATES)
    plan_text = normalize_plan_schema_arity(domain, problem_text, plan_text)
    return domain, problem_text, plan_text, ["simplify_non_bowl_pose_contracts"]


APPROVED_UNUSED_PREDICATES = {
    "black_box", "bowl_base", "bowl_block", "bowl_target", "box_block",
    "can_place_on", "capped", "child_lock_button", "dispenser_button",
    "drawer_of", "floor_of", "handle_button", "handle_of", "kitchen_scale",
    "leaning", "power_switch", "spatial_marker", "stack_target", "supports",
    "tilted",
}

APPROVED_DRAWER_TASKS = {279, 280, 281}
APPROVED_WASHING_TASKS = {14, 261, 262, 263}
APPROVED_KETTLE_TASKS = {270, 271, 272, 276, 277, 278}
APPROVED_LID_TASKS = {26, 270, 271, 272, 276, 277, 278, 291, 292, 293}


def _predicate_names(node: Node) -> set[str]:
    names: set[str] = set()

    def visit(item: Node) -> None:
        if isinstance(item, str) or not item:
            return
        head = item[0]
        if head in {"and", "or"}:
            for child in item[1:]:
                visit(child)
        elif head == "not" and len(item) == 2:
            visit(item[1])
        elif isinstance(head, str):
            names.add(head)

    visit(node)
    return names


def _remove_approved_init_only_predicates(
    domain: str, problem_text: str
) -> tuple[str, str, bool]:
    action_usage = set().union(*(
        _predicate_names(action.pre) | _predicate_names(action.eff)
        for action in domain_actions(domain).values()
    )) if domain_actions(domain) else set()
    problem = parse_problem(problem_text)
    goal_usage = set().union(*(
        _predicate_names(fact) for fact in problem_facts(problem, ":goal")
    )) if problem_facts(problem, ":goal") else set()
    unsafe = APPROVED_UNUSED_PREDICATES & (action_usage | goal_usage)
    if unsafe:
        raise ValueError(
            "approved init-only predicates became semantic: " + ", ".join(sorted(unsafe))
        )
    init = problem_facts(problem, ":init")
    filtered = [
        fact for fact in init
        if not (
            isinstance(fact, list)
            and fact
            and fact[0] in APPROVED_UNUSED_PREDICATES
        )
    ]
    changed = filtered != init
    if changed:
        set_problem_facts(problem, ":init", filtered)
        problem_text = render_problem(problem)
    new_domain = rewrite_predicates(
        domain, remove_names=APPROVED_UNUSED_PREDICATES
    )
    return new_domain, problem_text, changed or new_domain != domain


def _ensure_initial_state(
    problem_text: str,
    *,
    object_kind: str,
    state_predicate: str,
    conflicting_states: set[str],
) -> str:
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    objects = typed_objects(problem, object_kind)
    changed = False
    for object_name in objects:
        if any(
            isinstance(fact, list)
            and len(fact) == 2
            and fact[0] in conflicting_states | {state_predicate}
            and fact[1] == object_name
            for fact in init
        ):
            continue
        init.append([state_predicate, object_name])
        changed = True
    if not changed:
        return problem_text
    set_problem_facts(problem, ":init", _unique_nodes(init))
    return render_problem(problem)


def _canonical_drawer_lock_action(action: Action) -> Action:
    unlock_names = {
        "turn_combination_lock_knob_unlock",
        "turn_lock_knob_unlocked",
        "unlock_drawer_with_combination_lock",
        "turn_knob_unlock_drawer",
    }
    lock_names = {
        "turn_combination_lock_knob_lock",
        "turn_lock_knob_locked",
        "lock_drawer_with_combination_lock",
        "turn_knob_lock_drawer",
    }
    if action.name not in unlock_names | lock_names:
        return action
    hand = unary_var(action.pre, {"hand"})
    knob = unary_var(action.pre, {"knob"})
    drawer = unary_var(action.pre, {"drawer"})
    if None in {hand, knob, drawer}:
        raise ValueError(f"cannot infer drawer-lock roles in {action.name}")
    unlocking = action.name in unlock_names
    hand, knob, drawer = "?h", "?k", "?d"
    action.name = (
        "turn_knob_unlock_drawer" if unlocking else "turn_knob_lock_drawer"
    )
    action.params = [hand, knob, drawer]
    action.pre = make_and([
        ["hand", hand], ["knob", knob], ["drawer", drawer],
        ["hand_free", hand], ["mounted_on", knob, drawer],
        *([] if unlocking else [["closed", drawer]]),
        ["locked" if unlocking else "unlocked", drawer],
    ])
    action.eff = make_and([
        ["not", ["locked" if unlocking else "unlocked", drawer]],
        ["unlocked" if unlocking else "locked", drawer],
    ])
    action.comment = (
        "; Turn knob ?k to unlock drawer ?d with hand ?h."
        if unlocking
        else "; Turn knob ?k to lock drawer ?d with hand ?h."
    )
    return action


def _drawer_mount_problem(problem_text: str, plan_text: str) -> str:
    problem = parse_problem(problem_text)
    init = problem_facts(problem, ":init")
    knobs = set(typed_objects(problem, "knob"))
    drawers = set(typed_objects(problem, "drawer"))
    pairs = {
        (str(fact[1]), str(fact[2]))
        for fact in init
        if isinstance(fact, list)
        and len(fact) == 3
        and fact[0] in {"on", "mounted_on"}
        and fact[1] in knobs
        and fact[2] in drawers
    }
    old_order = {
        "turn_combination_lock_knob_unlock": (2, 3),
        "turn_combination_lock_knob_lock": (2, 3),
        "turn_lock_knob_unlocked": (2, 3),
        "turn_lock_knob_locked": (2, 3),
        "unlock_drawer_with_combination_lock": (3, 2),
        "lock_drawer_with_combination_lock": (3, 2),
        "turn_knob_unlock_drawer": (2, 3),
        "turn_knob_lock_drawer": (2, 3),
    }
    for line in plan_text.splitlines():
        tokens = line.strip().strip("()").split() if line.strip().startswith("(") else []
        positions = old_order.get(tokens[0]) if tokens else None
        if positions and len(tokens) > max(positions):
            pairs.add((tokens[positions[0]], tokens[positions[1]]))
    if not pairs and len(knobs) == len(drawers) == 1:
        pairs.add((next(iter(knobs)), next(iter(drawers))))
    if not pairs:
        raise ValueError("cannot infer any knob-drawer mounting relation")
    init = [
        fact for fact in init
        if not (
            isinstance(fact, list)
            and len(fact) == 3
            and fact[0] == "on"
            and (str(fact[1]), str(fact[2])) in pairs
        )
    ]
    init.extend(["mounted_on", knob, drawer] for knob, drawer in sorted(pairs))
    set_problem_facts(problem, ":init", _unique_nodes(init))
    return render_problem(problem)


def _approved_washing_action(action: Action) -> Action:
    machine = unary_var(action.pre, {"washing_machine"})
    if machine is None:
        return action
    if action.name == "start_washing_machine":
        action.pre = add_literal(action.pre, ["is_off", machine])
        effects = [
            literal for literal in conjunction(action.eff)
            if literal != ["is_off", machine]
        ]
        action.eff = make_and(_unique_nodes([
            *effects, ["not", ["is_off", machine]], ["started", machine],
        ]))
        return action
    if (
        "wash_clothes_in_washing_machine" in action.name
        and has_literal(action.pre, "started", machine)
    ):
        action.eff = make_and(_unique_nodes([
            *conjunction(action.eff), ["not", ["started", machine]],
            ["is_off", machine],
        ]))
    return action


def _normalize_washing_completion_goal(
    domain: str, problem_text: str
) -> str:
    has_completion = any(
        unary_var(action.pre, {"washing_machine"}) is not None
        and any(
            isinstance(literal, list)
            and literal
            and literal[0] in {"clean", "washed"}
            for literal in conjunction(action.eff)
        )
        and unary_var(action.pre, {"started"}) is not None
        for action in domain_actions(domain).values()
    )
    if not has_completion:
        return problem_text
    problem = parse_problem(problem_text)
    goal = problem_facts(problem, ":goal")
    if not any(
        isinstance(fact, list) and fact and fact[0] in {"clean", "washed"}
        for fact in goal
    ):
        return problem_text
    started_machines = [
        str(fact[1]) for fact in goal
        if isinstance(fact, list) and len(fact) == 2 and fact[0] == "started"
    ]
    if not started_machines:
        return problem_text
    goal = [
        fact for fact in goal
        if not (
            isinstance(fact, list)
            and len(fact) == 2
            and fact[0] == "started"
            and str(fact[1]) in started_machines
        )
    ]
    goal.extend(["is_off", machine] for machine in started_machines)
    set_problem_facts(problem, ":goal", _unique_nodes(goal))
    return render_problem(problem)


def _approved_kettle_pick(action: Action, tid: int) -> Action:
    if not action.name.startswith("pick_kettle_from_box_top_under_water_pump"):
        return action
    hand = unary_var(action.pre, {"hand"})
    kettle = unary_var(action.pre, {"kettle"})
    box = unary_var(action.pre, {"box"})
    pump = unary_var(action.pre, {"water_pump"})
    if None in {hand, kettle, box, pump}:
        raise ValueError(f"cannot infer kettle-pick roles in {action.name}")
    assert hand is not None and kettle is not None and box is not None and pump is not None
    action.name = "pick_kettle_from_box_top_under_water_pump"
    action.params = [hand, kettle, box, pump]
    action.pre = make_and([
        ["hand", hand], ["kettle", kettle], ["box", box],
        ["water_pump", pump], ["hand_free", hand], ["on", kettle, box],
        *([["closed", kettle]] if tid != 276 else []),
        ["is_off", pump], ["under", kettle, pump],
    ])
    action.eff = make_and([
        ["not", ["hand_free", hand]], ["holding", hand, kettle],
        ["not", ["on", kettle, box]], ["not", ["under", kettle, pump]],
    ])
    action.comment = "; Pick kettle ?k from box top ?b after stopping water pump ?p."
    return action


def _approved_lid_placement(action: Action) -> Action:
    if action.name not in {"place_lid_on_kettle", "place_kettle_lid_on_kettle"}:
        return action
    hand = unary_var(action.pre, {"hand"})
    lid = unary_var(action.pre, {"lid"})
    kettle = unary_var(action.pre, {"kettle"})
    if None in {hand, lid, kettle}:
        raise ValueError(f"cannot infer lid-placement roles in {action.name}")
    assert hand is not None and lid is not None and kettle is not None
    action.name = "place_lid_on_kettle"
    action.params = [hand, lid, kettle]
    action.pre = make_and([
        ["hand", hand], ["lid", lid], ["kettle", kettle],
        ["holding", hand, lid], ["open", kettle],
    ])
    action.eff = make_and([
        ["not", ["holding", hand, lid]], ["hand_free", hand],
        ["not", ["open", kettle]], ["closed", kettle], ["on", lid, kettle],
    ])
    action.comment = "; Place lid ?l on kettle ?k with hand ?h."
    return action


def apply_user_approved_fixes(
    domain: str,
    problem_text: str,
    plan_text: str,
    *,
    tid: int,
    dataset: str,
) -> tuple[str, str, str, list[str]]:
    """Apply only the operator changes explicitly approved in the current audit."""
    notes: list[str] = []
    domain = replace_predicate_heads(domain, {"off": "is_off"})
    problem_text = replace_predicate_heads(problem_text, {"off": "is_off"})

    if tid in APPROVED_DRAWER_TASKS:
        aliases = {
            "combination_lock_knob": "knob",
            "lock_knob": "knob",
            "combination_lock": "knob",
        }
        domain = replace_predicate_heads(domain, aliases)
        problem_text = replace_predicate_heads(problem_text, aliases)
        problem_text = _drawer_mount_problem(problem_text, plan_text)
        domain, edits = rewrite_actions(domain, _canonical_drawer_lock_action)
        domain = rewrite_predicates(
            domain, ensure=[["mounted_on", "?k", "?d"]]
        )
        notes.append("canonical_drawer_lock")

    if dataset == "human_aug" and tid in APPROVED_WASHING_TASKS:
        domain, edits = rewrite_actions(domain, _approved_washing_action)
        plan_text = rewrite_plan(plan_text, edits)
        problem_text = _normalize_washing_completion_goal(domain, problem_text)
        if "start_washing_machine" in domain_actions(domain):
            problem_text = _ensure_initial_state(
                problem_text,
                object_kind="washing_machine",
                state_predicate="is_off",
                conflicting_states={"is_on", "started"},
            )
        notes.append("washing_machine_lifecycle")

    if dataset == "human_aug" and tid in APPROVED_KETTLE_TASKS:
        aliases = {"glass_kettle": "kettle", "electric_kettle": "kettle"}
        domain = replace_predicate_heads(domain, aliases)
        problem_text = replace_predicate_heads(problem_text, aliases)
        domain, edits = rewrite_actions(
            domain, lambda action: _approved_kettle_pick(action, tid)
        )
        plan_text = rewrite_plan(plan_text, edits)
        if any(
            name.startswith("pick_kettle_from_box_top_under_water_pump")
            for name in domain_actions(domain)
        ):
            problem_text = _ensure_initial_state(
                problem_text,
                object_kind="water_pump",
                state_predicate="is_off",
                conflicting_states={"is_on"},
            )
        notes.append("minimal_kettle_pick")

    if dataset == "human_aug" and tid in APPROVED_LID_TASKS:
        domain = replace_predicate_heads(domain, {"kettle_lid": "lid"})
        problem_text = replace_predicate_heads(problem_text, {"kettle_lid": "lid"})
        domain, edits = rewrite_actions(domain, _approved_lid_placement)
        plan_text = rewrite_plan(plan_text, edits)
        notes.append("minimal_lid_placement")

    if dataset == "human_aug":
        domain, problem_text, removed = _remove_approved_init_only_predicates(
            domain, problem_text
        )
        if removed:
            notes.append("remove_init_only_predicates")

    domain = ensure_action_predicates_declared(domain)
    domain = rewrite_predicates(
        domain,
        remove_names={
            "off", "combination_lock_knob", "lock_knob", "combination_lock",
            "kettle_lid", "glass_kettle", "electric_kettle",
        },
    )
    plan_text = normalize_plan_schema_arity(domain, problem_text, plan_text)
    return domain, problem_text, plan_text, notes


def clean_round_user_approved(
    round_dir: Path,
) -> tuple[dict[str, str], list[str]]:
    domain_path = round_dir / "domain.pddl"
    problem_path = round_dir / "problem.pddl"
    plan_path = round_dir / "plan.txt"
    original_domain = domain_path.read_text(encoding="utf-8")
    original_problem = problem_path.read_text(encoding="utf-8")
    original_plan = plan_path.read_text(encoding="utf-8")
    domain, problem_text, _projected_plan, notes = apply_user_approved_fixes(
        original_domain,
        original_problem,
        original_plan,
        tid=task_id(round_dir),
        dataset=dataset_name(round_dir),
    )
    changed = {}
    if domain != original_domain:
        changed["domain.pddl"] = domain
    if problem_text != original_problem:
        changed["problem.pddl"] = problem_text
    return changed, notes if changed else []


def clean_round(
    round_dir: Path,
    input_mapping_dir: Path | None = None,
) -> tuple[dict[str, str], list[str]]:
    tid = task_id(round_dir)
    domain_path = resolve_input_file(round_dir, "domain.pddl", input_mapping_dir)
    problem_path = resolve_input_file(round_dir, "problem.pddl", input_mapping_dir)
    plan_path = resolve_input_file(round_dir, "plan.txt", input_mapping_dir)
    domain = domain_path.read_text(encoding="utf-8")
    problem_text = problem_path.read_text(encoding="utf-8")
    plan_text = plan_path.read_text(encoding="utf-8")
    original = {
        "domain.pddl": domain,
        "problem.pddl": problem_text,
        "plan.txt": plan_text,
    }
    notes: list[str] = []

    episode = int(round_dir.parent.name.removeprefix("episode_"))
    if tid == 297 and dataset_name(round_dir) == "human_aug" and episode in KEY_SET_AUG_EPISODES:
        domain, problem_text, plan_text = canonical_key_set_round(
            domain,
            problem_text,
            plan_text,
            drop_leading_pick=episode == 7,
        )
        notes.append("key_set_contract")
    else:
        if tid == 88:
            domain, problem_text, plan_text = normalize_laptop_task_88(domain, problem_text, plan_text)

        # Normalize a generated category spelling before any role inference.
        # The object identity stays unchanged; only its unary type predicate is
        # canonicalized in both domain and problem.
        hand_aliases = {"robot_hand": "hand"}
        domain = replace_predicate_heads(domain, hand_aliases)
        problem_text = replace_predicate_heads(problem_text, hand_aliases)
        closure_aliases = {
            "bottle_cap": "cap",
            "detergent_cap": "cap",
            "kettle_lid": "lid",
        }
        domain = replace_predicate_heads(domain, closure_aliases)
        problem_text = replace_predicate_heads(problem_text, closure_aliases)
        if tid in {270, 271, 272, 276, 277, 278}:
            kettle_aliases = {"glass_kettle": "kettle", "electric_kettle": "kettle"}
            domain = replace_predicate_heads(domain, kettle_aliases)
            problem_text = replace_predicate_heads(problem_text, kettle_aliases)
        plan_text = normalize_plan_schema_arity(domain, problem_text, plan_text)
        domain = restore_grounded_pick_source_types(domain, problem_text, plan_text)

        if tid in {14, 261, 262, 263}:
            washing_aliases = {
                "program_selected": "cycle_selected",
                "washing": "started",
                "running": "started",
                "wash_cycle_dial": "dial",
                "cycle_dial": "dial",
            }
            domain = replace_predicate_heads(domain, washing_aliases)
            problem_text = replace_predicate_heads(problem_text, washing_aliases)
            domain, washing_expansions = split_washing_start_completion(domain)
            if washing_expansions:
                plan_text = expand_plan_actions(plan_text, washing_expansions)
                notes.append("split_washing_start_completion")

        if tid in {6, 7, 16}:
            aliases = {"lid": "cap", "bottle_cap": "cap"}
            domain = replace_predicate_heads(domain, aliases)
            problem_text = replace_predicate_heads(problem_text, aliases)

        if tid == 291:
            liquid_aliases = {
                "has_liquid": "has_water",
                "received_spoonful": "has_water",
            }
            domain = replace_predicate_heads(domain, liquid_aliases)
            problem_text = normalize_liquid_problem_predicates(
                problem_text, normalize_spoonful=True, normalize_empty=False
            )
        if tid in {104, 105, 106, 107, 108}:
            empty_aliases = {"poured": "empty", "drained": "empty"}
            domain = replace_predicate_heads(domain, empty_aliases)
            problem_text = normalize_liquid_problem_predicates(
                problem_text, normalize_spoonful=False, normalize_empty=True
            )

        domain, problem_text = normalize_drawer_clearance_predicate(
            domain, problem_text
        )

        infinite_changed = False
        legacy_dispense_bindings = source_dispense_bindings(domain)
        legacy_dispense_pairs = ground_source_dispense_pairs(
            domain, plan_text, legacy_dispense_bindings
        )
        direct_microwave_relation = infer_direct_microwave_relation(domain)
        pending_pose_effects: dict[str, list[tuple[bool, str]]] = {}
        removed_pose_actions: set[str] = set()

        bowl_rewritten_domain, bowl_edits = rewrite_actions(
            domain, _canonical_bowl_stack_action
        )
        migrated_domain, migrated_problem = migrate_bowl_stack_contract(
            domain, problem_text
        )
        if migrated_domain != domain or migrated_problem != problem_text:
            notes.append("bowl_stack_on_contract")
            domain, problem_text = migrated_domain, migrated_problem
            if bowl_rewritten_domain != domain and not bowl_edits:
                raise ValueError("bowl migration produced inconsistent domain rewrites")
            plan_text = rewrite_plan(plan_text, bowl_edits)

        domain, plan_expansions = expand_pick_place_macros(domain)
        if plan_expansions:
            plan_text = expand_plan_actions(plan_text, plan_expansions)
            notes.extend(f"expand:{name}" for name in sorted(plan_expansions))

        if tid in {270, 271, 272, 276, 277, 278}:
            water_legacy_domain = domain
            domain, water_edits = normalize_water_supply_contracts(domain)
            if water_edits:
                plan_text = rewrite_water_supply_plan(
                    water_legacy_domain, domain, plan_text, water_edits
                )
                notes.append("water_supply_contract")

        clear_repair_families = stacking_clear_repair_families(domain)
        paired_press_kinds = held_press_control_kinds(domain)
        if clear_repair_families:
            notes.extend(
                f"stacking_clear_contract:{family}"
                for family in sorted(clear_repair_families)
            )

        def transform(action: Action) -> Action | None:
            nonlocal infinite_changed
            old = action.name
            action = normalize_pick_up(
                action,
                preserve_container_state=tid in {113, 116, 270, 271, 272, 276, 277, 278, 291, 292, 293},
            )
            action = normalize_place_hand_interface(action)
            action = restore_named_manipulation_type(action)
            action = misc_transform(action, tid)
            action = _canonical_bowl_stack_action(action)
            for family in sorted(clear_repair_families):
                action = canonical_family_clear_action(action, family)
            action = preserve_pick_pose(action, tid, pending_pose_effects)
            oriented = orientation_transform(action, tid, pending_pose_effects)
            if oriented is None:
                notes.append(f"remove:{old}")
                removed_pose_actions.add(old)
                return None
            action = oriented
            action = microwave_open_transform(action)
            action = microwave_door_transform(action)
            action = microwave_contract_transform(action, direct_microwave_relation)
            action = canonical_turntable_pick(action)
            action = washing_machine_contract_transform(action)
            action = connected_lid_transform(action, tid)
            action, threaded = threaded_cap_transform(action, tid)
            if not threaded:
                action = separable_lid_transform(action, tid)
            action = canonical_named_cap_action(action)
            action = faucet_transform(action, tid)
            action = canonical_faucet_toggle(action)
            action = canonical_water_button_contract(action)
            action = canonical_child_lock_contract(action)
            action = canonical_simple_power_toggle(action)
            action = canonical_kettle_start_contract(action)
            action = canonical_detergent_drawer_contract(action)
            action = running_water_transform(action, tid)
            action = water_supply_action_contract(action)
            action = canonical_press_release_contract(action, paired_press_kinds)
            action, source_changed = infinite_source_action(action)
            action = canonical_finite_water_end_contract(action)
            action = finite_liquid_contract_transform(action)
            action = wipe_contract_transform(action)
            action, finite_changed = finite_transfer_action(action)
            action = normalize_processed_manipulation(action)
            action = canonical_power_connection_contract(action)
            action = normalize_spatial_action_contract(action)
            action = canonical_key_contract(action)
            action = drawer_contract_transform(action)
            action = canonical_pick_contract(action)
            action = canonical_place_contract(action)
            action = canonical_open_close_contract(action)
            action = normalize_relation_lexemes(action)
            # Some legacy power actions still use "into" in their names. Run
            # the contract after lexical normalization so those aliases cannot
            # retain a separate plugged_in/plugged_into predicate family.
            action = canonical_power_connection_contract(action)
            action = minimal_contrast_name_transform(action)
            infinite_changed = infinite_changed or source_changed
            if finite_changed:
                notes.append(f"finite_transfer:{action.name}")
            if action.name != old:
                notes.append(f"rename:{old}->{action.name}")
            return action

        domain, edits = rewrite_actions(domain, transform)
        plan_text = rewrite_plan(plan_text, edits)

        # A base-name projection can make a generated button action recognizable
        # as a microwave transition. Converge that contract in the same pass so
        # parameter projection and dead-button cleanup do not wait for a rerun.
        def converge_semantic_contract(action: Action) -> Action:
            action = microwave_contract_transform(
                action, direct_microwave_relation
            )
            return minimal_contrast_name_transform(action)

        domain, convergence_edits = rewrite_actions(
            domain, converge_semantic_contract
        )
        if convergence_edits:
            plan_text = rewrite_plan(plan_text, convergence_edits)
        if removed_pose_actions:
            plan_text = remove_plan_actions(plan_text, removed_pose_actions)

        if tid == 275:
            domain, problem_text, plan_text = normalize_task_275_bookkeeping(
                domain, problem_text, plan_text
            )
            notes.append("remove_task_275_bookkeeping")

        for family in sorted(clear_repair_families):
            domain = rewrite_predicates(domain, ensure=[["clear", "?o"]])
            problem_text = normalize_family_clear_init(problem_text, family)
        if tid == 275 and dataset_name(round_dir) == "human_aug" and episode == 98:
            plan_text = canonical_task_275_episode_98_plan(domain, plan_text)
            notes.append("canonical_task_275_episode_98_plan")

        domain, problem_text, plan_text, early_pose_changed = normalize_early_pick_pose_contract(
            domain, problem_text, plan_text, tid
        )
        if early_pose_changed:
            notes.append("pick_pose_contract")

        domain, problem_text, plan_text, rinsed_pick_changed = canonical_rinsed_bowl_pick(
            domain, problem_text, plan_text, tid
        )
        if rinsed_pick_changed:
            notes.append("rinsed_bowl_pick_contract")

        domain, problem_text, source_problem_changed = normalize_infinite_sources(
            domain, problem_text, plan_text, legacy_dispense_pairs
        )
        domain, problem_text, plan_text = split_refill_liquid_identity(
            domain, problem_text, plan_text
        )
        if infinite_changed or source_problem_changed:
            domain = rewrite_predicates(domain, ensure=[["dispenses", "?s", "?l"]])
            notes.append("infinite_source_dispenses")

        if any(
            name.startswith(("turn_on_microwave", "turn_off_microwave"))
            for name in domain_actions(domain)
        ):
            domain, problem_text = remove_dead_microwave_buttons(domain, problem_text)
            problem_text = ensure_microwave_heating_type_facts(
                domain, problem_text, plan_text
            )

        if any(
            re.fullmatch(r"turn_(?:on|off)_(?:hot|cold)_water_button.*", name)
            or name in {"lock_child_lock", "unlock_child_lock"}
            for name in domain_actions(domain)
        ):
            problem_text = ensure_water_button_type_facts(
                domain, problem_text, plan_text
            )

        if any(
            name in {
                "insert_plug_in_wall_outlet", "plug_plug_in_wall_outlet",
                "plug_power_base_cord_in_wall_outlet", "insert_plug_in_outlet",
                "plug_power_base_cord_in_outlet", "turn_on_kettle_when_plug_inserted",
            }
            for name in domain_actions(domain)
        ):
            problem_text = ensure_power_connection_type_facts(
                domain, problem_text, plan_text
            )
            problem_text = normalize_power_connection_problem_predicates(problem_text)

        if "pick_paper_cup_from_microwave_turntable" in domain_actions(domain):
            problem_text = ensure_turntable_type_facts(
                domain, problem_text, plan_text
            )

        problem_text = normalize_initial_pressed_control_interface(domain, problem_text)

        if tid == 60:
            domain, problem_text = normalize_task_60(domain, problem_text)
            notes.append("box_open_closed")

        domain = ensure_action_predicates_declared(domain)
        problem_text = normalize_problem_spatial_contract(problem_text)
        plan_text = normalize_declared_plan_objects(problem_text, plan_text)
        problem_text = normalize_closure_problem_state(problem_text)
        domain, problem_text = remove_redundant_lid_clear_goals(domain, problem_text)
        domain = remove_unused_named_predicates(
            domain, problem_text, {"dispensing"}
        )
        problem_text = normalize_problem_state_for_plan(problem_text, plan_text)
        plan_text = schedule_blocking_release(domain, problem_text, plan_text)
        repaired_plan = repair_plan_hand_interfaces(domain, problem_text, plan_text)
        if repaired_plan != plan_text:
            notes.append("repair_hand_interface_plan")
            plan_text = repaired_plan

        repaired_problem, repaired_plan = repair_audited_round_anomalies(
            problem_text,
            plan_text,
            tid=tid,
            dataset=dataset_name(round_dir),
            episode=episode,
        )
        if repaired_problem != problem_text or repaired_plan != plan_text:
            notes.append("repair_audited_round_anomaly")
            problem_text, plan_text = repaired_problem, repaired_plan
            plan_text = rewrite_plan(plan_text, edits)

    # Specialized round normalizers can bypass the main action transform chain.
    # Project every emitted domain through the same final naming policy.
    domain, final_name_edits = rewrite_actions(
        domain, minimal_contrast_name_transform
    )
    if final_name_edits:
        plan_text = rewrite_plan(plan_text, final_name_edits)
        notes.extend(
            f"rename:{old}->{edit.new_name}"
            for old, edit in sorted(final_name_edits.items())
            if edit.new_name is not None and edit.new_name != old
        )

    domain, problem_text, plan_text, requested_notes = apply_requested_contract_fixes(
        domain,
        problem_text,
        plan_text,
        tid=tid,
        dataset=dataset_name(round_dir),
        episode=episode,
    )
    notes.extend(requested_notes)

    domain, problem_text, plan_text, drawer_notes = simplify_drawer_role_predicates(
        domain, problem_text, plan_text
    )
    notes.extend(drawer_notes)

    domain, problem_text, plan_text, pose_notes = simplify_non_bowl_pose_contracts(
        domain, problem_text, plan_text, tid=tid
    )
    notes.extend(pose_notes)

    result = {
        "domain.pddl": domain,
        "problem.pddl": problem_text,
        "plan.txt": plan_text,
    }
    changed = {name: text for name, text in result.items() if text != original[name]}
    return changed, sorted(set(notes))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stage_changes(changes: dict[Path, dict[str, str]], root: Path) -> None:
    for round_dir, files in changes.items():
        relative = round_dir.relative_to(PROJECT_ROOT)
        target_dir = root / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, text in files.items():
            (target_dir / filename).write_text(text, encoding="utf-8")


APPROVED_PLAN_NAME_ALIASES = {
    "turn_combination_lock_knob_unlock": "turn_knob_unlock_drawer",
    "turn_lock_knob_unlocked": "turn_knob_unlock_drawer",
    "unlock_drawer_with_combination_lock": "turn_knob_unlock_drawer",
    "turn_combination_lock_knob_lock": "turn_knob_lock_drawer",
    "turn_lock_knob_locked": "turn_knob_lock_drawer",
    "lock_drawer_with_combination_lock": "turn_knob_lock_drawer",
    "place_kettle_lid_on_kettle": "place_lid_on_kettle",
}


def _normalized_plan_actions(plan_text: str) -> list[str]:
    names: list[str] = []
    for line in plan_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("("):
            continue
        tokens = stripped.strip("()").split()
        if not tokens:
            continue
        name = tokens[0].lower()
        if name.startswith("pick_kettle_from_box_top_under_water_pump"):
            name = "pick_kettle_from_box_top_under_water_pump"
        names.append(APPROVED_PLAN_NAME_ALIASES.get(name, name))
    return names


def _critical_process_actions(names: list[str]) -> list[str]:
    markers = (
        "knob_", "drawer", "water_pump", "lid_on_kettle",
        "start_washing_machine", "wash_clothes_in_washing_machine",
    )
    return [name for name in names if any(marker in name for marker in markers)]


def solve_user_approved_changes(
    changes: dict[Path, dict[str, str]],
    *,
    max_workers: int,
) -> list[dict[str, str]]:
    """Genuinely solve every changed episode and reject semantic shortcuts."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from swm.pddl.planner import solve_pddl

    def solve_one(
        item: tuple[Path, dict[str, str]],
    ) -> tuple[Path, str | None, str | None]:
        round_dir, files = item
        domain_text = files.get(
            "domain.pddl", (round_dir / "domain.pddl").read_text(encoding="utf-8")
        )
        problem_text = files.get(
            "problem.pddl", (round_dir / "problem.pddl").read_text(encoding="utf-8")
        )
        old_plan = (round_dir / "plan.txt").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="swm_approved_solve_") as directory:
            candidate_dir = Path(directory)
            domain_path = candidate_dir / "domain.pddl"
            problem_path = candidate_dir / "problem.pddl"
            domain_path.write_text(domain_text, encoding="utf-8")
            problem_path.write_text(problem_text, encoding="utf-8")
            if not solve_pddl(domain_path, problem_path, reorder_plan=False):
                error_path = candidate_dir / "error.log"
                detail = (
                    error_path.read_text(encoding="utf-8", errors="replace")
                    if error_path.is_file()
                    else "planner returned no diagnostic"
                )
                return round_dir, None, "solve failed: " + re.sub(r"\s+", " ", detail)[:500]
            candidate_plan = (candidate_dir / "plan.txt").read_text(encoding="utf-8")

        old_names = _normalized_plan_actions(old_plan)
        new_names = _normalized_plan_actions(candidate_plan)
        if Counter(old_names) != Counter(new_names):
            removed = Counter(old_names) - Counter(new_names)
            added = Counter(new_names) - Counter(old_names)
            return round_dir, None, (
                f"new plan is not action-equivalent; removed={dict(removed)}, "
                f"added={dict(added)}"
            )
        if task_id(round_dir) == 276:
            required = [
                "turn_off_water_pump_after_filling",
                "pick_kettle_from_box_top_under_water_pump",
                "place_kettle_on_power_base",
                "place_lid_on_kettle",
            ]
            positions = [new_names.index(name) for name in required if name in new_names]
            if positions != sorted(positions):
                return round_dir, None, (
                    "task 276 plan differs from kf process order; "
                    f"required={required}, positions={positions}"
                )
        else:
            old_critical = _critical_process_actions(old_names)
            new_critical = _critical_process_actions(new_names)
            if old_critical != new_critical:
                return round_dir, None, (
                    "critical process order changed; "
                    f"old={old_critical}, new={new_critical}"
                )
        return round_dir, candidate_plan, None

    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(solve_one, item) for item in changes.items()]
        for future in concurrent.futures.as_completed(futures):
            round_dir, candidate_plan, error = future.result()
            if error is not None:
                failures.append({
                    "round": str(round_dir.relative_to(PROJECT_ROOT)),
                    "error": error,
                })
                continue
            assert candidate_plan is not None
            changes[round_dir]["plan.txt"] = candidate_plan
    return sorted(failures, key=lambda item: item["round"])


def validate_staged(
    changes: dict[Path, dict[str, str]],
    datasets: list[str],
    selected_tasks: set[int] | None,
    staging_root: Path,
) -> list[dict[str, str]]:
    import validate_operator_cleanup as validator

    resolver = validator.CandidateResolver(staging_root)
    failures: list[dict[str, str]] = []
    changed_rounds = set(changes)
    regression_codes = {
        "missing_file", "parse_or_declaration", "plan_grounding",
        "action_not_applicable", "goal_not_satisfied", "state_invariant",
    }

    def issue_key(item: object) -> tuple[str, str]:
        code = str(getattr(item, "code"))
        detail = str(getattr(item, "detail"))
        if code == "state_invariant":
            detail = detail.split(": ", 1)[-1]
        if code == "parse_or_declaration":
            detail = re.sub(r"^.*?: (?=(?:init|goal|action|predicate|problem|domain)[: ])", "", detail)
        return code, detail

    for dataset in datasets:
        for round_dir in latest_rounds(EVAL_ROOT / dataset, selected_tasks):
            if round_dir not in changed_rounds:
                continue
            issues = validator.validate_round(round_dir, resolver)
            baseline_issues = validator.validate_round(
                round_dir, validator.CandidateResolver(None)
            )
            baseline_counts = Counter(
                issue_key(item) for item in baseline_issues if item.code in regression_codes
            )
            candidate_counts = Counter(
                issue_key(item) for item in issues if item.code in regression_codes
            )
            regressions = candidate_counts - baseline_counts
            fatal: list[object] = []
            for item in issues:
                key = issue_key(item)
                if regressions[key] > 0:
                    fatal.append(item)
                    regressions[key] -= 1
            if fatal:
                failures.append({
                    "round": str(round_dir.relative_to(PROJECT_ROOT)),
                    "error": "; ".join(
                        f"{item.code} at step {item.step or '-'}"
                        f" {item.action or ''}: {item.detail}" for item in fatal[:3]
                    ),
                })
    return failures


def atomic_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["human", "human_aug", "both"], default="both")
    parser.add_argument("--apply", action="store_true", help="write changes; default is dry-run")
    parser.add_argument(
        "--user-approved-fixes-only",
        action="store_true",
        help=(
            "apply only the approved drawer, washing, kettle, lid, off-state, "
            "and init-only predicate fixes, then genuinely re-solve every change"
        ),
    )
    parser.add_argument(
        "--solve-workers", type=int, default=8,
        help="parallel Fast Downward workers for --user-approved-fixes-only",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip staged STRIPS validation (not allowed with --apply)",
    )
    parser.add_argument("--task", type=int, action="append", dest="tasks")
    parser.add_argument(
        "--input-mapping-dir",
        type=Path,
        help="optional sparse candidate tree to use as cleanup input",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        help=(
            "persist sparse candidate files here for independent validation "
            "and contract auditing; the directory must be empty"
        ),
    )
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "operator_cleanup_report.json")
    args = parser.parse_args()
    if args.apply and args.skip_validation:
        parser.error("--apply cannot be combined with --skip-validation")
    if args.apply and args.input_mapping_dir is not None:
        parser.error("--apply cannot be combined with --input-mapping-dir")
    if args.user_approved_fixes_only and args.input_mapping_dir is not None:
        parser.error("--user-approved-fixes-only cannot use --input-mapping-dir")
    if args.solve_workers < 1:
        parser.error("--solve-workers must be positive")
    input_mapping_dir = (
        args.input_mapping_dir.resolve()
        if args.input_mapping_dir is not None else None
    )
    if input_mapping_dir is not None and not input_mapping_dir.is_dir():
        parser.error("--input-mapping-dir must be an existing directory")
    if args.staging_dir is not None:
        staging_root = args.staging_dir.resolve()
        if staging_root == PROJECT_ROOT or PROJECT_ROOT in staging_root.parents:
            parser.error("--staging-dir must be outside the project tree")
        if staging_root.exists() and any(staging_root.iterdir()):
            parser.error("--staging-dir must be empty")
        staging_root.mkdir(parents=True, exist_ok=True)
    else:
        staging_root = None

    datasets = ["human", "human_aug"] if args.dataset == "both" else [args.dataset]
    selected_tasks = set(args.tasks) if args.tasks else None
    report: dict[str, object] = {
        "applied": args.apply,
        "datasets": datasets,
        "staging_dir": str(staging_root) if staging_root is not None else None,
        "input_mapping_dir": str(input_mapping_dir) if input_mapping_dir else None,
        "user_approved_fixes_only": args.user_approved_fixes_only,
        "rounds": [],
    }
    changed_rounds = 0
    changed_files = 0
    errors: list[dict[str, str]] = []
    all_changes: dict[Path, dict[str, str]] = {}

    for dataset in datasets:
        for round_dir in latest_rounds(EVAL_ROOT / dataset, selected_tasks):
            try:
                if args.user_approved_fixes_only:
                    changed, notes = clean_round_user_approved(round_dir)
                else:
                    changed, notes = clean_round(round_dir, input_mapping_dir)
                if not changed:
                    continue
                changed_rounds += 1
                changed_files += len(changed)
                all_changes[round_dir] = changed
                entry = {
                    "round": str(round_dir.relative_to(PROJECT_ROOT)),
                    "files": sorted(changed),
                    "notes": notes,
                    "before": {},
                    "after": {},
                }
                for filename, new_text in changed.items():
                    path = round_dir / filename
                    old_path = resolve_input_file(
                        round_dir, filename, input_mapping_dir
                    )
                    old_text = old_path.read_text(encoding="utf-8")
                    entry["before"][filename] = sha256(old_text)
                    entry["after"][filename] = sha256(new_text)
                report["rounds"].append(entry)
            except Exception as exc:  # keep a complete audit of heterogeneous augmented episodes
                errors.append({"round": str(round_dir.relative_to(PROJECT_ROOT)), "error": str(exc)})

    if all_changes and args.user_approved_fixes_only and not errors:
        errors.extend(solve_user_approved_changes(
            all_changes, max_workers=args.solve_workers
        ))

    entries_by_round = {
        str(entry["round"]): entry for entry in report["rounds"]
    }
    for round_dir, files in all_changes.items():
        relative = str(round_dir.relative_to(PROJECT_ROOT))
        entry = entries_by_round[relative]
        entry["files"] = sorted(files)
        entry["before"] = {}
        entry["after"] = {}
        for filename, new_text in files.items():
            old_path = resolve_input_file(round_dir, filename, input_mapping_dir)
            old_text = (
                old_path.read_text(encoding="utf-8") if old_path.is_file() else ""
            )
            entry["before"][filename] = sha256(old_text)
            entry["after"][filename] = sha256(new_text)
    changed_files = sum(len(files) for files in all_changes.values())

    validation_errors: list[dict[str, str]] = []
    if staging_root is not None and input_mapping_dir is not None:
        shutil.copytree(input_mapping_dir, staging_root, dirs_exist_ok=True)
    if all_changes and staging_root is not None:
        stage_changes(all_changes, staging_root)
    if all_changes and not errors and not args.skip_validation:
        if staging_root is not None:
            validation_errors = validate_staged(
                all_changes, datasets, selected_tasks, staging_root
            )
        else:
            with tempfile.TemporaryDirectory(prefix="swm_operator_cleanup_") as directory:
                temporary_staging_root = Path(directory)
                stage_changes(all_changes, temporary_staging_root)
                validation_errors = validate_staged(
                    all_changes, datasets, selected_tasks, temporary_staging_root
                )
    errors.extend(validation_errors)

    if args.apply and not errors:
        for round_dir, files in all_changes.items():
            for filename, new_text in files.items():
                atomic_write(round_dir / filename, new_text)

    report["summary"] = {
        "changed_rounds": changed_rounds,
        "changed_files": changed_files,
        "errors": len(errors),
        "validated": not args.skip_validation,
        "written": changed_files if args.apply and not errors else 0,
    }
    report["errors"] = errors
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"report: {args.report}")
    if errors:
        for error in errors[:20]:
            print(f"ERROR {error['round']}: {error['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
