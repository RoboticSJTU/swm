#!/usr/bin/env python3
"""Validate cleaned PDDL rounds without modifying the dataset.

The validator selects the highest-numbered round in every episode, parses its
domain/problem/plan, replays the grounded plan with STRIPS semantics, and checks
the final goal plus state invariants used by the operator cleanup.

An optional mapping directory may sparsely override candidate files.  For a
source file such as::

    eval_results/gpt-5.6-sol/human/task_60/episode_60/round1/domain.pddl

the resolver accepts any of these layouts below ``--mapping-dir``::

    eval_results/gpt-5.6-sol/human/task_60/episode_60/round1/domain.pddl
    human/task_60/episode_60/round1/domain.pddl
    task_60/episode_60/round1/domain.pddl

Missing overrides fall back to the source file, so sparse cleanup previews can
be validated before they are written back.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = PROJECT_ROOT / "eval_results" / "gpt-5.6-sol"
ROUND_RE = re.compile(r"roun(?:d)?[_-]?(\d+)$", re.IGNORECASE)
TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")

Literal = tuple[str, ...]
Node = str | list["Node"]


@dataclass(frozen=True)
class ActionSchema:
    name: str
    parameters: tuple[str, ...]
    positive_preconditions: frozenset[Literal]
    negative_preconditions: frozenset[Literal]
    add_effects: frozenset[Literal]
    delete_effects: frozenset[Literal]


@dataclass(frozen=True)
class GroundAction:
    name: str
    arguments: tuple[str, ...]
    positive_preconditions: frozenset[Literal]
    negative_preconditions: frozenset[Literal]
    add_effects: frozenset[Literal]
    delete_effects: frozenset[Literal]

    def render(self) -> str:
        suffix = " " + " ".join(self.arguments) if self.arguments else ""
        return f"({self.name}{suffix})"


@dataclass
class DomainModel:
    name: str
    predicate_arities: dict[str, int]
    actions: dict[str, ActionSchema]


@dataclass
class ProblemModel:
    domain_name: str
    objects: set[str]
    initial_state: set[Literal]
    positive_goals: set[Literal]
    negative_goals: set[Literal]


@dataclass
class ValidationIssue:
    round: str
    code: str
    detail: str
    step: int | None = None
    action: str | None = None


def strip_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def parse_sexp(text: str, source: Path) -> Node:
    tokens = [token.lower() for token in TOKEN_RE.findall(strip_comments(text))]
    position = 0

    def parse_one() -> Node:
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"{source}: unexpected end of S-expression")
        token = tokens[position]
        if token != "(":
            if token == ")":
                raise ValueError(f"{source}: unexpected closing parenthesis")
            position += 1
            return token
        position += 1
        result: list[Node] = []
        while position < len(tokens) and tokens[position] != ")":
            result.append(parse_one())
        if position >= len(tokens):
            raise ValueError(f"{source}: missing closing parenthesis")
        position += 1
        return result

    if not tokens:
        raise ValueError(f"{source}: empty PDDL file")
    root = parse_one()
    if position != len(tokens):
        raise ValueError(f"{source}: trailing S-expression tokens")
    return root


def require_list(node: Node, context: str) -> list[Node]:
    if not isinstance(node, list):
        raise ValueError(f"{context}: expected a list")
    return node


def strip_typed_items(node: Node, context: str) -> list[str]:
    items = require_list(node, context)
    output: list[str] = []
    index = 0
    while index < len(items):
        item = items[index]
        if not isinstance(item, str):
            raise ValueError(f"{context}: nested list in name sequence")
        if item == "-":
            raise ValueError(f"{context}: '-' has no preceding names")
        output.append(item)
        index += 1
        if index < len(items) and items[index] == "-":
            if index + 1 >= len(items) or not isinstance(items[index + 1], str):
                raise ValueError(f"{context}: invalid type annotation")
            index += 2
    return output


def literal(node: Node, context: str) -> Literal:
    atom = require_list(node, context)
    if not atom or not isinstance(atom[0], str):
        raise ValueError(f"{context}: invalid literal")
    if atom[0] in {"and", "not", "or", "when", "forall", "exists"}:
        raise ValueError(f"{context}: expected an atomic literal, got '{atom[0]}'")
    if any(not isinstance(argument, str) for argument in atom[1:]):
        raise ValueError(f"{context}: nested term in literal")
    return tuple(str(item) for item in atom)


def read_literals(node: Node, context: str) -> tuple[set[Literal], set[Literal]]:
    expression = require_list(node, context)
    if not expression:
        raise ValueError(f"{context}: empty expression")
    if expression[0] == "and":
        positive: set[Literal] = set()
        negative: set[Literal] = set()
        for index, child in enumerate(expression[1:], start=1):
            child_positive, child_negative = read_literals(child, f"{context}[{index}]")
            positive.update(child_positive)
            negative.update(child_negative)
        return positive, negative
    if expression[0] == "not":
        if len(expression) != 2:
            raise ValueError(f"{context}: not must have exactly one argument")
        return set(), {literal(expression[1], context)}
    if expression[0] in {"or", "when", "forall", "exists", "imply"}:
        raise ValueError(f"{context}: unsupported PDDL construct '{expression[0]}'")
    return {literal(expression, context)}, set()


def named_declaration(root: list[Node], keyword: str, source: Path) -> str:
    declarations = [
        item for item in root[1:]
        if isinstance(item, list) and item and item[0] == keyword
    ]
    if len(declarations) != 1 or len(declarations[0]) != 2 or not isinstance(declarations[0][1], str):
        raise ValueError(f"{source}: expected exactly one ({keyword} NAME) declaration")
    return str(declarations[0][1])


def validate_literal_signature(
    item: Literal,
    predicate_arities: dict[str, int],
    context: str,
) -> None:
    name = item[0]
    if name not in predicate_arities:
        raise ValueError(f"{context}: undeclared predicate '{name}'")
    actual = len(item) - 1
    expected = predicate_arities[name]
    if actual != expected:
        raise ValueError(
            f"{context}: predicate '{name}' expects {expected} arguments, got {actual}"
        )


def parse_domain(path: Path) -> DomainModel:
    root = require_list(parse_sexp(path.read_text(encoding="utf-8"), path), str(path))
    if not root or root[0] != "define":
        raise ValueError(f"{path}: expected (define ...)")
    name = named_declaration(root, "domain", path)

    predicate_sections = [
        item for item in root[1:]
        if isinstance(item, list) and item and item[0] == ":predicates"
    ]
    if len(predicate_sections) != 1:
        raise ValueError(f"{path}: expected exactly one :predicates section")
    predicate_arities: dict[str, int] = {}
    for index, declaration in enumerate(predicate_sections[0][1:], start=1):
        values = require_list(declaration, f"{path}: predicate {index}")
        if not values or not isinstance(values[0], str):
            raise ValueError(f"{path}: invalid predicate declaration {index}")
        predicate_name = values[0]
        arguments = strip_typed_items(values[1:], f"{path}: predicate {predicate_name}")
        if any(not argument.startswith("?") for argument in arguments):
            raise ValueError(f"{path}: predicate '{predicate_name}' has a non-variable argument")
        if predicate_name in predicate_arities:
            raise ValueError(f"{path}: duplicate predicate declaration '{predicate_name}'")
        predicate_arities[predicate_name] = len(arguments)

    actions: dict[str, ActionSchema] = {}
    for item in root[1:]:
        if not isinstance(item, list) or not item or item[0] != ":action":
            continue
        if len(item) < 2 or not isinstance(item[1], str):
            raise ValueError(f"{path}: action without a valid name")
        action_name = item[1]
        if action_name in actions:
            raise ValueError(f"{path}: duplicate action declaration '{action_name}'")
        fields: dict[str, Node] = {}
        index = 2
        while index < len(item):
            if index + 1 >= len(item) or not isinstance(item[index], str):
                raise ValueError(f"{path}: malformed fields in action '{action_name}'")
            field_name = str(item[index])
            if field_name in fields:
                raise ValueError(f"{path}: duplicate {field_name} in action '{action_name}'")
            fields[field_name] = item[index + 1]
            index += 2
        required = {":parameters", ":precondition", ":effect"}
        missing_fields = required - fields.keys()
        extra_fields = fields.keys() - required
        if missing_fields or extra_fields:
            raise ValueError(
                f"{path}: action '{action_name}' fields: "
                f"missing={sorted(missing_fields)}, unsupported={sorted(extra_fields)}"
            )
        parameters = strip_typed_items(
            fields[":parameters"], f"{path}: parameters of {action_name}"
        )
        if len(parameters) != len(set(parameters)):
            raise ValueError(f"{path}: duplicate parameter in action '{action_name}'")
        if any(not parameter.startswith("?") for parameter in parameters):
            raise ValueError(f"{path}: non-variable parameter in action '{action_name}'")
        positive_preconditions, negative_preconditions = read_literals(
            fields[":precondition"], f"{path}: precondition of {action_name}"
        )
        add_effects, delete_effects = read_literals(
            fields[":effect"], f"{path}: effect of {action_name}"
        )
        for group_name, literals in (
            ("precondition", positive_preconditions | negative_preconditions),
            ("effect", add_effects | delete_effects),
        ):
            for item_literal in literals:
                validate_literal_signature(
                    item_literal, predicate_arities,
                    f"{path}: {group_name} of {action_name}",
                )
                undeclared = {
                    token for token in item_literal[1:]
                    if token.startswith("?") and token not in parameters
                }
                if undeclared:
                    raise ValueError(
                        f"{path}: action '{action_name}' uses undeclared variables "
                        f"{sorted(undeclared)}"
                    )
        overlap = add_effects & delete_effects
        if overlap:
            raise ValueError(
                f"{path}: action '{action_name}' both adds and deletes {sorted(overlap)}"
            )
        actions[action_name] = ActionSchema(
            action_name,
            tuple(parameters),
            frozenset(positive_preconditions),
            frozenset(negative_preconditions),
            frozenset(add_effects),
            frozenset(delete_effects),
        )
    if not actions:
        raise ValueError(f"{path}: domain declares no actions")
    return DomainModel(name, predicate_arities, actions)


def find_section(root: list[Node], keyword: str, source: Path) -> list[Node]:
    matches = [
        item for item in root[1:]
        if isinstance(item, list) and item and item[0] == keyword
    ]
    if len(matches) != 1:
        raise ValueError(f"{source}: expected exactly one {keyword} section")
    return matches[0]


def parse_problem(path: Path, domain: DomainModel) -> ProblemModel:
    root = require_list(parse_sexp(path.read_text(encoding="utf-8"), path), str(path))
    if not root or root[0] != "define":
        raise ValueError(f"{path}: expected (define ...)")
    named_declaration(root, "problem", path)
    domain_section = find_section(root, ":domain", path)
    if len(domain_section) != 2 or not isinstance(domain_section[1], str):
        raise ValueError(f"{path}: invalid :domain section")
    domain_name = domain_section[1]
    if domain_name != domain.name:
        raise ValueError(
            f"{path}: problem names domain '{domain_name}', domain file names '{domain.name}'"
        )

    object_section = find_section(root, ":objects", path)
    objects = strip_typed_items(object_section[1:], f"{path}: objects")
    if len(objects) != len(set(objects)):
        raise ValueError(f"{path}: duplicate object declaration")
    object_set = set(objects)

    init_section = find_section(root, ":init", path)
    initial_state: set[Literal] = set()
    explicit_negative: set[Literal] = set()
    for index, fact in enumerate(init_section[1:], start=1):
        positive, negative = read_literals(fact, f"{path}: init fact {index}")
        if len(positive) + len(negative) != 1:
            raise ValueError(f"{path}: init fact {index} is not atomic")
        initial_state.update(positive)
        explicit_negative.update(negative)
    contradiction = initial_state & explicit_negative
    if contradiction:
        raise ValueError(f"{path}: contradictory init facts {sorted(contradiction)}")

    goal_section = find_section(root, ":goal", path)
    if len(goal_section) != 2:
        raise ValueError(f"{path}: invalid :goal section")
    positive_goals, negative_goals = read_literals(goal_section[1], f"{path}: goal")

    for context, literals in (
        ("init", initial_state | explicit_negative),
        ("goal", positive_goals | negative_goals),
    ):
        for item_literal in literals:
            validate_literal_signature(item_literal, domain.predicate_arities, f"{path}: {context}")
            undeclared_objects = set(item_literal[1:]) - object_set
            if undeclared_objects:
                raise ValueError(
                    f"{path}: {context} uses undeclared objects {sorted(undeclared_objects)}"
                )
    return ProblemModel(
        domain_name,
        object_set,
        initial_state,
        positive_goals,
        negative_goals,
    )


def parse_plan(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    actions: list[tuple[str, tuple[str, ...]]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split(";", 1)[0].strip().lower()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^\s*\d+(?:\.\d+)?\s*:\s*", "", line)
        line = re.sub(r"\s*\[\s*[\d.]+\s*\]\s*$", "", line)
        match = re.fullmatch(r"\(([^()]*)\)", line)
        if not match:
            raise ValueError(f"{path}:{line_number}: invalid plan line: {raw_line.strip()}")
        tokens = match.group(1).split()
        if not tokens:
            raise ValueError(f"{path}:{line_number}: empty plan action")
        actions.append((tokens[0], tuple(tokens[1:])))
    if not actions:
        raise ValueError(f"{path}: plan contains no actions")
    return actions


def ground_action(
    name: str,
    arguments: tuple[str, ...],
    domain: DomainModel,
    problem: ProblemModel,
) -> GroundAction:
    if name not in domain.actions:
        raise ValueError(f"action '{name}' does not exist in the domain")
    schema = domain.actions[name]
    if len(arguments) != len(schema.parameters):
        raise ValueError(
            f"action '{name}' expects {len(schema.parameters)} arguments, got {len(arguments)}"
        )
    undeclared = set(arguments) - problem.objects
    if undeclared:
        raise ValueError(f"action '{name}' uses undeclared objects {sorted(undeclared)}")
    bindings = dict(zip(schema.parameters, arguments))

    def substitute(items: Iterable[Literal]) -> frozenset[Literal]:
        grounded: set[Literal] = set()
        for item in items:
            result = tuple(bindings.get(token, token) for token in item)
            unresolved = [token for token in result[1:] if token.startswith("?")]
            if unresolved:
                raise ValueError(
                    f"action '{name}' leaves variables unbound: {sorted(set(unresolved))}"
                )
            grounded.add(result)
        return frozenset(grounded)

    return GroundAction(
        name,
        arguments,
        substitute(schema.positive_preconditions),
        substitute(schema.negative_preconditions),
        substitute(schema.add_effects),
        substitute(schema.delete_effects),
    )


LIQUID_TYPE_NAMES = {
    "beverage", "broth", "cleaner", "coffee", "content", "contents",
    "detergent", "disinfectant", "drink", "juice", "liquid", "milk",
    "oil", "soap", "solution", "soup", "tea", "water",
}
LIQUID_TYPE_SUFFIXES = ("_juice", "_liquid", "_oil", "_solution", "_water")


def is_liquid_type(predicate: str) -> bool:
    return predicate in LIQUID_TYPE_NAMES or predicate.endswith(LIQUID_TYPE_SUFFIXES)


def finite_liquid_objects(state: set[Literal]) -> set[str]:
    liquids = {
        item[1] for item in state
        if len(item) == 2 and is_liquid_type(item[0])
    }
    liquids.update(
        item[2] for item in state
        if len(item) == 3 and item[0] == "dispenses"
    )
    return liquids


def invariant_violations(state: set[Literal]) -> list[str]:
    violations: list[str] = []
    hand_free = {item[1] for item in state if len(item) == 2 and item[0] == "hand_free"}
    holding_hands = {item[1] for item in state if len(item) == 3 and item[0] == "holding"}
    for hand in sorted(hand_free & holding_hands):
        held = sorted(item[2] for item in state if len(item) == 3 and item[:2] == ("holding", hand))
        violations.append(f"hand_free/holding conflict for '{hand}', holding={held}")

    for positive_name, negative_name in (("open", "closed"), ("is_on", "is_off")):
        positive_objects = {
            item[1] for item in state if len(item) == 2 and item[0] == positive_name
        }
        negative_objects = {
            item[1] for item in state if len(item) == 2 and item[0] == negative_name
        }
        for obj in sorted(positive_objects & negative_objects):
            violations.append(f"{positive_name}/{negative_name} conflict for '{obj}'")

    liquid_owners: dict[str, set[str]] = defaultdict(set)
    liquids = finite_liquid_objects(state)
    for item in state:
        if len(item) == 3 and item[0] == "in" and item[1] in liquids:
            liquid_owners[item[1]].add(item[2])
    for liquid_name, owners in sorted(liquid_owners.items()):
        if len(owners) > 1:
            violations.append(
                f"finite liquid '{liquid_name}' is in multiple vessels {sorted(owners)}"
            )
    return violations


def latest_rounds(dataset_root: Path, task_ids: set[int] | None) -> list[Path]:
    output: list[Path] = []
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
            output.append(max(candidates, key=lambda pair: pair[0])[1])
    return output


class CandidateResolver:
    def __init__(self, mapping_dir: Path | None):
        self.mapping_dir = mapping_dir.resolve() if mapping_dir else None
        self.override_files = 0

    def resolve(self, source_round: Path, filename: str) -> Path:
        source = source_round / filename
        if self.mapping_dir is None:
            return source
        relative_candidates = [
            source.relative_to(PROJECT_ROOT),
            source.relative_to(EVAL_ROOT),
            source.relative_to(EVAL_ROOT / source_round.parents[2].name),
        ]
        for relative in relative_candidates:
            candidate = self.mapping_dir / relative
            if candidate.is_file():
                self.override_files += 1
                return candidate
        return source


def issue(
    round_dir: Path,
    code: str,
    detail: str,
    *,
    step: int | None = None,
    action: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        str(round_dir.relative_to(PROJECT_ROOT)), code, detail, step, action
    )


def validate_round(round_dir: Path, resolver: CandidateResolver) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        domain_path = resolver.resolve(round_dir, "domain.pddl")
        problem_path = resolver.resolve(round_dir, "problem.pddl")
        plan_path = resolver.resolve(round_dir, "plan.txt")
        for name, path in (
            ("domain.pddl", domain_path),
            ("problem.pddl", problem_path),
            ("plan.txt", plan_path),
        ):
            if not path.is_file():
                return [issue(round_dir, "missing_file", f"{name} not found at {path}")]
        domain = parse_domain(domain_path)
        problem = parse_problem(problem_path, domain)
        raw_plan = parse_plan(plan_path)
    except Exception as exc:
        return [issue(round_dir, "parse_or_declaration", str(exc))]

    grounded_plan: list[GroundAction] = []
    for step_number, (name, arguments) in enumerate(raw_plan, start=1):
        rendered = f"({name}{' ' if arguments else ''}{' '.join(arguments)})"
        try:
            grounded_plan.append(ground_action(name, arguments, domain, problem))
        except Exception as exc:
            return [
                issue(
                    round_dir, "plan_grounding", str(exc),
                    step=step_number, action=rendered,
                )
            ]

    state = set(problem.initial_state)
    for detail in invariant_violations(state):
        issues.append(issue(round_dir, "state_invariant", f"initial state: {detail}", step=0))

    replay_failed = False
    for step_number, action_model in enumerate(grounded_plan, start=1):
        missing = action_model.positive_preconditions - state
        violated = action_model.negative_preconditions & state
        if missing or violated:
            parts: list[str] = []
            if missing:
                parts.append(f"missing positive preconditions {sorted(missing)}")
            if violated:
                parts.append(f"violated negative preconditions {sorted(violated)}")
            issues.append(
                issue(
                    round_dir, "action_not_applicable", "; ".join(parts),
                    step=step_number, action=action_model.render(),
                )
            )
            replay_failed = True
            break
        state.difference_update(action_model.delete_effects)
        state.update(action_model.add_effects)
        for detail in invariant_violations(state):
            issues.append(
                issue(
                    round_dir, "state_invariant", f"after step {step_number}: {detail}",
                    step=step_number, action=action_model.render(),
                )
            )

    if not replay_failed:
        missing_goals = problem.positive_goals - state
        violated_goals = problem.negative_goals & state
        if missing_goals or violated_goals:
            detail_parts: list[str] = []
            if missing_goals:
                detail_parts.append(f"missing positive goals {sorted(missing_goals)}")
            if violated_goals:
                detail_parts.append(f"violated negative goals {sorted(violated_goals)}")
            issues.append(issue(round_dir, "goal_not_satisfied", "; ".join(detail_parts)))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("human", "human_aug", "both"), default="both",
        help="dataset to validate (default: both)",
    )
    parser.add_argument(
        "--task", type=int, action="append", dest="tasks",
        help="task id to validate; repeat for multiple tasks",
    )
    parser.add_argument(
        "--mapping-dir", type=Path,
        help="optional sparse directory containing candidate cleaned files",
    )
    parser.add_argument(
        "--report", type=Path,
        help="optional JSON report path; no report file is written by default",
    )
    parser.add_argument(
        "--show-errors", type=int, default=20,
        help="maximum errors printed to stdout (default: 20)",
    )
    args = parser.parse_args()

    if args.mapping_dir is not None and not args.mapping_dir.is_dir():
        parser.error(f"--mapping-dir is not a directory: {args.mapping_dir}")
    if args.show_errors < 0:
        parser.error("--show-errors must be non-negative")

    datasets = ("human", "human_aug") if args.dataset == "both" else (args.dataset,)
    selected_tasks = set(args.tasks) if args.tasks else None
    resolver = CandidateResolver(args.mapping_dir)
    rounds: list[Path] = []
    for dataset in datasets:
        rounds.extend(latest_rounds(EVAL_ROOT / dataset, selected_tasks))

    all_issues: list[ValidationIssue] = []
    failed_rounds = 0
    for round_dir in rounds:
        round_issues = validate_round(round_dir, resolver)
        if round_issues:
            failed_rounds += 1
            all_issues.extend(round_issues)

    counts = Counter(item.code for item in all_issues)
    summary = {
        "validated_rounds": len(rounds),
        "passed_rounds": len(rounds) - failed_rounds,
        "failed_rounds": failed_rounds,
        "issues": len(all_issues),
        "issue_counts": dict(sorted(counts.items())),
        "mapping_override_files": resolver.override_files,
    }
    report = {
        "datasets": list(datasets),
        "tasks": sorted(selected_tasks) if selected_tasks is not None else None,
        "mapping_dir": str(args.mapping_dir.resolve()) if args.mapping_dir else None,
        "summary": summary,
        "issues": [asdict(item) for item in all_issues],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(summary, ensure_ascii=False))
    for item in all_issues[:args.show_errors]:
        location = item.round
        if item.step is not None:
            location += f":step{item.step}"
        action_text = f" {item.action}" if item.action else ""
        print(f"ERROR [{item.code}] {location}{action_text}: {item.detail}")
    if args.report is not None:
        print(f"report: {args.report}")
    raise SystemExit(1 if all_issues else 0)


if __name__ == "__main__":
    main()
