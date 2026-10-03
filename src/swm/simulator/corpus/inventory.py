from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from swm.simulator.alignment.predicates import PredicateInterface

from swm.pddl.strips import (
    ActionSchema,
    DomainSchemas,
    GroundAction,
    ProblemModel,
    goals_satisfied,
    ground_plan,
    parse_domain,
    parse_sexpr_file,
    parse_plan,
    parse_problem_model,
    rollout,
    validate_untyped_pddl,
)

SCHEMA_VERSION = "domain_logical_simulator_corpus_v2"
ROUND_RE = re.compile(r"^round(\d+)$")
TASK_RE = re.compile(r"^task_(\d+)$")


@dataclass(frozen=True)
class CorpusCase:
    task_id: int
    round_number: int
    task_dir: str
    round_dir: str
    domain_path: str
    problem_path: str
    plan_path: str
    source_sha256: tuple[tuple[str, str], ...]

    def hashes(self) -> dict[str, str]:
        return dict(self.source_sha256)


@dataclass(frozen=True)
class OperatorObservation:
    task_id: int
    round_number: int
    action_name: str
    parameter_names: tuple[str, ...]
    preconditions_positive: tuple[tuple[str, ...], ...]
    preconditions_negative: tuple[tuple[str, ...], ...]
    effects_add: tuple[tuple[str, ...], ...]
    effects_delete: tuple[tuple[str, ...], ...]
    signature: str
    source_domain: str
    source_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numbered(path: Path, pattern: re.Pattern[str], kind: str) -> int:
    match = pattern.fullmatch(path.name)
    if not match:
        raise ValueError(f"invalid {kind} directory: {path}")
    return int(match.group(1))


def select_highest_rounds(corpus_root: Path) -> list[CorpusCase]:
    """Select one maximum numeric round for each task without touching sources."""
    if not corpus_root.is_dir():
        raise FileNotFoundError(f"corpus root not found: {corpus_root}")
    task_dirs = sorted(
        (path for path in corpus_root.iterdir() if TASK_RE.fullmatch(path.name)),
        key=lambda path: _numbered(path, TASK_RE, "task"),
    )
    cases: list[CorpusCase] = []
    for task_dir in task_dirs:
        task_id = _numbered(task_dir, TASK_RE, "task")
        episode_dir = task_dir / f"episode_{task_id}"
        if not episode_dir.is_dir():
            raise FileNotFoundError(f"missing episode directory: {episode_dir}")
        rounds = sorted(
            (path for path in episode_dir.iterdir() if ROUND_RE.fullmatch(path.name)),
            key=lambda path: _numbered(path, ROUND_RE, "round"),
        )
        if not rounds:
            raise FileNotFoundError(f"no rounds found in {episode_dir}")
        round_dir = rounds[-1]
        files = {
            "domain": round_dir / "domain.pddl",
            "problem": round_dir / "problem.pddl",
            "plan": round_dir / "plan.txt",
        }
        missing = [name for name, path in files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"highest round {round_dir} misses required files: {missing}"
            )
        cases.append(
            CorpusCase(
                task_id=task_id,
                round_number=_numbered(round_dir, ROUND_RE, "round"),
                task_dir=str(task_dir.resolve()),
                round_dir=str(round_dir.resolve()),
                domain_path=str(files["domain"].resolve()),
                problem_path=str(files["problem"].resolve()),
                plan_path=str(files["plan"].resolve()),
                source_sha256=tuple(
                    (name, sha256_file(path)) for name, path in sorted(files.items())
                ),
            )
        )
    return cases


@dataclass(frozen=True)
class CandidateDomainRecovery:
    """Safe plan-scoped repairs made only in a temporary candidate-domain copy."""

    ignored_unused_actions: tuple[str, ...] = ()
    deduplicated_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "ignored_unused_actions": list(self.ignored_unused_actions),
            "deduplicated_actions": list(self.deduplicated_actions),
        }


def _sexpr_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "(" + " ".join(_sexpr_text(item) for item in value) + ")"
    raise ValueError("candidate domain contains a non-symbol S-expression value")


def parse_candidate_domain_readonly(
    path: Path,
    used_actions: Iterable[str],
) -> tuple[DomainSchemas, CandidateDomainRecovery]:
    """Parse a candidate domain without allowing irrelevant bad operators to veto a plan.

    Candidate domains are generated artifacts.  A malformed operator that cannot
    occur in the submitted plan has no execution semantics, while two bytewise
    identical definitions of a used operator have exactly one semantics.  Both
    cases can be normalized in a temporary copy.  A malformed or conflicting
    definition of a *used* operator is deliberately still rejected.
    """
    validate_untyped_pddl(path.read_text(encoding="utf-8"))
    try:
        return parse_domain(path), CandidateDomainRecovery()
    except ValueError as error:
        original_error = error

    try:
        root = parse_sexpr_file(path)
    except ValueError:
        raise original_error
    if not isinstance(root, list) or not root or root[0] != "define":
        raise original_error

    required = {name.lower() for name in used_actions}
    action_groups: dict[str, list[list[Any]]] = {}
    for item in root[1:]:
        if (
            isinstance(item, list)
            and len(item) >= 2
            and item[0] == ":action"
            and isinstance(item[1], str)
        ):
            action_groups.setdefault(item[1], []).append(item)

    kept_actions: set[str] = set()
    ignored: list[str] = []
    deduplicated: list[str] = []
    projected: list[Any] = [root[0]]
    for item in root[1:]:
        if not (
            isinstance(item, list)
            and item
            and item[0] == ":action"
        ):
            projected.append(item)
            continue
        if len(item) < 2 or not isinstance(item[1], str):
            # Such an action cannot be named by a valid plan invocation.
            ignored.append("<malformed>")
            continue
        name = item[1]
        if name not in required:
            ignored.append(name)
            continue
        definitions = action_groups[name]
        if len(definitions) > 1:
            if any(definition != definitions[0] for definition in definitions[1:]):
                raise ValueError(
                    f"{path}: conflicting duplicate definition for used action '{name}'"
                ) from original_error
            if name not in deduplicated:
                deduplicated.append(name)
        if name in kept_actions:
            continue
        kept_actions.add(name)
        projected.append(item)

    if not ignored and not deduplicated:
        raise original_error
    try:
        with tempfile.TemporaryDirectory(prefix="logical-sim-candidate-domain-") as directory:
            repaired = Path(directory) / "domain.pddl"
            repaired.write_text(_sexpr_text(projected), encoding="utf-8")
            schemas = parse_domain(repaired)
    except ValueError:
        raise original_error
    return schemas, CandidateDomainRecovery(
        tuple(sorted(set(ignored))), tuple(sorted(deduplicated))
    )


def load_case(
    case: CorpusCase,
) -> tuple[DomainSchemas, ProblemModel, list[GroundAction]]:
    schemas = parse_domain(Path(case.domain_path))
    problem = parse_problem_model(Path(case.problem_path), schemas)
    raw_plan, _ = parse_plan(Path(case.plan_path))
    plan = ground_plan(raw_plan, schemas, problem.objects)
    if not plan:
        raise ValueError(f"empty reference plan: {case.plan_path}")
    return schemas, problem, plan


def _canonical_literal(
    literal: tuple[str, ...], mapping: dict[str, str]
) -> tuple[str, ...]:
    return (literal[0], *(mapping.get(token, token) for token in literal[1:]))


def alpha_signature(schema: ActionSchema) -> str:
    mapping = {parameter: f"?v{index}" for index, parameter in enumerate(schema.params)}

    def normalized(literals: Iterable[tuple[str, ...]]) -> list[list[str]]:
        return [
            list(item)
            for item in sorted(_canonical_literal(literal, mapping) for literal in literals)
        ]

    payload: dict[str, Any] = {
        "arity": len(schema.params),
        "pre_pos": normalized(schema.pre_pos),
        "pre_neg": normalized(schema.pre_neg),
        "add": normalized(schema.add_eff),
        "delete": normalized(schema.del_eff),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def observation_from_schema(
    case: CorpusCase, schema: ActionSchema
) -> OperatorObservation:
    mapping = {parameter: f"?v{index}" for index, parameter in enumerate(schema.params)}

    def literals(values: Iterable[tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted(_canonical_literal(value, mapping) for value in values))

    return OperatorObservation(
        task_id=case.task_id,
        round_number=case.round_number,
        action_name=schema.name,
        parameter_names=tuple(mapping[parameter] for parameter in schema.params),
        preconditions_positive=literals(schema.pre_pos),
        preconditions_negative=literals(schema.pre_neg),
        effects_add=literals(schema.add_eff),
        effects_delete=literals(schema.del_eff),
        signature=alpha_signature(schema),
        source_domain=case.domain_path,
        source_sha256=case.hashes()["domain"],
    )


def action_family(name: str) -> str:
    normalized = name.lower()
    if normalized.startswith("pick_") or normalized.startswith("pickup_"):
        return "pick"
    if normalized.startswith("place_") or normalized.startswith("put_"):
        return "place"
    if normalized.startswith("open_"):
        return "open"
    if normalized.startswith("close_"):
        return "close"
    if normalized.startswith("turn_") or normalized.startswith("switch_"):
        return "turn"
    if normalized.startswith("pour_"):
        return "pour"
    if normalized.startswith("remove_"):
        return "remove"
    if normalized.startswith("unscrew_"):
        return "unscrew"
    if normalized.startswith("screw_"):
        return "screw"
    if normalized.startswith("insert_"):
        return "insert"
    if normalized.startswith("scoop_"):
        return "scoop"
    return normalized.split("_", 1)[0]


def audit_corpus(cases: Iterable[CorpusCase]) -> dict[str, Any]:
    observations: list[OperatorObservation] = []
    action_names: set[str] = set()
    predicates: set[str] = set()
    categories: set[str] = set()
    family_steps: Counter[str] = Counter()
    plan_lengths: list[int] = []
    selected = list(cases)
    source_before = {
        (case.task_id, name): digest
        for case in selected
        for name, digest in case.source_sha256
    }
    case_rows = []
    for case in selected:
        schemas, problem, plan = load_case(case)
        final_state = rollout(problem.init_state, plan)
        if not goals_satisfied(
            final_state, problem.goal_positive, problem.goal_negative
        ):
            raise ValueError(f"reference plan does not reach goal: task {case.task_id}")
        for schema in schemas.values():
            observations.append(observation_from_schema(case, schema))
        action_names.update(schemas)
        predicates.update(schemas.predicate_arities)
        categories.update(item.raw_name for item in PredicateInterface.from_schemas(schemas).descriptions
                          if item.static and item.arity == 1)
        for action in plan:
            family_steps[action_family(action.name)] += 1
        plan_lengths.append(len(plan))
        case_rows.append(
            {
                "task_id": case.task_id,
                "round": case.round_number,
                "operator_count": len(schemas),
                "reference_steps": len(plan),
                "paths": {
                    "domain": case.domain_path,
                    "problem": case.problem_path,
                    "plan": case.plan_path,
                },
                "sha256": case.hashes(),
            }
        )
    source_after = {
        (case.task_id, name): sha256_file(
            Path(
                {
                    "domain": case.domain_path,
                    "problem": case.problem_path,
                    "plan": case.plan_path,
                }[name]
            )
        )
        for case in selected
        for name, _ in case.source_sha256
    }
    if source_before != source_after:
        raise RuntimeError("source corpus changed during read-only audit")

    clusters: dict[str, list[OperatorObservation]] = defaultdict(list)
    by_name: dict[str, set[str]] = defaultdict(set)
    for observation in observations:
        clusters[observation.signature].append(observation)
        by_name[observation.action_name].add(observation.signature)
    conflicts = [
        {
            "action_name": name,
            "schema_count": len(signatures),
            "signatures": sorted(signatures),
        }
        for name, signatures in sorted(by_name.items())
        if len(signatures) > 1
    ]
    sorted_lengths = sorted(plan_lengths)
    median = (
        sorted_lengths[len(sorted_lengths) // 2]
        if sorted_lengths
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "selection": {
            "policy": "maximum numeric round per task_<id>/episode_<id>",
            "task_count": len(selected),
            "source_round_count": sum(
                1
                for case in selected
                for path in Path(case.task_dir).joinpath(f"episode_{case.task_id}").iterdir()
                if ROUND_RE.fullmatch(path.name) and (path / "domain.pddl").is_file()
            ),
        },
        "counts": {
            "operators": len(observations),
            "unique_action_names": len(action_names),
            "unique_predicates": len(predicates),
            "unary_categories": len(categories),
            "transition_schemas": len(clusters),
            "reference_steps": sum(plan_lengths),
        },
        "plan_length": {
            "minimum": min(plan_lengths) if plan_lengths else None,
            "median": median,
            "maximum": max(plan_lengths) if plan_lengths else None,
        },
        "action_family_steps": dict(sorted(family_steps.items())),
        "predicates": sorted(predicates),
        "categories": sorted(categories),
        "schema_conflicts": conflicts,
        "cases": case_rows,
        "observations": [asdict(item) for item in observations],
    }


def build_inventory(corpus_root: Path) -> dict[str, Any]:
    return audit_corpus(select_highest_rounds(corpus_root))


def render_summary(report: dict[str, Any]) -> str:
    counts = report["counts"]
    selection = report["selection"]
    lines = [
        "# Human 300 Corpus Inventory",
        "",
        f"Schema: `{report['schema_version']}`",
        "",
        "## Selection",
        "",
        f"- Policy: {selection['policy']}",
        f"- Selected tasks: {selection['task_count']}",
        f"- Source round directories: {selection['source_round_count']}",
        "",
        "## Audited Counts",
        "",
        f"- Operators: {counts['operators']}",
        f"- Unique action names: {counts['unique_action_names']}",
        f"- Unique predicates: {counts['unique_predicates']}",
        f"- Unary categories: {counts['unary_categories']}",
        f"- Transition schemas: {counts['transition_schemas']}",
        f"- Reference steps: {counts['reference_steps']}",
        "",
        "Every selected domain/problem/plan parsed, grounded, replayed under its own",
        "PDDL, and reached its goal. Source hashes were unchanged by the audit.",
        "",
        "## Action Families",
        "",
    ]
    total = counts["reference_steps"]
    for family, count in report["action_family_steps"].items():
        lines.append(f"- `{family}`: {count} ({count / total:.2%})")
    lines.extend(
        [
            "",
            "## Conflicts",
            "",
            f"{len(report['schema_conflicts'])} repeated action names have multiple schemas.",
            "The machine-readable report retains every signature and its provenance; no",
            "conflict is promoted directly to a runtime rule.",
            "",
        ]
    )
    return "\n".join(lines)


def write_inventory_reports(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "corpus_inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (report_dir / "CORPUS_SUMMARY.md").write_text(
        render_summary(report), encoding="utf-8"
    )
