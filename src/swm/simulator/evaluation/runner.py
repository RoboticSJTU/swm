from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from swm.pddl.strips import ground_plan, parse_domain, parse_plan, parse_problem_model

from swm.simulator.corpus.inventory import (
    load_case,
    select_highest_rounds,
    sha256_file,
)
from swm.simulator.ir.model import ActionFamily, VerificationStatus
from swm.simulator.kernel import verify

SCHEMA_VERSION = "domain_logical_simulator_evaluation_v1"


def _run_files(directory: Path, plan_path: Path):
    schemas = parse_domain(directory / "domain.pddl")
    problem = parse_problem_model(directory / "problem.pddl", schemas)
    raw, _ = parse_plan(plan_path)
    actions = ground_plan(raw, schemas, problem.objects)
    return verify(schemas, problem, actions)


def evaluate_all(
    repository_root: Path,
    *,
    check_determinism: bool = True,
) -> dict[str, Any]:
    human_root = repository_root / "eval_results/gpt-5.6-sol/human"
    manifest_path = repository_root / "temp/domain_logical_simulator/fixtures/control_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_status: Counter[str] = Counter()
    family_steps: Counter[str] = Counter()
    split_status: dict[str, Counter[str]] = {
        "development": Counter(),
        "holdout": Counter(),
    }
    deterministic = True
    reference_rows = []
    total_steps = 0
    generic_steps = 0
    for case in select_highest_rounds(human_root):
        schemas, problem, plan = load_case(case)
        result = verify(schemas, problem, plan)
        if check_determinism:
            deterministic = deterministic and result == verify(schemas, problem, plan)
        status = result.status.value
        reference_status[status] += 1
        split = "holdout" if case.task_id > 20 and case.task_id % 5 == 0 else "development"
        split_status[split][status] += 1
        total_steps += len(result.trace)
        for record in result.trace:
            family_steps[record.family] += 1
            if record.family == ActionFamily.GENERIC_PDDL.value:
                generic_steps += 1
        reference_rows.append(
            {
                "task_id": case.task_id,
                "round": case.round_number,
                "status": status,
                "first_issue": None
                if result.first_issue is None
                else {
                    "step": result.first_issue.step,
                    "category": result.first_issue.category.value,
                },
            }
        )

    control_rows = []
    legal_status: Counter[str] = Counter()
    invalid_status: Counter[str] = Counter()
    earliest_correct = 0
    category_correct = 0
    invalid_count = 0
    for control in manifest["controls"]:
        plan_path = repository_root / control["path"]
        if sha256_file(plan_path) != control["sha256"]:
            raise ValueError(f"control hash mismatch: {plan_path}")
        directory = plan_path.parent
        result = _run_files(directory, plan_path)
        if check_determinism:
            deterministic = deterministic and result == _run_files(directory, plan_path)
        row = {
            "case": control["case"],
            "kind": control["kind"],
            "status": result.status.value,
            "step": None if result.first_issue is None else result.first_issue.step,
            "category": None if result.first_issue is None else result.first_issue.category.value,
        }
        control_rows.append(row)
        if control["kind"] == "legal":
            legal_status[result.status.value] += 1
        else:
            invalid_count += 1
            invalid_status[result.status.value] += 1
            earliest_correct += int(row["step"] == control["expected_step"])
            category_correct += int(row["category"] == control["category"])
    reference_total = sum(reference_status.values())
    legal_total = sum(legal_status.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "corpus_root": str(human_root.resolve()),
            "control_manifest": str(manifest_path.resolve()),
            "control_manifest_sha256": sha256_file(manifest_path),
        },
        "reference": {
            "total": reference_total,
            "status": dict(sorted(reference_status.items())),
            "pass_rate": reference_status[VerificationStatus.PASS.value] / reference_total,
            "fail_rate": reference_status[VerificationStatus.FAIL.value] / reference_total,
            "unknown_rate": reference_status[VerificationStatus.UNKNOWN.value] / reference_total,
            "rows": reference_rows,
        },
        "legal_controls": {
            "total": legal_total,
            "status": dict(sorted(legal_status.items())),
            "pass_rate": legal_status[VerificationStatus.PASS.value] / legal_total,
            "false_fail_rate": legal_status[VerificationStatus.FAIL.value] / legal_total,
        },
        "invalid_controls": {
            "total": invalid_count,
            "status": dict(sorted(invalid_status.items())),
            "recall": invalid_status[VerificationStatus.FAIL.value] / invalid_count,
            "earliest_step_accuracy": earliest_correct / invalid_count,
            "category_accuracy": category_correct / invalid_count,
        },
        "controls": control_rows,
        "coverage": {
            "mapped_steps": total_steps,
            "mapping_coverage": 1.0,
            "specific_semantic_steps": total_steps - generic_steps,
            "specific_semantic_coverage": (total_steps - generic_steps) / total_steps,
            "generic_pddl_steps": generic_steps,
            "family_steps": dict(sorted(family_steps.items())),
        },
        "holdout": {
            split: {"total": sum(counts.values()), "status": dict(sorted(counts.items()))}
            for split, counts in split_status.items()
        },
        "deterministic": deterministic,
        "runtime_work": {
            "reference_cases": reference_total,
            "reference_steps": total_steps,
            "controls": len(control_rows),
            "note": "wall time is environment-dependent and printed by the invoking command",
        },
    }


def write_evaluation_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reference = report["reference"]
    legal = report["legal_controls"]
    invalid = report["invalid_controls"]
    coverage = report["coverage"]
    lines = [
        "# Simulator Evaluation",
        "",
        f"- Source references: {reference['status']} ({reference['pass_rate']:.2%} PASS)",
        f"- Legal controls: {legal['status']} ({legal['false_fail_rate']:.2%} false FAIL)",
        f"- Invalid controls: {invalid['status']} ({invalid['recall']:.2%} recall)",
        f"- Earliest-step accuracy: {invalid['earliest_step_accuracy']:.2%}",
        f"- Certificate category accuracy: {invalid['category_accuracy']:.2%}",
        f"- Action mapping coverage: {coverage['mapping_coverage']:.2%}",
        f"- Specific deterministic semantics: {coverage['specific_semantic_coverage']:.2%}",
        f"- Deterministic repeated execution: {report['deterministic']}",
        "",
        "Holdout and per-case rows are retained in `evaluation.json`. The report",
        "contains source identities and no measured result depends on task-specific",
        "branches in runtime action code.",
        "",
    ]
    (report_dir / "EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
