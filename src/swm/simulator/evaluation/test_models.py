from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from swm.simulator.alignment import VLMAlignmentAdvisor
from swm.simulator.cli import build_vlm_advisor, parse_cli

from .gt_semantic import evaluate_semantic_gt, original_judge_pass

SCHEMA_VERSION = "domain_logical_simulator_test_models_v4"


def _new_metrics() -> dict[str, Any]:
    return {
        "tasks": 0,
        "simulator_status": Counter(),
        "ground_truth_sources": Counter(),
        "labeled_tasks": 0,
        "scored_tasks": 0,
        "abstentions": 0,
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }


def _record_metrics(
    metrics: dict[str, Any],
    status: str | None,
    judge_current_pass: bool | None,
) -> None:
    metrics["tasks"] += 1
    normalized_status = status or "INPUT_ERROR"
    metrics["simulator_status"][normalized_status] += 1
    prediction = True if normalized_status == "PASS" else False if normalized_status == "FAIL" else None
    if isinstance(judge_current_pass, bool):
        label = judge_current_pass
        source = "judge_current"
    else:
        label = None
        source = "unlabeled"
    metrics["ground_truth_sources"][source] += 1
    if label is None:
        return
    metrics["labeled_tasks"] += 1
    if prediction is None:
        metrics["abstentions"] += 1
        return
    metrics["scored_tasks"] += 1
    if prediction and label:
        metrics["TP"] += 1
    elif not prediction and not label:
        metrics["TN"] += 1
    elif prediction:
        metrics["FP"] += 1
    else:
        metrics["FN"] += 1


def _finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    scored = metrics["scored_tasks"]
    correct = metrics["TP"] + metrics["TN"]
    return {
        **{
            key: value
            for key, value in metrics.items()
            if key not in {"simulator_status", "ground_truth_sources"}
        },
        "simulator_status": dict(sorted(metrics["simulator_status"].items())),
        "effective_status": dict(sorted(metrics["simulator_status"].items())),
        "ground_truth_sources": dict(sorted(metrics["ground_truth_sources"].items())),
        "correct": correct,
        "accuracy": None if not scored else correct / scored,
    }


def _has_applied_label_correction(
    judge: dict[str, Any],
    current_pass: bool | None,
) -> bool:
    correction = judge.get("label_correction")
    if (
        isinstance(correction, dict)
        and isinstance(correction.get("previous_pass"), bool)
        and correction["previous_pass"] != current_pass
    ):
        return True
    history = judge.get("label_correction_history")
    return isinstance(history, list) and any(
        isinstance(item, dict) and item.get("corrected_pass") == current_pass
        for item in history
    )


def evaluate_test_models(
    gt_root: Path,
    models_root: Path,
    *,
    alignment_advisor: VLMAlignmentAdvisor | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    gt_root = gt_root.resolve()
    models_root = models_root.resolve()
    model_reports: dict[str, Any] = {}
    conflicts = []
    current_label_conflicts = []
    label_corrections = []
    label_correction_history = []
    judge_internal_inconsistencies = []
    failure_certificates = []
    aggregate_status: Counter[str] = Counter()
    aggregate_metrics = _new_metrics()
    split_metrics: dict[str, dict[str, Any]] = {}
    model_metrics: dict[str, dict[str, Any]] = {}
    model_split_metrics: dict[str, dict[str, dict[str, Any]]] = {}

    model_dirs = sorted(item for item in models_root.iterdir() if item.is_dir())
    if alignment_advisor is not None and workers > 1:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(model_dirs)))) as executor:
            futures = {
                model_dir.name: executor.submit(
                    evaluate_semantic_gt,
                    gt_root,
                    model_dir,
                    alignment_advisor=alignment_advisor,
                )
                for model_dir in model_dirs
            }
            reports = {name: future.result() for name, future in futures.items()}
    else:
        reports = {
            model_dir.name: evaluate_semantic_gt(
                gt_root, model_dir, alignment_advisor=alignment_advisor
            )
            for model_dir in model_dirs
        }

    for model_dir in model_dirs:
        report = reports[model_dir.name]
        model_reports[model_dir.name] = report
        model_metric = _new_metrics()
        model_metrics[model_dir.name] = model_metric
        model_split_metrics[model_dir.name] = {}
        for split, split_report in report["splits"].items():
            aggregate_status.update(split_report["candidate_status"])
            split_metric = split_metrics.setdefault(split, _new_metrics())
            model_split_metric = _new_metrics()
            model_split_metrics[model_dir.name][split] = model_split_metric
            for row in split_report["rows"]:
                status = row.get("status")
                task_id = row["task_id"]
                task_dir = model_dir / split / f"task_{task_id}"
                judge_path = task_dir / "judge.json"
                judge = None
                judge_pass = row.get("judge_pass")
                judge_current_pass = row.get("judge_current_pass")
                if judge_path.exists():
                    judge = json.loads(judge_path.read_text(encoding="utf-8"))
                    judge_current_pass = judge.get("pass")
                    correction = judge.get("label_correction")
                    judge_pass = original_judge_pass(judge, judge_current_pass)
                    row["judge_pass"] = judge_pass
                    row["judge_current_pass"] = judge_current_pass
                    text = " ".join(
                        str(judge.get(key, "")) for key in ("reasoning", "feedback")
                    ).lower()
                    decision = re.findall(r"\bdecision\s*:\s*(pass|fail)\b", text)
                    if (
                        decision
                        and isinstance(judge_current_pass, bool)
                        and not _has_applied_label_correction(judge, judge_current_pass)
                    ):
                        text_pass = decision[-1] == "pass"
                        if text_pass != judge_current_pass:
                            judge_internal_inconsistencies.append(
                                {
                                    "model": model_dir.name,
                                    "split": split,
                                    "task_id": task_id,
                                    "current_pass": judge_current_pass,
                                    "reasoning_decision": decision[-1],
                                }
                            )
                    if isinstance(correction, dict):
                        label_corrections.append(
                            {
                                "model": model_dir.name,
                                "split": split,
                                "task_id": task_id,
                                "previous_pass": correction.get("previous_pass"),
                                "corrected_pass": judge.get("pass"),
                                "source": correction.get("source"),
                            }
                        )
                    history = judge.get("label_correction_history")
                    if isinstance(history, list):
                        for item in history:
                            if isinstance(item, dict):
                                label_correction_history.append(
                                    {
                                        "model": model_dir.name,
                                        "split": split,
                                        "task_id": task_id,
                                        **item,
                                    }
                                )
                for metrics in (aggregate_metrics, split_metric, model_metric, model_split_metric):
                    _record_metrics(metrics, status, judge_current_pass)
                plan_path = task_dir / "plan.txt"
                plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest() if plan_path.exists() else None
                prediction = True if status == "PASS" else False if status == "FAIL" else None
                if status != "PASS":
                    failure_certificates.append(
                        {
                            "model": model_dir.name,
                            "split": split,
                            "task_id": task_id,
                            "status": status,
                            "judge_current_pass": judge_current_pass,
                            "first_issue": row.get("first_issue"),
                            "error": row.get("error"),
                            "plan_sha256": plan_sha256,
                        }
                    )
                if (
                    isinstance(judge_current_pass, bool)
                    and (
                        prediction is None or prediction != judge_current_pass
                    )
                ):
                    current_label_conflicts.append(
                        {
                            "model": model_dir.name,
                            "split": split,
                            "task_id": task_id,
                            "status": status,
                            "judge_current_pass": judge_current_pass,
                            "first_issue": row.get("first_issue"),
                            "error": row.get("error"),
                            "plan_sha256": plan_sha256,
                        }
                    )
                if not isinstance(judge_pass, bool):
                    continue
                if prediction is not None and prediction == judge_pass:
                    continue
                conflicts.append(
                    {
                        "model": model_dir.name,
                        "split": split,
                        "task_id": task_id,
                        "status": status,
                        "judge_pass": judge_pass,
                        "first_issue": row.get("first_issue"),
                        "error": row.get("error"),
                        "plan_sha256": plan_sha256,
                    }
                )

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for conflict in conflicts:
        issue = conflict.get("first_issue") or {}
        key = (
            conflict["split"],
            conflict["task_id"],
            conflict["status"],
            conflict["judge_pass"],
            issue.get("category"),
            conflict["plan_sha256"],
        )
        group = groups.setdefault(
            key,
            {
                "split": conflict["split"],
                "task_id": conflict["task_id"],
                "status": conflict["status"],
                "judge_pass": conflict["judge_pass"],
                "issue_category": issue.get("category"),
                "plan_sha256": conflict["plan_sha256"],
                "models": [],
                "example": conflict,
            },
        )
        group["models"].append(conflict["model"])

    conflict_groups = sorted(
        groups.values(),
        key=lambda item: (
            item["split"],
            item["task_id"],
            item["status"],
            item["plan_sha256"] or "",
        ),
    )
    for group in conflict_groups:
        group["models"].sort()

    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {"gt_root": str(gt_root), "models_root": str(models_root)},
        "model_count": len(model_reports),
        "aggregate_status": dict(sorted(aggregate_status.items())),
        "effective_status": dict(sorted(aggregate_status.items())),
        "conflict_count": len(conflicts),
        "current_label_conflict_count": len(current_label_conflicts),
        "current_label_conflicts": current_label_conflicts,
        "judge_internal_inconsistency_count": len(judge_internal_inconsistencies),
        "judge_internal_inconsistencies": judge_internal_inconsistencies,
        "conflict_group_count": len(conflict_groups),
        "label_correction_count": len(label_corrections),
        "label_corrections": label_corrections,
        "label_correction_history_count": len(label_correction_history),
        "label_correction_history": label_correction_history,
        "metrics": {
            "overall": _finalize_metrics(aggregate_metrics),
            "by_split": {
                split: _finalize_metrics(metrics)
                for split, metrics in sorted(split_metrics.items())
            },
            "by_model": {
                model: _finalize_metrics(metrics)
                for model, metrics in sorted(model_metrics.items())
            },
            "by_model_and_split": {
                model: {
                    split: _finalize_metrics(metrics)
                    for split, metrics in sorted(values.items())
                }
                for model, values in sorted(model_split_metrics.items())
            },
        },
        "vlm_mapping": None
        if alignment_advisor is None
        else {
            "model": alignment_advisor.model,
            "cache_only": not alignment_advisor.allow_network,
            "stats": alignment_advisor.stats(),
        },
        "conflicts": conflicts,
        "conflict_groups": conflict_groups,
        "failure_certificates": failure_certificates,
        "models": model_reports,
    }


def write_test_models_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "test_models_semantic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = report["metrics"]

    def accuracy(value: dict[str, Any]) -> str:
        return "n/a" if value["accuracy"] is None else f"{value['accuracy']:.2%}"

    overall = metrics["overall"]
    lines = [
        "# Test Model Semantic Evaluation",
        "",
        f"- Models: {report['model_count']}",
        f"- Aggregate status: {report['aggregate_status']}",
        f"- Decision status (unavailable inputs are not FAIL): {report['effective_status']}",
        f"- Simulator/original-judge conflicts: {report['conflict_count']}",
        f"- Current-label disagreements or unavailable decisions: "
        f"{report['current_label_conflict_count']}",
        f"- Current label/reasoning contradictions: {report['judge_internal_inconsistency_count']}",
        f"- Unique conflict groups: {report['conflict_group_count']}",
        f"- Previously modified judge labels: {report['label_correction_count']}",
        f"- New GT corrections in this audit: {report['label_correction_history_count']}",
        f"- Non-PASS records retained: {len(report['failure_certificates'])}",
        f"- Labeled binary agreement (not independently audited accuracy): "
        f"{accuracy(overall)} ({overall['correct']}/{overall['scored_tasks']}; "
        f"TP={overall['TP']}, TN={overall['TN']}, FP={overall['FP']}, FN={overall['FN']})",
    ]
    if report.get("vlm_mapping") is not None:
        lines.append(f"- VLM mapping: {report['vlm_mapping']}")
    lines.extend(
        [
            "",
            "## Dataset Results",
            "",
            "| Dataset | Tasks | Simulator PASS | Simulator FAIL | TP | TN | FP | FN | Label agreement |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, value in metrics["by_split"].items():
        effective = value["effective_status"]
        lines.append(
            f"| `{split}` | {value['tasks']} | {effective.get('PASS', 0)} | "
            f"{effective.get('FAIL', 0)} | {value['TP']} | {value['TN']} | "
            f"{value['FP']} | {value['FN']} | {accuracy(value)} |"
        )
    lines.extend(
        [
            "",
            "## Model Results",
            "",
            "| Model | SWM Judge agreement | UniDomain Judge agreement | Overall Judge agreement | TP | TN | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, value in metrics["by_model"].items():
        by_split = metrics["by_model_and_split"][model]
        lines.append(
            f"| `{model}` | {accuracy(by_split['swm'])} | "
            f"{accuracy(by_split['unidomain'])} | {accuracy(value)} | "
            f"{value['TP']} | {value['TN']} | {value['FP']} | {value['FN']} |"
        )
    lines.extend(["", "## Current-label Disagreements Or Unavailable Decisions", ""])
    if not report["current_label_conflicts"]:
        lines.append("- None.")
    else:
        for conflict in report["current_label_conflicts"]:
            issue = conflict.get("first_issue") or {}
            lines.append(
                f"- `{conflict['model']}/{conflict['split']}/task_{conflict['task_id']}`: "
                f"simulator `{conflict['status']}`, current label "
                f"`{conflict['judge_current_pass']}`, category `{issue.get('category')}`"
            )
    failure_categories = Counter(
        (item.get("first_issue") or {}).get("category")
        or str(item.get("status", "input_error")).lower()
        for item in report["failure_certificates"]
    )
    lines.extend(
        [
            "",
            "## Failure-certificate Categories",
            "",
            "| Category | Count |",
            "|---|---:|",
        ]
    )
    for category, count in failure_categories.most_common():
        lines.append(f"| `{category}` | {count} |")
    lines.extend(["", "## Original-label Conflict Groups", ""])
    for group in report["conflict_groups"]:
        models = ", ".join(group["models"])
        lines.append(
            f"- `{group['split']}/task_{group['task_id']}`: simulator "
            f"`{group['status']}`, judge `{group['judge_pass']}`, category "
            f"`{group['issue_category']}`, models: {models}"
        )
    (report_dir / "TEST_MODELS_SEMANTIC.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        root = Path(__file__).resolve().parents[4]
        argv = [str(root / path) for path in (
            "eval_results/gt", "eval_results/test",
            "temp/domain_logical_simulator/reports",
        )]
    paths, options, flags = parse_cli(
        argv,
        3,
        {
            "vlm-model", "vlm-base-url", "vlm-cache", "vlm-workers", "env-file",
            "vlm-api-key-env", "vlm-reasoning-effort",
        },
        {"vlm-mapping", "vlm-cache-only", "vlm-json-mode"},
    )
    advisor = build_vlm_advisor(options, flags)
    report = evaluate_test_models(
        Path(paths[0]),
        Path(paths[1]),
        alignment_advisor=advisor,
        workers=int(options.get("vlm-workers", 16)),
    )
    write_test_models_report(report, Path(paths[2]))
    print(
        f"models={report['model_count']} conflicts={report['conflict_count']} "
        f"groups={report['conflict_group_count']} status={report['aggregate_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
