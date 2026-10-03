from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from swm.pddl.strips import ground_plan, parse_domain, parse_plan, parse_problem_model

from swm.simulator.cli import parse_cli
from swm.simulator.kernel import verify

SCHEMA_VERSION = "domain_logical_simulator_gt_candidate_preflight_v1"
SPLITS = ("swm", "unidomain")


def select_highest_round(task_dir: Path) -> Path:
    rounds = []
    for path in task_dir.glob("round*"):
        suffix = path.name.removeprefix("round")
        if path.is_dir() and suffix.isdigit():
            rounds.append((int(suffix), path))
    if not rounds:
        raise FileNotFoundError(f"no round directory in {task_dir}")
    return max(rounds)[1]


def _task_id(path: Path) -> int:
    suffix = path.name.removeprefix("task_")
    if not suffix.isdigit():
        raise ValueError(f"invalid task directory: {path}")
    return int(suffix)


def _input_error(error: Exception) -> tuple[str, str]:
    message = str(error).splitlines()[0]
    if isinstance(error, KeyError):
        return "UNKNOWN_ACTION", message
    if "Arity mismatch" in message:
        return "ARITY_MISMATCH", message
    if "uses undeclared object" in message:
        return "OBJECT_MISMATCH", message
    if "Type mismatch" in message:
        return "TYPE_MISMATCH", message
    return "INPUT_ERROR", message


def _self_replay(domain_path: Path, problem_path: Path, plan_path: Path) -> dict[str, Any]:
    """Internal preflight check; the public evaluator is cross-domain only."""
    try:
        schemas = parse_domain(domain_path)
        problem = parse_problem_model(problem_path, schemas)
        raw, _ = parse_plan(plan_path)
        actions = ground_plan(raw, schemas, problem.objects)
        result = verify(schemas, problem, actions).to_dict()
        issue = result["first_issue"]
        return {
            "classification": result["status"],
            "issue": None if issue is None else {
                key: issue[key] for key in ("step", "category", "action", "detail")
            },
        }
    except Exception as error:
        classification, message = _input_error(error)
        return {"classification": classification, "error": message}


def _judge_label(candidate_dir: Path) -> bool | None:
    path = candidate_dir / "judge.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    label = value.get("pass")
    return label if isinstance(label, bool) else None


def evaluate_gt_candidate_preflight(
    gt_root: Path,
    candidate_root: Path,
    *,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    gt_root = gt_root.resolve()
    candidate_root = candidate_root.resolve()
    split_reports: dict[str, Any] = {}
    for split in splits:
        gt_tasks = {
            _task_id(path): path for path in (gt_root / split).glob("task_*")
            if path.is_dir()
        }
        candidate_tasks = {
            _task_id(path): path for path in (candidate_root / split).glob("task_*")
            if path.is_dir()
        }
        task_ids = sorted(set(gt_tasks) | set(candidate_tasks))
        gt_status: Counter[str] = Counter()
        candidate_status: Counter[str] = Counter()
        direct_status: Counter[str] = Counter()
        judge_matrix: Counter[str] = Counter()
        rows = []
        for task_id in task_ids:
            gt_task = gt_tasks.get(task_id)
            candidate_dir = candidate_tasks.get(task_id)
            row: dict[str, Any] = {"task_id": task_id}
            if gt_task is None:
                row["gt"] = {"classification": "MISSING_TASK"}
                gt_status["MISSING_TASK"] += 1
                gt_dir = None
            else:
                try:
                    gt_dir = select_highest_round(gt_task)
                    result = _self_replay(
                        gt_dir / "domain.pddl",
                        gt_dir / "problem.pddl",
                        gt_dir / "plan.txt",
                    )
                    row["gt"] = {
                        "round": int(gt_dir.name.removeprefix("round")),
                        "classification": result["classification"],
                        "issue": result.get("issue"),
                    }
                    gt_status[result["classification"]] += 1
                except Exception as error:
                    classification, message = _input_error(error)
                    row["gt"] = {
                        "classification": classification,
                        "error": message,
                    }
                    gt_status[classification] += 1
                    gt_dir = None

            if candidate_dir is None:
                row["candidate"] = {"classification": "MISSING_TASK"}
                candidate_status["MISSING_TASK"] += 1
                direct_status["MISSING_TASK"] += 1
                rows.append(row)
                continue
            plan_path = candidate_dir / "plan.txt"
            judge_label = _judge_label(candidate_dir)
            if not plan_path.exists():
                row["candidate"] = {
                    "classification": "MISSING_PLAN",
                    "judge_pass": judge_label,
                }
                row["direct_gt_replay"] = {"classification": "MISSING_PLAN"}
                candidate_status["MISSING_PLAN"] += 1
                direct_status["MISSING_PLAN"] += 1
                rows.append(row)
                continue

            try:
                result = _self_replay(
                    candidate_dir / "domain.pddl",
                    candidate_dir / "problem.pddl",
                    plan_path,
                )
                candidate_classification = result["classification"]
                row["candidate"] = {
                    "classification": candidate_classification,
                    "issue": result.get("issue"),
                    "judge_pass": judge_label,
                }
            except Exception as error:
                candidate_classification, message = _input_error(error)
                row["candidate"] = {
                    "classification": candidate_classification,
                    "error": message,
                    "judge_pass": judge_label,
                }
            candidate_status[candidate_classification] += 1
            if judge_label is not None:
                judge_matrix[f"{candidate_classification}|judge_{str(judge_label).lower()}"] += 1

            if gt_dir is None:
                direct = {"classification": "GT_INPUT_ERROR"}
            else:
                direct = _self_replay(
                    gt_dir / "domain.pddl", gt_dir / "problem.pddl", plan_path
                )
            row["direct_gt_replay"] = direct
            direct_status[direct["classification"]] += 1
            rows.append(row)

        direct_executable = sum(
            direct_status[status] for status in ("PASS", "FAIL", "UNKNOWN")
        )
        candidate_plans = sum(candidate_status.values()) - candidate_status["MISSING_PLAN"]
        split_reports[split] = {
            "tasks": len(task_ids),
            "gt_reference": {"status": dict(sorted(gt_status.items()))},
            "candidate_self_check": {
                "status": dict(sorted(candidate_status.items())),
                "plans": candidate_plans,
            },
            "direct_gt_replay": {
                "status": dict(sorted(direct_status.items())),
                "executable": direct_executable,
                "compatibility_rate": direct_executable / len(task_ids) if task_ids else 0.0,
            },
            "candidate_judge_matrix": dict(sorted(judge_matrix.items())),
            "rows": rows,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "gt_root": str(gt_root),
            "candidate_root": str(candidate_root),
            "gt_round_policy": "highest numeric round",
        },
        "semantic_gt_evaluation": {
            "available": False,
            "reason": (
                "candidate plans use task-local action signatures and object names; "
                "direct grounding against GT cannot distinguish vocabulary mismatch "
                "from an invalid physical plan"
            ),
            "required_adapter": (
                "deterministic object alignment plus canonical action/role translation "
                "before replay on the GT scene and goal"
            ),
        },
        "splits": split_reports,
    }


def write_gt_candidate_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "gt_9b_sft_preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# GT / 9B SFT Simulator Preflight",
        "",
        "This report separates candidate self-consistency from GT-grounded correctness.",
        "A vocabulary or arity mismatch is an input compatibility result, not a physical FAIL.",
        "",
    ]
    for split, value in report["splits"].items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Tasks: {value['tasks']}",
                f"- GT reference: {value['gt_reference']['status']}",
                f"- Candidate self-check: {value['candidate_self_check']['status']}",
                f"- Direct GT replay: {value['direct_gt_replay']['status']}",
                f"- Direct compatibility: {value['direct_gt_replay']['compatibility_rate']:.2%}",
                f"- Candidate self-check / judge matrix: {value['candidate_judge_matrix']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "",
            "The current simulator can batch-check each candidate's own PDDL, but the",
            "candidate plans cannot be used as GT-grounded plans without a deterministic",
            "cross-domain object and action-role adapter. The JSON retains per-task",
            "classifications needed to design and test that adapter.",
            "",
        ]
    )
    (report_dir / "GT_9B_SFT_PREFLIGHT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        root = Path(__file__).resolve().parents[4]
        argv = [str(root / path) for path in (
            "eval_results/gt", "eval_results/test/9B_sft",
            "temp/domain_logical_simulator/reports",
        )]
    paths, _, _ = parse_cli(argv, 3)
    report = evaluate_gt_candidate_preflight(Path(paths[0]), Path(paths[1]))
    write_gt_candidate_report(report, Path(paths[2]))
    for split, value in report["splits"].items():
        print(
            split,
            {
                "gt": value["gt_reference"]["status"],
                "candidate": value["candidate_self_check"]["status"],
                "direct_gt": value["direct_gt_replay"]["status"],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
