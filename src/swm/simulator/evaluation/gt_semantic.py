from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from swm.pddl.strips import ground_plan, parse_domain, parse_plan, parse_problem_model

from swm.simulator.alignment import (
    compile_step,
    compile_world,
    replay_on_gt,
    VLMAlignmentAdvisor,
    VLMAlignmentError,
)
from swm.simulator.cli import build_vlm_advisor, parse_cli
from swm.simulator.cross_domain import run_cross_domain_files

from .gt_candidate import SPLITS, _task_id, select_highest_round

SCHEMA_VERSION = "domain_logical_simulator_semantic_gt_v3"


def _stable_error(error: Exception) -> str:
    message = str(error).splitlines()[0]
    message = re.sub(
        r"/tmp/logical-sim-domain-[^/\s:]+/",
        "<temporary-domain>/",
        message,
    )
    return f"{type(error).__name__}: {message}"


def original_judge_pass(
    judge: dict[str, Any], current_pass: bool | None
) -> bool | None:
    """Recover the earliest retained label without treating it as unquestioned GT."""
    correction = judge.get("label_correction")
    if isinstance(correction, dict) and isinstance(correction.get("previous_pass"), bool):
        return correction["previous_pass"]
    history = judge.get("label_correction_history")
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict) and isinstance(item.get("previous_pass"), bool):
                return item["previous_pass"]
    return current_pass


def _task_dirs(path: Path) -> dict[int, Path]:
    return {
        _task_id(item): item
        for item in path.glob("task_*")
        if item.is_dir()
    }


def _load(directory: Path):
    schemas = parse_domain(directory / "domain.pddl")
    problem = parse_problem_model(directory / "problem.pddl", schemas)
    world = compile_world(schemas, problem)
    return schemas, problem, world


def _translated_initial_facts(candidate_world, mapping: dict[str, str]) -> frozenset[tuple[str, ...]]:
    facts = set()
    for fact in candidate_world.facts:
        if fact[0].startswith(("kind:", "static:")):
            continue
        translated = [fact[0]]
        for argument in fact[1:]:
            target = mapping.get(argument)
            if target is None:
                break
            translated.append(target)
        else:
            facts.add(tuple(translated))
    return frozenset(facts)


def _initial_comparison(candidate_world, gt_world, mapping: dict[str, str]) -> dict[str, Any]:
    predicates = {
        "hand_free", "holding", "on", "in", "inserted", "open", "closed",
        "power_on", "power_off", "locked", "unlocked", "upright",
        "upside_down", "flat", "vertical", "blocks", "blocks_opening",
        "blocks_closing", "under",
    }
    candidate_facts = {item for item in _translated_initial_facts(candidate_world, mapping) if item[0] in predicates}
    gt_facts = {item for item in gt_world.facts if item[0] in predicates}
    shared = candidate_facts & gt_facts
    return {
        "shared": len(shared),
        "candidate_only": [list(item) for item in sorted(candidate_facts - gt_facts)],
        "gt_only": [list(item) for item in sorted(gt_facts - candidate_facts)],
        "agreement": len(shared) / len(candidate_facts | gt_facts) if candidate_facts or gt_facts else 1.0,
    }


def _run_plan(
    directory: Path,
    schemas,
    problem,
    source_world,
    gt_world,
    mapping,
    *,
    instruction: str,
    trusted_declared_semantics: bool = False,
):
    raw, _ = parse_plan(directory / "plan.txt")
    actions = ground_plan(raw, schemas, problem.objects)
    steps = tuple(
        compile_step(
            action,
            source_world,
            mapping,
            trusted_declared_semantics=trusted_declared_semantics,
        )
        for action in actions
    )
    return replay_on_gt(gt_world, steps, instruction=instruction), steps


def _instructions(gt_root: Path, split: str) -> dict[str, str]:
    path = gt_root.parents[1] / "tasks" / "instructions" / f"instructions_{split}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get(split, {})


def _task_image(gt_root: Path, split: str, task_id: int) -> Path | None:
    root = gt_root.parents[1] / "tasks" / "images"
    directories = (
        (root / "swm_100" / "swm", root / "swm")
        if split == "swm"
        else (root / "unidomain", root / "swm_100" / "unidomain")
    )
    for directory in directories:
        matches = sorted(directory.glob(f"task_{task_id}.*"))
        if matches:
            return matches[0]
    return None


def evaluate_semantic_gt(
    gt_root: Path,
    candidate_root: Path,
    *,
    splits: tuple[str, ...] = SPLITS,
    alignment_advisor: VLMAlignmentAdvisor | None = None,
) -> dict[str, Any]:
    gt_root = gt_root.resolve()
    candidate_root = candidate_root.resolve()
    reports: dict[str, Any] = {}
    for split in splits:
        instructions = _instructions(gt_root, split)
        gt_tasks = _task_dirs(gt_root / split)
        candidate_tasks = _task_dirs(candidate_root / split)
        task_ids = sorted(set(gt_tasks) | set(candidate_tasks))
        gt_status: Counter[str] = Counter()
        candidate_status: Counter[str] = Counter()
        issue_categories: Counter[str] = Counter()
        judge_matrix: Counter[str] = Counter()
        alignment_status: Counter[str] = Counter()
        rows = []
        for task_id in task_ids:
            row: dict[str, Any] = {"task_id": task_id}
            instruction = instructions.get(f"task_{task_id}")
            if not isinstance(instruction, str) or not instruction.strip():
                row["status"] = "INPUT_ERROR"
                row["error"] = f"missing instruction for {split}/task_{task_id}"
                candidate_status["INPUT_ERROR"] += 1
                rows.append(row)
                continue
            gt_task = gt_tasks.get(task_id)
            candidate_dir = candidate_tasks.get(task_id)
            if gt_task is None:
                row["status"] = "MISSING_GT_TASK"
                candidate_status["MISSING_GT_TASK"] += 1
                rows.append(row)
                continue
            gt_dir = select_highest_round(gt_task)
            try:
                gt_schemas, gt_problem, gt_world = _load(gt_dir)
            except Exception as error:
                gt_status["INPUT_ERROR"] += 1
                row["gt_reference"] = {
                    "status": "INPUT_ERROR",
                    "error": _stable_error(error),
                }
                candidate_status["GT_INPUT_ERROR"] += 1
                rows.append(row)
                continue
            try:
                identity = {item.name: item.name for item in gt_world.objects}
                gt_result, _ = _run_plan(
                    gt_dir,
                    gt_schemas,
                    gt_problem,
                    gt_world,
                    gt_world,
                    identity,
                    instruction=instruction,
                    trusted_declared_semantics=True,
                )
                gt_status[gt_result.status.value] += 1
                row["gt_reference"] = {
                    "round": int(gt_dir.name.removeprefix("round")),
                    "status": gt_result.status.value,
                    "first_issue": None
                    if gt_result.first_issue is None
                    else gt_result.first_issue.__dict__,
                }
            except Exception as error:
                gt_status["INPUT_ERROR"] += 1
                row["gt_reference"] = {
                    "status": "INPUT_ERROR",
                    "error": _stable_error(error),
                }
            if candidate_dir is None or not (candidate_dir / "plan.txt").exists():
                row["status"] = "MISSING_PLAN"
                candidate_status["MISSING_PLAN"] += 1
                rows.append(row)
                continue
            try:
                run = run_cross_domain_files(
                    gt_dir / "domain.pddl",
                    gt_dir / "problem.pddl",
                    candidate_dir / "domain.pddl",
                    candidate_dir / "problem.pddl",
                    candidate_dir / "plan.txt",
                    instruction,
                    alignment_advisor,
                    _task_image(gt_root, split, task_id),
                )
                for item in run.alignment.objects:
                    alignment_status[item.status] += 1
                if (
                    alignment_advisor is not None
                    and alignment_advisor.require_complete
                    and run.result.status.value == "UNKNOWN"
                ):
                    category = (
                        run.result.first_issue.category
                        if run.result.first_issue is not None
                        else "unknown"
                    )
                    if category == "ambiguous_object_mapping":
                        raise VLMAlignmentError(
                            "decision-complete evaluation retained UNKNOWN: " + category
                        )
                candidate_status[run.result.status.value] += 1
                if run.result.first_issue is not None:
                    issue_categories[run.result.first_issue.category] += 1
                judge_path = candidate_dir / "judge.json"
                judge_pass = None
                judge_current_pass = None
                if judge_path.exists():
                    judge_value = json.loads(judge_path.read_text(encoding="utf-8"))
                    judge_current_pass = judge_value.get("pass")
                    judge_pass = original_judge_pass(
                        judge_value, judge_current_pass
                    )
                    if isinstance(judge_pass, bool):
                        judge_matrix[f"{run.result.status.value}|judge_{str(judge_pass).lower()}"] += 1
                row.update(
                    {
                        "status": run.result.status.value,
                        "first_issue": None
                        if run.result.first_issue is None
                        else run.result.first_issue.__dict__,
                        "judge_pass": judge_pass,
                        "judge_current_pass": judge_current_pass,
                        "alignment": run.alignment.to_dict(),
                        "identity_resolution": run.identity_resolution,
                        "candidate_domain_recovery": run.candidate_domain_recovery.to_dict(),
                        "initial_comparison": _initial_comparison(
                            run.candidate_world, run.gt_world, run.mapping
                        ),
                        "steps": len(run.steps),
                        "goal_contract": {
                            "positive": [list(item) for item in sorted(run.goal_contract.positive)],
                            "negative": [list(item) for item in sorted(run.goal_contract.negative)],
                        },
                    }
                )
            except Exception as error:
                candidate_status["INPUT_ERROR"] += 1
                issue_categories["input_error"] += 1
                row.update(
                    {
                        "status": "INPUT_ERROR",
                        "error": _stable_error(error),
                    }
                )
            rows.append(row)
        reports[split] = {
            "tasks": len(task_ids),
            "gt_reference_status": dict(sorted(gt_status.items())),
            "candidate_status": dict(sorted(candidate_status.items())),
            "issue_categories": dict(sorted(issue_categories.items())),
            "alignment_status": dict(sorted(alignment_status.items())),
            "judge_matrix": dict(sorted(judge_matrix.items())),
            "rows": rows,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "gt_root": str(gt_root),
            "candidate_root": str(candidate_root),
            "gt_round_policy": "highest numeric round",
        },
        "method": {
            "state_source": "GT canonical init",
            "goal_source": "instruction projection over GT canonical goal evidence",
            "candidate_usage": "object descriptions plus canonicalized operator/plan intent",
            "unmapped_policy": (
                "one image-grounded VLM completion for compatible aliases; FAIL only when no "
                "category-compatible GT object exists"
                if alignment_advisor is not None
                and alignment_advisor.require_complete
                else "UNKNOWN for compatible but unresolved aliases; FAIL only when no "
                "category-compatible GT object exists"
            ),
            "vlm_mapping": "disabled; unresolved plan objects remain UNKNOWN"
            if alignment_advisor is None
            else f"advisory ambiguous-object refinement via {alignment_advisor.model}",
        },
        "splits": reports,
    }


def write_semantic_gt_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "gt_9b_sft_semantic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Canonical GT Evaluation",
        "",
        "Candidate objects, predicates, and operators are compiled to one canonical",
        "interface. Replay starts from GT init. Completion uses the instruction-projected",
        "subset of GT goal evidence, so incidental demonstration staging is not required.",
        "",
    ]
    for split, value in report["splits"].items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Tasks: {value['tasks']}",
                f"- Canonical GT reference: {value['gt_reference_status']}",
                f"- Candidate result: {value['candidate_status']}",
                f"- Failure/unknown categories: {value['issue_categories']}",
                f"- Object alignment: {value['alignment_status']}",
                f"- Candidate / judge matrix: {value['judge_matrix']}",
                "",
            ]
        )
    (report_dir / "REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        root = Path(__file__).resolve().parents[4]
        argv = [str(root / path) for path in (
            "eval_results/gt", "eval_results/test/9B_sft",
            "temp/domain_logical_simulator/reports",
        )]
    paths, options, flags = parse_cli(
        argv,
        3,
        {"vlm-model", "vlm-base-url", "vlm-cache", "env-file", "vlm-api-key-env", "vlm-reasoning-effort"},
        {"vlm-mapping", "vlm-cache-only", "vlm-json-mode"},
    )
    advisor = build_vlm_advisor(options, flags)
    report = evaluate_semantic_gt(
        Path(paths[0]), Path(paths[1]), alignment_advisor=advisor
    )
    write_semantic_gt_report(report, Path(paths[2]))
    for split, value in report["splits"].items():
        print(split, value["candidate_status"], value["judge_matrix"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
