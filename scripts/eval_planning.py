from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import tempfile
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "src"))

from swm.llm import call_gpt
from swm.pddl.attribution import attribute_failure
from swm.pddl.eval_report import render_markdown_report
from swm.pddl.generation import parse_pddl_output, strip_code_block
from swm.pddl.planner import solve_pddl
from swm.pddl.reorder import plan_reorder
from swm.pddl.validate import validate_plan_feasibility

eval_model = "gpt-5.6-sol"
eval_base_url = None  # 默认由 swm.llm 路由；自定义本地服务时填写地址
# prompt_file = "training_input.txt"
prompt_file = "vlm_pddl.txt"
eval_mode = "pddl"  # pddl；problem_only；nl
prompt_path = root_dir / "src/swm/prompt_templates" / prompt_file
eval_root = root_dir / "eval_results" / eval_model / prompt_path.stem

judge_model = "deepseek/deepseek-flash"
# datasets = {
#     "swm_v3": "single_arm",
#     "unidomain": "single_arm",
#     "swm_dual_v2": "dual_arm",
# }
datasets = {
    "swm_v3": "single_arm",
    "swm_dual_v2": "dual_arm",
}
generation_max_workers = 100
enable_attribution = False  # 设为 True 启用失败归因
attribution_model = judge_model
judge_max_workers = generation_max_workers
attribution_max_workers = generation_max_workers
logger = logging.getLogger("swm.eval_planning")


def number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match[1]) if match else 0


def task_identifier(task: dict) -> str:
    return str(number(task["episode"])) if task["flat"] else f"{task['task']}/{task['episode']}"


def generation_complete(task: dict) -> bool:
    names = ("plan.txt",) if eval_mode == "nl" else ("domain.pddl", "problem.pddl", "plan.txt")
    files = (task["save_dir"] / name for name in names)
    return all(path.is_file() and path.read_text(encoding="utf-8").strip() for path in files)


def load_cached_status(task: dict):
    save_dir = task["save_dir"]
    judge_file = save_dir / "judge.json"
    if judge_file.exists():
        return "judge", json.loads(judge_file.read_text(encoding="utf-8")).get("pass") is True
    if (save_dir / "error.log").exists():
        return "error", False
    return None, False


def load_tasks() -> list[dict]:
    """统一读取扁平/嵌套任务，并确定各阶段共用的结果目录与 GT 目录。"""
    tasks = []
    for dataset_name, robot_configuration in datasets.items():
        instructions = json.loads(
            (root_dir / f"tasks/instructions/instructions_{dataset_name}.json").read_text(encoding="utf-8")
        )
        steps_data = json.loads(
            (root_dir / f"tasks/steps/steps_{dataset_name}.json").read_text(encoding="utf-8")
        )
        image_root = root_dir / "tasks/images" / dataset_name
        if dataset_name in instructions and isinstance(instructions[dataset_name], dict):
            instructions = instructions[dataset_name]
        if dataset_name in steps_data and isinstance(steps_data[dataset_name], dict):
            steps_data = steps_data[dataset_name]

        flat = all(isinstance(value, str) for value in instructions.values())
        groups = {dataset_name: instructions} if flat else instructions
        for task_name in sorted(groups, key=number):
            episodes = groups[task_name]
            if not isinstance(episodes, dict):
                continue
            task_steps = steps_data if flat else steps_data.get(task_name)
            image_dirs = [image_root] if flat else [image_root / task_name, image_root]
            for episode_name in sorted(episodes, key=number):
                images = (directory / f"{episode_name}.{ext}"
                          for ext in ("png", "jpg", "jpeg") for directory in image_dirs)
                raw_steps = task_steps.get(episode_name) if isinstance(task_steps, dict) else None
                if isinstance(raw_steps, dict):
                    raw_steps = [raw_steps[key] for key in sorted(raw_steps, key=number)]
                if isinstance(raw_steps, list):
                    raw_steps = "\n".join(map(str, raw_steps))

                relative = Path(dataset_name) / episode_name if flat else Path(dataset_name) / task_name / episode_name
                gt_dir = root_dir / "eval_results/gt" / relative
                task = {
                    "dataset": dataset_name, "task": task_name, "episode": episode_name,
                    "robot_configuration": robot_configuration,
                    "instruction": episodes[episode_name],
                    "image": next((path for path in images if path.exists()), None),
                    "kf_actions": "" if raw_steps is None else str(raw_steps), "flat": flat,
                    "save_dir": eval_root / relative, "gt_dir": gt_dir,
                }
                if eval_mode == "problem_only":
                    round_dir = max(
                        (p for p in gt_dir.glob("round*") if p.is_dir() and re.fullmatch(r"round\d+", p.name)),
                        key=lambda p: number(p.name), default=None,
                    )
                    if round_dir is None:
                        raise FileNotFoundError(f"No GT round directory in {gt_dir}")
                    task["gt_domain"] = (round_dir / "domain.pddl").read_text(encoding="utf-8")
                    task["kf_actions"] = (gt_dir / "kf_actions.txt").read_text(encoding="utf-8")
                    if not task["gt_domain"].strip() or not task["kf_actions"].strip():
                        raise ValueError(f"Empty GT domain or reference actions in {gt_dir}")
                tasks.append(task)
    return tasks


def generate_one(task: dict):
    image_path = task["image"]
    save_dir = task["save_dir"]
    domain_file = save_dir / "domain.pddl"
    problem_file = save_dir / "problem.pddl"
    plan_file = save_dir / "plan.txt"
    try:
        if generation_complete(task):
            return task, True, "cached"

        if image_path is None or not image_path.exists():
            return task, False, "missing_image"

        save_dir.mkdir(parents=True, exist_ok=True)

        prompt = prompt_path.read_text(encoding="utf-8").format(
            instruction=task["instruction"],
            instructions=task["instruction"],
            robot_configuration=task["robot_configuration"],
            domain=f"\n{task['gt_domain']}\n" if eval_mode == "problem_only" else "",
        )
        for retries_left in range(10, -1, -1):
            output = call_gpt(eval_model, prompt, [image_path], base_url=eval_base_url)
            try:
                if eval_mode == "nl":
                    data = json.loads(strip_code_block(output))
                    steps = data["plan_sequence"]
                    if not isinstance(steps, list) or not steps or any(
                        not isinstance(step, str) or not step.strip() for step in steps
                    ):
                        raise ValueError("plan_sequence must be a nonempty list of non-blank strings")
                elif eval_mode == "problem_only":
                    domain = task["gt_domain"]
                    match = re.search(r"<problem>\s*(.*?)\s*</problem>", output, flags=re.S)
                    if match is None or not (problem := strip_code_block(match[1])):
                        raise ValueError("expected a nonempty <problem> block")
                else:
                    domain, problem = parse_pddl_output(output)
                break
            except (ValueError, KeyError, TypeError):
                if retries_left == 0:
                    raise

        if eval_mode == "nl":
            if "reasoning" in data and (reasoning := str(data["reasoning"]).strip()):
                (save_dir / "reasoning.txt").write_text(reasoning, encoding="utf-8")
            plan_file.write_text("\n".join(step.strip() for step in steps), encoding="utf-8")
            (save_dir / "error.log").unlink(missing_ok=True)
            return task, True, "nl"

        domain_file.write_text(domain, encoding="utf-8")
        problem_file.write_text(problem, encoding="utf-8")
        if not solve_pddl(domain_file, problem_file, reorder_plan=False):
            return task, False, "pddl_unsolvable"
        if not plan_file.exists() or not plan_file.read_text(encoding="utf-8").strip():
            return task, False, "missing_pddl_plan"
        return task, True, "pddl"

    except Exception as e:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        return task, False, str(e)


def judge_one(task: dict):
    image_path = task["image"]
    save_dir = task["save_dir"]
    plan_file = save_dir / "plan.txt"
    judge_file = save_dir / "judge.json"

    try:
        cached_status, passed = load_cached_status(task)
        if cached_status == "judge":
            return task, True, passed, "cached"

        if image_path is None or not image_path.exists():
            return task, False, False, "missing_image"

        if not plan_file.is_file():
            return task, False, False, "missing_plan"

        if eval_mode != "nl":
            # 新求解和缓存的 plan 均先重排，失败时回退后继续验证。
            original_plan = plan_file.read_text(encoding="utf-8")
            try:
                plan_reorder(
                    save_dir / "domain.pddl",
                    save_dir / "problem.pddl",
                    plan_file,
                    plan_file,
                )
            except Exception as e:
                plan_file.write_text(original_plan, encoding="utf-8")
                logger.warning("Plan reordering failed; using original plan %s: %s", plan_file, e)

        candidate_plan = plan_file.read_text(encoding="utf-8").strip()
        if not candidate_plan:
            return task, False, False, "empty_plan"

        result = validate_plan_feasibility(
            model=judge_model,
            first_img=image_path,
            instruction=task["instruction"],
            kf_actions=task["kf_actions"],
            candidate_plan=candidate_plan,
            predicted_domain=(save_dir / "domain.pddl") if eval_mode != "nl" else None,
            pddl_plan=plan_file if eval_mode != "nl" else None,
            **({"scene_context": f"The robot configuration is {task['robot_configuration']}."}
               if eval_mode == "nl" and "robot_configuration" in task else {}),
        )

        judge_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (save_dir / "judge_error.log").unlink(missing_ok=True)
        return task, True, result["pass"] is True, "done"

    except Exception as e:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "judge_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        return task, False, False, str(e)


def write_summary(all_tasks: list[dict]) -> Path:
    """汇总本次评测结果与失败归因。"""
    records = {dataset: [] for dataset in datasets}
    for task in all_tasks:
        records[task["dataset"]].append({
            "id": task_identifier(task), "directory": task["save_dir"],
            "solved": generation_complete(task), "passed": load_cached_status(task)[1],
            "attribution": task.get("attribution") if eval_mode != "nl" else None,
        })
    eval_root.mkdir(parents=True, exist_ok=True)
    dataset_tag = "_".join(datasets)
    report_path = eval_root / f"summary_{dataset_tag}.md"
    report_text, attribution_summary = render_markdown_report(records, eval_root, pddl=eval_mode != "nl")
    report_path.write_text(report_text, encoding="utf-8")
    if (legacy_path := report_path.with_suffix(".log")).is_file():
        legacy_path.write_text(f"Report moved to Markdown: {report_path.name}\n", encoding="utf-8")
    if enable_attribution and eval_mode != "nl":
        (eval_root / f"attribution_summary_{dataset_tag}.json").write_text(
            json.dumps(attribution_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return report_path


def prepare_task(task):
    status, _ = load_cached_status(task)
    if status == "judge" or (
        status == "error" and eval_mode != "nl"
        and all((task["save_dir"] / name).is_file() for name in ("domain.pddl", "problem.pddl"))
    ) or generation_complete(task):
        return task, True, "cached"
    return generate_one(task)


def generate_tasks(tasks, progress):
    progress("generation", 0, len(tasks))
    with ThreadPoolExecutor(max_workers=generation_max_workers) as executor:
        futures = [executor.submit(prepare_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            task, ok, info = future.result()
            if not ok:
                logger.warning("Generation failed %s/%s: %s", task["dataset"], task_identifier(task), info)
            progress("generation", completed, len(tasks))
    progress("generation", len(tasks), len(tasks), state="done")


def attribute_one(task):
    with tempfile.TemporaryDirectory() as directory:
        for name in ("domain.pddl", "problem.pddl", "plan.txt", "judge.json"):
            source = task["save_dir"] / name
            if source.is_file():
                shutil.copy2(source, directory)
        return attribute_failure(
            directory, task["gt_dir"], task["instruction"], task["image"],
            model=attribution_model, reasoning_effort="medium",
        )


def judge_and_attribute(all_tasks, progress):
    """Run independent pools; a failed verdict immediately schedules attribution."""
    attribution_enabled = enable_attribution and eval_mode != "nl"
    pending_judges, known_failures = [], []
    judged = 0
    for task in all_tasks:
        status, passed = load_cached_status(task)
        complete = generation_complete(task)
        if complete:
            if status == "judge":
                judged += 1
            else:
                pending_judges.append(task)
        if attribution_enabled and not passed and (status == "judge" or not complete):
            known_failures.append(task)

    solved = judged + len(pending_judges)
    stats = {"attributed": 0, "classified": 0, "else_count": 0}
    attribution_total = len(known_failures)
    errors = []

    def report(*, judge=False, done=False):
        if judge:
            state = ("failed" if errors else "done") if done or judged == solved else "running"
            progress("judge", judged, solved, state=state)
        if attribution_enabled:
            progress("attribution", stats["attributed"], attribution_total, state="done" if done else "running")

    report(judge=True)
    with ThreadPoolExecutor(max_workers=judge_max_workers) as judges, \
         ThreadPoolExecutor(max_workers=attribution_max_workers) as attributions:
        futures = {attributions.submit(attribute_one, task): ("attribution", task)
                   for task in known_failures}
        # Probe the service once before sending a large batch. Attribution of
        # already-known failures continues while this first request is slow.
        probe = judges.submit(judge_one, pending_judges[0]) if pending_judges else None
        if probe is not None:
            futures[probe] = ("judge", pending_judges[0])
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                stage, task = futures.pop(future)
                if stage == "attribution":
                    result = future.result()
                    task["attribution"] = result
                    stats["attributed"] += 1
                    if result is not None:
                        stats["else_count" if result["label"] == "else" else "classified"] += 1
                else:
                    _, ok, passed, info = future.result()
                    judged += 1
                    if not ok:
                        message = f"{task['dataset']}/{task_identifier(task)}: {info}"
                        errors.append(message)
                        logger.error("Judge request failed %s", message)
                    else:
                        if not passed and attribution_enabled:
                            futures[attributions.submit(attribute_one, task)] = ("attribution", task)
                            attribution_total += 1
                        if future is probe:
                            for remaining in pending_judges[1:]:
                                futures[judges.submit(judge_one, remaining)] = ("judge", remaining)
                report(judge=stage == "judge")
    report(judge=True, done=True)
    if errors:
        raise RuntimeError(f"Judge errors={len(errors)}; first error: {errors[0]}. Rerun to resume.")
    return stats


def main() -> int:
    if eval_mode not in {"pddl", "problem_only", "nl"}:
        raise ValueError(f"Unknown eval_mode: {eval_mode!r}")
    print(f"full evaluation | model={eval_model} | prompt={prompt_file} | generation_workers={generation_max_workers} "
          f"judge_workers={judge_max_workers} attribution_enabled={enable_attribution} "
          f"attribution_workers={attribution_max_workers}", flush=True)
    previous = {}

    def progress(stage, completed, total, *, state="running"):
        mark = state, completed * 2 // max(total, 1)
        if previous.get(stage) != mark:
            previous[stage] = mark
            print(f"{eval_model} | {stage} {completed}/{total} {state}", flush=True)

    try:
        eval_root.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=eval_root / "evaluation.log", level=logging.WARNING, force=True,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s", encoding="utf-8")
        logger.setLevel(logging.INFO)
        logger.info("Evaluation started model=%s prompt=%s", eval_model, prompt_file)
        tasks = load_tasks()
        generate_tasks(tasks, progress)
        stats = judge_and_attribute(tasks, progress)
        report = write_summary(tasks)
        statuses = [load_cached_status(task) for task in tasks]
        logger.info("Evaluation done %s", dict(total=len(tasks), solved=sum(generation_complete(task) for task in tasks),
                    judged=sum(status == "judge" for status, _ in statuses),
                    passed=sum(passed for _, passed in statuses), **stats, report=str(report)))
    except Exception as error:
        logger.exception("Evaluation failed")
        print(f"{eval_model} | failed | {type(error).__name__}: {error}", flush=True)
        return 1
    print(f"{eval_model} | done | {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
