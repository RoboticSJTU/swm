from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory

root_dir = Path("/inspire/hdd/global_user/xiaoyunxiao-240108120113/swm")
sys.path.insert(0, str(root_dir / "src"))

from swm.llm import call_gpt
from swm.pddl.attribution import attribute_failure
from swm.pddl.eval_progress import ProgressDisplay, print_results
from swm.pddl.eval_report import render_markdown_report
from swm.pddl.generation import parse_pddl_output, strip_code_block
from swm.pddl.planner import solve_pddl
from swm.pddl.postprocess import repair_pddl
from swm.pddl.reorder import plan_reorder
from swm.pddl.strips import format_action_conditions, validate_untyped_pddl
from swm.pddl.validate import validate_plan_feasibility

# 基本配置
eval_models = ["9B_1031"]  # 按顺序对应本地端口 8001、8002、8003……
judge_model = "Qwen3.8-27B"
# ROBOT_CONFIGURATION = "single_arm"
# datasets = ["swm", "swm_v2", "unidomain"]
ROBOT_CONFIGURATION = "dual_arm"
datasets = ["swm_dual"]
generation_max_workers = 200
attribution_model = judge_model
judge_max_workers = generation_max_workers
attribution_max_workers = generation_max_workers
logger = logging.getLogger("swm.eval_planning")
prompt_files = ["training_input.txt"]


def configure_eval(model: str, prompt_file: str, port: int = 8001):
    global eval_model, eval_base_url, prompt_path, problem_only, eval_mode, eval_root
    eval_model = model
    eval_base_url = f"http://127.0.0.1:{port}/v1"
    prompt_path = root_dir / "src/swm/prompt_templates" / prompt_file
    problem_only = prompt_path.name == "training_input_only_problem.txt"
    if prompt_path.name in {"training_input.txt", "training_input_only_problem.txt"}:
        eval_mode = "pddl"
    elif prompt_path.name == "vlm_cot.txt":
        eval_mode = "nl"
    else:
        raise ValueError(f"Unknown prompt template: {prompt_path.name}")
    eval_root = root_dir / "eval_results" / eval_model
    if problem_only:
        eval_root /= "only_problem"


configure_eval(eval_models[0], prompt_files[0])


def number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match[1]) if match else 0


def task_identifier(task: dict) -> str:
    return str(number(task["episode"])) if task["flat"] else f"{task['task']}/{task['episode']}"


def generation_complete(task: dict) -> bool:
    names = ("domain.pddl", "problem.pddl", "plan.txt") if eval_mode == "pddl" else ("plan.txt",)
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
    for dataset_name in datasets:
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
                    "instruction": episodes[episode_name],
                    "image": next((path for path in images if path.exists()), None),
                    "kf_actions": "" if raw_steps is None else str(raw_steps), "flat": flat,
                    "save_dir": eval_root / relative, "gt_dir": gt_dir,
                }
                if problem_only:
                    rounds = [p for p in gt_dir.glob("round*")
                              if p.is_dir() and re.fullmatch(r"round\d+", p.name)]
                    if not rounds:
                        raise FileNotFoundError(f"No GT round directory in {gt_dir}")
                    round_dir = max(rounds, key=lambda p: number(p.name))
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
            robot_configuration=ROBOT_CONFIGURATION,
            domain=f"\n{task['gt_domain']}\n" if problem_only else "",
        )
        output = call_gpt(eval_model, prompt, [image_path], base_url=eval_base_url)

        if eval_mode != "pddl":
            candidate_plan = strip_code_block(output)
            reasoning = ""
            try:
                data = json.loads(candidate_plan)
                plan = "\n".join(step for item in data["plan_sequence"] if (step := str(item).strip()))
                if plan:
                    reasoning = str(data["reasoning"]).strip() if "reasoning" in data else ""
                    candidate_plan = plan
            except Exception:
                pass
            if not candidate_plan:
                raise ValueError("empty natural-language plan")
            if reasoning:
                (save_dir / "reasoning.txt").write_text(reasoning, encoding="utf-8")
            plan_file.write_text(candidate_plan, encoding="utf-8")
            return task, True, "nl"

        if problem_only:
            domain = task["gt_domain"]
            match = re.search(r"<problem>\s*(.*?)\s*</problem>", output, flags=re.S)
            if match is None:
                raise ValueError("cannot parse problem-only PDDL output")
            problem = strip_code_block(match.group(1))
            if not problem:
                raise ValueError("empty problem-only PDDL output")
        else:
            domain, problem = parse_pddl_output(output)
        # 暂存原文，修复失败时仍可保留输入供归因。
        domain_file.write_text(domain, encoding="utf-8")
        problem_file.write_text(problem, encoding="utf-8")
        plan_file.unlink(missing_ok=True)
        domain, problem, changes = repair_pddl(domain, problem)
        validate_untyped_pddl(domain)
        validate_untyped_pddl(problem)
        if not problem_only:
            domain = format_action_conditions(domain)
        domain_file.write_text(domain, encoding="utf-8")
        problem_file.write_text(problem, encoding="utf-8")
        if changes:
            (save_dir / "postprocess.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n")
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

        if eval_mode == "pddl":
            # 新求解和缓存的 plan 均先重排，失败时回退后继续验证。
            original_plan = plan_file.read_text(encoding="utf-8")
            try:
                plan_reorder(
                    save_dir / "domain.pddl",
                    save_dir / "problem.pddl",
                    plan_file,
                    plan_file,
                    allow_effect_overlap=True,
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
            predicted_domain=(save_dir / "domain.pddl") if eval_mode == "pddl" else None,
            pddl_plan=plan_file if eval_mode == "pddl" else None,
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
        save_dir = task["save_dir"]
        records[task["dataset"]].append({
            "id": task_identifier(task), "directory": save_dir,
            "solved": generation_complete(task), "passed": load_cached_status(task)[1],
            "attribution": task.get("attribution") if eval_mode == "pddl" else None,
        })
    eval_root.mkdir(parents=True, exist_ok=True)
    dataset_tag = "_".join(datasets)
    report_path = eval_root / f"summary_{dataset_tag}.md"
    report_text, attribution_summary = render_markdown_report(records, eval_root, pddl=eval_mode == "pddl")
    report_path.write_text(report_text, encoding="utf-8")
    legacy_path = report_path.with_suffix(".log")
    if legacy_path.is_file():
        legacy_path.write_text(f"Report moved to Markdown: {report_path.name}\n", encoding="utf-8")
    if eval_mode == "pddl":
        (eval_root / f"attribution_summary_{dataset_tag}.json").write_text(
            json.dumps(attribution_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return report_path


def repair_saved_task(task: dict):
    """Apply the same built-in repair to cached candidates before cache reuse."""
    save_dir = task["save_dir"]
    try:
        domain, problem, changes = repair_pddl(
            (save_dir / "domain.pddl").read_text(encoding="utf-8"),
            (save_dir / "problem.pddl").read_text(encoding="utf-8"),
        )
    except (OSError, ValueError, NotImplementedError):
        return False
    if not changes:
        return False
    for name in ("plan.txt", "judge.json", "attribution.json", "error.log"):
        (save_dir / name).unlink(missing_ok=True)
    shutil.rmtree(save_dir / "attribution_solver", ignore_errors=True)
    (save_dir / "domain.pddl").write_text(domain, encoding="utf-8")
    (save_dir / "problem.pddl").write_text(problem, encoding="utf-8")
    (save_dir / "postprocess.json").write_text(
        json.dumps(changes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    solve_pddl(save_dir / "domain.pddl", save_dir / "problem.pddl", reorder_plan=False)
    return True


def prepare_task(task):
    """Repair and inspect a cache before deciding whether generation is needed."""
    if eval_mode == "pddl":
        repair_saved_task(task)
    status, _ = load_cached_status(task)
    failed_candidate = status == "error" and (
        all((task["save_dir"] / name).is_file() for name in ("domain.pddl", "problem.pddl"))
        if eval_mode == "pddl" else generation_complete(task)
    )
    if status == "judge" or failed_candidate or generation_complete(task):
        return task, True, "cached"
    return generate_one(task)


def generate_tasks(tasks, progress):
    stats = {"solved": 0, "failed": 0, "cached": 0}
    progress("generation", 0, len(tasks), **stats)
    with ThreadPoolExecutor(max_workers=generation_max_workers) as executor:
        futures = [executor.submit(prepare_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            task, ok, info = future.result()
            solved = generation_complete(task)
            stats["solved"] += solved
            stats["failed"] += not solved
            stats["cached"] += info == "cached"
            if not ok:
                logger.warning("Generation failed %s/%s: %s", task["dataset"], task_identifier(task), info)
            progress("generation", completed, len(tasks), **stats)
    progress("generation", len(tasks), len(tasks), state="done", **stats)


def attribute_one(task):
    with TemporaryDirectory(prefix="swm_attribution_") as temporary:
        artifact_dir = Path(temporary)
        result = attribute_failure(
            task["save_dir"], task["gt_dir"], task["instruction"], task["image"],
            model=attribution_model, reasoning_effort="medium", artifact_dir=artifact_dir,
        )
        # Preserve parser evidence for the report after temporary files are removed.
        solver_path = artifact_dir / "attribution_solver/result.json"
        if result is not None and solver_path.is_file():
            result["checks"]["fast_downward"] = json.loads(solver_path.read_text(encoding="utf-8"))
        return result


def judge_and_attribute(all_tasks, progress):
    """Run independent pools; a failed verdict immediately schedules attribution."""
    pending_judges, known_failures = [], []
    judged = passed_count = failed_count = cached = solved = errors = 0
    for task in all_tasks:
        status, passed = load_cached_status(task)
        complete = generation_complete(task)
        if complete:
            solved += 1
            if status == "judge":
                judged += 1
                cached += 1
                passed_count += passed
                failed_count += not passed
            else:
                pending_judges.append(task)
        if not passed and (status == "judge" or not complete):
            known_failures.append(task)

    stats = {"attributed": 0, "classified": 0, "else_count": 0}
    attribution_total = len(known_failures) if eval_mode == "pddl" else 0

    def report(*, judge=False, done=False):
        if judge:
            state = ("failed" if errors else "done") if done or judged == solved else "running"
            progress("judge", judged, solved, state=state, passed=passed_count,
                     failed=failed_count, cached=cached, errors=errors)
        if eval_mode == "pddl":
            progress("attribution", stats["attributed"], attribution_total, state="done" if done else "running",
                     classified=stats["classified"], else_count=stats["else_count"],
                     **{"unjudged" if errors and done else "waiting": solved - judged})

    report(judge=True)
    first_error = None
    with ThreadPoolExecutor(max_workers=judge_max_workers) as judges, \
         ThreadPoolExecutor(max_workers=attribution_max_workers) as attributions:
        futures = {attributions.submit(attribute_one, task): ("attribution", task)
                   for task in known_failures} if eval_mode == "pddl" else {}
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
                        errors += 1
                        message = f"{task['dataset']}/{task_identifier(task)}: {info}"
                        first_error = first_error or message
                        logger.error("Judge request failed %s", message)
                    else:
                        passed_count += passed
                        failed_count += not passed
                        if not passed and eval_mode == "pddl":
                            futures[attributions.submit(attribute_one, task)] = ("attribution", task)
                            attribution_total += 1
                        if future is probe:
                            for remaining in pending_judges[1:]:
                                futures[judges.submit(judge_one, remaining)] = ("judge", remaining)
                report(judge=stage == "judge")
    report(judge=True, done=True)
    if errors:
        raise RuntimeError(f"Judge errors={errors}; first error: {first_error}. Rerun to resume.")
    return stats


def evaluate_model(model: str, port: int, *, model_index, events):
    """Workers only publish progress; the parent owns all terminal output."""
    results = []
    for prompt_index, prompt_file in enumerate(prompt_files):
        configure_eval(model, prompt_file, port)
        eval_root.mkdir(parents=True, exist_ok=True)
        log_path = eval_root / "evaluation.log"
        logging.basicConfig(filename=log_path, level=logging.WARNING, force=True,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s", encoding="utf-8")
        logger.setLevel(logging.INFO)
        label = model if len(prompt_files) == 1 else f"{model}/{prompt_path.stem}"
        result = {"run": f"{model_index}:{prompt_index}", "label": label, "log": str(log_path)}

        def progress(stage, completed, total, *, state="running", **stats):
            events.put({"run": result["run"], "stage": stage, "completed": completed,
                        "total": total, "state": state, "stats": stats})

        try:
            logger.info("Evaluation started model=%s prompt=%s", model, prompt_file)
            tasks = load_tasks()
            generate_tasks(tasks, progress)
            stats = judge_and_attribute(tasks, progress)
            report = write_summary(tasks)
            statuses = [load_cached_status(task) for task in tasks]
            result.update(total=len(tasks), solved=sum(generation_complete(task) for task in tasks),
                          judged=sum(status == "judge" for status, _ in statuses),
                          passed=sum(passed for _, passed in statuses), **stats, report=str(report), status="done")
            logger.info("Evaluation done %s", result)
        except Exception as error:
            logger.exception("Evaluation failed")
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        results.append(result)
    return results


def run_evaluations():
    if not eval_models or len(set(eval_models)) != len(eval_models):
        raise ValueError("At least one model is required; model names must be unique")
    runs = []
    for model_index, model in enumerate(eval_models):
        for prompt_index, prompt_file in enumerate(prompt_files):
            stages = ("generation", "judge") if prompt_file == "vlm_cot.txt" else ("generation", "judge", "attribution")
            runs.append({"id": f"{model_index}:{prompt_index}",
                         "label": model if len(prompt_files) == 1 else f"{model}/{Path(prompt_file).stem}", "stages": stages})
    print(f"full evaluation | models={len(eval_models)} | pddl_workers={generation_max_workers} "
          f"judge_workers={judge_max_workers} attribution_workers={attribution_max_workers}", flush=True)
    print("Attribution total grows as judge failures arrive. Details are saved in reports and logs.", flush=True)
    results = []
    with Manager() as manager, ProcessPoolExecutor(max_workers=len(eval_models)) as executor:
        events = manager.Queue()
        # Start workers before tqdm creates its terminal monitor thread.
        pending = {executor.submit(evaluate_model, model, 8001 + index,
                   model_index=index, events=events): index
                   for index, model in enumerate(eval_models)}
        display = ProgressDisplay(runs)
        try:
            while pending:
                try:
                    display.update(events.get(timeout=0.2))
                except Empty:
                    pass
                for future in list(pending):
                    if future.done():
                        index = pending.pop(future)
                        try:
                            results.extend(future.result())
                        except Exception as error:
                            results.append({"run": f"{index}:0", "label": eval_models[index], "status": "failed",
                                            "error": str(error), "log": str(root_dir / "eval_results" / eval_models[index] / "evaluation.log")})
                display.heartbeat()
            while True:
                try:
                    display.update(events.get_nowait())
                except Empty:
                    break
        finally:
            display.close()
    results.sort(key=lambda item: tuple(map(int, item["run"].split(":"))))
    print_results(results)
    return results


if __name__ == "__main__":
    results = run_evaluations()
    raise SystemExit(int(any(result["status"] != "done" for result in results)))
