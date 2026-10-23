from __future__ import annotations

import json
import re
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

root_dir = Path("/home/xyx/下载/swm")
sys.path.insert(0, str(root_dir / "src"))

from swm.llm import call_gpt
from swm.pddl.attribution import attribute_failure
from swm.pddl.eval_report import render_markdown_report
from swm.pddl.generation import parse_pddl_output, strip_code_block
from swm.pddl.planner import solve_pddl
from swm.pddl.postprocess import repair_pddl
from swm.pddl.reorder import plan_reorder
from swm.pddl.strips import format_action_conditions, validate_untyped_pddl
from swm.pddl.validate import validate_plan_feasibility

# 基本配置
eval_models = ["9B_full_1019", "9B_full_1019_nopro"]  # 按顺序对应本地端口 8001、8002、8003……
judge_model = "Qwen3.8-27B"
attribution_model = "Qwen3.8-27B"
ROBOT_CONFIGURATION = "single_arm"
datasets = ["swm", "swm_v2", "unidomain"]
generation_max_workers = 200
judge_max_workers = generation_max_workers
prompt_files = ["training_input.txt", "training_input_only_problem.txt"]


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
        raise ValueError(f"未知 prompt 文件名，无法判断评测模式: {prompt_path.name}")
    eval_root = root_dir / "eval_results" / eval_model
    if problem_only:
        eval_root /= "only_problem"


configure_eval(eval_models[0], prompt_files[0])


def number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match[1]) if match else 0


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
            # PFV 前统一按人类偏好重排一次，覆盖新求解和缓存的 plan。
            plan_reorder(save_dir / "domain.pddl", save_dir / "problem.pddl", plan_file, plan_file)

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
        return task, True, result["pass"] is True, "done"

    except Exception as e:
        return task, False, False, str(e)


def write_summary(all_tasks: list[dict]) -> Path:
    """汇总本次评测结果与失败归因。"""
    records = {dataset: [] for dataset in datasets}
    for task in all_tasks:
        save_dir = task["save_dir"]
        attribution_path = save_dir / "attribution.json"
        result = None
        if eval_mode == "pddl" and attribution_path.is_file():
            try:
                result = json.loads(attribution_path.read_text(encoding="utf-8"))
                if not isinstance(result, dict):
                    result = None
            except (OSError, ValueError):
                pass
        identifier = str(number(task["episode"])) if task["flat"] else f"{task['task']}/{task['episode']}"
        records[task["dataset"]].append({
            "id": identifier, "directory": save_dir,
            "solved": generation_complete(task), "passed": load_cached_status(task)[1],
            "attribution": result,
        })
    eval_root.mkdir(parents=True, exist_ok=True)
    dataset_tag = "_".join(datasets)
    report_path = eval_root / f"summary_{dataset_tag}.md"
    report_text, attribution_summary = render_markdown_report(records, eval_root, pddl=eval_mode == "pddl")
    report_path.write_text(report_text, encoding="utf-8")
    legacy_path = report_path.with_suffix(".log")
    if legacy_path.is_file():
        legacy_path.write_text(f"报告已改为 Markdown：{report_path.name}\n", encoding="utf-8")
    if eval_mode == "pddl":
        (eval_root / f"attribution_summary_{dataset_tag}.json").write_text(
            json.dumps(attribution_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    print(f"报告已保存到: {report_path}")
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


def main():
    all_tasks = load_tasks()
    prefix = f"[{eval_model}/{prompt_path.stem}]"
    print(f"{prefix} 开始：{len(all_tasks)} 个任务", flush=True)
    if eval_mode == "pddl":
        with ThreadPoolExecutor(max_workers=generation_max_workers) as executor:
            list(executor.map(repair_saved_task, all_tasks))

    generation_tasks = []
    judge_tasks = []
    for task in all_tasks:
        cached_status, _ = load_cached_status(task)
        failed_candidate = cached_status == "error" and (
            all((task["save_dir"] / name).is_file() for name in ("domain.pddl", "problem.pddl"))
            if eval_mode == "pddl" else generation_complete(task)
        )
        if cached_status == "judge" or failed_candidate:
            continue
        if generation_complete(task):
            judge_tasks.append(task)
        else:
            generation_tasks.append(task)

    skipped = len(all_tasks) - len(generation_tasks) - len(judge_tasks)
    print(f"{prefix} 缓存 {skipped}，待生成 {len(generation_tasks)}，待判定 {len(judge_tasks)}", flush=True)
    with ThreadPoolExecutor(max_workers=generation_max_workers) as executor:
        futures = [executor.submit(generate_one, task) for task in generation_tasks]
        for future in as_completed(futures):
            task, ok, _ = future.result()
            if ok:
                judge_tasks.append(task)

    print(f"{prefix} 判定 {len(judge_tasks)} 个任务", flush=True)
    # 先验证 Judge 可用，再并发判定其余任务。
    if judge_tasks:
        _, ok, _, info = judge_one(judge_tasks[0])
        if not ok:
            raise SystemExit(
                f"❌ Judge 生成失败: {info}\n"
                "PDDL 结果已保留；请在可访问 Judge API 的环境中重新运行本脚本。"
            )

    with ThreadPoolExecutor(max_workers=judge_max_workers) as executor:
        futures = [executor.submit(judge_one, task) for task in judge_tasks[1:]]

        for future in as_completed(futures):
            task, ok, _, info = future.result()
            if not ok:
                raise SystemExit(
                    f"❌ Judge 生成失败: "
                    f"{task['dataset']}/{task['task']}/{task['episode']}: {info}\n"
                    "已生成的 PDDL 和 Judge 结果均已保留，重新运行将自动续传。"
                )

    # 只给非通过的 PDDL 任务归因；不可解、无 plan/无 judge 同样进入。
    if eval_mode == "pddl":
        attribution_tasks = [task for task in all_tasks if not load_cached_status(task)[1]]
        print(f"{prefix} 归因 {len(attribution_tasks)} 个任务", flush=True)
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(
                attribute_failure, task["save_dir"], task["gt_dir"], task["instruction"], task["image"],
                model=attribution_model, reasoning_effort="medium",
            ) for task in attribution_tasks]
            for future in as_completed(futures):
                future.result()
    write_summary(all_tasks)


def evaluate_model(model: str, port: int):
    """Each model has its own process; run both prompts with isolated globals."""
    for prompt_file in prompt_files:
        configure_eval(model, prompt_file, port)
        print(f"\n模型: {eval_model} | 服务: {eval_base_url} | 模板: {prompt_file}", flush=True)
        main()


if __name__ == "__main__":
    if len(set(eval_models)) != len(eval_models):
        raise ValueError("模型名不能重复，否则评测结果目录会冲突")
    with ProcessPoolExecutor(max_workers=len(eval_models)) as executor:
        futures = [executor.submit(evaluate_model, model, 8001 + i)
                   for i, model in enumerate(eval_models)]
        for future in as_completed(futures):
            future.result()
