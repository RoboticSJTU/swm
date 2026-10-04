"""Evaluate the prepared external-paper datasets; keep eval_planning.py independent."""
from __future__ import annotations

import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import traceback
from swm.llm import call_gpt
from swm.pddl.judge import judge_pddl
from swm.pddl.gt_validation import VALIDATOR_VERSION, validate_gt_plan
from swm.pddl.planner import solve_pddl
from swm.pddl.strips import format_action_conditions, validate_untyped_pddl

# =========================
# 基本配置
# =========================
# 只需修改以下四项配置。
eval_model = "9B_sft_full"
judge_model = "Qwen3.8-27B"
max_workers = 200
datasets = [
    "vision_pddl_blocksworld_real",
    "vision_pddl_alfred_multi",
    "prodg_blocksworld",
    "prodg_cooking_diagnostic",
    "prodg_hanoi_ood",
    # ViPlan 以下三套使用初始观测进行静态评测。
    "viplan_bw_simple",
    "viplan_bw_medium",
    "viplan_bw_hard",
]

# 固定本机路径；沿用原全生成结果目录。
root_dir = Path("/home/xyx/下载/swm")
task_root = Path("/home/xyx/下载/swm/additional_experiments/task")
eval_root = Path("/home/xyx/下载/swm/eval_results") / eval_model / "additional_experiments" / "swm_full_generation"
prompt_path = root_dir / "src/swm/prompt_templates/training_input.txt"


# =========================
# 基础函数
# =========================
def number(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def strip_code_block(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:\w+)?\s*\n?(.*?)\n?```$", text, flags=re.S)
    return m.group(1).strip() if m else text


def steps_to_text(raw_steps) -> str:
    if isinstance(raw_steps, list):
        return "\n".join(str(x) for x in raw_steps)
    if isinstance(raw_steps, dict):
        return "\n".join(str(raw_steps[k]) for k in sorted(raw_steps.keys(), key=number))
    if raw_steps is None:
        return ""
    return str(raw_steps)


def parse_pddl_output(output: str) -> tuple[str, str]:
    output = output.strip()

    for domain_tag, problem_tag in [("domain", "problem"), ("domain_pddl", "problem_pddl")]:
        domain_match = re.search(rf"<{domain_tag}>\s*(.*?)\s*</{domain_tag}>", output, flags=re.S)
        problem_match = re.search(rf"<{problem_tag}>\s*(.*?)\s*</{problem_tag}>", output, flags=re.S)

        if domain_match and problem_match:
            domain = strip_code_block(domain_match.group(1))
            problem = strip_code_block(problem_match.group(1))
            return domain, problem

    data = json.loads(strip_code_block(output))
    if "domain" in data and "problem" in data:
        return str(data["domain"]).strip(), str(data["problem"]).strip()

    raise ValueError("cannot parse PDDL output")


def get_save_dir(task: dict) -> Path:
    return eval_root / task["dataset"] / task["episode"]


def generation_complete(task: dict) -> bool:
    save_dir = get_save_dir(task)
    return all(
        (save_dir / name).is_file() and (save_dir / name).read_text(encoding="utf-8").strip()
        for name in ("domain.pddl", "problem.pddl", "plan.txt")
    )


def load_cached_status(task: dict):
    save_dir = get_save_dir(task)
    judge_file = save_dir / "judge.json"
    error_file = save_dir / "error.log"

    if judge_file.exists():
        result = json.loads(judge_file.read_text(encoding="utf-8"))
        if result.get("gt_validation_version") != VALIDATOR_VERSION:
            return None, False  # Re-evaluate caches from older validation rules once.
        passed = result.get("pass") is True
        return "judge", passed

    if error_file.exists():
        return "error", False

    return None, False


# =========================
# 读取任务
# =========================
def load_tasks_one(dataset_name: str) -> list[dict]:
    instructions = json.loads((task_root / "instructions" / f"instructions_{dataset_name}.json").read_text(encoding="utf-8"))[dataset_name]
    steps = json.loads((task_root / "steps" / f"steps_{dataset_name}.json").read_text(encoding="utf-8"))[dataset_name]
    inputs = json.loads((task_root / "inputs" / f"inputs_{dataset_name}.json").read_text(encoding="utf-8"))[dataset_name]
    metadata = json.loads((task_root / "metadata" / f"metadata_{dataset_name}.json").read_text(encoding="utf-8"))
    if set(inputs) != set(instructions) or set(steps) != set(instructions):
        raise ValueError(f"Input/instruction/steps keys differ for {dataset_name}")
    settings = metadata["input_settings"]
    tasks = []
    for task_name in sorted(instructions, key=number):
        images = [task_root / relative for relative in inputs[task_name]]
        if not images or any(not image.is_file() for image in images):
            raise FileNotFoundError(f"Incomplete observations: {dataset_name}/{task_name}")
        expected_views = settings["views_per_task"] or metadata["tasks"][task_name]["image_count"]
        if len(images) != expected_views:
            raise ValueError(f"{dataset_name}/{task_name}: expected {expected_views} images, got {len(images)}")
        tasks.append({
            "dataset": dataset_name, "episode": task_name,
            "instruction": instructions[task_name], "images": images,
            "kf_actions": steps_to_text(steps[task_name]),
            "robot_configuration": settings["robot_configuration"],
            "gt_dir": task_root / "gt" / dataset_name / task_name,
            "source_context": metadata["tasks"][task_name].get("source_context", ""),
        })
    return tasks


def load_tasks() -> list[dict]:
    all_tasks = []

    for dataset_name in datasets:
        dataset_tasks = load_tasks_one(dataset_name)
        print(f"数据集: {dataset_name}")
        print(f"任务总数: {len(dataset_tasks)}")
        all_tasks.extend(dataset_tasks)

    return all_tasks


# =========================
# 单任务生成
# =========================
def generate_one(task: dict):
    instruction = task["instruction"]
    images = task["images"]

    save_dir = get_save_dir(task)
    domain_file = save_dir / "domain.pddl"
    problem_file = save_dir / "problem.pddl"
    plan_file = save_dir / "plan.txt"

    try:
        if generation_complete(task):
            return task, True, "cached"

        if not images or any(not image.is_file() for image in images):
            return task, False, "missing_image"

        save_dir.mkdir(parents=True, exist_ok=True)

        prompt = prompt_path.read_text(encoding="utf-8").format(
            instruction=instruction,
            robot_configuration=task["robot_configuration"],
        )
        if len(images) > 1:
            prompt = "All attached images are views of the same initial scene. Use every view jointly; an object appearing in multiple views is the same object.\n\n" + prompt
        if task.get("source_context"):
            prompt = "Official benchmark context:\n" + task["source_context"] + "\n\n" + prompt
        output = call_gpt(eval_model, prompt, images)

        domain, problem = parse_pddl_output(output)
        domain_file.write_text(domain, encoding="utf-8")
        problem_file.write_text(problem, encoding="utf-8")
        plan_file.unlink(missing_ok=True)
        validate_untyped_pddl(domain)
        validate_untyped_pddl(problem)
        domain_file.write_text(format_action_conditions(domain), encoding="utf-8")

        if not solve_pddl(domain_file, problem_file):
            return task, False, "pddl_unsolvable"
        if not plan_file.is_file() or not plan_file.read_text(encoding="utf-8").strip():
            return task, False, "missing_pddl_plan"
        return task, True, "pddl"

    except Exception as e:
        print(traceback.format_exc())
        return task, False, str(e)


# =========================
# 单任务 Judge
# =========================
def judge_one(task: dict):
    instruction = task["instruction"]
    images = task["images"]
    kf_actions = task["kf_actions"]

    save_dir = get_save_dir(task)
    plan_file = save_dir / "plan.txt"
    judge_file = save_dir / "judge.json"

    try:
        if judge_file.exists():
            result = json.loads(judge_file.read_text(encoding="utf-8"))
            if result.get("gt_validation_version") == VALIDATOR_VERSION:
                return task, True, result.get("pass") is True, "cached"

        if not images or any(not image.is_file() for image in images):
            return task, False, False, "missing_image"

        if not plan_file.is_file():
            return task, False, False, "missing_plan"

        candidate_plan = plan_file.read_text(encoding="utf-8").strip()
        if not candidate_plan:
            return task, False, False, "empty_plan"

        mapping_instruction = instruction
        if task.get("source_context"):
            mapping_instruction += "\nOfficial scene context:\n" + task["source_context"]
        verification = validate_gt_plan(
            save_dir, task["gt_dir"], instruction=mapping_instruction,
            images=images, model=judge_model,
        )
        (save_dir / "gt_validation.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
        if verification["status"] != "UNKNOWN":
            result = {
                "evaluation_method": ("vlm_mapping_gt_replay" if verification["mapping_source"] == "vlm"
                                      else "official_gt_replay"),
                "gt_validation_version": VALIDATOR_VERSION,
                "pass": verification["pass"], "reasoning": verification["reason"],
                "feedback": "" if verification["pass"] else verification["reason"],
            }
            judge_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return task, True, result["pass"], "done"

        result = judge_pddl(
            model=judge_model,
            first_img=images if len(images) > 1 else images[0],
            instruction=instruction,
            kf_actions=kf_actions,
            candidate_plan=candidate_plan,
            predicted_domain=save_dir / "domain.pddl",
            pddl_plan=save_dir / "plan.txt",
            **({"scene_context": task["source_context"]} if task.get("source_context") else {}),
        )

        result["evaluation_method"] = "llm_judge"
        result["gt_validation_version"] = VALIDATOR_VERSION
        result["program_validation_unavailable"] = verification["reason"]
        judge_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        passed = result.get("pass") is True
        return task, True, passed, "done"

    except Exception as e:
        return task, False, False, str(e)


def write_summary(all_tasks: list[dict]) -> Path:
    """Report solvability and task correctness, both over all tasks in each dataset."""
    grouped = defaultdict(list)
    for task in all_tasks:
        grouped[task["dataset"]].append(task)
    blocks = []
    for dataset in datasets:
        tasks = grouped[dataset]
        total = len(tasks)
        solved = sum(generation_complete(task) for task in tasks)
        correct = sum(load_cached_status(task)[1] for task in tasks)
        blocks.append("\n".join([
            "=" * 80, f"{dataset} ({total})", "",
            f"可解率：{100 * solved / total if total else 0:.1f}% ({solved}/{total})",
            f"正确率：{100 * correct / total if total else 0:.1f}% ({correct}/{total})",
        ]))
    eval_root.mkdir(parents=True, exist_ok=True)
    report_path = eval_root / f"summary_{'_'.join(datasets)}.log"
    report_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"报告已保存到: {report_path}")
    return report_path


# =========================
# 主流程
# =========================
def main():
    all_tasks = load_tasks()
    total_all = len(all_tasks)

    generation_tasks = []
    judge_tasks = []
    skipped = 0

    for task in all_tasks:
        cached_status, _ = load_cached_status(task)

        if cached_status in {"judge", "error"}:
            skipped += 1
            continue

        if generation_complete(task):
            judge_tasks.append(task)
        else:
            generation_tasks.append(task)

    generation_total = len(generation_tasks)

    print(f"测试集: {datasets}")
    print(f"任务总数: {total_all}")
    print(f"已有 Judge 或已记录错误: {skipped}")
    print(f"已有生成结果、待 Judge: {len(judge_tasks)}")
    print(f"待生成 PDDL 结果: {generation_total}")

    # =========================
    # 阶段一：所有测试集一起并发生成
    # =========================
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, task) for task in generation_tasks]

        for i, future in enumerate(as_completed(futures), 1):
            task, ok, info = future.result()

            dataset_name = task["dataset"]
            episode_name = task["episode"]
            print(f"\n[生成 {i}/{generation_total}] {dataset_name}/{episode_name}")

            if ok:
                judge_tasks.append(task)
                print("✅ 可解")
            else:
                if info == "pddl_unsolvable":
                    print("⚠️ PDDL 不可解")
                else:
                    print(f"❌ 生成失败: {info}")

    print("\n" + "=" * 80)
    print(f"开始 Judge 共 {len(judge_tasks)} 个任务")

    # =========================
    # 阶段二：先串行生成第一个 Judge，失败则立即退出
    # =========================
    if judge_tasks:
        first_task = judge_tasks[0]
        task, ok, passed, info = judge_one(first_task)
        dataset_name = task["dataset"]
        episode_name = task["episode"]

        print(f"\n[Judge 1/{len(judge_tasks)}] {dataset_name}/{episode_name}")

        if not ok:
            raise SystemExit(
                f"❌ Judge 生成失败: {info}\n"
                "PDDL 结果已保留；请在可访问 Judge API 的环境中重新运行本脚本。"
            )

        if passed:
            print("✅ Judge 通过")
        else:
            print("⚠️ Judge 未通过")

    # 首个 Judge 成功说明在线服务可用，再并发处理其余任务。
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(judge_one, task) for task in judge_tasks[1:]]

        for i, future in enumerate(as_completed(futures), 2):
            task, ok, passed, info = future.result()

            dataset_name = task["dataset"]
            episode_name = task["episode"]

            print(f"\n[Judge {i}/{len(judge_tasks)}] {dataset_name}/{episode_name}")

            if ok:
                if passed:
                    print("✅ Judge 通过")

                else:
                    print("⚠️ Judge 未通过")
            else:
                raise SystemExit(
                    f"❌ Judge 生成失败: "
                    f"{dataset_name}/{episode_name}: {info}\n"
                    "已生成的 PDDL 和 Judge 结果均已保留，重新运行将自动续传。"
                )

    write_summary(all_tasks)


if __name__ == "__main__":
    main()
