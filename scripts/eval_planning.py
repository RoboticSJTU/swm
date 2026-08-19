from __future__ import annotations

import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import traceback
from swm.llm import call_gpt, call_gpt_json
from swm.pddl.judge import judge_pddl, latest_round_problem
from swm.pddl.planner import solve_pddl

# =========================
# 基本配置
# =========================
root_dir = Path(__file__).resolve().parent.parent

eval_model = "9B_3e"
judge_model = "gpt-5.6-sol"
translate_model = "qwen3.7-plus"  
ROBOT_CONFIGURATION = "single-arm"

# swm swm_2 unidomain
datasets = ["swm", "unidomain"]

N = 1
generation_max_workers = 200
judge_max_workers = generation_max_workers

prompt_path = root_dir / "src" / "swm" / "prompt_templates" / "training_input.txt"
# prompt_path = root_dir / "src" / "swm" / "prompt_templates" / "vlm_cot.txt"

if prompt_path.name == "training_input.txt":
    eval_mode = "pddl"
elif prompt_path.name == "vlm_cot.txt":
    eval_mode = "nl"
else:
    raise ValueError(f"未知 prompt 文件名，无法判断评测模式: {prompt_path.name}")

eval_root = root_dir / "eval_results" / eval_model


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


def has_pddl_actions(plan: str) -> bool:
    return any(
        line.strip() and not line.lstrip().startswith((";", "#"))
        for line in plan.splitlines()
    )


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


def parse_nl_output(output: str) -> tuple[str, str]:
    text = strip_code_block(output)

    try:
        data = json.loads(text)

        if "plan_sequence" in data:
            reasoning = str(data["reasoning"]).strip() if "reasoning" in data else ""
            plan_lines = []

            for step in data["plan_sequence"]:
                step = str(step).strip()
                if step:
                    plan_lines.append(step)

            plan = "\n".join(plan_lines).strip()
            if not plan:
                raise ValueError("empty plan_sequence")

            return reasoning, plan

    except Exception:
        pass

    if not text.strip():
        raise ValueError("empty natural-language plan")

    return "", text.strip()


def get_save_dir(task: dict) -> Path:
    dataset_name = task["dataset"]
    task_name = task["task"]
    episode_name = task["episode"]

    if task["flat"]:
        return eval_root / dataset_name / episode_name

    return eval_root / dataset_name / task_name / episode_name


def generation_complete(task: dict) -> bool:
    save_dir = get_save_dir(task)

    if eval_mode == "pddl":
        output_files = [
            save_dir / "domain.pddl",
            save_dir / "problem.pddl",
            save_dir / "plan.txt",
        ]
    else:
        output_files = [save_dir / "plan_nl.txt"]

    return all(
        path.is_file() and path.read_text(encoding="utf-8").strip()
        for path in output_files
    )


def new_stats() -> dict:
    return {
        "total": 0,
        "generation_success": 0,
        "failed_tasks": [],
        "judge_pass": 0,
        "judge_passed_tasks": [],
        "judge_fail": 0,
        "judge_failed_tasks": [],
    }


def load_cached_status(task: dict):
    save_dir = get_save_dir(task)
    judge_file = save_dir / "judge.json"
    error_file = save_dir / "error.log"

    if judge_file.exists():
        result = json.loads(judge_file.read_text(encoding="utf-8"))
        passed = bool(result["pass"]) if "pass" in result else False
        return "judge", passed

    if error_file.exists():
        return "error", False

    return None, False


# =========================
# PDDL plan 翻译为自然语言 plan
# =========================
def translate_pddl_plan_to_nl(save_dir: Path) -> None:
    domain_file = save_dir / "domain.pddl"
    problem_file = save_dir / "problem.pddl"
    plan_file = save_dir / "plan.txt"
    nl_plan_file = save_dir / "plan_nl.txt"

    domain = domain_file.read_text(encoding="utf-8").strip()
    problem = problem_file.read_text(encoding="utf-8").strip()
    plan = plan_file.read_text(encoding="utf-8").strip()

    if not has_pddl_actions(plan):
        nl_plan_file.write_text("", encoding="utf-8")
        return

    prompt = f"""
You are a robot task-planning expert.

Your task is to translate a PDDL plan into a clear natural-language action plan.

You are given:
1. The PDDL domain, which defines action meanings.
2. The PDDL problem, which defines objects, initial state, and goal.
3. The PDDL plan, which contains the executable action sequence.

Requirements:
- Translate each PDDL action in the plan into one concise natural-language step.
- Keep the same action order.
- Do not add new actions.
- Do not remove actions.
- Do not explain PDDL syntax.
- Do not include reasoning.
- Use object names from the problem as much as possible.
- Return JSON only.

Expected JSON format:
{{
  "plan_sequence": [
    "First natural-language action.",
    "Second natural-language action."
  ]
}}

[PDDL Domain]
{domain}

[PDDL Problem]
{problem}

[PDDL Plan]
{plan}
""".strip()

    result = call_gpt_json(translate_model, prompt, [])

    if isinstance(result, str):
        result = json.loads(strip_code_block(result))

    if isinstance(result, list):
        plan_lines = [str(x).strip() for x in result if str(x).strip()]
    else:
        plan_lines = [str(x).strip() for x in result["plan_sequence"] if str(x).strip()]

    if not plan_lines:
        raise ValueError("empty translated natural-language plan")

    nl_plan_file.write_text("\n".join(plan_lines), encoding="utf-8")


# =========================
# 读取任务
# =========================
def load_tasks_one(dataset_name: str) -> list[dict]:
    instructions_path = root_dir / "tasks" / "instructions" / f"instructions_{dataset_name}.json"
    steps_path = root_dir / "tasks" / "steps" / f"steps_{dataset_name}.json"
    image_root = root_dir / "tasks" / "images" / dataset_name

    instructions = json.loads(instructions_path.read_text(encoding="utf-8"))
    steps_data = json.loads(steps_path.read_text(encoding="utf-8"))

    if dataset_name in instructions and isinstance(instructions[dataset_name], dict):
        instructions = instructions[dataset_name]

    if dataset_name in steps_data and isinstance(steps_data[dataset_name], dict):
        steps_data = steps_data[dataset_name]

    tasks = []

    # flat:
    # {
    #   "episode_1": "...",
    #   "episode_2": "..."
    # }
    if all(isinstance(v, str) for v in instructions.values()):
        for episode_name in sorted(instructions.keys(), key=number):
            instruction = instructions[episode_name]

            image_path = None
            for ext in ("png", "jpg", "jpeg"):
                p = image_root / f"{episode_name}.{ext}"
                if p.exists():
                    image_path = p
                    break

            kf_plan = ""
            if episode_name in steps_data:
                kf_plan = steps_to_text(steps_data[episode_name])

            tasks.append(
                {
                    "dataset": dataset_name,
                    "task": dataset_name,
                    "episode": episode_name,
                    "instruction": instruction,
                    "image": image_path,
                    "kf_plan": kf_plan,
                    "flat": True,
                }
            )

        return tasks

    # nested:
    # {
    #   "task_1": {
    #     "episode_1": "..."
    #   }
    # }
    for task_name in sorted(instructions.keys(), key=number):
        episode_map = instructions[task_name]
        if not isinstance(episode_map, dict):
            continue

        for episode_name in sorted(episode_map.keys(), key=number):
            instruction = episode_map[episode_name]

            image_path = None
            for ext in ("png", "jpg", "jpeg"):
                p1 = image_root / task_name / f"{episode_name}.{ext}"
                p2 = image_root / f"{episode_name}.{ext}"

                if p1.exists():
                    image_path = p1
                    break
                if p2.exists():
                    image_path = p2
                    break

            kf_plan = ""
            if task_name in steps_data and isinstance(steps_data[task_name], dict):
                if episode_name in steps_data[task_name]:
                    kf_plan = steps_to_text(steps_data[task_name][episode_name])

            tasks.append(
                {
                    "dataset": dataset_name,
                    "task": task_name,
                    "episode": episode_name,
                    "instruction": instruction,
                    "image": image_path,
                    "kf_plan": kf_plan,
                    "flat": False,
                }
            )

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
    task_name = task["task"]
    episode_name = task["episode"]
    instruction = task["instruction"]
    image_path = task["image"]

    save_dir = get_save_dir(task)
    domain_file = save_dir / "domain.pddl"
    problem_file = save_dir / "problem.pddl"
    pddl_plan_file = save_dir / "plan.txt"
    nl_plan_file = save_dir / "plan_nl.txt"
    reasoning_file = save_dir / "reasoning.txt"

    try:
        if generation_complete(task):
            return task, True, "cached"

        if image_path is None or not image_path.exists():
            return task, False, "missing_image"

        save_dir.mkdir(parents=True, exist_ok=True)

        prompt = prompt_path.read_text(encoding="utf-8").format(
            instruction=instruction,
            robot_configuration=ROBOT_CONFIGURATION,
        )
        output = call_gpt(eval_model, prompt, [image_path])

        if eval_mode == "pddl":
            domain, problem = parse_pddl_output(output)

            domain_file.write_text(domain, encoding="utf-8")
            problem_file.write_text(problem, encoding="utf-8")

            if not solve_pddl(domain_file, problem_file):
                return task, False, "pddl_unsolvable"

            if not pddl_plan_file.exists() or not pddl_plan_file.read_text(encoding="utf-8").strip():
                return task, False, "missing_pddl_plan"

            return task, True, "pddl"

        reasoning, nl_plan = parse_nl_output(output)

        if reasoning:
            reasoning_file.write_text(reasoning, encoding="utf-8")

        nl_plan_file.write_text(nl_plan, encoding="utf-8")
        return task, True, "nl"

    except Exception as e:
        print(traceback.format_exc())
        return task, False, str(e)


# =========================
# 单任务 Judge
# =========================
def judge_one(task: dict):
    instruction = task["instruction"]
    image_path = task["image"]
    kf_plan = task["kf_plan"]

    save_dir = get_save_dir(task)
    plan_file = save_dir / "plan_nl.txt"
    judge_file = save_dir / "judge.json"

    try:
        if judge_file.exists():
            result = json.loads(judge_file.read_text(encoding="utf-8"))
            passed = bool(result["pass"]) if "pass" in result else False
            return task, True, passed, "cached"

        if image_path is None or not image_path.exists():
            return task, False, False, "missing_image"

        if not plan_file.exists():
            if eval_mode == "pddl":
                translate_pddl_plan_to_nl(save_dir)
            else:
                return task, False, False, "missing_plan_nl"

        pred_plan = plan_file.read_text(encoding="utf-8").strip()
        if not pred_plan:
            if eval_mode != "pddl":
                return task, False, False, "empty_plan_nl"

            pddl_plan = (save_dir / "plan.txt").read_text(encoding="utf-8")
            if has_pddl_actions(pddl_plan):
                return task, False, False, "empty_plan_nl"

        result = judge_pddl(
            model=judge_model,
            first_img=image_path,
            instruction=instruction,
            kf_plan=kf_plan,
            nl_plan=pred_plan,
            n=N,
            predicted_problem=(save_dir / "problem.pddl") if eval_mode == "pddl" else None,
            ground_truth_problem=latest_round_problem(
                root_dir / "eval_results" / "gt" / task["dataset"] / task["episode"]
            ),
        )

        judge_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        passed = bool(result["pass"]) if "pass" in result else False
        return task, True, passed, "done"

    except Exception as e:
        return task, False, False, str(e)


# =========================
# 主流程
# =========================
def main():
    all_tasks = load_tasks()
    total_all = len(all_tasks)

    stats = defaultdict(new_stats)
    generation_tasks = []
    judge_tasks = []
    skipped = 0

    for task in all_tasks:
        dataset_name = task["dataset"]
        episode_name = task["episode"]
        stats[dataset_name]["total"] += 1

        cached_status, passed = load_cached_status(task)

        if cached_status == "judge":
            skipped += 1
            stats[dataset_name]["generation_success"] += 1

            if passed:
                stats[dataset_name]["judge_pass"] += 1
                stats[dataset_name]["judge_passed_tasks"].append(str(number(episode_name)))
            else:
                stats[dataset_name]["judge_fail"] += 1
                stats[dataset_name]["judge_failed_tasks"].append(str(number(episode_name)))

            continue

        if cached_status == "error":
            skipped += 1
            stats[dataset_name]["failed_tasks"].append(str(number(episode_name)))
            continue

        if generation_complete(task):
            stats[dataset_name]["generation_success"] += 1
            judge_tasks.append(task)
        else:
            generation_tasks.append(task)

    generation_total = len(generation_tasks)

    print(f"测试集: {datasets}")
    print(f"任务总数: {total_all}")
    print(f"已有 Judge 或已记录错误: {skipped}")
    print(f"已有生成结果、待 Judge: {len(judge_tasks)}")
    print(f"待生成 {eval_mode.upper()} 结果: {generation_total}")

    # =========================
    # 阶段一：所有测试集一起并发生成
    # =========================
    with ThreadPoolExecutor(max_workers=generation_max_workers) as executor:
        futures = [executor.submit(generate_one, task) for task in generation_tasks]

        for i, future in enumerate(as_completed(futures), 1):
            task, ok, info = future.result()

            dataset_name = task["dataset"]
            task_name = task["task"]
            episode_name = task["episode"]
            # 注意，dataset_name一定等于task_name，所以可以简化相关代码。
            print(f"\n[生成 {i}/{generation_total}] {dataset_name}/{episode_name}")

            if ok:
                stats[dataset_name]["generation_success"] += 1
                judge_tasks.append(task)
                print("✅ 可解")
            else:
                stats[dataset_name]["failed_tasks"].append(str(number(episode_name)))

                if info == "pddl_unsolvable":
                    print("⚠️ PDDL 不可解")
                else:
                    print(f"❌ 生成失败: {info}")

    print("\n" + "=" * 80)
    print(f"阶段一完成，开始 Judge 共 {len(judge_tasks)} 个任务")

    # =========================
    # 阶段二：先串行生成第一个 Judge，失败则立即退出
    # =========================
    if judge_tasks:
        first_task = judge_tasks[0]
        task, ok, passed, info = judge_one(first_task)
        dataset_name = task["dataset"]
        task_name = task["task"]
        episode_name = task["episode"]

        print(f"\n[Judge 1/{len(judge_tasks)}] {dataset_name}/{task_name}/{episode_name}")

        if not ok:
            raise SystemExit(
                f"❌ Judge 生成失败: {info}\n"
                "PDDL 结果已保留；请在可访问 Judge API 的环境中重新运行本脚本。"
            )

        if passed:
            stats[dataset_name]["judge_pass"] += 1
            stats[dataset_name]["judge_passed_tasks"].append(str(number(episode_name)))
            print("✅ Judge 通过")
        else:
            stats[dataset_name]["judge_fail"] += 1
            stats[dataset_name]["judge_failed_tasks"].append(str(number(episode_name)))
            print("⚠️ Judge 未通过")

    # 首个 Judge 成功说明在线服务可用，再并发处理其余任务。
    with ThreadPoolExecutor(max_workers=judge_max_workers) as executor:
        futures = [executor.submit(judge_one, task) for task in judge_tasks[1:]]

        for i, future in enumerate(as_completed(futures), 2):
            task, ok, passed, info = future.result()

            dataset_name = task["dataset"]
            task_name = task["task"]
            episode_name = task["episode"]

            print(f"\n[Judge {i}/{len(judge_tasks)}] {dataset_name}/{task_name}/{episode_name}")

            if ok:
                if passed:
                    stats[dataset_name]["judge_pass"] += 1
                    stats[dataset_name]["judge_passed_tasks"].append(str(number(episode_name)))
                    print("✅ Judge 通过")

                else:
                    stats[dataset_name]["judge_fail"] += 1
                    stats[dataset_name]["judge_failed_tasks"].append(str(number(episode_name)))
                    print("⚠️ Judge 未通过")
            else:
                raise SystemExit(
                    f"❌ Judge 生成失败: "
                    f"{dataset_name}/{task_name}/{episode_name}: {info}\n"
                    "已生成的 PDDL 和 Judge 结果均已保留，重新运行将自动续传。"
                )

    # =========================
    # 总结报告
    # 保持原始 log 风格，只是每个 dataset 单独写一块
    # =========================
    report_lines = []

    for dataset_name in datasets:
        s = stats[dataset_name]

        s["failed_tasks"].sort(key=int)
        s["judge_passed_tasks"].sort(key=int)
        s["judge_failed_tasks"].sort(key=int)

        dataset_total = s["total"]
        generation_success = s["generation_success"]
        judge_pass = s["judge_pass"]
        
        failed_all_tasks = sorted(
            set(s["failed_tasks"] + s["judge_failed_tasks"]),
            key=int,
        )
        failed_all = len(failed_all_tasks)

        generation_rate = 100 * generation_success / dataset_total if dataset_total else 0.0
        judge_rate = 100 * judge_pass / dataset_total if dataset_total else 0.0
        failed_rate = 100 * failed_all / dataset_total if dataset_total else 0.0

        block = [
            "=" * 80,
            f"数据集: {dataset_name}, 总任务数: {dataset_total}",
            f"可解: {generation_success}, 率: {generation_rate:.1f}%",
            f"通过: {judge_pass}, 率: {judge_pass}/{dataset_total} = {judge_rate:.1f}%, [{', '.join(s['judge_passed_tasks'])}]",
            f"未通过: [{', '.join(failed_all_tasks)}]",
        ]

        report_lines.extend(block)
        report_lines.append("")

    report_text = "\n".join(report_lines).rstrip()

    eval_root.mkdir(parents=True, exist_ok=True)

    dataset_tag = "_".join(datasets)
    report_path = eval_root / f"summary_{dataset_tag}.log"
    report_path.write_text(report_text + "\n", encoding="utf-8")

    print("\n" + report_text)
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
