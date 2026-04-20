from __future__ import annotations

import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from swm.utils.plan_learning import get_prompt_from_template
from swm.utils.apis import call_gpt
from swm.utils.pddl.judge import judge_pddl
from swm.utils.pddl.planer import solve_pddl
from swm.utils.pddl.translate import translate_pddl_plan


# =========================
# 基本配置
# =========================
root_dir = Path(__file__).resolve().parent.parent

model = "8B_tag_no_all_com"
judge_model = "gemini-3-flash-preview"
# swm unidomain
dataset = "swm"
max_workers = 100

# prompt_path = root_dir / "src" / "swm" / "prompt_templates" / "vlm_cot.txt"
prompt_path = root_dir / "src" / "swm" / "prompt_templates" / "training_input.txt"
instructions_path = root_dir / "tasks" / "instructions" / f"instructions_{dataset}.json"
steps_path = root_dir / "tasks" / "steps" / f"steps_{dataset}.json"
image_root = root_dir / "tasks" / "images" / dataset
eval_root = root_dir / "eval_results" / model


# =========================
# 小工具
# =========================
def number(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def strip_code_block(text: str) -> str:
    text = text.strip()
    m = re.match(r"^\s*```(?:\w+)?\s*\n?(.*?)\n?\s*```\s*$", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text


def parse_output(output: str) -> tuple[str, str, str]:
    """
    返回:
    mode: "json_plan" 或 "pddl"
    a:
      - json_plan 时为 reasoning
      - pddl 时为 domain
    b:
      - json_plan 时为 nl_plan
      - pddl 时为 problem
    """
    output = output.strip()

    # 先尝试 JSON plan_sequence 格式
    try:
        text = strip_code_block(output)
        data = json.loads(text)
        if isinstance(data, dict) and "plan_sequence" in data:
            reasoning = str(data["reasoning"]).strip() if "reasoning" in data else ""
            plan_sequence = data["plan_sequence"]
            if not isinstance(plan_sequence, list):
                raise ValueError("plan_sequence must be a list")

            steps = []
            for x in plan_sequence:
                step = str(x).strip()
                if step:
                    steps.append(step)

            if not steps:
                raise ValueError("plan_sequence is empty")

            nl_plan = "\n".join(steps)
            return "json_plan", reasoning, nl_plan
    except Exception:
        pass

    # 再尝试带标签的 PDDL 输出
    for domain_tag, problem_tag in [("domain", "problem"), ("domain_pddl", "problem_pddl")]:
        domain_match = re.search(rf"<{domain_tag}>\s*(.*?)\s*</{domain_tag}>", output, flags=re.S)
        problem_match = re.search(rf"<{problem_tag}>\s*(.*?)\s*</{problem_tag}>", output, flags=re.S)
        if domain_match and problem_match:
            domain = strip_code_block(domain_match.group(1))
            problem = strip_code_block(problem_match.group(1))
            return "pddl", domain, problem

    # 最后尝试 JSON 的 domain/problem
    try:
        text = strip_code_block(output)
        data = json.loads(text)
        if isinstance(data, dict) and "domain" in data and "problem" in data:
            domain = str(data["domain"]).strip()
            problem = str(data["problem"]).strip()
            return "pddl", domain, problem
    except Exception:
        pass

    raise ValueError("cannot parse model output")


# =========================
# 读取任务
# =========================

def load_tasks() -> list[dict]:
    instructions = json.loads(instructions_path.read_text(encoding="utf-8"))
    steps_data = json.loads(steps_path.read_text(encoding="utf-8"))

    tasks = []

    # 格式1:
    # {
    #   "task_1": {"episode_1": "...", "episode_2": "..."},
    #   "task_2": {"episode_1": "..."}
    # }
    #
    # 格式2:
    # {
    #   "unidomain": {
    #     "task_1": "...",
    #     "task_2": "..."
    #   }
    # }
    #
    # 这里统一展开成 task / episode / instruction / image / kf_plan

    if (
        isinstance(instructions, dict)
        and len(instructions) == 1
        and dataset in instructions
        and isinstance(instructions[dataset], dict)
    ):
        inner = instructions[dataset]
        all_values_are_string = True
        for v in inner.values():
            if isinstance(v, dict):
                all_values_are_string = False
                break

        if all_values_are_string:
            for episode_name in sorted(inner.keys(), key=number):
                instruction = inner[episode_name]
                task_name = dataset

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
                if task_name in steps_data and isinstance(steps_data[task_name], dict) and episode_name in steps_data[task_name]:
                    raw_steps = steps_data[task_name][episode_name]
                    if isinstance(raw_steps, list):
                        kf_plan = "\n".join(str(x) for x in raw_steps)
                    elif isinstance(raw_steps, dict):
                        lines = []
                        for k in sorted(raw_steps.keys(), key=number):
                            lines.append(str(raw_steps[k]))
                        kf_plan = "\n".join(lines)
                    elif raw_steps is not None:
                        kf_plan = str(raw_steps)

                tasks.append(
                    {
                        "task": task_name,
                        "episode": episode_name,
                        "instruction": instruction,
                        "image": image_path,
                        "kf_plan": kf_plan,
                    }
                )
            return tasks

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
            if task_name in steps_data and isinstance(steps_data[task_name], dict) and episode_name in steps_data[task_name]:
                raw_steps = steps_data[task_name][episode_name]
                if isinstance(raw_steps, list):
                    kf_plan = "\n".join(str(x) for x in raw_steps)
                elif isinstance(raw_steps, dict):
                    lines = []
                    for k in sorted(raw_steps.keys(), key=number):
                        lines.append(str(raw_steps[k]))
                    kf_plan = "\n".join(lines)
                elif raw_steps is not None:
                    kf_plan = str(raw_steps)

            tasks.append(
                {
                    "task": task_name,
                    "episode": episode_name,
                    "instruction": instruction,
                    "image": image_path,
                    "kf_plan": kf_plan,
                }
            )

    return tasks


# =========================
# 单任务生成
# =========================

def generate_one(task: dict):
    task_name = task["task"]
    episode_name = task["episode"]
    instruction = task["instruction"]
    image_path = task["image"]

    save_dir = eval_root / task_name / episode_name
    domain_file = save_dir / "domain.pddl"
    problem_file = save_dir / "problem.pddl"
    plan_file = save_dir / "plan.txt"
    nl_plan_file = save_dir / "plan_nl.txt"
    reasoning_file = save_dir / "reasoning.log"

    if nl_plan_file.exists() or (domain_file.exists() and problem_file.exists() and plan_file.exists()):
        return task, True, "cached"

    try:
        if image_path is None or not image_path.exists():
            return task, False, "missing_image"

        save_dir.mkdir(parents=True, exist_ok=True)

        prompt = get_prompt_from_template(prompt_path, instruction=instruction)
        output = call_gpt(model, prompt, [image_path])

        mode, a, b = parse_output(output)

        if mode == "json_plan":
            reasoning = a
            nl_plan = b
            reasoning_file.write_text(reasoning, encoding="utf-8")
            nl_plan_file.write_text(nl_plan, encoding="utf-8")
            return task, True, "json_plan"

        domain = a
        problem = b
        domain_file.write_text(domain, encoding="utf-8")
        problem_file.write_text(problem, encoding="utf-8")

        if not solve_pddl(domain_file, problem_file):
            return task, False, "pddl_unsolvable"

        # 过翻译模块
        # translated = translate_pddl_plan(domain_file, plan_file)
        # if isinstance(translated, str) and translated.strip():
        #     nl_plan_file.write_text(translated.strip(), encoding="utf-8")

        plan_text = plan_file.read_text(encoding="utf-8").strip()
        if plan_text:
            nl_plan_file.write_text(plan_text, encoding="utf-8")
    
        return task, True, "pddl"

    except Exception as e:
        save_dir.mkdir(parents=True, exist_ok=True)
        reasoning_file.write_text(
            f"=== Dataset ===\n{dataset}\n\n"
            f"=== Task ===\n{task_name} / {episode_name}\n\n"
            f"=== Instruction ===\n{instruction}\n\n"
            f"=== Error ===\n{e}\n",
            encoding="utf-8",
        )
        return task, False, f"exception: {e}"


# =========================
# 单任务 Judge
# =========================

def judge_one(task: dict):
    task_name = task["task"]
    episode_name = task["episode"]
    instruction = task["instruction"]
    image_path = task["image"]
    kf_plan = task["kf_plan"]

    save_dir = eval_root / task_name / episode_name
    nl_plan_file = save_dir / "plan_nl.txt"
    judge_file = save_dir / "judge.json"

    try:
        if judge_file.exists():
            data = json.loads(judge_file.read_text(encoding="utf-8"))
            passed = bool(data["pass"]) if isinstance(data, dict) and "pass" in data else False
            return task, True, passed, "cached"

        if image_path is None or not image_path.exists():
            return task, False, False, "missing_image"

        if not nl_plan_file.exists():
            return task, False, False, "missing_nl_plan"

        nl_plan = nl_plan_file.read_text(encoding="utf-8")
        result = judge_pddl(
            model=judge_model,
            first_img=image_path,
            instruction=instruction,
            kf_plan=kf_plan,
            nl_plan=nl_plan,
            n=1
        )

        judge_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        passed = bool(result["pass"]) if isinstance(result, dict) and "pass" in result else False
        return task, True, passed, "done"

    except Exception as e:
        return task, False, False, f"judge_exception: {e}"


# =========================
# 主流程
# =========================

def main():
    tasks = load_tasks()
    total_tasks = len(tasks)

    print(f"数据集: {dataset}")
    print(f"任务总数: {total_tasks}")

    solved_tasks = []
    failed_tasks = []

    generation_success = 0
    parse_or_exception_fail = 0
    pddl_unsolvable = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, task) for task in tasks]

        for i, future in enumerate(as_completed(futures), 1):
            task, ok, info = future.result()
            task_name = task["task"]
            episode_name = task["episode"]

            print(f"\n[{i}/{total_tasks}] {task_name} / {episode_name}")

            if ok:
                generation_success += 1
                solved_tasks.append(task)
                print("✅ 成功")
            else:
                failed_tasks.append((task_name, episode_name))
                if info == "pddl_unsolvable":
                    pddl_unsolvable += 1
                    print("⚠️ PDDL不可解")
                else:
                    parse_or_exception_fail += 1
                    print(f"❌ {info}")

    print("\n" + "=" * 80)
    print(f"开始进行 Judge 评测，共 {len(solved_tasks)} 个任务")

    judge_success = 0
    judge_pass = 0
    judge_fail = 0
    judge_error = 0
    judge_passed_tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(judge_one, task) for task in solved_tasks]

        for i, future in enumerate(as_completed(futures), 1):
            task, ok, passed, info = future.result()
            task_name = task["task"]
            episode_name = task["episode"].split("_")[1]

            print(f"\n[Judge {i}/{len(solved_tasks)}] {task_name} / {episode_name}")

            if ok:
                judge_success += 1
                if passed:
                    judge_pass += 1
                    judge_passed_tasks.append(episode_name)
                    print("✅ Judge通过")
                else:
                    judge_fail += 1
                    print("⚠️ Judge未通过")
            else:
                judge_error += 1
                print(f"❌ {info}")

    failed_tasks.sort(key=lambda x: (number(x[0]), number(x[1])))
    judge_passed_tasks.sort(key=int)

    generation_rate = 100 * generation_success / total_tasks if total_tasks else 0.0
    judge_rate = 100 * judge_pass / total_tasks if total_tasks else 0.0

    report_lines = [
        "=" * 80,
        "📊 评估完成统计报告",
        f"📦 数据集: {dataset}",
        f"📋 总任务数: {total_tasks}",
        f"✅ 第一阶段成功数: {generation_success}, 成功率 {generation_rate:.1f}%",
        f"❌ 第一阶段解析/异常失败数: {parse_or_exception_fail}",
        f"⚠️ 第一阶段PDDL不可解数: {pddl_unsolvable}",
        f"🧪 Judge执行成功数: {judge_success}",
        f"🧪 Judge通过数: {judge_pass}",
        f"🧪 Judge通过任务: {judge_passed_tasks}",
        f"🧪 Judge未通过数: {judge_fail}",
        f"🧪 Judge异常数: {judge_error}",
        f"🏁 最终Judge成功率: {judge_pass}/{total_tasks} = {judge_rate:.1f}%",
    ]

    if failed_tasks:
        report_lines.append(f"📋 第一阶段失败任务: {failed_tasks}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    eval_root.mkdir(parents=True, exist_ok=True)
    report_path = eval_root / f"summary_{dataset}.log"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    print(f"\n📝 报告已保存到: {report_path}")


if __name__ == "__main__":
    main()