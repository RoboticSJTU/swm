import base64
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from swm.utils.plan_learning import get_prompt_from_template
from swm.utils.apis import call_gpt
from openai import OpenAI
from swm.utils.pddl.judge import judge_pddl
from swm.utils.pddl.planer import solve_pddl
from swm.utils.pddl.translate import translate_pddl_plan

root_dir = Path(__file__).parent.parent
model = "gemini-3-flash-preview"
judge_model = "gemini-3-flash-preview"

# task = unidomain droid_100_eval   human_30
task = "swm_100" 
instructions_path = root_dir / "tasks" / "instructions" / f"instructions_{task}.json"
steps_path = root_dir / "tasks" / "steps" / f"steps_{task}.json"
eval_root = root_dir / "eval_results" / model
max_workers = 20

def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def process_task(task_info):
    task_domain, task_index, instruction, image_path, _ = task_info
    save_dir = eval_root / task_domain / task_index
    domain_file = save_dir / "domain.pddl"
    problem_file = save_dir / "problem.pddl"
    reasoning_file = save_dir / "reasoning.log"
    plan_file = save_dir / "plan.txt"

    if domain_file.exists():
        return task_info, "success", None

    try:
        if image_path is None or not image_path.exists():
            return task_info, "failed", "missing_image"

        save_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = Path("/home/xyx/下载/swm/src/swm/prompt_templates/training_input.txt")
        prompt = get_prompt_from_template(prompt_path, instruction=instruction)

        try:
            output = call_gpt(model, prompt, [image_path])
            data = json.loads(output)
            domain_content = data["domain"]
            problem_content = data["problem"]
        except Exception:
            reasoning_file.write_text(
                f"=== Task {task_domain}-{task_index} ===\n"
                f"Instruction: {instruction}\n\n"
                f"=== GPT Output ===\n{output}",
                encoding="utf-8",
            )
            return task_info, "failed", "json_parse_failed"

        domain_file.write_text(domain_content, encoding="utf-8")
        problem_file.write_text(problem_content, encoding="utf-8")

        if not solve_pddl(domain_file, problem_file):
            return task_info, "failed", "pddl_unsolvable"

        translate_pddl_plan(domain_file, plan_file)
        return task_info, "success", None

    except Exception as e:
        return task_info, "failed", f"exception: {e}"


def process_judge(task_info):
    task_domain, task_index, instruction, image_path, kf_plan = task_info
    save_dir = eval_root / task_domain / task_index
    nl_plan_path = save_dir / "plan_nl.txt"
    plan_path = save_dir / "plan.txt"
    judge_path = save_dir / "judge.json"

    try:
        if judge_path.exists():
            data = json.loads(judge_path.read_text(encoding="utf-8"))
            return task_info, "success", bool(data.get("pass", False))

        if not nl_plan_path.exists():
            return task_info, "failed", "missing_nl_plan"

        judge_result = judge_pddl(
            model=judge_model,
            first_img=image_path,
            instruction=instruction,
            kf_plan=kf_plan,
            nl_plan=nl_plan_path.read_text(encoding="utf-8"),
        )

        judge_path.write_text(json.dumps(judge_result, ensure_ascii=False, indent=2), encoding="utf-8")
        return task_info, "success", bool(judge_result.get("pass", False))

    except Exception as e:
        return task_info, "failed", f"judge_exception: {e}"

if __name__ == "__main__":
    instructions = json.loads(instructions_path.read_text(encoding="utf-8"))
    steps_data = json.loads(steps_path.read_text(encoding="utf-8"))

    tasks = []
    for task_domain, items in instructions.items():
        task_dir = root_dir / "tasks" / "images" / task_domain 
        for task_name, instruction in items.items():
            task_index = task_name.split("_")[1]
            image_path = next(
                (task_dir / f"task_{task_index}.{ext}" for ext in ("jpg", "png") if (task_dir / f"task_{task_index}.{ext}").exists()),
                None,
            )
            steps = steps_data[task_domain][task_name]
            kf_plan = "\n".join(steps) if isinstance(steps, list) else str(steps)
            tasks.append((task_domain, task_index, instruction, image_path, kf_plan))

    total_tasks = len(tasks)
    print(f"任务总数: {total_tasks}")

    success = 0
    json_parse_fail = 0
    pddl_unsolvable = 0
    other_fail = 0
    failed_tasks = []
    solved_tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            task_info, status, info = fut.result()
            task_domain, task_index = task_info[:2]
            print(f"\n[{i}/{total_tasks}] {task_domain}-{task_index}")

            if status == "success":
                success += 1
                solved_tasks.append(task_info)
                print("✅ 成功")
            else:
                failed_tasks.append((task_domain, task_index))
                if info == "json_parse_failed":
                    json_parse_fail += 1
                    print("❌ JSON解析失败")
                elif info == "pddl_unsolvable":
                    pddl_unsolvable += 1
                    print("⚠️ PDDL不可解")
                else:
                    other_fail += 1
                    print(f"❌ {info}")

    print("\n" + "=" * 80)
    print(f"开始进行 Judge 评测，共 {len(solved_tasks)} 个可解任务")

    judge_success = 0
    judge_pass = 0
    judge_fail = 0
    judge_error = 0
    judge_passed_task_indices = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_judge, t): t for t in solved_tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            task_info, status, info = fut.result()
            task_domain, task_index = task_info[:2]
            print(f"\n[Judge {i}/{len(solved_tasks)}] {task_domain}-{task_index}")

            if status == "success":
                judge_success += 1
                if info:
                    judge_pass += 1
                    judge_passed_task_indices.append(int(task_index))
                    print("✅ Judge通过")
                else:
                    judge_fail += 1
                    print("⚠️ Judge未通过")
            else:
                judge_error += 1
                print(f"❌ {info}")

    overall_judge_rate = judge_pass / total_tasks * 100 if total_tasks else 0
    pddl_rate = success / total_tasks * 100 if total_tasks else 0
    judge_passed_task_indices.sort()

    report_lines = [
        "=" * 80,
        "📊 评估完成统计报告",
        f"📋 总任务数: {total_tasks}",
        f"✅ PDDL可解数: {success}, 可解率 {pddl_rate:.1f}%",
        f"❌ JSON解析失败: {json_parse_fail}",
        f"⚠️ PDDL不可解: {pddl_unsolvable}",
        f"❌ 其他失败: {other_fail}",
        f"🧪 Judge执行成功数: {judge_success}",
        f"🧪 Judge通过数: {judge_pass}",
        f"🧪 Judge通过任务序号: {judge_passed_task_indices}",
        f"🧪 Judge未通过数: {judge_fail}",
        f"🧪 Judge异常数: {judge_error}",
        f"🏁 最终Judge成功率: {judge_pass}/{total_tasks} = {overall_judge_rate:.1f}%",
    ]

    if failed_tasks:
        report_lines.append(f"📋 第一阶段失败任务序号: {sorted(int(x[1]) for x in failed_tasks)}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_log_path = eval_root / next(iter(instructions.keys())) / "summary.log"
    report_log_path.parent.mkdir(parents=True, exist_ok=True)
    report_log_path.write_text(report_text + "\n", encoding="utf-8")
    print(f"\n📝 报告已保存到: {report_log_path}")