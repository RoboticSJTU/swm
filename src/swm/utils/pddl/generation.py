from pathlib import Path
import re
from swm.utils.pddl.planer import solve_pddl
import json
from swm.utils.construct_prompt import construct_prompt_with_feedback
from swm.utils.apis import call_gpt_json
from swm.utils.pddl.translate import translate_pddl_plan
from dataclasses import dataclass
@dataclass
class RetryState:
    solver_feedback: str = ""
    judge_feedback: str = ""
    prev_domain: str = ""
    prev_problem: str = ""
    prev_plan: str = ""
    
def generate_pddl(
    generate_pddl_model_name: str,
    task_img: Path,
    instruction_with_steps: str,
    save_dir: Path,
    attempt: int,
    retry_state: RetryState,
    action_template: str = ""
):
    """
    执行单个任务的一轮 PDDL 生成与规划。

    输入：
    - task_img: 当前任务对应的图像
    - instruction_with_steps: 已经拼接了步骤信息的详细任务描述
    - save_dir: 当前任务的保存目录
    - attempt: 当前是第几轮尝试
    - retry_state: 上一轮失败后留下来的反馈信息，用于本轮纠错生成

    输出：
    - 如果本轮失败，返回 ok=False, 并附带 solver_feedback
    - 如果本轮成功，返回 ok=True, 以及当前轮的 domain/problem/plan/nl_plan
    """
    # 结果单独保存在 round1 / round2 / ... 
    round_dir = save_dir / f"round{attempt}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # 当前轮的几个关键文件路径
    domain_path = round_dir / "domain.pddl"
    problem_path = round_dir / "problem.pddl"
    plan_path = round_dir / "plan.txt"
    nl_plan_path = round_dir / "plan_nl.txt"

    # 汇总上一轮失败反馈
    feedback_parts = []
    if retry_state.solver_feedback:
        feedback_parts.append("[Solver failure]\n" + retry_state.solver_feedback.strip())
    if retry_state.judge_feedback:
        feedback_parts.append("[Judge failure]\n" + retry_state.judge_feedback.strip())
    feedback = "\n".join(feedback_parts)
    
    prompt = construct_prompt_with_feedback(
        instruction_with_steps,
        feedback,
        failed_domain=retry_state.prev_domain,
        failed_problem=retry_state.prev_problem,
        failed_plan=retry_state.prev_plan,
        action_template=action_template,
    )

    data = call_gpt_json(generate_pddl_model_name, prompt, [task_img])
    domain_str = data["domain"]
    problem_str = data["problem"]
    domain_path.write_text(domain_str, encoding="utf-8")
    problem_path.write_text(problem_str, encoding="utf-8")
    
    ok = solve_pddl(domain_path, problem_path) or fix_pddl(domain_path, problem_path)
    if not ok:
        solver_feedback = (round_dir / "error.log").read_text(encoding="utf-8")
        return {"ok": False,"round_dir": round_dir,"domain": domain_str,"problem": problem_str,"plan": "","nl_plan": "","solver_feedback": solver_feedback}

    plan_text = plan_path.read_text(encoding="utf-8")
    translate_pddl_plan(domain_path, plan_path)
    nl_plan_text = nl_plan_path.read_text(encoding="utf-8")

    return {"ok": True,"round_dir": round_dir,"domain": domain_str,"problem": problem_str,"plan": plan_text,"nl_plan": nl_plan_text,"solver_feedback": ""}


def is_task_finished(save_dir: Path, max_attempts: int) -> bool:
    if not save_dir.exists():
        return False

    round_dirs = [
        d for d in save_dir.iterdir()
        if d.is_dir() and d.name.startswith("round") and d.name[5:].isdigit()
    ]
    if not round_dirs:
        return False

    max_round_dir = max(round_dirs, key=lambda d: int(d.name[5:]))
    max_round = int(max_round_dir.name[5:])
    judge_path = max_round_dir / "judge.json"

    # 1. 最大 round 的 judge.json 存在且 pass=True，则跳过
    if judge_path.exists():
        try:
            judge_data = json.loads(judge_path.read_text(encoding="utf-8"))
            if bool(judge_data.get("pass", False)):
                return True
        except Exception:
            pass

    # 2. 已达到最大尝试次数，也跳过
    if max_round >= max_attempts:
        return True

    return False


def fix_pddl(domain_path: Path, problem_path: Path) -> bool:
    err_path = domain_path.parent / "error.log"

    orig_problem = problem_path.read_text(encoding="utf-8")
    orig_domain = domain_path.read_text(encoding="utf-8")
    orig_err = err_path.read_text(encoding="utf-8") if err_path.exists() else None

    def restore():
        problem_path.write_text(orig_problem, encoding="utf-8")
        domain_path.write_text(orig_domain, encoding="utf-8")
        if orig_err is not None:
            err_path.write_text(orig_err, encoding="utf-8")
        elif err_path.exists():
            err_path.unlink()

    def try_solve(problem_text: str, domain_text: str) -> bool:
        problem_path.write_text(problem_text, encoding="utf-8")
        domain_path.write_text(domain_text, encoding="utf-8")
        return solve_pddl(domain_path, problem_path)

    # 读 domain 名
    m = re.search(
        r"\(define\s*\(\s*domain\s+([^\s\)]+)\s*\)",
        orig_domain,
        flags=re.IGNORECASE,
    )
    domain_name = m.group(1) if m else None

    problem = orig_problem

    # 必做 1：删除 (:objects ...) 里的类型标注
    m = re.search(r"(\(:objects\b)(.*?)(\))", problem, flags=re.IGNORECASE | re.DOTALL)
    if m:
        obj_block = re.sub(r"\s-\s*[^\s\)]+", "", m.group(2))
        problem = problem[:m.start(2)] + obj_block + problem[m.end(2):]

    # 必做 2：缺少 (:domain xxx) 就补上
    if domain_name and "(:domain" not in problem.lower():
        new_problem = re.sub(
            r"(\(define\s*\(\s*problem\s+[^\s\)]+\s*\))",
            rf"\1\n  (:domain {domain_name})",
            problem,
            count=1,
            flags=re.IGNORECASE,
        )
        if new_problem != problem:
            problem = new_problem
        else:
            pos = re.search(r"(?m)^\s*\(:", problem)
            if pos:
                problem = problem[:pos.start()] + f"  (:domain {domain_name})\n" + problem[pos.start():]

    # 先测“必做后”的版本
    if try_solve(problem, orig_domain):
        return True

    # 尝试 1：problem 末尾补 1 个 )
    if try_solve(problem.rstrip() + ")", orig_domain):
        return True

    # 尝试 2：problem 末尾补 2 个 ))
    if try_solve(problem.rstrip() + "))", orig_domain):
        return True

    # 尝试 3：domain 末尾补 1 个 )
    if try_solve(problem, orig_domain.rstrip() + ")"):
        return True

    restore()
    return False


def format_numbered_steps(steps):
    if isinstance(steps, str):
        steps = steps.splitlines()

    return "\n".join(
        f"{i}. {s.strip()}"
        for i, s in enumerate(steps, 1)
        if s and s.strip()
    )
    
def find_task_image(task_img_dir: Path, task_id: str) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        img_path = task_img_dir / f"{task_id}{ext}"
        if img_path.exists():
            return img_path
    raise FileNotFoundError(f"Image not found for {task_id} in {task_img_dir}")