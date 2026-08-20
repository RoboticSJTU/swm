from dataclasses import dataclass
from pathlib import Path

from swm.llm import call_gpt_json
from swm.pddl.planner import solve_pddl, summarize_solver_error
from swm.pddl.postprocess import PDDLPostprocessError, postprocess_pddl
from swm.pddl.translate import translate_pddl_plan
from swm.prompts import construct_prompt_with_feedback


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
    action_template: str = "",
    robot_configuration: str = "dual-arm",
):
    round_dir = save_dir / f"round{attempt}"
    round_dir.mkdir(parents=True, exist_ok=True)
    domain_path = round_dir / "domain.pddl"
    problem_path = round_dir / "problem.pddl"
    plan_path = round_dir / "plan.txt"
    nl_plan_path = round_dir / "plan_nl.txt"

    feedback_parts = []
    if retry_state.solver_feedback:
        feedback_parts.append(
            "[Solver failure]\n" + retry_state.solver_feedback.strip()
        )
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
        robot_configuration=robot_configuration,
    )

    data = call_gpt_json(generate_pddl_model_name, prompt, [task_img])
    domain_str = data["domain"]
    problem_str = data["problem"]

    for path in (
        plan_path,
        nl_plan_path,
        round_dir / "judge.json",
        round_dir / "error.log",
    ):
        path.unlink(missing_ok=True)

    try:
        domain_str, problem_str = postprocess_pddl(domain_str, problem_str)
    except PDDLPostprocessError as error:
        domain_path.write_text(domain_str, encoding="utf-8")
        problem_path.write_text(problem_str, encoding="utf-8")
        return {
            "ok": False,
            "round_dir": round_dir,
            "domain": domain_str,
            "problem": problem_str,
            "plan": "",
            "nl_plan": "",
            "solver_feedback": f"PDDL post-processing failed: {error}",
        }

    domain_path.write_text(domain_str, encoding="utf-8")
    problem_path.write_text(problem_str, encoding="utf-8")

    if not solve_pddl(domain_path, problem_path):
        error_log = (round_dir / "error.log").read_text(encoding="utf-8")
        solver_feedback = summarize_solver_error(error_log)
        return {
            "ok": False,
            "round_dir": round_dir,
            "domain": domain_str,
            "problem": problem_str,
            "plan": "",
            "nl_plan": "",
            "solver_feedback": solver_feedback,
        }

    plan_text = plan_path.read_text(encoding="utf-8")
    translate_pddl_plan(domain_path, plan_path)
    nl_plan_text = nl_plan_path.read_text(encoding="utf-8")

    return {
        "ok": True,
        "round_dir": round_dir,
        "domain": domain_str,
        "problem": problem_str,
        "plan": plan_text,
        "nl_plan": nl_plan_text,
        "solver_feedback": "",
    }
