from typing import List
from pathlib import Path
from swm.utils.plan_learning import get_prompt_from_template

def construct_instruction_with_steps(original_instruction: str, steps: List[str]) -> str:
    usable = [s for s in steps if s.strip().lower() != "none"]
    lines = [f"{original_instruction} Steps:"]

    for i, s in enumerate(usable, 1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines).strip() + "\n"


def construct_prompt_with_feedback(
    instruction_with_steps: str,
    feedback: str,
    failed_domain: str = "",
    failed_problem: str = "",
    failed_plan: str = "",
    action_template: str = "",
) -> str:
    
    prompt_path = Path(__file__).parent.parent / "prompt_templates" / "pddl_generation.txt"
    prompt = get_prompt_from_template(prompt_path, instruction_with_steps=instruction_with_steps, action_template=action_template)
    
    if not feedback.strip():
        return prompt

    prompt_with_feedback_path = Path(__file__).parent.parent / "prompt_templates" / "pddl_generation_with_feedback.txt"
    prompt_with_feedback = get_prompt_from_template(
        prompt_with_feedback_path, 
        instruction_with_steps=instruction_with_steps, 
        failed_domain = failed_domain, 
        failed_problem = failed_problem, 
        failed_plan = failed_plan if failed_plan else "Unsolveable.",  
        feedback=feedback,
        action_template=action_template)
    return prompt_with_feedback