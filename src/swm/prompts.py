from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompt_templates"


def read_prompt(file_name: str) -> str:
    return (PROMPT_DIR / file_name).read_text(encoding="utf-8").strip()


def construct_instruction_with_steps(
    original_instruction: str, steps: list[str]
) -> str:
    usable = [step for step in steps if step.strip().lower() != "none"]
    lines = [f"{original_instruction} Steps:"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(usable, 1))
    return "\n".join(lines).strip() + "\n"


def construct_prompt_with_feedback(
    instruction_with_steps: str,
    feedback: str,
    failed_domain: str = "",
    failed_problem: str = "",
    failed_plan: str = "",
    action_template: str = "",
    robot_configuration: str = "dual-arm",
) -> str:
    if robot_configuration not in {"single-arm", "dual-arm"}:
        raise ValueError(f"Unsupported robot configuration: {robot_configuration}")

    template = read_prompt(
        "pddl_generation_with_feedback.txt"
        if feedback.strip()
        else "pddl_generation.txt"
    )
    return template.format(
        instruction_with_steps=instruction_with_steps.strip(),
        robot_configuration=robot_configuration,
        failed_domain=failed_domain,
        failed_problem=failed_problem,
        failed_plan=failed_plan or "Unsolveable.",
        feedback=feedback,
        rules=read_prompt("RULES.txt"),
        action_template=action_template.strip(),
    )
