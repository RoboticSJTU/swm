import re
from functools import lru_cache
from pathlib import Path
from typing import List


PROMPT_DIR = Path(__file__).parent.parent / "prompt_templates"
SECTION_PATTERN = re.compile(
    r"^(?P<level>#{1,3}) (?P<kind>FILE|MODE|COMPONENT): "
    r"(?P<name>[a-z][a-z0-9_.-]*)$",
    re.MULTILINE,
)


@lru_cache(maxsize=1)
def load_prompt_components() -> dict:
    text = (PROMPT_DIR / "components.md").read_text(encoding="utf-8")
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches or text[:matches[0].start()].strip():
        raise ValueError("Invalid content before the first prompt file section")

    components = {}
    file_name = mode = None
    expected_levels = {"FILE": "#", "MODE": "##", "COMPONENT": "###"}
    for index, match in enumerate(matches):
        kind, name = match["kind"], match["name"]
        if match["level"] != expected_levels[kind]:
            raise ValueError(f"Invalid heading level for {kind}: {name}")
        if kind == "FILE":
            file_name, mode = name, None
            components.setdefault(file_name, {})
        elif kind == "MODE":
            if file_name is None:
                raise ValueError(f"MODE without FILE: {name}")
            mode = name
            components[file_name].setdefault(mode, {})
        else:
            if file_name is None or mode is None:
                raise ValueError(f"COMPONENT without FILE/MODE: {name}")
            target = components[file_name][mode]
            if name in target:
                raise ValueError(f"Duplicate component: {file_name}/{mode}/{name}")
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            target[name] = text[match.end():end].strip()
    return components


def get_prompt_component(file_name: str, mode: str, name: str) -> str:
    try:
        return load_prompt_components()[file_name][mode][name]
    except KeyError as error:
        raise ValueError(f"Missing prompt component: {file_name}/{mode}/{name}") from error


def construct_instruction_with_steps(original_instruction: str, steps: List[str]) -> str:
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

    components = load_prompt_components()["pddl_generation.txt"]
    common = components["common"]
    mode = "feedback" if feedback.strip() else "initial"
    selected = components[mode]
    failure_context = ""
    if mode == "feedback":
        failure_context = selected["failure_context"].format(
            failed_domain=failed_domain,
            failed_problem=failed_problem,
            failed_plan=failed_plan or "Unsolveable.",
            feedback=feedback,
        )

    template = (PROMPT_DIR / "pddl_generation.txt").read_text(encoding="utf-8")
    return template.format(
        task_description=selected["task_description"],
        instruction_with_steps=instruction_with_steps.strip(),
        robot_configuration=robot_configuration,
        arm_abstraction=common["arm_abstraction"],
        failure_context=failure_context,
        rules=common["rules"],
        action_template=action_template.strip(),
        reasoning_process=selected["reasoning_process"],
        reasoning_schema=selected["reasoning_schema"],
    )
