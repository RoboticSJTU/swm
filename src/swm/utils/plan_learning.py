import re
from pathlib import Path

from swm.utils.apis import call_gpt_json


def get_prompt_from_template(path: Path, **values) -> str:
    return path.read_text(encoding="utf-8").format(**values)


def learn_steps_from_keyframes(
    model_name: str,
    keyframe_dir: Path,
    instruction: str,
    save_dir: Path,
    max_backtracks: int = 10,
) -> tuple[Path, list[str]]:
    segment_images = []
    for segment_dir in sorted(keyframe_dir.glob("seg_*")):
        images = sorted(segment_dir.glob("*.png"), key=lambda path: int(path.stem))
        if images:
            segment_images.append(images)

    if not segment_images:
        raise ValueError(f"No keyframes found in {keyframe_dir}")

    first_image = segment_images[0][0]
    plan_path = save_dir / "kf_plan.txt"
    group_plan_path = save_dir / "kf_plan_group.txt"
    debug_path = save_dir / "pair_debug.log"
    prompt_dir = Path(__file__).parent.parent / "prompt_templates"

    plan_lines = []
    history = []
    group_meta = []
    retry_hints = {}
    group_index = 0
    backtracks = 0

    while group_index < len(segment_images):
        images = segment_images[group_index]
        history_text = "\n".join(f"{i}. {action}" for i, action in enumerate(history, 1)) or "none"

        if group_index in retry_hints:
            prompt_path = prompt_dir / "plan_learning_with_feedback.txt"
            prompt = get_prompt_from_template(
                prompt_path,
                instruction=instruction,
                history=history_text,
                group_idx=group_index,
                num_groups=len(segment_images),
                error_action=retry_hints[group_index]["error_action"],
                feedback=retry_hints[group_index]["feedback"],
            )
        else:
            prompt_path = prompt_dir / "plan_learning.txt"
            prompt = get_prompt_from_template(
                prompt_path,
                instruction=instruction,
                history=history_text,
                group_idx=group_index,
                num_groups=len(segment_images),
            )

        data = call_gpt_json(model_name, prompt, images)
        debug_lines = [
            "=" * 24 + " PAIR " + "=" * 24,
            f"[G{group_index}]: {images[0].name}->{images[-1].name}",
            f"[history_actions]\n{history_text}",
            f"[history_reasoning] {data['history_reasoning']}",
            f"[history_check] {data['history_check']}",
            f"[history_feedback] {data['history_feedback']}",
            f"[action_reasoning] {data['action_reasoning']}",
            f"[action] {data['action']}",
            "",
        ]
        with debug_path.open("a", encoding="utf-8") as debug_file:
            debug_file.write("\n".join(debug_lines))

        if not data["history_check"]:
            backtracks += 1
            if backtracks > max_backtracks:
                raise RuntimeError(f"Too many backtracks: {data['history_feedback']}")

            rollback_index = int(data["earliest_bad_group"])
            bad_group = group_meta[rollback_index]
            retry_hints[rollback_index] = {
                "error_action": bad_group["raw_action"],
                "feedback": data["history_feedback"],
            }

            retry_hints = {
                index: hint
                for index, hint in retry_hints.items()
                if index <= rollback_index
            }
            del plan_lines[-sum(group["plan_count"] for group in group_meta[rollback_index:]):]
            history_count = sum(group["history_count"] for group in group_meta[rollback_index:])
            if history_count:
                del history[-history_count:]
            del group_meta[rollback_index:]

            plan_path.write_text(
                "\n".join(plan_lines) + ("\n" if plan_lines else ""),
                encoding="utf-8",
            )
            group_index = rollback_index
            continue

        raw_action = data["action"]
        action_text = "\n".join(str(action) for action in raw_action) if isinstance(raw_action, list) else str(raw_action)
        action_text = action_text.replace("\\n", "\n")
        actions = []
        for line in action_text.splitlines():
            line = line.strip()
            line = re.sub(r"^\s*[\-\*\u2022]\s+", "", line)
            line = re.sub(r"^\s*\(?\d+\)?[.)]\s+", "", line).strip()
            line = re.sub(r"\\([.,;:!?])", r"\1", line)
            if line and line.lower().strip(".") != "none":
                actions.append(line if line.endswith(".") else line + ".")
        if not actions:
            actions = ["none"]

        plan_lines.extend(actions)
        history_actions = [action for action in actions if action != "none"]
        history.extend(f"[G{group_index}] {action}" for action in history_actions)
        group_meta.append(
            {
                "raw_action": raw_action,
                "actions": actions,
                "plan_count": len(actions),
                "history_count": len(history_actions),
            }
        )
        if group_index in retry_hints:
            del retry_hints[group_index]

        plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
        group_index += 1

    cleaned_plan = []
    previous = ""
    for action in plan_lines:
        key = action.lower().strip().strip(".")
        if key == "none":
            continue
        if key != previous:
            cleaned_plan.append(action)
        previous = key
    plan_path.write_text(
        "\n".join(cleaned_plan) + ("\n" if cleaned_plan else ""),
        encoding="utf-8",
    )

    group_lines = []
    for index, group in enumerate(group_meta):
        previous = ""
        for action in group["actions"]:
            key = action.lower().strip().strip(".")
            if key == "none":
                continue
            if key != previous:
                group_lines.append(f"[G{index}] {action}")
            previous = key
    group_plan_path.write_text(
        "\n".join(group_lines) + ("\n" if group_lines else ""),
        encoding="utf-8",
    )

    return first_image, cleaned_plan
