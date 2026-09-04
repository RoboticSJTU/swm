import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.keyframe.actions_extraction import extract_keyframe_actions
from swm.pddl.generation import RetryState, generate_pddl
from swm.pddl.judge import judge_pddl
from swm.prompts import construct_instruction_with_steps, read_prompt

ROOT_DIR = Path(__file__).resolve().parent.parent
STEP_SOURCE = "video"  # "video" or "steps_json"
TASK_DOMAIN = "human_aug"

PDDL_MODEL = "gpt-5.6-sol"
ACTION_EXTRACTION_MODEL = "gemini-3.7-flash"
JUDGE_MODEL = PDDL_MODEL

ROBOT_CONFIGURATION = "single-arm"  # "single-arm" or "dual-arm"
ACTION_TEMPLATE_MODE = "retrieved"  # "fixed" or "retrieved"
ACTION_TEMPLATE_DOMAIN = "human"
ACTION_TEMPLATE_MODEL = PDDL_MODEL

TASK_WORKERS = 30  # 主线程并发数
MAX_PLAN_ATTEMPTS = 3
PREPROCESS_WORKERS = 16  # 关键帧提取并发


def load_tasks() -> list[dict]:
    instructions_path = ROOT_DIR / "tasks" / "instructions" / f"instructions_{TASK_DOMAIN}.json"
    instructions = json.loads(instructions_path.read_text(encoding="utf-8"))

    if STEP_SOURCE == "steps_json":
        steps_path = ROOT_DIR / "tasks" / "steps" / f"steps_{TASK_DOMAIN}.json"
        all_steps = json.loads(steps_path.read_text(encoding="utf-8"))

    tasks = []
    for task_id, episodes in sorted(instructions.items()):
        for episode_id, instruction in sorted(episodes.items()):
            task = {
                "dataset": TASK_DOMAIN,
                "task_id": task_id,
                "episode_id": episode_id,
                "instruction": str(instruction).strip(),
                "save_dir": ROOT_DIR / "eval_results" / PDDL_MODEL / TASK_DOMAIN / task_id / episode_id,
            }

            if STEP_SOURCE == "video":
                task["video_path"] = ROOT_DIR / "dataset" / "videos" / TASK_DOMAIN / f"{episode_id}.mp4"
                task["frames_dir"] = ROOT_DIR / "dataset" / "frames" / TASK_DOMAIN / episode_id
                task["keyframe_dir"] = ROOT_DIR / "dataset" / "keyframes" / TASK_DOMAIN / task_id / episode_id
            else:
                task["image_path"] = ROOT_DIR / "tasks" / "images" / TASK_DOMAIN / task_id / f"{episode_id}.png"
                task["steps"] = [str(step).strip() for step in all_steps[task_id][episode_id]]

            tasks.append(task)

    return tasks


def temporal_gradient_radius(frame_count: int, dataset: str = "human") -> int:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    is_human = dataset == "human" or dataset.startswith("human_")
    base, step = (10, 10) if is_human else (20, 20)
    return min(90, base + step * max(0, (frame_count - 1) // 500))


def _keyframe_segments(keyframe_dir: Path) -> list[dict]:
    segments = []
    for segment_dir in sorted(keyframe_dir.glob("seg_*")):
        files = sorted(segment_dir.glob("*.png"), key=lambda path: int(path.stem))
        if files:
            segments.append(
                {
                    "segment": segment_dir.name,
                    "keyframe_indices": [int(path.stem) - 1 for path in files],
                    "keyframe_files": [str(path.resolve()) for path in files],
                }
            )
    return segments


def prepare_temporal_gradient_keyframes(task: dict) -> dict:
    from swm.keyframe.kf_extraction import (
        extract_frames,
        extract_temporal_gradient_keyframes,
    )

    dataset = task.get("dataset", TASK_DOMAIN)
    metadata_path = task.get(
        "keyframe_metadata_path", task["keyframe_dir"] / "metadata.json"
    )

    extract_frames(task["video_path"], task["frames_dir"])

    frame_count = len(list(task["frames_dir"].glob("*.png")))
    radius = temporal_gradient_radius(frame_count, dataset)

    cached_metadata = {}
    if metadata_path.is_file():
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    keyframes_exist = any(task["keyframe_dir"].glob("seg_*/*.png"))
    cache_matches = (
        cached_metadata.get("dataset") == dataset
        and cached_metadata.get("frame_count") == frame_count
        and cached_metadata.get("radius") == radius
    )

    if keyframes_exist and not cache_matches:
        for segment_dir in task["keyframe_dir"].glob("seg_*"):
            if segment_dir.is_dir():
                shutil.rmtree(segment_dir)
        (task["keyframe_dir"] / "energy_curve.png").unlink(missing_ok=True)
        keyframes_exist = False

    if not keyframes_exist:
        extract_temporal_gradient_keyframes(
            task["frames_dir"],
            task["keyframe_dir"],
            radius=radius,
            smooth_k=5,
            merge_pct=0.5,
            plot_energy=True,
        )

    segments = _keyframe_segments(task["keyframe_dir"])
    if not segments:
        raise ValueError(f"No keyframes found in {task['keyframe_dir']}")
    metadata = {
        "dataset": dataset,
        "task": task["task_id"],
        "episode": task["episode_id"],
        "frame_count": frame_count,
        "radius": radius,
        "keyframe_indices": sorted(
            {index for segment in segments for index in segment["keyframe_indices"]}
        ),
        "segments": segments,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def find_task_action_template(task_id: str) -> str:
    task_dir = ROOT_DIR / "eval_results" / ACTION_TEMPLATE_MODEL / ACTION_TEMPLATE_DOMAIN / task_id
    episode_dirs = [path for path in task_dir.glob("episode_*") if path.is_dir()]
    if not episode_dirs:
        return ""

    first_episode = min(episode_dirs, key=lambda path: int(path.name.rsplit("_", 1)[1]))
    passed_rounds = []
    for judge_path in first_episode.glob("round*/judge.json"):
        round_dir = judge_path.parent
        round_number = round_dir.name.removeprefix("round")
        if round_number.isdigit() and judge_path.is_file():
            judge_result = json.loads(judge_path.read_text(encoding="utf-8"))
            if judge_result["pass"]:
                passed_rounds.append((int(round_number), round_dir))

    if not passed_rounds:
        return ""

    domain_path = max(passed_rounds)[1] / "domain.pddl"
    if not domain_path.is_file():
        return ""

    lines = domain_path.read_text(encoding="utf-8").splitlines()
    first_action = next(
        (index for index, line in enumerate(lines) if "(:action" in line), None
    )
    if first_action is None:
        return ""

    action_start = first_action - 1
    while action_start >= 0 and (
        not lines[action_start].strip()
        or lines[action_start].lstrip().startswith(";")
    ):
        action_start -= 1
    action_section = "\n".join(lines[action_start + 1 : -1]).strip()
    return "\n\n".join(
        block.strip() for block in action_section.split("\n\n") if block.strip()
    )


def run_task(task: dict, action_template: str) -> tuple[bool, bool]:
    save_dir = task["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    kf_actions_path = save_dir / "kf_actions.txt"

    if STEP_SOURCE == "video":
        cached_actions = (
            kf_actions_path.read_text(encoding="utf-8")
            if kf_actions_path.is_file()
            else ""
        )
        if cached_actions.strip():
            steps = [
                re.sub(r"^(?:\[G\d+\]|\d+[.)])\s*", "", line.strip())
                for line in cached_actions.splitlines()
                if line.strip()
            ]
            images = sorted(
                task["keyframe_dir"].glob("seg_*/*.png"),
                key=lambda path: (path.parent.name, int(path.stem)),
            )
            if not images:
                raise ValueError(f"No keyframes found in {task['keyframe_dir']}")
            task_img = images[0]
        else:
            task_img, steps = extract_keyframe_actions(
                model_name=ACTION_EXTRACTION_MODEL,
                keyframe_dir=task["keyframe_dir"],
                instruction=task["instruction"],
                save_dir=save_dir,
            )
    else:
        task_img = task["image_path"]
        steps = task["steps"]
        kf_actions_path.write_text("\n".join(steps) + "\n", encoding="utf-8")

    instruction_with_steps = construct_instruction_with_steps(task["instruction"], steps)
    numbered_steps = "\n".join(
        f"{i}. {step.strip()}"
        for i, step in enumerate(steps, 1)
        if step.strip()
    )
    retry_state = RetryState()
    planning_success = False

    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        round_result = generate_pddl(
            generate_pddl_model_name=PDDL_MODEL,
            task_img=task_img,
            instruction_with_steps=instruction_with_steps,
            save_dir=save_dir,
            attempt=attempt,
            retry_state=retry_state,
            action_template=action_template,
            robot_configuration=ROBOT_CONFIGURATION,
        )

        retry_state.prev_domain = round_result["domain"]
        retry_state.prev_problem = round_result["problem"]

        if not round_result["ok"]:
            retry_state.solver_feedback = round_result["solver_feedback"]
            retry_state.judge_feedback = ""
            retry_state.prev_plan = ""
            continue

        planning_success = True
        retry_state.solver_feedback = ""
        judge_out = judge_pddl(
            model=JUDGE_MODEL,
            first_img=task_img,
            instruction=task["instruction"],
            kf_actions=numbered_steps,
            candidate_plan=round_result["plan"],
            predicted_domain=round_result["round_dir"] / "domain.pddl",
            predicted_problem=round_result["round_dir"] / "problem.pddl",
            pddl_plan=round_result["round_dir"] / "plan.txt",
        )
        (round_result["round_dir"] / "judge.json").write_text(
            json.dumps(judge_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if judge_out["pass"]:
            return True, True

        retry_state.judge_feedback = judge_out["feedback"]
        retry_state.prev_plan = round_result["plan"]

    return planning_success, False


def main() -> None:
    if STEP_SOURCE not in ("video", "steps_json"):
        raise ValueError('STEP_SOURCE must be "video" or "steps_json"')

    tasks = load_tasks()
    total_loaded = len(tasks)
    failed = 0
    tasks = [
        task
        for task in tasks
        if not any(
            json.loads(path.read_text(encoding="utf-8"))["pass"]
            for path in task["save_dir"].glob("round*/judge.json")
        )
    ]
    already_passed = total_loaded - len(tasks)

    task_ids = {task["task_id"] for task in tasks}
    if ACTION_TEMPLATE_MODE == "fixed":
        action_templates = dict.fromkeys(task_ids, read_prompt("ACTION_TEMPLATE.txt"))
    elif ACTION_TEMPLATE_MODE == "retrieved":
        action_templates = {task_id: find_task_action_template(task_id) for task_id in task_ids}
        missing = sorted(task_id for task_id, template in action_templates.items() if not template)
        if missing:
            raise RuntimeError(f"Missing retrieved action templates: {', '.join(missing)}")
    else:
        raise ValueError('ACTION_TEMPLATE_MODE must be "fixed" or "retrieved"')

    if STEP_SOURCE == "video":
        ready_tasks = []
        with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
            futures = {
                executor.submit(prepare_temporal_gradient_keyframes, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                    ready_tasks.append(task)
                except Exception as error:
                    failed += 1
                    print(f"[preprocess failed] {task['task_id']}/{task['episode_id']}: {error}")
        tasks = ready_tasks

    print(
        f"loaded samples: {total_loaded}, to run: {len(tasks)}, "
        f"already passed: {already_passed}"
    )

    passed = 0
    planning_succeeded = 0
    with ThreadPoolExecutor(max_workers=TASK_WORKERS) as executor:
        futures = {
            executor.submit(run_task, task, action_templates[task["task_id"]]): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            tag = f"{TASK_DOMAIN}/{task['task_id']}/{task['episode_id']}"
            try:
                planning_success, judge_pass = future.result()
                planning_succeeded += int(planning_success)
                passed += int(judge_pass)
                status = "judge pass" if judge_pass else "planning/judge failed"
            except Exception as error:
                failed += 1
                status = f"crashed: {error}"
            print(f"[{index}/{len(tasks)}] {tag} {status}")

    print(f"planning success: {planning_succeeded}/{len(tasks)}")
    print(f"judge pass: {passed}/{len(tasks)}")
    print(f"crashed: {failed}/{total_loaded}")


if __name__ == "__main__":
    main()
