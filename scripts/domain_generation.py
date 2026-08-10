import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.keyframe.extraction import extract_frames, extract_keyframes
from swm.utils.construct_prompt import construct_instruction_with_steps, get_prompt_component
from swm.utils.pddl.generation import RetryState, generate_pddl
from swm.utils.pddl.judge import judge_pddl
from swm.utils.plan_learning import learn_steps_from_keyframes

"""
video 输入 原始视频/已有关键帧/kf_plan.txt”，
prepared 输入 steps_json + 首帧图像”。
"""

# =========================
# Configuration
# =========================
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_MODE = "video"  # "video" or "prepared"
TASK_DOMAIN = "agibot"

PDDL_MODEL = "gpt-5.6-sol"
LEARN_STEPS_MODEL = PDDL_MODEL
JUDGE_MODEL = PDDL_MODEL

TASK_WORKERS = 30 # 主线程并发数
MAX_STEP_BACKTRACKS = 10
MAX_PLAN_ATTEMPTS = 3
PREPROCESS_WORKERS = 16 # 关键帧提取并发

ROBOT_CONFIGURATION = "dual-arm"  # "single-arm" or "dual-arm"
ACTION_TEMPLATE_MODE = "retrieved"  # "fixed" or "retrieved"
ACTION_TEMPLATE_DOMAIN = "agibot"
ACTION_TEMPLATE_MODEL = "gpt-5.6-sol"

def load_tasks(root_dir: Path, task_domain: str, input_mode: str) -> list[dict]:
    instructions_path = root_dir / "tasks" / "instructions" / f"instructions_{task_domain}.json"
    instructions = json.loads(instructions_path.read_text(encoding="utf-8"))

    if input_mode == "prepared":
        steps_path = root_dir / "tasks" / "steps" / f"steps_{task_domain}.json"
        all_steps = json.loads(steps_path.read_text(encoding="utf-8"))

    tasks = []
    for task_id, episodes in sorted(instructions.items()):
        for episode_id, instruction in sorted(episodes.items()):
            task = {
                "task_domain": task_domain,
                "task_id": task_id,
                "episode_id": episode_id,
                "instruction": str(instruction).strip(),
                "save_dir": root_dir / "eval_results" / PDDL_MODEL / task_domain / task_id / episode_id,
            }

            if input_mode == "video":
                task["video_path"] = root_dir / "dataset" / "videos" / task_domain / f"{episode_id}.mp4"
                task["frames_dir"] = root_dir / "dataset" / "frames" / task_domain / episode_id
                task["keyframe_dir"] = root_dir / "dataset" / "keyframes" / task_domain / task_id / episode_id
            else:
                task["image_path"] = root_dir / "tasks" / "images" / task_domain / task_id / f"{episode_id}.png"
                task["steps"] = [str(step).strip() for step in all_steps[task_id][episode_id]]

            tasks.append(task)

    return tasks


def prepare_video_task(task: dict) -> None:
    # Existing nested keyframes are the final cache and need no preprocessing.
    if list(task["keyframe_dir"].glob("seg_*/*.png")):
        return

    extract_frames(task["video_path"], task["frames_dir"])
    extract_keyframes(
        task["frames_dir"],
        task["keyframe_dir"],
        smooth_k=5,
        merge_pct=0.5,
        plot_energy=True,
    )


def find_task_action_template(
    root_dir: Path,
    model_name: str,
    task_domain: str,
    task_id: str,
) -> str:
    task_dir = root_dir / "eval_results" / model_name / task_domain / task_id
    episode_dirs = [path for path in task_dir.glob("episode_*") if path.is_dir()]
    if not episode_dirs:
        return ""

    first_episode = min(
        episode_dirs,
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )
    passed_rounds = []
    for round_dir in first_episode.glob("round*"):
        round_number = round_dir.name.removeprefix("round")
        judge_path = round_dir / "judge.json"
        if round_number.isdigit() and judge_path.is_file():
            judge_result = json.loads(judge_path.read_text(encoding="utf-8"))
            if judge_result["pass"]:
                passed_rounds.append(round_dir)

    if not passed_rounds:
        return ""

    passed_round = max(
        passed_rounds,
        key=lambda path: int(path.name.removeprefix("round")),
    )
    domain_path = passed_round / "domain.pddl"
    if not domain_path.is_file():
        return ""

    lines = domain_path.read_text(encoding="utf-8").splitlines()
    action_blocks = []
    line_index = 0
    while line_index < len(lines):
        if "(:action" not in lines[line_index]:
            line_index += 1
            continue

        action_start = line_index
        comment_index = line_index - 1
        while comment_index >= 0 and not lines[comment_index].strip():
            comment_index -= 1
        while comment_index >= 0 and lines[comment_index].lstrip().startswith(";"):
            action_start = comment_index
            comment_index -= 1

        depth = 0
        action_end = line_index
        while action_end < len(lines):
            depth += lines[action_end].count("(") - lines[action_end].count(")")
            if depth == 0:
                break
            action_end += 1

        action_blocks.append("\n".join(lines[action_start:action_end + 1]).strip())
        line_index = action_end + 1

    return "\n\n".join(action_blocks)


def run_task(task: dict, input_mode: str, action_template: str) -> tuple[bool, bool]:
    save_dir = task["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    kf_plan_path = save_dir / "kf_plan.txt"

    if input_mode == "video":
        first_segment = task["keyframe_dir"] / "seg_00"
        task_img = sorted(first_segment.glob("*.png"), key=lambda path: int(path.stem))[0]

        if kf_plan_path.is_file() and kf_plan_path.read_text(encoding="utf-8").strip():
            steps = [line.strip() for line in kf_plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            task_img, steps = learn_steps_from_keyframes(
                model_name=LEARN_STEPS_MODEL,
                keyframe_dir=task["keyframe_dir"],
                instruction=task["instruction"],
                save_dir=save_dir,
                max_backtracks=MAX_STEP_BACKTRACKS,
            )
    else:
        task_img = task["image_path"]
        steps = task["steps"]
        kf_plan_path.write_text("\n".join(steps) + "\n", encoding="utf-8")

    instruction_with_steps = construct_instruction_with_steps(task["instruction"], steps)
    retry_state = RetryState()
    planning_success = False
    judge_pass = False

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
        pddl_plan = round_result["plan"]
        nl_plan = round_result["nl_plan"]

        numbered_steps = "\n".join(
            f"{i}. {step.strip()}"
            for i, step in enumerate(steps, 1)
            if step.strip()
        )
        numbered_plan = "\n".join(
            f"{i}. {step.strip()}"
            for i, step in enumerate(nl_plan.splitlines(), 1)
            if step.strip()
        )
        judge_out = judge_pddl(
            JUDGE_MODEL,
            task_img,
            task["instruction"],
            numbered_steps,
            numbered_plan,
        )
        (round_result["round_dir"] / "judge.json").write_text(
            json.dumps(judge_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        judge_pass = judge_out["pass"]
        if judge_pass:
            break

        retry_state.judge_feedback = judge_out["feedback"]
        retry_state.prev_plan = pddl_plan

    return planning_success, judge_pass


def has_passed(save_dir: Path) -> bool:
    return any(
        json.loads(path.read_text(encoding="utf-8"))["pass"]
        for path in save_dir.glob("round*/judge.json")
    )


def main() -> None:
    if INPUT_MODE not in ("video", "prepared"):
        raise ValueError('INPUT_MODE must be "video" or "prepared"')

    tasks = load_tasks(ROOT_DIR, TASK_DOMAIN, INPUT_MODE)
    total_loaded = len(tasks)
    failed = 0

    task_ids = {task["task_id"] for task in tasks}
    if ACTION_TEMPLATE_MODE == "fixed":
        fixed_template = get_prompt_component(
            "pddl_generation.txt", "common", "action_template"
        )
        action_templates = dict.fromkeys(task_ids, fixed_template)
    elif ACTION_TEMPLATE_MODE == "retrieved":
        action_templates = {
            task_id: find_task_action_template(
                root_dir=ROOT_DIR,
                model_name=ACTION_TEMPLATE_MODEL,
                task_domain=ACTION_TEMPLATE_DOMAIN,
                task_id=task_id,
            )
            for task_id in task_ids
        }
        missing = sorted(task_id for task_id, template in action_templates.items() if not template)
        if missing:
            raise RuntimeError(f"Missing retrieved action templates: {', '.join(missing)}")
    else:
        raise ValueError('ACTION_TEMPLATE_MODE must be "fixed" or "retrieved"')

    if INPUT_MODE == "video":
        ready_tasks = []
        with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as executor:
            futures = {executor.submit(prepare_video_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                    ready_tasks.append(task)
                except Exception as error:
                    failed += 1
                    print(f"[preprocess failed] {task['task_id']}/{task['episode_id']}: {error}")
        tasks = ready_tasks

    total_ready = len(tasks)
    tasks = [task for task in tasks if not has_passed(task["save_dir"])]

    print(f"loaded samples: {total_loaded}, to run: {len(tasks)}, already passed: {total_ready - len(tasks)}")

    passed = 0
    planning_succeeded = 0
    with ThreadPoolExecutor(max_workers=TASK_WORKERS) as executor:
        futures = {
            executor.submit(
                run_task,
                task,
                INPUT_MODE,
                action_templates[task["task_id"]],
            ): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            tag = f"{task['task_domain']}/{task['task_id']}/{task['episode_id']}"
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
