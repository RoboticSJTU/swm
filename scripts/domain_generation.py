import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.keyframe.actions_extraction import extract_keyframe_actions
from swm.pddl.generation import RetryState, generate_pddl
from swm.pddl.validate import (
    SymbolicTraceError,
    _validate_verdict,
    save_validation,
    validate_plan_feasibility,
    validate_plan_soundness,
    format_kf_actions,
)
from swm.pddl.reorder import ReorderModel
from swm.pddl.strips import format_action_conditions, validate_untyped_pddl
from swm.prompts import construct_instruction_with_steps

ROOT_DIR = Path(__file__).resolve().parent.parent
STEP_SOURCE = "steps_json"  # "video" or "steps_json"
TASK_DOMAIN = "human_aug"

# 写死了save_dir = "gpt-5.6-sol"
PDDL_MODEL = "deepseek/deepseek-flash"
ACTION_EXTRACTION_MODEL = PDDL_MODEL
JUDGE_MODEL = PDDL_MODEL

ROBOT_CONFIGURATION = "single_arm"  # "single_arm" or "dual_arm"
ACTION_TEMPLATE_MODE = "retrieved"  # None: 不传；"fixed": 固定模板；"retrieved": 参考 domain 的 action 片段
ACTION_TEMPLATE_MODEL = "gpt-5.6-sol"
ACTION_TEMPLATE_DOMAIN = "human"


MAX_PLAN_ATTEMPTS = 3
TASK_WORKERS = 100  # 主线程并发数
PREPROCESS_WORKERS = TASK_WORKERS  # 关键帧提取并发

FIXED_TEMPORAL_GRADIENT_RADIUS = {
    "bridgedata_v2": 4,
    "human": 21,
    "agibot": 78,
    "droid": 39,
}


def temporal_gradient_radius(frame_count: int, dataset: str = TASK_DOMAIN) -> int:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    try:
        return FIXED_TEMPORAL_GRADIENT_RADIUS[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported fixed-radius dataset: {dataset}") from error

def load_tasks() -> list[dict]:
    instructions_path = ROOT_DIR / "tasks" / "instructions" / f"instructions_{TASK_DOMAIN}.json"
    instructions = json.loads(instructions_path.read_text(encoding="utf-8"))

    if STEP_SOURCE == "video":
        video_dir = ROOT_DIR / "dataset" / "videos" / TASK_DOMAIN
        videos = {path.stem: path for path in video_dir.glob("*.mp4")}
    else:
        steps_path = ROOT_DIR / "tasks" / "steps" / f"steps_{TASK_DOMAIN}.json"
        all_steps = json.loads(steps_path.read_text(encoding="utf-8"))
    tasks = []
    for task_id, episodes in sorted(instructions.items()):
        for episode_id, instruction in sorted(episodes.items()):
            if STEP_SOURCE == "video" and episode_id not in videos:
                continue

            task = {
                "dataset": TASK_DOMAIN,
                "task_id": task_id,
                "episode_id": episode_id,
                "instruction": str(instruction).strip(),
                "save_dir": ROOT_DIR / "eval_results" / "gpt-5.6-sol" / TASK_DOMAIN / task_id / episode_id,
            }

            if STEP_SOURCE == "video":
                task["video_path"] = videos[episode_id]
                task["frames_dir"] = ROOT_DIR / "dataset" / "frames" / TASK_DOMAIN / episode_id
                task["keyframe_dir"] = ROOT_DIR / "dataset" / "keyframes" / TASK_DOMAIN / task_id / episode_id
            else:
                task["image_path"] = ROOT_DIR / "tasks" / "images" / TASK_DOMAIN / task_id / f"{episode_id}.png"
                task["steps"] = [str(step).strip() for step in all_steps[task_id][episode_id]]

            tasks.append(task)

    return tasks

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


def _round_has_verified_plan(round_dir: Path) -> bool:
    judge_path = round_dir / "judge.json"
    try:
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        return judge['pfv']['pass'] is True and judge['psv']['pass'] is True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _latest_verified_round(task: dict) -> Path | None:
    rounds = []
    for round_dir in task['save_dir'].glob("round*"):
        suffix = round_dir.name.removeprefix("round")
        if suffix.isdigit() and _round_has_verified_plan(round_dir):
            rounds.append((int(suffix), round_dir))
    return max(rounds, default=(0, None), key=lambda item: item[0])[1]


def _extract_action_template(domain_path: Path) -> str:
    domain_text = domain_path.read_text(encoding="utf-8")
    validate_untyped_pddl(domain_text)
    lines = format_action_conditions(domain_text).splitlines()
    blocks = []
    for start, line in enumerate(lines):
        if "(:action" not in line:
            continue
        depth = 0
        for end in range(start, len(lines)):
            depth += lines[end].count("(") - lines[end].count(")")
            if depth == 0:
                blocks.append("\n".join(lines[start : end + 1]).strip())
                break
    return "\n\n".join(block for block in blocks if block)


def find_task_action_template(task_id: str) -> str:
    task_dir = ROOT_DIR / "eval_results" / ACTION_TEMPLATE_MODEL / ACTION_TEMPLATE_DOMAIN / task_id
    episode_dirs = sorted(
        (path for path in task_dir.glob("episode_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )
    if not episode_dirs:
        return ""

    passed_rounds = []
    for round_dir in episode_dirs[0].glob("round*"):
        suffix = round_dir.name.removeprefix("round")
        domain_path = round_dir / "domain.pddl"
        judge_path = round_dir / "judge.json"
        if not suffix.isdigit() or not domain_path.is_file() or not judge_path.is_file():
            continue
        try:
            judge = json.loads(judge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pfv = judge.get("pfv")
        psv = judge.get("psv")
        passed = judge.get("pass") is True or (
            isinstance(pfv, dict)
            and isinstance(psv, dict)
            and pfv.get("pass") is True
            and psv.get("pass") is True
        )
        if passed:
            passed_rounds.append((int(suffix), domain_path))

    if not passed_rounds:
        return ""
    return _extract_action_template(max(passed_rounds, key=lambda item: item[0])[1])


def _action_template_for_task(task: dict) -> str:
    if ACTION_TEMPLATE_MODE is None:
        return ""
    if ACTION_TEMPLATE_MODE == "fixed":
        path = ROOT_DIR / "src" / "swm" / "prompt_templates" / "ACTION_TEMPLATE.txt"
        return path.read_text(encoding="utf-8")
    if ACTION_TEMPLATE_MODE == "retrieved":
        return find_task_action_template(task["task_id"])
    raise ValueError(
        'ACTION_TEMPLATE_MODE must be None, "fixed", or "retrieved"'
    )


def run_task(task: dict, action_template: str | None = None) -> tuple[bool, bool]:
    save_dir = task["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    kf_actions_path = save_dir / "kf_actions.txt"
    if action_template is None:
        action_template = _action_template_for_task(task)

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
    numbered_steps = format_kf_actions("\n".join(steps))
    retry_state = RetryState()
    planning_success = False
    rounds = [p for p in save_dir.glob('round*') if p.is_dir() and p.name[5:].isdigit()]
    latest = max(rounds, key=lambda p: int(p.name[5:]), default=None)
    next_round = int(latest.name[5:]) + 1 if latest else 1
    pending = latest if latest and all((latest / n).is_file() for n in (
        'domain.pddl', 'problem.pddl', 'plan.txt', 'judge.json')) else None
    attempts = 0
    while pending is not None or attempts < MAX_PLAN_ATTEMPTS:
        if pending is not None:
            round_result = {'ok': True, 'round_dir': pending,
                            'domain': (pending / 'domain.pddl').read_text(),
                            'problem': (pending / 'problem.pddl').read_text(),
                            'plan': (pending / 'plan.txt').read_text()}
            pending = None
            try:
                ReorderModel(*(round_result['round_dir'] / n for n in ('domain.pddl', 'problem.pddl', 'plan.txt')))
            except (ValueError, NotImplementedError):
                continue
        else:
            generate_kwargs = dict(
                generate_pddl_model_name=PDDL_MODEL,
                task_img=task_img,
                instruction_with_steps=instruction_with_steps,
                save_dir=save_dir,
                attempt=next_round,
                retry_state=retry_state,
                robot_configuration=ROBOT_CONFIGURATION,
            )
            if action_template:
                generate_kwargs["action_template"] = action_template
            round_result = generate_pddl(**generate_kwargs)
            attempts += 1
            next_round += 1

        retry_state.prev_domain = round_result["domain"]
        retry_state.prev_problem = round_result["problem"]

        if not round_result["ok"]:
            retry_state.solver_feedback = round_result["solver_feedback"]
            retry_state.judge_feedback = ""
            retry_state.prev_plan = ""
            continue

        planning_success = True
        retry_state.solver_feedback = ""
        directory = round_result['round_dir']
        saved = {}
        if (directory / 'judge.json').is_file():
            try:
                saved = json.loads((directory / 'judge.json').read_text())
                if not isinstance(saved, dict):
                    saved = {}
            except json.JSONDecodeError:
                pass
        pfv = None
        if saved:
            try:
                pfv = _validate_verdict(saved['pfv'])
            except (ValueError, KeyError):
                pass
        try:
            if pfv is None:
                save_validation(directory, None, None)
                pfv = validate_plan_feasibility(
                    model=JUDGE_MODEL,
                    first_img=task_img,
                    instruction=task['instruction'],
                    kf_actions=numbered_steps,
                    candidate_plan=round_result['plan'],
                    predicted_domain=directory / 'domain.pddl',
                    pddl_plan=directory / 'plan.txt',
                )
        except SymbolicTraceError as error:
            (directory / 'judge.json').unlink(missing_ok=True)
            retry_state.solver_feedback = str(error)
            retry_state.judge_feedback = ""
            retry_state.prev_plan = round_result["plan"]
            (round_result["round_dir"] / "error.log").write_text(str(error), encoding="utf-8")
            continue
        except (RuntimeError, ValueError) as error:
            save_validation(directory, None, None)
            (directory / 'error.log').write_text(str(error), encoding='utf-8')
            return planning_success, False

        result = save_validation(directory, pfv, None)
        psv = None
        if pfv['pass']:
            psv = validate_plan_soundness(JUDGE_MODEL, task_img, task['instruction'], numbered_steps, directory)
            result = save_validation(directory, pfv, psv)

        if result['pass'] is True:
            return True, True
        if result['pass'] is None:
            return planning_success, False

        retry_state.judge_feedback = result['feedback']
        retry_state.prev_plan = psv['counterexample']['plan'] if psv is not None else round_result['plan']

    return planning_success, False


def main() -> None:
    if STEP_SOURCE not in ("video", "steps_json"):
        raise ValueError('STEP_SOURCE must be "video" or "steps_json"')

    tasks = load_tasks()
    total_loaded = len(tasks)
    failed = 0
    verified = [
        (task, _latest_verified_round(task)) for task in tasks
    ]
    tasks = [task for task, round_dir in verified if round_dir is None]
    already_passed = total_loaded - len(tasks)

    if ACTION_TEMPLATE_MODE is None:
        action_templates = {}
    elif ACTION_TEMPLATE_MODE == "fixed":
        fixed_path = ROOT_DIR / "src" / "swm" / "prompt_templates" / "ACTION_TEMPLATE.txt"
        fixed_template = fixed_path.read_text(encoding="utf-8")
        action_templates = {task["task_id"]: fixed_template for task in tasks}
    elif ACTION_TEMPLATE_MODE == "retrieved":
        action_templates = {
            task["task_id"]: find_task_action_template(task["task_id"])
            for task in tasks
        }
        missing = sorted(task_id for task_id, template in action_templates.items() if not template)
        if missing:
            raise RuntimeError(f"Missing retrieved action templates: {', '.join(missing)}")
    else:
        raise ValueError(
            'ACTION_TEMPLATE_MODE must be None, "fixed", or "retrieved"'
        )

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
        if ACTION_TEMPLATE_MODE is None:
            futures = {executor.submit(run_task, task): task for task in tasks}
        else:
            futures = {
                executor.submit(run_task, task, action_templates[task["task_id"]]): task
                for task in tasks
            }
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            tag = f"{TASK_DOMAIN}/{task['task_id']}/{task['episode_id']}"
            try:
                planning_success, validation_pass = future.result()
                planning_succeeded += int(planning_success)
                passed += int(validation_pass)
                status = 'PFV + PSV passed' if validation_pass else 'planning/validation unsuccessful'
            except Exception as error:
                failed += 1
                status = f"crashed: {error}"
            print(f"[{index}/{len(tasks)}] {tag} {status}")

    print(f"planning success: {planning_succeeded}/{len(tasks)}")
    print(f"PFV + PSV passed: {passed}/{len(tasks)}")
    print(f"crashed: {failed}/{total_loaded}")


if __name__ == "__main__":
    main()
