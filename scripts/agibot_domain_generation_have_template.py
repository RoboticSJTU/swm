import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from swm.utils.plan_learning import learn_steps_from_keyframes, get_first_keyframe_image
from swm.utils.construct_prompt import construct_instruction_with_steps
from swm.utils.pddl.judge import judge_pddl
from swm.utils.pddl.generation import is_task_finished, find_task_image, format_numbered_steps, RetryState, generate_pddl
import traceback
from swm.keyframe.extraction import extract_frames_from_video, extract_keyframes_from_frames

def load_tasks(root_dir: Path, task_domain: str) -> List[Dict[str, Any]]:
    """
    加载一个 task_domain 下的所有任务，并为每个任务整理统一输入来源。

    每个 task 包含：
    - task_domain: 任务所属 domain
    - task_id: 任务 id
    - instruction: tasks/instructions 中的任务指令
    - keyframe_dir: dataset/keyframes 下该任务的关键帧目录
    - image_path: tasks/images 下该任务的场景图，没有则为 None
    - steps: tasks/steps 中该任务的步骤，没有则为空列表
    """

    tasks_path = root_dir / "tasks" / "instructions" / f"instructions_{task_domain}.json"
    steps_path = root_dir / "tasks" / "steps" / f"steps_{task_domain}.json"
    task_img_dir = root_dir / "tasks" / "images" / task_domain

    videos_root = root_dir / "dataset" / "videos" / task_domain
    frames_root = root_dir / "dataset" / "frames" / task_domain
    keyframes_root = root_dir / "dataset" / "keyframes" / task_domain

    # 关键帧提取
    # try:
    #     extract_frames_from_video(videos_root, frames_root, max_workers=16)
    #     extract_keyframes_from_frames(frames_root, keyframes_root, smooth_k=5, merge_pct=0.5, max_workers=16, plot_energy=True)
    # except Exception as e:
    #     print(f"[Warn] keyframe extraction failed for domain {task_domain}: {e}")

    instructions_all_task = json.loads(tasks_path.read_text(encoding="utf-8"))

    if steps_path.is_file():
        steps_all_task = json.loads(steps_path.read_text(encoding="utf-8"))
    else:
        steps_all_task = {}

    tasks = []
    for task_id, ep2instruction in instructions_all_task.items():
        episode_items = sorted(ep2instruction.items())
        
        # # 只跑一个
        # episode_items = episode_items[:1]

        task_steps = steps_all_task.get(task_id, {})

        for episode_id, instruction in episode_items:
            instruction = str(instruction).strip()
            if not instruction:
                continue
            
            # 找首帧图
            img = None
            for name in [episode_id, episode_id.replace("episode_", "", 1)] if episode_id.startswith("episode_") else [episode_id]:
                for ext in ("png", "jpg", "jpeg", "webp"):
                    p = task_img_dir / task_id / f"{name}.{ext}"
                    if p.is_file():
                        img = p
                        break
                if img is not None:
                    break

            json_steps = task_steps.get(episode_id, [])
            steps = [str(step).strip() for step in json_steps if str(step).strip()]

            keyframe_dir = keyframes_root / task_id / episode_id
            if not keyframe_dir.is_dir() and episode_id.startswith("episode_"):
                alt = keyframes_root / task_id / episode_id.replace("episode_", "")
                if alt.is_dir():
                    keyframe_dir = alt

            tasks.append({
                "task_domain": task_domain,
                "task_id": task_id,
                "episode_id": episode_id,
                "instruction": instruction,
                "keyframe_dir": str(keyframe_dir),
                "image_path": str(img) if img is not None else None,
                "steps": steps,
            })

    return tasks


def get_task_image_and_steps(task: dict, save_dir: Path):
    """
    为单个任务准备后续规划所需的两个核心输入：
    1. task_img
    2. steps

    统一规则：
    - 读取steps 优先级：kf_plan.txt > keyframes > steps.json
    - 读取task_img 优先级：first keyframe image > task image in tasks/images
    """
    kf_plan_path = save_dir / "kf_plan.txt"
    keyframe_dir = Path(task["keyframe_dir"])
    image_path = Path(task["image_path"]) if task.get("image_path") else None

    # 1. 优先复用已有 kf_plan.txt
    if kf_plan_path.is_file() and kf_plan_path.read_text(encoding="utf-8").strip():
        steps = [line.strip() for line in kf_plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if keyframe_dir.is_dir():
            task_img = get_first_keyframe_image(keyframe_dir)
        elif image_path is not None and image_path.is_file():
            task_img = image_path
        else:
            raise ValueError(f"kf_plan exists but no task image available: {task['task_domain']}/{task['task_id']}")
        return task_img, steps

    # 2. 没有 kf_plan.txt，则走 keyframes
    if keyframe_dir.is_dir():
        task_img, steps = learn_steps_from_keyframes(
            model_name=Learn_steps_MODEL,
            keyframe_dir=keyframe_dir,
            instruction=task["instruction"],
            save_dir=save_dir,
            max_backtracks=MAX_STEP_BACKTRACKS,
        )
        return task_img, steps
    
    # 3. 最后退回 steps.json
    steps = task["steps"]
    kf_plan_path.write_text("\n".join(steps) + ("\n" if steps else ""), encoding="utf-8")
    return image_path, steps


def run_all_tasks(root_dir: Path, tasks: List[Dict[str, Any]]):
    total_loaded = len(tasks)

    # 过滤已经完成的任务
    tasks = [
        task for task in tasks
        if not is_task_finished(
            save_dir=root_dir / "eval_results" / PDDL_MODEL / task["task_domain"] / task["task_id"] / task["episode_id"],
            max_attempts=MAX_PLAN_ATTEMPTS,
        )
    ]

    total = len(tasks)
    print(f"loaded: {total_loaded}, to run: {total}, skipped: {total_loaded - total}")

    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_task = {
            ex.submit(run_single_task, task, root_dir): task
            for task in tasks
        }

        for done_i, future in enumerate(as_completed(future_to_task), 1):
            task = future_to_task[future]
            tag = f"{task['task_domain']}/{task['task_id']}/{task['episode_id']}"

            try:
                result = future.result()
            except Exception:
                print(f"[{done_i}/{total}] ❌ {tag} crashed:")
                print(traceback.format_exc())
                result = {
                    "task_domain": task["task_domain"],
                    "task_id": task["task_id"],
                    "completed": False,
                    "planning_success": False,
                    "judge_pass": False,
                    "sharegpt_sample": None,
                }

            all_results.append(result)

            if result["completed"] and result["judge_pass"]:
                msg = "✅ judge pass"
            elif result["completed"]:
                msg = "⚠️ finished (judge/planning fail)"
            else:
                msg = "❌ failed"

            print(f"[{done_i}/{total}] {tag} {msg}")

    finished = sum(r["completed"] for r in all_results)
    passed = sum(r["completed"] and r["judge_pass"] for r in all_results)
    print(f"finished: {finished}/{len(all_results)}")
    print(f"judge pass: {passed}/{len(all_results)}")


def run_single_task(task: dict, root_dir: Path):
    """
    跑单个任务：
    1. 创建当前任务的保存目录
    2. 获取任务图像和动作步骤
    3. 构造带步骤的详细任务描述
    4. 最多尝试 MAX_PLAN_ATTEMPTS 轮 PDDL 生成
    5. 每轮规划成功后，再调用 judge 检查 plan 是否符合任务意图
    6. 若 judge 通过，则生成 sharegpt 样本
    """

    task_domain = task["task_domain"]
    task_id = task["task_id"]
    episode_id = task["episode_id"]
    instruction = task["instruction"]

    # 当前任务的总保存目录：
    save_dir = root_dir / "eval_results" / PDDL_MODEL / task_domain / task_id / episode_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    task_img, steps = get_task_image_and_steps(task, save_dir)

    # 固定传入human里面的示例！！！！！
    action_template = find_task_action_template(
        root_dir=root_dir,
        model_name=PDDL_MODEL,
        task_domain="human",
        task_id=task_id,
    )

    # pddl生成
    instruction_with_steps = construct_instruction_with_steps(instruction, steps)
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
        )

        retry_state.prev_domain = round_result["domain"]
        retry_state.prev_problem = round_result["problem"]

        # 求解失败
        if not round_result["ok"]:
            retry_state.solver_feedback = round_result["solver_feedback"]
            retry_state.judge_feedback = ""
            retry_state.prev_plan = ""
            continue

        # 求解成功
        planning_success = True
        retry_state.solver_feedback = ""
        pddl_plan = round_result["plan"]
        nl_plan = round_result["nl_plan"]
        round_dir = round_result["round_dir"]
        judge_path = round_dir / "judge.json"
        
        judge_out = judge_pddl(
            JUDGE_MODEL,
            task_img,
            instruction,
            format_numbered_steps(steps),
            format_numbered_steps(nl_plan),
        )
        
        judge_path.write_text(json.dumps(judge_out, ensure_ascii=False, indent=2),encoding="utf-8")
        judge_pass = judge_out["pass"]
        if judge_pass:
            break

        # judge 不通过, 把 judge 给出的反馈和当前 plan 保存下来，供下一轮纠错生成
        retry_state.judge_feedback = judge_out["feedback"]
        retry_state.prev_plan = pddl_plan

    result = {
    "task_domain": task_domain,
    "task_id": task_id,
    "episode_id": episode_id,
    "completed": True,
    "planning_success": planning_success,
    "judge_pass": judge_pass,
    "sharegpt_sample": None,
}
    return result


def find_task_action_template(root_dir: Path, model_name: str, task_domain: str, task_id: str) -> str:
    task_dir = root_dir / "eval_results" / model_name / task_domain / task_id
    if not task_dir.is_dir():
        return ""

    for ep_dir in sorted(task_dir.iterdir()):
        if not ep_dir.is_dir():
            continue

        round_dirs = sorted(
            [p for p in ep_dir.iterdir() if p.is_dir() and p.name.startswith("round") and p.name[5:].isdigit()],
            key=lambda p: int(p.name[5:]),
        )
        if not round_dirs:
            continue

        domain_path = round_dirs[-1] / "domain.pddl"
        if not domain_path.is_file():
            continue

        lines = domain_path.read_text(encoding="utf-8").splitlines()
        blocks = []
        i = 0

        while i < len(lines):
            if "(:action" not in lines[i]:
                i += 1
                continue

            start = i
            k = i - 1
            while k >= 0 and not lines[k].strip():
                k -= 1
            while k >= 0 and lines[k].lstrip().startswith(";"):
                start = k
                k -= 1

            depth = 0
            j = i
            while j < len(lines):
                depth += lines[j].count("(") - lines[j].count(")")
                if depth == 0:
                    break
                j += 1

            blocks.append("\n".join(lines[start:j + 1]).strip())
            i = j + 1

        return "\n\n".join(blocks)

    return ""


"""
注意! task_domain遵循以下路径命名: 
tasks_path = root_dir / "tasks" / "instructions" / f"instructions_{task_domain}.json"
- keyframes: 
        keyframes_root = root_dir / "dataset" / "keyframes" / {task_domain}
- json:  
        img_dir = root_dir / "tasks" / "images" / {task_domain} 
        steps_path = root_dir / "tasks" / "steps" / f"steps_{task_domain}.json"
"""
if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    PDDL_MODEL = "gemini-3-flash-preview"
    Learn_steps_MODEL = "gemini-3-flash-preview"
    JUDGE_MODEL = "gemini-3-flash-preview"
    MAX_STEP_BACKTRACKS = 10
    MAX_PLAN_ATTEMPTS = 3
    MAX_WORKERS = 50

    task_domain = "agibot_aug_v1"
    tasks = load_tasks(root_dir, task_domain)
    run_all_tasks(root_dir, tasks)