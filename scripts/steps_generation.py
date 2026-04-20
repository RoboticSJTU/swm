"""
遍历 /inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/tasks/images/swm 下的任务图片，
读取对应的 instructions_swm.json，
并发调用 VLM 为每个 task 生成基于单张图像 + 指令的机器人原子动作 steps，
最终保存为 /inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/tasks/steps/steps_swm.json

本版本的原则：
1. 保留单臂机器人身份、视觉检查、禁止低层原语等提示约束。
2. 删除会导致结果被拒收的“硬校验”机制，尤其是不再检查“place 的对象必须与当前 held object 严格一致”。
3. 只做温和清洗（格式规范化、去重、过滤明显低层动作），不再因为动作逻辑问题判失败。
4. 只要 VLM 最终返回了可解析的 steps，就保存结果；人工再二次修改即可。
"""

import json
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Any

ROOT = Path("/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm")
TASK_DOMAIN = "swm"
IMAGE_DIR = ROOT / "tasks/images" / TASK_DOMAIN
INSTRUCTION_PATH = ROOT / "tasks/instructions" / f"instructions_{TASK_DOMAIN}.json"
SAVE_PATH = ROOT / "tasks/steps" / f"steps_{TASK_DOMAIN}.json"
FAILED_PATH = ROOT / "tasks/steps" / f"steps_{TASK_DOMAIN}_failed.json"

MODEL = "gemini-3-flash-preview"
MAX_WORKERS = 100
SAVE_EVERY = 20
MAX_RETRIES = 3

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from swm.utils.apis import call_gpt_json


LOW_LEVEL_PATTERNS = [
    r"\bgrasp\b",
    r"\brelease\b",
    r"\breach\b",
    r"\bapproach\b",
    r"\bmove\s+to\b",
    r"\bgo\s+to\b",
    r"\bnavigate\s+to\b",
    r"\bwalk\s+to\b",
]


def task_sort_key(task_id: str) -> int:
    m = re.search(r"\d+", task_id)
    return int(m.group()) if m else 10**18


def build_prompt(task_id: str, instruction: str, feedback: str = "") -> str:
    feedback_block = ""
    if feedback:
        feedback_block = f"""
Previous output had formatting problems:
{feedback}
Please regenerate the full plan from scratch and return clean JSON.
"""

    return f"""You are an expert single-arm robot task decomposer.

Robot identity and capability constraints:
1. The robot has only one arm and one gripper.
2. It can hold at most one object at a time.
3. If an object needs to be placed, put somewhere, inserted, hung, stacked, or transferred by the gripper, it should be picked first.
4. Do not assume a second hand, a helper hand, or simultaneous bimanual manipulation.

Inputs:
1. A single image showing the current scene.
2. A task instruction.

Your job:
Infer a minimal sequence of robot atomic actions that can complete the instruction from the shown scene.

Critical visual grounding requirements:
1. You must carefully inspect the image before planning.
2. You must identify each relevant object's visible initial location and visible initial state from the image.
3. Every action must be grounded in the visible scene. Do not invent hidden objects, hidden contents, invisible states, or unsupported placements.
4. If there are multiple similar objects or candidate targets, disambiguate only with clearly visible cues such as left/right, top/bottom, front/back, color, size, material, relative position, or visible contents.
5. Picks should refer to the object's visible source location when that source is visually clear.
6. Opens/closes/pulls/pushes should respect the object's visible initial state. For example, do not close something that already appears closed, and do not open something that already appears open.

No low-level primitives:
- Do not output low-level primitives such as grasp, release, reach, approach, move to, go to, or navigate to.
- Output manipulation-level actions only.

Output requirements:
1. Output JSON only. No markdown, no code fence, no extra text.
2. The JSON format must be exactly:
{{
  "steps": [
    "Action 1.",
    "Action 2."
  ]
}}
3. Each step must be one robot atomic action in concise imperative English.
4. Prefer manipulation verbs such as pick, place, put, open, close, push, pull, slide, pour, insert, remove, turn on, turn off, cut, wash, stir, stack, hang, transfer.
5. Do not output low-level motion fragments.
6. Keep the plan minimal but sufficient.
7. Do not add uncertain details, invisible states, or unnecessary attributes.
8. If repeated identical operations are needed on multiple similar items and the exact count is unclear from the image, you may use a representative per-item atomic sequence instead of duplicating the same step many times.
9. Prefer explicit source and target mentions whenever they are visually clear.

Good examples:
- "Pick the cucumber from the left shelf."
- "Open the top drawer."
- "Place the cucumber into the plastic bag in the cart."
- "Push the microwave button to open the microwave."

Bad examples:
- "Reach the cucumber."
- "Approach the drawer."
- "Move to the sink."

Task ID: {task_id}
Instruction: {instruction}
{feedback_block}
"""


def extract_steps_from_any_response(response: Any) -> Optional[List[str]]:
    if isinstance(response, dict):
        for key in ["steps", "actions", "plan", "result", "solution"]:
            if key in response:
                value = response[key]
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    return [x.strip() for x in re.split(r"\n+", value) if x.strip()]

        flat_lines = []
        for v in response.values():
            if isinstance(v, str):
                flat_lines.extend([x.strip() for x in re.split(r"\n+", v) if x.strip()])
        return flat_lines or None

    if isinstance(response, list):
        return response

    if isinstance(response, str):
        return [x.strip() for x in re.split(r"\n+", response) if x.strip()]

    return None


def normalize_steps(response: Any) -> Optional[List[str]]:
    steps = extract_steps_from_any_response(response)
    if not isinstance(steps, list):
        return None

    cleaned = []
    seen = set()
    for s in steps:
        if not isinstance(s, str):
            continue
        s = s.strip().strip('"').strip("'")
        s = re.sub(r"^\s*(?:[-*]|\d+[.)]|Step\s*\d+\s*[:.-])\s*", "", s, flags=re.I)
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue
        if s[-1] not in ".!?":
            s += "."
        key = s.lower()
        if key not in seen:
            cleaned.append(s)
            seen.add(key)

    return cleaned if cleaned else None


def contains_low_level_primitive(step: str) -> bool:
    s = step.lower()
    return any(re.search(p, s) for p in LOW_LEVEL_PATTERNS)


def soft_clean_steps(steps: List[str]) -> Tuple[List[str], List[str]]:
    """只做温和清洗，不再因为规则问题拒收结果。"""
    cleaned = []
    warnings = []

    for i, step in enumerate(steps, start=1):
        s = re.sub(r"\s+", " ", step).strip()
        if not s:
            continue
        if s[-1] not in ".!?":
            s += "."

        if contains_low_level_primitive(s):
            warnings.append(f"Step {i} removed for low-level primitive: {s}")
            continue

        cleaned.append(s)

    if not cleaned and steps:
        # 万一全被过滤掉，宁可回退保留原始 steps，也不要判失败
        cleaned = [re.sub(r"\s+", " ", s).strip() if isinstance(s, str) else str(s) for s in steps]
        cleaned = [s + "." if s and s[-1] not in ".!?" else s for s in cleaned if s]
        warnings.append("All steps were filtered during soft cleaning, so raw normalized steps were restored.")

    return cleaned, warnings


def call_once(task_id: str, image_path: Path, instruction: str, feedback: str = ""):
    prompt = build_prompt(task_id, instruction, feedback=feedback)
    response = call_gpt_json(MODEL, prompt, image_paths=[image_path])
    steps = normalize_steps(response)
    return response, steps


def run_one(task_id: str, image_path: Path, instruction: str):
    feedback = ""
    last_error = ""
    last_steps = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response, steps = call_once(task_id, image_path, instruction, feedback=feedback)
            if not steps:
                last_error = f"Invalid response structure: {response}"
                feedback = last_error
                continue

            cleaned_steps, warnings = soft_clean_steps(steps)
            if cleaned_steps:
                warn_msg = " | ".join(warnings) if warnings else None
                return task_id, cleaned_steps, warn_msg

            last_steps = steps
            last_error = "No valid steps remained after cleaning."
            feedback = last_error
        except Exception as e:
            last_error = str(e)
            feedback = f"The previous attempt failed with this error: {last_error}"

    # 最后兜底：只要拿到过 steps，就直接保存，不再判失败
    if last_steps:
        restored_steps, warnings = soft_clean_steps(last_steps)
        warn_msg = "Fallback saved after retries."
        if warnings:
            warn_msg += " | " + " | ".join(warnings)
        return task_id, restored_steps or last_steps, warn_msg

    return task_id, None, last_error


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_result() -> dict:
    if SAVE_PATH.exists():
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if TASK_DOMAIN in existing and isinstance(existing[TASK_DOMAIN], dict):
            return existing
        return {TASK_DOMAIN: existing}
    return {TASK_DOMAIN: {}}


def load_failed_record() -> dict:
    failed = {
        "missing_instruction": [],
        "missing_image": [],
        "vlm_failed": {},
        "warnings": {},
    }
    if FAILED_PATH.exists():
        try:
            with open(FAILED_PATH, "r", encoding="utf-8") as f:
                old_failed = json.load(f)
            if isinstance(old_failed, dict):
                failed.update(old_failed)
                failed.setdefault("warnings", {})
        except Exception:
            pass
    return failed


def build_image_map() -> dict:
    image_map = {}
    for p in sorted(IMAGE_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            if re.fullmatch(r"task_\d+", p.stem) and p.stem not in image_map:
                image_map[p.stem] = p
    return image_map


def main():
    with open(INSTRUCTION_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if TASK_DOMAIN in raw and isinstance(raw[TASK_DOMAIN], dict):
        task2inst = raw[TASK_DOMAIN]
    else:
        task2inst = raw

    image_map = build_image_map()
    result = load_existing_result()
    failed = load_failed_record()

    done_tasks = set(result[TASK_DOMAIN].keys())
    all_task_ids = sorted(set(task2inst.keys()) | set(image_map.keys()), key=task_sort_key)

    jobs = []
    for task_id in all_task_ids:
        if task_id in done_tasks:
            continue

        has_inst = task_id in task2inst
        has_img = task_id in image_map

        if not has_inst:
            failed["missing_instruction"].append(task_id)
            continue
        if not has_img:
            failed["missing_image"].append(task_id)
            continue

        jobs.append((task_id, image_map[task_id], task2inst[task_id]))

    failed["missing_instruction"] = sorted(set(failed["missing_instruction"]), key=task_sort_key)
    failed["missing_image"] = sorted(set(failed["missing_image"]), key=task_sort_key)

    print(f"已有结果: {len(done_tasks)}")
    print(f"图片数量: {len(image_map)}")
    print(f"指令数量: {len(task2inst)}")
    print(f"待处理任务: {len(jobs)}")

    if not jobs:
        save_json(SAVE_PATH, result)
        save_json(FAILED_PATH, failed)
        print("没有需要处理的新任务。")
        return

    finished = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(run_one, task_id, image_path, instruction): task_id
            for task_id, image_path, instruction in jobs
        }

        for future in as_completed(futures):
            task_id = futures[future]
            try:
                task_id, steps, msg = future.result()
                if steps:
                    result[TASK_DOMAIN][task_id] = steps
                    failed["vlm_failed"].pop(task_id, None)
                    if msg:
                        failed.setdefault("warnings", {})[task_id] = msg
                        print(f"[OK-WARN] {task_id}: {len(steps)} steps | {msg}")
                    else:
                        failed.setdefault("warnings", {}).pop(task_id, None)
                        print(f"[OK] {task_id}: {len(steps)} steps")
                else:
                    failed["vlm_failed"][task_id] = msg or "Unknown error"
                    print(f"[FAIL] {task_id}: {msg}")
            except Exception as e:
                failed["vlm_failed"][task_id] = str(e)
                print(f"[EXCEPTION] {task_id}: {e}")

            finished += 1
            if finished % SAVE_EVERY == 0:
                result[TASK_DOMAIN] = dict(sorted(result[TASK_DOMAIN].items(), key=lambda x: task_sort_key(x[0])))
                failed["vlm_failed"] = dict(sorted(failed["vlm_failed"].items(), key=lambda x: task_sort_key(x[0])))
                failed["warnings"] = dict(sorted(failed.get("warnings", {}).items(), key=lambda x: task_sort_key(x[0])))
                save_json(SAVE_PATH, result)
                save_json(FAILED_PATH, failed)
                print(f"已阶段性保存: {finished}/{len(jobs)}")

    result[TASK_DOMAIN] = dict(sorted(result[TASK_DOMAIN].items(), key=lambda x: task_sort_key(x[0])))
    failed["vlm_failed"] = dict(sorted(failed["vlm_failed"].items(), key=lambda x: task_sort_key(x[0])))
    failed["warnings"] = dict(sorted(failed.get("warnings", {}).items(), key=lambda x: task_sort_key(x[0])))

    save_json(SAVE_PATH, result)
    save_json(FAILED_PATH, failed)

    print("\n处理完成")
    print(f"成功: {len(result[TASK_DOMAIN])}")
    print(f"缺少指令: {len(failed['missing_instruction'])}")
    print(f"缺少图片: {len(failed['missing_image'])}")
    print(f"VLM失败: {len(failed['vlm_failed'])}")
    print(f"带警告保存: {len(failed.get('warnings', {}))}")
    print(f"结果已保存到: {SAVE_PATH}")
    print(f"失败记录已保存到: {FAILED_PATH}")


if __name__ == "__main__":
    main()