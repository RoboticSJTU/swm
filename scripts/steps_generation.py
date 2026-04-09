"""
遍历 /home/xyx/下载/swm/tasks/images/swm_100 下的任务图片，
读取对应的 instructions_swm_100.json，
并发调用 VLM 为每个 task 生成基于单张图像 + 指令的机器人原子动作 steps，
最终保存为 /home/xyx/下载/swm/tasks/steps/steps_swm_100.json

输出格式：
{
  "swm_100": {
    "task_1": [
      "Pick cucumber from the shelf.",
      ...
    ]
  }
}

说明：
1. 支持断点续跑：如果 steps_swm_100.json 已存在，会跳过已完成 task。
2. 支持并发调用 VLM。
3. 支持失败记录：会额外保存一个 steps_swm_100_failed.json。
4. 默认使用 gemini-3-flash-preview，可自行修改 MODEL。
"""
# 需要加入单臂人设
import json
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path("/home/xyx/下载/swm")
TASK_DOMAIN = "swm_100"
IMAGE_DIR = ROOT / "tasks/images" / TASK_DOMAIN
INSTRUCTION_PATH = ROOT / "tasks/instructions" / f"instructions_{TASK_DOMAIN}.json"
SAVE_PATH = ROOT / "tasks/steps" / f"steps_{TASK_DOMAIN}.json"
FAILED_PATH = ROOT / "tasks/steps" / f"steps_{TASK_DOMAIN}_failed.json"

MODEL = "gemini-3-flash-preview"
MAX_WORKERS = 16
SAVE_EVERY = 20

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from swm.utils.apis import call_gpt_json


def task_sort_key(task_id: str) -> int:
    m = re.search(r"\d+", task_id)
    return int(m.group()) if m else 10**18


def build_prompt(task_id: str, instruction: str) -> str:
    return f"""You are an expert robot task decomposer.

You are given:
1. A single image showing the current scene.
2. A task instruction.

Your job:
Infer a sequence of robot atomic actions that can complete the instruction from the shown scene.

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
4. Use robot manipulation actions such as pick, place, put, open, close, push, pull, slide, move, pour, insert, remove, turn on, turn off, cut, wash, stir, stack, hang, transfer.
5. Make actions specific by including the immediate purpose or effect when visually clear.
6. If multiple similar objects, object parts, or target locations are present, disambiguate them only with clearly visible cues such as top/middle/bottom, left/right, front/back, color, size, material, or contents.
7. Do not add uncertain details, invisible states, or unnecessary attributes.
8. Do not output low-level primitives such as move away, reach, or approach.
9. Keep the plan minimal but sufficient.
10. Prefer concrete actions such as:
   - "Push the microwave button to open the microwave."
   - "Move the kettle under the faucet."
   - "Open the top drawer."
11. If repeated identical operations are needed on multiple similar items and the exact count is unclear, you may use a representative per-item atomic sequence instead of duplicating the same step many times.

Task ID: {task_id}
Instruction: {instruction}
"""


def normalize_steps(response):
    if not isinstance(response, dict):
        return None

    steps = None
    for key in ["steps", "actions", "plan", "result", "solution"]:
        if key in response:
            steps = response[key]
            break

    if steps is None:
        return None

    if isinstance(steps, str):
        steps = [x.strip() for x in re.split(r"\n+", steps) if x.strip()]

    if not isinstance(steps, list):
        return None

    cleaned = []
    last = None

    for s in steps:
        if not isinstance(s, str):
            continue
        s = s.strip()
        s = re.sub(r"^\s*(?:[-*]|\d+[.)]|Step\s*\d+\s*[:.-])\s*", "", s, flags=re.I)
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue
        if s[-1] not in ".!?":
            s += "."
        if s != last:
            cleaned.append(s)
            last = s

    return cleaned if cleaned else None


def run_one(task_id: str, image_path: Path, instruction: str):
    prompt = build_prompt(task_id, instruction)

    for _ in range(2):
        try:
            response = call_gpt_json(MODEL, prompt, image_paths=[image_path])
            steps = normalize_steps(response)
            if steps:
                return task_id, steps, None
        except Exception as e:
            last_error = str(e)
        else:
            last_error = f"Invalid response: {response}"

    return task_id, None, last_error


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    with open(INSTRUCTION_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if TASK_DOMAIN in raw and isinstance(raw[TASK_DOMAIN], dict):
        task2inst = raw[TASK_DOMAIN]
    else:
        task2inst = raw

    image_map = {}
    for p in sorted(IMAGE_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            if re.fullmatch(r"task_\d+", p.stem):
                if p.stem not in image_map:
                    image_map[p.stem] = p

    if SAVE_PATH.exists():
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if TASK_DOMAIN in existing and isinstance(existing[TASK_DOMAIN], dict):
            result = existing
        else:
            result = {TASK_DOMAIN: existing}
    else:
        result = {TASK_DOMAIN: {}}

    failed = {
        "missing_instruction": [],
        "missing_image": [],
        "vlm_failed": {}
    }
    if FAILED_PATH.exists():
        try:
            with open(FAILED_PATH, "r", encoding="utf-8") as f:
                old_failed = json.load(f)
            if isinstance(old_failed, dict):
                failed = old_failed
        except Exception:
            pass

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
                task_id, steps, err = future.result()
                if steps:
                    result[TASK_DOMAIN][task_id] = steps
                    print(f"[OK] {task_id}: {len(steps)} steps")
                else:
                    failed["vlm_failed"][task_id] = err or "Unknown error"
                    print(f"[FAIL] {task_id}: {err}")
            except Exception as e:
                failed["vlm_failed"][task_id] = str(e)
                print(f"[EXCEPTION] {task_id}: {e}")

            finished += 1
            if finished % SAVE_EVERY == 0:
                result[TASK_DOMAIN] = dict(sorted(result[TASK_DOMAIN].items(), key=lambda x: task_sort_key(x[0])))
                failed["vlm_failed"] = dict(sorted(failed["vlm_failed"].items(), key=lambda x: task_sort_key(x[0])))
                save_json(SAVE_PATH, result)
                save_json(FAILED_PATH, failed)
                print(f"已阶段性保存: {finished}/{len(jobs)}")

    result[TASK_DOMAIN] = dict(sorted(result[TASK_DOMAIN].items(), key=lambda x: task_sort_key(x[0])))
    failed["vlm_failed"] = dict(sorted(failed["vlm_failed"].items(), key=lambda x: task_sort_key(x[0])))

    save_json(SAVE_PATH, result)
    save_json(FAILED_PATH, failed)

    print("\n处理完成")
    print(f"成功: {len(result[TASK_DOMAIN])}")
    print(f"缺少指令: {len(failed['missing_instruction'])}")
    print(f"缺少图片: {len(failed['missing_image'])}")
    print(f"VLM失败: {len(failed['vlm_failed'])}")
    print(f"结果已保存到: {SAVE_PATH}")
    print(f"失败记录已保存到: {FAILED_PATH}")


if __name__ == "__main__":
    main()