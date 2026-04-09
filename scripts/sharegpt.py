import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swm.utils.plan_learning import get_prompt_from_template

"""
构建 ShareGPT 风格的 PDDL 训练数据：
1. 读取 instructions_json 中的 task -> episode -> instruction 映射。
2. 遍历 keyframes_root / task_domain / task_xxx / episode_xxx / seg_00
   对每个 episode 选取 seg_00 内文件名数值最小的 png 作为输入图像。
4. 去 eval_root / task_xxx / episode_xxx 下查找编号最大的 round* 目录里的 domain.pddl 和 problem.pddl。
"""


PROMPT_PATH = Path(
    "/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/src/swm/prompt_templates/training_input.txt"
)


def build_sharegpt_pddl(
    instructions_json: Path,
    keyframes_root: Path,
    eval_root: Path,
    out_path: Path,
    max_workers: int = 1280,
) -> list:
    data = json.loads(instructions_json.read_text(encoding="utf-8"))

    first_key = next(iter(data))
    if first_key.startswith("task_"):
        task2ep2inst = data
    else:
        task2ep2inst = data[first_key]

    task_domain = eval_root.name
    keyframe_domain_dir = keyframes_root / task_domain

    img_map = {}
    for task_dir in keyframe_domain_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task = task_dir.name
        img_map[task] = {}

        for ep_dir in task_dir.iterdir():
            if not ep_dir.is_dir():
                continue

            seg_dir = ep_dir / "seg_00"
            if not seg_dir.is_dir():
                continue

            pngs = [
                p for p in seg_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".png" and p.stem.isdigit()
            ]
            if not pngs:
                continue

            first_img = min(pngs, key=lambda p: int(p.stem))
            img_map[task][ep_dir.name] = str(first_img)

    items = []
    for task in task2ep2inst:
        for ep in task2ep2inst[task]:
            instruction = task2ep2inst[task][ep]
            if isinstance(instruction, list):
                instruction = "\n".join(instruction)
            items.append((task, ep, instruction))

    def process(item):
        task, ep, instruction = item

        if task not in img_map:
            return None
        if ep not in img_map[task]:
            return None

        img = img_map[task][ep]
        ep_eval_dir = eval_root / task / ep
        if not ep_eval_dir.is_dir():
            return None

        rounds = []
        for d in ep_eval_dir.iterdir():
            if d.is_dir() and d.name.startswith("round") and d.name[5:].isdigit():
                rounds.append(d)
        if not rounds:
            return None

        round_dir = max(rounds, key=lambda d: int(d.name[5:]))
        domain_p = round_dir / "domain.pddl"
        problem_p = round_dir / "problem.pddl"
        if not domain_p.is_file() or not problem_p.is_file():
            return None

        return {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\n" + get_prompt_from_template(PROMPT_PATH, instruction=instruction),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "domain": domain_p.read_text(encoding="utf-8"),
                            "problem": problem_p.read_text(encoding="utf-8"),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "images": [img],
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        samples = [x for x in ex.map(process, items) if x]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build_sharegpt_pddl] saved={len(samples)} -> {out_path}")
    return samples


if __name__ == "__main__":
    ROOT = Path("/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm")

    build_sharegpt_pddl(
        instructions_json=ROOT / "tasks/instructions/instructions_agibot.json",
        keyframes_root=ROOT / "dataset/keyframes",
        eval_root=ROOT / "eval_results/gemini-3-flash-preview/agibot",
        out_path=ROOT / "eval_results/gemini-3-flash-preview/agibot_sharegpt.json",
    )