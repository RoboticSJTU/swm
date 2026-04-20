import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from swm.utils.plan_learning import get_prompt_from_template


def load_instruction_records(instructions_json: Path, task_domain: str):
    """
    读取 instruction json，并统一展开成 records。
    同时返回 allowed_keys，用于判断某个 episode 是否在指令 json 中。

    统一格式：
    {
        "task_domain": str,
        "task_id": str | None,
        "episode_id": str,
        "instruction": str,
    }

    支持两种输入格式：
    1.
    {
        "task_327": {
            "episode_648642": "xxx"
        },
        ...
    }

    2.
    {
        "human": {
            "episode_1": "xxx",
            ...
        }
    }
    """
    data = json.loads(instructions_json.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"instructions_json 必须是 dict: {instructions_json}")

    records = []
    allowed_keys = set()

    # 格式2：顶层就是 task_domain
    if len(data) == 1 and task_domain in data and isinstance(data[task_domain], dict):
        for episode_id, instruction in data[task_domain].items():
            if isinstance(instruction, str):
                instruction_text = instruction
            elif isinstance(instruction, list) and all(isinstance(x, str) for x in instruction):
                instruction_text = "\n".join(instruction)
            else:
                raise ValueError(
                    f"非法 instruction 类型: {type(instruction)}, episode={episode_id}"
                )

            record = {
                "task_domain": task_domain,
                "task_id": None,
                "episode_id": episode_id,
                "instruction": instruction_text,
            }
            records.append(record)
            allowed_keys.add((None, episode_id))

        return records, allowed_keys

    # 格式1：顶层是多个 task
    for task_id, episode_map in data.items():
        if not isinstance(episode_map, dict):
            raise ValueError(f"非法 json 格式: 顶层 {task_id} 对应的值不是 dict")

        for episode_id, instruction in episode_map.items():
            if isinstance(instruction, str):
                instruction_text = instruction
            elif isinstance(instruction, list) and all(isinstance(x, str) for x in instruction):
                instruction_text = "\n".join(instruction)
            else:
                raise ValueError(
                    f"非法 instruction 类型: {type(instruction)}, task={task_id}, episode={episode_id}"
                )

            record = {
                "task_domain": task_domain,
                "task_id": task_id,
                "episode_id": episode_id,
                "instruction": instruction_text,
            }
            records.append(record)
            allowed_keys.add((task_id, episode_id))

    return records, allowed_keys


def clean_invalid_episodes(eval_root: Path, instructions_json: Path):
    """
    以 instructions_json 为唯一真值源，删除无效 episode。

    删除条件：
    1. 不在指令 json 中
    2. 没有 roundN
    3. 最新 round 下没有 problem.pddl
    4. 最新 round 下没有 judge.json
    5. judge.json 里没有 pass 字段
    6. judge["pass"] 不是 True

    支持两种 eval 目录结构：
    1. eval_root / task_id / episode_id / roundN
    2. eval_root / episode_id / roundN
    """
    eval_root = Path(eval_root)
    task_domain = eval_root.name
    _, allowed_keys = load_instruction_records(instructions_json, task_domain)

    stats = {
        "total_episode_dirs": 0,
        "kept": 0,
        "deleted_not_in_instruction": 0,
        "deleted_no_round": 0,
        "deleted_no_problem": 0,
        "deleted_no_judge": 0,
        "deleted_no_pass_field": 0,
        "deleted_failed": 0,
        "removed_empty_task_dirs": 0,
        "error": 0,
    }

    episode_dirs = sorted(
        [p for p in eval_root.rglob("*") if p.is_dir() and p.name.startswith("episode")],
        key=lambda p: str(p),
    )

    for ep_dir in episode_dirs:
        stats["total_episode_dirs"] += 1

        try:
            rel = ep_dir.relative_to(eval_root)
            parts = rel.parts

            if len(parts) == 1:
                task_id = None
                episode_id = parts[0]
            elif len(parts) == 2:
                task_id = parts[0]
                episode_id = parts[1]
            else:
                stats["error"] += 1
                print(f"[ERROR] unexpected episode path: {ep_dir}")
                continue

            if (task_id, episode_id) not in allowed_keys:
                shutil.rmtree(ep_dir)
                stats["deleted_not_in_instruction"] += 1
                print(f"[DELETE] {ep_dir} -> not_in_instruction")
                continue

            round_dirs = [
                d for d in ep_dir.iterdir()
                if d.is_dir() and d.name.startswith("round") and d.name[5:].isdigit()
            ]
            if not round_dirs:
                shutil.rmtree(ep_dir)
                stats["deleted_no_round"] += 1
                print(f"[DELETE] {ep_dir} -> no_round")
                continue

            latest_round = max(round_dirs, key=lambda d: int(d.name[5:]))

            problem_path = latest_round / "problem.pddl"
            if not problem_path.is_file():
                shutil.rmtree(ep_dir)
                stats["deleted_no_problem"] += 1
                print(f"[DELETE] {ep_dir} -> no_problem")
                continue

            judge_path = latest_round / "judge.json"
            if not judge_path.is_file():
                shutil.rmtree(ep_dir)
                stats["deleted_no_judge"] += 1
                print(f"[DELETE] {ep_dir} -> no_judge")
                continue

            judge = json.loads(judge_path.read_text(encoding="utf-8"))
            if "pass" not in judge:
                shutil.rmtree(ep_dir)
                stats["deleted_no_pass_field"] += 1
                print(f"[DELETE] {ep_dir} -> no_pass_field")
                continue

            if judge["pass"] is not True:
                shutil.rmtree(ep_dir)
                stats["deleted_failed"] += 1
                print(f"[DELETE] {ep_dir} -> judge_failed")
                continue

            stats["kept"] += 1

        except Exception as e:
            stats["error"] += 1
            print(f"[ERROR] {ep_dir}: {e}")

    # 清理已经空掉的 task 目录
    for p in sorted(eval_root.iterdir(), key=lambda x: str(x)):
        if p.is_dir() and not p.name.startswith("episode"):
            if not any(p.iterdir()):
                p.rmdir()
                stats["removed_empty_task_dirs"] += 1
                print(f"[DELETE] empty task dir -> {p}")

    print("\n[clean_invalid_episodes] done")
    for k, v in stats.items():
        print(f"{k:<28}: {v}")

    return stats


def remove_problem_types(eval_root: Path):
    """
    遍历 eval_root 下所有保留下来的 episode，
    仅对其最新 round 的 problem.pddl 做一件事：
    删除 :objects 里的类型标注。

    例如：
        a b c - object
    变成：
        a b c
    """
    eval_root = Path(eval_root)
    objects_re = re.compile(r"(\(\s*:objects\b)(.*?)(\n\s*\))", re.S)

    stats = {
        "total_episode_dirs": 0,
        "modified": 0,
        "unchanged": 0,
        "skipped_no_round": 0,
        "skipped_no_problem": 0,
        "error": 0,
    }

    episode_dirs = sorted(
        [p for p in eval_root.rglob("*") if p.is_dir() and p.name.startswith("episode")],
        key=lambda p: str(p),
    )

    for ep_dir in episode_dirs:
        stats["total_episode_dirs"] += 1

        try:
            round_dirs = [
                d for d in ep_dir.iterdir()
                if d.is_dir() and d.name.startswith("round") and d.name[5:].isdigit()
            ]
            if not round_dirs:
                stats["skipped_no_round"] += 1
                continue

            latest_round = max(round_dirs, key=lambda d: int(d.name[5:]))
            problem_path = latest_round / "problem.pddl"

            if not problem_path.is_file():
                stats["skipped_no_problem"] += 1
                continue

            text = problem_path.read_text(encoding="utf-8")
            match = objects_re.search(text)
            if match is None:
                stats["unchanged"] += 1
                continue

            old_objects = match.group(2)
            new_objects = re.sub(r"\s*-\s*[^\s()]+", "", old_objects)

            if new_objects == old_objects:
                stats["unchanged"] += 1
                continue

            new_text = text[:match.start(2)] + new_objects + text[match.end(2):]
            problem_path.write_text(new_text, encoding="utf-8")
            stats["modified"] += 1
            print(problem_path)

        except Exception as e:
            stats["error"] += 1
            print(f"[ERROR] {ep_dir}: {e}")

    print("\n[remove_problem_types] done")
    for k, v in stats.items():
        print(f"{k:<28}: {v}")

    return stats


def build_sharegpt_pddl(
    instructions_json: Path,
    keyframes_root: Path,
    images_root: Path,
    eval_root: Path,
    prompt_path: Path,
    out_path: Path,
    max_workers: int = 128,
):
    """
    单纯构造 sharegpt 数据。

    数据来源以 instructions_json 为准。
    但只有同时满足下面条件的 record 才会被保存：
    1. 对应 episode 目录存在
    2. 有最新 round
    3. 最新 round 下有 domain.pddl 和 problem.pddl
    4. 能找到图像

    支持两种 eval 路径：
    1. eval_root / task_id / episode_id / roundN
    2. eval_root / episode_id / roundN

    支持两种图像路径：
    1. keyframes_root / task_domain / task_id / episode_id / seg_00 / 首帧.png
    2. keyframes_root / task_domain / episode_id / seg_00 / 首帧.png
    3. images_root / task_domain / task_id / episode_id.png
    4. images_root / task_domain / episode_id.png
    """
    instructions_json = Path(instructions_json)
    keyframes_root = Path(keyframes_root)
    images_root = Path(images_root)
    eval_root = Path(eval_root)
    prompt_path = Path(prompt_path)
    out_path = Path(out_path)

    task_domain = eval_root.name
    records, _ = load_instruction_records(instructions_json, task_domain)

    def process(record):
        task_id = record["task_id"]
        episode_id = record["episode_id"]

        # 找 episode 目录
        episode_dirs = []
        if task_id is not None:
            episode_dirs.append(eval_root / task_id / episode_id)
        episode_dirs.append(eval_root / episode_id)

        ep_dir = None
        for candidate in episode_dirs:
            if candidate.is_dir():
                ep_dir = candidate
                break

        if ep_dir is None:
            return "skipped_missing_episode", None

        # 找最新 round
        round_dirs = [
            d for d in ep_dir.iterdir()
            if d.is_dir() and d.name.startswith("round") and d.name[5:].isdigit()
        ]
        if not round_dirs:
            return "skipped_no_round", None

        latest_round = max(round_dirs, key=lambda d: int(d.name[5:]))

        domain_path = latest_round / "domain.pddl"
        problem_path = latest_round / "problem.pddl"
        if not domain_path.is_file() or not problem_path.is_file():
            return "skipped_missing_pddl", None

        # 找图像
        image_path = None

        seg_dirs = []
        if task_id is not None:
            seg_dirs.append(keyframes_root / task_domain / task_id / episode_id / "seg_00")
        seg_dirs.append(keyframes_root / task_domain / episode_id / "seg_00")

        for seg_dir in seg_dirs:
            if not seg_dir.is_dir():
                continue

            imgs = [
                p for p in seg_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.stem.isdigit()
            ]
            if imgs:
                imgs.sort(key=lambda p: int(p.stem))
                image_path = str(imgs[0])
                break

        if image_path is None and task_id is not None:
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = images_root / task_domain / task_id / f"{episode_id}{ext}"
                if candidate.is_file():
                    image_path = str(candidate)
                    break

        if image_path is None:
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = images_root / task_domain / f"{episode_id}{ext}"
                if candidate.is_file():
                    image_path = str(candidate)
                    break

        if image_path is None:
            return "skipped_missing_image", None

        domain_text = domain_path.read_text(encoding="utf-8")
        problem_text = problem_path.read_text(encoding="utf-8")

        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\n" + get_prompt_from_template(
                        prompt_path,
                        instruction=record["instruction"],
                    ),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "domain": domain_text,
                            "problem": problem_text,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "images": [image_path],
        }
        return "saved", sample

    stats = {
        "total_records": len(records),
        "saved": 0,
        "skipped_missing_episode": 0,
        "skipped_no_round": 0,
        "skipped_missing_pddl": 0,
        "skipped_missing_image": 0,
    }

    samples = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for status, sample in executor.map(process, records):
            stats[status] += 1
            if sample is not None:
                samples.append(sample)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[build_sharegpt_pddl] done")
    for k, v in stats.items():
        print(f"{k:<28}: {v}")
    print(f"[build_sharegpt_pddl] output -> {out_path}")

    return samples


if __name__ == "__main__":
    root_dir = Path("/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm")
    task_domain = "unidomain"

    instructions_json = root_dir / f"tasks/instructions/instructions_{task_domain}.json"
    eval_root = root_dir / f"eval_results/gemini-3-flash-preview/{task_domain}"

    # 1. 先以指令 json 为准，删除所有无效 episode
    clean_invalid_episodes(eval_root=eval_root,instructions_json=instructions_json)

    # 2. 再单独移除 problem.pddl 中的类型标注
    remove_problem_types(eval_root)

    # 3. 最后单独构造 sharegpt 数据
    build_sharegpt_pddl(
        instructions_json=instructions_json,
        keyframes_root=root_dir / "dataset/keyframes",
        images_root=root_dir / "tasks/images",
        eval_root=eval_root,
        prompt_path=root_dir / "src/swm/prompt_templates/training_input.txt",
        out_path=root_dir / f"eval_results/gemini-3-flash-preview/{task_domain}_sharegpt.json",
        max_workers=1280,
    )