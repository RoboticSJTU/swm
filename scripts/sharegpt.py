import json
import re
import shutil
from pathlib import Path

"""
本脚本用于将通过 judge 的 PDDL 结果整理成 ShareGPT 训练数据。

核心原则：
1. 原始 domain.pddl / problem.pddl 文件绝对不修改。
2. 删除注释、统一 domain/problem 名字、删除 objects 类型标注等清理操作，
   只作用于最终写入 ShareGPT JSON 的 assistant 输出。
3. clean_eval_results 只负责删除无效 episode，不清理、不覆盖 PDDL 文件。
"""

# ============================================================
# 配置区：只需要改这里
# ============================================================

ROOT_DIR = Path("/home/xyx/下载/swm")
TASK_DOMAIN = "human_aug_v6"
PDDL_DOMAIN_NAME = "single_arm"

INSTRUCTIONS_JSON = ROOT_DIR / f"tasks/instructions/instructions_{TASK_DOMAIN}.json"
MODEL_NAME = "gemini-3-flash-preview"
EVAL_ROOT = ROOT_DIR / f"eval_results/{MODEL_NAME}/{TASK_DOMAIN}"
KEYFRAMES_ROOT = ROOT_DIR / "dataset/keyframes"
IMAGES_ROOT = ROOT_DIR / "tasks/images"
PROMPT_PATH = ROOT_DIR / "src/swm/prompt_templates/training_input.txt"

OUT_PATH = ROOT_DIR / f"eval_results/{MODEL_NAME}/{TASK_DOMAIN}_sharegpt_tag_no_all_com.json"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


# ============================================================
# 读取 instruction json
# ============================================================

def instruction_to_text(instruction):
    if isinstance(instruction, str):
        return instruction

    if isinstance(instruction, list) and all(isinstance(x, str) for x in instruction):
        return "\n".join(instruction)

    raise ValueError(f"非法 instruction 类型: {type(instruction)}")


def load_instruction_records(instructions_json, task_domain):
    data = json.loads(instructions_json.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"instructions_json 必须是 dict: {instructions_json}")

    records = []

    # 格式 1：
    # {
    #   "human": {
    #       "episode_1": "xxx"
    #   }
    # }
    if len(data) == 1 and task_domain in data and isinstance(data[task_domain], dict):
        for episode_id, instruction in data[task_domain].items():
            records.append({
                "task_id": None,
                "episode_id": episode_id,
                "instruction": instruction_to_text(instruction),
            })
        return records

    # 格式 2：
    # {
    #   "task_1": {
    #       "episode_1": "xxx"
    #   }
    # }
    for task_id, episode_map in data.items():
        if not isinstance(episode_map, dict):
            raise ValueError(f"非法 json 格式: {task_id} 对应的值不是 dict")

        for episode_id, instruction in episode_map.items():
            records.append({
                "task_id": task_id,
                "episode_id": episode_id,
                "instruction": instruction_to_text(instruction),
            })

    return records


# ============================================================
# 仅用于写入 JSON 的 PDDL 清理
# 注意：这些函数不能写回原始 .pddl 文件
# ============================================================

def remove_comments_for_json(text):
    lines = []

    for line in text.splitlines():
        line = line.split(";", 1)[0].rstrip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def clean_domain_for_json(domain_text, domain_name):
    domain_text = remove_comments_for_json(domain_text)

    domain_text = re.sub(
        r"\(\s*define\s*\(\s*domain\s+[^()\s]+\s*\)",
        f"(define (domain {domain_name})",
        domain_text,
        count=1,
        flags=re.IGNORECASE,
    )

    return domain_text.strip()


def clean_problem_for_json(problem_text, domain_name):
    problem_text = remove_comments_for_json(problem_text)

    # 删除 (:objects ...) 中的类型标注：
    #   obj1 obj2 - object
    # 变成：
    #   obj1 obj2
    def remove_object_types(match):
        head = match.group(1)
        body = match.group(2)
        tail = match.group(3)

        body = re.sub(r"\s*-\s*[^\s()]+", "", body)
        return head + body + tail

    problem_text = re.sub(
        r"(\(\s*:objects\b)(.*?)(\n\s*\))",
        remove_object_types,
        problem_text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )

    problem_text = re.sub(
        r"\(\s*define\s*\(\s*problem\s+[^()\s]+\s*\)",
        "(define (problem task)",
        problem_text,
        count=1,
        flags=re.IGNORECASE,
    )

    problem_text = re.sub(
        r"\(\s*:domain\s+[^()\s]+\s*\)",
        f"(:domain {domain_name})",
        problem_text,
        count=1,
        flags=re.IGNORECASE,
    )

    return problem_text.strip()


# ============================================================
# eval 目录辅助逻辑
# ============================================================

def latest_round_dir(episode_dir):
    round_dirs = [
        p for p in episode_dir.iterdir()
        if p.is_dir() and p.name.startswith("round") and p.name[5:].isdigit()
    ]

    if not round_dirs:
        return None

    return max(round_dirs, key=lambda p: int(p.name[5:]))


def find_episode_dir(eval_root, task_id, episode_id):
    candidates = []

    if task_id is not None:
        candidates.append(eval_root / task_id / episode_id)

    candidates.append(eval_root / episode_id)

    for path in candidates:
        if path.is_dir():
            return path

    return None


def find_image(task_domain, task_id, episode_id, keyframes_root, images_root):
    seg_dirs = []

    if task_id is not None:
        seg_dirs.append(keyframes_root / task_domain / task_id / episode_id / "seg_00")

    seg_dirs.append(keyframes_root / task_domain / episode_id / "seg_00")

    for seg_dir in seg_dirs:
        if not seg_dir.is_dir():
            continue

        images = [
            p for p in seg_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
            and p.stem.isdigit()
        ]

        if images:
            images.sort(key=lambda p: int(p.stem))
            return str(images[0])

    if task_id is not None:
        for suffix in IMAGE_SUFFIXES:
            path = images_root / task_domain / task_id / f"{episode_id}{suffix}"
            if path.is_file():
                return str(path)

    for suffix in IMAGE_SUFFIXES:
        path = images_root / task_domain / f"{episode_id}{suffix}"
        if path.is_file():
            return str(path)

    return None


# ============================================================
# 1. 清理无效 episode
# 注意：这里不再修改 domain.pddl / problem.pddl
# ============================================================

def clean_eval_results(records, eval_root):
    allowed_keys = {(r["task_id"], r["episode_id"]) for r in records}

    stats = {
        "total_episode_dirs": 0,
        "kept": 0,
        "deleted_not_in_instruction": 0,
        "deleted_no_round": 0,
        "deleted_no_domain": 0,
        "deleted_no_problem": 0,
        "deleted_no_judge": 0,
        "deleted_no_pass_field": 0,
        "deleted_failed": 0,
        "unexpected_path": 0,
        "removed_empty_task_dirs": 0,
    }

    episode_dirs = sorted(
        [p for p in eval_root.rglob("*") if p.is_dir() and p.name.startswith("episode")],
        key=lambda p: str(p),
    )

    def delete_episode(ep_dir, stat_key, reason):
        shutil.rmtree(ep_dir)
        stats[stat_key] += 1
        print(f"[DELETE] {ep_dir} -> {reason}")

    for ep_dir in episode_dirs:
        stats["total_episode_dirs"] += 1

        parts = ep_dir.relative_to(eval_root).parts

        if len(parts) == 1:
            task_id = None
            episode_id = parts[0]
        elif len(parts) == 2:
            task_id = parts[0]
            episode_id = parts[1]
        else:
            stats["unexpected_path"] += 1
            print(f"[SKIP] unexpected episode path: {ep_dir}")
            continue

        if (task_id, episode_id) not in allowed_keys:
            delete_episode(ep_dir, "deleted_not_in_instruction", "not_in_instruction")
            continue

        round_dir = latest_round_dir(ep_dir)
        if round_dir is None:
            delete_episode(ep_dir, "deleted_no_round", "no_round")
            continue

        domain_path = round_dir / "domain.pddl"
        problem_path = round_dir / "problem.pddl"
        judge_path = round_dir / "judge.json"

        if not domain_path.is_file():
            delete_episode(ep_dir, "deleted_no_domain", "no_domain")
            continue

        if not problem_path.is_file():
            delete_episode(ep_dir, "deleted_no_problem", "no_problem")
            continue

        if not judge_path.is_file():
            delete_episode(ep_dir, "deleted_no_judge", "no_judge")
            continue

        judge = json.loads(judge_path.read_text(encoding="utf-8"))

        if "pass" not in judge:
            delete_episode(ep_dir, "deleted_no_pass_field", "no_pass_field")
            continue

        if judge["pass"] is not True:
            delete_episode(ep_dir, "deleted_failed", "judge_failed")
            continue

        stats["kept"] += 1

    for path in sorted(eval_root.iterdir(), key=lambda p: str(p)):
        if path.is_dir() and not path.name.startswith("episode") and not any(path.iterdir()):
            path.rmdir()
            stats["removed_empty_task_dirs"] += 1
            print(f"[DELETE] empty task dir -> {path}")

    print("\n[clean_eval_results] done")
    for key, value in stats.items():
        print(f"{key:<28}: {value}")

    return stats


# ============================================================
# 2. 构造 ShareGPT 数据
# 只在这里清理注释，原始 PDDL 文件不会被修改
# ============================================================

def build_sharegpt(records, task_domain, keyframes_root, images_root, eval_root, prompt_path, out_path, domain_name):
    prompt_template = prompt_path.read_text(encoding="utf-8")

    stats = {
        "total_records": len(records),
        "saved": 0,
        "skipped_missing_episode": 0,
        "skipped_no_round": 0,
        "skipped_missing_pddl": 0,
        "skipped_missing_image": 0,
    }

    samples = []

    for record in records:
        task_id = record["task_id"]
        episode_id = record["episode_id"]

        ep_dir = find_episode_dir(eval_root, task_id, episode_id)
        if ep_dir is None:
            stats["skipped_missing_episode"] += 1
            continue

        round_dir = latest_round_dir(ep_dir)
        if round_dir is None:
            stats["skipped_no_round"] += 1
            continue

        domain_path = round_dir / "domain.pddl"
        problem_path = round_dir / "problem.pddl"

        if not domain_path.is_file() or not problem_path.is_file():
            stats["skipped_missing_pddl"] += 1
            continue

        image_path = find_image(
            task_domain=task_domain,
            task_id=task_id,
            episode_id=episode_id,
            keyframes_root=keyframes_root,
            images_root=images_root,
        )

        if image_path is None:
            stats["skipped_missing_image"] += 1
            continue

        # 关键点：
        # 这里只清理写入 JSON 的文本，不写回原始 .pddl 文件。
        domain_text = clean_domain_for_json(
            domain_path.read_text(encoding="utf-8"),
            domain_name,
        )

        problem_text = clean_problem_for_json(
            problem_path.read_text(encoding="utf-8"),
            domain_name,
        )

        user_content = "<image>\n" + prompt_template.replace(
            "{instruction}",
            record["instruction"],
        )

        assistant_content = (
            f"<domain>\n{domain_text}\n</domain>\n"
            f"<problem>\n{problem_text}\n</problem>"
        )

        samples.append({
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
            ],
            "images": [image_path],
        })

        stats["saved"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[build_sharegpt] done")
    for key, value in stats.items():
        print(f"{key:<28}: {value}")
    print(f"[build_sharegpt] output -> {out_path}")

    return samples


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    records = load_instruction_records(
        instructions_json=INSTRUCTIONS_JSON,
        task_domain=TASK_DOMAIN,
    )

    clean_eval_results(
        records=records,
        eval_root=EVAL_ROOT,
    )

    build_sharegpt(
        records=records,
        task_domain=TASK_DOMAIN,
        keyframes_root=KEYFRAMES_ROOT,
        images_root=IMAGES_ROOT,
        eval_root=EVAL_ROOT,
        prompt_path=PROMPT_PATH,
        out_path=OUT_PATH,
        domain_name=PDDL_DOMAIN_NAME,
    )