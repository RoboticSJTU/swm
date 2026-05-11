import json
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
批量清洗评测结果并生成 ShareGPT 训练数据。

整体流程：
1. 遍历多个 task domain，例如 human、human_aug_v1 到 human_aug_v6；
2. 读取每个 domain 对应的 instruction 文件，只支持嵌套 JSON 结构；
3. 并发清理无效的 eval episode 目录：
   - 删除不在 instruction 中的 episode；
   - 删除没有 round 结果的 episode；
   - 删除缺少 domain.pddl、problem.pddl 或 judge.json 的 episode；
   - 删除 judge.json 中没有 pass 字段或 pass != true 的 episode；
   - 删除清理后产生的空 task 目录；
4. 基于保留下来的有效 episode 并发构造 ShareGPT 样本：
   - 读取 prompt 模板并填入 instruction；
   - 读取最新 round 中的 domain.pddl 和 problem.pddl；
   - 仅在写入 JSON 时清理 PDDL 文本，不修改原始 PDDL 文件；
   - 统一 domain/problem 名称，去除 PDDL 注释，并移除 problem objects 中的类型标注；
   - 将 domain 中 action 的 :precondition 和 :effect 统一压缩为单行；
   - 查找对应 episode 的首帧 keyframe 图像，找不到时回退到 tasks/images 中查找；
5. 并发处理多个 domain，将所有有效样本合并后写入统一的 sharegpt.json。
"""

ROOT_DIR = Path("/home/xyx/下载/swm")
MODEL_NAME = "gemini-3-flash-preview"
PDDL_DOMAIN_NAME = "single_arm"

TASK_DOMAINS = [
    "human",
    "human_aug_v0",
]

KEYFRAMES_ROOT = ROOT_DIR / "dataset/keyframes"
IMAGES_ROOT = ROOT_DIR / "tasks/images"
PROMPT_PATH = ROOT_DIR / "src/swm/prompt_templates/training_input.txt"

OUT_PATH = ROOT_DIR / f"eval_results/{MODEL_NAME}/sharegpt.json"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

MAX_DOMAIN_WORKERS = 7
MAX_EPISODE_WORKERS = 512

PRINT_DELETES = False


# ============================================================
# 只用于写入 ShareGPT JSON 的 PDDL 清理
# 原始 domain.pddl / problem.pddl 不会被修改
# ============================================================

def compact_domain_precondition_effect(text):
    for keyword in [":precondition", ":effect"]:
        pos = 0

        while True:
            start = text.find(keyword, pos)
            if start == -1:
                break

            expr_start = start + len(keyword)

            while expr_start < len(text) and text[expr_start].isspace():
                expr_start += 1

            if expr_start >= len(text) or text[expr_start] != "(":
                pos = start + len(keyword)
                continue

            depth = 0
            end = expr_start

            while end < len(text):
                if text[end] == "(":
                    depth += 1
                elif text[end] == ")":
                    depth -= 1
                    if depth == 0:
                        end += 1
                        break
                end += 1

            expr = text[expr_start:end]
            expr = re.sub(r"\s+", " ", expr.strip())

            replacement = f"{keyword} {expr}"
            text = text[:start] + replacement + text[end:]
            pos = start + len(replacement)

    return text


def clean_pddl_for_json(text, is_domain):
    lines = []

    for line in text.splitlines():
        line = line.split(";", 1)[0].rstrip()
        if line:
            lines.append(line)

    text = "\n".join(lines)

    if is_domain:
        text = re.sub(
            r"\(\s*define\s*\(\s*domain\s+[^()\s]+\s*\)",
            f"(define (domain {PDDL_DOMAIN_NAME})",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        text = compact_domain_precondition_effect(text)
        return text.strip()

    def remove_object_types(match):
        body = re.sub(r"\s*-\s*[^\s()]+", "", match.group(2))
        return match.group(1) + body + match.group(3)

    text = re.sub(
        r"(\(\s*:objects\b)(.*?)(\n\s*\))",
        remove_object_types,
        text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"\(\s*define\s*\(\s*problem\s+[^()\s]+\s*\)",
        "(define (problem task)",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\(\s*:domain\s+[^()\s]+\s*\)",
        f"(:domain {PDDL_DOMAIN_NAME})",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ============================================================
# 并发检查并清理单个 episode
# ============================================================

def check_and_clean_episode(ep_dir, eval_root, allowed_keys):
    parts = ep_dir.relative_to(eval_root).parts

    if len(parts) == 1:
        task_id = None
        episode_id = parts[0]
    elif len(parts) == 2:
        task_id = parts[0]
        episode_id = parts[1]
    else:
        return {
            "status": "unexpected_path",
            "reason": "unexpected_path",
            "ep_dir": ep_dir,
            "task_id": None,
            "episode_id": None,
            "round_dir": None,
            "deleted": False,
        }

    if (task_id, episode_id) not in allowed_keys:
        shutil.rmtree(ep_dir)
        return {
            "status": "deleted_not_in_instruction",
            "reason": "not_in_instruction",
            "ep_dir": ep_dir,
            "task_id": task_id,
            "episode_id": episode_id,
            "round_dir": None,
            "deleted": True,
        }

    round_dirs = [
        p for p in ep_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"round\d+", p.name)
    ]
    round_dirs.sort(key=lambda p: int(p.name.replace("round", "")))

    if not round_dirs:
        shutil.rmtree(ep_dir)
        return {
            "status": "deleted_no_round",
            "reason": "no_round",
            "ep_dir": ep_dir,
            "task_id": task_id,
            "episode_id": episode_id,
            "round_dir": None,
            "deleted": True,
        }

    round_dir = round_dirs[-1]
    domain_path = round_dir / "domain.pddl"
    problem_path = round_dir / "problem.pddl"
    judge_path = round_dir / "judge.json"

    if not domain_path.is_file():
        shutil.rmtree(ep_dir)
        return {
            "status": "deleted_no_domain",
            "reason": "no_domain",
            "ep_dir": ep_dir,
            "task_id": task_id,
            "episode_id": episode_id,
            "round_dir": None,
            "deleted": True,
        }

    if not problem_path.is_file():
        shutil.rmtree(ep_dir)
        return {
            "status": "deleted_no_problem",
            "reason": "no_problem",
            "ep_dir": ep_dir,
            "task_id": task_id,
            "episode_id": episode_id,
            "round_dir": None,
            "deleted": True,
        }

    if not judge_path.is_file():
        shutil.rmtree(ep_dir)
        return {
            "status": "deleted_no_judge",
            "reason": "no_judge",
            "ep_dir": ep_dir,
            "task_id": task_id,
            "episode_id": episode_id,
            "round_dir": None,
            "deleted": True,
        }

    judge = json.loads(judge_path.read_text(encoding="utf-8"))

    if "pass" not in judge:
        shutil.rmtree(ep_dir)
        return {
            "status": "deleted_no_pass_field",
            "reason": "no_pass_field",
            "ep_dir": ep_dir,
            "task_id": task_id,
            "episode_id": episode_id,
            "round_dir": None,
            "deleted": True,
        }

    if judge["pass"] is not True:
        shutil.rmtree(ep_dir)
        return {
            "status": "deleted_failed",
            "reason": "judge_failed",
            "ep_dir": ep_dir,
            "task_id": task_id,
            "episode_id": episode_id,
            "round_dir": None,
            "deleted": True,
        }

    return {
        "status": "kept",
        "reason": "kept",
        "ep_dir": ep_dir,
        "task_id": task_id,
        "episode_id": episode_id,
        "round_dir": round_dir,
        "deleted": False,
    }


# ============================================================
# 并发构造单个 ShareGPT 样本
# ============================================================

def build_one_sample(record, task_domain, prompt_template, valid_round_map):
    task_id = record["task_id"]
    episode_id = record["episode_id"]

    round_dir = valid_round_map.get((task_id, episode_id))

    if round_dir is None:
        return "skipped_missing_episode", None

    domain_path = round_dir / "domain.pddl"
    problem_path = round_dir / "problem.pddl"

    if not domain_path.is_file() or not problem_path.is_file():
        return "skipped_missing_pddl", None

    image_path = None
    keyframe_dirs = []

    if task_id is not None:
        keyframe_dirs.append(
            KEYFRAMES_ROOT / task_domain / task_id / episode_id / "seg_00"
        )

    keyframe_dirs.append(
        KEYFRAMES_ROOT / task_domain / episode_id / "seg_00"
    )

    for seg_dir in keyframe_dirs:
        if not seg_dir.is_dir():
            continue

        images = [
            p for p in seg_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
            and p.stem.isdigit()
        ]
        images.sort(key=lambda p: int(p.stem))

        if images:
            image_path = str(images[0])
            break

    if image_path is None:
        image_candidates = []

        if task_id is not None:
            for suffix in IMAGE_SUFFIXES:
                image_candidates.append(
                    IMAGES_ROOT / task_domain / task_id / f"{episode_id}{suffix}"
                )

        for suffix in IMAGE_SUFFIXES:
            image_candidates.append(
                IMAGES_ROOT / task_domain / f"{episode_id}{suffix}"
            )

        for path in image_candidates:
            if path.is_file():
                image_path = str(path)
                break

    if image_path is None:
        return "skipped_missing_image", None

    domain_text = clean_pddl_for_json(
        domain_path.read_text(encoding="utf-8"),
        is_domain=True,
    )

    problem_text = clean_pddl_for_json(
        problem_path.read_text(encoding="utf-8"),
        is_domain=False,
    )

    sample = {
        "messages": [
            {
                "role": "user",
                "content": "<image>\n" + prompt_template.replace(
                    "{instruction}",
                    record["instruction"],
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"<domain>\n{domain_text}\n</domain>\n"
                    f"<problem>\n{problem_text}\n</problem>"
                ),
            },
        ],
        "images": [image_path],
    }

    return "saved", sample


# ============================================================
# 处理单个 domain
# ============================================================

def process_one_domain(task_domain):
    instructions_json = ROOT_DIR / f"tasks/instructions/instructions_{task_domain}.json"
    eval_root = ROOT_DIR / f"eval_results/{MODEL_NAME}/{task_domain}"

    print(f"\n========== start {task_domain} ==========")

    # ------------------------------------------------------------
    # 1. 读取 instructions
    # 只支持嵌套结构：
    #   {
    #     "task_1": {
    #       "episode_1": "..."
    #     }
    #   }
    # ------------------------------------------------------------

    data = json.loads(instructions_json.read_text(encoding="utf-8"))
    records = []

    for task_id, episode_map in data.items():
        for episode_id, instruction in episode_map.items():
            if isinstance(instruction, list):
                instruction = "\n".join(instruction)

            records.append({
                "task_id": task_id,
                "episode_id": episode_id,
                "instruction": instruction,
            })

    allowed_keys = {(r["task_id"], r["episode_id"]) for r in records}

    # ------------------------------------------------------------
    # 2. 并发清理无效 eval episode
    # 同时记录每个有效 episode 的最新 round_dir，后续构造数据时不再重复遍历 round
    # ------------------------------------------------------------

    clean_stats = {
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
        [
            p for p in list(eval_root.glob("task_*/episode_*")) + list(eval_root.glob("episode_*"))
            if p.is_dir()
        ],
        key=lambda p: str(p),
    )

    clean_stats["total_episode_dirs"] = len(episode_dirs)

    valid_round_map = {}

    with ThreadPoolExecutor(max_workers=MAX_EPISODE_WORKERS) as executor:
        futures = [
            executor.submit(check_and_clean_episode, ep_dir, eval_root, allowed_keys)
            for ep_dir in episode_dirs
        ]

        for future in as_completed(futures):
            result = future.result()
            status = result["status"]

            clean_stats[status] += 1

            if status == "kept":
                valid_round_map[(result["task_id"], result["episode_id"])] = result["round_dir"]
            elif PRINT_DELETES and result["deleted"]:
                print(f"[{task_domain}] [DELETE] {result['ep_dir']} -> {result['reason']}")
            elif PRINT_DELETES and status == "unexpected_path":
                print(f"[{task_domain}] [SKIP] unexpected episode path: {result['ep_dir']}")

    for path in sorted(eval_root.iterdir(), key=lambda p: str(p)):
        if path.is_dir() and not path.name.startswith("episode") and not any(path.iterdir()):
            path.rmdir()
            clean_stats["removed_empty_task_dirs"] += 1
            if PRINT_DELETES:
                print(f"[{task_domain}] [DELETE] empty task dir -> {path}")

    # ------------------------------------------------------------
    # 3. 并发构造当前 domain 的 ShareGPT 数据
    # ------------------------------------------------------------

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    build_stats = {
        "total_records": len(records),
        "saved": 0,
        "skipped_missing_episode": 0,
        "skipped_no_round": 0,
        "skipped_missing_pddl": 0,
        "skipped_missing_image": 0,
    }

    samples = []

    with ThreadPoolExecutor(max_workers=MAX_EPISODE_WORKERS) as executor:
        futures = [
            executor.submit(
                build_one_sample,
                record,
                task_domain,
                prompt_template,
                valid_round_map,
            )
            for record in records
        ]

        for future in as_completed(futures):
            status, sample = future.result()
            build_stats[status] += 1

            if sample is not None:
                samples.append(sample)

    print(f"\n[{task_domain}] clean_eval_results")
    for key, value in clean_stats.items():
        print(f"{key:<28}: {value}")

    print(f"\n[{task_domain}] build_sharegpt")
    for key, value in build_stats.items():
        print(f"{key:<28}: {value}")

    print(f"========== done {task_domain} ==========")

    return {
        "task_domain": task_domain,
        "samples": samples,
        "clean_stats": clean_stats,
        "build_stats": build_stats,
    }


# ============================================================
# 并发处理多个 domain，并合并输出为一个 sharegpt.json
# ============================================================

def process_many_domains(task_domains):
    all_samples = []
    results = []

    with ThreadPoolExecutor(max_workers=MAX_DOMAIN_WORKERS) as executor:
        future_map = {}

        for task_domain in task_domains:
            future = executor.submit(process_one_domain, task_domain)
            future_map[future] = task_domain

        for future in as_completed(future_map):
            task_domain = future_map[future]
            result = future.result()

            results.append(result)
            all_samples.extend(result["samples"])

            print(
                f"\n[MERGE] {task_domain}: "
                f"{len(result['samples'])} samples merged"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(all_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== final summary ==========")
    print(f"domains          : {len(task_domains)}")
    print(f"total samples    : {len(all_samples)}")
    print(f"output           : {OUT_PATH}")

    print("\n========== per-domain saved samples ==========")
    for result in sorted(results, key=lambda x: x["task_domain"]):
        task_domain = result["task_domain"]
        saved = result["build_stats"]["saved"]
        total = result["build_stats"]["total_records"]
        print(f"{task_domain:<16}: {saved} / {total}")

    return all_samples


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    process_many_domains(TASK_DOMAINS)