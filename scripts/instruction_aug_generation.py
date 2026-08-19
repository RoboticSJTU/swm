#!/usr/bin/env python3
"""根据视频关键帧分段和剩余动作生成多条增强指令。

脚本支持断点续跑，并保存增强后的指令、步骤、元数据及对应图片软链接。
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.llm import call_gpt_json


# 修改这些配置后直接运行本文件，无需传入命令行参数。
SOURCE_TASK_DOMAIN = "human"
AUG_TASK_DOMAIN = f"{SOURCE_TASK_DOMAIN}_aug"
PLAN_MODEL_NAME = "gpt-5.6-sol"
CALL_GPT_MODEL = "gpt-5.6-sol"
AUG_TASK_IDS = [276]  # "ALL" 表示全部，或填写 [1, 5] 仅增强 task_1 和 task_5

AUG_FACTOR = 1
MAX_WORKERS = 50
MAX_RETRY = 10
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

STYLE_GUIDES = [
    "Use a short direct command.",
    "Use a polite but concise request.",
    "Phrase the instruction around the final goal.",
    "Emphasize explicit final states without explaining hidden steps.",
    "Use a different high-level verb such as put, place, set, move, stow, store, tidy, or arrange.",
    "Make the final task objects prominent.",
    "Make the final target region or destination prominent.",
    "Use natural household-task wording.",
    "Use neutral assistant-like wording.",
    "Use one compact sentence.",
    "Clearly preserve explicit final constraints such as closed, locked, returned, turned off, or placed.",
    "Stay close to the high-level meaning but change the wording enough to be distinct.",
]

ROOT = Path(__file__).resolve().parents[1]


def natural_key(text):
    """让 task_2 排在 task_10 前面。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def collect_jobs(source_data, progress_counts):
    """为每个关键帧分段整理原指令、已完成动作、剩余动作和图片。"""
    jobs = []
    selected_tasks = None if AUG_TASK_IDS == "ALL" else {f"task_{task_id}" for task_id in AUG_TASK_IDS}
    for group_id, episodes in sorted(source_data.items(), key=lambda item: natural_key(item[0])):
        if selected_tasks is not None and group_id not in selected_tasks:
            continue
        for episode_id, instruction in sorted(episodes.items(), key=lambda item: natural_key(item[0])):
            instruction = " ".join(str(instruction).strip(" \"'").split())
            if not instruction:
                continue

            keyframe_dir = ROOT / "dataset" / "keyframes" / SOURCE_TASK_DOMAIN / group_id / episode_id
            group_file = (
                ROOT
                / "eval_results"
                / PLAN_MODEL_NAME
                / SOURCE_TASK_DOMAIN
                / group_id
                / episode_id
                / "kf_plan_group.txt"
            )

            if not keyframe_dir.is_dir() or not group_file.is_file():
                continue

            actions = {}
            for line in group_file.read_text(encoding="utf-8").splitlines():
                match = re.match(r"^\[G(\d+)\]\s*(.+?)\s*$", line.strip())
                if match:
                    gid = int(match.group(1))
                    actions.setdefault(gid, []).append(" ".join(match.group(2).split()))

            for segment_dir in sorted(keyframe_dir.glob("seg_*"), key=lambda path: natural_key(path.name)):
                match = re.fullmatch(r"seg_(\d+)", segment_dir.name)
                if not segment_dir.is_dir() or not match:
                    continue

                start_gid = int(match.group(1))
                if start_gid not in actions:
                    continue

                images = sorted(
                    [path for path in segment_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS],
                    key=lambda path: natural_key(path.name),
                )
                if not images:
                    continue

                completed = []
                remaining = []
                for gid in sorted(actions):
                    if gid < start_gid:
                        completed.extend(actions[gid])
                    else:
                        remaining.extend(actions[gid])

                pair = (str(group_id), str(episode_id), start_gid)
                existing = progress_counts[pair] if pair in progress_counts else 0
                target = len(remaining) * AUG_FACTOR
                if target > existing:
                    jobs.append(
                        {
                            "group_id": str(group_id),
                            "episode_id": str(episode_id),
                            "instruction": instruction,
                            "start_gid": start_gid,
                            "image_path": images[0],
                            "completed": completed,
                            "remaining": remaining,
                            "target": target,
                            "existing": existing,
                            "need": target - existing,
                        }
                    )

    return sorted(
        jobs,
        key=lambda job: (
            natural_key(job["group_id"]),
            natural_key(job["episode_id"]),
            job["start_gid"],
        ),
    )


def generate_instructions(prompt_template, job, existing_instructions):
    """调用模型生成一个分段的全部增强指令；格式错误时重新生成。"""
    error = ""
    banned_words = [
        "image",
        "photo",
        "scene",
        "camera",
        "frame",
        "robot",
        "left hand",
        "right hand",
        "both hands",
        "remaining steps",
        "completed actions",
    ]

    def normalize(text):
        text = re.sub(r"[^\w\s]", " ", " ".join(str(text).lower().split()))
        text = re.sub(r"\b(the|a|an|please|could|would|can|just)\b", " ", text)
        return " ".join(text.split())

    for retry in range(1, MAX_RETRY + 1):
        styles = [
            f"{slot}. slot={slot}: {STYLE_GUIDES[(job['existing'] + slot - 1) % len(STYLE_GUIDES)]}"
            for slot in range(1, job["need"] + 1)
        ]
        prompt = prompt_template.format(
            instruction=job["instruction"],
            completed_actions="\n".join(
                f"{index}. {action}" for index, action in enumerate(job["completed"], 1)
            )
            or "- none",
            remaining_actions="\n".join(
                f"{index}. {action}" for index, action in enumerate(job["remaining"], 1)
            )
            or "- none",
            existing_block="\n".join(f"- {text}" for text in existing_instructions) or "none",
            style_slots="\n".join(styles),
            need=job["need"],
            retry_block=(
                f"Previous output problem:\n{error}\nRegenerate the full JSON and fix it." if error else ""
            ),
        )

        try:
            data = call_gpt_json(CALL_GPT_MODEL, prompt, [job["image_path"]])
            variants = sorted(data["variants"], key=lambda item: int(item["slot"]))
            instructions = [" ".join(str(item["instruction"]).strip(" \"'").split()) for item in variants]
            normalized = [normalize(text) for text in instructions]

            if len(variants) != job["need"]:
                raise ValueError(f"expected {job['need']} variants, got {len(variants)}")
            if [int(item["slot"]) for item in variants] != list(range(1, job["need"] + 1)):
                raise ValueError("slots must be consecutive and unique")
            if any(len(text.split()) < 3 for text in normalized):
                raise ValueError("an instruction is too short")
            if any(word in text for text in normalized for word in banned_words):
                raise ValueError("an instruction contains banned text")
            if len(set(normalized)) != len(normalized):
                raise ValueError("generated instructions contain duplicates")
            if set(normalized) & {normalize(text) for text in existing_instructions}:
                raise ValueError("an instruction duplicates existing data")

            for item, instruction in zip(variants, instructions):
                item["instruction"] = instruction

            return {
                "ok": True,
                "job": job,
                "variants": variants,
                "shared_goal": data["shared_goal"],
                "hidden_details_not_mentioned": data["hidden_details_not_mentioned"],
                "retry": retry,
                "error": "",
            }
        except Exception as exception:
            error = str(exception)

    return {"ok": False, "job": job, "variants": [], "retry": MAX_RETRY, "error": error}


def main():
    prompt_path = ROOT / "src" / "swm" / "prompt_templates" / "instruction_aug.txt"
    source_path = ROOT / "tasks" / "instructions" / f"instructions_{SOURCE_TASK_DOMAIN}.json"
    instruction_path = ROOT / "tasks" / "instructions" / f"instructions_{AUG_TASK_DOMAIN}.json"
    steps_path = ROOT / "tasks" / "steps" / f"steps_{AUG_TASK_DOMAIN}.json"
    meta_path = ROOT / "tasks" / "meta" / f"meta_{AUG_TASK_DOMAIN}.json"
    image_root = ROOT / "tasks" / "images" / AUG_TASK_DOMAIN

    prompt_template = prompt_path.read_text(encoding="utf-8")
    source_data = json.loads(source_path.read_text(encoding="utf-8"))
    instruction_data = json.loads(instruction_path.read_text(encoding="utf-8")) if instruction_path.is_file() else {}
    steps_data = json.loads(steps_path.read_text(encoding="utf-8")) if steps_path.is_file() else {}
    meta_data = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    # 已生成数量用于断点续跑，已有文本用于避免重复。
    progress_counts = {}
    existing_texts = {}
    for group_id, episodes in meta_data.items():
        for episode_id, meta in episodes.items():
            if not isinstance(meta, dict) or not {"source_group_id", "source_episode_id", "start_gid"} <= meta.keys():
                continue
            pair = (str(meta["source_group_id"]), str(meta["source_episode_id"]), int(meta["start_gid"]))
            progress_counts[pair] = progress_counts[pair] + 1 if pair in progress_counts else 1
            if group_id in instruction_data and episode_id in instruction_data[group_id]:
                existing_texts.setdefault(pair, []).append(instruction_data[group_id][episode_id])

    jobs = collect_jobs(source_data, progress_counts)
    print(f"jobs: {len(jobs)}")
    if not jobs:
        print("No jobs to run.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as executor:
        futures = [
            executor.submit(
                generate_instructions,
                prompt_template,
                job,
                existing_texts[pair] if pair in existing_texts else [],
            )
            for job in jobs
            for pair in [(job["group_id"], job["episode_id"], job["start_gid"])]
        ]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            job = result["job"]
            print(
                f"[{index}/{len(futures)}] {'OK' if result['ok'] else 'FAIL'} "
                f"{job['group_id']}/{job['episode_id']}/G{job['start_gid']} "
                f"target={job['target']} existing={job['existing']} need={job['need']} "
                f"new={len(result['variants'])} retry={result['retry']} {result['error']}"
            )

    results.sort(
        key=lambda result: (
            natural_key(result["job"]["group_id"]),
            natural_key(result["job"]["episode_id"]),
            result["job"]["start_gid"],
        )
    )

    next_episode_id = {}
    for group_id, episodes in instruction_data.items():
        numbers = [int(match.group(1)) for name in episodes if (match := re.search(r"(\d+)$", name))]
        next_episode_id[group_id] = max(numbers) + 1 if numbers else 1

    created = 0
    for result in results:
        if not result["ok"]:
            continue

        job = result["job"]
        group_id = job["group_id"]
        if group_id not in instruction_data:
            instruction_data[group_id] = {}
            steps_data[group_id] = {}
            meta_data[group_id] = {}
            next_episode_id[group_id] = 1

        for variant in result["variants"]:
            episode_id = f"episode_{next_episode_id[group_id]}"
            next_episode_id[group_id] += 1
            output_image = image_root / group_id / f"{episode_id}.png"
            output_image.parent.mkdir(parents=True, exist_ok=True)
            if output_image.exists() or output_image.is_symlink():
                output_image.unlink()
            output_image.symlink_to(os.path.relpath(job["image_path"].resolve(), output_image.parent))

            instruction_data[group_id][episode_id] = variant["instruction"]
            steps_data[group_id][episode_id] = job["remaining"]
            meta_data[group_id][episode_id] = {
                "source_group_id": group_id,
                "source_episode_id": job["episode_id"],
                "source_instruction": job["instruction"],
                "start_gid": job["start_gid"],
                "slot": int(variant["slot"]),
                "style_label": variant["style_label"],
                "shared_goal": result["shared_goal"],
                "hidden_details_not_mentioned": result["hidden_details_not_mentioned"],
                "source_image_path": str(job["image_path"]),
                "image_path": str(output_image.relative_to(ROOT)),
                "steps": job["remaining"],
                "completed_actions": job["completed"],
                "target_count": job["target"],
                "existing_count_before_run": job["existing"],
                "need_count_before_run": job["need"],
                "aug_version": "simple_grouped_implicit_instruction_aug_v3",
            }
            created += 1

    for path, data in [(instruction_path, instruction_data), (steps_path, steps_data), (meta_path, meta_data)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"created: {created}")
    print(f"total: {sum(len(episodes) for episodes in instruction_data.values())}")
    print(f"output instruction json: {instruction_path}")
    print(f"output steps json: {steps_path}")
    print(f"output meta json: {meta_path}")
    print(f"output image root: {image_root}")


if __name__ == "__main__":
    main()
