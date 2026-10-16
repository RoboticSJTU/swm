#!/usr/bin/env python3
"""根据关键帧动作边界生成完整、语义分支及其后缀任务指令。"""

import json
import os
import re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from swm.llm import call_gpt_json


# 修改这些配置后直接运行本文件，无需传入命令行参数。
SOURCE_TASK_DOMAIN = "agibotworld2026_balanced"
AUG_TASK_DOMAIN = f"{SOURCE_TASK_DOMAIN}_aug"
PLAN_MODEL_NAME = "gpt-5.6-sol"
CALL_GPT_MODEL = "gpt-5.6-sol"
AUG_TASK_IDS = "ALL"  # "ALL" 表示全部，或填写 [1, 5] 仅增强 task_1 和 task_5
ALLOW_EXTERNAL_API = False

MAX_WORKERS = 20
MAX_RETRY = 3
MIN_START_GID = 1  # 普通后缀默认从 seg_01 开始；语义分支仍保留各自的基准样本。
DIVERSITY_SIMILARITY_THRESHOLD = 0.90
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
BRANCH_ORDER = {"full": 0, "phase_a": 1, "phase_b": 2}

STYLE_GUIDES = [
    "Use a concise direct command; identify objects with grounded attributes when useful.",
    "Frame the request around the desired end state and a task-relevant spatial relation.",
    "Use the achieved current state as context without requesting completed work again.",
    "Make the destination or device state the information focus.",
    "Make the manipulated object and its final relation the information focus.",
    "Express the ordered outcome naturally without turning it into low-level steps.",
    "Identify the target through a grounded nearby object or scene location.",
    "Use a natural state-transition command and avoid stock placement wording.",
]

ROOT = Path(__file__).resolve().parents[1]


def natural_key(text):
    """让 task_2 排在 task_10 前面。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def normalize_instruction(text):
    """用于去重比较；不改变实际写入的自然语言。"""
    text = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return " ".join(text.split())


def target_signature(actions):
    """目标语义相同但仅执行手不同的动作归为一组。"""
    signature = []
    for action in actions:
        normalized = normalize_instruction(action)
        normalized = re.sub(r"\b(?:the )?(?:left|right|both) hands?\b", "", normalized)
        signature.append(" ".join(normalized.split()))
    return tuple(signature)


def similar_instruction(text, candidates, threshold=DIVERSITY_SIMILARITY_THRESHOLD):
    normalized = normalize_instruction(text)
    for candidate in candidates:
        candidate_normalized = normalize_instruction(candidate)
        if normalized == candidate_normalized:
            return candidate
        if SequenceMatcher(None, normalized, candidate_normalized).ratio() >= threshold:
            return candidate
    return None


def read_indexed_actions(path):
    """读取 ``[Gn] action``。"""
    actions = {}
    previous_gid = -1
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"\[G(\d+)\]\s+(.+?)\s*", line)
        if not match:
            raise ValueError(f"{path}:{line_number}: expected '[Gn] action', got {line!r}")
        gid = int(match.group(1))
        if gid < previous_gid:
            raise ValueError(f"{path}:{line_number}: action indices are not ordered")
        previous_gid = gid
        actions.setdefault(gid, []).append(" ".join(match.group(2).split()))
    if not actions:
        raise ValueError(f"{path}: no indexed actions")
    return actions


def resolve_boundary(rule, action_count, group_id):
    boundary = action_count // 2 if rule == "half" else rule
    if not isinstance(boundary, int) or isinstance(boundary, bool):
        raise ValueError(f"{group_id}: semantic boundary must be an integer or 'half'")
    if not 0 < boundary < action_count:
        raise ValueError(
            f"{group_id}: semantic boundary G{boundary} is invalid for {action_count} actions"
        )
    return boundary


def branch_ranges(action_count, boundary):
    if boundary is None:
        return [("full", 0, action_count, range(MIN_START_GID, action_count))]
    return [
        ("full", 0, action_count, range(0, action_count)),
        ("phase_a", 0, boundary, range(0, boundary)),
        ("phase_b", boundary, action_count, range(boundary, action_count)),
    ]


def collect_jobs(source_data, boundary_rules, completed_keys):
    """整理每个完整/语义分支及其有效后缀切点。"""
    jobs = []
    next_episode_index = {}
    selected_tasks = None if AUG_TASK_IDS == "ALL" else {f"task_{task_id}" for task_id in AUG_TASK_IDS}
    for group_id, episodes in sorted(source_data.items(), key=lambda item: natural_key(item[0])):
        if selected_tasks is not None and group_id not in selected_tasks:
            continue
        for episode_id, instruction in sorted(episodes.items(), key=lambda item: natural_key(item[0])):
            instruction = " ".join(str(instruction).strip(" \"'").split())
            if not instruction:
                continue

            keyframe_dir = ROOT / "dataset" / "keyframes" / SOURCE_TASK_DOMAIN / group_id / episode_id
            action_file = (
                ROOT
                / "eval_results"
                / PLAN_MODEL_NAME
                / SOURCE_TASK_DOMAIN
                / group_id
                / episode_id
                / "kf_actions.txt"
            )
            if not keyframe_dir.is_dir():
                raise FileNotFoundError(keyframe_dir)
            if not action_file.is_file():
                raise FileNotFoundError(action_file)
            actions = read_indexed_actions(action_file)
            gids = sorted(actions)
            if gids != list(range(len(gids))):
                raise ValueError(f"{action_file}: expected continuous action indices from G0")
            segment_dirs = {
                int(match.group(1)): segment_dir
                for segment_dir in keyframe_dir.glob("seg_*")
                if segment_dir.is_dir()
                and (match := re.fullmatch(r"seg_(\d+)", segment_dir.name))
            }
            metadata_path = keyframe_dir / "metadata.json"
            if metadata_path.is_file():
                segments = json.loads(metadata_path.read_text(encoding="utf-8")).get("segments", [])
                if segments and all("step" in segment for segment in segments):
                    if [segment["step"] for segment in segments] != [
                        action for gid in gids for action in actions[gid]
                    ] or any(len(actions[gid]) != 1 for gid in gids):
                        raise ValueError(f"{metadata_path}: segment steps do not match action groups")
                    segment_dirs = {
                        gid: keyframe_dir / segment["segment"]
                        for gid, segment in enumerate(segments)
                    }
            if set(segment_dirs) != set(gids):
                raise ValueError(f"{keyframe_dir}: segment ids do not match action ids")

            boundary = None
            if group_id in boundary_rules:
                boundary = resolve_boundary(boundary_rules[group_id], len(gids), group_id)

            for branch, branch_start, branch_end, starts in branch_ranges(len(gids), boundary):
                for start_gid in starts:
                    local_cut_index = start_gid - branch_start
                    output_index = next_episode_index.get(group_id, 0)
                    next_episode_index[group_id] = output_index + 1
                    output_episode_id = f"episode_{output_index}"
                    key = (str(group_id), str(episode_id), branch, local_cut_index)
                    if key in completed_keys:
                        continue

                    segment_dir = segment_dirs[start_gid]
                    images = sorted(
                        [path for path in segment_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS],
                        key=lambda path: natural_key(path.name),
                    )
                    if not images:
                        raise ValueError(f"{segment_dir}: no images")

                    completed = [
                        action
                        for gid in range(branch_start, start_gid)
                        for action in actions[gid]
                    ]
                    target = [
                        action
                        for gid in range(start_gid, branch_end)
                        for action in actions[gid]
                    ]
                    excluded = [
                        action
                        for gid in [*range(0, branch_start), *range(branch_end, len(gids))]
                        for action in actions[gid]
                    ]
                    style_index = (
                        int(str(group_id).removeprefix("task_"))
                        + int(str(episode_id).removeprefix("episode_"))
                        + BRANCH_ORDER[branch]
                        + local_cut_index
                    ) % len(STYLE_GUIDES)
                    job = {
                        "group_id": str(group_id),
                        "episode_id": str(episode_id),
                        "instruction": instruction,
                        "branch": branch,
                        "semantic_boundary_gid": boundary,
                        "branch_start_gid": branch_start,
                        "start_gid": start_gid,
                        "end_gid": branch_end,
                        "local_cut_index": local_cut_index,
                        "image_path": images[0],
                        "completed": completed,
                        "target": target,
                        "excluded": excluded,
                        "style_guide": STYLE_GUIDES[style_index],
                        "output_episode_id": output_episode_id,
                    }
                    jobs.append(job)

    jobs = sorted(
        jobs,
        key=lambda job: (
            natural_key(job["group_id"]),
            natural_key(job["episode_id"]),
            BRANCH_ORDER[job["branch"]],
            job["local_cut_index"],
        ),
    )
    output_keys = [(job["group_id"], job["output_episode_id"]) for job in jobs]
    if len(output_keys) != len(set(output_keys)):
        raise ValueError("output episode-id collision")
    return jobs


def generate_instruction(prompt_template, job):
    """为一个后缀任务生成一条指令；格式错误时带错误信息重试。"""
    error = ""
    banned_pattern = re.compile(
        r"\b(?:image|photo|scene|camera|frame|robot|left hand|right hand|both hands)\b"
        r"|\b(?:remaining steps|completed actions)\b",
        re.IGNORECASE,
    )

    for retry in range(1, MAX_RETRY + 1):
        prompt = prompt_template.format(
            instruction=job["instruction"],
            completed_actions="\n".join(
                f"{index}. {action}" for index, action in enumerate(job["completed"], 1)
            )
            or "- none",
            target_actions="\n".join(
                f"{index}. {action}" for index, action in enumerate(job["target"], 1)
            ),
            excluded_actions="\n".join(
                f"{index}. {action}" for index, action in enumerate(job["excluded"], 1)
            )
            or "- none",
            style_guide=job["style_guide"],
            retry_block=(
                f"Previous output problem:\n{error}\nReturn a corrected JSON object." if error else ""
            ),
        )

        try:
            data = call_gpt_json(CALL_GPT_MODEL, prompt, [job["image_path"]], attempts=1)
            instruction = " ".join(str(data["instruction"]).strip(" \"'").split())
            if not instruction:
                raise ValueError("instruction is empty")
            if banned_pattern.search(instruction):
                raise ValueError("instruction contains forbidden implementation context")
            return {
                "ok": True,
                "job": job,
                "instruction": instruction,
                "retry": retry,
                "error": "",
            }
        except Exception as exception:
            error = str(exception)

    return {
        "ok": False,
        "job": job,
        "instruction": "",
        "retry": MAX_RETRY,
        "error": error,
    }


def diversity_audit(instruction_data, steps_data):
    """仅在同一任务、同一目标动作组内报告重复。"""
    exact = 0
    near = 0
    for group_id, episodes in instruction_data.items():
        grouped = {}
        group_steps = steps_data.get(group_id, {})
        for episode_id, instruction in episodes.items():
            if episode_id not in group_steps:
                continue
            signature = target_signature(group_steps[episode_id])
            grouped.setdefault(signature, []).append(normalize_instruction(instruction))
        for normalized in grouped.values():
            exact += len(normalized) - len(set(normalized))
            for index, value in enumerate(normalized):
                near += sum(
                    SequenceMatcher(None, value, other).ratio()
                    >= DIVERSITY_SIMILARITY_THRESHOLD
                    for other in normalized[index + 1 :]
                )
    return exact, near


def load_journal(path):
    """读取逐条断点记录；仅容忍崩溃留下的不完整末行。"""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            print(f"Ignoring incomplete final journal line in {path}")
            valid_text = "\n".join(lines[:index])
            path.write_text(valid_text + ("\n" if valid_text else ""), encoding="utf-8")
    return records


def append_journal(path, record):
    """先持久化单条成功结果，避免长任务中断后重新调用模型。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def ensure_image_link(meta):
    source = Path(meta["source_image_path"])
    output = ROOT / meta["image_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        if output.resolve() == source.resolve():
            return
        output.unlink()
    elif output.exists():
        raise FileExistsError(f"refusing to replace non-symlink image: {output}")
    output.symlink_to(os.path.relpath(source.resolve(), output.parent))


def apply_record(record, instruction_data, steps_data, meta_data):
    group_id = record["group_id"]
    episode_id = record["episode_id"]
    instruction_group = instruction_data.setdefault(group_id, {})
    steps_group = steps_data.setdefault(group_id, {})
    meta_group = meta_data.setdefault(group_id, {})
    values = (
        (instruction_group, record["instruction"]),
        (steps_group, record["steps"]),
        (meta_group, record["meta"]),
    )
    for target, value in values:
        if episode_id in target and target[episode_id] != value:
            raise ValueError(f"checkpoint conflicts with existing {group_id}/{episode_id}")
        target[episode_id] = value
    ensure_image_link(record["meta"])


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_record(job, instruction, image_root):
    group_id = job["group_id"]
    output_episode_id = job["output_episode_id"]
    output_image = image_root / group_id / f"{output_episode_id}.png"
    meta = {
        "source_group_id": group_id,
        "source_episode_id": job["episode_id"],
        "source_instruction": job["instruction"],
        "branch": job["branch"],
        "semantic_boundary_gid": job["semantic_boundary_gid"],
        "local_cut_index": job["local_cut_index"],
        "clip_start_gid": job["start_gid"],
        "clip_end_gid": job["end_gid"],
        "cut_index": job["start_gid"],
        "source_segment_id": job["image_path"].parent.name,
        "source_frame_index": int(job["image_path"].stem),
        "style_guide": job["style_guide"],
        "source_image_path": str(job["image_path"].resolve()),
        "image_path": str(output_image.relative_to(ROOT)),
        "steps": job["target"],
        "completed_actions": job["completed"],
        "excluded_actions": job["excluded"],
        "aug_version": "semantic_suffix_instruction_aug_v6",
    }
    return {
        "group_id": group_id,
        "episode_id": output_episode_id,
        "instruction": instruction,
        "steps": job["target"],
        "meta": meta,
    }


def completed_job_keys(meta_data):
    keys = set()
    for episodes in meta_data.values():
        for meta in episodes.values():
            if not isinstance(meta, dict):
                continue
            if not {"source_group_id", "source_episode_id"} <= meta.keys():
                continue
            branch = str(meta.get("branch", "full"))
            local_cut_index = meta.get("local_cut_index")
            if local_cut_index is None:
                local_cut_index = meta.get("start_gid", meta.get("cut_index"))
            if local_cut_index is None:
                continue
            keys.add(
                (
                    str(meta["source_group_id"]),
                    str(meta["source_episode_id"]),
                    branch,
                    int(local_cut_index),
                )
            )
    return keys


def main():
    prompt_path = ROOT / "src" / "swm" / "prompt_templates" / "instruction_aug.txt"
    boundary_path = ROOT / "tasks" / "meta" / f"semantic_boundaries_{SOURCE_TASK_DOMAIN}.json"
    source_path = ROOT / "tasks" / "instructions" / f"instructions_{SOURCE_TASK_DOMAIN}.json"
    instruction_path = ROOT / "tasks" / "instructions" / f"instructions_{AUG_TASK_DOMAIN}.json"
    steps_path = ROOT / "tasks" / "steps" / f"steps_{AUG_TASK_DOMAIN}.json"
    meta_path = ROOT / "tasks" / "meta" / f"meta_{AUG_TASK_DOMAIN}.json"
    journal_path = meta_path.with_name(f".{meta_path.stem}.progress.jsonl")
    manual_path = meta_path.with_name(f".{meta_path.stem}.manual.json")
    image_root = ROOT / "tasks" / "images" / AUG_TASK_DOMAIN

    prompt_template = prompt_path.read_text(encoding="utf-8")
    source_data = json.loads(source_path.read_text(encoding="utf-8"))
    boundary_rules = json.loads(boundary_path.read_text(encoding="utf-8")) if boundary_path.is_file() else {}
    instruction_data = json.loads(instruction_path.read_text(encoding="utf-8")) if instruction_path.is_file() else {}
    steps_data = json.loads(steps_path.read_text(encoding="utf-8")) if steps_path.is_file() else {}
    meta_data = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    journal_records = load_journal(journal_path)
    for record in journal_records:
        apply_record(record, instruction_data, steps_data, meta_data)
    if journal_records:
        print(f"replayed checkpoint records: {len(journal_records)}")

    done = completed_job_keys(meta_data)
    jobs = collect_jobs(source_data, boundary_rules, done)
    print(f"completed samples: {len(done)}")
    print(f"jobs: {len(jobs)}")

    manual_data = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.is_file() else {}
    manual_created = 0
    pending_jobs = []
    for job in jobs:
        instruction = manual_data.get(job["group_id"], {}).get(job["output_episode_id"])
        if instruction is None:
            pending_jobs.append(job)
            continue
        instruction = " ".join(str(instruction).strip(" \"'").split())
        if not instruction:
            raise ValueError(
                f"empty manual instruction for {job['group_id']}/{job['output_episode_id']}"
            )
        record = build_record(job, instruction, image_root)
        append_journal(journal_path, record)
        apply_record(record, instruction_data, steps_data, meta_data)
        manual_created += 1
    if manual_created:
        print(f"imported manual samples: {manual_created}")
    jobs = pending_jobs
    print(f"pending manual samples: {len(jobs)}")

    if jobs and not ALLOW_EXTERNAL_API:
        print(f"External API disabled; add instructions to {manual_path}")
        return

    created = 0
    failures = []
    accepted_instructions = {}
    for group_id, episodes in instruction_data.items():
        for episode_id, instruction in episodes.items():
            steps = steps_data.get(group_id, {}).get(episode_id)
            if steps is None:
                continue
            key = (group_id, target_signature(steps))
            accepted_instructions.setdefault(key, []).append(instruction)
    if jobs:
        executor = ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs)))
        try:
            futures = [executor.submit(generate_instruction, prompt_template, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                job = result["job"]
                print(
                    f"[{index}/{len(futures)}] {'OK' if result['ok'] else 'FAIL'} "
                    f"{job['group_id']}/{job['episode_id']}/{job['branch']}/"
                    f"seg_{job['start_gid']:02d} "
                    f"retry={result['retry']} {result['error']}"
                )
                if not result["ok"]:
                    failures.append(result)
                    continue

                duplicate_key = (job["group_id"], target_signature(job["target"]))
                candidates = accepted_instructions.get(duplicate_key, [])
                duplicate = similar_instruction(result["instruction"], candidates)
                if duplicate is not None:
                    result["ok"] = False
                    result["error"] = "instruction duplicates or is too similar to an existing instruction"
                    failures.append(result)
                    continue

                record = build_record(job, result["instruction"], image_root)
                append_journal(journal_path, record)
                apply_record(record, instruction_data, steps_data, meta_data)
                accepted_instructions.setdefault(duplicate_key, []).append(result["instruction"])
                created += 1
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()

    write_json_atomic(instruction_path, instruction_data)
    write_json_atomic(steps_path, steps_data)
    write_json_atomic(meta_path, meta_data)
    journal_path.unlink(missing_ok=True)

    print(f"created: {created}")
    print(f"total: {sum(len(episodes) for episodes in instruction_data.values())}")
    print(f"output instruction json: {instruction_path}")
    print(f"output steps json: {steps_path}")
    print(f"output meta json: {meta_path}")
    print(f"output image root: {image_root}")
    exact_duplicates, near_duplicates = diversity_audit(instruction_data, steps_data)
    print(
        "instruction diversity audit: "
        f"same_target_exact_extras={exact_duplicates} same_target_near_pairs={near_duplicates}"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; rerun to resume them")


if __name__ == "__main__":
    main()
