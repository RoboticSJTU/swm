import json
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from swm.utils.apis import call_gpt_json


SOURCE_TASK_DOMAIN = "human"
AUG_TASK_DOMAIN = f"{SOURCE_TASK_DOMAIN}_aug_v1"

GEN_MODEL_NAME = "gemini-3-flash-preview"
MAX_WORKERS = 50

# "concise": 每个源 pair 只生成 concise
# "variants": 每个源 pair 生成 concise / appearance_location / verb_synonym
OUTPUT_MODE = "variants"


def clean_text(text) -> str:
    return re.sub(r"\s+", " ", str(text).strip().strip('"').strip("'"))


def natural_key(text: str):
    parts = re.split(r"(\d+)", str(text))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_items(source_path: Path):
    """
    支持两种输入格式：

    1. 多 task/group 嵌套:
    {
      "task_327": {
        "episode_648642": "xxx"
      }
    }

    2. 单 domain 平铺:
    {
      "human": {
        "episode_1": "xxx"
      }
    }

    返回:
    - items: [{"group_id", "episode_id", "instruction"}]
    - flat_mode:
        True  -> 路径按 domain/episode
        False -> 路径按 domain/group/episode
    """
    data = read_json(source_path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid source json: {source_path}")

    # 情况2：只有一个顶层 key，且它就是 SOURCE_TASK_DOMAIN，并且内部 value 都是字符串
    if (
        len(data) == 1
        and SOURCE_TASK_DOMAIN in data
        and isinstance(data[SOURCE_TASK_DOMAIN], dict)
        and all(not isinstance(v, dict) for v in data[SOURCE_TASK_DOMAIN].values())
    ):
        items = []
        for episode_id, instruction in sorted(data[SOURCE_TASK_DOMAIN].items(), key=lambda x: natural_key(x[0])):
            instruction = clean_text(instruction)
            if instruction:
                items.append({
                    "group_id": SOURCE_TASK_DOMAIN,
                    "episode_id": str(episode_id),
                    "instruction": instruction,
                })
        return items, True

    # 情况1：一般化嵌套格式
    items = []
    for group_id, episode_map in sorted(data.items(), key=lambda x: natural_key(x[0])):
        if not isinstance(episode_map, dict):
            raise ValueError(f"Unsupported source json format: top-level value of '{group_id}' must be a dict")

        for episode_id, instruction in sorted(episode_map.items(), key=lambda x: natural_key(x[0])):
            if isinstance(instruction, dict):
                raise ValueError(
                    f"Unsupported source json format: value of '{group_id}/{episode_id}' must be a string"
                )

            instruction = clean_text(instruction)
            if instruction:
                items.append({
                    "group_id": str(group_id),
                    "episode_id": str(episode_id),
                    "instruction": instruction,
                })

    return items, False


def load_saved_nested_json(path: Path):
    data = read_json(path)
    if not isinstance(data, dict):
        return {}

    out = {}
    for group_id, episode_map in data.items():
        if isinstance(episode_map, dict):
            out[str(group_id)] = dict(episode_map)
    return out


def collect_existing_pairs(meta_data: dict):
    """
    已存在样本的去重粒度严格按 source pair:
    (source_group_id, source_episode_id, start_gid)
    """
    existing_pairs = set()

    for episode_map in meta_data.values():
        for meta in episode_map.values():
            if not isinstance(meta, dict):
                continue

            if "source_group_id" in meta and "source_episode_id" in meta and "start_gid" in meta:
                existing_pairs.add((
                    str(meta["source_group_id"]),
                    str(meta["source_episode_id"]),
                    int(meta["start_gid"]),
                ))
            elif "source_task_id" in meta and "source_episode_id" in meta and "start_gid" in meta:
                existing_pairs.add((
                    str(meta["source_task_id"]),
                    str(meta["source_episode_id"]),
                    int(meta["start_gid"]),
                ))

    return existing_pairs


def collect_jobs(root_dir: Path, source_items: list, flat_mode: bool, existing_pairs: set):
    """
    按“真实 seg 原理”构造 job：

    1. 读取 kf_plan_group.txt 中真实存在的 G0/G1/G2...
    2. 枚举 keyframe 下真实存在的 seg_00/seg_01/...
    3. 只有 seg_xx 且 xx 恰好存在于 Gxx 时，才参与增强
    4. 一个 source pair = (group_id, episode_id, start_gid)
    """
    jobs = []

    for item in source_items:
        group_id = item["group_id"]
        episode_id = item["episode_id"]
        instruction = item["instruction"]

        if flat_mode:
            keyframe_dir = root_dir / "dataset" / "keyframes" / SOURCE_TASK_DOMAIN / episode_id
            group_path = (
                root_dir
                / "eval_results"
                / GEN_MODEL_NAME
                / SOURCE_TASK_DOMAIN
                / episode_id
                / "kf_plan_group.txt"
            )
        else:
            keyframe_dir = root_dir / "dataset" / "keyframes" / SOURCE_TASK_DOMAIN / group_id / episode_id
            group_path = (
                root_dir
                / "eval_results"
                / GEN_MODEL_NAME
                / SOURCE_TASK_DOMAIN
                / group_id
                / episode_id
                / "kf_plan_group.txt"
            )

        if not keyframe_dir.is_dir() or not group_path.is_file():
            continue

        # 读取真实 group
        group_to_actions = {}
        for line in group_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\[G(\d+)\]\s*(.+?)\s*$", line.strip())
            if m:
                gid = int(m.group(1))
                action = clean_text(m.group(2))
                group_to_actions.setdefault(gid, []).append(action)

        if not group_to_actions:
            continue

        group_ids = sorted(group_to_actions)

        # 枚举真实 seg_00 / seg_01 ...
        seg_infos = []
        for p in keyframe_dir.iterdir():
            if p.is_dir():
                m = re.fullmatch(r"seg_(\d+)", p.name)
                if m:
                    seg_infos.append((int(m.group(1)), p))
        seg_infos.sort(key=lambda x: x[0])

        for start_gid, seg_dir in seg_infos:
            pair = (group_id, episode_id, start_gid)

            # 关键原则：只有 seg_xx 对应真实 group id 时才参与
            if start_gid not in group_to_actions:
                continue

            if pair in existing_pairs:
                continue

            images = sorted(
                [p for p in seg_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}],
                key=lambda p: natural_key(p.name)
            )
            if not images:
                continue
            image_path = images[0]

            completed = []
            remaining = []
            for gid in group_ids:
                if gid < start_gid:
                    completed.extend(group_to_actions[gid])
                else:
                    remaining.extend(group_to_actions[gid])

            if not remaining:
                continue

            jobs.append({
                "group_id": group_id,
                "episode_id": episode_id,
                "instruction": instruction,
                "start_gid": start_gid,
                "image_path": image_path,
                "completed": completed,
                "remaining": remaining,
            })

    jobs.sort(key=lambda x: (
        natural_key(x["group_id"]),
        natural_key(x["episode_id"]),
        x["start_gid"],
    ))
    return jobs


def build_generation_prompt(instruction: str, completed: list[str], remaining: list[str]):
    completed_text = "- none" if not completed else "\n".join(f"- {x}" for x in completed)
    remaining_text = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(remaining))

    return f"""You are an expert in robot task instruction rewriting.
Your task is to rewrite the remaining task from the current image and the Remaining steps.
The image is the main evidence. The Remaining steps is a reliable description of what still needs to be done.
Completed actions are already finished and must not be asked again.

Global rules:
- describe only the remaining task
- do not redo completed parts
- preserve the original goal and important final constraints. The instruction must be inferable to the given steps.
- do not add new actions, goals, or constraints
- use natural imperative instructions
- do not mention left hand, right hand, both hands, or any hand information

canonical:
- write a complete, clear, high-level instruction for the remaining task

concise:
- shorten the canonical instruction without changing its meaning
- Omit the source or location of the item being operated on. For example, simplify "pick apple from the shelf" to "pick apple".
- Omit redundant low-level actions when they are already implied by the main action.
  Example:
  - steps: pick apple from table + place apple on plate
  - instruction: Place the apple on the plate.

appearance_location:
- rewrite the concise instruction by adding visible appearance and current-location details
- mainly enrich noun phrases, while keeping the task meaning unchanged
- only add details that are clearly visible and reliable in the image.

verb_synonym:
- rewrite the concise instruction by mainly replacing action verbs with natural synonyms

Original instruction:
{instruction}

Completed actions:
{completed_text}

Remaining steps:
{remaining_text}

Return JSON only:
{{
  "reasoning": "step by step think.",
  "canonical": "...",
  "variants": {{
    "concise": "...",
    "appearance_location": "...",
    "verb_synonym": "..."
  }}
}}"""


def rewrite_one_job(job: dict):
    result = call_gpt_json(
        GEN_MODEL_NAME,
        build_generation_prompt(job["instruction"], job["completed"], job["remaining"]),
        [job["image_path"]],
    )

    return {
        "source_group_id": job["group_id"],
        "source_episode_id": job["episode_id"],
        "source_instruction": job["instruction"],
        "start_gid": job["start_gid"],
        "image_src": str(job["image_path"]),
        "steps": job["remaining"],
        "variants": {
            "concise": clean_text(result["variants"]["concise"]),
            "appearance_location": clean_text(result["variants"]["appearance_location"]),
            "verb_synonym": clean_text(result["variants"]["verb_synonym"]),
        },
    }


def copy_image_as_png(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
    else:
        Image.open(src).save(dst)


def sort_nested_dict(data: dict):
    return {
        group_id: dict(sorted(episode_map.items(), key=lambda x: natural_key(x[0])))
        for group_id, episode_map in sorted(data.items(), key=lambda x: natural_key(x[0]))
    }


def main():
    if OUTPUT_MODE == "concise":
        output_variants = ["concise"]
    elif OUTPUT_MODE == "variants":
        output_variants = ["concise", "appearance_location", "verb_synonym"]
    else:
        raise ValueError(f"Unsupported OUTPUT_MODE: {OUTPUT_MODE}")

    root_dir = Path(__file__).resolve().parent.parent

    src_instruction_path = root_dir / "tasks" / "instructions" / f"instructions_{SOURCE_TASK_DOMAIN}.json"
    out_instruction_path = root_dir / "tasks" / "instructions" / f"instructions_{AUG_TASK_DOMAIN}.json"
    out_steps_path = root_dir / "tasks" / "steps" / f"steps_{AUG_TASK_DOMAIN}.json"
    out_meta_path = root_dir / "tasks" / "meta" / f"meta_{AUG_TASK_DOMAIN}.json"
    out_image_root = root_dir / "tasks" / "images" / AUG_TASK_DOMAIN

    out_instruction_path.parent.mkdir(parents=True, exist_ok=True)
    out_steps_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)
    out_image_root.mkdir(parents=True, exist_ok=True)

    # 1. 读取源数据
    source_items, flat_mode = load_source_items(src_instruction_path)

    # 2. 读取已有输出
    instructions_data = load_saved_nested_json(out_instruction_path)
    steps_data = load_saved_nested_json(out_steps_path)
    meta_data = load_saved_nested_json(out_meta_path)

    existing_pairs = collect_existing_pairs(meta_data)

    # 3. 构造所有待生成 job
    jobs = collect_jobs(root_dir, source_items, flat_mode, existing_pairs)
    print(f"jobs: {len(jobs)}")

    # 4. 并发生成
    samples = []
    if jobs:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as executor:
            futures = [executor.submit(rewrite_one_job, job) for job in jobs]
            total = len(futures)

            for i, future in enumerate(as_completed(futures), 1):
                sample = future.result()
                samples.append(sample)
                print(
                    f'[{i}/{total}] '
                    f'{sample["source_group_id"]}/{sample["source_episode_id"]}/G{sample["start_gid"]}'
                )

    samples.sort(key=lambda x: (
        natural_key(x["source_group_id"]),
        natural_key(x["source_episode_id"]),
        x["start_gid"],
    ))

    # 5. 为每个 group 计算下一个 episode 索引
    next_episode_idx = {}
    for group_id, episode_map in instructions_data.items():
        max_idx = 0
        for episode_id in episode_map:
            m = re.search(r"(\d+)$", str(episode_id))
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        next_episode_idx[group_id] = max_idx + 1

    # 6. 写入新样本
    created = 0

    for sample in samples:
        pair = (
            sample["source_group_id"],
            sample["source_episode_id"],
            sample["start_gid"],
        )
        if pair in existing_pairs:
            continue

        group_id = sample["source_group_id"]

        if group_id not in instructions_data:
            instructions_data[group_id] = {}
        if group_id not in steps_data:
            steps_data[group_id] = {}
        if group_id not in meta_data:
            meta_data[group_id] = {}
        if group_id not in next_episode_idx:
            next_episode_idx[group_id] = 1

        for variant_name in output_variants:
            new_episode_id = f"episode_{next_episode_idx[group_id]}"
            next_episode_idx[group_id] += 1

            dst_image = out_image_root / group_id / f"{new_episode_id}.png"
            copy_image_as_png(Path(sample["image_src"]), dst_image)

            instructions_data[group_id][new_episode_id] = sample["variants"][variant_name]
            steps_data[group_id][new_episode_id] = sample["steps"]
            meta_data[group_id][new_episode_id] = {
                "source_group_id": sample["source_group_id"],
                "source_episode_id": sample["source_episode_id"],
                "source_instruction": sample["source_instruction"],
                "start_gid": sample["start_gid"],
                "variant_name": variant_name,
                "source_image_path": sample["image_src"],
                "image_path": str(dst_image.relative_to(root_dir)),
                "output_mode": OUTPUT_MODE,
            }

            created += 1

        # 一个 source pair 一旦生成完所有 variant，就整体记为已完成
        existing_pairs.add(pair)

    # 7. 排序并保存
    instructions_data = sort_nested_dict(instructions_data)
    steps_data = sort_nested_dict(steps_data)
    meta_data = sort_nested_dict(meta_data)

    out_instruction_path.write_text(
        json.dumps(instructions_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_steps_path.write_text(
        json.dumps(steps_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_meta_path.write_text(
        json.dumps(meta_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total = sum(len(v) for v in instructions_data.values())
    print(f"created: {created}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()