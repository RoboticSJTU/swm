import json
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from swm.utils.apis import call_gpt_json


SOURCE_TASK_DOMAIN = "agibot"
AUG_TASK_DOMAIN = "agibot_aug_v1_2"

GEN_MODEL_NAME = "Qwen3.5-397B-A17B"

MAX_WORKERS = 50
MAX_REWRITE_ATTEMPTS = 3


def clean_text(x) -> str:
    return re.sub(r"\s+", " ", str(x).strip().strip('"').strip("'"))


def natural_key(x: str):
    parts = re.split(r"(\d+)", str(x))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_items(path: Path):
    """
    仅支持两种 source instruction JSON 格式：

    格式1：多组嵌套
    {
      "task_327": {
        "episode_648642": "xxx",
        "episode_685201": "xxx"
      },
      "task_351": {
        "episode_776577": "xxx"
      }
    }

    格式2：单个 task_domain 包住所有 episode
    {
      "human": {
        "episode_1": "xxx",
        "episode_2": "xxx"
      }
    }

    返回：
    - items: [{"group_id": ..., "episode_id": ..., "instruction": ...}, ...]
    - flat_mode: True 表示路径按 domain/episode 组织；False 表示按 domain/group/episode 组织
    """
    data = load_json(path, {})

    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid source json: {path}")

    # 格式2：顶层只有一个 key，且这个 key 就是 SOURCE_TASK_DOMAIN，
    # 下面直接是 episode -> instruction
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

    # 格式1：多组嵌套，顶层每个 key 下面都是 episode -> instruction
    items = []
    for group_id, episode_map in sorted(data.items(), key=lambda x: natural_key(x[0])):
        if not isinstance(episode_map, dict):
            raise ValueError(
                f"Unsupported source json format: top-level value of '{group_id}' must be a dict"
            )

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
    """
    读取已经生成过的 instructions / steps / meta。
    不限制 key 名字，只要求是两层 dict。
    """
    data = load_json(path, {})

    if not isinstance(data, dict):
        return {}

    out = {}
    for group_id, episode_map in data.items():
        if isinstance(episode_map, dict):
            out[str(group_id)] = dict(episode_map)
    return out


def sort_nested(data: dict):
    return {
        group_id: dict(sorted(episode_map.items(), key=lambda x: natural_key(x[0])))
        for group_id, episode_map in sorted(data.items(), key=lambda x: natural_key(x[0]))
    }


def next_episode_index(episode_map: dict) -> int:
    max_idx = 0
    for episode_id in episode_map:
        m = re.search(r"(\d+)$", str(episode_id))
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def load_group_actions(group_path: Path):
    group_to_actions = {}
    for line in group_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\[G(\d+)\]\s*(.+?)\s*$", line.strip())
        if m:
            gid = int(m.group(1))
            group_to_actions.setdefault(gid, []).append(clean_text(m.group(2)))
    return group_to_actions


def list_seg_dirs(keyframe_dir: Path):
    seg_dirs = []
    for p in keyframe_dir.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"seg_(\d+)", p.name)
            if m:
                seg_dirs.append((int(m.group(1)), p))
    seg_dirs.sort(key=lambda x: x[0])
    return seg_dirs


def first_image_in_dir(seg_dir: Path):
    imgs = sorted(
        [
            p for p in seg_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ],
        key=lambda p: natural_key(p.name)
    )
    return imgs[0] if imgs else None


def build_jobs(root_dir: Path, item: dict, flat_mode: bool, existing_pairs: set):
    group_id = item["group_id"]
    episode_id = item["episode_id"]
    instruction = item["instruction"]

    if flat_mode:
        keyframe_dir = root_dir / "dataset" / "keyframes" / SOURCE_TASK_DOMAIN / episode_id
        group_path = (
            root_dir
            / "eval_results"
            / "gemini-3-flash-preview"
            / SOURCE_TASK_DOMAIN
            / episode_id
            / "kf_plan_group.txt"
        )
    else:
        keyframe_dir = root_dir / "dataset" / "keyframes" / SOURCE_TASK_DOMAIN / group_id / episode_id
        group_path = (
            root_dir
            / "eval_results"
            / "gemini-3-flash-preview"
            / SOURCE_TASK_DOMAIN
            / group_id
            / episode_id
            / "kf_plan_group.txt"
        )

    if not keyframe_dir.is_dir() or not group_path.is_file():
        return []

    group_to_actions = load_group_actions(group_path)
    if not group_to_actions:
        return []

    jobs = []
    group_ids = sorted(group_to_actions)

    for start_gid, seg_dir in list_seg_dirs(keyframe_dir):
        pair = (group_id, episode_id, start_gid)
        if start_gid not in group_to_actions or pair in existing_pairs:
            continue

        image_path = first_image_in_dir(seg_dir)
        if image_path is None:
            continue

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


def extract_concise(gen: dict) -> str:
    if isinstance(gen, dict) and isinstance(gen.get("variants"), dict):
        if gen["variants"].get("concise"):
            return clean_text(gen["variants"]["concise"])
    if isinstance(gen, dict) and gen.get("concise"):
        return clean_text(gen["concise"])
    return ""


def rewrite_one_group(job: dict):
    concise = ""
    attempt_history = []

    for attempt in range(1, MAX_REWRITE_ATTEMPTS + 1):
        try:
            gen = call_gpt_json(
                GEN_MODEL_NAME,
                build_generation_prompt(job["instruction"], job["completed"], job["remaining"]),
                [job["image_path"]],
            )
            concise = extract_concise(gen)
        except Exception as e:
            concise = ""
            attempt_history.append({
                "attempt": attempt,
                "concise_text": "",
                "error": f"generation_error: {clean_text(e)}",
            })
            continue

        if concise:
            return {
                "status": "passed",
                "source_group_id": job["group_id"],
                "source_episode_id": job["episode_id"],
                "source_instruction": job["instruction"],
                "start_gid": job["start_gid"],
                "image_src": str(job["image_path"]),
                "steps": job["remaining"],
                "concise": concise,
                "attempts_used": attempt,
            }

        attempt_history.append({
            "attempt": attempt,
            "concise_text": "",
            "error": "generation returned empty concise instruction",
        })

    return {
        "status": "failed",
        "source_group_id": job["group_id"],
        "source_episode_id": job["episode_id"],
        "source_instruction": job["instruction"],
        "start_gid": job["start_gid"],
        "image_src": str(job["image_path"]),
        "steps": job["remaining"],
        "concise": concise,
        "attempts_used": MAX_REWRITE_ATTEMPTS,
        "attempt_history": attempt_history,
    }


def copy_image_as_png(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
    else:
        Image.open(src).save(dst)


def main():
    root_dir = Path(__file__).resolve().parent.parent

    src_instruction_path = root_dir / "tasks" / "instructions" / f"instructions_{SOURCE_TASK_DOMAIN}.json"

    out_instruction_path = root_dir / "tasks" / "instructions" / f"instructions_{AUG_TASK_DOMAIN}.json"
    out_steps_path = root_dir / "tasks" / "steps" / f"steps_{AUG_TASK_DOMAIN}.json"
    out_meta_path = root_dir / "tasks" / "meta" / f"meta_{AUG_TASK_DOMAIN}.json"
    out_failed_path = root_dir / "tasks" / "meta" / f"meta_{AUG_TASK_DOMAIN}_failed.json"
    out_image_root = root_dir / "tasks" / "images" / AUG_TASK_DOMAIN

    out_instruction_path.parent.mkdir(parents=True, exist_ok=True)
    out_steps_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)
    out_image_root.mkdir(parents=True, exist_ok=True)

    if not src_instruction_path.is_file():
        raise FileNotFoundError(src_instruction_path)

    source_items, flat_mode = load_source_items(src_instruction_path)

    instructions_data = load_saved_nested_json(out_instruction_path)
    steps_data = load_saved_nested_json(out_steps_path)
    meta_data = load_saved_nested_json(out_meta_path)

    # 读取已经生成过的 pair，避免重复生成
    existing_pairs = set()
    for _, episode_map in meta_data.items():
        for _, meta in episode_map.items():
            if not isinstance(meta, dict):
                continue

            if all(k in meta for k in ["source_group_id", "source_episode_id", "start_gid"]):
                existing_pairs.add((
                    meta["source_group_id"],
                    meta["source_episode_id"],
                    int(meta["start_gid"]),
                ))
            elif all(k in meta for k in ["source_task_id", "source_episode_id", "start_gid"]):
                # 兼容旧版 meta
                existing_pairs.add((
                    meta["source_task_id"],
                    meta["source_episode_id"],
                    int(meta["start_gid"]),
                ))

    jobs = []
    for item in source_items:
        jobs.extend(build_jobs(root_dir, item, flat_mode, existing_pairs))

    jobs.sort(key=lambda x: (
        natural_key(x["group_id"]),
        natural_key(x["episode_id"]),
        x["start_gid"],
    ))
    print(f"jobs: {len(jobs)}")

    all_samples = []
    if jobs:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as ex:
            futures = [ex.submit(rewrite_one_group, job) for job in jobs]
            total = len(futures)

            for idx, future in enumerate(as_completed(futures), 1):
                sample = future.result()
                all_samples.append(sample)
                print(
                    f'[{idx}/{total}] {sample["status"].upper()} '
                    f'{sample["source_group_id"]}/{sample["source_episode_id"]}/G{sample["start_gid"]} '
                    f'attempts={sample["attempts_used"]}'
                )

    all_samples.sort(key=lambda x: (
        natural_key(x["source_group_id"]),
        natural_key(x["source_episode_id"]),
        x["start_gid"],
    ))

    next_episode_idx = {
        group_id: next_episode_index(episode_map)
        for group_id, episode_map in instructions_data.items()
    }

    created = 0
    failed_samples = []

    for sample in all_samples:
        if sample["status"] != "passed":
            failed_samples.append({
                "source_group_id": sample["source_group_id"],
                "source_episode_id": sample["source_episode_id"],
                "start_gid": sample["start_gid"],
                "source_image_path": sample["image_src"],
                "concise_text": sample["concise"],
                "attempts_used": sample["attempts_used"],
                "attempt_history": sample["attempt_history"],
            })
            continue

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

        new_episode_id = f"episode_{next_episode_idx[group_id]}"
        next_episode_idx[group_id] += 1

        dst_image = out_image_root / group_id / f"{new_episode_id}.png"
        copy_image_as_png(Path(sample["image_src"]), dst_image)

        instructions_data[group_id][new_episode_id] = sample["concise"]
        steps_data[group_id][new_episode_id] = sample["steps"]
        meta_data[group_id][new_episode_id] = {
            "source_group_id": sample["source_group_id"],
            "source_episode_id": sample["source_episode_id"],
            "source_instruction": sample["source_instruction"],
            "start_gid": sample["start_gid"],
            "source_image_path": sample["image_src"],
            "image_path": str(dst_image.relative_to(root_dir)),
            "concise_text": sample["concise"],
            "attempts_used": sample["attempts_used"],
        }

        existing_pairs.add(pair)
        created += 1

    instructions_data = sort_nested(instructions_data)
    steps_data = sort_nested(steps_data)
    meta_data = sort_nested(meta_data)

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
    out_failed_path.write_text(
        json.dumps(failed_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total = sum(len(v) for v in instructions_data.values())
    print(f"created: {created}")
    print(f"failed: {len(failed_samples)}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()