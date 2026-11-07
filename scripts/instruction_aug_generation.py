#!/usr/bin/env python3
"""Import reviewed instructions for keyframe suffixes, entirely offline.

Every selected segment-start image receives exactly AUG_FACTOR instructions.
Edit instruction_aug_manual.json after reviewing source actions and images; a
changed action file requires that review before any dataset files are written.
For explicit, metadata-free suffix imports use import_reviewed_suffixes instead
of the legacy three-variant main entry point.
"""

import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

SOURCE_TASK_DOMAIN = "human"
AUG_TASK_DOMAIN = f"{SOURCE_TASK_DOMAIN}_aug"
PLAN_MODEL_NAME = "gpt-5.6-sol"
AUG_TASK_IDS = list(range(301, 321))
SKIP_TASK_IDS = set()  # Temporarily exclude tasks only while their source data is under review.
AUG_FACTOR = 3
MIN_START_GID = 1  # Retain the existing suffix scope, including a sparse first G1.
ALLOW_EXTERNAL_API = False
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ROOT = Path(__file__).resolve().parents[1]
MANUAL_PATH = Path(__file__).with_name("instruction_aug_manual.json")
BANNED_CONTEXT = re.compile(
    r"\b(?:image|photo|scene|camera|frame|robot|left hand|right hand|both hands)\b"
    r"|\b(?:remaining steps|completed actions)\b", re.IGNORECASE,
)


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(text))]


def normalize_instruction(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_indexed_actions(path):
    """Preserve original GIDs and all actions within each group, including gaps."""
    actions = {}
    previous = -1
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"\[G(\d+)\]\s+(.+?)\s*", line.strip())
        if not match or int(match[1]) < previous:
            raise ValueError(f"{path}:{number}: invalid or unordered indexed action")
        if re.search(r"\|\s*(?:Before|After)\s*:", match[2], re.IGNORECASE):
            raise ValueError(f"{path}:{number}: use a plain action without Before/After annotations")
        previous = int(match[1])
        actions.setdefault(previous, []).append(" ".join(match[2].split()))
    if not actions:
        raise ValueError(f"{path}: no indexed actions")
    return actions


def write_json_atomic(path, data):
    if path.is_file() and read_json(path) == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def ensure_image_link(root, meta):
    source = Path(meta["source_image_path"])
    output = root / meta["image_path"]
    if output.is_symlink():
        if output.resolve() == source.resolve():
            return
        output.unlink()
    elif output.exists():
        raise FileExistsError(f"refusing to replace non-symlink image: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.symlink_to(os.path.relpath(source, output.parent))


def existing_slots(episodes):
    """Use source GID and variant slot as identity, never its ordinal position."""
    slots = {}
    for episode, meta in episodes.items():
        key = (meta["source_episode_id"], meta.get("branch", "full"),
               int(meta["cut_index"]), int(meta.get("slot", 1)))
        if key in slots:
            raise ValueError(f"duplicate source/slot identity: {key}")
        slots[key] = episode
    return slots


def prepare_records(root, source_data, manual_data, instruction_data, steps_data, meta_data):
    """Validate every input and construct all replacements before changing files."""
    records = []
    for task_id in AUG_TASK_IDS:
        if task_id in SKIP_TASK_IDS:
            continue
        group = f"task_{task_id}"
        authored = manual_data[group]
        if authored.get("review_status", "reviewed") != "reviewed":
            raise ValueError(f"{group}: source data review is incomplete")
        source_episode = authored["source_episode_id"]
        source_instruction = source_data[group][source_episode]
        action_path = (root / "eval_results" / PLAN_MODEL_NAME / SOURCE_TASK_DOMAIN
                       / group / source_episode / "kf_actions.txt")
        actions = read_indexed_actions(action_path)
        if {str(gid): values for gid, values in actions.items()} != authored["reviewed_actions"]:
            raise ValueError(f"{group}: kf_actions changed; review local instructions first")
        starts = [gid for gid in actions if gid >= MIN_START_GID]
        if set(authored["instructions"]) != {str(gid) for gid in starts}:
            raise ValueError(f"{group}: authored GIDs do not cover the current suffixes")
        old_episodes = instruction_data.get(group, {})
        if set(old_episodes) != set(steps_data.get(group, {})):
            raise ValueError(f"{group}: existing instruction/step episode IDs disagree")
        slots = existing_slots(meta_data.get(group, {}))
        names = set(old_episodes) | set(steps_data.get(group, {})) | set(meta_data.get(group, {}))
        image_dir = root / "tasks/images" / AUG_TASK_DOMAIN / group
        names.update(path.stem for path in image_dir.glob("episode_*.*"))
        results_dir = root / "eval_results" / PLAN_MODEL_NAME / AUG_TASK_DOMAIN / group
        names.update(path.name for path in results_dir.glob("episode_*"))
        next_id = max((int(match[1]) for name in names
                       if (match := re.fullmatch(r"episode_(\d+)", name))), default=0) + 1
        assigned = set()
        for gid in starts:
            variants = authored["instructions"][str(gid)]
            if not isinstance(variants, list) or len(variants) != AUG_FACTOR:
                raise ValueError(f"{group}/G{gid}: expected {AUG_FACTOR} authored instructions")
            segment = (root / "dataset/keyframes" / SOURCE_TASK_DOMAIN / group
                       / source_episode / f"seg_{gid:02d}")
            images = sorted((p for p in segment.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                            key=lambda p: natural_key(p.name))
            if not images:
                raise ValueError(f"{segment}: no images")
            source_image = images[0].resolve(strict=True)
            target = [a for g, values in actions.items() if g >= gid for a in values]
            completed = [a for g, values in actions.items() if g < gid for a in values]
            for slot, instruction in enumerate(variants, 1):
                if (not isinstance(instruction, str) or len(instruction.split()) < 3
                        or BANNED_CONTEXT.search(instruction)):
                    raise ValueError(f"{group}/G{gid}/{slot}: invalid instruction")
                key = (source_episode, "full", gid, slot)
                episode = slots.get(key)
                if episode is None:
                    episode = f"episode_{next_id}"
                    next_id += 1
                assigned.add(episode)
                output_image = image_dir / f"{episode}.png"
                if output_image.exists() and not output_image.is_symlink():
                    raise FileExistsError(f"refusing to replace non-symlink image: {output_image}")
                meta = {
                    "source_group_id": group,
                    "source_episode_id": source_episode,
                    "source_instruction": source_instruction,
                    "branch": "full",
                    "semantic_boundary_gid": None,
                    "branch_start_gid": min(actions),
                    "local_cut_index": list(actions).index(gid),
                    "clip_start_gid": gid,
                    "clip_end_gid": max(actions),
                    "cut_index": gid,
                    "slot": slot,
                    "augmentation_factor": AUG_FACTOR,
                    "source_segment_id": segment.name,
                    "source_frame_index": int(source_image.stem),
                    "generation_source": "assistant_authored_local",
                    "external_api_calls": 0,
                    "source_image_path": str(source_image),
                    "image_path": str(output_image.relative_to(root)),
                    "steps": target,
                    "completed_actions": completed,
                    "excluded_actions": [],
                    "aug_version": "offline_three_variants_v7",
                }
                records.append({"group_id": group, "episode_id": episode,
                                "instruction": instruction, "steps": target, "meta": meta})
        unmanaged = (set(old_episodes) | set(meta_data.get(group, {}))) - assigned
        if unmanaged:
            raise ValueError(f"{group}: existing episodes need explicit review: {sorted(unmanaged)}")
    return records


def audit_records(source_data, records, manual_data):
    """Check all texts together per task, and count variants by resolved source image."""
    tasks = {}
    for task_id in AUG_TASK_IDS:
        if task_id in SKIP_TASK_IDS:
            continue
        group = f"task_{task_id}"
        selected = [record for record in records if record["group_id"] == group]
        texts = [(f"original/{episode}", text) for episode, text in source_data[group].items()]
        texts.extend((f"aug/{r['episode_id']}", r["instruction"]) for r in selected)
        normalized = [normalize_instruction(text) for _, text in texts]
        duplicates = [text for text, count in Counter(normalized).items() if count > 1]
        if duplicates:
            raise ValueError(f"{group}: original/aug duplicate instructions: {duplicates}")
        image_counts = Counter(r["meta"]["source_image_path"] for r in selected)
        if not image_counts or set(image_counts.values()) != {AUG_FACTOR}:
            raise ValueError(f"{group}: expected {AUG_FACTOR} variants per source image")
        near = []
        for index, (episode, text) in enumerate(texts):
            for other_episode, other_text in texts[index + 1:]:
                ratio = SequenceMatcher(None, normalize_instruction(text),
                                        normalize_instruction(other_text)).ratio()
                if ratio >= 0.9:
                    near.append({"episodes": [episode, other_episode], "similarity": round(ratio, 4)})
        tasks[group] = {
            "original_episodes": len(source_data[group]),
            "augmented_episodes": len(selected),
            "source_images": len(image_counts),
            "variants_per_image": AUG_FACTOR,
            "normalized_duplicate_count": 0,
            "similar_pairs_at_0_90": near,
            "review_method": "assistant_self_review_of_images_actions_and_each_instruction",
            "independent_review": False,
            "review_notes": manual_data[group]["review_notes"],
            "reviewed_on": manual_data[group].get("reviewed_on"),
            "reviewed_contact_sheet": manual_data[group].get("reviewed_contact_sheet"),
            "original_instruction_corrected": "original_instruction" in manual_data[group],
            "reviewed_source_instructions": source_data[group],
            "reviewed_images": [
                {"source_image_path": path,
                 "start_gid": next(r["meta"]["cut_index"] for r in selected
                                   if r["meta"]["source_image_path"] == path)}
                for path in image_counts
            ],
        }
    return {"external_api_calls": 0, "augmentation_factor": AUG_FACTOR,
            "skipped_tasks": {f"task_{task_id}": "Skipped at user request: source data errors"
                              for task_id in sorted(SKIP_TASK_IDS)},
            "source_images": sum(t["source_images"] for t in tasks.values()),
            "augmented_episodes": len(records), "tasks": tasks}


def suffix_augmentation_factor(completed_groups, total_groups):
    """Use ordinal KF-group progress, including sparse GIDs and grouped actions."""
    if not 0 < completed_groups < total_groups:
        raise ValueError("a suffix must exclude the full-task start and empty end")
    if 2 * completed_groups < total_groups:
        return 10
    if 4 * completed_groups < 3 * total_groups:
        return 5
    return 2


def import_reviewed_suffixes(root, reviewed_data, task_ids=range(321, 331)):
    """Append explicitly reviewed suffixes without reading or writing metadata.

    Each task supplies reviewed_actions and cuts. A cut names start_gid, an
    existing source_image_path, and zero or more reviewed instructions. Optional
    episode_ids reserve explicit new IDs; existing samples are never replaced.
    The returned records and cut shortfalls are in-memory review/PDDL inputs.
    All inputs are checked before any dataset file is changed.
    """
    root = Path(root).resolve()
    source_path = root / "tasks/instructions" / f"instructions_{SOURCE_TASK_DOMAIN}.json"
    instruction_path = root / "tasks/instructions" / f"instructions_{AUG_TASK_DOMAIN}.json"
    steps_path = root / "tasks/steps" / f"steps_{AUG_TASK_DOMAIN}.json"
    source_data = read_json(source_path)
    instruction_data = read_json(instruction_path)
    steps_data = read_json(steps_path)
    seen = {}
    for label, data in (("original", source_data), ("existing_aug", instruction_data)):
        for group, episodes in data.items():
            for episode, instruction in episodes.items():
                seen.setdefault(normalize_instruction(instruction),
                                f"{label}/{group}/{episode}")
    records, cuts = [], []
    task_ids = list(task_ids)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task IDs")
    for task_id in task_ids:
        group = f"task_{task_id}"
        authored = reviewed_data[group]
        if authored.get("review_status", "reviewed") != "reviewed":
            raise ValueError(f"{group}: source data review is incomplete")
        source_episode = authored.get("source_episode_id", f"episode_{task_id}")
        if source_episode not in source_data[group]:
            raise ValueError(f"{group}: unknown source episode {source_episode}")
        action_path = (root / "eval_results" / PLAN_MODEL_NAME / SOURCE_TASK_DOMAIN
                       / group / source_episode / "kf_actions.txt")
        actions = read_indexed_actions(action_path)
        if {str(gid): values for gid, values in actions.items()} != authored["reviewed_actions"]:
            raise ValueError(f"{group}: kf_actions changed; review local instructions first")
        gids = list(actions)
        old_instructions = instruction_data.get(group, {})
        old_steps = steps_data.get(group, {})
        if set(old_instructions) != set(old_steps):
            raise ValueError(f"{group}: existing instruction/step episode IDs disagree")
        image_dir = root / "tasks/images" / AUG_TASK_DOMAIN / group
        results_dir = root / "eval_results" / PLAN_MODEL_NAME / AUG_TASK_DOMAIN / group
        names = set(old_instructions) | set(old_steps)
        names.update(path.stem for path in image_dir.glob("episode_*.*"))
        names.update(path.name for path in results_dir.glob("episode_*"))
        next_id = max((int(match[1]) for name in names
                       if (match := re.fullmatch(r"episode_(\d+)", name))), default=0) + 1
        source_dir = (root / "dataset/keyframes" / SOURCE_TASK_DOMAIN
                      / group / source_episode).resolve()
        selected_gids, selected_images = set(), set()
        for cut in authored["cuts"]:
            gid = int(cut["start_gid"])
            if gid not in actions or gid in selected_gids:
                raise ValueError(f"{group}/G{gid}: unknown or duplicate cut GID")
            selected_gids.add(gid)
            completed_groups = gids.index(gid)
            target_count = suffix_augmentation_factor(completed_groups, len(gids))
            source_image = Path(cut["source_image_path"])
            if not source_image.is_absolute():
                source_image = root / source_image
            source_image = source_image.resolve(strict=True)
            if (not source_image.is_file() or source_image.suffix.lower() not in IMAGE_EXTS
                    or source_image.parent.parent != source_dir
                    or not re.fullmatch(r"seg_\d+", source_image.parent.name)):
                raise ValueError(f"{group}/G{gid}: image is not an existing source keyframe")
            if source_image in selected_images:
                raise ValueError(f"{group}/G{gid}: source image reused by another cut")
            selected_images.add(source_image)
            variants = cut["instructions"]
            if not isinstance(variants, list) or len(variants) > target_count:
                raise ValueError(f"{group}/G{gid}: expected at most {target_count} instructions")
            explicit_ids = cut.get("episode_ids")
            if explicit_ids is not None and (not isinstance(explicit_ids, list)
                                             or len(explicit_ids) != len(variants)):
                raise ValueError(f"{group}/G{gid}: episode_ids must match instructions")
            target = [a for g in gids[completed_groups:] for a in actions[g]]
            indexed = [f"[G{g}] {a}" for g in gids[completed_groups:] for a in actions[g]]
            completed = [a for g in gids[:completed_groups] for a in actions[g]]
            shortfall = target_count - len(variants)
            cut_result = {"group_id": group, "source_episode_id": source_episode,
                          "start_gid": gid, "source_image_path": str(source_image),
                          "completed_groups": completed_groups, "total_groups": len(gids),
                          "target_count": target_count, "actual_count": len(variants),
                          "shortfall": shortfall, "episode_ids": []}
            cuts.append(cut_result)
            for slot, instruction in enumerate(variants):
                if (not isinstance(instruction, str) or len(instruction.split()) < 3
                        or "\n" in instruction or BANNED_CONTEXT.search(instruction)):
                    raise ValueError(f"{group}/G{gid}/{slot + 1}: invalid instruction")
                normalized = normalize_instruction(instruction)
                if normalized in seen:
                    raise ValueError(f"{group}/G{gid}/{slot + 1}: duplicate instruction with "
                                     f"{seen[normalized]}")
                seen[normalized] = f"new/{group}/G{gid}/{slot + 1}"
                episode = explicit_ids[slot] if explicit_ids is not None else f"episode_{next_id}"
                if (not isinstance(episode, str)
                        or not re.fullmatch(r"episode_[1-9]\d*", episode)):
                    raise ValueError(f"{group}/G{gid}: invalid episode ID {episode}")
                if episode in names:
                    raise FileExistsError(f"{group}/{episode}: episode ID collision")
                names.add(episode)
                next_id = max(next_id, int(episode.split("_")[1]) + 1)
                image_path = image_dir / f"{episode}.png"
                records.append({**cut_result, "episode_id": episode, "slot": slot + 1,
                                "instruction": instruction, "steps": target,
                                "indexed_actions": indexed, "completed_actions": completed,
                                "image_path": str(image_path.relative_to(root))})
                cut_result["episode_ids"].append(episode)
        if selected_gids != set(gids[1:]):
            raise ValueError(f"{group}: reviewed cuts must cover every noninitial KF group; "
                             "use an empty instruction list for a rejected cut")
    # Validation and all collision checks finish before the first write.
    for record in records:
        ensure_image_link(root, record)
        group, episode = record["group_id"], record["episode_id"]
        directory = root / "eval_results" / PLAN_MODEL_NAME / AUG_TASK_DOMAIN / group / episode
        directory.mkdir(parents=True)
        (directory / "kf_actions.txt").write_text(
            "\n".join(record["indexed_actions"]) + "\n", encoding="utf-8")
        instruction_data.setdefault(group, {})[episode] = record["instruction"]
        steps_data.setdefault(group, {})[episode] = record["steps"]
    write_json_atomic(instruction_path, instruction_data)
    write_json_atomic(steps_path, steps_data)
    return {"records": records, "cuts": cuts, "normalized_duplicate_count": 0,
            "augmented_episodes": len(records), "source_images": len(cuts),
            "shortfall": sum(cut["shortfall"] for cut in cuts), "external_api_calls": 0}


def main(root=ROOT, manual_path=MANUAL_PATH):
    if ALLOW_EXTERNAL_API:
        raise ValueError("Only offline, reviewed instructions are supported")
    root = Path(root).resolve()
    source_path = root / "tasks/instructions" / f"instructions_{SOURCE_TASK_DOMAIN}.json"
    instruction_path = root / "tasks/instructions" / f"instructions_{AUG_TASK_DOMAIN}.json"
    steps_path = root / "tasks/steps" / f"steps_{AUG_TASK_DOMAIN}.json"
    meta_path = root / "tasks/metadata" / f"meta_{AUG_TASK_DOMAIN}.json"
    review_path = root / "tasks/metadata/instruction_aug_review_301_320.json"
    source_data = read_json(source_path)
    instruction_data = read_json(instruction_path)
    steps_data = read_json(steps_path)
    meta_data = read_json(meta_path)
    manual_data = read_json(Path(manual_path))
    for task_id in AUG_TASK_IDS:
        if task_id in SKIP_TASK_IDS:
            continue
        group = f"task_{task_id}"
        item = manual_data[group]
        if "original_instruction" in item:
            source_data[group][item["source_episode_id"]] = item["original_instruction"]
    records = prepare_records(root, source_data, manual_data, instruction_data, steps_data, meta_data)
    report = audit_records(source_data, records, manual_data)
    for task_id in AUG_TASK_IDS:
        if task_id in SKIP_TASK_IDS:
            continue
        group = f"task_{task_id}"
        selected = [record for record in records if record["group_id"] == group]
        instruction_data[group] = {r["episode_id"]: r["instruction"] for r in selected}
        steps_data[group] = {r["episode_id"]: r["steps"] for r in selected}
        meta_data[group] = {r["episode_id"]: r["meta"] for r in selected}
        task_report = report["tasks"][group]
        print(f"{group}: images={task_report['source_images']} instructions={len(selected)} "
              f"duplicates=0 near_pairs={len(task_report['similar_pairs_at_0_90'])}")
    # All source, coverage, duplicate, and image collision checks have passed.
    for record in records:
        ensure_image_link(root, record["meta"])
    for path, data in [(source_path, source_data), (instruction_path, instruction_data),
                       (steps_path, steps_data), (meta_path, meta_data), (review_path, report)]:
        write_json_atomic(path, data)
    print(f"Offline augmentation complete: {report['source_images']} images, {len(records)} instructions")
    print(f"Review report: {review_path}")
    return report


if __name__ == "__main__":
    main()
