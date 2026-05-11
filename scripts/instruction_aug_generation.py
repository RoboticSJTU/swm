#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from swm.utils.apis import call_gpt_json


# ============================================================
# Config
# ============================================================

SOURCE_TASK_DOMAIN = "human"
AUG_TASK_DOMAIN = f"{SOURCE_TASK_DOMAIN}_aug_v6"

GEN_MODEL_NAME = "gemini-3-flash-preview"
CALL_GPT_MODEL = "gpt-5.2"

MAX_WORKERS = 300
MAX_RETRY = 5
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


# ============================================================
# Basic helpers
# ============================================================

def clean(text):
    return re.sub(r"\s+", " ", str(text).strip().strip('"').strip("'"))


def norm(text):
    text = clean(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(the|a|an|please|could|would|can|just)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def nkey(text):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(text))]


def read_json(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_root():
    here = Path(__file__).resolve().parent
    if (here / "tasks").is_dir() and (here / "dataset").is_dir():
        return here
    if (here.parent / "tasks").is_dir() and (here.parent / "dataset").is_dir():
        return here.parent
    return here


def format_actions(actions):
    if not actions:
        return "- none"
    return "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))


def sort_nested(data):
    sorted_data = {}

    for group_id, episode_map in sorted(data.items(), key=lambda x: nkey(x[0])):
        sorted_data[group_id] = dict(sorted(episode_map.items(), key=lambda x: nkey(x[0])))

    return sorted_data


# ============================================================
# Data loading
# ============================================================

def load_source_items(path):
    data = read_json(path)

    flat_mode = (
        len(data) == 1
        and SOURCE_TASK_DOMAIN in data
        and isinstance(data[SOURCE_TASK_DOMAIN], dict)
        and all(not isinstance(v, dict) for v in data[SOURCE_TASK_DOMAIN].values())
    )

    items = []

    if flat_mode:
        for episode_id, instruction in sorted(data[SOURCE_TASK_DOMAIN].items(), key=lambda x: nkey(x[0])):
            instruction = clean(instruction)
            if instruction:
                items.append((SOURCE_TASK_DOMAIN, str(episode_id), instruction))
        return items, flat_mode

    for group_id, episode_map in sorted(data.items(), key=lambda x: nkey(x[0])):
        for episode_id, instruction in sorted(episode_map.items(), key=lambda x: nkey(x[0])):
            instruction = clean(instruction)
            if instruction:
                items.append((str(group_id), str(episode_id), instruction))

    return items, flat_mode


def read_group_actions(path):
    actions = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\[G(\d+)\]\s*(.+?)\s*$", line.strip())
        if not m:
            continue

        gid = int(m.group(1))
        action = clean(m.group(2))

        if gid not in actions:
            actions[gid] = []
        actions[gid].append(action)

    return actions


def load_existing_progress(meta_data, instruction_data):
    count_by_pair = {}
    texts_by_pair = {}

    for group_id in meta_data:
        if not isinstance(meta_data[group_id], dict):
            continue

        for episode_id in meta_data[group_id]:
            meta = meta_data[group_id][episode_id]
            if not isinstance(meta, dict):
                continue

            if "source_group_id" in meta:
                source_group_id = meta["source_group_id"]
            elif "source_task_id" in meta:
                source_group_id = meta["source_task_id"]
            else:
                continue

            if "source_episode_id" not in meta or "start_gid" not in meta:
                continue

            pair = (
                str(source_group_id),
                str(meta["source_episode_id"]),
                int(meta["start_gid"]),
            )

            if pair not in count_by_pair:
                count_by_pair[pair] = 0
            count_by_pair[pair] += 1

            if group_id in instruction_data and episode_id in instruction_data[group_id]:
                text = clean(instruction_data[group_id][episode_id])
                if text:
                    if pair not in texts_by_pair:
                        texts_by_pair[pair] = []
                    texts_by_pair[pair].append(text)

    return count_by_pair, texts_by_pair


# ============================================================
# Job collection
# ============================================================

def collect_jobs(root, source_items, flat_mode, count_by_pair):
    jobs = []

    for group_id, episode_id, instruction in source_items:
        if flat_mode:
            keyframe_dir = root / "dataset" / "keyframes" / SOURCE_TASK_DOMAIN / episode_id
            group_file = root / "eval_results" / GEN_MODEL_NAME / SOURCE_TASK_DOMAIN / episode_id / "kf_plan_group.txt"
        else:
            keyframe_dir = root / "dataset" / "keyframes" / SOURCE_TASK_DOMAIN / group_id / episode_id
            group_file = root / "eval_results" / GEN_MODEL_NAME / SOURCE_TASK_DOMAIN / group_id / episode_id / "kf_plan_group.txt"

        if not keyframe_dir.is_dir() or not group_file.is_file():
            continue

        group_actions = read_group_actions(group_file)
        if not group_actions:
            continue

        group_ids = sorted(group_actions)

        seg_dirs = []
        for path in keyframe_dir.iterdir():
            m = re.fullmatch(r"seg_(\d+)", path.name)
            if path.is_dir() and m:
                seg_dirs.append((int(m.group(1)), path))
        seg_dirs.sort(key=lambda x: x[0])

        for start_gid, seg_dir in seg_dirs:
            if start_gid not in group_actions:
                continue

            images = [p for p in seg_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            images.sort(key=lambda p: nkey(p.name))

            if not images:
                continue

            completed = []
            remaining = []

            for gid in group_ids:
                if gid < start_gid:
                    completed.extend(group_actions[gid])
                else:
                    remaining.extend(group_actions[gid])

            if not remaining:
                continue

            pair = (group_id, episode_id, start_gid)
            existing = count_by_pair[pair] if pair in count_by_pair else 0
            target = len(remaining)
            need = target - existing

            if need <= 0:
                continue

            jobs.append({
                "group_id": group_id,
                "episode_id": episode_id,
                "instruction": instruction,
                "start_gid": start_gid,
                "image_path": images[0],
                "completed": completed,
                "remaining": remaining,
                "target": target,
                "existing": existing,
                "need": need,
            })

    jobs.sort(key=lambda x: (nkey(x["group_id"]), nkey(x["episode_id"]), x["start_gid"]))
    return jobs


# ============================================================
# GPT generation
# ============================================================

def build_prompt(job, existing_texts, last_error):
    styles = []
    for i in range(job["need"]):
        style = STYLE_GUIDES[(job["existing"] + i) % len(STYLE_GUIDES)]
        styles.append(f"{i + 1}. slot={i + 1}: {style}")

    if existing_texts:
        existing_block = "\n".join(f"- {x}" for x in existing_texts)
    else:
        existing_block = "none"

    retry_block = ""
    if last_error:
        retry_block = f"""
Previous output problem:
{last_error}
Regenerate the full JSON and fix it.
"""

    return f"""You are generating augmented robot task instructions for the SAME image and the SAME remaining task.

The dataset trains a vision-language robot planner.
The language instruction should describe the high-level goal.
The image should preserve hidden executable preconditions.
Do not turn the instruction into a step-by-step robot plan.

Core rule:
- Keep WHAT final state should be achieved.
- Remove HOW the robot should make the task executable.
- Do not leak implicit constraints that are visible in the image.

Usually remove:
- source locations,
- current supports,
- blockers or occluders,
- intermediate tools,
- hidden state activations,
- low-level pick/lift/grab/take/remove steps,
- action ordering caused only by executability.

Usually keep:
- final target objects,
- final destination or container,
- final spatial relation,
- explicit final states such as closed, locked, returned, turned off, or placed,
- tools or objects only if they are part of the final requested result.

Examples:
Bad: Lift the green bowl out of the yellow bowl and place it on the white plate.
Good: Place the green bowl on the white plate.

Bad: Use the key to open the drawer.
Good: Open the drawer.

Good: Put the pill bottle into the top drawer, then close and lock the drawer and return the key.

Inputs:

Original full instruction:
{job["instruction"]}

Completed actions already done:
{format_actions(job["completed"])}

Remaining steps for reference only:
{format_actions(job["remaining"])}

Existing augmented instructions for this same source segment:
{existing_block}

Style slots:
{chr(10).join(styles)}

Task:
Generate exactly {job["need"]} new instructions.

Requirements:
- Describe only the remaining task.
- Preserve the remaining final goal.
- Do not ask for completed parts again.
- Do not invent new objects, tools, destinations, states, or goals.
- Do not mention image, photo, scene, camera, frame, or robot.
- Do not mention Remaining steps or Completed actions.
- Avoid exact duplicates and near-identical templates.
- Avoid repeating existing augmented instructions.
- Keep every instruction concise.

Return JSON only:
{{
  "shared_goal": "brief summary of the common high-level remaining goal",
  "hidden_details_not_mentioned": ["brief hidden details intentionally kept out of language"],
  "variants": [
    {{
      "slot": 1,
      "instruction": "generated instruction",
      "style_label": "short style label"
    }}
  ]
}}

Output rules:
- variants must contain exactly {job["need"]} items.
- slot must be integers from 1 to {job["need"]}, each exactly once.
- Do not output markdown.
- Do not output reasoning outside JSON.
{retry_block}"""


def validate(job, data, existing_texts):
    if not isinstance(data, dict):
        return False, [], "response is not a JSON dict"

    if "variants" not in data or not isinstance(data["variants"], list):
        return False, [], "missing variants list"

    if len(data["variants"]) != job["need"]:
        return False, [], f"expected {job['need']} variants, got {len(data['variants'])}"

    existing_norms = {norm(x) for x in existing_texts}
    variants = {}

    for item in data["variants"]:
        if not isinstance(item, dict):
            return False, [], "one variant is not a dict"

        if "slot" not in item or "instruction" not in item:
            return False, [], "missing slot or instruction"

        try:
            slot = int(item["slot"])
        except Exception:
            return False, [], "slot is not an integer"

        if slot < 1 or slot > job["need"]:
            return False, [], f"unexpected slot: {slot}"

        if slot in variants:
            return False, [], f"duplicate slot: {slot}"

        instruction = clean(item["instruction"])
        instruction_norm = norm(instruction)

        if len(instruction_norm.split()) < 3:
            return False, [], f"slot {slot}: instruction too short"

        banned = [
            "image", "photo", "scene", "camera", "frame", "robot",
            "left hand", "right hand", "both hands",
            "remaining steps", "completed actions",
        ]

        for word in banned:
            if word in instruction_norm:
                return False, [], f"slot {slot}: contains banned text: {word}"

        if instruction_norm in existing_norms:
            return False, [], f"slot {slot}: duplicates existing instruction"

        if "style_label" in item:
            style_label = clean(item["style_label"])
        else:
            style_label = f"style_{slot}"

        variants[slot] = {
            "slot": slot,
            "instruction": instruction,
            "style_label": style_label,
        }

    if set(variants) != set(range(1, job["need"] + 1)):
        return False, [], "missing slots"

    generated_norms = [norm(variants[i]["instruction"]) for i in range(1, job["need"] + 1)]
    if len(set(generated_norms)) != len(generated_norms):
        return False, [], "duplicate generated instructions"

    return True, [variants[i] for i in range(1, job["need"] + 1)], ""


def generate(job, existing_texts):
    last_error = ""

    for retry in range(1, MAX_RETRY + 1):
        try:
            prompt = build_prompt(job, existing_texts, last_error)
            data = call_gpt_json(CALL_GPT_MODEL, prompt, [job["image_path"]])
            ok, variants, error = validate(job, data, existing_texts)

            if ok:
                hidden = []
                if "hidden_details_not_mentioned" in data:
                    hidden = data["hidden_details_not_mentioned"]

                shared_goal = ""
                if "shared_goal" in data:
                    shared_goal = clean(data["shared_goal"])

                return {
                    "ok": True,
                    "job": job,
                    "variants": variants,
                    "shared_goal": shared_goal,
                    "hidden": hidden,
                    "retry": retry,
                    "error": "",
                }

            last_error = error

        except Exception as e:
            last_error = str(e)

    return {
        "ok": False,
        "job": job,
        "variants": [],
        "shared_goal": "",
        "hidden": [],
        "retry": MAX_RETRY,
        "error": last_error,
    }


# ============================================================
# Saving
# ============================================================

def next_episode_id_map(instruction_data):
    next_ids = {}

    for group_id in instruction_data:
        max_id = 0

        for episode_id in instruction_data[group_id]:
            m = re.search(r"(\d+)$", str(episode_id))
            if m:
                max_id = max(max_id, int(m.group(1)))

        next_ids[group_id] = max_id + 1

    return next_ids


def link_image(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    src = Path(src).resolve()
    rel_src = Path(os.path.relpath(src, start=dst.parent))
    dst.symlink_to(rel_src)


def save_results(root, results, instruction_data, steps_data, meta_data, image_root):
    next_ids = next_episode_id_map(instruction_data)
    created = 0

    for result in results:
        if not result["ok"]:
            continue

        job = result["job"]
        group_id = job["group_id"]

        if group_id not in instruction_data:
            instruction_data[group_id] = {}
        if group_id not in steps_data:
            steps_data[group_id] = {}
        if group_id not in meta_data:
            meta_data[group_id] = {}
        if group_id not in next_ids:
            next_ids[group_id] = 1

        for variant in result["variants"]:
            episode_id = f"episode_{next_ids[group_id]}"
            next_ids[group_id] += 1

            dst_image = image_root / group_id / f"{episode_id}.png"
            link_image(job["image_path"], dst_image)

            instruction_data[group_id][episode_id] = variant["instruction"]
            steps_data[group_id][episode_id] = job["remaining"]

            meta_data[group_id][episode_id] = {
                "source_group_id": job["group_id"],
                "source_episode_id": job["episode_id"],
                "source_instruction": job["instruction"],
                "start_gid": job["start_gid"],
                "slot": variant["slot"],
                "style_label": variant["style_label"],
                "shared_goal": result["shared_goal"],
                "hidden_details_not_mentioned": result["hidden"],
                "source_image_path": str(job["image_path"]),
                "image_path": str(dst_image.relative_to(root)),
                "steps": job["remaining"],
                "completed_actions": job["completed"],
                "target_count": job["target"],
                "existing_count_before_run": job["existing"],
                "need_count_before_run": job["need"],
                "aug_version": "simple_grouped_implicit_instruction_aug_v2",
            }

            created += 1

    return created


# ============================================================
# Main
# ============================================================

def main():
    root = find_root()

    src_instruction_path = root / "tasks" / "instructions" / f"instructions_{SOURCE_TASK_DOMAIN}.json"
    out_instruction_path = root / "tasks" / "instructions" / f"instructions_{AUG_TASK_DOMAIN}.json"
    out_steps_path = root / "tasks" / "steps" / f"steps_{AUG_TASK_DOMAIN}.json"
    out_meta_path = root / "tasks" / "meta" / f"meta_{AUG_TASK_DOMAIN}.json"
    out_image_root = root / "tasks" / "images" / AUG_TASK_DOMAIN

    out_image_root.mkdir(parents=True, exist_ok=True)

    source_items, flat_mode = load_source_items(src_instruction_path)

    instruction_data = read_json(out_instruction_path)
    steps_data = read_json(out_steps_path)
    meta_data = read_json(out_meta_path)

    count_by_pair, texts_by_pair = load_existing_progress(meta_data, instruction_data)
    jobs = collect_jobs(root, source_items, flat_mode, count_by_pair)

    print(f"jobs: {len(jobs)}")

    if not jobs:
        print("No jobs to run.")
        return

    results = []
    workers = min(MAX_WORKERS, len(jobs))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []

        for job in jobs:
            pair = (job["group_id"], job["episode_id"], job["start_gid"])
            if pair in texts_by_pair:
                existing_texts = texts_by_pair[pair]
            else:
                existing_texts = []

            futures.append(executor.submit(generate, job, existing_texts))

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            job = result["job"]
            status = "OK" if result["ok"] else "FAIL"

            print(
                f"[{i}/{len(futures)}] {status} "
                f"{job['group_id']}/{job['episode_id']}/G{job['start_gid']} "
                f"target={job['target']} "
                f"existing={job['existing']} "
                f"need={job['need']} "
                f"new={len(result['variants'])} "
                f"retry={result['retry']} "
                f"{result['error']}"
            )

    results.sort(key=lambda r: (
        nkey(r["job"]["group_id"]),
        nkey(r["job"]["episode_id"]),
        r["job"]["start_gid"],
    ))

    created = save_results(
        root,
        results,
        instruction_data,
        steps_data,
        meta_data,
        out_image_root,
    )

    instruction_data = sort_nested(instruction_data)
    steps_data = sort_nested(steps_data)
    meta_data = sort_nested(meta_data)

    write_json(out_instruction_path, instruction_data)
    write_json(out_steps_path, steps_data)
    write_json(out_meta_path, meta_data)

    total = sum(len(v) for v in instruction_data.values())

    print(f"created: {created}")
    print(f"total: {total}")
    print(f"output instruction json: {out_instruction_path}")
    print(f"output steps json: {out_steps_path}")
    print(f"output meta json: {out_meta_path}")
    print(f"output image root: {out_image_root}")


if __name__ == "__main__":
    main()