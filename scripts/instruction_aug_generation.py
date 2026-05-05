import json
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from swm.utils.apis import call_gpt_json


SOURCE_TASK_DOMAIN = "human"
AUG_TASK_DOMAIN = f"{SOURCE_TASK_DOMAIN}_aug_v6"
GEN_MODEL_NAME = "gemini-3-flash-preview"
CALL_GPT_MODEL = "gpt-4.1"

MAX_WORKERS = 300
MAX_STYLE_BATCH = 8
MAX_REFILL_ROUNDS = 15
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "on", "in", "into", "onto", "from", "with",
    "for", "at", "by", "back", "up", "down", "out", "off", "over", "under", "inside", "outside",
    "then", "first", "next", "finally", "please", "could", "would", "can", "just"
}

BANNED_START_PATTERNS = [
    r"^after\b", r"^with\b", r"^once\b", r"^before\b", r"^when\b",
    r"^end with\b", r"^have\b", r"^leave\b", r"^the\b.*\bshould end up\b",
]

BANNED_FRAGMENTS = [
    "left hand", "right hand", "both hands",
    "image", "photo", "scene", "camera", "frame", "robot",
    "you already", "already picked up", "currently",
    "completed actions", "remaining steps",
]

STYLE_FAMILIES = [
    ("concise_command", "Use the most direct high-level command. Keep it short and clear."),
    ("visual_appearance", "Highlight stable visible appearance cues such as color, material, or shape in the noun phrases."),
    ("spatial_relation", "Highlight stable spatial or source-to-target relations that are clearly visible and reliable."),
    ("goal_oriented", "Phrase it around the goal to achieve, but do not present the final state as already true."),
    ("step_by_step", "Use a natural ordered sentence with first, then, and finally, following only the remaining steps."),
    ("constraint_cautious", "Emphasize task-intrinsic completion constraints or required state changes, without inventing extra risks."),
    ("polite_conversational", "Use a polite conversational request while keeping the same task meaning."),
    ("human_intent_safe", "Optionally add a short generic purpose clause like 'to put things away', but do not invent personal stories."),
    ("verb_synonym", "Use different natural verbs while preserving meaning and task scope."),
    ("state_transition", "Emphasize required state transitions such as unlock, open, place, close, and lock."),
    ("source_to_target", "Emphasize moving items from their current source region to the target region."),
    ("completion_focus", "Emphasize the completed arrangement to achieve, but keep the wording forward and executable."),
]

REFERENCE_MODES = [
    ("plain", "Prefer plain object references unless extra detail is truly helpful."),
    ("appearance", "Prefer visible appearance details when they are clearly reliable."),
    ("location", "Prefer reliable current-location details when they are clearly visible."),
    ("appearance_location", "Mix stable appearance details and reliable current-location details when helpful."),
]

TONES = [
    ("direct", "Use a direct instruction tone."),
    ("polite", "Use a polite tone."),
    ("neutral", "Use a neutral assistant-like tone."),
]

LENGTHS = [
    ("short", "Keep it compact."),
    ("standard", "Use a normal one-sentence or two-clause instruction."),
]


STYLE_POOL = [
    {
        "style_code": f"{family}__{length}__{tone}__{ref}",
        "family": family,
        "length": length,
        "tone": tone,
        "reference": ref,
        "description": f"{family_desc} {length_desc} {tone_desc} {ref_desc}",
    }
    for length, length_desc in LENGTHS
    for tone, tone_desc in TONES
    for ref, ref_desc in REFERENCE_MODES
    for family, family_desc in STYLE_FAMILIES
]


def clean(text):
    return re.sub(r"\s+", " ", str(text).strip().strip('"').strip("'"))


def normalize(text):
    text = clean(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(the|a|an|please|could|would|can|just)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def natural_key(text):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(text))]


def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_root_dir():
    here = Path(__file__).resolve().parent
    if (here / "tasks").exists() and (here / "dataset").exists():
        return here
    if (here.parent / "tasks").exists() and (here.parent / "dataset").exists():
        return here.parent
    return here


def load_source_items(path):
    data = read_json(path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid source json: {path}")

    flat_mode = (
        len(data) == 1
        and SOURCE_TASK_DOMAIN in data
        and isinstance(data[SOURCE_TASK_DOMAIN], dict)
        and all(not isinstance(v, dict) for v in data[SOURCE_TASK_DOMAIN].values())
    )

    items = []
    if flat_mode:
        for episode_id, instruction in sorted(data[SOURCE_TASK_DOMAIN].items(), key=lambda x: natural_key(x[0])):
            instruction = clean(instruction)
            if instruction:
                items.append({
                    "group_id": SOURCE_TASK_DOMAIN,
                    "episode_id": str(episode_id),
                    "instruction": instruction,
                })
        return items, True

    for group_id, episode_map in sorted(data.items(), key=lambda x: natural_key(x[0])):
        if not isinstance(episode_map, dict):
            raise ValueError(f"Top-level value of '{group_id}' must be a dict.")
        for episode_id, instruction in sorted(episode_map.items(), key=lambda x: natural_key(x[0])):
            if isinstance(instruction, dict):
                raise ValueError(f"Instruction of '{group_id}/{episode_id}' must be a string.")
            instruction = clean(instruction)
            if instruction:
                items.append({
                    "group_id": str(group_id),
                    "episode_id": str(episode_id),
                    "instruction": instruction,
                })
    return items, False


def load_nested_output(path):
    data = read_json(path)
    if not isinstance(data, dict):
        return {}
    return {
        str(group_id): dict(episode_map)
        for group_id, episode_map in data.items()
        if isinstance(episode_map, dict)
    }


def collect_existing_progress(meta_data):
    count_by_pair = {}
    used_styles_by_pair = {}

    for episode_map in meta_data.values():
        for meta in episode_map.values():
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

            pair = (str(source_group_id), str(meta["source_episode_id"]), int(meta["start_gid"]))
            count_by_pair[pair] = count_by_pair[pair] + 1 if pair in count_by_pair else 1

            if "style_code" in meta:
                style_code = clean(meta["style_code"])
                if style_code:
                    if pair not in used_styles_by_pair:
                        used_styles_by_pair[pair] = set()
                    used_styles_by_pair[pair].add(style_code)

    return count_by_pair, used_styles_by_pair


def read_group_actions(path):
    group_to_actions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\[G(\d+)\]\s*(.+?)\s*$", line.strip())
        if match:
            gid = int(match.group(1))
            action = clean(match.group(2))
            if gid not in group_to_actions:
                group_to_actions[gid] = []
            group_to_actions[gid].append(action)
    return group_to_actions


def collect_jobs(root_dir, source_items, flat_mode, count_by_pair):
    jobs = []

    for item in source_items:
        group_id = item["group_id"]
        episode_id = item["episode_id"]

        if flat_mode:
            keyframe_dir = root_dir / "dataset" / "keyframes" / SOURCE_TASK_DOMAIN / episode_id
            group_path = root_dir / "eval_results" / GEN_MODEL_NAME / SOURCE_TASK_DOMAIN / episode_id / "kf_plan_group.txt"
        else:
            keyframe_dir = root_dir / "dataset" / "keyframes" / SOURCE_TASK_DOMAIN / group_id / episode_id
            group_path = root_dir / "eval_results" / GEN_MODEL_NAME / SOURCE_TASK_DOMAIN / group_id / episode_id / "kf_plan_group.txt"

        if not keyframe_dir.is_dir() or not group_path.is_file():
            continue

        group_to_actions = read_group_actions(group_path)
        if not group_to_actions:
            continue

        group_ids = sorted(group_to_actions)
        seg_dirs = []
        for path in keyframe_dir.iterdir():
            if path.is_dir():
                match = re.fullmatch(r"seg_(\d+)", path.name)
                if match:
                    seg_dirs.append((int(match.group(1)), path))
        seg_dirs.sort(key=lambda x: x[0])

        for start_gid, seg_dir in seg_dirs:
            if start_gid not in group_to_actions:
                continue

            images = sorted(
                [p for p in seg_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS],
                key=lambda p: natural_key(p.name),
            )
            if not images:
                continue

            completed, remaining = [], []
            for gid in group_ids:
                if gid < start_gid:
                    completed.extend(group_to_actions[gid])
                else:
                    remaining.extend(group_to_actions[gid])

            if not remaining:
                continue

            pair = (group_id, episode_id, start_gid)
            target_count = len(remaining)
            existing_count = count_by_pair[pair] if pair in count_by_pair else 0
            need_count = max(0, target_count - existing_count)
            if need_count == 0:
                continue

            jobs.append({
                "group_id": group_id,
                "episode_id": episode_id,
                "instruction": item["instruction"],
                "start_gid": start_gid,
                "image_path": images[0],
                "completed": completed,
                "remaining": remaining,
                "target_count": target_count,
                "existing_count": existing_count,
                "need_count": need_count,
            })

    jobs.sort(key=lambda x: (natural_key(x["group_id"]), natural_key(x["episode_id"]), x["start_gid"]))
    return jobs


def format_actions(actions, numbered=False):
    if not actions:
        return "- none"
    if numbered:
        return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(actions))
    return "\n".join(f"- {x}" for x in actions)


def build_anchor_prompt(job):
    return f"""You are rewriting robot task instructions from an image and a remaining action sequence.

Goal:
Write one canonical instruction and a small semantic anchor for the remaining task only.

Grounding priority:
1. Remaining steps define what still needs to be done.
2. The image defines which appearance or location details are safe to mention.
3. Completed actions are already done and must never be asked again.

Hard rules:
- Describe only the remaining task.
- Preserve the original goal and any still-relevant final constraints.
- Do not add new actions, objects, goals, states, assumptions, or hidden history.
- Do not redo completed parts.
- Write the canonical instruction as a forward executable instruction.
- Start with an imperative verb.
- Keep it high-level, not a detailed plan.
- Do not use these openings: After..., With..., Once..., Before..., When..., End with..., Have..., Leave....
- Do not present the final state as if it is already true.
- Do not mention hand allocation, the image, the camera, the scene, or the robot.
- Only mention appearance or location details if they are clearly visible and reliable.
- Do not hallucinate colors, materials, counts, states, or locations.

For must_keep:
- objects: short noun phrases for key remaining task objects that should still be mentioned.
- target_region: short destination/container/area phrase if important.
- final_constraints: short labels only, such as close drawer, lock drawer, return key, turn off faucet.

Original instruction:
{job["instruction"]}

Completed actions:
{format_actions(job["completed"])}

Remaining steps:
{format_actions(job["remaining"], numbered=True)}

Return JSON only:
{{
  "canonical": "...",
  "must_keep": {{
    "objects": ["..."],
    "target_region": "...",
    "final_constraints": ["..."]
  }}
}}"""


def build_variant_prompt(job, canonical, must_keep, style_cards):
    style_text = []
    for i, card in enumerate(style_cards, 1):
        style_text.append(
            f"{i}. style_code={card['style_code']}\n"
            f"   - family={card['family']}\n"
            f"   - length={card['length']}\n"
            f"   - tone={card['tone']}\n"
            f"   - reference={card['reference']}\n"
            f"   - guidance={card['description']}"
        )

    return f"""You are generating multiple style-controlled rewrites of the same robot task instruction.

Meaning anchor:
- Canonical instruction is the semantic anchor.
- must_keep lists the objects, target region, and final constraints that must remain preserved.
- Remaining steps define what still needs to be accomplished.
- Completed actions are already done and must never be asked again.

Global hard rules for every variant:
- Preserve exactly the same task meaning as the canonical instruction.
- Describe only the remaining task.
- Do not redo completed parts.
- Do not add new actions, objects, goals, states, constraints, assumptions, or hidden history.
- Keep forward executable wording.
- Do not present the final state as if it is already true.
- Do not use these openings: After..., With..., Once..., Before..., When..., End with..., Have..., Leave....
- Do not use phrases such as you already..., currently..., already picked up....
- Do not mention hand allocation, the image, the camera, the scene, or the robot.
- Only mention appearance or location details that are clearly visible and reliable.
- Do not hallucinate colors, materials, counts, states, or locations.
- Do not use numbering or bullet points inside the instruction text.
- Do not output reasoning or notes.

Family-specific rules:
- concise_command: short direct command.
- visual_appearance: add stable visible appearance cues mainly in noun phrases.
- spatial_relation: emphasize reliable spatial or source-target relations.
- goal_oriented: describe the goal to achieve, but keep it forward and executable.
- step_by_step: may use first, then, finally, but must follow only the remaining steps.
- constraint_cautious: emphasize required completion constraints or state changes, not imaginary dangers.
- polite_conversational: use a natural polite request style.
- human_intent_safe: may add a short generic purpose clause, but do not invent personal motives or stories.
- verb_synonym: vary natural verbs while preserving meaning.
- state_transition: highlight required state changes if they are part of the task.
- source_to_target: emphasize moving items from source region to target region.
- completion_focus: emphasize the arrangement to achieve, but do not state it as already completed.

Diversity rules:
- Variants must be clearly different in style, not just one-word swaps.
- Do not repeat the same opening phrase across many variants.
- Do not repeat the same sentence skeleton across many variants.
- Change at least two of: opening, tone, information focus, reference density, or verb choice.

Original instruction:
{job["instruction"]}

Canonical instruction:
{canonical}

must_keep:
{json.dumps(must_keep, ensure_ascii=False, indent=2)}

Completed actions:
{format_actions(job["completed"])}

Remaining steps:
{format_actions(job["remaining"], numbered=True)}

Style cards to realize:
{chr(10).join(style_text)}

Return JSON only:
{{
  "variants": [
    {{
      "style_code": "exactly one requested style_code",
      "instruction": "..."
    }}
  ]
}}"""


def parse_anchor(data):
    if not isinstance(data, dict) or "canonical" not in data or "must_keep" not in data:
        raise ValueError("Invalid anchor response.")

    canonical = clean(data["canonical"])
    must_keep_raw = data["must_keep"]
    if not isinstance(must_keep_raw, dict):
        raise ValueError("Invalid must_keep response.")

    objects = must_keep_raw["objects"] if "objects" in must_keep_raw and isinstance(must_keep_raw["objects"], list) else []
    constraints = (
        must_keep_raw["final_constraints"]
        if "final_constraints" in must_keep_raw and isinstance(must_keep_raw["final_constraints"], list)
        else []
    )
    target = must_keep_raw["target_region"] if "target_region" in must_keep_raw else ""

    must_keep = {
        "objects": [clean(x) for x in objects if clean(x)],
        "target_region": clean(target),
        "final_constraints": [clean(x) for x in constraints if clean(x)],
    }

    if not canonical:
        raise ValueError("Empty canonical instruction.")

    return canonical, must_keep


def text_has_phrase_token(text_low, phrase):
    tokens = [t for t in normalize(phrase).split() if len(t) > 1 and t not in STOPWORDS]
    return True if not tokens else any(t in text_low for t in tokens)


def satisfies_constraint(text_low, label):
    label_low = normalize(label)
    tokens = [t for t in label_low.split() if len(t) > 1 and t not in STOPWORDS]
    if not tokens:
        return True

    if "return" in label_low and "key" in label_low:
        return "key" in text_low and any(v in text_low for v in ["return", "put", "place", "set"])
    if "put back" in label_low and "key" in label_low:
        return "key" in text_low and any(v in text_low for v in ["return", "put", "place", "set"])
    if "lock" in label_low and "drawer" in label_low:
        return "lock" in text_low and "drawer" in text_low
    if ("close" in label_low or "shut" in label_low) and "drawer" in label_low:
        return "drawer" in text_low and any(v in text_low for v in ["close", "shut"])
    if "unlock" in label_low:
        return "unlock" in text_low
    if "open" in label_low and "drawer" in label_low:
        return "open" in text_low and "drawer" in text_low
    if "turn off" in label_low:
        return "off" in text_low and any(v in text_low for v in ["turn", "switch"])
    if "turn on" in label_low:
        return "on" in text_low and any(v in text_low for v in ["turn", "switch"])

    if len(tokens) <= 2:
        return all(t in text_low for t in tokens)
    return sum(t in text_low for t in tokens) >= len(tokens) - 1


def usable_instruction(text, canonical, must_keep):
    text = clean(text)
    text_low = normalize(text)

    if not text_low or len(text_low.split()) < 4:
        return False
    if text_low == normalize(canonical):
        return False
    if any(re.match(pattern, text_low) for pattern in BANNED_START_PATTERNS):
        return False
    if any(fragment in text_low for fragment in BANNED_FRAGMENTS):
        return False

    for obj in must_keep["objects"]:
        if not text_has_phrase_token(text_low, obj):
            return False

    if must_keep["target_region"] and not text_has_phrase_token(text_low, must_keep["target_region"]):
        return False

    for label in must_keep["final_constraints"]:
        if not satisfies_constraint(text_low, label):
            return False

    return True


def generate_one_sample(job, used_style_codes):
    anchor_data = call_gpt_json(CALL_GPT_MODEL, build_anchor_prompt(job), [job["image_path"]])
    canonical, must_keep = parse_anchor(anchor_data)

    need = job["need_count"]
    candidate_cards = [card for card in STYLE_POOL if card["style_code"] not in used_style_codes][: max(need * 2, need)]
    collected, collected_codes, collected_texts = [], set(), set()

    round_cards = candidate_cards
    for _ in range(MAX_REFILL_ROUNDS):
        if len(collected) >= need or not round_cards:
            break

        retry_cards = []
        for i in range(0, len(round_cards), MAX_STYLE_BATCH):
            batch = round_cards[i: i + MAX_STYLE_BATCH]
            requested_codes = {card["style_code"] for card in batch}

            data = call_gpt_json(
                CALL_GPT_MODEL,
                build_variant_prompt(job, canonical, must_keep, batch),
                [job["image_path"]],
            )

            accepted_codes = set()
            if isinstance(data, dict) and "variants" in data and isinstance(data["variants"], list):
                for item in data["variants"]:
                    if not isinstance(item, dict):
                        continue
                    if "style_code" not in item or "instruction" not in item:
                        continue

                    style_code = clean(item["style_code"])
                    instruction = clean(item["instruction"])
                    norm_text = normalize(instruction)

                    if style_code not in requested_codes:
                        continue
                    if style_code in collected_codes:
                        continue
                    if norm_text in collected_texts:
                        continue
                    if not usable_instruction(instruction, canonical, must_keep):
                        continue

                    collected.append({
                        "style_code": style_code,
                        "instruction": instruction,
                    })
                    collected_codes.add(style_code)
                    collected_texts.add(norm_text)
                    accepted_codes.add(style_code)

                    if len(collected) >= need:
                        break

            for card in batch:
                if card["style_code"] not in accepted_codes and card["style_code"] not in used_style_codes:
                    retry_cards.append(card)

            if len(collected) >= need:
                break

        round_cards = retry_cards

    return {
        "source_group_id": job["group_id"],
        "source_episode_id": job["episode_id"],
        "source_instruction": job["instruction"],
        "start_gid": job["start_gid"],
        "image_src": str(job["image_path"]),
        "steps": job["remaining"],
        "canonical": canonical,
        "must_keep": must_keep,
        "target_count": job["target_count"],
        "existing_count": job["existing_count"],
        "need_count": job["need_count"],
        "variants": collected[:need],
    }


def copy_image_as_png(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
    else:
        with Image.open(src) as image:
            image.save(dst)


def sort_nested(data):
    return {
        group_id: dict(sorted(episode_map.items(), key=lambda x: natural_key(x[0])))
        for group_id, episode_map in sorted(data.items(), key=lambda x: natural_key(x[0]))
    }


def next_episode_indices(instructions_data):
    indices = {}
    for group_id, episode_map in instructions_data.items():
        max_idx = 0
        for episode_id in episode_map:
            match = re.search(r"(\d+)$", str(episode_id))
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        indices[group_id] = max_idx + 1
    return indices


def save_samples(root_dir, samples, instructions_data, steps_data, meta_data, out_image_root):
    next_idx = next_episode_indices(instructions_data)
    created = 0

    for sample in samples:
        group_id = sample["source_group_id"]

        if group_id not in instructions_data:
            instructions_data[group_id] = {}
        if group_id not in steps_data:
            steps_data[group_id] = {}
        if group_id not in meta_data:
            meta_data[group_id] = {}
        if group_id not in next_idx:
            next_idx[group_id] = 1

        for variant in sample["variants"]:
            new_episode_id = f"episode_{next_idx[group_id]}"
            next_idx[group_id] += 1

            dst_image = out_image_root / group_id / f"{new_episode_id}.png"
            copy_image_as_png(Path(sample["image_src"]), dst_image)

            instructions_data[group_id][new_episode_id] = variant["instruction"]
            steps_data[group_id][new_episode_id] = sample["steps"]
            meta_data[group_id][new_episode_id] = {
                "source_group_id": sample["source_group_id"],
                "source_episode_id": sample["source_episode_id"],
                "source_instruction": sample["source_instruction"],
                "start_gid": sample["start_gid"],
                "style_code": variant["style_code"],
                "canonical_instruction": sample["canonical"],
                "must_keep": sample["must_keep"],
                "source_image_path": sample["image_src"],
                "image_path": str(dst_image.relative_to(root_dir)),
                "target_count": sample["target_count"],
                "existing_count_before_run": sample["existing_count"],
                "need_count_before_run": sample["need_count"],
                "aug_version": "simple_two_stage_anchor_plus_style_plan_refactored",
            }
            created += 1

    return created


def main():
    root_dir = detect_root_dir()

    src_instruction_path = root_dir / "tasks" / "instructions" / f"instructions_{SOURCE_TASK_DOMAIN}.json"
    out_instruction_path = root_dir / "tasks" / "instructions" / f"instructions_{AUG_TASK_DOMAIN}.json"
    out_steps_path = root_dir / "tasks" / "steps" / f"steps_{AUG_TASK_DOMAIN}.json"
    out_meta_path = root_dir / "tasks" / "meta" / f"meta_{AUG_TASK_DOMAIN}.json"
    out_image_root = root_dir / "tasks" / "images" / AUG_TASK_DOMAIN

    out_instruction_path.parent.mkdir(parents=True, exist_ok=True)
    out_steps_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)
    out_image_root.mkdir(parents=True, exist_ok=True)

    source_items, flat_mode = load_source_items(src_instruction_path)

    instructions_data = load_nested_output(out_instruction_path)
    steps_data = load_nested_output(out_steps_path)
    meta_data = load_nested_output(out_meta_path)

    count_by_pair, used_styles_by_pair = collect_existing_progress(meta_data)
    jobs = collect_jobs(root_dir, source_items, flat_mode, count_by_pair)

    print(f"jobs: {len(jobs)}")

    samples = []
    if jobs:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as executor:
            futures = []
            for job in jobs:
                pair = (job["group_id"], job["episode_id"], job["start_gid"])
                used_styles = used_styles_by_pair[pair] if pair in used_styles_by_pair else set()
                futures.append(executor.submit(generate_one_sample, job, used_styles))

            total = len(futures)
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    sample = future.result()
                    samples.append(sample)
                    print(
                        f"[{i}/{total}] "
                        f"{sample['source_group_id']}/{sample['source_episode_id']}/G{sample['start_gid']} "
                        f"target={sample['target_count']} "
                        f"existing={sample['existing_count']} "
                        f"new={len(sample['variants'])}"
                    )
                except Exception as error:
                    print(f"[ERROR {i}/{total}] {error}")

    samples.sort(key=lambda x: (
        natural_key(x["source_group_id"]),
        natural_key(x["source_episode_id"]),
        x["start_gid"],
    ))

    created = save_samples(root_dir, samples, instructions_data, steps_data, meta_data, out_image_root)

    instructions_data = sort_nested(instructions_data)
    steps_data = sort_nested(steps_data)
    meta_data = sort_nested(meta_data)

    write_json(out_instruction_path, instructions_data)
    write_json(out_steps_path, steps_data)
    write_json(out_meta_path, meta_data)

    total = sum(len(episode_map) for episode_map in instructions_data.values())
    print(f"created: {created}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()