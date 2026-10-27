"""Extract grounded actions: ordered Generator groups, one episode Compiler."""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from ..llm import call_gpt

_IMAGE_LOCK = Lock()
MAX_JSON_ATTEMPTS = 3
REQUEST_SETTINGS = {"reasoning_effort": "medium", "temperature": 0, "max_retries": 0}
DEFAULT_COMPILER_FRAME_SIZE = (640, 360)
FORMAT_VERSION = 3
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)
LOCAL_FRAME_RE = re.compile(r"\bK\d+\b")
FRAME_LABEL = r"\b(?:K\d+|img_\d+)\b"
FRAME_LIST = FRAME_LABEL + r"(?:\s*(?:,|/|->|→|and|to|through|-)\s*" + FRAME_LABEL + r")*"


class OutputContractError(ValueError):
    pass


def _normalize_action_text(value):
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\n\r|"):
        raise OutputContractError("action must be one non-empty line")
    # Frame citations are display metadata. Keep the physical claim and the
    # separately validated evidence, without spending a model retry on wording.
    value = re.sub(r"[\[(]\s*" + FRAME_LIST + r"\s*[\])]", "", value)
    value = re.sub(r"\b(?:at|in|from|between|during|through|see)\s+(?:frames?\s+)?" + FRAME_LIST, "", value, flags=re.IGNORECASE)
    value = re.sub(FRAME_LABEL, "", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(value.split()))
    value = re.sub(r"[\[(]\s*[\])]", "", value).strip(" ,;:")
    action = re.sub(r"^\s*[-*•]\s+", "", value.strip())
    action = re.sub(r"^\s*\(?\d+\)?[.)]\s+", "", action).strip()
    action = re.sub(r"\\([.,;:!?])", r"\1", action)
    key = action.lower().strip(". ")
    if (not key or key in {"none", "null", "continue", "continuation", "no action", "no actions"}
            or key.startswith("continuation of ")):
        raise OutputContractError("use actions: [] when no independent action completes")
    return action if action.endswith(".") else action + "."


def _normalize_frames(value, allowed):
    if not isinstance(value, list) or len(value) < 2:
        raise OutputContractError("frames must cite at least two distinct chronological images")
    if any(not isinstance(frame, str) or frame not in allowed for frame in value):
        raise OutputContractError("frames reference an image outside this group or the attached episode images")
    positions = [allowed[frame] for frame in value]
    if positions != sorted(set(positions)):
        raise OutputContractError("frames must be distinct and in chronological order")
    return list(value)


def normalize_group_result(data, frame_count):
    if not isinstance(data, dict) or set(data) != {"changes", "actions"}:
        raise OutputContractError("generator output must contain only changes and actions")
    if not isinstance(data["changes"], list) or not isinstance(data["actions"], list):
        raise OutputContractError("generator changes and actions must be lists; no actions means []")
    allowed = {f"K{i}": i for i in range(frame_count)}
    changes = []
    for item in data["changes"]:
        if not isinstance(item, dict) or set(item) != {"frames", "change"}:
            raise OutputContractError("each change needs only frames and change")
        frames = _normalize_frames(item["frames"], allowed)
        if not isinstance(item["change"], str) or not item["change"].strip():
            raise OutputContractError("change must contain non-empty observation text")
        changes.append({"frames": frames, "change": item["change"].strip()})
    expected = [[f"K{i}", f"K{i+1}"] for i in range(frame_count - 1)]
    if [item["frames"] for item in changes] != expected:
        raise OutputContractError("changes must cover every adjacent pair exactly once in order")
    actions = []
    for item in data["actions"]:
        if not isinstance(item, dict) or set(item) != {"frames", "action"}:
            raise OutputContractError("each action needs only frames and action")
        actions.append({"frames": _normalize_frames(item["frames"], allowed),
                        "action": _normalize_action_text(item["action"])})
    # Different events can share evidence. Their physical order cannot be inferred
    # reliably from the first/last cited frame, so preserve the model's sequence.
    return {"changes": changes, "actions": actions}


def _frame_id(path):
    return f"img_{int(path.stem)}"


def _global_text(text, images):
    mapping = {f"K{i}": _frame_id(path) for i, path in enumerate(images)}
    return LOCAL_FRAME_RE.sub(lambda match: mapping.get(match[0], match[0]), text)


def normalize_compiler_result(data, generator_results, segment_images):
    keys = [f"G{i}" for i in range(len(segment_images))]
    if not isinstance(data, dict) or list(data) != keys:
        raise OutputContractError("compiler must return every group exactly once in order: " + ", ".join(keys))
    if len(generator_results) != len(segment_images):
        raise OutputContractError("generator results do not cover every group")
    allowed = {f"img_{frame['frame']}": frame["frame"]
               for frame in _plan_compiler_images(segment_images)["frames"]}
    result = {}
    for key in keys:
        item = data[key]
        if not isinstance(item, dict) or set(item) != {"review", "actions"}:
            raise OutputContractError(f"{key} needs only review and actions")
        if not isinstance(item["actions"], list):
            raise OutputContractError(f"{key} actions must be a list, including [] for no actions")
        actions = []
        for action in item["actions"]:
            if not isinstance(action, dict) or set(action) != {"frames", "action"}:
                raise OutputContractError(f"{key} each action needs only frames and action")
            actions.append({"frames": _normalize_frames(action["frames"], allowed),
                            "action": _normalize_action_text(action["action"])})
        review = item["review"]
        if not isinstance(review, list) or not review:
            raise OutputContractError(f"{key} review must be a non-empty evidence list")
        normalized = []
        for note in review:
            if not isinstance(note, dict) or set(note) != {"frames", "reason"}:
                raise OutputContractError(f"{key} each review needs only frames and reason")
            if not isinstance(note["reason"], str) or not note["reason"].strip():
                raise OutputContractError(f"{key} review reason must state visible evidence for the decision")
            normalized.append({"frames": _normalize_frames(note["frames"], allowed),
                               "reason": " ".join(note["reason"].split())})
        result[key] = {"review": normalized, "actions": actions}
    return result


def _format_generator_output(results, groups=None):
    summary = [f"[G{group}] {item['action']}" for group, result in enumerate(results)
               for item in result["actions"]]
    lines = ["Actions:", *summary] if summary else ["Actions: []"]
    for group, result in enumerate(results):
        def refs(item):
            frames = item["frames"]
            if groups is not None:
                frames = [_frame_id(groups[group][int(frame[1:])]) for frame in frames]
            return "[" + ", ".join(frames) + "]"
        lines.extend(("", f"[G{group}]", "changes:"))
        lines.extend(f"- {refs(item)} " + (_global_text(item["change"], groups[group])
                     if groups is not None else item["change"]) for item in result["changes"])
        lines.append("Actions:" if result["actions"] else "Actions: []")
        lines.extend(f"- {refs(item)} {item['action']}" for item in result["actions"])
    return "\n".join(lines) + "\n"


def _format_compiler_output(result):
    lines = []
    for key, item in result.items():
        if lines:
            lines.append("")
        lines.extend((f"[{key}]", "review:"))
        for note in item["review"]:
            lines.append(f"- [{', '.join(note['frames'])}] {note['reason']}")
    return "\n".join(lines) + "\n"


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path, value):
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _read_records(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_bytes().splitlines():
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A process can stop during its final append; retain the partial
            # record on disk without treating it as a successful response.
            continue
    return records


def _append_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OutputContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _call_json(model, prompt, images, normalize, *, records_path, label, run_id=None):
    paths = [str(path.resolve()) for path in images]
    previous = [record for record in _read_records(records_path)
                if record.get("event") == "request" and record.get("run_id") == run_id
                and record.get("label") == label and record.get("base_prompt") == prompt
                and record.get("model") == model and record.get("image_paths") == paths
                and record.get("settings") == REQUEST_SETTINGS]
    for record in previous:
        if record.get("success"):
            return normalize(record["parsed_output"])
    if len(previous) >= MAX_JSON_ATTEMPTS:
        raise RuntimeError(f"{label} exhausted {MAX_JSON_ATTEMPTS} logical attempts")
    last_failure = previous[-1].get("retry_feedback", "") if previous else ""
    for attempt in range(len(previous) + 1, MAX_JSON_ATTEMPTS + 1):
        request_prompt = prompt + (
            f"\n\nAttempt {attempt-1} failed: {last_failure}\n"
            "Regenerate ONE complete valid JSON object for the entire original scope. "
            "Include all required groups and fields; do not append another root object."
            if last_failure else "")
        record = {"event": "request", "run_id": run_id, "label": label, "attempt": attempt,
                  "model": model, "settings": dict(REQUEST_SETTINGS), "base_prompt": prompt,
                  "prompt": request_prompt, "image_paths": paths, "started_at": _now(),
                  "raw_output": None, "parsed_output": None, "normalized_output": None,
                  "capture": {"usage": None}, "success": False}
        capture = {}
        error = None
        try:
            record["raw_output"] = call_gpt(model, request_prompt, images, capture=capture, **REQUEST_SETTINGS)
            if capture.get("finish_reason") == "length":
                raise OutputContractError("Response hit the token limit; return the complete original scope with shorter observations/reasons.")
            if capture.get("finish_reason") not in (None, "stop"):
                raise OutputContractError("Response did not finish normally")
            if not isinstance(record["raw_output"], str) or not record["raw_output"].strip():
                raise OutputContractError("Response has no JSON text")
            text = record["raw_output"].strip()
            match = JSON_FENCE_RE.fullmatch(text)
            record["parsed_output"] = json.loads(match.group(1) if match else text,
                                                object_pairs_hook=_unique_object)
            record["normalized_output"] = normalize(record["parsed_output"])
            record["success"] = True
        except BaseException as caught:
            error = caught
            last_failure = str(caught) if isinstance(caught, (OutputContractError, json.JSONDecodeError)) else "The request/response did not complete successfully."
            record.update(error_type=type(caught).__name__, error=str(caught), retry_feedback=last_failure)
        finally:
            record.update(capture={"usage": None, **capture}, finished_at=_now())
            _append_record(records_path, record)
        if error is None:
            return record["normalized_output"]
        if not isinstance(error, Exception):
            raise error
        if getattr(error, "status_code", None) in (401, 403, 404):
            raise error
        if attempt < MAX_JSON_ATTEMPTS:
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"{label} exhausted {MAX_JSON_ATTEMPTS} logical attempts: {last_failure}") from error


def _prepare_generator_images(images, directory, prefix):
    if len(images) not in {3, 4, 5, 6}:
        raise ValueError("Generator comparison needs 3..6 chronological keyframes")
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(images[0]) as image:
        shapes = [image.size]
    for path in images[1:]:
        with Image.open(path) as image:
            shapes.append(image.size)
    portrait = sum(height > width for width, height in shapes) > len(shapes) / 2
    width, height = (180, 320) if portrait else (320, 180)
    header = 24
    if len(images) == 3:
        rows, columns = 1, 3
    elif len(images) == 4:
        rows, columns = 2, 2
    else:
        rows, columns = (2, 3) if portrait else (3, 2)
    prepared = list(images)
    if len(images) >= 5:
        prepared = []
        cache = directory.parent / "image_cache"
        cache.mkdir(parents=True, exist_ok=True)
        for source in images:
            target = cache / f"{_frame_id(source)}.jpg"
            if not target.exists():
                with Image.open(source) as image:
                    image = image.convert("RGB")
                    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                    temporary = target.with_suffix(".jpg.tmp")
                    image.save(temporary, "JPEG", quality=90, optimize=True)
                    temporary.replace(target)
            prepared.append(target)
    timeline = Image.new("RGB", (columns * width, rows * (height + header)), (24, 24, 24))
    draw = ImageDraw.Draw(timeline)
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    for index, source in enumerate(images):
        row, column = divmod(index, columns)
        x, y = column * width, row * (height + header)
        draw.text((x + 3, y + 1), f"K{index}", font=font, fill="white")
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            timeline.paste(image, (x + (width-image.width)//2, y + header + (height-image.height)//2))
    overview = directory / f"{prefix}.jpg"
    timeline.save(overview, "JPEG", quality=90, optimize=True)
    return [overview, *prepared]


def _plan_compiler_images(groups):
    frames, group_plan, rows = {}, [], []
    for group, paths in enumerate(groups):
        ids = [int(path.stem) for path in paths]
        if len(ids) not in {3, 4, 5, 6} or ids != sorted(set(ids)):
            raise ValueError("grouped_3x3 needs groups of 3..6 distinct chronological frames")
        selected = ids[:1] + ids[2:] if len(ids) == 4 else ids
        for frame, path in zip(ids, paths):
            if frame not in selected:
                continue
            entry = frames.setdefault(frame, {"frame": frame, "source": str(path.resolve()), "groups": []})
            entry["groups"].append(group)
        start = len(rows)
        chunks = [selected[i:i+3] for i in range(0, len(selected), 3)]
        for part, chunk in enumerate(chunks, 1):
            rows.append({"row_index": len(rows), "group": group, "part": part,
                         "parts": len(chunks), "frames": chunk})
        group_plan.append({"group": group, "original_frames": ids, "selected_frames": selected,
                           "omitted_frames": ids[1:2] if len(ids) == 4 else [],
                           "row_start": start, "row_end": len(rows)})
    if not rows:
        raise ValueError("No groups to display")
    ordered = [{"time_index": index, **frames[frame]} for index, frame in enumerate(sorted(frames))]
    pages, start = [], 0
    while start < len(group_plan):
        end, used = start, 0
        while end < len(group_plan):
            count = group_plan[end]["row_end"] - group_plan[end]["row_start"]
            if used + count > 3:
                break
            used += count
            end += 1
        first, last = group_plan[start]["row_start"], group_plan[end-1]["row_end"]
        page_rows = [dict(row) for row in rows[first:last]]
        page_rows.extend({"group": None, "frames": []} for _ in range(3-used))
        pages.append({"page": len(pages)+1, "groups": list(range(start, end)), "rows": page_rows})
        start = end
    return {"all_frames": ordered, "frames": ordered, "groups": group_plan, "rows": rows, "pages": pages}


def _compiler_frame_size(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(n) is not int or n <= 0 for n in value) or value[0] < value[1]):
        raise ValueError("compiler_frame_size must be positive integer (landscape width, height), width >= height")
    return tuple(value)


def _prepare_compiler_images(groups, directory, *, frame_size=DEFAULT_COMPILER_FRAME_SIZE):
    landscape_width, landscape_height = _compiler_frame_size(frame_size)
    plan = _plan_compiler_images(groups)
    directory.mkdir(parents=True, exist_ok=True)
    portrait_count = 0
    for frame in plan["all_frames"]:
        with Image.open(frame["source"]) as image:
            portrait_count += image.height > image.width
    portrait = portrait_count > len(plan["all_frames"])/2
    width, height = ((landscape_height, landscape_width) if portrait
                     else (landscape_width, landscape_height))
    header = max(1, round(24 * landscape_height / 180))
    font = ImageFont.truetype("DejaVuSans.ttf", max(1, round(16 * landscape_height / 180)))
    lookup = {frame["frame"]: frame for frame in plan["all_frames"]}
    prepared = []
    layout = ["Each image is a 3x3 timeline; read left to right, then top to bottom. "
              "[G#] identifies a group, img_N a real source moment. "
              "Repeated img_N is the same moment, not another event. Blank cells are padding."]
    for page in plan["pages"]:
        canvas = Image.new("RGB", (3*width, 3*(height+header)), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        descriptions = []
        for row, entry in enumerate(page["rows"]):
            for column, frame_id in enumerate(entry["frames"]):
                label = f"[G{entry['group']}] img_{frame_id}" if column == 0 else f"img_{frame_id}"
                x, y = column*width, row*(height+header)
                draw.text((x+3, y+1), label, font=font, fill="white")
                with Image.open(lookup[frame_id]["source"]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((width, height), Image.Resampling.LANCZOS)
                    canvas.paste(image, (x+(width-image.width)//2, y+header+(height-image.height)//2))
            if entry["frames"]:
                descriptions.append(f"row {row+1} [G{entry['group']}] [" +
                                    ", ".join(f"img_{n}" for n in entry["frames"]) + "]")
        target = directory / f"grouped_{page['page']:02d}.jpg"
        canvas.save(target, "JPEG", quality=90, optimize=True)
        prepared.append(target)
        layout.append(f"Image {page['page']}: " + "; ".join(descriptions) + ".")
    layout.append(f"Episode final frame: img_{plan['all_frames'][-1]['frame']}.")
    _write_json(directory / "source_mapping.json", {"cell_long_side": max(width, height),
                "cell_size": [width, height], "landscape_cell_size": [landscape_width, landscape_height],
                "orientation": "portrait" if portrait else "landscape", "plan": plan,
                "attachments": [str(path.resolve()) for path in prepared]})
    return prepared, "\n".join(layout)


def extract_keyframe_actions(
    model_name: str,
    keyframe_dir: Path,
    instruction: str,
    save_dir: Path,
    *,
    compiler_frame_size: tuple[int, int] = DEFAULT_COMPILER_FRAME_SIZE,
) -> tuple[Path, list[str]]:
    compiler_frame_size = _compiler_frame_size(compiler_frame_size)
    keyframe_dir, save_dir = Path(keyframe_dir), Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    actions_path = save_dir / "kf_actions.txt"
    records_path = save_dir / "api_calls.jsonl"
    last_run = next((record for record in reversed(_read_records(records_path))
                     if record.get("event") == "run"), None)
    if last_run is None:
        actions_path.unlink(missing_ok=True)
    segments = sorted(keyframe_dir.glob("seg_*"), key=lambda path: int(path.name.split("_")[-1]))
    groups = [sorted(segment.glob("*.png"), key=lambda path: int(path.stem)) for segment in segments]
    groups = [images for images in groups if images]
    if not groups:
        raise ValueError(f"No keyframes found in {keyframe_dir}")
    if any(len(images) not in {3, 4, 5, 6} for images in groups):
        raise ValueError("Action extraction needs 3..6 chronological keyframes per group")
    for images in groups:
        ids = [int(path.stem) for path in images]
        if ids != sorted(set(ids)):
            raise ValueError("Keyframes must be distinct chronological source indices")
    checkpoint_path = save_dir / "actions_checkpoint.json"
    prompt_dir = Path(__file__).parents[1] / "prompt_templates"
    common, generator, compiler = [(prompt_dir / name).read_text(encoding="utf-8") for name in
                                   ("kf_actions_common.txt", "kf_actions_generator.txt", "kf_actions_compiler.txt")]
    inputs = {"format_version": FORMAT_VERSION, "model": model_name, "instruction": instruction,
              "settings": dict(REQUEST_SETTINGS), "prompts": [common, generator, compiler],
              "image_settings": {"generator_cell_long_side": 320, "single_long_side": 1280,
                                 "compiler_frame_size": list(compiler_frame_size), "jpeg_quality": 90,
                                 "compiler_drop_second_of_four": True},
              "frames": [[[str(image.resolve()), image.stat().st_mtime_ns, image.stat().st_size]
                          for image in images] for images in groups]}
    if last_run is not None and last_run["inputs"] != inputs:
        raise ValueError("This output directory belongs to different inputs or settings; use a new save_dir.")
    actions_path.unlink(missing_ok=True)
    checkpoint = {}
    try:
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if candidate["inputs"] == inputs and len(candidate["groups"]) <= len(groups):
            candidate["groups"] = [normalize_group_result(value, len(groups[i]))
                                   for i, value in enumerate(candidate["groups"])]
            checkpoint = candidate
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if not checkpoint and last_run is not None:
        checkpoint = {"inputs": inputs, "run_id": last_run["run_id"], "groups": []}
    if not checkpoint:
        # These are only this extractor's generated images in the explicitly
        # selected output directory. Existing request history is retained.
        for folder, pattern in [("generator_images", "group*.jpg"), ("compiler_images", "grouped_*.jpg"),
                                ("image_cache", "img_*.jpg")]:
            for path in (save_dir / folder).glob(pattern):
                path.unlink()
            (save_dir / folder / "source_mapping.json").unlink(missing_ok=True)
        (save_dir / "compiler_output.txt").unlink(missing_ok=True)
        for name in ("pair_debug.log", "ledger.txt", "drafts.json", "compiled.json"):
            (save_dir / name).unlink(missing_ok=True)
        checkpoint = {"inputs": inputs, "run_id": uuid4().hex, "groups": []}
        _write_json(checkpoint_path, checkpoint)
        _append_record(records_path, {"event": "run", "run_id": checkpoint["run_id"],
                                      "started_at": _now(), "inputs": inputs})
    else:
        _write_json(checkpoint_path, checkpoint)
    completed = checkpoint["groups"]
    _write_text(save_dir / "generator_output.txt", _format_generator_output(completed))
    mapping_path = save_dir / "generator_images/source_mapping.json"
    mapping = {"cell_long_side": 320, "groups": []}
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    previous = [f"[G{g}] {item['action']}" for g, value in enumerate(completed) for item in value["actions"]]
    stage = "generator"
    try:
        for group, images in enumerate(groups):
            if group < len(completed):
                continue
            with _IMAGE_LOCK:
                attachments = _prepare_generator_images(images, save_dir / "generator_images", f"group{group}")
            entry = {"group": group, "comparison": str(attachments[0].resolve()),
                     "frames": [{"label": f"K{i}", "image_id": _frame_id(path), "source": str(path.resolve()),
                                 "comparison_cell": i, "attachment_index": i+1,
                                 "attachment": str(attachments[i+1].resolve())} for i, path in enumerate(images)],
                     "attachments": [str(path.resolve()) for path in attachments]}
            mapping["groups"] = [value for value in mapping["groups"] if value["group"] != group] + [entry]
            _write_json(mapping_path, mapping)
            frame_order = ("The first image is the comparison grid; following images show the same moments "
                           "individually. Read K0 through " + f"K{len(images)-1}; repeated views are not new moments.")
            result = _call_json(model_name, generator.format(shared_rules=common, instruction=instruction,
                                group=group, last_group=len(groups)-1, frame_order=frame_order,
                                previous_actions="\n".join(previous) or "(none)"), attachments,
                                lambda data, count=len(images): normalize_group_result(data, count),
                                records_path=records_path, label=f"[G{group}]", run_id=checkpoint["run_id"])
            completed.append(result)
            previous.extend(f"[G{group}] {item['action']}" for item in result["actions"])
            _write_json(checkpoint_path, checkpoint)
            _write_text(save_dir / "generator_output.txt", _format_generator_output(completed))
        stage = "compiler"
        with _IMAGE_LOCK:
            attachments, frame_order = _prepare_compiler_images(
                groups, save_dir / "compiler_images", frame_size=compiler_frame_size)
        if "compiled" in checkpoint:
            compiled = normalize_compiler_result(checkpoint["compiled"], completed, groups)
        else:
            compiled = _call_json(model_name, compiler.format(shared_rules=common, instruction=instruction,
                                  generator_output=_format_generator_output(completed, groups), frame_order=frame_order,
                                  group_keys=", ".join(f"G{i}" for i in range(len(groups)))), attachments,
                                  lambda data: normalize_compiler_result(data, completed, groups),
                                  records_path=records_path, label="[COMPILER]", run_id=checkpoint["run_id"])
            checkpoint["compiled"] = compiled
            _write_json(checkpoint_path, checkpoint)
        _write_text(save_dir / "compiler_output.txt", _format_compiler_output(compiled))
        final = [(group, action["action"]) for group, value in enumerate(compiled.values()) for action in value["actions"]]
        _append_record(records_path, {"event": "complete", "run_id": checkpoint["run_id"], "finished_at": _now(),
                                      "action_count": len(final)})
        checkpoint_path.unlink(missing_ok=True)
        _write_text(actions_path, "".join(f"[G{group}] {action}\n" for group, action in final))
        return groups[0][0], [action for _, action in final]
    except BaseException as error:
        actions_path.unlink(missing_ok=True)
        _write_text(save_dir / "compiler_output.txt", f"status: failed\nstage: {stage}\nerror: {type(error).__name__}: {error}\n")
        _append_record(records_path, {"event": "failed", "run_id": checkpoint["run_id"], "stage": stage,
                                      "finished_at": _now(), "error_type": type(error).__name__, "error": str(error)})
        raise
