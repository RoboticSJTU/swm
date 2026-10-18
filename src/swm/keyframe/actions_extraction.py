"""Extract grounded actions and states: ordered Generator groups, one Compiler."""

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from threading import Lock

from PIL import Image, ImageDraw, ImageFont

from ..llm import call_gpt

_IMAGE_LOCK = Lock()

RETRY_BACKOFF_MAX_SECONDS = 30
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)
PAIR_RE = re.compile(r"^K(\d+)\s*(?:->|→)\s*K(\d+)$", re.IGNORECASE)
EVIDENCE_RE = re.compile(r"\bG(\d+)\s*:?\s+img_(\d+)\s*(?:->|→)\s*img_(\d+)\b", re.IGNORECASE)


class OutputContractError(ValueError):
    pass


def _normalize_pair(value: str, label: str) -> tuple[str, int, int]:
    match = PAIR_RE.fullmatch(value.strip()) if isinstance(value, str) else None
    if not match:
        raise OutputContractError(f"{label} must be an adjacent pair such as K0->K1")
    start, end = map(int, match.groups())
    if end != start + 1:
        raise OutputContractError(f"{label} must reference adjacent images")
    return f"K{start}->K{end}", start, end


def _normalize_action_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value or "|" in value:
        raise OutputContractError("action must be one non-empty line")
    action = re.sub(r"^\s*[-*•]\s+", "", value.strip())
    action = re.sub(r"^\s*\(?\d+\)?[.)]\s+", "", action).strip()
    action = re.sub(r"\\([.,;:!?])", r"\1", action)
    key = action.lower().strip(".")
    if not action or key in {"none", "continuation"} or key.startswith("continuation of "):
        raise OutputContractError("action must name a manipulation event")
    return action if action.endswith(".") else action + "."


def _normalize_state_text(value: str) -> str:
    if not isinstance(value, str) or "|" in value:
        raise OutputContractError("before and after must be strings without '|'")
    return " ".join(value.split())


def format_step(item: dict) -> str:
    parts = [item["action"]]
    for field in ("before", "after"):
        if item[field]:
            parts.append(f"{field.title()}: {item[field]}")
    return " | ".join(parts)


def normalize_group_result(data: dict, frame_count: int) -> dict:
    if not isinstance(data, dict) or set(data) != {"changes", "actions"}:
        raise OutputContractError("generator output must contain only changes and actions")
    raw_changes = data["changes"]
    raw_actions = data["actions"]
    if not isinstance(raw_changes, list) or not isinstance(raw_actions, list):
        raise OutputContractError("generator changes and actions must be lists")

    expected_pairs = [f"K{index}->K{index + 1}" for index in range(frame_count - 1)]
    changes = []
    for index, item in enumerate(raw_changes, 1):
        if not isinstance(item, dict) or set(item) != {"pair", "change"}:
            raise OutputContractError(f"change {index} must contain only pair and change")
        pair, _, _ = _normalize_pair(item["pair"], f"change {index} pair")
        change = item["change"]
        if not isinstance(change, str) or not change.strip():
            raise OutputContractError(f"change {index} must contain non-empty text")
        changes.append({"pair": pair, "change": change.strip()})
    if [item["pair"] for item in changes] != expected_pairs:
        raise OutputContractError("changes must cover every adjacent pair exactly once in order")

    pair_positions = {pair: index for index, pair in enumerate(expected_pairs)}
    actions = []
    for index, item in enumerate(raw_actions, 1):
        if (not isinstance(item, dict) or set(item) != {"action", "before", "after", "evidence"}
            or not isinstance(item["evidence"], list)):
            raise OutputContractError(
                f"action {index} must contain only action, before, after, and evidence"
            )
        evidence = [
            _normalize_pair(value, f"action {index} evidence")[0]
            for value in item["evidence"]
        ]
        if not evidence or len(evidence) != len(set(evidence)):
            raise OutputContractError(f"action {index} evidence must be non-empty and unique")
        if any(pair not in pair_positions for pair in evidence):
            raise OutputContractError(f"action {index} cites a pair outside this group")
        evidence.sort(key=pair_positions.__getitem__)
        actions.append(
            {
                "action": _normalize_action_text(item["action"]),
                "before": _normalize_state_text(item["before"]),
                "after": _normalize_state_text(item["after"]),
                "evidence": evidence,
            }
        )

    first_evidence = [pair_positions[item["evidence"][0]] for item in actions]
    if any(left > right for left, right in zip(first_evidence, first_evidence[1:])):
        raise OutputContractError("generator actions must be in temporal order")
    return {"changes": changes, "actions": actions}


def normalize_compiler_result(data: dict, drafts: list[dict], segment_images: list[list[Path]]) -> dict:
    if not isinstance(data, dict) or set(data) != {"removed", "actions"}:
        raise OutputContractError("compiler output must contain removed and actions")
    if not isinstance(data["removed"], list) or not isinstance(data["actions"], list):
        raise OutputContractError("compiler removed and actions must be lists")
    if not data["actions"]:
        raise OutputContractError("the complete episode needs a non-empty supported action sequence")
    originals = {item["id"]: item for item in drafts}
    spans = {
        (group, int(start.stem), int(end.stem))
        for group, images in enumerate(segment_images)
        for index, start in enumerate(images)
        for end in images[index + 1:]
    }

    def evidence(value: str, required: bool) -> str:
        if not isinstance(value, str):
            raise OutputContractError("reason must be a string")
        reason = " ".join(value.split())
        if not reason and not required:
            return ""
        citations = [tuple(map(int, match)) for match in EVIDENCE_RE.findall(reason)]
        if not citations:
            raise OutputContractError(
                f"Changed item lacks a span citation: {reason!r}. "
                "Copy a complete G# img_N->img_M span from the ledger, then explain the observed fact."
            )
        invalid = [citation for citation in citations if citation not in spans]
        if invalid:
            raise OutputContractError(
                f"Invalid group/frame spans: {invalid}. "
                "Copy spans from the ledger; each span must stay within its stated group."
            )
        return reason

    removed = []
    removed_ids = set()
    for item in data["removed"]:
        if not isinstance(item, dict) or set(item) != {"draft", "reason"}:
            raise OutputContractError("each removal needs draft and reason")
        draft_id = item["draft"]
        if type(draft_id) is not int or draft_id not in originals or draft_id in removed_ids:
            raise OutputContractError("removed draft IDs must be known and unique")
        removed.append({"draft": draft_id, "reason": evidence(item["reason"], True)})
        removed_ids.add(draft_id)

    actions = []
    uses = Counter()
    for index, item in enumerate(data["actions"], 1):
        if not isinstance(item, dict) or set(item) != {"drafts", "reason", "group", "action", "before", "after"}:
            raise OutputContractError(f"compiled action {index} needs drafts, reason, group, action, before and after")
        sources = item["drafts"]
        if (not isinstance(sources, list)
                or any(type(source) is not int or source not in originals for source in sources)
                or len(sources) != len(set(sources))):
            raise OutputContractError(f"compiled action {index} has invalid source Draft IDs")
        group = item["group"]
        if type(group) is not int or not 0 <= group < len(segment_images):
            raise OutputContractError(f"compiled action {index} has an invalid group")
        actions.append({
            "drafts": sources,
            "reason": item["reason"],
            "group": group,
            "action": _normalize_action_text(item["action"]),
            "before": _normalize_state_text(item["before"]),
            "after": _normalize_state_text(item["after"]),
        })
        uses.update(sources)
    missing = set(originals) - set(uses)
    if removed_ids != missing:
        raise OutputContractError(
            f"unrepresented Drafts must be explicitly removed with evidence; "
            f"missing removal IDs={sorted(missing - removed_ids)}, still-used removal IDs={sorted(removed_ids - missing)}"
        )
    fields = ("group", "action", "before", "after")
    reordered = set()
    for position, left in enumerate(actions):
        for right in actions[position + 1:]:
            if any(a > b for a in left["drafts"] for b in right["drafts"]):
                reordered.update(left["drafts"])
                reordered.update(right["drafts"])
    for item in actions:
        sources = item["drafts"]
        unchanged = (len(sources) == 1 and uses[sources[0]] == 1
                     and sources[0] not in reordered
                     and all(item[field] == originals[sources[0]][field] for field in fields))
        item["reason"] = evidence(item["reason"], not unchanged)
    if any(first["group"] > second["group"] for first, second in zip(actions, actions[1:])):
        raise OutputContractError("compiled actions must have nondecreasing groups")
    return {"removed": removed, "actions": actions}


def _write_trace(path: Path, trace: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _call_json(
    model: str,
    prompt: str,
    images: list[Path],
    normalize,
    *,
    trace_path: Path | None = None,
    trace_images: list[Path] | None = None,
) -> dict:
    last_failure = ""
    trace = {
        "model": model,
        "prompt": prompt,
        "images": [str(path.resolve()) for path in (trace_images or images)],
        "request_images": [str(path.resolve()) for path in images],
        "retry_policy": "until_success",
        "retry_backoff_max_seconds": RETRY_BACKOFF_MAX_SECONDS,
        "input_order": "text_then_images",
        "attempts": [],
    }
    trace["image_sizes"] = []
    for path in images:
        with Image.open(path) as image:
            trace["image_sizes"].append(list(image.size))
    for attempt in count(1):
        raw_output = None
        parsed_output = None
        json_text = ""
        api_capture = {"reasoning_effort": "medium", "temperature": 0}
        feedback = f"Attempt {attempt - 1} failed: {last_failure}" if last_failure else ""
        request_prompt = prompt + (
            f"\n\n{feedback}\nRegenerate ONE complete valid JSON object for the entire original scope. "
            "Include all required fields and exactly matching brackets/braces; do not append another root object."
            if feedback else ""
        )
        record = {
            "attempt": attempt, "retry_feedback": feedback, "request_prompt": request_prompt,
            "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_output": None, "parsed_output": None, "api": api_capture,
        }
        trace["attempts"].append(record)
        if trace_path is not None:
            _write_trace(trace_path, trace)
        try:
            raw_output = call_gpt(
                model,
                request_prompt,
                images,
                capture=api_capture,
                reasoning_effort="medium",
                temperature=0,
            )
            if api_capture.get("finish_reason") == "length":
                raise OutputContractError(
                    "Response hit the token limit before a complete JSON result. "
                    "Keep the entire scope and all events; shorten evidence reasons and states."
                )
            if not isinstance(raw_output, str):
                raise OutputContractError("Response has no JSON text; return the complete required JSON object.")
            if api_capture.get("finish_reason") not in (None, "stop"):
                raise OutputContractError("Response did not finish normally; return the complete required JSON object.")
            text = raw_output.strip()
            match = JSON_FENCE_RE.fullmatch(text)
            json_text = match.group(1).strip() if match else text
            parsed_output = json.loads(json_text)
            result = normalize(parsed_output)
        except Exception as error:  # noqa: BLE001 - retry model and contract failures
            if isinstance(error, json.JSONDecodeError):
                context = json_text[max(0, error.pos - 80):error.pos + 80]
                last_failure = f"{error}. JSON error context: {context!r}"
            elif isinstance(error, OutputContractError):
                last_failure = str(error)
            else:
                last_failure = "The request/response did not complete successfully."
            record.update(status="error", raw_output=raw_output, parsed_output=parsed_output,
                          error_type=type(error).__name__, error=str(error))
            if trace_path is not None:
                _write_trace(trace_path, trace)
            delay = min(RETRY_BACKOFF_MAX_SECONDS, 2 ** min(attempt - 1, 5))
            print(f"Request attempt {attempt} failed ({type(error).__name__}); retrying in {delay}s.", flush=True)
            time.sleep(delay)
            continue
        record.update(status="ok", raw_output=raw_output, parsed_output=parsed_output,
                      normalized_output=result)
        if trace_path is not None:
            _write_trace(trace_path, trace)
        return result


def _frame_id(path: Path) -> str:
    return f"img_{int(path.stem)}"


def _frame_span(pair: str, frame_ids: list[str]) -> str:
    _, start, end = _normalize_pair(pair, "evidence pair")
    return f"{frame_ids[start]}->{frame_ids[end]}"


def _prepare_generator_images(images: list[Path], directory: Path, prefix: str) -> list[Path]:
    if len(images) not in {3, 4, 5, 6}:
        raise ValueError("Generator comparison needs 3..6 chronological keyframes")
    cell_long_side = 320
    directory.mkdir(parents=True, exist_ok=True)
    shapes = []
    for source in images:
        with Image.open(source) as image:
            shapes.append(image.size)
    portrait = sum(height > width for width, height in shapes) > len(shapes) / 2
    width, height = cell_long_side, cell_long_side * 9 // 16
    if portrait:
        width, height = height, width
    header = cell_long_side * 3 // 40
    if len(images) == 3:
        rows, columns = 1, 3
    elif len(images) == 4:
        rows, columns = 2, 2
    else:
        rows, columns = (2, 3) if portrait else (3, 2)
    prepared = list(images)
    if len(images) >= 5:
        prepared = []
        for index, source in enumerate(images):
            target = directory / f"{prefix}_{index}.jpg"
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                image.save(target, "JPEG", quality=90, optimize=True)
            prepared.append(target)
    timeline = Image.new("RGB", (columns * width, rows * (height + header)), (24, 24, 24))
    draw = ImageDraw.Draw(timeline)
    font = ImageFont.truetype("DejaVuSans.ttf", cell_long_side // 20)
    cells = []
    for index, source in enumerate(images):
        row, column = divmod(index, columns)
        x, y = column * width, row * (height + header)
        draw.text((x + 3, y + 1), f"K{index}", font=font, fill="white")
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            timeline.paste(image, (x + (width - image.width) // 2,
                                   y + header + (height - image.height) // 2))
        cells.append({"label": f"K{index}", "frame": int(source.stem),
                      "source": str(source.resolve()), "source_size": list(shapes[index]),
                      "individual_image": str(prepared[index].resolve()), "row": row, "column": column})
    overview = directory / f"{prefix}_timeline.jpg"
    timeline.save(overview, "JPEG", quality=90, optimize=True)
    (directory / f"{prefix}_layout.json").write_text(json.dumps({
        "mode": "group_comparison", "orientation": "portrait" if portrait else "landscape",
        "rows": rows, "columns": columns, "scene_cell_size": [width, height], "scene_long_side": cell_long_side,
        "header_height": header, "font_size": font.size, "image_size": list(timeline.size),
        "blank_cells": rows * columns - len(images), "spatial_crops": False, "cells": cells,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [overview, *prepared]


def _plan_compiler_images(groups):
    """Normalize complete groups to three-frame rows, then pack fixed-size pages."""
    frames, group_plan, rows = {}, [], []
    for group, paths in enumerate(groups):
        ids = [int(path.stem) for path in paths]
        if len(ids) not in {3, 4, 5, 6} or ids != sorted(set(ids)):
            raise ValueError("grouped_3x3 needs groups of 3..6 distinct chronological frames")
        for frame, path in zip(ids, paths):
            entry = frames.setdefault(frame, {"frame": frame, "source": str(path.resolve()), "groups": []})
            entry["groups"].append(group)
        selected = ids
        if len(ids) == 4:
            middle = min(ids[1:3], key=lambda n: (abs(2 * n - ids[0] - ids[-1]), n))
            selected = [ids[0], middle, ids[-1]]
        chunks = [selected] if len(selected) == 3 else (
            [selected[:3], selected[2:]] if len(selected) == 5 else [selected[:3], selected[3:]])
        start = len(rows)
        for part, chunk in enumerate(chunks, 1):
            rows.append({"row_index": len(rows), "group": group, "part": part,
                         "parts": len(chunks), "frames": chunk})
        group_plan.append({"group": group, "original_frames": ids,
                           "selected_frames": selected, "omitted_frames": sorted(set(ids) - set(selected)),
                           "row_start": start, "row_end": len(rows)})
    if not rows:
        raise ValueError("No groups to display")
    ordered = [{"time_index": index, **frames[frame]} for index, frame in enumerate(sorted(frames))]
    selected_ids = {frame for row in rows for frame in row["frames"]}
    pages, start = [], 0
    while start < len(group_plan):
        end, used = start, 0
        while end < len(group_plan):
            count = group_plan[end]["row_end"] - group_plan[end]["row_start"]
            if used + count > 3:
                break
            used += count
            end += 1
        first, last = group_plan[start]["row_start"], group_plan[end - 1]["row_end"]
        missing = 3 - used
        before = rows[max(0, first - missing):first]
        after = rows[last:last + missing - len(before)]
        # An episode with fewer than three rows has no outside context to add.
        padding = [rows[0]] * (missing - len(before) - len(after))
        page_rows = [{**r, "context": True} for r in padding + before]
        page_rows += [{**r, "context": False} for r in rows[first:last]]
        page_rows += [{**r, "context": True} for r in after]
        assert len(page_rows) == 3 and all(len(r["frames"]) == 3 for r in page_rows)
        pages.append({"page": len(pages) + 1, "groups": list(range(start, end)), "rows": page_rows})
        start = end
    return {"all_frames": ordered, "frames": [f for f in ordered if f["frame"] in selected_ids],
            "groups": group_plan, "rows": rows, "pages": pages}


def _prepare_compiler_images(groups, directory):
    """Use one fixed orientation per episode; cell_width bounds the long side."""
    cell_width = 320
    plan = _plan_compiler_images(groups)
    directory.mkdir(parents=True, exist_ok=True)
    orientation_counts = {"landscape": 0, "portrait": 0}
    for frame in plan["all_frames"]:
        with Image.open(frame["source"]) as image:
            orientation_counts["portrait" if image.height > image.width else "landscape"] += 1
    # Decide once for the episode, never per page. Mixed inputs retain all content;
    # the majority of unique moments chooses the template, with landscape on a tie.
    orientation = ("portrait" if orientation_counts["portrait"] > orientation_counts["landscape"]
                   else "landscape")
    width, height, header = cell_width, cell_width * 9 // 16, cell_width * 3 // 40
    if orientation == "portrait":
        width, height = height, width
    font = ImageFont.truetype("DejaVuSans.ttf", cell_width // 20)
    lookup = {frame["frame"]: frame for frame in plan["all_frames"]}
    prepared, pages = [], []
    layout = [
        "Each image is a 3x3 timeline; read left to right, then top to bottom. "
        "G# identifies a group, img_N a moment, and * repeated context. "
        "Repeated img_N is the same moment. Labels match the ledger.",
    ]
    for page in plan["pages"]:
        canvas = Image.new("RGB", (3 * width, 3 * (height + header)), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        cells, descriptions = [], []
        for row, entry in enumerate(page["rows"]):
            is_bridge = len(plan["groups"][entry["group"]]["original_frames"]) == 5 and entry["part"] == 2
            for column, frame_id in enumerate(entry["frames"]):
                frame = lookup[frame_id]
                group_label = f"G{entry['group']}" + ("*" if entry["context"] else "")
                frame_label = f"img_{frame_id}"
                label = f"{group_label} | {frame_label}" if column == 0 else frame_label
                x, y = column * width, row * (height + header)
                draw.text((x + max(3, width // 120), y + 1), label, font=font, fill="white")
                with Image.open(frame["source"]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((width, height), Image.Resampling.LANCZOS)
                    canvas.paste(image, (x + (width - image.width) // 2,
                                         y + header + (height - image.height) // 2))
                cells.append({**frame, "group": entry["group"], "part": entry["part"],
                              "context": entry["context"], "bridge_repeat": is_bridge and column == 0,
                              "row": row, "column": column, "label": label})
            descriptions.append(f"row {row + 1} {group_label} "
                                + "[" + ",".join(f"img_{n}" for n in entry["frames"]) + "]")
        target = directory / f"grouped_{page['page']:02d}.jpg"
        canvas.save(target, "JPEG", quality=90, optimize=True)
        prepared.append(target)
        pages.append({**page, "cells": cells})
        layout.append(f"Image {page['page']}: " + "; ".join(descriptions) + ".")
    layout.append(f"Episode final frame: img_{plan['all_frames'][-1]['frame']}.")
    provenance = {"mode": "grouped_3x3", "scene_cell_size": [width, height],
                  "orientation": orientation, "source_orientation_counts": orientation_counts,
                  "header_height": header, "image_size": [3 * width, 3 * (height + header)],
                  "label_style": "row_group_frame", "frame_label_prefix": "img_", "label_scale": 2, "font_size": font.size,
                  "grid_only": True, "spatial_crops": False, "frames": plan["frames"],
                  "all_frames": plan["all_frames"], "groups": plan["groups"], "pages": pages,
                  "group_frames": [[str(p.resolve()) for p in group] for group in groups]}
    (directory / "source_mapping.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [Path(frame["source"]) for frame in plan["frames"]], prepared, "\n".join(layout)


def extract_keyframe_actions(
    model_name: str,
    keyframe_dir: Path,
    instruction: str,
    save_dir: Path,
    trace_dir: Path | None = None,
) -> tuple[Path, list[str]]:
    groups = [sorted(segment.glob("*.png"), key=lambda p: int(p.stem))
              for segment in sorted(keyframe_dir.glob("seg_*"))]
    groups = [images for images in groups if images]
    if not groups:
        raise ValueError(f"No keyframes found in {keyframe_dir}")
    if any(len(images) not in {3, 4, 5, 6} for images in groups):
        raise ValueError("Action extraction needs 3..6 chronological keyframes per group")
    save_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = trace_dir if trace_dir is not None else save_dir / "action_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    actions_path = save_dir / "kf_actions.txt"
    actions_path.unlink(missing_ok=True)
    prompt_dir = Path(__file__).parents[1] / "prompt_templates"
    common = (prompt_dir / "kf_actions_common.txt").read_text(encoding="utf-8").strip()
    generator = (prompt_dir / "kf_actions_generator.txt").read_text(encoding="utf-8")
    compiler = (prompt_dir / "kf_actions_compiler.txt").read_text(encoding="utf-8")
    previous, ledger, drafts = [], [], []
    for group, images in enumerate(groups):
        with _IMAGE_LOCK:
            attachments = _prepare_generator_images(images, save_dir / "generator_images", f"group{group}")
        frame_ids = [_frame_id(path) for path in images]
        frame_order = (
            "The first image compares this group; following images show the same frames individually. "
            "Read increasing K labels. Repeated views are the same moments. "
            + ", ".join(f"K{i}={frame}" for i, frame in enumerate(frame_ids)) + "."
        )
        result = _call_json(model_name, generator.format(
            shared_rules=common, instruction=instruction, group=group, last_group=len(groups)-1,
            frame_order=frame_order, previous_actions="\n".join(previous) or "(none)"),
            attachments, lambda value, count=len(images): normalize_group_result(value, count),
            trace_path=trace_dir / f"generator_g{group:02d}_retry00.json", trace_images=images)
        ledger.extend((f"[G{group}]", "Observations:"))
        ledger.extend(f"- G{group} {_frame_span(item['pair'], frame_ids)}: {item['change']}"
                      for item in result["changes"])
        ledger.append("Drafts:")
        if not result["actions"]:
            ledger.append("- (none)")
        for item in result["actions"]:
            previous.append(f"[G{group}] {item['action']}")
            evidence = [f"G{group} {_frame_span(pair, frame_ids)}" for pair in item["evidence"]]
            drafts.append(dict(id=len(drafts)+1, group=group,
                               **{k: item[k] for k in ("action", "before", "after")}))
            ledger.append(f"- D{len(drafts)} | {', '.join(evidence)}: {format_step(item)}")
        ledger.append("")
        (trace_dir / "evidence_ledger.txt").write_text("\n".join(ledger).rstrip()+"\n", encoding="utf-8")
    with _IMAGE_LOCK:
        originals, attachments, frame_order = _prepare_compiler_images(groups, save_dir / "compiler_images")
    (save_dir / "compiler_images/layout.txt").write_text(frame_order+"\n", encoding="utf-8")
    compiled = _call_json(model_name, compiler.format(
        shared_rules=common, instruction=instruction, evidence_ledger="\n".join(ledger).rstrip(),
        example_span=f"G0 {_frame_id(groups[0][0])}->{_frame_id(groups[0][-1])}", frame_order=frame_order),
        attachments, lambda value: normalize_compiler_result(value, drafts, groups),
        trace_path=trace_dir / "compiler.json", trace_images=originals)
    _write_trace(trace_dir / "compiled.json", compiled)
    steps = [format_step(item) for item in compiled["actions"]]
    actions_path.write_text("".join(f"[G{a['group']}] {step}\n" for a, step in zip(compiled["actions"], steps)), encoding="utf-8")
    return groups[0][0], steps
