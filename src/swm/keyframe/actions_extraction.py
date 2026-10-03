"""Keyframe-based Action Sequence Extraction."""

import json
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from ..llm import call_gpt

API_ATTEMPTS = 20
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)
PAIR_RE = re.compile(r"^K(\d+)\s*(?:->|→)\s*K(\d+)$", re.IGNORECASE)
GROUP_REF_RE = re.compile(r"\bG\d+\b", re.IGNORECASE)
FRAME_SPAN_RE = re.compile(
    r"\bimg_\d+\s*(?:->|→)\s*img_\d+\b",
    re.IGNORECASE,
)


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
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise OutputContractError("action must be one non-empty line")
    action = re.sub(r"^\s*[-*•]\s+", "", value.strip())
    action = re.sub(r"^\s*\(?\d+\)?[.)]\s+", "", action).strip()
    action = re.sub(r"\\([.,;:!?])", r"\1", action)
    key = action.lower().strip(".")
    if not action or key in {"none", "continuation"} or key.startswith("continuation of "):
        raise OutputContractError("action is not atomic")
    return action if action.endswith(".") else action + "."


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
        if (not isinstance(item, dict) or set(item) != {"action", "evidence"}
            or not isinstance(item["evidence"], list)):
            raise OutputContractError(
                f"action {index} must contain only action and evidence"
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
                "evidence": evidence,
            }
        )

    first_evidence = [pair_positions[item["evidence"][0]] for item in actions]
    if any(left > right for left, right in zip(first_evidence, first_evidence[1:])):
        raise OutputContractError("generator actions must be in temporal order")
    return {"changes": changes, "actions": actions}


def normalize_compiler_result(data: dict, drafts: list[dict], group_count: int) -> dict:
    if not isinstance(data, dict) or set(data) != {"decisions"}:
        raise OutputContractError("compiler output must contain only decisions")
    raw_decisions = data["decisions"]
    if not isinstance(raw_decisions, list):
        raise OutputContractError("compiler decisions must be a list")

    draft_ids = {item["id"] for item in drafts}
    draft_groups = {item["id"]: item["group"] for item in drafts}
    inserts = {}
    changes = {}
    decisions = []
    decision_fields = {"reason", "op", "draft", "group", "action"}
    for index, item in enumerate(raw_decisions, 1):
        if not isinstance(item, dict) or set(item) != decision_fields:
            raise OutputContractError(
                f"compiler decision {index} must have the fixed five fields"
            )
        operation = item["op"]
        if operation not in {"insert", "replace", "delete"}:
            raise OutputContractError(f"compiler decision {index} has an invalid op")
        reason = item["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise OutputContractError(f"compiler decision {index} needs a reason")
        reason = " ".join(reason.split())
        if not GROUP_REF_RE.search(reason) or not FRAME_SPAN_RE.search(reason):
            raise OutputContractError(
                f"compiler decision {index} reason must cite G# and img_N->img_M"
            )

        draft_id = item["draft"]
        group = item["group"]
        if type(draft_id) is not int or type(group) is not int:
            raise OutputContractError(
                f"compiler decision {index} draft and group must be integers"
            )
        if not 0 <= group < group_count:
            raise OutputContractError(f"compiler decision {index} has an invalid group")

        if operation == "insert":
            if draft_id not in draft_ids | {0}:
                raise OutputContractError(f"compiler insert {index} has an invalid position")
            action = _normalize_action_text(item["action"])
            inserts.setdefault(draft_id, []).append({"group": group, "action": action})
        else:
            if draft_id not in draft_ids or draft_id in changes:
                raise OutputContractError(
                    f"compiler decision {index} has an invalid draft id"
                )
            if group != draft_groups[draft_id]:
                raise OutputContractError(
                    f"compiler decision {index} group does not match its draft"
                )
            if operation == "replace":
                action = _normalize_action_text(item["action"])
            elif item["action"] is not None:
                raise OutputContractError(
                    f"compiler delete {index} action must be null"
                )
            else:
                action = None

        decision = {"reason": reason, "op": operation, "draft": draft_id,
                    "group": group, "action": action}
        if operation != "insert":
            changes[draft_id] = decision
        decisions.append(decision)

    actions = list(inserts.get(0, []))
    for draft in drafts:
        decision = changes.get(draft["id"])
        if decision is None or decision["op"] == "replace":
            actions.append({"group": draft["group"],
                            "action": draft["action"] if decision is None else decision["action"]})
        actions.extend(inserts.get(draft["id"], []))

    if any(first["group"] > second["group"] for first, second in zip(actions, actions[1:])):
        raise OutputContractError("compiler decisions produce invalid chronological groups")
    return {"decisions": decisions, "actions": actions}


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
    last_error = None
    trace = {
        "model": model,
        "prompt": prompt,
        "images": [str(path.resolve()) for path in (trace_images or images)],
        "attempts": [],
    }
    for attempt in range(API_ATTEMPTS):
        raw_output = None
        parsed_output = None
        api_capture = {}
        try:
            raw_output = call_gpt(
                model,
                prompt,
                images,
                capture=api_capture,
            )
            text = raw_output.strip()
            match = JSON_FENCE_RE.fullmatch(text)
            parsed_output = json.loads(match.group(1).strip() if match else text)
            result = normalize(parsed_output)
            trace["attempts"].append(
                {
                    "attempt": attempt + 1,
                    "status": "ok",
                    "raw_output": raw_output,
                    "parsed_output": parsed_output,
                    "normalized_output": result,
                    "api": api_capture,
                }
            )
            if trace_path is not None:
                _write_trace(trace_path, trace)
            return result
        except Exception as error:  # noqa: BLE001 - retry model and contract failures
            last_error = error
            trace["attempts"].append(
                {
                    "attempt": attempt + 1,
                    "status": "error",
                    "raw_output": raw_output,
                    "parsed_output": parsed_output,
                    "api": api_capture,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if trace_path is not None:
                _write_trace(trace_path, trace)
            if attempt + 1 < API_ATTEMPTS:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(
        f"model did not satisfy the output contract after {API_ATTEMPTS} attempts"
    ) from last_error


def _frame_id(path: Path) -> str:
    return f"img_{int(path.stem)}"


def _frame_span(pair: str, frame_ids: list[str]) -> str:
    _, start, end = _normalize_pair(pair, "evidence pair")
    return f"{frame_ids[start]}->{frame_ids[end]}"


def extract_keyframe_actions(
    model_name: str,
    keyframe_dir: Path,
    instruction: str,
    save_dir: Path,
    trace_dir: Path | None = None,
) -> tuple[Path, list[str]]:
    segment_images = []
    for segment_dir in sorted(keyframe_dir.glob("seg_*")):
        images = sorted(segment_dir.glob("*.png"), key=lambda path: int(path.stem))
        if images:
            segment_images.append(images)
    if not segment_images:
        raise ValueError(f"No keyframes found in {keyframe_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)
    actions_path = save_dir / "kf_actions.txt"
    actions_path.unlink(missing_ok=True)

    prompt_dir = Path(__file__).parents[1] / "prompt_templates"
    generator_template = (prompt_dir / "kf_actions_generator.txt").read_text(
        encoding="utf-8"
    )
    compiler_template = (prompt_dir / "kf_actions_compiler.txt").read_text(
        encoding="utf-8"
    )
    prior_actions = []
    ledger = []
    drafts = []

    with TemporaryDirectory() as temporary_dir:
        temporary_dir = Path(temporary_dir)
        for group, images in enumerate(segment_images):
            if len(images) < 5:
                request_images = images
            else:
                request_images = []
                for image_path in images:
                    target = temporary_dir / f"{group}_{image_path.stem}.jpg"
                    with Image.open(image_path) as image:
                        image = image.convert("RGB")
                        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                        image.save(target, "JPEG", quality=85, optimize=True)
                    request_images.append(target)

            frame_order = " -> ".join(f"K{index}" for index in range(len(images)))
            frame_ids = [_frame_id(path) for path in images]
            prompt = generator_template.format(
                instruction=instruction,
                group=group,
                last_group=len(segment_images) - 1,
                frame_order=frame_order,
                previous_actions="\n".join(prior_actions) or "(none)",
            )
            result = _call_json(
                model_name,
                prompt,
                request_images,
                lambda value, count=len(images): normalize_group_result(value, count),
                trace_path=(trace_dir / f"generator_g{group:02d}_retry00.json"
                            if trace_dir is not None else None),
                trace_images=images,
            )
            ledger.extend((f"[G{group}]", "Observations:"))
            for item in result["changes"]:
                ledger.append(
                    f"- {_frame_span(item['pair'], frame_ids)}: {item['change']}"
                )
            ledger.append("Drafts:")
            if not result["actions"]:
                ledger.append("- (none)")
            for item in result["actions"]:
                prior_actions.append(f"[G{group}] {item['action']}")
                evidence = [_frame_span(pair, frame_ids) for pair in item["evidence"]]
                drafts.append(
                    {
                        "id": len(drafts) + 1,
                        "group": group,
                        "action": item["action"],
                        "evidence": evidence,
                    }
                )
                ledger.append(
                    f"- D{len(drafts)} | {', '.join(evidence)}: {item['action']}"
                )
            ledger.append("")
            if trace_dir is not None:
                trace_dir.mkdir(parents=True, exist_ok=True)
                (trace_dir / "evidence_ledger.txt").write_text(
                    "\n".join(ledger).rstrip() + "\n", encoding="utf-8"
                )

    first_image = segment_images[0][0]
    compiled = _call_json(
        model_name,
        compiler_template.format(
            instruction=instruction,
            evidence_ledger="\n".join(ledger).rstrip(),
            initial_frame=_frame_id(first_image),
        ),
        [first_image],
        lambda value: normalize_compiler_result(value, drafts, len(segment_images)),
        trace_path=trace_dir / "compiler.json" if trace_dir is not None else None,
    )
    if trace_dir is not None:
        (trace_dir / "compiled.json").write_text(
            json.dumps(compiled, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    actions_path.write_text(
        "".join(
            f"[G{item['group']}] {item['action']}\n" for item in compiled["actions"]
        ),
        encoding="utf-8",
    )
    return first_image, [item["action"] for item in compiled["actions"]]
