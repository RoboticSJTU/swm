"""Keyframe-based Action Sequence Extraction."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from swm.llm import call_gpt

API_ATTEMPTS = 10
MAX_HAND_RETRIES = 2
DELTA_RE = re.compile(
    r"^([+-])\s+([a-z][a-z0-9_]*)\s*\(\s*"
    r"([a-z][a-z0-9_]*(?:\s*,\s*[a-z][a-z0-9_]*)*)\s*\)\s*$",
    re.IGNORECASE,
)
PICKUP_RE = re.compile(r"\b(pick\s+up|grasp|lift)\b", re.IGNORECASE)
PLACEMENT_RE = re.compile(
    r"\b(place|put|set\s+down|return|insert)\b", re.IGNORECASE
)


class OutputContractError(ValueError):
    pass


def _normalize_delta(value: str) -> str:
    match = DELTA_RE.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise OutputContractError(
            "every state_change item must be '+ predicate' or '- predicate'"
        )
    sign, name = match.group(1), match.group(2).lower()
    arguments = tuple(part.strip().lower() for part in match.group(3).split(","))
    if name not in {"holding", "hand_free"}:
        raise OutputContractError(
            "state_change may contain only holding(hand,obj) or hand_free(hand)"
        )
    arity = 2 if name == "holding" else 1
    if len(arguments) != arity:
        raise OutputContractError(f"predicate {name} expects {arity} arguments")
    if arguments[0] not in {"left_hand", "right_hand"}:
        raise OutputContractError("hand argument must be left_hand or right_hand")
    return f"{sign} {name}({','.join(arguments)})"


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


def normalize_group_result(data: dict, frames: int | list[str]) -> dict:
    changes = data["changes"].strip()
    actions = data["actions"]
    if not changes or not isinstance(actions, list):
        raise OutputContractError("generator changes must be text and actions must be a list")

    frame_count = frames if type(frames) is int else len(frames)
    if type(frames) is not int:
        for index, (first, second) in enumerate(zip(frames, frames[1:])):
            changes = re.sub(
                rf"{re.escape(first)}\s*(?:->|→)\s*{re.escape(second)}\s*:",
                f"K{index}->K{index + 1}:",
                changes,
                flags=re.IGNORECASE,
            )
    for index in range(frame_count - 1):
        if not re.search(
            rf"K{index}\s*(?:->|→)\s*K{index + 1}\s*:", changes, re.IGNORECASE
        ):
            raise OutputContractError(f"changes must describe K{index}->K{index + 1}")

    normalized_actions = []
    for index, item in enumerate(actions, 1):
        state_change = item["state_change"]
        if not isinstance(state_change, list):
            raise OutputContractError(f"action {index} state_change must be a list")
        state_change = [_normalize_delta(value) for value in state_change]
        added, removed = _deltas_by_sign(state_change)
        if len(state_change) != len(set(state_change)) or added & removed:
            raise OutputContractError(f"action {index} has inconsistent state_change")
        normalized_actions.append(
            {"action": _normalize_action_text(item["action"]), "state_change": state_change}
        )
    return {"changes": changes, "actions": normalized_actions}


def normalize_compiler_result(data: dict, drafts: list[dict], group_count: int) -> dict:
    audit_data = data["audit"]
    raw_edits = data["edits"]
    if not isinstance(raw_edits, list) or any(
        not isinstance(audit_data[name], list)
        for name in ("coverage_issues", "draft_issues")
    ):
        raise OutputContractError("compiler audit and edits must be lists")

    audit = {
        name: [issue.strip() for issue in audit_data[name]]
        for name in ("coverage_issues", "draft_issues")
    }
    if any(not issue for issues in audit.values() for issue in issues):
        raise OutputContractError("compiler audit issues must be non-empty text")

    draft_ids = {item["id"] for item in drafts}
    inserts = {}
    changes = {}
    edits = []
    for index, item in enumerate(raw_edits, 1):
        operation = item["op"]
        if operation == "insert":
            after = item["after"]
            group = item["group"]
            if after not in draft_ids | {0} or not 0 <= group < group_count:
                raise OutputContractError(f"compiler insert {index} has an invalid position")
            action = _normalize_action_text(item["action"])
            edit = {"op": operation, "after": after, "group": group, "action": action}
            inserts.setdefault(after, []).append({"group": group, "action": action})
        elif operation in {"replace", "delete"}:
            draft_id = item["id"]
            if draft_id not in draft_ids or draft_id in changes:
                raise OutputContractError(f"compiler edit {index} has an invalid draft id")
            edit = {"op": operation, "id": draft_id}
            if operation == "replace":
                edit["action"] = _normalize_action_text(item["action"])
            changes[draft_id] = edit
        else:
            raise OutputContractError(f"compiler edit {index} has an invalid operation")
        edits.append(edit)

    if any(audit.values()) != bool(edits):
        raise OutputContractError("compiler audit and edits disagree")

    actions = list(inserts.get(0, []))
    for draft in drafts:
        edit = changes.get(draft["id"])
        if edit is None:
            actions.append({"group": draft["group"], "action": draft["action"]})
        elif edit["op"] == "replace":
            actions.append({"group": draft["group"], "action": edit["action"]})
        actions.extend(inserts.get(draft["id"], []))

    if any(first["group"] > second["group"] for first, second in zip(actions, actions[1:])):
        raise OutputContractError("compiler edits produce invalid chronological groups")
    return {"audit": audit, "edits": edits, "actions": actions}


def _deltas_by_sign(state_change: list[str]) -> tuple[set[str], set[str]]:
    return (
        {value[2:] for value in state_change if value.startswith("+ ")},
        {value[2:] for value in state_change if value.startswith("- ")},
    )


def _holdings(atoms: set[str]) -> list[tuple[str, str]]:
    return [tuple(atom[8:-1].split(",")) for atom in atoms if atom.startswith("holding(")]


def _same_held_object_label(first: str, second: str) -> bool:
    return (
        first == second
        or first.endswith(f"_{second}")
        or second.endswith(f"_{first}")
        or re.sub(r"_\d+$", "", first) == second
        or re.sub(r"_\d+$", "", second) == first
    )


def validate_hand_actions(
    actions: list[dict], current_state: set[str], allow_discontinuity: bool = False
) -> tuple[set[str] | None, str | None]:
    state = set(current_state)
    for index, item in enumerate(actions, 1):
        action = item["action"]
        prefix = f"Action {index} ({action})"
        holdings = _holdings(state)
        state_change = []
        for delta in item["state_change"]:
            if delta.startswith("- holding("):
                hand, _ = _holdings({delta[2:]})[0]
                held_objects = [obj for known_hand, obj in holdings if known_hand == hand]
                if len(held_objects) == 1:
                    delta = f"- holding({hand},{held_objects[0]})"
            state_change.append(delta)
        added, removed = _deltas_by_sign(state_change)

        if allow_discontinuity:
            for hand, obj in _holdings(removed):
                holdings = _holdings(state)
                matching = {
                    f"holding({known_hand},{known_obj})"
                    for known_hand, known_obj in holdings
                    if _same_held_object_label(obj, known_obj)
                }
                state.difference_update(
                    matching
                    or {
                        f"holding({known_hand},{known_obj})"
                        for known_hand, known_obj in holdings
                        if known_hand == hand
                    }
                )
                state.discard(f"hand_free({hand})")
            acquired_hands = {hand for hand, _ in _holdings(added)}
            state.difference_update(
                f"holding({hand},{obj})"
                for hand, obj in _holdings(state)
                if hand in acquired_hands
            )

        held = _holdings(state)
        if PICKUP_RE.search(action):
            acquired = _holdings(added)
            if not acquired:
                return None, f"{prefix}: {action} must add holding(hand,obj)"
            for hand, _ in acquired:
                free = f"hand_free({hand})"
                if any(held_hand == hand for held_hand, _ in held):
                    return None, f"{prefix}: {action} requires {free} before pickup"
                if free not in removed:
                    return None, f"{prefix}: {action} must remove {free}"

        if PLACEMENT_RE.search(action):
            released = _holdings(removed)
            if not released:
                return None, f"{prefix}: {action} must remove holding(hand,obj)"
            for hand, obj in released:
                holding = f"holding({hand},{obj})"
                free = f"hand_free({hand})"
                known_held = any(held_hand == hand for held_hand, _ in held)
                held_elsewhere = next(
                    (
                        known_hand
                        for known_hand, known_obj in held
                        if known_hand != hand
                        and _same_held_object_label(obj, known_obj)
                    ),
                    None,
                )
                if holding not in state and held_elsewhere:
                    return None, (
                        f"{prefix}: "
                        f"{action} releases {obj} from {hand}, but it is held by "
                        f"{held_elsewhere}"
                    )
                if holding not in state and (free in state or known_held):
                    return None, f"{prefix}: {action} requires {holding} before placement"
                if free not in added:
                    return None, f"{prefix}: {action} must add {free}"

        next_state = state - removed
        next_state.update(added)
        next_holdings = _holdings(next_state)
        for hand in dict.fromkeys(hand for hand, _ in next_holdings):
            objects = {obj for held_hand, obj in next_holdings if held_hand == hand}
            if f"hand_free({hand})" in next_state:
                return None, f"{prefix}: {hand} is both hand_free and holding {min(objects)}"
            if len(objects) > 1:
                return None, f"{prefix}: {hand} holds multiple objects: {', '.join(sorted(objects))}"
        state = next_state
    return state, None


def _drop_repeated_boundary_release(
    actions: list[dict], current_state: set[str], previous_action: dict | None
) -> list[dict]:
    if (
        not actions
        or previous_action is None
        or PICKUP_RE.search(actions[0]["action"])
        or actions[0]["action"].lower().rstrip(".")
        != previous_action["action"].lower().rstrip(".")
    ):
        return actions

    current = actions[0]
    _, removed = _deltas_by_sign(current["state_change"])
    _, previous_removed = _deltas_by_sign(previous_action["state_change"])
    releases = _holdings(removed)
    previous_releases = _holdings(previous_removed)
    state_holdings = _holdings(current_state)
    if not releases:
        return actions

    if PLACEMENT_RE.search(current["action"]):
        if not PLACEMENT_RE.search(previous_action["action"]) or not all(
            any(previous_hand == hand and _same_held_object_label(obj, previous_obj)
                for previous_hand, previous_obj in previous_releases)
            for hand, obj in releases
        ):
            return actions
    elif set(current["state_change"]) != {
        delta
        for hand, obj in releases
        for delta in (f"- holding({hand},{obj})", f"+ hand_free({hand})")
    }:
        return actions

    held_hands = {hand for hand, _ in state_holdings}
    if any(
        f"hand_free({hand})" not in current_state or hand in held_hands
        for hand, _ in releases
    ):
        return actions
    return actions[1:]


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
            raw_output = call_gpt(model, prompt, images, capture=api_capture)
            parsed_output = json.loads(raw_output)
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
    generator_template = (prompt_dir / "kf_actions_extraction.txt").read_text(
        encoding="utf-8"
    )
    compiler_template = (prompt_dir / "kf_actions_compilation.txt").read_text(
        encoding="utf-8"
    )
    history = []
    ledger = []
    drafts = []
    current_state = set()
    previous_action = None

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
            retry_feedback = ""
            for retry in range(MAX_HAND_RETRIES + 1):
                retry_block = ""
                if retry_feedback:
                    retry_block = (
                        "Retry correction:\n"
                        "The previous output failed deterministic hand-state "
                        "validation:\n"
                        f"{retry_feedback}\n"
                        "Re-examine the images and regenerate the complete JSON "
                        "for this group."
                    )
                prompt = generator_template.format(
                    instruction=instruction,
                    group=group,
                    frame_order=frame_order,
                    previous_history="\n".join(history) or "(no accepted actions)",
                    retry_block=retry_block,
                )
                result = _call_json(
                    model_name,
                    prompt,
                    request_images,
                    lambda value, count=len(images): normalize_group_result(value, count),
                    trace_path=(
                        trace_dir / f"generator_g{group:02d}_retry{retry:02d}.json"
                        if trace_dir is not None
                        else None
                    ),
                    trace_images=images,
                )
                result["actions"] = _drop_repeated_boundary_release(
                    result["actions"],
                    current_state,
                    previous_action,
                )
                next_state, error = validate_hand_actions(result["actions"], current_state)
                if error and retry == MAX_HAND_RETRIES:
                    recovered_state, retry_error = validate_hand_actions(
                        result["actions"], current_state, allow_discontinuity=True
                    )
                    if retry_error is None:
                        next_state, error = recovered_state, None

                if error is None:
                    current_state = next_state
                    previous_action = result["actions"][-1] if result["actions"] else None
                    ledger.append(f"G{group}: {result['changes']}")
                    if not result["actions"]:
                        ledger.append("Draft: (none)")
                    for item in result["actions"]:
                        history.append(f"[G{group}] {item['action']}")
                        drafts.append(
                            {"id": len(drafts) + 1, "group": group, "action": item["action"]}
                        )
                        ledger.append(f"[D{len(drafts)}] {item['action']}")
                    break
                retry_feedback = error
            else:
                raise RuntimeError(
                    f"G{group} failed hand validation after {MAX_HAND_RETRIES} retries: "
                    f"{retry_feedback}"
                )

    first_image = segment_images[0][0]
    compiled = _call_json(
        model_name,
        compiler_template.format(
            instruction=instruction,
            evidence_ledger="\n".join(ledger),
        ),
        [first_image],
        lambda value: normalize_compiler_result(value, drafts, len(segment_images)),
        trace_path=trace_dir / "compiler.json" if trace_dir is not None else None,
    )
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "evidence_ledger.txt").write_text(
            "\n".join(ledger) + "\n", encoding="utf-8"
        )
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
