#!/usr/bin/env python3
"""Regenerate natural-language plans for the latest round of each episode."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = PROJECT_ROOT / "eval_results" / "gpt-5.6-sol"
ROUND_RE = re.compile(r"roun(?:d)?[_-]?(\d+)$", re.IGNORECASE)


def replace_vars_exact(template: str, mapping: dict[str, str]) -> str:
    """Replace complete PDDL variable tokens without substring collisions."""
    return re.sub(r"\?\w+", lambda match: mapping.get(match.group(0), match.group(0)), template)


def _space_before_var(text: str) -> str:
    return re.sub(r"(?<=[A-Za-z0-9_])\?", " ?", text)


def _clean_template(comment_line: str, unary_preds: set[str]) -> str:
    text = comment_line.lstrip(";").strip()
    text = re.sub(r"\bobject\s+(\?\w+)\b", r"\1", text, flags=re.IGNORECASE)
    if unary_preds:
        predicate_pattern = "|".join(
            map(re.escape, sorted(unary_preds, key=str.lower))
        )
        text = re.sub(
            rf"\b({predicate_pattern})\s+(\?\w+)\b",
            r"\2",
            text,
            flags=re.IGNORECASE,
        )
    text = _space_before_var(text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def _action_templates(domain_text: str) -> dict[str, tuple[str, list[str]]]:
    """Extract action comments and parameter order using translate.py semantics."""
    action_info: dict[str, tuple[str, list[str]]] = {}
    lines = domain_text.splitlines()
    index = 0

    while index < len(lines):
        if "(:action" not in lines[index]:
            index += 1
            continue

        action_match = re.search(r"\(:action\s+([^\s()]+)", lines[index])
        if not action_match:
            index += 1
            continue
        action_name = action_match.group(1)

        comment_index = index - 1
        while comment_index >= 0 and not lines[comment_index].strip():
            comment_index -= 1
        comment = (
            lines[comment_index].strip()
            if comment_index >= 0 and lines[comment_index].lstrip().startswith(";")
            else ""
        )

        balance = 0
        block_lines: list[str] = []
        block_end = index
        while block_end < len(lines):
            block_lines.append(lines[block_end])
            balance += lines[block_end].count("(") - lines[block_end].count(")")
            if block_end > index and balance <= 0:
                break
            block_end += 1
        block = "\n".join(block_lines)

        parameters_match = re.search(
            r":parameters\s*\((.*?)\)", block, flags=re.DOTALL | re.IGNORECASE
        )
        parameters = re.findall(
            r"\?\w+", parameters_match.group(1) if parameters_match else ""
        )
        precondition_match = re.search(
            r":precondition\s*\((.*?)\)\s*:effect",
            block,
            flags=re.DOTALL | re.IGNORECASE,
        )
        precondition = precondition_match.group(1) if precondition_match else ""
        unary_preds = {
            predicate
            for predicate, _variable in re.findall(
                r"\(\s*([A-Za-z_][\w-]*)\s+(\?\w+)\s*\)", precondition
            )
            if predicate.lower() not in {"and", "or", "not", "=", "imply"}
        }
        template = _clean_template(comment, unary_preds) if comment else ""
        action_info[action_name] = (template, parameters)
        index = block_end + 1

    return action_info


def translate_plan_text(domain_text: str, plan_text: str) -> str:
    """Translate a symbolic plan without reading or writing filesystem state."""
    action_info = _action_templates(domain_text)
    output: list[str] = []

    for raw_line in plan_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith((";", "#")):
            continue
        line = re.sub(r";.*$", "", line).strip()
        line = re.sub(r"^\s*\d+\s*:\s*", "", line)
        line = re.sub(r"\s*\[\s*[\d.]+\s*\]\s*$", "", line)
        line = line.strip().strip("()").strip()
        if not line:
            continue

        tokens = line.split()
        action_name, arguments = tokens[0], tokens[1:]
        template, parameters = action_info.get(action_name, ("", []))
        if not template:
            output.append(
                action_name + (" " + " ".join(arguments) if arguments else "")
            )
            continue

        mapping = {
            parameters[position]: arguments[position]
            for position in range(min(len(parameters), len(arguments)))
        }
        translated = replace_vars_exact(_space_before_var(template), mapping)
        translated = re.sub(r"\s+", " ", translated).strip()
        translated = re.sub(r"\s+([.,;:!?])", r"\1", translated)
        output.append(translated)

    return "\n".join(output) + ("\n" if output else "")


def latest_rounds(dataset_root: Path, task_ids: set[int] | None = None) -> list[Path]:
    """Return only the highest-numbered PDDL round in every selected episode."""
    rounds: list[Path] = []
    episodes = sorted(
        dataset_root.glob("task_*/episode_*"),
        key=lambda path: (
            int(path.parent.name.removeprefix("task_")),
            int(path.name.removeprefix("episode_")),
        ),
    )
    for episode in episodes:
        task_match = re.fullmatch(r"task_(\d+)", episode.parent.name)
        if not task_match:
            continue
        if task_ids is not None and int(task_match.group(1)) not in task_ids:
            continue

        candidates: list[tuple[int, Path]] = []
        for child in episode.iterdir():
            round_match = ROUND_RE.fullmatch(child.name) if child.is_dir() else None
            if not round_match:
                continue
            if (child / "domain.pddl").is_file() and (child / "plan.txt").is_file():
                candidates.append((int(round_match.group(1)), child))
        if candidates:
            rounds.append(max(candidates, key=lambda candidate: candidate[0])[1])
    return rounds


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate plan_nl.txt for the latest round of each episode."
    )
    parser.add_argument(
        "--dataset", choices=["human", "human_aug", "both"], default="both"
    )
    parser.add_argument(
        "--task",
        type=int,
        action="append",
        dest="tasks",
        help="Restrict processing to a task ID; repeat for multiple tasks.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report stale or missing plan_nl.txt files without writing them.",
    )
    args = parser.parse_args()

    datasets = ["human", "human_aug"] if args.dataset == "both" else [args.dataset]
    selected_tasks = set(args.tasks) if args.tasks else None
    summary = {
        "rounds": 0,
        "changed": 0,
        "unchanged": 0,
        "written": 0,
        "check": args.check,
    }

    for dataset in datasets:
        for round_dir in latest_rounds(EVAL_ROOT / dataset, selected_tasks):
            summary["rounds"] += 1
            domain_text = (round_dir / "domain.pddl").read_text(
                encoding="utf-8", errors="ignore"
            )
            plan_text = (round_dir / "plan.txt").read_text(
                encoding="utf-8", errors="ignore"
            )
            translated = translate_plan_text(domain_text, plan_text)
            output_path = round_dir / "plan_nl.txt"
            existing = (
                output_path.read_text(encoding="utf-8", errors="ignore")
                if output_path.is_file()
                else None
            )
            if existing == translated:
                summary["unchanged"] += 1
                continue

            summary["changed"] += 1
            if not args.check:
                _atomic_write(output_path, translated)
                summary["written"] += 1

    print(json.dumps(summary, ensure_ascii=False))
    if args.check and summary["changed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
