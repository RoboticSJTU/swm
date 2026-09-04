from __future__ import annotations

import re
from pathlib import Path

from swm.llm import call_gpt_json
from swm.pddl.init_state_precheck import (
    compare_initial_states,
    explicit_tool_possession_conflicts,
    implicit_running_device_start_conflicts,
    parse_problem_text,
    read_problem_source,
    unfinished_started_process_conflicts,
)
from swm.pddl.strips import ground_plan, parse_domain, parse_plan

JUDGE_REASONING_EFFORT = "xhigh"
JUDGE_TEMPERATURE = 1


def latest_round_problem(task_dir: Path) -> Path | None:
    """Return the problem from the highest numeric round that contains one."""
    candidates = []
    if task_dir.is_dir():
        for round_dir in task_dir.glob("round*"):
            suffix = round_dir.name.removeprefix("round")
            problem_path = round_dir / "problem.pddl"
            if suffix.isdigit() and problem_path.is_file():
                candidates.append((int(suffix), problem_path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _validated_vlm_result(result: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Judge response is not a JSON object")  # noqa: TRY004
    required_keys = {"reasoning", "pass", "feedback"}
    if set(result) != required_keys:
        raise ValueError(
            "Judge response must contain exactly reasoning, pass, and feedback"
        )

    reasoning = result["reasoning"]
    passed = result["pass"]
    feedback = result["feedback"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Judge response has invalid reasoning")
    if type(passed) is not bool:
        raise ValueError("Judge response pass must be a JSON boolean")
    if not isinstance(feedback, str):
        raise ValueError("Judge response feedback must be a string")  # noqa: TRY004

    reasoning = reasoning.strip()
    feedback = feedback.strip()
    if passed and feedback:
        raise ValueError("Passing judge response must have empty feedback")
    if not passed and not feedback:
        raise ValueError("Failing judge response must have non-empty feedback")

    return {"reasoning": reasoning, "pass": passed, "feedback": feedback}


def _call_validated_judge(
    model: str,
    prompt: str,
    first_img: Path,
) -> dict:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            return _validated_vlm_result(
                call_gpt_json(
                    model,
                    prompt,
                    [first_img],
                    attempts=1,
                    reasoning_effort=JUDGE_REASONING_EFFORT,
                    temperature=JUDGE_TEMPERATURE,
                )
            )
        except (RuntimeError, ValueError) as error:
            last_error = error
    raise ValueError(
        "Judge did not return the required flat judge schema after 3 attempts: "
        f"{last_error}"
    )


def _evaluated_symbolic_trace(
    candidate_plan: str,
    predicted_domain: str | Path | None,
    pddl_plan: str | Path | None,
) -> str:
    if not isinstance(predicted_domain, Path) or not isinstance(pddl_plan, Path):
        return candidate_plan

    try:
        schemas = parse_domain(predicted_domain)
        raw_plan, _ = parse_plan(pddl_plan)
        if not raw_plan:
            return "Candidate trace contains zero actions."
        actions = ground_plan(raw_plan, schemas)
    except (OSError, KeyError, ValueError, NotImplementedError) as error:
        return f"{candidate_plan}\n\nSymbolic details unavailable: {error}"

    dynamic_predicates = {
        literal[0]
        for schema in schemas.values()
        for literal in schema.add_eff | schema.del_eff
    }

    def format_literals(literals: set[tuple[str, ...]]) -> list[str]:
        return sorted(
            f"{literal[0]}({', '.join(literal[1:])})"
            for literal in literals
            if literal[0] in dynamic_predicates
        )

    lines = []
    for index, action in enumerate(actions, start=1):
        manipulators = [
            argument
            for argument in action.args
            if any(part in {"arm", "hand", "gripper"} for part in argument.split("_"))
        ]
        objects = [argument for argument in action.args if argument not in manipulators]
        before = format_literals(action.pre_pos)
        before.extend(f"not {literal}" for literal in format_literals(action.pre_neg))
        changes = [f"+{literal}" for literal in format_literals(action.add_eff)]
        changes.extend(f"-{literal}" for literal in format_literals(action.del_eff))

        action_text = f"{action.name}({', '.join(objects)})"
        if manipulators:
            action_text += f" with {' and '.join(manipulators)}"
        lines.append(f"{index}. {action_text}")
        if before:
            lines.append(f"   Before: {', '.join(before)}")
        if changes:
            lines.append(f"   State change: {', '.join(changes)}")
    return "\n".join(lines)


def _render_literal(literal: tuple[str, ...]) -> str:
    return "(" + " ".join(literal) + ")"


def _candidate_initial_state(predicted_problem: str | Path | None) -> str:
    if predicted_problem is None:
        return "Candidate Initial State: unavailable."
    try:
        parsed = parse_problem_text(read_problem_source(predicted_problem))
    except (OSError, TypeError, ValueError) as error:
        return f"Candidate Initial State: unavailable ({error})."

    positive = [_render_literal(literal) for literal in sorted(parsed.positive_init)]
    negative = [
        f"(not {_render_literal(literal)})" for literal in sorted(parsed.negative_init)
    ]
    return "\n".join(
        [
            "Positive facts:",
            *(positive or ["(none)"]),
            "Negative facts:",
            *(negative or ["(none)"]),
        ]
    )


def _candidate_goal(predicted_problem: str | Path | None) -> str:
    if predicted_problem is None:
        return "Candidate Goal: unavailable."
    try:
        text = read_problem_source(predicted_problem)
        match = re.search(r"\(\s*:goal\b", text, re.IGNORECASE)
        if not match:
            raise ValueError("missing :goal section")
        depth = 0
        in_comment = False
        for index in range(match.start(), len(text)):
            character = text[index]
            if character == "\n":
                in_comment = False
            elif in_comment:
                continue
            elif character == ";":
                in_comment = True
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    return text[match.start() : index + 1]
        raise ValueError("unclosed :goal section")
    except (OSError, TypeError, ValueError) as error:
        return f"Candidate Goal: unavailable ({error})."


def _render_judge_prompt(
    instruction: str,
    kf_actions: str,
    candidate_plan: str,
    predicted_domain: str | Path | None,
    predicted_problem: str | Path | None,
    ground_truth_problem: str | Path | None,
    pddl_plan: str | Path | None,
) -> str:
    prompt_path = Path(__file__).parent.parent / "prompt_templates" / "pddl_judge.txt"
    return prompt_path.read_text(encoding="utf-8").format(
        instruction=instruction,
        kf_actions=kf_actions,
        candidate_initial_state=_candidate_initial_state(predicted_problem),
        candidate_goal=_candidate_goal(predicted_problem),
        programmatic_findings=_programmatic_findings(
            predicted_domain,
            predicted_problem,
            pddl_plan,
            ground_truth_problem,
        ),
        evaluated_symbolic_trace=_evaluated_symbolic_trace(
            candidate_plan,
            predicted_domain,
            pddl_plan,
        ),
    )


def _programmatic_findings(
    predicted_domain: str | Path | None,
    predicted_problem: str | Path | None,
    pddl_plan: str | Path | None,
    ground_truth_problem: str | Path | None,
) -> str:
    findings = []

    if predicted_problem is not None and ground_truth_problem is not None:
        findings.extend(
            f"- Init: {contradiction}"
            for contradiction in compare_initial_states(
                read_problem_source(predicted_problem),
                read_problem_source(ground_truth_problem),
            ).contradictions
        )

    if all(
        isinstance(source, Path)
        for source in (predicted_domain, predicted_problem, pddl_plan)
    ):
        findings.extend(
            f"- Tool: {conflict}"
            for conflict in explicit_tool_possession_conflicts(
                predicted_domain,
                predicted_problem,
                pddl_plan,
            )
        )
        findings.extend(
            f"- Device: {conflict}"
            for conflict in implicit_running_device_start_conflicts(
                predicted_domain,
                pddl_plan,
            )
        )

    if findings:
        return "\n".join(findings)
    if ground_truth_problem is None:
        return "GT PDDL was not supplied; only Candidate-only checks were available."
    return "None."


def judge_pddl(
    model: str,
    first_img: Path,
    instruction: str,
    kf_actions: str,
    candidate_plan: str,
    n: int = 1,
    predicted_problem: str | Path | None = None,
    ground_truth_problem: str | Path | None = None,
    predicted_domain: str | Path | None = None,
    pddl_plan: str | Path | None = None,
):
    if n < 1:
        raise ValueError("n must be at least 1")
    candidate_plan = candidate_plan.strip()
    if not candidate_plan:
        raise ValueError("candidate_plan must be non-empty")

    if all(
        isinstance(source, Path)
        for source in (predicted_domain, predicted_problem, pddl_plan)
    ):
        conflicts = unfinished_started_process_conflicts(
            predicted_domain,
            predicted_problem,
            pddl_plan,
        )
        if conflicts:
            failure = "; ".join(conflicts)
            return {
                "reasoning": "The Candidate leaves a task-started process active: "
                + failure,
                "pass": False,
                "feedback": "Stop the started process before completion. " + failure,
            }

    prompt = _render_judge_prompt(
        instruction=instruction,
        kf_actions=kf_actions,
        candidate_plan=candidate_plan,
        predicted_domain=predicted_domain,
        predicted_problem=predicted_problem,
        ground_truth_problem=ground_truth_problem,
        pddl_plan=pddl_plan,
    )
    results = [_call_validated_judge(model, prompt, first_img) for _ in range(n)]
    majority = sum(result["pass"] for result in results) > n // 2
    return next(result for result in results if result["pass"] == majority)
