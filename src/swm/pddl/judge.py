from __future__ import annotations

from pathlib import Path

from swm.llm import call_gpt_json
from swm.pddl.init_state_precheck import (
    compare_initial_states,
    explicit_tool_possession_conflicts,
    implicit_running_device_start_conflicts,
    read_problem_source,
    reference_contract_conflicts,
)
from swm.pddl.strips import ground_plan, parse_domain, parse_plan


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
    last_error: ValueError | None = None
    for _ in range(3):
        try:
            return _validated_vlm_result(call_gpt_json(model, prompt, [first_img]))
        except ValueError as error:
            last_error = error
    raise ValueError(
        "Judge did not return the required flat judge schema after 3 attempts: "
        f"{last_error}"
    )


def _pddl_context(
    predicted_domain: str | Path | None,
    pddl_plan: str | Path | None,
) -> str:
    if predicted_domain is None or pddl_plan is None:
        return "No symbolic PDDL artifacts were supplied."
    try:
        domain_path = predicted_domain if isinstance(predicted_domain, Path) else None
        plan_path = pddl_plan if isinstance(pddl_plan, Path) else None
        if domain_path is None or plan_path is None:
            plan = (
                pddl_plan.read_text(encoding="utf-8")
                if isinstance(pddl_plan, Path)
                else pddl_plan
            )
            return "[Authoritative symbolic plan]\n" + plan

        schemas = parse_domain(domain_path)
        raw_plan, _ = parse_plan(plan_path)
        actions = ground_plan(raw_plan, schemas)
        dynamic_predicates = {
            literal[0]
            for schema in schemas.values()
            for literal in schema.add_eff | schema.del_eff
        }

        def format_literal(literal: tuple[str, ...]) -> str:
            return f"({' '.join(literal)})"

        def format_literals(literals: set[tuple[str, ...]]) -> list[str]:
            return sorted(
                format_literal(literal)
                for literal in literals
                if literal[0] in dynamic_predicates
            )

        def format_action(action) -> str:
            arguments = f" {' '.join(action.args)}" if action.args else ""
            return f"({action.name}{arguments})"

        lines = ["[Candidate mechanically grounded symbolic trace]"]
        for index, action in enumerate(actions, start=1):
            before = format_literals(action.pre_pos)
            before.extend(f"not {literal}" for literal in format_literals(action.pre_neg))
            changes = [f"+{literal}" for literal in format_literals(action.add_eff)]
            changes.extend(f"-{literal}" for literal in format_literals(action.del_eff))

            lines.append(f"{index}. {format_action(action)}")
            if before:
                lines.append(f"   Before: {', '.join(before)}")
            if changes:
                lines.append(f"   State change: {', '.join(changes)}")
        return "\n".join(lines)
    except (OSError, KeyError, ValueError, NotImplementedError) as error:
        return f"Symbolic PDDL artifacts could not be read: {error}"


def _initial_state_crosscheck(
    predicted_problem: str | Path | None,
    ground_truth_problem: str | Path | None,
) -> str:
    """Report auxiliary PDDL disagreements without deciding the visual verdict."""
    if predicted_problem is None or ground_truth_problem is None:
        return "No auxiliary reference PDDL initial-state comparison was supplied."
    try:
        result = compare_initial_states(
            read_problem_source(predicted_problem),
            read_problem_source(ground_truth_problem),
        )
    except (OSError, ValueError) as error:
        return f"Auxiliary initial-state comparison is unavailable: {error}"
    if not result.contradictions:
        return "No explicit auxiliary initial-state disagreement was detected."
    return "\n".join(
        [
            "Candidate and reference PDDL disagree on:",
            *[f"- {contradiction}" for contradiction in result.contradictions],
            "This is an auxiliary warning, not scene evidence.",
        ]
    )


def judge_pddl(
    model: str,
    first_img: Path,
    instruction: str,
    kf_plan: str,
    nl_plan: str,
    n: int = 1,
    predicted_problem: str | Path | None = None,
    ground_truth_problem: str | Path | None = None,
    predicted_domain: str | Path | None = None,
    pddl_plan: str | Path | None = None,
):
    if n < 1:
        raise ValueError("n must be at least 1")

    if all(
        isinstance(source, Path)
        for source in (predicted_domain, predicted_problem, pddl_plan)
    ):
        conflicts = explicit_tool_possession_conflicts(
            predicted_domain,
            predicted_problem,
            pddl_plan,
        )
        if conflicts:
            failure = "; ".join(conflicts)
            return {
                "reasoning": "The evaluated plan explicitly uses a tool without holding or mechanically binding it: " + failure,
                "pass": False,
                "feedback": "Hold or mechanically bind the named tool before using it. " + failure,
            }

    if all(isinstance(source, Path) for source in (predicted_domain, pddl_plan)):
        conflicts = implicit_running_device_start_conflicts(
            predicted_domain,
            pddl_plan,
        )
        if conflicts:
            failure = "; ".join(conflicts)
            return {
                "reasoning": "The evaluated action has an impossible device-state transition: " + failure,
                "pass": False,
                "feedback": "Turn on the device before the running-device action. " + failure,
            }

    if all(
        isinstance(source, Path)
        for source in (
            predicted_domain,
            predicted_problem,
            pddl_plan,
            ground_truth_problem,
        )
    ):
        reference_domain = ground_truth_problem.parent / "domain.pddl"
        reference_plan = ground_truth_problem.parent / "plan.txt"
        if reference_domain.is_file() and reference_plan.is_file():
            conflicts = reference_contract_conflicts(
                predicted_domain,
                predicted_problem,
                pddl_plan,
                reference_domain,
                ground_truth_problem,
                reference_plan,
                instruction,
            )
            if conflicts:
                failure = "; ".join(conflicts)
                return {
                    "reasoning": "The candidate violates a reference-backed semantic contract: " + failure,
                    "pass": False,
                    "feedback": "Preserve the required object count, relation specificity, and independently meaningful state transitions. " + failure,
                }

    prompt_path = Path(__file__).parent.parent / "prompt_templates" / "pddl_judge.txt"
    prompt = prompt_path.read_text(encoding="utf-8").format(
        instruction=instruction,
        kf_plan=kf_plan,
        nl_plan=nl_plan,
        initial_state_crosscheck=_initial_state_crosscheck(
            predicted_problem,
            ground_truth_problem,
        ),
        pddl_context=_pddl_context(
            predicted_domain,
            pddl_plan,
        ),
    )

    results = [_call_validated_judge(model, prompt, first_img) for _ in range(n)]

    passed = sum(r["pass"] for r in results) > n // 2

    for r in results:
        if r["pass"] == passed:
            return r
