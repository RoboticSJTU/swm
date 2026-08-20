from __future__ import annotations

from pathlib import Path

from swm.llm import call_gpt_json
from swm.pddl.init_state_precheck import (
    PrecheckResult,
    compare_initial_states,
    read_problem_source,
)


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


def judge_pddl(
    model: str,
    first_img: Path,
    instruction: str,
    kf_plan: str,
    nl_plan: str,
    n: int = 1,
    predicted_problem: str | Path | None = None,
    ground_truth_problem: str | Path | None = None,
):
    if n < 1:
        raise ValueError("n must be at least 1")

    if predicted_problem is not None and ground_truth_problem is not None:
        try:
            precheck = compare_initial_states(
                read_problem_source(predicted_problem),
                read_problem_source(ground_truth_problem),
            )
        except (OSError, TypeError, ValueError) as error:
            # Missing/unreadable auxiliary inputs must not suppress the VLM judge.
            precheck = PrecheckResult(
                decision="defer",
                reason=f"Auxiliary problem input could not be read: {error}",
            )

        if precheck.should_reject:
            first_error = precheck.contradictions[0]
            return {
                "reasoning": (
                    "The predicted PDDL initial state explicitly contradicts the "
                    "ground-truth initial state: " + " ".join(precheck.contradictions)
                ),
                "pass": False,
                "feedback": (
                    "Correct the predicted PDDL :init facts to match the initial "
                    f"scene. Earliest contradiction: {first_error}"
                ),
            }

    prompt_path = Path(__file__).parent.parent / "prompt_templates" / "pddl_judge.txt"
    prompt = prompt_path.read_text(encoding="utf-8").format(
        instruction=instruction,
        kf_plan=kf_plan,
        nl_plan=nl_plan,
    )

    results = [_call_validated_judge(model, prompt, first_img) for _ in range(n)]

    passed = sum(r["pass"] for r in results) > n // 2

    for r in results:
        if r["pass"] == passed:
            return r
