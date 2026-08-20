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
        raise ValueError("Judge response is not a JSON object")
    stage_1 = result.get("stage_1")
    stage_2 = result.get("stage_2")
    if not isinstance(stage_1, dict) or not isinstance(stage_2, dict):
        raise ValueError("Judge response is missing stage_1 or stage_2")

    stage_1_pass = stage_1.get("pass") is True
    stage_1_reasoning = str(stage_1.get("reason", stage_1.get("reasoning", ""))).strip()
    stage_2_raw_pass = stage_2.get("pass")
    stage_2_evaluated = (
        stage_2.get("evaluated") is True
        if "evaluated" in stage_2
        else stage_2_raw_pass is not None
    )
    stage_2_pass = stage_2_raw_pass is True
    stage_2_reasoning = str(stage_2.get("reason", stage_2.get("reasoning", ""))).strip()
    if not stage_1_pass:
        stage_2_evaluated = False
        stage_2_pass = False
        stage_2_reasoning = "Not evaluated because Stage 1 failed."
    passed = stage_1_pass and stage_2_evaluated and stage_2_pass
    if not stage_1_reasoning:
        raise ValueError("Judge response has empty Stage 1 reason")
    if stage_2_evaluated and not stage_2_reasoning:
        raise ValueError("Judge response has empty Stage 2 reason")

    reasoning = f"Stage 1: {stage_1_reasoning} Stage 2: {stage_2_reasoning}"
    feedback = (
        "" if passed else (stage_1_reasoning if not stage_1_pass else stage_2_reasoning)
    )

    return {
        "pass": passed,
        "stage_1": {
            "pass": stage_1_pass,
            "reasoning": stage_1_reasoning,
        },
        "stage_2": {
            "evaluated": stage_2_evaluated,
            "pass": stage_2_pass,
            "reasoning": stage_2_reasoning,
        },
        "reasoning": reasoning,
        "feedback": feedback,
    }


def _call_validated_judge(
    model: str,
    prompt: str,
    first_img: Path,
) -> dict:
    last_error: ValueError | None = None
    for _ in range(3):
        try:
            return _validated_vlm_result(
                call_gpt_json(model, prompt, [first_img])
            )
        except ValueError as error:
            last_error = error
    raise ValueError(f"Judge did not return the required two-stage schema after 3 attempts: {last_error}")


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

    precheck = None
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
                "pass": False,
                "judge_source": "programmatic_precheck",
                "stage_1": {
                    "pass": False,
                    "reasoning": (
                        "The predicted PDDL initial state explicitly contradicts "
                        f"the ground-truth initial state: {first_error}"
                    ),
                },
                "stage_2": {
                    "evaluated": False,
                    "pass": False,
                    "reasoning": "Not evaluated because the programmatic precheck failed.",
                },
                "reasoning": (
                    "Programmatic initial-state precheck failed before VLM judging. "
                    + " ".join(precheck.contradictions)
                ),
                "feedback": (
                    "Correct the predicted PDDL :init facts to match the initial "
                    f"scene. Earliest contradiction: {first_error}"
                ),
                "precheck": precheck.to_dict(),
            }

    if precheck is None or not precheck.mapping_details:
        object_mapping = "No structured correspondences are available."
    else:
        object_mapping = "\n".join(
            f"- {match.predicted} = {match.ground_truth} "
            f"(method: {match.method}, confidence: {match.score:.3f})"
            for match in precheck.mapping_details
        )

    prompt_path = Path(__file__).parent.parent / "prompt_templates" / "pddl_judge.txt"
    prompt = prompt_path.read_text(encoding="utf-8").format(
        instruction=instruction,
        kf_plan=kf_plan,
        nl_plan=nl_plan,
        object_mapping=object_mapping,
    )

    results = [_call_validated_judge(model, prompt, first_img) for _ in range(n)]

    passed = sum(r["pass"] for r in results) > n // 2

    for r in results:
        if r["pass"] == passed:
            r["judge_source"] = "vlm"
            if precheck is not None:
                r["precheck"] = precheck.to_dict()
            return r
