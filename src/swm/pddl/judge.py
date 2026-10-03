from __future__ import annotations

from pathlib import Path

from swm.llm import call_gpt_json
from swm.pddl.strips import (
    ground_plan,
    parse_domain,
    parse_plan,
    parse_sexpr_file,
)


def _call_validated_judge(
    model: str, prompt: str, first_img: Path, capture: dict | None = None
) -> dict:
    last_error: Exception | None = None
    attempts = []
    for attempt in range(1, 4):
        attempt_capture = {"attempt": attempt}
        try:
            call_kwargs = {"attempts": 1}
            if capture is not None:
                call_kwargs["capture"] = attempt_capture
            result = call_gpt_json(model, prompt, [first_img], **call_kwargs)
            if not isinstance(result, dict):
                raise ValueError("Judge response is not a JSON object")
            if set(result) != {"reasoning", "pass", "feedback"}:
                raise ValueError("Judge response must contain exactly reasoning, pass, and feedback")
            reasoning, passed, feedback = result["reasoning"], result["pass"], result["feedback"]
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError("Judge response has invalid reasoning")
            if type(passed) is not bool:
                raise ValueError("Judge response pass must be a JSON boolean")
            if not isinstance(feedback, str):
                raise ValueError("Judge response feedback must be a string")
            reasoning, feedback = reasoning.strip(), feedback.strip()
            if passed and feedback:
                raise ValueError("Passing judge response must have empty feedback")
            if not passed and not feedback:
                raise ValueError("Failing judge response must have non-empty feedback")
            attempts.append(attempt_capture)
            if capture is not None:
                capture["attempts"] = attempts
            return {"reasoning": reasoning, "pass": passed, "feedback": feedback}
        except (RuntimeError, ValueError) as error:
            attempt_capture["error_type"] = type(error).__name__
            attempt_capture["error"] = str(error)
            attempts.append(attempt_capture)
            last_error = error
    if capture is not None:
        capture["attempts"] = attempts
    raise ValueError(
        "Judge did not return the required flat judge schema after 3 attempts: "
        f"{last_error}"
    )


def _evaluated_symbolic_trace(
    candidate_plan: str,
    predicted_domain: str | Path | None,
    pddl_plan: str | Path | None,
    predicted_problem: str | Path | None = None,
) -> str:
    if not isinstance(predicted_domain, Path) or not isinstance(pddl_plan, Path):
        return candidate_plan

    try:
        raw_plan, _ = parse_plan(pddl_plan)
        if not raw_plan:
            return "Candidate trace contains zero actions."
        schemas = parse_domain(predicted_domain)
        actions = ground_plan(raw_plan, schemas)
        source_actions = {
            node[1]: dict(zip(node[2::2], node[3::2]))
            for node in parse_sexpr_file(predicted_domain)[2:]
            if isinstance(node, list) and node[:1] == [":action"]
        }
    except (OSError, KeyError, ValueError, NotImplementedError) as error:
        return f"{candidate_plan}\n\nSymbolic details unavailable: {error}"

    dynamic_predicates = {
        literal[0]
        for schema in schemas.values()
        for literals in (schema.add_eff, schema.del_eff)
        for literal in literals
    }

    def format_literals(expression, bindings, *, effect=False) -> list[str]:
        if expression[0] == "and":
            return [
                text
                for child in expression[1:]
                for text in format_literals(child, bindings, effect=effect)
            ]
        negative = expression[0] == "not"
        atom = expression[1] if negative else expression
        if atom[0] not in dynamic_predicates:
            return []
        prefix = ("-" if negative else "+") if effect else ("not " if negative else "")
        arguments = ", ".join(bindings.get(argument, argument) for argument in atom[1:])
        return [f"{prefix}{atom[0]}({arguments})"]

    lines = []
    for index, action in enumerate(actions, start=1):
        manipulators = [
            argument
            for argument in action.args
            if any(part in {"arm", "hand", "gripper"} for part in argument.split("_"))
        ]
        objects = [argument for argument in action.args if argument not in manipulators]
        bindings = dict(zip(schemas[action.name].params, action.args))
        fields = source_actions[action.name]
        before = format_literals(fields[":precondition"], bindings)
        changes = format_literals(fields[":effect"], bindings, effect=True)

        action_text = f"{action.name}({', '.join(objects)})"
        if manipulators:
            action_text += f" with {' and '.join(manipulators)}"
        lines.append(f"{index}. {action_text}")
        if before:
            lines.append(f"   Before: {', '.join(before)}")
        if changes:
            lines.append(f"   State change: {', '.join(changes)}")
    return "\n".join(lines)


def judge_pddl(
    model: str,
    first_img: Path,
    instruction: str,
    kf_actions: str,
    candidate_plan: str,
    predicted_problem: str | Path | None = None,
    predicted_domain: str | Path | None = None,
    pddl_plan: str | Path | None = None,
    capture: dict | None = None,
):
    candidate_plan = candidate_plan.strip()
    prompt_path = Path(__file__).parent.parent / "prompt_templates" / "pddl_judge.txt"
    prompt = prompt_path.read_text(encoding="utf-8").format(
        instruction=instruction,
        kf_actions=kf_actions,
        evaluated_symbolic_trace=_evaluated_symbolic_trace(
            candidate_plan, predicted_domain, pddl_plan, predicted_problem
        ),
    )
    if capture is not None:
        capture.update({"decision_source": "vlm", "model": model})
    return _call_validated_judge(model, prompt, first_img, capture)
