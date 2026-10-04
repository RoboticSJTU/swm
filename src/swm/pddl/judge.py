from __future__ import annotations

from pathlib import Path

from swm.llm import call_gpt_json
from swm.pddl.strips import parse_sexpr_file


def _call_validated_judge(
    model: str, prompt: str, first_img: Path | list[Path], capture: dict | None = None
) -> dict:
    if isinstance(first_img, (list, tuple)) and len(first_img) > 1:
        prompt = "All attached images are views of the same initial scene. Use the complete set of views and preserve object identity across views.\n\n" + prompt
    last_error: Exception | None = None
    attempts = []
    for attempt in range(1, 4):
        attempt_capture = {"attempt": attempt}
        try:
            call_kwargs = {"attempts": 1}
            if capture is not None:
                call_kwargs["capture"] = attempt_capture
            images = list(first_img) if isinstance(first_img, (list, tuple)) else [first_img]
            result = call_gpt_json(model, prompt, images, **call_kwargs)
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


class SymbolicTraceError(ValueError):
    """PDDL trace preparation failed; do not fall back to an NL judgment."""


def _evaluated_symbolic_trace(
    candidate_plan: str,
    predicted_domain: str | Path | None,
    pddl_plan: str | Path | None,
) -> str:
    if predicted_domain is None and pddl_plan is None:
        return candidate_plan  # Explicit natural-language evaluation mode.
    if predicted_domain is None or pddl_plan is None:
        raise SymbolicTraceError("PDDL judging requires both domain and plan paths")

    try:
        domain_path, plan_path = Path(predicted_domain), Path(pddl_plan)
        raw_plan = []
        for line in plan_path.read_text(encoding="utf-8").splitlines():
            line = line.split(";", 1)[0].strip()
            if line and (not line.startswith("(") or not line.endswith(")")
                         or "(" in line[1:] or ")" in line[:-1]):
                raise ValueError(f"Malformed plan line: {line}")
            if line:
                parts = line[1:-1].lower().split()
                if not parts:
                    raise ValueError("Empty plan action")
                raw_plan.append((parts[0], parts[1:]))
        if not raw_plan:
            return "Candidate trace contains zero actions."
        root = parse_sexpr_file(domain_path)
        source_actions = {
            node[1]: dict(zip(node[2::2], node[3::2]))
            for node in root[1:]
            if isinstance(node, list) and node[:1] == [":action"]
        }
        derived = [node for node in root[1:] if isinstance(node, list) and node[:1] == [":derived"]]

        def effect_predicates(node):
            if not node:
                return set()
            if node[0] == "and":
                return set().union(*(effect_predicates(child) for child in node[1:]))
            if node[0] in {"forall", "when"}:
                return effect_predicates(node[2])
            if node[0] == "not":
                return {node[1][0]}
            return {node[0]}

        dynamic_predicates = {node[1][0] for node in derived}
        for fields in source_actions.values():
            dynamic_predicates.update(effect_predicates(fields[":effect"]))

        def sexpr(node, bindings):
            if isinstance(node, str):
                return bindings.get(node, node)
            if node[0] in {"forall", "exists"}:
                # Quantified variables shadow action parameters of the same name.
                local = {key: value for key, value in bindings.items() if key not in node[1]}
                return f"({node[0]} ({' '.join(node[1])}) {sexpr(node[2], local)})"
            return "(" + " ".join(sexpr(child, bindings) for child in node) + ")"

        def format_literals(expression, bindings, *, effect=False):
            if not expression:
                return []
            if expression[0] == "and":
                return [text for child in expression[1:]
                        for text in format_literals(child, bindings, effect=effect)]
            negative = expression[0] == "not"
            atom = expression[1] if negative else expression
            if atom[0] in {"forall", "exists", "when", "or", "imply", "and"}:
                # Preserve guards and logical structure, without reading candidate init.
                return [sexpr(expression, bindings)]
            if atom[0] not in dynamic_predicates and atom[0] != "=":
                return []
            prefix = ("-" if negative else "+") if effect else ("not " if negative else "")
            arguments = ", ".join(sexpr(argument, bindings) for argument in atom[1:])
            return [f"{prefix}{atom[0]}({arguments})"]

        lines = []
        if derived:
            lines.append("Derived rules: " + " ".join(sexpr(node, {}) for node in derived))
        for index, (name, args) in enumerate(raw_plan, start=1):
            fields = source_actions[name]
            parameters = [token for token in fields[":parameters"]
                          if isinstance(token, str) and token.startswith("?")]
            if len(args) != len(parameters):
                raise ValueError(f"Arity mismatch for {name}: expected {len(parameters)}, got {len(args)}")
            bindings = dict(zip(parameters, args))
            manipulators = [arg for arg in args
                            if any(part in {"arm", "hand", "gripper"} for part in arg.split("_"))]
            objects = [arg for arg in args if arg not in manipulators]
            before = format_literals(fields[":precondition"], bindings)
            changes = format_literals(fields[":effect"], bindings, effect=True)
            action_text = f"{name}({', '.join(objects)})"
            if manipulators:
                action_text += f" with {' and '.join(manipulators)}"
            lines.append(f"{index}. {action_text}")
            if before:
                lines.append(f"   Before: {', '.join(before)}")
            if changes:
                lines.append(f"   State change: {', '.join(changes)}")
        return "\n".join(lines)
    except (OSError, KeyError, ValueError, NotImplementedError, IndexError, TypeError) as error:
        raise SymbolicTraceError(f"Cannot prepare PDDL judge trace: {error}") from error


def judge_pddl(
    model: str,
    first_img: Path | list[Path],
    instruction: str,
    kf_actions: str,
    candidate_plan: str,
    predicted_domain: str | Path | None = None,
    pddl_plan: str | Path | None = None,
    capture: dict | None = None,
    scene_context: str = "",
):
    candidate_plan = candidate_plan.strip()
    prompt_path = Path(__file__).parent.parent / "prompt_templates" / "pddl_judge.txt"
    prompt = prompt_path.read_text(encoding="utf-8").format(
        instruction=instruction,
        kf_actions=kf_actions,
        evaluated_symbolic_trace=_evaluated_symbolic_trace(
            candidate_plan, predicted_domain, pddl_plan
        ),
    )
    if scene_context:
        prompt = "Official benchmark context:\n" + scene_context + "\n\n" + prompt
    if capture is not None:
        capture.update({"decision_source": "vlm", "model": model})
    return _call_validated_judge(model, prompt, first_img, capture)
