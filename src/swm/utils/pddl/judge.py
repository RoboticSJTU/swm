from pathlib import Path
from swm.utils.plan_learning import get_prompt_from_template
from swm.utils.apis import call_gpt_json

def judge_pddl(
    model: str,
    first_img: Path,
    instruction: str,
    kf_plan: str,
    nl_plan: str,
    n: int = 1,
):
    prompt_path = Path(__file__).parent.parent.parent / "prompt_templates" / "pddl_judge.txt"
    prompt = get_prompt_from_template(
        prompt_path,
        instruction=instruction,
        kf_plan=kf_plan,
        nl_plan=nl_plan,
    )

    results = [call_gpt_json(model, prompt, [first_img]) for _ in range(n)]

    passed = sum(r["pass"] for r in results) > n // 2

    for r in results:
        if r["pass"] == passed:
            return {
                "pass": passed,
                "reasoning": r["reasoning"],
                "feedback": r["feedback"] if not passed else "",
            }