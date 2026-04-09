from pathlib import Path
from swm.utils.plan_learning import get_prompt_from_template
from swm.utils.apis import call_gpt_json


def judge_pddl(
    model:str,
    first_img: Path,
    instruction: str,
    kf_plan: str,
    nl_plan: str,
):
    prompt_path = Path(__file__).parent.parent.parent / "prompt_templates" / "pddl_judge.txt"
    prompt = get_prompt_from_template(prompt_path, instruction=instruction,kf_plan=kf_plan, nl_plan=nl_plan)
    # print(prompt)
    data = call_gpt_json(model, prompt, [first_img])
    passed = data["pass"]
    reasoning = data["reasoning"]
    feedback = data["feedback"]
    judge_out = {"pass": passed, "reasoning": reasoning, "feedback": feedback}
    return judge_out