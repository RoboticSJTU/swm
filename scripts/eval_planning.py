from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import tempfile
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

from openai import APIConnectionError, APIStatusError

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "src"))

from swm.llm import call_gpt
from swm.pddl.attribution import attribute_failure
from swm.pddl.eval_report import render_markdown_report
from swm.pddl.generation import parse_pddl_output, strip_code_block
from swm.pddl.planner import solve_pddl
from swm.pddl.postprocess import repair_pddl
from swm.pddl.reorder import plan_reorder
from swm.pddl.validate import validate_plan_feasibility

ACTIVE_BASELINE = "vlm_nl"
generation_max_workers = 200
rule_postprocess = False     # True：PDDL 求解失败后尝试静态修复
pddl_vote = False            # True：每个任务采样 5 份 PDDL 并按计划投票
PDDL_VOTE_SAMPLES = 5

BASELINES = {
    "9B_sft": dict(
        model="9B_1126", reasoning_effort=None, base_url="http://127.0.0.1:8001/v1", mode="pddl",
        prompt_path=root_dir / "src/swm/prompt_templates/training_input.txt",
    ),
    "vlm_nl": dict(
        model="gpt-6-sol", reasoning_effort="none", base_url=None, mode="nl",
        prompt_path=root_dir / "src/swm/prompt_templates/vlm_nl.txt",
    ),
    "vlm_pddl": dict(
        model="gpt-6-sol", reasoning_effort="none", base_url=None, mode="pddl",
        prompt_path=root_dir / "src/swm/prompt_templates/vlm_pddl.txt",
    ),

}
baseline = BASELINES[ACTIVE_BASELINE]
eval_model = baseline["model"]
eval_mode = baseline["mode"]
prompt_path = baseline["prompt_path"]
eval_root = root_dir / "eval_results" / eval_model / prompt_path.stem
if pddl_vote and eval_mode != "nl":
    eval_root = eval_root.with_name(f"{eval_root.name}_vote{PDDL_VOTE_SAMPLES}")

judge_model = "deepseek/deepseek-flash"
judge_reasoning_effort = "high"

# judge_model = "Qwen3.8-27B"
# judge_reasoning_effort = None

datasets = {
    "anydomain/singe_arm": "single_arm",
    "anydomain/dual_arm": "dual_arm",
    "unidomain/unidomain": "single_arm",
}
enable_attribution = False  # 设为 True 启用失败归因
attribution_model = judge_model
judge_max_workers = generation_max_workers
attribution_max_workers = generation_max_workers
PDDL_FILES = ("domain.pddl", "problem.pddl")
logger = logging.getLogger("swm.eval_planning")


def number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match[1]) if match else 0


def task_identifier(task: dict) -> str:
    return str(number(task["episode"])) if task["flat"] else f"{task['task']}/{task['episode']}"


def generation_complete(task: dict) -> bool:
    names = ("plan.txt",) if eval_mode == "nl" else (*PDDL_FILES, "plan.txt")
    files = (task["save_dir"] / name for name in names)
    return all(path.is_file() and path.read_text(encoding="utf-8").strip() for path in files)


def read_judge(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_judge(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_cached_status(task: dict):
    save_dir = task["save_dir"]
    passed = read_judge(save_dir / "judge.json").get("pass")
    if isinstance(passed, bool):
        return "judge", passed
    if (save_dir / "error.log").exists():
        return "error", False
    return None, False


def postprocess_enabled() -> bool:
    return rule_postprocess and eval_mode != "nl"


def load_tasks() -> list[dict]:
    tasks = []
    for dataset_name, robot_configuration in datasets.items():
        source_dataset, _, group_name = dataset_name.partition("/")
        instructions, steps_data = [
            json.loads((root_dir / f"tasks/{name}/{name}_{source_dataset}.json").read_text(encoding="utf-8"))
            for name in ("instructions", "steps")
        ]
        if group_name:
            instructions, steps_data = instructions[group_name], steps_data[group_name]
        else:
            if isinstance(instructions.get(source_dataset), dict):
                instructions = instructions[source_dataset]
            if isinstance(steps_data.get(source_dataset), dict):
                steps_data = steps_data[source_dataset]

        flat = all(isinstance(value, str) for value in instructions.values())
        groups = {dataset_name: instructions} if flat else instructions
        image_root = root_dir / "tasks/images" / source_dataset
        for task_name in sorted(groups, key=number):
            episodes = groups[task_name]
            if not isinstance(episodes, dict):
                continue
            task_steps = steps_data if flat else steps_data.get(task_name)
            image_dirs = (image_root,) if flat else (image_root / task_name, image_root)
            task_dir = Path(dataset_name) if flat else Path(dataset_name) / task_name
            for episode_name in sorted(episodes, key=number):
                images = (directory / f"{episode_name}.{ext}"
                          for ext in ("png", "jpg", "jpeg") for directory in image_dirs)
                raw_steps = task_steps.get(episode_name) if isinstance(task_steps, dict) else None
                if isinstance(raw_steps, dict):
                    raw_steps = [raw_steps[key] for key in sorted(raw_steps, key=number)]
                if isinstance(raw_steps, list):
                    raw_steps = "\n".join(map(str, raw_steps))

                relative = task_dir / episode_name
                gt_dir = root_dir / "eval_results/gt" / relative
                task = {
                    "dataset": dataset_name, "task": task_name, "episode": episode_name,
                    "robot_configuration": robot_configuration,
                    "instruction": episodes[episode_name],
                    "image": next((path for path in images if path.exists()), None),
                    "kf_actions": "" if raw_steps is None else str(raw_steps), "flat": flat,
                    "save_dir": eval_root / relative, "gt_dir": gt_dir,
                }
                if eval_mode == "problem_only":
                    round_dir = max(
                        (p for p in gt_dir.glob("round*") if p.is_dir() and re.fullmatch(r"round\d+", p.name)),
                        key=lambda p: number(p.name), default=None,
                    )
                    if round_dir is None:
                        raise FileNotFoundError(f"No GT round directory in {gt_dir}")
                    task["gt_domain"] = (round_dir / "domain.pddl").read_text(encoding="utf-8")
                    task["kf_actions"] = (gt_dir / "kf_actions.txt").read_text(encoding="utf-8")
                    if not task["gt_domain"].strip() or not task["kf_actions"].strip():
                        raise ValueError(f"Empty GT domain or reference actions in {gt_dir}")
                tasks.append(task)
    return tasks


def parse_generated_pddl(task: dict, output: str) -> tuple[str, str]:
    if eval_mode == "problem_only":
        match = re.search(r"<problem>\s*(.*?)\s*</problem>", output, flags=re.S)
        if match is None or not (problem := strip_code_block(match[1])):
            raise ValueError("expected a nonempty <problem> block")
        return task["gt_domain"], problem
    return parse_pddl_output(output)


def solve_generated_pddl(task: dict, domain: str, problem: str) -> tuple[bool, str]:
    save_dir = task["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    domain_file, problem_file = (save_dir / name for name in PDDL_FILES)
    domain_file.write_text(domain, encoding="utf-8")
    problem_file.write_text(problem, encoding="utf-8")
    (save_dir / "postprocess.json").unlink(missing_ok=True)
    solved = solve_pddl(domain_file, problem_file, reorder_plan=False)
    if solved and generation_complete(task):
        return True, "pddl"
    if postprocess_enabled() and repair_saved_task(task):
        return True, "pddl_repaired"
    return False, "missing_pddl_plan" if solved else "pddl_unsolvable"


def vote_pddl(task: dict, outputs: list[str]) -> tuple[bool, str]:
    if len(outputs) != PDDL_VOTE_SAMPLES:
        raise ValueError(f"Expected {PDDL_VOTE_SAMPLES} PDDL samples, got {len(outputs)}")
    temp_root = root_dir / "temp"
    temp_root.mkdir(exist_ok=True)
    votes = Counter()
    candidates = {}
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        for index, output in enumerate(outputs):
            try:
                domain, problem = parse_generated_pddl(task, output or "")
            except (ValueError, KeyError, TypeError):
                continue
            candidate_dir = Path(directory) / str(index)
            ok, status = solve_generated_pddl({**task, "save_dir": candidate_dir}, domain, problem)
            plan = (candidate_dir / "plan.txt").read_bytes() if ok else None
            candidates.setdefault(plan, (candidate_dir, ok, status))
            if plan is not None:
                votes[plan] += 1

        if not candidates:
            raise ValueError("No parseable PDDL samples")
        selected = max(votes, key=votes.get) if votes else None
        candidate_dir, ok, status = candidates[selected]
        save_dir = task["save_dir"]
        (save_dir / "plan.txt").unlink(missing_ok=True)
        for name in (*PDDL_FILES, "postprocess.json", "error.log", "plan.txt"):
            source, target = candidate_dir / name, save_dir / name
            if source.is_file():
                shutil.move(source, target)
            else:
                target.unlink(missing_ok=True)
        return ok, status


def generate_one(task: dict):
    image_path = task["image"]
    save_dir = task["save_dir"]
    plan_file = save_dir / "plan.txt"
    generation_api = None
    try:
        if generation_complete(task):
            return task, True, "cached"

        if image_path is None or not image_path.exists():
            return task, False, "missing_image"

        save_dir.mkdir(parents=True, exist_ok=True)
        instruction = task["instruction"]
        prompt = prompt_path.read_text(encoding="utf-8").format(
            instruction=instruction, instructions=instruction,
            robot_configuration=task["robot_configuration"],
            domain=f"\n{task['gt_domain']}\n" if eval_mode == "problem_only" else "",
        )
        voting = pddl_vote and eval_mode != "nl"
        generation_api = {"calls": 0, "seconds": 0.0}
        for retries_left in reversed(range(10)):
            attempt = {"calls": 0, "seconds": 0.0}
            try:
                output = call_gpt(
                    eval_model, image_paths=[image_path], prompt=prompt,
                    base_url=baseline["base_url"], reasoning_effort=baseline["reasoning_effort"],
                    max_retries=0, api_metrics=attempt, n=PDDL_VOTE_SAMPLES if voting else 1,
                )
                if voting:
                    if not output or not any(sample and sample.strip() for sample in output):
                        raise ValueError("Empty VLM response")
                elif eval_mode == "nl":
                    data = json.loads(strip_code_block(output or ""))
                    steps = data["plan_sequence"]
                    if not isinstance(steps, list) or not steps or any(
                        not isinstance(step, str) or not step.strip() for step in steps
                    ):
                        raise ValueError("plan_sequence must be a nonempty list of non-blank strings")
                else:
                    domain, problem = parse_generated_pddl(task, output or "")
            except APIStatusError as error:
                if not retries_left or error.status_code not in (408, 502, 503, 504):
                    raise
            except (APIConnectionError, AttributeError, IndexError, KeyError, TypeError, ValueError):
                if not retries_left:
                    raise
            else:
                if voting:
                    ok, status = vote_pddl(task, output)
                    generation_api = attempt
                    return task, ok, status
                generation_api = attempt
                break

        if eval_mode == "nl":
            if reasoning := str(data.get("reasoning", "")).strip():
                (save_dir / "reasoning.txt").write_text(reasoning, encoding="utf-8")
            plan_file.write_text("\n".join(step.strip() for step in steps), encoding="utf-8")
            (save_dir / "error.log").unlink(missing_ok=True)
            return task, True, "nl"

        ok, status = solve_generated_pddl(task, domain, problem)
        return task, ok, status

    except Exception as e:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        return task, False, str(e)
    finally:
        if generation_api is not None:
            judge_file = save_dir / "judge.json"
            record = read_judge(judge_file)
            generation_api["seconds"] = round(generation_api["seconds"], 6)
            record["generation_api"] = generation_api
            write_judge(judge_file, record)


def judge_one(task: dict):
    image_path = task["image"]
    save_dir = task["save_dir"]
    plan_file = save_dir / "plan.txt"
    judge_file = save_dir / "judge.json"
    pddl = eval_mode != "nl"
    domain_file, problem_file = (save_dir / name for name in PDDL_FILES)

    try:
        cached_status, passed = load_cached_status(task)
        if cached_status == "judge":
            return task, True, passed, "cached"

        if image_path is None or not image_path.exists():
            return task, False, False, "missing_image"

        if not plan_file.is_file():
            return task, False, False, "missing_plan"

        if pddl:
            original_plan = plan_file.read_text(encoding="utf-8")
            try:
                plan_reorder(domain_file, problem_file, plan_file, plan_file)
            except Exception as e:
                plan_file.write_text(original_plan, encoding="utf-8")
                logger.warning("Plan reordering failed; using original plan %s: %s", plan_file, e)

        candidate_plan = plan_file.read_text(encoding="utf-8").strip()
        if not candidate_plan:
            return task, False, False, "empty_plan"

        result = validate_plan_feasibility(
            model=judge_model,
            reasoning_effort=judge_reasoning_effort,
            first_img=image_path,
            instruction=task["instruction"],
            kf_actions=task["kf_actions"],
            candidate_plan=candidate_plan,
            predicted_domain=domain_file if pddl else None,
            pddl_plan=plan_file if pddl else None,
            **({"scene_context": f"The robot configuration is {task['robot_configuration']}."}
               if not pddl and "robot_configuration" in task else {}),
        )

        write_judge(judge_file, {**read_judge(judge_file), **result})
        (save_dir / "judge_error.log").unlink(missing_ok=True)
        return task, True, result["pass"] is True, "done"

    except Exception as e:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "judge_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        return task, False, False, str(e)


def write_summary(all_tasks: list[dict]) -> Path:
    pddl = eval_mode != "nl"
    records = {dataset: [] for dataset in datasets}
    for task in all_tasks:
        records[task["dataset"]].append({
            "id": task_identifier(task), "directory": task["save_dir"],
            "solved": generation_complete(task), "passed": load_cached_status(task)[1],
            "attribution": task.get("attribution") if pddl else None,
        })
    report_root = eval_root
    report_root.mkdir(parents=True, exist_ok=True)
    dataset_tag = "_".join(datasets).replace("/", "_")
    report_path = report_root / f"summary_{dataset_tag}.md"
    report_text, attribution_summary = render_markdown_report(records, report_root, pddl=pddl)
    report_path.write_text(report_text, encoding="utf-8")
    if (legacy_path := report_path.with_suffix(".log")).is_file():
        legacy_path.write_text(f"Report moved to Markdown: {report_path.name}\n", encoding="utf-8")
    if enable_attribution and pddl:
        (report_root / f"attribution_summary_{dataset_tag}.json").write_text(
            json.dumps(attribution_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return report_path


def repair_saved_task(task: dict) -> bool:
    save_dir = task["save_dir"]
    domain_file, problem_file = (save_dir / name for name in PDDL_FILES)
    try:
        original_domain = domain_file.read_text(encoding="utf-8")
        domain, problem, changes = repair_pddl(original_domain, problem_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, NotImplementedError):
        return False
    if not changes or (eval_mode == "problem_only" and domain != original_domain):
        return False

    judge_file = save_dir / "judge.json"
    try:
        generation_api = read_judge(judge_file).get("generation_api")
    except ValueError:
        generation_api = None
    if generation_api:
        write_judge(judge_file, {"generation_api": generation_api})
    else:
        judge_file.unlink(missing_ok=True)
    for name in ("plan.txt", "error.log", "judge_error.log", "attribution.json"):
        (save_dir / name).unlink(missing_ok=True)
    shutil.rmtree(save_dir / "attribution_solver", ignore_errors=True)
    domain_file.write_text(domain, encoding="utf-8")
    problem_file.write_text(problem, encoding="utf-8")
    (save_dir / "postprocess.json").write_text(
        json.dumps(changes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return solve_pddl(domain_file, problem_file, reorder_plan=False) and generation_complete(task)


def prepare_task(task):
    status, _ = load_cached_status(task)
    if status == "judge" or generation_complete(task):
        return task, True, "cached"
    if (status == "error" and eval_mode != "nl"
        and all((task["save_dir"] / name).is_file() for name in PDDL_FILES)
    ):
        if postprocess_enabled() and repair_saved_task(task):
            return task, True, "pddl_repaired"
        return task, True, "cached"
    return generate_one(task)


def generate_tasks(tasks, progress):
    total = len(tasks)
    progress("generation", 0, total)
    with ThreadPoolExecutor(max_workers=generation_max_workers) as executor:
        futures = [executor.submit(prepare_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            task, ok, info = future.result()
            if not ok:
                logger.warning("Generation failed %s/%s: %s", task["dataset"], task_identifier(task), info)
            progress("generation", completed, total)
    progress("generation", total, total, state="done")


def attribute_one(task):
    with tempfile.TemporaryDirectory() as directory:
        for name in (*PDDL_FILES, "plan.txt", "judge.json"):
            source = task["save_dir"] / name
            if source.is_file():
                shutil.copy2(source, directory)
        return attribute_failure(
            directory, task["gt_dir"], task["instruction"], task["image"],
            model=attribution_model, reasoning_effort="medium",
        )


def judge_and_attribute(all_tasks, progress):
    attribution_enabled = enable_attribution and eval_mode != "nl"
    pending_judges, known_failures = [], []
    judged = 0
    for task in all_tasks:
        status, passed = load_cached_status(task)
        complete = generation_complete(task)
        if complete and status == "judge":
            judged += 1
        elif complete:
            pending_judges.append(task)
        if attribution_enabled and not passed and (status == "judge" or not complete):
            known_failures.append(task)

    solved = judged + len(pending_judges)
    stats = {"attributed": 0, "classified": 0, "else_count": 0}
    attribution_total = len(known_failures)
    errors = []

    def report(*, judge=False, done=False):
        if judge:
            state = ("failed" if errors else "done") if done or judged == solved else "running"
            progress("judge", judged, solved, state=state)
        if attribution_enabled:
            progress("attribution", stats["attributed"], attribution_total, state="done" if done else "running")

    report(judge=True)
    with (
        ThreadPoolExecutor(max_workers=judge_max_workers) as judges,
        ThreadPoolExecutor(max_workers=attribution_max_workers) as attributions,
    ):
        futures = {attributions.submit(attribute_one, task): ("attribution", task) for task in known_failures}
        futures.update({judges.submit(judge_one, task): ("judge", task) for task in pending_judges})
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                stage, task = futures.pop(future)
                if stage == "attribution":
                    result = future.result()
                    task["attribution"] = result
                    stats["attributed"] += 1
                    if result is not None:
                        stats["else_count" if result["label"] == "else" else "classified"] += 1
                else:
                    _, ok, passed, info = future.result()
                    judged += 1
                    if not ok:
                        message = f"{task['dataset']}/{task_identifier(task)}: {info}"
                        errors.append(message)
                        logger.error("Judge request failed %s", message)
                    elif attribution_enabled and not passed:
                        futures[attributions.submit(attribute_one, task)] = ("attribution", task)
                        attribution_total += 1
                report(judge=stage == "judge")
    report(judge=True, done=True)
    if errors:
        raise RuntimeError(f"Judge errors={len(errors)}; first error: {errors[0]}. Rerun to resume.")
    return stats


def main() -> int:
    if eval_mode not in {"pddl", "problem_only", "nl"}:
        raise ValueError(f"Unknown eval_mode: {eval_mode!r}")
    print(f"full evaluation | model={eval_model} | prompt={prompt_path.name} | generation_workers={generation_max_workers} "
          f"judge_workers={judge_max_workers} attribution_enabled={enable_attribution} "
          f"attribution_workers={attribution_max_workers} rule_postprocess={postprocess_enabled()} "
          f"pddl_vote={pddl_vote and eval_mode != 'nl'}", flush=True)
    previous = {}

    def progress(stage, completed, total, *, state="running"):
        mark = state, completed * 2 // max(total, 1)
        if previous.get(stage) != mark:
            previous[stage] = mark
            print(f"{eval_model} | {stage} {completed}/{total} {state}", flush=True)

    try:
        eval_root.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=eval_root / "evaluation.log", level=logging.WARNING, force=True,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s", encoding="utf-8")
        logger.setLevel(logging.INFO)
        logger.info("Evaluation started model=%s prompt=%s rule_postprocess=%s",
                    eval_model, prompt_path.name, postprocess_enabled())
        tasks = load_tasks()
        generate_tasks(tasks, progress)
        stats = judge_and_attribute(tasks, progress)
        report = write_summary(tasks)
        statuses = [load_cached_status(task) for task in tasks]
        logger.info("Evaluation done %s", dict(total=len(tasks), solved=sum(generation_complete(task) for task in tasks),
                    judged=sum(status == "judge" for status, _ in statuses),
                    passed=sum(passed for _, passed in statuses), **stats, report=str(report)))
    except Exception as error:
        logger.exception("Evaluation failed")
        print(f"{eval_model} | failed | {type(error).__name__}: {error}", flush=True)
        return 1
    print(f"{eval_model} | done | {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
