"""Export plan-pruned PDDL to ShareGPT and save the pruned source domain."""

import json
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

# ============================================================
# 配置：只需要修改这里
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swm.pddl.export_format import (
    ACTION_HEADS,
    expression_head,
    format_problem,
    normalize_names,
    parse_pddl,
    pddl_line,
    render_domain,
    reorder_stacks as reorder_stacks,
)
from swm.pddl.strips import parse_plan

MODEL_NAME = "gpt-5.6-sol"
EXPORT_ONLY_PROBLEM = False # 额外导出only_problem
NORMALIZE_PDDL_NAMES = True  # 仅在导出的 JSON 中将 domain/problem 分别命名为 domain_name/problem_name

# ROBOT_CONFIGURATION = "single_arm"
# TASK_DOMAINS = ["droid", "bridgedata_v2"]

# ROBOT_CONFIGURATION = "dual_arm"
# TASK_DOMAINS = ["agibot", "agibot_aug", "agibotworld2026_balanced", "agibotworld2026_balanced_aug", "gelaxea", "gelaxea_aug"]

ROBOT_CONFIGURATION = "single_arm"
TASK_DOMAINS = ["human", "human_aug"]


KEYFRAMES_ROOT = ROOT_DIR / "dataset/keyframes"
IMAGES_ROOT = ROOT_DIR / "tasks/images"
PROMPT_PATH = ROOT_DIR / "src/swm/prompt_templates/training_input.txt"
ONLY_PROBLEM_PROMPT_PATH = ROOT_DIR / "src/swm/prompt_templates/training_input_only_problem.txt"
OUT_JSON_PATH = ROOT_DIR / f"eval_results/{MODEL_NAME}/data/{'_'.join(TASK_DOMAINS)}.json"
ONLY_PROBLEM_OUT_JSON_PATH = OUT_JSON_PATH.with_name(OUT_JSON_PATH.stem + "_only_problem.json")
ERROR_LOG_PATH = OUT_JSON_PATH.with_suffix(".error.log")

MAX_WORKERS = 100


def predicate_name(expression):
    if expression_head(expression) == "not":
        return expression_head(expression[1])
    return expression_head(expression)


def format_domain(domain, problem, action_names):
    """按 plan 首次出现的顺序保留动作，并裁剪未引用的谓词。"""
    declarations = [
        predicate
        for section in domain[2:]
        if expression_head(section) == ":predicates"
        for predicate in section[1:]
    ]
    declared = {predicate_name(predicate) for predicate in declarations}
    action_order = {
        name: index
        for index, name in enumerate(dict.fromkeys(name.lower() for name in action_names))
    }
    referenced = set()

    def collect(expression):
        if not isinstance(expression, list):
            return
        name = expression_head(expression)
        if name in declared:
            referenced.add(name)
        for child in expression:
            collect(child)

    for section in domain[2:]:
        name = expression_head(section)
        if name != ":predicates" and (
            name not in ACTION_HEADS or section[1].lower() in action_order
        ):
            collect(section)
    collect(problem)

    declarations = [
        declaration
        for declaration in declarations
        if predicate_name(declaration) in referenced
    ]
    selected_actions = [
        section
        for section in domain[2:]
        if expression_head(section) in ACTION_HEADS
        and section[1].lower() in action_order
    ]
    selected_actions.sort(key=lambda action: action_order[action[1].lower()])

    sections = [
        [":predicates", *declarations] if expression_head(section) == ":predicates" else section
        for section in domain[2:]
        if expression_head(section) not in ACTION_HEADS
    ]
    return render_domain([*domain[:2], *sections, *selected_actions])


def atomic_write(path, text):
    """替换成功前保留已有文件。"""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def inspect_init(problem):
    """返回 :init 冲突和手部状态。"""
    for section in problem[2:]:
        if expression_head(section) != ":init":
            continue

        supports = {}
        held = set()
        clear = set()
        hand_states = []
        for atom in section[1:]:
            name = expression_head(atom)
            if name in ("hand_free", "holding"):
                hand_states.append(pddl_line(atom))
            if name == "on" and len(atom) == 3:
                supports.setdefault(atom[1], []).append(atom[2])
            elif name == "holding" and len(atom) == 3:
                held.add(atom[2])
            elif name == "clear" and len(atom) == 2:
                clear.add(atom[1])

        conflicts = []
        for above, belows in supports.items():
            if len(set(belows)) > 1:
                conflicts.extend(f"(on {above} {below})" for below in belows)
        conflicts.extend(
            f"(holding * {obj}) + (clear {obj})" for obj in sorted(held & clear)
        )
        return conflicts, hand_states
    return [], []


# ============================================================
# 生成 ShareGPT 数据
# ============================================================

def find_image(task_domain, task_id, episode_id):
    if task_domain == "gelaxea":
        image_path = KEYFRAMES_ROOT / task_domain / task_id / episode_id / "initial_views" / "initial.png"
        if image_path.is_file():
            return str(image_path)

    # 增强指令的官方映射可能指向中间帧，优先尊重该映射。
    image_path = IMAGES_ROOT / task_domain / task_id / f"{episode_id}.png"
    if image_path.is_file():
        return str(image_path)

    directory = KEYFRAMES_ROOT / task_domain / task_id / episode_id / "seg_00"
    try:
        with os.scandir(directory) as entries:
            images = [
                Path(entry.path)
                for entry in entries
                if entry.name.lower().endswith(".png")
                and entry.name[:-4].isdigit()
                and entry.is_file()
            ]
    except (FileNotFoundError, NotADirectoryError):
        images = []
    if images:
        return str(min(images, key=lambda path: int(path.stem)))

    return None


def find_task_rounds(item):
    """选择包含 PDDL 的最大 round；不回退到较旧的有效模型。"""
    task_dir, allowed_episodes = item
    task_id = Path(task_dir).name
    with os.scandir(task_dir) as entries:
        episodes = sorted(
            (entry.name, entry.path)
            for entry in entries
            if entry.name.startswith("episode_")
            and entry.name in allowed_episodes
            and entry.is_dir()
        )

    valid_rounds = []
    warnings = []
    for episode_id, episode_dir in episodes:
        with os.scandir(episode_dir) as entries:
            rounds = [
                Path(entry.path)
                for entry in entries
                if re.fullmatch(r"round\d+", entry.name)
                and entry.is_dir()
                and any((Path(entry.path) / name).is_file() for name in ("domain.pddl", "problem.pddl"))
            ]
        round_dir = max(rounds, key=lambda path: int(path.name[5:]), default=None)
        if round_dir is None:
            continue
        if not all(
            (round_dir / name).is_file()
            for name in ("domain.pddl", "problem.pddl")
        ):
            warnings.append((
                (task_id, episode_id),
                f"[PDDL ERROR] {round_dir.relative_to(ROOT_DIR)}: 缺少 domain.pddl 或 problem.pddl；排除该 episode，不回退旧 round",
            ))
            continue
        valid_rounds.append(((task_id, episode_id), round_dir))
    return valid_rounds, warnings


def prepare_round(item):
    key, round_dir = item
    prepared = None
    review = None
    init_review = None
    try:
        source_paths = [
            round_dir / name
            for name in ("domain.pddl", "problem.pddl", "plan.txt")
        ]
        domain_raw, problem_raw = (
            path.read_text(encoding="utf-8") for path in source_paths[:2]
        )
        domain = parse_pddl(domain_raw)
        problem = parse_pddl(problem_raw)
        raw_plan, _ = parse_plan(source_paths[2])
        if not raw_plan:
            raise ValueError("没有可用 action")
        # 信任已有求解结果，仅读取动作名称用于裁剪 domain。
        conflicts, hand_states = inspect_init(problem)
        if conflicts:
            problem_path = source_paths[1].relative_to(ROOT_DIR)
            message = f"{problem_path}: " + " | ".join(conflicts)
            review = "[INIT CONFLICT] " + message
            init_review = message

        source_domain = format_domain(domain, problem, (name for name, _ in raw_plan)).strip() + "\n"
        source_problem = format_problem(problem).strip() + "\n"
        prepared = (source_domain, source_problem)
        if NORMALIZE_PDDL_NAMES:
            normalize_names(domain, problem)
            prepared = (
                format_domain(domain, problem, (name for name, _ in raw_plan)).strip() + "\n",
                format_problem(problem).strip() + "\n",
            )

        if ROBOT_CONFIGURATION == "single_arm" and len(hand_states) > 1:
            message = (
                f"{source_paths[1].relative_to(ROOT_DIR)}: "
                + " | ".join(hand_states)
            )
            hand_review = "[HAND STATE] " + message
            review = f"{review}\n{hand_review}" if review else hand_review

        if source_domain != domain_raw:
            atomic_write(source_paths[0], source_domain)

        return key, prepared, review, init_review, None
    except Exception as error:  # noqa: BLE001 - 单条失败记录后排除，继续处理其余样本
        message = f"{round_dir.relative_to(ROOT_DIR)}: {error}"
        warning = "[PDDL ERROR] " + message
        review = f"{review}\n{warning}" if review else warning
        return key, None, review, init_review, message


def prepare_episode(item):
    task_domain, key, round_dir = item
    result = prepare_round((key, round_dir))
    image_path = find_image(task_domain, *key) if result[1] is not None else None
    return result, image_path


def process_domain(task_domain, prompt_template, only_problem_template=None):
    started = perf_counter()
    eval_root = ROOT_DIR / f"eval_results/{MODEL_NAME}/{task_domain}"
    instruction_path = ROOT_DIR / f"tasks/instructions/instructions_{task_domain}.json"
    instructions = json.loads(instruction_path.read_text(encoding="utf-8"))
    records = sorted(
        (
            task_id,
            episode_id,
            instruction,
        )
        for task_id, episodes in instructions.items()
        for episode_id, instruction in episodes.items()
    )
    print(f"[{task_domain}] scanning {len(records)} episode(s)", flush=True)
    try:
        with os.scandir(eval_root) as entries:
            task_dirs = sorted(
                (entry.path, set(instructions[entry.name]))
                for entry in entries
                if entry.name.startswith("task_")
                and entry.name in instructions
                and entry.is_dir()
            )
    except (FileNotFoundError, NotADirectoryError):
        task_dirs = []

    valid_rounds = {}
    scan_warnings = {}
    if task_dirs:
        # 与 merge.py 一样，重叠共享文件系统的元数据等待，保留排序。
        with ProcessPoolExecutor(
            max_workers=min(32, MAX_WORKERS, len(task_dirs))
        ) as executor:
            for rounds, warnings in executor.map(find_task_rounds, task_dirs):
                valid_rounds.update(rounds)
                scan_warnings.update(warnings)
    scan_seconds = perf_counter() - started

    review_messages = list(scan_warnings.values())
    init_reviews = []
    prepared_rounds = {}
    pddl_warnings = list(scan_warnings.values())

    workers = min(MAX_WORKERS, len(valid_rounds)) if valid_rounds else 1
    print(
        f"[{task_domain}] preparing {len(valid_rounds)} episode(s) "
        f"with {workers} processes (scan={scan_seconds:.1f}s)",
        flush=True,
    )
    if valid_rounds:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(
                prepare_episode,
                ((task_domain, key, path) for key, path in valid_rounds.items()),
                chunksize=max(1, min(50, len(valid_rounds) // workers)),
            )
            for (key, prepared, review, init_review, pddl_warning), image_path in results:
                if prepared is not None:
                    prepared_rounds[key] = (*prepared, image_path)
                if review is not None:
                    review_messages.append(review)
                if init_review is not None:
                    init_reviews.append(init_review)
                if pddl_warning is not None:
                    pddl_warnings.append(pddl_warning)
    prepare_seconds = perf_counter() - started - scan_seconds

    samples = []
    only_problem_samples = []
    missing_episode = 0
    missing_image = 0
    for task_id, episode_id, instruction in records:
        key = (task_id, episode_id)
        if key not in valid_rounds:
            missing_episode += 1
            continue
        if key not in prepared_rounds:
            continue

        domain_text, problem_text, image_path = prepared_rounds[key]
        if image_path is None:
            missing_image += 1
            continue

        problem_output = f"<problem>\n{problem_text}\n</problem>"
        domain_output = f"<domain>\n{domain_text}\n</domain>"
        datasets = [
            (samples, prompt_template, domain_output + "\n" + problem_output),
        ]
        if only_problem_template is not None:
            datasets.append((only_problem_samples, only_problem_template, problem_output))
        for dataset, template, answer in datasets:
            dataset.append({
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>\n" + template.format(
                            instruction=instruction,
                            robot_configuration=ROBOT_CONFIGURATION,
                            domain=f"\n{domain_text}\n",
                        ),
                    },
                    {"role": "assistant", "content": answer},
                ],
                "images": [image_path],
            })

    print(
        f"[{task_domain}] saved={len(samples)}/{len(records)}  "
        f"missing_episode={missing_episode}  "
        f"missing_image={missing_image}  init_review={len(init_reviews)}  "
        f"pddl_warnings={len(pddl_warnings)}  "
        f"time={perf_counter() - started:.1f}s "
        f"(scan={scan_seconds:.1f}s, prepare+images={prepare_seconds:.1f}s)",
        flush=True,
    )

    return samples, only_problem_samples, review_messages


def main():
    started = perf_counter()
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    only_problem_template = (
        ONLY_PROBLEM_PROMPT_PATH.read_text(encoding="utf-8")
        if EXPORT_ONLY_PROBLEM
        else None
    )
    samples = []
    only_problem_samples = []
    review_messages = []
    for task_domain in TASK_DOMAINS:
        domain_samples, domain_only_problem_samples, domain_reviews = process_domain(
            task_domain, prompt_template, only_problem_template,
        )
        samples.extend(domain_samples)
        only_problem_samples.extend(domain_only_problem_samples)
        review_messages.extend(domain_reviews)

    output_files = [(OUT_JSON_PATH, samples)]
    if EXPORT_ONLY_PROBLEM:
        output_files.append((ONLY_PROBLEM_OUT_JSON_PATH, only_problem_samples))
    for path, dataset in output_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(dataset, ensure_ascii=False, indent=2))
        print(f"[done] samples={len(dataset)} -> {path}")

    if review_messages:
        atomic_write(ERROR_LOG_PATH, "\n".join(review_messages) + "\n")
        print(f"[review] {len(review_messages)} warning(s) -> {ERROR_LOG_PATH}")
    else:
        ERROR_LOG_PATH.unlink(missing_ok=True)
    print(f"[done] total time={perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
