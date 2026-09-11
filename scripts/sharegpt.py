"""Export validated PDDL rounds and normalize each source PDDL layout."""

import json
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# ============================================================
# 配置：只需要修改这里
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swm.pddl.planner import solve_pddl
from swm.pddl.strips import (
    goals_satisfied,
    ground_plan,
    parse_domain,
    parse_plan,
    parse_problem_model,
    rollout,
)

MODEL_NAME = "gpt-5.6-sol"
TASK_DOMAINS = ["human", "human_aug"]
ROBOT_CONFIGURATION = "single-arm"
PDDL_DOMAIN_NAME = ROBOT_CONFIGURATION.replace("-", "_")

KEYFRAMES_ROOT = ROOT_DIR / "dataset/keyframes"
IMAGES_ROOT = ROOT_DIR / "tasks/images"
PROMPT_PATH = ROOT_DIR / "src/swm/prompt_templates/training_input.txt"
OUT_JSON_PATH = ROOT_DIR / f"eval_results/{MODEL_NAME}/data/human.json"
ERROR_LOG_PATH = OUT_JSON_PATH.with_suffix(".error.log")

MAX_WORKERS = 40


# ============================================================
# PDDL 解析与格式化
# ============================================================

def parse_pddl(text):
    text = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    root = []
    stack = [root]

    for token in tokens:
        if token == "(":
            expression = []
            stack[-1].append(expression)
            stack.append(expression)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("PDDL 括号不匹配")
            stack.pop()
        else:
            stack[-1].append(token)

    if len(stack) != 1 or len(root) != 1 or not isinstance(root[0], list):
        raise ValueError("PDDL 结构不完整")
    return root[0]


def pddl_line(expression):
    if isinstance(expression, str):
        return expression
    return "(" + " ".join(pddl_line(item) for item in expression) + ")"


def expression_head(expression):
    if expression and isinstance(expression[0], str):
        return expression[0].lower()
    return ""


def predicate_name(expression):
    if expression_head(expression) == "not":
        return expression_head(expression[1])
    return expression_head(expression)


def format_domain(domain, problem, action_names, *, preserve_identity=False):
    declarations = [
        predicate
        for section in domain[2:]
        if expression_head(section) == ":predicates"
        for predicate in section[1:]
    ]
    declared = {predicate_name(predicate) for predicate in declarations}
    selected_names = set(action_names)
    referenced = set()

    def collect(expression):
        if not isinstance(expression, list):
            return
        name = expression_head(expression)
        if name in declared:
            referenced.add(name)
        for child in expression[1:]:
            collect(child)

    for section in domain[2:]:
        name = expression_head(section)
        if name != ":predicates" and (
            name != ":action" or section[1].lower() in selected_names
        ):
            collect(section)
    collect(problem)

    if preserve_identity:
        selected_actions = [
            section
            for section in domain[2:]
            if expression_head(section) == ":action"
        ]
    else:
        declarations = [
            declaration
            for declaration in declarations
            if predicate_name(declaration) in referenced
        ]
        actions = {
            section[1].lower(): section
            for section in domain[2:]
            if expression_head(section) == ":action"
        }
        selected_actions = [actions[name] for name in action_names]

    domain_name = domain[1][1] if preserve_identity else PDDL_DOMAIN_NAME
    lines = [f"(define (domain {domain_name})"]
    for section in domain[2:]:
        name = expression_head(section)
        if name == ":action":
            continue
        if name == ":requirements":
            lines.append("  (:requirements " + " ".join(section[1:]) + ")")
        elif name == ":predicates":
            lines.append("  (:predicates")
            lines.extend("    " + pddl_line(item) for item in declarations)
            lines.append("  )")
        else:
            lines.append("  " + pddl_line(section))

    for action in selected_actions:
        if len(lines) > 1:
            lines.append("")
        lines.append(f"  (:action {action[1]}")
        for key, value in zip(action[2::2], action[3::2]):
            lines.append(f"    {key} {pddl_line(value)}")
        lines.append("  )")
    lines.append(")")
    return "\n".join(lines)


def reorder_stacks(atoms, goal=False):
    """将相连的 (on A B) 按照从上到下的顺序排列。"""
    on_atoms = [
        (index, atom)
        for index, atom in enumerate(atoms)
        if len(atom) == 3 and expression_head(atom) == "on"
    ]
    if len(on_atoms) < 2:
        return atoms

    successors = {index: set() for index, _ in on_atoms}
    indegree = {index: 0 for index, _ in on_atoms}
    for upper_index, upper in on_atoms:
        for lower_index, lower in on_atoms:
            if upper_index != lower_index and upper[2] == lower[1]:
                successors[upper_index].add(lower_index)
                indegree[lower_index] += 1

    if not any(successors.values()):
        return atoms

    ready = sorted(index for index, degree in indegree.items() if degree == 0)
    ordered_indexes = []
    while ready:
        index = ready.pop(0)
        ordered_indexes.append(index)
        for successor in sorted(successors[index]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        ready.sort()

    if len(ordered_indexes) != len(on_atoms):
        return atoms

    atoms_by_index = dict(on_atoms)
    ordered_on = [atoms_by_index[index] for index in ordered_indexes]

    if goal:
        non_on = [
            atom
            for atom in atoms
            if expression_head(atom) != "on" or len(atom) != 3
        ]
        return ordered_on + non_on

    result = list(atoms)
    on_positions = [
        index
        for index, atom in enumerate(atoms)
        if len(atom) == 3 and expression_head(atom) == "on"
    ]
    for index, atom in zip(on_positions, ordered_on):
        result[index] = atom
    return result


def format_problem(problem, *, preserve_identity=False):
    objects = None
    init = None
    goal = None
    domain_section = None
    extra_sections = []
    for section in problem[2:]:
        name = expression_head(section)
        if name == ":domain" and domain_section is None:
            domain_section = section
        elif name == ":objects" and objects is None:
            objects = section
        elif name == ":init":
            init = section[1:]
        elif name == ":goal":
            if len(section) != 2:
                raise ValueError(":goal 格式错误")
            goal = section[1]
        elif name not in {":domain", ":objects"}:
            extra_sections.append(section)

    if preserve_identity:
        if domain_section is None:
            raise ValueError("problem 缺少 :domain")
        lines = ["(define " + pddl_line(problem[1]), "  " + pddl_line(domain_section)]
    else:
        lines = ["(define (problem task)", f"  (:domain {PDDL_DOMAIN_NAME})"]
    if objects is not None:
        lines.append("  " + pddl_line(objects))
    if init is not None:
        init = reorder_stacks(init)
        lines.append("  (:init")
        lines.extend("    " + pddl_line(atom) for atom in init)
        lines.append("  )")
    if goal is not None:
        atoms = goal[1:] if expression_head(goal) == "and" else [goal]
        atoms = reorder_stacks(atoms, goal=True)
        lines.extend(["  (:goal", "    (and"])
        lines.extend("      " + pddl_line(atom) for atom in atoms)
        lines.extend(["    )", "  )"])
    lines.extend("  " + pddl_line(section) for section in extra_sections)
    lines.append(")")
    return "\n".join(lines)


def atomic_write(path, text):
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


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


def validate_round(domain_path, problem_path, plan_path):
    schemas = parse_domain(domain_path)
    problem = parse_problem_model(problem_path, schemas)
    raw_plan, _ = parse_plan(plan_path)
    if not raw_plan:
        raise ValueError("没有可用 action")
    actions = ground_plan(raw_plan, schemas, problem.object_types)
    final_state = rollout(problem.init_state, actions)
    if not goals_satisfied(
        final_state,
        problem.goal_positive,
        problem.goal_negative,
    ):
        raise ValueError("plan 未达到 goal")


# ============================================================
# 生成 ShareGPT 数据
# ============================================================

def find_image(task_domain, task_id, episode_id):
    directory = KEYFRAMES_ROOT / task_domain / task_id / episode_id / "seg_00"
    if directory.is_dir():
        images = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".png"
            and path.stem.isdigit()
        ]
        if images:
            return str(min(images, key=lambda path: int(path.stem)))

    image_path = IMAGES_ROOT / task_domain / task_id / f"{episode_id}.png"
    if image_path.is_file():
        return str(image_path)
    return None


def prepare_round(item):
    key, round_dir = item
    try:
        source_paths = [
            round_dir / name
            for name in ("domain.pddl", "problem.pddl", "plan.txt")
        ]
        domain_raw, problem_raw, plan_raw = (
            path.read_text(encoding="utf-8") for path in source_paths
        )

        problem = parse_pddl(problem_raw)
        conflicts, hand_states = inspect_init(problem)
        if conflicts:
            problem_path = source_paths[1].relative_to(ROOT_DIR)
            message = f"{problem_path}: " + " | ".join(conflicts)
            return key, None, "[INIT CONFLICT] " + message, message, None

        validate_round(*source_paths)
        domain = parse_pddl(domain_raw)
        source_domain = format_domain(
            domain,
            problem,
            [],
            preserve_identity=True,
        ).strip() + "\n"
        source_problem = format_problem(problem, preserve_identity=True).strip() + "\n"
        source_plan = plan_raw
        source_changed = source_domain != domain_raw or source_problem != problem_raw
        old_actions, _ = parse_plan(source_paths[2])
        new_actions = old_actions
        if source_changed:
            with tempfile.TemporaryDirectory(prefix="sharegpt_source_pddl_") as directory:
                temporary_dir = Path(directory)
                temporary_domain = temporary_dir / "domain.pddl"
                temporary_problem = temporary_dir / "problem.pddl"
                temporary_domain.write_text(source_domain, encoding="utf-8")
                temporary_problem.write_text(source_problem, encoding="utf-8")
                if not solve_pddl(temporary_domain, temporary_problem):
                    error = (temporary_dir / "error.log").read_text(encoding="utf-8")
                    raise ValueError("PDDL 格式化后重新求解失败: " + error)
                source_plan = (temporary_dir / "plan.txt").read_text(encoding="utf-8")
                validate_round(
                    temporary_domain,
                    temporary_problem,
                    temporary_dir / "plan.txt",
                )
                new_actions, _ = parse_plan(temporary_dir / "plan.txt")

            old_steps = Counter((name, tuple(arguments)) for name, arguments in old_actions)
            new_steps = Counter((name, tuple(arguments)) for name, arguments in new_actions)
            if old_steps != new_steps:
                raise ValueError("PDDL 格式化后的新 plan 与原 plan 动作不等价")

        raw_plan = new_actions
        if not raw_plan:
            raise ValueError("没有可用 action")
        # dict 保留 plan 首次出现顺序，同时去掉重复执行的 action 名。
        action_names = list(dict.fromkeys(name for name, _arguments in raw_plan))
        domain_text = format_domain(domain, problem, action_names)
        prepared = (
            domain_text.strip() + "\n",
            format_problem(problem).strip() + "\n",
        )

        with tempfile.TemporaryDirectory(prefix="sharegpt_pddl_") as directory:
            export_paths = [Path(directory) / path.name for path in source_paths]
            for path, text in zip(export_paths, (*prepared, source_plan)):
                path.write_text(text, encoding="utf-8")
            validate_round(*export_paths)

        if source_changed:
            atomic_write(source_paths[2], source_plan)
            atomic_write(source_paths[1], source_problem)
            atomic_write(source_paths[0], source_domain)

        review = None
        if ROBOT_CONFIGURATION == "single-arm" and len(hand_states) > 1:
            message = (
                f"{source_paths[1].relative_to(ROOT_DIR)}: "
                + " | ".join(hand_states)
            )
            review = "[HAND STATE] " + message

        return key, prepared, review, None, None
    except Exception as error:  # noqa: BLE001 - 单个 episode 失败不应中断批量导出
        message = f"{round_dir.relative_to(ROOT_DIR)}: {error}"
        return key, None, "[PDDL SKIP] " + message, None, message


def process_domain(task_domain, prompt_template):
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
    allowed = {(task_id, episode_id) for task_id, episode_id, _ in records}

    valid_rounds = {}
    episodes = sorted(
        path
        for path in eval_root.glob("task_*/episode_*")
        if path.is_dir()
    )
    for episode in episodes:
        task_id, episode_id = episode.relative_to(eval_root).parts
        rounds = [
            path
            for path in episode.iterdir()
            if path.is_dir() and re.fullmatch(r"round\d+", path.name)
        ]
        round_dir = max(rounds, key=lambda path: int(path.name[5:]), default=None)
        if (task_id, episode_id) not in allowed or round_dir is None:
            continue

        domain_path = round_dir / "domain.pddl"
        problem_path = round_dir / "problem.pddl"
        judge_path = round_dir / "judge.json"
        if not (
            domain_path.is_file()
            and problem_path.is_file()
            and judge_path.is_file()
        ):
            continue
        try:
            judge = json.loads(judge_path.read_text(encoding="utf-8"))
            if judge.get("pass") is not True:
                continue
        except json.JSONDecodeError:
            continue
        valid_rounds[(task_id, episode_id)] = round_dir

    review_messages = []
    init_reviews = []
    prepared_rounds = {}
    pddl_errors = []

    workers = min(MAX_WORKERS, len(valid_rounds)) if valid_rounds else 1
    print(
        f"[{task_domain}] preparing {len(valid_rounds)} episode(s) "
        f"with {workers} processes"
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            prepare_round,
            list(valid_rounds.items()),
            chunksize=50,
        )
        for key, prepared, review, init_review, pddl_error in results:
            if prepared is not None:
                prepared_rounds[key] = prepared
            if review is not None:
                review_messages.append(review)
            if init_review is not None:
                init_reviews.append(init_review)
            if pddl_error is not None:
                pddl_errors.append(pddl_error)
                valid_rounds.pop(key)

    samples = []
    missing_episode = 0
    missing_image = 0
    for task_id, episode_id, instruction in records:
        key = (task_id, episode_id)
        if key not in valid_rounds:
            missing_episode += 1
            continue
        if key not in prepared_rounds:
            continue

        image_path = find_image(task_domain, task_id, episode_id)
        if image_path is None:
            missing_image += 1
            continue

        domain_text, problem_text = prepared_rounds[key]
        samples.append({
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\n" + prompt_template.replace(
                        "{instruction}", instruction
                    ).replace("{robot_configuration}", ROBOT_CONFIGURATION),
                },
                {
                    "role": "assistant",
                    "content": f"<domain>\n{domain_text}\n</domain>\n<problem>\n{problem_text}\n</problem>",
                },
            ],
            "images": [image_path],
        })

    print(
        f"[{task_domain}] saved={len(samples)}/{len(records)}  "
        f"missing_episode={missing_episode}  "
        f"missing_image={missing_image}  init_review={len(init_reviews)}  "
        f"pddl_skip={len(pddl_errors)}"
    )
    if init_reviews:
        print("  [REVIEW init] :init 中发现冲突状态，以下 episode 已保留但未写入训练集：")
        for message in init_reviews:
            print("   - " + message)

    return samples, review_messages


def main():
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    samples = []
    review_messages = []
    for task_domain in TASK_DOMAINS:
        domain_samples, domain_reviews = process_domain(task_domain, prompt_template)
        samples.extend(domain_samples)
        review_messages.extend(domain_reviews)

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if review_messages:
        ERROR_LOG_PATH.write_text("\n".join(review_messages) + "\n", encoding="utf-8")
        print(f"[review] {len(review_messages)} issue(s) -> {ERROR_LOG_PATH}")
    else:
        ERROR_LOG_PATH.unlink(missing_ok=True)

    print(f"[done] samples={len(samples)} -> {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
