"""
将 eval_results 中通过评测的 PDDL 数据整理成 ShareGPT 格式，用于多模态 SFT。

主要流程：
1. 读取 instruction 与只读 unified domain 的谓词标签，删除无效 episode；
2. 对每个有效最新 round 检查 :init 冲突，安全改名对象并严格校验 plan；
3. 按 plan 裁剪 action/predicate，并按语义重排 declaration、precondition、effect、init、goal；
4. 回写清洗后的 source PDDL（保留 operator 注释），同步对象改名涉及的已有计划文本；
5. 从同一份清洗结果生成无注释的 ShareGPT JSON。

注意：无效 episode 会直接删除；:init 冲突的 episode 保留 source 文件但不写入训练集；
整个流程不调用 LLM、solver 或重规划。
"""

import json
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path


# ============================================================
# 配置：只需要修改这里
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swm.pddl.postprocess import (
    build_predicate_labels,
    sort_action,
    sort_facts,
    sort_logic,
    sort_predicate_declarations,
)

MODEL_NAME = "gpt-5.6-sol"
TASK_DOMAINS = ["human", "human_aug"]
ROBOT_CONFIGURATION = "single-arm"
PDDL_DOMAIN_NAME = ROBOT_CONFIGURATION.replace("-", "_")

KEYFRAMES_ROOT = ROOT_DIR / "dataset/keyframes"
IMAGES_ROOT = ROOT_DIR / "tasks/images"
PROMPT_PATH = ROOT_DIR / "src/swm/prompt_templates/training_input.txt"
OUT_JSON_PATH = ROOT_DIR / f"eval_results/{MODEL_NAME}/{'_'.join(TASK_DOMAINS)}.json"
ERROR_LOG_PATH = OUT_JSON_PATH.with_suffix(".error.log")

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
MAX_WORKERS = 40



# ============================================================
# PDDL 解析与格式化
# ============================================================

def remove_comments(text):
    """删除 PDDL 注释，并把整个文件解析成嵌套列表。"""
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def parse_pddl(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", remove_comments(text))
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


def action_sections(domain):
    """按大小写无关的 action 名建立映射。"""
    return {
        section[1].lower(): section
        for section in domain[2:]
        if expression_head(section) == ":action"
    }


def plan_actions(plan_text):
    """严格读取 plan 中的 ground action。"""
    actions = []
    for raw_line in plan_text.splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line or raw_line.lstrip().startswith((";", "#")):
            continue
        line = re.sub(r"^[\d.]+\s*:\s*", "", line)
        line = re.sub(r"\s*\[\s*[\d.]+\s*\]\s*$", "", line).strip()
        match = re.fullmatch(r"\(\s*([^\s()]+)(?:\s+([^()]*?))?\s*\)", line)
        if match is None:
            raise ValueError(f"无法解析 plan 行: {raw_line.strip()}")
        arguments = tuple(match.group(2).split()) if match.group(2) else ()
        actions.append((match.group(1).lower(), arguments))
    if not actions:
        raise ValueError("没有可用 action")
    return actions


def plan_action_names(plan_text):
    """返回 plan 中 action 的首次出现顺序。"""
    actions = []
    seen = set()
    for name, _arguments in plan_actions(plan_text):
        if name not in seen:
            actions.append(name)
            seen.add(name)
    return actions


def ordered_actions(domain, plan_actions):
    actions = action_sections(domain)
    return [actions[name] for name in plan_actions]


def predicate_declarations(domain):
    return {
        predicate_name(predicate)
        for section in domain[2:]
        if expression_head(section) == ":predicates"
        for predicate in section[1:]
    }


def referenced_predicates(domain_text, problem_text, plan_actions):
    """返回保留 operator 和 problem 实际使用的已声明谓词名。"""
    domain = parse_pddl(domain_text)
    problem = parse_pddl(problem_text)
    declared = predicate_declarations(domain)
    selected_actions = set(plan_actions)
    referenced = set()

    def collect(expression):
        if not isinstance(expression, list):
            return
        name = expression_head(expression)
        if name in declared:
            referenced.add(name)
        for item in expression[1:]:
            collect(item)

    for section in domain[2:]:
        section_name = expression_head(section)
        if section_name == ":predicates":
            continue
        if section_name == ":action" and section[1].lower() not in selected_actions:
            continue
        collect(section)
    collect(problem)
    return referenced


def action_comments(domain_text):
    """保留每个 action 前连续的说明注释。"""
    comments = {}
    lines = domain_text.splitlines()
    for index, line in enumerate(lines):
        match = re.search(r"\(\s*:action\s+([^\s()]+)", line, re.IGNORECASE)
        if match is None or ";" in line[:match.start()]:
            continue
        previous = index - 1
        action_comments = []
        while previous >= 0:
            stripped = lines[previous].strip()
            if stripped.startswith(";"):
                action_comments.append(lines[previous])
            elif stripped:
                break
            previous -= 1
        if action_comments:
            comments[match.group(1).lower()] = list(reversed(action_comments))
    return comments


def append_action(lines, action, comments=None):
    name = action[1]
    if comments is not None and name.lower() in comments:
        lines.extend(comments[name.lower()])

    lines.append(f"  (:action {name}")
    for key, value in zip(action[2::2], action[3::2]):
        lines.append(f"    {key} {pddl_line(value)}")
    lines.append("  )")


def domain_name(domain):
    return domain[1][1]


def render_domain(domain_text, referenced, plan_actions, labels, source=False):
    """渲染同一份已裁剪、已排序的 action 子集。"""
    domain = parse_pddl(domain_text)
    selected_actions = ordered_actions(domain, plan_actions)
    comments = action_comments(domain_text) if source else None
    name = domain_name(domain) if source else PDDL_DOMAIN_NAME
    declarations = [
        predicate
        for section in domain[2:]
        if expression_head(section) == ":predicates"
        for predicate in section[1:]
        if predicate_name(predicate) in referenced
    ]
    declarations = sort_predicate_declarations(declarations, labels)
    declaration_order = {
        predicate_name(predicate): index
        for index, predicate in enumerate(declarations)
    }
    for action in selected_actions:
        sort_action(action, labels, declaration_order)
    lines = [f"(define (domain {name})"]

    for section in domain[2:]:
        section_name = expression_head(section)
        if section_name == ":action":
            continue
        if section_name == ":requirements":
            lines.append("  (:requirements " + " ".join(section[1:]) + ")")
        elif section_name == ":predicates":
            lines.append("  (:predicates")
            lines.extend("    " + pddl_line(predicate) for predicate in declarations)
            lines.append("  )")
        else:
            lines.append("  " + pddl_line(section))

    for action in selected_actions:
        if len(lines) > 1:
            lines.append("")
        append_action(lines, action, comments)

    lines.append(")")
    return "\n".join(lines) + ("\n" if source else "")


def format_domain(domain_text, referenced, plan_actions, labels):
    """ShareGPT 使用的简化 domain；action 结构与源文件保持同一顺序。"""
    return render_domain(
        domain_text,
        referenced,
        plan_actions,
        labels,
        source=False,
    )


def format_source_domain(domain_text, problem_text, plan_actions, labels):
    """用于回写的 domain，保留原 domain 名、requirements 与 action 注释。"""
    referenced = referenced_predicates(domain_text, problem_text, plan_actions)
    return render_domain(
        domain_text,
        referenced,
        plan_actions,
        labels,
        source=True,
    )


def init_conflicts(problem_text):
    """返回 :init 中需要人工复核的冲突状态。"""
    problem = parse_pddl(problem_text)
    for section in problem[2:]:
        if expression_head(section) != ":init":
            continue

        conflicts = []
        supports = {}
        held = set()
        clear = set()

        for atom in section[1:]:
            name = expression_head(atom)
            if name == "on" and len(atom) == 3:
                supports.setdefault(atom[1], []).append(atom[2])
            elif name == "holding" and len(atom) == 3:
                held.add(atom[2])
            elif name == "clear" and len(atom) == 2:
                clear.add(atom[1])

        for above, belows in supports.items():
            if len(set(belows)) > 1:
                conflicts.extend(f"(on {above} {below})" for below in belows)
        conflicts.extend(f"(holding * {obj}) + (clear {obj})" for obj in sorted(held & clear))
        return conflicts
    return []


def domain_predicate_order(domain_text):
    domain = parse_pddl(domain_text)
    for section in domain[2:]:
        if expression_head(section) == ":predicates":
            return {
                predicate_name(predicate): index
                for index, predicate in enumerate(section[1:])
            }
    raise ValueError("domain 缺少 :predicates")


def sorted_problem_parts(problem, labels, declaration_order):
    init = None
    goal = None
    for section in problem[2:]:
        if expression_head(section) == ":init":
            init = sort_facts(section[1:], labels, declaration_order)
        elif expression_head(section) == ":goal":
            if len(section) != 2:
                raise ValueError(":goal 格式错误")
            goal = sort_logic(section[1], labels, declaration_order, {})
    return init, goal


def format_problem(problem_text, labels, declaration_order):
    problem = parse_pddl(problem_text)
    objects = None
    extra_sections = []

    for section in problem[2:]:
        name = expression_head(section)
        if name == ":objects" and objects is None:
            objects = section
        elif name not in {":domain", ":objects", ":init", ":goal"}:
            extra_sections.append(section)

    init, goal = sorted_problem_parts(problem, labels, declaration_order)

    lines = ["(define (problem task)", f"  (:domain {PDDL_DOMAIN_NAME})"]

    if objects is not None:
        object_names = []
        index = 1
        while index < len(objects):
            if objects[index] == "-":
                index += 2
            else:
                object_names.append(objects[index])
                index += 1
        lines.append("  (:objects " + " ".join(object_names) + ")")

    if init is not None:
        lines.append("  (:init")
        lines.extend("    " + pddl_line(atom) for atom in init)
        lines.append("  )")

    if goal is not None:
        lines.append("  (:goal")
        condition = goal
        atoms = condition[1:] if expression_head(condition) == "and" else [condition]
        lines.append("    (and")
        lines.extend("      " + pddl_line(atom) for atom in atoms)
        lines.extend(["    )", "  )"])

    lines.extend("  " + pddl_line(section) for section in extra_sections)
    lines.append(")")
    return "\n".join(lines)


def format_source_problem(problem_text, labels, declaration_order):
    """保留 source problem 的名称、对象与其他 section，只重排 literal。"""
    problem = parse_pddl(problem_text)
    init, goal = sorted_problem_parts(problem, labels, declaration_order)
    lines = ["(define " + pddl_line(problem[1])]
    for section in problem[2:]:
        name = expression_head(section)
        if name == ":init":
            lines.append("  (:init")
            lines.extend("    " + pddl_line(atom) for atom in init)
            lines.append("  )")
        elif name == ":goal":
            lines.append("  (:goal")
            if expression_head(goal) == "and":
                lines.append("    (and")
                lines.extend("      " + pddl_line(atom) for atom in goal[1:])
                lines.append("    )")
            else:
                lines.append("    " + pddl_line(goal))
            lines.append("  )")
        else:
            lines.append("  " + pddl_line(section))
    lines.append(")")
    return "\n".join(lines) + "\n"


# ============================================================
# episode 预处理：对象改名
# ============================================================

def latest_round(episode_dir):
    rounds = [
        path for path in episode_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"round\d+", path.name)
    ]
    return max(rounds, key=lambda path: int(path.name[5:]), default=None)


def episode_dirs(eval_root):
    paths = list(eval_root.glob("task_*/episode_*")) + list(eval_root.glob("episode_*"))
    return sorted(path for path in paths if path.is_dir())


def init_hand_states(problem_text):
    problem = parse_pddl(problem_text)
    for section in problem[2:]:
        if expression_head(section) == ":init":
            return [
                pddl_line(atom) for atom in section[1:]
                if expression_head(atom) in ("hand_free", "holding")
            ]
    return []


def numbered_object_renames(text):
    """返回可安全改为无编号形式的唯一对象映射。"""
    objects = []
    for section in parse_pddl(text)[2:]:
        if expression_head(section) != ":objects":
            continue
        index = 1
        while index < len(section):
            if section[index] == "-":
                index += 2
            else:
                objects.append(section[index])
                index += 1
        break

    groups = {}
    for name in objects:
        match = re.fullmatch(r"(.+?)(?:_)?(\d+)", name)
        if match is None:
            continue
        base, number = match.group(1), int(match.group(2))
        if base not in groups:
            groups[base] = []
        groups[base].append((name, number))

    rename_map = {}
    for base, items in groups.items():
        numbers = {number for _, number in items}
        if 1 not in numbers:
            continue
        names_ending_in_one = [name for name, number in items if number == 1]
        if numbers == {1} and len(names_ending_in_one) == 1 and base not in objects:
            rename_map[names_ending_in_one[0]] = base
    return rename_map


def replace_object_tokens(text, rename_map):
    if not rename_map:
        return text
    replacements = {name.lower(): value for name, value in rename_map.items()}
    names = sorted(rename_map, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_\-])(" + "|".join(re.escape(name) for name in names) + r")(?![A-Za-z0-9_\-])",
        re.IGNORECASE,
    )
    return pattern.sub(lambda match: replacements[match.group(1).lower()], text)


def action_parameters(action):
    for key, value in zip(action[2::2], action[3::2]):
        if key.lower() == ":parameters" and isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item.startswith("?")]
    raise ValueError(f"action {action[1]} 缺少 :parameters")


def problem_objects(problem_text):
    for section in parse_pddl(problem_text)[2:]:
        if expression_head(section) == ":objects":
            objects = set()
            index = 1
            while index < len(section):
                if section[index] == "-":
                    index += 2
                else:
                    objects.add(section[index].lower())
                    index += 1
            return objects
    raise ValueError("problem 缺少 :objects")


def validate_plan(domain_text, problem_text, plan_text):
    actions = action_sections(parse_pddl(domain_text))
    objects = problem_objects(problem_text)
    plan = plan_actions(plan_text)
    for name, arguments in plan:
        if name not in actions:
            raise ValueError(f"plan action '{name}' 未在 domain 定义")
        if len(arguments) != len(action_parameters(actions[name])):
            raise ValueError(f"plan action '{name}' 参数数量不匹配")
        unknown = [argument for argument in arguments if argument.lower() not in objects]
        if unknown:
            raise ValueError(f"plan action '{name}' 使用未声明对象: {', '.join(unknown)}")
    return plan


def atomic_write(path, text):
    """避免求解或格式化异常时留下半写入的单个 PDDL 文件。"""
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




# ============================================================
# 清理 episode 并生成 ShareGPT 数据
# ============================================================

def read_instructions(task_domain):
    path = ROOT_DIR / f"tasks/instructions/instructions_{task_domain}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for task_id, episodes in data.items():
        for episode_id, instruction in episodes.items():
            if isinstance(instruction, list):
                instruction = "\n".join(instruction)
            records.append((task_id, episode_id, instruction))
    return sorted(records, key=lambda record: (record[0], record[1]))


def find_image(task_domain, task_id, episode_id):
    keyframe_dirs = []
    if task_id is not None:
        keyframe_dirs.append(KEYFRAMES_ROOT / task_domain / task_id / episode_id / "seg_00")
    keyframe_dirs.append(KEYFRAMES_ROOT / task_domain / episode_id / "seg_00")

    for directory in keyframe_dirs:
        if directory.is_dir():
            images = [
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem.isdigit()
            ]
            if images:
                return str(min(images, key=lambda path: int(path.stem)))

    image_paths = []
    if task_id is not None:
        image_paths.extend(
            IMAGES_ROOT / task_domain / task_id / f"{episode_id}{suffix}"
            for suffix in IMAGE_SUFFIXES
        )
    image_paths.extend(
        IMAGES_ROOT / task_domain / f"{episode_id}{suffix}"
        for suffix in IMAGE_SUFFIXES
    )
    for path in image_paths:
        if path.is_file():
            return str(path)
    return None


def prepare_round(item, labels):
    key, round_dir = item
    try:
        domain_raw = (round_dir / "domain.pddl").read_text(encoding="utf-8")
        problem_raw = (round_dir / "problem.pddl").read_text(encoding="utf-8")
        plan_raw = (round_dir / "plan.txt").read_text(encoding="utf-8")

        conflicts = init_conflicts(problem_raw)
        if conflicts:
            problem_path = (round_dir / "problem.pddl").relative_to(ROOT_DIR)
            message = f"{problem_path}: " + " | ".join(conflicts)
            return key, None, "[INIT CONFLICT] " + message, message, None

        rename_map = numbered_object_renames(problem_raw)
        renamed_problem = replace_object_tokens(problem_raw, rename_map)
        renamed_plan = replace_object_tokens(plan_raw, rename_map)
        plan = validate_plan(domain_raw, renamed_problem, renamed_plan)
        plan_names = []
        for name, _arguments in plan:
            if name not in plan_names:
                plan_names.append(name)

        source_domain = format_source_domain(
            domain_raw,
            renamed_problem,
            plan_names,
            labels,
        )
        declaration_order = domain_predicate_order(source_domain)
        source_problem = format_source_problem(
            renamed_problem,
            labels,
            declaration_order,
        )
        referenced = predicate_declarations(parse_pddl(source_domain))
        prepared = (
            format_domain(source_domain, referenced, plan_names, labels),
            format_problem(source_problem, labels, declaration_order),
        )

        updates = [
            (round_dir / "domain.pddl", source_domain),
            (round_dir / "problem.pddl", source_problem),
            (round_dir / "plan.txt", renamed_plan),
        ]
        for name in ("plan_nl.txt",):
            path = round_dir / name
            if path.is_file():
                updates.append((path, replace_object_tokens(path.read_text(encoding="utf-8"), rename_map)))
        kf_plan = round_dir.parent / "kf_plan.txt"
        if kf_plan.is_file():
            updates.append((kf_plan, replace_object_tokens(kf_plan.read_text(encoding="utf-8"), rename_map)))
        for path, text in updates:
            if path.read_text(encoding="utf-8") != text:
                atomic_write(path, text)

        review = None
        if ROBOT_CONFIGURATION == "single-arm":
            hand_states = init_hand_states(source_problem)
            if len(hand_states) > 1:
                message = (
                    f"{(round_dir / 'problem.pddl').relative_to(ROOT_DIR)}: "
                    + " | ".join(hand_states)
                )
                review = "[HAND STATE] " + message

        return key, prepared, review, None, None
    except Exception as error:
        message = f"{round_dir.relative_to(ROOT_DIR)}: {error}"
        return key, None, "[PDDL SKIP] " + message, None, message


def process_domain(task_domain, prompt_template):
    eval_root = ROOT_DIR / f"eval_results/{MODEL_NAME}/{task_domain}"
    records = read_instructions(task_domain)
    allowed = {(task_id, episode_id) for task_id, episode_id, _ in records}
    labels = build_predicate_labels(
        parse_pddl((eval_root / "unified_domain.pddl").read_text(encoding="utf-8"))
    )

    valid_rounds = {}
    removed = 0
    for episode in episode_dirs(eval_root):
        parts = episode.relative_to(eval_root).parts
        task_id, episode_id = (None, parts[0]) if len(parts) == 1 else parts
        round_dir = latest_round(episode)
        valid = (task_id, episode_id) in allowed and round_dir is not None

        if valid:
            domain_path = round_dir / "domain.pddl"
            problem_file = round_dir / "problem.pddl"
            judge_path = round_dir / "judge.json"
            valid = domain_path.is_file() and problem_file.is_file() and judge_path.is_file()
        if valid:
            try:
                judge = json.loads(judge_path.read_text(encoding="utf-8"))
                valid = judge.get("pass") is True
            except json.JSONDecodeError:
                valid = False

        if valid:
            valid_rounds[(task_id, episode_id)] = round_dir
        else:
            shutil.rmtree(episode)
            removed += 1

    for path in eval_root.iterdir():
        if path.is_dir() and not path.name.startswith("episode") and not any(path.iterdir()):
            path.rmdir()

    review_messages = []
    init_reviews = []
    prepared_rounds = {}
    pddl_errors = []

    workers = min(MAX_WORKERS, len(valid_rounds)) if valid_rounds else 1
    print(f"[{task_domain}] preparing {len(valid_rounds)} episode(s) with {workers} processes")
    invalid_keys = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            partial(prepare_round, labels=labels),
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
                invalid_keys.append(key)

    for key in invalid_keys:
        shutil.rmtree(valid_rounds.pop(key).parent)
        removed += 1

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
        f"removed={removed}  missing_episode={missing_episode}  "
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
