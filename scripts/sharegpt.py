"""
将 eval_results 中通过评测的 PDDL 数据整理成 ShareGPT 格式，用于多模态 SFT。

主要流程：
1. 读取 instruction，规范唯一对象编号和 on 堆叠顺序；
2. 删除缺少文件、评测未通过或不在 instruction 中的 episode；
3. 检查 :init 中冲突状态，冲突 episode 保留但不写入训练集；
4. 仅保留 plan 使用的 action/predicate；
5. 回写清洗后的 domain/problem，并生成 ShareGPT JSON。

注意：无效 episode 会直接删除；需要人工复核的 :init 冲突只在终端汇报并跳过训练输出。
"""

import json
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


# ============================================================
# 配置：只需要修改这里
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
MAX_WORKERS = 16



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
            stack.pop()
        else:
            stack[-1].append(token)

    return root[0]


def pddl_line(expression):
    if isinstance(expression, str):
        return expression
    return "(" + " ".join(pddl_line(item) for item in expression) + ")"


def section_span(text, section_name):
    """返回 (:init ...)、(:goal ...) 等 section 在原文中的位置。"""
    match = re.search(r"\(\s*" + re.escape(section_name) + r"(?=\s|\))", text, re.I)
    if match is None:
        return None

    depth = 0
    for index in range(match.start(), len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return match.start(), index + 1


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


def plan_action_names(plan_text):
    """读取 plan 中 action 的首次出现顺序。"""
    actions = []
    seen = set()
    for raw_line in plan_text.splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line or raw_line.lstrip().startswith((";", "#")):
            continue
        line = re.sub(r"^[\d.]+\s*:\s*", "", line)
        line = re.sub(r"\s*\[\s*[\d.]+\s*\]\s*$", "", line).strip()
        match = re.fullmatch(r"\(\s*([^\s()]+)(?:\s+[^()]*)?\)", line)
        if match is None:
            continue
        name = match.group(1).lower()
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
    comments = {}
    lines = domain_text.splitlines()
    for index, line in enumerate(lines):
        match = re.search(r"\(\s*:action\s+([^\s()]+)", line, re.IGNORECASE)
        if match is None or ";" in line[:match.start()]:
            continue
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous >= 0 and lines[previous].lstrip().startswith(";"):
            comments[match.group(1).lower()] = lines[previous].strip()
    return comments


def expression_end(text, start):
    """找到从 start 处左括号开始的 PDDL 表达式末尾，忽略行注释。"""
    depth = 0
    in_comment = False
    for index in range(start, len(text)):
        character = text[index]
        if in_comment:
            if character == "\n":
                in_comment = False
            continue
        if character == ";":
            in_comment = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


def raw_action_blocks(domain_text):
    """按 action 名提取原始文本，避免改写 LLM 给出的 pre/eff 排版。"""
    blocks = {}
    pattern = re.compile(r"\(\s*:action\s+([^\s()]+)", re.IGNORECASE)
    for match in pattern.finditer(domain_text):
        line_start = domain_text.rfind("\n", 0, match.start()) + 1
        if ";" in domain_text[line_start:match.start()]:
            continue
        name = match.group(1).lower()
        end = expression_end(domain_text, match.start())
        blocks[name] = domain_text[match.start():end].rstrip()
    return blocks


def append_action(lines, action, blocks=None, comments=None):
    name = action[1]
    if comments is not None and name.lower() in comments:
        lines.append("  " + comments[name.lower()])
    if blocks is not None:
        lines.append("  " + blocks[name.lower()])
        return

    lines.append(f"  (:action {name}")
    for key, value in zip(action[2::2], action[3::2]):
        lines.append(f"    {key} {pddl_line(value)}")
    lines.append("  )")


def domain_name(domain):
    return domain[1][1]


def render_domain(domain_text, referenced, plan_actions, source=False):
    """渲染同一份 action 子集；回写源文件时保留 action 原文。"""
    domain = parse_pddl(domain_text)
    selected_actions = ordered_actions(domain, plan_actions)
    blocks = raw_action_blocks(domain_text) if source else None
    comments = action_comments(domain_text) if source else None
    name = domain_name(domain) if source else PDDL_DOMAIN_NAME
    lines = [f"(define (domain {name})"]

    for section in domain[2:]:
        section_name = expression_head(section)
        if section_name == ":action":
            continue
        if section_name == ":requirements":
            lines.append("  (:requirements " + " ".join(section[1:]) + ")")
        elif section_name == ":predicates":
            lines.append("  (:predicates")
            lines.extend(
                "    " + pddl_line(predicate)
                for predicate in section[1:]
                if predicate_name(predicate) in referenced
            )
            lines.append("  )")
        else:
            lines.append("  " + pddl_line(section))

    for action in selected_actions:
        if len(lines) > 1:
            lines.append("")
        append_action(lines, action, blocks, comments)

    lines.append(")")
    return "\n".join(lines) + ("\n" if source else "")


def format_domain(domain_text, referenced, plan_actions):
    """ShareGPT 使用的简化 domain；action 结构与源文件保持同一顺序。"""
    return render_domain(
        domain_text,
        referenced,
        plan_actions,
        source=False,
    )


def format_source_domain(domain_text, problem_text, plan_actions):
    """用于回写的 domain，保留原 domain 名、requirements 与 action 注释。"""
    referenced = referenced_predicates(domain_text, problem_text, plan_actions)
    return render_domain(
        domain_text,
        referenced,
        plan_actions,
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


def reorder_stacks(atoms, goal=False):
    """将相连的 (on A B) 按照从上到下的顺序排列。"""
    on_atoms = [
        (index, atom)
        for index, atom in enumerate(atoms)
        if len(atom) == 3 and expression_head(atom) == "on"
    ]
    if len(on_atoms) < 2:
        return atoms

    same_above = {}
    for index, atom in on_atoms:
        if atom[1] not in same_above:
            same_above[atom[1]] = []
        same_above[atom[1]].append((index, atom))

    # 冲突关系已在 episode 处理阶段拦截；这里仍不对歧义关系做自动猜测。
    links = {
        above: records[0]
        for above, records in same_above.items()
        if len(records) == 1
    }
    below = {atom[2] for _, atom in links.values()}
    tops = sorted(
        (above for above in links if above not in below),
        key=lambda above: links[above][0],
    )

    chains = []
    used = set()
    for top in tops:
        chain = []
        seen = set()
        current = top
        while current in links and current not in seen:
            seen.add(current)
            index, atom = links[current]
            chain.append((index, atom))
            current = atom[2]
        if len(chain) >= 2 and current not in seen:
            chains.append(chain)
            used.update(index for index, _ in chain)

    if not chains:
        return atoms

    chains.sort(key=lambda chain: min(index for index, _ in chain))
    ordered_on = [atom for chain in chains for _, atom in chain]
    ordered_on.extend(atom for index, atom in on_atoms if index not in used)

    if goal:
        non_on = [atom for atom in atoms if expression_head(atom) != "on" or len(atom) != 3]
        return ordered_on + non_on

    result = list(atoms)
    on_positions = [index for index, atom in enumerate(atoms) if len(atom) == 3 and expression_head(atom) == "on"]
    if len(on_positions) != len(ordered_on):
        return atoms
    for index, atom in zip(on_positions, ordered_on):
        result[index] = atom
    return result


def format_problem(problem_text):
    problem = parse_pddl(problem_text)
    objects = None
    init = None
    goal = None
    extra_sections = []

    for section in problem[2:]:
        name = expression_head(section)
        if name == ":objects" and objects is None:
            objects = section
        elif name == ":init" and init is None:
            init = section
        elif name == ":goal" and goal is None:
            goal = section
        elif name != ":domain":
            extra_sections.append(section)

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
        atoms = reorder_stacks(init[1:])
        lines.append("  (:init")
        lines.extend("    " + pddl_line(atom) for atom in atoms)
        lines.append("  )")

    if goal is not None:
        lines.append("  (:goal")
        condition = goal[1]
        atoms = condition[1:] if expression_head(condition) == "and" else [condition]
        lines.append("    (and")
        lines.extend("      " + pddl_line(atom) for atom in reorder_stacks(atoms, goal=True))
        lines.extend(["    )", "  )"])

    lines.extend("  " + pddl_line(section) for section in extra_sections)
    lines.append(")")
    return "\n".join(lines)


def reorder_problem_text(text):
    """只重排 :init/:goal 中已存在的堆叠关系，不直接写文件。"""
    replacements = []

    for section_name in (":init", ":goal"):
        span = section_span(text, section_name)
        if span is None:
            continue

        section = parse_pddl(text[span[0]:span[1]])
        if section_name == ":init":
            old_atoms = section[1:]
            new_atoms = reorder_stacks(old_atoms)
            lines = ["  (:init"]
            lines.extend("    " + pddl_line(atom) for atom in new_atoms)
            lines.append("  )")
        else:
            condition = section[1]
            old_atoms = condition[1:] if expression_head(condition) == "and" else [condition]
            new_atoms = reorder_stacks(old_atoms, goal=True)
            lines = ["  (:goal", "    (and"]
            lines.extend("      " + pddl_line(atom) for atom in new_atoms)
            lines.extend(["    )", "  )"])

        if new_atoms != old_atoms:
            replacements.append((span[0], span[1], "\n".join(lines)))

    for start, end, section in reversed(replacements):
        text = text[:start] + section + text[end:]
    return text


# ============================================================
# episode 预处理：对象改名和堆叠重排
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


def rename_numbered_objects_text(text):
    """仅将唯一的 object1 改为 object；存在多实例歧义时不改名。"""
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
    needs_review = False
    for base, items in groups.items():
        numbers = {number for _, number in items}
        if 1 not in numbers:
            continue
        names_ending_in_one = [name for name, number in items if number == 1]
        if numbers != {1} or len(names_ending_in_one) != 1 or base in objects:
            needs_review = True
        else:
            rename_map[names_ending_in_one[0]] = base

    if needs_review or not rename_map:
        return text

    names = sorted(rename_map, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_\-])(" + "|".join(re.escape(name) for name in names) + r")(?![A-Za-z0-9_\-])"
    )
    return pattern.sub(lambda match: rename_map[match.group(1)], text)


def preprocess_problem_text(text):
    """把 problem 的局部规范化纳入同一次事务式回写。"""
    return reorder_problem_text(rename_numbered_objects_text(text))


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


def prepare_round(item):
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

        source_problem = preprocess_problem_text(problem_raw)
        plan_actions = plan_action_names(plan_raw)
        source_domain = format_source_domain(domain_raw, source_problem, plan_actions)

        if source_domain != domain_raw:
            atomic_write(round_dir / "domain.pddl", source_domain)
        if source_problem != problem_raw:
            atomic_write(round_dir / "problem.pddl", source_problem)

        referenced = referenced_predicates(domain_raw, source_problem, plan_actions)
        prepared = (
            format_domain(domain_raw, referenced, plan_actions),
            format_problem(source_problem),
        )

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
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(prepare_round, valid_rounds.items(), chunksize=50)
        for key, prepared, review, init_review, pddl_error in results:
            if prepared is not None:
                prepared_rounds[key] = prepared
            if review is not None:
                review_messages.append(review)
            if init_review is not None:
                init_reviews.append(init_review)
            if pddl_error is not None:
                pddl_errors.append(pddl_error)

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
