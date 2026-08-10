"""
将 eval_results 中通过评测的 PDDL 数据整理成 ShareGPT 格式，用于多模态 SFT。

主要流程：
1. 读取 instruction，规范唯一对象编号和 on 堆叠顺序；
2. 删除缺少文件、评测未通过或不在 instruction 中的 episode；
3. 删除未使用谓词，统一 domain/problem 的 PDDL 格式，并查找对应图片；
4. 组装 user/assistant 消息，合并各数据集后写入 JSON。

注意：运行时会改写 problem.pddl，并直接删除无效 episode 目录。
"""

import json
import re
import shutil
from pathlib import Path


# ============================================================
# 配置：只需要修改这里
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "gpt-5.6-sol"
PDDL_DOMAIN_NAME = "single_arm"
TASK_DOMAINS = ["human", "human_aug"]
ROBOT_CONFIGURATION = "single-arm"

KEYFRAMES_ROOT = ROOT_DIR / "dataset/keyframes"
IMAGES_ROOT = ROOT_DIR / "tasks/images"
PROMPT_PATH = ROOT_DIR / "src/swm/prompt_templates/training_input.txt"
OUT_JSON_PATH = ROOT_DIR / f"eval_results/{MODEL_NAME}/{'_'.join(TASK_DOMAINS)}.json"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# block1 和 wooden_block1 表示真实的多个实例，不删除编号。
KEEP_NUMBERED_PREFIXES = ("block", "wooden_block")


# ============================================================
# PDDL 解析与格式化
# ============================================================

def remove_comments(text):
    """删除 PDDL 注释，并把整个文件解析成嵌套列表。"""
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def parse_pddl(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", remove_comments(text))
    stack = []
    root = None

    for token in tokens:
        if token == "(":
            expression = []
            if stack:
                stack[-1].append(expression)
            stack.append(expression)
        elif token == ")":
            root = stack.pop()
        else:
            stack[-1].append(token)

    if stack:
        raise ValueError("PDDL 括号不完整")
    return root


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


def referenced_predicates(domain_text, problem_text):
    """返回 operator 和 problem 实际使用的已声明谓词名。"""
    domain = parse_pddl(domain_text)
    problem = parse_pddl(problem_text)
    declared = {
        predicate_name(predicate)
        for section in domain[2:]
        if expression_head(section) == ":predicates"
        for predicate in section[1:]
    }
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
        if expression_head(section) == ":action":
            collect(section)
    collect(problem)
    return referenced


def predicate_order(domain_text):
    """
    按 :predicates 中的空行分组和定义顺序，记录每个谓词的排序位置。
    """
    text = remove_comments(domain_text)
    span = section_span(text, ":predicates")
    if span is None:
        return {}

    section = text[span[0]:span[1]]
    body_start = re.match(r"\(\s*:predicates\b", section, re.I).end()
    order = {}
    group = 0
    predicate_index = 0
    previous_end = body_start
    depth = 0
    start = 0

    for index in range(body_start, len(section) - 1):
        if section[index] == "(":
            if depth == 0:
                start = index
            depth += 1
        elif section[index] == ")":
            depth -= 1
            if depth == 0:
                if predicate_index and re.search(r"\n[ \t]*\n", section[previous_end:start]):
                    group += 1
                expression = parse_pddl(section[start:index + 1])
                order[predicate_name(expression)] = (group, predicate_index)
                predicate_index += 1
                previous_end = index + 1

    return order


def sort_facts(facts, order, keep_order_inside_group=False):
    def rank(item):
        index, fact = item
        name = predicate_name(fact)
        if name in order:
            group, position = order[name]
        else:
            group, position = 10_000, 100_000 + index
        if keep_order_inside_group:
            position = index
        return group, position, index

    return [fact for _, fact in sorted(enumerate(facts), key=rank)]


def format_domain(domain_text, order, referenced):
    domain = parse_pddl(domain_text)
    if expression_head(domain) != "define":
        raise ValueError("domain.pddl 缺少 define")
    lines = [f"(define (domain {PDDL_DOMAIN_NAME})"]

    for section in domain[2:]:
        name = expression_head(section)
        if name in (":requirements", ":types"):
            continue

        if len(lines) > 1:
            lines.append("")

        if name == ":predicates":
            lines.append("  (:predicates")
            lines.extend(
                "    " + pddl_line(predicate)
                for predicate in section[1:]
                if predicate_name(predicate) in referenced
            )
            lines.append("  )")
        elif name == ":action":
            lines.append(f"  (:action {section[1]}")
            for key in (":parameters", ":precondition", ":effect"):
                if key not in section:
                    continue
                value = section[section.index(key) + 1]
                if key != ":parameters" and expression_head(value) == "and":
                    facts = sort_facts(value[1:], order, keep_order_inside_group=True)
                    value = ["and", *facts]
                lines.append(f"    {key} {pddl_line(value)}")
            lines.append("  )")
        else:
            lines.append("  " + pddl_line(section))

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

    same_above = {}
    for index, atom in on_atoms:
        if atom[1] not in same_above:
            same_above[atom[1]] = []
        same_above[atom[1]].append((index, atom))

    # 如果同一物体同时在多个物体上，该关系有歧义，不加入链。
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


def format_problem(problem_text, order):
    problem = parse_pddl(problem_text)
    if expression_head(problem) != "define":
        raise ValueError("problem.pddl 缺少 define")
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
        atoms = reorder_stacks(sort_facts(init[1:], order))
        lines.append("  (:init")
        lines.extend("    " + pddl_line(atom) for atom in atoms)
        lines.append("  )")

    if goal is not None:
        lines.append("  (:goal")
        condition = goal[1]
        if expression_head(condition) == "and":
            atoms = reorder_stacks(sort_facts(condition[1:], order), goal=True)
            lines.append("    (and")
            lines.extend("      " + pddl_line(atom) for atom in atoms)
            lines.append("    )")
        else:
            lines.append("    " + pddl_line(condition))
        lines.append("  )")

    lines.extend("  " + pddl_line(section) for section in extra_sections)
    lines.append(")")
    return "\n".join(lines)


def reorder_problem_file(problem_path):
    """只在原 problem.pddl 的 :init/:goal 中写回堆叠顺序。"""
    text = problem_path.read_text(encoding="utf-8")
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
    if replacements:
        problem_path.write_text(text, encoding="utf-8")


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


def problem_path(episode_dir):
    round_dir = latest_round(episode_dir)
    if round_dir is None or not (round_dir / "problem.pddl").is_file():
        return None
    return round_dir / "problem.pddl"


def rename_numbered_objects(path):
    """仅将唯一的 object1 改为 object；存在多实例歧义时不改名。"""
    text = path.read_text(encoding="utf-8")
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
        if match is None or name.lower().startswith(KEEP_NUMBERED_PREFIXES):
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
        return

    names = sorted(rename_map, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_\-])(" + "|".join(re.escape(name) for name in names) + r")(?![A-Za-z0-9_\-])"
    )
    text = pattern.sub(lambda match: rename_map[match.group(1)], text)
    path.write_text(text, encoding="utf-8")


def preprocess_problems(eval_root):
    for episode in episode_dirs(eval_root):
        path = problem_path(episode)
        if path is not None:
            try:
                rename_numbered_objects(path)
                reorder_problem_file(path)
            except Exception as error:
                print(f"[PDDL ERROR] {path}: {error}")


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


def process_domain(task_domain, prompt_template):
    print(f"\n========== {task_domain} ==========")
    eval_root = ROOT_DIR / f"eval_results/{MODEL_NAME}/{task_domain}"
    records = read_instructions(task_domain)
    allowed = {(task_id, episode_id) for task_id, episode_id, _ in records}

    # 先预处理 problem.pddl，再删除无效 episode。
    preprocess_problems(eval_root)

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
                valid = "pass" in judge and judge["pass"] is True
            except json.JSONDecodeError:
                valid = False

        if valid:
            valid_rounds[(task_id, episode_id)] = round_dir
        else:
            shutil.rmtree(episode)
            removed += 1

    # 删除 episode 清理后产生的空 task 目录。
    for path in eval_root.iterdir():
        if path.is_dir() and not path.name.startswith("episode") and not any(path.iterdir()):
            path.rmdir()

    samples = []
    missing_episode = 0
    missing_image = 0
    for task_id, episode_id, instruction in records:
        key = (task_id, episode_id)
        if key not in valid_rounds:
            missing_episode += 1
            continue

        image_path = find_image(task_domain, task_id, episode_id)
        if image_path is None:
            missing_image += 1
            continue

        round_dir = valid_rounds[key]
        try:
            domain_raw = (round_dir / "domain.pddl").read_text(encoding="utf-8")
            problem_raw = (round_dir / "problem.pddl").read_text(encoding="utf-8")
            order = predicate_order(domain_raw)
            referenced = referenced_predicates(domain_raw, problem_raw)
            domain_text = format_domain(domain_raw, order, referenced)
            problem_text = format_problem(problem_raw, order)
        except Exception as error:
            print(f"[PDDL ERROR] {task_id}/{episode_id}: {error}")
            continue

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

    print(f"instructions    : {len(records)}")
    print(f"removed episode: {removed}")
    print(f"missing episode: {missing_episode}")
    print(f"missing image  : {missing_image}")
    print(f"saved samples  : {len(samples)}")
    return samples


def main():
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    samples = []
    for task_domain in TASK_DOMAINS:
        samples.extend(process_domain(task_domain, prompt_template))

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\ntotal samples: {len(samples)}")
    print(f"output       : {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
