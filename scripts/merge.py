"""Merge the latest episode domains into one deduplicated operator library."""

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swm.pddl.typing import TypeHierarchy, render_typed_symbols, typed_symbol_map


# 只需要修改这里。
DATASET = "human"
DATASET_ROOT = PROJECT_ROOT / "eval_results" / "gpt-5.6-sol"
MAX_WORKERS = 20
TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")

PddlNode = Union[str, list["PddlNode"]]


@dataclass
class PredicateItem:
    name: str
    arity: int
    expr: str
    leading_comments: list[str] = field(default_factory=list)
    inline_comment: str = ""
    sources: list[str] = field(default_factory=list)
    argument_names: tuple[str, ...] = ()


@dataclass
class ActionItem:
    name: str
    param_arity: int
    signature: str
    block_text: str
    leading_comments: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    parameter_types: tuple[str, ...] = ()


def remove_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def find_matching_paren(text: str, start: int) -> int:
    if text[start] != "(":
        raise ValueError(f"位置 {start} 不是左括号")

    depth = 0
    in_comment = False
    for index in range(start, len(text)):
        char = text[index]
        if in_comment:
            if char == "\n":
                in_comment = False
        elif char == ";":
            in_comment = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index

    raise ValueError("括号不匹配")


def find_token(text: str, token: str, start: int = 0) -> int:
    token = token.lower()
    in_comment = False
    for index in range(start, len(text) - len(token) + 1):
        char = text[index]
        if in_comment:
            if char == "\n":
                in_comment = False
        elif char == ";":
            in_comment = True
        elif text[index : index + len(token)].lower() == token:
            return index
    return -1


def parse_sexp(text: str) -> PddlNode:
    root: list[PddlNode] = []
    stack = [root]

    for token in TOKEN_RE.findall(text):
        if token == "(":
            node: list[PddlNode] = []
            stack[-1].append(node)
            stack.append(node)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("PDDL 表达式存在多余右括号")
            stack.pop()
        else:
            stack[-1].append(token)

    if len(stack) != 1:
        raise ValueError("PDDL 表达式缺少右括号")
    if len(root) != 1:
        raise ValueError("PDDL 表达式解析后仍有剩余 token")
    return root[0]


def canonical_signature(
    parameters_text: str,
    precondition_text: str,
    effect_text: str,
) -> tuple[int, str, tuple[str, ...]]:
    """Normalize variable names and unordered Boolean terms for action deduplication."""
    parameters = parse_sexp(parameters_text)
    if not isinstance(parameters, list):
        raise ValueError("action parameters must be a list")
    param_vars, param_types, _ = typed_symbol_map(
        parameters,
        context="action parameters",
    )
    param_vars = [variable.lower() for variable in param_vars]
    precondition = parse_sexp(precondition_text)
    effect = parse_sexp(effect_text)
    usage = {variable: [] for variable in param_vars}

    def collect_usage(node: PddlNode, prefix: str) -> None:
        if not isinstance(node, list) or not node or not isinstance(node[0], str):
            return
        head = node[0].lower()
        if head in {"and", "or"}:
            for child in node[1:]:
                collect_usage(child, prefix)
        elif head == "not":
            if len(node) == 2:
                collect_usage(node[1], prefix + ":not")
        else:
            for index, argument in enumerate(node[1:]):
                if isinstance(argument, str) and argument.startswith("?"):
                    variable = argument.lower()
                    usage.setdefault(variable, []).append(f"{prefix}:{head}:{index}")

    collect_usage(precondition, "pre")
    collect_usage(effect, "eff")
    ordered_vars = sorted(
        param_vars,
        key=lambda variable: (tuple(sorted(usage[variable])), variable),
    )
    var_map = {
        variable: f"?v{index}" for index, variable in enumerate(ordered_vars)
    }

    def canon(node: PddlNode) -> str:
        if isinstance(node, str):
            token = node.lower()
            if token.startswith("?"):
                if token not in var_map:
                    var_map[token] = f"?v{len(var_map)}"
                return var_map[token]
            return token
        if not node:
            return "()"
        if not isinstance(node[0], str):
            raise ValueError("非法 PDDL 表达式：head 是 list")

        head = node[0].lower()
        if head in {"and", "or"}:
            children = []
            for child in node[1:]:
                if (
                    isinstance(child, list)
                    and child
                    and isinstance(child[0], str)
                    and child[0].lower() == head
                ):
                    children.extend(canon(grandchild) for grandchild in child[1:])
                else:
                    children.append(canon(child))
            children.sort()
        else:
            children = [canon(child) for child in node[1:]]
        return "(" + " ".join([head, *children]) + ")"

    ordered_types = tuple(param_types[variable] for variable in ordered_vars)
    signature = (
        f"arity={len(param_vars)} | types={ordered_types} | "
        f"pre={canon(precondition)} | eff={canon(effect)}"
    )
    parameter_types = tuple(param_types[variable] for variable in param_vars)
    return len(param_vars), signature, parameter_types


def find_domain_files(root_dir: Path) -> list[Path]:
    """Find the highest numbered round containing a domain for every episode."""
    domain_files = []
    for episode_dir in sorted(root_dir.glob("task_*/episode_*")):
        if not episode_dir.is_dir():
            continue

        candidates = []
        for round_dir in sorted(
            episode_dir.iterdir(), key=lambda path: path.name.lower()
        ):
            match = re.fullmatch(
                r"roun(?:d)?[_\-]?(\d+)", round_dir.name, re.IGNORECASE
            )
            domain_path = round_dir / "domain.pddl"
            if not round_dir.is_dir() or not match or not domain_path.exists():
                continue
            key = (
                int(match.group(1)),
                int(round_dir.name.lower().startswith("round")),
            )
            candidates.append((key, domain_path))

        if candidates:
            domain_files.append(max(candidates, key=lambda item: item[0])[1])
    return domain_files


def parse_domain_file(domain_path: Path):
    """Parse one domain without letting a malformed file stop the full merge."""
    try:
        text = domain_path.read_text(encoding="utf-8")
        source = str(domain_path)
        domain = parse_sexp(remove_comments(text))
        type_sections = [
            section
            for section in domain[2:]
            if isinstance(section, list)
            and section
            and str(section[0]).lower() == ":types"
        ]
        if len(type_sections) > 1:
            raise ValueError("重复 :types 块")
        type_items = type_sections[0][1:] if type_sections else []
        type_parents = TypeHierarchy.from_declaration(type_items).parents
        type_order = list(type_parents)

        predicates = []
        pred_start = find_token(text, "(:predicates")
        if pred_start == -1:
            raise ValueError("没有找到 :predicates 块")
        pred_end = find_matching_paren(text, pred_start)
        pred_block = text[pred_start : pred_end + 1]
        header = re.match(r"\(\s*:predicates\b", pred_block, re.IGNORECASE)
        if not header:
            raise ValueError(":predicates 块格式错误")

        body = pred_block[header.end() : -1]
        position = 0
        while position < len(body):
            leading_comments = []
            while position < len(body):
                while position < len(body) and body[position].isspace():
                    position += 1
                if position == len(body) or body[position] != ";":
                    break
                line_end = body.find("\n", position)
                if line_end == -1:
                    line_end = len(body)
                leading_comments.append(body[position:line_end].strip())
                position = line_end

            if position == len(body):
                break
            if body[position] != "(":
                position += 1
                continue

            expr_end = find_matching_paren(body, position)
            expr = body[position : expr_end + 1].strip()
            position = expr_end + 1
            while position < len(body) and body[position] in " \t":
                position += 1

            inline_comment = ""
            if position < len(body) and body[position] == ";":
                line_end = body.find("\n", position)
                if line_end == -1:
                    line_end = len(body)
                inline_comment = body[position:line_end].strip()
                position = line_end

            declaration = parse_sexp(remove_comments(expr))
            if (
                not isinstance(declaration, list)
                or not declaration
                or not isinstance(declaration[0], str)
            ):
                raise ValueError(f"无法解析 predicate: {expr}")
            name = declaration[0]
            argument_names, _, _ = typed_symbol_map(
                declaration[1:],
                context=f"predicate {name}",
            )
            predicates.append(
                PredicateItem(
                    name=name,
                    arity=len(argument_names),
                    expr="(" + " ".join([name, *argument_names]) + ")",
                    leading_comments=leading_comments,
                    inline_comment=inline_comment,
                    sources=[source],
                    argument_names=tuple(argument_names),
                )
            )

        actions = []
        position = 0
        while True:
            action_start = find_token(text, "(:action", position)
            if action_start == -1:
                break
            action_end = find_matching_paren(text, action_start)
            action_block = text[action_start : action_end + 1].strip()

            preceding_lines = text[position:action_start].splitlines()
            while preceding_lines and not preceding_lines[-1].strip():
                preceding_lines.pop()
            leading_comments = []
            while preceding_lines and preceding_lines[-1].lstrip().startswith(";"):
                leading_comments.append(preceding_lines.pop().strip())
            leading_comments.reverse()

            clean_action = remove_comments(action_block)
            action_match = re.search(
                r"\(\s*:action\s+([^\s()]+)", clean_action, re.IGNORECASE
            )
            if not action_match:
                raise ValueError("action 块开头格式错误")
            action_name = action_match.group(1).strip()

            fields = {}
            for keyword in (":parameters", ":precondition", ":effect"):
                field_match = re.search(
                    rf"{re.escape(keyword)}\b", clean_action, re.IGNORECASE
                )
                if not field_match:
                    raise ValueError(f"{action_name} 没有找到 {keyword}")
                field_start = field_match.end()
                while (
                    field_start < len(clean_action)
                    and clean_action[field_start].isspace()
                ):
                    field_start += 1
                if field_start == len(clean_action) or clean_action[field_start] != "(":
                    raise ValueError(
                        f"{action_name} 的 {keyword} 后面不是括号表达式"
                    )
                field_end = find_matching_paren(clean_action, field_start)
                fields[keyword] = clean_action[field_start : field_end + 1].strip()

            param_arity, signature, parameter_types = canonical_signature(
                fields[":parameters"],
                fields[":precondition"],
                fields[":effect"],
            )
            actions.append(
                ActionItem(
                    name=action_name,
                    param_arity=param_arity,
                    signature=signature,
                    block_text=action_block,
                    leading_comments=leading_comments,
                    sources=[source],
                    parameter_types=parameter_types,
                )
            )
            position = action_end + 1

        return True, source, predicates, actions, (type_order, type_parents), ""
    except Exception as error:
        return False, str(domain_path), [], [], ([], {}), str(error)


def merge_predicates(predicates: list[PredicateItem]) -> list[PredicateItem]:
    merged: dict[str, tuple[PredicateItem, set[str]]] = {}

    for item in predicates:
        key = item.name.lower()
        if key not in merged:
            merged[key] = item, set(item.sources)
            continue

        old, seen_sources = merged[key]
        if old.arity != item.arity:
            raise ValueError(
                f"predicate arity conflicts require semantic cleanup: {item.name}"
            )
        if not (old.leading_comments or old.inline_comment) and (
            item.leading_comments or item.inline_comment
        ):
            old.leading_comments = item.leading_comments
            old.inline_comment = item.inline_comment

        for source in item.sources:
            if source not in seen_sources:
                seen_sources.add(source)
                old.sources.append(source)

    return sorted(
        (item for item, _ in merged.values()),
        key=lambda item: (item.name.lower(), item.arity, item.expr.lower()),
    )


def resolve_predicate_arity_collisions(
    predicates: list[PredicateItem],
    actions: list[ActionItem],
) -> dict[str, list[int]]:
    """Reject predicate overloading, which classic PDDL does not support."""
    arities_by_name: dict[str, set[int]] = {}
    for item in predicates:
        arities_by_name.setdefault(item.name.lower(), set()).add(item.arity)

    conflicts = {
        name: sorted(arities)
        for name, arities in arities_by_name.items()
        if len(arities) > 1
    }
    if not conflicts:
        return {}

    sources_by_name: dict[str, dict[str, set[int]]] = {}
    for item in predicates:
        name = item.name.lower()
        if name in conflicts:
            for source in item.sources:
                sources_by_name.setdefault(name, {}).setdefault(source, set()).add(
                    item.arity
                )
    within_source = [
        name
        for name, sources in sources_by_name.items()
        if any(len(arities) > 1 for arities in sources.values())
    ]
    if within_source:
        raise ValueError(
            "单个 source 内 predicate arity 冲突: "
            + ", ".join(sorted(within_source))
        )
    raise ValueError(
        "predicate arity conflicts require semantic cleanup: "
        + ", ".join(sorted(conflicts))
    )


def merge_actions(actions: list[ActionItem]) -> list[tuple[str, ActionItem]]:
    """Merge equivalent schemas and number same-name contract variants."""
    grouped: dict[str, list[ActionItem]] = {}
    display_names: dict[str, str] = {}

    for item in actions:
        if re.search(r"(?:_|-)\d+$", item.name):
            print(f"[WARN] numeric action suffix: {item.name} from {item.sources[:3]}")
        group_key = item.name.lower()
        display_names.setdefault(group_key, item.name)
        grouped.setdefault(group_key, []).append(item)

    final_actions = []
    for group_key, variants in grouped.items():
        ordered_variants = sorted(variants, key=lambda item: item.signature)
        if len(ordered_variants) == 1:
            final_actions.append((display_names[group_key], ordered_variants[0]))
        else:
            final_actions.extend(
                (f"{display_names[group_key]}_{index}", item)
                for index, item in enumerate(ordered_variants, start=1)
            )

    def natural_name_key(item: tuple[str, ActionItem]):
        return tuple(
            (1, int(part)) if part.isdigit() else (0, part)
            for part in re.split(r"(\d+)", item[0].lower())
            if part
        )

    return sorted(final_actions, key=natural_name_key)


def write_outputs(
    type_order: list[str],
    type_parents: dict[str, str],
    predicates: list[PredicateItem],
    actions: list[tuple[str, ActionItem]],
    domain_name: str,
    output_path: Path,
    source_json_path: Path,
) -> None:
    lines = [
        f"(define (domain {domain_name})",
        "  (:requirements :strips :typing :negative-preconditions :equality)",
        "  (:types " + " ".join(render_typed_symbols(type_order, type_parents)) + ")",
        "  (:predicates",
    ]

    for item in predicates:
        lines.extend(f"    {comment}" for comment in item.leading_comments)
        expr_lines = item.expr.strip().splitlines()
        if item.inline_comment:
            expr_lines[-1] = f"{expr_lines[-1].rstrip()} {item.inline_comment}"
        lines.extend(f"    {line.rstrip()}" for line in expr_lines)

    lines.extend(("  )", ""))
    source_json = []
    for final_name, item in actions:
        lines.extend(f"  {comment}" for comment in item.leading_comments)
        renamed_block = re.sub(
            r"(\(\s*:action\s+)([^\s()]+)",
            lambda match: match.group(1) + final_name,
            item.block_text,
            count=1,
            flags=re.IGNORECASE,
        )
        lines.extend(f"  {line.rstrip()}" for line in renamed_block.strip().splitlines())
        lines.append("")
        source_json.append(
            {
                "final_name": final_name,
                "original_name": item.name,
                "param_arity": item.param_arity,
                "signature": item.signature,
                "sources": sorted(item.sources),
            }
        )

    lines[-1] = ")"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_json_path.write_text(
        json.dumps(source_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    root_dir = DATASET_ROOT / DATASET
    output_path = root_dir / "unified_domain.pddl"
    source_json_path = root_dir / "unified_operator_sources.json"
    domain_files = find_domain_files(root_dir)
    if not domain_files:
        raise FileNotFoundError(f"在 {root_dir} 下没有找到 domain.pddl")

    print(f"[INFO] ROOT_DIR: {root_dir}")
    print(f"[INFO] DOMAIN_NAME: {DATASET}")
    print(f"[INFO] 找到 {len(domain_files)} 个最大 round 的 domain.pddl")

    # 边解析边去重，避免大数据集在内存中保留每个重复对象。
    predicate_index: dict[
        tuple[str, int], tuple[PredicateItem, set[str]]
    ] = {}
    action_index: dict[
        tuple[str, str], tuple[ActionItem, set[str]]
    ] = {}
    combined_type_parents: dict[str, str] = {}
    raw_predicate_count = 0
    raw_action_count = 0
    parsed_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(
        max_workers=min(MAX_WORKERS, len(domain_files))
    ) as executor:
        results = executor.map(parse_domain_file, domain_files, chunksize=50)
        for ok, path, predicates, actions, type_info, error in results:
            if not ok:
                skipped_count += 1
                print(f"[WARN] 跳过解析失败文件: {path}")
                print(f"       原因: {error}")
                continue

            parsed_count += 1
            type_order, type_parents = type_info
            for type_name in type_order:
                parent = type_parents[type_name]
                if (
                    type_name in combined_type_parents
                    and combined_type_parents[type_name] != parent
                ):
                    raise ValueError(
                        f"type parent conflict for {type_name}: "
                        f"{combined_type_parents[type_name]} / {parent}"
                    )
                combined_type_parents[type_name] = parent

            raw_predicate_count += len(predicates)
            for item in predicates:
                key = (item.name.lower(), item.arity)
                if key not in predicate_index:
                    predicate_index[key] = item, set(item.sources)
                    continue
                existing, seen_sources = predicate_index[key]
                if not (existing.leading_comments or existing.inline_comment) and (
                    item.leading_comments or item.inline_comment
                ):
                    existing.expr = item.expr
                    existing.leading_comments = item.leading_comments
                    existing.inline_comment = item.inline_comment
                for source in item.sources:
                    if source not in seen_sources:
                        seen_sources.add(source)
                        existing.sources.append(source)

            raw_action_count += len(actions)
            for item in actions:
                key = (item.name.lower(), item.signature)
                if key not in action_index:
                    action_index[key] = item, set(item.sources)
                    continue
                existing, seen_sources = action_index[key]
                if not existing.leading_comments and item.leading_comments:
                    existing.leading_comments = item.leading_comments
                    existing.block_text = item.block_text
                for source in item.sources:
                    if source not in seen_sources:
                        seen_sources.add(source)
                        existing.sources.append(source)

    if not predicate_index and not action_index:
        raise RuntimeError("所有 domain.pddl 都解析失败，无法生成 unified domain")

    print("[INFO] 开始合并")
    all_predicates = [item for item, _ in predicate_index.values()]
    all_actions = [item for item, _ in action_index.values()]
    predicate_arity_renames = resolve_predicate_arity_collisions(
        all_predicates, all_actions
    )
    hierarchy = TypeHierarchy(combined_type_parents)
    for type_name in combined_type_parents:
        hierarchy.ancestors(type_name)
    merged_predicates = merge_predicates(all_predicates)
    merged_actions = merge_actions(all_actions)

    action_contract_counts: dict[str, int] = {}
    for name, _ in action_index:
        action_contract_counts.setdefault(name, 0)
        action_contract_counts[name] += 1
    action_conflicts = [
        count for count in action_contract_counts.values() if count > 1
    ]

    write_outputs(
        list(combined_type_parents),
        combined_type_parents,
        merged_predicates,
        merged_actions,
        DATASET,
        output_path,
        source_json_path,
    )

    print()
    print(f"[INFO] 成功解析 domain 文件数: {parsed_count}")
    print(f"[INFO] 跳过解析失败文件数: {skipped_count}")
    print(f"[INFO] 原始 predicate 数: {raw_predicate_count}")
    print(f"[INFO] 合并后 predicate 数: {len(merged_predicates)}")
    print(f"[WARN] predicate arity 冲突名数: {len(predicate_arity_renames)}")
    for name, arities in sorted(predicate_arity_renames.items()):
        print(f"[WARN]   {name}: arity={arities}")
    print(f"[INFO] 原始 action 数: {raw_action_count}")
    print(f"[INFO] 合并后 action 数: {len(merged_actions)}")
    print(f"[INFO] 同名多合同 action 名数: {len(action_conflicts)}")
    print(f"[INFO] 数字后缀 action 变体数: {sum(action_conflicts)}")
    print(f"[INFO] unified domain 已保存到: {output_path}")
    print(f"[INFO] operator 来源 json 已保存到: {source_json_path}")


if __name__ == "__main__":
    main()
