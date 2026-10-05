"""Merge the latest episode domains into one deduplicated operator library."""

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swm.pddl.strips import format_action_conditions, validate_untyped_pddl


# 只需要修改这里。
DATASET = "human_aug"
DATASET_ROOT = PROJECT_ROOT / "eval_results" / "gpt-5.6-sol"
MAX_WORKERS = 100
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
) -> tuple[int, str]:
    """Normalize variable names and unordered Boolean terms for action deduplication."""
    parameters = parse_sexp(parameters_text)
    if not isinstance(parameters, list):
        raise ValueError("action parameters must be a list")
    if any(not isinstance(value, str) or not value.startswith("?") for value in parameters):
        raise ValueError("action parameters must be untyped variables")
    param_vars = [variable.lower() for variable in parameters]
    if len(set(param_vars)) != len(param_vars):
        raise ValueError("duplicate action parameter")
    precondition = parse_sexp(precondition_text)
    effect = parse_sexp(effect_text)
    usage = {variable: [] for variable in param_vars}

    def collect_usage(
        node: PddlNode, prefix: str, bound: frozenset[str] = frozenset()
    ) -> None:
        if not isinstance(node, list) or not node or not isinstance(node[0], str):
            return
        head = node[0].lower()
        if head in {"and", "or"}:
            for child in node[1:]:
                collect_usage(child, prefix, bound)
        elif head == "not":
            if len(node) == 2:
                collect_usage(node[1], prefix + ":not", bound)
        elif head in {"forall", "exists"}:
            collect_usage(
                node[2], prefix + ":" + head,
                bound | frozenset(variable.lower() for variable in node[1]),
            )
        elif head in {"when", "imply"}:
            collect_usage(node[1], prefix + ":" + head + ":condition", bound)
            collect_usage(node[2], prefix + ":" + head + ":body", bound)
        else:
            for index, argument in enumerate(node[1:]):
                if (
                    isinstance(argument, str)
                    and argument.startswith("?")
                    and argument.lower() not in bound
                ):
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

    def canon(
        node: PddlNode, bound: dict[str, str] | None = None, depth: int = 0
    ) -> str:
        bound = {} if bound is None else bound
        if isinstance(node, str):
            token = node.lower()
            if token in bound:
                return bound[token]
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
        if head in {"forall", "exists"}:
            local = dict(bound)
            variables = [variable.lower() for variable in node[1]]
            names = [f"?q{depth + index}" for index in range(len(variables))]
            local.update(zip(variables, names))
            return (
                f"({head} ({' '.join(names)}) "
                + canon(node[2], local, depth + len(variables))
                + ")"
            )
        if head in {"and", "or"}:
            children = []
            for child in node[1:]:
                if (
                    isinstance(child, list)
                    and child
                    and isinstance(child[0], str)
                    and child[0].lower() == head
                ):
                    children.extend(
                        canon(grandchild, bound, depth) for grandchild in child[1:]
                    )
                else:
                    children.append(canon(child, bound, depth))
            children.sort()
        else:
            children = [canon(child, bound, depth) for child in node[1:]]
        return "(" + " ".join([head, *children]) + ")"

    signature = (
        f"arity={len(param_vars)} | "
        f"pre={canon(precondition)} | eff={canon(effect)}"
    )
    return len(param_vars), signature


def find_task_domain_files(task_dir: str) -> list[Path]:
    """Scan one task, keeping episode order and the latest available round."""
    round_pattern = re.compile(r"roun(?:d)?[_\-]?(\d+)", re.IGNORECASE)
    with os.scandir(task_dir) as entries:
        episode_dirs = sorted(
            entry.path for entry in entries
            if entry.name.startswith("episode_") and entry.is_dir()
        )

    domain_files = []
    for episode_dir in episode_dirs:
        candidates = []
        with os.scandir(episode_dir) as entries:
            for entry in entries:
                match = round_pattern.fullmatch(entry.name)
                if not match or not entry.is_dir():
                    continue
                key = (
                    -int(match.group(1)),
                    -int(entry.name.lower().startswith("round")),
                    entry.name.lower(),
                )
                candidates.append((key, entry.path))

        # Only probe older rounds when the latest round has no domain.
        for _, round_dir in sorted(candidates, key=lambda item: item[0]):
            domain_path = Path(round_dir) / "domain.pddl"
            if domain_path.exists():
                domain_files.append(domain_path)
                break
    return domain_files


def find_domain_files(root_dir: Path) -> list[Path]:
    """Find the highest numbered round containing a domain for every episode."""
    try:
        with os.scandir(root_dir) as entries:
            task_dirs = sorted(
                entry.path for entry in entries
                if entry.name.startswith("task_") and entry.is_dir()
            )
    except (FileNotFoundError, NotADirectoryError):
        return []
    if not task_dirs:
        return []

    # Overlap shared-filesystem metadata latency; ordered map keeps output stable.
    with ProcessPoolExecutor(max_workers=min(32, MAX_WORKERS, len(task_dirs))) as executor:
        return [
            path
            for paths in executor.map(find_task_domain_files, task_dirs)
            for path in paths
        ]


def parse_domain_file(domain_path: Path):
    """Extract domain content, reporting input issues without discarding the file."""
    source = str(domain_path)
    predicates = []
    actions = []
    warnings = []
    try:
        text = domain_path.read_text(encoding="utf-8")
        try:
            validate_untyped_pddl(text)
        except ValueError as error:
            warnings.append(f"检查未通过，继续提取：{error}")

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
            argument_names = [
                argument for argument in declaration[1:]
                if isinstance(argument, str) and argument.startswith("?")
            ]
            predicates.append(
                PredicateItem(
                    name=name,
                    arity=len(argument_names),
                    expr="(" + " ".join(declaration) + ")",
                    leading_comments=leading_comments,
                    inline_comment=inline_comment,
                    sources=[source],
                    argument_names=tuple(argument_names),
                )
            )

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
            try:
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

                param_arity, signature = canonical_signature(
                    fields[":parameters"],
                    fields[":precondition"],
                    fields[":effect"],
                )
                block_text = format_action_conditions(action_block)
            except (ValueError, TypeError, IndexError) as error:
                warnings.append(f"{action_name}：{error}；保留原始动作")
                param_arity = sum(
                    token.startswith("?")
                    for token in TOKEN_RE.findall(fields.get(":parameters", ""))
                )
                signature = "raw=" + " ".join(TOKEN_RE.findall(clean_action)).lower()
                block_text = action_block
            actions.append(
                ActionItem(
                    name=action_name,
                    param_arity=param_arity,
                    signature=signature,
                    block_text=block_text,
                    leading_comments=leading_comments,
                    sources=[source],
                )
            )
            position = action_end + 1

        return True, source, predicates, actions, "；".join(warnings)
    except Exception as error:
        warnings.append(
            f"无法继续提取：{error}；已保留 {len(predicates)} 个谓词、{len(actions)} 个动作"
        )
        return False, source, predicates, actions, "；".join(warnings)


def merge_predicates(predicates: list[PredicateItem]) -> list[PredicateItem]:
    merged: dict[tuple[str, int], tuple[PredicateItem, set[str]]] = {}

    for item in predicates:
        if item.name == "=":
            if item.arity != 2:
                raise ValueError("built-in equality must have two arguments")
            continue
        key = (item.name.lower(), item.arity)
        if key not in merged:
            merged[key] = item, set(item.sources)
            continue

        old, seen_sources = merged[key]
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
    """Warn about predicate overloading without rewriting source semantics."""
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

    examples = ", ".join(
        f"{name}={arities}" for name, arities in sorted(conflicts.items())[:3]
    )
    print(
        f"[WARN] 谓词元数冲突 {len(conflicts)} 个（示例：{examples}）；"
        "保留各元数声明，标准 PDDL 求解前仍需清理"
    )
    return conflicts


def merge_actions(actions: list[ActionItem]) -> list[tuple[str, ActionItem]]:
    """Merge equivalent schemas and number same-name contract variants."""
    grouped: dict[str, list[ActionItem]] = {}
    display_names: dict[str, str] = {}
    numeric_suffix_names = set()

    for item in actions:
        if re.search(r"(?:_|-)\d+$", item.name):
            numeric_suffix_names.add(item.name.lower())
        group_key = item.name.lower()
        display_names.setdefault(group_key, item.name)
        grouped.setdefault(group_key, []).append(item)

    if numeric_suffix_names:
        print(f"[WARN] 输入中已有数字后缀的 action 名称：{len(numeric_suffix_names)} 个")

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
    predicates: list[PredicateItem],
    actions: list[tuple[str, ActionItem]],
    domain_name: str,
    output_path: Path,
    source_json_path: Path,
) -> None:
    uses_adl = any(
        re.search(r"\(\s*(?:forall|exists|when|or|imply)\b", item.block_text, re.IGNORECASE)
        for _, item in actions
    )
    requirements = ":adl" if uses_adl else ":strips :negative-preconditions :equality"
    lines = [
        f"(define (domain {domain_name})",
        f"  (:requirements {requirements})",
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
    started = perf_counter()
    root_dir = DATASET_ROOT / DATASET
    output_path = root_dir / "unified_domain.pddl"
    source_json_path = root_dir / "unified_operator_sources.json"
    print(f"[INFO] 数据集：{DATASET}", flush=True)
    domain_files = find_domain_files(root_dir)
    scan_seconds = perf_counter() - started
    if not domain_files:
        raise FileNotFoundError(f"在 {root_dir} 下没有找到 domain.pddl")

    print(
        f"[INFO] 找到 {len(domain_files)} 个 domain",
        flush=True,
    )

    # 边解析边去重，避免大数据集在内存中保留每个重复对象。
    predicate_index: dict[
        tuple[str, int], tuple[PredicateItem, set[str]]
    ] = {}
    action_index: dict[
        tuple[str, str], tuple[ActionItem, set[str]]
    ] = {}
    raw_predicate_count = 0
    raw_action_count = 0
    processed_count = 0
    warning_count = 0
    warning_examples = []

    with ProcessPoolExecutor(
        max_workers=min(MAX_WORKERS, len(domain_files))
    ) as executor:
        results = executor.map(parse_domain_file, domain_files, chunksize=50)
        for _, path, predicates, actions, warning in results:
            if warning:
                warning_count += 1
                if len(warning_examples) < 3:
                    warning_examples.append((path, warning))

            processed_count += 1
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

    parse_seconds = perf_counter() - started - scan_seconds
    if warning_count:
        print(f"[WARN] 输入检查或提取警告：{warning_count} 个文件（最多显示 3 个示例）")
        for path, warning in warning_examples:
            print(f"[WARN] {Path(path).relative_to(root_dir)}：{' '.join(warning.split())}")

    if not predicate_index and not action_index:
        print("[WARN] 未提取到谓词或动作，将输出空算子库")

    all_predicates = [item for item, _ in predicate_index.values()]
    all_actions = [item for item, _ in action_index.values()]
    resolve_predicate_arity_collisions(all_predicates, all_actions)
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
        merged_predicates,
        merged_actions,
        DATASET,
        output_path,
        source_json_path,
    )

    total_seconds = perf_counter() - started
    print(
        f"[INFO] 完成：处理 domain {processed_count}/{len(domain_files)}；"
        f"predicate {raw_predicate_count} → {len(merged_predicates)}；"
        f"action {raw_action_count} → {len(merged_actions)}"
    )
    print(
        f"[INFO] 同名 action 多版本：{len(action_conflicts)} 个名称、"
        f"{sum(action_conflicts)} 个变体；耗时 {total_seconds:.1f}s"
        f"（扫描 {scan_seconds:.1f}s，解析去重 {parse_seconds:.1f}s，"
        f"合并写出 {total_seconds - scan_seconds - parse_seconds:.1f}s）"
    )
    print(f"[INFO] 输出：    {output_path}\n[INFO] 来源：    {source_json_path}")


if __name__ == "__main__":
    main()
