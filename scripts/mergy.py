# Merge latest episode PDDL domains into a deduplicated unified operator library with source tracking.

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Union


# =========================
# 配置
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "eval_results" / "gpt-5.6-sol"
DATASET_CHOICES = ("human", "human_aug")
MAX_WORKERS = 16

VAR_RE = re.compile(r"\?[A-Za-z0-9_\-]+")
TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


# =========================
# 数据结构
# =========================

@dataclass
class PredicateItem:
    name: str
    arity: int
    expr: str
    leading_comments: List[str] = field(default_factory=list)
    inline_comment: str = ""
    sources: List[str] = field(default_factory=list)

    def has_comment(self) -> bool:
        return bool(self.leading_comments or self.inline_comment)


@dataclass
class ActionItem:
    name: str
    param_arity: int
    signature: str
    block_text: str
    leading_comments: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    def has_comment(self) -> bool:
        return bool(self.leading_comments)


PddlNode = Union[str, List["PddlNode"]]


# =========================
# PDDL 基础解析
# =========================

def remove_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        pos = line.find(";")
        if pos != -1:
            line = line[:pos]
        lines.append(line)
    return "\n".join(lines)


def find_matching_paren(text: str, start: int) -> int:
    if text[start] != "(":
        raise ValueError(f"位置 {start} 不是左括号")

    depth = 0
    in_comment = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_comment:
            if ch == "\n":
                in_comment = False
            continue

        if ch == ";":
            in_comment = True
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i

    raise ValueError("括号不匹配")


def find_token(text: str, token: str, start: int = 0) -> int:
    token = token.lower()
    in_comment = False
    i = start

    while i <= len(text) - len(token):
        ch = text[i]

        if in_comment:
            if ch == "\n":
                in_comment = False
            i += 1
            continue

        if ch == ";":
            in_comment = True
            i += 1
            continue

        if text[i:i + len(token)].lower() == token:
            return i

        i += 1

    return -1


def parse_sexp(text: str) -> PddlNode:
    tokens = TOKEN_RE.findall(text)
    pos = 0

    def parse_one() -> PddlNode:
        nonlocal pos

        if pos >= len(tokens):
            raise ValueError("PDDL 表达式提前结束")

        tok = tokens[pos]

        if tok == "(":
            pos += 1
            node = []
            while pos < len(tokens) and tokens[pos] != ")":
                node.append(parse_one())

            if pos >= len(tokens):
                raise ValueError("PDDL 表达式缺少右括号")

            pos += 1
            return node

        if tok == ")":
            raise ValueError("PDDL 表达式存在多余右括号")

        pos += 1
        return tok

    node = parse_one()

    if pos != len(tokens):
        raise ValueError("PDDL 表达式解析后仍有剩余 token")

    return node


def canonical_signature(parameters_text: str, precondition_text: str, effect_text: str) -> Tuple[int, str]:
    """
    构造 action 的语义签名：
    1. 变量名统一改成 ?v0, ?v1, ...
    2. and/or 内部子表达式排序
    3. 参数表顺序不同但语义相同的 action 可以被合并
    """
    param_vars = [v.lower() for v in VAR_RE.findall(parameters_text)]
    pre_tree = parse_sexp(precondition_text)
    eff_tree = parse_sexp(effect_text)

    usage: Dict[str, List[str]] = {}
    for v in param_vars:
        usage[v] = []

    def collect_usage(node: PddlNode, prefix: str) -> None:
        if isinstance(node, str) or not node:
            return

        head = node[0]
        if isinstance(head, list):
            return

        head = str(head).lower()

        if head in {"and", "or"}:
            for child in node[1:]:
                collect_usage(child, prefix)
            return

        if head == "not":
            if len(node) == 2:
                collect_usage(node[1], prefix + ":not")
            return

        for i, arg in enumerate(node[1:]):
            if isinstance(arg, str) and arg.startswith("?"):
                arg = arg.lower()
                if arg not in usage:
                    usage[arg] = []
                usage[arg].append(f"{prefix}:{head}:{i}")

    collect_usage(pre_tree, "pre")
    collect_usage(eff_tree, "eff")

    ordered_vars = sorted(param_vars, key=lambda v: (tuple(sorted(usage[v])), v))

    var_map = {}
    for i, v in enumerate(ordered_vars):
        var_map[v] = f"?v{i}"

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

        head = node[0]
        if isinstance(head, list):
            raise ValueError("非法 PDDL 表达式：head 是 list")

        head = str(head).lower()

        if head in {"and", "or"}:
            children = []

            for child in node[1:]:
                if (
                    isinstance(child, list)
                    and child
                    and isinstance(child[0], str)
                    and child[0].lower() == head
                ):
                    for grand_child in child[1:]:
                        children.append(canon(grand_child))
                else:
                    children.append(canon(child))

            children.sort()
            return "(" + " ".join([head] + children) + ")"

        children = [canon(child) for child in node[1:]]
        return "(" + " ".join([head] + children) + ")"

    signature = f"arity={len(param_vars)} | pre={canon(pre_tree)} | eff={canon(eff_tree)}"
    return len(param_vars), signature


# =========================
# 查找 domain 文件
# =========================

def find_domain_files(root_dir: Path) -> List[Path]:
    """
    查找每个 task_x/episode_y 下最大 round 的 domain.pddl。
    """
    domain_files = []

    for episode_dir in sorted(root_dir.glob("task_*/episode_*")):
        if not episode_dir.is_dir():
            continue

        best_round_key = (-1, -1)
        best_domain = None

        for round_dir in sorted(episode_dir.iterdir(), key=lambda path: path.name.lower()):
            if not round_dir.is_dir():
                continue

            # Some historical cloud folders are named `roun3` (missing `d`).
            m = re.fullmatch(
                r"roun(?:d)?[_\-]?(\d+)",
                round_dir.name,
                flags=re.IGNORECASE,
            )
            if not m:
                continue

            domain_path = round_dir / "domain.pddl"
            if not domain_path.exists():
                continue

            round_id = int(m.group(1))
            is_canonical_spelling = int(round_dir.name.lower().startswith("round"))
            round_key = (round_id, is_canonical_spelling)
            if round_key > best_round_key:
                best_round_key = round_key
                best_domain = domain_path

        if best_domain is not None:
            domain_files.append(best_domain)

    return domain_files


# =========================
# 解析一个 domain 文件
# =========================

def parse_domain_file(domain_path: Path):
    try:
        text = domain_path.read_text(encoding="utf-8")
        source = str(domain_path)

        predicates = []
        actions = []

        # ---------- predicates ----------
        pred_start = find_token(text, "(:predicates")
        if pred_start == -1:
            raise ValueError("没有找到 :predicates 块")

        pred_end = find_matching_paren(text, pred_start)
        pred_block = text[pred_start:pred_end + 1]

        m = re.match(r"\(\s*:predicates\b", pred_block, flags=re.IGNORECASE)
        if not m:
            raise ValueError(":predicates 块格式错误")

        body = pred_block[m.end():-1]
        i = 0

        while i < len(body):
            leading_comments = []

            while i < len(body):
                while i < len(body) and body[i].isspace():
                    i += 1

                if i < len(body) and body[i] == ";":
                    line_end = body.find("\n", i)
                    if line_end == -1:
                        line_end = len(body)
                    leading_comments.append(body[i:line_end].strip())
                    i = line_end
                    continue

                break

            if i >= len(body):
                break

            if body[i] != "(":
                i += 1
                continue

            j = find_matching_paren(body, i)
            expr = body[i:j + 1].strip()

            k = j + 1
            while k < len(body) and body[k] in " \t":
                k += 1

            inline_comment = ""
            if k < len(body) and body[k] == ";":
                line_end = body.find("\n", k)
                if line_end == -1:
                    line_end = len(body)
                inline_comment = body[k:line_end].strip()
                k = line_end

            clean_expr = remove_comments(expr)
            tokens = re.findall(r"[^\s()]+", clean_expr)

            if not tokens:
                raise ValueError(f"无法解析 predicate: {expr}")

            name = tokens[0]
            arity = 0
            for tok in tokens[1:]:
                if tok.startswith("?"):
                    arity += 1

            predicates.append(
                PredicateItem(
                    name=name,
                    arity=arity,
                    expr=expr,
                    leading_comments=leading_comments,
                    inline_comment=inline_comment,
                    sources=[source],
                )
            )

            i = k

        # ---------- actions ----------
        pos = 0

        while True:
            action_start = find_token(text, "(:action", pos)
            if action_start == -1:
                break

            action_end = find_matching_paren(text, action_start)
            action_block = text[action_start:action_end + 1].strip()

            before_action = text[pos:action_start].splitlines()
            idx = len(before_action) - 1

            while idx >= 0 and before_action[idx].strip() == "":
                idx -= 1

            leading_comments = []
            while idx >= 0 and before_action[idx].lstrip().startswith(";"):
                leading_comments.append(before_action[idx].strip())
                idx -= 1

            leading_comments.reverse()

            clean_action = remove_comments(action_block)

            m = re.search(r"\(\s*:action\s+([^\s()]+)", clean_action, flags=re.IGNORECASE)
            if not m:
                raise ValueError("action 块开头格式错误")

            action_name = m.group(1).strip()

            def read_field(keyword: str) -> str:
                field_match = re.search(rf"{re.escape(keyword)}\b", clean_action, flags=re.IGNORECASE)
                if not field_match:
                    raise ValueError(f"{action_name} 没有找到 {keyword}")

                p = field_match.end()
                while p < len(clean_action) and clean_action[p].isspace():
                    p += 1

                if p >= len(clean_action) or clean_action[p] != "(":
                    raise ValueError(f"{action_name} 的 {keyword} 后面不是括号表达式")

                q = find_matching_paren(clean_action, p)
                return clean_action[p:q + 1].strip()

            parameters_text = read_field(":parameters")
            precondition_text = read_field(":precondition")
            effect_text = read_field(":effect")

            param_arity, signature = canonical_signature(
                parameters_text,
                precondition_text,
                effect_text,
            )

            actions.append(
                ActionItem(
                    name=action_name,
                    param_arity=param_arity,
                    signature=signature,
                    block_text=action_block,
                    leading_comments=leading_comments,
                    sources=[source],
                )
            )

            pos = action_end + 1

        return True, source, predicates, actions, ""

    except Exception as e:
        return False, str(domain_path), [], [], str(e)


# =========================
# 合并
# =========================

def merge_predicates(predicates: List[PredicateItem]) -> List[PredicateItem]:
    merged: Dict[Tuple[str, int], PredicateItem] = {}
    source_sets: Dict[Tuple[str, int], set[str]] = {}

    for item in predicates:
        key = (item.name.lower(), item.arity)

        if key not in merged:
            merged[key] = item
            source_sets[key] = set(item.sources)
        else:
            old = merged[key]

            if not old.has_comment() and item.has_comment():
                old.expr = item.expr
                old.leading_comments = item.leading_comments
                old.inline_comment = item.inline_comment

            seen_sources = source_sets[key]
            for src in item.sources:
                if src not in seen_sources:
                    seen_sources.add(src)
                    old.sources.append(src)

    return sorted(
        merged.values(),
        key=lambda x: (x.name.lower(), x.arity, x.expr.lower()),
    )


def rewrite_pddl_head_tokens(text: str, replacements: Dict[str, str]) -> str:
    """Rewrite S-expression head tokens while preserving comments and layout."""
    if not replacements:
        return text

    normalized = {name.lower(): replacement for name, replacement in replacements.items()}
    output: List[str] = []
    index = 0
    in_comment = False

    while index < len(text):
        char = text[index]

        if in_comment:
            output.append(char)
            if char == "\n":
                in_comment = False
            index += 1
            continue

        if char == ";":
            in_comment = True
            output.append(char)
            index += 1
            continue

        if char != "(":
            output.append(char)
            index += 1
            continue

        output.append(char)
        index += 1
        while index < len(text) and text[index].isspace():
            output.append(text[index])
            index += 1

        token_start = index
        while index < len(text) and not text[index].isspace() and text[index] not in "()":
            index += 1
        token = text[token_start:index]
        output.append(normalized.get(token.lower(), token))

    return "".join(output)


def resolve_predicate_arity_collisions(
    predicates: List[PredicateItem],
    actions: List[ActionItem],
) -> Dict[str, Dict[int, str]]:
    """Reject predicate overloading instead of silently renaming contracts."""
    arities_by_name: Dict[str, set[int]] = {}
    arities_by_source_name: Dict[Tuple[str, str], set[int]] = {}

    for item in predicates:
        name = item.name.lower()
        arities_by_name.setdefault(name, set()).add(item.arity)
        for source in item.sources:
            arities_by_source_name.setdefault((source, name), set()).add(item.arity)

    invalid_sources = {
        key: sorted(arities)
        for key, arities in arities_by_source_name.items()
        if len(arities) > 1
    }
    if invalid_sources:
        details = ", ".join(
            f"{source}:{name}={arities}"
            for (source, name), arities in sorted(invalid_sources.items())
        )
        raise ValueError(f"单个 source 内 predicate arity 冲突，无法安全合并: {details}")

    collisions = {
        name: sorted(arities)
        for name, arities in arities_by_name.items()
        if len(arities) > 1
    }
    if not collisions:
        return {}
    details = ", ".join(
        f"{name}={arities}" for name, arities in sorted(collisions.items())
    )
    raise ValueError(
        "predicate arity conflicts require semantic cleanup before merge: "
        + details
    )


def merge_actions(actions: List[ActionItem]) -> List[Tuple[str, ActionItem]]:
    """Merge equivalent schemas and number same-name contract variants."""
    grouped: Dict[str, Dict[str, ActionItem]] = {}
    display_names: Dict[str, str] = {}
    source_sets: Dict[Tuple[str, str], set[str]] = {}

    for item in actions:
        if re.search(r"(?:_|-)\d+$", item.name):
            raise ValueError(
                f"numeric action suffix is forbidden: {item.name} "
                f"from {item.sources[:3]}"
            )
        group_key = item.name.lower()
        display_names.setdefault(group_key, item.name)
        variants = grouped.setdefault(group_key, {})
        old = variants.get(item.signature)
        if old is None:
            variants[item.signature] = item
            source_sets[(group_key, item.signature)] = set(item.sources)
            continue
        if not old.has_comment() and item.has_comment():
            old.leading_comments = item.leading_comments
            old.block_text = item.block_text
        seen_sources = source_sets[(group_key, item.signature)]
        for src in item.sources:
            if src not in seen_sources:
                seen_sources.add(src)
                old.sources.append(src)

    final_actions: List[Tuple[str, ActionItem]] = []
    for group_key, variants in grouped.items():
        ordered_variants = [
            item for _, item in sorted(variants.items(), key=lambda pair: pair[0])
        ]
        if len(ordered_variants) == 1:
            final_actions.append((display_names[group_key], ordered_variants[0]))
            continue
        for index, item in enumerate(ordered_variants, start=1):
            final_actions.append((f"{display_names[group_key]}_{index}", item))

    def natural_name_key(item: Tuple[str, ActionItem]):
        return tuple(
            (1, int(part)) if part.isdigit() else (0, part)
            for part in re.split(r"(\d+)", item[0].lower())
            if part
        )

    final_actions.sort(key=natural_name_key)
    return final_actions


# =========================
# 写出结果
# =========================

def write_outputs(
    predicates: List[PredicateItem],
    actions: List[Tuple[str, ActionItem]],
    domain_name: str,
    output_path: Path,
    source_json_path: Path,
) -> None:
    lines = [
        f"(define (domain {domain_name})",
        "  (:requirements :strips :typing :negative-preconditions :disjunctive-preconditions :equality)",
        "  (:predicates",
    ]

    for item in predicates:
        for c in item.leading_comments:
            lines.append(f"    {c}")

        expr_lines = item.expr.strip().splitlines()
        for i, line in enumerate(expr_lines):
            line = line.rstrip()
            if i == len(expr_lines) - 1 and item.inline_comment:
                lines.append(f"    {line} {item.inline_comment}")
            else:
                lines.append(f"    {line}")

    lines.append("  )")
    lines.append("")

    source_json = []

    for final_name, item in actions:
        for c in item.leading_comments:
            lines.append(f"  {c}")

        renamed_block = re.sub(
            r"(\(\s*:action\s+)([^\s()]+)",
            lambda m: m.group(1) + final_name,
            item.block_text,
            count=1,
            flags=re.IGNORECASE,
        )

        for line in renamed_block.strip().splitlines():
            lines.append(f"  {line.rstrip()}")

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

    if lines[-1] == "":
        lines.pop()

    lines.append(")")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    source_json_path.write_text(
        json.dumps(source_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================
# 主流程
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the maximum-round domains for one evaluation dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_CHOICES,
        help="Dataset to merge. This must be selected explicitly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = DATASET_ROOT / args.dataset
    domain_name = args.dataset
    output_path = root_dir / "unified_domain.pddl"
    source_json_path = root_dir / "unified_operator_sources.json"

    domain_files = find_domain_files(root_dir)

    if not domain_files:
        raise FileNotFoundError(f"在 {root_dir} 下没有找到 domain.pddl")

    print(f"[INFO] ROOT_DIR: {root_dir}")
    print(f"[INFO] DOMAIN_NAME: {domain_name}")
    print(f"[INFO] 找到 {len(domain_files)} 个最大 round 的 domain.pddl")

    all_predicates = []
    all_actions = []
    parsed_count = 0
    skipped_count = 0

    workers = min(MAX_WORKERS, len(domain_files))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(parse_domain_file, domain_files, chunksize=50)

        for ok, path, predicates, actions, error in results:
            if ok:
                all_predicates.extend(predicates)
                all_actions.extend(actions)
                parsed_count += 1
            else:
                skipped_count += 1
                print(f"[WARN] 跳过解析失败文件: {path}")
                print(f"       原因: {error}")

    if not all_predicates and not all_actions:
        raise RuntimeError("所有 domain.pddl 都解析失败，无法生成 unified domain")

    print("[INFO] 开始合并")

    predicate_arity_renames = resolve_predicate_arity_collisions(
        all_predicates,
        all_actions,
    )
    merged_predicates = merge_predicates(all_predicates)
    merged_actions = merge_actions(all_actions)
    action_contract_signatures: Dict[str, set[str]] = {}
    for item in all_actions:
        action_contract_signatures.setdefault(item.name.lower(), set()).add(
            item.signature
        )
    action_conflicts = {
        name: len(signatures)
        for name, signatures in action_contract_signatures.items()
        if len(signatures) > 1
    }

    write_outputs(
        merged_predicates,
        merged_actions,
        domain_name,
        output_path,
        source_json_path,
    )

    print()
    print(f"[INFO] 成功解析 domain 文件数: {parsed_count}")
    print(f"[INFO] 跳过解析失败文件数: {skipped_count}")
    print(f"[INFO] 原始 predicate 数: {len(all_predicates)}")
    print(f"[INFO] 合并后 predicate 数: {len(merged_predicates)}")
    print(f"[INFO] predicate arity 冲突名数: {len(predicate_arity_renames)}")
    for name, variants in sorted(predicate_arity_renames.items()):
        rendered = ", ".join(
            f"arity {arity} -> {final_name}"
            for arity, final_name in sorted(variants.items())
        )
        print(f"[INFO]   {name}: {rendered}")
    print(f"[INFO] 原始 action 数: {len(all_actions)}")
    print(f"[INFO] 合并后 action 数: {len(merged_actions)}")
    print(f"[INFO] 同名多合同 action 名数: {len(action_conflicts)}")
    print(f"[INFO] 数字后缀 action 变体数: {sum(action_conflicts.values())}")
    print(f"[INFO] unified domain 已保存到: {output_path}")
    print(f"[INFO] operator 来源 json 已保存到: {source_json_path}")


if __name__ == "__main__":
    main()
