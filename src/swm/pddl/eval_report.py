"""Evaluation reports rendered from saved evidence."""
import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from unicodedata import east_asian_width

from swm.pddl.attribution import LABELS, OPPOSITES, SPATIAL, load_world
from swm.pddl.strips import parse_domain


def compact_ids(values):
    values = {str(value) for value in values}
    if not all(value.isdigit() for value in values):
        return ", ".join(sorted(values))  # Keep nested task/episode identities intact.
    ranges = []
    for number in sorted(map(int, values)):
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


def _pad(text, width):
    size = sum(2 if east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - size)


def _solver_reason(directory):
    path = directory / "attribution_solver/result.json"
    if not path.is_file():
        return "Fast Downward 无法解析或处理 PDDL；详细日志缺失。"
    solver = json.loads(path.read_text())
    log = "\n".join(solver.get(key) or "" for key in ("stdout", "stderr"))
    action = re.findall(r"Parsing action '([^']+)'", log)
    where = f"动作 {action[-1]}" if action else "PDDL"
    match = re.search(r"(?:Undefined|Undeclared)\s+(predicate|object|variable)\s*\nGot:\s*([^\n]+)", log, re.I)
    if match:
        kind, name = match.groups()
        description = {"predicate": "未声明谓词", "object": "未声明对象", "variable": "未绑定变量"}[kind.lower()]
        return f"{where} 使用了{description} {name.strip()}。"
    match = re.search(r"Expected logical operator or predicate name\s*\nGot:\s*([^\n]+)", log)
    if match:
        return f"{where} 使用了未声明谓词 {match[1].strip()}。"
    match = re.search(r"Predicate '([^']+)' of arity (\d+) used\s+with (\d+) arguments", log, re.I)
    if match:
        return f"{where} 的谓词 {match[1]} 参数数量错误：应有 {match[2]} 个，实际 {match[3]} 个。"
    match = re.search(
        r"Expected a non-empty block starting with any of the following words:\s*([^\n]+)\n"
        r"(?:Syntax:[^\n]*\n)?Got:\s*\(([^()\s]+)(?:\s+([^()\s]+))?", log,
    )
    if match:
        expected, actual, following = match.groups()
        if ":action" in expected.split(", ") and following in {":parameters", ":precondition", ":effect"}:
            return f"动作声明格式错误：{actual} 应为 :action，且缺少动作名。"
        return f"PDDL 结构错误：此处应以 {expected} 开头，实际为 {actual}。"
    if "Missing ')'" in log:
        return "PDDL 括号不配对：缺少右括号 )。"
    if "Tokens remaining after parsing:" in log:
        return "PDDL 主体结束后仍有多余内容，请检查括号和文件末尾。"
    if "Non-ASCII character outside comment:" in log:
        return "PDDL 包含不支持的非 ASCII 字符，请检查对象、谓词或动作名称。"
    return f"Fast Downward 无法处理 PDDL（返回码 {solver.get('returncode', '未知')}）；详见 attribution_solver/result.json。"


def describe_error(result, directory):
    """Explain one observed error; never infer a missing action's identity."""
    label, checks = result["label"], result.get("checks", {})
    if label == "pddl_invalid":
        return _solver_reason(directory)
    if label == "operator_missing":
        candidate = parse_domain(directory / "domain.pddl")
        gt = parse_domain(Path(result["gt_directory"]) / "domain.pddl")
        candidate_only = [name for name in candidate if name not in gt]
        gt_only = [name for name in gt if name not in candidate]
        return (f"Candidate 独有 ({len(candidate_only)})：[{', '.join(candidate_only)}]\n"
                f"GT 独有 ({len(gt_only)})：[{', '.join(gt_only)}]")
    if label == "operator_semantics":
        plans = checks["plans"]
        return ("action 名称及次数一致，但顺序不同；"
                f"Candidate [{', '.join(plans['candidate_names'])}]；GT [{', '.join(plans['gt_names'])}]。")
    if label == "else":
        return ""

    world = load_world(directory)
    inverse = {target: source for source, target in checks.get("object_mapping", {}).get("mapping", {}).items()}

    def gt_name(name):
        # Keep the GT identity visible when no candidate counterpart is mapped.
        return inverse.get(name, f"{name}[GT]")

    def fact_text(fact, negative=False):
        predicate, *args = fact
        # Use the candidate's predicate spelling when it has an unambiguous alias.
        aliases = [p for p in world.interface.descriptions
                   if p.canonical_name == predicate and p.arity == len(args) and not p.static]
        if len(aliases) == 1:
            description = aliases[0]
            predicate = description.raw_name
            reordered = list(args)
            for canonical_index, raw_index in enumerate(description.argument_order):
                reordered[raw_index] = args[canonical_index]
            args = reordered
        text = "(" + " ".join([predicate, *args]) + ")"
        return f"(not {text})" if negative else text

    if label == "init_error":
        item = next(c for c in checks["init"]["checks"] if c["status"] == "fail")
        if item["kind"] == "unary_state":
            predicate = item["gt"][0]
            positive = item["positive"]
            negatives = {tuple(f) for f in item.get("candidate_negative_states", [])}
            for actual in item["candidate_states"]:
                is_negative = tuple(actual) in negatives
                if (positive and ((actual[0] == predicate and is_negative) or
                                  (actual[0] == OPPOSITES.get(predicate) and not is_negative))) or (
                        not positive and actual[0] == predicate and not is_negative):
                    expected = [predicate, actual[1]]
                    return f"{fact_text(actual, is_negative)} → {fact_text(expected, not positive)}"
            return "初始显式状态与 GT 相矛盾；详见 attribution.json。"
        mapping = checks["object_mapping"]["mapping"]
        area = next(a for a in item["candidate_areas"] if a in mapping and mapping[a] not in item["gt_areas"])
        actual = next(f for f in sorted(world.facts) if len(f) == 3 and f[0] in SPATIAL
                      and f[1:] == (item["candidate_object"], area))
        expected = [fact_text([*actual[:2], gt_name(a)]) for a in item["gt_areas"]]
        return f"{fact_text(actual)} → {' / '.join(expected)}"
    if label == "goal_error":
        item = checks["goal_error"]["matches"][0]
        actual = item["candidate_goal"]
        if item["kind"] == "reversed_arguments":
            expected = [item["gt_goal"][0], *(gt_name(a) for a in item["gt_goal"][1:])]
            return f"{fact_text(actual)} → {fact_text(expected)}"
        expected = [fact_text([*actual[:2], gt_name(a)]) for a in item["gt_areas"]]
        return f"{fact_text(actual)} → {' / '.join(expected)}"
    return "尚未生成有效归因结果。"


def render_dataset_report(dataset, records, *, pddl=True):
    """Each non-passing task occurs in exactly one category, or pending attribution."""
    total = len(records)
    solved = sum(row["solved"] for row in records)
    passed = [row["id"] for row in records if row["passed"]]
    failures = [row for row in records if not row["passed"]]
    lines = ["=" * 80, f"{dataset} ({total})", "",
             f"{'可解率' if pddl else '生成率'}：{100 * solved / total if total else 0:.1f}% ({solved}/{total})",
             f"通过率：{100 * len(passed) / total if total else 0:.1f}% ({len(passed)}/{total})", "",
             f"通过 ({len(passed)})：", textwrap.fill(f"[{compact_ids(passed)}]", width=88, subsequent_indent=" ", break_on_hyphens=False), ""]
    if not pddl:
        lines.append(f"失败任务：[{compact_ids(row['id'] for row in failures)}]")
        return "\n".join(lines), {}
    groups = defaultdict(list)
    for row in failures:
        label = (row.get("attribution") or {}).get("label")
        groups[label if label in LABELS else "待归因"].append(row)
    labels = [*LABELS, *(["待归因"] if groups["待归因"] else [])]
    lines += [_pad("错误类型", 20) + " 数量  具体任务", "-" * 80]
    for label in labels:
        prefix = f"{_pad(label, 20)} {len(groups[label]):>4}  "
        ids = compact_ids(row["id"] for row in groups[label]) or "-"
        lines.append(textwrap.fill(prefix + ids, width=88, subsequent_indent=" " * 27, break_on_hyphens=False))
    lines += ["-" * 80, f"{_pad('合计', 20)} {len(failures):>4}"]
    if failures:
        lines += ["", "具体错误"]
    for label in labels:
        if not groups[label]:
            continue
        lines += ["", f"[{label}]"]
        if label == "else":
            lines.append(textwrap.fill(compact_ids(row["id"] for row in groups[label]), width=100,
                                       break_on_hyphens=False))
            continue
        reasons = defaultdict(list)
        for row in groups[label]:
            try:
                reason = describe_error(row["attribution"], Path(row["directory"])) if label != "待归因" else "尚未生成有效归因结果。"
            except (OSError, ValueError, KeyError, TypeError, StopIteration, NotImplementedError):
                reason = "归因证据不完整或不可读取，详见 attribution.json。"
            reasons[reason].append(row["id"])
        for reason, ids in reasons.items():
            if "\n" in reason:
                lines.append(f"{compact_ids(ids)}：")
                lines.extend(f"  {detail}" for detail in reason.splitlines())
            else:
                lines.append(textwrap.fill(f"{compact_ids(ids)}：{reason}", width=100, subsequent_indent="  ",
                                           break_long_words=False, break_on_hyphens=False))
    return "\n".join(lines), {label: len(groups[label]) for label in labels}
