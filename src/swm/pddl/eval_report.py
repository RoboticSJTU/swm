"""Plain-text evaluation summaries rendered from saved attribution evidence."""
import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from unicodedata import east_asian_width

from swm.pddl.attribution import LABELS, OPPOSITES, SPATIAL, load_world
from swm.pddl.planner import summarize_solver_error


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
    return summarize_solver_error(log)


def describe_error(result, directory):
    """Explain one observed error; never infer a missing action's identity."""
    label, checks = result["label"], result.get("checks", {})
    if label == "pddl_invalid":
        return _solver_reason(directory)
    if label == "operator_missing":
        counts = checks["operator_counts"]
        return f"operator 数量不足：Candidate {counts['candidate']} < GT {counts['gt']}。"
    if label == "operator_semantics":
        plans = checks["plans"]
        return ("action 名称及次数一致，但顺序不同；"
                f"Candidate [{', '.join(plans['candidate_names'])}]；GT [{', '.join(plans['gt_names'])}]。")
    if label == "else":
        if checks.get("mapping_error"):
            return "对象映射失败，待人工复查。"
        if checks.get("input_error"):
            return f"输入不可用：{checks['input_error']}；待人工复查。"
        solver = checks.get("fast_downward", {})
        if solver.get("returncode") in {20, 21, 22, 23, 24}:
            return f"Fast Downward 超时或内存受限（返回码 {solver['returncode']}），待人工复查。"
        if solver.get("error") or (solver.get("returncode") or 0) < 0:
            return f"Fast Downward 未正常完成（{solver.get('error') or solver['returncode']}），待人工复查。"
        if result.get("reason") == "init_not_confirmed":
            return "init 证据不足，未命中现有归因规则，待人工复查。"
        return "未命中现有归因规则，待人工复查。"

    world = load_world(directory)
    inverse = {target: source for source, target in checks.get("object_mapping", {}).get("mapping", {}).items()}

    def gt_name(name):
        return inverse.get(name, f"{name}（GT）")

    def gt_area(name):
        source = inverse.get(name)
        if source == name:
            return name
        if source:
            return f"{name}（GT；对应 Candidate {source}）"
        return f"{name}（GT）"

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
                    return f"{fact_text(actual, is_negative)} → 应为 {fact_text(expected, not positive)}。"
            return "初始显式状态与 GT 相矛盾；详见 attribution.json。"
        mapping = checks["object_mapping"]["mapping"]
        area = next(a for a in item["candidate_areas"] if a in mapping and mapping[a] not in item["gt_areas"])
        actual = next(f for f in sorted(world.facts) if len(f) == 3 and f[0] in SPATIAL
                      and f[1:] == (item["candidate_object"], area))
        return f"{fact_text(actual)} → 所在区域应为 {', '.join(gt_area(a) for a in item['gt_areas'])}。"
    if label == "goal_error":
        item = checks["goal_error"]["matches"][0]
        actual = fact_text(item["candidate_goal"])
        if item["kind"] == "reversed_arguments":
            expected = [item["gt_goal"][0], *(gt_name(a) for a in item["gt_goal"][1:])]
            return f"{actual} → 应为 {fact_text(expected)}。"
        return f"{actual} → 目标区域应为 {', '.join(gt_area(a) for a in item['gt_areas'])}。"
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
        reasons = defaultdict(list)
        for row in groups[label]:
            try:
                reason = describe_error(row["attribution"], Path(row["directory"])) if label != "待归因" else "尚未生成有效归因结果。"
            except (OSError, ValueError, KeyError, TypeError, StopIteration, NotImplementedError):
                reason = "归因证据不完整或不可读取，详见 attribution.json。"
            reasons[reason].append(row["id"])
        for reason, ids in reasons.items():
            lines.append(textwrap.fill(f"{compact_ids(ids)}：{reason}", width=100, subsequent_indent="  ",
                                       break_long_words=False, break_on_hyphens=False))
    return "\n".join(lines), {label: len(groups[label]) for label in labels}
