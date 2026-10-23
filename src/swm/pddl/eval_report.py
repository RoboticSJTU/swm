"""Evaluation reports rendered from saved evidence."""
import json
import re
import textwrap
from collections import defaultdict
from os.path import relpath
from pathlib import Path
from unicodedata import east_asian_width
from urllib.parse import quote

from swm.pddl.attribution import (
    CLASSIFICATION_CRITERIA, LABELS, OPPOSITES, SPATIAL, TAXONOMY, TAXONOMY_VERSION,
    load_world, taxonomy_for,
)
from swm.pddl.strips import parse_domain
from swm.reporting import markdown_details, markdown_table


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
        duplicates = [item for item in checks.get("declarations", {}).get("issues", [])
                      if item["kind"] == "duplicate_declaration"]
        if duplicates:
            item = duplicates[0]
            return f"{item['file']} 的 {item['declaration']} 重复声明：{', '.join(item['symbols'])}。"
        return _solver_reason(directory)
    if label == "operator_missing":
        missing = checks.get("operator_coverage", {}).get("matches", [])
        if missing:
            return "缺少参考计划所需的操作能力：" + "; ".join(
                f"{item['family']}（GT 动作 {item['gt_action']}，状态效果 {', '.join(item['gt_effects'])}）"
                for item in missing) + "。"
        candidate = parse_domain(directory / "domain.pddl", allow_effect_overlap=True)
        gt = parse_domain(Path(result["gt_directory"]) / "domain.pddl", allow_effect_overlap=True)
        candidate_only = [name for name in candidate if name not in gt]
        gt_only = [name for name in gt if name not in candidate]
        return ("算子数量少于 GT；这是数量启发式，尚未确认缺少哪种必需操作能力。\n"
                f"Candidate 独有 ({len(candidate_only)})：[{', '.join(candidate_only)}]\n"
                f"GT 独有 ({len(gt_only)})：[{', '.join(gt_only)}]")
    if label == "operator_semantics":
        if order := checks.get("instruction_order", {}).get("matches"):
            item = order[0]
            return (f"指令明确要求“{item['first_clause']}”先于“{item['second_clause']}”；"
                    f"候选先在第 {', '.join(map(str, item['second_steps']))} 步执行后项，"
                    f"直到第 {min(item['first_steps'])} 步才执行前项。"
                    f"后项执行时前项对象仍在已确认来源 ({' '.join(item['observations'][0])})，"
                    "此前没有对应转移或可解释该转移的复合动作。")
        if physical := checks.get("physical_transition_semantics", {}).get("matches"):
            item = physical[0]
            return (f"第 {item['step']} 步 {item['action']} 必须先满足 ({' '.join(item['gt_precondition'])})；"
                    "全部 GT 建立方式都要求该条件，但候选没有交代所需的拿取、移位或移除障碍。"
                    "依据现有评测约定，这些实际物理转移不能作为隐含步骤补出。")
        if effects := checks.get("effect_reachability", {}).get("matches"):
            item = effects[0]
            fact = item["ineffective_effects"][0]
            operation = "添加" if fact["positive"] else "删除"
            return (f"动作 {item['action']} 的处理效果{operation} ({' '.join(fact['fact'])}) 永远不能改变状态；"
                    "完整实例化及可达性检查后，剩余有效效果只有位置、持握等结构变化。")
        causal = checks.get("causal_reachability", {}).get("matches", [])
        if causal:
            item = causal[0]
            if item["kind"] == "causal_goal_mutex":
                return ("目标 " + " 与 ".join("(" + " ".join(f) + ")" for f in item["goals"])
                        + " 各自可达，却可证明无法同时成立；有效 GT 计划的最终状态同时满足二者。")
            text = "(" + " ".join(item["candidate_goal"]) + ")"
            if dead := item.get("uninstantiable_producers"):
                return (f"目标 {text} 的因果链不可达；动作 {dead[0]['action']} 没有任何满足静态条件的参数绑定。")
            if blocked := item.get("blocked_producers"):
                first = blocked[0]
                guards = " 与 ".join("(" + " ".join(f) + ")" for f in first["conditions"])
                return (f"目标 {text} 的所有建立路径都被阻断；例如 {first['action']} 所需的 {guards} "
                        + ("无法同时成立。" if len(first["conditions"]) == 2 else "不可达。"))
            return f"GT 要求的目标 {text} 在候选模型中不可达；完整实例化后没有建立该事实的动作。"
        matches = checks.get("strict_operator_semantics", {}).get("matches", []) or checks.get("operator_semantics", {}).get("matches", [])
        if matches:
            item = matches[0]
            if item["kind"] == "reference_state_mutex":
                proof = item["reference_mutex_proof"]
                return (f"第 {item['step']} 步 {item['action']} 声明的动作后状态同时包含 "
                        f"({' '.join(proof['anchor'])}) 与 ({' '.join(proof['observation'])})；"
                        "二者各自在 GT 中可建立，但完整可达性检查证明不能共存。")
            if item["kind"] == "semantic_noop":
                return f"动作 {item['action']}（{item['family']}）的效果不会改变任何可达状态，无法记录或保证该操作完成。"
            if item["kind"] == "semantic_effect_missing":
                return f"动作 {item['action']} 声明为 {item['family']}，效果却只有位置、持握等结构状态变化，缺少相应处理效果。"
            if item["kind"] == "exclusive_resource_goals":
                return (f"目标要求同一对象 {item['object']} 同时位于多个容器，但初始状态及所有转移动作"
                        "都保持它至多位于一个容器；该资源表示无法满足多目标。")
            if item["kind"] == "lost_required_state":
                lost = item["lost_at"]
                return (f"第 {item['step']} 步 {item['action']} 需要 ({' '.join(item['gt_precondition'])})，"
                        f"但第 {lost['step']} 步 {lost['action']} 已破坏该条件，此后未恢复。")
            if item["kind"] == "unrestored_initial_state":
                return (f"第 {item['step']} 步 {item['action']} 需要 ({' '.join(item['gt_precondition'])})，"
                        f"GT 初始状态为 ({' '.join(item['gt_initial_opposite'])})，此前未操作该对象或相关部件；"
                        "GT 可达性检查也排除了所声明的动作后状态。")
            if item["kind"] == "missing_semantic_precondition":
                return (f"第 {item['step']} 步 {item['action']} 绕过已建模的处理条件 "
                        f"({' '.join(item['gt_precondition'])})；候选具备建立它的动作，但执行到此时尚未满足。")
            if item["kind"] == "violated_negative_gt_guard":
                return (f"第 {item['step']} 步 {item['action']} 违反 GT 的否定前置条件 "
                        f"({' '.join(item['gt_precondition'])})；候选状态中被禁止的事实仍成立。")
            if item["kind"] == "inconsistent_effects":
                return (f"第 {item['step']} 步 {item['action']} 的效果导致互斥状态同时成立："
                        + " / ".join(str(f) for f in item["conflicts"][0]) + "。")
            return (f"第 {item['step']} 步 {item['action']} 违反对应 GT 操作的前置条件 "
                    f"({' '.join(item['gt_precondition'])})；GT 可达性证明其声明的动作后状态中存在互斥事实。")
        unreachable = checks.get("goal_error", {}).get("unachievable", [])
        if unreachable:
            item = next((i for i in unreachable if i.get("reference_supported")), unreachable[0])
            verb = "添加" if item["positive"] else "删除"
            return f"目标 ({' '.join(item['candidate_fact'])}) 需要{verb}，但没有匹配的算子效果；GT 支持该要求。"
        plans = checks["plans"]
        return ("action 名称及次数一致，但顺序不同；"
                f"Candidate [{', '.join(plans['candidate_names'])}]；GT [{', '.join(plans['gt_names'])}]。")
    if label == "else":
        if error := checks.get("input_error"):
            return f"归因输入未能解析：{error}"
        evidence = checks.get("unresolved", {})
        if result.get("reason") == "candidate_trace_unavailable":
            return ("求解器已确认可解，但被评测的 plan.txt 缺失或不能合法回放；"
                    "需要恢复有效候选轨迹才能完成语义归因，不能将此直接当作 PDDL 模型错误。")
        if unreachable := evidence.get("unreachable_goals"):
            return (f"已证明候选目标 ({' '.join(unreachable[0])}) 不可达；"
                    "现有证据尚不能区分初始信息、目标要求或算子中的具体责任。")
        if mutexes := evidence.get("goal_mutexes"):
            return ("已证明候选目标 " + " 与 ".join("(" + " ".join(f) + ")" for f in mutexes[0])
                    + " 无法同时满足；尚未确认这两项要求是否都由任务支持。")
        if guards := evidence.get("uncertified_guards"):
            item = guards[0]
            return (f"第 {item['step']} 步 {item['action']} 存在 GT 条件 ({' '.join(item['gt_precondition'])}) 的冲突线索；"
                    "尚未获得排除合法复合动作的证明，归入 else。")
        if uncertain := evidence.get("uncertain_goal_encodings"):
            item = uncertain[0]
            return (f"GT 要求 ({' '.join(item['gt_goal'])})，但候选在相关对象上使用了其他目标、处理编码或转移实现；"
                    "尚未确认其语义对应关系，不能确定为任务要求遗漏。")
        if interfaces := evidence.get("reference_goal_interfaces"):
            return (f"GT 要求 ({' '.join(interfaces[0])})；候选采用不同表示，"
                    "尚缺可靠的状态或事件对应关系，不能仅凭谓词不同判错。")
        if constraints := evidence.get("reference_only_initial_constraints"):
            return (f"GT 记录了候选未表示的初始约束 ({' '.join(constraints[0])})；"
                    "仍需确认候选如何处理该约束，不能仅凭缺少同名谓词判错。")
        if result.get("reason") == "init_not_confirmed":
            return "对象映射或初始状态证据不完整，现有局部检查未找到确定错误。"
        if result.get("reason") == "fast_downward_inconclusive":
            return "求解器超时或运行失败，现有局部检查未找到确定错误。"
        return "现有证据未能证明具体错误；需要进一步检查任务顺序、数量或对象建模。"

    if cardinality := checks.get("cardinality"):
        item = cardinality[0]
        count = len(item["required_objects"])
        if item["kind"] == "semantic_goal_cardinality":
            return (f"GT 要求对 {count} 个独立 {item['kind_name']} 完成 {item['predicate']}，"
                    f"候选计划至多处理了 {len(item['candidate_touched_objects'])} 个不同实例；重复处理同一实例不能补足数量。")
        if item["kind"] == "declared_instance_deficit":
            return (f"GT 要求 {count} 个独立 {item['kind_name']}；在已确认的一对一映射下，"
                    f"候选至多有 {len(item['candidate_objects'])} 个对象可对应这些实例，数量不足。")
        return (f"GT 要求 {count} 个独立对象到达 {item['candidate_area']}，"
                f"候选目标仅约束 {len(item['candidate_goal_objects'])} 个，计划最终至多到达 {len(item['candidate_final_objects'])} 个。")
    if raw := checks.get("causal_reachability", {}).get("raw_init_matches"):
        item = raw[0]
        actual = " / ".join("(" + " ".join(f) + ")" for f in item["present_aliases"])
        pattern = item.get("required_pattern", [])
        if any(arg.startswith("?") for arg in pattern[1:]):
            return (f"初始状态使用 {actual}，但因果链至少需要一个满足 ({' '.join(pattern)}) 的原始事实；"
                    "所有参数绑定均无初始事实或建立方式，不能用 in/on 的区域比较约定替代原始 PDDL 条件。")
        return (f"初始状态使用 {actual}，但因果链需要 ({' '.join(item['raw_fact'])})；"
                "语义别名不等于 PDDL 中的同一事实，候选也无法建立所需原始事实。")
    if access := checks.get("initial_access", {}).get("matches"):
        item = access[0]
        proof = item["reference_mutex_proof"]
        return (f"初始状态声明 ({' '.join(item['candidate_fact'])})，同时保留已确认的遮挡关系 "
                f"({' '.join(proof['observation'])})；GT 可达性证明这两项状态不能共存。")
    issues = checks.get("declarations", {}).get("issues", [])
    section = ":init" if label == "init_error" else ":goal"
    if conflicting := next((item for item in issues if item.get("section") == section), None):
        if conflicting["kind"] == "reflexive_location":
            return f"{section} 声明对象位于自身：({' '.join(conflicting['facts'][0])})；空间关系 in/on 不允许自包含或自支撑。"
        return f"{section} 同时要求互斥事实：" + " / ".join(
            "(" + " ".join(fact) + ")" for fact in conflicting["conflicts"][0]) + "。"

    world = load_world(directory)
    inverse = {target: source for source, target in checks.get("object_mapping", {}).get("mapping", {}).items()}

    def gt_name(name):
        # Keep the GT identity visible when no candidate counterpart is mapped.
        return inverse.get(name, f"{name}[GT]")

    def fact_text(fact, negative=False):
        predicate, *args = fact
        # Use the candidate's predicate spelling when it has an unambiguous alias.
        aliases = [p for p in world.interface.descriptions
            if p.canonical_name == predicate and p.arity == len(args)]
        if predicate.startswith(("kind:", "static:")):
            canonical = predicate.split(":", 1)[1]
            aliases = [p for p in world.interface.descriptions if p.static and p.arity == len(args)
                       and p.raw_name.replace("-", "_") == canonical]
        if predicate == "contains_water" and len(args) == 1 and not aliases:
            water = [o.name for o in world.objects if "water" in o.kinds]
            if len(water) == 1:
                predicate, args = "in", [water[0], *args]
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
        if item["kind"] == "required_fact_missing":
            return (f"初始状态缺少 {fact_text(item['candidate_fact'])}；"
                    "GT 中该事实成立，候选前置条件或目标使用它，且没有算子能够建立它。")
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
        if item["kind"] in {"omitted_goal", "wrong_goal_state", "goal_without_producer", "missing_goal_witness"}:
            expected = fact_text(item["candidate_fact"], not item.get("positive", True))
            if item["kind"] == "missing_goal_witness":
                return (f"目标仅保留初始已成立的 {expected}，遗漏 GT 的操作完成标记 "
                        f"({' '.join(item['gt_goal'])})；候选计划没有执行相应放置。")
            if item["kind"] == "omitted_goal":
                return f"目标遗漏 {expected}，且候选计划的最终状态未满足该 GT 要求。"
            if item["kind"] == "wrong_goal_state":
                return f"目标状态与 GT 要求 {expected} 相矛盾。"
            return f"额外目标 {expected} 在初始状态不成立，Candidate 和 GT 均无匹配的建立效果。"
        actual = item["candidate_goal"]
        if item["kind"] == "reversed_arguments":
            expected = [item["gt_goal"][0], *(gt_name(a) for a in item["gt_goal"][1:])]
            return f"{fact_text(actual)} → {fact_text(expected)}"
        expected = [fact_text([*actual[:2], gt_name(a)]) for a in item["gt_areas"]]
        return f"{fact_text(actual)} → {' / '.join(expected)}"
    return "尚未生成有效归因结果。"


def render_dataset_report(dataset, records, *, pddl=True, details=True):
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
    groups = defaultdict(lambda: defaultdict(list))
    legacy_counts = dict.fromkeys(LABELS, 0)
    for row in failures:
        result = row.get("attribution")
        category, subcategory = taxonomy_for(result, row["directory"])
        groups[category or "待归因"][subcategory].append(row)
        label = (result or {}).get("label")
        if label in legacy_counts:
            legacy_counts[label] += 1
    categories = [*TAXONOMY, *(["待归因"] if "待归因" in groups else [])]

    def category_name(category):
        return TAXONOMY[category]["name"] if category in TAXONOMY else category

    def ordered_subcategories(category):
        return [sub for sub in TAXONOMY.get(category, {}).get("subcategories", {}) if sub in groups[category]] + (
            [None] if None in groups[category] else [])

    lines += [_pad("错误大类", 20) + " 数量  具体任务", "-" * 80]
    for category in categories:
        members = [row for group in groups[category].values() for row in group]
        prefix = f"{_pad(category_name(category), 20)} {len(members):>4}  "
        ids = compact_ids(row["id"] for row in members) or "-"
        lines.append(textwrap.fill(prefix + ids, width=88, subsequent_indent=" " * 27, break_on_hyphens=False))
    lines += ["-" * 80, f"{_pad('合计', 20)} {len(failures):>4}"]
    if failures:
        lines += ["", "具体错误（按小类组织；每个任务仅计入一个小类）"]
    summary = {"taxonomy_version": TAXONOMY_VERSION, "total": total, "solved": solved,
               "passed": len(passed), "failed": len(failures), "categories": {}, "legacy_labels": legacy_counts}
    for category in categories:
        count = sum(len(group) for group in groups[category].values())
        summary["categories"][category] = {"name": category_name(category), "count": count, "subcategories": {}}
        if not count:
            continue
        lines += ["", f"[{category_name(category)} / {category}]"]
        for subcategory in ordered_subcategories(category):
            members = groups[category][subcategory]
            name = TAXONOMY.get(category, {}).get("subcategories", {}).get(subcategory, "小类未明确")
            criterion = CLASSIFICATION_CRITERIA.get(subcategory)
            summary["categories"][category]["subcategories"][subcategory or "unresolved"] = {
                "name": name, "count": len(members), "criterion": criterion}
            if not details:
                continue
            if category != "待归因":
                lines += ["", f"  {name} ({len(members)})"]
                if criterion:
                    lines.append(f"  判定特征：{criterion}")
            reasons = defaultdict(list)
            for row in members:
                try:
                    reason = describe_error(row["attribution"], Path(row["directory"])) if category != "待归因" else "尚未生成有效归因结果。"
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
    return "\n".join(lines), summary


def _fact_code(fact):
    if fact[0] == "not":
        inner = fact[1] if len(fact) == 2 and isinstance(fact[1], list) else fact[1:]
        return "`(not (" + " ".join(inner) + "))`"
    return "`(" + " ".join(fact) + ")`"


def describe_error_brief(result, directory):
    """Keep the concrete defect or uncertainty; full evidence stays in JSON."""
    if not result:
        return "尚未完成归因。"
    checks, reason = result.get("checks", {}), result.get("reason")
    if result["label"] == "else":
        evidence = checks.get("unresolved", {})
        if goals := evidence.get("unreachable_goals"):
            return f"目标 {_fact_code(goals[0])} 已证实不可达，责任尚未定位。"
        if pairs := evidence.get("goal_mutexes"):
            return "目标 " + " 与 ".join(map(_fact_code, pairs[0])) + " 互斥，尚未确认均为任务要求。"
        if guards := evidence.get("uncertified_guards"):
            item = guards[0]
            return f"第 {item['step']} 步存在 {_fact_code(item['gt_precondition'])} 条件冲突线索，尚不能排除复合动作。"
        if encodings := evidence.get("uncertain_goal_encodings"):
            return f"要求 {_fact_code(encodings[0]['gt_goal'])}，候选实现的语义对应未确认。"
        if interfaces := evidence.get("reference_goal_interfaces"):
            return f"要求 {_fact_code(interfaces[0])}，候选表示的语义对应未确认。"
        if constraints := evidence.get("reference_only_initial_constraints"):
            return f"初始约束 {_fact_code(constraints[0])} 在候选中的对应未确认。"
        objects = evidence.get("unmapped_candidate_objects", []) + evidence.get("unmapped_reference_goal_objects", [])
        if objects:
            return "对象映射未确认：" + "、".join(f"`{name}`" for name in dict.fromkeys(objects)) + "。"
        if checks.get("input_error") or reason in {"candidate_trace_unavailable", "fast_downward_inconclusive", "input_unavailable"}:
            return describe_error(result, directory)
        return "现有检查尚未定位具体原因。"
    if result["label"] == "pddl_invalid":
        text = describe_error(result, directory)
        if match := re.search(r"(未声明谓词|未声明对象|未绑定变量) ([^。\n]+)", text):
            return f"{match[1]} `{match[2]}`。"
        return text
    if reason in {"gt_init_mismatch", "wrong_destination"}:
        return re.sub(r"\([^()]*\)", lambda match: f"`{match[0]}`", describe_error(result, directory))
    if reason == "initial_access_conflict":
        item = checks["initial_access"]["matches"][0]
        return f"初始 {_fact_code(item['candidate_fact'])} 与 {_fact_code(item['reference_mutex_proof']['observation'])} 冲突。"
    if reason == "wrong_goal_state":
        item = checks["goal_error"]["matches"][0]
        expected = item["candidate_fact"] if item.get("positive", True) else ["not", *item["candidate_fact"]]
        return f"目标状态写反，应为 {_fact_code(expected)}。"
    if result["label"] == "operator_missing" and (matches := checks.get("operator_coverage", {}).get("matches")):
        return "缺少 " + "、".join(f"`{item['family']}`" for item in matches) + " 操作能力。"
    if reason == "explicit_task_order_violation":
        item = checks["instruction_order"]["matches"][0]
        return (f"顺序颠倒：要求先“{item['first_clause']}”，再“{item['second_clause']}”；"
                f"候选第 {min(item['second_steps'])} 步先做后项，第 {min(item['first_steps'])} 步才做前项。")
    if reason == "unaccounted_physical_precondition":
        item = checks["physical_transition_semantics"]["matches"][0]
        return f"第 {item['step']} 步缺少建立 {_fact_code(item['gt_precondition'])} 的移位/移障操作。"
    if reason == "unreachable_causal_goal":
        item = checks["causal_reachability"]["matches"][0]
        goal = _fact_code(item["candidate_goal"])
        if blocked := item.get("blocked_producers"):
            conditions = blocked[0]["conditions"]
            if len(conditions) == 1:
                return f"目标 {goal} 不可达：所需 {_fact_code(conditions[0])} 无法建立。"
            return f"目标 {goal} 不可达：所需 " + " 与 ".join(map(_fact_code, conditions)) + " 无法同时成立。"
        if not item.get("uninstantiable_producers"):
            return f"目标 {goal} 没有建立动作。"
    matches = checks.get("strict_operator_semantics", {}).get("matches") or checks.get("operator_semantics", {}).get("matches")
    if matches:
        item = matches[0]
        if reason in {"violated_gt_guard", "violated_negative_gt_guard", "missing_semantic_precondition"}:
            return f"第 {item['step']} 步在未满足 {_fact_code(item['gt_precondition'])} 时执行。"
        if reason == "unrestored_initial_state":
            return (f"第 {item['step']} 步需 {_fact_code(item['gt_precondition'])}，"
                    f"初始为 {_fact_code(item['gt_initial_opposite'])}，此前未改变该状态。")
        if reason == "semantic_effect_missing":
            return f"`{item['family']}` 动作仅改变位置/持握，缺少对应处理效果。"
        if reason == "reference_state_mutex":
            proof = item["reference_mutex_proof"]
            return f"第 {item['step']} 步后，{_fact_code(proof['anchor'])} 与 {_fact_code(proof['observation'])} 无法共存。"
        if reason == "exclusive_resource_goals":
            return f"目标要求同一 `{item['object']}` 同时在多个容器，模型却只允许它位于一处。"
    return describe_error(result, directory)


def render_markdown_report(records, report_root, *, pddl=True):
    """One table of rates per total tasks, then grouped, collapsible evidence."""
    summaries = {dataset: render_dataset_report(dataset, rows, pddl=pddl, details=False)[1]
                 for dataset, rows in records.items()}
    totals = [{"total": len(rows), "solved": sum(row["solved"] for row in rows),
               "passed": sum(row["passed"] for row in rows)} for rows in records.values()]
    aggregate = {key: sum(row[key] for row in totals) for key in ("total", "solved", "passed")}
    columns = [*totals, aggregate]
    headers = ["指标", *(f"{dataset}（{len(rows)}）" for dataset, rows in records.items()), f"合计（{aggregate['total']}）"]

    def rates(values):
        return [f"{100 * value / row['total'] if row['total'] else 0:.1f}".removesuffix(".0")
                for value, row in zip(values, columns)]

    table = []
    for key, label in (("solved", "可解率" if pddl else "生成率"), ("passed", "通过率")):
        cells = rates(row[key] for row in columns)
        table.append([label, *(f"**{cell}**" if key == "passed" else cell for cell in cells)])
    if pddl:
        def counts(category, subcategory=None):
            values = []
            for summary in summaries.values():
                group = summary["categories"].get(category, {})
                values.append(group.get("subcategories", {}).get(subcategory, {}).get("count", 0)
                              if subcategory else group.get("count", 0))
            return [*values, sum(values)]

        def split_rates(totals, parts):
            first = rates(total - part for total, part in zip(totals, parts))
            return [f"{a}/{b}" for a, b in zip(first, rates(parts))]

        init_conflicts = counts("init_error", "init_inconsistent")
        missing = counts("operator_error", "operator_missing")
        table += [
            ["init错误率 (写错/冲突)", *split_rates(counts("init_error"), init_conflicts)],
            ["goal错误率", *rates(counts("goal_error"))],
            ["oper错误率 (建模/缺失)", *split_rates(counts("operator_error"), missing)],
            ["else", *rates(known + pending for known, pending in zip(counts("else"), counts("待归因")))],
        ]
    lines = markdown_table(headers, table)
    lines.append("")
    for dataset, rows in records.items():
        failures = [row for row in rows if not row["passed"]]
        if not failures:
            continue
        groups = defaultdict(lambda: defaultdict(list))
        for row in failures:
            result = row.get("attribution")
            category, subcategory = taxonomy_for(result, row["directory"]) if pddl else (None, None)
            try:
                reason = describe_error_brief(result, Path(row["directory"])) if pddl else "未通过。"
            except (OSError, ValueError, KeyError, TypeError, StopIteration, NotImplementedError):
                reason = "归因证据不完整，需检查原始记录。"
            groups[category or "else", subcategory][reason].append(row)
        order = {(category, subcategory): index for index, (category, subcategory) in enumerate(
            (category, subcategory) for category, group in TAXONOMY.items() for subcategory in group["subcategories"])}
        body = [f"失败task: {compact_ids(row['id'] for row in failures)}", ""]
        for category, subcategory in sorted(groups, key=lambda key: order.get(key, len(order))):
            reasons = groups[category, subcategory]
            name = TAXONOMY.get(category, {}).get("subcategories", {}).get(subcategory, TAXONOMY.get(category, {}).get("name", "else"))
            if name == "goal写错":
                name = "goal错误"
            body += [f"**{name}（{sum(map(len, reasons.values()))}）**", ""]
            for reason, members in reasons.items():
                links = []
                for row in members:
                    directory = Path(row["directory"])
                    evidence = directory / "attribution.json"
                    if not evidence.is_file():
                        evidence = next((directory / name for name in ("judge.json", "error.log") if (directory / name).is_file()), directory)
                    label = str(row["id"])
                    target = quote(relpath(evidence, report_root), safe="/._-")
                    links.append(f"[{label}]({target})" if evidence.exists() else label)
                body.append("- " + "、".join(links) + "：" + " ".join(reason.splitlines()))
            body.append("")
        lines += markdown_details(f"{dataset}：{len(failures)} 个任务", body)
    return "\n".join(lines).rstrip() + "\n", summaries
