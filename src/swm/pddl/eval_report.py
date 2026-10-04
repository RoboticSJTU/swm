"""Evaluation reports rendered from saved evidence."""
import json
import os
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from unicodedata import east_asian_width

from swm.pddl.attribution import LABELS, OPPOSITES, SPATIAL, load_world
from swm.pddl.planner import summarize_solver_error
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


ABSTRACT_STATUS_LABELS = {
    'success': '任务完成',
    'invalid_output': '模型回答格式无效',
    'invalid_pddl': 'PDDL 不合法',
    'unsupported_pddl': 'PDDL 功能不受支持',
    'candidate_unsolvable': '生成的模型无解',
    'search_incomplete': '搜索未找到计划（未证明无解）',
    'solver_timeout': '求解超时',
    'solver_memory_limit': '求解内存不足',
    'solver_error': '求解器异常',
    'invalid_plan': '计划格式无效',
    'interface_mismatch': '动作接口不匹配',
    'candidate_plan_invalid': '计划未通过 PDDL 验证',
    'illegal_environment_action': '环境中出现非法动作',
    'real_goal_not_reached': '未达成真实目标',
    'oracle_disagreement': '参考验证与环境结果不一致',
    'infrastructure_error': '模型调用或运行异常',
    'rule_mismatch': '任务完成，但规则检查未通过',
}


def _abstract_error_details(result, task):
    """Describe recorded failures and one counterexample without inferring a cause."""
    lines = []
    error, solver = result.get('error'), result.get('solver', {})
    if error:
        if solver and not solver.get('solved'):
            lines.append(summarize_solver_error(error) if result['status'] in {
                'invalid_pddl', 'unsupported_pddl', 'solver_error'
            } else f"{ABSTRACT_STATUS_LABELS[result['status']]}；求解器返回码：{solver['returncode']}")
        else:
            lines.append(error)
    if result.get('interface_ok') is False:
        expected, actual = task['operation_signatures'], result['operation_signatures']
        missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
        if missing:
            lines.append('缺少动作：' + ', '.join(missing))
        if extra:
            lines.append('多出动作：' + ', '.join(extra))
        for name in sorted(expected.keys() & actual.keys()):
            if expected[name] != actual[name]:
                lines.append(f'动作 {name} 参数数量：期望 {expected[name]}，实际 {actual[name]}')
    environment = result.get('environment', {})
    if environment.get('failed_step') is not None:
        action = '(' + ' '.join(environment['action']) + ')'
        lines.append(f"真实环境第 {environment['failed_step']} 步非法：{action}")
    elif environment.get('goal_reached') is False:
        goal = json.loads(Path(task['state_path']).read_text())['goal']
        lines.append('真实目标：' + json.dumps(goal, ensure_ascii=False, sort_keys=True))
        lines.append(f"计划长度：{result['plan_length']} 步；实际执行未达成目标")
    if result.get('candidate_val_pass') is False:
        lines.append('候选 PDDL 验证未通过')
    if result.get('reference_val_pass') is False:
        lines.append('参考 PDDL 验证未通过')
    probes = result['probes']
    failures = [(i, probe) for i, probe in enumerate(probes['results']) if not probe['pass']]
    if failures:
        lines.append(f"规则检查：{probes['passed']}/{probes['total']} 通过")
    if failures and result['status'] not in {'invalid_pddl', 'unsupported_pddl', 'solver_error'}:
        index, failure = failures[0]
        reference = json.loads(Path(task['probes_path']).read_text())[index]
        kind = failure.get('mechanic') or failure['kind']
        lines.append(f"首个失败检查：#{index + 1}（{kind}）")
        actions = reference['actions']
        steps = [o['failed_step'] for o in [reference, failure['actual']] if o.get('failed_step') is not None]
        stop = min(min(steps, default=len(actions)), len(actions))
        start = max(0, stop - 3)
        if actions:
            snippet = ' → '.join('(' + ' '.join(a) + ')' for a in actions[start:stop])
            lines.append(f'轨迹片段：第 {start + 1}–{stop} 步（共 {len(actions)} 步）：{snippet}')
        else:
            lines.append('检查轨迹：空轨迹')
        for label, outcome in [('参考规则', reference), ('候选模型', failure['actual'])]:
            executable = {True: '可执行', False: '不可执行', None: '无法判断'}[outcome.get('executable')]
            reached = {True: '已达成', False: '未达成', None: '无法判断'}[outcome.get('goal_reached')]
            detail = f'{label}：{executable}；目标{reached}'
            if outcome.get('failed_step') is not None:
                detail += f"；失败在第 {outcome['failed_step']} 步"
            if outcome.get('bad_operator'):
                detail += '；动作名称或参数无法匹配'
            if outcome.get('error') and label == '候选模型':
                detail += '；' + outcome['error']
            lines.append(detail)
    return '\n'.join(lines) or ABSTRACT_STATUS_LABELS.get(result['status'], result['status'])


def render_abstract_report(summary, results, tasks, output_root):
    """Mirror the dataset summary: passing tasks, indexed error groups and evidence."""
    stats = summary['overall']
    solved = sum(r.get('solver', {}).get('solved', False) for r in results)
    lines = [f"# 抽象规划评测：{summary['model']}", '',
             f"任务总数：{stats['total']}  ",
             f"求解成功：{solved}/{stats['total']}（{solved / stats['total']:.1%}）  ",
             f"任务完成：{stats['task_success']}/{stats['total']}（{stats['task_success_rate']:.1%}）  ",
             f"任务完成且全部规则检查通过：{stats['behavior_success']}/{stats['total']}（{stats['behavior_success_rate']:.1%}）  ",
             f"规则检查通过：{stats['probe_passed']}/{stats['probe_total']}", '',
             '求解成功表示生成的 PDDL 找到了计划（含空计划）；任务完成还要求动作接口正确、PDDL 验证通过，并在真实规则下达到目标。',
             '以下通过任务和错误分类以“任务完成且全部规则检查通过”为口径，每个未通过任务只计入一种错误。', '',
             '## 测试集总览', '',
             '| 测试集 | 任务数 | 求解成功 | 任务完成 | 任务完成且全部规则通过 | 规则检查通过 |',
             '|---|---:|---:|---:|---:|---:|']
    split_names = {'a_id': '已知游戏·同分布', 'a_scale': '已知游戏·规模外推',
                   'b_base': '新游戏·基础规则', 'b_rule': '新游戏·规则变化'}
    by_split = defaultdict(list)
    for result in sorted(results, key=lambda r: r['id']):
        by_split[result['split']].append(result)
    for split, records in by_split.items():
        part = summary['by_split'][split]
        dataset = records[0]['id'].split('/')[0]
        n = part['total']
        count = sum(r.get('solver', {}).get('solved', False) for r in records)
        lines.append(f"| [{dataset} · {split_names.get(split, split)}](#{dataset}) | {n} | {count}/{n} | "
                     f"{part['task_success']}/{n} | {part['behavior_success']}/{n} | {part['probe_passed']}/{part['probe_total']} |")
    lines += ['', '任务索引保留 task 和 episode 编号；点击任务可查看单项结果以及动作模型、目标和计划。',
              '未执行的规则检查仍计入评分分母；具体错误只展示已经观察到的证据。']
    for split, records in by_split.items():
        dataset = records[0]['id'].split('/')[0]

        def task_link(result):
            label = '/'.join(result['id'].split('/')[1:])
            return f"[{label}]({result['id']}/result.md)"

        passed = [r for r in records if r['behavior_success']]
        lines += ['', f'<a id="{dataset}"></a>', '',
                  f'## {dataset} · {split_names.get(split, split)}（{len(records)} 个任务）', '',
                  f'通过（{len(passed)}）：' + ('、'.join(task_link(r) for r in passed) or '无'), '',
                  '### 错误类型与任务索引', '',
                  '| 错误类型 | 数量 | 具体任务 |', '|---|---:|---|']
        groups = defaultdict(list)
        for result in records:
            if not result['behavior_success']:
                category = 'rule_mismatch' if result['task_success'] else result['status']
                groups[category].append(result)
        for category, failures in groups.items():
            label = ABSTRACT_STATUS_LABELS.get(category, category)
            lines.append(f"| {label}（`{category}`） | {len(failures)} | " + '、'.join(task_link(r) for r in failures) + ' |')
        lines.append(f"| 未通过合计 | {len(records) - len(passed)} | |")
        if not groups:
            continue
        lines += ['', '### 具体错误']
        for category, failures in groups.items():
            lines += ['', f'#### {ABSTRACT_STATUS_LABELS.get(category, category)}（{len(failures)}）']
            reasons = defaultdict(list)
            for result in failures:
                reasons[_abstract_error_details(result, tasks[result['id']])].append(result)
            for reason, affected in reasons.items():
                games = ', '.join(sorted({r['domain'] for r in affected}))
                first = affected[0]
                task = tasks[first['id']]
                links = [f'[{label}]({first["id"]}/{name})' for name, label in [
                    ('domain.pddl', '动作模型'), ('problem.pddl', '初始状态与目标'), ('plan.txt', '求解计划'),
                    ('.cache.json', '原始回答与检查结果')]
                    if (output_root / first['id'] / name).is_file()]
                links.append(f"[完整规则检查轨迹]({os.path.relpath(task['probes_path'], output_root)})")
                lines += ['', '任务：' + '、'.join(task_link(r) for r in affected) + f' · 游戏：`{games}`', '',
                          '相关文件（首个任务）：' + ' · '.join(links), '', '~~~text', reason, '~~~']
                if category == 'solver_error' and '\n' in first.get('error', ''):
                    lines += ['', '<details>', '<summary>首个任务的原始错误日志</summary>', '',
                              '~~~text', first['error'], '~~~', '', '</details>']
                cache_path = output_root / first['id'] / '.cache.json'
                if category == 'invalid_output' and cache_path.is_file():
                    raw = json.loads(cache_path.read_text()).get('raw_output')
                    if isinstance(raw, str):
                        lines += ['', '<details>', '<summary>首个任务的模型回答开头（最多 20 行）</summary>', '',
                                  '~~~text', '\n'.join(raw.splitlines()[:20]), '~~~', '',
                                  f"[完整原始回答与评测数据]({first['id']}/.cache.json)", '', '</details>']
    return '\n'.join(lines) + '\n'
