"""Compact reports from the abstract evaluator's recorded task decisions."""
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from swm.reporting import markdown_details, markdown_table

SPLIT_LABELS = {'seen': 'Seen', 'unseen_composition': 'Unseen composition'}
EVALUATION_LABELS = {'seen_rule':'已见规则的新场景', 'scale':'规模外推',
                     'composition_base':'规则组合对照', 'unseen_rule_combination':'未见规则组合',
                     'unseen_game':'未见游戏', 'unseen_game_rule':'未见游戏的规则变体',
                     'unseen_game_rule_change':'未见游戏的规则变体'}
ERROR_TYPES = {'init_mismatch':'init写错','init_inconsistent':'init冲突','goal_error':'goal错误',
               'operator_modeling':'oper建模错误','operator_missing':'oper整体缺失','output_error':'输出错误'}
GROUP_NAMES = {**ERROR_TYPES,'plan_error':'计划错误','unattributed':'待归因','pending':'评测未完成'}


def diagnostic_group(result, unresolved):
    status = result['status']
    if status in unresolved:
        return 'pending'
    if result['task_success']:
        return None
    if result.get('output_format') == 'nl':
        return 'output_error' if status == 'invalid_output' else 'plan_error'
    proof = result.get('diagnosis') or {}
    category, subcategory = proof.get('category'), proof.get('subcategory')
    if category == 'goal_error':
        return 'goal_error'
    if ((category == 'init_error' and subcategory in {'init_mismatch', 'init_inconsistent'})
            or (category == 'operator_error' and subcategory in {'operator_modeling', 'operator_missing'})):
        return subcategory
    return 'output_error' if status in {'invalid_output', 'invalid_pddl'} else 'unattributed'


def _brief_error(result):
    proof = result.get('diagnosis') or {}
    if proof.get('evidence'):
        return proof['evidence']
    status, error = result['status'], str(result.get('error') or '')
    if status == 'invalid_pddl':
        match = re.search(r'(?:Undefined|Undeclared)\s+(predicate|object|variable)\s*\nGot:\s*([^\n]+)', error, re.I)
        if match:
            kind = {'predicate': '谓词', 'object': '对象', 'variable': '变量'}[match[1].lower()]
            return f'未声明{kind} `{match[2].strip()}`。'
        if match := re.search(r"Predicate '([^']+)' of arity (\d+) used\s+with (\d+) arguments", error, re.I):
            return f'谓词 `{match[1]}` 需要 {match[2]} 个参数，实际 {match[3]} 个。'
        if match := re.search(r'Expected logical operator or predicate name\s*\nGot:\s*([^\n]+)', error):
            return f'未声明谓词 `{match[1].strip()}`。'
        diagnostic = next((line.strip() for line in error.splitlines()
                           if re.search(r'Error:|Exception:|Expected |Missing |Unsupported |Invalid ', line)), None)
        return 'PDDL 解析失败' + (f'：{diagnostic}' if diagnostic else '，详见求解日志。')
    if status == 'interface_mismatch':
        missing = result.get('missing_operations', [])
        return ('动作接口不匹配，缺少 ' + '、'.join(f'`{name}`' for name in missing) + '。'
                if missing else '候选包含任务接口之外的动作。')
    if status == 'illegal_environment_action':
        environment = result.get('environment', {})
        action = ' '.join(environment.get('action', []))
        return f"第 {environment.get('failed_step', '?')} 步 `{action}` 在真实环境中不可执行。"
    messages = {'candidate_unsolvable': '候选 PDDL 无解，尚不能定位到初始状态、目标或动作。',
                'candidate_plan_invalid': '计划未通过候选 PDDL 的合法性验证。',
                'real_goal_not_reached': '计划可以执行，但未满足真实目标。',
                'search_incomplete': '搜索提前结束，尚未确认是否可解。',
                'solver_timeout': '求解达到时间限制，任务尚未评测完成。',
                'solver_memory_limit': '求解达到内存限制，任务尚未评测完成。'}
    if status in messages:
        return messages[status]
    if status == 'invalid_output':
        if result.get('output_format') == 'nl':
            return '输出结构无法解析：' + error
        for pattern,message in [("Missing ')'", 'PDDL 括号不配对：缺少右括号。'),
                                ('Unparsed tokens remain', 'PDDL 主体后存在多余内容。'),
                                ('Expecting value', '输出无法解析为 PDDL/JSON。')]:
            if pattern in error:
                return message
    if status == 'mapping_error':
        for pattern, message in [('Unknown mapping target', '评测映射未完成：映射目标不在参考对象集合中。'),
                                 ('model did not return a JSON object', '评测映射未完成：响应无法解析。'),
                                 ('Object mapping cancelled', '评测映射已中断。')]:
            if pattern in error:
                return message
    labels = {'invalid_output': '输出结构无法解析', 'generation_error': '模型调用失败',
              'invalid_plan_step': '动作无法映射',
              'mapping_error': '评测映射未完成', 'mapping_incomplete': '评测映射不完整',
              'action_mapping_ambiguous': '评测动作映射存在歧义', 'unsupported_pddl': '求解器不支持该 PDDL',
              'infrastructure_error': '评测未完成'}
    label = labels.get(status, status)
    return label + ('：' + ' '.join(error.splitlines())[:240] if error else '。')


def render_abstract_report(results, summary, report_root, unresolved):
    """Compare the splits and link each nonpassing task to its recorded evidence."""
    nl = summary.get('output_format') == 'nl'
    present, overall = summary['by_split'], summary['overall']
    splits = [name for name in SPLIT_LABELS.values() if name in present]
    splits += sorted(present.keys() - set(splits))
    statistics = [present[split] for split in splits] + [overall]
    headers = ['指标', *(f"{split}（{present[split]['total']}）" for split in splits), f"合计（{overall['total']}）"]
    counts, overall_counts, failures = defaultdict(Counter), Counter(), defaultdict(list)
    for result in results:
        category = diagnostic_group(result, unresolved)
        counts[result['split']][category] += 1
        overall_counts[category] += 1
        if category:
            failures[result['split']].append((category, result))
    diagnostics = [counts[split] for split in splits] + [overall_counts]

    def rate(count, total):
        return f'{100 * count / total:.1f}'.removesuffix('.0') if total else '—'

    table = [['Acc（全部任务）', *(f"**{rate(row['task_success'], row['total'])}**" for row in statistics)]]
    metrics = [('spl', 'SPL')]
    if not nl:
        metrics += [('solv', 'Solv'), ('init_acc', 'Init Acc（可比子集）')]
    for key, label in metrics:
        table.append([label, *(f'{100 * row[key]:.1f}' if row.get(key) is not None else '—' for row in statistics)])
    if not nl:
        table.append(['Init可比任务数', *(str(row.get('init_comparable', 0)) for row in statistics)])
    if pairs := summary.get('pairs'):
        by_id = {result['id']: result for result in results}
        cells = []
        for split in [*splits, None]:
            matching = [p for p in pairs if split is None or by_id[p['variant']]['split'] == split]
            cells.append(rate(sum(pair['both_tasks_pass'] for pair in matching), len(matching)))
        table.append(['原规则与新规则均成功率', *cells])
    if not nl:
        for label, keys in [('init错误率 (写错/冲突)', ['init_mismatch', 'init_inconsistent']),
                            ('goal错误率', ['goal_error']),
                            ('oper错误率 (建模/缺失)', ['operator_modeling', 'operator_missing'])]:
            table.append([label, *('/'.join(rate(count[key], row['evaluated']) for key in keys)
                                   if row['evaluated'] else '—' for count, row in zip(diagnostics, statistics))])
    for key, label in [('plan_error', '计划错误率'), ('output_error', '输出错误率'), ('unattributed', '待归因率')]:
        if overall_counts[key]:
            table.append([label, *(rate(count[key], row['total']) for count, row in zip(diagnostics, statistics))])
    table.append(['评测完成率', *(rate(row['evaluated'], row['total']) for row in statistics)])
    if overall['unresolved']:
        table.append(['评测未完成率', *(rate(row['unresolved'], row['total']) for row in statistics)])
    lines = []
    if summary.get('complete') is False:
        lines += ['**评测未完成：本文件是中间诊断，不能作为最终结果。**', '']
    lines += [*markdown_table(headers, table), '']
    if 'prompt_template' in summary:
        filename = 'summary_abs.json' if summary['complete'] else 'summary_abs.partial.json'
        lines += [f'本次评测配置的 prompt 模板全文见 [汇总 JSON]({filename})。', '']
    if kinds := summary.get('by_evaluation_kind'):
        rows = [[EVALUATION_LABELS.get(kind, kind), row['total'], row['evaluated'], row['task_success'],
                 rate(row['task_success'], row['evaluated']), row['unresolved']] for kind, row in kinds.items()]
        lines += [*markdown_table(['评测条件', '任务数', '已完成数', '成功数', '成功率（%）', '未完成数'], rows), '']
    report_root = Path(report_root)
    for split in splits:
        evidence = defaultdict(lambda: defaultdict(list))
        for category, result in sorted(failures[split], key=lambda item: [int(part) if part.isdigit() else part
                                                                        for part in re.split(r'(\d+)', item[1]['id'])]):
            target = f"{result['dataset']}/{result['task']}/result.json"
            number = re.search(r'\d+$', result['task'])
            identifier = number[0] if number else result['task']
            link = f"[{identifier}]({quote(target, safe='/._-')})" if (report_root / target).is_file() else identifier
            evidence[category][_brief_error(result)].append(link)
        if evidence:
            body = []
            for category, label in GROUP_NAMES.items():
                if category in evidence:
                    body += [f'**{label}**', '', *('- ' + '、'.join(links) + '：' + reason
                                                  for reason, links in evidence[category].items()), '']
            lines += markdown_details(split, body)
    return '\n'.join(lines).rstrip() + '\n'
