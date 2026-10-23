"""Compact reports from the abstract evaluator's recorded task decisions."""
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from swm.reporting import markdown_details, markdown_table

SPLIT_LABELS = {'abs_a_id': 'A-ID', 'abs_a_scale': 'A-scale', 'abs_b_base': 'B-base', 'abs_b_rule': 'B-rule'}
ERROR_TYPES = {'init_mismatch':'init写错','init_inconsistent':'init冲突','goal_error':'goal错误',
               'operator_modeling':'oper建模错误','operator_missing':'oper整体缺失','else':'else'}
GROUP_NAMES = {**ERROR_TYPES,'pending':'评测未完成'}


def diagnostic_group(result, unresolved):
    if result['status'] in unresolved:
        return 'pending'
    if result['task_success']:
        return None
    proof = result.get('diagnosis') or {}
    if proof.get('category') == 'goal_error':
        return 'goal_error'
    subcategories = {'init_error':{'init_mismatch','init_inconsistent'},
                     'operator_error':{'operator_modeling','operator_missing'}}
    if proof.get('subcategory') in subcategories.get(proof.get('category'),set()):
        return proof['subcategory']
    return 'else'


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
        for pattern,message in [("Missing ')'", 'PDDL 括号不配对：缺少右括号。'),
                                ('Unparsed tokens remain', 'PDDL 主体后存在多余内容。'),
                                ('Expecting value', '输出无法解析为 PDDL/JSON。')]:
            if pattern in error:
                return message
    if status == 'mapping_error' and 'Unknown mapping target' in error:
        return '评测映射未完成：映射目标不在参考对象集合中。'
    if status == 'mapping_error' and 'model did not return a JSON object' in error:
        return '评测映射未完成：响应无法解析。'
    if status == 'mapping_error' and 'Object mapping cancelled' in error:
        return '评测映射已中断。'
    labels = {'invalid_output': '输出结构无法解析', 'generation_error': '模型调用失败',
              'mapping_error': '评测映射未完成', 'mapping_incomplete': '评测映射不完整',
              'action_mapping_ambiguous': '评测动作映射存在歧义', 'unsupported_pddl': '求解器不支持该 PDDL',
              'infrastructure_error': '评测未完成'}
    label = labels.get(status, status)
    return label + ('：' + ' '.join(error.splitlines())[:240] if error else '。')


def render_abstract_report(results, summary, report_root, unresolved):
    """Compare splits once, then show only nonpassing tasks with linked evidence."""
    present = summary['by_split']
    splits = [name for name in SPLIT_LABELS.values() if name in present]
    splits += sorted(set(present) - set(splits))
    groups = [[result for result in results if result['split'] == split] for split in splits] + [results]
    statistics = [present[split] for split in splits] + [summary['overall']]
    headers = ['指标', *(f"{split}（{present[split]['total']}）" for split in splits), f"合计（{summary['overall']['total']}）"]
    def rate(count,total):
        return f'{100 * count / total:.1f}'.removesuffix('.0') if total else '—'
    cells = [rate(row['task_success'],row['evaluated']) for row in statistics]
    table = [['任务成功率', *(f'**{cell}**' for cell in cells)]]
    if summary.get('pairs'):
        by_id = {result['id']: result for result in results}
        cells = []
        for split in [*splits, None]:
            pairs = [pair for pair in summary['pairs'] if pair['evaluated'] and
                     (split is None or by_id[pair['variant']]['split'] == split)]
            passed = sum(pair['both_tasks_pass'] for pair in pairs)
            cells.append(rate(passed,len(pairs)))
        table.append(['原规则与新规则均成功率', *cells])
    diagnostics = [Counter(diagnostic_group(result, unresolved) for result in group) for group in groups]
    for label,keys in [('init错误率 (写错/冲突)',['init_mismatch','init_inconsistent']),
                       ('goal错误率',['goal_error']),
                       ('oper错误率 (建模/缺失)',['operator_modeling','operator_missing']),
                       ('else',['else'])]:
        table.append([label, *('/'.join(rate(count[key],row['evaluated']) for key in keys)
                              if row['evaluated'] else '—' for count,row in zip(diagnostics,statistics))])
    table.append(['评测未完成率',*(rate(row['unresolved'],row['total']) for row in statistics)])
    lines = [*markdown_table(headers, table), '']
    for split, group in zip(splits, groups):
        failures = [result for result in group if diagnostic_group(result, unresolved)]
        if not failures:
            continue
        evidence = defaultdict(lambda: defaultdict(list))
        for result in sorted(failures, key=lambda result: [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', result['id'])]):
            evidence[diagnostic_group(result, unresolved)][_brief_error(result)].append(result)
        body = []
        for category, label in GROUP_NAMES.items():
            if category not in evidence:
                continue
            reasons = evidence[category]
            body += [f'**{label}**', '']
            for reason, members in reasons.items():
                links = []
                for result in members:
                    directory = Path(report_root) / result['dataset'] / result['task']
                    target = quote(f"{result['dataset']}/{result['task']}/result.json", safe='/._-')
                    number = re.search(r'\d+$', result['task'])
                    identifier = number[0] if number else result['task']
                    links.append(f'[{identifier}]({target})' if (directory / 'result.json').is_file() else identifier)
                body.append('- ' + '、'.join(links) + '：' + reason)
            body.append('')
        lines += markdown_details(split, body)
    return '\n'.join(lines).rstrip() + '\n'
