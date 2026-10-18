"""Present complete plan orders, or factor their shared branches.

This represents action orders only. It neither merges execution states nor
identifies plans as equivalent. A continuation is executed for every branch.
"""


def order_similar_plans(candidates: list[dict]) -> list[dict]:
    """Medoid first, then longest shared prefix and smallest Kendall distance.

    Only action orders and anonymous IDs enter this ordering. No states,
    equivalence judgments, reference verdicts or task-specific rules are used.
    """
    if not candidates:
        raise ValueError('Expected at least one candidate')
    positions = {c['id']: {action: i for i, action in enumerate(c['order'])} for c in candidates}

    def distance(a, b):
        order, target = a['order'], positions[b['id']]
        return sum(target[x] > target[y] for i, x in enumerate(order) for y in order[i+1:])

    first = min(candidates, key=lambda a: (sum(distance(a, b) for b in candidates), a['id']))
    ordered, remaining = [first], [c for c in candidates if c is not first]
    while remaining:
        previous = ordered[-1]

        def priority(candidate):
            shared = 0
            for a, b in zip(previous['order'], candidate['order']):
                if a != b:
                    break
                shared += 1
            return -shared, distance(previous, candidate), candidate['id']

        chosen = min(remaining, key=priority)
        remaining.remove(chosen)
        ordered.append(chosen)
    return ordered


def render_named_orders(candidates: list[dict], actions: dict[int, str], *, sort_similar: bool = True) -> str:
    """Complete linear orders with the existing bound operation's headline."""
    titles = {identifier: description.splitlines()[0] for identifier, description in actions.items()}
    return '\n'.join(candidate['id'] + ': ' + ' -> '.join(
        f'A{identifier:02d}[{titles[identifier]}]' for identifier in candidate['order'])
        for candidate in (order_similar_plans(candidates) if sort_similar else candidates))


def factor_plan_orders(candidates: list[dict]) -> dict:
    """Build a compressed prefix tree, factoring each node's common suffix.

    Callers supply distinct IDs and valid orders of the same action multiset.
    Step ranges always refer to positions in the complete original plans.
    """
    if not candidates or len({len(c['order']) for c in candidates}) != 1:
        raise ValueError('Expected nonempty candidates with equal-length orders')

    def build(rows, begin, end):
        first = rows[0]['order']
        left, right = begin, end
        while left < right and all(c['order'][left] == first[left] for c in rows):
            left += 1
        # Bound the suffix by the unconsumed interval: never overlap a prefix.
        while right > left and all(c['order'][right-1] == first[right-1] for c in rows):
            right -= 1
        groups = {}
        if left < right:
            for candidate in rows:
                groups.setdefault(candidate['order'][left], []).append(candidate)
        return {'candidate_ids': [c['id'] for c in rows], 'start_step': begin + 1, 'end_step': end,
                'prefix': list(first[begin:left]),
                # Finish small branches before descending into the remaining
                # shared workflow; preserve input order for equal-size groups.
                'branches': [build(group, left, right) for group in sorted(groups.values(), key=len)],
                'continuation': list(first[right:end])}

    return build(candidates, 0, len(candidates[0]['order']))


def expand_plan_orders(tree: dict) -> dict[str, list[int]]:
    """Recover exactly the original labelled orders, without branch mixing."""
    prefix, continuation = tree['prefix'], tree['continuation']
    if not tree['branches']:
        return {identifier: prefix + continuation for identifier in tree['candidate_ids']}
    result = {}
    for branch in tree['branches']:
        for identifier, order in expand_plan_orders(branch).items():
            if identifier in result:
                raise ValueError('A candidate occurs on multiple branches')
            result[identifier] = prefix + order + continuation
    return result


def render_plan_orders(tree: dict) -> str:
    """Each line names its plans and absolute steps; tree lines show forks."""
    lines = []

    def segment(node, actions, start, indent, connector='', continuation=False):
        if not actions:
            return
        steps = str(start) if len(actions) == 1 else f'{start}-{start+len(actions)-1}'
        qualifier = '; continue separately' if continuation else ''
        lines.append(indent + connector + ', '.join(node['candidate_ids']) +
                     f' [steps {steps}{qualifier}]: ' + ' -> '.join(f'A{i:02d}' for i in actions))

    def emit(node, indent='', connector=''):
        segment(node, node['prefix'], node['start_step'], indent, connector)
        child_indent = indent + ('   ' if connector == '└─ ' else '│  ' if connector else '')
        for index, child in enumerate(node['branches']):
            emit(child, child_indent, '└─ ' if index == len(node['branches'])-1 else '├─ ')
        segment(node, node['continuation'], node['end_step']-len(node['continuation'])+1,
                child_indent, continuation=True)

    emit(tree)
    return '\n'.join(lines)
