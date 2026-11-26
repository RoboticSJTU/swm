"""Group PSV representatives and encode their original grounded orders for review."""
from fractions import Fraction
from itertools import takewhile
from pathlib import Path
from typing import NamedTuple

import numpy as np


MAX_CLASSES = 50


class Merge(NamedTuple):
    left: int
    right: int
    score: Fraction  # Mean LCS length over every cross-child pair (in actions).


def validated_orders(model, representatives):
    """Order leaves by plan content; keep class ranks only to identify the final members."""
    canonical = {action.line: index for index, action in enumerate(sorted(model.actions, key=lambda a: a.line))}
    action_keys = [canonical[action.line] for action in model.actions]
    rows = []
    for row in representatives:
        order = row["order"]
        if not model.replay(order):
            raise ValueError("Representative is not executable to the candidate model's goal")
        rows.append((tuple(action_keys[i - 1] for i in order), row["class_rank"]))
    if len({rank for _, rank in rows}) != len(rows):
        raise ValueError("Repeated class rank in one PDDL instance")
    if len({order for order, _ in rows}) != len(rows):
        raise ValueError("Repeated grounded plan in the pending class inventory")
    rows.sort()
    return [rank for _, rank in rows], [order for order, _ in rows]


def lcs_sums(orders, dtype):
    """Initial cross-cluster sums. A bit per matching position retains repeated actions."""
    count = len(orders)
    sums = np.zeros((count, count), dtype=dtype)
    for i, order in enumerate(orders):
        positions = {}
        for offset, action in enumerate(order):
            positions[action] = positions.get(action, 0) | (1 << offset)
        for j in range(i):
            # Set bits mark increases along a row of the standard LCS dynamic program.
            frontier = 0
            for action in orders[j]:
                matches = frontier | positions.get(action, 0)
                frontier = matches & ~(matches - ((frontier << 1) | 1))
            sums[i, j] = sums[j, i] = frontier.bit_count()
    return sums


def merge_history(orders):
    """Globally highest average linkage, with content-sorted leaf indices breaking ties."""
    count = len(orders)
    if count < 2:
        return []
    length = len(orders[0])
    max_pairs = (count // 2) * (count - count // 2)
    # Rational means differ by at least 1/max_pairs**2. Use exact Fraction scores
    # when binary64 cannot guarantee the same ordering and tie decisions.
    float_safe = length * max_pairs**2 < 2**51
    sum_type = np.int64 if length * max_pairs <= np.iinfo(np.int64).max else object
    sums = lcs_sums(orders, sum_type)
    scores = sums.astype(float if float_safe else object)
    np.fill_diagonal(scores, -1)
    slots = np.arange(count)
    sizes = np.ones(count, dtype=np.int64)
    nearest = scores.argmax(axis=1)
    history = []
    for _ in range(count - 1):
        left = int(scores[slots, nearest].argmax())
        right = int(nearest[left])
        score = Fraction(int(sums[left, right]), int(sizes[left]) * int(sizes[right]))
        history.append(Merge(left, right, score))

        sizes[left] += sizes[right]
        sizes[right] = 0
        other = slots[(sizes > 0) & (slots != left)]
        sums[left, other] += sums[right, other]
        sums[other, left] = sums[left, other]
        denominators = sizes[left] * sizes[other]
        if float_safe:
            updated = sums[left, other] / denominators
        else:
            updated = [Fraction(int(sums[left, j]), int(d)) for j, d in zip(other, denominators)]
        scores[left, other] = scores[other, left] = updated
        scores[right, :] = scores[:, right] = -1
        # A weighted average of the two old similarities cannot exceed both.
        # Only rows whose previous winner was consumed need a fresh search.
        refresh = slots[(sizes > 0) & ((nearest == left) | (nearest == right) | (slots == left))]
        for row in refresh:
            nearest[row] = scores[row].argmax()
    return history


def cut_count(merges):
    """Cut before the unique drop that exceeds the runner-up by at least one aligned action."""
    drops = [older.score - newer.score for older, newer in zip(merges, merges[1:])]
    if not drops:
        return len(merges)
    top, runner_up = (sorted(drops, reverse=True) + [0])[:2]
    return drops.index(top) + 1 if top - runner_up >= 1 else len(merges)


def natural_clusters(ranks, merges):
    """Return each cluster's tree leaves and linkage strengths at every adjacent-leaf boundary."""
    nodes = {i: ([rank], []) for i, rank in enumerate(ranks)}
    for left, right, score in merges[:cut_count(merges)]:
        leaves, strengths = nodes[left]
        next_leaves, next_strengths = nodes.pop(right)
        leaves.extend(next_leaves)
        strengths.append(score)
        strengths.extend(next_strengths)
    return [nodes[i] for i in sorted(nodes)]


def packet_segments(seams, capacity=MAX_CLASSES):
    """At minimum packet count, minimize severed linkage; prefer balanced packets on ties."""
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Packet capacity must be a positive integer")
    n = len(seams) + 1
    count = (n + capacity - 1) // capacity
    if count == 1:
        return [(0, n)]
    costs = {0: (Fraction(0), 0)}
    parents = []
    for k in range(1, count + 1):
        # Restrict endpoints to those that leave room for all remaining packets.
        remaining = count - k
        first_end = max(k, n - capacity * remaining)
        last_end = min(k * capacity, n - remaining)
        previous, next_costs = {}, {}
        for end in range(first_end, last_end + 1):
            strength, squares, start = min(
                (total, balance + (end - start)**2, start)
                for start, (total, balance) in costs.items() if 1 <= end - start <= capacity
            )
            previous[end] = start
            next_costs[end] = (strength + (seams[end - 1] if end < n else 0), squares)
        parents.append(previous)
        costs = next_costs
    segments = []
    end = n
    for previous in reversed(parents):
        start = previous[end]
        segments.append((start, end))
        end = start
    return segments[::-1]



def group_clusters(model, representatives, *, capacity=MAX_CLASSES):
    """Return logical clusters and submission packets in the same class-rank namespace."""
    if capacity is not None and (type(capacity) is not int or capacity < 1):
        raise ValueError("Packet capacity must be a positive integer or None")
    ranks, orders = validated_orders(model, representatives)
    clusters = natural_clusters(ranks, merge_history(orders))
    logical = [sorted(leaves) for leaves, _ in clusters]
    if capacity is None:
        return logical, logical
    packets = [sorted(leaves[start:end]) for leaves, seams in clusters
               for start, end in packet_segments(seams, capacity)]
    return logical, packets


_TREE_HEADER = ("Plan orders: each path extends the named parent by the actions on its line. "
                "Each ID names its complete path. ROOT is the empty start.")


def encode_plan_tree(candidates):
    """Render shared prefixes once; check that every ID recovers its exact action sequence."""
    orders = {item["id"]: list(item["order"]) for item in candidates}
    if not orders or len(orders) != len(candidates):
        raise ValueError("Expected distinct class IDs in a nonempty packet")
    lines = [_TREE_HEADER]

    def visit(members, depth, parent):
        branches = {}
        for name, seq in members:
            if len(seq) == depth:
                lines.append(f"{name}: {parent}")
            else:
                branches.setdefault(seq[depth], []).append((name, seq))
        for branch in branches.values():
            columns = zip(*(seq[depth:] for _, seq in branch))
            prefix = [col[0] for col in takewhile(lambda col: len(set(col)) == 1, columns)]
            node = f"path{len(lines)}"
            labels = " ".join(f"A{action:02d}" for action in prefix)
            lines.append(f"{node} = {parent} + {labels}")
            visit(branch, depth + len(prefix), node)

    visit(orders.items(), 0, "ROOT")
    text = "\n".join(lines)
    if decode_plan_tree(text) != orders:
        raise ValueError("Printed tree does not reconstruct the original grounded orders")
    return text


def decode_plan_tree(text):
    """Decode the exact shared-prefix format presented to the judge."""
    header, *lines = text.splitlines()
    if header != _TREE_HEADER:
        raise ValueError("Unknown plan tree format")
    paths = {"ROOT": []}
    orders = {}
    for line in lines:
        if " = " in line:
            name, extension = line.split(" = ", 1)
            parent, actions = extension.split(" + ", 1)
            tokens = actions.split()
            if name in paths or parent not in paths or not tokens:
                raise ValueError("Invalid plan tree path")
            if any(token[:1] != "A" or not token[1:].isdigit() or int(token[1:]) < 1 for token in tokens):
                raise ValueError("Invalid grounded action label")
            paths[name] = paths[parent] + [int(token[1:]) for token in tokens]
        else:
            identifier, path = line.split(": ", 1)
            if identifier in orders or path not in paths:
                raise ValueError("Invalid or duplicate tree endpoint")
            orders[identifier] = paths[path]
    return orders


def review_inputs(round_dir, images, instruction, kf_actions, model, robot_configuration):
    """Local file revisions and textual evidence used to decide whether a round can resume."""
    def revision(path):
        path = Path(path)
        stat = path.stat()
        return [str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]

    directory = Path(round_dir)
    return {
        "round_files": [revision(directory / name) for name in ("domain.pddl", "problem.pddl", "plan.txt")],
        "images": [revision(path) for path in images],
        "instruction": instruction,
        "kf_actions": kf_actions,
        "model": model,
        "robot_configuration": robot_configuration,
    }
