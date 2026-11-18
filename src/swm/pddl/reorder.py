from __future__ import annotations

from bisect import bisect_left
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Sequence
from contextlib import redirect_stdout
from copy import copy
import heapq
import io
import itertools
import json
import math
import mmap
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

from swm.pddl.strips import (
    GroundAction,
    Literal,
    apply_action,
    assert_goals,
    goals_satisfied,
    ground_plan,
    parse_domain,
    parse_plan,
    parse_problem_model,
    rollout,
    parse_sexpr,
)

ResourceSig = tuple[str, ...]
ObjectSig = tuple[str, ...]


def _successors(state, remaining, transition):
    """The common legal-action expansion for preferred and complete search."""
    for index in bit_indices(remaining):
        after = transition(state, index)
        if after is not None:
            yield index, after


@dataclass
class OccupancyModel:
    free_pred: str
    occ_pred: str
    occ_arity: int
    resource_positions: tuple[int, ...]
    object_positions: tuple[int, ...]
    resources: set[ResourceSig]
    implicit_resource: ResourceSig | None = None

    def project(self, lit: Literal) -> tuple[ResourceSig, ObjectSig] | None:
        if lit[0] != self.occ_pred or len(lit) != self.occ_arity + 1:
            return None

        if self.resource_positions:
            resource = tuple(lit[1 + i] for i in self.resource_positions)
            if resource not in self.resources:
                return None
        else:
            if self.implicit_resource is None:
                return None
            resource = self.implicit_resource

        obj = tuple(lit[1 + i] for i in self.object_positions)
        return (resource, obj) if obj else None

    def build(self, resource: ResourceSig, obj: ObjectSig) -> Literal:
        args: list[str | None] = [None] * self.occ_arity

        if self.resource_positions:
            if len(resource) != len(self.resource_positions):
                raise ValueError("Resource signature arity mismatch")
            for pos, value in zip(self.resource_positions, resource):
                args[pos] = value
        elif self.implicit_resource != resource:
            raise ValueError("Invalid implicit resource")

        if len(obj) != len(self.object_positions):
            raise ValueError("Object signature arity mismatch")
        for pos, value in zip(self.object_positions, obj):
            args[pos] = value

        if any(value is None for value in args):
            raise ValueError("Could not rebuild occupancy literal")
        return (self.occ_pred, *[value for value in args if value is not None])


def infer_occupancy_model(
    init_state: set[Literal], plan: list[GroundAction]
) -> OccupancyModel | None:
    states = [set(init_state)]
    for action in plan:
        states.append(apply_action(states[-1], action))

    pred_arity: dict[str, int] = {}
    pred_literals: dict[str, list[Literal]] = defaultdict(list)
    add_count: Counter[str] = Counter()
    del_count: Counter[str] = Counter()

    def record(lit: Literal) -> None:
        pred = lit[0]
        arity = len(lit) - 1
        if pred in pred_arity and pred_arity[pred] != arity:
            raise ValueError(f"Predicate {pred} appears with inconsistent arity")
        pred_arity[pred] = arity
        pred_literals[pred].append(lit)

    for state in states:
        for lit in state:
            record(lit)

    for action in plan:
        for lit in action.pre_pos | action.pre_neg | action.add_eff | action.del_eff:
            record(lit)
        for lit in action.add_eff:
            add_count[lit[0]] += 1
        for lit in action.del_eff:
            del_count[lit[0]] += 1

    def score_model(model: OccupancyModel) -> tuple | None:
        # A resource must be either free or occupied by exactly one object in
        # every observed state. Names and approximate correlations are not proof.
        previous_free = None
        acquire = release = state_matches = 0
        free_init_count = occ_init_count = 0
        for state in states:
            free_now = {tuple(lit[1:]) for lit in state if lit[0] == model.free_pred}
            occupied = set()
            for lit in state:
                if lit[0] != model.occ_pred:
                    continue
                projected = model.project(lit)
                if projected is None or projected[0] in occupied:
                    return None
                occupied.add(projected[0])
            if free_now & occupied or free_now | occupied != model.resources:
                return None
            if previous_free is not None:
                acquire += len(previous_free - free_now)
                release += len(free_now - previous_free)
            else:
                free_init_count, occ_init_count = len(free_now), len(occupied)
            previous_free = free_now
            state_matches += len(occupied)

        if not state_matches or not (acquire + release):
            return None
        return (
            acquire + release,
            bool(model.resource_positions),
            -pred_arity[model.free_pred],
            min(acquire, release),
            free_init_count,
            -occ_init_count,
            state_matches,
        )

    if (
        "hand_free" in pred_arity
        and "holding" in pred_arity
        and pred_arity["hand_free"] == 1
        and pred_arity["holding"] == 2
        and (hands := {tuple(lit[1:]) for lit in pred_literals["hand_free"]})
    ):
        model = OccupancyModel(
            free_pred="hand_free",
            occ_pred="holding",
            occ_arity=2,
            resource_positions=(0,),
            object_positions=(1,),
            resources=hands,
        )
        if score_model(model) is not None:
            return model

    # A valid episode may consist only of an acquisition or only of a release.
    dynamic_preds = sorted(
        pred for pred in pred_arity if add_count[pred] or del_count[pred]
    )

    candidates: list[tuple[tuple, OccupancyModel]] = []

    for free_pred in dynamic_preds:
        resources = {tuple(lit[1:]) for lit in pred_literals[free_pred]}
        if not resources:
            continue

        for occ_pred in dynamic_preds:
            if occ_pred == free_pred:
                continue

            occ_arity = pred_arity[occ_pred]
            positions = list(range(occ_arity))
            for count in range(occ_arity + 1):
                for resource_positions in combinations(positions, count):
                    object_positions = tuple(
                        pos for pos in positions if pos not in resource_positions
                    )
                    if not object_positions:
                        continue

                    implicit_resource: ResourceSig | None = None
                    if not resource_positions:
                        if len(resources) != 1:
                            continue
                        implicit_resource = next(iter(resources))

                    model = OccupancyModel(
                        free_pred=free_pred,
                        occ_pred=occ_pred,
                        occ_arity=occ_arity,
                        resource_positions=tuple(resource_positions),
                        object_positions=object_positions,
                        resources=resources,
                        implicit_resource=implicit_resource,
                    )

                    score = score_model(model)
                    if score is not None:
                        candidates.append((score, model))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


# ============================================================
# Strict DAG constraints
# ============================================================


def build_predecessors(
    init_state: set[Literal],
    goal_pos: set[Literal],
    goal_neg: set[Literal],
    plan: list[GroundAction],
    occupancy_model: OccupancyModel | None,
) -> tuple[
    dict[int, set[int]],
    list[tuple[int, int, Literal]],
    list[tuple[int, int, Literal]],
]:
    """Preserve the reference's non-resource causal links and protect their uses."""
    n = len(plan)
    preds = {i: set() for i in range(n)}
    resource_preds: set[str] = set()

    if occupancy_model is not None:
        resource_preds.add(occupancy_model.free_pred)
        resource_preds.add(occupancy_model.occ_pred)

    def add_edge(before: int, after: int) -> None:
        if before != after:
            preds[after].add(before)

    def latest_supporter(end: int, lit: Literal, positive: bool) -> int | None:
        if positive:
            provider = -1 if lit in init_state else None
            for i in range(end):
                if lit in plan[i].del_eff:
                    provider = None
                if lit in plan[i].add_eff:
                    provider = i
        else:
            provider = -1 if lit not in init_state else None
            for i in range(end):
                if lit in plan[i].add_eff:
                    provider = None
                if lit in plan[i].del_eff:
                    provider = i
        return provider

    pos_links: list[tuple[int, int, Literal]] = []
    neg_links: list[tuple[int, int, Literal]] = []

    for consumer, action in enumerate(plan):
        for lit in action.pre_pos:
            if lit[0] in resource_preds:
                continue
            provider = latest_supporter(consumer, lit, positive=True)
            if provider is None:
                raise ValueError(
                    f"Original plan invalid: no supporter for positive precondition {lit} "
                    f"of step {consumer}: {action.to_line()}"
                )
            pos_links.append((provider, consumer, lit))
            if provider >= 0:
                add_edge(provider, consumer)

        for lit in action.pre_neg:
            if lit[0] in resource_preds:
                continue
            provider = latest_supporter(consumer, lit, positive=False)
            if provider is None:
                raise ValueError(
                    f"Original plan invalid: no supporter for negative precondition (not {lit}) "
                    f"of step {consumer}: {action.to_line()}"
                )
            neg_links.append((provider, consumer, lit))
            if provider >= 0:
                add_edge(provider, consumer)

    for provider, consumer, lit in pos_links:
        for k, action in enumerate(plan):
            if k == consumer or lit not in action.del_eff:
                continue
            if provider == -1:
                add_edge(consumer, k)
            elif provider >= 0 and k < provider:
                add_edge(k, provider)
            elif provider >= 0 and k > consumer:
                add_edge(consumer, k)

    for provider, consumer, lit in neg_links:
        for k, action in enumerate(plan):
            if k == consumer or lit not in action.add_eff:
                continue
            if provider == -1:
                add_edge(consumer, k)
            elif provider >= 0 and k < provider:
                add_edge(k, provider)
            elif provider >= 0 and k > consumer:
                add_edge(consumer, k)

    for lit in goal_pos:
        provider = latest_supporter(n, lit, positive=True)
        if provider is None:
            raise ValueError(f"Original plan invalid: positive goal {lit} not achieved")
        if provider >= 0 and lit[0] not in resource_preds:
            for k, action in enumerate(plan):
                if k < provider and lit in action.del_eff:
                    add_edge(k, provider)

    for lit in goal_neg:
        provider = latest_supporter(n, lit, positive=False)
        if provider is None:
            raise ValueError(
                f"Original plan invalid: negative goal (not {lit}) not achieved"
            )
        if provider >= 0 and lit[0] not in resource_preds:
            for k, action in enumerate(plan):
                if k < provider and lit in action.add_eff:
                    add_edge(k, provider)

    return preds, pos_links, neg_links


# ============================================================
# Heuristic DFS reordering
# ============================================================


def reorder_by_search(
    init_state: set[Literal],
    goal_pos: set[Literal],
    goal_neg: set[Literal],
    plan: list[GroundAction],
    preds: dict[int, set[int]],
    pos_links: list[tuple[int, int, Literal]],
    neg_links: list[tuple[int, int, Literal]],
    occupancy_model: OccupancyModel | None,
    *,
    max_states: int = 50000,
    timeout: float = 2.0,
) -> list[GroundAction]:
    """Keep active causal tasks together, retaining the reference on budget exhaustion."""
    if max_states < 1 or not timeout > 0:
        raise ValueError("Invalid reordering search budget")
    deadline = time.perf_counter() + timeout
    n = len(plan)
    if not n:
        return plan
    done_all = (1 << n) - 1
    # The agenda only ranks legal successors. A fully exhausted state/mask is
    # infeasible regardless of the agenda used to reach it.
    failed: set[tuple[int, frozenset[Literal]]] = set()

    links_by_consumer: dict[int, list[tuple[int, Literal, bool]]] = defaultdict(list)
    for provider, consumer, lit in pos_links:
        if provider >= 0:
            links_by_consumer[consumer].append((provider, lit, True))
    for provider, consumer, lit in neg_links:
        if provider >= 0:
            links_by_consumer[consumer].append((provider, lit, False))

    causal_support = Counter(
        provider
        for provider, _consumer, _lit in pos_links + neg_links
        if provider >= 0
    )

    # Occupancy links guide preparation, but do not restrict legal reorderings.
    causal_links = [(p, c, lit, True) for p, c, lit in pos_links]
    causal_links += [(p, c, lit, False) for p, c, lit in neg_links]
    holders = {
        lit: -1 for lit in init_state
        if occupancy_model is not None and occupancy_model.project(lit) is not None
    }
    holding_roots = dict.fromkeys(holders, 0)
    episode_ends = defaultdict(int)
    acquisitions = []
    for consumer, action in enumerate(plan):
        for lit in action.pre_pos & holders.keys():
            causal_links.append((holders[lit], consumer, lit, True))
        released = action.del_eff & holders.keys()
        origins = 0
        for lit in released:
            origins |= holding_roots[lit]
        for lit in action.del_eff:
            holders.pop(lit, None)
            holding_roots.pop(lit, None)
        acquired = set()
        if occupancy_model is not None:
            acquired = {
                lit for lit in action.add_eff
                if occupancy_model.project(lit) is not None
                and lit not in holders
            }
            holders.update((lit, consumer) for lit in acquired)
            holding_roots.update((lit, origins if released else 1 << consumer) for lit in acquired)
        if not acquired:
            for root in bit_indices(origins):
                episode_ends[root] |= 1 << consumer
        acquisitions.append(bool(acquired))

    def ancestor_masks(predecessors):
        result = list(predecessors)
        for provider in range(n):
            for consumer in range(n):
                if result[consumer] & (1 << provider):
                    result[consumer] |= result[provider]
        return result

    support_preds = [0] * n
    readers = defaultdict(int)
    for provider, consumer, lit, positive in causal_links:
        readers[provider, lit, positive] |= 1 << consumer
        if provider >= 0:
            support_preds[consumer] |= 1 << provider
    support_ancestors = ancestor_masks(support_preds)
    work_preds = list(support_preds)
    ancestors = [sum(1 << p for p in preds[i]) | support_preds[i] for i in range(n)]

    # A consuming use ends an episode only after its other readers have finished.
    closures = []
    for provider, consumer, lit, positive in causal_links:
        opposite = plan[consumer].del_eff if positive else plan[consumer].add_eff
        others = readers[provider, lit, positive] & ~(1 << consumer)
        # A handoff closes the old holding even though it acquires a new one.
        holding = occupancy_model is not None and occupancy_model.project(lit) is not None
        # Initial holdings are already open episodes. Other initial conditions
        # become commitments only once at least one of their readers has run.
        activation = (1 << provider) if provider >= 0 else (0 if holding else others)
        if (lit in opposite and (not acquisitions[consumer] or holding)
                and (activation or holding)):
            ancestors[consumer] |= others
            work_preds[consumer] |= others
            closures.append((activation, consumer, others, lit, positive, holding))

    ancestors = ancestor_masks(ancestors)
    work_ancestors = ancestor_masks(work_preds)

    # Finish a terminal causal chain or one input branch of a join. Threat
    # ordering is needed for legality, but does not identify a coherent task.
    providers = 0
    for mask in support_preds:
        providers |= mask
    task_ends = set(bit_indices(done_all & ~providers))
    for mask in support_preds:
        inputs = [p for p in bit_indices(mask)
                  if not any(support_ancestors[q] & (1 << p) for q in bit_indices(mask))]
        if len(inputs) > 1:
            task_ends.update(p for p in inputs if not acquisitions[p])

    # Include cleanup only when all its causal work belongs to this task.
    # Cleanup shared by independent tasks remains outside their contexts.
    cleanup = []
    for _activation, end, _others, _lit, _positive, _holding in closures:
        cleanup.append((end, work_ancestors[end]))
    context_ends = set(task_ends)
    for end, work in cleanup:
        if end not in context_ends:
            continue
        other_work = 0
        for other in task_ends:
            if other != end and not support_ancestors[other] & (1 << end):
                other_work |= support_ancestors[other] | (1 << other)
        if work and not work & ~other_work:
            # Closing shared setup is coordination of the tasks that use it,
            # rather than a reason to batch their prefixes ahead of their ends.
            context_ends.remove(end)
    bundles = {}
    for end in sorted(context_ends):
        core = work_ancestors[end] | (1 << end)
        members = ancestors[end] | (1 << end)
        owned_cleanup = 0
        for close, work in cleanup:
            if work and not work & ~core:
                members |= ancestors[close] | (1 << close)
                owned_cleanup |= 1 << close
        core |= owned_cleanup
        bundles[members] = bundles.get(members, 0) | core

    # Prepare the independent inputs of a chain before starting it. Holdings
    # retain their complete episode, including later grips and release.
    preparation_ends = dict(episode_ends)
    for end, providers in enumerate(support_preds):
        for root in bit_indices(providers):
            if not acquisitions[root]:
                preparation_ends[root] = preparation_ends.get(root, 0) | (1 << end)

    preparations = defaultdict(int)
    for root, endings in preparation_ends.items():
        for end in bit_indices(endings):
            for i in bit_indices(ancestors[end]):
                if i != root and not ancestors[i] & (1 << root):
                    preparations[root] |= 1 << i

    def active_closures(state, done_mask):
        return [(ancestors[end] | (1 << end), holding)
                for activation, end, _others, lit, positive, holding in closures
                if (not activation or done_mask & activation)
                and ((lit in state) == positive) and not done_mask & (1 << end)]

    def advance_agenda(state, done_mask, agenda, last=None):
        """Retain unfinished contexts and enter a branch when work on it starts."""
        agenda = list(agenda)
        while agenda and not agenda[-1] & ~done_mask:
            agenda.pop()
        active = [(mask & ~done_mask, holding) for mask, holding in active_closures(state, done_mask)]

        def choose(parent=None):
            candidates = {}
            parent_remaining = parent & ~done_mask if parent is not None else None
            excluded = {mask & ~done_mask for mask in agenda}
            for members, core in bundles.items():
                remaining = members & ~done_mask
                if not remaining or remaining in excluded:
                    continue
                if parent_remaining is not None and (remaining == parent_remaining or remaining & ~parent_remaining):
                    continue
                if last is not None and not core & (1 << last):
                    continue
                holds = any(holding and not mask & ~remaining for mask, holding in active)
                closes = holds or any(not mask & ~remaining for mask, _ in active)
                if closes or core & done_mask:
                    candidates.setdefault(remaining, (members, holds, closes))
            if any(row[1] for row in candidates.values()):
                candidates = {mask: row for mask, row in candidates.items() if row[1]}
            elif any(row[2] for row in candidates.values()):
                candidates = {mask: row for mask, row in candidates.items() if row[2]}
            maximal = [mask for mask in candidates
                       if not any(mask != other and not mask & ~other for other in candidates)]
            if len(maximal) == 1:
                return candidates[maximal[0]][0]
            if maximal and any(candidates[mask][1] for mask in maximal):
                return candidates[min(maximal, key=lambda mask: (mask.bit_count(), mask))][0]
            # Shared preparation alone does not commit to either independent
            # branch. Bind the context when work on one branch actually starts.
            return None

        if agenda and last is not None and not agenda[-1] & (1 << last):
            bridge = choose()
            if bridge is not None:
                agenda.append(bridge)
        if not agenda:
            root = choose()
            if root is not None:
                agenda.append(root)
        while agenda:
            child = choose(agenda[-1])
            if child is None:
                break
            agenda.append(child)
        return tuple(agenda)

    def current_occupancy(state: set[Literal]) -> dict[ResourceSig, ObjectSig]:
        if occupancy_model is None:
            return {}

        buckets: dict[ResourceSig, set[ObjectSig]] = defaultdict(set)
        for lit in state:
            projected = occupancy_model.project(lit)
            if projected is not None:
                resource, obj = projected
                buckets[resource].add(obj)

        return {
            resource: next(iter(objects))
            for resource, objects in buckets.items()
            if len(objects) == 1
        }

    def ready_actions(state: set[Literal], done_mask: int) -> list[int]:
        def transition(state, index):
            action = plan[index]
            if action.pre_pos <= state and not action.pre_neg & state:
                return apply_action(state, action)
            return None

        return [
            index
            for index, _ in _successors(state, done_all ^ done_mask, transition)
            if all(done_mask & (1 << pred) for pred in preds[index])
        ]

    def rank_ready_actions(state: set[Literal], done_mask: int, agenda=()) -> list[int]:
        ready = ready_actions(state, done_mask)
        if not ready:
            return []

        ready_now = set(ready)
        occupancy_now = current_occupancy(state)
        scored: list[tuple[tuple, int]] = []

        for i in ready:
            if time.perf_counter() >= deadline:
                return []
            next_state = apply_action(state, plan[i])
            next_mask = done_mask | (1 << i)
            newly_ready = len(set(ready_actions(next_state, next_mask)) - ready_now)
            completed_links = sum(
                1
                for provider, _lit, _positive in links_by_consumer[i]
                if done_mask & (1 << provider)
            )

            resource_continuity = 0
            if occupancy_model is not None:
                resource_continuity = sum(
                    occupancy_model.build(resource, obj) in plan[i].pre_pos
                    for resource, obj in occupancy_now.items()
                )

            occupancy_next = current_occupancy(next_state)
            new_resource_acquisitions = len(
                set(occupancy_next.items()) - set(occupancy_now.items())
            )
            new_holdings = {
                lit for lit in plan[i].add_eff - state
                if occupancy_model is not None and occupancy_model.project(lit) is not None
            }
            uses = [
                action for j, action in enumerate(plan)
                if not (next_mask & (1 << j)) and action.pre_pos & new_holdings
            ]
            preserving_uses = [
                action for action in uses if (action.pre_pos - action.del_eff) & new_holdings
            ]
            support = sum(len((a.pre_pos - a.del_eff) & new_holdings) for a in preserving_uses)
            # Prepare missing use conditions before occupying a hand; an early
            # release does not replace the work that preserves the holding.
            waiting = min(
                (len(a.pre_pos - next_state) + len(a.pre_neg & next_state)
                 for a in preserving_uses or uses),
                default=0,
            )
            if i in preparations:
                waiting = (preparations[i] & ~done_mask).bit_count()
            # Prefer the nearest completion of an established or already-used condition.
            finish = max(
                (
                    n - bin(ancestors[consumer] & ~done_mask).count("1")
                    for activation, consumer, others, lit, positive, _holding in closures
                    if (not activation or done_mask & activation)
                    and ((lit in state) == positive)
                    and not (done_mask & (1 << consumer))
                    and (i != consumer or done_mask & others == others)
                    and (ancestors[consumer] | (1 << consumer)) & (1 << i)
                ),
                default=0,
            )
            score = (
                bool(agenda and agenda[-1] & (1 << i)),
                finish,
                -waiting,
                completed_links,
                causal_support[i],
                newly_ready,
                resource_continuity,
                -new_resource_acquisitions,
                support,
                -i,
            )
            scored.append((score, i))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [i for _, i in scored]

    # Explicit DFS frames avoid tying valid plan length to Python's call stack.
    path: list[int] = []
    agenda = advance_agenda(init_state, 0, ())
    stack = [(0, frozenset(init_state), agenda, iter(rank_ready_actions(init_state, 0, agenda)))]
    expanded = 1
    while stack:
        if time.perf_counter() >= deadline:
            return plan
        done_mask, frozen_state, agenda, choices = stack[-1]
        i = next(choices, None)
        if i is None:
            failed.add((done_mask, frozen_state))
            stack.pop()
            if path:
                path.pop()
            continue

        next_mask = done_mask | (1 << i)
        next_state = apply_action(set(frozen_state), plan[i])
        key = (next_mask, frozenset(next_state))
        if key in failed:
            continue
        if next_mask == done_all:
            if goals_satisfied(next_state, goal_pos, goal_neg):
                return [plan[index] for index in [*path, i]]
            failed.add(key)
            continue
        if expanded >= max_states:
            return plan
        expanded += 1
        next_agenda = advance_agenda(next_state, next_mask, agenda, i)
        stack.append((next_mask, key[1], next_agenda,
                      iter(rank_ready_actions(next_state, next_mask, next_agenda))))
        path.append(i)

    return plan


# ============================================================
# Main entry
# ============================================================


def plan_reorder(
    domain_path: Path, problem_path: Path, plan_path: Path, output_path: Path
) -> None:
    try:
        schemas = parse_domain(domain_path)
        problem = parse_problem_model(problem_path, schemas)
        init_state = problem.init_state
        goal_pos = problem.goal_positive
        goal_neg = problem.goal_negative
        raw_plan, comments = parse_plan(plan_path)
        plan = ground_plan(raw_plan, schemas, problem.objects)

        original_final = rollout(init_state, plan)
        assert_goals(
            original_final, goal_pos, goal_neg, "Original plan does not satisfy goal."
        )

        occupancy_model = infer_occupancy_model(init_state, plan)
        preds, pos_links, neg_links = build_predecessors(
            init_state, goal_pos, goal_neg, plan, occupancy_model
        )
        reordered = reorder_by_search(
            init_state,
            goal_pos,
            goal_neg,
            plan,
            preds,
            pos_links,
            neg_links,
            occupancy_model,
        )

        final_state = rollout(init_state, reordered)
        assert_goals(
            final_state,
            goal_pos,
            goal_neg,
            "Final reordered plan does not satisfy goal.",
        )

        lines = [action.to_line() for action in reordered] + comments
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    except NotImplementedError as e:
        print(f"[plan_reorder] {e}. Skip reordering and keep original plan.")
        output_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")


# Complete fixed-multiset search used by Plan Soundness Validation.

def typed_symbols(items):
    result, pending = {}, []
    for index, token in enumerate(items):
        if not isinstance(token, str):
            raise NotImplementedError("Union types are not supported")
        if token == "-":
            if not pending or index + 1 == len(items):
                raise ValueError("Malformed typed declaration")
            kind = items[index + 1]
            for name in pending:
                if name in result:
                    raise ValueError(f"Duplicate symbol: {name}")
                result[name] = kind
            pending = []
        elif index == 0 or items[index - 1] != "-":
            pending.append(token)
    for name in pending:
        if name in result:
            raise ValueError(f"Duplicate symbol: {name}")
        result[name] = "object"
    return result

def plan_actions(path):
    raw = []
    for line in path.read_text().splitlines():
        line = line.split(";", 1)[0].strip()
        if not line:
            continue
        item = parse_sexpr(line)
        if not isinstance(item, list) or not item or not all(isinstance(x, str) for x in item):
            raise ValueError(f"Invalid plan line: {line}")
        raw.append((item[0], item[1:]))
    return raw

def conjunction(items):
    positive = negative = 0
    other = []
    for item in items:
        if item is False:
            return False
        if item is True:
            continue
        if item[0] == "mask":
            positive |= item[1]
            negative |= item[2]
        else:
            other.append(item)
    if positive & negative:
        return False
    if positive or negative:
        other.append(("mask", positive, negative))
    return True if not other else other[0] if len(other) == 1 else ("and", tuple(other))

def disjunction(items):
    values = []
    for item in items:
        if item is True:
            return True
        if item is not False and item not in values:
            values.append(item)
    return False if not values else values[0] if len(values) == 1 else ("or", tuple(values))

def evaluate(expression, state):
    if isinstance(expression, bool):
        return expression
    kind = expression[0]
    if kind == "mask":
        return state & expression[1] == expression[1] and not state & expression[2]
    if kind == "and":
        return all(evaluate(child, state) for child in expression[1])
    return any(evaluate(child, state) for child in expression[1])

def required_masks(expression):
    """Literals necessary in every satisfying branch; a safe pruning bound."""
    if isinstance(expression, bool):
        return 0, 0
    if expression[0] == "mask":
        return expression[1:]
    children = [required_masks(child) for child in expression[1]]
    positive, negative = children[0]
    for pos, neg in children[1:]:
        if expression[0] == "and":
            positive, negative = positive | pos, negative | neg
        else:
            positive, negative = positive & pos, negative & neg
    return positive, negative

@dataclass
class _CompiledAction:
    line: str
    args: tuple
    guard: object
    effects: tuple  # (pre-state guard, add mask, delete mask)
    possible_add: int
    possible_delete: int

def effect_predicates(node):
    if not node:
        return set()
    head = node[0]
    if head == "and":
        return set().union(*(effect_predicates(child) for child in node[1:]))
    if head in {"forall", "when"}:
        return effect_predicates(node[2])
    return {node[1][0] if head == "not" else head}

class ReorderModel:
    def __init__(self, domain, problem, plan, *, max_groundings=200000):
        self.domain_path, self.problem_path, self.plan_path = map(Path, (domain, problem, plan))
        domain = parse_sexpr(self.domain_path.read_text())
        problem = parse_sexpr(self.problem_path.read_text())
        for tree, allowed in ((domain, {"domain", ":requirements", ":types", ":constants", ":predicates", ":action"}),
                              (problem, {"problem", ":domain", ":objects", ":init", ":goal"})):
            if not isinstance(tree, list) or tree[:1] != ["define"]:
                raise ValueError("Expected complete PDDL")
            for node in tree[1:]:
                if not isinstance(node, list) or not node or node[0] not in allowed:
                    raise NotImplementedError("Unsupported section: numeric/temporal/derived/trajectory features are not flattened")
        self.parents, constants, self.predicates, schemas = {}, {}, {}, {}
        for node in domain[1:]:
            if node[0] == ":types":
                self.parents = typed_symbols(node[1:])
            elif node[0] == ":constants":
                constants = typed_symbols(node[1:])
            elif node[0] == ":predicates":
                for predicate in node[1:]:
                    if predicate[0] in self.predicates:
                        raise ValueError("Duplicate predicate")
                    self.predicates[predicate[0]] = tuple(typed_symbols(predicate[1:]).values())
            elif node[0] == ":action":
                fields = dict(zip(node[2::2], node[3::2]))
                if len(node) != 8 or set(fields) != {":parameters", ":precondition", ":effect"} or node[1] in schemas:
                    raise ValueError("Malformed/duplicate action")
                schemas[node[1]] = fields
        sections = {}
        for node in problem[1:]:
            if node[0] in sections:
                raise ValueError("Duplicate problem section")
            sections[node[0]] = node[1:]
        if not {":domain", ":objects", ":init", ":goal"} <= sections.keys() or len(sections[":goal"]) != 1:
            raise ValueError("Missing or malformed problem section")
        names = [node[1] for node in domain[1:] if node[:1] == ["domain"] and len(node) == 2]
        if len(names) != 1 or names != sections[":domain"]:
            raise ValueError("Domain/problem mismatch")
        self.objects = typed_symbols(sections[":objects"])
        if set(constants) & self.objects.keys():
            raise ValueError("Duplicate object/constant")
        self.objects.update(constants)
        kinds = {"object", *self.parents}
        if any(kind not in kinds for kind in [*self.parents.values(), *self.objects.values(),
                                             *(t for types in self.predicates.values() for t in types)]):
            raise ValueError("Undeclared type")
        for kind in self.parents:
            self.subtype(kind, "object")  # Also detect cycles, including unused types.
        self.dynamic = set().union(*(effect_predicates(s[":effect"]) for s in schemas.values()))
        self.atoms, self.bits, self.initial_facts = [], {}, set()
        negative = set()
        for node in sections[":init"]:
            negated = node[:1] == ["not"]
            fact = self.atom(node[1] if negated else node, {})
            (negative if negated else self.initial_facts).add(fact)
        if self.initial_facts & negative:
            raise ValueError("Contradictory initial state")
        self.initial = 0
        for fact in sorted(self.initial_facts):
            if fact[0] in self.dynamic:
                self.initial |= self.bit(fact)
        self.grounding_count, self.max_groundings = 0, max_groundings
        self.actions, intern = [], {}
        for name, args in plan_actions(self.plan_path):
            if name not in schemas:
                raise ValueError(f"Unknown action: {name}")
            fields = schemas[name]
            params = typed_symbols(fields[":parameters"])
            if len(args) != len(params):
                raise ValueError(f"Arity mismatch: {name}")
            for arg, kind in zip(args, params.values()):
                if arg not in self.objects or not self.subtype(self.objects[arg], kind):
                    raise ValueError(f"Illegal parameter binding: {name}, {arg}")
            line = "(" + " ".join([name, *args]) + ")"
            if line not in intern:
                bindings = dict(zip(params, args))
                guard = self.condition(fields[":precondition"], bindings)
                effects = tuple(self.effects(fields[":effect"], bindings))
                add = delete = 0
                for _, pos, neg in effects:
                    add |= pos
                    delete |= neg
                intern[line] = _CompiledAction(line, tuple(args), guard, effects, add, delete)
            self.actions.append(intern[line])
        self.goal = self.condition(sections[":goal"][0], {})
        self.goal_required = required_masks(self.goal)
        self.transition_cache, self.activated_effects = {}, {}
        self.before, self.after = [], []
        state = self.initial
        for index, action in enumerate(self.actions):
            self.before.append(state)
            state = self.apply(state, index)
            if state is None:
                raise ValueError(f"Saved reference is inapplicable at step {index + 1}: {action.line}")
            self.after.append(state)
        if not evaluate(self.goal, state):
            raise ValueError("Saved reference does not reach the original goal")

    def subtype(self, actual, required):
        visited = set()
        while actual != "object":
            if actual in visited:
                raise ValueError("Cyclic type hierarchy")
            visited.add(actual)
            if actual == required:
                return True
            actual = self.parents.get(actual, "object")
        return required == "object"

    def atom(self, node, bindings):
        if not node or any(not isinstance(part, str) for part in node):
            raise ValueError("Expected an atomic predicate; functions/numeric terms are unsupported")
        fact = tuple(bindings.get(part, part) for part in node)
        if fact[0] not in self.predicates or len(fact) - 1 != len(self.predicates[fact[0]]):
            raise ValueError(f"Unknown predicate or arity: {fact}")
        for arg, kind in zip(fact[1:], self.predicates[fact[0]]):
            if arg not in self.objects or not self.subtype(self.objects[arg], kind):
                raise ValueError(f"Illegal atom binding: {fact}")
        return fact

    def bit(self, fact):
        if fact not in self.bits:
            self.bits[fact] = 1 << len(self.atoms)
            self.atoms.append(fact)
        return self.bits[fact]

    def quantified(self, declarations, bindings):
        params = typed_symbols(declarations)
        if any(kind not in {"object", *self.parents} for kind in params.values()):
            raise ValueError("Undeclared quantified type")
        values = [tuple(obj for obj in sorted(self.objects) if self.subtype(self.objects[obj], kind)) for kind in params.values()]
        for args in itertools.product(*values):
            self.grounding_count += 1
            if self.grounding_count > self.max_groundings:
                raise NotImplementedError("Quantifier grounding budget exceeded")
            yield {**bindings, **dict(zip(params, args))}

    def condition(self, node, bindings, negate=False):
        if node == []:
            return not negate
        if not isinstance(node, list) or not node:
            raise ValueError("Malformed logical expression")
        head = node[0]
        if head == "not":
            if len(node) != 2:
                raise ValueError("Malformed negation")
            return self.condition(node[1], bindings, not negate)
        if head in {"and", "or"}:
            combine = conjunction if (head == "and") != negate else disjunction
            return combine(self.condition(child, bindings, negate) for child in node[1:])
        if head == "imply":
            if len(node) != 3:
                raise ValueError("Malformed implication")
            return self.condition(["or", ["not", node[1]], node[2]], bindings, negate)
        if head in {"forall", "exists"}:
            if len(node) != 3:
                raise ValueError("Malformed quantifier")
            combine = conjunction if (head == "forall") != negate else disjunction
            return combine(self.condition(node[2], local, negate) for local in self.quantified(node[1], bindings))
        if head == "=":
            if len(node) != 3 or any(isinstance(part, list) for part in node):
                raise NotImplementedError("Only object equality is supported")
            args = [bindings.get(part, part) for part in node[1:]]
            if any(arg not in self.objects for arg in args):
                raise ValueError("Unbound equality term")
            return (args[0] == args[1]) != negate
        fact = self.atom(node, bindings)
        if head not in self.dynamic:
            return (fact in self.initial_facts) != negate
        bit = self.bit(fact)
        return ("mask", 0 if negate else bit, bit if negate else 0)

    def effects(self, node, bindings, guard=True):
        if node == []:
            return
        if not isinstance(node, list) or not node:
            raise ValueError("Malformed effect")
        head = node[0]
        if head == "and":
            for child in node[1:]:
                yield from self.effects(child, bindings, guard)
        elif head == "forall":
            for local in self.quantified(node[1], bindings):
                yield from self.effects(node[2], local, guard)
        elif head == "when":
            local_guard = conjunction((guard, self.condition(node[1], bindings)))
            yield from self.effects(node[2], bindings, local_guard)
        else:
            negated = head == "not"
            bit = self.bit(self.atom(node[1] if negated else node, bindings))
            if guard is not False:
                yield guard, 0 if negated else bit, bit if negated else 0

    def apply(self, state, index):
        action = self.actions[index]
        key = state, action.line
        if key not in self.transition_cache:
            if not evaluate(action.guard, state):
                self.transition_cache[key] = None
            else:
                add = delete = 0
                for guard, pos, neg in action.effects:
                    if evaluate(guard, state):  # Every conditional guard sees the same PRE-state.
                        add |= pos
                        delete |= neg
                self.activated_effects[key] = add, delete
                self.transition_cache[key] = (state & ~delete) | add
        return self.transition_cache[key]

    def activated(self, state, index):
        self.apply(state, index)
        return self.activated_effects[state, self.actions[index].line]

    def facts(self, state):
        return {fact for index, fact in enumerate(self.atoms) if state & (1 << index)}

    def replay(self, order):
        if (not isinstance(order, (list, tuple)) or any(type(i) is not int for i in order)
                or sorted(order) != list(range(1, len(self.actions) + 1))):
            raise ValueError("Reordering must preserve every action occurrence")
        state = self.initial
        for identifier in order:
            state = self.apply(state, identifier - 1)
            if state is None:
                return False
        return evaluate(self.goal, state)

    def order_for_plan(self, path):
        """Recover canonical occurrence IDs without conflating repeated actions."""
        occurrences = defaultdict(deque)
        for identifier, action in enumerate(self.actions, 1):
            occurrences[action.line].append(identifier)
        order = []
        for name, arguments in plan_actions(Path(path)):
            line = "(" + " ".join([name, *arguments]) + ")"
            if not occurrences[line]:
                raise ValueError("Plan does not preserve the reference action multiset")
            order.append(occurrences[line].popleft())
        if not self.replay(order):
            raise ValueError("Plan is not executable to the original goal")
        return order

def expression_bits(expression):
    """All reads, including disjunctions and negative/conditional literals."""
    if isinstance(expression, bool):
        return 0
    if expression[0] == "mask":
        return expression[1] | expression[2]
    result = 0
    for child in expression[1]:
        result |= expression_bits(child)
    return result

@dataclass
class _SearchNode:
    remaining: int
    state: int
    edges: dict = field(default_factory=dict)

def bit_indices(mask):
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length() - 1
        mask ^= lowest

def build_reorder_graph(model, *, max_states=50000, timeout=2.0, completion_bound=False):
    began, n = time.perf_counter(), len(model.actions)
    nodes = [_SearchNode((1 << n) - 1, model.initial)]
    lookup = {(nodes[0].remaining, nodes[0].state): 0}

    def connect(parent, action, state):
        remaining = nodes[parent].remaining & ~(1 << action)
        key = remaining, state
        if key not in lookup:
            lookup[key] = len(nodes)
            nodes.append(_SearchNode(remaining, state))
        nodes[parent].edges[action] = lookup[key]
        return lookup[key]

    # Keep the validated reference path even if an otherwise broad search hits a budget.
    parent = 0
    for index in range(n):
        parent = connect(parent, index, model.after[index])
    earlier_identical = []
    for index, action in enumerate(model.actions):
        earlier_identical.append(sum(1 << j for j in range(index) if model.actions[j].line == action.line))
    remaining_effects = {0: (0, 0)}

    def future_effects(mask):
        if mask not in remaining_effects:
            lowest = mask & -mask
            pos, neg = future_effects(mask ^ lowest)
            action = model.actions[lowest.bit_length() - 1]
            remaining_effects[mask] = pos | action.possible_add, neg | action.possible_delete
        return remaining_effects[mask]

    requirements=[required_masks(a.guard) for a in model.actions]
    impossible=set()
    def can_complete(state,remaining):
        # Monotone closure over POSSIBLE Boolean values. A fact may become
        # both possibly true and false. Only necessary guard literals are
        # required; conditional effects are optimistically all available.
        # Every real suffix is included by induction on its actual actions.
        # Stalling therefore proves a dead prefix, including circular support
        # which the old one-action-at-a-time availability bound missed.
        positive, negative = state, ((1 << len(model.atoms))-1) ^ state
        pending = list(bit_indices(remaining))
        while pending:
            waiting = []
            for i in pending:
                need_pos, need_neg = requirements[i]
                if need_pos & ~positive or need_neg & ~negative:
                    waiting.append(i)
                else:
                    positive |= model.actions[i].possible_add
                    negative |= model.actions[i].possible_delete
            if len(waiting) == len(pending):
                return False
            pending = waiting
        return True
    cursor, goal_pruned, completion_pruned, stop = 0, 0, 0, "complete"
    while cursor < len(nodes):
        if time.perf_counter() - began >= timeout:
            stop = "time_budget_exhausted"
            break
        node = nodes[cursor]
        for index, state in _successors(node.state, node.remaining, model.apply):
            if earlier_identical[index] & node.remaining:
                continue
            remaining = node.remaining & ~(1 << index)
            pos, neg = future_effects(remaining)
            if model.goal_required[0] & ~(state | pos) or model.goal_required[1] & state & ~neg:
                goal_pruned += 1
                continue
            key=remaining,state
            if completion_bound and key not in lookup:
                if key in impossible or not can_complete(state,remaining):
                    impossible.add(key)
                    completion_pruned+=1
                    continue
            if (remaining, state) not in lookup and len(nodes) >= max_states:
                stop = "state_budget_exhausted"
                break
            connect(cursor, index, state)
        if stop != "complete":
            break
        cursor += 1
    return nodes, {"status": stop, "nodes": len(nodes), "expanded_nodes": cursor,
                   "edges": sum(len(node.edges) for node in nodes), "goal_bound_pruned_edges": goal_pruned,
                   "completion_bound_pruned_edges":completion_pruned,
                   "completion_bound_kind":"monotone_necessary_literal_closure" if completion_bound else "disabled",
                   "transition_cache_entries": len(model.transition_cache), "seconds": time.perf_counter() - began}


def _search_classes(model, observer, max_states, max_histories, timeout):
    """Count complete paths while retaining their per-occurrence observations.

    A state/remaining-action node can share suffixes. Prefixes at that node
    merge only if their observation histories also agree. Budgets produce an
    explicitly incomplete result, never a certificate of soundness.
    """
    began = time.perf_counter()
    nodes, graph = build_reorder_graph(
        model, max_states=max_states, timeout=min(timeout * .25, 15), completion_bound=True
    )
    n = len(model.actions)
    layers = defaultdict(list)
    for i, node in enumerate(nodes):
        layers[n - node.remaining.bit_count()].append(i)
    suffix_count = [0] * len(nodes)
    for depth in range(n, -1, -1):
        for parent in layers[depth]:
            node = nodes[parent]
            suffix_count[parent] = (int(evaluate(model.goal, node.state)) if not node.remaining
                                    else sum(suffix_count[child] for child in node.edges.values()))
    raw_count = suffix_count[0]
    # A common fast case: every action sees the same state in all live paths.
    uniform = graph['status'] == 'complete' and max_histories >= n + 1
    uniform_tokens = [0] * n
    finals = set()
    for depth in range(n, -1, -1):
        for parent in layers[depth]:
            node = nodes[parent]
            if uniform and suffix_count[parent]:
                if not node.remaining:
                    finals.add(node.state)
                for index, child in node.edges.items():
                    if not suffix_count[child]:
                        continue
                    _, token = observer.event(node.state, index + 1)
                    if uniform_tokens[index] and uniform_tokens[index] != token:
                        uniform = False
                        break
                    uniform_tokens[index] = token
    terminal = {}
    expanded, created, merges, status = 0, 1, 0, 'complete'
    if uniform and len(finals) == 1 and time.perf_counter() - began < timeout:
        assert tuple(uniform_tokens) == observer.reference[1]
        terminal[observer.reference] = [raw_count, tuple(range(1, n + 1))]
    else:
        width = (len(observer.events) + max_histories * n + 1).bit_length()
        field = (1 << width) - 1
        path_width = (n + 1).bit_length()
        path_field = (1 << path_width) - 1

        def decode(code):
            order = []
            for _ in range(n):
                order.append(code & path_field)
                code >>= path_width
            return tuple(reversed(order))

        labels = defaultdict(dict)
        labels[0][0] = [1, 0]
        for depth in range(n + 1):
            for parent in layers[depth]:
                node = nodes[parent]
                edges = sorted(node.edges.items())
                for history, (count, prefix) in labels[parent].items():
                    if expanded >= max_histories or time.perf_counter() - began >= timeout:
                        status = 'history_budget_exhausted' if expanded >= max_histories else 'time_budget_exhausted'
                        break
                    expanded += 1
                    if depth == n:
                        if evaluate(model.goal, node.state):
                            tokens = tuple((history >> (i * width)) & field for i in range(n))
                            terminal[(node.state, tokens, ())] = [count, decode(prefix)]
                        continue
                    for index, child in edges:
                        if not suffix_count[child]:
                            continue
                        _, token = observer.event(node.state, index + 1)
                        following = history | (token << (index * width))
                        path = (prefix << path_width) | (index + 1)
                        if following in labels[child]:
                            labels[child][following][0] += count
                            merges += 1
                        else:
                            if created >= max_histories:
                                status = 'history_budget_exhausted'
                                break
                            labels[child][following] = [count, path]
                            created += 1
                    if status != 'complete':
                        break
                if status != 'complete':
                    break
                labels.pop(parent, None)
            if status != 'complete':
                break
    if graph['status'] != 'complete' and status == 'complete':
        status = 'base_graph_incomplete'
    terminal.setdefault(observer.reference, [1, tuple(range(1, n + 1))])
    representatives, classes = [], []
    for signature, (count, order) in terminal.items():
        reference = signature == observer.reference
        candidate = {'id': 'reference' if reference else f'cf_{len(representatives) + 1}',
                     'order': list(range(1, n + 1)) if reference else list(order)}
        representatives.append(candidate)
        classes.append({'representative_id': candidate['id'], 'count': count,
                        'final_state': signature[0], 'events': list(signature[1]), 'merge_eligible': True})
    accounted = sum(c['count'] for c in classes)
    if status == 'complete' and accounted != raw_count:
        raise AssertionError('Complete reordering search lost or duplicated a path')
    return representatives, {
        'coverage_complete': status == 'complete', 'status': status,
        'raw_orders_including_reference': raw_count, 'raw_order_count_exact': graph['status'] == 'complete',
        'accounted_orders': accounted, 'classes': classes, 'class_count': len(classes),
        'base_graph': graph, 'created_histories': created, 'expanded_histories': expanded,
        'prefix_merges': merges, 'seconds': time.perf_counter() - began,
    }


def _dependency_components(model, reads, writes):
    remaining, components = set(range(len(model.actions))), []
    while remaining:
        todo = [min(remaining)]
        remaining.remove(todo[0])
        component = []
        while todo:
            i = todo.pop()
            component.append(i + 1)
            adjacent = {j for j in remaining if writes[i] & (reads[j] | writes[j])
                        or writes[j] & reads[i] or model.actions[i].line == model.actions[j].line}
            remaining.difference_update(adjacent)
            todo.extend(sorted(adjacent, reverse=True))
        components.append(sorted(component))
    return components


def search_plan_classes(model, observer, *, max_states=300000, max_histories=300000, timeout=120):
    """Complete fixed-multiset reordering, factoring only proved independence.

    Unlike the preferred-plan query, this query fixes no reference supporter.
    Reads include observation selectors, so moving containers/resources cannot
    invalidate the proof that two components are independent.
    """
    if max_states < 1 or max_histories < 1 or timeout < 0:
        raise ValueError('Invalid reordering search budget')
    began = time.perf_counter()
    writes = [a.possible_add | a.possible_delete for a in model.actions]
    components = _dependency_components(model, observer.guards, writes)

    def fallback(reason):
        representatives, proof = _search_classes(
            model, observer, max_states, max_histories, max(0, timeout - (time.perf_counter() - began))
        )
        proof['factorization'] = {'applied': False, 'reason': reason, 'components': components}
        return representatives, proof

    if len(components) < 2:
        return fallback('one_dependency_component')
    components = _dependency_components(model, observer.process.reads, writes)
    if len(components) < 2:
        return fallback('observation_selectors_couple_components')
    parts, states_used, histories_used = [], 0, 0
    for identifiers in components:
        if time.perf_counter() - began >= timeout or states_used >= max_states or histories_used >= max_histories:
            return fallback('factorization_budget_exhausted')
        local = copy(model)
        local.actions = [model.actions[i - 1] for i in identifiers]
        local.goal, local.goal_required = True, (0, 0)
        local.transition_cache, local.activated_effects = {}, {}
        local.before, local.after = [], []
        state = local.initial
        for index in range(len(local.actions)):
            local.before.append(state)
            state = local.apply(state, index)
            if state is None:
                raise AssertionError('Independent component became inapplicable')
            local.after.append(state)
        restricted = copy(observer)
        restricted.model = local
        restricted.process = observer.process.restrict(identifiers, local)
        restricted.guards = [observer.guards[i - 1] for i in identifiers]
        restricted.cache, restricted.tokens, restricted.events = {}, {}, {}
        restricted.reference = restricted.profile(list(range(1, len(identifiers) + 1)))
        reps, proof = _search_classes(local, restricted, max_states - states_used,
                                     max_histories - histories_used, max(0, timeout - (time.perf_counter() - began)))
        if not proof['coverage_complete']:
            return fallback('component_search_incomplete')
        states_used += proof['base_graph']['nodes']
        histories_used += proof['created_histories']
        parts.append((identifiers, reps, proof))
    combinations = math.prod(len(reps) for _, reps, _ in parts)
    if combinations + histories_used > max_histories:
        return fallback('class_product_budget_exhausted')
    interleavings = math.factorial(len(model.actions))
    for identifiers in components:
        interleavings //= math.factorial(len(identifiers))
    choices = [list(zip(reps, proof['classes'])) for _, reps, proof in parts]
    terminal = {}
    for choice in itertools.product(*choices):
        if time.perf_counter() - began >= timeout:
            return fallback('class_product_time_budget_exhausted')
        order = [part[0][i - 1] for part, (candidate, _) in zip(parts, choice) for i in candidate['order']]
        state = model.initial
        for identifier in order:
            state = model.apply(state, identifier - 1)
            if state is None:
                raise AssertionError('Independent component product became inapplicable')
        if not evaluate(model.goal, state):
            continue
        signature = observer.profile(order)
        count = interleavings * math.prod(row['count'] for _, row in choice)
        if signature in terminal:
            terminal[signature][0] += count
        else:
            terminal[signature] = [count, order]
    if observer.reference not in terminal:
        raise AssertionError('Independent component product lost the reference')
    representatives, classes = [], []
    for signature, (count, order) in terminal.items():
        reference = signature == observer.reference
        candidate = {'id': 'reference' if reference else f'cf_{len(representatives) + 1}',
                     'order': list(range(1, len(model.actions) + 1)) if reference else order}
        representatives.append(candidate)
        classes.append({'representative_id': candidate['id'], 'count': count,
                        'final_state': signature[0], 'events': list(signature[1]), 'merge_eligible': True})
    count = sum(row['count'] for row in classes)
    return representatives, {
        'coverage_complete': True, 'status': 'complete', 'raw_orders_including_reference': count,
        'raw_order_count_exact': True, 'accounted_orders': count, 'class_count': len(classes), 'classes': classes,
        'base_graph': {'nodes': states_used, 'edges': sum(p['base_graph']['edges'] for _, _, p in parts)},
        'created_histories': histories_used + combinations,
        'factorization': {'applied': True, 'components': components,
                          'interleavings_per_combination': interleavings, 'class_products': combinations},
        'seconds': time.perf_counter() - began,
    }


# 全绑定等长模型：参考计划只确定步数，不限制动作组成。
_FD_LOCK = threading.Lock()


def _reachable_bindings(domain: Path, problem: Path):
    """FD's delete relaxation is a superset of executable action bindings.

    Read the binding table, not the SAS task: SAS relevance simplification and
    dropping no-ops preserve solvability but need not preserve exact-L traces.
    Compile the original PDDL afterward, so normalized conditional effects and
    disjunctive variants never become extra externally visible operations.
    """
    source = Path(__file__).resolve().parents[3] / "downward" / "src"
    if not source.is_dir():
        raise FileNotFoundError("Fast Downward source is required at downward/src")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from translate import instantiate, normalize, options, pddl_parser
    # FD's translator owns process-global options. The pipeline runs episodes
    # in threads, so only this short instantiation section is serialized.
    with _FD_LOCK, redirect_stdout(io.StringIO()):
        options.set_options([str(domain), str(problem), "--keep-no-ops"])
        task = pddl_parser.open(str(domain), str(problem))
        normalize.normalize(task)
        _, _, _, _, _, parameters = instantiate.explore(task)
    bindings = set()
    for schema, values in parameters.items():
        count = schema.num_external_parameters
        for arguments in values:
            bindings.add((schema.name, tuple(arguments[:count])))
    return sorted(bindings)


class SameLengthModel(ReorderModel):
    """Original PDDL semantics over all FD-reachable original bindings."""

    def __init__(self, domain, problem, plan, *, max_groundings=200000):
        began = time.perf_counter()
        super().__init__(domain, problem, plan, max_groundings=max_groundings)
        reference_lines = [a.line for a in self.actions]
        self.length = len(reference_lines)
        tree = parse_sexpr(self.domain_path.read_text())
        schemas = {node[1]: dict(zip(node[2::2], node[3::2]))
                   for node in tree[1:] if node[:1] == [":action"]}
        ground = _reachable_bindings(self.domain_path, self.problem_path)
        self.actions = []
        for name, arguments in ground:
            fields = schemas[name]
            parameters = typed_symbols(fields[":parameters"])
            bindings = dict(zip(parameters, arguments))
            guard = self.condition(fields[":precondition"], bindings)
            if guard is False:
                continue
            effects = tuple(self.effects(fields[":effect"], bindings))
            add = delete = 0
            for _, pos, neg in effects:
                add |= pos
                delete |= neg
            line = "(" + " ".join((name, *arguments)) + ")"
            self.actions.append(_CompiledAction(line, arguments, guard, effects, add, delete))
        identifiers = {a.line: i for i, a in enumerate(self.actions, 1)}
        if any(line not in identifiers for line in reference_lines):
            raise ValueError("FD relaxed reachability omitted an executable reference binding")
        self.reference_order = [identifiers[line] for line in reference_lines]
        self.transition_cache, self.activated_effects = {}, {}
        self.before, self.after = [], []
        state = self.initial
        for identifier in self.reference_order:
            self.before.append(state)
            state = self.apply(state, identifier - 1)
            self.after.append(state)
        self.initialization_seconds = time.perf_counter() - began
        self.ground_actions = len(self.actions)
        requirements = [required_masks(a.guard) for a in self.actions]
        frequency = Counter(bit for positive, _ in requirements for bit in _bits(positive))
        self.successor_buckets, self.unanchored = defaultdict(list), []
        self.successor_cache = {}
        for index, (positive, negative) in enumerate(requirements):
            row = index, positive, negative
            if positive:
                anchor = min(_bits(positive), key=lambda bit: (frequency[bit], bit))
                self.successor_buckets[anchor].append(row)
            else:
                self.unanchored.append(row)

    def successors(self, state):
        """Index one necessary positive guard literal, then check the full guard."""
        if state in self.successor_cache:
            return self.successor_cache[state]
        rows = list(self.unanchored)
        for bit in _bits(state):
            rows.extend(self.successor_buckets.get(bit, ()))
        result = []
        for index, positive, negative in rows:
            if state & positive == positive and not state & negative:
                after = self.apply(state, index)
                if after is not None:
                    result.append((index, after))
        self.successor_cache[state] = tuple(result)
        return self.successor_cache[state]

    def replay(self, order):
        if (not isinstance(order, (list, tuple)) or len(order) != self.length
                or any(type(i) is not int or not 1 <= i <= len(self.actions) for i in order)):
            raise ValueError("Expected an exact-reference-length sequence of action IDs")
        state = self.initial
        for identifier in order:
            state = self.apply(state, identifier - 1)
            if state is None:
                return False
        return evaluate(self.goal, state)

    def order_for_plan(self, path):
        identifiers = {a.line: i for i, a in enumerate(self.actions, 1)}
        order = []
        for name, arguments in plan_actions(Path(path)):
            line = "(" + " ".join((name, *arguments)) + ")"
            if line not in identifiers:
                raise ValueError("Unknown or unreachable action binding: " + line)
            order.append(identifiers[line])
        if not self.replay(order):
            raise ValueError("Plan is not executable to the original goal at length L")
        return order


def _bits(mask):
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit


# 事件语言与计数语言共享后缀；不展开全部原始计划。
class _Language:
    def __init__(self, max_nodes, deadline):
        self.nodes = [()]
        self.counts = [1]
        self.interned = {(): 0}
        self.union_cache = {}
        self.max_nodes, self.deadline = max_nodes, deadline

    def node(self, edges):
        edges = tuple(sorted(edges))
        if edges not in self.interned:
            if len(self.nodes) >= self.max_nodes:
                raise OverflowError("Class-language node budget exhausted")
            if time.perf_counter() >= self.deadline:
                raise TimeoutError("Class-language time budget exhausted")
            self.interned[edges] = len(self.nodes)
            self.nodes.append(edges)
            self.counts.append(sum(self.counts[child] for _, child in edges))
        return self.interned[edges]

    def _known_union(self, a, b):
        if a == b:
            return a
        if a < 0:
            return b
        if b < 0:
            return a
        if a > b:
            a, b = b, a
        if a == 0:
            raise ValueError("Event words must have a distinct final-state marker")
        return self.union_cache.get((a, b))

    def union(self, a, b):
        known = self._known_union(a, b)
        if known is not None:
            return known
        root = (min(a, b), max(a, b))
        stack = [(*root, False)]
        while stack:
            left_id, right_id, expanded = stack.pop()
            if self._known_union(left_id, right_id) is not None:
                continue
            left_id, right_id = min(left_id, right_id), max(left_id, right_id)
            left = dict(self.nodes[left_id])
            if not expanded:
                stack.append((left_id, right_id, True))
                for token, child in self.nodes[right_id]:
                    if token in left and self._known_union(left[token], child) is None:
                        stack.append((left[token], child, False))
                continue
            for token, child in self.nodes[right_id]:
                left[token] = self._known_union(left[token], child) if token in left else child
            self.union_cache[left_id, right_id] = self.node(left.items())
        return self.union_cache[root]

    def count(self, node):
        return 0 if node < 0 else self.counts[node]

    def word(self, node, rank):
        word = []
        while node:
            for token, child in self.nodes[node]:
                count = self.count(child)
                if rank < count:
                    word.append(token)
                    node = child
                    break
                rank -= count
            else:
                raise IndexError("Class rank out of bounds")
        return tuple(word)

    def rank(self, node, word):
        rank = 0
        for expected in word:
            for token, child in self.nodes[node]:
                if token == expected:
                    node = child
                    break
                rank += self.count(child)
            else:
                raise ValueError("Word absent from class language")
        if node:
            raise ValueError("Truncated class word")
        return rank

    def contains(self, node, word, start):
        for expected in word[start:]:
            node = next((child for token, child in self.nodes[node] if token == expected), -1)
            if node < 0:
                return False
        return node == 0


class EventLanguageCover(Sequence):
    """Exact, rankable representatives; iteration performs no Judge calls."""

    encoding = "ordered_event_language_dag_v2"

    def __init__(self, model, observer, states, edges, tokens, roots, language, finals, common,
                 token_labels):
        self.model, self.observer = model, observer
        self.states, self.edges, self.tokens = states, edges, tokens
        self.roots, self.language, self.finals = roots, language, finals
        self.token_labels = token_labels
        self.event_keys = {label: observer.events[token] for token, label in token_labels.items()}
        self.common_channels = tuple(sorted(common))
        self.count = language.count(roots[0])
        self.class_count = self.count
        self.representative_count = self.count
        self.reference_class_id = "reference"
        self.action_ids = tuple(sorted({identifier for parent, identifier in tokens}))
        state, word = model.initial, []
        for identifier in model.reference_order:
            state, token = observer.event(state, identifier)
            word.append(token_labels[token])
        word.append(finals[observer.finish(state, observer.initial_profile())[0]])
        self.reference_rank = language.rank(roots[0], tuple(word))

    def __len__(self):
        # Python's len() is limited to Py_ssize_t; class_count is an arbitrary
        # precision integer and is the coverage/streaming interface.
        return self.count

    def __iter__(self):
        for index in range(self.count):
            yield self[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.count))]
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError("Class rank out of bounds")
        if index == self.reference_rank:
            return {"id": "reference", "order": list(self.model.reference_order)}
        word = self.language.word(self.roots[0], index)
        parent, order = 0, []
        for position, token in enumerate(word[:-1]):
            for identifier, child in sorted(self.edges[parent]):
                if (self.tokens.get((parent, identifier)) == token
                        and self.roots[child] >= 0
                        and self.language.contains(self.roots[child], word, position + 1)):
                    order.append(identifier)
                    parent = child
                    break
            else:
                raise AssertionError("Class word lost its original bound-action realization")
        return {"id": f"cf_{index + 1}", "order": order}

    def class_count_for(self, index):
        """Number of original bound-action paths in one represented class."""
        word = self.language.word(self.roots[0], index)
        counts = {0: 1}
        for token in word[:-1]:
            following = {}
            for parent, count in counts.items():
                for identifier, child in self.edges[parent]:
                    if self.tokens.get((parent, identifier)) == token:
                        following[child] = following.get(child, 0) + count
            counts = following
        total = 0
        for parent, count in counts.items():
            final = self.observer.finish(self.states[parent], self.observer.initial_profile())[0]
            if self.finals[final] == word[-1]:
                total += count
        return total

    def metadata(self):
        return {"encoding": self.encoding, "count": self.count,
                "language_nodes": len(self.language.nodes),
                "common_channels": list(self.common_channels),
                "reference_rank": self.reference_rank,
                "representatives_materialized": False}


def build_event_language_cover(model, observer, states, edges, layers, suffix,
                              *, max_nodes, deadline):
    """Encode complete event words when the observer records every occurrence."""
    if not hasattr(observer, "token_channels") or not hasattr(observer, "channels"):
        return None
    common = set(range(len(observer.channels)))
    tokens = {}
    for layer in layers[:-1]:
        for parent in layer:
            if not suffix[parent]:
                continue
            for identifier, child in edges[parent]:
                if not suffix[child]:
                    continue
                _, token = observer.event(states[parent], identifier)
                tokens[parent, identifier] = token
                common.intersection_update(observer.token_channels[token])
                if not common:
                    return None
    # Stable semantic labels and lexicographic original bindings keep ranks and
    # representatives identical across Python/native graph traversal orders.
    live_tokens = set(tokens.values())
    token_labels = {token: i for i, token in enumerate(
        sorted(live_tokens, key=observer.events.__getitem__), 1)}
    tokens = {edge: token_labels[token] for edge, token in tokens.items()}
    language = _Language(max_nodes, deadline)
    roots = [-1] * len(states)
    terminal_keys = {parent: observer.finish(states[parent], observer.initial_profile())[0]
                     for parent in layers[-1] if suffix[parent]}
    finals = {final: -i - 1 for i, final in enumerate(sorted(set(terminal_keys.values())))}
    for parent in layers[-1]:
        if suffix[parent]:
            roots[parent] = language.node(((finals[terminal_keys[parent]], 0),))
    for layer in reversed(layers[:-1]):
        for parent in layer:
            grouped = {}
            for identifier, child in edges[parent]:
                if suffix[child]:
                    token = tokens[parent, identifier]
                    grouped[token] = language.union(grouped.get(token, -1), roots[child])
            if grouped:
                roots[parent] = language.node(grouped.items())
    return EventLanguageCover(model, observer, states, edges, tokens, roots, language, finals, common,
                              token_labels)


class _CountLanguage(_Language):
    """A canonical trie DAG for fixed-length sorted multisets."""

    def __init__(self, max_nodes, deadline):
        super().__init__(max_nodes, deadline)
        self.insert_cache = {}
        self.insert_calls = self.union_calls = 0

    def node(self, edges):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("Count-language time budget exhausted")
        return super().node(edges)

    def union(self, a, b):
        self.union_calls += 1
        if self.union_calls % 1024 == 0 and time.perf_counter() >= self.deadline:
            raise TimeoutError("Context-count language time budget exhausted")
        return super().union(a, b)

    def insert(self, token, root):
        """Insert one occurrence into every sorted word in root's language."""
        if root < 0:
            return -1
        self.insert_calls += 1
        key = token, root
        known = self.insert_cache.get(key)
        if known is not None:
            return known
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if (token, node) in self.insert_cache:
                continue
            if time.perf_counter() >= self.deadline:
                raise TimeoutError("Context-count language insertion time budget exhausted")
            if node == 0:
                self.insert_cache[token, node] = self.node(((token, 0),))
                continue
            edges = self.nodes[node]
            if not expanded:
                stack.append((node, True))
                stack.extend((child, False) for label, child in edges
                             if label < token and (token, child) not in self.insert_cache)
                continue
            lower = [(label, self.insert_cache[token, child])
                     for label, child in edges if label < token]
            upper = tuple((label, child) for label, child in edges if label >= token)
            if upper:
                # Every upper word receives token before its original head.
                # Lower heads retain their order and recurse into the tail.
                lower.append((token, self.node(upper)))
            self.insert_cache[token, node] = self.node(lower)
        return self.insert_cache[key]


def _topological(edges, deadline=math.inf):
    incoming = [0] * len(edges)
    for outgoing in edges:
        for _, child in outgoing:
            incoming[child] += 1
    queue = deque(i for i, count in enumerate(incoming) if not count)
    order = []
    while queue:
        if len(order) % 256 == 0 and time.perf_counter() >= deadline:
            raise TimeoutError("Count graph traversal time budget exhausted")
        parent = queue.popleft()
        order.append(parent)
        for _, child in edges[parent]:
            incoming[child] -= 1
            if not incoming[child]:
                queue.append(child)
    if len(order) != len(edges):
        raise ValueError("Context cover requires an acyclic layered graph")
    return order


def _reorder_events(ordered, deadline):
    """Change only internal event variable order, preserving semantic records.

    Cyclic physical regions are ordered topologically.  Within each region,
    minimum-degree elimination keeps physically connected transitions near one
    another.  This is a representation optimization, never a class merge.
    """
    began = time.perf_counter()
    physical = list(dict.fromkeys(ordered.states))
    indices = {state: i for i, state in enumerate(physical)}
    forward, reverse = [set() for _ in physical], [set() for _ in physical]
    links = {}
    for parent, outgoing in enumerate(ordered.edges):
        if parent % 256 == 0 and time.perf_counter() >= deadline:
            raise TimeoutError("Count variable ordering time budget exhausted")
        for identifier, child in outgoing:
            # The search graph can retain prefixes with no exact-L goal
            # suffix. Only live edges belong to the event language.
            if ordered.roots[parent] < 0 or ordered.roots[child] < 0:
                continue
            source, target = indices[ordered.states[parent]], indices[ordered.states[child]]
            forward[source].add(target)
            reverse[target].add(source)
            token = ordered.tokens[parent, identifier]
            links.setdefault(token, set()).add((source, target))
    visited, finished = set(), []
    for seed in range(len(physical)):
        if seed in visited:
            continue
        stack = [(seed, False)]
        while stack:
            vertex, expanded = stack.pop()
            if expanded:
                finished.append(vertex)
                continue
            if vertex in visited:
                continue
            visited.add(vertex)
            stack.append((vertex, True))
            stack.extend((child, False) for child in forward[vertex] if child not in visited)
    components, members = {}, []
    for seed in reversed(finished):
        if seed in components:
            continue
        index = len(members)
        members.append([])
        stack = [seed]
        components[seed] = index
        while stack:
            vertex = stack.pop()
            members[index].append(vertex)
            for child in reverse[vertex]:
                if child not in components:
                    components[child] = index
                    stack.append(child)
    maximum_degree = 0
    elimination = []
    for group in members:
        vertices = set(group)
        neighbors = {vertex: (forward[vertex] | reverse[vertex]) & vertices - {vertex}
                     for vertex in group}
        queue = [(len(adjacent), vertex) for vertex, adjacent in neighbors.items()]
        heapq.heapify(queue)
        while queue:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Count variable elimination time budget exhausted")
            degree, vertex = heapq.heappop(queue)
            if vertex not in neighbors or degree != len(neighbors[vertex]):
                continue
            maximum_degree = max(maximum_degree, degree)
            adjacent = neighbors.pop(vertex)
            elimination.append(vertex)
            for other in adjacent:
                neighbors[other].remove(vertex)
                neighbors[other].update(adjacent - {other})
                heapq.heappush(queue, (len(neighbors[other]), other))
    ranks = {vertex: rank for rank, vertex in enumerate(elimination)}

    def ordering(token):
        return (min(max(ranks[source], ranks[target]) for source, target in links[token]), token)
    mapping = {token: index for index, token in enumerate(sorted(links, key=ordering), 1)}
    result = copy(ordered)
    result.tokens = {edge: mapping[token] for edge, token in ordered.tokens.items()}
    result.event_keys = {mapping[token]: event for token, event in ordered.event_keys.items()}
    # Keep the returned ordered cover coherent as well: its old ranks can
    # change under the new alphabet order, but its words and bound paths do not.
    result.token_labels = {token: mapping[label] for token, label in ordered.token_labels.items()}
    source_language = ordered.language
    renamed_language = _Language(source_language.max_nodes, deadline)
    renamed_language.nodes = [tuple(sorted((mapping.get(label, label), child) for label, child in edges))
                              for edges in source_language.nodes]
    renamed_language.counts = list(source_language.counts)
    renamed_language.interned = {edges: node for node, edges in enumerate(renamed_language.nodes)}
    result.language = renamed_language
    reference_word = source_language.word(ordered.roots[0], ordered.reference_rank)
    result.reference_rank = renamed_language.rank(ordered.roots[0],
                                                  tuple(mapping.get(label, label) for label in reference_word))
    result.context_event_order = {"strategy": "scc_min_degree", "seconds": time.perf_counter() - began,
                                  "physical_states": len(physical), "physical_sccs": len(members),
                                  "maximum_elimination_degree": maximum_degree}
    return result


class _ClassPaths:
    def __init__(self, cover, rank, timeout, max_states):
        self.cover, self.rank = cover, rank
        self.deadline = time.perf_counter() + timeout
        signature = cover.signature(rank)
        if not signature or signature[0] >= 0 or any(x <= 0 for x in signature[1:]):
            raise AssertionError("A class must contain one final marker and positive events")
        self.final, self.remaining = signature[0], signature[1:]
        self.memo = {}
        self.started_states = 0
        self.max_states = max_states

    def count(self, parent, remaining):
        requested = parent, remaining
        stack = [(parent, remaining, False)]
        while stack:
            if time.perf_counter() >= self.deadline:
                raise TimeoutError("Class member query time budget exhausted")
            parent, remaining, expanded = stack.pop()
            key = parent, remaining
            if key in self.memo:
                continue
            if not expanded:
                if self.started_states >= self.max_states:
                    raise OverflowError("Class member query state budget exhausted")
                self.started_states += 1
                root = self.cover.roots[parent]
                if root < 0 or not self.cover.language.contains(root, (self.final, *remaining), 0):
                    self.memo[key] = 0
                    continue
                if not remaining:
                    self.memo[key] = int(self.cover.terminal_markers.get(parent) == self.final)
                    continue
            children = []
            for identifier, child in self.cover.edges[parent]:
                token = self.cover.tokens.get((parent, identifier))
                position = bisect_left(remaining, token) if token is not None else len(remaining)
                if position < len(remaining) and remaining[position] == token:
                    children.append((child, remaining[:position] + remaining[position + 1:]))
            if expanded:
                self.memo[key] = sum(self.memo[child] for child in children)
            else:
                stack.append((parent, remaining, True))
                stack.extend((child, after, False) for child, after in children
                             if (child, after) not in self.memo)
        return self.memo[requested]

    def path(self, rank):
        total = self.count(0, self.remaining)
        if not 0 <= rank < total:
            raise IndexError("Class member rank out of bounds")
        parent, remaining, order = 0, self.remaining, []
        while remaining:
            for identifier, child in sorted(self.cover.edges[parent]):
                token = self.cover.tokens.get((parent, identifier))
                position = bisect_left(remaining, token) if token is not None else len(remaining)
                if position >= len(remaining) or remaining[position] != token:
                    continue
                after = remaining[:position] + remaining[position + 1:]
                count = self.count(child, after)
                if rank < count:
                    order.append(identifier)
                    parent, remaining = child, after
                    break
                rank -= count
            else:
                raise AssertionError("Class signature lost its original binding witness")
        if self.cover.terminal_markers.get(parent) != self.final:
            raise AssertionError("Class witness ended in a different canonical final state")
        return order


class CountCover(Sequence):
    """Rankable exact event-count classes and original-binding member queries."""

    encoding = "context_count_language_dag_v1"

    def __init__(self, ordered, language, roots, terminal_markers, raw_suffix,
                 began, query_timeout=120, query_max_states=1000000, resources=None,
                 construction_deadline=math.inf):
        self._resources, self._closed = resources, False
        self.ordered = ordered
        self.model, self.observer = ordered.model, ordered.observer
        self.states, self.edges, self.tokens = ordered.states, ordered.edges, ordered.tokens
        self.event_keys, self.finals = ordered.event_keys, ordered.finals
        self.language, self.roots, self.terminal_markers = language, roots, terminal_markers
        self.raw_suffix = raw_suffix
        self.class_count = self.representative_count = self.count = language.count(roots[0])
        self.raw_count = raw_suffix[0]
        self.action_ids = ordered.action_ids
        self.query_timeout, self.query_max_states = query_timeout, query_max_states
        self.reference_class_id = "reference"
        reference = self._signature_for_order(self.model.reference_order, construction_deadline)
        self.reference_rank = self.language.rank(self.roots[0], reference)
        self.construction_seconds = time.perf_counter() - began
        self._query_cache = OrderedDict()
        self.backend = "python"
        self.backend_metrics = {}

    def __len__(self):
        return self.class_count

    def __iter__(self):
        for rank in range(self.class_count):
            yield self[rank]

    def signature(self, rank):
        if self._closed:
            raise ValueError("Count cover is closed")
        if not 0 <= rank < self.class_count:
            raise IndexError("Context class rank out of bounds")
        deadline = time.perf_counter() + self.query_timeout
        word = self.language.word(self.roots[0], rank)
        if time.perf_counter() >= deadline:
            raise TimeoutError("Class signature query time budget exhausted")
        return word

    def signature_for_order(self, order):
        return self._signature_for_order(order, time.perf_counter() + self.query_timeout)

    def _signature_for_order(self, order, deadline):
        if self._closed:
            raise ValueError("Count cover is closed")
        if (not isinstance(order, (tuple, list)) or len(order) != self.model.length
                or any(type(i) is not int or i < 1 for i in order)):
            raise ValueError("Expected an exact-reference-length original binding path")
        parent, tokens = 0, []
        for identifier in order:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Class membership query time budget exhausted")
            matches = [(op, child) for op, child in self.edges[parent] if op == identifier]
            if len(matches) != 1:
                raise ValueError("Path is not in the complete original-binding graph")
            tokens.append(self.tokens[parent, identifier])
            parent = matches[0][1]
        if parent not in self.terminal_markers:
            raise ValueError("Path does not end at an exact-length goal")
        return tuple(sorted((self.terminal_markers[parent], *tokens)))

    def rank_for_order(self, order):
        deadline = time.perf_counter() + self.query_timeout
        rank = self.language.rank(self.roots[0], self._signature_for_order(order, deadline))
        if time.perf_counter() >= deadline:
            raise TimeoutError("Class rank query time budget exhausted")
        return rank

    def _query(self, rank):
        deadline = time.perf_counter() + self.query_timeout
        if rank not in self._query_cache:
            self._query_cache[rank] = _ClassPaths(self, rank, self.query_timeout, self.query_max_states)
            if len(self._query_cache) > 8:
                self._query_cache.popitem(last=False)
        self._query_cache.move_to_end(rank)
        query = self._query_cache[rank]
        # A cached query can be used long after construction; each public call
        # receives a fresh bounded deadline while retaining proven counts.
        query.deadline = deadline
        if time.perf_counter() >= deadline:
            raise TimeoutError("Class member query preparation time budget exhausted")
        return query

    def __getitem__(self, rank):
        if self._closed:
            raise ValueError("Count cover is closed")
        if isinstance(rank, slice):
            return [self[i] for i in range(*rank.indices(self.class_count))]
        if rank < 0:
            rank += self.class_count
        if not 0 <= rank < self.class_count:
            raise IndexError("Context class rank out of bounds")
        if rank == self.reference_rank:
            return {"id": "reference", "order": list(self.model.reference_order)}
        return {"id": f"ctx_{rank + 1}", "order": self._query(rank).path(0)}

    def class_count_for(self, rank):
        query = self._query(rank)
        return query.count(0, query.remaining)

    def witnesses(self, rank, count=3):
        if type(count) is not int or count < 1:
            raise ValueError("Witness count must be positive")
        query = self._query(rank)
        total = query.count(0, query.remaining)
        count = min(count, total)
        ranks = [i * (total - 1) // max(1, count - 1) for i in range(count)]
        result = []
        for path_rank in ranks:
            order = query.path(path_rank)
            result.append({"id": "reference" if order == self.model.reference_order else f"ctx_{rank + 1}_path_{path_rank}",
                           "order": order, "path_rank": path_rank})
        return result

    def metadata(self):
        return {"encoding": self.encoding, "coverage_complete": True,
                "count": self.class_count, "raw_count": self.raw_count,
                "ordered_class_count": self.ordered.class_count,
                "reference_rank": self.reference_rank,
                "language_nodes": len(self.language.nodes),
                "insert_cache_entries": len(self.language.insert_cache),
                "union_cache_entries": len(self.language.union_cache),
                "construction_seconds": self.construction_seconds,
                "event_order": getattr(self.ordered, "context_event_order", {"strategy": "semantic"}),
                "retained_event_observation": self.observer.version,
                "query_states": sum(len(query.memo) for query in self._query_cache.values()),
                "backend": self.backend, "backend_metrics": self.backend_metrics}

    def close(self):
        if getattr(self, '_closed', True):
            return
        self._closed = True
        self._query_cache.clear()
        close = getattr(self.language, 'close', None)
        if close is not None:
            close()
        if self._resources is not None:
            self._resources.cleanup()
            self._resources = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _python_cover(ordered_cover, *, deadline, began, max_nodes,
                  query_timeout, query_max_states):
    """Compute all event-count/final classes from a complete ordered cover."""
    required = ("states", "edges", "tokens", "event_keys", "finals", "roots", "language")
    if any(not hasattr(ordered_cover, name) for name in required):
        raise TypeError("An exact EventLanguageCover is required")
    language = _CountLanguage(max_nodes, deadline)
    roots, raw_suffix = [-1] * len(ordered_cover.states), [0] * len(ordered_cover.states)
    terminal_markers = {}
    order = _topological(ordered_cover.edges, deadline)
    for parent in reversed(order):
        if time.perf_counter() >= language.deadline:
            raise TimeoutError("Context cover construction time budget exhausted")
        if ordered_cover.roots[parent] < 0:
            continue
        outgoing = ordered_cover.edges[parent]
        if not outgoing:
            # The complete layered graph has only goal leaves.  Retain its
            # existing canonical full-final marker without a fresh policy.
            word = ordered_cover.language.word(ordered_cover.roots[parent], 0)
            if len(word) != 1 or word[0] >= 0:
                raise ValueError("Live layered leaf lacks one canonical final marker")
            marker = word[0]
            roots[parent] = language.node(((marker, 0),))
            terminal_markers[parent] = marker
            raw_suffix[parent] = 1
            continue
        root = -1
        for identifier, child in outgoing:
            if roots[child] < 0:
                continue
            raw_suffix[parent] += raw_suffix[child]
            token = ordered_cover.tokens[parent, identifier]
            root = language.union(root, language.insert(token, roots[child]))
        roots[parent] = root
    result = CountCover(ordered_cover, language, roots, terminal_markers, raw_suffix,
                        began, query_timeout, query_max_states, construction_deadline=deadline)
    if result.class_count > ordered_cover.class_count:
        raise AssertionError("Forgetting event order increased the class count")
    if time.perf_counter() >= deadline:
        raise TimeoutError("Count-language construction deadline exhausted")
    return result


def build_count_cover(ordered, *, timeout=120, max_nodes=20000000, native="auto",
                      query_timeout=120, query_max_states=1000000):
    """Build complete event-count classes without enumerating original paths.

    Construction shares one deadline through ordering, a bounded Python probe,
    native admission, export and construction. Each later public member query
    has its own finite query_timeout, shared by that query's recursive work.
    Missing native support or uint64 overflow retains arbitrary-precision
    Python, capped at two million nodes and sharing admission with native
    construction. Budget failures never certify a partial language.
    """
    if native is True:
        native = "native"
    elif native is False:
        native = "python"
    if (native not in {"auto", "python", "native"}
            or type(max_nodes) is not int or max_nodes < 1
            or not math.isfinite(timeout)
            or not math.isfinite(query_timeout) or query_timeout <= 0
            or type(query_max_states) is not int or query_max_states < 1):
        raise ValueError("Invalid count-cover backend or finite budget")
    if timeout <= 0:
        raise TimeoutError("Count-language construction deadline exhausted")
    began = time.perf_counter()
    deadline = began + timeout
    ordered = _reorder_events(ordered, deadline)
    kwargs = dict(began=began, query_timeout=query_timeout, query_max_states=query_max_states)
    if native == "python":
        return _python_cover(ordered, deadline=deadline, max_nodes=max_nodes, **kwargs)
    probe_error = None
    if native == "auto":
        try:
            return _python_cover(ordered, deadline=min(deadline, time.perf_counter() + .35),
                                 max_nodes=min(max_nodes, 200000), **kwargs)
        except (TimeoutError, OverflowError) as error:
            probe_error = type(error).__name__
    try:
        return build_native_count(ordered, deadline=deadline, max_nodes=min(max_nodes, 20000000), **kwargs)
    except NativeCountUnsupported as error:
        remaining = deadline - time.perf_counter()
        if remaining <= 0 or not _COUNT_JOBS.acquire(timeout=remaining):
            raise TimeoutError("Python count fallback admission deadline exhausted") from error
        try:
            limit = min(max_nodes, 2000000)
            result = _python_cover(ordered, deadline=deadline, max_nodes=limit, **kwargs)
            result.backend_metrics = {"native_fallback": str(error), "python_probe_failure": probe_error,
                                      "python_node_budget": limit}
            return result
        finally:
            _COUNT_JOBS.release()


# 原生计数仅替换存储表示；能力不足时保留精确 Python 后备。
ROOT = Path(__file__).resolve().parents[3]


BINARY = ROOT / '.build' / 'psv_count_cover'


_COUNT_JOBS = threading.BoundedSemaphore(1)


class NativeCountUnsupported(ValueError):
    """Native capability is insufficient; the exact Python backend may proceed."""


class _Nodes(Sequence):
    def __init__(self, memory, node_base, edge_base, count):
        self.memory, self.node_base, self.edge_base, self.size = memory, node_base, edge_base, count
        self.cache = OrderedDict()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        if index not in self.cache:
            offset, degree, _, _ = struct.unpack_from('<QIIQ', self.memory, self.node_base + index * 24)
            self.cache[index] = tuple(struct.unpack_from('<iI', self.memory, self.edge_base + (offset + i) * 8)
                                      for i in range(degree))
            if len(self.cache) > 4096:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


class MMapLanguage(_Language):
    """Read-only flat nodes; each instance owns its bounded cache and mapping."""

    def __init__(self, path):
        self.file = Path(path).open('rb')
        try:
            self.memory = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            if len(self.memory) < 32:
                raise NativeCountUnsupported('truncated count-language header')
            magic, original_nodes, node_count, edge_count, self.root_count = struct.unpack_from('<8sIIQQ', self.memory, 0)
            self.node_base = 32 + 4 * original_nodes
            self.edge_base = self.node_base + 24 * node_count
            if magic != b'CCLDAG01' or not original_nodes or not node_count or self.edge_base + edge_count * 8 != len(self.memory):
                raise NativeCountUnsupported('invalid native count-language format')
            self.roots = [value if value != 0xffffffff else -1
                          for value in struct.unpack_from(f'<{original_nodes}I', self.memory, 32)]
            if any(root >= node_count for root in self.roots):
                raise NativeCountUnsupported('native root index outside node arena')
            self.nodes = _Nodes(self.memory, self.node_base, self.edge_base, node_count)
            self.insert_cache, self.union_cache = {}, {}
            if self.count(self.roots[0]) != self.root_count:
                raise NativeCountUnsupported('native root count mismatch')
        except Exception:
            self.close()
            raise

    def count(self, node):
        return struct.unpack_from('<Q', self.memory, self.node_base + node * 24 + 16)[0] if node >= 0 else 0

    def close(self):
        nodes = getattr(self, 'nodes', None)
        if nodes is not None:
            nodes.cache.clear()
            nodes.memory = None
        memory = getattr(self, 'memory', None)
        if memory is not None:
            self.memory = None
            memory.close()
        file = getattr(self, 'file', None)
        if file is not None:
            self.file = None
            file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _remaining(deadline):
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError('Count-language construction deadline exhausted')
    return remaining


def _export(ordered, path, deadline):
    terminal_markers, raw_suffix = {}, [0] * len(ordered.states)
    for parent in reversed(_topological(ordered.edges, deadline)):
        _remaining(deadline)
        if ordered.roots[parent] < 0:
            continue
        if not ordered.edges[parent]:
            word = ordered.language.word(ordered.roots[parent], 0)
            if len(word) != 1 or word[0] >= 0:
                raise ValueError('Live layered leaf lacks a canonical final marker')
            terminal_markers[parent], raw_suffix[parent] = word[0], 1
        else:
            raw_suffix[parent] = sum(raw_suffix[child] for _, child in ordered.edges[parent])
    with Path(path).open('w') as file:
        file.write(str(len(ordered.states)) + '\n')
        for parent, outgoing in enumerate(ordered.edges):
            _remaining(deadline)
            live = [(identifier, child) for identifier, child in outgoing
                    if ordered.roots[parent] >= 0 and ordered.roots[child] >= 0]
            file.write(f'{terminal_markers.get(parent, 0)} {len(live)}')
            for identifier, child in live:
                file.write(f' {ordered.tokens[parent, identifier]} {child}')
            file.write('\n')
    return terminal_markers, raw_suffix


def build_native_count(ordered, *, deadline, began, max_nodes, query_timeout, query_max_states):
    if not BINARY.is_file():
        raise NativeCountUnsupported('optional .build/psv_count_cover binary is missing')
    if not _COUNT_JOBS.acquire(timeout=_remaining(deadline)):
        raise TimeoutError('Count-language native admission deadline exhausted')
    temporary = None
    language = None
    try:
        temporary = TemporaryDirectory(prefix='swm-count-cover-')
        directory = Path(temporary.name)
        terminal_markers, raw_suffix = _export(ordered, directory / 'graph.txt', deadline)
        remaining = _remaining(deadline)
        try:
            result = subprocess.run([str(BINARY), str(directory / 'graph.txt'), str(directory / 'language.bin'),
                                     str(max_nodes), str(remaining)],
                                    text=True, capture_output=True, timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError('Native count-language deadline exhausted') from error
        except OSError as error:
            raise NativeCountUnsupported(str(error)) from error
        _remaining(deadline)
        if result.returncode:
            reason = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f'native exit {result.returncode}'
            if 'TIME_BUDGET' in reason:
                raise TimeoutError(reason)
            if 'BUDGET_EXHAUSTED' in reason or result.returncode < 0:
                raise OverflowError(reason)
            raise NativeCountUnsupported(reason)
        metrics = json.loads(result.stdout.strip().splitlines()[-1])
        if metrics.get('complete') is not True:
            raise NativeCountUnsupported('native did not report a complete language')
        language = MMapLanguage(directory / 'language.bin')
        if language.root_count != metrics['classes'] or len(language.roots) != len(ordered.states):
            raise NativeCountUnsupported('native count/graph dimensions mismatch')
        cover = CountCover(ordered, language, language.roots, terminal_markers, raw_suffix,
                           began, query_timeout, query_max_states, resources=temporary,
                           construction_deadline=deadline)
        if cover.class_count > ordered.class_count:
            raise NativeCountUnsupported('count classes exceed complete ordered classes')
        _remaining(deadline)
        cover.backend = 'native'
        cover.backend_metrics = metrics
        language = temporary = None  # Ownership is transferred to the cover.
        return cover
    finally:
        if language is not None:
            language.close()
        if temporary is not None:
            temporary.cleanup()
        _COUNT_JOBS.release()


# PSV 使用此入口；上方 search_plan_classes 仍提供原有固定动作重排接口。
def search_same_length_classes(model, observer):
    from swm.pddl.planner import build_fd_graph
    graph = build_fd_graph(model, timeout=120, max_states=1000000,
                           pdb_states=100000, verify_edges=True)
    if not all(graph.proof[key] is True for key in
               ('coverage_complete', 'original_edges_verified', 'raw_order_count_exact')):
        raise ValueError('Exact-L search coverage is incomplete')
    ordered = build_event_language_cover(model, observer, graph.states, graph.edges,
        graph.layers, graph.suffix, max_nodes=3000000, deadline=time.perf_counter() + 120)
    if ordered is None:
        raise ValueError('Some original action edges have no observation')
    cover = build_count_cover(ordered, timeout=120, max_nodes=10000000,
                              native='auto', query_timeout=120)
    try:
        if cover.raw_count != graph.suffix[0]:
            raise ValueError('Deduplication lost original paths')
        if cover.class_count > 100000:
            raise ValueError('More than 100000 representative plans; review remains incomplete')
        representatives = []
        for rank in range(cover.class_count):
            order = cover[rank]['order']
            if not model.replay(order) or cover.rank_for_order(order) != rank:
                raise ValueError('Representative replay/rank mismatch')
            representatives.append(dict(id='reference' if rank == cover.reference_rank else f'P{rank:04d}',
                class_rank=rank, original_plans=cover.class_count_for(rank), order=order))
        if sum(row['original_plans'] for row in representatives) != cover.raw_count:
            raise ValueError('Class multiplicities do not conserve original plans')
        proof = dict(status='complete', coverage_complete=True,
            raw_orders_including_reference=cover.raw_count, raw_order_count_exact=True,
            accounted_orders=cover.raw_count, classes=representatives,
            reference_rank=cover.reference_rank,
            factorization={'graph': graph.proof, 'counts': cover.metadata(), 'observer': observer.metadata()})
        return representatives, proof
    finally:
        cover.close()
