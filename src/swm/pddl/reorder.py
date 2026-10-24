from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import copy
import itertools
import math
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

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

    if (
        "hand_free" in pred_arity
        and "holding" in pred_arity
        and pred_arity["hand_free"] == 1
        and pred_arity["holding"] == 2
        and (hands := {tuple(lit[1:]) for lit in pred_literals["hand_free"]})
    ):
        return OccupancyModel(
            free_pred="hand_free",
            occ_pred="holding",
            occ_arity=2,
            resource_positions=(0,),
            object_positions=(1,),
            resources=hands,
        )

    dynamic_preds = [
        pred for pred in pred_arity if add_count[pred] > 0 and del_count[pred] > 0
    ]
    if not dynamic_preds:
        return None

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

                    free_init_count = sum(1 for lit in states[0] if lit[0] == free_pred)
                    occ_init_count = sum(1 for lit in states[0] if lit[0] == occ_pred)
                    state_matches = 0
                    free_overlap = 0
                    mutex_conflicts = 0
                    object_conflicts = 0
                    outside_resource = 0
                    occ_seen = 0

                    for state in states:
                        free_now = {
                            tuple(lit[1:])
                            for lit in state
                            if lit[0] == free_pred
                            and len(lit) == pred_arity[free_pred] + 1
                        }
                        resource_to_objects: dict[ResourceSig, set[ObjectSig]] = (
                            defaultdict(set)
                        )
                        object_to_resources: dict[ObjectSig, set[ResourceSig]] = (
                            defaultdict(set)
                        )

                        for lit in state:
                            if lit[0] == occ_pred and len(lit) == occ_arity + 1:
                                occ_seen += 1

                            projected = model.project(lit)
                            if (
                                lit[0] == occ_pred
                                and len(lit) == occ_arity + 1
                                and projected is None
                            ):
                                outside_resource += 1
                            if projected is None:
                                continue

                            resource, obj = projected
                            state_matches += 1
                            if resource in free_now:
                                free_overlap += 1
                            resource_to_objects[resource].add(obj)
                            object_to_resources[obj].add(resource)

                        for objects in resource_to_objects.values():
                            if len(objects) > 1:
                                mutex_conflicts += len(objects) - 1
                        for object_resources in object_to_resources.values():
                            if len(object_resources) > 1:
                                object_conflicts += len(object_resources) - 1

                    if occ_seen == 0:
                        continue

                    acquire = 0
                    release = 0
                    for action in plan:
                        deleted_free = {
                            tuple(lit[1:])
                            for lit in action.del_eff
                            if lit[0] == free_pred
                            and len(lit) == pred_arity[free_pred] + 1
                        }
                        added_free = {
                            tuple(lit[1:])
                            for lit in action.add_eff
                            if lit[0] == free_pred
                            and len(lit) == pred_arity[free_pred] + 1
                        }
                        added_occ_resources: set[ResourceSig] = set()
                        deleted_occ_resources: set[ResourceSig] = set()

                        for lit in action.add_eff:
                            projected = model.project(lit)
                            if projected is not None:
                                added_occ_resources.add(projected[0])
                        for lit in action.del_eff:
                            projected = model.project(lit)
                            if projected is not None:
                                deleted_occ_resources.add(projected[0])

                        acquire += len(deleted_free & added_occ_resources)
                        release += len(added_free & deleted_occ_resources)

                    if acquire + release == 0 and state_matches == 0:
                        continue

                    score = (
                        1 if acquire + release > 0 else 0,
                        acquire + release,
                        min(acquire, release),
                        free_init_count,
                        -pred_arity[free_pred],
                        -occ_init_count,
                        state_matches,
                        -free_overlap,
                        -mutex_conflicts,
                        -object_conflicts,
                        -outside_resource,
                        1 if resource_positions else 0,
                    )
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
    n = len(plan)
    preds = {i: set() for i in range(n)}
    resource_preds = {"hand_free", "holding"}

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
            if provider == -1 and k < consumer:
                add_edge(consumer, k)
            elif provider >= 0 and k < provider:
                add_edge(k, provider)
            elif provider >= 0 and k > consumer:
                add_edge(consumer, k)

    for provider, consumer, lit in neg_links:
        for k, action in enumerate(plan):
            if k == consumer or lit not in action.add_eff:
                continue
            if provider == -1 and k < consumer:
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
) -> list[GroundAction]:
    n = len(plan)
    done_all = (1 << n) - 1
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
    acquisitions = []
    for consumer, action in enumerate(plan):
        for lit in action.pre_pos & holders.keys():
            causal_links.append((holders[lit], consumer, lit, True))
        for lit in action.del_eff:
            holders.pop(lit, None)
        acquired = set()
        if occupancy_model is not None:
            acquired = {
                lit for lit in action.add_eff
                if occupancy_model.project(lit) is not None
            }
            holders.update((lit, consumer) for lit in acquired)
        acquisitions.append(bool(acquired))

    ancestors = [sum(1 << p for p in preds[i]) for i in range(n)]
    readers = defaultdict(int)
    for provider, consumer, lit, positive in causal_links:
        readers[provider, lit, positive] |= 1 << consumer
        if provider >= 0:
            ancestors[consumer] |= 1 << provider

    # A consuming use ends an episode only after its other readers have finished.
    closures = []
    for provider, consumer, lit, positive in causal_links:
        opposite = plan[consumer].del_eff if positive else plan[consumer].add_eff
        others = readers[provider, lit, positive] & ~(1 << consumer)
        if lit in opposite and not acquisitions[consumer] and (provider >= 0 or others):
            ancestors[consumer] |= others
            closures.append((provider, consumer, others))

    for provider in range(n):
        for consumer in range(n):
            if ancestors[consumer] & (1 << provider):
                ancestors[consumer] |= ancestors[provider]

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

    def rank_ready_actions(state: set[Literal], done_mask: int) -> list[int]:
        ready = ready_actions(state, done_mask)
        if not ready:
            return []

        ready_now = set(ready)
        occupancy_now = current_occupancy(state)
        scored: list[tuple[tuple, int]] = []

        for i in ready:
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
            # Prefer the nearest completion of an established or already-used condition.
            finish = max(
                (
                    n - bin(ancestors[consumer] & ~done_mask).count("1")
                    for provider, consumer, others in closures
                    if (done_mask & others if provider < 0 else done_mask & (1 << provider))
                    and not (done_mask & (1 << consumer))
                    and (i != consumer or done_mask & others == others)
                    and (ancestors[consumer] | (1 << consumer)) & (1 << i)
                ),
                default=0,
            )
            score = (
                finish,
                completed_links,
                -waiting,
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

    def dfs(done_mask: int, frozen_state: frozenset[Literal]) -> list[int] | None:
        if done_mask == done_all:
            return (
                [] if goals_satisfied(set(frozen_state), goal_pos, goal_neg) else None
            )

        key = (done_mask, frozen_state)
        if key in failed:
            return None

        state = set(frozen_state)
        for i in rank_ready_actions(state, done_mask):
            next_state = apply_action(state, plan[i])
            suffix = dfs(done_mask | (1 << i), frozenset(next_state))
            if suffix is not None:
                return [i] + suffix

        failed.add(key)
        return None

    order = dfs(0, frozenset(init_state))
    return plan if order is None else [plan[index] for index in order]


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
