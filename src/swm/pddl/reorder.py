from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
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
    parse_problem,
    rollout,
)

ResourceSig = tuple[str, ...]
ObjectSig = tuple[str, ...]


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


@dataclass
class Commitment:
    kind: str
    target: int | None
    distance: int
    weight: float


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
) -> dict[int, set[int]]:
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
        if provider >= 0:
            for k, action in enumerate(plan):
                if k < provider and lit in action.del_eff:
                    add_edge(k, provider)

    for lit in goal_neg:
        provider = latest_supporter(n, lit, positive=False)
        if provider is None:
            raise ValueError(
                f"Original plan invalid: negative goal (not {lit}) not achieved"
            )
        if provider >= 0:
            for k, action in enumerate(plan):
                if k < provider and lit in action.add_eff:
                    add_edge(k, provider)

    return preds


# ============================================================
# Heuristic DFS reordering
# ============================================================


def reorder_by_search(
    init_state: set[Literal],
    goal_pos: set[Literal],
    goal_neg: set[Literal],
    plan: list[GroundAction],
    preds: dict[int, set[int]],
    occupancy_model: OccupancyModel | None,
) -> list[GroundAction]:
    n = len(plan)
    done_all = (1 << n) - 1
    failed: set[tuple[int, frozenset[Literal]]] = set()

    state_preds = {lit[0] for action in plan for lit in action.add_eff | action.del_eff}
    resource_preds = {"hand_free"}
    if occupancy_model is not None:
        resource_preds.add(occupancy_model.free_pred)
        resource_preds.add(occupancy_model.occ_pred)

    pos_supporters: dict[tuple[int, Literal], list[int]] = {}
    neg_supporters: dict[tuple[int, Literal], list[int]] = {}
    use_pos: dict[Literal, list[int]] = defaultdict(list)
    use_neg: dict[Literal, list[int]] = defaultdict(list)
    adders: dict[Literal, list[int]] = defaultdict(list)
    deleters: dict[Literal, list[int]] = defaultdict(list)
    state_items: set[tuple[str, Literal]] = set()

    for idx, action in enumerate(plan):
        for lit in action.pre_pos:
            providers = [-1] if lit in init_state else []
            providers += [i for i in range(idx) if lit in plan[i].add_eff]
            pos_supporters[(idx, lit)] = providers

            if lit[0] in state_preds and lit[0] not in resource_preds:
                state_items.add(("pos", lit))
                use_pos[lit].append(idx)

        for lit in action.pre_neg:
            providers = [-1] if lit not in init_state else []
            providers += [i for i in range(idx) if lit in plan[i].del_eff]
            neg_supporters[(idx, lit)] = providers

            if lit[0] in state_preds and lit[0] not in resource_preds:
                state_items.add(("neg", lit))
                use_neg[lit].append(idx)

        for lit in action.add_eff:
            adders[lit].append(idx)
        for lit in action.del_eff:
            deleters[lit].append(idx)

    for lit in goal_pos:
        if lit[0] in state_preds and lit[0] not in resource_preds:
            state_items.add(("pos", lit))
    for lit in goal_neg:
        if lit[0] in state_preds and lit[0] not in resource_preds:
            state_items.add(("neg", lit))

    ordered_state_items = sorted(state_items)

    def potential(distance: int) -> float:
        return 1.0 / (distance + 1.0)

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

    def non_resource_preconditions_ready(target: int, state: set[Literal]) -> bool:
        action = plan[target]
        for lit in action.pre_pos:
            if (
                lit[0] in state_preds
                and lit[0] not in resource_preds
                and lit not in state
            ):
                return False
        for lit in action.pre_neg:
            if lit[0] in state_preds and lit[0] not in resource_preds and lit in state:
                return False
        return True

    def make_distance_fn(state: set[Literal], done_mask: int):
        memo: dict[int, frozenset[int] | None] = {}
        visiting: set[int] = set()
        impossible = frozenset(range(n))

        def needed(idx: int) -> frozenset[int] | None:
            if done_mask & (1 << idx):
                return None
            if idx in memo:
                return memo[idx]
            if idx in visiting:
                return impossible

            visiting.add(idx)
            need: set[int] = set()

            for pred in preds[idx]:
                if done_mask & (1 << pred):
                    continue
                sub = needed(pred)
                if sub is None:
                    visiting.remove(idx)
                    memo[idx] = None
                    return None
                need.add(pred)
                need.update(sub)

            for lit in plan[idx].pre_pos:
                if lit in state:
                    continue

                best: frozenset[int] | None = None
                for provider in pos_supporters[(idx, lit)]:
                    if provider == -1:
                        continue
                    if done_mask & (1 << provider):
                        continue
                    sub = needed(provider)
                    if sub is None:
                        continue
                    cand = frozenset(set(sub) | {provider})
                    if best is None or len(cand) < len(best):
                        best = cand

                if best is None:
                    visiting.remove(idx)
                    memo[idx] = None
                    return None
                need.update(best)

            for lit in plan[idx].pre_neg:
                if lit not in state:
                    continue

                best: frozenset[int] | None = None
                for provider in neg_supporters[(idx, lit)]:
                    if provider == -1:
                        continue
                    if done_mask & (1 << provider):
                        continue
                    sub = needed(provider)
                    if sub is None:
                        continue
                    cand = frozenset(set(sub) | {provider})
                    if best is None or len(cand) < len(best):
                        best = cand

                if best is None:
                    visiting.remove(idx)
                    memo[idx] = None
                    return None
                need.update(best)

            visiting.remove(idx)
            memo[idx] = frozenset(need)
            return memo[idx]

        def distance(idx: int) -> int | None:
            required = needed(idx)
            return None if required is None else len(required)

        return distance

    def collect_commitments(
        state: set[Literal], done_mask: int, distance
    ) -> dict[object, Commitment]:
        commitments: dict[object, Commitment] = {}
        done_steps = [i for i in range(n) if done_mask & (1 << i)]

        for sign, lit in ordered_state_items:
            lit_true = lit in state
            if sign == "pos" and not lit_true:
                continue
            if sign == "neg" and lit_true:
                continue

            if sign == "pos":
                use_list = use_pos[lit]
                establish_list = adders[lit]
                flip_list = deleters[lit]
                goal_needed = lit in goal_pos
            else:
                use_list = use_neg[lit]
                establish_list = deleters[lit]
                flip_list = adders[lit]
                goal_needed = lit in goal_neg

            started = any(
                step in use_list or step in establish_list for step in done_steps
            )
            if not started:
                continue

            target: int | None = None
            weight = 1.0

            for step in use_list:
                if not (done_mask & (1 << step)):
                    target = step
                    if not non_resource_preconditions_ready(step, state):
                        weight = 0.0
                    break

            if target is None and not goal_needed:
                for step in flip_list:
                    if not (done_mask & (1 << step)):
                        target = step
                        break

            if target is None:
                continue

            d = distance(target)
            if d is None:
                continue

            key = ("state", target)
            cand = Commitment("state", target, d + 1, weight)
            if key not in commitments or cand.weight > commitments[key].weight:
                commitments[key] = cand

        if occupancy_model is not None:
            for resource, obj in current_occupancy(state).items():
                occ_lit = occupancy_model.build(resource, obj)
                best_distance: int | None = None
                best_target: int | None = None

                for step, action in enumerate(plan):
                    if done_mask & (1 << step):
                        continue
                    if occ_lit not in action.del_eff:
                        continue

                    d = distance(step)
                    if d is None:
                        continue
                    cand = d + 1
                    if best_distance is None or cand < best_distance:
                        best_distance = cand
                        best_target = step

                if best_distance is not None:
                    commitments[("holding", resource, obj)] = Commitment(
                        "holding", best_target, best_distance, 1.0
                    )

        return commitments

    def rank_ready_actions(state: set[Literal], done_mask: int) -> list[int]:
        ready = [
            index
            for index, action in enumerate(plan)
            if not done_mask & (1 << index)
            and all(done_mask & (1 << pred) for pred in preds[index])
            and action.pre_pos <= state
            and not action.pre_neg & state
        ]
        if not ready:
            return []

        distance_now = make_distance_fn(state, done_mask)
        commitments_now = collect_commitments(state, done_mask, distance_now)
        scored: list[tuple[tuple, int]] = []

        for i in ready:
            next_state = apply_action(state, plan[i])
            next_mask = done_mask | (1 << i)
            distance_next = make_distance_fn(next_state, next_mask)
            commitments_next = collect_commitments(next_state, next_mask, distance_next)

            progress = 0.0
            for key, before in commitments_now.items():
                if before.kind == "state":
                    if before.target == i:
                        gain = potential(0) - potential(before.distance)
                    elif key in commitments_next:
                        after = commitments_next[key]
                        gain = (
                            potential(after.distance) - potential(before.distance)
                        ) * after.weight
                    else:
                        continue
                else:
                    after_distance = (
                        commitments_next[key].distance if key in commitments_next else 0
                    )
                    gain = potential(after_distance) - potential(before.distance)

                if gain > 0:
                    progress += gain

            bonus = 0.0
            for key, after in commitments_next.items():
                if key not in commitments_now:
                    bonus = max(bonus, potential(after.distance) * after.weight)

            score = (1 if progress > 0 else 0, progress, bonus, -i)
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
        init_state, goal_pos, goal_neg = parse_problem(problem_path)
        raw_plan, comments = parse_plan(plan_path)
        plan = ground_plan(raw_plan, schemas)

        original_final = rollout(init_state, plan)
        assert_goals(
            original_final, goal_pos, goal_neg, "Original plan does not satisfy goal."
        )

        occupancy_model = infer_occupancy_model(init_state, plan)
        preds = build_predecessors(
            init_state, goal_pos, goal_neg, plan, occupancy_model
        )
        reordered = reorder_by_search(
            init_state, goal_pos, goal_neg, plan, preds, occupancy_model
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
