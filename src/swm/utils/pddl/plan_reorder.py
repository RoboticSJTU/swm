from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


Literal = tuple[str, ...]
ResourceSig = tuple[str, ...]
ObjectSig = tuple[str, ...]


@dataclass
class ActionSchema:
    name: str
    params: list[str]
    pre_pos: set[Literal]
    pre_neg: set[Literal]
    add_eff: set[Literal]
    del_eff: set[Literal]


@dataclass
class GroundAction:
    name: str
    args: list[str]
    pre_pos: set[Literal]
    pre_neg: set[Literal]
    add_eff: set[Literal]
    del_eff: set[Literal]

    def to_line(self) -> str:
        return f"({self.name} {' '.join(self.args)})"


@dataclass
class OccupancyModel:
    free_pred: str
    occ_pred: str
    occ_arity: int
    resource_positions: tuple[int, ...]
    object_positions: tuple[int, ...]
    resources: set[ResourceSig]
    implicit_resource: ResourceSig | None

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
        else:
            if self.implicit_resource is None or resource != self.implicit_resource:
                raise ValueError("Invalid implicit resource")

        if len(obj) != len(self.object_positions):
            raise ValueError("Object signature arity mismatch")
        for pos, value in zip(self.object_positions, obj):
            args[pos] = value

        if any(x is None for x in args):
            raise ValueError("Could not rebuild occupancy literal")

        return (self.occ_pred, *[x for x in args if x is not None])


# =========================
# PDDL 解析
# =========================
def load_sexpr(path: Path):
    text = re.sub(r";[^\n]*", "", path.read_text(encoding="utf-8")).lower()
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()

    def parse():
        if not tokens:
            raise ValueError(f"Unexpected EOF in {path}")

        token = tokens.pop(0)
        if token == "(":
            out = []
            while tokens and tokens[0] != ")":
                out.append(parse())
            if not tokens:
                raise ValueError(f"Missing ')' in {path}")
            tokens.pop(0)
            return out

        if token == ")":
            raise ValueError(f"Unexpected ')' in {path}")

        return token

    root = parse()
    if tokens:
        raise ValueError(f"Unparsed tokens remain in {path}")
    return root


def strip_types(items: list[str]) -> list[str]:
    out = []
    skip = False
    for x in items:
        if x == "-":
            skip = True
        elif skip:
            skip = False
        else:
            out.append(x)
    return out


def read_literals(expr) -> tuple[set[Literal], set[Literal]]:
    if expr is None:
        return set(), set()

    if isinstance(expr, str):
        raise ValueError(f"Unexpected atom: {expr}")

    if expr[0] == "and":
        pos, neg = set(), set()
        for sub in expr[1:]:
            p, n = read_literals(sub)
            pos |= p
            neg |= n
        return pos, neg

    if expr[0] == "not":
        return set(), {tuple(expr[1])}

    return {tuple(expr)}, set()


def parse_domain(path: Path) -> dict[str, ActionSchema]:
    root = load_sexpr(path)
    if root[0] != "define":
        raise ValueError(f"{path} is not a valid domain file")

    schemas = {}
    for item in root[1:]:
        if not isinstance(item, list) or not item or item[0] != ":action":
            continue

        name = item[1]
        params: list[str] = []
        precondition = None
        effect = None

        i = 2
        while i < len(item):
            key = item[i]
            value = item[i + 1]
            if key == ":parameters":
                params = strip_types(value)
            elif key == ":precondition":
                precondition = value
            elif key == ":effect":
                effect = value
            i += 2

        pre_pos, pre_neg = read_literals(precondition)
        add_eff, del_eff = read_literals(effect)
        schemas[name] = ActionSchema(name, params, pre_pos, pre_neg, add_eff, del_eff)

    return schemas


def parse_problem(path: Path) -> tuple[set[Literal], set[Literal], set[Literal]]:
    root = load_sexpr(path)
    if root[0] != "define":
        raise ValueError(f"{path} is not a valid problem file")

    init_state: set[Literal] = set()
    init_neg: set[Literal] = set()
    goal_pos: set[Literal] = set()
    goal_neg: set[Literal] = set()

    for item in root[1:]:
        if not isinstance(item, list) or not item:
            continue

        if item[0] == ":init":
            for lit in item[1:]:
                if isinstance(lit, list) and lit and lit[0] == "not":
                    atom = tuple(lit[1])
                    if atom in init_state:
                        raise ValueError(f"{path}: contradictory init literal {atom}")
                    init_neg.add(atom)
                else:
                    atom = tuple(lit)
                    if atom in init_neg:
                        raise ValueError(f"{path}: contradictory init literal {atom}")
                    init_state.add(atom)

        elif item[0] == ":goal":
            goal_pos, goal_neg = read_literals(item[1])

    return init_state, goal_pos, goal_neg


def parse_plan(path: Path) -> tuple[list[tuple[str, list[str]]], list[str]]:
    plan = []
    comments = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith(";"):
            comments.append(line)
            continue

        m = re.match(r"^\(([^()]*)\)$", line.lower())
        if not m:
            continue

        parts = m.group(1).split()
        plan.append((parts[0], parts[1:]))

    return plan, comments


# =========================
# grounding + 执行
# =========================
def ground_plan(raw_plan, schemas):
    def substitute(lit, mapping):
        return tuple(mapping.get(x, x) for x in lit)

    plan = []
    for name, args in raw_plan:
        if name not in schemas:
            raise KeyError(f"Action '{name}' not found in domain")

        schema = schemas[name]
        if len(args) != len(schema.params):
            raise ValueError(
                f"Arity mismatch for action {name}: expected {len(schema.params)}, got {len(args)}"
            )

        mapping = dict(zip(schema.params, args))

        pre_pos = {substitute(lit, mapping) for lit in schema.pre_pos}
        pre_neg = {substitute(lit, mapping) for lit in schema.pre_neg}
        add_eff = {substitute(lit, mapping) for lit in schema.add_eff}
        del_eff = {substitute(lit, mapping) for lit in schema.del_eff}

        # 关键修复：若同一 grounded literal 同时 add 和 del，按 apply_action 的语义保留 add
        del_eff -= add_eff

        plan.append(
            GroundAction(
                name=name,
                args=args,
                pre_pos=pre_pos,
                pre_neg=pre_neg,
                add_eff=add_eff,
                del_eff=del_eff,
            )
        )
    return plan


def apply_action(state: set[Literal], action: GroundAction) -> set[Literal]:
    missing = action.pre_pos - state
    violated = action.pre_neg & state

    if missing or violated:
        msg = [f"Action not applicable: {action.to_line()}"]
        if missing:
            msg.append(f"Missing positive preconditions: {sorted(missing)}")
        if violated:
            msg.append(f"Violated negative preconditions: {sorted(violated)}")
        raise ValueError("\n".join(msg))

    new_state = set(state)
    new_state.difference_update(action.del_eff)
    new_state.update(action.add_eff)
    return new_state


def rollout(
    init_state: set[Literal],
    plan: list[GroundAction],
    trace: bool = False,
) -> set[Literal] | list[set[Literal]]:
    state = set(init_state)
    states = [set(state)] if trace else None

    for action in plan:
        state = apply_action(state, action)
        if trace:
            states.append(set(state))

    return states if trace else state


def goals_satisfied(state: set[Literal], goal_pos: set[Literal], goal_neg: set[Literal]) -> bool:
    return goal_pos <= state and not (goal_neg & state)


# =========================
# occupancy 推理
# =========================
def infer_occupancy_model(
    init_state: set[Literal],
    plan: list[GroundAction],
) -> OccupancyModel | None:
    states = rollout(init_state, plan, trace=True)
    assert isinstance(states, list)

    pred_arities: dict[str, int] = {}
    pred_literals: dict[str, list[Literal]] = {}
    add_count: dict[str, int] = {}
    del_count: dict[str, int] = {}

    def record(lit: Literal):
        pred = lit[0]
        arity = len(lit) - 1
        old = pred_arities.get(pred)
        if old is not None and old != arity:
            raise ValueError(f"Predicate {pred} appears with inconsistent arity")
        pred_arities[pred] = arity
        pred_literals.setdefault(pred, []).append(lit)

    for state in states:
        for lit in state:
            record(lit)

    for action in plan:
        for lit in action.pre_pos | action.pre_neg | action.add_eff | action.del_eff:
            record(lit)
        for lit in action.add_eff:
            add_count[lit[0]] = add_count.get(lit[0], 0) + 1
        for lit in action.del_eff:
            del_count[lit[0]] = del_count.get(lit[0], 0) + 1

    # 第一层：标准锚点 hand_free / holding
    if pred_arities.get("hand_free") == 1 and pred_arities.get("holding") == 2:
        resources = {tuple(lit[1:]) for lit in pred_literals.get("hand_free", [])}
        if resources:
            return OccupancyModel(
                free_pred="hand_free",
                occ_pred="holding",
                occ_arity=2,
                resource_positions=(0,),
                object_positions=(1,),
                resources=resources,
                implicit_resource=None,
            )

    # 第二层：泛化占用推理
    dynamic_preds = [
        pred
        for pred in pred_arities
        if add_count.get(pred, 0) > 0 and del_count.get(pred, 0) > 0
    ]
    if not dynamic_preds:
        return None

    def all_position_subsets(arity: int):
        positions = list(range(arity))
        for r in range(arity + 1):
            for subset in combinations(positions, r):
                yield tuple(subset)

    def project_candidate(
        lit: Literal,
        occ_pred: str,
        occ_arity: int,
        resource_positions: tuple[int, ...],
        object_positions: tuple[int, ...],
        resources: set[ResourceSig],
        implicit_resource: ResourceSig | None,
    ) -> tuple[ResourceSig, ObjectSig] | None:
        if lit[0] != occ_pred or len(lit) != occ_arity + 1:
            return None

        if resource_positions:
            resource = tuple(lit[1 + i] for i in resource_positions)
            if resource not in resources:
                return None
        else:
            if implicit_resource is None:
                return None
            resource = implicit_resource

        obj = tuple(lit[1 + i] for i in object_positions)
        return (resource, obj) if obj else None

    candidates: list[tuple[tuple, OccupancyModel]] = []

    for free_pred in dynamic_preds:
        resources = {tuple(lit[1:]) for lit in pred_literals.get(free_pred, [])}
        if not resources:
            continue

        for occ_pred in dynamic_preds:
            if occ_pred == free_pred:
                continue

            occ_arity = pred_arities[occ_pred]
            for resource_positions in all_position_subsets(occ_arity):
                object_positions = tuple(i for i in range(occ_arity) if i not in resource_positions)
                if not object_positions:
                    continue

                implicit_resource = None
                if not resource_positions:
                    if len(resources) != 1:
                        continue
                    implicit_resource = next(iter(resources))

                acquire = 0
                release = 0
                state_matches = 0
                free_overlap = 0
                mutex_conflicts = 0
                object_conflicts = 0
                outside_resource = 0
                occ_seen = 0
                free_init_count = sum(1 for lit in states[0] if lit[0] == free_pred)
                occ_init_count = sum(1 for lit in states[0] if lit[0] == occ_pred)

                for state in states:
                    free_now = {
                        tuple(lit[1:])
                        for lit in state
                        if lit[0] == free_pred and len(lit) == pred_arities[free_pred] + 1
                    }
                    resource_to_objects: dict[ResourceSig, set[ObjectSig]] = {}
                    object_to_resources: dict[ObjectSig, set[ResourceSig]] = {}

                    for lit in state:
                        projected = project_candidate(
                            lit,
                            occ_pred,
                            occ_arity,
                            resource_positions,
                            object_positions,
                            resources,
                            implicit_resource,
                        )

                        if lit[0] == occ_pred and len(lit) == occ_arity + 1:
                            occ_seen += 1
                            if projected is None:
                                outside_resource += 1
                                continue

                        if projected is None:
                            continue

                        resource, obj = projected
                        state_matches += 1
                        if resource in free_now:
                            free_overlap += 1
                        resource_to_objects.setdefault(resource, set()).add(obj)
                        object_to_resources.setdefault(obj, set()).add(resource)

                    mutex_conflicts += sum(len(v) - 1 for v in resource_to_objects.values() if len(v) > 1)
                    object_conflicts += sum(len(v) - 1 for v in object_to_resources.values() if len(v) > 1)

                if occ_seen == 0:
                    continue

                for action in plan:
                    deleted_free = {
                        tuple(lit[1:])
                        for lit in action.del_eff
                        if lit[0] == free_pred and len(lit) == pred_arities[free_pred] + 1
                    }
                    added_free = {
                        tuple(lit[1:])
                        for lit in action.add_eff
                        if lit[0] == free_pred and len(lit) == pred_arities[free_pred] + 1
                    }

                    added_occ_resources = set()
                    deleted_occ_resources = set()

                    for lit in action.add_eff:
                        projected = project_candidate(
                            lit,
                            occ_pred,
                            occ_arity,
                            resource_positions,
                            object_positions,
                            resources,
                            implicit_resource,
                        )
                        if projected is not None:
                            added_occ_resources.add(projected[0])

                    for lit in action.del_eff:
                        projected = project_candidate(
                            lit,
                            occ_pred,
                            occ_arity,
                            resource_positions,
                            object_positions,
                            resources,
                            implicit_resource,
                        )
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
                    -pred_arities[free_pred],
                    -occ_init_count,
                    state_matches,
                    -free_overlap,
                    -mutex_conflicts,
                    -object_conflicts,
                    -outside_resource,
                    1 if resource_positions else 0,
                )

                candidates.append(
                    (
                        score,
                        OccupancyModel(
                            free_pred=free_pred,
                            occ_pred=occ_pred,
                            occ_arity=occ_arity,
                            resource_positions=resource_positions,
                            object_positions=object_positions,
                            resources=resources,
                            implicit_resource=implicit_resource,
                        ),
                    )
                )

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# =========================
# 严格前驱 DAG
# =========================
def build_predecessors(
    init_state: set[Literal],
    goal_pos: set[Literal],
    goal_neg: set[Literal],
    plan: list[GroundAction],
) -> dict[int, set[int]]:
    n = len(plan)
    preds = {i: set() for i in range(n)}

    def add_edge(i: int, j: int):
        if i != j:
            preds[j].add(i)

    def latest_supporter(end: int, lit: Literal, positive: bool) -> int | None:
        if positive:
            provider = -1 if lit in init_state else None
        else:
            provider = -1 if lit not in init_state else None

        for i in range(end):
            if positive:
                if lit in plan[i].del_eff:
                    provider = None
                if lit in plan[i].add_eff:
                    provider = i
            else:
                if lit in plan[i].add_eff:
                    provider = None
                if lit in plan[i].del_eff:
                    provider = i

        return provider

    pos_links = []
    neg_links = []

    for consumer, action in enumerate(plan):
        for lit in action.pre_pos:
            provider = latest_supporter(consumer, lit, True)
            if provider is None:
                raise ValueError(
                    f"Original plan invalid: no supporter for positive precondition {lit} "
                    f"of step {consumer}: {action.to_line()}"
                )
            pos_links.append((provider, consumer, lit))
            if provider >= 0:
                add_edge(provider, consumer)

        for lit in action.pre_neg:
            provider = latest_supporter(consumer, lit, False)
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
                if k < consumer:
                    add_edge(consumer, k)
            else:
                if k < provider:
                    add_edge(k, provider)
                elif k > consumer:
                    add_edge(consumer, k)

    for provider, consumer, lit in neg_links:
        for k, action in enumerate(plan):
            if k == consumer or lit not in action.add_eff:
                continue

            if provider == -1:
                if k < consumer:
                    add_edge(consumer, k)
            else:
                if k < provider:
                    add_edge(k, provider)
                elif k > consumer:
                    add_edge(consumer, k)

    for lit in goal_pos:
        provider = latest_supporter(n, lit, True)
        if provider is None:
            raise ValueError(f"Original plan invalid: positive goal {lit} not achieved")
        if provider >= 0:
            for k, action in enumerate(plan):
                if k < provider and lit in action.del_eff:
                    add_edge(k, provider)

    for lit in goal_neg:
        provider = latest_supporter(n, lit, False)
        if provider is None:
            raise ValueError(f"Original plan invalid: negative goal (not {lit}) not achieved")
        if provider >= 0:
            for k, action in enumerate(plan):
                if k < provider and lit in action.add_eff:
                    add_edge(k, provider)

    return preds


# =========================
# 搜索重排
# =========================
def reorder_by_search(
    init_state: set[Literal],
    goal_pos: set[Literal],
    goal_neg: set[Literal],
    plan: list[GroundAction],
    preds: dict[int, set[int]],
    occupancy_model: OccupancyModel | None,
) -> list[GroundAction]:
    n = len(plan)
    all_done = (1 << n) - 1
    failed = set()

    pos_supporters: dict[tuple[int, Literal], list[int]] = {}
    neg_supporters: dict[tuple[int, Literal], list[int]] = {}

    for consumer, action in enumerate(plan):
        for lit in action.pre_pos:
            providers = []
            if lit in init_state:
                providers.append(-1)
            providers.extend(i for i in range(consumer) if lit in plan[i].add_eff)
            pos_supporters[(consumer, lit)] = providers

        for lit in action.pre_neg:
            providers = []
            if lit not in init_state:
                providers.append(-1)
            providers.extend(i for i in range(consumer) if lit in plan[i].del_eff)
            neg_supporters[(consumer, lit)] = providers

    state_preds = {lit[0] for action in plan for lit in (action.add_eff | action.del_eff)}
    excluded = {"hand_free"}
    if occupancy_model is not None:
        excluded.add(occupancy_model.free_pred)
        excluded.add(occupancy_model.occ_pred)

    state_chain_requirements: dict[int, set[tuple[str, Literal]]] = {}
    for i, action in enumerate(plan):
        reqs = set()
        for lit in action.pre_pos:
            if lit[0] in state_preds and lit[0] not in excluded:
                reqs.add(("pos", lit))
        for lit in action.pre_neg:
            if lit[0] in state_preds and lit[0] not in excluded:
                reqs.add(("neg", lit))
        state_chain_requirements[i] = reqs

    def potential(distance: int) -> float:
        return 1.0 / (distance + 1.0)

    def current_holding(state: set[Literal]) -> dict[ResourceSig, ObjectSig]:
        if occupancy_model is None:
            return {}

        buckets: dict[ResourceSig, set[ObjectSig]] = {}
        for lit in state:
            projected = occupancy_model.project(lit)
            if projected is None:
                continue
            resource, obj = projected
            buckets.setdefault(resource, set()).add(obj)

        out = {}
        for resource, objects in buckets.items():
            if len(objects) == 1:
                out[resource] = next(iter(objects))
        return out

    def open_state_chains(state: set[Literal], done_mask: int) -> set[int]:
        active = set()
        for consumer in range(n):
            if done_mask & (1 << consumer):
                continue

            for sign, lit in state_chain_requirements[consumer]:
                if sign == "pos":
                    if lit in state and any(
                        (done_mask & (1 << i)) and lit in plan[i].add_eff
                        for i in range(consumer)
                    ):
                        active.add(consumer)
                        break
                else:
                    if lit not in state and any(
                        (done_mask & (1 << i)) and lit in plan[i].del_eff
                        for i in range(consumer)
                    ):
                        active.add(consumer)
                        break
        return active

    def make_distance(state: set[Literal], done_mask: int):
        memo = {}
        visiting = set()
        impossible = frozenset(range(n))

        def needed(idx: int):
            if done_mask & (1 << idx):
                return None
            if idx in memo:
                return memo[idx]
            if idx in visiting:
                return impossible

            visiting.add(idx)
            need = set()

            for p in preds[idx]:
                if done_mask & (1 << p):
                    continue
                sub = needed(p)
                if sub is None:
                    visiting.remove(idx)
                    memo[idx] = None
                    return None
                need.add(p)
                need.update(sub)

            for lit in plan[idx].pre_pos:
                if lit in state:
                    continue

                best = None
                for provider in pos_supporters[(idx, lit)]:
                    if provider == -1:
                        continue

                    if done_mask & (1 << provider):
                        if lit in state:
                            best = frozenset()
                            break
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

                best = None
                for provider in neg_supporters[(idx, lit)]:
                    if provider == -1:
                        continue

                    if done_mask & (1 << provider):
                        if lit not in state:
                            best = frozenset()
                            break
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

        def dist(idx: int) -> int | None:
            x = needed(idx)
            return None if x is None else len(x)

        return dist

    def finish_state_chain(dist_fn, idx: int, executed_now: bool = False) -> int | None:
        if executed_now:
            return 0
        d = dist_fn(idx)
        return None if d is None else d + 1

    def finish_holding(
        state: set[Literal],
        dist_fn,
        done_mask: int,
        resource: ResourceSig,
        obj: ObjectSig,
    ) -> int | None:
        if current_holding(state).get(resource) != obj:
            return 0

        assert occupancy_model is not None
        lit = occupancy_model.build(resource, obj)

        best = None
        for j, action in enumerate(plan):
            if done_mask & (1 << j):
                continue
            if lit not in action.del_eff:
                continue

            d = dist_fn(j)
            if d is None:
                continue

            cand = d + 1
            if best is None or cand < best:
                best = cand

        return best

    def current_chains(state: set[Literal], done_mask: int) -> list[tuple]:
        chains = [("state", idx) for idx in sorted(open_state_chains(state, done_mask))]
        for resource, obj in sorted(current_holding(state).items()):
            chains.append(("holding", resource, obj))
        return chains

    def new_chain_bonus(
        chain_set_now: set[tuple],
        chains_next: list[tuple],
        next_state: set[Literal],
        next_mask: int,
        dist_next,
    ) -> float:
        best = 0.0
        for chain in chains_next:
            if chain in chain_set_now:
                continue

            if chain[0] == "state":
                distance = finish_state_chain(dist_next, chain[1])
            else:
                _, resource, obj = chain
                distance = finish_holding(next_state, dist_next, next_mask, resource, obj)

            if distance is not None:
                best = max(best, potential(distance))

        return best

    def rank_ready_actions(state: set[Literal], done_mask: int) -> list[int]:
        ready = []
        for i, action in enumerate(plan):
            if done_mask & (1 << i):
                continue
            if not all(done_mask & (1 << p) for p in preds[i]):
                continue
            if not (action.pre_pos <= state):
                continue
            if action.pre_neg & state:
                continue
            ready.append(i)

        if not ready:
            return []

        dist_now = make_distance(state, done_mask)
        chains_now = current_chains(state, done_mask)
        chain_set_now = set(chains_now)
        scored = []

        for i in ready:
            next_state = apply_action(state, plan[i])
            next_mask = done_mask | (1 << i)
            dist_next = make_distance(next_state, next_mask)
            chains_next = current_chains(next_state, next_mask)

            progress = 0.0
            for chain in chains_now:
                if chain[0] == "state":
                    idx = chain[1]
                    before = finish_state_chain(dist_now, idx)
                    after = finish_state_chain(dist_next, idx, executed_now=(idx == i))
                else:
                    _, resource, obj = chain
                    before = finish_holding(state, dist_now, done_mask, resource, obj)
                    after = finish_holding(next_state, dist_next, next_mask, resource, obj)

                if before is None or after is None:
                    continue

                gain = potential(after) - potential(before)
                if gain > 0:
                    progress += gain

            bonus = new_chain_bonus(chain_set_now, chains_next, next_state, next_mask, dist_next)
            scored.append(((1 if progress > 0 else 0, progress, bonus, -i), i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scored]

    def dfs(done_mask: int, state: frozenset[Literal]):
        if done_mask == all_done:
            return [] if goals_satisfied(set(state), goal_pos, goal_neg) else None

        key = (done_mask, state)
        if key in failed:
            return None

        for i in rank_ready_actions(set(state), done_mask):
            next_state = apply_action(set(state), plan[i])
            suffix = dfs(done_mask | (1 << i), frozenset(next_state))
            if suffix is not None:
                return [i] + suffix

        failed.add(key)
        return None

    order = dfs(0, frozenset(init_state))
    if order is None:
        return plan

    reordered = [plan[i] for i in order]
    final_state = rollout(init_state, reordered)
    assert isinstance(final_state, set)

    return reordered if goals_satisfied(final_state, goal_pos, goal_neg) else plan


# =========================
# 主流程
# =========================
def plan_reorder(domain_path: Path, problem_path: Path, plan_path: Path, output_path: Path) -> None:
    schemas = parse_domain(domain_path)
    init_state, goal_pos, goal_neg = parse_problem(problem_path)
    raw_plan, comments = parse_plan(plan_path)
    plan = ground_plan(raw_plan, schemas)

    original_final = rollout(init_state, plan)
    assert isinstance(original_final, set)
    if not goals_satisfied(original_final, goal_pos, goal_neg):
        missing = sorted(goal_pos - original_final)
        violated = sorted(goal_neg & original_final)
        msg = ["Original plan does not satisfy goal."]
        if missing:
            msg.append(f"Missing positive goals: {missing}")
        if violated:
            msg.append(f"Violated negative goals: {violated}")
        raise ValueError("\n".join(msg))

    occupancy_model = infer_occupancy_model(init_state, plan)
    preds = build_predecessors(init_state, goal_pos, goal_neg, plan)
    reordered = reorder_by_search(init_state, goal_pos, goal_neg, plan, preds, occupancy_model)

    final_state = rollout(init_state, reordered)
    assert isinstance(final_state, set)
    if not goals_satisfied(final_state, goal_pos, goal_neg):
        missing = sorted(goal_pos - final_state)
        violated = sorted(goal_neg & final_state)
        msg = ["Final reordered plan does not satisfy goal."]
        if missing:
            msg.append(f"Missing positive goals: {missing}")
        if violated:
            msg.append(f"Violated negative goals: {violated}")
        raise ValueError("\n".join(msg))

    lines = [action.to_line() for action in reordered] + comments
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    round_dir = Path("/home/xyx/下载/swm/test_pddl")
    plan_reorder(
        domain_path=round_dir / "temp_domain.pddl",
        problem_path=round_dir / "temp_problem.pddl",
        plan_path=round_dir / "plan.txt",
        output_path=round_dir / "plan_reorder.txt",
    )