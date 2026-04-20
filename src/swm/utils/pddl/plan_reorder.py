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
        if not obj:
            return None
        return resource, obj

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
# PDDL parsing
# =========================
def load_sexpr(path: Path):
    text = re.sub(r";[^\n]*", "", path.read_text(encoding="utf-8")).lower()
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    idx = 0

    def parse():
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError(f"Unexpected EOF in {path}")

        tok = tokens[idx]
        idx += 1
        if tok == "(":
            out = []
            while idx < len(tokens) and tokens[idx] != ")":
                out.append(parse())
            if idx >= len(tokens):
                raise ValueError(f"Missing ')' in {path}")
            idx += 1
            return out
        if tok == ")":
            raise ValueError(f"Unexpected ')' in {path}")
        return tok

    root = parse()
    if idx != len(tokens):
        raise ValueError(f"Unparsed tokens remain in {path}")
    return root


def ensure_supported_pddl(path: Path) -> None:
    root = load_sexpr(path)
    stack = [root]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            if cur in {"when", "forall", "or"}:
                raise NotImplementedError(f"Unsupported PDDL construct '{cur}' in {path}")
            continue
        if isinstance(cur, list):
            stack.extend(cur)


def strip_types(items: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for x in items:
        if x == "-":
            skip_next = True
        elif skip_next:
            skip_next = False
        else:
            out.append(x)
    return out


def read_literals(expr) -> tuple[set[Literal], set[Literal]]:
    if expr is None:
        return set(), set()
    if isinstance(expr, str):
        raise ValueError(f"Unexpected atom: {expr}")
    if expr[0] == "and":
        pos: set[Literal] = set()
        neg: set[Literal] = set()
        for sub in expr[1:]:
            sub_pos, sub_neg = read_literals(sub)
            pos |= sub_pos
            neg |= sub_neg
        return pos, neg
    if expr[0] == "not":
        return set(), {tuple(expr[1])}
    return {tuple(expr)}, set()


def parse_domain(path: Path) -> dict[str, ActionSchema]:
    root = load_sexpr(path)
    if root[0] != "define":
        raise ValueError(f"{path} is not a valid domain file")

    schemas: dict[str, ActionSchema] = {}
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
    actions: list[tuple[str, list[str]]] = []
    comments: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(";"):
            comments.append(line)
            continue
        m = re.match(r"^\(([^()]*)\)$", line.lower())
        if m:
            parts = m.group(1).split()
            actions.append((parts[0], parts[1:]))
    return actions, comments


# =========================
# Grounding and execution
# =========================
def ground_plan(raw_plan: list[tuple[str, list[str]]], schemas: dict[str, ActionSchema]) -> list[GroundAction]:
    plan: list[GroundAction] = []
    for name, args in raw_plan:
        if name not in schemas:
            raise KeyError(f"Action '{name}' not found in domain")
        schema = schemas[name]
        if len(args) != len(schema.params):
            raise ValueError(
                f"Arity mismatch for action {name}: expected {len(schema.params)}, got {len(args)}"
            )

        mapping = dict(zip(schema.params, args))
        pre_pos = {tuple(mapping.get(x, x) for x in lit) for lit in schema.pre_pos}
        pre_neg = {tuple(mapping.get(x, x) for x in lit) for lit in schema.pre_neg}
        add_eff = {tuple(mapping.get(x, x) for x in lit) for lit in schema.add_eff}
        del_eff = {tuple(mapping.get(x, x) for x in lit) for lit in schema.del_eff}
        del_eff -= add_eff

        plan.append(GroundAction(name, args, pre_pos, pre_neg, add_eff, del_eff))
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


def rollout(init_state: set[Literal], plan: list[GroundAction], trace: bool = False):
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
# Occupancy inference
# =========================
def infer_occupancy_model(init_state: set[Literal], plan: list[GroundAction]) -> OccupancyModel | None:
    states = rollout(init_state, plan, trace=True)
    assert isinstance(states, list)

    pred_arity: dict[str, int] = {}
    pred_literals: dict[str, list[Literal]] = {}
    add_count: dict[str, int] = {}
    del_count: dict[str, int] = {}

    def record(lit: Literal) -> None:
        pred = lit[0]
        arity = len(lit) - 1
        if pred in pred_arity and pred_arity[pred] != arity:
            raise ValueError(f"Predicate {pred} appears with inconsistent arity")
        pred_arity[pred] = arity
        if pred not in pred_literals:
            pred_literals[pred] = []
        pred_literals[pred].append(lit)

    for state in states:
        for lit in state:
            record(lit)
    for action in plan:
        for lit in action.pre_pos | action.pre_neg | action.add_eff | action.del_eff:
            record(lit)
        for lit in action.add_eff:
            pred = lit[0]
            if pred not in add_count:
                add_count[pred] = 0
            add_count[pred] += 1
        for lit in action.del_eff:
            pred = lit[0]
            if pred not in del_count:
                del_count[pred] = 0
            del_count[pred] += 1

    if pred_arity.get("hand_free") == 1 and pred_arity.get("holding") == 2:
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

    dynamic_preds = [
        pred for pred in pred_arity if add_count.get(pred, 0) > 0 and del_count.get(pred, 0) > 0
    ]
    if not dynamic_preds:
        return None

    candidates: list[tuple[tuple, OccupancyModel]] = []
    for free_pred in dynamic_preds:
        resources = {tuple(lit[1:]) for lit in pred_literals.get(free_pred, [])}
        if not resources:
            continue

        for occ_pred in dynamic_preds:
            if occ_pred == free_pred:
                continue

            occ_arity = pred_arity[occ_pred]
            positions = list(range(occ_arity))
            for r in range(occ_arity + 1):
                for resource_positions in combinations(positions, r):
                    object_positions = tuple(i for i in positions if i not in resource_positions)
                    if not object_positions:
                        continue

                    implicit_resource: ResourceSig | None = None
                    if not resource_positions:
                        if len(resources) != 1:
                            continue
                        implicit_resource = next(iter(resources))

                    def project(lit: Literal) -> tuple[ResourceSig, ObjectSig] | None:
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
                        if not obj:
                            return None
                        return resource, obj

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
                            if lit[0] == free_pred and len(lit) == pred_arity[free_pred] + 1
                        }
                        resource_to_objects: dict[ResourceSig, set[ObjectSig]] = {}
                        object_to_resources: dict[ObjectSig, set[ResourceSig]] = {}

                        for lit in state:
                            if lit[0] == occ_pred and len(lit) == occ_arity + 1:
                                occ_seen += 1
                            projected = project(lit)
                            if lit[0] == occ_pred and len(lit) == occ_arity + 1 and projected is None:
                                outside_resource += 1
                            if projected is None:
                                continue

                            resource, obj = projected
                            state_matches += 1
                            if resource in free_now:
                                free_overlap += 1
                            if resource not in resource_to_objects:
                                resource_to_objects[resource] = set()
                            resource_to_objects[resource].add(obj)
                            if obj not in object_to_resources:
                                object_to_resources[obj] = set()
                            object_to_resources[obj].add(resource)

                        for values in resource_to_objects.values():
                            if len(values) > 1:
                                mutex_conflicts += len(values) - 1
                        for values in object_to_resources.values():
                            if len(values) > 1:
                                object_conflicts += len(values) - 1

                    if occ_seen == 0:
                        continue

                    for action in plan:
                        deleted_free = {
                            tuple(lit[1:])
                            for lit in action.del_eff
                            if lit[0] == free_pred and len(lit) == pred_arity[free_pred] + 1
                        }
                        added_free = {
                            tuple(lit[1:])
                            for lit in action.add_eff
                            if lit[0] == free_pred and len(lit) == pred_arity[free_pred] + 1
                        }
                        added_occ_resources: set[ResourceSig] = set()
                        deleted_occ_resources: set[ResourceSig] = set()

                        for lit in action.add_eff:
                            projected = project(lit)
                            if projected is not None:
                                added_occ_resources.add(projected[0])
                        for lit in action.del_eff:
                            projected = project(lit)
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
                    candidates.append(
                        (
                            score,
                            OccupancyModel(
                                free_pred=free_pred,
                                occ_pred=occ_pred,
                                occ_arity=occ_arity,
                                resource_positions=tuple(resource_positions),
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
# Strict DAG
# =========================
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

    def add_edge(i: int, j: int) -> None:
        if i != j:
            preds[j].add(i)

    def latest_supporter(end: int, lit: Literal, positive: bool) -> int | None:
        provider = (-1 if lit in init_state else None) if positive else (-1 if lit not in init_state else None)
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

    pos_links: list[tuple[int, int, Literal]] = []
    neg_links: list[tuple[int, int, Literal]] = []

    for consumer, action in enumerate(plan):
        for lit in action.pre_pos:
            if lit[0] in resource_preds:
                continue
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
            if lit[0] in resource_preds:
                continue
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
# Reordering search
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
    done_all = (1 << n) - 1
    failed: set[tuple[int, frozenset[Literal]]] = set()

    pos_supporters: dict[tuple[int, Literal], list[int]] = {}
    neg_supporters: dict[tuple[int, Literal], list[int]] = {}
    state_preds = {lit[0] for action in plan for lit in (action.add_eff | action.del_eff)}
    resource_preds = {"hand_free"}
    if occupancy_model is not None:
        resource_preds.add(occupancy_model.free_pred)
        resource_preds.add(occupancy_model.occ_pred)

    use_pos: dict[Literal, list[int]] = {}
    use_neg: dict[Literal, list[int]] = {}
    adders: dict[Literal, list[int]] = {}
    deleters: dict[Literal, list[int]] = {}
    state_items: set[tuple[str, Literal]] = set()

    for idx, action in enumerate(plan):
        for lit in action.pre_pos:
            providers = []
            if lit in init_state:
                providers.append(-1)
            providers.extend(i for i in range(idx) if lit in plan[i].add_eff)
            pos_supporters[(idx, lit)] = providers
            if lit[0] in state_preds and lit[0] not in resource_preds:
                state_items.add(("pos", lit))
                if lit not in use_pos:
                    use_pos[lit] = []
                use_pos[lit].append(idx)

        for lit in action.pre_neg:
            providers = []
            if lit not in init_state:
                providers.append(-1)
            providers.extend(i for i in range(idx) if lit in plan[i].del_eff)
            neg_supporters[(idx, lit)] = providers
            if lit[0] in state_preds and lit[0] not in resource_preds:
                state_items.add(("neg", lit))
                if lit not in use_neg:
                    use_neg[lit] = []
                use_neg[lit].append(idx)

        for lit in action.add_eff:
            if lit not in adders:
                adders[lit] = []
            adders[lit].append(idx)
        for lit in action.del_eff:
            if lit not in deleters:
                deleters[lit] = []
            deleters[lit].append(idx)

    for lit in goal_pos:
        if lit[0] in state_preds and lit[0] not in resource_preds:
            state_items.add(("pos", lit))
    for lit in goal_neg:
        if lit[0] in state_preds and lit[0] not in resource_preds:
            state_items.add(("neg", lit))

    state_items = sorted(state_items)

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
            if resource not in buckets:
                buckets[resource] = set()
            buckets[resource].add(obj)
        out: dict[ResourceSig, ObjectSig] = {}
        for resource, objects in buckets.items():
            if len(objects) == 1:
                out[resource] = next(iter(objects))
        return out

    def target_state_ready(target: int, state: set[Literal]) -> bool:
        action = plan[target]
        for lit in action.pre_pos:
            if lit[0] in state_preds and lit[0] not in resource_preds and lit not in state:
                return False
        for lit in action.pre_neg:
            if lit[0] in state_preds and lit[0] not in resource_preds and lit in state:
                return False
        return True

    def state_targets(state: set[Literal], done_mask: int) -> dict[int, float]:
        targets: dict[int, float] = {}
        done_steps = [i for i in range(n) if done_mask & (1 << i)]

        for sign, lit in state_items:
            if sign == "pos":
                if lit not in state:
                    continue
                use_list = use_pos.get(lit, [])
                establish_list = adders.get(lit, [])
                flip_list = deleters.get(lit, [])
                goal_needed = lit in goal_pos
            else:
                if lit in state:
                    continue
                use_list = use_neg.get(lit, [])
                establish_list = deleters.get(lit, [])
                flip_list = adders.get(lit, [])
                goal_needed = lit in goal_neg

            started = False
            for i in done_steps:
                if i in use_list or i in establish_list:
                    started = True
                    break
            if not started:
                continue

            target: int | None = None
            is_support = False
            for j in use_list:
                if not (done_mask & (1 << j)):
                    target = j
                    is_support = True
                    break

            if target is None and not goal_needed:
                for j in flip_list:
                    if not (done_mask & (1 << j)):
                        target = j
                        break

            if target is None:
                continue

            weight = 1.0
            if is_support and not target_state_ready(target, state):
                weight = 0.0

            if target not in targets or weight > targets[target]:
                targets[target] = weight

        return targets

    def make_distance(state: set[Literal], done_mask: int):
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
                best: frozenset[int] | None = None
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
        holding_lit = occupancy_model.build(resource, obj)
        best: int | None = None
        for j, action in enumerate(plan):
            if done_mask & (1 << j):
                continue
            if holding_lit not in action.del_eff:
                continue
            d = dist_fn(j)
            if d is None:
                continue
            cand = d + 1
            if best is None or cand < best:
                best = cand
        return best

    def rank_ready_actions(state: set[Literal], done_mask: int) -> list[int]:
        ready: list[int] = []
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
        state_targets_now = state_targets(state, done_mask)
        holding_now = current_holding(state)
        scored: list[tuple[tuple, int]] = []

        for i in ready:
            next_state = apply_action(state, plan[i])
            next_mask = done_mask | (1 << i)
            dist_next = make_distance(next_state, next_mask)
            state_targets_next = state_targets(next_state, next_mask)
            holding_next = current_holding(next_state)

            progress = 0.0
            for target in state_targets_now:
                before = dist_now(target)
                before = None if before is None else before + 1
                if before is None:
                    continue
                after = 0 if target == i else dist_next(target)
                after = after if target == i or after is None else after + 1
                if after is None:
                    continue
                gain = potential(after) - potential(before)
                if target != i:
                    gain *= state_targets_next.get(target, 0.0)
                if gain > 0:
                    progress += gain

            for resource, obj in holding_now.items():
                before = finish_holding(state, dist_now, done_mask, resource, obj)
                after = finish_holding(next_state, dist_next, next_mask, resource, obj)
                if before is None or after is None:
                    continue
                gain = potential(after) - potential(before)
                if gain > 0:
                    progress += gain

            bonus = 0.0
            for target, weight in state_targets_next.items():
                if target in state_targets_now:
                    continue
                d = dist_next(target)
                if d is None:
                    continue
                bonus = max(bonus, potential(d + 1) * weight)

            for resource, obj in holding_next.items():
                if holding_now.get(resource) == obj:
                    continue
                d = finish_holding(next_state, dist_next, next_mask, resource, obj)
                if d is None:
                    continue
                bonus = max(bonus, potential(d))

            scored.append(((1 if progress > 0 else 0, progress, bonus, -i), i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scored]

    def dfs(done_mask: int, state: frozenset[Literal]) -> list[int] | None:
        if done_mask == done_all:
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
# Main entry
# =========================
def plan_reorder(domain_path: Path, problem_path: Path, plan_path: Path, output_path: Path) -> None:
    try:
        ensure_supported_pddl(domain_path)
        ensure_supported_pddl(problem_path)

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
        preds = build_predecessors(init_state, goal_pos, goal_neg, plan, occupancy_model)
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

    except NotImplementedError as e:
        print(f"[plan_reorder] {e}. Skip reordering and keep original plan.")
        output_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    round_dir = Path("/home/xyx/下载/swm/test_pddl")
    plan_reorder(
        domain_path=round_dir / "temp_domain.pddl",
        problem_path=round_dir / "temp_problem.pddl",
        plan_path=round_dir / "plan.txt",
        output_path=round_dir / "plan_reorder.txt",
    )