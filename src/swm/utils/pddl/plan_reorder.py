from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
Literal = tuple[str, ...]

@dataclass
class ActionSchema:
    name: str
    params: list[str]
    pre_pos: set[Literal]
    add_eff: set[Literal]
    del_eff: set[Literal]


@dataclass
class GroundAction:
    idx: int
    name: str
    args: list[str]
    pre_pos: set[Literal]
    add_eff: set[Literal]
    del_eff: set[Literal]

    def to_line(self) -> str:
        return f"({self.name} {' '.join(self.args)})"

# =========================
# PDDL 解析
# =========================
def load_sexpr(path: Path):
    text = re.sub(r";[^\n]*", "", path.read_text(encoding="utf-8")).lower()
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()

    def parse():
        tok = tokens.pop(0)
        if tok == "(":
            out = []
            while tokens[0] != ")":
                out.append(parse())
            tokens.pop(0)
            return out
        if tok == ")":
            raise ValueError(f"Unexpected ')' in {path}")
        return tok

    root = parse()
    if tokens:
        raise ValueError(f"Unparsed tokens remain in {path}")
    return root


def strip_types(tokens: list[str]) -> list[str]:
    out = []
    skip = False
    for tok in tokens:
        if tok == "-":
            skip = True
        elif skip:
            skip = False
        else:
            out.append(tok)
    return out


def parse_literals(expr) -> tuple[set[Literal], set[Literal]]:
    if expr is None:
        return set(), set()
    if isinstance(expr, str):
        raise ValueError(f"Unexpected atom: {expr}")
    if expr[0] == "and":
        pos, neg = set(), set()
        for sub in expr[1:]:
            p, n = parse_literals(sub)
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
        params = []
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

        pre_pos, _ = parse_literals(precondition)
        add_eff, del_eff = parse_literals(effect)
        schemas[name] = ActionSchema(name, params, pre_pos, add_eff, del_eff)

    return schemas


def parse_problem(path: Path) -> tuple[set[str], set[Literal], set[Literal]]:
    root = load_sexpr(path)
    if root[0] != "define":
        raise ValueError(f"{path} is not a valid problem file")

    objects = set()
    init_state = set()
    goal = set()

    for item in root[1:]:
        if not isinstance(item, list) or not item:
            continue
        if item[0] == ":objects":
            objects |= set(strip_types(item[1:]))
        elif item[0] == ":init":
            for lit in item[1:]:
                init_state.add(tuple(lit))
        elif item[0] == ":goal":
            goal, _ = parse_literals(item[1])

    return objects, init_state, goal


def parse_plan(path: Path) -> tuple[list[tuple[str, list[str]]], list[str]]:
    raw_plan = []
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
        raw_plan.append((parts[0], parts[1:]))

    return raw_plan, comments


# =========================
# grounding + 状态执行
# =========================
def ground_plan(raw_plan: list[tuple[str, list[str]]], schemas: dict[str, ActionSchema]) -> list[GroundAction]:
    plan = []

    for idx, (name, args) in enumerate(raw_plan):
        if name not in schemas:
            raise KeyError(f"Action '{name}' not found in domain")

        schema = schemas[name]
        if len(args) != len(schema.params):
            raise ValueError(
                f"Arity mismatch for action {name}: expected {len(schema.params)}, got {len(args)}"
            )

        sub = dict(zip(schema.params, args))
        plan.append(
            GroundAction(
                idx=idx,
                name=name,
                args=args,
                pre_pos={tuple(sub.get(x, x) for x in lit) for lit in schema.pre_pos},
                add_eff={tuple(sub.get(x, x) for x in lit) for lit in schema.add_eff},
                del_eff={tuple(sub.get(x, x) for x in lit) for lit in schema.del_eff},
            )
        )

    return plan


def apply_action(state: set[Literal], step: GroundAction) -> set[Literal]:
    if not step.pre_pos <= state:
        missing = sorted(step.pre_pos - state)
        raise ValueError(f"Action not applicable: {step.to_line()}\nMissing preconditions: {missing}")

    nxt = set(state)
    nxt.difference_update(step.del_eff)
    nxt.update(step.add_eff)
    return nxt


def rollout(init_state: set[Literal], plan: list[GroundAction]) -> set[Literal]:
    state = set(init_state)
    for step in plan:
        state = apply_action(state, step)
    return state


# =========================
# hand / holding 推理
# =========================
def infer_holding(
    objects: set[str],
    init_state: set[Literal],
    plan: list[GroundAction],
) -> tuple[set[str], str | None, int | None, int | None]:
    # 先推断哪些 object 是手
    hands = {x for x in objects if any(tok in x for tok in ("hand", "gripper", "arm"))}
    if not hands:
        for lit in init_state:
            for x in lit[1:]:
                if any(tok in x for tok in ("hand", "gripper", "arm")):
                    hands.add(x)

    if not hands:
        return hands, None, None, None

    # rollout 一次，holding 推理全部基于整条状态轨迹
    states = [set(init_state)]
    state = set(init_state)
    for step in plan:
        state = apply_action(state, step)
        states.append(set(state))

    binary_preds = {lit[0] for s in states for lit in s if len(lit) == 3}

    def has_valid_example(pred: str, hand_pos: int, obj_pos: int) -> bool:
        for s in states:
            for lit in s:
                if lit[0] != pred or len(lit) != 3:
                    continue
                hand = lit[1 + hand_pos]
                obj = lit[1 + obj_pos]
                if hand in hands and obj not in hands:
                    return True
        return False

    def score(pred: str, hand_pos: int, obj_pos: int) -> tuple[int, int, int, int]:
        matches = 0
        changes = 0
        hand_conflicts = 0
        obj_conflicts = 0

        for s in states:
            hand_to_objs = {}
            obj_to_hands = {}

            for lit in s:
                if lit[0] != pred or len(lit) != 3:
                    continue
                hand = lit[1 + hand_pos]
                obj = lit[1 + obj_pos]
                if hand not in hands or obj in hands:
                    continue

                matches += 1
                hand_to_objs.setdefault(hand, set()).add(obj)
                obj_to_hands.setdefault(obj, set()).add(hand)

            hand_conflicts += sum(len(v) - 1 for v in hand_to_objs.values() if len(v) > 1)
            obj_conflicts += sum(len(v) - 1 for v in obj_to_hands.values() if len(v) > 1)

        for step in plan:
            for lit in step.add_eff | step.del_eff:
                if lit[0] != pred or len(lit) != 3:
                    continue
                hand = lit[1 + hand_pos]
                obj = lit[1 + obj_pos]
                if hand in hands and obj not in hands:
                    changes += 1

        return matches, changes, -hand_conflicts, -obj_conflicts

    candidates = []

    # 第一层：优先标准 holding
    if "holding" in binary_preds:
        for hand_pos, obj_pos in ((0, 1), (1, 0)):
            if has_valid_example("holding", hand_pos, obj_pos):
                candidates.append((1, score("holding", hand_pos, obj_pos), "holding", hand_pos, obj_pos))

    # 第二层：没有标准 holding 时，再自动猜最像 holding 的二元谓词
    if not candidates:
        for pred in binary_preds:
            for hand_pos, obj_pos in ((0, 1), (1, 0)):
                if has_valid_example(pred, hand_pos, obj_pos):
                    candidates.append((0, score(pred, hand_pos, obj_pos), pred, hand_pos, obj_pos))

    if not candidates:
        return hands, None, None, None

    candidates.sort(reverse=True)
    _, _, pred, hand_pos, obj_pos = candidates[0]
    return hands, pred, hand_pos, obj_pos


# =========================
# 构建前驱约束 DAG
# =========================
def build_predecessors(init_state: set[Literal], goal: set[Literal], plan: list[GroundAction]) -> dict[int, set[int]]:
    def latest_supporter(upto: int, lit: Literal) -> int | None:
        provider = -1 if lit in init_state else None
        for i in range(upto):
            if lit in plan[i].add_eff:
                provider = i
            if lit in plan[i].del_eff:
                provider = None
        return provider

    # 先直接在这里构建 support records，不再单独拆函数
    support_records = []
    for j, step in enumerate(plan):
        for lit in step.pre_pos:
            provider = latest_supporter(j, lit)
            if provider is None:
                raise ValueError(
                    f"Original plan invalid: no supporter for precondition {lit} of step {j}: {step.to_line()}"
                )
            support_records.append((provider, j, lit))

    preds = {i: set() for i in range(len(plan))}

    def add_edge(i: int, j: int):
        if i != j:
            preds[j].add(i)

    # 1. support edge
    for provider, consumer, _ in support_records:
        if provider >= 0:
            add_edge(provider, consumer)

    # 2. threat protection
    for provider, consumer, lit in support_records:
        for k, step in enumerate(plan):
            if k == consumer or lit not in step.del_eff:
                continue

            if provider == -1:
                if k < consumer:
                    add_edge(consumer, k)
            else:
                if k < provider:
                    add_edge(k, provider)
                elif k > consumer:
                    add_edge(consumer, k)

    # 3. goal protection
    for lit in goal:
        provider = latest_supporter(len(plan), lit)
        if provider is None:
            raise ValueError(f"Original plan invalid: goal {lit} not achieved")

        if provider >= 0:
            for k, step in enumerate(plan):
                if lit in step.del_eff and k < provider:
                    add_edge(k, provider)

    return preds


# =========================
# DFS 搜索重排
# =========================
def reorder_by_search(
    init_state: set[Literal],
    goal: set[Literal],
    plan: list[GroundAction],
    preds: dict[int, set[int]],
    hands: set[str],
    holding_pred: str | None,
    hand_pos: int | None,
    obj_pos: int | None,
) -> list[GroundAction]:
    def holding_map(lits: set[Literal]) -> dict[str, str]:
        if holding_pred is None or hand_pos is None or obj_pos is None:
            return {}
        out = {}
        for lit in lits:
            if lit[0] != holding_pred or len(lit) != 3:
                continue
            hand = lit[1 + hand_pos]
            obj = lit[1 + obj_pos]
            if hand in hands and obj not in hands:
                out[hand] = obj
        return out

    def step_uses_hand(step: GroundAction, hand: str) -> bool:
        if hand in step.args:
            return True
        return any(hand in lit[1:] for lit in step.pre_pos | step.add_eff | step.del_eff)

    def step_mentions_obj(step: GroundAction, obj: str) -> bool:
        if obj in step.args:
            return True
        return any(obj in lit[1:] for lit in step.pre_pos | step.add_eff | step.del_eff)

    def step_holding_objs(step: GroundAction) -> set[str]:
        if holding_pred is None or hand_pos is None or obj_pos is None:
            return set()

        objs = set()
        for lit in step.pre_pos | step.add_eff | step.del_eff:
            if lit[0] != holding_pred or len(lit) != 3:
                continue
            hand = lit[1 + hand_pos]
            obj = lit[1 + obj_pos]
            if hand in hands and obj not in hands:
                objs.add(obj)
        return objs

    def future_needs_obj(unscheduled: set[int], current_idx: int, obj: str) -> bool:
        for j in unscheduled:
            if j == current_idx:
                continue

            step = plan[j]
            if not step_mentions_obj(step, obj):
                continue

            if obj in step_holding_objs(step):
                return True

            if any(x in hands for x in step.args):
                return True

            if any(any(x in hands for x in lit[1:]) for lit in step.pre_pos | step.add_eff | step.del_eff):
                return True

        return False

    all_mask = (1 << len(plan)) - 1
    failed = set()

    def select_ready(state: set[Literal], done_mask: int) -> list[int]:
        unscheduled = {i for i in range(len(plan)) if not (done_mask & (1 << i))}
        ready = [
            i for i in unscheduled
            if all(done_mask & (1 << p) for p in preds[i]) and plan[i].pre_pos <= state
        ]
        if not ready:
            return []

        held_now = holding_map(state)

        # 规则 1：手里正拿着某个物体，就优先继续做这只手/这个物体的链条
        if held_now:
            cont = []
            for i in ready:
                for hand, obj in held_now.items():
                    if step_uses_hand(plan[i], hand) and step_mentions_obj(plan[i], obj):
                        cont.append(i)
                        break
            if cont:
                return sorted(cont)

        # 规则 2：如果某步能安全放下当前物体，就优先收尾
        cleanup = []
        for i in ready:
            removed = holding_map(plan[i].del_eff)
            for hand, obj in removed.items():
                if hand in held_now and held_now[hand] == obj and not future_needs_obj(unscheduled, i, obj):
                    cleanup.append(i)
                    break
        if cleanup:
            return sorted(cleanup)

        return sorted(ready)

    def dfs(done_mask: int, state_fs: frozenset[Literal]):
        if done_mask == all_mask:
            return [] if goal <= set(state_fs) else None

        key = (done_mask, state_fs)
        if key in failed:
            return None

        state = set(state_fs)
        ready = select_ready(state, done_mask)
        if not ready:
            failed.add(key)
            return None

        for i in ready:
            next_state = apply_action(state, plan[i])
            suffix = dfs(done_mask | (1 << i), frozenset(next_state))
            if suffix is not None:
                return [i] + suffix

        failed.add(key)
        return None

    order = dfs(0, frozenset(init_state))
    if order is None:
        return plan

    reordered = [plan[i] for i in order]
    return reordered if goal <= rollout(init_state, reordered) else plan


# =========================
# 主流程
# =========================
def plan_reorder(domain_path: Path, problem_path: Path, plan_path: Path, output_path: Path) -> None:
    schemas = parse_domain(domain_path)
    objects, init_state, goal = parse_problem(problem_path)
    raw_plan, comments = parse_plan(plan_path)
    plan = ground_plan(raw_plan, schemas)

    original_final = rollout(init_state, plan)
    if not goal <= original_final:
        raise ValueError(f"Original plan does not satisfy goal. Missing: {sorted(goal - original_final)}")

    hands, holding_pred, hand_pos, obj_pos = infer_holding(objects, init_state, plan)
    preds = build_predecessors(init_state, goal, plan)

    reordered = reorder_by_search(
        init_state=init_state,
        goal=goal,
        plan=plan,
        preds=preds,
        hands=hands,
        holding_pred=holding_pred,
        hand_pos=hand_pos,
        obj_pos=obj_pos,
    )

    final_state = rollout(init_state, reordered)
    if not goal <= final_state:
        raise ValueError(f"Final reordered plan does not satisfy goal. Missing: {sorted(goal - final_state)}")

    lines = [step.to_line() for step in reordered] + comments
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    ROUND_DIR = Path(
"/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/eval_results/gemini-3-flash-preview/agibot/task_351/episode_776577/round1"        
        
        
        )
    DOMAIN_PATH = ROUND_DIR / "domain.pddl"
    PROBLEM_PATH = ROUND_DIR / "problem.pddl"
    PLAN_PATH = ROUND_DIR / "plan.txt"
    OUTPUT_PATH = ROUND_DIR / "plan_reorder.txt"
    plan_reorder(DOMAIN_PATH, PROBLEM_PATH, PLAN_PATH, OUTPUT_PATH)