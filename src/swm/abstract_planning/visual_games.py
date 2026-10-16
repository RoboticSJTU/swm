"""Independent visual game variants with explicit native states and unit actions.

No benchmark test instances are used. Randomness for appearance is independent
of task generation; native transitions do not interpret PDDL.
"""
from __future__ import annotations

import copy
import itertools
import random
from pathlib import Path

from .model import action, make_domain, make_problem, parse_sexpr, section, native_plan

GAMES = ("cooperative_kitchen", "rush_hour", "color_sort", "blocks_3d", "numeric_sliding")
COLORS = {"red": "#c85650", "blue": "#4789c6", "yellow": "#dfb841",
          "green": "#70a767", "purple": "#aa75b6"}
LETTERS = {"red": "R", "blue": "B", "yellow": "Y", "green": "G", "purple": "P"}
SEEDS = {g: 2046101000 + i * 100000 for i, g in enumerate(GAMES)}
LENGTHS = {"cooperative_kitchen": ((12, 30), (31, 50), (51, 100)),
           "rush_hour": ((3, 8), (9, 15), (16, 32)),
           "color_sort": ((3, 10), (11, 20), (21, 50)),
           "blocks_3d": ((4, 14), (15, 28), (29, 64)),
           "numeric_sliding": ((4, 10), (11, 18), (19, 28))}
ROOT = Path(__file__).resolve().parents[3]


def cell(r, c):
    return f"r{r + 1:02d}c{c + 1:02d}"


def rc(p):
    return int(p[1:3]) - 1, int(p[4:6]) - 1


def adjacent(a, b):
    x, y = rc(a), rc(b)
    return abs(x[0] - y[0]) + abs(x[1] - y[1]) == 1


def categories(name, values):
    return [[name, v] for v in values]


def make(game, candidate, spec, state, objects, static, goal, actions, rules, requirements=()):
    # Existing formatter preserves identities and uses one line per condition.
    import sharegpt
    task = dict(game=game, candidate=candidate, seed=SEEDS[game] + candidate,
                spec=spec, state=state)
    initial = static + [list(a) for a in sorted(projection(task, state))]
    domain = parse_sexpr(make_domain(game, actions, initial, goal))
    section(domain, ":requirements")[:] = [":requirements", ":strips", *requirements]
    problem = parse_sexpr(make_problem(game, objects, initial, goal))
    problem[1][1] = f"{game}_{candidate:06d}"
    task["domain"] = sharegpt.format_domain(domain, problem, [a[1] for a in actions], preserve_identity=True) + "\n"
    task["problem"] = sharegpt.format_problem(problem, preserve_identity=True) + "\n"
    from .instructions import operation_list
    task["instruction"] = rules + "\n" + operation_list(task["domain"])
    visual_rng = random.Random(3046101000 + GAMES.index(game) * 100000 + candidate)
    spec["appearance"] = dict(style=visual_rng.choice(["flat", "bevel", "outline", "night"]),
                              scale=visual_rng.choice([112, 132, 152]),
                              camera=visual_rng.choice([0, 1, 2]),
                              palette=visual_rng.randrange(5), spacing=visual_rng.randrange(3))
    return task


def kitchen(candidate, difficulty, size=None):
    rng = random.Random(SEEDS["cooperative_kitchen"] + candidate)
    h = rng.choice([3, 4, 5]); w = rng.choice([5, 6, 7])
    if size: h, w = size["board"]
    barrier = rng.randrange(2, w - 2)
    vertical = rng.choice([True, False])
    def loc(r, c):
        return cell(r, c) if vertical else cell(c, r)
    counter_r = rng.randrange(h)
    floor_coords = [(r, c) for r in range(h) for c in range(w) if c != barrier]
    # Obstacles are kept only when each region stays connected.
    def connected(coords):
        if not coords: return False
        reached = {coords[0]}
        while True:
            new = reached | {x for x in coords if any(abs(x[0]-y[0])+abs(x[1]-y[1]) == 1 for y in reached)}
            if new == reached: return len(new) == len(coords)
            reached = new
    for _ in range(rng.randrange(3)):
        choices = [x for x in floor_coords if x not in [(counter_r, barrier-1), (counter_r, barrier+1)]]
        remove = rng.choice(choices)
        remaining = [x for x in floor_coords if x != remove]
        if all(connected([x for x in remaining if (x[1] < barrier) == side]) for side in [True, False]):
            floor_coords = remaining
    left = [loc(*x) for x in floor_coords if x[1] < barrier]
    right = [loc(*x) for x in floor_coords if x[1] > barrier]
    n = rng.choice([[1, 2], [2, 3], [3, 4]][difficulty])
    foods = [f"f{i+1}" for i in range(n)]
    positions = rng.sample(left, min(len(left), n + 2))
    chop = positions[0]; plate, delivery, chef2 = rng.sample(right, 3)
    spec = dict(h=h if vertical else w, w=w if vertical else h,
                floors=[loc(*x) for x in floor_coords], counter=loc(counter_r, barrier),
                chop=chop, delivery=delivery, foods=foods,
                required=foods[:], hands={"a1": "hand1", "a2": "hand2"},
                food_kinds={f: rng.choice(["tomato", "lettuce", "onion"]) for f in foods})
    state = dict(chefs={"a1": positions[1], "a2": chef2}, holding={"hand1": None, "hand2": None},
                 food_places={f: ["at", positions[(i+2) % len(positions)]] for i, f in enumerate(foods)},
                 prepared=rng.sample(foods, rng.randrange(min(n, 2))), plate_place=["at", plate], served=False)
    if rng.random() < .2:
        f = rng.choice(foods); state["food_places"][f] = ["on", "pass1"]
    if rng.random() < .18:
        f = rng.choice(foods)
        if state["food_places"][f][0] == "at":
            state["food_places"][f] = ["held", "hand1"]; state["holding"]["hand1"] = f
    # Never make an easy task finishable solely by the second chef.
    if all(state["food_places"][f][0] == "on" and f in state["prepared"] for f in foods):
        state["prepared"] = []
    floors = spec["floors"]
    static = categories("chef", spec["hands"]) + categories("hand", spec["hands"].values())
    static += categories("food", foods) + [["plate", "p1"], ["counter", "pass1"]] + categories("location", floors)
    static += [["has_hand", a, h] for a, h in spec["hands"].items()]
    static += [["adjacent", a, b] for a in floors for b in floors if adjacent(a, b)]
    static += [["counter_reachable", a, "pass1"] for a in floors if adjacent(a, spec["counter"])]
    static += [["chop_station", chop], ["delivery_station", delivery]] + [["required", f, "p1"] for f in foods]
    actions = [
        action("move", "?chef ?from ?to", "(and (chef ?chef) (location ?from) (location ?to) (at ?chef ?from) (adjacent ?from ?to))", "(and (not (at ?chef ?from)) (at ?chef ?to))"),
        action("pickup_ingredient", "?food ?loc ?hand ?chef", "(and (food ?food) (location ?loc) (hand ?hand) (chef ?chef) (has_hand ?chef ?hand) (at ?chef ?loc) (at ?food ?loc) (hand_free ?hand))", "(and (not (at ?food ?loc)) (not (hand_free ?hand)) (holding ?hand ?food))"),
        action("chop", "?food ?loc ?hand ?chef", "(and (food ?food) (location ?loc) (hand ?hand) (chef ?chef) (has_hand ?chef ?hand) (at ?chef ?loc) (chop_station ?loc) (holding ?hand ?food) (not (prepared ?food)))", "(prepared ?food)"),
        action("put_on_counter", "?food ?counter ?loc ?hand ?chef", "(and (food ?food) (counter ?counter) (location ?loc) (hand ?hand) (chef ?chef) (has_hand ?chef ?hand) (at ?chef ?loc) (counter_reachable ?loc ?counter) (holding ?hand ?food) (counter_free ?counter))", "(and (not (holding ?hand ?food)) (hand_free ?hand) (not (counter_free ?counter)) (on ?food ?counter))"),
        action("take_from_counter", "?food ?counter ?loc ?hand ?chef", "(and (food ?food) (counter ?counter) (location ?loc) (hand ?hand) (chef ?chef) (has_hand ?chef ?hand) (at ?chef ?loc) (counter_reachable ?loc ?counter) (hand_free ?hand) (on ?food ?counter))", "(and (not (on ?food ?counter)) (counter_free ?counter) (not (hand_free ?hand)) (holding ?hand ?food))"),
        action("put_on_plate", "?food ?plate ?loc ?hand ?chef", "(and (food ?food) (plate ?plate) (location ?loc) (hand ?hand) (chef ?chef) (has_hand ?chef ?hand) (at ?chef ?loc) (holding ?hand ?food) (prepared ?food) (at ?plate ?loc) (required ?food ?plate))", "(and (not (holding ?hand ?food)) (hand_free ?hand) (on ?food ?plate))"),
        action("pickup_plate", "?plate ?loc ?hand ?chef", "(and (plate ?plate) (location ?loc) (hand ?hand) (chef ?chef) (has_hand ?chef ?hand) (at ?chef ?loc) (at ?plate ?loc) (hand_free ?hand) (forall (?food) (imply (required ?food ?plate) (on ?food ?plate))))", "(and (not (at ?plate ?loc)) (not (hand_free ?hand)) (holding ?hand ?plate))"),
        action("deliver", "?plate ?loc ?hand ?chef", "(and (plate ?plate) (location ?loc) (hand ?hand) (chef ?chef) (has_hand ?chef ?hand) (at ?chef ?loc) (delivery_station ?loc) (holding ?hand ?plate))", "(and (served ?plate) (at ?plate ?loc) (not (holding ?hand ?plate)) (hand_free ?hand))"),
    ]
    rules = (f"Deliver salad p1 containing {', '.join(foods)}, all chopped.\n"
             "Rules: Chefs move one orthogonal floor cell; dark walls and pass1 block walking. "
             "Chef aN uses handN and holds one item. Pick up items in the current cell. "
             "Chop held food at CHOP; a slash on food means already chopped. "
             "The single-slot pass1 accepts or releases one food to a chef on an adjacent floor cell. "
             "Put required chopped food on p1 in the current cell. Pick up p1 only after all ingredients are on it, then deliver at SERVE. "
             "Actions are sequential. Small labeled items within a chef's hand badge are held; plate contents move with p1. "
             "Coordinates start at r01c01 in the top-left.")
    return make("cooperative_kitchen", candidate, spec, state, ["a1", "a2", "hand1", "hand2", "p1", "pass1", *foods, *floors],
                static, [["served", "p1"]], actions, rules, [":negative-preconditions", ":disjunctive-preconditions", ":universal-preconditions"])


def footprint(car, anchor):
    r, c = rc(anchor)
    return [cell(r + j*(car["axis"] == "vertical"), c + j*(car["axis"] == "horizontal")) for j in range(car["length"])]


def rush(candidate, difficulty, size=None):
    from build_visual_game_pilots import rush_distance
    rng = random.Random(SEEDS["rush_hour"] + candidate)
    for _ in range(60):
        h, w = rng.choice([(5, 5), (5, 6), (6, 6), (6, 7), (7, 7)])
        if size: h, w = size["board"]
        row = rng.randrange(1, h - 1)
        cars = {"v0": dict(axis="horizontal", length=2)}
        state = {"v0": cell(row, rng.randrange(min(2, w-3)))}
        for col in rng.sample(list(range(rc(state["v0"])[1]+2, w)), rng.choice([1, 2])):
            length = rng.choice([2, 3]); start = rng.randrange(max(0, row-length+1), min(row, h-length)+1)
            name = f"v{len(cars)}"; cars[name] = dict(axis="vertical", length=length); state[name] = cell(start, col)
        wanted = rng.choice([[4, 5, 6], [6, 7, 8, 9], [9, 10, 11, 12]][difficulty])
        for _ in range(wanted-len(cars)):
            taken = {p for v, a in state.items() for p in footprint(cars[v], a)}
            choices = []
            for axis, length, r, c in itertools.product(["horizontal", "vertical"], [2, 3], range(h), range(w)):
                car = dict(axis=axis, length=length); span = footprint(car, cell(r, c))
                if all(0 <= rc(p)[0] < h and 0 <= rc(p)[1] < w for p in span) and not set(span) & taken:
                    choices.append((car, cell(r, c)))
            if choices:
                car, a = rng.choice(choices); name = f"v{len(cars)}"; cars[name] = car; state[name] = a
        spec = dict(h=h, w=w, cars=cars, target_row=row)
        distance = rush_distance(spec, state, limit=120000)
        lo, hi = LENGTHS["rush_hour"][difficulty]
        if distance is not None and lo <= distance <= hi: break
    else: raise ValueError("No board in the requested optimal-distance band")
    spec["shortest_native_plan"] = distance
    cells = [cell(r, c) for r in range(h) for c in range(w)]
    static = categories("vehicle", cars) + categories("location", cells) + [["target", "v0"]]
    for v, car in cars.items():
        r, c = rc(state[v])
        anchors = [cell(r, j) for j in range(w-car["length"]+1)] if car["axis"] == "horizontal" else [cell(j, c) for j in range(h-car["length"]+1)]
        for a, b in itertools.permutations(anchors, 2):
            if adjacent(a, b):
                old, new = set(footprint(car, a)), set(footprint(car, b))
                static.append(["shift", v, a, b, next(iter(old-new)), next(iter(new-old))])
    anchor = cell(row, w-2); static.append(["exit_pattern", "v0", anchor, *footprint(cars["v0"], anchor)])
    actions = [
        action("move_vehicle", "?vehicle ?from ?to ?rear ?front", "(and (vehicle ?vehicle) (location ?from) (location ?to) (location ?rear) (location ?front) (at ?vehicle ?from) (shift ?vehicle ?from ?to ?rear ?front) (empty ?front))", "(and (not (at ?vehicle ?from)) (at ?vehicle ?to) (empty ?rear) (not (empty ?front)))"),
        action("exit_vehicle", "?vehicle ?from ?rear ?front", "(and (vehicle ?vehicle) (location ?from) (location ?rear) (location ?front) (target ?vehicle) (at ?vehicle ?from) (exit_pattern ?vehicle ?from ?rear ?front))", "(and (not (at ?vehicle ?from)) (escaped ?vehicle) (empty ?rear) (empty ?front))"),
    ]
    rules = ("Move red vehicle v0 out through EXIT.\nRules: Each vehicle moves one cell along its long axis, "
             "keeping its shown length, orientation and lane. The new front cell must be empty; the old rear becomes empty. "
             "Vehicles cannot turn or overlap. Only v0 can exit, after reaching the right edge; exiting frees its two cells. "
             "Coordinates start at r01c01 in the top-left. A vehicle's position is its topmost or leftmost occupied cell. "
             "Derive shift and exit patterns from these grid rules.")
    return make("rush_hour", candidate, spec, state, [*cars, *cells], static, [["escaped", "v0"]], actions, rules)


def slot(bolt, height):
    return f"{bolt}s{height}"


def color_sort(candidate, difficulty, size=None):
    rng = random.Random(SEEDS["color_sort"] + candidate)
    capacity = rng.choice([[2, 3], [3, 4], [3, 4]][difficulty])
    if size: capacity = size["capacity"]
    nc = rng.choice([[2, 3], [3, 4], [4, 5]][difficulty])
    colors_used = rng.sample(list(COLORS), nc)
    nuts = [f"n{i+1}" for i in range(nc*capacity)]; colors = dict(zip(nuts, [c for c in colors_used for _ in range(capacity)]))
    order = nuts[:]; rng.shuffle(order)
    piles = [order[i:i+capacity] for i in range(0, len(order), capacity)] + [[], []]
    rng.shuffle(piles)
    state = {f"b{i+1}": p for i, p in enumerate(piles)}
    # Some partially filled starts are reached by legal moves, preserving solvability.
    spec = dict(capacity=capacity, colors=colors, bolts=list(state),
                targets=dict(zip(colors_used, rng.sample(list(state), nc))))
    for _ in range(rng.randrange(4)):
        options = color_successors(spec, state)
        if options: state = rng.choice(list(options.values()))
    slots = [slot(b, i) for b in state for i in range(capacity+1)]
    static = categories("nut", nuts) + categories("bolt", state) + categories("slot", slots)
    static += categories("color", colors_used) + [["has_color", n, c] for n, c in colors.items()]
    static += [["same_color", a, b] for a in nuts for b in nuts if colors[a] == colors[b]]
    static += [["slot_of", slot(b, i), b] for b in state for i in range(capacity+1)]
    static += [["next_slot", slot(b, i), slot(b, i+1)] for b in state for i in range(capacity)]
    static += [["base", slot(b, 0)] for b in state]
    actions = [
        action("move_to_empty", "?nut ?source ?target ?from ?source_after ?target_before ?to",
               "(and (nut ?nut) (bolt ?source) (bolt ?target) (slot ?from) (slot ?source_after) (slot ?target_before) (slot ?to) (not (= ?source ?target)) (slot_of ?from ?source) (slot_of ?to ?target) (top ?source ?from) (at ?nut ?from) (next_slot ?source_after ?from) (top ?target ?target_before) (base ?target_before) (next_slot ?target_before ?to))",
               "(and (not (at ?nut ?from)) (at ?nut ?to) (not (in ?nut ?source)) (in ?nut ?target) (not (top ?source ?from)) (top ?source ?source_after) (not (top ?target ?target_before)) (top ?target ?to))"),
        action("move_onto", "?nut ?support ?source ?target ?from ?source_after ?target_before ?to",
               "(and (nut ?nut) (nut ?support) (bolt ?source) (bolt ?target) (slot ?from) (slot ?source_after) (slot ?target_before) (slot ?to) (not (= ?source ?target)) (slot_of ?from ?source) (slot_of ?to ?target) (top ?source ?from) (at ?nut ?from) (next_slot ?source_after ?from) (top ?target ?target_before) (at ?support ?target_before) (same_color ?nut ?support) (next_slot ?target_before ?to))",
               "(and (not (at ?nut ?from)) (at ?nut ?to) (not (in ?nut ?source)) (in ?nut ?target) (not (top ?source ?from)) (top ?source ?source_after) (not (top ?target ?target_before)) (top ?target ?to))"),
    ]
    goal = [["in", n, spec["targets"][colors[n]]] for n in nuts]
    rules = ("Sort by color: " + "; ".join(f"{c} on {b}" for c, b in spec["targets"].items()) + ".\n"
             f"Rules: Each bolt holds at most {capacity} nuts. One move transfers exactly one top nut to a different bolt, "
             "either empty or topped by a nut of the same color. Nuts below the top cannot move. "
             "Nut IDs and color letters are printed on the nuts (R red, B blue, Y yellow, G green, P purple). "
             f"Slot bNs0 is bolt bN's empty base; bNs1 through bNs{capacity} are its bottom-to-top slots. "
             "All nuts of each specified color must end on its target bolt; nut order within a color is unrestricted.")
    return make("color_sort", candidate, spec, state, [*nuts, *state, *slots, *colors_used], static, goal, actions, rules, [":negative-preconditions", ":equality"])


def supports(piles):
    return {b: lower for p in piles for b, lower in zip(p, ["table", *p[:-1]])}


def block_piles(rng, names, roots=None):
    order = rng.sample(names, len(names)); roots = roots or rng.randrange(max(2, (len(names)+4)//5), min(5, len(names))+1)
    piles = [[b] for b in order[:roots]]
    for b in order[roots:]:
        rng.choice([p for p in piles if len(p) < 5]).append(b)
    return piles


def blocks(candidate, difficulty, size=None):
    rng = random.Random(SEEDS["blocks_3d"] + candidate)
    n = rng.choice([[4, 5, 6], [7, 8, 9], [10, 11, 12]][difficulty])
    if size: n = size["blocks"]
    names = [f"b{i+1}" for i in range(n)]
    piles = block_piles(rng, names)
    state = dict(on=supports(piles), held=None)
    if rng.random() < .12:
        held = rng.choice(piles).pop(); del state["on"][held]; state["held"] = held
    goal_on = supports(block_piles(rng, names))
    mode = rng.choice(["full", "full", "partial", "clear", "table"])
    if mode == "partial":
        keep = rng.sample(names, max(2, n//2)); goal_on = {b: goal_on[b] for b in keep}
    elif mode == "clear":
        goal_on = {}; clear_goal = rng.sample(names, max(2, n//2))
    elif mode == "table": goal_on = dict.fromkeys(names, "table")
    spec = dict(blocks=names, initial_roots=[p[0] for p in piles if p], goal_on=goal_on,
                clear_goal=clear_goal if mode == "clear" else [], goal_mode=mode)
    static = categories("block", names) + [["table", "table"], ["hand", "hand1"]]
    # Exactly the same domain/action interfaces as the existing 2D blocks family.
    actions = [
        action("unstack", "?ob ?underob ?h", "(and (block ?ob) (block ?underob) (hand ?h) (hand_free ?h) (clear ?ob) (on ?ob ?underob))", "(and (not (hand_free ?h)) (holding ?h ?ob) (not (clear ?ob)) (clear ?underob) (not (on ?ob ?underob)))"),
        action("putdown", "?ob ?h ?surface", "(and (block ?ob) (hand ?h) (table ?surface) (holding ?h ?ob))", "(and (not (holding ?h ?ob)) (hand_free ?h) (clear ?ob) (on ?ob ?surface))"),
        action("stack", "?ob ?underob ?h", "(and (block ?ob) (block ?underob) (hand ?h) (holding ?h ?ob) (clear ?underob))", "(and (not (holding ?h ?ob)) (hand_free ?h) (not (clear ?underob)) (clear ?ob) (on ?ob ?underob))"),
        action("pickup", "?ob ?h ?surface", "(and (block ?ob) (hand ?h) (table ?surface) (hand_free ?h) (clear ?ob) (on ?ob ?surface))", "(and (not (hand_free ?h)) (holding ?h ?ob) (not (clear ?ob)) (not (on ?ob ?surface)))"),
    ]
    goal = [["on", b, lower] for b, lower in goal_on.items()] + [["clear", b] for b in spec["clear_goal"]] + [["hand_free", "hand1"]]
    if mode == "clear": target = "Leave " + ", ".join(spec["clear_goal"]) + " clear."
    elif mode == "table": target = "Place every block directly on the table."
    else: target = "Arrange direct supports: " + "; ".join(f"{b} on {lower}" for b, lower in goal_on.items()) + "."
    rules = (target + " Finish with hand1 empty.\nRules: One hand holds one block. "
             "pickup lifts a clear block from the table with an empty hand; unstack lifts a clear block from its direct support. "
             "putdown places the held block on the table; stack places it on a clear block. Both free the hand. "
             "A block is clear when supported and has nothing directly above it. Any block can support any other block; "
             "color, size and table position impose no restrictions. The table is one support named table. "
             "The gripper is hand1; a block between its fingers is held.")
    return make("blocks_3d", candidate, spec, state, [*names, "table", "hand1"], static, goal, actions, rules)


def sliding(candidate, difficulty, size=None):
    rng = random.Random(SEEDS["numeric_sliding"] + candidate)
    h, w = rng.choice([[(2, 3), (2, 4), (3, 3)], [(3, 3), (3, 4), (4, 3), (4, 4)], [(3, 3), (3, 4), (4, 3), (4, 4)]][difficulty])
    if size: h, w = size["board"]
    cells = [cell(r, c) for r in range(h) for c in range(w)]; tiles = [f"t{i+1}" for i in range(len(cells)-1)]
    pattern = rng.choice(["row", "row", "column", "snake", "descending"])
    if pattern == "column": order = [cell(r, c) for c in range(w) for r in range(h)]
    elif pattern == "snake": order = [cell(r, c) for r in range(h) for c in (range(w) if r % 2 == 0 else range(w-1, -1, -1))]
    else: order = cells
    numbers = tiles[::-1] if pattern == "descending" else tiles
    goal_state = dict(zip(order, [*numbers, None])); state = dict(goal_state); previous = None
    moves = rng.randint(*[(4, 12), (12, 22), (22, 36)][difficulty])
    if size: moves = size.get("scramble", moves)
    for _ in range(moves):
        empty = next(p for p, t in state.items() if t is None)
        target = rng.choice([p for p in cells if adjacent(empty, p) and p != previous])
        state[empty], state[target] = state[target], None; previous = empty
    spec = dict(h=h, w=w, goal_state=goal_state, goal_pattern=pattern, scramble_steps=moves)
    static = categories("tile", tiles) + categories("position", cells)
    actions = []
    for direction, (dr, dc) in [("north", (-1, 0)), ("east", (0, 1)), ("south", (1, 0)), ("west", (0, -1))]:
        pred = direction + "_of"
        for origin in cells:
            r, c = rc(origin); target = cell(r+dr, c+dc)
            if target in cells: static.append([pred, target, origin])
        actions.append(action("move_blank_" + direction, "?tile ?from ?to",
                      f"(and (tile ?tile) (position ?from) (position ?to) (at ?tile ?from) (empty ?to) ({pred} ?from ?to))",
                      "(and (not (at ?tile ?from)) (at ?tile ?to) (not (empty ?to)) (empty ?from))"))
    goals = {"row": "Place numbers in increasing row-major order, with the blank last (bottom-right).",
             "column": "Place numbers in increasing column-major order (top-to-bottom, then left-to-right), with the blank last.",
             "snake": "Place numbers in increasing snake order: first row left-to-right, second right-to-left, alternating; put the blank last.",
             "descending": "Place numbers in decreasing row-major order, with the blank last (bottom-right)."}
    rules = (goals[pattern] + "\nRules: Number N identifies tile tN. Coordinates start at r01c01 in the top-left. "
             "Each move swaps one tile with the orthogonally adjacent blank; no diagonal moves or jumps. "
             "Operation names specify the BLANK's movement, opposite to the tile's movement. "
             "There is exactly one blank.")
    return make("numeric_sliding", candidate, spec, state, [*tiles, *cells], static,
                [list(a) for a in sorted(sliding_projection(goal_state))], actions, rules)


GENERATORS = dict(zip(GAMES, [kitchen, rush, color_sort, blocks, sliding]))


def kitchen_projection(s):
    out = {("at", a, loc) for a, loc in s["chefs"].items()}
    out |= {("hand_free", h) if x is None else ("holding", h, x) for h, x in s["holding"].items()}
    out |= {("prepared", f) for f in s["prepared"]}
    out |= {(rel, f, loc) for f, (rel, loc) in s["food_places"].items() if rel != "held"}
    if not any(p == ["on", "pass1"] for p in s["food_places"].values()): out.add(("counter_free", "pass1"))
    if s["plate_place"][0] == "at": out.add(("at", "p1", s["plate_place"][1]))
    if s["served"]: out.add(("served", "p1"))
    return out


def kitchen_successors(spec, s):
    result = {}
    def add(a, change):
        t = copy.deepcopy(s); change(t); result[tuple(a)] = t
    for chef, loc in s["chefs"].items():
        hand = spec["hands"][chef]; held = s["holding"][hand]
        for to in spec["floors"]:
            if adjacent(loc, to): add(["move", chef, loc, to], lambda t, to=to: t["chefs"].__setitem__(chef, to))
        for f in spec["foods"]:
            place = s["food_places"][f]
            if held is None and place == ["at", loc]:
                def pick(t, f=f): t["holding"][hand] = f; t["food_places"][f] = ["held", hand]
                add(["pickup_ingredient", f, loc, hand, chef], pick)
            if held == f and loc == spec["chop"] and f not in s["prepared"]:
                add(["chop", f, loc, hand, chef], lambda t, f=f: t["prepared"].append(f))
            if adjacent(loc, spec["counter"]):
                if held == f and not any(p == ["on", "pass1"] for p in s["food_places"].values()):
                    def put(t, f=f): t["holding"][hand] = None; t["food_places"][f] = ["on", "pass1"]
                    add(["put_on_counter", f, "pass1", loc, hand, chef], put)
                if held is None and place == ["on", "pass1"]:
                    def take(t, f=f): t["holding"][hand] = f; t["food_places"][f] = ["held", hand]
                    add(["take_from_counter", f, "pass1", loc, hand, chef], take)
            if held == f and f in s["prepared"] and f in spec["required"] and s["plate_place"] == ["at", loc]:
                def pack(t, f=f): t["holding"][hand] = None; t["food_places"][f] = ["on", "p1"]
                add(["put_on_plate", f, "p1", loc, hand, chef], pack)
        if held is None and s["plate_place"] == ["at", loc] and all(s["food_places"][f] == ["on", "p1"] for f in spec["required"]):
            def pick_plate(t): t["holding"][hand] = "p1"; t["plate_place"] = ["held", hand]
            add(["pickup_plate", "p1", loc, hand, chef], pick_plate)
        if held == "p1" and loc == spec["delivery"]:
            def deliver(t): t["holding"][hand] = None; t["plate_place"] = ["at", loc]; t["served"] = True
            add(["deliver", "p1", loc, hand, chef], deliver)
    return result


def rush_projection(spec, s):
    occupied = {p for v, a in s.items() if a is not None for p in footprint(spec["cars"][v], a)}
    return {("empty", cell(r, c)) for r in range(spec["h"]) for c in range(spec["w"]) if cell(r, c) not in occupied} | {("escaped", v) if a is None else ("at", v, a) for v, a in s.items()}


def rush_successors(spec, s):
    result = {}; occupied = {p for v, a in s.items() if a is not None for p in footprint(spec["cars"][v], a)}
    for v, anchor in s.items():
        if anchor is None: continue
        car = spec["cars"][v]; r, c = rc(anchor); own = footprint(car, anchor)
        for sign in [-1, 1]:
            other = cell(r + sign*(car["axis"] == "vertical"), c + sign*(car["axis"] == "horizontal"))
            span = footprint(car, other)
            if not all(0 <= rc(p)[0] < spec["h"] and 0 <= rc(p)[1] < spec["w"] for p in span): continue
            entering = set(span)-set(own); leaving = set(own)-set(span)
            if not entering & occupied:
                t = dict(s); t[v] = other
                result[("move_vehicle", v, anchor, other, next(iter(leaving)), next(iter(entering)))] = t
        if v == "v0" and c + car["length"] == spec["w"]:
            t = dict(s); t[v] = None; result[("exit_vehicle", v, anchor, *own)] = t
    return result


def color_projection(s):
    return {("top", b, slot(b, len(p))) for b, p in s.items()} | {fact for b, p in s.items() for i, n in enumerate(p, 1) for fact in [("in", n, b), ("at", n, slot(b, i))]}


def color_successors(spec, s):
    result = {}
    for source, target in itertools.permutations(s, 2):
        a, b = s[source], s[target]
        if not a or len(b) >= spec["capacity"] or (b and spec["colors"][a[-1]] != spec["colors"][b[-1]]): continue
        n = a[-1]; t = copy.deepcopy(s); t[target].append(t[source].pop())
        positions = (source, target, slot(source, len(a)), slot(source, len(a)-1), slot(target, len(b)), slot(target, len(b)+1))
        operation = ("move_onto", n, b[-1], *positions) if b else ("move_to_empty", n, *positions)
        result[operation] = t
    return result


def block_projection(s):
    out = {("on", b, lower) for b, lower in s["on"].items()} | {("clear", b) for b in s["on"] if b not in s["on"].values()}
    out.add(("hand_free", "hand1") if s["held"] is None else ("holding", "hand1", s["held"]))
    return out


def block_successors(spec, s):
    result = {}; clear = [b for b in s["on"] if b not in s["on"].values()]
    if s["held"] is None:
        for b in clear:
            lower = s["on"][b]; t = copy.deepcopy(s); del t["on"][b]; t["held"] = b
            a = ("pickup", b, "hand1", "table") if lower == "table" else ("unstack", b, lower, "hand1")
            result[a] = t
    else:
        b = s["held"]
        for lower in ["table", *clear]:
            t = copy.deepcopy(s); t["held"] = None; t["on"][b] = lower
            a = ("putdown", b, "hand1", "table") if lower == "table" else ("stack", b, lower, "hand1")
            result[a] = t
    return result


def sliding_projection(s):
    return {("empty", p) if t is None else ("at", t, p) for p, t in s.items()}


def sliding_successors(spec, s):
    empty = next(p for p, t in s.items() if t is None); result = {}
    for p, tile in s.items():
        if tile and adjacent(p, empty):
            t = dict(s); t[empty], t[p] = tile, None
            r, c = rc(p); er, ec = rc(empty)
            direction = "south" if r > er else "north" if r < er else "east" if c > ec else "west"
            result[("move_blank_" + direction, tile, p, empty)] = t
    return result


def projection(task, state):
    g, spec = task["game"], task["spec"]
    if g == "rush_hour": return rush_projection(spec, state)
    return {"cooperative_kitchen": kitchen_projection, "color_sort": color_projection,
            "blocks_3d": block_projection, "numeric_sliding": sliding_projection}[g](state)


def successors(task, state):
    return {"cooperative_kitchen": kitchen_successors, "rush_hour": rush_successors,
            "color_sort": color_successors, "blocks_3d": block_successors,
            "numeric_sliding": sliding_successors}[task["game"]](task["spec"], state)


def native_goal(task, s):
    g, spec = task["game"], task["spec"]
    if g == "cooperative_kitchen": return s["served"] and all(s["food_places"][f] == ["on", "p1"] and f in s["prepared"] for f in spec["required"])
    if g == "rush_hour": return s["v0"] is None
    if g == "color_sort": return all(all(spec["targets"][spec["colors"][n]] == b for n in stack) for b, stack in s.items())
    if g == "blocks_3d": return s["held"] is None and all(s["on"].get(b) == lower for b, lower in spec["goal_on"].items()) and all(b in s["on"] and b not in s["on"].values() for b in spec["clear_goal"])
    return s == spec["goal_state"]


def simulate(task, plan):
    """Replay actions with native rules, independently of PDDL."""
    state = copy.deepcopy(task["state"])
    invariant(task, state)
    for index, operation in enumerate(native_plan(task, plan), 1):
        available = successors(task, state)
        if tuple(operation) not in available:
            raise ValueError(f"Illegal native action at step {index}: {operation}")
        state = available[tuple(operation)]
        invariant(task, state)
    if not native_goal(task, state):
        raise ValueError("Native game goal not reached")
    return state


def invariant(task, s):
    spec, g = task["spec"], task["game"]
    if g == "numeric_sliding":
        assert sum(t is None for t in s.values()) == 1
        assert sorted(t for t in s.values() if t) == sorted(t for t in spec["goal_state"].values() if t)
    elif g == "rush_hour":
        occupied = [p for v, a in s.items() if a is not None for p in footprint(spec["cars"][v], a)]
        assert len(occupied) == len(set(occupied))
        assert all(0 <= rc(p)[0] < spec["h"] and 0 <= rc(p)[1] < spec["w"] for p in occupied)
    elif g == "color_sort":
        assert sorted(n for stack in s.values() for n in stack) == sorted(spec["colors"])
        assert all(len(stack) <= spec["capacity"] for stack in s.values())
    elif g == "blocks_3d":
        assert set(s["on"]) | ({s["held"]} if s["held"] else set()) == set(spec["blocks"])
        assert s["held"] not in s["on"]
        supports_used = [x for x in s["on"].values() if x != "table"]
        assert len(supports_used) == len(set(supports_used)) and set(supports_used) <= set(s["on"])
        for b in s["on"]:
            seen = set()
            while b != "table":
                assert b not in seen; seen.add(b); b = s["on"][b]
    else:
        assert all(p in spec["floors"] for p in s["chefs"].values())
        assert len([p for p in s["food_places"].values() if p == ["on", "pass1"]]) <= 1
        assert set(s["prepared"]) <= set(spec["foods"])
        for h, x in s["holding"].items():
            if x and x != "p1": assert s["food_places"][x] == ["held", h]
        for f, (rel, loc) in s["food_places"].items():
            if rel == "held": assert s["holding"][loc] == f
            if rel == "on" and loc == "p1": assert f in s["prepared"]
        if s["plate_place"][0] == "held": assert s["holding"][s["plate_place"][1]] == "p1"


def structure(task):
    """Comparable structures, not hashes. Ignore cosmetic styles and ID renaming."""
    g, s, spec = task["game"], task["state"], task["spec"]
    if g == "rush_hour":
        return (spec["h"], spec["w"], s["v0"], tuple(sorted((car["axis"], car["length"], s[v]) for v, car in spec["cars"].items() if v != "v0")))
    if g == "numeric_sliding": return (spec["h"], spec["w"], spec["goal_pattern"], tuple(sorted(s.items())))
    if g == "color_sort":
        return (spec["capacity"], tuple(tuple(spec["colors"][n] for n in p) for p in s.values()), tuple(sorted(spec["targets"].items())))
    if g == "cooperative_kitchen":
        return (spec["h"], spec["w"], tuple(spec["floors"]), spec["counter"], spec["chop"], spec["delivery"], tuple(sorted(s["chefs"].items())),
                tuple(sorted(s["holding"].items())), tuple(sorted((f, tuple(p)) for f, p in s["food_places"].items())), tuple(sorted(s["prepared"])), tuple(s["plate_place"]))
    roots = [b for b, lower in s["on"].items() if lower == "table"]
    upper = {lower: b for b, lower in s["on"].items() if lower != "table"}
    piles = []
    for root in roots:
        p = [root]
        while p[-1] in upper: p.append(upper[p[-1]])
        piles.append(p)
    piles.sort(key=lambda p: (-len(p), p))
    order = [b for p in piles for b in p] + ([s["held"]] if s["held"] else [])
    names = {b: i for i, b in enumerate(order)} | {"table": -1}
    return (tuple(map(len, piles)), s["held"] is not None, spec["goal_mode"],
            tuple(sorted((names[b], names[lower]) for b, lower in spec["goal_on"].items())), tuple(sorted(names[b] for b in spec["clear_goal"])))
