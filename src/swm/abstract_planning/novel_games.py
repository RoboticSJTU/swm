"""Four held-out games, with geometric simulators independent of PDDL."""

from collections import deque
import random

from .grid_games import DIRECTIONS, cell, rc
from .model import action, make_domain, make_problem, parse_sexpr, section, fields, condition, apply_effect
from .render import Scene, INK

GAMES = ["phase_gates", "energy_route", "coupled_tokens", "conveyor_route"]


def step_cell(pos, direction):
    r, c = rc(pos)
    dr, dc = DIRECTIONS[direction]
    return cell(r + dr, c + dc)


def landing(spec, entry):
    """Resolve forced motion; blocked exits and cycles invalidate the whole move."""
    seen = set()
    while entry in spec["belts"]:
        if entry in seen:
            return None
        seen.add(entry)
        entry = step_cell(entry, spec["belts"][entry])
        if entry not in spec["cells"]:
            return None
    return entry


def successors(game, spec, state):
    """Enumerate legal public actions using only board geometry and game rules."""
    pos = state[0]
    if game == "coupled_tokens":
        other = state[1]
        for i, direction in enumerate(DIRECTIONS):
            a = step_cell(pos, direction)
            opposite = list(DIRECTIONS)[(i + 2) % 4]
            b = step_cell(other, opposite)
            if a in spec["cells"] and b in spec["cells"] and a != b:
                yield ("move_" + direction, pos, a, other, b), (a, b)
        return
    for direction in DIRECTIONS:
        entry = step_cell(pos, direction)
        if entry not in spec["cells"]:
            continue
        if game == "phase_gates":
            phase = state[1]
            color = spec["gates"].get(entry)
            required = {"red": spec["red_phase"], "blue": 1 - spec["red_phase"]}
            if color and phase != required[color]:
                continue
            yield ("move", pos, entry), (entry, phase)
        elif game == "energy_route":
            energy = state[1]
            cost = spec["rough_cost"] if entry in spec["rough"] else 1
            if energy >= cost:
                yield ("move", pos, entry, f"n{energy}", f"n{energy - cost}"), (entry, energy - cost)
        else:
            target = landing(spec, entry)
            if target is not None and target != pos:
                yield ("move", pos, entry, target), (target,)
    if game == "phase_gates" and pos in spec["consoles"]:
        yield ("toggle", pos, f"phase{state[1]}", f"phase{1-state[1]}"), (pos, 1-state[1])
    if game == "energy_route" and pos in spec["chargers"] and state[1] < 4:
        yield ("charge", pos, f"n{state[1]}"), (pos, 4)


def explore(game, spec):
    start = tuple(spec["start"])
    parents = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for act, following in successors(game, spec, current):
            if following not in parents:
                parents[following] = (current, act)
                queue.append(following)
    return parents


def path_to(parents, target):
    plan = []
    while parents[target] is not None:
        target, act = parents[target]
        plan.append(act)
    return list(reversed(plan))


def simulate(scenario, plan):
    state = tuple(scenario["novel"]["start"])
    trace = [state]
    for act in plan:
        possible = dict(successors(scenario["game"], scenario["novel"], state))
        if tuple(act) not in possible:
            raise ValueError(f"Illegal game action: {act}, state={state}")
        state = possible[tuple(act)]
        trace.append(state)
    goal = scenario["novel"]["goal"]
    if tuple(state[:len(goal)]) != tuple(goal):
        raise ValueError("Independent game goal not reached")
    return trace


def formalize(game, seed, spec):
    cells = spec["cells"]
    edges = [[a, step_cell(a, d), d] for a in cells for d in DIRECTIONS if step_cell(a, d) in cells]
    initial = [["cell", c] for c in cells]
    objects = list(cells)
    vocab = {"at_player": "the player is at cell {0}", "cell": "{0} is a walkable cell"}
    if game == "coupled_tokens":
        initial += [["at_a", spec["start"][0]], ["at_b", spec["start"][1]]]
        actions = []
        for i, d in enumerate(DIRECTIONS):
            opposite = list(DIRECTIONS)[(i + 2) % 4]
            initial += [[d + "_step", a, b] for a, b, dr in edges if dr == d]
            vocab[d + "_step"] = "cell {1} is immediately " + d + " of cell {0}"
            actions.append(action("move_"+d, "?a_from ?a_to ?b_from ?b_to",
                f"(and (at_a ?a_from) (at_b ?b_from) ({d}_step ?a_from ?a_to) ({opposite}_step ?b_from ?b_to) (not (= ?a_to ?b_to)))",
                "(and (not (at_a ?a_from)) (at_a ?a_to) (not (at_b ?b_from)) (at_b ?b_to))"))
        vocab |= {"at_a": "token A is at cell {0}", "at_b": "token B is at cell {0}"}
        goal = [["at_a", spec["goal"][0]], ["at_b", spec["goal"][1]]]
        rules = "Each command moves A one cell in its named direction and B one cell oppositely. Both destinations must be floor and different; swapping is allowed. If either token hits a wall, neither moves."
    else:
        initial += [["at_player", spec["start"][0]]]
        initial += [["neighbor", a, b] for a, b, d in edges]
        vocab["neighbor"] = "cells {0} and {1} are orthogonally adjacent"
        goal = [["at_player", spec["goal"][0]]]
        if game == "phase_gates":
            objects += ["phase0", "phase1"]
            initial += [["phase", f"phase{spec['start'][1]}"], ["flip", "phase0", "phase1"], ["flip", "phase1", "phase0"]]
            initial += [["console", c] for c in spec["consoles"]]
            for c in cells:
                color = spec["gates"].get(c)
                phases = [spec["red_phase"] if color == "red" else 1-spec["red_phase"]] if color else [0, 1]
                initial += [["allows", c, f"phase{p}"] for p in phases]
            actions = [action("move", "?from ?to", "(and (at_player ?from) (neighbor ?from ?to) (exists (?p) (and (phase ?p) (allows ?to ?p))))", "(and (not (at_player ?from)) (at_player ?to))"),
                       action("toggle", "?loc ?previous ?result", "(and (at_player ?loc) (console ?loc) (phase ?previous) (flip ?previous ?result))", "(and (not (phase ?previous)) (phase ?result))")]
            vocab |= {"phase": "the global phase is {0}", "flip": "{0} toggles to {1}", "console": "cell {0} has a phase console", "allows": "cell {0} permits entry during {1}"}
            rules = f"Move one orthogonal floor cell. Enter red gates only in phase{spec['red_phase']}, blue gates only in phase{1-spec['red_phase']}. Other cells allow either phase. At a dial, toggle phase without moving; movement preserves phase."
        elif game == "energy_route":
            objects += [f"n{i}" for i in range(5)]
            initial += [["energy", f"n{spec['start'][1]}"], *[["charger", c] for c in spec["chargers"]]]
            for c in cells:
                cost = spec["rough_cost"] if c in spec["rough"] else 1
                initial += [["spend", c, f"n{i}", f"n{i-cost}"] for i in range(cost, 5)]
            initial += [["below_full", f"n{i}"] for i in range(4)]
            actions = [action("move", "?from ?to ?previous ?result", "(and (at_player ?from) (neighbor ?from ?to) (energy ?previous) (spend ?to ?previous ?result))", "(and (not (at_player ?from)) (at_player ?to) (not (energy ?previous)) (energy ?result))"),
                       action("charge", "?loc ?previous", "(and (at_player ?loc) (charger ?loc) (energy ?previous) (below_full ?previous))", "(and (not (energy ?previous)) (energy n4))")]
            vocab |= {"energy": "the remaining energy is {0}", "charger": "cell {0} is a charger", "spend": "entering {0} changes energy {1} to {2}", "below_full": "energy {0} is below full"}
            rules = f"Move one orthogonal floor cell, spending {spec['rough_cost']} energy on entering striped terrain and 1 elsewhere. Insufficient energy blocks movement. At lightning, charge restores energy to 4 if below 4; charging is explicit and does not move. nN means N energy."
        else:
            initial += [["landing", c, t] for c in cells if (t := landing(spec, c)) is not None]
            actions = [action("move", "?from ?entry ?to", "(and (at_player ?from) (neighbor ?from ?entry) (landing ?entry ?to) (not (= ?from ?to)))", "(and (not (at_player ?from)) (at_player ?to))")]
            vocab["landing"] = "entering cell {0} follows the arrow chain and stops at {1}"
            rules = "Enter an orthogonally adjacent floor cell. On arrows, automatically follow the arrow chain until an unmarked cell. A chain hitting a wall, looping, or returning to the start makes the whole move illegal. entry is the adjacent cell; to is the final landing cell."
    constants = ["n4"] if game == "energy_route" else []
    domain = make_domain(game, actions, initial, goal, constants)
    prompt = "Goal: " + (f"put token A at {spec['goal'][0]} and token B at {spec['goal'][1]}" if game == "coupled_tokens" else f"reach {spec['goal'][0]}") + ".\nRules: " + rules + "\n"
    prompt += "r01c01 is top-left; rows increase down, columns right. Walls and boundaries block movement.\n"
    legend = {"phase_gates": "Arches are gates; the header shows phase.", "energy_route": "The header shows current energy.", "coupled_tokens": "North is up; A/B label the tokens.", "conveyor_route": ""}[game]
    if legend: prompt += legend + "\n"
    prompt += "Actions:\n" + "\n".join(a[1]+"("+", ".join(x.lstrip("?") for x in fields(a)[":parameters"])+")" for a in actions) + "\n"
    return {"game": game, "seed": seed, "variant": "base" if spec.get("red_phase", 0) == 0 and spec.get("rough_cost", 2) == 2 else "rule_changed", "domain": domain, "problem": make_problem(game, [x for x in objects if x not in constants], initial, goal), "vocabulary": vocab, "notes": [rules], "novel": spec, "instruction": prompt, "source": {"repository": "local held-out geometric game rules"}}


def generate(game, seed):
    rng = random.Random(seed)
    h = w = 6
    cells = [cell(r,c) for r in range(h) for c in range(w) if rng.random() > 0.16]
    if len(cells) < 24:
        raise ValueError("too_many_walls")
    start = rng.choice(cells)
    spec = {"height": h, "width": w, "cells": cells}
    if game == "phase_gates":
        special = rng.sample([c for c in cells if c != start], 8)
        spec |= {"gates": dict(zip(special[:6], ["red", "blue"]*3)), "consoles": special[6:], "red_phase": 0, "start": [start, 0]}
    elif game == "energy_route":
        spec |= {"rough": rng.sample([c for c in cells if c != start], 7), "chargers": rng.sample(cells, 9), "rough_cost": 2, "start": [start, 4]}
    elif game == "coupled_tokens":
        spec["start"] = [start, rng.choice([c for c in cells if c != start])]
    else:
        sites = rng.sample([c for c in cells if c != start], 12)
        spec |= {"belts": {c: rng.choice(list(DIRECTIONS)) for c in sites}, "start": [start]}
    parents = explore(game, spec)
    candidates = []
    for state in parents:
        plan = path_to(parents, state)
        if len(plan) < (6 if game == "coupled_tokens" else 8):
            continue
        if game == "phase_gates" and (not any(a[0] == "toggle" for a in plan) or not {"red", "blue"} <= {spec["gates"].get(a[2]) for a in plan if a[0] == "move"}):
            continue
        if game == "energy_route" and (not any(a[0] == "charge" for a in plan) or not any(a[0] == "move" and a[2] in spec["rough"] for a in plan)):
            continue
        if game == "conveyor_route" and not any(a[2] != a[3] for a in plan):
            continue
        if game in {"phase_gates", "energy_route"}:
            changed = spec | ({"red_phase": 1} if game == "phase_gates" else {"rough_cost": 3})
            others = explore(game, changed)
            targets = [s for s in others if s[0] == state[0]]
            if not targets or min(len(path_to(others, s)) for s in targets) < 8:
                continue
        candidates.append(state)
    if not candidates:
        raise ValueError("no_nontrivial_mechanism_goal")
    target = rng.choice(sorted(candidates))
    spec["goal"] = list(target if game == "coupled_tokens" else target[:1])
    return formalize(game, seed, spec)


def render(scenario):
    s = Scene(scenario)
    spec, game = scenario["novel"], scenario["game"]
    unit, ox, oy = 132, 210, 170
    s.label((ox, 55), game.replace("_", " "), 28, True)
    if game == "phase_gates":
        s.label((ox, 102), f"phase{spec['start'][1]}", 25, True)
        s.fact("phase", f"phase{spec['start'][1]}")
    elif game == "energy_route":
        s.label((ox, 102), f"energy {spec['start'][1]}/4", 25, True)
        s.fact("energy", f"n{spec['start'][1]}")
    for r in range(spec["height"]):
        s.label((ox-14, oy+r*unit+54), f"r{r+1:02d}", 22, True, anchor="right")
        for c in range(spec["width"]):
            name = cell(r,c)
            x,y = ox+c*unit, oy+r*unit
            if r == 0:
                s.label((x+unit/2, oy-30), f"c{c+1:02d}", 22, True, anchor="center")
            s.texture((x+1,y+1,x+unit-1,y+unit-1), "tile" if name in spec["cells"] else "brick")
            if name not in spec["cells"]:
                continue
            s.regions[name] = [x,y,x+unit,y+unit]
            if game == "phase_gates":
                if name in spec["gates"]:
                    color = "#c73c3c" if spec["gates"][name] == "red" else "#2f68cf"
                    s.draw.arc((x+20,y+18,x+112,y+113), 180, 360, fill=color, width=12)
                    s.draw.line((x+20,y+65,x+20,y+115),fill=color,width=12)
                    s.draw.line((x+112,y+65,x+112,y+115),fill=color,width=12)
                if name in spec["consoles"]:
                    s.draw.ellipse((x+35,y+25,x+97,y+87), fill="#ead886",outline=INK,width=4)
                    s.draw.line((x+66,y+56,x+82,y+37),fill=INK,width=4)
            elif game == "energy_route":
                if name in spec["rough"]:
                    s.draw.rectangle((x+8,y+8,x+124,y+124),fill="#c6a477")
                    for t in range(15,120,15):
                        s.draw.line((x+10,y+t,x+122,y+t),fill="#8e6e40",width=3)
                if name in spec["chargers"]:
                    s.draw.polygon([(x+70,y+20),(x+41,y+70),(x+65,y+67),(x+56,y+112),(x+96,y+53),(x+72,y+58)],fill="#f4d12f",outline=INK)
            elif game == "conveyor_route" and name in spec["belts"]:
                dr,dc = DIRECTIONS[spec["belts"][name]]
                s.arrow((x+66-dc*30,y+66-dr*30),(x+66+dc*32,y+66+dr*32),fill="#24738a",width=9)
            if game == "coupled_tokens":
                for i,label in enumerate(["A","B"]):
                    if name == spec["start"][i]:
                        s.object(label,"player",(x+66,y+55),58,location=name, label=False)
                        s.label((x+66,y+99),label,22,True,anchor="center")
                        s.fact("at_"+label.lower(),name)
            elif name == spec["start"][0]:
                s.object("player","player",(x+66,y+55),52,location=name, label=False)
                s.label((x+66,y+102),"player",18,True,anchor="center")
                s.fact("at_player",name)
    return s.finish()


def verify_transitions(scenario):
    """Check positive and negative actions at every independently reachable state."""
    game, spec = scenario["game"], scenario["novel"]
    domain, problem = parse_sexpr(scenario["domain"]), parse_sexpr(scenario["problem"])
    ops = {a[1]:fields(a) for a in domain if isinstance(a,list) and a[:1]==[":action"]}
    objects = section(problem, ":objects")[1:] + section(domain, ":constants", [":constants"])[1:]
    initial = {tuple(a) for a in section(problem, ":init")[1:]}
    dynamic = {"at_player","phase","energy","at_a","at_b"}
    static = {a for a in initial if a[0] not in dynamic}
    def projection(state):
        if game == "coupled_tokens":
            return {("at_a",state[0]),("at_b",state[1])}
        result = {("at_player",state[0])}
        if game == "phase_gates": result.add(("phase",f"phase{state[1]}"))
        if game == "energy_route": result.add(("energy",f"n{state[1]}"))
        return result
    positives = negatives = 0
    reachable = explore(game,spec)
    for state in reachable:
        legal = dict(successors(game,spec,state))
        candidates = set(legal)
        if game == "phase_gates":
            candidates.update(("move",state[0],c) for c in spec["cells"])
            candidates.update(("toggle",c,f"phase{i}",f"phase{j}") for c in spec["cells"] for i in range(2) for j in range(2))
        elif game == "energy_route":
            candidates.update(("move",state[0],c,f"n{i}",f"n{j}") for c in spec["cells"] for i in range(5) for j in range(5))
            candidates.update(("charge",c,f"n{i}") for c in spec["cells"] for i in range(5))
        elif game == "conveyor_route":
            candidates.update(("move",state[0],a,b) for a in spec["cells"] for b in spec["cells"])
        else:
            for d in DIRECTIONS:
                # Every pair of candidate destinations, including swaps and collisions.
                candidates.update(("move_"+d,state[0],a,state[1],b) for a in spec["cells"] for b in spec["cells"])
        before = static | projection(state)
        for act in sorted(candidates):
            op = ops[act[0]]
            binding = dict(zip(op[":parameters"],act[1:]))
            applicable = condition(op[":precondition"],before,binding,objects)
            assert applicable == (act in legal), (game,state,act,"legality mismatch")
            if applicable:
                add,delete = set(),set()
                apply_effect(op[":effect"],before,binding,objects,add,delete)
                assert (before-delete)|add == static|projection(legal[act]), (game,state,act,"effect mismatch")
                positives += 1
            else:
                negatives += 1
    return {"reachable_states":len(reachable),"legal_transitions_checked":positives,"illegal_actions_checked":negatives,"all_checked_transitions_match":True}


def verify_edge_rules():
    # Fixed small rule tests, unrelated to the selected evaluation scenes.
    a,b = cell(0,0),cell(0,1)
    spec={"cells":[a,b],"belts":{a:"east",b:"west"}}
    assert landing(spec,a) is None
    spec["belts"]={b:"east"}
    assert landing(spec,b) is None
    spec["belts"]={a:"east"}
    assert landing(spec,a)==b
    swapped=dict(successors("coupled_tokens",{"cells":[a,b]},(a,b)))
    assert swapped[("move_east",a,b,b,a)]==(b,a)
    spec={"cells":[a,b],"gates":{b:"red"},"consoles":[a],"red_phase":0}
    assert ("move",a,b) in dict(successors("phase_gates",spec,(a,0)))
    assert ("move",a,b) not in dict(successors("phase_gates",spec,(a,1)))
    spec={"cells":[a,b],"rough":[b],"chargers":[a],"rough_cost":3}
    assert not any(act[0]=="move" for act,_ in successors("energy_route",spec,(a,2)))
    assert not any(act[0]=="charge" for act,_ in successors("energy_route",spec,(a,4)))
    return True
