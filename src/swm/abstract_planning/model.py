"""PDDL adaptation, literal English rules, and simultaneous state execution."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

from swm.pddl.strips import parse_sexpr, validate_untyped_pddl


def sexpr(node):
    return "(" + " ".join(map(sexpr, node)) + ")" if isinstance(node, list) else node


def section(tree, name, default=None):
    return next((x for x in tree if isinstance(x, list) and x[:1] == [name]), default)


def fields(action):
    return dict(zip(action[2::2], action[3::2]))


def typed_symbols(items):
    result, pending, i = [], [], 0
    while i < len(items):
        if items[i] == "-":
            result.extend((x, items[i + 1]) for x in pending)
            pending = []
            i += 2
        else:
            pending.append(items[i])
            i += 1
    return result + [(x, "object") for x in pending]


def untype(domain_text, problem_text):
    """Replace typing with guards; remove only optimization cost bookkeeping."""
    domain_text = safe_variables(domain_text)
    domain, problem = (
        parse_sexpr(domain_text.lower()),
        parse_sexpr(problem_text.lower()),
    )
    parents = dict(typed_symbols(section(domain, ":types", [":types"])[1:]))
    categories = set(parents) | {"object"}

    def guards(items):
        pairs = typed_symbols(items)
        categories.update(t for _, t in pairs)
        return [x for x, _ in pairs], [[t, x] for x, t in pairs]

    def convert(node, effect=False):
        if not isinstance(node, list):
            return node
        if node[:1] == ["increase"]:
            if node[1] != ["total_cost"]:
                raise ValueError("Cannot discard a resource update: " + sexpr(node))
            return ["and"]
        if node[:1] in [["forall"], ["exists"]]:
            params, checks = guards(node[1])
            body = convert(node[2], effect)
            if node[0] == "exists":
                body = ["and", *checks, body]
            else:
                body = ["when" if effect else "imply", ["and", *checks], body]
            return [node[0], params, body]
        return [convert(x, effect) for x in node]

    output = ["define", domain[1]]
    output.append([":requirements", ":adl"])
    declared = section(domain, ":predicates")[1:]
    declared = [[p[0], *guards(p[1:])[0]] for p in declared]
    constants = typed_symbols(section(domain, ":constants", [":constants"])[1:])
    objects = typed_symbols(section(problem, ":objects", [":objects"])[1:])
    categories.update(t for _, t in constants + objects)
    actions = []
    for a in domain:
        if not isinstance(a, list) or a[:1] != [":action"]:
            continue
        f = fields(a)
        params, checks = guards(f[":parameters"])
        actions.append(
            [
                ":action",
                a[1],
                ":parameters",
                params,
                ":precondition",
                ["and", *checks, convert(f[":precondition"])],
                ":effect",
                convert(f[":effect"], True),
            ]
        )
    if constants:
        output.append([":constants", *(x for x, _ in constants)])
    names = {p[0] for p in declared}
    output.append(
        [":predicates", *declared, *([c, "?x"] for c in sorted(categories - names))]
    )
    output.extend(actions)
    initial = [
        x for x in section(problem, ":init")[1:] if x[:1] not in [["="], ["not"]]
    ]
    for x, category in objects + constants:
        seen = set()
        while category not in seen:
            seen.add(category)
            if [category, x] not in initial:
                initial.append([category, x])
            if category == "object":
                break
            category = parents.get(category, "object")
    result = [
        "define",
        problem[1],
        section(problem, ":domain"),
        [":objects", *(x for x, _ in objects)],
        [":init", *initial],
        [":goal", convert(section(problem, ":goal")[1])],
    ]
    dtext, ptext = sexpr(output) + "\n", sexpr(result) + "\n"
    validate_untyped_pddl(dtext)
    validate_untyped_pddl(ptext)
    return dtext, ptext


def condition(node, state, binding, objects):
    head = node[0]
    if head == "and":
        return all(condition(x, state, binding, objects) for x in node[1:])
    if head == "or":
        return any(condition(x, state, binding, objects) for x in node[1:])
    if head == "not":
        return not condition(node[1], state, binding, objects)
    if head == "imply":
        return not condition(node[1], state, binding, objects) or condition(
            node[2], state, binding, objects
        )
    if head in {"forall", "exists"}:
        values = (
            condition(node[2], state, binding | dict(zip(node[1], args)), objects)
            for args in itertools.product(objects, repeat=len(node[1]))
        )
        return all(values) if head == "forall" else any(values)
    fact = tuple(binding.get(x, x) for x in node)
    return fact[1] == fact[2] if head == "=" else fact in state


def apply_effect(node, state, binding, objects, add, delete):
    head = node[0]
    if head == "and":
        for x in node[1:]:
            apply_effect(x, state, binding, objects, add, delete)
    elif head == "when":
        if condition(node[1], state, binding, objects):
            apply_effect(node[2], state, binding, objects, add, delete)
    elif head == "forall":
        for args in itertools.product(objects, repeat=len(node[1])):
            apply_effect(
                node[2], state, binding | dict(zip(node[1], args)), objects, add, delete
            )
    elif head == "not":
        delete.add(tuple(binding.get(x, x) for x in node[1]))
    else:
        add.add(tuple(binding.get(x, x) for x in node))


def replay(domain_text, problem_text, plan):
    domain, problem = parse_sexpr(domain_text), parse_sexpr(problem_text)
    objects = (
        section(problem, ":objects")[1:]
        + section(domain, ":constants", [":constants"])[1:]
    )
    state = {tuple(a) for a in section(problem, ":init")[1:]}
    actions = {
        a[1]: fields(a) for a in domain if isinstance(a, list) and a[:1] == [":action"]
    }
    trace = [state]
    for step, action in enumerate(plan, 1):
        f = actions[action[0]]
        if len(f[":parameters"]) != len(action) - 1 or not set(action[1:]) <= set(
            objects
        ):
            raise ValueError(f"Invalid interface at step {step}: {action}")
        binding = dict(zip(f[":parameters"], action[1:]))
        if not condition(f[":precondition"], state, binding, objects):
            raise ValueError(f"Illegal action at step {step}: {action}")
        add, delete = set(), set()
        apply_effect(f[":effect"], state, binding, objects, add, delete)
        state = (state - delete) | add
        trace.append(state)
    if not condition(section(problem, ":goal")[1], state, {}, objects):
        raise ValueError("Plan does not reach the task goal")
    return trace


def verify_invariants(scenario, plan, trace):
    """Check physical conservation separately from the PDDL action interpreter."""
    game = scenario["game"]

    def facts(state, p):
        return [a[1:] for a in state if a[0] == p]

    initial = trace[0]
    for step, state in enumerate(trace):

        def require(test, meaning):
            if not test:
                raise ValueError(f"{game} invariant failed at step {step}: {meaning}")

        at = facts(state, "at")
        if game in {"blocks", "kitchen", "printer"}:
            held = facts(state, "holding")
            require(
                len(held) <= 1 and (("arm_empty",) in state) == (not held),
                "one hand and consistent empty flag",
            )
        if game in {"sokoban", "package", "printer"}:
            if game == "sokoban":
                blockers = {a[0] for a in facts(initial, "box")} | {"player"}
            else:
                blockers = {
                    a[0]
                    for a in facts(
                        initial, "package" if game == "package" else "printer"
                    )
                }
            if game == "printer":
                blockers |= {a[0] for a in facts(initial, "table")}
            blockers.add("player")
            occupied = {loc for obj, loc in at if obj in blockers}
            cells = set(scenario["scene"]["cells"])
            require(
                {a[0] for a in facts(state, "empty")} == cells - occupied,
                "empty cells agree with physical occupants",
            )
        if game == "sokoban":
            require(len({loc for obj, loc in at}) == len(at), "no player/box overlap")
            goals = {a[0] for a in facts(initial, "goal_cell")}
            require(
                {a[0] for a in facts(state, "on_goal")}
                == {obj for obj, loc in at if obj != "player" and loc in goals},
                "target flags agree with box positions",
            )
        elif game == "sliding_puzzle":
            positions = [loc for obj, loc in at] + [a[0] for a in facts(state, "empty")]
            require(
                len(facts(state, "empty")) == 1
                and len(set(positions)) == len(positions),
                "exactly one blank and no tile overlap",
            )
            require(
                set(positions) == {a[0] for a in facts(initial, "position")},
                "every cell is occupied by a tile or the blank",
            )
        elif game == "kitchen":
            for (ingredient,) in facts(initial, "ingredient"):
                require(
                    sum(
                        a[0] == ingredient
                        for a in at + facts(state, "holding") + facts(state, "on_plate")
                    )
                    == 1,
                    "ingredient is on a cell, in hand or on one plate",
                )
                require(
                    sum(a[0] == ingredient for a in facts(state, "stage")) == 1,
                    "one processing stage per ingredient",
                )
            for (plate,) in facts(initial, "plate"):
                require(
                    sum(a[0] == plate for a in at + facts(state, "holding")) == 1,
                    "one location per plate",
                )
        elif game == "printer":
            for (printer,) in facts(initial, "printer"):
                installed = [
                    a for a in facts(state, "installed_cartridge") if a[0] == printer
                ]
                require(
                    len(installed) <= 1
                    and (("installed", printer) in state) == bool(installed),
                    "at most one cartridge per printer",
                )
                require(
                    sum(a[0] == printer for a in facts(state, "paper_level")) == 1,
                    "one paper counter per printer",
                )
            for (table,) in facts(initial, "table"):
                mounted = [a for a in facts(state, "on") if a[1] == table]
                require(
                    len(mounted) <= 1
                    and (("table_free", table) in state) == (not mounted),
                    "one printer per table",
                )
        elif game == "lights_out":
            connected = facts(state, "connected")
            require(
                len(connected) <= 1
                and (("controller_free",) in state) == (not connected),
                "one panel connector",
            )
            if step and plan[step - 1][0] == "press_button":
                button = plan[step - 1][1]
                affected = {a[1] for a in facts(initial, "affects") if a[0] == button}
                before = {a[0] for a in facts(trace[step - 1], "lit")}
                require(
                    {a[0] for a in facts(state, "lit")} == before ^ affected,
                    "all affected lights flip using the previous state",
                )
        elif game == "briefcase":
            case = facts(state, "is_at")
            require(len(case) == 1, "one briefcase location")
            for (item,) in facts(initial, "portable"):
                places = [loc for obj, loc in at if obj == item]
                require(len(places) == 1, "one location per item")
                if ("in", item) in state:
                    require(
                        places[0] == case[0][0],
                        "contained items travel with the briefcase",
                    )
        elif game == "assembly":
            for (whole,) in facts(state, "complete"):
                permanent = {p for p, w in facts(state, "part-of") if w == whole}
                temporary = {p for p, w in facts(state, "transient-part") if w == whole}
                installed = {p for p, w in facts(state, "incorporated") if w == whole}
                require(
                    permanent <= installed and not temporary & installed,
                    "complete assemblies contain all permanent parts and no temporary fixtures",
                )
        elif game == "floortile":
            positions = facts(state, "robot_at")
            painted = facts(state, "painted")
            require(
                len({r for r, t in positions}) == len(positions)
                and len({t for r, t in positions}) == len(positions),
                "one location per robot and no overlap",
            )
            require(
                len({t for t, c in painted}) == len(painted),
                "one paint colour per tile",
            )
            unavailable = {t for r, t in positions} | {t for t, c in painted}
            require(
                {t for (t,) in facts(state, "clear")}
                == {t for (t,) in facts(initial, "tile")} - unavailable,
                "painted and occupied tiles cannot be clear",
            )


def phrase(node, vocabulary, binding=None, effect=False):
    binding = binding or {}
    head = node[0]
    if head == "and":
        return (
            "; ".join(phrase(x, vocabulary, binding, effect) for x in node[1:])
            or "no additional condition"
        )
    if head == "or":
        return (
            "at least one of ["
            + " OR ".join("[" + phrase(x, vocabulary, binding) + "]" for x in node[1:])
            + "]"
        )
    if head == "not":
        return (
            ("make false: " if effect else "it is false that ")
            + "["
            + phrase(node[1], vocabulary, binding)
            + "]"
        )
    if head in {"when", "imply"}:
        return (
            "if ["
            + phrase(node[1], vocabulary, binding)
            + "], then ["
            + phrase(node[2], vocabulary, binding, effect)
            + "]"
        )
    if head in {"forall", "exists"}:
        quantifier = "for every" if head == "forall" else "for at least one"
        return (
            quantifier
            + " "
            + ", ".join(x.removeprefix("?") for x in node[1])
            + ": ["
            + phrase(node[2], vocabulary, binding, effect)
            + "]"
        )
    args = [binding.get(x, x.removeprefix("?")) for x in node[1:]]
    if head == "=":
        return f"{args[0]} equals {args[1]}"
    if head not in vocabulary:
        if len(args) != 1:
            raise KeyError("Missing literal English meaning: " + sexpr(node))
        return f"{args[0]} is an entity of category {head.replace('_', ' ').replace('-', ' ')}"
    return vocabulary[head].format(*args)


def instruction(domain_text, problem_text, vocabulary, notes, game):
    """Short game rules; instance positions and mutable state come from pixels."""
    domain, problem = parse_sexpr(domain_text), parse_sexpr(problem_text)
    initial = section(problem, ":init")[1:]
    rules = {
        "frozenlake": "Move one orthogonal cell onto safe ice. Water is impassable. Cracked ice collapses permanently when left. A checkpoint is recorded only by record_checkpoint while standing on it, once only.",
        "maze": "Turn left/right by 90 degrees. Move forward one cell through an open passage in the current heading. Doors block passage until unlocked. Pick up a key in the current cell; face its adjacent matching door to unlock it. Unlocking consumes the key and is permanent. Keys have unlimited inventory capacity; keyN fits doorN.",
        "sokoban": "Move one orthogonal cell into an empty floor cell. Push one adjacent box forward into an empty cell, moving the player into its old cell. Pulling and pushing multiple boxes are forbidden. push_to_goal ends on a marked target; push_to_nongoal ends on an unmarked floor cell. A box is on_goal exactly when on a target.",
        "package": "Turn by 90 degrees; move forward into an empty cell. Packages block movement even when open. Small boxes inside a larger box are accessible only after their immediate parent opens. Collect cutter/pliers in the current cell; tools are reusable and inventory is unlimited. Face the adjacent package to cut tape with cutter, remove its metal clip with pliers, or open it. Opening requires no tape and no clip. Tape-cut means no tape remains. Cutting and removing clips also require the parent open.",
        "printer": "Turn by 90 degrees; move forward into an empty cell. Printers and tables block movement; cartridges do not. One hand holds one object. Face the adjacent printer/table for printer operations. Pick up an unmounted printer with an empty hand and mount it on a free table; mounting frees the hand and is permanent. Pick up or put down a cartridge in the current cell. Install a matching held cartridge in a mounted, powered-off printer with no cartridge; removing a cartridge requires power off and an empty hand. Both reset calibration. Loading requires a mounted printer and an empty hand and adds one sheet. Power-on requires mounting and a cartridge. Toggle power only to the opposite state; power-off resets calibration. Calibrate a powered, uncalibrated printer. Print an unprinted assigned document using its required cartridge, power and calibration, consuming one sheet. nN denotes N sheets; counts are n0..n3. cartridgeI_J fits printerI.",
        "kitchen": "Move one orthogonal cell; objects and stations do not block movement. One hand holds one ingredient or plate. Pick up in the current cell with an empty hand; put_down frees it. Process a held ingredient at its matching clean station, in recipe order; advance its phase and dirty the tool. The final phase makes it prepared. Clean a dirty tool at its station with an empty hand. Put a prepared held ingredient onto its assigned plate in the current cell, freeing the hand. Hold a plate containing all required prepared ingredients to finish its recipe at the matching clean station; finish only an unready dish, making it ready and dirtying the tool. Cooling a ready held plate at the rack does not dirty it. Deliver a ready held plate at the green delivery tile; baked dishes require cooling. Delivery places the plate there and frees the hand. Plate contents travel with it. METHOD_station supports METHOD; phases count processing steps from phase0(raw).",
        "blocks": "One arm holds one block. pickup lifts a clear block from the table with an empty arm; unstack lifts a clear block from its direct supporting block, clearing that support. putdown places the held block on the table; stack places it on a clear block. Both free the arm. clear means supported on the table or another block with nothing directly above. Any block may support any other block.",
        "hanoi": "Move a top, clear disk from its direct support onto an empty peg or a top, clear larger disk. The old support becomes clear and the new support occupied. Disk dN has size N; d1 is smallest. Pegs are supports, never movable disks.",
        "sliding_puzzle": "Each move swaps one tile with the orthogonally adjacent blank. Operation names describe BLANK movement, opposite to tile movement. In parameters (tile, from, to), from is the tile cell and to the blank cell before the swap. There is exactly one blank.",
        "gripper": "The robot can move directly between any rooms. Its left and right grippers each hold one ball. pick requires the ball and robot in the same room and the selected gripper free; drop places its held ball in the current room and frees that gripper. Carried balls have no independent room location.",
        "miconic": "The lift may travel directly to any higher floor using up or any lower floor using down. board picks up an unserved waiting passenger at their origin; depart removes that passenger only at their destination and marks them served. Boarding consumes one free slot and departing restores one. Floor fN has height N. nN means N free slots.",
        "lights_out": "Pressing buttonR_C simultaneously flips lightR_C and its orthogonal neighbours using their previous states. A button belongs to row panelR. Connect a panel only when the shared connector is free; disconnecting frees it. Press only buttons of the connected panel. Buttons may be repeated. Rows and columns are zero-based from top-left.",
        "floortile": "Move a robot one orthogonal tile into a clear tile, clearing its old tile. Paint only the clear tile immediately above or below the robot using its current colour; painting permanently makes that tile unavailable for entry or repainting. change_color selects any available colour. A tile is clear exactly when unpainted and without a robot. tile_R_C rows start at the bottom with row0 and increase upward; columns increase rightward.",
        "key_delivery": "Move along drawn room passages only after their doors are unlocked. One hand holds one key or relic. Pick up in the current room with an empty hand. A held keyN unlocks doorN from either adjacent room; this consumes the key and frees the hand. Doors stay unlocked. Put down a held relic in any current room, freeing the hand; delivery is recorded only in its assigned destination. Relics may be temporarily put down.",
        "termes": "Move to an orthogonal neighbour with equal height, or exactly one level up/down. One block can be carried. create_block at a depot with an empty carrier obtains a block; destroy_block at a depot discards it. place_block requires a carried block and an adjacent non-depot tile at the robot height; raise that tile by one and empty the carrier. remove_block requires an empty carrier and an adjacent tile exactly one level higher than the robot; lower it by one and carry that block. hN means height N.",
        "freecell": "Move only one exposed card at a time. A column card may stack on an exposed card of the opposite colour with rank exactly one greater (only cards of rank 2 or higher can stack), go to an empty freecell, start an empty column, or go to its same-suit foundation in increasing rank. A card in a freecell may move to a legal column, empty column, or foundation. Foundation cards cannot leave. The home state identifies only the current foundation top; sending a card replaces that top. -b actions move a bottom column card and free its column; other column-source actions uncover oldcard. Freecell/empty-column counters decrease on entry and increase on exit. clear applies to column tops only; on is the immediate column support. C/S suits are black, H/D red. Rank-zero cards are fixed empty-foundation bases. Card prefixes c,h,s,d identify suits; numeric suffixes are ranks and a is ace (rank 1); suit IDs are c, h, s, d. nN is a rank, cellnN a freecell count and colnN an empty-column count.",
        "briefcase": "Move the briefcase directly between any rooms. put_in requires a loose item in the same room as the case; take_out removes a contained item in the current room. Capacity is unlimited. Moving simultaneously moves every contained item to the new room; loose items stay. A contained item retains the room location of the case.",
    }
    if game not in rules:
        raise ValueError("Not a visual game: " + game)
    goal_node = section(problem, ":goal")[1]
    goal = "Achieve this final state: " + phrase(goal_node, vocabulary)
    if game == "sokoban":
        goal = "Put every box on a marked target"
    elif game == "lights_out":
        goal = "Turn off every light and disconnect all panels"
    elif game == "miconic":
        goal = "Serve every passenger"
    elif game == "freecell":
        goal = "Complete each same-suit foundation up to its highest card rank"
    elif game == "briefcase":
        goals = goal_node[1:]
        goal = "Deliver " + "; ".join(
            f"{a[1]} to {a[2]}" for a in goals if a[0] == "at"
        )
        outside = [a[1][1] for a in goals if a[0] == "not" and a[1][0] == "in"]
        if outside:
            goal += ". Leave " + ", ".join(outside) + " outside the briefcase"
        for a in goals:
            if a[0] == "is_at":
                goal += ". Finish with the briefcase in " + a[1]
    lines = [
        goal + ".",
        "Rules: " + rules[game],
    ]
    if game in {
        "frozenlake",
        "maze",
        "sokoban",
        "package",
        "printer",
        "kitchen",
        "sliding_puzzle",
        "termes",
    }:
        lines.append(
            "r01c01 is top-left; rows increase down, columns right. North/east/south/west mean up/right/down/left."
        )
    if game in {"maze", "package", "printer"}:
        lines.append(
            "The player arrow shows heading. For movement/interactions, from/to are player/target cells and d the heading; turns take old/new headings."
        )
    if game == "package":
        lines.append(
            "pN labels denote packageN; cutaway views show nested boxes. A beige strip is tape; a grey loop is a clip."
        )
    elif game == "printer":
        lines.append(
            "pN/tN/cI_J labels denote printerN/tableN/cartridgeI_J. A printer badge counts sheets; its dark indicator is power off."
        )
    elif game == "kitchen":
        lines.append(
            "Food badges count completed processing steps; a green station dot means clean. METHOD labels denote METHOD_station."
        )
    elif game == "frozenlake":
        lines.append("cpN denotes checkpointN.")
    if game == "frozenlake":
        if any(a[0] == "before" for a in initial):
            lines.append("Record checkpoints in increasing checkpoint number.")
        else:
            lines.append("Checkpoints may be recorded in any order.")
    elif game == "printer":
        jobs = {a[1]: a[2] for a in initial if a[0] == "job_for"}
        lines.append(
            "Jobs: "
            + "; ".join(
                f"{a[1]} uses {jobs[a[1]]}/{a[2]}"
                for a in initial
                if a[0] == "document_cartridge"
            )
            + "."
        )
        if "recalibration" in " ".join(notes).lower() or any(
            "Printing also consumes calibration" in n for n in notes
        ):
            lines.append(
                "Printing consumes calibration; recalibrate for each document."
            )
    elif game == "kitchen":
        for note in notes:
            if note.startswith("Ingredient stage sequences:") or note.startswith(
                "Dish assignments:"
            ):
                lines.append(
                    note.replace("Ingredient stage sequences:", "Recipes:").replace(
                        "Dish assignments:", "Dishes:"
                    )
                )
    elif game == "miconic":
        counts = [a[1] for a in initial if a[0] == "free_slots"]
        lines.append(
            "Capacity: "
            + (str(int(counts[0][1:])) if counts else "unlimited")
            + " passengers. Passenger badges show their destination floor."
        )
    elif game == "floortile":
        lines.append(
            "Paint colours are the swatches in the scene; a robot badge shows its current colour."
        )
    elif game == "key_delivery":
        lines.append(
            "Deliver: "
            + "; ".join(f"{a[1]} to {a[2]}" for a in initial if a[0] == "delivery_room")
            + "."
        )
    elif game == "termes":
        heights = sorted({x for a in initial if a[0] == "numb" for x in a[1:]})
        lines.append(
            "Height levels: "
            + ", ".join(heights)
            + ". The outlined depot supplies blocks."
        )
    elif game == "freecell":
        for pred, label in [
            ("cellnum", "Freecell counts"),
            ("colnum", "Empty-column counts"),
            ("num", "Ranks"),
        ]:
            lines.append(
                label + ": " + ", ".join(a[1] for a in initial if a[0] == pred) + "."
            )
    lines.append("Available operation names and parameter order:")
    for a in domain:
        if isinstance(a, list) and a[:1] == [":action"]:
            lines.append(
                a[1]
                + "("
                + ", ".join(p.removeprefix("?") for p in fields(a)[":parameters"])
                + ")"
            )
    lines.append(
        "Use visible IDs and these interfaces; choose predicate names while preserving the rules. Encode the depicted state and rule-implied bookkeeping; other facts are false. Unchanged facts persist."
    )
    return "\n".join(lines) + "\n"


def make_problem(name, objects, initial, goal):
    return (
        sexpr(
            [
                "define",
                ["problem", name],
                [":domain", name],
                [":objects", *objects],
                [":init", *initial],
                [":goal", ["and", *goal]],
            ]
        )
        + "\n"
    )


def make_domain(name, actions, initial, goal, constants=()):
    arities = {}
    logical = {"and", "or", "not", "forall", "exists", "when", "imply", "="}

    def collect(node):
        if not isinstance(node, list) or not node:
            return
        if node[0] in {"forall", "exists"}:
            collect(node[2])
        elif node[0] not in logical and all(isinstance(x, str) for x in node):
            arities[node[0]] = len(node) - 1
        else:
            for x in node[1:]:
                collect(x)

    for a in actions:
        f = fields(a)
        collect(f[":precondition"])
        collect(f[":effect"])
    for atom in initial + goal:
        collect(atom)
    body = ["define", ["domain", name], [":requirements", ":adl"]]
    if constants:
        body.append([":constants", *constants])
    body.extend(
        [
            [
                ":predicates",
                *(
                    [p, *(f"?x{i}" for i in range(n))]
                    for p, n in sorted(arities.items())
                ),
            ],
            *actions,
        ]
    )
    return sexpr(body) + "\n"


def action(name, params, pre, effects):
    """Parse compact, standard PDDL clauses; no separate operator DSL."""
    return [
        ":action",
        name,
        ":parameters",
        safe_variables(params).split(),
        ":precondition",
        parse_sexpr(safe_variables(pre)),
        ":effect",
        parse_sexpr(safe_variables(effects)),
    ]


def safe_variables(text):
    # VAL treats "after" as a keyword even in some variable declarations.
    return re.sub(
        r"\?(before|after)\b",
        lambda m: "?previous" if m[1] == "before" else "?result",
        text,
    )


def read_plan(path):
    return [
        tuple(re.search(r"\(([^()]+)\)", line)[1].lower().split())
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(";")
    ]
