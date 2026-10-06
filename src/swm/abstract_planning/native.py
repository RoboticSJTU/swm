"""IPC domains with generated task structures and explicit rule variants."""

from __future__ import annotations

import random
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from swm.pddl.canonical import canonicalize_pddl_text
from .model import fields, parse_sexpr, section, sexpr, typed_symbols, untype
from .vocabulary import VOCABULARY

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / "downloads/pddl-generators"
FOLDERS = {
    "blocks": "blocksworld",
    "hanoi": "hanoi",
    "sliding_puzzle": "npuzzle",
    "gripper": "gripper",
    "miconic": "miconic",
    "floortile": "floortile",
    "termes": "termes",
    "freecell": "freecell",
    "briefcase": "briefcaseworld",
}


def command(folder, args):
    # Upstream tools may write fixed names such as STATES in their cwd.
    with tempfile.TemporaryDirectory(prefix="swm-generator-") as working:
        result = subprocess.run(
            [str(x) for x in args],
            cwd=working,
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ | {"PYTHONHASHSEED": "0"},
        )
    if result.returncode:
        raise ValueError(f"{folder}: {(result.stdout + result.stderr)[-1200:]}")
    start = result.stdout.lower().find("(define")
    if start < 0:
        raise ValueError(f"{folder} did not generate PDDL: {result.stdout[-300:]}")
    return canonicalize_pddl_text(result.stdout[start:])


def native_problem(domain, categories, initial, goal):
    objects = []
    for kind, values in categories.items():
        if values:
            objects += [*values, "-", kind]
    return sexpr(
        [
            "define",
            ["problem", "instance"],
            [":domain", parse_sexpr(domain.lower())[1][1]],
            [":objects", *objects],
            [":init", *initial],
            [":goal", ["and", *goal]],
        ]
    )


def generate(game, seed, size=None):
    size = size or {}
    rng = random.Random(seed)
    folder = FOLDERS[game]
    domain = canonicalize_pddl_text((SOURCES / folder / "domain.pddl").read_text())
    vocabulary = dict(VOCABULARY[game])
    notes = [
        "The image shows the current scene. Static capabilities, resource requirements and directed relations are specified below. Layout positions without a location predicate are illustrative and impose no movement constraint.",
        "The task requires feasibility, not minimizing an omitted numeric action cost.",
    ]
    variant, derived = "standard", []
    if game == "blocks":
        n = size.get("blocks", rng.randint(6, 15))
        problem = command(folder, [SOURCES / folder / "blocksworld", 4, n, seed])
        variant = "support_restrictions" if seed % 3 else "unrestricted_supports"
        if seed % 3:
            d, p = parse_sexpr(domain), parse_sexpr(problem.lower())
            section(d, ":predicates").append(["can_support", "?lower", "?upper"])
            stack = next(
                a for a in d if isinstance(a, list) and a[:2] == [":action", "stack"]
            )
            f = fields(stack)
            f[":precondition"].append(
                ["can_support", f[":parameters"][1], f[":parameters"][0]]
            )
            nodes = section(p, ":objects")[1:]
            required = {
                (a[2], a[1])
                for a in section(p, ":init")[1:] + section(p, ":goal")[1][1:]
                if a[0] == "on"
            }
            permitted = required | {
                (a, b) for a in nodes for b in nodes if a != b and rng.random() < 0.45
            }
            section(p, ":init").extend(
                ["can_support", a, b] for a, b in sorted(permitted)
            )
            domain, problem = sexpr(d), sexpr(p)
            vocabulary["can_support"] = "block {0} is permitted to support block {1}"
        notes.append(
            "There is one arm. A block is supported directly by the table or by the block shown; support permissions are explicit, not inferred from appearance."
        )
    elif game == "hanoi":
        n = size.get("disks", rng.randint(5, 9))
        disks, pegs = [f"d{i}" for i in range(1, n + 1)], ["peg1", "peg2", "peg3"]

        def piles():
            result = [[] for _ in pegs]
            for disk in reversed(disks):
                result[rng.randrange(3)].append(disk)
            return result

        initial, goal = [], []
        start_piles, goal_piles = piles(), piles()
        if "goal_walk" in size:
            # A larger legal configuration with a reachable full-layout goal.
            goal_piles = [list(stack) for stack in start_piles]
            previous = None
            for _ in range(size["goal_walk"]):
                moves = [(a, b) for a in range(3) for b in range(3)
                         if a != b and goal_piles[a]
                         and (not goal_piles[b] or int(goal_piles[a][-1][1:]) < int(goal_piles[b][-1][1:]))
                         and (a, b) != previous]
                a, b = rng.choice(moves)
                goal_piles[b].append(goal_piles[a].pop())
                previous = (b, a)
        for stacks, dest in [(start_piles, initial), (goal_piles, goal)]:
            for peg, stack in zip(pegs, stacks):
                for below, above in zip([peg, *stack], stack):
                    dest.append(["on", above, below])
                if dest is initial:
                    dest.append(["clear", stack[-1] if stack else peg])
        initial += [
            ["smaller", support, disk]
            for disk in disks
            for support in pegs + disks
            if support in pegs or int(support[1:]) > int(disk[1:])
        ]
        problem = native_problem(domain, {"object": disks + pegs}, initial, goal)
        notes.append(
            "d1 is the smallest disk; size increases with the number. An empty peg is clear. The support relation permits a peg or a strictly larger disk underneath a disk."
        )
        derived = ["smaller"]
        variant = "distributed_towers_and_compound_goal"
    elif game == "sliding_puzzle":
        h, w = size.get("board", (3, 3) if seed % 2 else (2, 4))
        cells = [f"r{r + 1:02d}c{c + 1:02d}" for r in range(h) for c in range(w)]
        tiles = [f"t{i + 1}" for i in range(len(cells) - 1)]
        goal_places = rng.sample(cells, len(cells))
        positions = dict(zip(tiles, goal_places[:-1]))
        blank = goal_places[-1]
        goal = [["at", t, c] for t, c in positions.items()] + [["empty", blank]]
        previous = None
        for _ in range(size.get("scramble", rng.randint(24, 60))):
            r, c = int(blank[1:3]), int(blank[4:6])
            choices = [
                cell
                for cell in cells
                if cell != previous
                and abs(int(cell[1:3]) - r) + abs(int(cell[4:6]) - c) == 1
            ]
            target = rng.choice(choices)
            tile = next(t for t, c in positions.items() if c == target)
            previous = blank
            positions[tile], blank = blank, target
        initial = [["at", t, c] for t, c in positions.items()] + [["empty", blank]]
        d = parse_sexpr(domain)
        original = next(a for a in d if isinstance(a, list) and a[:1] == [":action"])
        d.remove(original)
        section(d, ":predicates")[:] = [
            ":predicates",
            ["at", "?tile", "?position"],
            ["empty", "?position"],
        ]
        for direction, delta in [
            ("north", (-1, 0)),
            ("east", (0, 1)),
            ("south", (1, 0)),
            ("west", (0, -1)),
        ]:
            pred = direction + "_of"
            section(d, ":predicates").append([pred, "?p1", "?p2"])
            a = parse_sexpr(sexpr(original))
            a[1] = "move_blank_" + direction
            f = fields(a)

            def change(node):
                if isinstance(node, list):
                    if node[:1] == ["neighbor"]:
                        node[0] = pred
                    for child in node:
                        change(child)

            change(f[":precondition"])
            d.append(a)
            for origin in cells:
                r, c = int(origin[1:3]), int(origin[4:6])
                target = f"r{r + delta[0]:02d}c{c + delta[1]:02d}"
                if target in cells:
                    initial.append([pred, target, origin])
            vocabulary[pred] = "cell {0} is immediately " + direction + " of cell {1}"
        domain = sexpr(d)
        problem = native_problem(
            domain, {"tile": tiles, "position": cells}, initial, goal
        )
        derived = [d + "_of" for d in ["north", "east", "south", "west"]]
        notes.append(
            "r01c01 is the top-left cell; rows increase down and columns right. Operation names describe BLANK movement, opposite to tile movement. Each move swaps a tile with the adjacent blank."
        )
        variant = f"{h}x{w}_compound_layout_goal"
    elif game == "gripper":
        rooms = [f"room{i}" for i in range(size.get("rooms", rng.randint(3, 6)))]
        balls = [f"ball{i}" for i in range(size.get("balls", rng.randint(5, 10)))]
        grips = ["left", "right"]
        initial = (
            [["room", r] for r in rooms]
            + [["ball", b] for b in balls]
            + [["gripper", g] for g in grips]
        )
        initial += [["free", g] for g in grips] + [["at_robby", rng.choice(rooms)]]
        goal = []
        for b in balls:
            start = rng.choice(rooms)
            target = rng.choice([r for r in rooms if r != start])
            initial.append(["at", b, start])
            goal.append(["at", b, target])
        goal += [["free", g] for g in grips]
        problem = native_problem(
            domain, {"object": rooms + balls + grips}, initial, goal
        )
        variant = "multiple_origins_and_destinations"
        notes.append(
            "Each of the two grippers holds at most one ball. Rooms permit direct robot movement between any two rooms. Balls carried in grippers do not have an independent room location."
        )
    elif game == "miconic":
        problem = command(
            folder,
            [
                SOURCES / folder / "miconic",
                "-f",
                size.get("floors", rng.randint(4, 9)),
                "-p",
                size.get("passengers", rng.randint(4, 9)),
                "-r",
                seed,
            ],
        )
        d, p = parse_sexpr(domain), parse_sexpr(problem.lower())
        passengers = [
            x for x, t in typed_symbols(section(p, ":objects")[1:]) if t == "passenger"
        ]
        section(p, ":init").extend(
            [pred, x] for x in passengers for pred in ["not_boarded", "not_served"]
        )
        board = next(
            a for a in d if isinstance(a, list) and a[:2] == [":action", "board"]
        )
        depart = next(
            a for a in d if isinstance(a, list) and a[:2] == [":action", "depart"]
        )
        fields(board)[":precondition"] += [["not_boarded", "?p"], ["not_served", "?p"]]
        board[board.index(":effect") + 1] = [
            "and",
            ["boarded", "?p"],
            ["not", ["not_boarded", "?p"]],
        ]
        fields(depart)[":effect"] += [
            ["not_boarded", "?p"],
            ["not", ["not_served", "?p"]],
        ]
        variant = "unlimited_capacity"
        if seed % 3:
            capacity = 1 + seed % 2
            section(d, ":types").append("slotcount")
            section(d, ":predicates").extend(
                [
                    ["free_slots", "?n", "-", "slotcount"],
                    ["slot_succ", "?more", "?less", "-", "slotcount"],
                ]
            )
            section(p, ":objects").extend(
                [*(f"n{i}" for i in range(capacity + 1)), "-", "slotcount"]
            )
            section(p, ":init").extend(
                [
                    ["free_slots", f"n{capacity}"],
                    *(["slot_succ", f"n{i + 1}", f"n{i}"] for i in range(capacity)),
                ]
            )
            for a, removing in [(board, True), (depart, False)]:
                f = fields(a)
                f[":parameters"] += ["?before", "?after", "-", "slotcount"]
                f[":precondition"] += [
                    ["free_slots", "?before"],
                    [
                        "slot_succ",
                        "?before" if removing else "?after",
                        "?after" if removing else "?before",
                    ],
                ]
                f[":effect"] += [
                    ["not", ["free_slots", "?before"]],
                    ["free_slots", "?after"],
                ]
            vocabulary.update(
                {
                    "free_slots": "the lift has {0} free passenger slots",
                    "slot_succ": "slot count {0} is exactly one greater than {1}",
                }
            )
            notes.append(
                "nN denotes N free passenger slots. Boarding consumes one slot; departing restores one. Boarding a served passenger is forbidden."
            )
            variant = f"capacity_{capacity}"
        domain, problem = sexpr(d), sexpr(p)
        notes.append(
            "Floors are ordered by the displayed above relations. The lift may travel directly between any two floors in the indicated up/down order."
        )
        derived = ["slot_succ"]
    elif game == "floortile":
        rows, columns, robots = rng.randint(3, 5), rng.randint(3, 5), rng.randint(1, 2)
        rows, columns = size.get("rows", rows), size.get("columns", columns)
        generator_seed = rng.randrange(1, 2**31)
        problem = command(
            folder,
            [
                sys.executable,
                SOURCES / folder / "floortile-generator.py",
                f"p{seed}",
                rows,
                columns,
                robots,
                "seq",
                generator_seed,
            ],
        )
        p = parse_sexpr(problem.lower())
        palette = ["white", "black"] + (["red"] if seed % 3 == 2 else [])
        if len(palette) == 3:
            section(p, ":objects").extend(["red", "-", "color"])
            section(p, ":init").append(["available_color", "red"])
        pattern = ["checkerboard", "row_bands", "column_bands", "block_mosaic"][
            seed % 4
        ]
        for goal in section(p, ":goal")[1][1:]:
            r, c = map(int, goal[1].removeprefix("tile_").split("_"))
            r -= 1
            c -= 1
            index = {
                "checkerboard": r + c,
                "row_bands": r,
                "column_bands": c,
                "block_mosaic": r // 2 + c // 2,
            }[pattern]
            goal[2] = palette[index % len(palette)]
        endpoints = rng.sample(range(1, columns + 1), robots)
        section(p, ":goal")[1].extend(
            ["robot_at", f"robot{i + 1}", f"tile_0_{c}"]
            for i, c in enumerate(endpoints)
        )
        problem = sexpr(p)
        notes.append(
            "In tile_R_C, rows R increase upward and columns C rightward; row 0 is the bottom parking row. Painted tiles cease to be clear and cannot be entered. Robots paint the cell immediately above or below their own cell. Direction relations refer to the board, not a robot heading. Finish with every robot at its required parking tile."
        )
        variant = f"{pattern}_{len(palette)}_colours_return_positions"
    elif game == "termes":
        h, w = size.get("board", (rng.randint(3, 5), rng.randint(3, 5)))
        cells = [f"r{r + 1:02d}c{c + 1:02d}" for r in range(h) for c in range(w)]
        depot = rng.choice(cells) if size.get("random_depot") else cells[0]
        targets = rng.sample([c for c in cells if c != depot], rng.randint(2, 4))
        heights = {c: (rng.randint(1, 2) if c in targets else 0) for c in cells}
        initial = [["at", depot], ["is_depot", depot]] + [
            ["height", c, "h0"] for c in cells
        ]
        initial += [["succ", "h1", "h0"], ["succ", "h2", "h1"]]
        for a in cells:
            ar, ac = int(a[1:3]), int(a[4:6])
            for b in cells:
                if abs(ar - int(b[1:3])) + abs(ac - int(b[4:6])) == 1:
                    initial.append(["neighbor", a, b])
        goal = [["height", c, f"h{height}"] for c, height in heights.items()] + [
            ["not", ["has_block"]],
            ["at", depot],
        ]
        problem = native_problem(
            domain, {"position": cells, "numb": ["h0", "h1", "h2"]}, initial, goal
        )
        notes.append(
            "r01c01 is top-left; rows increase down and columns right. hN denotes height N. Neighbours are orthogonal. The robot carries at most one block. Temporary scaffolding must be removed wherever the goal requires height zero."
        )
        derived = ["succ", "neighbor"]
        variant = "build_towers_and_remove_scaffolding"
    elif game == "freecell":
        rank = size.get("rank", rng.randint(3, 5))
        columns = rng.randint(3, 5)
        args = [
            SOURCES / folder / "freecell",
            "-f",
            rng.randint(1, 3),
            "-c",
            columns,
            "-s",
            2,
            "-i",
            rng.randint(2, min(4, columns)),
            "-0",
            rank,
            "-1",
            rank,
            "-r",
            seed,
        ]
        problem = command(folder, args)
        notes.append(
            "Tableau stacking uses the explicit allowed-card pairs; foundation moves require the same suit and the next rank. Freecells and empty columns are limited resources. Counter successor relations specify the exact before/after capacity updates."
        )
        notes.append(
            "nN denotes rank N. Rank-zero cards are fixed foundation bases, representing an empty foundation rather than a playable card; rank one is an ace. Suits c and s are black, h and d are red."
        )
        variant = "limited_buffers_and_foundation_order"
    elif game == "briefcase":
        rooms = [f"room{i}" for i in range(size.get("rooms", rng.randint(3, 6)))]
        items = [f"item{i}" for i in range(size.get("items", rng.randint(5, 9)))]
        start = rng.choice(rooms)
        initial = [["is_at", start]]
        goal = []
        for item in items:
            origin = rng.choice(rooms)
            target = rng.choice([r for r in rooms if r != origin])
            initial.append(["at", item, origin])
            goal += [["at", item, target], ["not", ["in", item]]]
            if origin == start and rng.random() < 0.3:
                initial.append(["in", item])
        goal.append(["is_at", start])
        problem = native_problem(
            domain, {"location": rooms, "portable": items}, initial, goal
        )
        notes.append(
            "All items inside the briefcase move with it simultaneously. An item in the briefcase also has the case location. Removing an item changes only containment, not location. All delivered items must finish outside the case."
        )
        variant = "automatic_contents_transport_and_multiple_dropoffs"
    else:
        raise KeyError(game)
    # Several upstream generators use a different domain label from domain.pddl.
    p = parse_sexpr(problem.lower())
    section(p, ":domain")[1] = parse_sexpr(domain)[1][1]
    domain, problem = untype(domain, sexpr(p))
    return {
        "game": game,
        "seed": seed,
        "variant": variant,
        "domain": domain,
        "problem": problem,
        "vocabulary": vocabulary,
        "notes": notes,
        "derived": derived,
        "source": {
            "repository": "https://github.com/AI-Planning/pddl-generators",
            "folder": folder,
        },
    }
