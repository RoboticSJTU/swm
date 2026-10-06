"""Grid tasks with causal subgoals, nested containers and device preparation."""

from __future__ import annotations

import random
from pathlib import Path
from functools import lru_cache

from .model import action, make_domain, make_problem

ROOT = Path(__file__).resolve().parents[3]
DIRECTIONS = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
GRID_NOTE = "r01c01 is the top-left cell; rows increase downward and columns rightward. North is up, east right, south down, west left. Left and right turns rotate the heading by 90 degrees counterclockwise and clockwise. Labels identify board cells. Walls and holes cannot be entered. Cell borders without an opening block movement."
BASE_VOCAB = {
    "at": "{0} is at cell {1}",
    "location": "{0} is a board cell",
    "adjacent": "cell {0} is orthogonally adjacent to cell {1}",
    "ahead": "cell {1} is immediately {2} of cell {0} through an open passage",
    "facing": "the player faces {0}",
    "empty": "cell {0} is free for movement (no player or blocking object)",
    "left_turn": "turning left changes direction {0} to {1}",
    "right_turn": "turning right changes direction {0} to {1}",
}


def cell(r, c):
    return f"r{r + 1:02d}c{c + 1:02d}"


def rc(name):
    return int(name[1:3]) - 1, int(name[4:6]) - 1


def board(rng, h=None, w=None):
    h, w = h or rng.randint(5, 7), w or rng.randint(5, 7)
    cells = [cell(r, c) for r in range(h) for c in range(w)]
    edges = []
    for a in cells:
        r, c = rc(a)
        for d, (dr, dc) in DIRECTIONS.items():
            b = cell(r + dr, c + dc)
            if b in cells:
                edges.append([a, b, d])
    return {"height": h, "width": w, "cells": cells, "edges": edges, "holes": []}


def finish(
    game,
    seed,
    variant,
    objects,
    initial,
    goal,
    actions,
    vocabulary,
    notes,
    scene,
    derived=(),
):
    return {
        "game": game,
        "seed": seed,
        "variant": variant,
        "domain": make_domain(game, actions, initial, goal, ["player"]),
        "problem": make_problem(game, sorted(set(objects) - {"player"}), initial, goal),
        "vocabulary": BASE_VOCAB | vocabulary,
        "notes": [GRID_NOTE, *notes],
        "scene": scene,
        "derived": list(derived),
        "source": {"repository": "local verified game rules"},
    }


def pose_initial(scene, rng, at):
    dirs = list(DIRECTIONS)
    return (
        [["location", c] for c in scene["cells"]]
        + [["direction", d] for d in dirs]
        + [
            ["at", "player", at],
            ["facing", rng.choice(dirs)],
            *(["ahead", a, b, d] for a, b, d in scene["edges"]),
            *(["left_turn", d, dirs[(i - 1) % 4]] for i, d in enumerate(dirs)),
            *(["right_turn", d, dirs[(i + 1) % 4]] for i, d in enumerate(dirs)),
        ]
    )


def pose_actions(movement_extra="", move_effect=""):
    return [
        action(
            "turn_left",
            "?from ?to",
            "(and (direction ?from) (direction ?to) (facing ?from) (left_turn ?from ?to))",
            "(and (not (facing ?from)) (facing ?to))",
        ),
        action(
            "turn_right",
            "?from ?to",
            "(and (direction ?from) (direction ?to) (facing ?from) (right_turn ?from ?to))",
            "(and (not (facing ?from)) (facing ?to))",
        ),
        action(
            "move_forward",
            "?from ?to ?d",
            f"(and (location ?from) (location ?to) (direction ?d) (at player ?from) (facing ?d) (ahead ?from ?to ?d) {movement_extra})",
            f"(and (not (at player ?from)) (at player ?to) {move_effect})",
        ),
    ]


def frozenlake(seed, size=None):
    rng = random.Random(seed)
    scene = board(rng, *(size or {}).get("board", (None, None)))
    start = rng.choice(scene["cells"])
    safe = {c for c in scene["cells"] if rng.random() > 0.22} | {start}
    scene["holes"] = sorted(set(scene["cells"]) - safe)
    markers = [f"checkpoint{i}" for i in range(rng.randint(2, 4))]
    sites = rng.sample(sorted(safe - {start}), len(markers) + 1)
    initial = (
        [["location", c] for c in scene["cells"]]
        + [["safe", c] for c in sorted(safe)]
        + [["at", "player", start]]
    )
    fragile = seed % 3 == 0
    if fragile:
        initial += [["fragile", c] for c in sorted(safe) if rng.random() < 0.18]
    actions = []
    vocab = {
        "safe": "cell {0} is safe",
        "checkpoint_at": "checkpoint {0} is at cell {1}",
        "marked": "checkpoint {0} has been recorded",
        "before": "checkpoint {0} must be recorded before checkpoint {1}",
        "fragile": "cell {0} collapses permanently after the player leaves it",
    }
    for d in DIRECTIONS:
        initial.extend([d + "_of", b, a] for a, b, dir in scene["edges"] if dir == d)
        effect = "(when (fragile ?from) (not (safe ?from)))" if fragile else ""
        actions.append(
            action(
                "move_" + d,
                "?from ?to",
                f"(and (location ?from) (location ?to) (at player ?from) (safe ?to) ({d}_of ?to ?from))",
                f"(and (not (at player ?from)) (at player ?to) {effect})",
            )
        )
        vocab[d + "_of"] = "cell {0} is immediately " + d + " of cell {1}"
    initial += [["checkpoint", m] for m in markers] + [
        ["checkpoint_at", m, c] for m, c in zip(markers, sites)
    ]
    if seed % 2:
        initial += [["before", a, b] for a, b in zip(markers, markers[1:])]
    actions.append(
        action(
            "record_checkpoint",
            "?m ?loc",
            "(and (checkpoint ?m) (location ?loc) (at player ?loc) (checkpoint_at ?m ?loc) (not (marked ?m)) (forall (?prev) (imply (before ?prev ?m) (marked ?prev))))",
            "(marked ?m)",
        )
    )
    variant = ("ordered_checkpoints" if seed % 2 else "multiple_checkpoints") + (
        "_fragile_ice" if fragile else ""
    )
    return finish(
        "frozenlake",
        seed,
        variant,
        scene["cells"] + markers,
        initial,
        [*(["marked", m] for m in markers), ["at", "player", sites[-1]]],
        actions,
        vocab,
        [
            "Entering a checkpoint does not record it automatically. Recording changes only its recorded state. The final location is required in addition to all checkpoint goals. Initially every non-hole cell is safe. Fragile tiles, when listed, become unsafe permanently after departure."
        ],
        scene,
        ["safe", *[d + "_of" for d in DIRECTIONS]],
    )


def maze(seed, size=None):
    rng = random.Random(seed)
    scene = board(rng, *(size or {}).get("board", (None, None)))
    root = rng.choice(scene["cells"])
    stack = [root]
    seen = {root}
    edges = []
    while stack:
        a = stack[-1]
        choices = [(b, d) for x, b, d in scene["edges"] if x == a and b not in seen]
        if not choices:
            stack.pop()
            continue
        b, d = rng.choice(choices)
        seen.add(b)
        stack.append(b)
        opposite = list(DIRECTIONS)[(list(DIRECTIONS).index(d) + 2) % 4]
        edges.extend([[a, b, d], [b, a, opposite]])
    scene["edges"] = edges
    initial = pose_initial(scene, rng, root)
    goal = [["at", "player", rng.choice([c for c in scene["cells"] if c != root])]]
    extra = ""
    actions = []
    objects = scene["cells"] + list(DIRECTIONS)
    vocab = {
        "edge_door": "door {2} blocks the passage from cell {0} to cell {1}",
        "unlocked": "door {0} is unlocked",
        "key_at": "key {0} rests at cell {1} outside the inventory",
        "has_key": "the player carries key {0}",
        "fits": "key {0} unlocks door {1}",
    }
    if seed % 3:
        chosen = rng.sample(edges, len(edges) // 2)[: rng.randint(1, 3)]
        doors = []
        for i, (a, b, d) in enumerate(chosen):
            door, key = f"door{i}", f"key{i}"
            doors.append({"id": door, "from": a, "to": b})
            objects += [door, key]
            initial += [
                ["door", door],
                ["key", key],
                ["edge_door", a, b, door],
                ["edge_door", b, a, door],
                ["key_at", key, root],
                ["fits", key, door],
            ]
            goal.append(["unlocked", door])
        scene["doors"] = doors
        extra = "(forall (?door) (imply (edge_door ?from ?to ?door) (unlocked ?door)))"
        actions += [
            action(
                "pickup_key",
                "?key ?loc",
                "(and (key ?key) (location ?loc) (at player ?loc) (key_at ?key ?loc))",
                "(and (not (key_at ?key ?loc)) (has_key ?key))",
            ),
            action(
                "unlock_door",
                "?door ?key ?from ?to ?d",
                "(and (door ?door) (key ?key) (at player ?from) (facing ?d) (ahead ?from ?to ?d) (edge_door ?from ?to ?door) (fits ?key ?door) (has_key ?key) (not (unlocked ?door)))",
                "(and (unlocked ?door) (not (has_key ?key)))",
            ),
        ]
    actions += pose_actions(extra)
    return finish(
        "maze",
        seed,
        "matching_keys_and_doors" if seed % 3 else "corridor_navigation",
        objects,
        initial,
        goal,
        actions,
        vocab,
        [
            "Turns happen in place. Forward movement follows the current heading through an open passage. For turns, from/to are directions; for movement they are cells. Unlocking consumes the matching key and does not move the player. Keys do not block movement; the inventory can hold multiple keys. Doors are initially locked unless explicitly shown unlocked."
        ],
        scene,
        ["ahead", "left_turn", "right_turn"],
    )


@lru_cache(maxsize=1)
def boxoban_levels():
    levels = []
    for path in sorted((ROOT / "downloads/boxoban").glob("*.txt")):
        current = []
        index = None
        for line in path.read_text().splitlines() + [";end"]:
            if line.startswith(";"):
                if current and len(current) == 10:
                    levels.append(
                        {"board": current, "source_file": path.name, "level": index}
                    )
                index = line[1:].strip()
                current = []
            elif line and set(line) <= set("#@$.*+ "):
                current.append(line)
    if not levels:
        raise ValueError("No downloaded Boxoban levels")
    random.Random(20261005).shuffle(levels)
    return levels


def sokoban(seed, size=None):
    level = boxoban_levels()[seed % len(boxoban_levels())]
    if size:
        # Stretch only existing passages: padding alone is not scale extrapolation.
        lines = level["board"]
        gap = size["stretch"]
        expanded = []
        for r, row in enumerate(lines):
            expanded.append(row)
            if r == 4:
                expanded.extend("".join(" " if row[c] != "#" and lines[r + 1][c] != "#" else "#"
                                        for c in range(10)) for _ in range(gap))
        lines = [row[:5] + (" " if row[4] != "#" and row[5] != "#" else "#") * gap + row[5:]
                 for row in expanded]
    else:
        lines = level["board"]
    scene = {"height": len(lines), "width": len(lines[0]), "cells": [], "edges": [], "holes": []}
    positions = {}
    goals = []
    stones = []
    for r, row in enumerate(lines):
        for c, ch in enumerate(row):
            if ch == "#":
                continue
            loc = cell(r, c)
            scene["cells"].append(loc)
            if ch in "@+":
                positions["player"] = loc
            if ch in "$*":
                stone = f"box{len(stones) + 1}"
                stones.append(stone)
                positions[stone] = loc
            if ch in ".*+":
                goals.append(loc)
    for a in scene["cells"]:
        r, c = rc(a)
        for d, (dr, dc) in DIRECTIONS.items():
            b = cell(r + dr, c + dc)
            if b in scene["cells"]:
                scene["edges"].append([a, b, d])
    initial = (
        [["location", c] for c in scene["cells"]]
        + [["direction", d] for d in DIRECTIONS]
        + [["box", s] for s in stones]
    )
    initial += [["at", o, c] for o, c in positions.items()] + [
        ["empty", c] for c in scene["cells"] if c not in positions.values()
    ]
    initial += [["ahead", a, b, d] for a, b, d in scene["edges"]] + [
        ["goal_cell", c] for c in goals
    ]
    initial += [["non_goal_cell", c] for c in scene["cells"] if c not in goals] + [
        ["on_goal", s] for s in stones if positions[s] in goals
    ]
    actions = [
        action(
            "move",
            "?from ?to ?d",
            "(and (at player ?from) (ahead ?from ?to ?d) (empty ?to))",
            "(and (not (at player ?from)) (at player ?to) (empty ?from) (not (empty ?to)))",
        )
    ]
    for name, predicate in [
        ("push_to_goal", "goal_cell"),
        ("push_to_nongoal", "non_goal_cell"),
    ]:
        effect = "(on_goal ?box)" if name == "push_to_goal" else "(not (on_goal ?box))"
        actions.append(
            action(
                name,
                "?box ?from ?boxloc ?to ?d",
                f"(and (box ?box) (at player ?from) (at ?box ?boxloc) (ahead ?from ?boxloc ?d) (ahead ?boxloc ?to ?d) (empty ?to) ({predicate} ?to))",
                f"(and (not (at player ?from)) (at player ?boxloc) (not (at ?box ?boxloc)) (at ?box ?to) (empty ?from) (not (empty ?to)) {effect})",
            )
        )
    vocab = {
        "box": "{0} is a box",
        "goal_cell": "cell {0} is a box target",
        "non_goal_cell": "cell {0} is not a box target",
        "on_goal": "box {0} is on a target",
    }
    scene["goals"] = goals
    result = finish(
        "sokoban",
        seed,
        "boxoban_" + level["source_file"].split("_")[0],
        scene["cells"] + stones + list(DIRECTIONS),
        initial,
        [["on_goal", s] for s in stones],
        actions,
        vocab,
        [
            "Boxes can only be pushed, never pulled. Player, box and destination must be three consecutive cells in the same direction. Each cell contains at most one player or box. A box pushed into a non-target corner may become permanently stuck. Targets are interchangeable."
        ],
        scene,
        ["ahead", "goal_cell", "non_goal_cell", "empty"],
    )
    result["source"] = {
        "repository": "https://github.com/google-deepmind/boxoban-levels",
        **level,
    }
    return result


def package(seed, size=None):
    rng = random.Random(seed)
    scene = board(rng, *(size or {}).get("board", (rng.randint(4, 5), rng.randint(4, 5))))
    count = rng.randint(3, 6)
    packages = [f"package{i}" for i in range(count)]
    sites = rng.sample(scene["cells"], count + 2)
    start, toolsite = sites[:2]
    clips = seed % 3 != 0
    initial = pose_initial(scene, rng, start) + [
        ["tool", "cutter"],
        ["tool_at", "cutter", toolsite],
        ["cuts_tape", "cutter"],
    ]
    initial += [["package", p] for p in packages] + [
        ["at", p, c] for p, c in zip(packages, sites[2:])
    ]
    for p in packages:
        initial.append(["taped" if rng.random() < 0.7 else "tape_cut", p])
    if clips:
        initial += [
            ["tool", "pliers"],
            ["tool_at", "pliers", toolsite],
            ["removes_clips", "pliers"],
        ]
        initial += [["clipped", p] for p in rng.sample(packages, rng.randint(1, count))]
    for i in range(1, count):
        if seed % 2 or i % 2:
            initial.append(["depends_on", packages[i], packages[rng.randrange(i)]])
    placements = {p: c for p, c in zip(packages, sites[2:])}
    for atom in initial:
        if atom[0] == "depends_on":
            placements[atom[1]] = placements[atom[2]]
    for atom in initial:
        if atom[0] == "at" and atom[1] in placements:
            atom[2] = placements[atom[1]]
    initial += [
        ["empty", c]
        for c in scene["cells"]
        if c not in placements.values() and c != start
    ]
    actions = pose_actions("(empty ?to)", "(empty ?from) (not (empty ?to))")
    actions += [
        action(
            "take_tool",
            "?tool ?loc",
            "(and (tool ?tool) (at player ?loc) (tool_at ?tool ?loc))",
            "(and (not (tool_at ?tool ?loc)) (has_tool ?tool))",
        ),
        action(
            "cut_tape",
            "?p ?from ?to ?d ?tool",
            "(and (package ?p) (at player ?from) (facing ?d) (ahead ?from ?to ?d) (at ?p ?to) (taped ?p) (tool ?tool) (has_tool ?tool) (forall (?other) (imply (depends_on ?p ?other) (open ?other))))",
            "(and (not (taped ?p)) (tape_cut ?p))",
        ),
        action(
            "open_package",
            "?p ?from ?to ?d",
            "(and (package ?p) (at player ?from) (facing ?d) (ahead ?from ?to ?d) (at ?p ?to) (tape_cut ?p) (not (clipped ?p)) (not (open ?p)) (forall (?other) (imply (depends_on ?p ?other) (open ?other))))",
            "(open ?p)",
        ),
    ]
    # Tool roles matter: a pair of pliers cannot cut tape.
    fields = actions[-2][actions[-2].index(":precondition") + 1]
    fields.append(["cuts_tape", "?tool"])
    if clips:
        actions.append(
            action(
                "remove_clip",
                "?p ?from ?to ?d ?tool",
                "(and (package ?p) (tool ?tool) (at player ?from) (facing ?d) (ahead ?from ?to ?d) (at ?p ?to) (clipped ?p) (has_tool ?tool) (removes_clips ?tool) (forall (?other) (imply (depends_on ?p ?other) (open ?other))))",
                "(not (clipped ?p))",
            )
        )
    return finish(
        "package",
        seed,
        "nested_tape_and_clips" if clips else "nested_tape",
        scene["cells"]
        + packages
        + list(DIRECTIONS)
        + ["cutter"]
        + (["pliers"] if clips else []),
        initial,
        [["open", p] for p in packages],
        actions,
        {
            "tool_at": "tool {0} rests at cell {1} outside the inventory",
            "has_tool": "the player has tool {0}",
            "cuts_tape": "tool {0} can cut tape",
            "removes_clips": "tool {0} can remove clips",
            "taped": "package {0} is taped shut",
            "tape_cut": "package {0} has no remaining tape",
            "clipped": "package {0} has a fastening clip",
            "open": "package {0} is open",
            "depends_on": "package {0} is accessible only after package {1} is open",
        },
        [
            "Packages block their cells even after opening. A dependent package is inside its parent at the same cell. Opening requires facing the adjacent package. Access dependencies apply to cutting tape, removing clips and opening. Tools do not block movement and are reusable; inventory can hold multiple tools. Only cutter cuts tape; only pliers removes clips. All remaining tape and clips must be removed before opening."
        ],
        scene,
        ["ahead", "left_turn", "right_turn", "empty"],
    )


def printer(seed, size=None):
    rng = random.Random(seed)
    scene = board(rng, *(size or {}).get("board", (3, rng.randint(3, 4))))
    count = 1 + seed % 2
    printers = [f"printer{i}" for i in range(count)]
    tables = [f"table{i}" for i in range(count)]
    swap = seed % 3 != 0
    reset = seed % 3 == 2
    cartridges = [
        f"cartridge{i}_{j}" for i in range(count) for j in range(2 if swap else 1)
    ]
    sites = rng.sample(scene["cells"], count * 2 + len(cartridges) + 1)
    start = sites[0]
    initial = pose_initial(scene, rng, start) + [["arm_empty"]]
    for i, p in enumerate(printers):
        initial += [
            ["printer", p],
            ["table", tables[i]],
            ["table_free", tables[i]],
            ["at", p, sites[1 + i]],
            ["at", tables[i], sites[1 + count + i]],
            ["paper_level", p, "n0"],
        ]
    for i, c in enumerate(cartridges):
        p = printers[int(c.removeprefix("cartridge").split("_")[0])]
        initial += [
            ["cartridge", c],
            ["at", c, sites[1 + count * 2 + i]],
            ["compatible", c, p],
        ]
    initial += [
        ["empty", c]
        for c in scene["cells"]
        if c not in sites[1 : 1 + count * 2] and c != start
    ]
    documents = [f"document{i}" for i in range(count * 2)]
    initial += [["document", d] for d in documents] + [
        ["job_for", d, printers[i // 2]] for i, d in enumerate(documents)
    ]
    initial += [
        ["document_cartridge", d, f"cartridge{i // 2}_{i % 2 if swap else 0}"]
        for i, d in enumerate(documents)
    ]
    initial += [["count", f"n{i}"] for i in range(4)] + [
        ["count_succ", f"n{i + 1}", f"n{i}"] for i in range(3)
    ]
    actions = pose_actions("(empty ?to)", "(empty ?from) (not (empty ?to))")
    front = "(at player ?from) (facing ?d) (ahead ?from ?to ?d) (at ?p ?to)"
    actions += [
        action(
            "pick_up_printer",
            "?p ?from ?to ?d",
            f"(and (printer ?p) {front} (arm_empty) (not (mounted ?p)))",
            "(and (holding ?p) (not (arm_empty)) (not (at ?p ?to)) (empty ?to))",
        ),
        action(
            "put_on_table",
            "?p ?table ?from ?to ?d",
            "(and (printer ?p) (table ?table) (table_free ?table) (holding ?p) (at player ?from) (facing ?d) (ahead ?from ?to ?d) (at ?table ?to))",
            "(and (not (holding ?p)) (arm_empty) (at ?p ?to) (on ?p ?table) (mounted ?p) (not (table_free ?table)))",
        ),
        action(
            "pick_cartridge",
            "?c ?loc",
            "(and (cartridge ?c) (arm_empty) (at player ?loc) (at ?c ?loc))",
            "(and (holding ?c) (not (arm_empty)) (not (at ?c ?loc)))",
        ),
        action(
            "install_cartridge",
            "?p ?c ?from ?to ?d",
            f"(and (printer ?p) (cartridge ?c) {front} (mounted ?p) (holding ?c) (compatible ?c ?p) (not (installed ?p)) (not (powered ?p)))",
            "(and (installed ?p) (installed_cartridge ?p ?c) (not (holding ?c)) (arm_empty) (not (calibrated ?p)))",
        ),
        action(
            "load_paper",
            "?p ?from ?to ?d ?before ?after",
            f"(and (printer ?p) {front} (mounted ?p) (arm_empty) (paper_level ?p ?before) (count_succ ?after ?before))",
            "(and (not (paper_level ?p ?before)) (paper_level ?p ?after))",
        ),
        action(
            "power_on",
            "?p ?from ?to ?d",
            f"(and (printer ?p) {front} (mounted ?p) (installed ?p) (not (powered ?p)))",
            "(powered ?p)",
        ),
        action(
            "calibrate",
            "?p ?from ?to ?d",
            f"(and (printer ?p) {front} (powered ?p) (not (calibrated ?p)))",
            "(calibrated ?p)",
        ),
        action(
            "print_document",
            "?p ?doc ?c ?from ?to ?d ?before ?after",
            f"(and (printer ?p) (document ?doc) (cartridge ?c) {front} (powered ?p) (calibrated ?p) (job_for ?doc ?p) (document_cartridge ?doc ?c) (installed_cartridge ?p ?c) (not (printed ?doc)) (paper_level ?p ?before) (count_succ ?before ?after))",
            f"(and (printed ?doc) (not (paper_level ?p ?before)) (paper_level ?p ?after) {'(not (calibrated ?p))' if reset else ''})",
        ),
    ]
    if swap:
        actions += [
            action(
                "power_off",
                "?p ?from ?to ?d",
                f"(and (printer ?p) {front} (powered ?p))",
                "(and (not (powered ?p)) (not (calibrated ?p)))",
            ),
            action(
                "remove_cartridge",
                "?p ?c ?from ?to ?d",
                f"(and (printer ?p) (cartridge ?c) {front} (installed_cartridge ?p ?c) (not (powered ?p)) (arm_empty))",
                "(and (not (installed_cartridge ?p ?c)) (not (installed ?p)) (holding ?c) (not (arm_empty)) (not (calibrated ?p)))",
            ),
            action(
                "put_cartridge",
                "?c ?loc",
                "(and (cartridge ?c) (holding ?c) (at player ?loc))",
                "(and (not (holding ?c)) (arm_empty) (at ?c ?loc))",
            ),
        ]
    goal = [["printed", d] for d in documents] + [
        ["on", p, t] for p, t in zip(printers, tables)
    ]
    vocab = {
        "at": "{0} has its own board position at cell {1}",
        "arm_empty": "the single arm is empty",
        "holding": "the arm holds {0}",
        "compatible": "cartridge {0} fits printer {1}",
        "paper_level": "printer {0} contains {1} sheets",
        "count_succ": "sheet count {0} is one above {1}",
        "table_free": "table {0} has no mounted printer",
        "document_cartridge": "document {0} requires cartridge {1}",
        "installed_cartridge": "printer {0} contains cartridge {1}",
        "job_for": "document {0} must be printed by printer {1}",
        "mounted": "printer {0} is mounted on a table",
        "on": "printer {0} is directly on table {1}",
        "installed": "printer {0} has a cartridge installed",
        "powered": "printer {0} is powered on",
        "calibrated": "printer {0} is calibrated",
        "printed": "document {0} has been printed",
    }
    variant = (
        "cartridge_change_recalibration"
        if reset
        else "cartridge_change"
        if swap
        else "single_cartridge"
    )
    return finish(
        "printer",
        seed,
        variant,
        scene["cells"]
        + printers
        + tables
        + cartridges
        + documents
        + list(DIRECTIONS)
        + [f"n{i}" for i in range(4)],
        initial,
        goal,
        actions,
        vocab,
        [
            "The arm holds at most one object. Printers and tables block movement; cartridges do not. Each table holds at most one printer; each printer holds at most one cartridge. Held objects and installed cartridges are located through their holder and have no separate board position; mounted printers retain the position of their table cell. Printer operations act on the adjacent cell directly ahead. Cartridges are picked up or put down in the same cell. nN means N sheets; loading adds one sheet from an unlimited supply, printing consumes one. Installing or removing a cartridge requires power off and resets calibration. Each document requires its listed cartridge. Mounted printers cannot be picked up again."
            + (
                " Printing also consumes calibration; calibrate again for each document."
                if reset
                else ""
            )
        ],
        scene,
        ["ahead", "left_turn", "right_turn", "empty", "count_succ"],
    )
