"""Recipes, guarded graph exploration, and Boolean circuit puzzles."""

from __future__ import annotations

import random

from .model import action, make_domain, make_problem
from .grid_games import BASE_VOCAB, GRID_NOTE, board


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
    scene=None,
    derived=(),
):
    result = {
        "game": game,
        "seed": seed,
        "variant": variant,
        "domain": make_domain(game, actions, initial, goal, ["player"]),
        "problem": make_problem(game, sorted(set(objects) - {"player"}), initial, goal),
        "vocabulary": vocabulary,
        "notes": notes,
        "derived": list(derived),
        "source": {"repository": "local formalized puzzle rules"},
    }
    if scene:
        result["scene"] = scene
    return result


RECIPES = {
    "salad": {
        "tomato": ["wash", "chop"],
        "onion": ["peel", "chop"],
        "herbs": ["wash", "chop"],
        "finish": "mix",
    },
    "soup": {
        "carrot": ["wash", "peel", "chop", "boil"],
        "potato": ["wash", "peel", "chop", "boil"],
        "beans": ["wash", "boil"],
        "finish": "mix",
    },
    "stir_fry": {
        "mushroom": ["wash", "chop", "fry"],
        "onion": ["peel", "chop", "fry"],
        "rice": ["wash", "boil"],
        "finish": "mix",
    },
    "bread": {
        "grain": ["grind"],
        "egg": ["crack", "whisk"],
        "water": ["measure"],
        "finish": "bake",
    },
    "pasta": {
        "pasta": ["boil"],
        "tomato": ["wash", "chop", "fry"],
        "herbs": ["wash", "chop"],
        "finish": "mix",
    },
    "roast": {
        "potato": ["wash", "peel", "chop"],
        "carrot": ["wash", "peel", "chop"],
        "mushroom": ["wash", "chop"],
        "finish": "bake",
    },
}


def kitchen(seed, size=None):
    rng = random.Random(seed)
    scene = board(rng, *(size or {}).get("board", (3, 4)))
    names = list(RECIPES)
    selected = [names[seed % len(names)]]
    if seed % 3 == 2:
        # Related recipes share tools while still requiring separate dish plans.
        selected.append("pasta" if selected[0] == "stir_fry" else "salad")
    dishes = [f"dish{i}" for i in range(len(selected))]
    plates = [f"plate{i}" for i in range(len(selected))]
    pantry, delivery, start = rng.sample(scene["cells"], 3)
    initial = [["location", c] for c in scene["cells"]] + [
        ["at", "player", start],
        ["arm_empty"],
        ["delivery", delivery],
    ]
    initial += [["adjacent", a, b] for a, b, d in scene["edges"]]
    ingredients = []
    methods = set()
    recipe_info = []
    dish_ingredients = {}
    max_stage = 0
    for index, name in enumerate(selected):
        recipe = RECIPES[name]
        dish, plate = dishes[index], plates[index]
        dish_ingredients[dish] = []
        initial += [
            ["dish", dish],
            ["plate", plate],
            ["item", plate],
            ["plate_for", plate, dish],
            ["at", plate, rng.choice(scene["cells"])],
        ]
        foods = [(food, steps) for food, steps in recipe.items() if food != "finish"]
        if len(selected) > 1:
            # Keep the longer processing chains in smaller two-dish tasks.
            foods = sorted(foods, key=lambda item: -len(item[1]))[:2]
        for food, steps in foods:
            item = f"{food}{index}"
            ingredients.append(item)
            methods.update(steps)
            max_stage = max(max_stage, len(steps))
            dish_ingredients[dish].append(item)
            initial += [
                ["ingredient", item],
                ["item", item],
                ["at", item, pantry],
                ["stage", item, "phase0"],
                ["requires", dish, item],
                ["final_stage", item, f"phase{len(steps)}"],
            ]
            for j, step in enumerate(steps):
                initial.append(["step_" + step, item, f"phase{j}", f"phase{j + 1}"])
            recipe_info.append(
                item
                + ": phase0(raw)"
                + "".join(f" --{step}--> phase{j + 1}" for j, step in enumerate(steps))
            )
        methods.add("finish_" + recipe["finish"])
        initial.append(["finish_" + recipe["finish"] + "_recipe", dish])
        if recipe["finish"] == "bake":
            initial.append(["needs_cooling", dish])
            methods.add("cool")
    tools = {method: method + "_station" for method in sorted(methods)}
    tool_sites = rng.sample([c for c in scene["cells"] if c != delivery], len(tools))
    for (method, tool), loc in zip(tools.items(), tool_sites):
        initial += [
            ["tool", tool],
            [method + "_tool", tool],
            ["tool_at", tool, loc],
            ["clean", tool],
        ]
    initial += [["phase", f"phase{i}"] for i in range(max_stage + 1)]
    vocabulary = BASE_VOCAB | {
        "at": "{0} has its own board position at cell {1}",
        "arm_empty": "the single hand is empty",
        "holding": "the hand holds {0}",
        "delivery": "cell {0} is a delivery station",
        "plate_for": "plate {0} is assigned to dish {1}",
        "stage": "ingredient {0} is at processing stage {1}",
        "requires": "dish {0} requires ingredient {1}",
        "final_stage": "ingredient {0} is fully processed at stage {1}",
        "prepared": "ingredient {0} is fully processed",
        "tool_at": "tool {0} is at cell {1}",
        "clean": "tool {0} is clean",
        "on_plate": "ingredient {0} is on plate {1}",
        "ready": "dish {0} is prepared",
        "needs_cooling": "dish {0} must be cooled before delivery",
        "cooled": "dish {0} has cooled",
        "delivered": "dish {0} has been delivered",
    }
    actions = [
        action(
            "move",
            "?from ?to",
            "(and (location ?from) (location ?to) (at player ?from) (adjacent ?from ?to))",
            "(and (not (at player ?from)) (at player ?to))",
        ),
        action(
            "pickup_ingredient",
            "?i ?loc",
            "(and (ingredient ?i) (at player ?loc) (at ?i ?loc) (arm_empty))",
            "(and (holding ?i) (not (at ?i ?loc)) (not (arm_empty)))",
        ),
        action(
            "put_down",
            "?item ?loc",
            "(and (item ?item) (at player ?loc) (holding ?item))",
            "(and (not (holding ?item)) (arm_empty) (at ?item ?loc))",
        ),
        action(
            "clean_tool",
            "?tool ?loc",
            "(and (tool ?tool) (tool_at ?tool ?loc) (at player ?loc) (arm_empty) (not (clean ?tool)))",
            "(clean ?tool)",
        ),
        action(
            "put_on_plate",
            "?i ?plate ?dish ?loc",
            "(and (ingredient ?i) (plate ?plate) (dish ?dish) (at player ?loc) (at ?plate ?loc) (plate_for ?plate ?dish) (requires ?dish ?i) (holding ?i) (prepared ?i))",
            "(and (on_plate ?i ?plate) (not (holding ?i)) (arm_empty))",
        ),
        action(
            "pickup_plate",
            "?plate ?loc",
            "(and (plate ?plate) (at player ?loc) (at ?plate ?loc) (arm_empty))",
            "(and (holding ?plate) (not (at ?plate ?loc)) (not (arm_empty)))",
        ),
        action(
            "deliver",
            "?dish ?plate ?loc",
            "(and (dish ?dish) (plate ?plate) (plate_for ?plate ?dish) (holding ?plate) (ready ?dish) (at player ?loc) (delivery ?loc) (imply (needs_cooling ?dish) (cooled ?dish)))",
            "(and (delivered ?dish) (not (holding ?plate)) (arm_empty) (at ?plate ?loc))",
        ),
    ]
    for method in sorted(methods):
        vocabulary[method + "_tool"] = "tool {0} supports " + method.replace("_", " ")
        if method == "cool":
            actions.append(
                action(
                    "cool",
                    "?dish ?plate ?tool ?loc",
                    "(and (dish ?dish) (plate ?plate) (plate_for ?plate ?dish) (holding ?plate) (ready ?dish) (needs_cooling ?dish) (tool_at ?tool ?loc) (cool_tool ?tool) (at player ?loc))",
                    "(cooled ?dish)",
                )
            )
        elif method.startswith("finish_"):
            vocabulary[method + "_recipe"] = (
                "dish {0} requires finishing by " + method.removeprefix("finish_")
            )
            actions.append(
                action(
                    method,
                    "?dish ?plate ?tool ?loc",
                    f"(and (dish ?dish) (plate ?plate) (plate_for ?plate ?dish) (holding ?plate) (at player ?loc) (tool_at ?tool ?loc) ({method}_tool ?tool) ({method}_recipe ?dish) (clean ?tool) (not (ready ?dish)) (forall (?i) (imply (requires ?dish ?i) (and (prepared ?i) (on_plate ?i ?plate)))))",
                    "(and (ready ?dish) (not (clean ?tool)))",
                )
            )
        else:
            vocabulary["step_" + method] = (
                "ingredient {0} must use "
                + method
                + " to advance from stage {1} to stage {2}"
            )
            actions.append(
                action(
                    method,
                    "?i ?tool ?loc ?before ?after",
                    f"(and (ingredient ?i) (tool ?tool) (phase ?before) (phase ?after) (holding ?i) (at player ?loc) (tool_at ?tool ?loc) ({method}_tool ?tool) (clean ?tool) (stage ?i ?before) (step_{method} ?i ?before ?after))",
                    "(and (not (stage ?i ?before)) (stage ?i ?after) (not (clean ?tool)) (when (final_stage ?i ?after) (prepared ?i)))",
                )
            )
    notes = [
        GRID_NOTE,
        "There is one hand, holding at most one ingredient or plate. Tools remain at their stations. Ingredients, plates and stations do not block movement. An item has its own board position only while placed directly on a cell; held items and plate contents are located through their holder. Ingredient processing is allowed only in the current required stage; ingredient processing and recipe finishing dirty their tools. Cooling uses a passive rack and leaves tool state unchanged. Cleaning uses unlimited cleaning supplies at that tool station and requires an empty hand. Contents on a plate move with that plate.",
        "Ingredient stage sequences: " + "; ".join(recipe_info) + ".",
        "Dish assignments: "
        + "; ".join(
            f"{dish} uses {plates[i]}, requires exactly "
            + ", ".join(dish_ingredients[dish])
            + f", and follows recipe {name}; finish by {RECIPES[name]['finish']}"
            + (
                "; cool at a cooling station before delivery"
                if RECIPES[name]["finish"] == "bake"
                else ""
            )
            for i, (dish, name) in enumerate(zip(dishes, selected))
        )
        + ".",
    ]
    objects = (
        scene["cells"]
        + ingredients
        + dishes
        + plates
        + list(tools.values())
        + [f"phase{i}" for i in range(max_stage + 1)]
    )
    return finish(
        "kitchen",
        seed,
        "+".join(selected),
        objects,
        initial,
        [["delivered", d] for d in dishes],
        actions,
        vocabulary,
        notes,
        scene,
        [
            "adjacent",
            "requires",
            "final_stage",
            *[p for p in vocabulary if p.startswith("step_")],
        ],
    )


def lights_out(seed, size=None):
    rng = random.Random(seed)
    h, w = (size or {}).get("board", rng.choice([(2, 3), (3, 3), (3, 4), (4, 4), (4, 5)]))
    lamps = [f"light{r}_{c}" for r in range(h) for c in range(w)]
    buttons = [f"button{r}_{c}" for r in range(h) for c in range(w)]
    panels = [f"panel{r}" for r in range(h)]
    initial = (
        [["light", l] for l in lamps]
        + [["button", b] for b in buttons]
        + [["panel", p] for p in panels]
        + [["controller_free"]]
    )
    links = {}
    for r in range(h):
        for c in range(w):
            button = f"button{r}_{c}"
            initial.append(["button_on", button, panels[r]])
            links[button] = [
                f"light{rr}_{cc}"
                for rr in range(h)
                for cc in range(w)
                if abs(rr - r) + abs(cc - c) <= 1
            ]
            initial += [["affects", button, l] for l in links[button]]
    on = set()
    for b in rng.sample(buttons, rng.randint(4, min(9, len(buttons)))):
        on.symmetric_difference_update(links[b])
    initial += [["lit", l] for l in sorted(on)]
    actions = [
        action(
            "connect_panel",
            "?p",
            "(and (panel ?p) (controller_free) (not (connected ?p)))",
            "(and (connected ?p) (not (controller_free)))",
        ),
        action(
            "disconnect_panel",
            "?p",
            "(and (panel ?p) (connected ?p))",
            "(and (not (connected ?p)) (controller_free))",
        ),
        action(
            "press_button",
            "?b ?p",
            "(and (button ?b) (panel ?p) (connected ?p) (button_on ?b ?p))",
            "(and (forall (?l) (when (and (light ?l) (affects ?b ?l) (lit ?l)) (not (lit ?l)))) (forall (?l) (when (and (light ?l) (affects ?b ?l) (not (lit ?l))) (lit ?l))))",
        ),
    ]
    goal = (
        [["not", ["lit", l]] for l in lamps]
        + [["controller_free"]]
        + [["not", ["connected", p]] for p in panels]
    )
    vocab = {
        "controller_free": "the shared panel connector is free",
        "button_on": "button {0} belongs to panel {1}",
        "affects": "button {0} flips light {1}",
        "lit": "light {0} is on",
        "connected": "panel {0} is connected",
    }
    notes = [
        "In lightR_C and buttonR_C, R is the zero-based row from top to bottom and C the zero-based column from left to right. Lights form a rectangular board. Pressing a button flips its own light and each orthogonally adjacent light; all flips use the previous state. Buttons may be pressed repeatedly. Only one row panel can be connected at a time; pressing requires its panel connected. The task also requires disconnecting every panel."
    ]
    return finish(
        "lights_out",
        seed,
        f"{h}x{w}_shared_connector",
        lamps + buttons + panels,
        initial,
        goal,
        actions,
        vocab,
        notes,
    )


def key_delivery(seed, size=None):
    rng = random.Random(seed)
    n = (size or {}).get("rooms", rng.randint(5, 9))
    rooms = [f"room{i}" for i in range(n)]
    doors = [f"door{i}" for i in range(n - 1)]
    keys = [f"key{i}" for i in range(n - 1)]
    relics = [f"relic{i}" for i in range(rng.randint(2, 4))]
    initial = (
        [["room", x] for x in rooms]
        + [["door", x] for x in doors]
        + [["key", x] for x in keys]
        + [["relic", x] for x in relics]
        + [["at_player", rooms[0]], ["hand_empty"]]
    )
    for i, door in enumerate(doors):
        a, b = rooms[i : i + 2]
        key = keys[i]
        initial += [
            ["road", a, b],
            ["road", b, a],
            ["edge_door", a, b, door],
            ["edge_door", b, a, door],
            ["fits", key, door],
            ["key_at", key, rng.choice(rooms[: i + 1])],
        ]
    goal = []
    for relic in relics:
        initial += [
            ["relic_at", relic, rng.choice(rooms[n // 2 :])],
            ["delivery_room", relic, rng.choice(rooms[: n // 2])],
        ]
        goal.append(["delivered", relic])
    goal.append(["at_player", rooms[0]])
    actions = [
        action(
            "move",
            "?from ?to",
            "(and (room ?from) (room ?to) (at_player ?from) (road ?from ?to) (forall (?d) (imply (edge_door ?from ?to ?d) (unlocked ?d))))",
            "(and (not (at_player ?from)) (at_player ?to))",
        ),
        action(
            "pickup_key",
            "?key ?room",
            "(and (key ?key) (at_player ?room) (key_at ?key ?room) (hand_empty))",
            "(and (holding_key ?key) (not (key_at ?key ?room)) (not (hand_empty)))",
        ),
        action(
            "unlock",
            "?door ?key ?from ?to",
            "(and (door ?door) (key ?key) (at_player ?from) (edge_door ?from ?to ?door) (holding_key ?key) (fits ?key ?door) (not (unlocked ?door)))",
            "(and (unlocked ?door) (not (holding_key ?key)) (hand_empty))",
        ),
        action(
            "pickup_relic",
            "?relic ?room",
            "(and (relic ?relic) (at_player ?room) (relic_at ?relic ?room) (hand_empty))",
            "(and (holding_relic ?relic) (not (relic_at ?relic ?room)) (not (hand_empty)))",
        ),
        action(
            "put_down_relic",
            "?relic ?room",
            "(and (relic ?relic) (room ?room) (at_player ?room) (holding_relic ?relic))",
            "(and (relic_at ?relic ?room) (not (holding_relic ?relic)) (hand_empty) (when (delivery_room ?relic ?room) (delivered ?relic)))",
        ),
    ]
    vocabulary = {
        "at_player": "the player is in room {0}",
        "hand_empty": "the single hand is empty",
        "road": "a directed passage leads from room {0} to {1}",
        "edge_door": "door {2} guards the passage from {0} to {1}",
        "fits": "key {0} opens door {1}",
        "key_at": "key {0} rests in room {1} outside the hand",
        "unlocked": "door {0} is unlocked",
        "holding_key": "the hand holds key {0}",
        "relic_at": "relic {0} rests in room {1} outside the hand",
        "delivery_room": "relic {0} must be delivered to room {1}",
        "delivered": "relic {0} has been delivered",
        "holding_relic": "the hand holds relic {0}",
    }
    return finish(
        "key_delivery",
        seed,
        "matched_consumable_keys_and_single_hand_delivery",
        rooms + doors + keys + relics,
        initial,
        goal,
        actions,
        vocabulary,
        [
            "One hand holds at most one key or relic. Unlocking consumes a matching key and frees the hand. Doors stay unlocked permanently. Relics can be put down temporarily; delivery is recorded only when put down in their assigned room."
        ],
    )
