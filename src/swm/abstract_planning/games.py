"""One registry: actual game families, never aliases or difficulty levels."""

from . import native, grid_games, causal_games

GAMES = [
    "frozenlake",
    "maze",
    "sokoban",
    "package",
    "printer",
    "kitchen",
    "blocks",
    "hanoi",
    "sliding_puzzle",
    "gripper",
    "miconic",
    "lights_out",
    "floortile",
    "key_delivery",
    "termes",
    "freecell",
    "briefcase",
]
MECHANISMS = {
    "frozenlake": "unsafe tiles, ordered checkpoint subgoals and spatial routing",
    "maze": "heading, passages, matching keys and persistent door unlocking",
    "sokoban": "push-only motion, blocking boxes, narrow passages and irreversible deadlocks",
    "package": "nested containment, access dependencies, tape and reusable tools",
    "printer": "mounting, cartridge compatibility, calibration and paper consumption",
    "kitchen": "six recipes, ingredient processing order, shared dirty tools and cooling",
    "blocks": "support occupancy, one arm, compound stack goals and permitted supports",
    "hanoi": "size ordering, clear supports and distributed multi-tower goals",
    "sliding_puzzle": "one blank, bidirectional tile/blank motion and full layout goals",
    "gripper": "two independent single-object grippers and multiple destinations",
    "miconic": "individual origins/destinations, ordered floors and capacity constraints",
    "lights_out": "conditional Boolean flips and a shared row-panel connector",
    "floortile": "irreversible paint, colour changes and movement frontier constraints",
    "key_delivery": "consumable matching keys, single hand and relic delivery",
    "termes": "height-dependent motion, block carrying and temporary scaffolding removal",
    "freecell": "alternating stacks, rank-ordered foundations and limited buffers",
    "briefcase": "conditional transport of every contained item and multiple dropoffs",
}

MECHANISMS = {game: MECHANISMS[game] for game in GAMES}


def generate(game, seed, size=None):
    if game in native.FOLDERS:
        return native.generate(game, seed, size)
    if hasattr(grid_games, game):
        return getattr(grid_games, game)(seed, size)
    return getattr(causal_games, game)(seed, size)
