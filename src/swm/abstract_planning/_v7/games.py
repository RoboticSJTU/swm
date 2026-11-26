"""One registry: actual game families, never aliases or difficulty levels."""

from . import native, grid_games, causal_games

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

GAMES = list(MECHANISMS)


def generate(game, seed, size=None, *, downloads=None):
    if game in native.FOLDERS:
        sources = native.SOURCES
        if downloads is not None:
            external = downloads / "pddl-generators"
            folder = native.FOLDERS[game]
            files = ["domain.pddl"]
            if game in {"blocks", "miconic", "freecell"}:
                files.append(folder)
            if (all((external / folder / name).is_file() for name in files)
                    or not all((sources / folder / name).is_file() for name in files)):
                sources = external
        return native.generate(game, seed, size, sources=sources)
    if game == "sokoban" and downloads is not None:
        return grid_games.sokoban(seed, size, levels_dir=downloads / "boxoban")
    if hasattr(grid_games, game):
        return getattr(grid_games, game)(seed, size)
    return getattr(causal_games, game)(seed, size)
