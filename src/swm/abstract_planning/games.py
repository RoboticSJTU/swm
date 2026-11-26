"""Generate the classic planning game families and check their rules."""
from pathlib import Path

from ._v7.games import GAMES, MECHANISMS, generate as generate_scene
from ._v7.rule_audit import audit_description, successors


DOWNLOADS = Path(__file__).resolve().parents[3] / "downloads"

# These options belonged to the benchmark scene builder. Classic training
# generation ignores them, as it did before sharing the generators.
BENCHMARK_OPTIONS = {
    "frozenlake": {"checkpoints"},
    "maze": {"doors", "key_placement"},
    "package": {"packages"},
    "printer": {"printers", "documents_per_printer"},
    "kitchen": {"dishes"},
    "key_delivery": {"relics"},
    "hanoi": {"goal_selection", "goal_minimum_distance"},
    "floortile": {"robots"},
    "termes": {"targets", "prebuilt"},
    "freecell": {"columns", "freecells"},
}


def generate(game, seed, size=None):
    if size and game in BENCHMARK_OPTIONS:
        size = {key: value for key, value in size.items() if key not in BENCHMARK_OPTIONS[game]}
    return generate_scene(game, seed, size, downloads=DOWNLOADS)
