from __future__ import annotations

from typing import TypeAlias

Literal: TypeAlias = tuple[str, ...]


PREDICATE_ALIASES = {
    "in_hand": "holding",
    "dispenses": "in",
    "has_water": "contains_water",
    "hand_empty": "hand_free",
    "free_hand": "hand_free",
    "inverted": "upside_down",
    "chopped": "cut",
    "scrunch": "scrunched",
    "crumple": "scrunched",
    "crumpled": "scrunched",
    "sidled": "slid",
    "dial_turned": "configured",
    "cycle_selected": "configured",
    "pressed": "activated",
    "pushed_state": "activated",
    "started": "power_on",
    "is_on": "power_on",
    "is_off": "power_off",
    "blocking": "blocks",
    "warmed": "heated",
    "warm_mixture": "heated",
    "hot": "heated",
    "wiped": "clean",
    "cleaned": "clean",
    "rinsed": "washed",
    "is_rinsed": "washed",
    "rinsed_thoroughly": "washed",
    "head_clean": "clean",
    "head_rinsed": "clean",
    "damp": "wet",
    "water_poured_out": "empty",
    "filled_with_hot_water": "contains_hot_water",
    "hot_water_added": "contains_hot_water",
    "purified_water_added": "contains_purified_water",
}

PREDICATE_ARGUMENT_ORDERS = {
    "in_hand": (1, 0),
    "dispenses": (1, 0),
}


def canonical_literal(literal: Literal) -> Literal:
    if not literal:
        return literal
    name = literal[0]
    arguments = literal[1:]
    order = PREDICATE_ARGUMENT_ORDERS.get(name)
    if order is not None and len(arguments) == len(order):
        arguments = tuple(arguments[index] for index in order)
    return (PREDICATE_ALIASES.get(name, name), *arguments)
