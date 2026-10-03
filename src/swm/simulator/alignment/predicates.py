from __future__ import annotations

import re
from dataclasses import dataclass

from swm.pddl.strips import DomainSchemas, ProblemModel

from swm.simulator.predicate_aliases import (
    PREDICATE_ALIASES,
    PREDICATE_ARGUMENT_ORDERS,
    canonical_literal,
)

Literal = tuple[str, ...]


TOKEN_ALIASES = {
    "arm": "hand",
    "bin": "can",
    "countertop": "counter",
    "bookstand": "rack",
    "socket": "outlet",
    "dettol": "disinfectant",
    "washer": "washing_machine",
    "dish": "plate",
    "glass": "cup",
    "mug": "cup",
    "clothes": "cloth",
    "clothing": "cloth",
    "dissinfectant": "disinfectant",
}

STATE_PREDICATES = {
    "hand_free",
    "holding",
    "on",
    "in",
    "inserted",
    "open",
    "closed",
    "power_on",
    "power_off",
    "locked",
    "unlocked",
    "upright",
    "upside_down",
    "flat",
    "vertical",
    "clear",
    "blocks",
    "blocks_opening",
    "blocks_closing",
    "under",
    "against",
    "away_from",
    "in_front_of",
    "clean",
    "washed",
    "wet",
    "dry",
    "slid",
    "pushed",
    "activated",
    "configured",
    "enabled",
    "disabled",
    "unblocked",
    "dirty",
    "cut",
    "uncut",
    "stirred",
    "ready",
    "folded",
    "unfolded",
    "scrunched",
    "empty",
    "contains_water",
    "contains_hot_water",
    "contains_purified_water",
    "heated",
}


def normalized_tokens(value: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z]+|\d+", value.lower())
    return tuple(TOKEN_ALIASES.get(token, token) for token in tokens)


def normalized_descriptor(value: str) -> str:
    return "_".join(normalized_tokens(value))


@dataclass(frozen=True)
class PredicateDescription:
    raw_name: str
    canonical_name: str
    arity: int
    static: bool
    argument_order: tuple[int, ...]


@dataclass(frozen=True)
class PredicateInterface:
    descriptions: tuple[PredicateDescription, ...]

    @classmethod
    def from_schemas(cls, schemas: DomainSchemas) -> "PredicateInterface":
        changed = {
            literal[0]
            for schema in schemas.values()
            for literal in schema.add_eff | schema.del_eff
        }
        descriptions = []
        for raw_name, arity in sorted(schemas.predicate_arities.items()):
            canonical = PREDICATE_ALIASES.get(raw_name, raw_name)
            order = PREDICATE_ARGUMENT_ORDERS.get(
                raw_name, tuple(range(arity))
            )
            # ``dispenses`` describes a source capability, not current
            # containment.  Keeping it static prevents a candidate-only
            # source relation from becoming a GT runtime precondition.
            if raw_name == "dispenses" and arity == 2:
                canonical = "dispenses"
                order = tuple(range(arity))
            elif raw_name == "has_water" and arity == 1:
                canonical = "contains_water"
            descriptions.append(
                PredicateDescription(
                    raw_name,
                    canonical,
                    arity,
                    raw_name not in changed and canonical not in STATE_PREDICATES,
                    order,
                )
            )
        return cls(tuple(descriptions))

    def description(self, raw_name: str) -> PredicateDescription | None:
        return next((item for item in self.descriptions if item.raw_name == raw_name), None)

    def canonicalize(self, literal: Literal) -> Literal:
        description = self.description(literal[0])
        if description is None:
            return canonical_literal(literal)
        arguments = literal[1:]
        ordered = tuple(arguments[index] for index in description.argument_order)
        if description.static and len(ordered) == 1:
            return (f"kind:{normalized_descriptor(description.raw_name)}", ordered[0])
        if description.static:
            return (f"static:{normalized_descriptor(description.raw_name)}", *ordered)
        return (description.canonical_name, *ordered)

    def is_static(self, raw_name: str) -> bool:
        description = self.description(raw_name)
        return bool(description and description.static)


@dataclass(frozen=True)
class CanonicalObjectDescription:
    name: str
    kinds: tuple[str, ...]
    name_tokens: tuple[str, ...]
    state_tags: tuple[str, ...]
    relation_profile: tuple[tuple[str, int], ...]
    identity_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalWorld:
    interface: PredicateInterface
    objects: tuple[CanonicalObjectDescription, ...]
    facts: frozenset[Literal]
    negative_facts: frozenset[Literal]
    goal_positive: frozenset[Literal]
    goal_negative: frozenset[Literal]
    schemas: DomainSchemas | None = None
    identity_facts: frozenset[Literal] = frozenset()

    def object(self, name: str) -> CanonicalObjectDescription | None:
        return next((item for item in self.objects if item.name == name), None)


def _describe_objects(
    schemas: DomainSchemas,
    problem: ProblemModel,
    facts: frozenset[Literal],
) -> tuple[CanonicalObjectDescription, ...]:
    result = []
    static_unary = {
        item.raw_name for item in PredicateInterface.from_schemas(schemas).descriptions
        if item.static and item.arity == 1
    }
    for name in sorted(problem.objects):
        kinds = {
            fact[0].removeprefix("kind:")
            for fact in facts
            if fact[0].startswith("kind:") and len(fact) == 2 and fact[1] == name
        }

        identity_kinds = {
            fact[0]
            for fact in problem.init_state
            if len(fact) == 2 and fact[0] in static_unary and fact[1] == name
        }

        state_tags = sorted(
            fact[0]
            for fact in facts
            if len(fact) == 2
            and fact[1] == name
            and not fact[0].startswith(("kind:", "static:"))
        )
        profile: dict[str, int] = {}
        for fact in facts:
            if len(fact) < 3:
                continue
            if fact[1] == name:
                key = f"{fact[0]}:out"
                profile[key] = profile.get(key, 0) + 1
            if fact[2] == name:
                key = f"{fact[0]}:in"
                profile[key] = profile.get(key, 0) + 1
        result.append(
            CanonicalObjectDescription(
                name,
                tuple(sorted(kinds)),
                normalized_tokens(name),
                tuple(state_tags),
                tuple(sorted(profile.items())),
                tuple(sorted(identity_kinds)),
            )
        )
    return tuple(result)


def compile_world(schemas: DomainSchemas, problem: ProblemModel) -> CanonicalWorld:
    interface = PredicateInterface.from_schemas(schemas)
    facts = frozenset(interface.canonicalize(item) for item in problem.init_state)
    negative = frozenset(interface.canonicalize(item) for item in problem.init_negative)
    goal_positive = frozenset(interface.canonicalize(item) for item in problem.goal_positive)
    goal_negative = frozenset(interface.canonicalize(item) for item in problem.goal_negative)
    return CanonicalWorld(
        interface,
        _describe_objects(schemas, problem, facts),
        facts,
        negative,
        goal_positive,
        goal_negative,
        schemas,
        frozenset(problem.init_state),
    )
