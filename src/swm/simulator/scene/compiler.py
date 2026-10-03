from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from swm.pddl.strips import DomainSchemas, ProblemModel
from swm.simulator.alignment.predicates import PredicateInterface

from swm.simulator.ir.model import (
    Capability,
    Closure,
    EvidenceValue,
    Location,
    LocationKind,
    LockState,
    Pose,
    Power,
    Provenance,
)
from swm.simulator.ontology.capabilities import infer_capabilities

Literal = tuple[str, ...]


@dataclass(frozen=True)
class Diagnostic:
    category: str
    objects: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    categories: tuple[str, ...]
    capabilities: tuple[tuple[Capability, EvidenceValue], ...]
    provenance: tuple[Provenance, ...]

    def capability(self, capability: Capability) -> EvidenceValue:
        return next((value for current, value in self.capabilities if current is capability), EvidenceValue.UNKNOWN)


@dataclass(frozen=True)
class SceneState:
    objects: tuple[ObjectSpec, ...]
    facts: frozenset[Literal]
    negative_facts: frozenset[Literal]
    locations: tuple[tuple[str, Location], ...]
    poses: tuple[tuple[str, Pose], ...]
    closures: tuple[tuple[str, Closure], ...]
    powers: tuple[tuple[str, Power], ...]
    locks: tuple[tuple[str, LockState], ...]
    diagnostics: tuple[Diagnostic, ...]

    def location(self, name: str) -> Location:
        return next((value for current, value in self.locations if current == name), Location(LocationKind.NONE))

    def pose(self, name: str) -> Pose:
        return next((value for current, value in self.poses if current == name), Pose.UNKNOWN)

    def closure(self, name: str) -> Closure:
        return next((value for current, value in self.closures if current == name), Closure.NOT_APPLICABLE)

    def power(self, name: str) -> Power:
        return next((value for current, value in self.powers if current == name), Power.NOT_APPLICABLE)

    def lock(self, name: str) -> LockState:
        return next((value for current, value in self.locks if current == name), LockState.NOT_APPLICABLE)

    def object(self, name: str) -> ObjectSpec | None:
        return next((item for item in self.objects if item.name == name), None)

    def relation(self, predicate: str, *arguments: str) -> bool:
        return (predicate, *arguments) in self.facts

    def children_on(self, target: str) -> tuple[str, ...]:
        return tuple(sorted(item[1] for item in self.facts if item[0] == "on" and item[2] == target))

    def children_in(self, target: str) -> tuple[str, ...]:
        return tuple(sorted(item[1] for item in self.facts if item[0] == "in" and item[2] == target))

    def is_clear(self, target: str) -> bool:
        return not self.children_on(target)

    def ancestors(self, name: str) -> tuple[str, ...]:
        locations = dict(self.locations)
        result: list[str] = []
        seen = {name}
        current = name
        while True:
            location = locations.get(current, Location(LocationKind.NONE))
            if location.kind not in {LocationKind.ON, LocationKind.IN} or location.parent is None:
                break
            if location.parent in seen:
                break
            seen.add(location.parent)
            result.append(location.parent)
            current = location.parent
        return tuple(result)

    def accessible(self, name: str) -> bool:
        locations = dict(self.locations)
        seen = {name}
        current = name
        while True:
            location = locations.get(current, Location(LocationKind.NONE))
            if location.kind not in {LocationKind.ON, LocationKind.IN} or location.parent is None:
                break
            parent = location.parent
            if parent in seen:
                break
            if location.kind is LocationKind.IN:
                if self.closure(parent) is Closure.CLOSED:
                    return False
                if self.lock(parent) is LockState.LOCKED:
                    return False
            seen.add(parent)
            current = parent
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "objects": [
                {
                    "name": item.name,
                    "categories": list(item.categories),
                    "capabilities": {cap.value: value.value for cap, value in item.capabilities},
                }
                for item in self.objects
            ],
            "facts": [list(item) for item in sorted(self.facts)],
            "negative_facts": [list(item) for item in sorted(self.negative_facts)],
            "locations": {
                name: {"kind": value.kind.value, "parent": value.parent}
                for name, value in self.locations
            },
            "poses": {name: value.value for name, value in self.poses},
            "closures": {name: value.value for name, value in self.closures},
            "powers": {name: value.value for name, value in self.powers},
            "locks": {name: value.value for name, value in self.locks},
            "diagnostics": [
                {"category": item.category, "objects": list(item.objects), "detail": item.detail}
                for item in self.diagnostics
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _exclusive_state(
    name: str,
    facts: frozenset[Literal],
    pairs: tuple[tuple[str, object], ...],
    default: object,
    diagnostics: list[Diagnostic],
) -> object:
    active = [(predicate, value) for predicate, value in pairs if (predicate, name) in facts]
    if len(active) > 1:
        diagnostics.append(
            Diagnostic(
                "contradictory_state",
                (name,),
                f"mutually exclusive predicates present: {[item[0] for item in active]}",
            )
        )
        return default
    return active[0][1] if active else default


def _locations(
    object_names: Iterable[str], facts: frozenset[Literal], diagnostics: list[Diagnostic]
) -> tuple[tuple[str, Location], ...]:
    result = []
    for name in sorted(object_names):
        held = sorted((item[1] for item in facts if item[0] == "holding" and item[2] == name))
        on = sorted((item[2] for item in facts if item[0] == "on" and item[1] == name))
        inside = sorted((item[2] for item in facts if item[0] == "in" and item[1] == name))
        if len(held) > 1 or len(on) > 1 or len(inside) > 1:
            diagnostics.append(Diagnostic("multiple_location_parent", (name,), "multiple parents in one relation"))
        if held and (on or inside):
            diagnostics.append(Diagnostic("multiple_location_kind", (name,), "held object also has a physical parent"))
        if held:
            location = Location(LocationKind.HELD, held[0])
        elif on:
            # An accompanying `in` fact is containment membership for a stack whose
            # direct support parent is represented by `on`.
            location = Location(LocationKind.ON, on[0])
        elif inside:
            location = Location(LocationKind.IN, inside[0])
        else:
            location = Location(LocationKind.NONE)
        result.append((name, location))
    return tuple(result)


def _location_cycles(locations: tuple[tuple[str, Location], ...]) -> list[Diagnostic]:
    mapping = dict(locations)
    diagnostics = []
    for start in sorted(mapping):
        seen = {start}
        current = start
        while True:
            location = mapping[current]
            parent = location.parent
            if location.kind not in {LocationKind.ON, LocationKind.IN} or parent not in mapping:
                break
            if parent in seen:
                diagnostics.append(Diagnostic("relation_cycle", tuple(sorted(seen | {parent})), f"cycle from {start}"))
                break
            seen.add(parent)
            current = parent
    unique = {(item.category, item.objects, item.detail): item for item in diagnostics}
    return [unique[key] for key in sorted(unique)]


def compile_scene(
    schemas: DomainSchemas,
    problem: ProblemModel,
    *,
    facts: Iterable[Literal] | None = None,
) -> SceneState:
    raw_facts = frozenset(problem.init_state if facts is None else facts)
    diagnostics: list[Diagnostic] = []
    registry = infer_capabilities(schemas)
    category_predicates = {
        item.raw_name for item in PredicateInterface.from_schemas(schemas).descriptions
        if item.static and item.arity == 1
    }
    objects = tuple(
        ObjectSpec(
            name,
            categories,
            tuple(
                (capability, registry.get(categories, capability))
                for capability in Capability
            ),
            tuple(Provenance("initial_category", name, category) for category in categories),
        )
        for name in sorted(problem.objects)
        for categories in [tuple(sorted(
            fact[0] for fact in problem.init_state
            if len(fact) == 2 and fact[1] == name and fact[0] in category_predicates
        ))]
    )
    locations = _locations(problem.objects, raw_facts, diagnostics)
    diagnostics.extend(_location_cycles(locations))
    poses = tuple(
        (
            name,
            _exclusive_state(
                name,
                raw_facts,
                (("upright", Pose.UPRIGHT), ("upside_down", Pose.UPSIDE_DOWN), ("flat", Pose.FLAT), ("vertical", Pose.VERTICAL)),
                Pose.UNKNOWN,
                diagnostics,
            ),
        )
        for name in sorted(problem.objects)
    )
    closures = []
    powers = []
    locks = []
    for item in objects:
        closure_default = Closure.UNKNOWN if item.capability(Capability.OPENABLE) is EvidenceValue.KNOWN else Closure.NOT_APPLICABLE
        power_default = Power.UNKNOWN if item.capability(Capability.DEVICE) is EvidenceValue.KNOWN else Power.NOT_APPLICABLE
        lock_default = LockState.UNKNOWN if item.capability(Capability.LOCKABLE) is EvidenceValue.KNOWN else LockState.NOT_APPLICABLE
        closures.append((item.name, _exclusive_state(
            item.name, raw_facts, (("open", Closure.OPEN), ("closed", Closure.CLOSED)),
            closure_default, diagnostics,
        )))
        powers.append((item.name, _exclusive_state(
            item.name, raw_facts, (("is_on", Power.ON), ("is_off", Power.OFF)),
            power_default, diagnostics,
        )))
        locks.append((item.name, _exclusive_state(
            item.name, raw_facts, (("locked", LockState.LOCKED), ("unlocked", LockState.UNLOCKED)),
            lock_default, diagnostics,
        )))
    return SceneState(
        objects,
        raw_facts,
        frozenset(problem.init_negative),
        locations,
        poses,
        tuple(closures),
        tuple(powers),
        tuple(locks),
        tuple(sorted(diagnostics, key=lambda item: (item.category, item.objects, item.detail))),
    )
