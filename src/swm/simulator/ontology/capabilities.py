from __future__ import annotations

from dataclasses import dataclass

from swm.pddl.strips import DomainSchemas
from swm.simulator.alignment.predicates import PredicateInterface

from swm.simulator.ir.model import Capability, EvidenceValue, Provenance


# Reviewed generic affordance profiles. These are unary category predicates, never task or
# object identities. They cover geometry that STRIPS predicates cannot distinguish.
REVIEWED_CATEGORY_CAPABILITIES = {
    "laptop": {Capability.OPENING_SURFACE},
}


@dataclass(frozen=True)
class CapabilityRegistry:
    values: tuple[tuple[tuple[str, ...], Capability, EvidenceValue], ...]
    provenance: tuple[tuple[tuple[str, ...], Capability, tuple[Provenance, ...]], ...]

    def get(self, categories: tuple[str, ...], capability: Capability) -> EvidenceValue:
        return next((value for required, cap, value in self.values
                     if cap is capability and set(required) <= set(categories)), EvidenceValue.UNKNOWN)


def infer_capabilities(schemas: DomainSchemas) -> CapabilityRegistry:
    evidence: dict[tuple[tuple[str, ...], Capability], list[Provenance]] = {}

    def add(categories: tuple[str, ...], capability: Capability, action_name: str, detail: str) -> None:
        if not categories:
            return
        evidence.setdefault((categories, capability), []).append(
            Provenance("pddl_schema", action_name, detail)
        )

    category_predicates = {
        item.raw_name for item in PredicateInterface.from_schemas(schemas).descriptions
        if item.static and item.arity == 1
    }
    for schema in schemas.values():
        parameter_categories = {
            parameter: tuple(sorted(literal[0] for literal in schema.pre_pos
                                    if len(literal) == 2 and literal[1] == parameter
                                    and literal[0] in category_predicates))
            for parameter in schema.params
        }
        literals = schema.pre_pos | schema.add_eff | schema.del_eff
        for literal in literals:
            if literal[0] == "holding" and len(literal) == 3:
                add(parameter_categories[literal[2]], Capability.PICKABLE, schema.name, "holding role")
            elif literal[0] == "on" and len(literal) == 3:
                add(parameter_categories[literal[2]], Capability.SUPPORT, schema.name, "on target")
            elif literal[0] == "in" and len(literal) == 3:
                add(parameter_categories[literal[2]], Capability.CONTAINER, schema.name, "in target")
            elif literal[0] in {"open", "closed"} and len(literal) == 2:
                add(parameter_categories[literal[1]], Capability.OPENABLE, schema.name, literal[0])
            elif literal[0] in {"locked", "unlocked"} and len(literal) == 2:
                add(parameter_categories[literal[1]], Capability.LOCKABLE, schema.name, literal[0])
            elif literal[0] in {"is_on", "is_off"} and len(literal) == 2:
                add(parameter_categories[literal[1]], Capability.DEVICE, schema.name, literal[0])
            elif literal[0] == "inserted" and len(literal) == 3:
                add(parameter_categories[literal[2]], Capability.SLOT_HOLDER, schema.name, "inserted target")
    for category in category_predicates:
        for capability in REVIEWED_CATEGORY_CAPABILITIES.get(category, set()):
            add((category,), capability, "reviewed_category_profile", "generic affordance profile")
    values = tuple(
        (categories, capability, EvidenceValue.KNOWN)
        for categories, capability in sorted(evidence, key=lambda item: (item[0], item[1].value))
    )
    provenance = tuple(
        (categories, capability, tuple(items))
        for (categories, capability), items in sorted(
            evidence.items(), key=lambda item: (item[0][0], item[0][1].value)
        )
    )
    return CapabilityRegistry(values, provenance)
