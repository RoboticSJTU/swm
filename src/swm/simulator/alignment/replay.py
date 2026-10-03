from __future__ import annotations

import re
from dataclasses import dataclass

from swm.simulator.ir.model import ActionFamily, VerificationStatus

from .actions import CanonicalStep, LOCATION_PREDICATES, POSE_PREDICATES, STRUCTURAL_PREDICATES
from .goals import instruction_action_issue, instruction_order_issue
from .predicates import CanonicalWorld, Literal, normalized_tokens

@dataclass(frozen=True)
class CrossDomainIssue:
    step: int
    category: str
    detail: str
    action: str


@dataclass(frozen=True)
class CrossDomainTrace:
    step: int
    action: str
    family: str
    added: tuple[Literal, ...]
    deleted: tuple[Literal, ...]


@dataclass(frozen=True)
class CrossDomainResult:
    status: VerificationStatus
    first_issue: CrossDomainIssue | None
    trace: tuple[CrossDomainTrace, ...]
    final_facts: frozenset[Literal]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "first_issue": None if self.first_issue is None else self.first_issue.__dict__,
            "trace": [item.__dict__ for item in self.trace],
            "final_facts": [list(item) for item in sorted(self.final_facts)],
        }


def _location(facts: set[Literal], obj: str) -> tuple[str, str] | None:
    holding = next((item for item in facts if item[0] == "holding" and item[2] == obj), None)
    if holding:
        return "holding", holding[1]
    relation = next(
        (
            item
            for item in facts
            if item[0] in {"on", "in", "inserted"}
            and item[1] == obj
        ),
        None,
    )
    if relation is None:
        relation = next(
            (
                item
                for item in facts
                if item[0]
                in {
                    "under", "against", "away_from", "in_front_of",
                    "blocks", "blocks_opening", "blocks_closing",
                }
                and len(item) == 3
                and item[1] == obj
            ),
            None,
        )
    return None if relation is None else (relation[0], relation[2])


def detach_location(facts: set[Literal], obj: str) -> None:
    facts.difference_update({
        item
        for item in facts
        if (item[0] == "holding" and len(item) == 3 and item[2] == obj)
        or (
            item[0] in LOCATION_PREDICATES
            and len(item) == 3
            and item[1] == obj
        )
    })


def attach_location(facts: set[Literal], relation: str, obj: str, target: str) -> None:
    """Move an object to one exclusive logical location."""
    detach_location(facts, obj)
    facts.add((relation, obj, target))


def update_hand_state(
    facts: set[Literal],
    hand: str,
    *,
    held_object: str | None = None,
    release_object: str | None = None,
) -> None:
    """Keep hand_free and holding mutually consistent."""
    if release_object is not None:
        facts.discard(("holding", hand, release_object))
    if held_object is not None:
        facts.discard(("hand_free", hand))
        facts.add(("holding", hand, held_object))
    elif release_object is not None:
        facts.add(("hand_free", hand))


def _set_pair(facts: set[Literal], positive: str, negative: str, target: str) -> None:
    facts.discard((negative, target))
    facts.add((positive, target))


def _set_closure_pair(
    world: CanonicalWorld,
    facts: set[Literal],
    positive: str,
    negative: str,
    target: str,
) -> None:
    """Project vessel-level open/closed actions onto modeled closure parts."""
    targets = (target, *_matching_components(world, target, {"cap", "door", "flap", "lid"}))
    for item in targets:
        _set_pair(facts, positive, negative, item)


def _accessible(facts: set[Literal], obj: str) -> bool:
    seen = {obj}
    current = obj
    while True:
        location = _location(facts, current)
        if location is None or location[0] not in {"on", "in"}:
            return True
        relation, parent = location
        if parent in seen:
            return False
        if relation == "in" and (("closed", parent) in facts or ("locked", parent) in facts):
            return False
        seen.add(parent)
        current = parent


def _children_on(facts: set[Literal], target: str) -> tuple[str, ...]:
    return tuple(sorted(item[1] for item in facts if item[0] == "on" and item[2] == target))


def _children_in(facts: set[Literal], target: str) -> tuple[str, ...]:
    return tuple(sorted(item[1] for item in facts if item[0] == "in" and item[2] == target))


def _has_location(facts: set[Literal], obj: str, parent: str) -> bool:
    return any(
        item[1] == obj and item[2] == parent
        for item in facts
        if item[0] in LOCATION_PREDICATES
        and len(item) == 3
    )


def _declared_kind_tokens(world: CanonicalWorld, name: str) -> set[str]:
    obj = world.object(name)
    if obj is None:
        return set()
    return {
        token
        for kind in obj.kinds
        for token in kind.split("_")
    }


def _has_closure_capability(world: CanonicalWorld, target: str) -> bool:
    if any((state, target) in world.facts for state in ("open", "closed")):
        return True
    target_kinds = _declared_kind_tokens(world, target)
    closure_kinds = {"cap", "door", "flap", "lid"}
    return any(
        target_kinds
        and target_kinds < _declared_kind_tokens(world, item.name)
        and _declared_kind_tokens(world, item.name) & closure_kinds
        and any((state, item.name) in world.facts for state in ("open", "closed"))
        for item in world.objects
    )


def _matching_components(
    world: CanonicalWorld,
    target: str,
    component_kinds: set[str],
) -> tuple[str, ...]:
    target_kinds = _declared_kind_tokens(world, target)
    return tuple(
        item.name
        for item in world.objects
        if target_kinds
        and target_kinds < _declared_kind_tokens(world, item.name)
        and _declared_kind_tokens(world, item.name) & component_kinds
    )


def _is_closed(world: CanonicalWorld, facts: set[Literal], target: str) -> bool:
    if ("closed", target) in facts:
        return True
    return any(
        ("closed", component) in facts
        for component in _matching_components(world, target, {"cap", "door", "flap", "lid"})
    )


def _is_open(world: CanonicalWorld, facts: set[Literal], target: str) -> bool:
    if ("open", target) in facts:
        return True
    return any(
        ("open", component) in facts
        for component in _matching_components(world, target, {"cap", "door", "flap", "lid"})
    )


def _transfer_requires_open_source(
    world: CanonicalWorld,
    action: CanonicalStep,
    source: str,
) -> bool:
    """Use GT transfer schemas to distinguish sealed vessels from pourable spouts."""
    if ("open", source) in action.positive_preconditions:
        return True
    if world.schemas is None:
        return False
    source_kinds = _declared_kind_tokens(world, source)
    for schema in world.schemas.values():
        preconditions = [world.interface.canonicalize(item) for item in schema.pre_pos]
        effects = [world.interface.canonicalize(item) for item in schema.add_eff]
        for effect in effects:
            if len(effect) != 3 or effect[0] != "in":
                continue
            content_variable, _receiver_variable = effect[1:]
            for precondition in preconditions:
                if (
                    len(precondition) != 3
                    or precondition[0] != "in"
                    or precondition[1] != content_variable
                ):
                    continue
                source_variable = precondition[2]
                open_variables = {
                    item[1]
                    for item in preconditions
                    if len(item) == 2 and item[0] == "open"
                }
                source_requires_open = source_variable in open_variables or any(
                    len(item) == 3
                    and item[0] in {"on", "in", "inserted"}
                    and item[1] in open_variables
                    and item[2] == source_variable
                    for item in preconditions
                )
                if not source_requires_open:
                    continue
                expected_kinds = _parameter_kind_tokens(world, schema, source_variable)
                expected_tokens = {
                    token
                    for kind in expected_kinds
                    for token in kind.split("_")
                }
                if not expected_tokens or expected_tokens & source_kinds:
                    return True
    return False


def _pose_precondition_satisfied(facts: set[Literal], item: Literal) -> bool:
    if item in facts:
        return True
    if len(item) != 2 or item[0] not in {"upright", "vertical"}:
        return False
    equivalent = "vertical" if item[0] == "upright" else "upright"
    return (equivalent, item[1]) in facts


def _state_preconditions_missing(
    world: CanonicalWorld,
    facts: set[Literal],
    action: CanonicalStep,
    predicates: set[str],
) -> tuple[Literal, ...]:
    missing = []
    for item in action.positive_preconditions:
        if item[0] not in predicates or _pose_precondition_satisfied(facts, item):
            continue
        if len(item) == 2 and item[0] == "closed" and _is_closed(world, facts, item[1]):
            continue
        if len(item) == 2 and item[0] == "open" and _is_open(world, facts, item[1]):
            continue
        if len(item) == 2 and item[0] == "clear" and not any(
            len(fact) == 3
            and fact[0] in {"on", "blocks", "blocks_opening", "blocks_closing"}
            and fact[2] == item[1]
            for fact in facts
        ):
            continue
        missing.append(item)
    return tuple(sorted(missing))


def _source_matches(
    world: CanonicalWorld,
    facts: set[Literal],
    obj: str,
    requested: str,
    action_name: str,
) -> bool:
    """Require the action's stated immediate source to match live state.

    A shared outer support does not prove a direct relation. For example, a
    kettle and its base may both be on a counter while the kettle is either on
    the base or beside it. Initial-scene visual evidence may resolve that
    ambiguity outside replay; the symbolic kernel itself stays strict.
    """
    if _has_location(facts, obj, requested):
        return True
    location = _location(facts, obj)
    if location is None or location[0] != "on":
        return False
    component = location[1]
    component_location = _location(facts, component)
    action_tokens = set(re.findall(r"[a-z0-9]+", action_name.lower()))
    return bool(
        "top" in action_tokens
        and component_location == ("on", requested)
        and _declared_kind_tokens(world, component) & {"cap", "door", "flap", "lid"}
    )


def _explicit_blockers(facts: set[Literal], target: str) -> tuple[str, ...]:
    """Return only blockers explicitly represented by the scene relation graph."""
    return tuple(sorted({
        item[1]
        for item in facts
        if len(item) == 3
        and item[0] in {"blocks", "blocks_opening", "blocks_closing"}
        and item[2] == target
    }))


def _receiver_blockers(
    world: CanonicalWorld,
    facts: set[Literal],
    target: str,
) -> tuple[str, ...]:
    """Return explicit blockers plus nested physical containers.

    Liquids and ordinary contents may coexist in a receiver. A second vessel
    occupying that receiver instead blocks access to its interior even when a
    generated domain incorrectly asserts ``clear``.
    """
    blockers = set(_explicit_blockers(facts, target))
    blockers.update(
        child
        for child in _children_in(facts, target)
        if _is_container_kind(world, child)
    )
    return tuple(sorted(blockers))


def _opening_blockers(facts: set[Literal], target: str) -> tuple[str, ...]:
    return tuple(sorted({
        item[1]
        for item in facts
        if item[0] in {"blocks", "blocks_opening"} and len(item) == 3 and item[2] == target
    }))


def _empty_subject(action: CanonicalStep) -> str | None:
    action_name = action.raw_action.lstrip("(").split(maxsplit=1)[0].lower()
    if not action_name.startswith("empty_"):
        return None
    transferred = next(
        (
            item[1]
            for item in action.add_effects
            if len(item) == 3 and item[0] == "in"
        ),
        None,
    )
    if transferred is not None:
        return transferred
    return action.role("object")


def _can_be_emptied(
    world: CanonicalWorld,
    target: str,
    action: CanonicalStep,
) -> bool:
    # Emptying may mean transferring a content object out of a source vessel;
    # the emptied object itself need not be a vessel.  The paired containment
    # transition is the structural evidence for that interpretation.
    if any(
        len(item) == 3 and item[0] == "in" and item[1] == target
        for item in action.add_effects
    ):
        return True
    # A declared kind is structural evidence; object-name tokens are not.
    if _is_container_kind(world, target):
        return True
    evidence = set(world.facts | world.goal_positive | world.goal_negative)
    if _children_in(evidence, target):
        return True
    return any(
        len(fact) == 2
        and fact[1] == target
        and fact[0].startswith("contains_")
        for fact in evidence
    )


def _is_container_kind(world: CanonicalWorld, target: str) -> bool:
    return bool(_declared_kind_tokens(world, target) & {
        "container", "basin", "bin", "bottle", "bowl", "box", "bucket",
        "cabinet", "can", "carton", "cup", "drawer", "glass", "kettle",
        "mug", "pan", "pot", "sink", "thermos", "tray",
    })


PHYSICAL_PRECONDITION_PREDICATES = {
    "hand_free", "holding", *LOCATION_PREDICATES, *POSE_PREDICATES,
    "open", "closed", "power_on", "power_off", "locked", "unlocked", "clear",
    "wet", "dry",
}

FAMILY_EFFECT_PREDICATES = {
    ActionFamily.PICK: {"holding", "hand_free", *LOCATION_PREDICATES},
    ActionFamily.PLACE_ON: {"holding", "hand_free", *LOCATION_PREDICATES, *POSE_PREDICATES},
    ActionFamily.PLACE_IN: {"holding", "hand_free", *LOCATION_PREDICATES, *POSE_PREDICATES},
    ActionFamily.INSERT: {"holding", "hand_free", *LOCATION_PREDICATES, *POSE_PREDICATES},
    ActionFamily.OPEN: {"open", "closed", "holding", "hand_free"},
    ActionFamily.CLOSE: {"open", "closed", "holding", "hand_free"},
    ActionFamily.TURN_ON: {
        "power_on", "power_off", "activated", "configured", "in", "under",
        "contains_water", "contains_hot_water", "contains_purified_water", "heated",
    },
    ActionFamily.TURN_OFF: {
        "power_on", "power_off", "activated", "configured", "in", "under",
        "contains_water", "contains_hot_water", "contains_purified_water", "heated",
    },
    ActionFamily.LOCK: {"locked", "unlocked"},
    ActionFamily.UNLOCK: {"locked", "unlocked"},
    ActionFamily.REMOVE_CLOSURE: {"holding", "hand_free", "on", "open", "closed"},
    ActionFamily.REPLACE_CLOSURE: {"holding", "hand_free", "on", "open", "closed"},
    ActionFamily.POUR: {"in", "holding", "hand_free", *POSE_PREDICATES},
    ActionFamily.SCOOP: {"in", "holding", "hand_free", *POSE_PREDICATES},
    ActionFamily.WIPE: {"clean", "washed", "dirty", "holding", "hand_free"},
    ActionFamily.STIR: {"stirred", "ready"},
    ActionFamily.CUT: {"cut", "uncut"},
    ActionFamily.WASH: {"washed", "clean", "dirty", "wet", "contains_water", "in"},
    ActionFamily.FOLD: {"folded", "unfolded"},
    ActionFamily.SCRUNCH: {"scrunched"},
    ActionFamily.FILL: {
        "in", "contains_water", "contains_hot_water", "contains_purified_water",
        "empty", "holding", "hand_free", *LOCATION_PREDICATES,
    },
    ActionFamily.WET: {"wet", "dry"},
    ActionFamily.EMPTY: {"empty", "in", "contains_water", *LOCATION_PREDICATES},
    ActionFamily.SWEEP: {"on", "in"},
    ActionFamily.ORIENT: POSE_PREDICATES,
    ActionFamily.CONFIGURE: {"configured"},
    ActionFamily.ACTIVATE: {"activated", "enabled", "disabled"},
    ActionFamily.SLIDE: {"slid", "on"},
    ActionFamily.PUSH: {"pushed", "on"},
    ActionFamily.UNBLOCK: {"blocks", "unblocked"},
}


def _passive_semantic_subjects(action: CanonicalStep) -> set[str]:
    """Find processed objects that the action does not physically acquire.

    A semantic operation may contain a malformed relation as an operation
    marker, such as moving a cut item onto the knife. When the same action has
    a schema-backed semantic result but no holding transition for that subject,
    the relation is not evidence of physical motion.
    """
    processing_families = {
        ActionFamily.WIPE,
        ActionFamily.STIR,
        ActionFamily.CUT,
        ActionFamily.WASH,
        ActionFamily.FOLD,
        ActionFamily.SCRUNCH,
        ActionFamily.WET,
    }
    if action.family not in processing_families:
        return set()
    subjects = {
        item[1]
        for item in action.semantic_effects
        if len(item) == 2 and item[0] not in PHYSICAL_PRECONDITION_PREDICATES
    }
    physically_moved = {
        item[2]
        for item in action.add_effects + action.delete_effects
        if len(item) == 3 and item[0] == "holding"
    }
    return subjects - physically_moved


def _apply_declared_effects(
    facts: set[Literal],
    action: CanonicalStep,
    allowed_predicates: set[str] | None = None,
) -> None:
    if action.trusted_declared_semantics:
        allowed = {
            item[0] for item in action.add_effects + action.delete_effects
        }
    else:
        allowed = set(FAMILY_EFFECT_PREDICATES.get(action.family, set()))
        # A compound action may combine a state transition with a structural
        # transfer, such as stopping a dispenser while adding water to a cup.
        # Relation effects remain claims about the executed action and are
        # translated only when their objects were mapped.
        allowed.update(LOCATION_PREDICATES)
        # Semantic actions can also carry an explicit manipulation transition
        # (for example, scrunch-and-pick). Preserve only those core predicates
        # that the action actually declares; do not infer them from its name.
        allowed.update(
            item[0]
            for item in action.add_effects + action.delete_effects
            if item[0] in {"holding", "hand_free", *LOCATION_PREDICATES}
        )
        if allowed_predicates is not None:
            allowed &= allowed_predicates
    passive_subjects = _passive_semantic_subjects(action)

    def applicable(item: Literal) -> bool:
        return not (
            len(item) == 3
            and item[0] in LOCATION_PREDICATES
            and item[1] in passive_subjects
        )

    delete_effects = {
        item
        for item in action.delete_effects
        if item[0] in allowed and applicable(item)
    }
    add_effects = {
        item
        for item in action.add_effects
        if item[0] in allowed and applicable(item)
    }
    facts.difference_update(delete_effects)
    for effect in add_effects:
        if len(effect) == 3 and effect[0] in LOCATION_PREDICATES:
            detach_location(facts, effect[1])
    for effect in add_effects:
        if len(effect) == 2 and effect[0] in POSE_PREDICATES:
            facts.difference_update(
                (pose, effect[1]) for pose in POSE_PREDICATES if pose != effect[0]
            )
    facts.update(add_effects)


def _generic_motion_issue(action: CanonicalStep) -> str | None:
    """A push may rearrange an object on a support but cannot replace placement into a container."""
    if action.family is not ActionFamily.PUSH:
        return None
    if any(
        len(effect) == 3 and effect[0] in {"in", "inserted"}
        for effect in action.add_effects
    ):
        return "a push action cannot create containment; a separate placement is required"
    return None


def _check_compound_transfer_receiver(
    world: CanonicalWorld,
    facts: set[Literal],
    action: CanonicalStep,
    stop,
) -> CrossDomainResult | None:
    """Apply GT access rules to transfer effects embedded in state actions."""
    for effect in action.add_effects + action.semantic_effects:
        if len(effect) == 3 and effect[0] == "in":
            receiver = effect[2]
        elif len(effect) == 2 and effect[0] in {
            "contains_water", "contains_hot_water", "contains_purified_water"
        }:
            receiver = effect[1]
        else:
            continue
        if _is_closed(world, facts, receiver) or not _accessible(facts, receiver):
            return stop(
                VerificationStatus.FAIL,
                "closed_target",
                f"receiver {receiver} is inaccessible",
            )
        blockers = _explicit_blockers(facts, receiver)
        if blockers:
            return stop(
                VerificationStatus.FAIL,
                "blocked_not_clear",
                f"receiver {receiver} has objects on top: {list(blockers)}",
            )
    return None


def _apply_semantic_effects(
    world: CanonicalWorld,
    facts: set[Literal],
    action: CanonicalStep,
) -> None:
    supported = {
        description.canonical_name
        for description in world.interface.descriptions
    }
    for item in action.semantic_effects:
        if item[0] not in supported:
            continue
        if len(item) == 2 and item[0] in POSE_PREDICATES:
            facts.difference_update(
                (pose, item[1]) for pose in POSE_PREDICATES if pose != item[0]
            )
            facts.discard(("horizontal", item[1]))
        facts.add(item)


def _has_pose_capability(world: CanonicalWorld, target: str) -> bool:
    return any(
        len(item) == 2
        and item[0] in {"upright", "upside_down", "flat", "vertical", "horizontal"}
        and item[1] == target
        for item in world.facts | world.goal_positive | world.goal_negative
    )


def _has_clear_evidence(world: CanonicalWorld, target: str) -> bool:
    """Whether the GT domain models ``target`` as a clear-sensitive support."""
    evidence = world.facts | world.goal_positive | world.goal_negative
    if ("clear", target) in evidence:
        return True
    if world.schemas is None:
        return False
    target_kinds = _declared_kind_tokens(world, target)
    for schema in world.schemas.values():
        preconditions = {
            world.interface.canonicalize(item) for item in schema.pre_pos
        }
        for clear in preconditions:
            if len(clear) != 2 or clear[0] != "clear":
                continue
            clear_parameter = clear[1]
            expected_kinds = _parameter_kind_tokens(world, schema, clear_parameter)
            if expected_kinds and not expected_kinds & target_kinds:
                continue
            # A clear precondition is itself GT evidence that the parameter is
            # a clearance-sensitive support.  The action that consumes the
            # clearance need not also contain the placement effect that
            # creates it (opening a lid is the canonical example).
            if clear_parameter in schema.params or expected_kinds:
                return True
            if any(
                len(effect) == 3
                and effect[0] in {"on", "in", "inserted"}
                and effect[2] == clear_parameter
                for effect in (
                    world.interface.canonicalize(item)
                    for item in schema.add_eff
                )
            ):
                return True
    return False


def _parameter_kind_tokens(
    world: CanonicalWorld,
    schema: object,
    parameter: str,
) -> set[str]:
    """Recover a parameter's static PDDL kind, from unary category guards."""
    kinds: set[str] = set()
    preconditions = getattr(schema, "pre_pos", set())
    for literal in preconditions:
        canonical = world.interface.canonicalize(literal)
        if (
            len(canonical) == 2
            and canonical[0].startswith("kind:")
            and canonical[1] == parameter
        ):
            kinds.add(canonical[0].removeprefix("kind:"))
    return kinds


def _direct_pick_supported(world: CanonicalWorld, target: str) -> bool:
    """Reject fixed powered devices and payload-only kinds without pick evidence."""
    if world.schemas is None:
        return True
    target_kinds = _declared_kind_tokens(world, target)
    payload_only = False
    for schema in world.schemas.values():
        preconditions = {
            world.interface.canonicalize(item) for item in schema.pre_pos
        }
        additions = {
            world.interface.canonicalize(item) for item in schema.add_eff
        }
        for effect in additions:
            if len(effect) == 3 and effect[0] == "holding":
                expected = _parameter_kind_tokens(world, schema, effect[2])
                expected = {
                    token for kind in expected for token in kind.split("_")
                }
                if not expected or expected & target_kinds:
                    return True
        for relation in preconditions | additions:
            if len(relation) != 3 or relation[0] != "in":
                continue
            expected = _parameter_kind_tokens(world, schema, relation[1])
            expected = {
                token for kind in expected for token in kind.split("_")
            }
            if expected and expected & target_kinds:
                payload_only = True
    powered = any(
        (state, target) in world.facts | world.goal_positive | world.goal_negative
        for state in ("power_on", "power_off")
    )
    return not (payload_only or powered)


def _schema_parameter_accepts(
    world: CanonicalWorld,
    schema: object,
    parameter: str,
    object_name: str,
) -> bool:
    expected = {
        token
        for kind in _parameter_kind_tokens(world, schema, parameter)
        for token in kind.split("_")
    }
    actual = _declared_kind_tokens(world, object_name)
    return not expected or bool(expected & actual)


def _transfer_signature(action: CanonicalStep) -> tuple[str, str, str] | None:
    """Return payload, source, and receiver for one declared transfer."""
    additions = [
        item for item in action.add_effects if len(item) == 3 and item[0] == "in"
    ]
    for addition in additions:
        content, receiver = addition[1:]
        source_relation = next(
            (
                item
                for item in action.delete_effects
                if len(item) == 3
                and item[0] == "in"
                and item[1] == content
                and item[2] != receiver
            ),
            None,
        )
        if source_relation is not None:
            return content, source_relation[2], receiver
    if action.family not in {ActionFamily.POUR, ActionFamily.SCOOP}:
        return None
    content = action.role("content")
    source = action.role("source")
    receiver = action.role("receiver")
    if content and source and receiver:
        return content, source, receiver
    return None


def _transfer_tool_supported(
    world: CanonicalWorld,
    action: CanonicalStep,
    content: str,
    source: str,
    receiver: str,
) -> bool:
    """Check the GT carrier/tool contract for a payload transfer.

    The receiver itself is not the carrier.  Evidence comes from GT schemas
    that can place this payload kind into this receiver kind: their source and
    held-object parameters define whether the candidate is using a compatible
    bottle, ladle, spoon, or other transfer carrier.
    """
    if world.schemas is None:
        return True
    held_objects = {
        item[2]
        for item in action.positive_preconditions
        if len(item) == 3 and item[0] == "holding"
    }
    saw_destination_schema = False
    for schema in world.schemas.values():
        preconditions = {
            world.interface.canonicalize(item) for item in schema.pre_pos
        }
        additions = {
            world.interface.canonicalize(item) for item in schema.add_eff
        }
        deletions = {
            world.interface.canonicalize(item) for item in schema.del_eff
        }
        for effect in additions:
            if len(effect) != 3 or effect[0] != "in":
                continue
            content_parameter, receiver_parameter = effect[1:]
            if not _schema_parameter_accepts(
                world, schema, content_parameter, content
            ) or not _schema_parameter_accepts(
                world, schema, receiver_parameter, receiver
            ):
                continue
            source_relations = [
                item
                for item in preconditions | deletions
                if len(item) == 3
                and item[0] == "in"
                and item[1] == content_parameter
                and item[2] != receiver_parameter
            ]
            if not source_relations:
                continue
            saw_destination_schema = True
            source_parameters = {item[2] for item in source_relations}
            if not any(
                _schema_parameter_accepts(world, schema, parameter, source)
                for parameter in source_parameters
            ):
                continue
            holdings = [
                item
                for item in preconditions
                if len(item) == 3 and item[0] == "holding"
            ]
            if not holdings:
                return True
            if any(
                _schema_parameter_accepts(world, schema, holding[2], held)
                for holding in holdings
                for held in held_objects
            ):
                return True
    return not saw_destination_schema


def _unstable_support_pair(
    world: CanonicalWorld,
    obj: str,
    target: str,
) -> bool:
    """Whether a rigid support is being balanced on a concave vessel."""
    object_kinds = _declared_kind_tokens(world, obj)
    target_kinds = _declared_kind_tokens(world, target)
    return bool(
        object_kinds & {"plate", "rack", "tray"}
        and target_kinds
        & {"basin", "bottle", "bowl", "cup", "glass", "kettle", "mug", "pot"}
    )


def _rack_support_issue(
    world: CanonicalWorld,
    facts: set[Literal],
    target: str,
) -> str | None:
    """Reject using a non-flat object stored in a rack as a support."""
    target_location = _location(facts, target)
    if (
        target_location is not None
        and target_location[0] in {"on", "in"}
        and "rack" in _declared_kind_tokens(world, target_location[1])
        and ("flat", target) not in facts
    ):
        return f"target {target} is still stored in rack {target_location[1]}"
    return None


def _is_clear(world: CanonicalWorld, facts: set[Literal], target: str) -> bool:
    """Derive clear from the current support graph when GT exposes that state."""
    if not _has_clear_evidence(world, target):
        return True
    return not _children_on(facts, target)


def _open_lock_dependencies(
    world: CanonicalWorld,
    facts: set[Literal],
    target: str,
) -> tuple[str, ...]:
    """Find explicitly open GT components required closed by a lock schema."""
    if world.schemas is None:
        return ()
    target_kinds = _declared_kind_tokens(world, target)
    dependencies: set[str] = set()
    for schema in world.schemas.values():
        effects = {
            world.interface.canonicalize(item) for item in schema.add_eff
        }
        locked_targets = {
            item[1]
            for item in effects
            if len(item) == 2 and item[0] == "locked"
        }
        if not locked_targets:
            continue
        if not any(
            _parameter_kind_tokens(world, schema, parameter) & target_kinds
            for parameter in locked_targets
        ):
            continue
        for precondition in (
            world.interface.canonicalize(item) for item in schema.pre_pos
        ):
            if len(precondition) != 2 or precondition[0] != "closed":
                continue
            parameter = precondition[1]
            expected_kinds = _parameter_kind_tokens(world, schema, parameter)
            for obj in world.objects:
                if (
                    expected_kinds
                    and not expected_kinds
                    & _declared_kind_tokens(world, obj.name)
                ):
                    continue
                if ("open", obj.name) in facts:
                    dependencies.add(obj.name)
    return tuple(sorted(dependencies))


def _derive_semantic_facts(world: CanonicalWorld, facts: set[Literal]) -> None:
    """Apply only predicate-level implications, never object-name heuristics."""
    additions: set[Literal] = set()
    for fact in facts:
        if len(fact) == 2 and fact[0] == "boiled":
            additions.add(("heated", fact[1]))
        if len(fact) == 2 and fact[0] == "washed":
            additions.add(("clean", fact[1]))
        if len(fact) == 2 and fact[0] == "stirred":
            additions.update(
                ("stirred", item[2])
                for item in facts
                if len(item) == 3 and item[0] == "in" and item[1] == fact[1]
            )
        if len(fact) == 3 and fact[0] == "in":
            content, container = fact[1], fact[2]
            kinds = _declared_kind_tokens(world, content)
            if "hot" in kinds and "water" in kinds:
                additions.add(("contains_hot_water", container))
            elif kinds & {"purified", "cold"} and "water" in kinds:
                additions.add(("contains_purified_water", container))
            elif "water" in kinds:
                additions.add(("contains_water", container))
    facts.update(additions)
    containers = {
        fact[1]
        for fact in facts
        if len(fact) == 2 and fact[0] in {"contains_hot_water", "contains_purified_water"}
    }
    for container in containers:
        if (
            ("contains_hot_water", container) in facts
            and ("contains_purified_water", container) in facts
        ):
            facts.add(("mixed_water", container))
            facts.add(("heated", container))

def _unordered_stack_satisfied(
    world: CanonicalWorld,
    facts: set[Literal],
    instruction: str,
) -> bool:
    text = " ".join(instruction.lower().split())
    if "top to bottom" in text or not re.search(
        r"\bstack\s+(?:them|all\b|the\s+(?:bowls?|blocks?|plates?|cups?))",
        text,
    ):
        return False
    goals = {item for item in world.goal_positive if len(item) == 3 and item[0] == "on"}
    subjects = {item[1] for item in goals}
    members = subjects | {
        item[2] for item in goals
        if item[2] not in subjects
        and _declared_kind_tokens(world, item[2]) & {"bowl", "block", "plate", "cup"}
    }
    if len(members) < 2:
        return False
    supports = {
        item[2] for item in world.facts | world.goal_positive
        if len(item) == 3 and item[0] == "on" and item[1] in members
        and item[2] not in members
        and _declared_kind_tokens(world, item[2]) & set(normalized_tokens(instruction))
    }
    if not supports:
        return False
    edges = {
        (item[1], item[2])
        for item in facts
        if len(item) == 3 and item[0] == "on" and item[1] in members
    }
    if len(edges) != len(members):
        return False
    parents = {child: parent for child, parent in edges}
    if set(parents) != members or sum(parent in supports for parent in parents.values()) != 1:
        return False
    for child in members:
        seen = {child}
        current = child
        while parents.get(current) in members:
            current = parents[current]
            if current in seen:
                return False
            seen.add(current)
        if parents.get(current) not in supports:
            return False
    return True


def _goal_present(facts: set[Literal], goal: Literal) -> bool:
    if goal in facts:
        return True
    if len(goal) == 2 and goal[0] == "clear":
        return not any(
            len(item) == 3
            and item[0] in {"on", "blocks", "blocks_opening", "blocks_closing"}
            and item[2] == goal[1]
            for item in facts
        )
    return False


def _location_conflicts(facts: set[Literal]) -> set[str]:
    locations: dict[str, set[Literal]] = {}
    for fact in facts:
        if len(fact) != 3:
            continue
        if fact[0] == "holding":
            obj = fact[2]
        elif fact[0] in {"on", "in", "inserted"}:
            obj = fact[1]
        else:
            continue
        locations.setdefault(obj, set()).add(fact)
    return {obj for obj, places in locations.items() if len(places) > 1}


def replay_on_gt(
    gt: CanonicalWorld,
    actions: tuple[CanonicalStep, ...],
    *,
    instruction: str | None = None,
    source_overrides: frozenset[tuple[int, str, str]] = frozenset(),
) -> CrossDomainResult:
    facts = set(gt.facts)
    _derive_semantic_facts(gt, facts)
    known_blockers = {
        fact
        for fact in facts
        if len(fact) == 3 and fact[0] in {"blocks", "blocks_opening", "blocks_closing"}
    }
    trace = []
    for step_number, action in enumerate(actions, 1):
        before = set(facts)
        facts = set(before)
        restored_blockers: tuple[Literal, ...] = ()
        known_blockers.update(
            fact
            for fact in action.positive_preconditions + action.delete_effects
            if len(fact) == 3 and fact[0] in {"blocks", "blocks_opening", "blocks_closing"}
        )

        def stop(
            status: VerificationStatus, category: str, detail: str
        ) -> CrossDomainResult:
            issue = CrossDomainIssue(
                step_number, category, detail, action.raw_action
            )
            return CrossDomainResult(status, issue, tuple(trace), frozenset(before))

        def commit(*, allowed: set[str] | None = None) -> CrossDomainResult | None:
            _apply_declared_effects(facts, action, allowed)
            _apply_semantic_effects(gt, facts, action)
            facts.update(restored_blockers)
            _derive_semantic_facts(gt, facts)
            for positive, negative in (("wet", "dry"), ("clean", "dirty")):
                for item in tuple(facts - before):
                    if len(item) == 2 and item[0] in {positive, negative}:
                        inverse = negative if item[0] == positive else positive
                        facts.discard((inverse, item[1]))
            new_conflicts = _location_conflicts(facts) - _location_conflicts(before)
            if new_conflicts:
                return stop(
                    VerificationStatus.FAIL,
                    "invalid_transition",
                    f"objects have multiple exclusive locations: {sorted(new_conflicts)}",
                )
            trace.append(CrossDomainTrace(
                step_number, action.raw_action, action.family.value,
                tuple(sorted(facts - before)), tuple(sorted(before - facts)),
            ))
            return None

        new_locations: dict[str, set[Literal]] = {}
        for effect in action.add_effects:
            if len(effect) == 3 and effect[0] in {"on", "in", "inserted"}:
                new_locations.setdefault(effect[1], set()).add(effect)
        if any(len(locations) > 1 for locations in new_locations.values()):
            return stop(
                VerificationStatus.FAIL,
                "invalid_transition",
                "one object cannot acquire multiple exclusive locations in one action",
            )

        if action.ambiguous_objects:
            detail = f"object identity is unresolved: {list(action.ambiguous_objects)}"
            return stop(VerificationStatus.UNKNOWN, "ambiguous_object_mapping", detail)
        if action.unmapped_objects:
            detail = f"candidate objects rejected by initial-scene check: {list(action.unmapped_objects)}"
            return stop(VerificationStatus.FAIL, "object_not_in_scene", detail)
        family = action.family
        role = dict(action.roles)
        hand = role.get("hand")
        obj = role.get("object")
        target = role.get("target")
        allowed_effects: set[str] | None = None
        transfer = _transfer_signature(action)
        if transfer is not None and not _transfer_tool_supported(
            gt, action, *transfer
        ):
            _, source, receiver = transfer
            return stop(
                VerificationStatus.FAIL,
                "unsupported_capability",
                f"GT transfer schemas do not support moving payload from {source} "
                f"to {receiver} with the candidate's held carrier",
            )
        empty_subject = _empty_subject(action)
        if empty_subject is not None and not _can_be_emptied(gt, empty_subject, action):
            return stop(VerificationStatus.FAIL, "unsupported_capability", f"{empty_subject} is not a container or content-bearing object")
        if family is ActionFamily.UNSUPPORTED:
            return stop(VerificationStatus.UNKNOWN, "unsupported_action", "no canonical action semantics")
        if family in {
            ActionFamily.GENERIC_PDDL,
            ActionFamily.WIPE,
            ActionFamily.STIR,
            ActionFamily.CUT,
            ActionFamily.WASH,
            ActionFamily.FOLD,
            ActionFamily.SCRUNCH,
            ActionFamily.FILL,
            ActionFamily.WET,
            ActionFamily.EMPTY,
            ActionFamily.SWEEP,
            ActionFamily.ORIENT,
            ActionFamily.CONFIGURE,
            ActionFamily.ACTIVATE,
            ActionFamily.SLIDE,
            ActionFamily.PUSH,
            ActionFamily.UNBLOCK,
        }:
            allowed = set(FAMILY_EFFECT_PREDICATES.get(family, set()))
            allowed.update(
                item[0]
                for item in action.add_effects + action.delete_effects
                if item[0] in {"holding", "hand_free", *LOCATION_PREDICATES}
            )
            allowed.update(item[0] for item in action.semantic_effects)
            allowed_effects = allowed
            supported = {
                description.canonical_name
                for description in gt.interface.descriptions
            }
            if not allowed and not action.trusted_declared_semantics:
                effects = action.add_effects + action.delete_effects + action.semantic_effects
                if not (
                    family is ActionFamily.GENERIC_PDDL
                    and effects
                    and all(
                        item[0] not in supported and item[0] not in STRUCTURAL_PREDICATES
                        for item in effects
                    )
                ):
                    return stop(
                        VerificationStatus.UNKNOWN,
                        "unsupported_action",
                        "action family has no reviewed transition signature",
                    )
            motion_issue = _generic_motion_issue(action)
            if motion_issue is not None:
                return stop(VerificationStatus.FAIL, "invalid_transition", motion_issue)
            precondition_predicates = (
                allowed | PHYSICAL_PRECONDITION_PREDICATES
            ) & supported
            missing = list(
                _state_preconditions_missing(
                    gt,
                    facts,
                    action,
                    precondition_predicates,
                )
            )
            forbidden = sorted(
                item
                for item in action.negative_preconditions
                if item[0] in precondition_predicates and item in facts
            )
            if missing or forbidden:
                return stop(VerificationStatus.FAIL, "missing_precondition", f"missing={missing}; forbidden={forbidden}")
            transfer_issue = _check_compound_transfer_receiver(gt, facts, action, stop)
            if transfer_issue is not None:
                return transfer_issue
            if (
                family is ActionFamily.WASH
                and (target := action.role("object")) is not None
                and _is_container_kind(gt, target)
                and any(
                    item.canonical_name == "contains_water"
                    for item in gt.interface.descriptions
                )
            ):
                facts.add(("contains_water", target))
        elif family is ActionFamily.PICK:
            source = role.get("source")
            if not hand or not obj:
                return stop(VerificationStatus.UNKNOWN, "missing_role", "pick requires hand and object")
            if not _direct_pick_supported(gt, obj):
                return stop(
                    VerificationStatus.FAIL,
                    "unsupported_capability",
                    f"GT schemas do not permit directly holding {obj}",
                )
            location = _location(facts, obj)
            held_objects = {
                item[2]
                for item in facts
                if len(item) == 3 and item[0] == "holding" and item[1] == hand
            }
            released_objects = {
                item[2]
                for item in action.delete_effects
                if len(item) == 3 and item[0] == "holding" and item[1] == hand
            }
            can_transfer_grasp = bool(held_objects) and held_objects <= released_objects
            if (
                ("hand_free", hand) not in facts
                and not can_transfer_grasp
            ) or location is None:
                return stop(VerificationStatus.FAIL, "missing_precondition", "hand is occupied or object has no location")
            source_is_visually_verified = (step_number, obj, source or "") in source_overrides
            if source and not source_is_visually_verified and not _source_matches(
                gt, facts, obj, source, action.raw_action
            ):
                return stop(VerificationStatus.FAIL, "source_mismatch", f"object is at {location}, not {source}")
            requires_clear = ("clear", obj) in action.positive_preconditions
            blocked = (
                not _accessible(facts, obj)
                or _explicit_blockers(facts, obj)
                or (requires_clear and _children_on(facts, obj))
            )
            if blocked:
                return stop(VerificationStatus.FAIL, "blocked_not_accessible", "object is blocked or inaccessible")
            detach_location(facts, obj)
            for released in released_objects:
                facts.discard(("holding", hand, released))
            update_hand_state(facts, hand, held_object=obj)
        elif family in {ActionFamily.PLACE_ON, ActionFamily.PLACE_IN, ActionFamily.INSERT}:
            if not hand or not obj or not target:
                return stop(VerificationStatus.UNKNOWN, "missing_role", "placement requires hand, object, and target")
            carrier = next(
                (
                    item[2]
                    for item in facts
                    if len(item) == 3
                    and item[0] == "in"
                    and item[1] == obj
                    and ("holding", hand, item[2]) in facts
                ),
                None,
            )
            if (
                ("holding", hand, obj) not in facts
                and carrier is None
            ) or not _accessible(facts, target):
                return stop(VerificationStatus.FAIL, "missing_precondition", "object is not held or target is inaccessible")
            relation = action.placement_relation or {
                ActionFamily.PLACE_ON: "on",
                ActionFamily.PLACE_IN: "in",
                ActionFamily.INSERT: "inserted",
            }[family]
            if relation == "on" and _unstable_support_pair(gt, obj, target):
                return stop(
                    VerificationStatus.FAIL,
                    "invalid_pose",
                    f"{target} is not a stable support for {obj}",
                )
            rack_issue = (
                _rack_support_issue(gt, facts, target)
                if relation == "on"
                else None
            )
            if rack_issue is not None:
                return stop(
                    VerificationStatus.FAIL,
                    "invalid_pose",
                    rack_issue,
                )
            if (
                family is ActionFamily.PLACE_ON
                and _has_pose_capability(gt, target)
                and any((pose, target) in facts for pose in ("vertical", "upside_down"))
            ):
                return stop(
                    VerificationStatus.FAIL,
                    "invalid_pose",
                    f"target {target} cannot support an object in its current pose",
                )
            pose_missing = tuple(
                item
                for item in action.positive_preconditions
                if item[0] in {"flat", "upright", "vertical"}
                and len(item) == 2
                and _has_pose_capability(gt, item[1])
                and not _pose_precondition_satisfied(facts, item)
                and not (
                    item[0] == "flat"
                    and len(item) == 2
                    and _location(facts, item[1]) is not None
                    and _location(facts, item[1])[0] == "on"
                    and ("vertical", item[1]) not in facts
                    and ("upside_down", item[1]) not in facts
                )
            )
            if pose_missing:
                return stop(VerificationStatus.FAIL, "invalid_pose", f"missing={list(pose_missing)}")
            if (
                ("clear", target) in action.positive_preconditions
                or _has_clear_evidence(gt, target)
            ) and not _is_clear(gt, facts, target):
                return stop(VerificationStatus.FAIL, "blocked_not_clear", f"target {target} is not clear")
            open_portals = {
                item[1]
                for item in action.positive_preconditions
                if item[0] == "open" and len(item) == 2 and item[1] != target
            }
            if (
                family is ActionFamily.PLACE_IN
                and ("closed", target) in facts
                and not any(("open", portal) in facts for portal in open_portals)
            ):
                return stop(VerificationStatus.FAIL, "closed_target", f"target {target} is closed")
            if family in {ActionFamily.PLACE_IN, ActionFamily.INSERT} and ("upside_down", target) in facts:
                return stop(VerificationStatus.FAIL, "invalid_pose", f"target {target} is upside down")
            attach_location(facts, relation, obj, target)
            if carrier is None:
                update_hand_state(facts, hand, release_object=obj)
            if relation == "in":
                restored_blockers = tuple(
                    fact
                    for fact in known_blockers
                    if fact[1:] == (obj, target)
                )
        elif family in {ActionFamily.OPEN, ActionFamily.CLOSE}:
            if not target:
                return stop(VerificationStatus.UNKNOWN, "missing_role", "open/close requires target")
            if family is ActionFamily.OPEN:
                if not _has_closure_capability(gt, target):
                    return stop(VerificationStatus.FAIL, "unsupported_capability", f"GT scene does not identify {target} as openable")
            missing = _state_preconditions_missing(
                gt, facts, action, {"hand_free", "holding", "open", "closed"}
            )
            if missing:
                return stop(VerificationStatus.FAIL, "missing_precondition", f"missing={list(missing)}")
            if family is ActionFamily.OPEN:
                if not _is_clear(gt, facts, target):
                    return stop(
                        VerificationStatus.FAIL,
                        "blocked_not_clear",
                        f"target {target} is not clear",
                    )
                blockers = tuple(
                    sorted(
                        set(_opening_blockers(facts, target))
                    )
                )
                if ("locked", target) in facts or blockers:
                    return stop(VerificationStatus.FAIL, "blocked_not_clear", f"target blockers={sorted(set(blockers))}")
                _set_closure_pair(gt, facts, "open", "closed", target)
            else:
                blockers = tuple(item[1] for item in facts if item[0] in {"blocks", "blocks_closing"} and item[2] == target)
                if blockers:
                    return stop(VerificationStatus.FAIL, "blocked_not_clear", f"target blockers={sorted(set(blockers))}")
                _set_closure_pair(gt, facts, "closed", "open", target)
        elif family in {ActionFamily.TURN_ON, ActionFamily.TURN_OFF, ActionFamily.LOCK, ActionFamily.UNLOCK}:
            if not target:
                return stop(VerificationStatus.UNKNOWN, "missing_role", "state action requires target")
            supported = {
                description.canonical_name
                for description in gt.interface.descriptions
            }
            state_predicates = {
                "hand_free", "holding", "under", "open", "closed",
                "in", "on", "inserted",
            }
            target_evidence = gt.facts | gt.goal_positive | gt.goal_negative
            for pair in (("power_on", "power_off"), ("locked", "unlocked")):
                if any(
                    (state, item[1]) in target_evidence
                    for item in action.positive_preconditions
                    if len(item) == 2 and item[0] in pair
                    for state in pair
                ):
                    state_predicates.update(pair)
            missing = _state_preconditions_missing(
                gt,
                facts,
                action,
                {
                    predicate for predicate in state_predicates
                    if predicate in supported
                },
            )
            if missing:
                return stop(VerificationStatus.FAIL, "missing_precondition", f"missing={list(missing)}")
            if family in {ActionFamily.TURN_ON, ActionFamily.TURN_OFF}:
                # Only the state target is operated by a turn-on/turn-off action.
                # Other arguments may be controls, payloads, or scene witnesses;
                # treating every argument as a locked device over-constrains
                # domains that represent a lock on a control component.
                locked = (
                    [target]
                    if target is not None and ("locked", target) in facts
                    else []
                )
                if locked:
                    return stop(VerificationStatus.FAIL, "missing_precondition", f"locked objects must be unlocked first: {locked}")
            if family is ActionFamily.LOCK:
                open_dependencies = _open_lock_dependencies(gt, facts, target)
                if open_dependencies:
                    return stop(
                        VerificationStatus.FAIL,
                        "missing_precondition",
                        f"lock dependencies are open: {list(open_dependencies)}",
                    )
                if not any(
                    (state, target) in gt.facts for state in ("locked", "unlocked")
                ):
                    return stop(VerificationStatus.FAIL, "unsupported_capability", f"GT scene does not identify {target} as lockable")
            pairs = {
                ActionFamily.TURN_ON: ("power_on", "power_off"),
                ActionFamily.TURN_OFF: ("power_off", "power_on"),
                ActionFamily.LOCK: ("locked", "unlocked"),
                ActionFamily.UNLOCK: ("unlocked", "locked"),
            }
            _set_pair(facts, *pairs[family], target)
            transfer_issue = _check_compound_transfer_receiver(gt, facts, action, stop)
            if transfer_issue is not None:
                return transfer_issue
        elif family in {ActionFamily.REMOVE_CLOSURE, ActionFamily.REPLACE_CLOSURE}:
            closure = role.get("closure") or obj
            vessel = role.get("vessel") or target
            if not hand or not closure or not vessel:
                return stop(VerificationStatus.UNKNOWN, "missing_role", "closure action requires hand, closure, and vessel")
            if family is ActionFamily.REMOVE_CLOSURE:
                takes_closure = ("holding", hand, closure) in action.add_effects
                blockers = tuple(
                    item[1]
                    for item in facts
                    if len(item) == 3
                    and item[0] in {"blocks", "blocks_opening"}
                    and item[2] in {closure, vessel}
                ) + tuple(
                    child for child in _opening_blockers(facts, vessel) if child != closure
                ) + _children_on(facts, closure)
                if blockers:
                    return stop(VerificationStatus.FAIL, "blocked_not_clear", f"closure blockers={sorted(set(blockers))}")
                if (takes_closure and ("hand_free", hand) not in facts) or not _has_location(facts, closure, vessel):
                    return stop(VerificationStatus.FAIL, "missing_precondition", "closure is not on vessel or hand is occupied")
                detach_location(facts, closure)
                if takes_closure:
                    update_hand_state(facts, hand, held_object=closure)
                _set_closure_pair(gt, facts, "open", "closed", vessel)
            else:
                if ("holding", hand, closure) not in facts:
                    return stop(VerificationStatus.FAIL, "missing_precondition", "closure is not held")
                blockers = tuple(
                    item[1] for item in facts
                    if len(item) == 3 and item[0] in {"blocks", "blocks_closing"}
                    and item[2] in {closure, vessel}
                )
                if blockers:
                    return stop(VerificationStatus.FAIL, "blocked_not_clear", f"closure blockers={sorted(set(blockers))}")
                attach_location(facts, "on", closure, vessel)
                update_hand_state(facts, hand, release_object=closure)
                _set_closure_pair(gt, facts, "closed", "open", vessel)
        elif family in {ActionFamily.POUR, ActionFamily.SCOOP}:
            content = role.get("content")
            source = role.get("source")
            receiver = role.get("receiver") or role.get("tool")
            transfer_effects = [
                item for item in action.add_effects if item[0] == "in" and len(item) == 3
            ]
            if (not content or not source or not receiver) and not transfer_effects:
                missing = sorted(set(action.positive_preconditions) - facts)
                forbidden = sorted(set(action.negative_preconditions) & facts)
                if missing or forbidden:
                    return stop(VerificationStatus.FAIL, "missing_precondition", f"missing={missing}; forbidden={forbidden}")
                issue = commit()
                if issue is not None:
                    return issue
                continue
            if not content or not source or not receiver:
                return stop(VerificationStatus.UNKNOWN, "missing_role", "transfer requires content, source, and receiver")
            if ("in", content, source) not in facts:
                return stop(VerificationStatus.FAIL, "missing_precondition", f"content is not in source {source}")
            if _is_closed(gt, facts, source) and _transfer_requires_open_source(
                gt, action, source
            ):
                return stop(VerificationStatus.FAIL, "closed_source", f"source {source} is closed")
            if _is_closed(gt, facts, receiver) or not _accessible(facts, receiver):
                return stop(VerificationStatus.FAIL, "closed_target", f"receiver {receiver} is inaccessible")
            receiver_location = _location(facts, receiver)
            if (
                receiver_location is not None
                and receiver_location[0] in {"on", "in"}
                and "rack" in _declared_kind_tokens(gt, receiver_location[1])
                and _declared_kind_tokens(gt, receiver)
                & {"basin", "bottle", "bowl", "cup", "glass", "kettle", "mug", "pot"}
            ):
                return stop(
                    VerificationStatus.FAIL,
                    "invalid_pose",
                    f"receiver {receiver} is still stored in rack {receiver_location[1]}",
                )
            if any((pose, receiver) in facts for pose in ("vertical", "upside_down")):
                return stop(VerificationStatus.FAIL, "invalid_pose", f"receiver {receiver} is not upright")
            blockers = _receiver_blockers(gt, facts, receiver)
            if blockers:
                return stop(VerificationStatus.FAIL, "blocked_not_clear", f"receiver {receiver} has objects on top: {list(blockers)}")
            facts.discard(("in", content, source))
            facts.add(("in", content, receiver))
        else:
            return stop(VerificationStatus.UNKNOWN, "unsupported_action", f"family {family.value} is not replayable")
        issue = commit(allowed=allowed_effects)
        if issue is not None:
            return issue
    if instruction:
        action_issue = instruction_action_issue(
            instruction, actions, gt, bind_named=True,
        )
        if action_issue is not None:
            step_number, detail = action_issue
            issue = CrossDomainIssue(
                step_number, "instruction_action_missing", detail, "<instruction>"
            )
            return CrossDomainResult(
                VerificationStatus.FAIL, issue, tuple(trace), frozenset(facts)
            )
        order_issue = instruction_order_issue(instruction, actions, gt)
        if order_issue is not None:
            step_number, detail = order_issue
            action = actions[step_number - 1]
            issue = CrossDomainIssue(
                step_number, "instruction_order", detail, action.raw_action
            )
            return CrossDomainResult(VerificationStatus.FAIL, issue, tuple(trace), frozenset(facts))
    unordered_stack = _unordered_stack_satisfied(gt, facts, instruction or "")
    # Heating is an operation-level goal, independent of its modeled subject.
    # Count only executed effects, including reheating an already heated item.
    performed_heating = any(
        len(item) == 2 and item[0] == "heated"
        for action, transition in zip(actions, trace)
        for item in transition.added + action.semantic_effects + (
            action.add_effects
            if action.trusted_declared_semantics
            or "heated" in FAMILY_EFFECT_PREDICATES.get(action.family, set())
            else ()
        )
    )
    missing = sorted(
        goal
        for goal in gt.goal_positive
        if not (
            unordered_stack
            and len(goal) == 3
            and goal[0] == "on"
            and goal[1] in {item[1] for item in gt.goal_positive if len(item) == 3 and item[0] == "on"}
        )
        and not (goal[0] == "heated" and performed_heating)
        and not _goal_present(facts, goal)
    )
    violated = sorted(gt.goal_negative & facts)
    if missing or violated:
        issue = CrossDomainIssue(
            len(actions) + 1,
            "final_goal_not_achieved",
            f"missing={missing}; violated={violated}",
            "<gt-goal>",
        )
        return CrossDomainResult(VerificationStatus.FAIL, issue, tuple(trace), frozenset(facts))
    return CrossDomainResult(VerificationStatus.PASS, None, tuple(trace), frozenset(facts))
