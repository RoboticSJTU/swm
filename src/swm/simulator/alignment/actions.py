from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from swm.pddl.strips import GroundAction

from swm.simulator.ir.mapping import map_ground_action, map_schema
from swm.simulator.ir.model import ActionFamily

from .predicates import CanonicalWorld, Literal


LOCATION_PREDICATES = {
    "on", "in", "inserted", "under", "against", "away_from", "in_front_of",
    "blocks", "blocks_opening", "blocks_closing",
}
POSE_PREDICATES = {"upright", "upside_down", "flat", "vertical"}
STRUCTURAL_PREDICATES = {
    "hand_free", "holding", *LOCATION_PREDICATES, "open", "closed",
    "power_on", "power_off", "locked", "unlocked", "clear",
    *POSE_PREDICATES,
}

_NAME_SEMANTIC_FAMILIES = (
    ({"wipe", "clean"}, ActionFamily.WIPE),
    ({"stir", "mix"}, ActionFamily.STIR),
    ({"cut", "chop", "slice"}, ActionFamily.CUT),
    ({"wash", "rinse"}, ActionFamily.WASH),
    ({"fold"}, ActionFamily.FOLD),
    ({"scrunch", "crumple"}, ActionFamily.SCRUNCH),
    ({"fill"}, ActionFamily.FILL),
    ({"wet", "dampen"}, ActionFamily.WET),
    ({"empty", "drain"}, ActionFamily.EMPTY),
    ({"sweep"}, ActionFamily.SWEEP),
    ({"slide"}, ActionFamily.SLIDE),
    ({"push"}, ActionFamily.PUSH),
    ({"press", "activate"}, ActionFamily.ACTIVATE),
    ({"configure", "select"}, ActionFamily.CONFIGURE),
)


@dataclass(frozen=True)
class CanonicalStep:
    raw_action: str
    family: ActionFamily
    roles: tuple[tuple[str, str], ...]
    positive_preconditions: tuple[Literal, ...]
    negative_preconditions: tuple[Literal, ...]
    add_effects: tuple[Literal, ...]
    delete_effects: tuple[Literal, ...]
    unmapped_objects: tuple[str, ...]
    ambiguous_objects: tuple[str, ...]
    objects: tuple[str, ...]
    placement_relation: str | None = None
    trusted_declared_semantics: bool = False
    semantic_effects: tuple[Literal, ...] = ()

    def role(self, name: str) -> str | None:
        return next((value for role, value in self.roles if role == name), None)


def required_action_objects(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None = None,
) -> set[str]:
    """Return action arguments that must denote canonical scene objects.

    Some generated domains add a witness parameter solely to state that no
    blocker relation exists. Canonical replay checks blockers from GT state, so
    that witness is not a physical participant. Arguments used by roles,
    effects, or positive dynamic preconditions remain mandatory.
    """
    mapped = map_ground_action(action)
    family = mapped.family
    if family is ActionFamily.UNSUPPORTED or (
        family is ActionFamily.GENERIC_PDDL
        and not _has_reviewed_name_semantics(action)
    ):
        return set(action.args)
    roles = set(dict(mapped.roles).values())
    optional_witnesses = _optional_precondition_witnesses(action, candidate_world, roles)
    liquid_contents = _abstract_liquid_effect_objects(
        action,
        candidate_world,
        reference_world,
    )
    required = set(roles)
    for literal in action.pre_pos | action.add_eff | action.del_eff:
        canonical = candidate_world.interface.canonicalize(literal)
        if candidate_world.interface.is_static(literal[0]):
            continue
        required.update(canonical[1:])
    blocker_predicates = {"blocks", "blocks_opening", "blocks_closing"}
    for literal in action.pre_neg:
        canonical = candidate_world.interface.canonicalize(literal)
        if canonical[0] not in blocker_predicates:
            required.update(canonical[1:])
    # A liquid payload may be omitted only when the reference interface has an
    # explicit abstract ``contains_*`` state for that operation.  If the GT
    # models the liquid as a physical object (ordinary ``in``), it remains a
    # required identity just like every other action argument.
    liquid_witnesses = liquid_contents - roles
    return required - optional_witnesses - liquid_witnesses


def _optional_precondition_witnesses(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    roles: set[str],
) -> set[str]:
    """Identify unobservable, precondition-only witnesses in known actions.

    Candidate domains may model latent food inside a food box or an external
    plug/outlet pair while the GT scene deliberately abstracts those details.
    A witness is optional only when it has no effect or canonical action role,
    has no unary state guard, and is not directly related to an action-role or
    effect object.  Direct payloads, tools, and stateful resources remain
    mandatory.
    """
    structural_unary = {
        "hand_free", "open", "closed", "power_on", "power_off", "locked",
        "unlocked", "upright", "upside_down", "flat", "vertical", "activated",
        "configured", "pushed", "slid",
    }
    liquid_contents = _liquid_effect_objects(action, candidate_world)
    effect_objects = {
        argument
        for literal in action.add_eff | action.del_eff
        if not candidate_world.interface.is_static(literal[0])
        for canonical in (candidate_world.interface.canonicalize(literal),)
        for argument in canonical[1:]
        if (
            canonical[0] in structural_unary
            or (
                len(canonical) >= 3
                and not (canonical[0] == "in" and argument in liquid_contents)
            )
        )
    }
    anchors = roles | effect_objects
    dynamic_preconditions = [
        candidate_world.interface.canonicalize(literal)
        for literal in action.pre_pos
        if not candidate_world.interface.is_static(literal[0])
    ]
    directly_required = set(anchors)
    for literal in dynamic_preconditions:
        if len(literal) >= 3 and any(argument in anchors for argument in literal[1:]):
            directly_required.update(
                argument for argument in literal[1:] if argument not in liquid_contents
            )
    witnesses = set()
    for argument in action.args:
        if argument in directly_required:
            continue
        occurrences = [
            literal for literal in dynamic_preconditions if argument in literal[1:]
        ]
        if occurrences and all(len(literal) >= 3 for literal in occurrences):
            witnesses.add(argument)
    return witnesses


def _liquid_effect_objects(
    action: GroundAction,
    candidate_world: CanonicalWorld,
) -> set[str]:
    liquid_tokens = {"water", "milk", "soup", "detergent", "disinfectant"}
    result = set()
    for literal in action.add_eff:
        if candidate_world.interface.is_static(literal[0]):
            continue
        canonical = candidate_world.interface.canonicalize(literal)
        if len(canonical) != 3 or canonical[0] != "in":
            continue
        content = candidate_world.object(canonical[1])
        if content is None:
            continue
        tokens = set(content.name_tokens)
        for kind in content.kinds:
            tokens.update(kind.split("_"))
        if liquid_tokens & tokens:
            result.add(canonical[1])
    return result


def _abstract_liquid_effect_objects(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
) -> set[str]:
    """Return payloads represented abstractly by the reference interface.

    The same candidate syntax, ``(in liquid container)``, is used for two
    different domain encodings: a physical liquid entity, or a shorthand for a
    GT state such as ``contains_hot_water(container)``.  The candidate alone
    cannot distinguish those encodings.  Only the reference interface can
    authorize the abstraction, which keeps mapping and replay responsibilities
    separate without relying on object or task names.
    """
    if reference_world is None:
        return set()
    abstract_predicates = {
        description.canonical_name
        for description in reference_world.interface.descriptions
        if description.canonical_name.startswith("contains_")
    }
    family = map_ground_action(action).family
    if family is ActionFamily.WASH:
        abstract_predicates.update(
            predicate
            for predicate in ("washed", "wet", "clean")
            if any(
                item.canonical_name == predicate
                for item in reference_world.interface.descriptions
            )
        )
    elif family is ActionFamily.WET and any(
        item.canonical_name == "wet"
        for item in reference_world.interface.descriptions
    ):
        abstract_predicates.add("wet")
    result = set()
    for content in _liquid_effect_objects(action, candidate_world):
        tokens = set()
        content_object = candidate_world.object(content)
        if content_object is not None:
            tokens.update(content_object.name_tokens)
            for kind in content_object.kinds:
                tokens.update(kind.split("_"))
        if "hot" in tokens and "water" in tokens:
            predicate = "contains_hot_water"
        elif "water" in tokens and ({"cold", "purified"} & tokens):
            predicate = "contains_purified_water"
        elif "water" in tokens:
            predicate = "contains_water"
        else:
            predicate = ""
        if predicate not in abstract_predicates:
            if family is ActionFamily.WASH:
                predicate = next(
                    (
                        candidate
                        for candidate in ("washed", "wet", "clean")
                        if candidate in abstract_predicates
                    ),
                    "",
                )
            elif family is ActionFamily.WET and "wet" in abstract_predicates:
                predicate = "wet"
        if predicate in abstract_predicates:
            result.add(content)
    return result


def abstract_liquid_effects(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
    object_mapping: dict[str, str],
) -> tuple[Literal, ...]:
    """Translate liquid shorthand into a declared GT unary effect.

    A candidate can express washing or filling as containment of a liquid
    payload.  The reference interface decides whether that payload is a real
    object or an abstract state transition.  Only the latter is projected,
    and only to a predicate declared by the GT domain.
    """
    if reference_world is None:
        return ()
    family = map_ground_action(action).family
    supported = {
        item.canonical_name for item in reference_world.interface.descriptions
    }
    effects: list[Literal] = []
    for content in _liquid_effect_objects(action, candidate_world):
        canonical = None
        for item in action.add_eff:
            candidate = candidate_world.interface.canonicalize(item)
            if len(candidate) == 3 and candidate[0] == "in" and candidate[1] == content:
                canonical = candidate
                break
        if canonical is None or len(canonical) != 3:
            continue
        receiver = object_mapping.get(canonical[2])
        if receiver is None:
            continue
        tokens = set()
        content_object = candidate_world.object(content)
        if content_object is not None:
            tokens.update(content_object.name_tokens)
            for kind in content_object.kinds:
                tokens.update(kind.split("_"))
        predicate = ""
        if "hot" in tokens and "water" in tokens:
            predicate = "contains_hot_water"
        elif "water" in tokens and ({"cold", "purified"} & tokens):
            predicate = "contains_purified_water"
        elif "water" in tokens:
            predicate = "contains_water"
        if predicate not in supported:
            predicate = next(
                (
                    candidate
                    for candidate in ("washed", "wet", "clean")
                    if candidate in supported
                ),
                "",
            ) if family is ActionFamily.WASH else (
                "wet" if family is ActionFamily.WET and "wet" in supported else ""
            )
        if predicate:
            effects.append((predicate, receiver))
    return tuple(sorted(set(effects)))


def _project_declared_semantic_effects(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
    object_mapping: dict[str, str],
) -> tuple[Literal, ...]:
    """Project a semantic result across harmless predicate-arity encodings.

    Candidate domains sometimes encode an operation as ``cut(object, tool)``
    while the reference domain stores the persistent result as
    ``cut(object)``.  The operation's subject remains the first argument; the
    extra arguments describe the tool or context.  This projection is allowed
    only for a predicate declared by the GT interface and never for structural
    relations such as ``on`` or ``in``.
    """
    if reference_world is None:
        return ()
    supported: dict[str, set[int]] = {}
    for description in reference_world.interface.descriptions:
        if not description.static:
            supported.setdefault(description.canonical_name, set()).add(
                description.arity
            )
    effects: set[Literal] = set()
    for item in action.add_eff:
        canonical = candidate_world.interface.canonicalize(item)
        predicate = canonical[0]
        if predicate in STRUCTURAL_PREDICATES or predicate not in supported:
            continue
        candidate_arguments = canonical[1:]
        for arity in sorted(supported[predicate]):
            if arity <= 0 or arity > len(candidate_arguments):
                continue
            mapped = tuple(
                object_mapping.get(argument)
                for argument in candidate_arguments[:arity]
            )
            if all(argument is not None for argument in mapped):
                effects.add((predicate, *mapped))
    return tuple(sorted(effects))


def _semantic_name_family(action: GroundAction) -> ActionFamily | None:
    tokens = set(action.name.lower().split("_"))
    for names, family in _NAME_SEMANTIC_FAMILIES:
        if tokens & names:
            return family
    return None


def _schema_matches_semantic_family(schema: object, family: ActionFamily) -> bool:
    """Prefer an explicit operation verb over a structural effect family."""
    named_family = _semantic_name_family(schema)  # type: ignore[arg-type]
    return named_family is family or (
        named_family is None and map_schema(schema)[0] is family
    )


def _translated_dynamic_literals(
    items: set[Literal],
    world: CanonicalWorld,
    object_mapping: dict[str, str],
) -> set[Literal]:
    translated: set[Literal] = set()
    for item in items:
        if world.interface.is_static(item[0]):
            continue
        canonical = world.interface.canonicalize(item)
        mapped = tuple(object_mapping.get(argument) for argument in canonical[1:])
        if all(argument is not None for argument in mapped):
            translated.add((canonical[0], *mapped))
    return translated


def _unify_schema_literal(
    pattern: Literal,
    grounded: Literal,
    bindings: dict[str, str],
) -> dict[str, str] | None:
    if len(pattern) != len(grounded) or pattern[0] != grounded[0]:
        return None
    result = dict(bindings)
    for variable, value in zip(pattern[1:], grounded[1:]):
        prior = result.get(variable)
        if prior is not None and prior != value:
            return None
        result[variable] = value
    return result


def _predicate_inflection_forms(predicate: str) -> frozenset[str]:
    """Return conservative surface forms for a predicate's final word.

    Independently generated domains often use a process/result inflection for
    the same state (for example ``boiling`` versus ``boiled``).  This helper is
    intentionally lexical: it does not declare semantic synonyms, and is used
    only after schema arguments and physical preconditions have grounded.
    """
    prefix, separator, word = predicate.rpartition("_")
    forms = {word}
    for suffix in ("ing", "ed"):
        if len(word) <= len(suffix) + 1 or not word.endswith(suffix):
            continue
        stem = word[: -len(suffix)]
        forms.update({stem, stem + "e"})
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            forms.add(stem[:-1])
    if len(word) > 3 and word.endswith("s"):
        forms.add(word[:-1])
    return frozenset(
        f"{prefix}{separator}{form}" if separator else form
        for form in forms
    )


def _predicates_inflection_compatible(left: str, right: str) -> bool:
    return bool(
        _predicate_inflection_forms(left)
        & _predicate_inflection_forms(right)
    )


def _unify_inflection_compatible_literal(
    pattern: Literal,
    grounded: Literal,
    bindings: dict[str, str],
) -> dict[str, str] | None:
    if (
        len(pattern) != len(grounded)
        or not _predicates_inflection_compatible(pattern[0], grounded[0])
    ):
        return None
    return _unify_schema_literal(
        pattern,
        (pattern[0], *grounded[1:]),
        bindings,
    )


def _schema_parameter_kinds(
    reference_world: CanonicalWorld,
    schema: object,
    variable: str,
) -> set[str]:
    kinds = {
        item[0].removeprefix("kind:")
        for raw in getattr(schema, "pre_pos", set())
        for item in (reference_world.interface.canonicalize(raw),)
        if len(item) == 2 and item[0].startswith("kind:") and item[1] == variable
    }
    return kinds


def _is_control_component(world: CanonicalWorld, name: str) -> bool:
    description = world.object(name)
    tokens = set(() if description is None else description.name_tokens)
    if description is not None:
        for kind in description.kinds:
            tokens.update(kind.split("_"))
    return bool(tokens & {"button", "control", "dial", "knob", "lever", "lock", "switch"})


def _complete_schema_bindings(
    reference_world: CanonicalWorld,
    schema: object,
    seed: dict[str, str],
) -> tuple[dict[str, str], ...]:
    variables = tuple(
        variable
        for variable in getattr(schema, "params", ())
        if variable not in seed
    )
    if not variables:
        return (dict(seed),)
    choices: list[tuple[str, ...]] = []
    for variable in variables:
        expected = _schema_parameter_kinds(reference_world, schema, variable)
        candidates = tuple(
            item.name
            for item in reference_world.objects
            if not expected or expected & set(item.kinds)
        )
        if not candidates:
            return ()
        choices.append(candidates)
    count = 1
    for values in choices:
        count *= len(values)
        if count > 512:
            return ()
    return tuple(
        {**seed, **dict(zip(variables, values))}
        for values in product(*choices)
    )


def _ground_schema_literal(
    item: Literal,
    bindings: dict[str, str],
) -> Literal | None:
    if not all(argument in bindings for argument in item[1:]):
        return None
    return (item[0], *(bindings[argument] for argument in item[1:]))


def _ground_schema_preconditions(
    world: CanonicalWorld,
    preconditions: set[Literal],
    bindings: dict[str, str],
) -> frozenset[Literal] | None:
    static = {
        _ground_schema_literal(item, bindings)
        for item in preconditions
        if item[0].startswith(("kind:", "static:"))
    }
    if None in static or not static <= world.facts:
        return None
    return frozenset(
        grounded
        for item in preconditions
        if not item[0].startswith(("kind:", "static:"))
        for grounded in (_ground_schema_literal(item, bindings),)
        if grounded is not None
    )


def _schema_backed_state_proxy_projection(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
    object_mapping: dict[str, str],
    family: ActionFamily,
) -> tuple[tuple[Literal, ...], tuple[Literal, ...], frozenset[str], str | None]:
    """Project a candidate-local control state onto its GT device state.

    A candidate may model a visible button or switch as the owner of the power
    state while the GT keeps that state on the parent appliance.  The proxy is
    omitted only when a uniquely grounded same-family GT schema is supported by
    the other mapped action participants and dynamic preconditions.
    """
    state_pairs = {
        ActionFamily.TURN_ON: ("power_on", "power_off"),
        ActionFamily.TURN_OFF: ("power_off", "power_on"),
        ActionFamily.LOCK: ("locked", "unlocked"),
        ActionFamily.UNLOCK: ("unlocked", "locked"),
    }
    pair = state_pairs.get(family)
    if pair is None or reference_world is None or reference_world.schemas is None:
        return (), (), frozenset(), None
    desired, inverse = pair
    local_state_owners = {
        item[1]
        for literal in action.add_eff | action.del_eff
        for item in (candidate_world.interface.canonicalize(literal),)
        if len(item) == 2
        and item[0] in {desired, inverse}
    }
    if not local_state_owners:
        return (), (), frozenset(), None

    candidate_pre = _translated_dynamic_literals(
        action.pre_pos, candidate_world, object_mapping
    )
    mapped_participants = {
        object_mapping[argument]
        for argument in action.args
        if argument in object_mapping
    }
    witnesses: list[tuple[int, Literal, frozenset[Literal]]] = []
    for schema in reference_world.schemas.values():
        if map_schema(schema)[0] is not family:
            continue
        schema_pre = {
            reference_world.interface.canonicalize(item) for item in schema.pre_pos
        }
        schema_add = {
            reference_world.interface.canonicalize(item) for item in schema.add_eff
        }
        schema_delete = {
            reference_world.interface.canonicalize(item) for item in schema.del_eff
        }
        seed: dict[str, str] = {}
        for variable in schema.params:
            expected = _schema_parameter_kinds(reference_world, schema, variable)
            if not expected:
                continue
            matches = [
                name
                for name in mapped_participants
                if (
                    (description := reference_world.object(name)) is not None
                    and expected & set(description.kinds)
                )
            ]
            if len(matches) == 1:
                seed[variable] = matches[0]
        participant_overlap = len(set(seed.values()))
        if participant_overlap < 2:
            continue
        for bindings in _complete_schema_bindings(reference_world, schema, seed):
            grounded_pre = _ground_schema_preconditions(reference_world, schema_pre, bindings)
            if grounded_pre is None:
                continue
            overlap = len(grounded_pre & candidate_pre)
            if overlap < 1:
                continue
            grounded_add = {
                grounded
                for item in schema_add
                for grounded in (_ground_schema_literal(item, bindings),)
                if grounded is not None
            }
            grounded_delete = {
                grounded
                for item in schema_delete
                for grounded in (_ground_schema_literal(item, bindings),)
                if grounded is not None
            }
            effects = [
                item
                for item in grounded_add
                if len(item) == 2
                and item[0] == desired
                and (inverse, item[1]) in grounded_pre | grounded_delete
            ]
            for effect in effects:
                witnesses.append(
                    (3 * overlap + participant_overlap, effect, grounded_pre)
                )
    if not witnesses:
        return (), (), frozenset(), None
    best = max(score for score, _, _ in witnesses)
    selected = [item for item in witnesses if item[0] == best]
    effects = {effect for _, effect, _ in selected}
    if len(effects) != 1:
        return (), (), frozenset(), None
    precondition_sets = [preconditions for _, _, preconditions in selected]
    common = set(precondition_sets[0])
    for preconditions in precondition_sets[1:]:
        common.intersection_update(preconditions)
    projected = next(iter(effects))
    proxies = {
        owner
        for owner in local_state_owners
        if object_mapping.get(owner) != projected[1]
    }
    if any(
        not _is_control_component(candidate_world, owner)
        for owner in proxies
    ):
        return (), (), frozenset(), None
    candidate_state_preconditions = {
        canonical
        for item in action.pre_pos
        for canonical in (candidate_world.interface.canonicalize(item),)
        if len(canonical) == 2
        and canonical[0] in {"power_on", "power_off", "locked", "unlocked"}
    }
    for predicate, local_owner in candidate_state_preconditions:
        reference_owners = {
            item[1]
            for item in common
            if len(item) == 2 and item[0] == predicate
        }
        if (
            len(reference_owners) == 1
            and object_mapping.get(local_owner) != next(iter(reference_owners))
            and _is_control_component(candidate_world, local_owner)
        ):
            proxies.add(local_owner)
    if not proxies:
        return (), (), frozenset(), None
    physical = tuple(sorted(
        item
        for item in common
        if len(item) == 2
        and item[0] in {"power_on", "power_off", "locked", "unlocked"}
    ))
    return (projected,), physical, frozenset(proxies), projected[1]


def _schema_backed_read_state_proxy_projection(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
    object_mapping: dict[str, str],
) -> tuple[tuple[Literal, ...], tuple[Literal, ...], frozenset[str]]:
    """Project a read-only local control state onto its GT state owner.

    Generated domains may split a GT transition into a control action followed
    by a wait action.  The wait reads state from a local switch object while a
    GT schema stores that state on the appliance.  A projection is accepted
    only when a mapped result of the wait uniquely grounds a GT schema that
    co-declares the same state predicate on a mapped action participant.
    """
    if reference_world is None or reference_world.schemas is None:
        return (), (), frozenset()
    state_predicates = {"power_on", "power_off", "locked", "unlocked"}
    candidate_pre_raw = {
        candidate_world.interface.canonicalize(item) for item in action.pre_pos
    }
    proxy_states = {
        item[1]: item[0]
        for item in candidate_pre_raw
        if len(item) == 2
        and item[0] in state_predicates
        and item[1] not in object_mapping
    }
    if not proxy_states:
        return (), (), frozenset()
    for proxy in proxy_states:
        if any(
            proxy in item[1:]
            for item in action.add_eff | action.del_eff
        ) or any(
            proxy in item[1:]
            and not (len(item) == 2 and item[0] in state_predicates)
            for item in candidate_pre_raw
        ):
            return (), (), frozenset()
    candidate_add = _translated_dynamic_literals(
        action.add_eff, candidate_world, object_mapping
    )
    if not candidate_add:
        return (), (), frozenset()
    candidate_pre = _translated_dynamic_literals(
        action.pre_pos, candidate_world, object_mapping
    )
    mapped_participants = {
        object_mapping[argument]
        for argument in action.args
        if argument in object_mapping
    }
    witnesses: dict[str, list[tuple[int, Literal, frozenset[Literal]]]] = {
        proxy: [] for proxy in proxy_states
    }
    for schema in reference_world.schemas.values():
        schema_pre = {
            reference_world.interface.canonicalize(item) for item in schema.pre_pos
        }
        schema_add = {
            reference_world.interface.canonicalize(item) for item in schema.add_eff
        }
        anchors = [
            (pattern, grounded)
            for pattern in schema_add
            if not pattern[0].startswith(("kind:", "static:"))
            and pattern[0] not in state_predicates
            for grounded in candidate_add
            if len(pattern) == len(grounded)
            and _predicates_inflection_compatible(pattern[0], grounded[0])
        ]
        for pattern, grounded in anchors:
            seed = _unify_inflection_compatible_literal(pattern, grounded, {})
            if seed is None:
                continue
            for bindings in _complete_schema_bindings(reference_world, schema, seed):
                grounded_pre = _ground_schema_preconditions(reference_world, schema_pre, bindings)
                if grounded_pre is None:
                    continue
                grounded_add = {
                    grounded_item
                    for item in schema_add
                    if not item[0].startswith(("kind:", "static:"))
                    for grounded_item in (_ground_schema_literal(item, bindings),)
                    if grounded_item is not None
                }
                matched_results = frozenset(
                    reference_result
                    for reference_result in grounded_add
                    if any(
                        len(reference_result) == len(candidate_result)
                        and reference_result[1:] == candidate_result[1:]
                        and _predicates_inflection_compatible(
                            reference_result[0], candidate_result[0]
                        )
                        for candidate_result in candidate_add
                    )
                )
                if not matched_results:
                    continue
                participant_overlap = len(
                    mapped_participants
                    & {argument for item in grounded_pre | grounded_add for argument in item[1:]}
                )
                pre_overlap = len(grounded_pre & candidate_pre)
                score = 8 * len(matched_results) + 2 * participant_overlap + pre_overlap
                for proxy, predicate in proxy_states.items():
                    owners = {
                        item
                        for item in grounded_pre | grounded_add
                        if len(item) == 2
                        and item[0] == predicate
                        and item[1] in mapped_participants
                    }
                    for owner in owners:
                        witnesses[proxy].append((score, owner, matched_results))
    projected: list[Literal] = []
    projected_effects: set[Literal] = set()
    for proxy, options in witnesses.items():
        if not options:
            return (), (), frozenset()
        best = max(score for score, _, _ in options)
        selected = [item for item in options if item[0] == best]
        owners = {owner for _, owner, _ in selected}
        if len(owners) != 1:
            return (), (), frozenset()
        common_effects = set(selected[0][2])
        for _, _, effects in selected[1:]:
            common_effects.intersection_update(effects)
        if not common_effects:
            return (), (), frozenset()
        projected.append(next(iter(owners)))
        projected_effects.update(common_effects)
    return (
        tuple(sorted(projected)),
        tuple(sorted(projected_effects)),
        frozenset(proxy_states),
    )


def _schema_backed_transition_preconditions(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
    object_mapping: dict[str, str],
    family: ActionFamily,
) -> tuple[Literal, ...]:
    """Recover physical guards from GT schemas that produce the same effect.

    Candidate domains often preserve the requested result while dropping the
    tool, support, or carrier needed to produce it.  A GT schema is evidence
    only when one of its non-hand add effects grounds to an explicit candidate
    add effect.  Ambiguous compatible schemas contribute only their common
    preconditions.
    """
    if reference_world is None or reference_world.schemas is None:
        return ()
    candidate_pre = _translated_dynamic_literals(
        action.pre_pos, candidate_world, object_mapping
    )
    candidate_add = _translated_dynamic_literals(
        action.add_eff, candidate_world, object_mapping
    )
    candidate_delete = _translated_dynamic_literals(
        action.del_eff, candidate_world, object_mapping
    )
    mapped_participants = {
        object_mapping[argument]
        for argument in action.args
        if argument in object_mapping
    }
    core_add = {
        item for item in candidate_add
        if item[0] not in {"hand_free", "holding"}
    }
    if not core_add:
        return ()
    intended = _semantic_name_family(action)
    if intended is None and family is ActionFamily.GENERIC_PDDL:
        return ()
    witnesses: list[tuple[int, frozenset[Literal]]] = []
    for schema in reference_world.schemas.values():
        schema_family = map_schema(schema)[0]
        if intended is not None:
            if schema_family is not intended:
                continue
        elif family is not ActionFamily.GENERIC_PDDL and schema_family is not family:
            continue
        schema_pre = {
            reference_world.interface.canonicalize(item)
            for item in schema.pre_pos
        }
        schema_add = {
            reference_world.interface.canonicalize(item)
            for item in schema.add_eff
        }
        schema_delete = {
            reference_world.interface.canonicalize(item)
            for item in schema.del_eff
        }
        anchors = [
            (pattern, grounded)
            for pattern in schema_add
            if not pattern[0].startswith(("kind:", "static:"))
            and pattern[0] not in {"hand_free", "holding"}
            for grounded in core_add
            if pattern[0] == grounded[0] and len(pattern) == len(grounded)
        ]
        seeds: list[tuple[dict[str, str], int]] = []
        for pattern, grounded in anchors:
            seed = _unify_schema_literal(pattern, grounded, {})
            if seed is not None:
                seeds.append((seed, 0))

        participant_seed: dict[str, str] = {}
        for variable in schema.params:
            expected = _schema_parameter_kinds(reference_world, schema, variable)
            if not expected:
                continue
            matches = [
                name
                for name in mapped_participants
                if (
                    (description := reference_world.object(name)) is not None
                    and expected & set(description.kinds)
                )
            ]
            if len(matches) == 1:
                participant_seed[variable] = matches[0]
        if len(set(participant_seed.values())) >= 2:
            seeds.append((participant_seed, len(set(participant_seed.values()))))

        for seed, participant_overlap in seeds:
            for bindings in _complete_schema_bindings(reference_world, schema, seed):
                grounded_pre = _ground_schema_preconditions(reference_world, schema_pre, bindings)
                if grounded_pre is None:
                    continue
                grounded_add = {
                    grounded_item
                    for item in schema_add
                    if not item[0].startswith(("kind:", "static:"))
                    for grounded_item in (_ground_schema_literal(item, bindings),)
                    if grounded_item is not None
                }
                if schema_family is ActionFamily.WIPE:
                    semantic_targets = {
                        item[1]
                        for item in grounded_add
                        if len(item) == 2 and item[0] not in STRUCTURAL_PREDICATES
                    }
                    local_tools = {
                        item[1]
                        for item in candidate_pre
                        if len(item) == 3
                        and item[0] in {"on", "in"}
                        and item[2] in semantic_targets
                    }
                    grounded_pre = frozenset(
                        item
                        for item in grounded_pre
                        if not (
                            len(item) == 3
                            and item[0] == "holding"
                            and item[2] in local_tools
                            and ("hand_free", item[1]) in candidate_pre
                        )
                    )
                grounded_delete = {
                    grounded_item
                    for item in schema_delete
                    if not item[0].startswith(("kind:", "static:"))
                    for grounded_item in (_ground_schema_literal(item, bindings),)
                    if grounded_item is not None
                }
                exact_effect_overlap = len(grounded_add & core_add)
                predicate_effect_overlap = len(
                    {item[0] for item in grounded_add}
                    & {item[0] for item in core_add}
                )
                state_family = family in {
                    ActionFamily.TURN_ON,
                    ActionFamily.TURN_OFF,
                    ActionFamily.LOCK,
                    ActionFamily.UNLOCK,
                    ActionFamily.OPEN,
                    ActionFamily.CLOSE,
                }
                if not exact_effect_overlap and not (
                    state_family
                    and participant_overlap >= 2
                    and predicate_effect_overlap
                ):
                    continue
                score = (
                    8 * exact_effect_overlap
                    + 4 * predicate_effect_overlap
                    + 3 * len(grounded_delete & candidate_delete)
                    + 2 * len(grounded_pre & candidate_pre)
                    + participant_overlap
                )
                witnesses.append((score, grounded_pre))
    if not witnesses:
        return ()
    best = max(score for score, _ in witnesses)
    selected = [preconditions for score, preconditions in witnesses if score == best]
    common = set(selected[0])
    for preconditions in selected[1:]:
        common.intersection_update(preconditions)
    state_targets = {
        item[1]
        for item in core_add
        if len(item) == 2 and item[0] in STRUCTURAL_PREDICATES
    }
    state_target_kinds = {
        token
        for target in state_targets
        for description in (reference_world.object(target),)
        if description is not None
        for kind in description.kinds
        for token in kind.split("_")
    }

    def justified_physical_guard(item: Literal) -> bool:
        if item in candidate_add:
            return False
        if len(item) == 3 and item[0] == "holding":
            return item[1] in mapped_participants or item[2] in mapped_participants
        if len(item) == 3 and item[0] in {"on", "in", "inserted"}:
            return item[2] in mapped_participants and not any(
                (pre[0] == "holding" and len(pre) == 3 and pre[2] == item[1])
                or (
                    pre[0] in LOCATION_PREDICATES
                    and len(pre) == 3
                    and pre[1:] == item[1:]
                    and pre[0] != item[0]
                )
                for pre in candidate_pre
            )
        if (
            len(item) != 2
            or item[0] not in STRUCTURAL_PREDICATES
            or item[0] == "hand_free"
        ):
            return False
        if item[0] in {"locked", "unlocked"}:
            return True
        subject = item[1]
        if subject in state_targets:
            return True
        description = reference_world.object(subject)
        subject_kinds = {
            token
            for kind in (() if description is None else description.kinds)
            for token in kind.split("_")
        }
        # Peer-state guards are physical when they constrain another component
        # of the same kind (for example, drawer interlocks).  A state on an
        # unrelated device is a workflow ordering guard, not a precondition of
        # the candidate transition itself.
        if state_target_kinds & subject_kinds:
            return family is ActionFamily.OPEN and item[0] == "closed"
        state_family = family in {
            ActionFamily.TURN_ON,
            ActionFamily.TURN_OFF,
            ActionFamily.LOCK,
            ActionFamily.UNLOCK,
            ActionFamily.OPEN,
            ActionFamily.CLOSE,
        }
        return not state_family and subject in mapped_participants

    return tuple(sorted(item for item in common if justified_physical_guard(item)))


def _schema_backed_compound_semantics(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
    object_mapping: dict[str, str],
) -> tuple[tuple[Literal, ...], tuple[Literal, ...]]:
    """Recover an omitted operation result from compound tool evidence.

    The action name selects an operation family but is never sufficient by
    itself.  Projection additionally requires a uniquely grounded GT schema,
    at least two matching dynamic preconditions, and a held tool distinct from
    the schema's semantic result subject.  This admits compound wipe/cut plus
    release actions while keeping ordinary named placement strict.
    """
    intended = _semantic_name_family(action)
    if intended is None or reference_world is None or reference_world.schemas is None:
        return (), ()
    candidate_pre = _translated_dynamic_literals(
        action.pre_pos, candidate_world, object_mapping
    )
    candidate_add = _translated_dynamic_literals(
        action.add_eff, candidate_world, object_mapping
    )
    candidate_delete = _translated_dynamic_literals(
        action.del_eff, candidate_world, object_mapping
    )
    structural_witnesses: list[tuple[int, Literal, frozenset[Literal]]] = []
    for schema in reference_world.schemas.values():
        if not _schema_matches_semantic_family(schema, intended):
            continue
        schema_pre = {
            reference_world.interface.canonicalize(item) for item in schema.pre_pos
        }
        schema_add = {
            reference_world.interface.canonicalize(item) for item in schema.add_eff
        }
        schema_delete = {
            reference_world.interface.canonicalize(item) for item in schema.del_eff
        }
        semantic_effects = [
            item for item in schema_add
            if len(item) == 2
            and item[0] not in STRUCTURAL_PREDICATES
            and not item[0].startswith(("kind:", "static:"))
        ]
        anchors = [
            (pattern, grounded)
            for pattern in schema_add | schema_delete | schema_pre
            if not pattern[0].startswith(("kind:", "static:"))
            for grounded in candidate_add | candidate_delete | candidate_pre
            if pattern[0] == grounded[0] and len(pattern) == len(grounded)
        ]
        for pattern, grounded in anchors:
            seed = _unify_schema_literal(pattern, grounded, {})
            if seed is None:
                continue
            for bindings in _complete_schema_bindings(reference_world, schema, seed):
                grounded_pre = _ground_schema_preconditions(reference_world, schema_pre, bindings)
                if grounded_pre is None:
                    continue
                grounded_add = {
                    grounded_item
                    for item in schema_add
                    if not item[0].startswith(("kind:", "static:"))
                    for grounded_item in (_ground_schema_literal(item, bindings),)
                    if grounded_item is not None
                }
                grounded_delete = {
                    grounded_item
                    for item in schema_delete
                    if not item[0].startswith(("kind:", "static:"))
                    for grounded_item in (_ground_schema_literal(item, bindings),)
                    if grounded_item is not None
                }
                structural_overlap = (
                    len((grounded_add & candidate_add) - set(semantic_effects))
                    + len(grounded_delete & candidate_delete)
                )
                pre_overlap = len(grounded_pre & candidate_pre)
                if structural_overlap < 3 or pre_overlap < 1:
                    continue
                for effect in semantic_effects:
                    grounded_effect = _ground_schema_literal(effect, bindings)
                    if grounded_effect is not None:
                        structural_witnesses.append(
                            (3 * structural_overlap + pre_overlap, grounded_effect, grounded_pre)
                        )
    if structural_witnesses:
        best = max(score for score, _, _ in structural_witnesses)
        selected = [item for item in structural_witnesses if item[0] == best]
        effects = {effect for _, effect, _ in selected}
        if len(effects) == 1:
            precondition_sets = [preconditions for _, _, preconditions in selected]
            common = set(precondition_sets[0])
            for preconditions in precondition_sets[1:]:
                common.intersection_update(preconditions)
            return (next(iter(effects)),), tuple(sorted(common))

    witnesses: list[tuple[int, Literal, frozenset[Literal]]] = []
    for schema in reference_world.schemas.values():
        if not _schema_matches_semantic_family(schema, intended):
            continue
        schema_pre = {
            reference_world.interface.canonicalize(item)
            for item in schema.pre_pos
        }
        schema_add = {
            reference_world.interface.canonicalize(item)
            for item in schema.add_eff
        }
        holding = next(
            (
                item for item in schema_pre
                if len(item) == 3 and item[0] == "holding"
            ),
            None,
        )
        effects = [
            item for item in schema_add
            if len(item) == 2
            and item[0] not in STRUCTURAL_PREDICATES
            and not item[0].startswith(("kind:", "static:"))
        ]
        if holding is None or not effects:
            continue
        seeds: list[dict[str, str]] = [{}]
        for pattern in schema_pre:
            if pattern[0].startswith(("kind:", "static:")):
                continue
            matches = [
                grounded for grounded in candidate_pre
                if grounded[0] == pattern[0] and len(grounded) == len(pattern)
            ]
            expanded = []
            for seed in seeds:
                expanded.extend(
                    binding
                    for grounded in matches
                    for binding in (_unify_schema_literal(pattern, grounded, seed),)
                    if binding is not None
                )
            if expanded:
                seeds.extend(expanded)
        for seed in seeds:
            for bindings in _complete_schema_bindings(reference_world, schema, seed):
                grounded_pre = _ground_schema_preconditions(reference_world, schema_pre, bindings)
                if grounded_pre is None:
                    continue
                overlap = len(grounded_pre & candidate_pre)
                grounded_holding = _ground_schema_literal(holding, bindings)
                if overlap < 2 or grounded_holding not in candidate_pre:
                    continue
                for effect in effects:
                    grounded_effect = _ground_schema_literal(effect, bindings)
                    if grounded_effect is None or grounded_effect[1] == grounded_holding[2]:
                        continue
                    witnesses.append((overlap, grounded_effect, grounded_pre))
    if not witnesses:
        return (), ()
    best = max(score for score, _, _ in witnesses)
    selected = {
        (effect, preconditions)
        for score, effect, preconditions in witnesses
        if score == best
    }
    effects = {effect for effect, _ in selected}
    if len(effects) != 1:
        return (), ()
    precondition_sets = [preconditions for _, preconditions in selected]
    common = set(precondition_sets[0])
    for preconditions in precondition_sets[1:]:
        common.intersection_update(preconditions)
    return (next(iter(effects)),), tuple(sorted(common))


def _schema_backed_semantic_marker_projection(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
    object_mapping: dict[str, str],
    family: ActionFamily,
) -> tuple[tuple[Literal, ...], tuple[Literal, ...], tuple[Literal, ...]]:
    """Recover a semantic result only from an otherwise non-physical marker.

    A malformed candidate sometimes represents an operation by adding a second,
    mutually exclusive location for an object that remains on the same support.
    The extra relation cannot be replayed as motion.  It may be treated as an
    operation marker only when a GT schema of the same family has the same
    subject/support and held-tool signature and declares a unary persistent
    result for that subject.

    The returned tuples are projected semantic effects, translated marker
    relations to suppress, and grounded GT preconditions justified by the
    matched schema.  A normal placement/release transition never satisfies this
    contract, so an action name alone cannot manufacture a semantic result.
    """
    if reference_world is None or reference_world.schemas is None:
        return (), (), ()

    candidate_pre = {
        candidate_world.interface.canonicalize(item) for item in action.pre_pos
    }
    candidate_add = {
        candidate_world.interface.canonicalize(item) for item in action.add_eff
    }
    candidate_delete = {
        candidate_world.interface.canonicalize(item) for item in action.del_eff
    }
    held = [
        item for item in candidate_pre
        if len(item) == 3 and item[0] == "holding"
    ]
    markers: list[tuple[Literal, Literal]] = []
    for added in candidate_add:
        if len(added) != 3 or added[0] not in LOCATION_PREDICATES:
            continue
        prior = next(
            (
                item for item in candidate_pre
                if len(item) == 3
                and item[0] in LOCATION_PREDICATES
                and item[0] != added[0]
                and item[1:] == added[1:]
                and item not in candidate_delete
            ),
            None,
        )
        if prior is not None:
            markers.append((prior, added))
    if not markers or not held:
        return (), (), ()

    supported = {
        item.canonical_name
        for item in reference_world.interface.descriptions
        if not item.static
    }
    witnesses: set[tuple[Literal, Literal, tuple[Literal, ...]]] = set()
    for prior, marker in markers:
        subject, context = prior[1:]
        mapped_subject = object_mapping.get(subject)
        mapped_context = object_mapping.get(context)
        if mapped_subject is None or mapped_context is None:
            continue
        if any(
            len(item) == 3
            and item[0] == "holding"
            and item[2] == subject
            for item in candidate_add | candidate_delete
        ):
            continue
        for candidate_holding in held:
            hand, tool = candidate_holding[1:]
            if tool == subject:
                continue
            mapped_hand = object_mapping.get(hand)
            mapped_tool = object_mapping.get(tool)
            if mapped_hand is None or mapped_tool is None:
                continue
            for schema in reference_world.schemas.values():
                if map_schema(schema)[0] is not family:
                    continue
                schema_pre = {
                    reference_world.interface.canonicalize(item)
                    for item in schema.pre_pos
                }
                schema_add = {
                    reference_world.interface.canonicalize(item)
                    for item in schema.add_eff
                }
                for effect in schema_add:
                    if (
                        len(effect) != 2
                        or effect[0] in STRUCTURAL_PREDICATES
                        or effect[0] not in supported
                    ):
                        continue
                    subject_variable = effect[1]
                    location = next(
                        (
                            item for item in schema_pre
                            if len(item) == 3
                            and item[0] == prior[0]
                            and item[1] == subject_variable
                        ),
                        None,
                    )
                    schema_holding = next(
                        (
                            item for item in schema_pre
                            if len(item) == 3
                            and item[0] == "holding"
                            and item[2] != subject_variable
                        ),
                        None,
                    )
                    if location is None or schema_holding is None:
                        continue
                    bindings = {
                        subject_variable: mapped_subject,
                        location[2]: mapped_context,
                        schema_holding[1]: mapped_hand,
                        schema_holding[2]: mapped_tool,
                    }
                    grounded_preconditions: set[Literal] = set()
                    valid = True
                    for precondition in schema_pre:
                        arguments = precondition[1:]
                        if not all(argument in bindings for argument in arguments):
                            continue
                        grounded = (
                            precondition[0],
                            *(bindings[argument] for argument in arguments),
                        )
                        if precondition[0].startswith(("kind:", "static:")):
                            if grounded not in reference_world.facts:
                                valid = False
                                break
                        else:
                            grounded_preconditions.add(grounded)
                    if valid:
                        translated_marker = (
                            marker[0], mapped_subject, mapped_context
                        )
                        witnesses.add(
                            (
                                (effect[0], mapped_subject),
                                translated_marker,
                                tuple(sorted(grounded_preconditions)),
                            )
                        )

    if len(witnesses) != 1:
        return (), (), ()
    semantic_effect, marker, preconditions = next(iter(witnesses))
    return (semantic_effect,), (marker,), preconditions


def _project_enclosing_semantic_effects(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
    object_mapping: dict[str, str],
    family: ActionFamily,
) -> tuple[Literal, ...]:
    """Project an explicit latent-object state onto the GT aggregate object.

    Independently generated domains may model a payload that the GT deliberately
    omits, such as ``food in food_box``.  Projection is allowed only when the
    candidate explicitly adds the semantic state to that payload and a
    same-family GT schema adds the same state to the mapped enclosing object under
    the same outer relation.  Static GT kinds must also match the proposed object.
    """
    if reference_world is None or reference_world.schemas is None:
        return ()
    candidate_pre = {
        candidate_world.interface.canonicalize(item) for item in action.pre_pos
    }
    candidate_add = {
        candidate_world.interface.canonicalize(item) for item in action.add_eff
    }
    projected: set[Literal] = set()
    for effect in candidate_add:
        if (
            len(effect) != 2
            or effect[0] in STRUCTURAL_PREDICATES
        ):
            continue
        payload = effect[1]
        enclosures = {
            item[2]
            for item in candidate_pre
            if len(item) == 3
            and item[0] == "in"
            and item[1] == payload
            and item[2] in object_mapping
        }
        for enclosure in enclosures:
            mapped_enclosure = object_mapping[enclosure]
            outer_relations = {
                item
                for item in candidate_pre
                if len(item) == 3
                and item[1] == enclosure
                and item[2] in object_mapping
            }
            for schema in reference_world.schemas.values():
                if map_schema(schema)[0] is not family:
                    continue
                schema_pre = {
                    reference_world.interface.canonicalize(item)
                    for item in schema.pre_pos
                }
                schema_add = {
                    reference_world.interface.canonicalize(item)
                    for item in schema.add_eff
                }
                for reference_effect in schema_add:
                    if len(reference_effect) != 2 or reference_effect[0] != effect[0]:
                        continue
                    subject_variable = reference_effect[1]
                    for outer in outer_relations:
                        anchor = next(
                            (
                                item for item in schema_pre
                                if len(item) == 3
                                and item[0] == outer[0]
                                and item[1] == subject_variable
                            ),
                            None,
                        )
                        if anchor is None:
                            continue
                        bindings = {
                            subject_variable: mapped_enclosure,
                            anchor[2]: object_mapping[outer[2]],
                        }
                        if all(
                            grounded in reference_world.facts
                            for grounded in (
                                (
                                    item[0],
                                    *(bindings[argument] for argument in item[1:]),
                                )
                                for item in schema_pre
                                if item[0].startswith(("kind:", "static:"))
                                and all(argument in bindings for argument in item[1:])
                            )
                        ):
                            projected.add((effect[0], mapped_enclosure))
    return tuple(sorted(projected))


def _project_held_tool_wipe(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
    object_mapping: dict[str, str],
) -> tuple[Literal, ...]:
    """Recognize a held tool wiping a surface when GT exposes that operation."""
    if reference_world is None or reference_world.schemas is None:
        return ()
    if "wipe" not in action.name.split("_"):
        return ()
    candidate_pre = {
        candidate_world.interface.canonicalize(item) for item in action.pre_pos
    }
    candidate_add = {
        candidate_world.interface.canonicalize(item) for item in action.add_eff
    }
    held = {item[2] for item in candidate_pre if len(item) == 3 and item[0] == "holding"}
    projected: set[Literal] = set()
    for effect in candidate_add:
        if len(effect) != 3 or effect[1] not in held:
            continue
        if effect[0] not in {"on", "wipe"}:
            continue
        tool = object_mapping.get(effect[1])
        surface = object_mapping.get(effect[2])
        if tool is None or surface is None:
            continue
        tool_description = reference_world.object(tool)
        if tool_description is None:
            continue
        for schema in reference_world.schemas.values():
            if map_schema(schema)[0] is not ActionFamily.WIPE:
                continue
            schema_pre = {
                reference_world.interface.canonicalize(item) for item in schema.pre_pos
            }
            schema_add = {
                reference_world.interface.canonicalize(item) for item in schema.add_eff
            }
            for result in schema_add:
                if len(result) != 2 or result[0] != "clean":
                    continue
                if not any(
                    len(item) == 3
                    and ((item[0] == "on" and item[2] == result[1])
                         or item[0] == "holding")
                    and (item[1] if item[0] == "on" else item[2]) != result[1]
                    and (not _schema_parameter_kinds(
                        reference_world, schema,
                        item[1] if item[0] == "on" else item[2],
                    ) or _schema_parameter_kinds(
                        reference_world, schema,
                        item[1] if item[0] == "on" else item[2],
                    ) & set(tool_description.kinds))
                    for item in schema_pre
                ):
                    continue
                if not any(
                    item[0].startswith(("kind:", "static:")) and result[1] in item[1:]
                    and (item[0], surface) in reference_world.facts
                    for item in schema_pre
                ):
                    continue
                projected.add(("clean", surface))
    return tuple(sorted(projected))


def _schema_backed_pose_effects(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    reference_world: CanonicalWorld | None,
    object_mapping: dict[str, str],
    family: ActionFamily,
) -> tuple[Literal, ...]:
    """Project a pose only when a matching GT placement schema proves it.

    ``upright`` and ``vertical`` are not global aliases.  They may correspond
    for one transition when candidate and GT schemas place the same subject on
    or in the same target, and either declare the corresponding subject pose or
    use the target pose to determine the placed object's pose.
    """
    if reference_world is None or reference_world.schemas is None:
        return ()

    def compatible(left: str, right: str) -> bool:
        return left == right or {left, right} <= {"upright", "vertical"}

    candidate_pre = {
        candidate_world.interface.canonicalize(item) for item in action.pre_pos
    }
    candidate_add = {
        candidate_world.interface.canonicalize(item) for item in action.add_eff
    }
    placements = [
        item
        for item in candidate_add
        if len(item) == 3 and item[0] in {"on", "in", "inserted"}
    ]
    witnesses: set[Literal] = set()
    for placement in placements:
        subject, target = placement[1:]
        mapped_subject = object_mapping.get(subject)
        mapped_target = object_mapping.get(target)
        if mapped_subject is None or mapped_target is None:
            continue
        for schema in reference_world.schemas.values():
            if map_schema(schema)[0] is not family:
                continue
            schema_pre = {
                reference_world.interface.canonicalize(item)
                for item in schema.pre_pos
            }
            schema_add = {
                reference_world.interface.canonicalize(item)
                for item in schema.add_eff
            }
            schema_placement = next(
                (
                    item
                    for item in schema_add
                    if len(item) == 3 and item[0] == placement[0]
                ),
                None,
            )
            if schema_placement is None:
                continue
            subject_variable, target_variable = schema_placement[1:]
            bindings = {
                subject_variable: mapped_subject,
                target_variable: mapped_target,
            }
            if any(
                (
                    item[0],
                    *(bindings[argument] for argument in item[1:]),
                )
                not in reference_world.facts
                for item in schema_pre
                if item[0].startswith(("kind:", "static:"))
                and all(argument in bindings for argument in item[1:])
            ):
                continue
            for effect in schema_add:
                if (
                    len(effect) != 2
                    or effect[0] not in POSE_PREDICATES
                    or effect[1] not in bindings
                ):
                    continue
                candidate_object = subject if effect[1] == subject_variable else target
                direct_pose = any(
                    len(item) == 2
                    and item[1] == candidate_object
                    and item[0] in POSE_PREDICATES
                    and compatible(item[0], effect[0])
                    for item in candidate_add
                )
                target_pose = (
                    effect[1] == subject_variable
                    and any(
                        len(item) == 2
                        and item[1] == target
                        and item[0] in POSE_PREDICATES
                        and any(
                            len(required) == 2
                            and required[1] == target_variable
                            and required[0] in POSE_PREDICATES
                            and compatible(item[0], required[0])
                            for required in schema_pre
                        )
                        for item in candidate_pre | candidate_add
                    )
                )
                if direct_pose or target_pose:
                    witnesses.add((effect[0], bindings[effect[1]]))
    return tuple(witnesses) if len(witnesses) == 1 else ()


def _has_reviewed_name_semantics(action: GroundAction) -> bool:
    """Whether generic replay has a deterministic adapter for this action."""
    tokens = set(action.name.lower().split("_"))
    return bool(tokens & {"wash", "washed", "rinse", "rinsed", "wet", "fill", "filled"})


def _translate_literal(
    literal: Literal,
    world: CanonicalWorld,
    object_mapping: dict[str, str],
) -> tuple[Literal | None, tuple[str, ...]]:
    canonical = world.interface.canonicalize(literal)
    translated = [canonical[0]]
    missing = []
    for argument in canonical[1:]:
        if argument not in object_mapping:
            missing.append(argument)
        else:
            translated.append(object_mapping[argument])
    return (tuple(translated) if not missing else None), tuple(missing)


def compile_step(
    action: GroundAction,
    candidate_world: CanonicalWorld,
    object_mapping: dict[str, str],
    *,
    reference_world: CanonicalWorld | None = None,
    ambiguous_objects: set[str] | None = None,
    trusted_declared_semantics: bool = False,
) -> CanonicalStep:
    canonical_action = map_ground_action(action)
    family = canonical_action.family
    if reference_world is not None:
        # A fixture's lock state and key slot belong to its image-verified
        # lock owner. Ordinary placement on the fixture keeps its identity.
        lock_targets = {
            item[1]
            for literal in action.pre_pos | action.add_eff | action.del_eff
            for item in (candidate_world.interface.canonicalize(literal),)
            if len(item) == 2 and item[0] in {"locked", "unlocked"}
            and family in {ActionFamily.LOCK, ActionFamily.UNLOCK}
        }
        lock_targets.update(
            item[2]
            for literal in action.pre_pos | action.add_eff | action.del_eff
            for item in (candidate_world.interface.canonicalize(literal),)
            if len(item) == 3 and item[0] == "inserted"
            and (payload := candidate_world.object(item[1])) is not None
            and "key" in payload.kinds
        )
        overrides = {}
        for target in lock_targets:
            owners = {
                item[2] for item in reference_world.facts
                if len(item) == 3 and item[0] == "lock_owner"
                and item[1] == object_mapping.get(target)
            }
            if len(owners) == 1:
                overrides[target] = next(iter(owners))
        object_mapping = object_mapping | overrides
    closure_relation = next(
        (item for item in action.del_eff if item[0] == "on" and len(item) == 3),
        None,
    )
    if (
        family is ActionFamily.OPEN
        and closure_relation is not None
        and action.name.startswith(("remove_", "unscrew_", "pick_lid_"))
    ):
        family = ActionFamily.REMOVE_CLOSURE
        canonical_action = map_ground_action(action, family)
    local_roles = dict(canonical_action.roles)
    (
        state_proxy_effects,
        state_proxy_preconditions,
        state_proxy_objects,
        state_proxy_target,
    ) = _schema_backed_state_proxy_projection(
        action,
        candidate_world,
        reference_world,
        object_mapping,
        family,
    )
    read_proxy_preconditions, read_proxy_effects, read_proxy_objects = (
        _schema_backed_read_state_proxy_projection(
            action,
            candidate_world,
            reference_world,
            object_mapping,
        )
    )
    required_objects = (
        set(action.args)
        if trusted_declared_semantics
        else required_action_objects(action, candidate_world, reference_world)
    ) - set(state_proxy_objects) - set(read_proxy_objects)
    ignored_objects = {
        item
        for item in action.args
        if item not in required_objects and item not in object_mapping
    } | {
        item
        for item in state_proxy_objects | read_proxy_objects
        if item not in object_mapping
    }
    projected_state_owners = state_proxy_objects | read_proxy_objects
    if "hand" not in local_roles:
        hand_candidates = [
            argument
            for argument in action.args
            if (description := candidate_world.object(argument)) is not None
            and "hand" in description.kinds
        ]
        if len(hand_candidates) == 1:
            local_roles["hand"] = hand_candidates[0]
    role_missing: set[str] = set()
    literal_missing: set[str] = set()
    roles = []
    for role, value in sorted(local_roles.items()):
        if role == "target" and state_proxy_target is not None:
            roles.append((role, state_proxy_target))
            continue
        mapped_object = object_mapping.get(value)
        if mapped_object is None:
            role_missing.add(value)
        else:
            roles.append((role, mapped_object))

    def translate(items: set[Literal]) -> tuple[Literal, ...]:
        result = []
        for item in sorted(items):
            canonical = candidate_world.interface.canonicalize(item)
            if candidate_world.interface.is_static(item[0]):
                continue
            if (
                len(canonical) == 2
                and canonical[0] in {"power_on", "power_off", "locked", "unlocked"}
                and canonical[1] in projected_state_owners
            ):
                continue
            if any(argument in ignored_objects for argument in canonical[1:]):
                continue
            translated, absent = _translate_literal(item, candidate_world, object_mapping)
            literal_missing.update(absent)
            if translated is not None and not translated[0].startswith("kind:"):
                result.append(translated)
        return tuple(result)

    placement = next(
        (
            item[0]
            for item in action.add_eff
            if item[0]
            in {"on", "in", "inserted", "under", "against", "away_from", "in_front_of"}
        ),
        None,
    )
    mapped_objects = tuple(
        sorted({object_mapping[item] for item in action.args if item in object_mapping})
    )
    semantic_effects = []
    semantic_effects.extend(state_proxy_effects)
    semantic_effects.extend(read_proxy_effects)
    semantic_effects.extend(
        abstract_liquid_effects(
            action,
            candidate_world,
            reference_world,
            object_mapping,
        )
    )
    semantic_effects.extend(
        _project_declared_semantic_effects(
            action,
            candidate_world,
            reference_world,
            object_mapping,
        )
    )
    semantic_effects.extend(
        _project_enclosing_semantic_effects(
            action,
            candidate_world,
            reference_world,
            object_mapping,
            family,
        )
    )
    semantic_effects.extend(
        _project_held_tool_wipe(action, candidate_world, reference_world, object_mapping)
    )
    semantic_effects.extend(
        _schema_backed_pose_effects(
            action,
            candidate_world,
            reference_world,
            object_mapping,
            family,
        )
    )
    schema_preconditions = set(
        _schema_backed_transition_preconditions(
            action,
            candidate_world,
            reference_world,
            object_mapping,
            family,
        )
    )
    schema_preconditions.update(state_proxy_preconditions)
    schema_preconditions.update(read_proxy_preconditions)
    compound_effects, compound_preconditions = _schema_backed_compound_semantics(
        action,
        candidate_world,
        reference_world,
        object_mapping,
    )
    semantic_effects.extend(compound_effects)
    schema_preconditions.update(compound_preconditions)
    marker_effects, ignored_add_effects, marker_preconditions = (
        _schema_backed_semantic_marker_projection(
            action,
            candidate_world,
            reference_world,
            object_mapping,
            family,
        )
    )
    semantic_effects.extend(marker_effects)
    schema_preconditions.update(marker_preconditions)
    supported = (
        {
            item.canonical_name
            for item in reference_world.interface.descriptions
        }
        if reference_world is not None
        else set()
    )
    for item in action.add_eff:
        canonical = candidate_world.interface.canonicalize(item)
        if len(canonical) != 2 or canonical[0] in STRUCTURAL_PREDICATES:
            continue
        mapped_object = object_mapping.get(canonical[1])
        if mapped_object is None:
            continue
        predicate = canonical[0]
        if predicate in supported:
            semantic_effects.append((predicate, mapped_object))
            continue
        # Some candidate domains encode a wet tool as ``contains_water``.
        # Preserve that abstraction only when the GT exposes ``wet`` and the
        # mapped object has no container category.
        if predicate != "contains_water" or "wet" not in supported:
            continue
        description = reference_world.object(mapped_object) if reference_world is not None else None
        kind_tokens = set()
        if description is not None:
            for kind in description.kinds:
                kind_tokens.update(kind.split("_"))
        if not kind_tokens & {
            "container", "basin", "bin", "bottle", "bowl", "box", "bucket",
            "cabinet", "can", "carton", "cup", "drawer", "glass", "kettle",
            "mug", "pan", "pot", "sink", "thermos", "tray",
        }:
            semantic_effects.append(("wet", mapped_object))
    unresolved = required_objects - set(object_mapping)
    ambiguous = unresolved & (ambiguous_objects or set())
    absent = unresolved - ambiguous
    positive_preconditions = tuple(
        sorted(set(translate(action.pre_pos)) | schema_preconditions)
    )
    add_effects = tuple(
        item
        for item in translate(action.add_eff)
        if item not in ignored_add_effects
    )
    return CanonicalStep(
        action.to_line(),
        family,
        tuple(roles),
        positive_preconditions,
        translate(action.pre_neg),
        add_effects,
        translate(action.del_eff),
        tuple(sorted(absent | role_missing | (literal_missing - ambiguous))),
        tuple(sorted(ambiguous)),
        mapped_objects,
        placement,
        trusted_declared_semantics,
        tuple(sorted(set(semantic_effects))),
    )
