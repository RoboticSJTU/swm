from __future__ import annotations

from swm.pddl.strips import ActionSchema, GroundAction

from swm.simulator.predicate_aliases import canonical_literal

from .model import ActionFamily, CanonicalAction, Provenance


_SEMANTIC_EFFECT_FAMILIES = (
    ("wet", ActionFamily.WET),
    ("washed", ActionFamily.WASH),
    ("stirred", ActionFamily.STIR),
    ("cut", ActionFamily.CUT),
    ("folded", ActionFamily.FOLD),
    ("scrunched", ActionFamily.SCRUNCH),
    ("empty", ActionFamily.EMPTY),
    ("upright", ActionFamily.ORIENT),
    ("flat", ActionFamily.ORIENT),
    ("upside_down", ActionFamily.ORIENT),
    ("configured", ActionFamily.CONFIGURE),
    ("activated", ActionFamily.ACTIVATE),
    ("slid", ActionFamily.SLIDE),
    ("pushed", ActionFamily.PUSH),
    ("unblocked", ActionFamily.UNBLOCK),
)


def _canonical_literals(literals: set[tuple[str, ...]]) -> set[tuple[str, ...]]:
    return {canonical_literal(literal) for literal in literals}


def _family_from_transition(
    name: str,
    pre_pos: set[tuple[str, ...]],
    add_eff: set[tuple[str, ...]],
    del_eff: set[tuple[str, ...]],
) -> tuple[ActionFamily, str]:
    pre_pos = _canonical_literals(pre_pos)
    add_eff = _canonical_literals(add_eff)
    del_eff = _canonical_literals(del_eff)
    added = {literal[0] for literal in add_eff}
    deleted = {literal[0] for literal in del_eff}
    lower = name.lower()
    # A tool-mediated transfer can be represented as a pair of containment
    # changes while the hand keeps holding the tool.  It is not a normal
    # placement of the tool itself: the moved object is the first argument of
    # both relations and the held object is only the carrier.
    held_tools = {
        literal[2]
        for literal in pre_pos
        if len(literal) == 3 and literal[0] == "holding"
    }
    removed_contents = {
        literal[1]
        for literal in del_eff
        if len(literal) == 3 and literal[0] in {"in", "inserted"}
    }
    added_contents = {
        literal[1]
        for literal in add_eff
        if len(literal) == 3 and literal[0] in {"in", "inserted"}
    }
    if held_tools and removed_contents & added_contents - held_tools:
        return ActionFamily.SCOOP, "tool-mediated containment transfer"
    # A compound action may also pick up a tool or object. Prefer an explicit
    # semantic state transition over the incidental holding transition; replay
    # can then apply the declared state effect without consulting the name.
    if not added & {"on", "in", "inserted"}:
        for predicate, semantic_family in _SEMANTIC_EFFECT_FAMILIES:
            if predicate in added:
                return semantic_family, f"adds {predicate} state"
    if "holding" in added:
        if "closed" in deleted and ("open" in added or "on" in deleted):
            return ActionFamily.REMOVE_CLOSURE, "removes closure while opening vessel"
        if lower == "open" or lower.startswith("open_"):
            return ActionFamily.OPEN, "open macro acquires the operated object"
        return ActionFamily.PICK, "adds holding"
    if "holding" in deleted:
        if "closed" in added and "open" in deleted and "on" in added:
            return ActionFamily.REPLACE_CLOSURE, "releases closure and closes vessel"
        if "inserted" in added:
            return ActionFamily.INSERT, "releases object into inserted relation"
        if "in" in added:
            return ActionFamily.PLACE_IN, "releases object into containment"
        if "on" in added:
            return ActionFamily.PLACE_ON, "releases object onto support"
        if lower.startswith("release_"):
            return ActionFamily.PLACE_ON, "release transition"
    if "open" in added and "closed" in deleted:
        return ActionFamily.OPEN, "open/closed inverse transition"
    if "closed" in added and "open" in deleted:
        return ActionFamily.CLOSE, "closed/open inverse transition"
    if "power_on" in added or "on_state" in added or "turned_on" in added:
        return ActionFamily.TURN_ON, "adds powered-on state"
    if "power_off" in added or "off_state" in added or "turned_off" in added:
        return ActionFamily.TURN_OFF, "adds powered-off state"
    if "locked" in added and "unlocked" in deleted:
        return ActionFamily.LOCK, "locked/unlocked inverse transition"
    if "unlocked" in added and "locked" in deleted:
        return ActionFamily.UNLOCK, "unlocked/locked inverse transition"
    if "inserted" in added:
        return ActionFamily.INSERT, "adds inserted relation"
    for predicate, semantic_family in _SEMANTIC_EFFECT_FAMILIES:
        if predicate in added:
            return semantic_family, f"adds {predicate} state"
    if lower.startswith("pour_"):
        return ActionFamily.POUR, "pour transition name with content effects"
    if lower.startswith("scoop_"):
        return ActionFamily.SCOOP, "scoop transition name with content effects"
    semantic_names = (
        (("wipe", "clean"), ActionFamily.WIPE),
        (("stir", "mix"), ActionFamily.STIR),
        (("cut", "chop", "slice"), ActionFamily.CUT),
        (("wash", "washed", "rinse", "rinsed"), ActionFamily.WASH),
        (("fold",), ActionFamily.FOLD),
        (("scrunch", "scrunched", "crumple", "crumpled"), ActionFamily.SCRUNCH),
        (("fill", "filling"), ActionFamily.FILL),
        (("wet", "dampen"), ActionFamily.WET),
        (("empty", "drain"), ActionFamily.EMPTY),
        (("sweep",), ActionFamily.SWEEP),
        (("slide",), ActionFamily.SLIDE),
        (("push",), ActionFamily.PUSH),
        (("press", "activate"), ActionFamily.ACTIVATE),
        (("configure", "select"), ActionFamily.CONFIGURE),
    )
    for words, semantic_family in semantic_names:
        name_tokens = set(lower.split("_"))
        if any(word in name_tokens for word in words):
            return semantic_family, f"reviewed semantic name {semantic_family.value}"
    prefixes = (
        ("pick_", ActionFamily.PICK),
        ("grasp_", ActionFamily.PICK),
        ("take_", ActionFamily.PICK),
        ("lift_", ActionFamily.PICK),
        ("place_", ActionFamily.PLACE_ON),
        ("put_", ActionFamily.PLACE_ON),
        ("drop_", ActionFamily.PLACE_ON),
        ("open_", ActionFamily.OPEN),
        ("close_", ActionFamily.CLOSE),
        ("turn_on_", ActionFamily.TURN_ON),
        ("turn_off_", ActionFamily.TURN_OFF),
        ("switch_on_", ActionFamily.TURN_ON),
        ("switch_off_", ActionFamily.TURN_OFF),
        ("remove_", ActionFamily.REMOVE_CLOSURE),
        ("unscrew_", ActionFamily.REMOVE_CLOSURE),
        ("screw_", ActionFamily.REPLACE_CLOSURE),
        ("insert_", ActionFamily.INSERT),
        ("lock_", ActionFamily.LOCK),
        ("unlock_", ActionFamily.UNLOCK),
    )
    for prefix, family in prefixes:
        if lower == prefix[:-1] or lower.startswith(prefix):
            return family, f"name fallback {prefix}"
    if add_eff or del_eff or pre_pos:
        return ActionFamily.GENERIC_PDDL, "covered by explicit PDDL transition only"
    return ActionFamily.UNSUPPORTED, "no recognized transition semantics"


def map_schema(schema: ActionSchema) -> tuple[ActionFamily, str]:
    return _family_from_transition(
        schema.name, schema.pre_pos, schema.add_eff, schema.del_eff
    )


def _find_argument(
    literals: set[tuple[str, ...]], predicate: str, position: int
) -> str | None:
    values = sorted(
        literal[position]
        for literal in literals
        if literal[0] == predicate and len(literal) > position
    )
    return values[0] if values else None


def map_ground_action(
    action: GroundAction, family: ActionFamily | None = None
) -> CanonicalAction:
    detected_family, reason = _family_from_transition(
        action.name, action.pre_pos, action.add_eff, action.del_eff
    )
    family = family or detected_family
    all_literals = _canonical_literals(action.pre_pos | action.add_eff | action.del_eff)
    canonical_add = _canonical_literals(action.add_eff)
    canonical_del = _canonical_literals(action.del_eff)
    roles: dict[str, str] = {}
    holding_add = next((x for x in canonical_add if x[0] == "holding"), None)
    holding_del = next((x for x in canonical_del if x[0] == "holding"), None)
    holding_pre = next((x for x in all_literals if x[0] == "holding"), None)
    holding = holding_add or holding_del or holding_pre
    if holding and len(holding) >= 3:
        roles["hand"] = holding[1]
        roles["object"] = holding[2]
    else:
        for literal in sorted(action.pre_pos):
            if len(literal) == 2 and literal[0] == "hand":
                roles.setdefault("hand", literal[1])
    relation_candidates = [
        literal
        for literal in sorted(all_literals)
        if literal[0]
        in {"on", "in", "inserted", "under", "against", "away_from", "in_front_of"}
        and len(literal) >= 3
    ]
    for relation in relation_candidates:
        if roles.get("object") == relation[1] or family in {
            ActionFamily.PLACE_ON,
            ActionFamily.PLACE_IN,
            ActionFamily.INSERT,
        }:
            roles.setdefault("object", relation[1])
            roles.setdefault("target", relation[2])
    if family is ActionFamily.PICK:
        removed = [literal for literal in canonical_del if literal in relation_candidates]
        if removed:
            roles.setdefault("object", removed[0][1])
            roles["source"] = removed[0][2]
    state_predicates = {
        ActionFamily.OPEN: ("open", "closed"),
        ActionFamily.CLOSE: ("closed", "open"),
        ActionFamily.TURN_ON: ("power_on", "power_off"),
        ActionFamily.TURN_OFF: ("power_off", "power_on"),
        ActionFamily.LOCK: ("locked", "unlocked"),
        ActionFamily.UNLOCK: ("unlocked", "locked"),
    }.get(family, ("open", "closed", "power_on", "power_off", "locked", "unlocked"))
    state_target = None
    for predicate in state_predicates:
        state_target = _find_argument(all_literals, predicate, 1)
        if state_target:
            roles.setdefault("target", state_target)
            break
    if family in {ActionFamily.OPEN, ActionFamily.CLOSE} and state_target is None:
        operated = roles.get("object")
        if operated is not None:
            roles.setdefault("target", operated)
    if family in {ActionFamily.POUR, ActionFamily.SCOOP}:
        moved = [
            literal for literal in canonical_add if literal[0] in {"in", "has_water", "has_content"}
        ]
        removed = [
            literal for literal in canonical_del if literal[0] in {"in", "has_water", "has_content"}
        ]
        if removed:
            roles.setdefault("content", removed[0][1])
            if len(removed[0]) >= 3:
                roles.setdefault("source", removed[0][2])
            else:
                roles.setdefault("source", removed[0][1])
        if moved:
            roles.setdefault("content", moved[0][1])
            if len(moved[0]) >= 3:
                roles.setdefault("receiver", moved[0][2])
            else:
                roles.setdefault("receiver", moved[0][1])
        non_hand = [
            arg
            for arg in action.args
            if arg != roles.get("hand") and ("hand", arg) not in action.pre_pos
        ]
        if family is ActionFamily.SCOOP and len(non_hand) >= 2:
            roles.setdefault("source", non_hand[-2])
            roles.setdefault("tool", non_hand[-1])
        elif family is ActionFamily.POUR and len(non_hand) >= 2:
            roles.setdefault("source", non_hand[-2])
            roles.setdefault("receiver", non_hand[-1])
    if family in {ActionFamily.REMOVE_CLOSURE, ActionFamily.REPLACE_CLOSURE}:
        relation = next(
            (literal for literal in all_literals if literal[0] == "on" and len(literal) == 3),
            None,
        )
        if relation:
            roles.setdefault("object", relation[1])
            roles["closure"] = roles["object"]
            roles.setdefault("vessel", relation[2])
        else:
            non_hand = [
                arg
                for arg in action.args
                if arg != roles.get("hand") and ("hand", arg) not in action.pre_pos
            ]
            if non_hand:
                roles.setdefault("closure", non_hand[0])
            if len(non_hand) >= 2:
                roles.setdefault("vessel", non_hand[-1])
                roles.setdefault("target", non_hand[-1])
        if state_target:
            roles["vessel"] = state_target
    return CanonicalAction(
        family=family,
        raw_name=action.name,
        arguments=tuple(action.args),
        roles=tuple(sorted(roles.items())),
        provenance=(Provenance("pddl_action", action.name, action.to_line()),),
        mapping_reason=reason,
    )
