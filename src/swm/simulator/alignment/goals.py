from __future__ import annotations

import re
from dataclasses import dataclass

from swm.simulator.ir.model import ActionFamily

from .predicates import CanonicalWorld, Literal, normalized_tokens


@dataclass(frozen=True)
class InstructionGoal:
    positive: frozenset[Literal]
    negative: frozenset[Literal]
    provenance: tuple[tuple[Literal, str], ...]


_PREDICATE_WORDS = {
    "open": {"open"},
    "closed": {"close", "closed", "shut"},
    "power_on": {"start", "started"},
    "power_off": {"off"},
    "locked": {"lock", "locked"},
    "unlocked": {"unlock", "unlocked"},
    "inserted": {"insert", "plug"},
    "upright": {"upright"},
    "flat": {"flat"},
    "vertical": {"vertical", "vertically"},
    "clean": {"clean", "wash", "washed", "rinse", "rinsed", "wipe", "wiped"},
    "washed": {"wash", "washed", "rinse", "rinsed"},
    "cut": {"cut", "chop", "chopped"},
    "heated": {"heat", "heated", "warm", "warmed"},
    "boiled": {"boil", "boiled"},
    "mixed_water": {"mix", "mixed", "mixture"},
    "contains_hot_water": {"hot"},
    "contains_purified_water": {"purified", "cold"},
    "stirred": {"stir", "stirred"},
    "folded": {"fold", "folded"},
    "scrunched": {"scrunch", "scrunched", "crumple", "crumpled"},
    "slid": {"slide", "slid"},
    "activated": {"press", "pressed", "push", "pushed"},
    "empty": {"empty"},
}

_GENERIC_TOKENS = {
    "object", "thing", "hand", "robot", "left", "right", "front", "rear",
    "kitchen", "office", "room", "white", "black",
}


def _instruction_tokens(instruction: str) -> set[str]:
    words = set(normalized_tokens(instruction))
    words.update(
        word[:-1] for word in tuple(words) if len(word) > 3 and word.endswith("s")
    )
    return words


def _object_tokens(world: CanonicalWorld, name: str) -> set[str]:
    obj = world.object(name)
    if obj is None:
        return set(normalized_tokens(name))
    result = set(obj.name_tokens)
    for kind in obj.kinds:
        result.update(normalized_tokens(kind))
    return result


def _mentioned(world: CanonicalWorld, name: str, words: set[str]) -> bool:
    tokens = _object_tokens(world, name) - _GENERIC_TOKENS
    obj = world.object(name)
    kind_tokens = {
        token
        for kind in (() if obj is None else obj.kinds)
        for token in normalized_tokens(kind)
    } - _GENERIC_TOKENS
    if kind_tokens and not (kind_tokens & words):
        return False
    return bool(tokens & words)


def _predicate_relevant(predicate: str, words: set[str]) -> bool:
    if predicate == "power_on":
        return "start" in words or ({"turn", "on"} <= words)
    if predicate == "power_off":
        return "off" in words or "shut" in words
    expected = _PREDICATE_WORDS.get(predicate)
    return bool(expected and expected & words)


def _literal_relevant(literal: Literal, world: CanonicalWorld, words: set[str]) -> bool:
    predicate = literal[0]
    if predicate in {"holding", "hand_free", "clear", "configured", "unblocked"}:
        return False
    if predicate in {"on", "in", "inserted", "under", "against", "away_from", "in_front_of"}:
        if (
            predicate == "in"
            and "fill" in words
            and len(literal) == 3
            and "water" in _object_tokens(world, literal[1])
            and _mentioned(world, literal[2], words)
        ):
            return True
        return len(literal) == 3 and all(
            _mentioned(world, argument, words) for argument in literal[1:]
        )
    if len(literal) == 2:
        if (predicate == "power_off" and "close" in words
            and "faucet" in _object_tokens(world, literal[1])):
            return _mentioned(world, literal[1], words)
        return _predicate_relevant(predicate, words) and _mentioned(world, literal[1], words)
    return _predicate_relevant(predicate, words)


def _relation_relevant_by_clause(
    literal: Literal,
    world: CanonicalWorld,
    instruction: str,
) -> bool:
    if len(literal) != 3:
        return False
    clauses = [
        clause.strip()
        for clause in re.split(r"[,.;:]|\bthen\b|\band\b", instruction.lower())
        if clause.strip()
    ]
    previous_focus: str | None = None
    previous_referents: set[str] = set()
    component_tokens = {
        "base", "button", "cap", "door", "lid", "outlet", "plug", "spout", "tray",
    }
    for clause in clauses:
        words = _instruction_tokens(clause)
        if (literal[0] == "in" and "fill" in words
            and "water" in _object_tokens(world, literal[1])
            and _mentioned(world, literal[2], words)):
            return True
        explicit = [obj.name for obj in world.objects if _mentioned(world, obj.name, words)]
        mentioned = set(explicit)
        pronouns = set(normalized_tokens(clause)) & {"it", "them"}
        if previous_focus is not None and "it" in pronouns:
            mentioned.add(previous_focus)
        if "them" in pronouns:
            mentioned.update(previous_referents)
        if all(argument in mentioned for argument in literal[1:]):
            return True
        if explicit:
            clause_tokens = normalized_tokens(clause)

            def rank(name: str) -> tuple[int, int, int]:
                object_tokens = _object_tokens(world, name)
                positions = [
                    index for index, token in enumerate(clause_tokens) if token in object_tokens
                ]
                extras = object_tokens - words - _GENERIC_TOKENS
                return (
                    min(positions, default=len(clause_tokens)),
                    len(extras & component_tokens),
                    len(extras),
                )

            best_rank = min(rank(name) for name in explicit)
            previous_referents = {
                name for name in explicit if rank(name) == best_rank
            }
            previous_focus = min(previous_referents)
    return False


def _ranked_clause_mentions(
    world: CanonicalWorld,
    clause_tokens: tuple[str, ...],
) -> set[str]:
    words = _instruction_tokens(" ".join(clause_tokens))
    candidates = [obj.name for obj in world.objects if _mentioned(world, obj.name, words)]
    if not candidates:
        return set()

    def rank(name: str) -> tuple[int, int, int]:
        object_tokens = _object_tokens(world, name)
        positions = [
            index for index, token in enumerate(clause_tokens)
            if token in object_tokens or (
                len(token) > 3 and token.endswith("s") and token[:-1] in object_tokens
            )
        ]
        extras = object_tokens - words - _GENERIC_TOKENS
        return (
            min(positions, default=len(clause_tokens)),
            len(extras),
            len(object_tokens),
        )

    best = min(rank(name) for name in candidates)
    return {name for name in candidates if rank(name) == best}


def _instruction_relation_by_clause(
    literal: Literal,
    world: CanonicalWorld,
    instruction: str,
) -> str | None:
    """Read an explicit final placement relation from the instruction.

    GT goals anchor the intended objects, but a mismatched GT relation must not
    override an explicit ``on``/``in`` phrase in the instruction.  Requiring the
    preposition to introduce the target object avoids confusing phrases such as
    ``fill the bowl on the counter with water`` with ``put water on the bowl``.
    """
    if len(literal) != 3 or literal[0] not in {"on", "in"}:
        return None
    clauses = [
        clause.strip()
        for clause in re.split(r"[,.;:]|\bthen\b|\band\b", instruction.lower())
        if clause.strip()
    ]
    previous_focus: str | None = None
    previous_referents: set[str] = set()
    relation_words = {"on": "on", "onto": "on", "in": "in", "into": "in", "inside": "in"}
    for clause in clauses:
        clause_tokens = normalized_tokens(clause)
        for index, token in enumerate(clause_tokens):
            relation = relation_words.get(token)
            if relation is None:
                continue
            targets = _ranked_clause_mentions(world, clause_tokens[index + 1:])
            if literal[2] not in targets:
                continue
            subjects = _ranked_clause_mentions(world, clause_tokens[:index])
            prefix = set(clause_tokens[:index])
            subject_matches = literal[1] in subjects
            subject_matches |= "it" in prefix and literal[1] == previous_focus
            subject_matches |= "them" in prefix and literal[1] in previous_referents
            if subject_matches:
                return relation

        explicit = [
            obj.name
            for obj in world.objects
            if _mentioned(world, obj.name, _instruction_tokens(clause))
        ]
        if explicit:
            ranked = _ranked_clause_mentions(world, clause_tokens)
            previous_referents = ranked
            previous_focus = min(ranked) if ranked else None
    return None


def _coordinated_destination_targets(
    instruction: str,
    world: CanonicalWorld,
) -> set[str]:
    """Find shared destinations in coordinated transfer phrases.

    For example, ``add A, B, and C to the basin`` gives one destination for
    all three contents.  This is deliberately limited to the grammatical
    coordination pattern; it does not treat every mentioned relation as a
    requested final state.
    """
    text = " ".join(instruction.lower().split())
    if not re.search(
        r"\b(?:add|pour|fill)\b[^.;]*,[^.;]*\band\b[^.;]*\b(?:to|into)\b",
        text,
    ):
        return set()
    targets: set[str] = set()
    for match in re.finditer(r"\b(?:to|into)\b", text):
        tail = re.split(r"[.;]|\bthen\b", text[match.end():], maxsplit=1)[0]
        candidates: list[tuple[int, int, str]] = []
        for obj in world.objects:
            positions = [
                tail.find(token)
                for token in _object_tokens(world, obj.name)
                if token and tail.find(token) >= 0
            ]
            if positions:
                candidates.append((-
                    len(positions),
                    min(positions),
                    obj.name,
                ))
        if candidates:
            candidates.sort()
            targets.add(candidates[0][2])
    return targets


def _closure_staging_relevant(
    literal: Literal,
    world: CanonicalWorld,
    instruction: str,
) -> bool:
    if len(literal) != 3 or literal[0] != "on":
        return True
    if not (_object_tokens(world, literal[1]) & {"lid", "cap"}):
        return True
    clauses = re.split(r"[,.;:]|\bthen\b|\band\b", instruction.lower())
    return any(
        _mentioned(world, literal[1], _instruction_tokens(clause))
        and _mentioned(world, literal[2], _instruction_tokens(clause))
        and set(normalized_tokens(clause)) & {"put", "place", "return", "replace"}
        for clause in clauses
    )


def _find_object(world: CanonicalWorld, label: str, kind_hint: str | None) -> str | None:
    label_tokens = set(normalized_tokens(label))
    candidates = []
    for obj in world.objects:
        tokens = _object_tokens(world, obj.name)
        if not label_tokens <= tokens:
            continue
        score = 10 * len(label_tokens & tokens)
        if kind_hint and kind_hint in tokens:
            score += 20
        candidates.append((score, obj.name))
    return max(candidates, default=(0, None))[1]


def _table(world: CanonicalWorld, words: set[str]) -> str | None:
    candidates = []
    for obj in world.objects:
        tokens = _object_tokens(world, obj.name)
        if not (tokens & {"table", "counter", "desk"}):
            continue
        score = 10 if tokens & words else 0
        candidates.append((score, obj.name))
    return max(candidates, default=(0, None))[1]


def _stack_goals(instruction: str, world: CanonicalWorld) -> frozenset[Literal]:
    text = " ".join(instruction.lower().split())
    if "top to bottom" not in text:
        return frozenset()
    kind_hint = "bowl" if "bowl" in text else "block" if "block" in text else None
    sequences: list[list[str]] = []
    for match in re.finditer(
        r"(?:first|second|single)?\s*stack[^.:]*?(?:blocks?|bowls?)?\s*"
        r"([a-z0-9, ]+?)\s+in order from top to bottom",
        text,
    ):
        labels = re.findall(
            r"\b(?:\d+|red|orange|yellow|green|blue|purple|pink|white)\b",
            match.group(1),
        )
        if len(labels) >= 2:
            sequences.append(labels)
    if not sequences:
        tail = text.split("top to bottom", 1)[1]
        labels = re.findall(
            r"\b(?:\d+|red|orange|yellow|green|blue|purple|pink|white)\b",
            tail,
        )
        if len(labels) >= 2:
            sequences.append(labels)
    support = _table(world, _instruction_tokens(instruction))
    goals: set[Literal] = set()
    for labels in sequences:
        objects = [_find_object(world, label, kind_hint) for label in labels]
        if support is None or any(item is None for item in objects):
            continue
        names = [item for item in objects if item is not None]
        goals.update(("on", top, lower) for top, lower in zip(names, names[1:]))
        goals.add(("on", names[-1], support))
    return frozenset(goals)


def _return_to_origin_goals(
    instruction: str,
    world: CanonicalWorld,
) -> frozenset[Literal]:
    """Resolve an explicit return to the object's original scene location."""
    clauses = [
        clause.strip()
        for clause in re.split(r"[,.;:]|\bthen\b|\band\b", instruction.lower())
        if clause.strip()
    ]
    previous_focus: str | None = None
    previous_plural: set[str] = set()
    result: set[Literal] = set()
    for clause in clauses:
        tokens = normalized_tokens(clause)
        words = _instruction_tokens(clause)
        explicit = {
            obj.name for obj in world.objects if _mentioned(world, obj.name, words)
        }
        subjects = set(explicit)
        if "it" in tokens and previous_focus is not None:
            subjects.add(previous_focus)
        if "them" in tokens:
            subjects.update(previous_plural)
        if (
            "back" in tokens
            or "return" in tokens
            or "returned" in tokens
        ):
            placement = re.search(r"\bback\s+(on|onto|in|into)\b", clause)
            requested_relation = (
                {"onto": "on", "into": "in"}.get(placement.group(1), placement.group(1))
                if placement else None
            )
            result.update(
                literal
                for literal in world.facts
                if len(literal) == 3
                and literal[0]
                in {"on", "in", "inserted", "against", "under", "in_front_of"}
                and literal[1] in subjects
                and (requested_relation is None or literal[0] == requested_relation)
            )
        if explicit:
            ranked = _ranked_clause_mentions(world, tokens)
            previous_focus = min(ranked) if ranked else min(explicit)
            if len(ranked) > 1:
                previous_plural = ranked
    return frozenset(result)


def _explicit_placement_goals(instruction: str, world: CanonicalWorld) -> frozenset[Literal]:
    """Keep an unambiguous final placement even when the GT goal omits it."""
    result: set[Literal] = set()
    previous_subject: str | None = None
    for clause in re.split(r"[,.;:]|\bthen\b|\band\b", instruction.lower()):
        match = re.search(r"\b(?:put|place)\b(.*?)\b(on|onto|in|into)\b(.+)", clause)
        if match:
            subject_words = normalized_tokens(match.group(1))
            if "it" in subject_words:
                subjects = {previous_subject} if previous_subject else set()
            else:
                subjects = _ranked_clause_mentions(world, subject_words)
            targets = _ranked_clause_mentions(world, normalized_tokens(match.group(3)))
            if len(subjects) == len(targets) == 1:
                subject, target = next(iter(subjects)), next(iter(targets))
                has_gt_destination = any(
                    len(goal) == 3 and goal[0] in {"on", "in", "inserted"}
                    and goal[1] == subject for goal in world.goal_positive
                )
                if subject != target and not has_gt_destination:
                    relation = "on" if match.group(2) in {"on", "onto"} else "in"
                    result.add((relation, subject, target))
        mentions = _ranked_clause_mentions(world, normalized_tokens(clause))
        previous_subject = (
            next(iter(mentions)) if len(mentions) == 1
            and not {"it", "them"} & set(normalized_tokens(clause)) else None
        )
        if re.search(r"\b(?:in|into|on|onto|from|with|under|over)\b", clause):
            previous_subject = None
    return frozenset(result)


def _fold_on_support_goals(instruction: str, world: CanonicalWorld) -> frozenset[Literal]:
    """Keep an explicitly requested final fold and its named support."""
    goals: set[Literal] = set()
    for match in re.finditer(r"\bfold\b[^.;]*?\bon(?:to)?\b([^.;,]*)", instruction.lower()):
        supports = _ranked_clause_mentions(world, normalized_tokens(match.group(1)))
        foldables = {
            item[1] for item in world.facts
            if len(item) == 2 and item[0] == "unfolded"
        }
        if len(foldables) != 1 or len(supports) != 1:
            continue
        subject = next(iter(foldables))
        support = next(iter(supports))
        if subject != support:
            goals.update({("folded", subject), ("on", subject, support)})
    return frozenset(goals)


def compile_instruction_goal(instruction: str, world: CanonicalWorld) -> InstructionGoal:
    words = _instruction_tokens(instruction)
    stack = _stack_goals(instruction, world)
    return_to_origin = _return_to_origin_goals(instruction, world)
    stack_subjects = {item[1] for item in stack}
    coordinated_targets = _coordinated_destination_targets(instruction, world)
    positive: set[Literal] = set()
    for literal in world.goal_positive:
        if not _literal_relevant(literal, world, words):
            continue
        if (
            literal[0]
            in {"on", "in", "inserted", "under", "against", "away_from", "in_front_of"}
            and not _relation_relevant_by_clause(literal, world, instruction)
            and not (
                literal[0] == "in"
                and literal[2] in coordinated_targets
                and all(_mentioned(world, argument, words) for argument in literal[1:])
            )
        ):
            continue
        if not _closure_staging_relevant(literal, world, instruction):
            continue
        if literal[0] == "on" and literal[1] in stack_subjects:
            continue
        relation = _instruction_relation_by_clause(literal, world, instruction)
        positive.add(literal if relation is None else (relation, *literal[1:]))
    positive.update(stack)
    positive.update(return_to_origin)
    positive.update(_explicit_placement_goals(instruction, world))
    positive.update(_fold_on_support_goals(instruction, world))
    negative = {
        literal
        for literal in world.goal_negative
        if _literal_relevant(literal, world, words)
    }
    if not positive and not negative:
        reason = "unsupported_instruction_projection"
    else:
        reason = "instruction_projection"
    provenance = tuple(
        (literal, reason) for literal in sorted(positive | negative)
    )
    return InstructionGoal(frozenset(positive), frozenset(negative), provenance)


_EVENT_SPECS = {
    "activate": ({ActionFamily.ACTIVATE, ActionFamily.TURN_ON}, {"activate", "press", "push"}, {"activated"}),
    "boil": ({ActionFamily.TURN_ON}, {"boil", "heat", "warm"}, {"boiled", "heated"}),
    "close": ({ActionFamily.CLOSE, ActionFamily.REPLACE_CLOSURE, ActionFamily.TURN_OFF}, {"close", "shut", "replace"}, {"closed", "power_off"}),
    "cut": ({ActionFamily.CUT}, {"cut", "chop", "slice"}, {"cut"}),
    "fill": ({ActionFamily.FILL, ActionFamily.POUR}, {"fill", "pour", "dispense"}, {"contains_water", "in"}),
    "fold": ({ActionFamily.FOLD}, {"fold"}, {"folded"}),
    "lock": ({ActionFamily.LOCK}, {"lock"}, {"locked"}),
    "open": ({ActionFamily.OPEN, ActionFamily.REMOVE_CLOSURE}, {"open", "remove", "unscrew"}, {"open"}),
    "pick": ({ActionFamily.PICK, ActionFamily.REMOVE_CLOSURE}, {"pick", "take", "remove", "lift", "grasp"}, {"holding"}),
    "place": ({ActionFamily.PLACE_ON, ActionFamily.PLACE_IN, ActionFamily.INSERT, ActionFamily.REPLACE_CLOSURE}, {"place", "put", "return", "insert"}, {"on", "in", "inserted"}),
    "pour": ({ActionFamily.POUR, ActionFamily.FILL}, {"pour", "fill", "dispense"}, {"in", "contains_water"}),
    "press": ({ActionFamily.ACTIVATE}, {"press"}, {"activated"}),
    "push": ({ActionFamily.PUSH}, {"push"}, {"pushed"}),
    "scrunch": ({ActionFamily.SCRUNCH}, {"scrunch", "crumple"}, {"scrunched"}),
    "slide": ({ActionFamily.SLIDE}, {"slide", "silde"}, {"slid"}),
    "stir": ({ActionFamily.STIR}, {"stir", "mix"}, {"stirred"}),
    "turn_off": ({ActionFamily.TURN_OFF}, set(), {"power_off"}),
    "turn_on": ({ActionFamily.TURN_ON}, {"start"}, {"power_on"}),
    "unlock": ({ActionFamily.UNLOCK}, {"unlock"}, {"unlocked"}),
    "wash": ({ActionFamily.WASH, ActionFamily.WET}, {"wash", "rinse", "wet"}, {"washed", "clean", "wet"}),
    "wipe": ({ActionFamily.WIPE}, {"wipe", "clean"}, {"clean", "wiped"}),
}

_EVENT_ALIASES = {
    "activate": "activate",
    "boil": "boil",
    "boiled": "boil",
    "chop": "cut",
    "close": "close",
    "closed": "close",
    "crumple": "scrunch",
    "cut": "cut",
    "fill": "fill",
    "fold": "fold",
    "heat": "boil",
    "insert": "place",
    "lift": "pick",
    "lock": "lock",
    "mix": "stir",
    "open": "open",
    "pick": "pick",
    "place": "place",
    "pour": "pour",
    "press": "press",
    "push": "push",
    "put": "place",
    "remove": "pick",
    "return": "place",
    "rinse": "wash",
    "scrunch": "scrunch",
    "shut": "close",
    "slice": "cut",
    "slide": "slide",
    "slid": "slide",
    "silde": "slide",
    "start": "turn_on",
    "stir": "stir",
    "take": "pick",
    "unlock": "unlock",
    "warm": "boil",
    "wash": "wash",
    "wet": "wash",
    "wipe": "wipe",
}

_ORDER_STOP_WORDS = {
    "a", "an", "and", "at", "back", "before", "after", "from", "in",
    "into", "it", "its", "of", "off", "on", "onto", "the", "then",
    "them", "to", "up", "with",
}


def _event_key(text: str, *, last: bool) -> str | None:
    tokens = normalized_tokens(text)
    events: list[tuple[int, str]] = []
    for index, token in enumerate(tokens):
        if token == "turn" and index + 1 < len(tokens):
            if tokens[index + 1] == "off":
                events.append((index, "turn_off"))
                continue
            if tokens[index + 1] == "on":
                events.append((index, "turn_on"))
                continue
        event = _EVENT_ALIASES.get(token)
        if event is not None:
            events.append((index, event))
    if not events:
        return None
    return events[-1][1] if last else events[0][1]


def _event_fragment(text: str, *, last: bool) -> tuple[str, str] | None:
    fragments = [item.strip() for item in re.split(r"[,;]", text) if item.strip()]
    ordered = reversed(fragments) if last else iter(fragments)
    for fragment in ordered:
        event = _event_key(fragment, last=last)
        if event is not None:
            return fragment, event
    event = _event_key(text, last=last)
    return None if event is None else (text, event)


def _event_positions(
    fragment: str,
    context: str,
    event: str,
    actions: tuple[object, ...],
    world: CanonicalWorld | None = None,
) -> tuple[int, ...]:
    families, keywords, effect_predicates = _EVENT_SPECS[event]
    fragment_tokens = normalized_tokens(fragment)
    context_tokens = normalized_tokens(context)
    relation_tokens = {
        "place": {"on", "onto", "in", "into", "to", "against", "under"},
        "pick": {"from"},
        "fill": {"into", "to"},
        "pour": {"into", "to"},
    }.get(event, set())
    target_tokens: set[str] = set()
    for index, token in enumerate(fragment_tokens):
        if token not in relation_tokens:
            continue
        for item in fragment_tokens[index + 1:]:
            if item == "and" or item in _EVENT_ALIASES:
                break
            if item not in _ORDER_STOP_WORDS and item != "number" and not item.isdigit():
                target_tokens.add(item)
        if target_tokens:
            break
    noun_tokens = {
        token
        for token in fragment_tokens
        if token not in _ORDER_STOP_WORDS and token not in _EVENT_ALIASES
    }
    if not noun_tokens:
        noun_tokens = {
            token
            for token in context_tokens
            if token not in _ORDER_STOP_WORDS and token not in _EVENT_ALIASES
        }
    required_tokens = target_tokens or noun_tokens
    pronoun_subject_tokens: set[str] = set()
    if set(fragment_tokens) & {"it", "them"}:
        pronoun_subject_tokens = {
            token
            for token in context_tokens
            if token not in _ORDER_STOP_WORDS
            and token not in _EVENT_ALIASES
            and token not in target_tokens
        }
    candidates: list[tuple[int, set[str]]] = []
    for index, action in enumerate(actions, 1):
        family = getattr(action, "family", None)
        raw = getattr(action, "raw_action", "")
        action_tokens = set(normalized_tokens(raw))
        semantic_predicates = {
            item[0] for item in getattr(action, "semantic_effects", ())
        }
        declared_predicates = {
            item[0] for item in getattr(action, "add_effects", ())
        }
        family_match = family in families
        keyword_match = bool(action_tokens & keywords)
        effect_match = bool(
            (semantic_predicates | declared_predicates) & effect_predicates
        )
        if not (family_match or keyword_match or effect_match):
            continue
        candidates.append((index, action_tokens))

    if event == "place" and "back" in fragment_tokens and world is not None:
        return tuple(
            index
            for index, action_tokens in candidates
            if any(
                item in world.goal_positive
                for item in getattr(actions[index - 1], "add_effects", ())
                if len(item) == 3 and item[0] in {"on", "in", "inserted"}
            )
        )

    if required_tokens and not target_tokens:
        frequencies = {
            token: sum(token in action_tokens for _, action_tokens in candidates)
            for token in required_tokens
        }
        positive = {token: count for token, count in frequencies.items() if count}
        if positive:
            least_common = min(positive.values())
            required_tokens = {
                token for token, count in positive.items() if count == least_common
            }

    mentioned_objects: set[str] = set()
    if world is not None:
        ranked = _ranked_clause_mentions(world, fragment_tokens)
        if len(ranked) == 1:
            mentioned_objects = ranked

    positions = []
    for index, action_tokens in candidates:
        action_objects = set(getattr(actions[index - 1], "objects", ()))
        if mentioned_objects and not mentioned_objects <= action_objects:
            continue
        if target_tokens and not target_tokens <= action_tokens:
            continue
        if required_tokens and not target_tokens and not (
            required_tokens & action_tokens
        ):
            continue
        if pronoun_subject_tokens and not target_tokens and not (
            pronoun_subject_tokens & action_tokens
        ):
            continue
        positions.append(index)
    return tuple(positions)


def _explicit_order_issue(
    instruction: str,
    actions: tuple[object, ...],
    world: CanonicalWorld | None = None,
) -> tuple[int, str] | None:
    text = " ".join(instruction.lower().split())
    constraints: list[tuple[str, str, str, str]] = []
    then_parts = re.split(r"\bthen\b", text)
    for left, right in zip(then_parts, then_parts[1:]):
        constraints.append((left, right, left, right))
    if " before " in text:
        left, right = text.split(" before ", 1)
        constraints.append((left, right, left, right))
    if " after " in text:
        after, before = text.split(" after ", 1)
        constraints.append((before, after, before, after))

    for before_text, after_text, before_context, after_context in constraints:
        before_event = _event_fragment(before_text, last=True)
        after_event = _event_fragment(after_text, last=False)
        if before_event is None or after_event is None:
            continue
        before_fragment, before_key = before_event
        after_fragment, after_key = after_event
        if before_key != "place":
            continue
        before_positions = _event_positions(
            before_fragment, before_context, before_key, actions, world
        )
        after_positions = _event_positions(
            after_fragment, after_context, after_key, actions, world
        )
        if not before_positions or not after_positions:
            continue
        before_step = min(before_positions)
        after_step = min(after_positions)
        if after_step < before_step:
            return (
                after_step,
                f"instruction requires {before_key} before {after_key}",
            )
    return None


def instruction_order_issue(
    instruction: str,
    actions: tuple[object, ...],
    world: CanonicalWorld | None = None,
) -> tuple[int, str] | None:
    text = " ".join(instruction.lower().split())
    explicit = _explicit_order_issue(instruction, actions, world)
    if explicit is not None:
        return explicit
    delayed_closure = re.search(
        r"(?:put|place|screw)(?:\s+\w+){0,4}\s+back|\breplac(?:e|ed|ing)\b",
        text,
    )
    if (
        "at the end" in text
        and any(word in text for word in ("lid", "cap"))
        and delayed_closure
    ):
        closure = next(
            (
                index
                for index, action in reversed(list(enumerate(actions, 1)))
                if getattr(action, "family", None).value == "replace_closure"
            ),
            None,
        )
        last_core = max(
            (
                index
                for index, action in enumerate(actions, 1)
                if getattr(action, "family", None).value
                in {"turn_off", "place_on", "place_in"}
            ),
            default=0,
        )
        if closure is not None and closure < last_core:
            return closure, "instruction requires replacing the lid/cap at the end"
    return None


def instruction_action_issue(
    instruction: str,
    actions: tuple[object, ...],
    world: CanonicalWorld | None = None,
    *,
    bind_named: bool = False,
) -> tuple[int, str] | None:
    words = _instruction_tokens(instruction)
    lines = [getattr(action, "raw_action", "").lower() for action in actions]
    action_names = [line.lstrip("(").split(maxsplit=1)[0] for line in lines]
    requirements = {
        "sweep": ("sweep",),
        "wipe": ("wipe",),
        "stir": ("stir",),
        "cut": ("cut", "chop"),
        "chop": ("cut", "chop"),
        "rinse": ("rinse", "wash"),
        "rinsed": ("rinse", "wash"),
        "wash": ("rinse", "wash"),
        "wet": ("wet",),
        "fold": ("fold",),
        "scrunch": ("scrunch", "crumple"),
        "slide": ("slide", "silde"),
        "silde": ("slide", "silde"),
        "press": ("press",),
        "push": ("push",),
    }

    family_requirements = {
        "sweep": {ActionFamily.SWEEP},
        "wipe": {ActionFamily.WIPE},
        "stir": {ActionFamily.STIR},
        "cut": {ActionFamily.CUT},
        "chop": {ActionFamily.CUT},
        "rinse": {ActionFamily.WASH},
        "rinsed": {ActionFamily.WASH},
        "wash": {ActionFamily.WASH},
        "wet": {ActionFamily.WET},
        "fold": {ActionFamily.FOLD},
        "scrunch": {ActionFamily.SCRUNCH},
        "slide": {ActionFamily.SLIDE},
        "silde": {ActionFamily.SLIDE},
        "press": {ActionFamily.ACTIVATE},
        "push": {ActionFamily.PUSH},
    }

    def has_required_action(word: str, aliases: tuple[str, ...]) -> bool:
        expected = family_requirements.get(word, set())
        if any(
            getattr(getattr(action, "family", None), "value", None)
            in {family.value for family in expected}
            for action in actions
        ):
            return True
        if any(
            any(effect[0] == word for effect in getattr(action, "semantic_effects", ()))
            for action in actions
        ):
            return True
        return any(
            any(alias in line for alias in aliases)
            for line in lines
        )

    for word, aliases in requirements.items():
        if word in words and not has_required_action(word, aliases):
            return len(actions) + 1, f"instruction requires an explicit {word} action"

    if bind_named:
        fragments = [
            fragment.strip()
            for fragment in re.split(r"[,.;:]|\bthen\b|\band\b", instruction.lower())
            if fragment.strip()
        ]
        for fragment in fragments:
            event = _event_key(fragment, last=False)
            if event != "push":
                continue
            if not _event_positions(fragment, instruction, event, actions, world):
                return len(actions) + 1, f"instruction requires {event} on the named object or target"
    if "against" in words and not any(
        "against" in line
        or any(effect[0] == "against" for effect in getattr(action, "add_effects", ()))
        for line, action in zip(lines, actions)
    ):
        return len(actions) + 1, "instruction requires an explicit placement against the named support"
    if {"fill", "water"} <= words:
        transfer_names = {"fill", "pour", "add", "dispense"}
        has_transfer = any(
            any(token in name for token in transfer_names)
            or any(
                effect[0] == "in" and "water" in " ".join(effect)
                for effect in getattr(action, "add_effects", ())
            )
            for name, action in zip(action_names, actions)
        )
        if not has_transfer:
            return len(actions) + 1, "instruction requires transferring water into the target"
    return None
