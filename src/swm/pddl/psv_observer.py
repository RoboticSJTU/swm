"""Extend operation-context events with the state of their bound resource roles."""
from collections import defaultdict
from typing import NamedTuple

from swm.pddl.reorder import expression_bits, required_masks


class _Role(NamedTuple):
    positions: tuple
    free: tuple
    custody: tuple
    held_bits: int
    required: bool
    exclusion_only: bool
    free_bits: int
    incident: tuple
    conflicts: int


def _positive_mentions(expression):
    if isinstance(expression, bool):
        return 0
    if expression[0] == "mask":
        return expression[1]
    bits = 0
    for child in expression[1]:
        bits |= _positive_mentions(child)
    return bits


class AvailabilityObserver:
    """Record physical resource roles as observations on already executable transitions.

    Class identity retains argument roles, availability and custody where relevant.
    No observation changes the PDDL preconditions, effects, or goal.
    """
    version = "availability_residual_v4"

    def __init__(self, base):
        self.base = base
        self.model = base.model
        self.channels = base.channels
        self.initial_profile = base.initial_profile
        self.finish = base.finish
        self.events = {}
        self.tokens = {}
        self.token_channels = {}
        self._roles = self._compile_roles()

    def _compile_roles(self):
        model, base = self.model, self.base
        free_names = set(base.schema.get("resource_free_predicates", ()))
        arities = base.schema.get("relation_arities", {})
        free = defaultdict(list)
        custody = defaultdict(list)
        free_masks = defaultdict(int)
        custody_masks = defaultdict(int)
        all_masks = defaultdict(int)
        incident = defaultdict(list)
        for hand in base.resource_set:
            for name in sorted(free_names):
                if len(model.predicates.get(name, ())) == 1:
                    fact = (name, hand)
                    bit = model.bits.get(fact, 0)
                    free[hand].append((name, bit, not bit and fact in model.initial_facts))
        for fact in sorted(set(model.bits) | model.initial_facts):
            name = fact[0]
            bit = model.bits.get(fact, 0)
            for position, obj in enumerate(fact[1:]):
                if bit and obj in base.resource_set:
                    all_masks[obj] |= bit
                    if name not in base.resource_relations and name not in free_names:
                        incident[obj].append((name, position, bit))
            if len(fact) == 2 and name in free_names:
                free_masks[fact[1]] |= bit
            role = base.resource_relations.get(name)
            if role and max(role.values()) < len(fact):
                hand = fact[role["resource"]]
                custody_masks[hand] |= bit
                if hand in base.resource_set and (name not in arities or len(fact) == arities[name] + 1):
                    custody[hand].append((name, fact[role["payload"]], bit,
                                          not bit and fact in model.initial_facts))

        # Find loads that may coexist with a declared-free hand after a faulty acquisition.
        conflicts = defaultdict(int)
        for action in model.actions:
            positive, _ = required_masks(action.guard)
            for hand in base.resource_set:
                if positive & free_masks[hand] and not action.possible_delete & free_masks[hand]:
                    conflicts[hand] |= action.possible_add & custody_masks[hand]

        result = []
        for action in model.actions:
            slots = defaultdict(list)
            for i, argument in enumerate(action.args):
                slots[argument].append(i)
            hands = [hand for hand in slots if hand in base.resource_set]
            positive, negative = required_masks(action.guard)
            guard_bits = expression_bits(action.guard)
            other_access = action.possible_add | action.possible_delete | _positive_mentions(action.guard)
            for condition, _, _ in action.effects:
                other_access |= expression_bits(condition)
            roles = []
            for hand in hands:
                links = tuple((name, tuple(slots.get(payload, ())), bit, static)
                              for name, payload, bit, static in custody[hand])
                held_bits = sum(bit for _, _, bit, _ in links)
                matched = sum(bit for _, positions, bit, _ in links if positions)
                read = guard_bits & all_masks[hand]
                # A secondary hand whose sole role is a negative custody test cannot
                # introduce an unrelated carried payload when a different hand acts.
                exclusion = (len(hands) > 1 and bool(read) and not read & ~(negative & matched)
                             and not other_access & all_masks[hand])
                available = free_masks[hand]
                roles.append(_Role(tuple(slots[hand]), tuple(free[hand]), links, held_bits,
                                   bool(positive & (available | held_bits)), exclusion, available,
                                   tuple(incident[hand]), conflicts[hand] if positive & available else 0))
            result.append(roles)
        return result

    def _resource_prestate(self, state, successor, index):
        add, delete = self.model.activated(state, index)
        roles = self._roles[index]
        changed = state ^ successor
        active = {r.positions for r in roles if r.required and changed & r.held_bits}
        scoped, residual, hidden_conflicts = [], [], []
        for role in roles:
            acquiring = bool(add & role.held_bits)
            released = delete & ~add & role.held_bits
            flags, occupied, matches = (), False, set()
            if not (role.exclusion_only and active - {role.positions}):
                if acquiring or not (released or role.required):
                    flags = tuple((name, bool(state & bit or static)) for name, bit, static in role.free)
                    occupied = any(state & bit or static for _, _, bit, static in role.custody)
                for name, positions, bit, static in role.custody:
                    if positions and (state & bit or static) and (released & bit or not (acquiring or released)):
                        matches.add((name, positions))
            scoped.append((role.positions, (acquiring, bool(released)), flags, occupied, tuple(sorted(matches))))

            if state & role.free_bits:
                others = {(name, position) for name, position, bit in role.incident if state & bit}
                if others:
                    residual.append((role.positions, tuple(sorted(others))))
                if not (flags or occupied) and state & role.conflicts:
                    hidden_conflicts.append(role.positions)
        return tuple(scoped), tuple(residual), tuple(hidden_conflicts)

    def event(self, state, identifier):
        successor, previous = self.base.event(state, identifier)
        signature = (*self.base.events[previous], self._resource_prestate(state, successor, identifier - 1))
        token = self.tokens.get(signature)
        if token is None:
            token = len(self.tokens) + 1
            self.tokens[signature] = token
            self.events[token] = signature
            self.token_channels[token] = (0,)
        return successor, token

    def metadata(self):
        return {"version": self.version, "resource_argument_roles": True,
                "base_observer": self.base.metadata(), "event_count": len(self.events)}
