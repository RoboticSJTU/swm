"""Deterministic PDDL failure attribution, evidence checks and error taxonomy.

VLM is used only to map object identities.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import product
from math import prod
from pathlib import Path

from swm.llm import call_gpt_json
from swm.pddl.planner import _run_fast_downward, fast_downward_path
from swm.pddl.strips import (
    ground_plan, parse_domain, parse_plan, parse_problem_model,
    parse_sexpr_file, read_literals, rollout,
)
from swm.simulator.alignment.objects import align_worlds
from swm.simulator.alignment.predicates import STATE_PREDICATES, compile_world, normalized_tokens
from swm.simulator.ir.mapping import map_schema
from swm.simulator.predicate_aliases import canonical_literal


LABELS = ("pddl_invalid", "init_error", "goal_error", "operator_missing", "operator_semantics", "else")
RULES_VERSION = 7


# Model and plan evidence.

SPATIAL = {"in", "on"}
OPPOSITES = dict(pair for a, b in (
    ("open", "closed"), ("power_on", "power_off"), ("locked", "unlocked"),
    ("upright", "upside_down"), ("flat", "vertical"), ("wet", "dry"),
    ("folded", "unfolded"), ("enabled", "disabled"), ("cut", "uncut"),
    ("clean", "dirty"),
) for pair in ((a, b), (b, a)))
SEMANTIC_FAMILIES = {"wipe", "stir", "cut", "wash", "fold", "scrunch", "wet", "slide", "push"}
GUARDS = {*SPATIAL, *OPPOSITES, "holding", "hand_free", "clear", "unblocked"}
SEMANTIC_STATES = {"clean", "washed", "stirred", "cut", "folded", "scrunched", "wet", "slid", "pushed"}
STRUCTURAL = {*SPATIAL, "holding", "hand_free", "clear", "open", "closed", "power_on", "power_off",
              "locked", "unlocked", "upright", "upside_down", "flat", "vertical", "under", "inserted"}
EVENT_PREFIXES = ("placed_", "picked_", "returned_", "removed_", "stored_")


def contradictions(positive, negative=()):
    positive, negative = set(positive), set(negative)
    conflicts = {tuple(sorted((fact, ("not", *fact)))) for fact in positive & negative}
    for fact in positive:
        if len(fact) == 2 and (other := OPPOSITES.get(fact[0])) and (other, fact[1]) in positive:
            conflicts.add(tuple(sorted((fact, (other, fact[1])))))
        if len(fact) == 3 and fact[0] == "holding" and ("hand_free", fact[1]) in positive:
            conflicts.add(tuple(sorted((fact, ("hand_free", fact[1])))))
    return [[list(fact) for fact in pair] for pair in sorted(conflicts)]


def check_declarations(directory):
    """FD tolerates duplicate names; do not lose these errors in our parser."""
    issues = []
    for filename in ("domain.pddl", "problem.pddl"):
        root = parse_sexpr_file(directory / filename)
        sections = [s for s in root[1:] if isinstance(s, list) and s]
        groups = {}
        for section in sections:
            head = section[0]
            if head == ":action":
                groups.setdefault("action", []).append(section[1])
                if ":parameters" in section:
                    params = section[section.index(":parameters") + 1]
                    groups[f"parameters:{section[1]}"] = params
            elif head == ":predicates":
                groups["predicate"] = [p[0] for p in section[1:] if isinstance(p, list) and p]
                for predicate in section[1:]:
                    if isinstance(predicate, list) and predicate:
                        groups[f"predicate_parameters:{predicate[0]}"] = predicate[1:]
            elif head == ":objects":
                groups["object"] = section[1:]
        for kind, names in groups.items():
            # Typed lists are outside this evaluator's contract, not duplicate names.
            if not all(isinstance(name, str) for name in names) or "-" in names:
                continue
            duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
            if duplicates:
                issues.append({"kind": "duplicate_declaration", "file": filename,
                               "declaration": kind, "symbols": duplicates})
        for section in sections:
            if section[0] not in {":init", ":goal"}:
                continue
            expression = ["and", *section[1:]] if section[0] == ":init" else section[1]
            positive, negative = read_literals(expression)
            # dispenses is a source capability, not current water containment.
            def canonical(fact):
                return fact if fact[0] == "dispenses" else canonical_literal(fact)
            conflicts = contradictions({canonical(f) for f in positive},
                                       {canonical(f) for f in negative})
            if conflicts:
                issues.append({"kind": "contradictory_literals", "section": section[0],
                               "conflicts": conflicts})
            reflexive = [list(f) for f in sorted(positive) if len(f) == 3 and f[0] in SPATIAL and f[1] == f[2]]
            if reflexive:
                issues.append({"kind": "reflexive_location", "section": section[0], "facts": reflexive})
    return {"issues": issues}


def substitute(literal, binding):
    return tuple(binding.get(token, token) for token in literal)


def unify(pattern, fact):
    if len(pattern) != len(fact) or pattern[0] != fact[0]:
        return None
    binding = {}
    for variable, value in zip(pattern[1:], fact[1:]):
        if variable.startswith("?"):
            if variable in binding and binding[variable] != value:
                return None
            binding[variable] = value
        elif variable != value:
            return None
    return binding


def producers(world, fact, *, positive=True, check_kinds=True):
    """Overapproximate producers, respecting repeated variables and bound kinds.

    An empty result proves there is no producer. A nonempty result does not
    claim that its preconditions are reachable. No Cartesian grounding needed.
    """
    found = []
    for schema in world.schemas.values():
        effects = schema.add_eff if positive else schema.del_eff - schema.add_eff
        for effect in effects:
            pattern = world.interface.canonicalize(effect)
            binding = unify(pattern, fact)
            if binding is None:
                continue
            if check_kinds and not compatible_kinds(world, schema, binding):
                continue
            found.append(schema.name)
            break
    return sorted(set(found))


def compatible_kinds(world, schema, binding, *, checking=None):
    for precondition in schema.pre_pos:
        if len(precondition) != 2 or not world.interface.is_static(precondition[0]):
            continue
        pattern = world.interface.canonicalize(precondition)
        if pattern[1] not in binding:
            continue
        grounded = substitute(pattern, binding)
        if grounded != checking and grounded not in world.facts:
            return False
    return True


def reference_producers(world, fact, *, positive=True, owners=None):
    # The existing attribution contract compares in/on by target area.
    predicates = SPATIAL if positive and fact[0] in SPATIAL else {fact[0]}
    targets = {fact[1]} if len(fact) == 2 else set()
    if fact[0] in {"open", "closed"} and owners:
        parent = owners.get(fact[1], fact[1])
        targets = {parent} | {lid for lid, owner in owners.items() if owner == parent}
    facts = {(predicate, *fact[1:]) for predicate in predicates}
    if targets:
        facts.update((fact[0], target) for target in targets)
    return [name for item in sorted(facts) for name in producers(world, item, positive=positive)]


def mapped_fact(fact, mapping):
    if all(arg in mapping for arg in fact[1:]):
        return (fact[0], *(mapping[arg] for arg in fact[1:]))
    return None


def equivalent(fact, facts):
    if fact in facts:
        return True
    return fact[0] in SPATIAL and any((p, *fact[1:]) in facts for p in SPATIAL)


def enclosing_objects(facts, objects):
    """Include spatial carriers and attached wholes in a physical frame."""
    enclosing = set(objects)
    while True:
        parents = {f[2] for f in facts if len(f) == 3 and f[1] in enclosing
                   and f[0] in SPATIAL | {"static:part_of", "static:drawer_of", "static:attached_to"}}
        if parents <= enclosing:
            return enclosing
        enclosing.update(parents)


def contradicted(fact, facts):
    if len(fact) == 2:
        if fact[0] == "clear":
            return any(len(f) == 3 and f[0] == "on" and f[2] == fact[1] for f in facts)
        if fact[0] == "unblocked":
            return any(len(f) == 3 and f[0] in {"blocks", "blocks_opening"} and f[2] == fact[1]
                       for f in facts)
        return (OPPOSITES.get(fact[0]), fact[1]) in facts or (
            fact[0] == "hand_free" and any(f[0] == "holding" and f[1] == fact[1] for f in facts))
    if len(fact) == 3 and fact[0] == "holding":
        return ("hand_free", fact[1]) in facts
    if len(fact) == 3 and fact[0] in SPATIAL:
        return any(len(f) == 3 and ((f[0] in SPATIAL and f[1] == fact[1] and f[2] != fact[2])
                                   or (f[0] == "holding" and f[2] == fact[1])) for f in facts)
    return False


def check_missing_init(candidate, gt, mapping):
    """A confirmed GT fact is missing, necessary, and has no producer."""
    inverse = {target: source for source, target in mapping.items()}
    matches = []
    necessary = necessary_facts(candidate)
    for expected in sorted(gt.facts):
        fact = mapped_fact(expected, inverse)
        if fact is None or fact not in necessary or equivalent(fact, candidate.facts) or producers(candidate, fact):
            continue
        if fact[0] in SPATIAL and any(len(f) == 3 and (
                (f[0] in SPATIAL and f[1] == fact[1]) or (f[0] == "holding" and f[2] == fact[1]))
                for f in candidate.facts):
            continue  # Existing location checks handle areas; do not require all GT supports.
        uses = [schema.name for schema in candidate.schemas.values() for pre in schema.pre_pos
                if (binding := unify(candidate.interface.canonicalize(pre), fact)) is not None
                and compatible_kinds(candidate, schema, binding, checking=fact)]
        if fact in candidate.goal_positive:
            uses.append(":goal")
        if not uses:
            continue
        matches.append({"kind": "required_fact_missing", "gt": list(expected),
                        "candidate_fact": list(fact), "used_by": sorted(set(uses)), "status": "fail",
                        "reason": "required GT initial fact is absent and has no candidate producer"})
    return {"matched": bool(matches), "matches": matches}


def ground_saved_plan(directory, world):
    path = directory / "plan.txt"
    raw, _ = parse_plan(path)
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith(";")]
    if len(raw) != len(lines):
        raise ValueError("Malformed saved plan; cannot use partial replay as evidence")
    return ground_plan(raw, world.schemas, frozenset(o.name for o in world.objects))


def replay_saved_plan(directory, world):
    """Validate raw PDDL goals before using a saved trace as evidence.

    Canonical aliases are for cross-model comparison. They cannot make a plan
    satisfy a different raw predicate, or erase a still-present alias fact.
    """
    problem = parse_problem_model(directory / "problem.pddl", world.schemas)
    plan = ground_saved_plan(directory, world)
    final = rollout(set(problem.init_state), plan)
    if not problem.goal_positive <= final or problem.goal_negative & final:
        raise ValueError("Saved plan does not satisfy its own raw PDDL goals")
    return plan, final


def necessary_facts(world, *, static_feasible=False, coeffects=False, problem=None, keep_unbound=False):
    """Ground achievement landmarks: common prerequisites and coeffects.

    Include every producer (optionally prune static incompatibility). A
    coeffect of every producer is necessarily achieved too, but need not remain
    true finally. Unbound arguments never establish a ground fact. Supplying
    the raw problem keeps distinct PDDL aliases distinct along every route.
    ``keep_unbound`` also records common existential patterns; these are never
    expanded recursively or treated as evidence for one particular binding.
    """
    canonical = world.interface.canonicalize if problem is None else tuple
    initial = world.facts if problem is None else problem.init_state
    needed = set(world.goal_positive if problem is None else problem.goal_positive)
    frontier = list(needed)
    while frontier:
        fact = frontier.pop()
        if fact in initial:
            continue
        alternatives = []
        for schema in world.schemas.values():
            for effect in schema.add_eff:
                if (binding := unify(canonical(effect), fact)) is None:
                    continue
                if static_feasible and not compatible_kinds(world, schema, binding):
                    continue
                for variable in schema.params:
                    if variable in binding:
                        continue
                    domain = {o.name for o in world.objects}
                    for pre in schema.pre_pos:
                        pattern = canonical(pre)
                        category = (pattern[0].startswith("kind:") if problem is None
                                    else world.interface.is_static(pre[0]))
                        if len(pattern) == 2 and pattern[1] == variable and category:
                            domain &= {f[1] for f in initial if len(f) == 2 and f[0] == pattern[0]}
                    if len(domain) == 1:
                        binding[variable] = next(iter(domain))
                witnesses = schema.pre_pos | (schema.add_eff if coeffects else set())
                alternatives.append({substitute(canonical(pre), binding)
                                     for pre in witnesses})
        common = set.intersection(*alternatives) if alternatives else set()
        for pre in common - needed:
            if any(arg.startswith("?") for arg in pre[1:]):
                if keep_unbound:
                    needed.add(pre)
                continue
            needed.add(pre)
            frontier.append(pre)
    return needed


def required_predicates(world):
    """Necessary predicate families, overapproximating every producer route."""
    needed = {f[0] for f in world.goal_positive - world.facts}
    frontier = list(needed)
    while frontier:
        predicate = frontier.pop()
        if any(f[0] == predicate for f in world.facts):
            continue
        alternatives = [{world.interface.canonicalize(pre)[0] for pre in schema.pre_pos}
                        for schema in world.schemas.values()
                        if any(world.interface.canonicalize(e)[0] == predicate for e in schema.add_eff)]
        common = set.intersection(*alternatives) if alternatives else set()
        for pre in common - needed:
            needed.add(pre)
            frontier.append(pre)
    return needed


def required_processing(world):
    """Required processing predicates, including common causal prerequisites."""
    grounded = {f[0] for f in necessary_facts(world, static_feasible=True, coeffects=True) - world.facts}
    return (required_predicates(world) | grounded) & SEMANTIC_STATES


def required_final_facts(world):
    """Only monotone processing landmarks extend the explicit final goals.

    A prerequisite can be transient. It supports a final-state comparison only
    if it must first be achieved and no reference action can delete it. Facts
    already true initially and incidental reference-plan effects add nothing.
    """
    deleted = {world.interface.canonicalize(f) for s in world.schemas.values() for f in s.del_eff}
    return set(world.goal_positive) | {
        f for f in necessary_facts(world, static_feasible=True, coeffects=True) - world.facts
        if f[0] in SEMANTIC_STATES and not any(unify(pattern, f) is not None for pattern in deleted)}


def unresolved_evidence(candidate, gt, mapping, checks):
    """Expose proof gaps without turning them into guessed error categories."""
    inverse = set(mapping.values())
    signatures = {(p.canonical_name, p.arity) for p in candidate.interface.descriptions if not p.static}
    if ("in", 2) in signatures and any("water" in o.kinds for o in candidate.objects):
        signatures.add(("contains_water", 1))
    represented = {mapped for fact in candidate.facts if (mapped := mapped_fact(fact, mapping)) is not None}
    needed = necessary_facts(gt, static_feasible=True)
    constraints = [list(f) for f in sorted(gt.facts) if not equivalent(f, represented)
                   and (f[0] in {"blocks", "blocks_opening", "covers"}
                        or (len(f) == 3 and f[0] == "on" and ("clear", f[2]) in needed))]
    return {
        "candidate_plan_unavailable": (checks.get("operator_semantics", {}).get("unavailable")
                                       if checks.get("reference_plan", {}).get("valid") else None),
        "unreachable_goals": checks.get("causal_reachability", {}).get("unreachable_goals", []),
        "goal_mutexes": checks.get("causal_reachability", {}).get("goal_mutexes", []),
        "unmapped_candidate_objects": sorted({o.name for o in candidate.objects} - mapping.keys()),
        "unmapped_reference_goal_objects": sorted({arg for f in gt.goal_positive | gt.goal_negative
                                                    for arg in f[1:]} - inverse),
        "reference_goal_interfaces": [list(f) for f in sorted(gt.goal_positive | gt.goal_negative)
                                      if not f[0].startswith(("kind:", "static:"))
                                      and (f[0], len(f) - 1) not in signatures],
        "reference_only_initial_constraints": constraints,
        "uncertified_guards": [item for item in checks.get("strict_operator_semantics", {}).get("guard_candidates", [])
                               if item["kind"] != "reference_state_mutex" and "reference_mutex_proof" not in item],
        "uncertain_goal_encodings": checks.get("goal_error", {}).get("uncertain", []),
        "scope": "formal barriers do not identify the faulty section; missing identity or predicate correspondence is not an error proof",
    }


def check_operator_coverage(candidate, gt, reference):
    """Missing semantic capabilities, including when extra operators mask counts."""
    try:
        used = {action.name for action in replay_saved_plan(reference, gt)[0]}
    except (OSError, ValueError, KeyError) as error:
        return {"matched": False, "matches": [], "unavailable": str(error)}
    families = {family for schema in candidate.schemas.values()
                for family in (map_schema(schema)[0].value, named_family(schema))}
    effects = {candidate.interface.canonicalize(f)[0]
               for s in candidate.schemas.values() for f in s.add_eff}
    matches = []
    required = required_processing(gt)
    initial = {f[0] for f in candidate.facts}
    for schema in gt.schemas.values():
        family = map_schema(schema)[0].value
        if schema.name not in used or family not in SEMANTIC_FAMILIES or family in families:
            continue
        # A macro with the same state effect can implement this capability.
        semantic_effects = {gt.interface.canonicalize(f)[0] for f in schema.add_eff
                            if len(f) == 2 and gt.interface.canonicalize(f)[0] in required}
        if not semantic_effects or semantic_effects & (effects | initial):
            continue
        # A negative state effect or unknown processing marker may be an
        # alternate encoding. Absence of familiar positive names is not proof.
        if alternate_state_effects(candidate, semantic_effects):
            continue
        matches.append({"kind": "missing_semantic_capability", "gt_action": schema.name,
                        "family": family, "gt_effects": sorted(semantic_effects)})
    return {"matched": bool(matches), "matches": matches}


def named_family(schema):
    return map_schema(replace(schema, pre_pos=set(), add_eff=set(), del_eff=set()))[0].value


def alternate_state_effects(world, predicates, *, actions=None, objects=None):
    """Possible encodings veto absence proofs; they do not establish success.

    A different action family may implement processing via an opaque marker or
    a negative state. With a validated ground trace, restrict uncertainty to
    the affected objects; schema-level capability checks keep all alternatives.
    """
    opposite = {OPPOSITES[p] for p in predicates if p in OPPOSITES}
    evidence = []
    for action in world.schemas.values() if actions is None else actions:
        added = {world.interface.canonicalize(f) for f in action.add_eff}
        deleted = {world.interface.canonicalize(f) for f in action.del_eff} - added
        unknown = {f for f in added | deleted if f[0] not in STATE_PREDICATES
                   and not f[0].startswith(("kind:", "static:"))}
        negative = {f for f in deleted if f[0] in opposite}
        participants = set(getattr(action, "args", ())) | {arg for f in unknown | negative for arg in f[1:]}
        if (unknown or negative) and (objects is None or not participants or participants & objects):
            evidence.append({"action": action.name, "unknown_effects": [list(f) for f in sorted(unknown)],
                             "negative_state_effects": [list(f) for f in sorted(negative)]})
    return evidence


def check_effect_invariants(candidate, gt, *, unsolved, mapping=None):
    matches, uncertain = [], []
    initial_predicates = {f[0] for f in candidate.facts}
    added_predicates = {candidate.interface.canonicalize(f)[0]
                        for schema in candidate.schemas.values() for f in schema.add_eff}
    required = required_processing(gt)
    reference_families = {named_family(schema) for schema in gt.schemas.values()
                          if any(gt.interface.canonicalize(f)[0] in required for f in schema.add_eff)}
    for schema in candidate.schemas.values():
        family = named_family(schema)
        if family not in SEMANTIC_FAMILIES or family not in reference_families:
            continue
        # Do not blame an optional bad variant if another action can establish
        # the required processing state (including an idempotent initial one).
        needed = {gt.interface.canonicalize(f)[0] for s in gt.schemas.values() if named_family(s) == family
                  for f in s.add_eff if gt.interface.canonicalize(f)[0] in required}
        if needed & (initial_predicates | added_predicates):
            continue
        if alternatives := alternate_state_effects(candidate, needed):
            uncertain.append({"action": schema.name, "required_predicates": sorted(needed),
                              "alternative_encodings": alternatives})
            continue
        positive = {candidate.interface.canonicalize(f) for f in schema.pre_pos}
        negative = {candidate.interface.canonicalize(f) for f in schema.pre_neg}
        added = {candidate.interface.canonicalize(f) for f in schema.add_eff}
        deleted = {candidate.interface.canonicalize(f) for f in schema.del_eff} - added
        if added <= positive and all(f in negative or f[0] not in initial_predicates | added_predicates for f in deleted):
            matches.append({"kind": "semantic_noop", "action": schema.name, "family": family})
        elif family in {"wipe", "stir", "cut"} and added | deleted and all(
                fact[0] in STRUCTURAL for fact in added | deleted):
            # These operations transform a surface/material, rather than merely
            # releasing a tool or moving its support. Unknown state encodings
            # and negative state effects are deliberately left unclassified.
            matches.append({"kind": "semantic_effect_missing", "action": schema.name,
                            "family": family, "effects": [list(f) for f in sorted(added | deleted)]})
    if unsolved:
        mapping = mapping if mapping is not None else ({o.name: o.name for o in candidate.objects} if candidate is gt else {})
        for subject in sorted({f[1] for f in candidate.goal_positive if f[0] == "in" and len(f) == 3}):
            goals = {f for f in candidate.goal_positive if f[0] == "in" and f[1] == subject}
            initial = {f for f in candidate.facts if f[0] == "in" and f[1] == subject}
            if len(goals) < 2 or len(initial) > 1:
                continue
            kinds = set(candidate.object(subject).kinds)
            destinations = {mapping.get(f[2]) for f in goals}
            supported = {f[2] for f in gt.goal_positive if len(f) == 3 and f[0] in SPATIAL
                         and (mapping.get(subject) == f[1] or kinds & set(gt.object(f[1]).kinds))}
            if "water" in kinds:
                supported.update(f[1] for f in gt.goal_positive if len(f) == 2 and f[0] == "contains_water")
            if None in destinations or not destinations <= supported:
                continue  # An arbitrary impossible extra goal does not prove an operator error.
            preserving = True
            for schema in candidate.schemas.values():
                # Match the subject only; destinations may be different constants.
                adds = {f for f in schema.add_eff if f[0] == "in" and len(f) == 3
                        and (f[1].startswith("?") or f[1] == subject)}
                if len(adds) > 1:
                    preserving = False
                    break
                for added in adds:
                    if added in schema.pre_pos:
                        continue
                    if not any(f[0] == "in" and len(f) == 3 and f[1] == added[1]
                               and f in schema.pre_pos for f in schema.del_eff):
                        preserving = False
            if preserving:
                matches.append({"kind": "exclusive_resource_goals", "object": subject,
                                "goals": [list(f) for f in sorted(goals)],
                                "reason": "initially at most one containment; every producer preserves that invariant"})
    return {"matched": bool(matches), "matches": matches, "uncertain": uncertain}


def project_water(world, facts):
    """Bridge the two existing encodings: in(water, bowl) and has_water(bowl)."""
    water = {obj.name for obj in world.objects if "water" in obj.kinds}
    return set(facts) | {("contains_water", fact[2]) for fact in facts
                         if len(fact) == 3 and fact[0] == "in" and fact[1] in water}


def project_closures(facts, owners):
    projected = set(facts)
    for fact in facts:
        if len(fact) == 2 and fact[0] in {"open", "closed"}:
            parent = owners.get(fact[1], fact[1])
            projected.update((fact[0], obj) for obj in {parent} | {lid for lid, owner in owners.items() if owner == parent}
                             if not any((p, obj) in facts for p in {"open", "closed"}))
    return projected


def check_causal_goals(candidate, gt, mapping, *, unsolved, final_state=None,
                       executed_effects=None, candidate_owners=None, gt_owners=None):
    """Compare shared state goals and prove impossible, unsupported extra goals."""
    inverse = {target: source for source, target in mapping.items()}
    signatures = {(p.canonical_name, p.arity) for p in candidate.interface.descriptions if not p.static}
    water = {obj.name for obj in candidate.objects if "water" in obj.kinds}
    if water and ("in", 2) in signatures:
        signatures.add(("contains_water", 1))
    candidate_owners = candidate_owners or {}
    if final_state is not None:
        final_state = project_water(candidate, project_closures(final_state, candidate_owners))
    matches, unsupported, unachievable, uncertain = [], [], [], []
    candidate_plan = executed_effects.get("candidate_plan", []) if executed_effects is not None else []

    def alternate_encoding(fact):
        evidence = alternate_state_effects(candidate, {fact[0]}, actions=candidate_plan, objects=set(fact[1:]))
        opposite = (OPPOSITES.get(fact[0]), *fact[1:])
        if opposite in candidate.goal_negative:
            evidence.append({"negative_goal": ["not", *opposite]})
        opaque_goals = [list(f) for f in sorted(candidate.goal_positive | candidate.goal_negative)
                        if f[0] not in STATE_PREDICATES and not f[0].startswith(("kind:", "static:"))
                        and (not f[1:] or set(f[1:]) & set(fact[1:]))]
        if opaque_goals:
            evidence.append({"unknown_goals": opaque_goals})
        return evidence

    gt_required_final = required_final_facts(gt)
    for positive, goals in ((True, gt.goal_positive), (False, gt.goal_negative)):
        for expected in sorted(goals):
            if expected[0].startswith(("kind:", "static:")):
                continue
            fact = mapped_fact(expected, inverse)
            if fact is None or (fact[0], len(fact) - 1) not in signatures:
                continue
            actual = project_closures(candidate.goal_positive if positive else candidate.goal_negative, candidate_owners)
            if positive or len(water) == 1:
                actual = project_water(candidate, actual)
            if (equivalent(fact, actual) if positive else fact in actual):
                continue
            opposite = ((OPPOSITES[fact[0]], *fact[1:]) if fact[0] in OPPOSITES else None)
            wrong = fact in (candidate.goal_negative if positive else candidate.goal_positive)
            if positive and opposite and opposite in candidate.goal_positive:
                wrong = True
            item = {"kind": "wrong_goal_state" if wrong else "omitted_goal",
                    "gt_goal": list(expected), "candidate_fact": list(fact), "positive": positive}
            if wrong:
                matches.append(item)
            elif final_state is not None and ((not equivalent(fact, final_state)) if positive else fact in final_state):
                # Omission alone is insufficient: the judged trace also misses it.
                alternatives = alternate_encoding(fact) if positive else []
                if alternatives:
                    uncertain.append({**item, "alternative_encodings": alternatives})
                else:
                    matches.append(item)
    if unsolved:
        for positive, goals in ((True, candidate.goal_positive), (False, candidate.goal_negative)):
            for fact in sorted(goals):
                already = fact in candidate.facts if positive else fact not in candidate.facts
                if already or producers(candidate, fact, positive=positive):
                    continue
                expected = mapped_fact(fact, mapping)
                item = {"kind": "goal_without_producer", "candidate_fact": list(fact), "positive": positive}
                item["has_matching_effect"] = bool(producers(candidate, fact, positive=positive, check_kinds=False))
                item["reference_supported"] = bool(expected is not None and (
                    equivalent(expected, gt_required_final) if positive else expected in gt.goal_negative))
                unachievable.append(item)
                if expected is None:
                    continue
                gt_signatures = {(gt.interface.canonicalize((p.raw_name, *("?x" for _ in range(p.arity))))[0], p.arity)
                                 for p in gt.interface.descriptions}
                if (expected[0], len(expected) - 1) not in gt_signatures:
                    continue  # An absent reference vocabulary is unknown, not a negative fact.
                gt_initial = project_closures(gt.facts, gt_owners or {})
                gt_already = equivalent(expected, gt_initial) if positive else expected not in gt_initial
                gt_goals = gt_required_final if positive else gt.goal_negative
                desired = equivalent(expected, gt_goals) if positive else expected in gt_goals
                if not gt_already and not desired and not reference_producers(gt, expected, positive=positive, owners=gt_owners):
                    unsupported.append(item)
    if executed_effects is not None:
        # A reference goal can record an event whose final location was already
        # true initially. Keeping only that location loses the required action.
        actual_effects = {tuple(f) for f in executed_effects["candidate"]}
        for expected in sorted(gt.goal_positive):
            if (expected in gt.facts or not expected[0].startswith(EVENT_PREFIXES)
                or (expected[0], len(expected) - 1) in signatures):
                continue
            # Intersect every possible event producer, not just the one chosen
            # by the saved reference plan. An alternate implementation may
            # record the event without making the same spatial change.
            achievers = []
            for schema in gt.schemas.values():
                for effect in schema.add_eff:
                    binding = unify(gt.interface.canonicalize(effect), expected)
                    if binding is not None:
                        achievers.append({substitute(gt.interface.canonicalize(f), binding) for f in schema.add_eff})
            if not achievers:
                continue
            for event in sorted(set.intersection(*achievers)):
                if event[0] not in SPATIAL:
                    continue
                if any(arg.startswith("?") for arg in event[1:]):
                    continue
                fact = mapped_fact(event, inverse)
                if fact and equivalent(fact, candidate.facts) and equivalent(fact, candidate.goal_positive) and not equivalent(fact, actual_effects):
                    item = {"kind": "missing_goal_witness", "gt_goal": list(expected),
                            "candidate_fact": list(fact), "positive": True}
                    alternatives = alternate_encoding(fact)
                    # A declared transfer macro may restore the initial area
                    # without exposing the same intermediate relation/effect.
                    transfers = []
                    for action in candidate_plan:
                        effects = {candidate.interface.canonicalize(f) for f in action.add_eff | action.del_eff}
                        moved = {f[2] if f[0] == "holding" else f[1] for f in effects
                                 if len(f) == 3 and f[0] in SPATIAL | {"holding", "inserted", "under"}}
                        if fact[1] in (moved or set(action.args)) and named_family(candidate.schemas[action.name]) in {
                                "pick", "place_on", "place_in", "insert", "pour", "scoop",
                                "remove_closure", "replace_closure"}:
                            transfers.append(action.name)
                    if alternatives or transfers:
                        uncertain.append({**item, "alternative_encodings": alternatives, "possible_transfers": transfers})
                    else:
                        matches.append(item)
    return {"matched": bool(matches or unsupported), "matches": matches + unsupported,
            "unachievable": unachievable, "uncertain": uncertain}


def reference_guards(gt, family, anchors, sources, *, limit=4096, details=False):
    """Intersect guards across all statically possible GT implementations.

    The reference plan is one implementation, not proof that its incidental
    ordering is mandatory. Unbound roles and alternate GT actions participate
    in this intersection; a grounding budget exhaustion yields no conclusion.
    With family=None, consider every statically feasible producer of the anchor.
    """
    alternatives = []
    names = {o.name for o in gt.objects}
    def canonical(fact):
        return fact if fact[0] == "=" else gt.interface.canonicalize(fact)

    for schema in gt.schemas.values():
        if family is not None and map_schema(schema)[0] != family:
            continue
        effects = {gt.interface.canonicalize(f) for f in schema.add_eff}
        preconditions = {canonical(f) for f in schema.pre_pos}
        bindings = [binding for effect in effects for anchor in anchors
                    if (binding := unify(effect, anchor)) is not None]
        if family is not None and family.value == "pick" and sources:
            bindings = [{**binding, **extra} for binding in bindings for pre in preconditions for source in sources
                        if ((pre[0] in SPATIAL and source[0] in SPATIAL) or pre[0] == source[0]) and (extra := unify(
                            substitute((source[0], *pre[1:]), binding), source)) is not None]
        for binding in bindings:
            if not compatible_kinds(gt, schema, binding):
                continue
            variables = [p for p in schema.params if p not in binding]
            domains = []
            for variable in variables:
                domain = set(names)
                for pre in preconditions:
                    if len(pre) == 2 and pre[1] == variable and pre[0].startswith("kind:"):
                        domain &= {f[1] for f in gt.facts if len(f) == 2 and f[0] == pre[0]}
                domains.append(sorted(domain))
            combinations = prod(map(len, domains))
            limit -= combinations
            if limit < 0:
                return {} if details else set()
            for values in product(*domains):
                grounded = {**binding, **dict(zip(variables, values))}
                pre = {substitute(f, grounded) for f in preconditions}
                if any(f[0].startswith(("kind:", "static:")) and f not in gt.facts for f in pre):
                    continue
                if any(f[0] == "=" and f[1] != f[2] for f in pre):
                    continue
                negative = {substitute(canonical(f), grounded) for f in schema.pre_neg}
                if any(f[0] == "=" and f[1] == f[2] for f in negative):
                    continue
                if any(f[0].startswith(("kind:", "static:")) and f in gt.facts for f in negative):
                    continue
                added = {substitute(f, grounded) for f in effects}
                deleted = {substitute(gt.interface.canonicalize(f), grounded) for f in schema.del_eff} - added
                alternatives.append((pre, negative, added, deleted))
    if not alternatives:
        return {} if details else set()
    if not details:
        return set.intersection(*(item[0] for item in alternatives))
    return {"positive": set.intersection(*(item[0] for item in alternatives)),
            "negative": set.intersection(*(item[1] for item in alternatives)),
            "add": set.intersection(*(item[2] for item in alternatives)),
            "delete": set.intersection(*(item[3] for item in alternatives)),
            "possible_add": set.union(*(item[2] for item in alternatives))}


def unique_state_proxies(candidate, gt, mapping):
    """An abstract device can expose its sole stateful component's power state.

    This changes unary state comparison only, never object identity or spatial
    locations. Multiple stateful components, or an explicit whole state, veto it.
    """
    family = {"power_on", "power_off"}
    represented = {f[1] for f in gt.facts if len(f) == 2 and f[0] in family}
    for obj in gt.objects:
        if any(producers(gt, (predicate, obj.name)) for predicate in family):
            represented.add(obj.name)
    proxies = {}
    for source, target in mapping.items():
        if target in represented or not any((predicate, source) in candidate.facts or producers(candidate, (predicate, source)) for predicate in family):
            continue
        children = {f[1] for f in gt.facts if len(f) == 3 and f[0] == "static:part_of"
                    and f[2] == target and f[1] in represented}
        if len(children) == 1:
            child = next(iter(children))
            if child not in set(mapping.values()) - {target}:
                for predicate in family:
                    proxies[(predicate, source)] = child
    return proxies


def check_plan_semantics(candidate, gt, mapping, directory, reference, *, strict=False,
                         candidate_owners=None, gt_owners=None):
    """Replay locally, then check explicit GT guards for matched state transitions.

    A different reference order is not itself an error. Guard conflicts must
    apply to every matched GT action and be supported by a reference invariant
    that also excludes a compound implementation of the proposed transition.
    """
    result = {"matched": False, "matches": [], "final_state": None, "executed_effects": None}
    try:
        plan, final = replay_saved_plan(directory, candidate)
        replay_saved_plan(reference, gt)
        raw_state = set(candidate.identity_facts)
        canonical_final = {candidate.interface.canonicalize(f) for f in final}
        result["final_state"] = [list(f) for f in sorted(canonical_final)]
        result["plan_actions"] = [action.name for action in plan]
        result["executed_effects"] = [list(f) for f in sorted({candidate.interface.canonicalize(f)
                                                              for a in plan for f in a.add_eff})]
        state = {candidate.interface.canonicalize(f) for f in raw_state}
        proxies = unique_state_proxies(candidate, gt, mapping) if strict else {}
        if proxies:
            result["state_proxies"] = [{"predicate": p, "candidate": obj, "gt_component": target}
                                       for (p, obj), target in sorted(proxies.items())]

        def project(fact):
            if len(fact) == 2 and (target := proxies.get((fact[0], fact[1]))) is not None:
                return (fact[0], target)
            return mapped_fact(fact, mapping)

        def project_state(facts):
            facts = project_water(candidate, facts) if strict else facts
            if strict:
                facts = project_closures(facts, candidate_owners or {})
            mapped = {f for raw in facts if (f := project(raw)) is not None}
            return project_closures(mapped, gt_owners or {}) if strict else mapped

        def related_objects(obj):
            # A component's operation may change its whole's state, and vice
            # versa. Such an operation vetoes an untouched-object frame proof.
            related = {obj}
            for link in gt.facts:
                if len(link) == 3 and link[0] in {"static:part_of", "static:drawer_of", "static:attached_to"} and obj in link[1:]:
                    related.update(link[1:])
            parent = (gt_owners or {}).get(obj, obj)
            return related | {parent} | {lid for lid, owner in (gt_owners or {}).items() if owner == parent}

        inverse = {target: source for source, target in mapping.items()}
        known_negative, touched, ambiguous = {}, set(), set()
        guard_candidates = []
        for index, action in enumerate(plan, 1):
            added = {candidate.interface.canonicalize(f) for f in action.add_eff}
            deleted = {candidate.interface.canonicalize(f) for f in action.del_eff}
            raw_next = (raw_state - action.del_eff) | action.add_eff
            next_state = {candidate.interface.canonicalize(f) for f in raw_next}
            new_conflicts = [c for c in contradictions(next_state) if c not in contradictions(state)]
            if new_conflicts:
                result["matches"].append({"kind": "inconsistent_effects", "step": index,
                                          "action": action.name, "conflicts": new_conflicts})
            family = map_schema(candidate.schemas[action.name])[0]
            anchors = {mapped for f in added - state if f[0] in GUARDS - {"hand_free"}
                       and (mapped := project(f)) is not None}
            sources = {f for raw in deleted if raw[0] in SPATIAL | {"inserted"}
                       and (f := project(raw)) is not None}
            unknown_source = family.value == "pick" and any(
                raw[0] in SPATIAL | {"inserted"} and project(raw) is None for raw in deleted)
            mapped_state, mapped_added = project_state(state), project_state(added)
            mapped_deleted = mapped_state - project_state(next_state)
            unknown_effect = any(f[0] not in STATE_PREDICATES and not f[0].startswith(("kind:", "static:"))
                                 for f in added | deleted)
            negative_processing = any(OPPOSITES.get(f[0]) in SEMANTIC_STATES for f in deleted - added)
            if unknown_effect or negative_processing:
                ambiguous.update(mapping[arg] for arg in action.args if arg in mapping)
                if unknown_effect and not action.args:
                    ambiguous.update(mapping.values())
            restoration_anchors = {f for raw in added if raw[0] in STATE_PREDICATES - {"hand_free", "clear"}
                                   and (f := project(raw)) is not None}
            restoration = reference_guards(gt, family, restoration_anchors, sources, details=True) if strict and known_negative else {}
            for fact, evidence in list(known_negative.items()):
                uncertain_restore = (unknown_effect or negative_processing) and any(related_objects(arg) & ambiguous for arg in fact[1:])
                if (fact in mapped_added or fact in restoration.get("possible_add", set()) or uncertain_restore
                    or any(tuple(proxy) in mapped_added for proxy in evidence.get("shared_deletions", []))):
                    del known_negative[fact]
            context = {}
            if anchors and not unknown_source:
                if strict:
                    context = reference_guards(gt, family, anchors, sources, details=True)
                    required = context.get("positive", set())
                else:
                    required = reference_guards(gt, family, anchors, sources)
                for fact in sorted(required):
                    if equivalent(fact, mapped_state) or equivalent(fact, mapped_added):
                        continue
                    kind, evidence = None, {}
                    if fact[0] in GUARDS and contradicted(fact, mapped_state):
                        kind = "violated_gt_guard"
                    elif strict and fact in known_negative:
                        kind, evidence = "lost_required_state", {"lost_at": known_negative[fact]}
                    elif (strict and fact[0] in SEMANTIC_STATES and not set(fact[1:]) & ambiguous
                          and (local := mapped_fact(fact, inverse)) is not None and producers(candidate, local)):
                        kind = "missing_semantic_precondition"
                    elif (strict and len(fact) == 2 and fact[0] in OPPOSITES
                          and fact[1] in inverse and not related_objects(fact[1]) & touched
                          and (OPPOSITES[fact[0]], fact[1]) in gt.facts):
                        kind, evidence = "unrestored_initial_state", {"gt_initial_opposite": [OPPOSITES[fact[0]], fact[1]]}
                    if kind:
                        post = project_state(next_state)
                        if kind == "violated_gt_guard":
                            observations = [list(f) for f in sorted(post) if contradicted(fact, {f})]
                        elif kind == "unrestored_initial_state":
                            observations = [evidence["gt_initial_opposite"]]
                        else:
                            observations = [["not", *fact]]
                        guard_candidates.append({"kind": kind, "step": index,
                                                 "action": action.name, "gt_precondition": list(fact),
                                                 "mapped_state": [list(f) for f in sorted(mapped_state) if f[0] in GUARDS],
                                                 "transition_anchors": [list(f) for f in sorted(anchors)],
                                                 "post_observations": observations, **evidence})
                for fact in sorted(context.get("negative", set())):
                    if fact in mapped_state and fact in project_state(next_state):
                        guard_candidates.append({"kind": "violated_negative_gt_guard", "step": index,
                                                 "action": action.name, "gt_precondition": ["not", *fact],
                                                 "transition_anchors": [list(f) for f in sorted(anchors)],
                                                 "post_observations": [list(fact)]})
            if strict:
                # Active processes and object handling must describe a jointly
                # realizable post-state, even when the source vocabulary differs
                # (e.g. a tray in one model and an outlet in the other).
                post = project_state(next_state)
                new = post - mapped_state
                for anchor in sorted(new):
                    if anchor[0] not in {"power_on", "power_off", "holding", "on", "in", "under"}:
                        continue
                    observations = [f for f in sorted(post - {anchor})
                                    if ((anchor[0] in {"power_on", "power_off"}
                                         and f[0] in {"holding", "on", "in", "under"})
                                        or (anchor[0] in {"holding", "on", "in", "under"}
                                            and f[0] in {"power_on", "power_off"}))]
                    if observations:
                        guard_candidates.append({"kind": "reference_state_mutex", "step": index,
                                                 "action": action.name, "transition_anchors": [list(anchor)],
                                                 "post_observations": [list(f) for f in observations]})
                shared = {f for f in mapped_deleted & context.get("delete", set())
                          if f[0] not in {"hand_free", "holding"}}
                for fact in context.get("delete", set()):
                    proxies_deleted = {f for f in shared if any(related_objects(arg) & set(f[1:]) for arg in fact[1:])}
                    if proxies_deleted and fact in gt.facts:
                        known_negative[fact] = {"step": index, "action": action.name,
                                                "shared_deletions": [list(f) for f in sorted(proxies_deleted)]}
                for fact in mapped_state - project_state(next_state):
                    known_negative[fact] = {"step": index, "action": action.name}
                for fact in context.get("possible_add", set()):
                    known_negative.pop(fact, None)
                touched.update(target for arg in action.args
                               if (target := mapping.get(arg)) is not None)
            state = next_state
            raw_state = raw_next
        if guard_candidates:
            certified, reason = certify_guard_conflicts(gt, reference, guard_candidates)
            result["matches"].extend(certified)
            result["guard_candidates"] = guard_candidates
            if reason:
                result["guard_proof_unavailable"] = reason
        result["matched"] = bool(result["matches"])
    except (OSError, ValueError, KeyError) as error:
        result["unavailable"] = str(error)
    return result


# Bounded reachability proofs. The pair relaxation overapproximates reachable
# states: unreachable facts/pairs are proofs, reachable ones are not plans.
# Grounding and closure budgets fail open.

def _bits(values):
    result = 0
    for value in values:
        result |= 1 << value
    return result


def _indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


@dataclass
class PairReachability:
    complete: bool = False
    reason: str = ""
    actions: list = field(default_factory=list)
    dead_schemas: list = field(default_factory=list)
    indices: dict = field(default_factory=dict)
    pairs: list = field(default_factory=list)
    rounds: int = 0

    def index(self, fact, positive=True):
        index = self.indices.get(tuple(fact))
        return None if index is None else index + (not positive)

    def possible(self, fact, positive=True):
        if not self.complete:
            return None
        index = self.index(fact, positive)
        # An atom outside the complete grounded universe is never established.
        return not positive if index is None else bool(self.pairs[index] & (1 << index))

    def together(self, left, right):
        if not self.complete:
            return None
        a, b = self.index(*left), self.index(*right)
        if a is None or b is None:
            return self.possible(*left) and self.possible(*right)
        return bool(self.pairs[a] & (1 << b))


def analyze_reachability(world, problem, *, grounding_limit=20000, atom_limit=1500,
                         work_limit=1000000, round_limit=128):
    result = PairReachability()
    initial = set(problem.init_state)
    objects = frozenset(problem.objects)
    changed = {f[0] for s in world.schemas.values() for f in s.add_eff | s.del_eff}
    static = set(world.schemas.predicate_arities) - changed
    for schema in world.schemas.values():
        domains, empty = [], []
        for variable in schema.params:
            domain = set(objects)
            kinds = [f[0] for f in schema.pre_pos if len(f) == 2 and f[0] in static and f[1] == variable]
            for kind in kinds:
                domain &= {f[1] for f in initial if len(f) == 2 and f[0] == kind}
            domains.append(sorted(domain))
            if not domain:
                empty.append({"parameter": variable, "predicates": sorted(kinds)})
        grounding_limit -= prod(map(len, domains))
        if grounding_limit < 0:
            result.reason = "grounding_budget_exhausted"
            return result
        count = len(result.actions)
        for arguments in product(*domains):
            action = ground_plan([(schema.name, list(arguments))], world.schemas, objects)[0]
            if any(f[0] in static and f not in initial for f in action.pre_pos):
                continue
            if any(f[0] in static and f in initial for f in action.pre_neg):
                continue
            if any(f[1] != f[2] for f in action.equality_preconditions):
                continue
            if any(f[1] == f[2] for f in action.inequality_preconditions):
                continue
            result.actions.append(action)
        if count == len(result.actions):
            result.dead_schemas.append({"action": schema.name, "empty_parameters": empty,
                                        "static_preconditions": [list(f) for f in sorted(schema.pre_pos) if f[0] in static]})
    universe = initial | problem.goal_positive | problem.goal_negative
    for action in result.actions:
        universe |= action.pre_pos | action.pre_neg | action.add_eff | action.del_eff
    if len(universe) > atom_limit:
        result.reason = "atom_budget_exhausted"
        return result
    result.indices = {fact: 2 * index for index, fact in enumerate(sorted(universe))}
    size = 2 * len(universe)
    starting = [index + (fact not in initial) for fact, index in result.indices.items()]
    starting_bits = _bits(starting)
    result.pairs = [starting_bits if index in starting else 0 for index in range(size)]
    encoded = []
    for action in result.actions:
        pre = [result.index(f) for f in action.pre_pos] + [result.index(f, False) for f in action.pre_neg]
        added = [result.index(f) for f in action.add_eff] + [result.index(f, False) for f in action.del_eff]
        encoded.append((pre, _bits(pre), added, _bits(added), _bits(i ^ 1 for i in added)))
    for iteration in range(round_limit):
        result.rounds = iteration + 1
        changed = False
        for pre, pre_bits, added, added_bits, deleted_bits in encoded:
            if any(result.pairs[p] & pre_bits != pre_bits for p in pre):
                continue
            surviving = (1 << size) - 1
            for p in pre:
                surviving &= result.pairs[p]
            if not pre:
                surviving = _bits(i for i in range(size) if result.pairs[i] & (1 << i))
            surviving &= ~deleted_bits
            work_limit -= surviving.bit_count() + len(added)
            if work_limit < 0:
                result.reason = "pair_closure_budget_exhausted"
                return result
            for effect in added:
                before = result.pairs[effect]
                result.pairs[effect] |= surviving | added_bits
                changed |= before != result.pairs[effect]
            for fact in _indices(surviving):
                before = result.pairs[fact]
                result.pairs[fact] |= added_bits
                changed |= before != result.pairs[fact]
        if not changed:
            result.complete = True
            return result
    result.reason = "pair_closure_round_budget_exhausted"
    return result


def _literal_text(signed):
    fact, positive = signed
    return list(fact) if positive else ["not", *fact]


def _blocked_conditions(analysis, action):
    pre = [(f, True) for f in sorted(action.pre_pos)] + [(f, False) for f in sorted(action.pre_neg)]
    missing = [f for f in pre if not analysis.possible(*f)]
    if missing:
        return [_literal_text(missing[0])]
    for index, left in enumerate(pre):
        for right in pre[index + 1:]:
            if not analysis.together(left, right):
                return [_literal_text(left), _literal_text(right)]
    return []


def certify_guard_conflicts(gt, reference, candidates, *, analysis=None):
    """A violated atomic guard alone cannot disprove a compound action.

    Require a reference invariant: the proposed post-transition observation
    and an achieved anchor cannot coexist under any reference action sequence.
    Both observations must be individually reachable. Thus an incomplete GT
    vocabulary/capability is not mistaken for a physical impossibility.
    """
    if analysis is None:
        analysis = analyze_reachability(gt, parse_problem_model(reference / "problem.pddl", gt.schemas))
    if not analysis.complete:
        return [], analysis.reason

    produced = {f for action in analysis.actions for f in action.add_eff}

    def raw_variants(fact, positive, *, opening=False, require_producer=False):
        variants = []
        for description in gt.interface.descriptions:
            if description.arity != len(fact) - 1:
                continue
            args = list(fact[1:])
            for canonical_index, raw_index in enumerate(description.argument_order):
                args[raw_index] = fact[canonical_index + 1]
            raw = (description.raw_name, *args)
            canonical = gt.interface.canonicalize(raw)
            agrees = equivalent(fact, {canonical}) if positive else fact == canonical
            if opening and fact[0] == "blocks_opening" and canonical == ("blocks", *fact[1:]):
                agrees = True
            if agrees and analysis.possible(raw, positive):
                variants.append(raw)
        # Never discard a reachable alternate spelling just because that
        # spelling has no producer; it might permit the allegedly invalid pair.
        return variants if not require_producer or all(f in produced for f in variants) else []

    certified = []
    for item in candidates:
        for anchor in item["transition_anchors"]:
            state_conflict = item["kind"] == "reference_state_mutex"
            anchors = raw_variants(tuple(anchor), True, require_producer=state_conflict)
            if not anchors:
                continue
            for signed in item["post_observations"]:
                positive = signed[0] != "not"
                observation = tuple(signed if positive else signed[1:])
                opening = anchor[0] in {"open", "unblocked"} and len(observation) == 3 and observation[2] == anchor[1]
                observations = raw_variants(observation, positive, opening=opening,
                                            require_producer=state_conflict)
                if observations and all(not analysis.together((a, True), (b, positive))
                                        for a in anchors for b in observations):
                    item["reference_mutex_proof"] = {"anchor": anchor, "observation": signed,
                                                       "grounded_actions": len(analysis.actions)}
                    certified.append(item)
                    break
            if item in certified:
                break
    return certified, ""


def check_initial_access(candidate, gt, mapping, reference):
    """Explicit clear/unblocked claims conflict with a represented obstacle.

    Do not infer an obstruction from the mere absence of clear. Require both
    candidate facts, a shared GT initial obstacle, and a reference mutex proof.
    """
    mapped = {f for raw in candidate.facts if (f := mapped_fact(raw, mapping)) is not None}
    candidates = []
    for raw in sorted(candidate.facts):
        if len(raw) != 2 or raw[0] not in {"clear", "unblocked"}:
            continue
        anchor = mapped_fact(raw, mapping)
        if anchor is None or anchor in gt.facts:
            continue
        obstacles = [f for f in sorted(mapped) if contradicted(anchor, {f})
                     and (f in gt.facts or (f[0] == "blocks_opening" and ("blocks", *f[1:]) in gt.facts))]
        if obstacles:
            candidates.append({"kind": "initial_access_conflict", "candidate_fact": list(raw),
                               "transition_anchors": [list(anchor)],
                               "post_observations": [list(f) for f in obstacles], "status": "fail"})
    if not candidates:
        return {"matched": False, "matches": []}
    certified, reason = certify_guard_conflicts(gt, reference, candidates)
    return {"matched": bool(certified), "matches": certified, "reason": reason}


def check_effect_reachability(candidate, gt, directory):
    """Discard processing effects that provably cannot change any state.

    A delete of wet(plate) does not implement processing if only towels can
    ever be wet. Unknown effects and alternate state encodings remain evidence
    against this rule. Complete grounding is required for every conclusion.
    """
    required = required_processing(gt)
    reference_families = {named_family(s) for s in gt.schemas.values()
                          if any(gt.interface.canonicalize(f)[0] in required for f in s.add_eff)}
    selected = [s for s in candidate.schemas.values()
                if named_family(s) in {"wipe", "stir", "cut"} & reference_families
                and any(candidate.interface.canonicalize(f)[0] not in STRUCTURAL
                        for f in s.add_eff | s.del_eff)]
    result = {"available": False, "matches": []}
    if not selected:
        return result
    analysis = analyze_reachability(candidate, parse_problem_model(directory / "problem.pddl", candidate.schemas))
    result.update(available=analysis.complete, reason=analysis.reason)
    if not analysis.complete:
        return result
    viable = [a for a in analysis.actions if not _blocked_conditions(analysis, a)]
    effective_families, effective_actions = set(), []
    for action in viable:
        added = {f for f in action.add_eff if f not in action.pre_pos and analysis.possible(f, False)}
        deleted = {f for f in action.del_eff if analysis.possible(f)}
        effective_actions.append(replace(action, add_eff=added, del_eff=deleted))
        if any(candidate.interface.canonicalize(f)[0] not in STRUCTURAL for f in added | deleted):
            effective_families.add(named_family(candidate.schemas[action.name]))
    for schema in selected:
        if named_family(schema) in effective_families:
            continue
        needed = {gt.interface.canonicalize(f)[0] for s in gt.schemas.values() if named_family(s) == named_family(schema)
                  for f in s.add_eff if gt.interface.canonicalize(f)[0] in required}
        if any(f[0] in needed for f in candidate.facts) or any(
                candidate.interface.canonicalize(f)[0] in needed for s in candidate.schemas.values() for f in s.add_eff):
            continue
        if alternate_state_effects(candidate, needed, actions=effective_actions):
            continue
        instances = [a for a in viable if a.name == schema.name]
        ineffective, remaining = set(), set()
        for action in instances:
            for fact in action.add_eff:
                if fact in action.pre_pos or not analysis.possible(fact, False):
                    ineffective.add((fact, True))
                else:
                    remaining.add(candidate.interface.canonicalize(fact))
            for fact in action.del_eff:
                if not analysis.possible(fact):
                    ineffective.add((fact, False))
                else:
                    remaining.add(candidate.interface.canonicalize(fact))
        semantic = [(fact, positive) for fact, positive in ineffective
                    if candidate.interface.canonicalize(fact)[0] not in STRUCTURAL]
        if semantic and remaining and all(f[0] in STRUCTURAL for f in remaining):
            result["matches"].append({"kind": "unreachable_processing_effect", "action": schema.name,
                                      "family": named_family(schema), "grounded_instances": len(instances),
                                      "ineffective_effects": [{"fact": list(f), "positive": p}
                                                              for f, p in sorted(semantic)],
                                      "remaining_effects": [list(f) for f in sorted(remaining)]})
    return result


def check_raw_init(candidate, gt, mapping, problem):
    """Canonical aliases cannot supply a literal missing from the actual PDDL."""
    inverse = {target: source for source, target in mapping.items()}
    initial = set(candidate.identity_facts)
    changed = {f[0] for s in candidate.schemas.values() for f in s.add_eff | s.del_eff}
    necessary = necessary_facts(candidate, problem=problem)
    matches = []
    for expected in sorted(gt.facts):
        fact = mapped_fact(expected, inverse)
        if fact is None:
            continue
        for description in candidate.interface.descriptions:
            if description.arity != len(fact) - 1:
                continue
            arguments = list(fact[1:])
            for canonical_index, raw_index in enumerate(description.argument_order):
                arguments[raw_index] = fact[canonical_index + 1]
            raw = (description.raw_name, *arguments)
            if candidate.interface.canonicalize(raw) != fact or raw in initial:
                continue
            if raw[0] in changed or raw not in necessary:
                continue
            users = [s.name for s in candidate.schemas.values() if any(
                (binding := unify(pre, raw)) is not None and compatible_kinds(candidate, s, binding)
                for pre in s.pre_pos)]
            if raw in problem.goal_positive:
                users.append(":goal")
            # A representation alias is already present, which is why the old
            # canonical-only comparison missed this missing raw declaration.
            aliases = [list(f) for f in sorted(initial) if candidate.interface.canonicalize(f) == fact]
            if users and aliases:
                matches.append({"kind": "required_raw_fact_missing", "raw_fact": list(raw),
                                "canonical_fact": list(fact), "gt_fact": list(expected),
                                "present_aliases": aliases, "used_by": users})
    return matches


def check_spatial_cardinality(candidate, gt, mapping, directory):
    """Count distinct required instances, using all final subjects as an upper bound.

    Repeated conjuncts and collective names do not create new objects under the
    evaluator's one-to-one object contract. No names or number words are parsed.
    """
    try:
        plan, raw_final = replay_saved_plan(directory, candidate)
        final = {candidate.interface.canonicalize(f) for f in raw_final}
    except (OSError, ValueError, KeyError):
        final = None
    inverse = {target: source for source, target in mapping.items()}
    matches = []
    for area in sorted({f[2] for f in gt.goal_positive if len(f) == 3 and f[0] in SPATIAL}):
        source = inverse.get(area)
        required = {f[1] for f in gt.goal_positive if len(f) == 3 and f[0] in SPATIAL and f[2] == area}
        if source is None or len(required) < 2:
            continue
        goals = {f[1] for f in candidate.goal_positive if len(f) == 3 and f[0] in SPATIAL and f[2] == source}
        if len(goals) >= len(required):
            continue
        if final is not None:
            actual = {f[1] for f in final if len(f) == 3 and f[0] in SPATIAL and f[2] == source}
            possible = {arg for f in final if len(f) >= 3 and f[0] not in STATE_PREDICATES and source in f[1:]
                        for arg in f[1:] if arg != source}
            possible.update(arg for f in candidate.goal_positive | candidate.goal_negative
                            if f[0] not in STATE_PREDICATES and not f[0].startswith(("kind:", "static:"))
                            for arg in f[1:])
            for action in plan:
                if alternate_state_effects(candidate, SPATIAL, actions=[action]):
                    possible.update(action.args or [obj.name for obj in candidate.objects])
            upper_bound = (actual | possible) - {source}
            if len(upper_bound) < len(required):
                matches.append({"kind": "spatial_goal_cardinality", "gt_area": area, "candidate_area": source,
                                "required_objects": sorted(required), "candidate_goal_objects": sorted(goals),
                                "candidate_final_objects": sorted(upper_bound),
                                "candidate_explicit_final_objects": sorted(actual),
                                "possible_alternate_objects": sorted(possible - {source})})
        else:
            kinds = [set(gt.object(obj).kinds) for obj in required]
            common = set.intersection(*kinds) - {"water", "milk", "soup", "liquid", "detergent", "disinfectant"}
            for kind in sorted(common):
                instances = {obj.name for obj in candidate.objects if kind in obj.kinds}
                # Kind membership can itself be incompletely declared. Count
                # every object not already assigned to a different GT object
                # as a possible instance, irrespective of its candidate name/type.
                available = {obj.name for obj in candidate.objects
                             if obj.name not in mapping or mapping[obj.name] in required}
                if len(available) < len(required):
                    matches.append({"kind": "declared_instance_deficit", "gt_area": area, "candidate_area": source,
                                    "kind_name": kind, "required_objects": sorted(required),
                                    "candidate_objects": sorted(available), "candidate_typed_objects": sorted(instances)})
    return matches


def check_semantic_cardinality(candidate, gt, directory, mapping=None):
    """An upper bound on distinct instances touched by required processing.

    This does not align bowl names: processing one of two bowls can cover at
    most one of two required instances under every possible one-to-one map.
    All same-kind participants of a macro count, so a batch action is allowed.
    """
    try:
        plan, _ = replay_saved_plan(directory, candidate)
    except (OSError, ValueError, KeyError):
        return []
    matches = []
    mapping = mapping if mapping is not None else {o.name: o.name for o in candidate.objects if gt.object(o.name) is not None}
    required_facts = necessary_facts(gt, static_feasible=True, coeffects=True) - gt.facts
    for predicate in sorted(SEMANTIC_STATES):
        required = {f[1] for f in required_facts if len(f) == 2 and f[0] == predicate}
        if len(required) < 2:
            continue
        families = {named_family(s) for s in gt.schemas.values()
                    if any(gt.interface.canonicalize(f)[0] == predicate for f in s.add_eff)} - {"unsupported", "generic_pddl"}
        for kind in sorted(set.intersection(*(set(gt.object(obj).kinds) for obj in required))):
            participants = {obj.name for obj in candidate.objects
                            if obj.name not in mapping or not gt.object(mapping[obj.name]).kinds
                            or kind in gt.object(mapping[obj.name]).kinds}
            if not participants:
                continue
            touched = {f[1] for f in candidate.facts if len(f) == 2 and f[0] == predicate and f[1] in participants}
            touched.update(f[1] for f in candidate.negative_facts if len(f) == 2
                           and f[0] == OPPOSITES.get(predicate) and f[1] in participants)
            for action in plan:
                effects = {candidate.interface.canonicalize(f)[0] for f in action.add_eff}
                deleted = {candidate.interface.canonicalize(f)[0] for f in action.del_eff}
                unknown = any(p not in STRUCTURAL | SEMANTIC_STATES
                              and not p.startswith(("kind:", "static:")) for p in effects | deleted)
                if (named_family(candidate.schemas[action.name]) in families or predicate in effects
                    or OPPOSITES.get(predicate) in deleted or unknown):
                    touched.update(set(action.args) & participants)
                    if unknown and not action.args:
                        touched.update(participants)
            if len(touched) < len(required):
                matches.append({"kind": "semantic_goal_cardinality", "predicate": predicate, "kind_name": kind,
                                "required_objects": sorted(required), "candidate_touched_objects": sorted(touched)})
    return matches


def check_causal_failure(candidate, gt, mapping, directory, reference):
    problem = parse_problem_model(directory / "problem.pddl", candidate.schemas)
    analysis = analyze_reachability(candidate, problem)
    result = {"available": analysis.complete, "reason": analysis.reason,
              "grounded_actions": len(analysis.actions), "rounds": analysis.rounds,
              "dead_schemas": analysis.dead_schemas, "matches": []}
    if not analysis.complete:
        return result
    result["raw_init_matches"] = check_raw_init(candidate, gt, mapping, problem)
    if result["raw_init_matches"]:
        return result
    try:
        _, final = replay_saved_plan(reference, gt)
    except (OSError, ValueError, KeyError) as error:
        result["reason"] = f"reference_plan_unavailable: {error}"
        return result
    gt_final = {gt.interface.canonicalize(f) for f in final}
    goals = [(f, True) for f in sorted(problem.goal_positive)] + [(f, False) for f in sorted(problem.goal_negative)]
    result["unreachable_goals"] = [_literal_text(f) for f in goals if not analysis.possible(*f)]
    result["goal_mutexes"] = [[_literal_text(left), _literal_text(right)]
                              for index, left in enumerate(goals) for right in goals[index + 1:]
                              if analysis.possible(*left) and analysis.possible(*right)
                              and not analysis.together(left, right)]
    # A missing on/in bootstrap cannot be supplied by its alternate spelling.
    # A blocker on one optional route need not be necessary on other routes.
    # Intersect raw prerequisites before diagnosing a particular bootstrap.
    needed = necessary_facts(candidate, problem=problem, keep_unbound=True)
    for alias in sorted(problem.init_state):
        if len(alias) != 3 or alias[0] not in SPATIAL:
            continue
        for predicate in sorted(SPATIAL - {alias[0]}):
            fact = (predicate, *alias[1:])
            expected = mapped_fact(candidate.interface.canonicalize(fact), mapping)
            if expected not in gt.facts:
                continue
            for pattern in sorted(needed):
                if unify(pattern, fact) is None:
                    continue
                # All bindings of this common requirement lack a bootstrap,
                # including alternative objects or producers on other routes.
                if any(unify(pattern, f) is not None for f in problem.init_state) or any(
                        unify(pattern, f) is not None for a in analysis.actions for f in a.add_eff):
                    continue
                result["raw_init_matches"].append({"kind": "initial_relation_vocabulary_mismatch", "raw_fact": list(fact),
                                                   "required_pattern": list(pattern), "gt_fact": list(expected),
                                                   "present_aliases": [list(alias)]})
                break
    if result["raw_init_matches"]:
        return result
    supported = []
    gt_required_final = required_final_facts(gt)
    for fact, positive in goals:
        expected = mapped_fact(candidate.interface.canonicalize(fact), mapping)
        gt_goals = gt_required_final if positive else gt.goal_negative
        desired = expected is not None and (equivalent(expected, gt_goals) if positive else expected in gt_goals)
        final_agrees = expected is not None and (equivalent(expected, gt_final) if positive else expected not in gt_final)
        if desired and not analysis.possible(fact, positive):
            producers = [a for a in analysis.actions if fact in (a.add_eff if positive else a.del_eff)]
            blocked = [{"action": a.to_line(), "conditions": _blocked_conditions(analysis, a)} for a in producers]
            dead = [item for item in analysis.dead_schemas if any(
                unify(effect, fact) is not None for effect in (
                    candidate.schemas[item["action"]].add_eff if positive else candidate.schemas[item["action"]].del_eff))]
            result["matches"].append({"kind": "unreachable_causal_goal", "candidate_goal": list(fact),
                                      "positive": positive, "gt_goal": list(expected),
                                      "blocked_producers": blocked, "uninstantiable_producers": dead})
        if final_agrees and analysis.possible(fact, positive):
            supported.append(((fact, positive), expected, desired))
    for index, (left, gt_left, left_required) in enumerate(supported):
        for right, gt_right, right_required in supported[index + 1:]:
            if left_required and right_required and not analysis.together(left, right):
                result["matches"].append({"kind": "causal_goal_mutex", "goals": [_literal_text(left), _literal_text(right)],
                                          "gt_goals": [_literal_text((gt_left, left[1])), _literal_text((gt_right, right[1]))],
                                          "reference_witness": "both hold in the validated GT final state"})
    return result


# Physical transitions under the explicit non-silent-transfer judge contract.

CONTRACT_QUOTE = "actual transfers, relocation, obstacle removal, or resource release cannot be supplied silently"
TRANSFERS = {"holding", "on", "in", "under", "inserted"}
ACQUIRE_NAMES = {"pick", "pickup", "grasp", "grab", "take", "lift", "acquire"}
MOVE_NAMES = {"place", "put", "move", "relocate", "position", "transfer", "return"}
CLEAR_NAMES = {"clear", "unblock", "uncover", "remove"}


def check_physical_transitions(candidate, gt, mapping, directory, reference):
    result = {"matched": False, "matches": []}
    contract_path = Path(__file__).resolve().parents[1] / "prompt_templates/pddl_judge.txt"
    try:
        if CONTRACT_QUOTE not in contract_path.read_text():
            result["unavailable"] = "explicit non-silent-transfer contract is absent"
            return result
        result["contract"] = {"path": str(contract_path), "quote": CONTRACT_QUOTE}
        plan, _ = replay_saved_plan(directory, candidate)
        replay_saved_plan(reference, gt)
    except (OSError, ValueError, KeyError) as error:
        result["unavailable"] = str(error)
        return result

    raw_state = set(candidate.identity_facts)
    touched, relocated, uncertain_objects, unknown_touched = set(), set(), set(), False

    def project(facts):
        return {f for raw in project_water(candidate, {candidate.interface.canonicalize(f) for f in facts})
                if (f := mapped_fact(raw, mapping)) is not None}

    for index, action in enumerate(plan, 1):
        next_state = (raw_state - action.del_eff) | action.add_eff
        before, after = project(raw_state), project(next_state)
        participants = {mapping[arg] for arg in action.args if arg in mapping}
        effects = {candidate.interface.canonicalize(f) for f in action.add_eff | action.del_eff}
        unknown_effect = any(f[0] not in STATE_PREDICATES and not f[0].startswith(("kind:", "static:"))
                             for f in effects)
        words = set(re.findall(r"[a-z]+", action.name.lower()))
        transfer_objects = {f[2] if f[0] == "holding" else f[1]
                            for f in effects if len(f) == 3 and f[0] in TRANSFERS}
        transfers = {mapping[obj] for obj in transfer_objects if obj in mapping}
        unknown_transfer = bool(transfer_objects - mapping.keys())

        def accounted(objects, names):
            return (unknown_effect or unknown_transfer or bool(transfers & objects)
                    or bool(words & names))

        # Every reference producer of each shared transition participates,
        # irrespective of its action family/name. Alternate implementations
        # without the same physical guard prevent a strong inference.
        anchors = sorted(f for f in after - before if f[0] in STATE_PREDICATES - {"hand_free", "clear"})
        for anchor in anchors:
            context = reference_guards(gt, None, {anchor}, set(), details=True)
            for required in sorted(context.get("positive", set())):
                if equivalent(required, before):
                    continue
                observations, missing = [], None
                if len(required) == 3 and required[0] == "holding":
                    hand, obj = required[1:]
                    if required in gt.facts and obj not in touched:
                        continue  # Do not trust a conflicting candidate init over known possession.
                    if obj not in participants:
                        continue  # A different source/tool may be an alternate method.
                    observations = [f for f in sorted(before) if (
                        f == ("hand_free", hand)
                        or (len(f) == 3 and f[0] == "holding" and f[1] == hand and f[2] != obj)
                        or (len(f) == 3 and f[0] in SPATIAL | {"under", "inserted"} and f[1] == obj))]
                    if observations and not accounted({obj}, ACQUIRE_NAMES):
                        missing = "acquire_required_object"
                elif len(required) == 3 and required[0] in SPATIAL | {"under", "inserted"}:
                    obj = required[1]
                    if equivalent(required, gt.facts) and obj not in touched:
                        continue
                    if obj not in anchor[1:]:
                        continue  # Incidental preparation on another object is not a site constraint.
                    if required[2] in enclosing_objects(before, {obj}):
                        continue  # A carrier can place its contents at the site.
                    # Holding alone does not say where the hand/object is;
                    # positioning a held vessel under running water can be local.
                    observations = [f for f in sorted(before) if len(f) == 3
                                    and f[0] in SPATIAL | {"under", "inserted"}
                                    and f[1] == obj and f[2] != required[2]]
                    if observations and not accounted({obj}, MOVE_NAMES):
                        missing = "move_to_required_site"
                elif len(required) == 2 and required[0] in {"clear", "unblocked"}:
                    predicates = {"on"} if required[0] == "clear" else {"blocks", "blocks_opening"}
                    obstacles = [f for f in gt.facts if len(f) == 3 and f[0] in predicates and f[2] == required[1]]
                    observations = [f for f in sorted(obstacles) if f[1] not in touched
                                    and not enclosing_objects(gt.facts | before, set(f[1:])) & (relocated | uncertain_objects)
                                    and not unknown_touched
                                    and not accounted({f[1]}, CLEAR_NAMES)]
                    if observations and required not in after:
                        missing = "remove_initial_obstacle"
                if missing:
                    result["matches"].append({"kind": "unaccounted_physical_precondition", "step": index,
                        "action": action.name, "gt_precondition": list(required), "transition_anchor": list(anchor),
                        "observations": [list(f) for f in observations],
                        "candidate_effects": [list(f) for f in sorted(effects)],
                        "reference_transition_proof": {"missing_transition": missing,
                            "all_reference_producers_require_guard": True,
                            "declared_transfer_objects": sorted(transfers), "contract": result["contract"]}})
            for forbidden in sorted(context.get("negative", set())):
                if forbidden[0] not in {"covers", "blocks", "blocks_opening"} or len(forbidden) != 3:
                    continue
                if forbidden not in gt.facts or set(forbidden[1:]) & touched or unknown_touched:
                    continue
                if enclosing_objects(gt.facts | before, set(forbidden[1:])) & (relocated | uncertain_objects):
                    continue
                if accounted({forbidden[1]}, CLEAR_NAMES) or forbidden not in after and any(
                        mapped_fact(f, mapping) == forbidden for f in effects):
                    continue
                result["matches"].append({"kind": "unaccounted_physical_precondition", "step": index,
                    "action": action.name, "gt_precondition": ["not", *forbidden],
                    "transition_anchor": list(anchor), "observations": [list(forbidden)],
                    "candidate_effects": [list(f) for f in sorted(effects)],
                    "reference_transition_proof": {"missing_transition": "remove_initial_obstacle",
                        "all_reference_producers_require_guard": True,
                        "declared_transfer_objects": sorted(transfers), "contract": result["contract"]}})
        touched.update(mapping[arg] for arg in action.args if arg in mapping)
        relocated.update(transfers)
        if words & MOVE_NAMES:
            relocated.update(participants)
        if unknown_effect:
            uncertain_objects.update(participants)
        unknown_touched |= any(arg not in mapping for arg in action.args) or (unknown_effect and not action.args)
        raw_state = next_state
        if result["matches"]:
            break
    result["matched"] = bool(result["matches"])
    return result


# Explicit instruction order: only simple affirmative transfers joined by then.

_SCOPE = re.compile(r"\b(if|unless|until|before|after|either|or|not|never|may|can|optionally|instead)\b", re.I)
_NOUN_SCOPE = {"and", "all", "each", "every", "both", "one", "two", "three", "it", "them", "this", "that",
               "these", "those", "its", "their", "from", "with", "without", "by", "to", "in", "into", "on"}
_COMMANDS = "start|stop|open|close|turn|press|put|place|add|pour|pick|take|wash|wipe|stir|return"
_TRANSFERS = {"load", "put", "place", "insert", "move", "relocate", "transfer", "return", "pour", "add", "fill",
              "pick", "pickup", "take", "lift", "grasp"}
_LOCAL = {"open", "close", "turn", "switch", "press", "select", "configure", "start", "stop", "screw", "unscrew"}


def _tokens(text):
    return tuple(part for token in normalized_tokens(text) for part in token.split("_"))


def _noun(text):
    tokens = _tokens(re.sub(r"^(?:the|a|an)\s+", "", text.strip(), flags=re.I))
    if not tokens or set(tokens) & _NOUN_SCOPE or any(t.isdigit() for t in tokens):
        return None
    return tokens


def parse_explicit_orders(instruction):
    """Extract a bounded grammar, never infer order from commas/``and`` alone."""
    orders = []
    if (not isinstance(instruction, str) or _SCOPE.search(instruction)
        or re.search(r"\b(order|sequence)\b", instruction, re.I)):
        return orders
    for sentence in re.split(r"[.!?]\s*", instruction):
        parts = re.split(r"\b(?:and\s+)?then\b", sentence, flags=re.I)
        if len(parts) != 2 or _SCOPE.search(sentence):
            continue
        first, second = (part.strip(" ,") for part in parts)
        # A following affirmative command is outside the second event. Do not
        # split arbitrary noun conjunctions, which may denote several objects.
        second = re.split(rf"\s+and\s+(?=(?:{_COMMANDS})\b)", second, maxsplit=1, flags=re.I)[0]
        clauses = []
        for text in (first, second):
            match = re.fullmatch(r"(?:please\s+)?(put|place|add)\s+([a-z][a-z _-]*?)\s+(into|in|on|to)\s+([a-z][a-z _-]*)", text, re.I)
            if match is None:
                break
            verb, subject, preposition, destination = match.groups()
            if (verb.lower() == "add") != (preposition.lower() == "to") or not _noun(subject) or not _noun(destination):
                break
            clauses.append({"text": text, "subject": subject, "destination": destination,
                            "predicate": "on" if preposition.lower() == "on" else "in"})
        if len(clauses) == 2:
            orders.append({"sentence": sentence.strip(), "clauses": clauses})
    return orders


def resolve_noun(world, text):
    """Require a unique typed noun head and support every descriptor token."""
    tokens = _noun(text)
    if tokens is None:
        return None
    matches = []
    for obj in world.objects:
        kinds = [_tokens(kind) for kind in obj.identity_kinds]
        vocabulary = set(_tokens(obj.name)) | {token for kind in kinds for token in kind}
        vocabulary.update(tag for tag in obj.state_tags if "_" not in tag)
        typed_head = any(kind and kind[-1] == tokens[-1] for kind in kinds)
        if (typed_head and set(tokens) <= vocabulary) or tokens == _tokens(obj.name):
            matches.append(obj.name)
    return matches[0] if len(matches) == 1 else None


def _trace(world, plan, mapping=None):
    def project(raw):
        canonical = {world.interface.canonicalize(f) for f in raw}
        return canonical if mapping is None else {f for item in canonical if (f := mapped_fact(item, mapping)) is not None}

    raw = set(world.identity_facts)
    states = [project(raw)]
    for action in plan:
        raw = (raw - action.del_eff) | action.add_eff
        states.append(project(raw))
    return states


def _events(states, fact):
    return [step for step in range(1, len(states)) if fact in states[step] and fact not in states[step - 1]]


def _possible_unrecorded_transfer(candidate, plan, mapping, first, *, require_untouched):
    """Veto macros, unknown encodings and prior operations on the first item."""
    subject, destination = first[1:]
    heads = {head for text in (subject, destination) for head in _tokens(text)[-1:]}
    for obj in candidate.objects:
        if mapping.get(obj.name) in {subject, destination}:
            heads.update(_tokens(kind)[-1] for kind in obj.identity_kinds if _tokens(kind))
            heads.update(_tokens(obj.name)[-1:])
    for action in plan:
        if not action.args or any(arg not in mapping for arg in action.args):
            return True
        participants = {mapping[arg] for arg in action.args}
        effects = {candidate.interface.canonicalize(f) for f in action.add_eff | action.del_eff}
        projected = [mapped_fact(f, mapping) for f in effects]
        if require_untouched and ({subject, destination} & participants or any(
                f and {subject, destination} & set(f[1:]) for f in projected)):
            return True
        unknown = [f for f in effects if f[0] not in STATE_PREDICATES and not f[0].startswith(("kind:", "static:"))]
        if unknown and ({subject, destination} & participants or any(
                (mapped := mapped_fact(f, mapping)) is None or {subject, destination} & set(mapped[1:]) for f in unknown)):
            return True
        name_tokens = _tokens(action.name)
        if not name_tokens:
            return True
        words = set(name_tokens)
        if require_untouched and words & heads:
            return True
        if words & _TRANSFERS and ({subject, destination} & participants or words & heads):
            return True
        # An unfamiliar operation on a relevant object may name a macro even
        # when its effects use a known predicate. Never exhaustively interpret
        # arbitrary verbs through a finite list of transfer keywords.
        if ({subject, destination} & participants or words & heads) and (
                name_tokens[0] not in _LOCAL or words & {"and", "then"}):
            return True
    return False


def _stationary_sources(candidate, gt, states, plan, mapping, observations, destination):
    """A basket may carry its contents; a persistent direct source is not enough."""
    sources = enclosing_objects(states[0] | gt.facts, {f[2] for f in observations})
    if destination in sources:
        return None  # Already nested within the requested destination.
    for action in plan:
        participants = {mapping[arg] for arg in action.args if arg in mapping}
        tokens = _tokens(action.name)
        effects = {candidate.interface.canonicalize(f) for f in action.add_eff | action.del_eff}
        moved = {mapping[obj] for f in effects if len(f) == 3 and f[0] in SPATIAL | {"holding", "under", "inserted"}
                 if (obj := f[2] if f[0] == "holding" else f[1]) in mapping}
        if participants & sources and (not tokens or tokens[0] not in _LOCAL
                                       or set(tokens) & (_TRANSFERS | {"and", "then"})) and (
                not moved or moved & sources or set(tokens) & {"and", "then"}):
            return None  # A named source/ancestor macro may carry the item.
        for raw in action.add_eff | action.del_eff:
            canonical = candidate.interface.canonicalize(raw)
            fact = mapped_fact(canonical, mapping)
            if fact and len(fact) == 3 and ((fact[0] in SPATIAL | {"under", "inserted"} and fact[1] in sources)
                                           or (fact[0] == "holding" and fact[2] in sources)):
                return None
            if canonical[0] not in STATE_PREDICATES and not canonical[0].startswith(("kind:", "static:")) and (
                    participants & sources or fact is None or sources & set(fact[1:])):
                return None
    return sorted(sources)


def check_instruction_order(candidate, gt, mapping, directory, reference, instruction):
    result = {"matched": False, "matches": [], "constraints": []}
    orders = parse_explicit_orders(instruction)
    if not orders:
        return result
    try:
        contract_path = Path(__file__).resolve().parents[1] / "prompt_templates/pddl_judge.txt"
        if CONTRACT_QUOTE not in contract_path.read_text():
            raise ValueError("Explicit non-silent-transfer contract is absent")
        plan, _ = replay_saved_plan(directory, candidate)
        reference_plan, _ = replay_saved_plan(reference, gt)
    except (OSError, ValueError, KeyError) as error:
        result["unavailable"] = str(error)
        return result
    if len(set(mapping.values())) != len(mapping):
        result["unavailable"] = "Object mapping is not one-to-one"
        return result
    states, reference_states = _trace(candidate, plan, mapping), _trace(gt, reference_plan)
    inverse = {target: source for source, target in mapping.items()}
    for order in orders:
        item = {**order, "status": "unknown"}
        result["constraints"].append(item)
        facts = []
        for clause in order["clauses"]:
            subject, destination = resolve_noun(gt, clause["subject"]), resolve_noun(gt, clause["destination"])
            if subject is None or destination is None or subject == destination or subject not in inverse or destination not in inverse:
                break
            facts.append((clause["predicate"], subject, destination))
        if len(facts) != 2 or facts[0] == facts[1]:
            item["reason"] = "Transfer objects/relations are not uniquely grounded and mapped"
            continue
        first, second = facts
        item["gt_facts"] = [list(f) for f in facts]
        item["candidate_facts"] = [[f[0], *(inverse[arg] for arg in f[1:])] for f in facts]
        first_steps, second_steps = _events(states, first), _events(states, second)
        reference_first, reference_second = _events(reference_states, first), _events(reference_states, second)
        item.update(first_steps=first_steps, second_steps=second_steps,
                    reference_first_steps=reference_first, reference_second_steps=reference_second)
        if first in states[0] or first in reference_states[0] or second in states[0]:
            item["reason"] = "An event condition already holds initially; redundant transfer is not required"
            continue
        if not reference_first or not reference_second or not any(a < b for a in reference_first for b in reference_second):
            item["reason"] = "No valid reference witness for the grounded instruction order"
            continue
        # Remember historical completion, even if the first state is later
        # undone. Any valid first->second witness or same-action macro vetoes.
        if not first_steps or not second_steps or max(second_steps) >= min(first_steps):
            item["reason"] = "No definite inversion of all recorded event occurrences"
            continue
        # Idempotent additions may still describe another dose/placement.
        # They cannot prove completion, but can refute a definite inversion.
        repeated_second = [step for step, action in enumerate(plan, 1) if step >= min(first_steps)
                           and any(mapped_fact(candidate.interface.canonicalize(f), mapping) == second
                                   for f in action.add_eff)]
        if repeated_second or _possible_unrecorded_transfer(
                candidate, plan[min(first_steps) - 1:], mapping, second, require_untouched=False):
            item["reason"] = "A simultaneous/later repeated second event or alternate encoding may satisfy the order"
            continue
        through = max(second_steps)
        observations = [f for f in sorted(states[0] & states[through - 1] & states[through] & gt.facts)
                        if len(f) == 3 and f[0] in SPATIAL and f[1] == first[1] and f[2] != first[2]]
        if not observations or _possible_unrecorded_transfer(
                candidate, plan[:through], mapping, first, require_untouched=True):
            item["reason"] = "No persistent confirmed source, or an earlier macro/unknown operation could account for transfer"
            continue
        stationary_sources = _stationary_sources(candidate, gt, states, plan[:through], mapping, observations, first[2])
        if stationary_sources is None:
            item["reason"] = "Source/ancestor may have carried the first item into the destination, or uses an unknown encoding"
            continue
        item.update(status="violated", reason="All second events precede the first; its subject remains at the confirmed source")
        result["matches"].append({"kind": "explicit_task_order_violation", "instruction": instruction,
                                  "first_clause": order["clauses"][0]["text"], "second_clause": order["clauses"][1]["text"],
                                  "step": through, "action": plan[through - 1].name,
                                  "first_steps": first_steps, "second_steps": second_steps,
                                  "gt_facts": item["gt_facts"], "candidate_facts": item["candidate_facts"],
                                  "observations": [list(f) for f in observations],
                                  "stationary_source_objects": stationary_sources,
                                  "reference_first_steps": reference_first, "reference_second_steps": reference_second,
                                  "contract": {"path": str(contract_path), "quote": CONTRACT_QUOTE}})
    result["matched"] = bool(result["matches"])
    return result


# Error categories describe the faulty model part; detailed rules supply evidence.

TAXONOMY_VERSION = 2
TAXONOMY = {
    "pddl_invalid": {
        "name": "PDDL 表达错误",
        "subcategories": {
            "pddl_syntax": "语法结构错误",
            "pddl_symbols": "使用未声明谓词/变量",
        },
    },
    "init_error": {
        "name": "init 错误",
        "subcategories": {
            "init_missing": "初始信息缺失",
            "init_mismatch": "init写错",
            "init_inconsistent": "init冲突",
        },
    },
    "goal_error": {
        "name": "goal 错误",
        "subcategories": {
            "goal_missing": "任务要求遗漏",
            "goal_mismatch": "goal写错",
            "goal_inconsistent": "目标自身矛盾",
        },
    },
    "operator_error": {
        "name": "算子错误",
        "subcategories": {
            "operator_missing": "oper整体缺失",
            "operator_modeling": "oper建模错误",
        },
    },
    "else": {
        "name": "else",
        "subcategories": {
            "else": "else",
        },
    },
}

# These describe the evidence required by the rules, not keyword heuristics.
# Keep them beside the taxonomy so saved JSON and rendered reports agree.
CLASSIFICATION_CRITERIA = {
    "pddl_syntax": "解析日志明确定位括号、关键字或表达式结构错误；仅有输入失败返回码不细分。",
    "pddl_symbols": "声明扫描或解析日志明确发现重复声明、未声明符号、未绑定变量或参数数量错误。",
    "init_missing": "已确认必要对象或初始事实缺失，且无法由候选操作建立；原始事实须按原始目标和全部替代路径证明必要，数量按可对应实例上界核实。",
    "init_mismatch": "已确认同一对象的显式初始状态或位置与 GT 冲突，或所有路径共有的原始关系在 init 中用词不一致且所有绑定均无建立方式；缺省或未映射附加位置不强判。",
    "init_inconsistent": "init 中直接出现互斥事实、自包含/自支撑，或已确认障碍与开敞声明有可达性互斥证明。",
    "goal_missing": "可信任务要求未被目标表达，且有效计划结果或操作见证也未覆盖；排除初始已完成事件、负状态和未知替代编码，数量按潜在覆盖实例上界核实。",
    "goal_mismatch": "已确认目标对象、目的地或状态与任务要求冲突，或额外目标在初始不成立且双方均无建立方式；不可达本身不足以判 goal 错。",
    "goal_inconsistent": "goal 直接要求互斥事实或对象位于自身；跨动作导致的不可达另按因果证据判断。",
    "operator_missing": "参考任务必需某操作能力，而候选既无该能力也无可替代的复合实现；不按算子数量或动作改名判断。",
    "operator_modeling": "动作条件、必需效果或跨动作衔接有确定错误；依据必要条件、明确指令顺序、效果语义或完整因果证明，排除合法替代编码与复合实现；分析不充分时归 else。",
}

_LEGACY_CATEGORIES = {key: key for key in TAXONOMY}
_LEGACY_CATEGORIES.update(operator_missing="operator_error", operator_semantics="operator_error")

_RULE_GROUPS = {
    "init_missing": {
        "required_initial_fact_missing", "required_raw_fact_missing", "declared_instance_deficit",
    },
    "init_mismatch": {"gt_init_mismatch", "initial_relation_vocabulary_mismatch"},
    "init_inconsistent": {"contradictory_literals", "reflexive_location", "initial_access_conflict"},
    "goal_missing": {"omitted_goal", "missing_goal_witness", "spatial_goal_cardinality", "semantic_goal_cardinality"},
    "goal_mismatch": {"wrong_goal_state", "wrong_destination", "reversed_arguments", "goal_without_producer",
                      "mapped_goal_arguments_or_destination_wrong"},
    "goal_inconsistent": {"contradictory_goal_literals", "reflexive_location"},
    "operator_modeling": {
        "violated_gt_guard", "violated_negative_gt_guard", "missing_semantic_precondition",
        "unrestored_initial_state", "unaccounted_physical_precondition", "explicit_task_order_violation",
        "semantic_noop", "semantic_effect_missing", "unreachable_processing_effect",
        "inconsistent_effects", "required_goal_has_no_producer", "reference_state_mutex",
        "unreachable_causal_goal", "causal_goal_mutex", "exclusive_resource_goals", "lost_required_state",
    },
}


def _pddl_subcategory(result, directory):
    if result.get("reason") == "duplicate_declaration" or any(
            issue.get("kind") == "duplicate_declaration"
            for issue in result.get("checks", {}).get("declarations", {}).get("issues", [])):
        return "pddl_symbols"
    solver = result.get("checks", {}).get("fast_downward", {})
    if directory is not None:
        try:
            saved = json.loads((Path(directory) / "attribution_solver/result.json").read_text())
            if isinstance(saved, dict):
                solver = saved
        except (OSError, ValueError):
            pass
    if not isinstance(solver, dict):
        return None
    log = "\n".join(solver.get(key) or "" for key in ("stdout", "stderr"))
    if re.search(r"(?:Undefined|Undeclared)\s+(?:predicate|object|variable)|"
                 r"Expected logical operator or predicate name|"
                 r"Predicate '[^']+' of arity \d+ used\s+with \d+ arguments", log, re.I):
        return "pddl_symbols"
    if re.search(r"Missing '\)'|Tokens remaining after parsing:|Non-ASCII character outside comment:|"
                 r"Expected a non-empty block starting with any of the following words:", log):
        return "pddl_syntax"
    # Input failure alone cannot distinguish bad syntax from bad symbols.
    return None


def taxonomy_for(result, directory=None):
    """Group existing evidence without changing its legacy attribution decision.

    Recompute from label/reason, so stale or manually mismatched taxonomy fields
    cannot override the underlying evidence. Missing attribution is distinct
    from an attribution that explicitly concluded ``else``.
    """
    if not isinstance(result, dict) or (category := _LEGACY_CATEGORIES.get(result.get("label"))) is None:
        return None, None
    reason = result.get("reason")
    if reason in {"candidate_operator_count_less_than_gt", "same_action_multiset_different_order"}:
        return "else", "else"  # Historical heuristics are not proofs.
    if category == "pddl_invalid":
        return category, _pddl_subcategory(result, directory)
    if result.get("label") == "operator_missing":
        return category, "operator_missing"
    if category == "else":
        return category, "else"
    for subcategory in TAXONOMY[category]["subcategories"]:
        if reason in _RULE_GROUPS.get(subcategory, set()):
            return category, subcategory
    # A known broad category need not imply a proven finer diagnosis.
    return category, None


def taxonomy_fields(result, directory=None):
    category, subcategory = taxonomy_for(result, directory)
    return {"taxonomy_version": TAXONOMY_VERSION, "category": category, "subcategory": subcategory,
            "classification_criterion": CLASSIFICATION_CRITERIA.get(subcategory)}


# Object alignment, shared-state checks and attribution workflow.

def load_world(directory):
    # FD accepts overlapping effects (add wins). Keep them inspectable instead
    # of dropping the whole task into else because the reorder parser is strict.
    schemas = parse_domain(directory / "domain.pddl", allow_effect_overlap=True)
    problem = parse_problem_model(directory / "problem.pddl", schemas)
    return compile_world(schemas, problem)


def object_payload(world):
    return {
        "objects": [{"name": o.name, "kinds": list(o.identity_kinds)} for o in world.objects],
        "init": [list(f) for f in sorted(world.facts)],
        "negative_init": [list(f) for f in sorted(world.negative_facts)],
    }


def validate_mapping(value, query_names, gt_names, fixed):
    if not isinstance(value, dict) or set(value) != {"objects"} or not isinstance(value["objects"], dict):
        raise ValueError("Expected objects mapping")
    returned = value["objects"]
    if not set(query_names) <= returned.keys() or set(returned) - set(query_names) - fixed.keys():
        raise ValueError("Every queried object must occur exactly once")
    mapped, occupied = dict(fixed), set(fixed.values())
    for candidate, target in value["objects"].items():
        if target is not None and not isinstance(target, str):
            raise ValueError("Mapping values must be GT names or null, never judgments")
        if candidate in fixed:
            if target != fixed[candidate]:
                raise ValueError("A fixed identity cannot be changed")
            continue  # Some providers echo exact fixed identities; no new decision.
        if target is None:
            continue
        if not isinstance(target, str) or target not in gt_names or target in occupied:
            raise ValueError("Mapping must use existing, distinct GT objects")
        mapped[candidate] = target
        occupied.add(target)
    return mapped


def lid_owners(world):
    """Unique cover/container pairs, for open/closed comparisons ONLY."""
    facts = world.facts
    names = {f[i] for f in facts for i in range(1, len(f))}
    kinds = {name: {f[0][5:] for f in facts if len(f) == 2 and f[1] == name and f[0].startswith("kind:")}
             for name in names}
    cover = {name for name in names if any(k.split("_")[-1] in {"lid", "cap", "door"} for k in kinds[name])
             or name.endswith(("_lid", "_cap", "_door"))}
    openable = {"container", "pot", "kettle", "bottle", "box", "thermos", "bowl", "cup", "microwave",
                "ice_maker", "washing_machine", "fridge", "refrigerator", "cabinet", "drawer"}
    possible = {}
    for lid in cover:
        parents = {f[2] for f in facts if len(f) == 3 and f[1] == lid
                   and f[0].removeprefix("static:") in {"on", "part_of", "attached_to"}
                   and kinds.get(f[2], set()) & openable}
        for suffix in ("_lid", "_cap", "_door"):
            if lid.endswith(suffix) and lid[:-len(suffix)] in names:
                parents.add(lid[:-len(suffix)])
        possible[lid] = parents
    # A container with multiple possible covers is ambiguous even if one cover
    # itself has a unique parent. Do not select a convenient match.
    counts = Counter(parent for parents in possible.values() for parent in parents)
    return {lid: next(iter(parents)) for lid, parents in possible.items()
            if len(parents) == 1 and counts[next(iter(parents))] == 1}


def check_init(candidate, gt, mapping):
    """Compare shared, explicitly represented states/locations; absence is skipped."""
    inverse = {target: source for source, target in mapping.items()}
    candidate_owners, gt_owners = lid_owners(candidate), lid_owners(gt)
    owner_mapping = {}
    for source, target in mapping.items():
        owner_mapping.setdefault(gt_owners.get(target, target), set()).add(candidate_owners.get(source, source))
    checks = []
    unary = [(f, True) for f in gt.facts if len(f) == 2 and not f[0].startswith(("kind:", "static:"))]
    unary += [(f, False) for f in gt.negative_facts if len(f) == 2 and not f[0].startswith(("kind:", "static:"))]
    for fact, positive in sorted(unary):
        predicate, target = fact
        targets = {target}
        sources = {inverse[target]} if target in inverse else set()
        if predicate in {"open", "closed"}:
            owner = gt_owners.get(target, target)
            targets = {owner} | {lid for lid, parent in gt_owners.items() if parent == owner}
            # Compare an explicitly represented object's own state first.
            # Its cover can have a separate state and cannot override it.
            direct = inverse.get(target)
            if direct is not None and any(len(f) == 2 and f[1] == direct and f[0] in {predicate, OPPOSITES.get(predicate)}
                                          for f in candidate.facts | candidate.negative_facts):
                sources, targets = {direct}, {target}
            else:
                owners = owner_mapping.get(owner, set())
                sources = set()
                if len(owners) == 1:
                    candidate_owner = next(iter(owners))
                    sources = {candidate_owner} | {lid for lid, parent in candidate_owners.items() if parent == candidate_owner}
        source = inverse.get(target) or (sorted(sources)[0] if sources else None)
        item = {"kind": "unary_state", "gt": list(fact), "positive": positive, "candidate_object": source}
        if predicate in {"open", "closed"}:
            item.update(gt_state_objects=sorted(targets), candidate_state_objects=sorted(sources))
        if not sources:
            item.update(status="unknown", reason="GT object has no confirmed candidate mapping")
        else:
            family = {predicate, OPPOSITES.get(predicate)}
            relevant = {f for f in candidate.facts | candidate.negative_facts
                        if len(f) == 2 and f[1] in sources and f[0] in family}
            item["candidate_states"] = [list(f) for f in sorted(relevant)]
            item["candidate_negative_states"] = [list(f) for f in sorted(relevant & candidate.negative_facts)]
            gt_conflict = positive and any((OPPOSITES.get(predicate), obj) in gt.facts or (predicate, obj) in gt.negative_facts for obj in targets)
            if gt_conflict:
                item.update(status="unknown", reason="GT has contradictory states on the container/cover")
            elif not relevant:
                item.update(status="skipped", reason="state family is not explicitly represented in candidate init")
            else:
                contradiction = any((predicate, obj) in candidate.negative_facts or (OPPOSITES.get(predicate), obj) in candidate.facts for obj in sources) if positive else any((predicate, obj) in candidate.facts for obj in sources)
                item.update(status="fail" if contradiction else "pass",
                            reason="explicit states contradict" if contradiction else "shared explicit states agree")
        checks.append(item)

    subjects = sorted({f[1] for f in gt.facts if len(f) == 3 and f[0] in SPATIAL})
    for target in subjects:
        expected = {f[2] for f in gt.facts if len(f) == 3 and f[0] in SPATIAL and f[1] == target}
        source = inverse.get(target)
        actual = {f[2] for f in candidate.facts if len(f) == 3 and f[0] in SPATIAL and f[1] == source}
        mapped_areas = {mapping[a] for a in actual if a in mapping}
        item = {"kind": "spatial_area", "gt_object": target, "candidate_object": source,
                "gt_areas": sorted(expected), "candidate_areas": sorted(actual), "mapped_areas": sorted(mapped_areas)}
        if source is None:
            item.update(status="unknown", reason="GT subject has no confirmed candidate mapping")
        elif actual - mapping.keys():
            item.update(status="unknown", reason="an unmapped candidate area may supply the shared GT area")
        elif mapped_areas and not mapped_areas & expected:
            item.update(status="fail", reason="candidate area maps to a different GT object/area")
        elif mapped_areas - expected:
            item.update(status="unknown", reason="shared area agrees; additional support relations are not confirmed")
        elif not actual:
            item.update(status="skipped", reason="location is not explicitly represented in candidate init")
        elif actual - mapping.keys() or expected - inverse.keys():
            item.update(status="unknown", reason="source/target area mapping is incomplete")
        else:
            item.update(status="pass", reason="shared areas agree (in/on ignored; GT-only areas not required)")
        checks.append(item)
    compared = sum(c["status"] in {"pass", "fail"} for c in checks)
    status = "fail" if any(c["status"] == "fail" for c in checks) else "unknown" if not compared or any(c["status"] == "unknown" for c in checks) else "pass"
    return {"status": status, "checks": checks, "compared": compared,
            "skipped": sum(c["status"] == "skipped" for c in checks),
            "scope": "shared explicit unary state families and in/on areas; absent facts skipped, unmapped identities unknown; pass is not full init validation"}


def check_goal_error(candidate, gt, mapping):
    """Explicit reversed arguments or a different mapped destination in in/on goals."""
    candidate_goals = {f for f in candidate.goal_positive if len(f) == 3 and f[0] in SPATIAL}
    gt_goals = {f for f in gt.goal_positive if len(f) == 3 and f[0] in SPATIAL}
    mapped = {tuple([f[0], *(mapping[a] for a in f[1:])]): f for f in candidate.goal_positive
              if len(f) == 3 and f[0] in SPATIAL and all(a in mapping for a in f[1:])}
    matches = []
    for fact in sorted(gt_goals):
        if fact[1] == fact[2]:
            continue
        reverse = (fact[0], fact[2], fact[1])
        if reverse in mapped and fact not in mapped and reverse not in gt_goals:
            matches.append({"kind": "reversed_arguments", "gt_goal": list(fact),
                            "candidate_goal": list(mapped[reverse]), "mapped_candidate_goal": list(reverse)})
    for subject in sorted({f[1] for f in gt_goals}):
        expected = {f[2] for f in gt_goals if f[1] == subject}
        actual = sorted(f for f in candidate_goals if mapping.get(f[1]) == subject)
        # Missing goals, unknown destinations, and goals that also include a
        # correct destination do not establish that the destination was replaced.
        if not actual or any(f[2] not in mapping for f in actual):
            continue
        if {mapping[f[2]] for f in actual} & expected:
            continue
        for fact in actual:
            matches.append({"kind": "wrong_destination", "gt_object": subject,
                            "gt_goals": [list(f) for f in sorted(gt_goals) if f[1] == subject],
                            "gt_areas": sorted(expected), "candidate_goal": list(fact),
                            "mapped_candidate_goal": [fact[0], subject, mapping[fact[2]]]})
    return {"matched": bool(matches), "matches": matches,
            "scope": "positive in/on goals: same-predicate reversed arguments, or shared subject with confirmed different target area (in/on ignored); no missing-goal or negation inference"}


def action_names(path):
    """Strictly read action names, preserving repetitions and ignoring arguments."""
    names = []
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split(";", 1)[0].strip().lower()
        if not line:
            continue
        match = re.fullmatch(r"\(\s*([a-z][a-z0-9_-]*)(?:\s+[^()]*)?\s*\)", line)
        if not match:
            raise ValueError(f"Malformed plan line {number}: {line}")
        names.append(match[1])
    return names


def compare_plans(candidate_path, gt_path):
    try:
        candidate, gt = action_names(candidate_path), action_names(gt_path)
    except (OSError, ValueError) as error:
        return {"available": False, "reason": str(error), "order_only_difference": False}
    same_count = len(candidate) == len(gt)
    same_multiset = Counter(candidate) == Counter(gt)
    different_order = candidate != gt
    return {"available": True, "candidate_names": candidate, "gt_names": gt,
            "same_count": same_count, "same_multiset": same_multiset, "different_order": different_order,
            "order_only_difference": same_count and same_multiset and different_order}


def fd_category(solver):
    code = solver.get("returncode")
    if code is None:
        return "else"  # No returned FD code: do not invent an input error.
    if code in {0, 1, 2, 3, 10, 11, 12}:
        return None
    if code == 31:  # TRANSLATE_INPUT_ERROR; search/driver errors are not PDDL syntax proofs.
        return "pddl_invalid"
    return "else"


def classify(solver, init, counts, judge_failed, plans, goal=None, *, coverage=None, semantics=None):
    """First matching rule wins. Unknown init is never treated as correct."""
    if (label := fd_category(solver)) == "pddl_invalid":
        return label, "fast_downward_returncode"
    if init["status"] == "fail":
        first = next((item for item in init.get("checks", []) if item.get("status") == "fail"), {})
        if first.get("kind") == "initial_access_conflict":
            return "init_error", "initial_access_conflict"
        return "init_error", "required_initial_fact_missing" if first.get("kind") == "required_fact_missing" else "gt_init_mismatch"
    if goal and goal.get("matched"):
        return "goal_error", (goal.get("matches") or [{}])[0].get("kind", "mapped_goal_mismatch")
    if coverage and coverage.get("matched"):
        return "operator_missing", "required_semantic_capability_missing"
    if semantics and semantics.get("matched"):
        return "operator_semantics", (semantics.get("matches") or [{}])[0].get("kind", "semantic_rule_matched")
    if goal and any(item.get("reference_supported") and not item.get("has_matching_effect")
                    for item in goal.get("unachievable", [])):
        return "operator_semantics", "required_goal_has_no_producer"
    # Fewer schemas can be a correct macro/parameterized model. Counts are
    # retained as evidence, but never prove that a required ability is missing.
    if label == "else":
        return "else", "fast_downward_inconclusive"
    if init["status"] != "pass":
        return "else", "init_not_confirmed"
    return "else", "no_rule_matched"


def run_solver(source, dest):
    """Keep the original judged plan intact; run FD in an attribution directory."""
    source, dest = Path(source).resolve(), Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "plan.txt").unlink(missing_ok=True)
    command = [sys.executable, str(fast_downward_path), "--overall-time-limit", "55s",
               "--plan-file", str(dest / "plan.txt"), str(source / "domain.pddl"),
               str(source / "problem.pddl"), "--search", "astar(lmcut())"]
    result = {"command": command, "returncode": None}
    try:
        stdout, stderr = _run_fast_downward(command, dest)
        result.update(returncode=0, stdout=stdout, stderr=stderr)
    except subprocess.CalledProcessError as error:
        result.update(returncode=error.returncode, stdout=error.stdout, stderr=error.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        result.update(error=f"{type(error).__name__}: {error}",
                      stdout=getattr(error, "stdout", ""), stderr=getattr(error, "stderr", ""))
    for key in ("stdout", "stderr"):
        if isinstance(result.get(key), bytes):
            result[key] = result[key].decode("utf-8", errors="replace")
    (dest / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: value for key, value in result.items() if key not in {"stdout", "stderr"}}


def map_objects(candidate, gt, instruction, image_path, model, reasoning_effort, evidence):
    fixed = align_worlds(candidate, gt).mapping()
    queries = sorted({o.name for o in candidate.objects} - fixed.keys())
    targets = sorted({o.name for o in gt.objects} - set(fixed.values()))
    evidence.update(model=model, reasoning_effort=reasoning_effort, fixed=fixed, attempts=[])
    if not queries:
        evidence.update(mapping=fixed, unmapped=[])
        return fixed
    images = list(image_path) if isinstance(image_path, (list, tuple)) else [image_path] if image_path is not None else []
    if not images or any(not Path(path).is_file() for path in images):
        raise ValueError("Object mapping requires all initial images")
    data = {"instruction": instruction, "fixed_mapping": fixed, "query_objects": queries,
            "eligible_gt_objects": targets, "candidate": object_payload(candidate), "gt": object_payload(gt)}
    template = Path(__file__).parents[1] / "prompt_templates/object_mapping.txt"
    prompt = template.read_text().replace("{mapping_input}", json.dumps(data, ensure_ascii=False, indent=2))
    for attempt in range(1, 3):
        capture = {"attempt": attempt}
        evidence["attempts"].append(capture)
        try:
            value = call_gpt_json(model, prompt, [Path(path) for path in images], attempts=1,
                                  response_format={"type": "json_object"}, max_tokens=8192,
                                  reasoning_effort=reasoning_effort, temperature=0, capture=capture)
            mapping = validate_mapping(value, queries, set(targets), fixed)
            evidence.update(mapping=mapping, unmapped=sorted(set(queries) - mapping.keys()))
            return mapping
        except Exception as error:
            capture["error"] = f"{type(error).__name__}: {error}"
    raise ValueError("Object mapping failed after two API/schema attempts")


def attribute_failure(candidate_dir, gt_task_dir, instruction, image_path, *,
                      model="Qwen3.8-27B", reasoning_effort="medium", offline=False):
    """Save one attribution.json for a non-passing task, including tasks without plans."""
    candidate_dir, gt_task_dir = Path(candidate_dir), Path(gt_task_dir)
    checks = {}
    label, reason = "else", "input_unavailable"
    result = {"model": model, "reasoning_effort": reasoning_effort,
              "rules_version": RULES_VERSION, "checks": checks}
    try:
        judge_path = candidate_dir / "judge.json"
        judge = json.loads(judge_path.read_text()) if judge_path.is_file() else {}
        if judge.get("pass") is True:
            return None
        for name in ("domain.pddl", "problem.pddl"):
            if not (candidate_dir / name).is_file():
                raise FileNotFoundError(f"Missing candidate {name}")
        cached_path = candidate_dir / "attribution.json"
        cached = json.loads(cached_path.read_text()) if offline and cached_path.is_file() else {}
        if offline:
            solver_path = candidate_dir / "attribution_solver/result.json"
            solver = (json.loads(solver_path.read_text()) if solver_path.is_file()
                      else cached.get("checks", {}).get("fast_downward", {"returncode": None}))
            solver = {k: v for k, v in solver.items() if k not in {"stdout", "stderr"}}
        else:
            solver = run_solver(candidate_dir, candidate_dir / "attribution_solver")
        checks["fast_downward"] = solver
        try:
            checks["declarations"] = check_declarations(candidate_dir)
        except (ValueError, NotImplementedError, IndexError, TypeError) as error:
            checks["declaration_scan_error"] = f"{type(error).__name__}: {error}"
        issues = checks.get("declarations", {}).get("issues", [])
        if (immediate := fd_category(solver)) == "pddl_invalid":
            label, reason = immediate, "fast_downward_returncode"
        elif any(item["kind"] == "duplicate_declaration" for item in issues):
            label, reason = "pddl_invalid", "duplicate_declaration"
        elif any(item.get("section") == ":init" for item in issues):
            label, reason = "init_error", next(item["kind"] for item in issues if item.get("section") == ":init")
        elif any(item.get("section") == ":goal" for item in issues):
            label, reason = "goal_error", "contradictory_goal_literals"
        else:
            rounds = [p for p in gt_task_dir.glob("round*") if p.is_dir() and p.name[5:].isdigit()]
            if not rounds:
                raise FileNotFoundError(f"No GT round directory in {gt_task_dir}")
            reference = max(rounds, key=lambda p: int(p.name[5:]))
            result["gt_directory"] = str(reference)
            candidate, gt = load_world(candidate_dir), load_world(reference)
            reference_issues = check_declarations(reference)["issues"]
            if reference_issues:
                checks["reference_model_issues"] = reference_issues
                raise ValueError("Reference PDDL has inconsistent facts or invalid declarations; cannot use it as attribution evidence")
            try:
                replay_saved_plan(reference, gt)
                reference_valid = True
                checks["reference_plan"] = {"valid": True}
            except (OSError, ValueError, KeyError) as error:
                reference_valid = False
                checks["reference_plan"] = {"valid": False, "reason": str(error)}
            counts = {"candidate": len(candidate.schemas), "gt": len(gt.schemas)}
            checks["operator_counts"] = counts
            if reference_valid:
                checks["reference_requirements"] = {
                    "processing_landmarks": [list(f) for f in sorted(
                        necessary_facts(gt, static_feasible=True, coeffects=True) - gt.facts)
                        if f[0] in SEMANTIC_STATES],
                    "derived_final_facts": [list(f) for f in sorted(required_final_facts(gt) - gt.goal_positive)],
                }
            mapping = {}
            checks["object_mapping"] = {}
            try:
                if offline:
                    evidence = checks["object_mapping"]
                    saved = cached.get("checks", {}).get("object_mapping", {})
                    fixed = align_worlds(candidate, gt).mapping()
                    names, targets = {o.name for o in candidate.objects}, {o.name for o in gt.objects}
                    same_reference = (cached.get("gt_directory") is not None
                                      and Path(cached["gt_directory"]).resolve() == reference.resolve())
                    previous = saved.get("mapping", {}) if same_reference else {}
                    proposed = {name: previous.get(name) for name in names - fixed.keys()}
                    mapping = validate_mapping({"objects": proposed}, sorted(proposed), targets, fixed)
                    evidence.update(saved)
                    evidence.update(mapping=mapping, fixed=fixed, source="cached_or_exact_identity",
                                    unmapped=sorted(names - mapping.keys()))
                else:
                    mapping = map_objects(candidate, gt, instruction, image_path, model,
                                          reasoning_effort, checks["object_mapping"])
                init = check_init(candidate, gt, mapping)
            except Exception as error:
                checks["mapping_error"] = f"{type(error).__name__}: {error}"
                # A failed VLM request does not discard exact object identities.
                mapping = align_worlds(candidate, gt).mapping()
                checks["object_mapping"].update(mapping=mapping, source="exact_identity_fallback")
                init = check_init(candidate, gt, mapping)
            if solver.get("returncode") in {10, 11, 12}:
                missing = check_missing_init(candidate, gt, mapping)
                checks["missing_init"] = missing
                if missing["matched"]:
                    init["checks"].extend(missing["matches"])
                    init["status"] = "fail"
            if reference_valid and init["status"] != "fail":
                access = check_initial_access(candidate, gt, mapping, reference)
                checks["initial_access"] = access
                if access["matched"]:
                    init["checks"].extend(access["matches"])
                    init["status"] = "fail"
            checks["init"] = init
            goal = check_goal_error(candidate, gt, mapping)
            semantics = check_plan_semantics(candidate, gt, mapping, candidate_dir, reference)
            invariants = (check_effect_invariants(candidate, gt, unsolved=solver.get("returncode") in {10, 11, 12}, mapping=mapping)
                          if reference_valid else {"matched": False, "matches": []})
            semantics["matches"].extend(invariants["matches"])
            semantics["matched"] = bool(semantics["matches"])
            semantics["effect_invariant_uncertainty"] = invariants.get("uncertain", [])
            checks["operator_semantics"] = semantics
            causal_goals = check_causal_goals(
                candidate, gt, mapping, unsolved=reference_valid and solver.get("returncode") in {10, 11, 12},
                final_state=({tuple(f) for f in semantics["final_state"]}
                             if semantics["final_state"] is not None else None),
                executed_effects=({"candidate": semantics["executed_effects"],
                                   "candidate_plan": ground_saved_plan(candidate_dir, candidate),
                                   "reference_plan": ground_saved_plan(reference, gt)}
                                  if semantics["executed_effects"] is not None else None),
                candidate_owners=lid_owners(candidate), gt_owners=lid_owners(gt),
            )
            goal["matches"].extend(causal_goals["matches"])
            goal.update(matched=bool(goal["matches"]), unachievable=causal_goals["unachievable"],
                        uncertain=causal_goals["uncertain"])
            if any(item.get("section") == ":goal" for item in issues):
                goal["matches"].append({"kind": "contradictory_goal_literals", "issues": issues})
                goal["matched"] = True
            checks["goal_error"] = goal
            coverage = check_operator_coverage(candidate, gt, reference)
            checks["operator_coverage"] = coverage
            plans = compare_plans(candidate_dir / "plan.txt", reference / "plan.txt")
            checks["plans"] = plans
            checks["judge_failed"] = judge.get("pass") is False
            label, reason = classify(solver, init, counts, checks["judge_failed"], plans, goal,
                                     coverage=coverage, semantics=semantics)
            if label == "else" and reference_valid:
                cardinality = check_spatial_cardinality(candidate, gt, mapping, candidate_dir)
                cardinality.extend(check_semantic_cardinality(candidate, gt, candidate_dir, mapping))
                checks["cardinality"] = cardinality
                if cardinality:
                    reason = cardinality[0]["kind"]
                    label = "init_error" if reason == "declared_instance_deficit" else "goal_error"
            if label == "else" and reference_valid:
                effects = check_effect_reachability(candidate, gt, candidate_dir)
                checks["effect_reachability"] = effects
                if effects["matches"]:
                    label, reason = "operator_semantics", effects["matches"][0]["kind"]
            if label == "else":
                strict_semantics = check_plan_semantics(
                    candidate, gt, mapping, candidate_dir, reference, strict=True,
                    candidate_owners=lid_owners(candidate), gt_owners=lid_owners(gt))
                checks["strict_operator_semantics"] = strict_semantics
                if strict_semantics["matched"]:
                    label, reason = "operator_semantics", strict_semantics["matches"][0]["kind"]
            if label == "else" and solver.get("returncode") in {10, 11, 12}:
                causal = check_causal_failure(candidate, gt, mapping, candidate_dir, reference)
                checks["causal_reachability"] = causal
                if causal.get("raw_init_matches"):
                    label, reason = "init_error", causal["raw_init_matches"][0]["kind"]
                elif causal["matches"]:
                    label, reason = "operator_semantics", causal["matches"][0]["kind"]
            if label == "else" and reference_valid:
                physical = check_physical_transitions(candidate, gt, mapping, candidate_dir, reference)
                checks["physical_transition_semantics"] = physical
                if physical["matched"]:
                    label, reason = "operator_semantics", physical["matches"][0]["kind"]
            if label == "else" and reference_valid:
                order = check_instruction_order(candidate, gt, mapping, candidate_dir, reference, instruction)
                checks["instruction_order"] = order
                if order["matched"]:
                    label, reason = "operator_semantics", order["matches"][0]["kind"]
            if label == "else":
                if reference_valid and solver.get("returncode") == 0 and semantics["final_state"] is None:
                    reason = "candidate_trace_unavailable"
                checks["unresolved"] = unresolved_evidence(candidate, gt, mapping, checks)
    except Exception as error:
        checks["input_error"] = f"{type(error).__name__}: {error}"
    result.update(label=label, reason=reason)
    result.update(taxonomy_fields(result, candidate_dir))
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "attribution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
