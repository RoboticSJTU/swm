from __future__ import annotations

from collections import defaultdict
from copy import copy
from functools import reduce
import heapq
import json
import re
from operator import or_
from pathlib import Path

from swm.llm import call_gpt_json
from swm.pddl.reorder import ReorderModel, evaluate, expression_bits, search_plan_classes
from swm.pddl.strips import parse_sexpr_file
from swm.pddl.plan_branches import factor_plan_orders, render_plan_orders, render_named_orders


def _validate_verdict(result):
    if not isinstance(result, dict) or set(result) != {'reasoning', 'pass', 'feedback'}:
        raise ValueError('Judge must return exactly reasoning, pass, and feedback')
    if not isinstance(result['reasoning'], str) or not result['reasoning'].strip():
        raise ValueError('Judge reasoning must be a nonempty string')
    if type(result['pass']) is not bool or not isinstance(result['feedback'], str):
        raise ValueError('Judge pass must be boolean and feedback must be a string')
    value = {**result, 'reasoning': result['reasoning'].strip(), 'feedback': result['feedback'].strip()}
    if bool(value['feedback']) == value['pass']:
        raise ValueError('Passing verdicts require empty feedback; rejections require evidence')
    return value


def _call_validated_judge(
    model: str, prompt: str, first_img: Path | list[Path], capture: dict | None = None
) -> dict:
    if isinstance(first_img, (list, tuple)) and len(first_img) > 1:
        prompt = "All attached images are views of the same initial scene. Use the complete set of views and preserve object identity across views.\n\n" + prompt
    last_error: Exception | None = None
    attempts = []
    for attempt in range(1, 4):
        attempt_capture = {"attempt": attempt}
        try:
            call_kwargs = {"attempts": 1}
            if capture is not None:
                call_kwargs["capture"] = attempt_capture
            images = list(first_img) if isinstance(first_img, (list, tuple)) else [first_img]
            result = call_gpt_json(model, prompt, images, **call_kwargs)
            result = _validate_verdict(result)
            attempts.append(attempt_capture)
            if capture is not None:
                capture["attempts"] = attempts
            return result
        except (RuntimeError, ValueError) as error:
            attempt_capture["error_type"] = type(error).__name__
            attempt_capture["error"] = str(error)
            attempts.append(attempt_capture)
            last_error = error
    if capture is not None:
        capture["attempts"] = attempts
    raise ValueError(
        "Judge did not return the required flat judge schema after 3 attempts: "
        f"{last_error}"
    )


class SymbolicTraceError(ValueError):
    """PDDL trace preparation failed; do not fall back to an NL judgment."""


def _evaluated_symbolic_trace(
    candidate_plan: str,
    predicted_domain: str | Path | None,
    pddl_plan: str | Path | None,
) -> str:
    if predicted_domain is None and pddl_plan is None:
        return candidate_plan  # Explicit natural-language evaluation mode.
    if predicted_domain is None or pddl_plan is None:
        raise SymbolicTraceError("PDDL judging requires both domain and plan paths")

    try:
        domain_path, plan_path = Path(predicted_domain), Path(pddl_plan)
        raw_plan = []
        for line in plan_path.read_text(encoding="utf-8").splitlines():
            line = line.split(";", 1)[0].strip()
            if line and (not line.startswith("(") or not line.endswith(")")
                         or "(" in line[1:] or ")" in line[:-1]):
                raise ValueError(f"Malformed plan line: {line}")
            if line:
                parts = line[1:-1].lower().split()
                if not parts:
                    raise ValueError("Empty plan action")
                raw_plan.append((parts[0], parts[1:]))
        if not raw_plan:
            return "Candidate trace contains zero actions."
        root = parse_sexpr_file(domain_path)
        source_actions = {
            node[1]: dict(zip(node[2::2], node[3::2]))
            for node in root[1:]
            if isinstance(node, list) and node[:1] == [":action"]
        }
        derived = [node for node in root[1:] if isinstance(node, list) and node[:1] == [":derived"]]

        def effect_predicates(node):
            if not node:
                return set()
            if node[0] == "and":
                return set().union(*(effect_predicates(child) for child in node[1:]))
            if node[0] in {"forall", "when"}:
                return effect_predicates(node[2])
            if node[0] == "not":
                return {node[1][0]}
            return {node[0]}

        dynamic_predicates = {node[1][0] for node in derived}
        for fields in source_actions.values():
            dynamic_predicates.update(effect_predicates(fields[":effect"]))

        def sexpr(node, bindings):
            if isinstance(node, str):
                return bindings.get(node, node)
            if node[0] in {"forall", "exists"}:
                # Quantified variables shadow action parameters of the same name.
                local = {key: value for key, value in bindings.items() if key not in node[1]}
                return f"({node[0]} ({' '.join(node[1])}) {sexpr(node[2], local)})"
            return "(" + " ".join(sexpr(child, bindings) for child in node) + ")"

        def format_literals(expression, bindings, *, effect=False):
            if not expression:
                return []
            if expression[0] == "and":
                return [text for child in expression[1:]
                        for text in format_literals(child, bindings, effect=effect)]
            negative = expression[0] == "not"
            atom = expression[1] if negative else expression
            if atom[0] in {"forall", "exists", "when", "or", "imply", "and"}:
                # Preserve guards and logical structure, without reading candidate init.
                return [sexpr(expression, bindings)]
            if atom[0] not in dynamic_predicates and atom[0] != "=":
                return []
            prefix = ("-" if negative else "+") if effect else ("not " if negative else "")
            arguments = ", ".join(sexpr(argument, bindings) for argument in atom[1:])
            return [f"{prefix}{atom[0]}({arguments})"]

        lines = []
        if derived:
            lines.append("Derived rules: " + " ".join(sexpr(node, {}) for node in derived))
        for index, (name, args) in enumerate(raw_plan, start=1):
            fields = source_actions[name]
            parameters = [token for token in fields[":parameters"]
                          if isinstance(token, str) and token.startswith("?")]
            if len(args) != len(parameters):
                raise ValueError(f"Arity mismatch for {name}: expected {len(parameters)}, got {len(args)}")
            bindings = dict(zip(parameters, args))
            manipulators = [arg for arg in args
                            if any(part in {"arm", "hand", "gripper"} for part in arg.split("_"))]
            objects = [arg for arg in args if arg not in manipulators]
            before = format_literals(fields[":precondition"], bindings)
            changes = format_literals(fields[":effect"], bindings, effect=True)
            action_text = f"{name}({', '.join(objects)})"
            if manipulators:
                action_text += f" with {' and '.join(manipulators)}"
            lines.append(f"{index}. {action_text}")
            if before:
                lines.append(f"   Before: {', '.join(before)}")
            if changes:
                lines.append(f"   State change: {', '.join(changes)}")
        return "\n".join(lines)
    except (OSError, KeyError, ValueError, NotImplementedError, IndexError, TypeError) as error:
        raise SymbolicTraceError(f"Cannot prepare PDDL judge trace: {error}") from error


def build_group_judge_prompt(
    action_library: list[dict],
    candidates: list[dict],
    *,
    instruction: str,
    kf_actions: str,
    comparison_question: str = "",
    multiview: bool = False,
    presentation: str = "plain",
) -> str:
    """Reuse the single-plan audit with shared operations and labelled paths.

    Only bound operator descriptions enter the trace. Predicted problem init
    and simulated states are deliberately absent from this interface.
    Plain preserves complete input paths; named adds bound headlines; linear
    also orders similar paths; branches retains the prefix-tree presentation.
    """
    actions = {item["action_id"]: item["description"] for item in action_library}
    if (not actions or len(actions) != len(action_library)
            or any(type(i) is not int or i < 1 for i in actions)
            or any(not isinstance(text, str) or not text.strip() for text in actions.values())):
        raise ValueError("Expected distinct positive action IDs and complete descriptions")
    ids = [candidate["id"] for candidate in candidates]
    if (not ids or any(not isinstance(i, str) or not i or any(c.isspace() or c in ':,[]' for c in i) for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Expected distinct single-line candidate IDs")
    for candidate in candidates:
        order = candidate["order"]
        if (not isinstance(order, (list, tuple)) or any(type(i) is not int for i in order)
                or sorted(order) != sorted(actions)):
            raise ValueError("Every candidate must preserve all action occurrences")

    if presentation not in {"branches", "linear", "plain", "named"}:
        raise ValueError("Unknown group presentation")
    if presentation in {"linear", "plain", "named"}:
        trace = [
            "All plans start from the same actual initial scene. Action IDs are unordered labels, not step numbers. "
            "Keep each plan's state, completed operations and verdict separate.",
            "Each row is one complete plan. Follow its action IDs left to right.",
        ]
    else:
        trace = [
            "All plans start from the same actual initial scene. Track a shared prefix once, "
            "then follow each plan's branch from that prefix state.",
            "Each line names its plans and execution steps. Read action IDs left to right; "
            "the action dictionary is unordered.",
            "Shared continuations apply separately to each branch using that branch's state and audit history. "
            "Keep states and verdicts separate after a fork.",
        ]
    trace.append("Shared actions:")
    definitions = {}
    for identifier, description in sorted(actions.items()):
        definitions.setdefault(description, []).append(f"A{identifier:02d}")
    trace.extend(", ".join(labels) + ": " + description for description, labels in definitions.items())
    if presentation in {"linear", "plain", "named"}:
        trace.append("Plan orders:")
        if presentation in {"linear", "named"}:
            trace.append(render_named_orders(candidates, actions, sort_similar=presentation == "linear"))
        else:
            trace.extend(candidate['id'] + ': ' + ' -> '.join(f'A{i:02d}' for i in candidate['order'])
                         for candidate in candidates)
    else:
        trace.append("Plan branches:")
        trace.append(render_plan_orders(factor_plan_orders(candidates)))
    if comparison_question:
        if presentation in {"linear", "plain", "named"}:
            trace.append("Shared review question: " + comparison_question + " Apply the complete audit to every plan.")
        else:
            trace.append("Shared review question: " + comparison_question)
            trace.append("Apply the full audit to every plan; sharing a question does not imply a shared verdict.")

    template = Path(__file__).parent.parent / "prompt_templates" / "pddl_judge.txt"
    audit, marker, _ = template.read_text(encoding="utf-8").rpartition("\n# Output\n")
    if not marker:
        raise ValueError("Judge template is missing its output section")
    prompt = audit.format(instruction=instruction, kf_actions=kf_actions,
                          evaluated_symbolic_trace="\n".join(trace))
    prompt += ('\n# Output\nReturn exactly one JSON object: {"results":[{"candidate_id":"...","reasoning":"...",'
               '"pass":true,"feedback":""}]}. '
               'Return one independent result for each supplied candidate ID. '
               'feedback must be empty if pass=true; otherwise identify the first invalid action or unmet requirement and its evidence.\n')
    if multiview:
        prompt = "All attached images show the same initial scene; preserve object identity across views.\n\n" + prompt
    return prompt


def validate_plan_feasibility(
    model: str,
    first_img: Path | list[Path],
    instruction: str,
    kf_actions: str,
    candidate_plan: str,
    predicted_domain: str | Path | None = None,
    pddl_plan: str | Path | None = None,
    capture: dict | None = None,
    scene_context: str = "",
):
    candidate_plan = candidate_plan.strip()
    prompt_path = Path(__file__).parent.parent / "prompt_templates" / "pddl_judge.txt"
    prompt = prompt_path.read_text(encoding="utf-8").format(
        instruction=instruction,
        kf_actions=kf_actions,
        evaluated_symbolic_trace=_evaluated_symbolic_trace(
            candidate_plan, predicted_domain, pddl_plan
        ),
    )
    if scene_context:
        prompt = "Official benchmark context:\n" + scene_context + "\n\n" + prompt
    if capture is not None:
        capture.update({"decision_source": "vlm", "model": model})
    return _call_validated_judge(model, prompt, first_img, capture)


# One declared state policy for single- and dual-arm plans.
PROCESS_SCHEMA = {'version': 2,
 'location_relations': {'on': [1, 2], 'in': [1, 2], 'at': [1, 2]},
 'hierarchy_relations': {'part_of': [1, 2], 'drawer_of': [1, 2], 'lock_of': [1, 2]},
 'record_relations': {'scanned_with': [1]},
 'resource_free_predicates': ['hand_free'],
 'resource_relations': {'holding': {'resource': 1, 'payload': 2},
                        'robot_holding': {'resource': 2, 'payload': 3}},
 'unknown_relations': 'protect_all_participants_and_current_connections',
 'reference_sites': 'only_receive_location_references_and_have_no_own_writes_or_other_relations',
 'description': 'An operation has focal objects and contextual supports/sources/destinations. '
                'Focal containers expose their contents; contextual containers expose their own '
                'state/access without reading sibling contents. Hierarchies propagate focal '
                'observation downwards and contextual access upwards. Objects whose state cells or '
                'resource payloads are directly used become focal; remaining location/reference '
                'arguments are contextual. Recorded identities preserve values without ongoing '
                'attachment. Unknown multi-object relations conservatively protect all '
                'participants. No action or episode names enter the policy.'}

class ProcessModel:
    """Project parameter, nested-content, enclosing-environment and resource state."""

    def __init__(self, model, observer):
        self.model, self.schema = (model, PROCESS_SCHEMA)
        facts = set(model.bits) | model.initial_facts
        self.resource_names = {f[1] for f in facts if len(f) == 2 and (f[0] == 'hand' or f[0] in self.schema['resource_free_predicates'])}
        location, records = (self.schema['location_relations'], self.schema['record_relations'])
        hierarchical = {**location, **self.schema['hierarchy_relations']}
        self.owners, self.resource_facts, self.link_facts = ({}, [], [])
        for fact in sorted(facts):
            args, names = (set(fact[1:]), set(fact[1:]) & self.resource_names)
            if names and fact[0] not in records:
                payloads = args - self.resource_names
                role = self.schema['resource_relations'].get(fact[0])
                if role and max(role.values()) < len(fact) and (fact[role['resource']] in names):
                    payloads = {fact[role['payload']]}
                self.resource_facts.append((fact, names, payloads))
            elif len(fact) > 1:
                if fact[0] in hierarchical:
                    payload, parent = hierarchical[fact[0]]
                    if not (1 <= payload < len(fact) and 1 <= parent < len(fact) and (payload != parent)):
                        raise ValueError('Location relation argument roles do not match ' + repr(fact))
                    owners = {fact[payload]}
                    self.link_facts.append((fact, {fact[payload], fact[parent]}, fact[payload], fact[parent]))
                elif fact[0] in records:
                    roles = records[fact[0]]
                    if not roles or not all((1 <= p < len(fact) for p in roles)):
                        raise ValueError('Record relation owner roles do not match ' + repr(fact))
                    owners = {fact[p] for p in roles}
                else:
                    owners = args - self.resource_names
                    if len(owners) > 1:
                        self.link_facts.append((fact, owners, None, None))
                self.owners[fact] = owners
        written = reduce(or_, (a.possible_add | a.possible_delete for a in model.actions), 0)
        written_owners = set().union(*(owners for f, owners in self.owners.items() if model.bits.get(f, 0) & written))
        self.references = set()
        for obj in set(model.objects) - self.resource_names - written_owners:
            relevant = [f for f in facts if obj in f[1:]]
            incoming = False
            for fact in relevant:
                if len(fact) == 2:
                    continue
                role = location.get(fact[0])
                if role and fact[role[1]] == obj and (fact[role[0]] != obj):
                    incoming = True
                else:
                    break
            else:
                if incoming:
                    self.references.add(obj)
        objects = sorted(set(model.objects) - self.resource_names - self.references)
        self.object_bits = {obj: 1 << i for i, obj in enumerate(objects)}
        self.owner_masks, self.selector_masks = ({o: 0 for o in objects}, {o: 0 for o in objects})
        self.global_mask = sum((bit for fact, bit in model.bits.items() if len(fact) == 1))
        for fact, owners in self.owners.items():
            for owner in owners:
                if owner in self.owner_masks:
                    self.owner_masks[owner] |= model.bits.get(fact, 0)
        self.links = []
        for fact, endpoints, child, parent in self.link_facts:
            members = sum((self.object_bits.get(o, 0) for o in endpoints))
            selector = model.bits.get(fact, 0)
            active_static = not selector and fact in model.initial_facts
            if parent in self.selector_masks:
                self.selector_masks[parent] |= selector
            if not endpoints & self.references:
                self.links.append((selector, active_static, members, self.object_bits.get(child, 0), self.object_bits.get(parent, 0), child is not None))
        self.resource_masks = {name: 0 for name in self.resource_names}
        self.resource_selectors = []
        for fact, names, payloads in self.resource_facts:
            bit = model.bits.get(fact, 0)
            for name in names:
                self.resource_masks[name] |= bit
            self.resource_selectors.append((bit, not bit and fact in model.initial_facts, sum((self.object_bits.get(o, 0) for o in payloads)), names))
        resource_roles = {f: (names, payloads) for f, names, payloads in self.resource_facts}
        self.roots, self.context_roots, self.dependencies, self.explicit = ([], [], [], [])
        for i, action in enumerate(model.actions):
            relevant = model.facts(observer.guards[i] | action.possible_add | action.possible_delete)
            mentioned = set(action.args) | {o for f in relevant for o in f[1:]}
            focal, context = (set(), set())
            for fact in relevant:
                if fact in resource_roles:
                    _, payloads = resource_roles[fact]
                    focal.update(payloads)
                    context.update(set(fact[1:]) - payloads - self.resource_names)
                elif fact in self.owners:
                    focal.update(self.owners[fact])
                    context.update(set(fact[1:]) - self.owners[fact] - self.resource_names)
            focal.update(mentioned - focal - context - self.resource_names)
            context.difference_update(focal)
            self.roots.append(sum((self.object_bits.get(o, 0) for o in focal)))
            self.context_roots.append(sum((self.object_bits.get(o, 0) for o in context)))
            self.dependencies.append(mentioned & self.resource_names)
            self.explicit.append(observer.guards[i])
        self.selection_bits = reduce(or_, (row[0] for row in self.links), 0) | reduce(or_, (row[0] for row in self.resource_selectors if row[2]), 0)
        self.mask_cache = [{} for _ in model.actions]
        self.reads = [self.project((1 << len(model.atoms)) - 1, i)[0] for i in range(len(model.actions))]

    def project(self, state, index):
        # Cache the selected objects/facts, never their current truth values.
        key = state & self.selection_bits
        observed = self.mask_cache[index].get(key)
        if observed is None:
            observed = self.select_mask(state, index)
            self.mask_cache[index][key] = observed
        return (observed, state & observed)

    def select_mask(self, state, index):
        focal, context = (self.roots[index], self.context_roots[index])
        changed = True
        while changed:
            changed = False
            for selector, static, members, child, parent, hierarchy in self.links:
                if not (static or state & selector):
                    continue
                old = (focal, context)
                if hierarchy:
                    if focal & parent:
                        focal |= child
                    if (focal | context) & child:
                        context |= parent
                elif (focal | context) & members:
                    focal |= members
                context &= ~focal
                changed |= (focal, context) != old
        scope = focal | context
        observed = self.global_mask | self.explicit[index]
        for obj, bit in self.object_bits.items():
            if scope & bit:
                observed |= self.owner_masks[obj]
            if focal & bit:
                observed |= self.selector_masks[obj]
        needed = set(self.dependencies[index])
        for selector, static, members, names in self.resource_selectors:
            if scope & members and (static or state & selector):
                needed.update(names)
        for name in needed:
            observed |= self.resource_masks[name]
        return observed

    def restrict(self, identifiers, model):
        result = copy(self)
        result.model = model
        result.mask_cache = [{} for _ in identifiers]
        for name in ('roots', 'context_roots', 'dependencies', 'explicit', 'reads'):
            setattr(result, name, [getattr(self, name)[i - 1] for i in identifiers])
        return result

    def mask_stable_under_action(self, index):
        """Prove once that an action cannot change its own observation scope.

        Changed hierarchy edges already have a focal child and a seeded parent;
        changed unknown edges have every endpoint focal. Changed resource links
        name resources already required by the action. Their activation/deletion
        therefore cannot add/remove any object/resource from this action's view.
        """
        action = self.model.actions[index]
        writes = action.possible_add | action.possible_delete
        focal, context = (self.roots[index], self.context_roots[index])
        for selector, _, members, child, parent, hierarchy in self.links:
            if not selector & writes:
                continue
            if hierarchy:
                if child & ~focal or parent & ~(focal | context):
                    return False
            elif members & ~focal:
                return False
        return all((not selector & writes or names <= self.dependencies[index] for selector, _, members, names in self.resource_selectors if members))

def shared_questions(model, representatives, observer, signatures):
    if not representatives:
        return []
    catalog = observer.catalog()
    variants = defaultdict(set)
    groups = {}
    for signature in [observer.reference, *signatures.values()]:
        for action, token in enumerate(signature[1], 1):
            variants[action].add(token)
    owners = observer.process.owners
    resources = observer.process.resource_names
    locations = observer.process.schema['location_relations']

    def key_for(fact):
        hands = tuple(sorted(set(fact[1:]) & resources))
        if hands:
            return ('resource', *hands)
        if fact[0] in locations:
            return ('contents_or_location', fact[locations[fact[0]][1]])
        return ('object', *sorted(owners.get(fact, set(fact[1:])))) if len(fact) > 1 else ('scene',)

    def add(key, row):
        if key not in groups:
            groups[key] = {'key': key, 'kind': key[0], 'subjects': list(key[1:]), 'rows': [], 'members': set()}
        groups[key]['rows'].append(row)
        for candidate in representatives:
            sig = signatures[candidate['id']]
            value = bool(sig[0] & model.bits[tuple(row['fact'])]) if row['phase'] == 'final' else row['variants'][str(sig[1][row['action_id'] - 1])]
            if value != row['baseline']:
                groups[key]['members'].add(candidate['id'])
    for action, tokens in sorted(variants.items()):
        if len(tokens) == 1:
            continue
        events = {str(t): catalog[str(t)] for t in tokens}
        baseline = str(observer.reference[1][action - 1])
        for phase in ('before', 'after'):
            observed = {t: set(map(tuple, e[phase + '_observed'])) for t, e in events.items()}
            truth = {t: set(map(tuple, e[phase + '_true'])) for t, e in events.items()}
            for fact in sorted(set().union(*observed.values())):
                values = {t: ('true' if fact in truth[t] else 'false') if fact in observed[t] else 'unobserved' for t in events}
                if len(set(values.values())) > 1:
                    add(key_for(fact), {'action_id': action, 'phase': phase, 'fact': list(fact), 'variants': values, 'baseline': values[baseline]})
        for field in ('add', 'delete'):
            sets = {t: set(map(tuple, e[field])) for t, e in events.items()}
            for fact in sorted(set().union(*sets.values())):
                values = {t: fact in fs for t, fs in sets.items()}
                if len(set(values.values())) > 1:
                    add(key_for(fact), {'action_id': action, 'phase': field, 'fact': list(fact), 'variants': values, 'baseline': values[baseline]})
        for field in ('branches', 'supports'):
            values = {t: json.dumps(e[field], separators=(',', ':')) for t, e in events.items()}
            if len(set(values.values())) > 1:
                add(('operation_outcome', action), {'action_id': action, 'phase': field, 'variants': values, 'baseline': values[baseline]})
    finals = {s[0] for s in [observer.reference, *signatures.values()]}
    different = 0
    for state in finals:
        different |= state ^ observer.reference[0]
    for fact in sorted(model.facts(different)):
        add(key_for(fact), {'action_id': 0, 'phase': 'final', 'fact': list(fact), 'bit': model.bits[fact], 'baseline': bool(observer.reference[0] & model.bits[fact])})
    return [{**g, 'members': sorted(g['members'])} for _, g in sorted(groups.items(), key=lambda p: repr(p[0]))]


class ExecutionObserver:
    """One observation policy for every arm count; no instruction-based merging."""

    def __init__(self, model):
        self.model = model
        self.guards = []
        for action in model.actions:
            reads = expression_bits(action.guard)
            for guard, _, _ in action.effects:
                reads |= expression_bits(guard)
            self.guards.append(reads)
        self.process = ProcessModel(model, self)
        self.simple_context = all(
            (a.guard is True or isinstance(a.guard, tuple) and a.guard[0] == 'mask')
            and all(g is True for g, _, _ in a.effects)
            and self.process.mask_stable_under_action(i)
            for i, a in enumerate(model.actions)
        )
        self.cache, self.tokens, self.events = {}, {}, {}
        self.reference = self.profile(list(range(1, len(model.actions) + 1)))

    def event(self, state, identifier):
        key = state, identifier
        if key not in self.cache:
            i, model = identifier - 1, self.model
            after = model.apply(state, i)
            if after is None:
                raise ValueError('Operation observation requires an applicable action')
            before = self.process.project(state, i)
            action = model.actions[i]
            if self.simple_context:
                fixed = 0 if action.guard is True else action.guard[1] | action.guard[2]
                token_key = identifier, (before[0] & ~fixed, before[1] & ~fixed)
                signature = (identifier, before, (before[0], after & before[0]),
                             (action.possible_add, action.possible_delete), tuple(True for _ in action.effects), ())
            else:
                signature = (identifier, before, self.process.project(after, i), model.activated(state, i),
                             tuple(evaluate(g, state) for g, _, _ in action.effects), ())
                token_key = signature
            if token_key not in self.tokens:
                token = len(self.tokens) + 1
                self.tokens[token_key] = token
                self.events[token] = signature
            self.cache[key] = after, self.tokens[token_key]
        return self.cache[key]

    def profile(self, order):
        if sorted(order) != list(range(1, len(self.model.actions) + 1)):
            raise ValueError('Reordering must preserve every action occurrence')
        state, tokens = self.model.initial, [0] * len(order)
        for identifier in order:
            state, tokens[identifier - 1] = self.event(state, identifier)
        if not evaluate(self.model.goal, state):
            raise ValueError('Expected a goal-reaching plan')
        return state, tuple(tokens), ()

    def catalog(self):
        def facts(mask):
            return [list(f) for f in sorted(self.model.facts(mask))]

        return {str(token): {'action_id': e[0], 'before_observed': facts(e[1][0]), 'before_true': facts(e[1][1]),
                            'after_observed': facts(e[2][0]), 'after_true': facts(e[2][1]),
                            'add': facts(e[3][0]), 'delete': facts(e[3][1]), 'branches': list(e[4]), 'supports': []}
                for token, e in self.events.items()}


def _action_library(model):
    trace = _evaluated_symbolic_trace('', model.domain_path, model.plan_path)
    chunks = re.findall(r'^\d+\. (.*?)(?=^\d+\. |\Z)', trace, re.M | re.S)
    if len(chunks) != len(model.actions):
        raise ValueError('Bound action descriptions do not match the plan')
    return [{'action_id': i, 'description': text.rstrip('\n')} for i, text in enumerate(chunks, 1)]


def group_plan_classes(model, representatives, observer, instruction, kf_actions, *, multiview=False):
    """Pack by shared state questions, retaining complete independent timelines."""
    signatures = {c['id']: observer.profile(c['order']) for c in representatives}
    if len(signatures) != len(representatives):
        raise ValueError('Duplicate representative ID')
    topics = shared_questions(model, representatives, observer, signatures)
    by_candidate = defaultdict(list)
    frequency = {repr(t['key']): len(t['members']) for t in topics}
    for topic in topics:
        for identifier in topic['members']:
            by_candidate[identifier].append(repr(topic['key']))
    catalog, library = observer.catalog(), _action_library(model)
    event_cost = {int(t): len(json.dumps(e, separators=(',', ':'))) for t, e in catalog.items()}

    def sort_key(candidate):
        shared = sorted(by_candidate[candidate['id']], key=lambda k: (-frequency[k], k))
        primary = shared[0] if shared else ''
        return (candidate['id'] != 'reference', -frequency.get(primary, 0), primary, tuple(shared),
                tuple(observer.events[t] for t in signatures[candidate['id']][1]), tuple(candidate['order']))

    queue = sorted(representatives, key=sort_key)
    rank = {c['id']: i for i, c in enumerate(queue)}
    event_sets = {i: set(s[1]) for i, s in signatures.items()}
    topic_sets = {i: set(by_candidate[i]) for i in signatures}

    def prompt(candidates):
        diversity = {}
        if len(candidates) > 1:
            for action in candidates[0]['order']:
                variants = set()
                for c in candidates:
                    event = catalog[str(signatures[c['id']][1][action - 1])]
                    variants.add((tuple(map(tuple, event['before_observed'])), tuple(map(tuple, event['before_true']))))
                if len(variants) > 1:
                    diversity[action] = len(variants)
        question = ''
        if diversity:
            focus = max(diversity, key=lambda a: (diversity[a], -a))
            question = (f"When A{focus:02d} occurs, do the objects' states, contents and enclosing environment "
                        'allow the operation and satisfy Instruction in each plan?')
        return build_group_judge_prompt(library, candidates, instruction=instruction, kf_actions=kf_actions,
                                        comparison_question=question, multiview=multiview)

    def choose(available, used_events, used_topics):
        return min(available, key=lambda c: (sum(event_cost[t] for t in event_sets[c['id']] - used_events),
                                            -len(topic_sets[c['id']] & used_topics), rank[c['id']]))

    batches, pending, rendered = [], [], ''
    while queue:
        if len(pending) == 10:
            batches.append({'candidates': pending, 'prompt': rendered})
            pending = []
        if not pending:
            proposed, available, used_events, used_topics = [], list(queue[:64]), set(), set()
            while available and len(proposed) < 10:
                c = available[0] if not proposed else choose(available, used_events, used_topics)
                proposed.append(c)
                available.remove(c)
                used_events.update(event_sets[c['id']])
                used_topics.update(topic_sets[c['id']])
            rendered = prompt(proposed)
            if len(rendered) <= 48000:
                batches.append({'candidates': proposed, 'prompt': rendered})
                taken = {c['id'] for c in proposed}
                queue = [c for c in queue if c['id'] not in taken]
                continue
        used_events = set().union(*(event_sets[c['id']] for c in pending))
        used_topics = set().union(*(topic_sets[c['id']] for c in pending))
        options = (heapq.nsmallest(10, queue[:64], key=lambda c: (
            sum(event_cost[t] for t in event_sets[c['id']] - used_events),
            -len(topic_sets[c['id']] & used_topics), rank[c['id']])) if pending else queue[:1])
        for candidate in options:
            trial = pending + [candidate]
            trial_prompt = prompt(trial)
            if len(trial_prompt) <= 48000:
                queue.remove(candidate)
                pending, rendered = trial, trial_prompt
                break
        else:
            if not pending:
                raise ValueError('One complete review prompt exceeds capacity')
            batches.append({'candidates': pending, 'prompt': rendered})
            pending = []
    if pending:
        batches.append({'candidates': pending, 'prompt': rendered})
    actual = [c['id'] for b in batches for c in b['candidates']]
    if len(actual) != len(representatives) or set(actual) != set(signatures):
        raise AssertionError('Grouping omitted or duplicated a representative')
    return library, batches


def validation_inputs(round_dir, model, first_img, instruction, kf_actions):
    """Bind resumable verdicts to the actual round and review evidence."""
    images = list(first_img) if isinstance(first_img, (list, tuple)) else [first_img]
    template = Path(__file__).parent.parent / 'prompt_templates/pddl_judge.txt'
    reference = Path(round_dir).parent / 'kf_actions.txt'
    return {
        'model': model, 'instruction': instruction, 'kf_actions': kf_actions,
        'files': {n: (Path(round_dir) / n).read_text() for n in ('domain.pddl', 'problem.pddl', 'plan.txt')},
        'images': [[str(Path(p).resolve()), Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in images],
        'audit_template': template.read_text(), 'state_policy': PROCESS_SCHEMA,
        'reference_source': reference.read_text() if reference.is_file() else None,
    }


def save_validation(round_dir, pfv, psv, inputs):
    """One atomic round result; pass is the final PFV-and-PSV decision."""
    passed, feedback = None, ''
    reasoning = 'PFV has not completed.' if pfv is None else pfv['reasoning']
    if pfv is not None and not pfv['pass']:
        passed, feedback = False, pfv['feedback']
    elif pfv is not None and psv is not None:
        if psv['status'] == 'rejected':
            passed = False
            feedback = '[PSV] ' + psv['counterexample']['feedback']
            reasoning += '\nPSV counterexample: ' + psv['counterexample']['reasoning']
        elif psv['status'] == 'passed':
            passed = True
            reasoning += '\nPSV: complete operation-logic class coverage; no rejected representative.'
        else:
            reasoning += '\nPSV incomplete; resume this round to finish review.'
    result = {'pass': passed, 'reasoning': reasoning, 'feedback': feedback,
              'pfv': pfv, 'psv': psv, 'inputs': inputs}
    path = Path(round_dir) / 'judge.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return result


def validate_plan_soundness(model, first_img, instruction, kf_actions, round_dir):
    """Review every other operation-logic class after this round's PFV passed.

    Search and transport failures remain incomplete. Only completed, validated
    responses produce verdicts. Removed orders inherit no instruction verdict.
    """
    directory = Path(round_dir)
    record = json.loads((directory / 'judge.json').read_text())
    inputs = validation_inputs(directory, model, first_img, instruction, kf_actions)
    if record['inputs'] != inputs or not record['pfv']['pass']:
        raise ValueError('PSV requires a passed PFV for these exact inputs')
    pfv, previous = record['pfv'], record['psv']
    psv = {'status': 'incomplete', 'search_complete': False, 'reviews': [], 'requests': [],
           'instruction_verdict_inherited': False, 'scope': 'fixed_reference_action_multiset'}
    if previous is not None:
        psv['requests'] = previous['requests']

    def save():
        save_validation(directory, pfv, psv, inputs)

    try:
        symbolic = ReorderModel(*(directory / n for n in ('domain.pddl', 'problem.pddl', 'plan.txt')))
        observer = ExecutionObserver(symbolic)
        representatives, proof = search_plan_classes(symbolic, observer)
        psv.update(search_complete=proof['coverage_complete'], search_status=proof['status'],
                   legal_orders=proof['raw_orders_including_reference'],
                   legal_order_count_exact=proof['raw_order_count_exact'], classes=len(representatives),
                   covered_orders=proof['accounted_orders'], class_counts=proof['classes'],
                   factorization=proof['factorization'])
        # The exact reference representative already has this round's PFV.
        pending = [c for c in representatives if c['id'] != 'reference']
        by_id = {c['id']: c for c in pending}
        found = {}
        if previous is not None:
            for row in previous['reviews']:
                if row['candidate_id'] in by_id and row['order'] == by_id[row['candidate_id']]['order']:
                    verdict = _validate_verdict({k: row[k] for k in ('reasoning', 'pass', 'feedback')})
                    found[row['candidate_id']] = {**row, **verdict}
        images = list(first_img) if isinstance(first_img, (list, tuple)) else [first_img]
        library, batches = group_plan_classes(symbolic, pending, observer, instruction, kf_actions,
                                             multiview=len(images) > 1)
        psv['groups'] = len(batches)
        psv['reference_review_source'] = 'pfv'

        def checkpoint():
            psv['reviews'] = [found[c['id']] for c in pending if c['id'] in found]
            rejected = [r for r in psv['reviews'] if not r['pass']]
            psv['unresolved_representatives'] = len(pending) - len(found)
            if rejected:
                psv['status'], psv['counterexample'] = 'rejected', rejected[0]
                psv['counterexample'] = {**rejected[0], 'plan': '\n'.join(
                    symbolic.actions[i - 1].line for i in rejected[0]['order']) + '\n'}
            elif proof['coverage_complete'] and not psv['unresolved_representatives']:
                psv['status'] = 'passed'
            else:
                psv['status'] = 'incomplete'
            save()

        checkpoint()

        def review(candidates, prompt):
            capture = {}
            request = {'candidates': candidates, 'prompt': prompt, 'capture': capture, 'status': 'in_progress'}
            psv['requests'].append(request)
            save()
            try:
                response = call_gpt_json(model, prompt, images, response_format={'type': 'json_object'},
                                         attempts=1, reasoning_effort='xhigh', max_tokens=16384,
                                         temperature=0, capture=capture)
                request['response'] = response
                if set(response) != {'results'} or not isinstance(response['results'], list):
                    raise ValueError('Group judge must return a results list')
                allowed = {c['id']: c for c in candidates}
                counts = defaultdict(int)
                for row in response['results']:
                    if isinstance(row, dict) and isinstance(row.get('candidate_id'), str):
                        counts[row['candidate_id']] += 1
                for row in response['results']:
                    if not isinstance(row, dict) or set(row) != {'candidate_id', 'reasoning', 'pass', 'feedback'}:
                        continue
                    identifier = row['candidate_id']
                    if not isinstance(identifier, str) or not identifier or not identifier.strip():
                        continue
                    original_id = identifier
                    if len(candidates) == len(response['results']) == 1:
                        identifier = candidates[0]['id']
                    if identifier not in allowed or counts[original_id] != 1:
                        continue
                    try:
                        verdict = _validate_verdict({k: row[k] for k in ('reasoning', 'pass', 'feedback')})
                    except ValueError:
                        continue
                    found[identifier] = {'candidate_id': identifier, 'order': allowed[identifier]['order'],
                                         **verdict, 'response_candidate_id': original_id,
                                         'request_index': len(psv['requests']) - 1}
                request['status'] = 'complete'
            except Exception as error:
                request.update(status='request_error', error_type=type(error).__name__, error=str(error))
            checkpoint()

        for batch in batches:
            candidates = [c for c in batch['candidates'] if c['id'] not in found]
            if not candidates:
                continue
            review(candidates, batch['prompt'] if len(candidates) == len(batch['candidates']) else
                   build_group_judge_prompt(library, candidates, instruction=instruction,
                                            kf_actions=kf_actions, multiview=len(images) > 1))
            # Invalid/missing rows get one independent retry. Valid rejections
            # and valid passes are never retried to seek a different answer.
            for candidate in candidates:
                if candidate['id'] not in found:
                    review([candidate], build_group_judge_prompt(library, [candidate], instruction=instruction,
                                                                kf_actions=kf_actions, multiview=len(images) > 1))
        checkpoint()
    except (OSError, ValueError, KeyError, NotImplementedError) as error:
        psv.update(status='incomplete', error_type=type(error).__name__, error=str(error))
        save()
    return psv

