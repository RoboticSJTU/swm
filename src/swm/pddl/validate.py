from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from swm.llm import call_gpt_json
from swm.pddl.psv import MAX_CLASSES, encode_plan_tree, group_clusters, review_inputs
from swm.pddl.psv_observer import AvailabilityObserver
from swm.pddl.reorder import (
    SameLengthModel, evaluate, expression_bits,
    search_same_length_classes as search_plan_classes,
)
from swm.pddl.strips import parse_sexpr_file
from swm.prompts import read_prompt

PSV_PROTOCOL = 'availability_v4_lcs_tree_cap50_v1'


def format_kf_actions(text: str) -> str:
    """Give judges numbered actions without extraction labels."""
    actions = []
    for line in text.splitlines():
        action = line.strip()
        action = re.sub(r'^(?:(?:\[G\d+\]|\d+[.)])\s*)+', '', action).strip()
        if action:
            actions.append(action)
    return '\n'.join(f'{i}. {action}' for i, action in enumerate(actions, 1))


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
    model: str, prompt: str, first_img: Path | list[Path], capture: dict | None = None,
    *, reasoning_effort: str | None = None,
) -> dict:
    if isinstance(first_img, (list, tuple)) and len(first_img) > 1:
        prompt = "All attached images are views of the same initial scene. Use the complete set of views and preserve object identity across views.\n\n" + prompt
    last_error: Exception | None = None
    attempts = []
    for attempt in range(1, 4):
        attempt_capture = {"attempt": attempt}
        try:
            call_kwargs = {"attempts": 1}
            if reasoning_effort is not None:
                call_kwargs["reasoning_effort"] = reasoning_effort
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
    pddl_plan: str | Path | list[str] | None,
) -> str:
    if predicted_domain is None and pddl_plan is None:
        return candidate_plan  # Explicit natural-language evaluation mode.
    if predicted_domain is None or pddl_plan is None:
        raise SymbolicTraceError("PDDL judging requires both domain and plan paths")

    try:
        domain_path = Path(predicted_domain)
        lines = pddl_plan if isinstance(pddl_plan, list) else Path(pddl_plan).read_text(encoding="utf-8").splitlines()
        raw_plan = []
        for line in lines:
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
            parameters = [token for token in fields.get(":parameters", [])
                          if isinstance(token, str) and token.startswith("?")]
            if len(args) != len(parameters):
                raise ValueError(f"Arity mismatch for {name}: expected {len(parameters)}, got {len(args)}")
            bindings = dict(zip(parameters, args))
            manipulators = [arg for arg in args
                            if any(part in {"arm", "hand", "gripper"} for part in arg.split("_"))]
            objects = [arg for arg in args if arg not in manipulators]
            before = format_literals(fields.get(":precondition", []), bindings)
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
    exact_length: int | None = None,
    comparison_order: list[int] | None = None,
    tree: bool = False,
    robot_configuration: str | None = None,
) -> str:
    """用原审核模板呈现完整计划；支持重复绑定及固定比较锚点。"""
    actions = {item["action_id"]: item["description"] for item in action_library}
    if (not actions or len(actions) != len(action_library)
            or any(type(i) is not int or i < 1 for i in actions)
            or any(not isinstance(text, str) or not text.strip() for text in actions.values())):
        raise ValueError("Expected distinct positive action IDs and complete descriptions")
    ids = [candidate["id"] for candidate in candidates]
    if (not ids or any(not isinstance(i, str) or not i or any(c.isspace() or c in ':,[]' for c in i) for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Expected distinct single-line candidate IDs")
    orders = [candidate['order'] for candidate in candidates]
    if comparison_order is not None:
        orders.append(comparison_order)
    for order in orders:
        if (not isinstance(order, (list, tuple)) or not order
                or any(type(i) is not int or i not in actions for i in order)
                or exact_length is not None and len(order) != exact_length):
            raise ValueError("Expected complete paths of known action IDs at the required length")

    trace = ["Shared actions:"]
    definitions = {}
    used = {identifier for order in orders for identifier in order}
    for identifier in sorted(used):
        definitions.setdefault(actions[identifier], []).append(f"A{identifier:02d}")
    trace.extend(", ".join(labels) + ": " + description for description, labels in definitions.items())
    trace.append("Every appearance executes that operation again.")
    if comparison_order is not None:
        trace.append("Comparison anchor (context only; no candidate verdict requested): " +
                     ' -> '.join(f'A{i:02d}' for i in comparison_order))
    if tree:
        trace.append(encode_plan_tree(candidates))
    else:
        trace.append("Plan orders:")
        trace.extend(candidate['id'] + ': ' + ' -> '.join(f'A{i:02d}' for i in candidate['order'])
                     for candidate in candidates)
    prompt = read_prompt("verify_soundness.txt").format(
        instruction=instruction, kf_actions=format_kf_actions(kf_actions),
        evaluated_symbolic_trace="\n".join(trace), comparison_question=comparison_question,
    )
    if multiview:
        prompt = "All attached images show the same initial scene; preserve object identity across views.\n\n" + prompt
    if robot_configuration:
        prompt = "Robot configuration: " + robot_configuration + "\n\n" + prompt
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
    *,
    reasoning_effort: str | None = None,
):
    kf_actions = format_kf_actions(kf_actions)
    candidate_plan = candidate_plan.strip()
    prompt = read_prompt("verify_feasibility.txt").format(
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
    return _call_validated_judge(model, prompt, first_img, capture, reasoning_effort=reasoning_effort)


PROCESS_SCHEMA = {'version': 3,
 'relation_arities': {'on': 2, 'in': 2, 'at': 2, 'part_of': 2, 'drawer_of': 2,
                      'lock_of': 2, 'scanned_with': 2, 'holding': 2, 'robot_holding': 3},
 'location_relations': {'on': [1, 2], 'in': [1, 2], 'at': [1, 2]},
 'hierarchy_relations': {'part_of': [1, 2], 'drawer_of': [1, 2], 'lock_of': [1, 2]},
 'record_relations': {'scanned_with': [1]},
 'resource_free_predicates': ['hand_free'],
 'resource_type_predicates': ['hand'],
 'resource_types': ['hand'],
 'resource_identity_source': 'explicit_types_or_initial_roles_only',
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



# PSV 等价去重：具体动作、相关前态及实际非手效果的计数多重集。
class LogicObserver:
    version = "hand_free_recursive_relevant_prestate_effect_counts_v2"
    class_equivalence = "context_counts"
    channel_history_semantics = True

    def __init__(self, model, *, schema=None):
        self.model = model
        self.schema = schema or PROCESS_SCHEMA
        self.resource_relations = self.schema.get("resource_relations", {})
        self.resource_set = self._hands()
        self.resource_names = tuple(sorted(self.resource_set))
        self.channels = (("all_events",),)
        self.tokens, self.events, self.token_channels = {}, {}, {}
        self.cache = {}
        self.trace_nodes, self.trace_intern = [None], {}
        self._scope_cache, self._link_cache = {}, {}

        arities = self.schema.get("relation_arities", {})
        self.hierarchy = {
            name: roles
            for field in ("location_relations", "hierarchy_relations")
            for name, roles in self.schema.get(field, {}).items()
            if name in model.predicates
            and len(model.predicates[name]) == arities.get(name, max(roles))
        }
        self.records = {
            name: roles for name, roles in self.schema.get("record_relations", {}).items()
            if name in model.predicates
            and len(model.predicates[name]) == arities.get(name, max(roles))
        }
        self.owner_masks = defaultdict(int)
        self.selector_masks = defaultdict(int)
        self.nonhand_mask = 0
        self.links, self.link_mask = [], 0
        # 每个状态位归属谁；包含关系同时保存父子链接及内容物成员位。
        for fact in set(model.bits) | model.initial_facts:
            bit = model.bits.get(fact, 0)
            if self._hand_fact(fact):
                continue
            self.nonhand_mask |= bit
            for owner in self._owners(fact):
                self.owner_masks[owner] |= bit
            if fact[0] in self.hierarchy:
                child_pos, parent_pos = self.hierarchy[fact[0]]
                child, parent = fact[child_pos], fact[parent_pos]
                self.links.append((bit, not bit and fact in model.initial_facts, child, parent))
                self.link_mask |= bit
                self.selector_masks[parent] |= bit

        self.operations, self.parameters, self.direct, self.explicit = [], [], [], []
        for action in model.actions:
            name = action.line[1:-1].split()[0]
            args = tuple(arg for arg in action.args if arg not in self.resource_set)
            self.operations.append((name, args))
            self.parameters.append(set(args))
            written = action.possible_add | action.possible_delete
            read = expression_bits(action.guard)
            for guard, _, _ in action.effects:
                read |= expression_bits(guard)
            self.direct.append(self._direct_roots(written, read))
            self.explicit.append((read | written) & self.nonhand_mask)
        self.reference = self.profile(model.reference_order)

    def _hands(self):
        model, schema = self.model, self.schema
        result = {fact[1] for fact in model.initial_facts
                  if len(fact) == 2 and fact[0] in schema.get("resource_type_predicates", ())}
        for obj, kind in model.objects.items():
            if any(t in model.parents and model.subtype(kind, t)
                   for t in schema.get("resource_types", ())):
                result.add(obj)
        if not result:
            result.update(f[1] for f in model.initial_facts
                          if len(f) == 2 and f[0] in schema.get("resource_free_predicates", ()))
            for fact in model.initial_facts:
                role = self.resource_relations.get(fact[0])
                if role and max(role.values()) < len(fact):
                    result.add(fact[role["resource"]])
        return result

    def _hand_fact(self, fact):
        return bool(set(fact[1:]) & self.resource_set)

    def _owners(self, fact):
        if fact[0] in self.hierarchy:
            return {fact[self.hierarchy[fact[0]][0]]}
        if fact[0] in self.records:
            return {fact[p] for p in self.records[fact[0]]}
        return set(fact[1:]) - self.resource_set

    def _direct_roots(self, written, read):
        # 直接对象由动作可能写入的状态确定；仅作支撑的对象不递归展开内容物。
        anchors, unknown = set(), set()
        for fact in self.model.facts(written):
            if self._hand_fact(fact):
                role = self.resource_relations.get(fact[0])
                if role and max(role.values()) < len(fact):
                    anchors.update({fact[role["payload"]]} - self.resource_set)
            elif len(fact) == 2:
                anchors.add(fact[1])
            elif fact[0] in self.hierarchy:
                anchors.add(fact[self.hierarchy[fact[0]][0]])
            else:
                unknown.update(self._owners(fact))
        parents = {
            fact[self.hierarchy[fact[0]][1]]
            for fact in self.model.facts(read | written)
            if not self._hand_fact(fact) and fact[0] in self.hierarchy
        }
        return anchors | (unknown - parents)

    def _adjacency(self, state):
        key = state & self.link_mask
        if key not in self._link_cache:
            children, parents = defaultdict(set), defaultdict(set)
            for bit, static, child, parent in self.links:
                if static or state & bit:
                    children[parent].add(child)
                    parents[child].add(parent)
            self._link_cache[key] = children, parents
        return self._link_cache[key]

    @staticmethod
    def _closure(seeds, adjacency):
        result, pending = set(seeds), list(seeds)
        while pending:
            node = pending.pop()
            for child in adjacency.get(node, ()):
                if child not in result:
                    result.add(child)
                    pending.append(child)
        return result

    def observation_mask(self, state, index):
        key = state & self.link_mask, index
        if key not in self._scope_cache:
            children, parents = self._adjacency(state)
            # 内容物向下递归，环境向上追溯；祖先的其他内容物不进入范围。
            focal = self._closure(self.direct[index], children)
            environment = self._closure(focal, parents)
            scope = self.parameters[index] | focal | environment
            observed = self.explicit[index]
            for obj in scope:
                observed |= self.owner_masks[obj]
            for obj in focal:
                observed |= self.selector_masks[obj]
            self._scope_cache[key] = observed
        return self._scope_cache[key]

    def event(self, state, identifier):
        if type(identifier) is not int or not 1 <= identifier <= len(self.model.actions):
            raise ValueError("Expected an original bound action ID")
        key = state, identifier
        if key not in self.cache:
            index = identifier - 1
            successor = self.model.apply(state, index)
            if successor is None:
                raise ValueError("Event action is not executable")
            observed = self.observation_mask(state, index)
            add, delete = self.model.activated(state, index)
            effect = add & self.nonhand_mask, delete & ~add & self.nonhand_mask
            # mask 本身也进签名，因此「未观察」与「观察到为假」可以区分。
            signature = self.operations[index], (observed, state & observed), effect
            token = self.tokens.get(signature)
            if token is None:
                token = len(self.tokens) + 1
                self.tokens[signature] = token
                self.events[token] = signature
                self.token_channels[token] = (0,)
            self.cache[key] = successor, token
        return self.cache[key]

    def initial_profile(self):
        return (0,)

    def extend(self, profile, token):
        key = profile[0], token
        if key not in self.trace_intern:
            self.trace_intern[key] = len(self.trace_nodes)
            self.trace_nodes.append(key)
        return (self.trace_intern[key],)

    def finish(self, state, profile):
        # Constant end marker: final state is not an extra equivalence condition.
        return (), profile

    def decode_history(self, profile):
        tokens, node = [], profile[0]
        while node:
            node, token = self.trace_nodes[node]
            tokens.append(token)
        return (tuple(reversed(tokens)),)

    def profile(self, order):
        if len(order) != self.model.length:
            raise ValueError("Expected reference length")
        state, counts = self.model.initial, Counter()
        for identifier in order:
            state, token = self.event(state, identifier)
            counts[token] += 1
        if not evaluate(self.model.goal, state):
            raise ValueError("Expected an original goal-reaching path")
        return (), tuple(sorted(counts.items()))

    def catalog(self):
        return {str(token): event for token, event in self.events.items()}

    def metadata(self):
        return {"version": self.version, "hand_objects_ignored": list(self.resource_names),
                "final_state_is_equivalence_key": False,
                "event_count": len(self.events),
                "hierarchy_relations": sorted(self.hierarchy),
                "direct_cargo_rule": "strong_anchors_plus_unknown_participants_except_unanchored_known_parents",
                "unknown_relation_state_ownership": "all_nonhand_participants_conservative",
                "actual_nonhand_effect_recorded": True}


# PSV 对照分组：匹配重复动作，再比较固定焦点与背景。
def action_key(value):
    """Return a hashable concrete non-hand operation from an event or action."""
    if isinstance(value, dict):
        value = value['action']
    return value[0], tuple(value[1])


def jaccard_bits(left, right):
    union = (left | right).bit_count()
    return ((left ^ right).bit_count() / union) if union else 0.0


def event_distances(left, right):
    """Exact positive/domain and signed-effect Jaccard distances in [0,1]."""
    lm, rm = left['mask'], right['mask']
    lt, rt = left['pre'] & lm, right['pre'] & rm
    context = .5 * (jaccard_bits(lt, rt) + jaccard_bits(lm, rm))
    union = (left['add'] | right['add']).bit_count() + (left['delete'] | right['delete']).bit_count()
    intersection = (left['add'] & right['add']).bit_count() + (left['delete'] & right['delete']).bit_count()
    effect = 1 - intersection / union if union else 0.0
    return context, effect


def assignment(cost):
    """Minimum-cost square assignment; no external library or service."""
    n = len(cost)
    if not n:
        return []
    if any(len(row) != n for row in cost):
        raise ValueError('Hungarian cost must be square')
    u, v, match, way = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for row in range(1, n + 1):
        match[0] = row
        column, minimum, used = 0, [float('inf')] * (n + 1), [False] * (n + 1)
        while True:
            used[column] = True
            active = match[column]
            delta, following = float('inf'), 0
            for target in range(1, n + 1):
                if used[target]:
                    continue
                value = cost[active - 1][target - 1] - u[active] - v[target]
                if value < minimum[target]:
                    minimum[target], way[target] = value, column
                if minimum[target] < delta:
                    delta, following = minimum[target], target
            for target in range(n + 1):
                if used[target]:
                    u[match[target]] += delta
                    v[target] -= delta
                else:
                    minimum[target] -= delta
            column = following
            if match[column] == 0:
                break
        while column:
            previous = way[column]
            match[column] = match[previous]
            column = previous
            if not column:
                break
    result = [0] * n
    for target in range(1, n + 1):
        result[match[target] - 1] = target - 1
    return result


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def action_comparison(action, left, right, *, details=False):
    original_left_count, original_right_count = len(left), len(right)
    left_contexts = Counter((e['mask'], e['pre'] & e['mask']) for e in left)
    right_contexts = Counter((e['mask'], e['pre'] & e['mask']) for e in right)
    differs = bool(left and right and any(
        left_contexts[key] * len(right) != right_contexts[key] * len(left)
        for key in left_contexts.keys() | right_contexts.keys()))
    # Canonical occurrence order and orientation make tie-breaking symmetric:
    # equal combined assignment costs cannot select different context/effect
    # subtotals merely because the caller reversed the two plans.
    key = lambda event: (event['mask'], event['pre'], event['add'], event['delete'])
    left_order = sorted(range(len(left)), key=lambda i: key(left[i]))
    right_order = sorted(range(len(right)), key=lambda i: key(right[i]))
    left = [left[i] for i in left_order]
    right = [right[i] for i in right_order]
    swapped = tuple(map(key, left)) > tuple(map(key, right))
    if swapped:
        left, right = right, left
        left_order, right_order = right_order, left_order
    n = max(len(left), len(right))
    if not n:
        raise ValueError('At least one occurrence is required')
    differences = [[event_distances(a, b) for b in right] for a in left]
    cost = [[sum(differences[i][j]) / 2 if i < len(left) and j < len(right) else 1.0
             for j in range(n)] for i in range(n)]
    pairs = assignment(cost)
    context, effect, matched_context, matched_effect = 0.0, 0.0, 0.0, 0.0
    matched = 0
    rows = []
    for i, j in enumerate(pairs):
        real = i < len(left) and j < len(right)
        c, e = differences[i][j] if real else (1.0, 1.0)
        context += c
        effect += e
        if real:
            matched += 1
            matched_context += c
            matched_effect += e
        if details:
            li = left_order[i] if i < len(left) else None
            rj = right_order[j] if j < len(right) else None
            if swapped:
                li, rj = rj, li
            rows.append({'left_occurrence': li, 'right_occurrence': rj,
                         'context_distance': c, 'effect_distance': e})
    result = {'action': [action[0], list(action[1])], 'left_count': original_left_count, 'right_count': original_right_count,
              'matched_count': matched, 'missing_occurrences': abs(len(left) - len(right)),
              'left_only_occurrences': max(0, original_left_count - original_right_count),
              'right_only_occurrences': max(0, original_right_count - original_left_count),
              'context_distance': context / n, 'effect_distance': effect / n,
              'matched_context_distance': matched_context / matched if matched else 0.0,
              'matched_effect_distance': matched_effect / matched if matched else 0.0,
              'context_distribution_differs': differs,
              'context_support_differs': left_contexts.keys() != right_contexts.keys(),
              'focal_eligible': differs}
    if details:
        result['occurrence_pairs'] = rows
    return result


def pair_compare(left, right, *, details=False):
    """Compare two saved plan dictionaries; action counts may differ.

    per_action is a JSON-compatible list. Every distance is symmetric. Focal
    eligibility requires the action in both plans and an actual change in its
    normalized pre-context distribution, not merely more identical repeats.
    """
    le, re = defaultdict(list), defaultdict(list)
    for event in left['events']:
        le[action_key(event)].append(event)
    for event in right['events']:
        re[action_key(event)].append(event)
    actions = sorted(le.keys() | re.keys())
    rows = [action_comparison(action, le[action], re[action], details=details) for action in actions]
    intersection = sum(min(len(le[action]), len(re[action])) for action in actions)
    union = sum(max(len(le[action]), len(re[action])) for action in actions)
    ls = [action_key(event) for event in left['events']]
    rs = [action_key(event) for event in right['events']]
    maximum, order_edit = max(len(ls), len(rs)), edit_distance(ls, rs)
    return {'left_class_rank': left.get('class_rank'), 'right_class_rank': right.get('class_rank'),
            'per_action': rows, 'shared_concrete_actions': sum(bool(le[a] and re[a]) for a in actions),
            'missing_occurrences': sum(row['missing_occurrences'] for row in rows),
            'composition_distance': 1 - intersection / union if union else 0.0,
            'composition_similarity': intersection / union if union else 1.0,
            'order_edit_distance': order_edit,
            'order_distance': order_edit / maximum if maximum else 0.0,
            'context_distance': sum(row['context_distance'] * max(row['left_count'], row['right_count'])
                                    for row in rows) / union if union else 0.0,
            'effect_distance': sum(row['effect_distance'] * max(row['left_count'], row['right_count'])
                                   for row in rows) / union if union else 0.0,
            'focal_actions': [row['action'] for row in rows if row['focal_eligible']]}


class Distances:
    def __init__(self, plans):
        self.plans = {p['class_rank']: p for p in plans}
        self.cache = {}
        self.focus_cache = {}

    def get(self, left, right):
        if left > right:
            left, right = right, left
        key = left, right
        if key not in self.cache:
            self.cache[key] = pair_compare(self.plans[left], self.plans[right])
        return self.cache[key]

    def focused(self, left, right, focus):
        key = min(left, right), max(left, right), focus
        if key in self.focus_cache:
            return self.focus_cache[key]
        comparison = self.get(left, right)
        rows = comparison['per_action']
        target = next((row for row in rows if action_key(row['action']) == focus), None)
        shared = [row for row in rows if row['matched_count']]
        strongest = max((row['matched_context_distance'] for row in shared), default=0)
        eligible = bool(target and target['focal_eligible'] and
                        target['matched_context_distance'] + 1e-12 >= strongest)
        background = [row for row in rows if action_key(row['action']) != focus]
        weight = sum(max(row['left_count'], row['right_count']) for row in background)
        distance = sum(max(row['left_count'], row['right_count']) *
                       (row['context_distance'] + row['effect_distance']) / 2
                       for row in background) / weight if weight else 0.0
        value = {'eligible': eligible, 'background_distance': distance,
                 'composition_distance': comparison['composition_distance'],
                 'order_distance': comparison['order_distance'],
                 'focus_context_distance': target['matched_context_distance'] if target else 0.0}
        self.focus_cache[key] = value
        return value


def group_metrics(ranks, anchor, focus, distances):
    anchor_rows = [distances.focused(anchor, rank, focus) for rank in ranks if rank != anchor]
    pair_rows = [distances.focused(left, right, focus)
                 for position, left in enumerate(ranks) for right in ranks[position + 1:]]
    return {'anchor_background_mean': sum(x['background_distance'] for x in anchor_rows) / len(anchor_rows)
                if anchor_rows else 0,
            'anchor_background_max': max((x['background_distance'] for x in anchor_rows), default=0),
            'pair_background_max': max((x['background_distance'] for x in pair_rows), default=0),
            'composition_distance_max': max((x['composition_distance'] for x in pair_rows), default=0),
            'anchor_order_distance_mean': sum(x['order_distance'] for x in anchor_rows) / len(anchor_rows)
                if anchor_rows else 0}


def logical_groups(case, config, distances):
    # 先列出每个“锚点＋焦点”的候选，再按人数、背景和顺序距离择优成组。
    plans = {p['class_rank']: p for p in case['plans']}
    ranks = sorted(plans)
    background, diameter, composition = (config['background_limit'], config['pair_background_limit'],
                                         config['composition_limit'])
    candidates = []
    for anchor in ranks:
        focuses = sorted({action_key(event) for event in plans[anchor]['events']})
        for focus in focuses:
            possible = []
            for other in ranks:
                if other == anchor:
                    continue
                value = distances.focused(anchor, other, focus)
                if (value['eligible'] and value['background_distance'] <= background + 1e-12
                        and value['composition_distance'] <= composition + 1e-12):
                    possible.append((value['background_distance'], value['composition_distance'],
                                     value['order_distance'], other))
            if possible:
                candidates.append((anchor, focus, sorted(possible)))
    available, groups = set(ranks), []
    while available:
        best = None
        for anchor, focus, possible in candidates:
            if anchor not in available:
                continue
            members = [anchor]
            for _, _, _, other in possible:
                if other not in available:
                    continue
                if any(distances.focused(member, other, focus)['background_distance'] > diameter + 1e-12
                       or distances.focused(member, other, focus)['composition_distance'] > composition + 1e-12
                       for member in members):
                    continue
                members.append(other)
            if len(members) < 2:
                continue
            metrics = group_metrics(members, anchor, focus, distances)
            score = (-len(members), metrics['anchor_background_mean'],
                     metrics['anchor_order_distance_mean'], focus, anchor, tuple(members))
            if best is None or score < best[0]:
                best = score, anchor, focus, members, metrics
        if best is None:
            anchor = min(available)
            members, focus, metrics = [anchor], None, {}
        else:
            _, anchor, focus, members, metrics = best
        groups.append({'id': f'g{len(groups):04d}', 'focus_action': focus,
                       'anchor_rank': anchor, 'class_ranks': sorted(members),
                       'size': len(members), 'metrics': metrics})
        available.difference_update(members)
    return groups


def _action_library(model):
    trace = _evaluated_symbolic_trace('', model.domain_path, [action.line for action in model.actions])
    chunks = re.findall(r'^\d+\. (.*?)(?=^\d+\. |\Z)', trace, re.M | re.S)
    if len(chunks) != len(model.actions):
        raise ValueError('Bound action descriptions do not match the action library')
    return [{'action_id': i, 'description': text.rstrip('\n')} for i, text in enumerate(chunks, 1)]


def group_plan_classes(model, representatives, observer, instruction, kf_actions, *, multiview=False):
    """Reproduce the historical fixed-focus protocol for its research callers."""
    plans = []
    for candidate in representatives:
        state, events = model.initial, []
        for identifier in candidate['order']:
            state, token = observer.event(state, identifier)
            action, (mask, pre), (add, delete) = observer.events[token]
            events.append(dict(action=[action[0], list(action[1])], mask=mask, pre=pre,
                               add=add, delete=delete))
        plans.append(dict(class_rank=candidate['class_rank'], events=events))
    config = dict(background_limit=.35, pair_background_limit=.50, composition_limit=.40)
    groups = logical_groups({'plans': plans}, config, Distances(plans))
    by_rank = {candidate['class_rank']: candidate for candidate in representatives}
    if not any(candidate['id'] != 'reference' for candidate in representatives):
        return [], []
    library, batches = _action_library(model), []
    for group in groups:
        members = [by_rank[rank] for rank in group['class_ranks'] if by_rank[rank]['id'] != 'reference']
        anchor = by_rank[group['anchor_rank']]['order']
        focus = group['focus_action']
        question = (f"When {focus[0]}({', '.join(focus[1])}) occurs, do the objects' states, contents "
                    'and enclosing environment allow the operation and satisfy Instruction in each plan?') if focus else ''

        def prompt(candidates):
            return build_group_judge_prompt(library, candidates, instruction=instruction,
                kf_actions=kf_actions, comparison_question=question, multiview=multiview,
                exact_length=model.length, comparison_order=anchor)

        current = []
        for position, candidate in enumerate(members):
            current.append(candidate)
            rendered = prompt(current)
            if len(rendered) > 48000:
                raise ValueError('One complete review prompt exceeds capacity')
            if len(current) < 10 and position + 1 < len(members):
                if len(prompt(current + [members[position + 1]])) <= 48000:
                    continue
            batches.append(dict(candidates=current, prompt=rendered, comparison_question=question,
                                comparison_order=anchor, group_id=group['id']))
            current = []
    actual = [candidate['id'] for batch in batches for candidate in batch['candidates']]
    expected = [candidate['id'] for candidate in representatives if candidate['id'] != 'reference']
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError('Grouping omitted or duplicated a representative')
    return library, batches


def _write_validation_json(path, value):
    stream = NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False)
    temp = Path(stream.name)
    try:
        with stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def save_validation(round_dir, pfv, psv, *, inputs=None):
    """Keep flat verdicts in judge.json and the current protocol's counts alongside them."""
    verdict = None
    if psv is not None and psv['status'] == 'rejected':
        verdict = {k: psv['counterexample'][k] for k in ('reasoning', 'pass', 'feedback')}
    elif psv is not None and psv['status'] == 'passed':
        verdict = {'reasoning': 'Complete operation-logic class coverage; no rejected representative.',
                   'pass': True, 'feedback': ''}
    result = {'pfv': pfv, 'psv': verdict}
    directory = Path(round_dir)
    meta_path = directory / 'psv_meta.json'
    # Invalidate the verdict first on a reset; a transient sidecar cannot certify it.
    _write_validation_json(directory / 'judge.json', result)
    if psv is not None and psv.get('protocol') == PSV_PROTOCOL:
        _write_validation_json(meta_path, {
            'protocol': PSV_PROTOCOL, 'capacity': MAX_CLASSES, 'status': psv['status'],
            'inputs': psv.get('inputs'),
            'raw_plans': psv.get('legal_orders'), 'classes': psv.get('classes'),
            'pending_classes': psv.get('pending_classes'), 'natural_clusters': psv.get('natural_clusters'),
            'first_packets': psv.get('groups'), 'request_attempts': len(psv['requests']),
        })
    elif inputs is not None:
        _write_validation_json(meta_path, {
            'protocol': PSV_PROTOCOL, 'status': 'pfv_only', 'inputs': inputs,
        })
    else:
        meta_path.unlink(missing_ok=True)
    passed, feedback = None, ''
    if pfv is not None and not pfv['pass']:
        passed, feedback = False, pfv['feedback']
    elif pfv is not None and verdict is not None:
        passed = verdict['pass']
        feedback = '[PSV] ' + verdict['feedback'] if not passed else ''
    return {**result, 'pass': passed, 'feedback': feedback}


def validate_plan_soundness(model, first_img, instruction, kf_actions, round_dir, *,
                            robot_configuration=None):
    """Review every other operation-logic class after this round's PFV passed.

    Search and transport failures remain incomplete. Only completed, validated
    responses produce verdicts. Removed orders inherit no instruction verdict.
    """
    kf_actions = format_kf_actions(kf_actions)
    directory = Path(round_dir)
    record = json.loads((directory / 'judge.json').read_text())
    pfv = _validate_verdict(record['pfv'])
    if not pfv['pass']:
        raise ValueError('PSV requires a passed PFV')
    psv = {'status': 'incomplete', 'protocol': PSV_PROTOCOL, 'search_complete': False,
           'reviews': [], 'requests': [],
           'instruction_verdict_inherited': False, 'scope': 'exact_reference_length_original_bindings'}
    def save():
        save_validation(directory, pfv, psv)

    save()
    try:
        images = list(first_img) if isinstance(first_img, (list, tuple)) else [first_img]
        psv['inputs'] = review_inputs(directory, images, instruction, kf_actions, model, robot_configuration)
        symbolic = SameLengthModel(*(directory / n for n in ('domain.pddl', 'problem.pddl', 'plan.txt')))
        observer = AvailabilityObserver(LogicObserver(symbolic))
        representatives, proof = search_plan_classes(symbolic, observer)
        psv.update(search_complete=proof['coverage_complete'], search_status=proof['status'],
                   legal_orders=proof['raw_orders_including_reference'],
                   legal_order_count_exact=proof['raw_order_count_exact'], classes=len(representatives),
                   covered_orders=proof['accounted_orders'], class_counts=proof['classes'],
                   factorization=proof['factorization'])
        if not proof['coverage_complete']:
            raise ValueError('Cannot group an incomplete class inventory')
        references = [c for c in representatives if c['id'] == 'reference']
        if len(references) != 1 or list(references[0]['order']) != list(symbolic.reference_order):
            raise ValueError('The PFV-covered class must represent this exact reference plan')
        pending = [c for c in representatives if c['id'] != 'reference']
        by_rank = {c['class_rank']: c for c in pending}
        if len(by_rank) != len(pending) or len({c['id'] for c in pending}) != len(pending):
            raise ValueError('Repeated pending rank or candidate ID')
        clusters, packets = group_clusters(symbolic, pending, capacity=MAX_CLASSES)
        if (sorted(rank for packet in packets for rank in packet) != sorted(by_rank)
                or any(not 1 <= len(packet) <= MAX_CLASSES for packet in packets)):
            raise ValueError('PSV packets must cover every pending class exactly once within capacity')
        psv.update(pending_classes=len(pending), natural_clusters=len(clusters), groups=len(packets),
                   reference_review_source='pfv')
        found = {}
        library = _action_library(symbolic) if packets else []

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

        def review(candidates):
            prompt = build_group_judge_prompt(
                library, candidates, instruction=instruction, kf_actions=kf_actions, tree=True,
                exact_length=symbolic.length, multiview=len(images) > 1,
                robot_configuration=robot_configuration,
            )
            capture = {}
            request = {'candidates': candidates, 'status': 'in_progress'}
            psv['requests'].append(request)
            save()
            try:
                response = call_gpt_json(model, prompt, images, response_format={'type': 'json_object'},
                                         attempts=1, reasoning_effort='xhigh',
                                         temperature=0, capture=capture)
                request['response'] = response
                if not isinstance(response, dict) or set(response) != {'results'} or not isinstance(response['results'], list):
                    raise ValueError('Group judge must return a results list')
                allowed = {c['id']: c for c in candidates}
                counts = Counter(row['candidate_id'] for row in response['results']
                                 if isinstance(row, dict) and isinstance(row.get('candidate_id'), str))
                accepted = 0
                for row in response['results']:
                    if not isinstance(row, dict) or set(row) != {'candidate_id', 'reasoning', 'pass', 'feedback'}:
                        continue
                    identifier = row['candidate_id']
                    if not isinstance(identifier, str) or identifier not in allowed or counts[identifier] != 1:
                        continue
                    try:
                        verdict = _validate_verdict({k: row[k] for k in ('reasoning', 'pass', 'feedback')})
                    except ValueError:
                        continue
                    found[identifier] = {'candidate_id': identifier, 'order': allowed[identifier]['order'],
                                         **verdict, 'response_candidate_id': identifier,
                                         'request_index': len(psv['requests']) - 1}
                    accepted += 1
                if accepted == len(allowed) == len(response['results']):
                    request['status'] = 'complete'
                else:
                    request['status'] = 'partial' if accepted else 'invalid_response'
            except Exception as error:
                request.update(status='request_error', error_type=type(error).__name__, error=str(error))
            request['capture'] = {k: v for k, v in capture.items() if k not in {'raw_output', 'json_text'}}
            checkpoint()

        for ranks in packets:
            candidates = [by_rank[rank] for rank in sorted(ranks)]
            review(candidates)
            # Invalid/missing rows get one independent retry. Valid rejections
            # and valid passes are never retried to seek a different answer.
            for candidate in candidates:
                if candidate['id'] not in found:
                    review([candidate])
    except (OSError, ValueError, KeyError, NotImplementedError, OverflowError, MemoryError) as error:
        psv['status'] = 'rejected' if 'counterexample' in psv else 'incomplete'
        psv.update(error_type=type(error).__name__, error=str(error))
        save()
    return psv
