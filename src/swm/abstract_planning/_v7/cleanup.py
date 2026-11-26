"""Goal-aware PDDL projection and concise rule text for the isolated v4 build.

Native physical transitions always receive their original complete state.
Only comparison/export states lose predicates unused by actions and the goal.
Action logic only loses redundant conjunction wrappers and duplicate clauses; task goals and physical
transition semantics remain unchanged.
"""
from __future__ import annotations
import copy
import re
from collections import defaultdict

from .model import check_operation_list, fields, operation_list, operation_names, parse_sexpr, section, sexpr

LOGICAL = {'and', 'or', 'not', 'forall', 'exists', 'when', 'imply', '='}
SEMANTIC_ONLY = {}


def atoms(node):
    if not isinstance(node, list) or not node:
        return
    if node[0] in {'forall', 'exists'}:
        yield from atoms(node[2])
    elif node[0] in LOGICAL:
        if node[0] == '=':
            yield node
        else:
            for child in node[1:]:
                yield from atoms(child)
    else:
        yield node


def positive_bindings(node):
    """Variables whose absent-object binding makes this conjunction false."""
    if not isinstance(node, list) or not node:
        return set()
    if node[0] == 'and':
        return set().union(*(positive_bindings(child) for child in node[1:]))
    if node[0] not in LOGICAL:
        return {value for value in node[1:] if isinstance(value, str) and value.startswith('?')}
    return set()


def unguarded_quantifiers(node, effect=False):
    """Conservatively retain the universe if unused objects could affect it."""
    if not isinstance(node, list) or not node:
        return []
    found = []
    head = node[0]
    if head in {'forall', 'exists'}:
        body = node[2]
        if head == 'exists' and not effect:
            guard = body
        elif head == 'forall' and body[:1] in [['when'], ['imply']]:
            guard = body[1]
        elif head == 'forall' and not effect and body[:1] == ['not']:
            guard = body[1]
        else:
            guard = []
        if not set(node[1]) <= positive_bindings(guard):
            found.append(sexpr(node))
        found.extend(unguarded_quantifiers(body, effect))
        return found
    if head == 'when':
        return unguarded_quantifiers(node[1]) + unguarded_quantifiers(node[2], True)
    for child in node[1:]:
        found.extend(unguarded_quantifiers(child, effect))
    return found


def controller_rules(task):
    """State the hidden controller values and only active dependencies.

    The scene contains game state only. Controller state therefore needs a
    short textual initial value, unlike positions and counters visible there.
    """
    cfg = task['cfg']
    prep, energy, cooling = cfg['preparation'], cfg['energy'], cfg['cooling']
    if prep == 'none' and not energy and not cooling:
        return ''
    from .rules_v2 import WORK, RESET_TEXT
    names = set(operation_names(task['domain']))
    work = [name for name in dict.fromkeys(WORK[task['family']]) if name in names]
    initial = []
    badge = task['badge']
    if prep != 'none':
        initial.append('enabled' if badge['ready'] else 'disabled')
    if energy:
        initial.append(f"energy {badge['charge']}/3")
    if cooling:
        initial.append('cooling busy' if badge['busy'] else 'cooling idle')
    text = ['Controller initially: ' + '; '.join(initial) + '.',
            'Guarded operations: ' + ', '.join(work) + '.']
    if prep != 'none':
        text += ['Guarded operations require the controller enabled. prepare_controller enables a disabled controller'
                 + (' while cooling is idle.' if cooling else '.')]
        if prep == 'per_work':
            text.append('Each guarded operation disables the controller.')
        elif prep == 'after_motion':
            if task['family'] in {'gripper', 'briefcase'}:
                text.append('Every move command disables the controller, including one naming the current room; guarded work preserves it.')
            else:
                text.append(RESET_TEXT.get(task['family'],
                    'Position-changing travel disables ARM; turns in place and guarded work preserve it.').replace('ARM', 'the controller'))
        else:
            text.append('Guarded operations preserve enabling.')
    if energy:
        text += ['Energy is an integer from zero to three; qN denotes N units.',
                 'Every guarded operation requires and consumes one unit. recharge_controller restores energy to three only when below three'
                 + (' and cooling is idle.' if cooling else '.')]
    if cooling:
        text.append('Guarded operations require idle cooling and make it busy. wait_for_controller requires busy cooling and restores idle, without requiring energy or enabling.')
    text.append('Control operations preserve game objects and all other control states.')
    return ' '.join(text)


def concise_instruction(task):
    text = task['instruction']
    # Policy sentences below restate constraints already supplied by the base game.
    text = text.replace(' Painted tiles cannot be repainted.', '')
    text = text.replace(' Paint still cannot be overwritten. Only unpainted unoccupied tiles may be painted, above or below.', ' Paint immediately above or below the robot.')
    text = text.replace(" Placement raises an adjacent tile at the robot's current height; removal lowers a tile one level higher.", '')
    text = text.replace(' All contained items still move together.', '')
    text = text.replace('Latch actions preserve contents and positions; contained items still move together.', 'Latch actions preserve contents and positions.')
    text = text.replace(' The final colour assignments remain required.', '')
    text = text.replace(' Final colour assignments remain required.', '')
    text = re.sub(r'(?m)^Allowed operations: .*\n?', '', text)
    text = re.sub(r'(?m)^Guarded operations: .*\n?', '', text)
    text = re.sub(r'(?m)^All action effects use the pre-action state, then update together\..*\n?', '', text)
    text = re.sub(r'(?m)^All effects read the pre-action state and update together; other facts persist\.\n?', '', text)
    # Card ranks and the number of freecell/tableau slots can be counted in
    # the image. Keep their naming convention, but do not enumerate the init.
    text = re.sub(r'(?m)^(?:Freecell counts|Empty-column counts|Ranks): .*\n?', '', text)
    if task['family'] == 'floortile':
        text = text.replace('Paint colours are the swatches in the scene; a robot badge shows its current colour.\n', '')
    if task['family'] == 'termes':
        # This is the legal height range, not a list of initial tile heights.
        text = text.replace('Height levels:', 'Allowed height levels:')
    if task['family'] == 'color_sort':
        text = re.sub(r'Each bolt holds at most \d+ nuts\.',
                      'Each bolt holds at most one nut per depicted slot.', text)
        text = re.sub(r"Slot bNs0 is bolt bN's empty base; bNs1 through bNs\d+ are its bottom-to-top slots\.",
                      "Slot bNs0 is bolt bN's empty base; bNsJ names its Jth depicted nut slot, counted bottom-to-top.", text)
    if task['family'] == 'reversi_solitaire':
        text = text.replace('Remove all white disks;', 'Turn all white disks black;')
        text = text.replace('Turn all white disks black; empty squares need not be filled.',
                            'Empty squares need not be filled.')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Base-game rules and the selected policy become one coherent rules block.
    rule_lines = [(i, line[len('Rules:'):].strip()) for i, line in enumerate(lines)
                  if line.startswith('Rules:')]
    if rule_lines:
        insert = rule_lines[0][0]
        rules = ' '.join(value for _, value in rule_lines if value)
        # An exact repeated goal sentence carries no additional rule meaning.
        if lines and not lines[0].startswith('Rules:') and lines[0] in rules:
            rules = rules.replace(lines[0], '', 1).strip()
        lines = [line for line in lines if not line.startswith('Rules:')]
        lines.insert(insert, 'Rules: ' + rules)
    control = controller_rules(task)
    if control:
        lines.append(control)
    if task['family'] == 'printer':
        lines.append('The paper-stack counter is feeder sheets; numbers on printers are their loaded sheets.')
    if task['family'] == 'floortile':
        lines.append('The bottom swatches show available paint colors; the dot beside each robot shows its current paint color.')
    if task['family'] == 'briefcase' and task['cfg']['policy'] in {'latched', 'two_slot_latch'}:
        lines.append('Open and closed padlock icons show the case latch state.')
    if task['family'] == 'rush_hour' and task['cfg']['policy'] in {'parking_brakes', 'glide_brakes'}:
        lines.append('A black triangular wheel chock marks an applied parking brake.')
    lines.append('Effects use the pre-action state and update together; other facts persist.')
    lines.append(operation_list(task['domain']).strip())
    result = '\n'.join(lines) + '\n'
    check_operation_list(result, task['domain'])
    assert result.count('Rules:') <= 1
    assert not re.search(r'\bARM\b|\bDRIVE\b|\bCOOL\b|bottom badge|shared task controller', result)
    return result


def clean_task(task):
    """Keep all action/goal dependencies and expose an explicit native projection."""
    if 'model_projection' in task:
        return task
    domain, problem = parse_sexpr(task['domain']), parse_sexpr(task['problem'])
    static_guard_proof = None
    if task['family'] == 'gripper':
        static_guard_proof = remove_implied_unary_guard(domain, problem, 'object')
    complement_proof = None
    if task['family'] == 'miconic':
        complement_proof = remove_miconic_complements(domain, problem)
    actions = [node for node in domain if isinstance(node, list) and node[:1] == [':action']]
    original_actions = copy.deepcopy(actions)
    original_goal = copy.deepcopy(section(problem, ':goal')[1])
    names = [action[1] for action in actions]
    roots = [fields(action)[key] for action in actions for key in (':precondition', ':effect')]
    roots.append(section(problem, ':goal')[1])
    required = {atom[0] for root in roots for atom in atoms(root) if atom[0] != '='}
    semantic = dict(SEMANTIC_ONLY.get(task['family'], {}))
    semantic.update({name: 'Explicit generator semantic requirement.'
                     for name in task.get('semantic_predicates', ())})
    required.update(semantic)
    declarations = section(domain, ':predicates')
    declared = {node[0] for node in declarations[1:]}
    removed = declared - required
    initial = section(problem, ':init')
    native_initial = set(task['initial'])
    removed_facts = [node for node in initial[1:] if node[0] in removed]
    initial[:] = [':init', *(node for node in initial[1:] if node[0] not in removed)]
    declarations[:] = [':predicates', *(node for node in declarations[1:] if node[0] not in removed)]
    referenced = {value for root in roots + initial[1:] for atom in atoms(root)
                  for value in atom[1:] if isinstance(value, str) and not value.startswith('?')}
    blockers = []
    for action in actions:
        model = fields(action)
        blockers += unguarded_quantifiers(model[':precondition'])
        blockers += unguarded_quantifiers(model[':effect'], True)
    blockers += unguarded_quantifiers(section(problem, ':goal')[1])
    objects = section(problem, ':objects', [':objects'])
    constants = section(domain, ':constants', [':constants'])
    old_objects, old_constants = list(objects[1:]), list(constants[1:])
    if not blockers:
        objects[:] = [':objects', *(value for value in old_objects if value in referenced)]
        constants[:] = [':constants', *(value for value in old_constants if value in referenced)]
        if len(constants) == 1 and constants in domain:
            domain.remove(constants)
    task['native_initial'] = native_initial
    task['model_projection'] = dict(
        removed_predicates=sorted(removed), retained_predicates=sorted(declared-removed),
        removed_initial_facts=copy.deepcopy(removed_facts),
        removed_objects=sorted(set(old_objects)-set(objects[1:])),
        removed_constants=sorted(set(old_constants)-set(constants[1:])),
        universe_retained_for_unguarded_quantifiers=blockers,
        semantic_predicates=semantic,
        action_names_unchanged=names,
        goal_unchanged=True,
    )
    if static_guard_proof:
        task['model_projection']['static_guard_elimination'] = static_guard_proof
    if complement_proof:
        task['model_projection']['complement_elimination'] = complement_proof
    domain, logic_stats = normalize_action_logic(domain)
    normalized_original, _ = normalize_action_logic(['define', *original_actions])
    original_actions = normalized_original[1:]
    task['model_projection']['logic_normalization'] = logic_stats
    task['domain'] = format_domain(domain, problem, names) + '\n'
    # The shared formatter infers features from operators. A negative goal also
    # needs the negative-preconditions declaration, even with positive-only
    # operator conditions (for example the plain Reversi puzzle).
    def negative_goal(node):
        return isinstance(node, list) and bool(node) and (
            node[0] == 'not' or any(negative_goal(child) for child in node[1:]))
    published_requirements = section(parse_sexpr(task['domain']), ':requirements')
    goal_requirements_added = []
    if negative_goal(original_goal) and ':negative-preconditions' not in published_requirements:
        published_requirements.append(':negative-preconditions')
        goal_requirements_added.append(':negative-preconditions')
        task['domain'] = re.sub(
            r'(?m)^  \(:requirements[^\n]*\)$',
            '  (:requirements ' + ' '.join(sorted(published_requirements[1:])) + ')',
            task['domain'], count=1)
    if goal_requirements_added:
        task['model_projection']['goal_requirements_added'] = goal_requirements_added
    task['problem'] = format_problem(problem) + '\n'
    published_domain = parse_sexpr(task['domain'])
    assert [node for node in published_domain
            if isinstance(node, list) and node[:1] == [':action']] == original_actions
    assert section(parse_sexpr(task['problem']), ':goal')[1] == original_goal
    task['initial'] = project_native_state(task, native_initial)
    task['instruction'] = concise_instruction(task)
    assert operation_names(task['domain']) == names
    assert task['initial'] == {tuple(atom) for atom in section(problem, ':init')[1:]}
    return task


def project_native_state(task, state):
    """Only discard certified absent model vocabulary; never infer game state."""
    removed = set(task.get('model_projection', {}).get('removed_predicates', ()))
    return {fact for fact in state if fact[0] not in removed}


def project_native_successors(task, native_state, controller=True):
    """Input is the FULL independent native state, not a projected model state.

    Keep the unprojected successors separately for replay and native invariants.
    """
    from .rules_v2_native import successors
    native_task = dict(task, initial=task.get('native_initial', task['initial']))
    return {operation: project_native_state(task, next_state)
            for operation, next_state in successors(native_task, native_state, controller).items()}


def remove_implied_unary_guard(domain, problem, predicate):
    """Remove all uses only after proving each positive conjunct redundant.

    Both predicates must be static. For each occurrence P(x), a distinct
    same-variable conjunct Q(x) must have its complete initial extension
    contained in P's extension. Any non-conjunctive, effect or goal use makes
    the transformation inapplicable. Vocabulary pruning stays with the caller.
    """
    actions = [a for a in domain if isinstance(a, list) and a[:1] == [':action']]
    if not any(d[0] == predicate for d in section(domain, ':predicates')[1:]):
        return None
    declaration = next(d for d in section(domain, ':predicates')[1:] if d[0] == predicate)
    if len(declaration) != 2:
        return None
    dynamic = {a[0] for action in actions for a in atoms(fields(action)[':effect'])}
    if predicate in dynamic or any(a[0] == predicate for a in atoms(section(problem, ':goal')[1])):
        return None
    extensions = {}
    for fact in section(problem, ':init')[1:]:
        if len(fact) == 2:
            extensions.setdefault(fact[0], set()).add(fact[1])
    target = extensions.get(predicate, set())
    edits, witnesses = [], []
    def conjuncts(node):
        if node[:1] == ['and']:
            return [clause for child in node[1:] for clause in conjuncts(child)]
        return [node]
    for action in actions:
        precondition = fields(action)[':precondition']
        clauses = conjuncts(precondition)
        references = [a for a in atoms(precondition) if a[0] == predicate]
        direct = [a for a in clauses if a[:1] == [predicate]]
        if len(references) != len(direct):
            return None
        for guard in direct:
            stronger = next((a for a in clauses if len(a) == 2 and a[1] == guard[1]
                             and a[0] not in LOGICAL | dynamic | {predicate}
                             and extensions.get(a[0], set()) <= target), None)
            if stronger is None:
                return None
            witnesses.append(dict(action=action[1], argument=guard[1], implied_by=stronger[0],
                                  narrower_values=sorted(extensions.get(stronger[0], set())),
                                  broad_values=sorted(target)))
        if direct:
            edits.append((action, [c for c in clauses if c not in direct]))
    if not witnesses:
        return None
    for action, clauses in edits:
        action[action.index(':precondition') + 1] = ['and', *clauses]
    return dict(predicate=predicate, removed_guards=len(witnesses),
                proof='Static unary extension inclusion with a same-argument positive conjunct for every use.',
                witnesses=witnesses, goal_and_effects_unchanged=True)


def remove_miconic_complements(domain,problem):
    """Prove the two unary complements are exact, then use negative literals.

    The old full native state remains unchanged. Each original initial passenger
    has exactly one side of each complement, every carrier is the single lift,
    and every unconditional write has its simultaneous opposite-side write.
    All other effects preserve both sides. This induction covers all reachable
    states, including arbitrary controller operations and non-plan branches.
    """
    complements={'not_boarded':('in','lift'),'not_served':('served',None)}
    initial={tuple(a) for a in section(problem,':init')[1:]}
    passengers={a[1] for a in initial if a[0]=='passenger'}
    assert passengers
    assert all(a[1] in passengers for a in initial if a[0] in {*complements,'in','served'})
    assert all(a[2]=='lift' for a in initial if a[0]=='in')
    for passenger in passengers:
        for negative,(positive,carrier) in complements.items():
            atom=(positive,passenger,carrier) if carrier else (positive,passenger)
            assert ((negative,passenger) in initial)!=(atom in initial)

    def literals(node):
        if node[0]=='and':
            return [v for child in node[1:] for v in literals(child)]
        if node[0] in {'when','forall'}:
            # The original physical writes are unconditional. Controller
            # conditionals cannot mention either side of these invariants.
            assert not any(mentions(child) for child in node[1:])
            return []
        return [node]

    def mentions(node):
        if not isinstance(node,list) or not node:return False
        return (isinstance(node[0],str) and node[0] in {*complements,'in','served'}) or any(
            mentions(child) for child in node[1:])

    def key(literal):
        return (False,literal[1]) if literal[0]=='not' else (True,literal)

    actions=[a for a in domain if isinstance(a,list) and a[:1]==[':action']]
    writes=0
    for action in actions:
        model=fields(action)
        precondition=model[':precondition']
        conjuncts=precondition[1:] if precondition[0]=='and' else [precondition]
        passenger_guards={tuple(c) for c in conjuncts if c[0]=='passenger'}
        effects=[key(v) for v in literals(model[':effect'])]
        for sign,atom in effects:
            if atom[0] in {*complements,'in','served'}:
                assert ('passenger',atom[1]) in passenger_guards,(action[1],atom)
            for negative,(positive,carrier) in complements.items():
                if atom[0]==negative:
                    opposite=[positive,atom[1],carrier] if carrier else [positive,atom[1]]
                    assert (not sign,opposite) in effects,(action[1],atom)
                    writes+=1
                elif atom[0]==positive:
                    assert (carrier is None or atom[2]==carrier)
                    assert (not sign,[negative,atom[1]]) in effects,(action[1],atom)

    def replace_condition(node):
        if node[0] in complements:
            positive,carrier=complements[node[0]]
            return ['not',[positive,node[1],carrier] if carrier else [positive,node[1]]]
        if node[0]=='not':
            child=replace_condition(node[1])
            return child[1] if child[:1]==['not'] else ['not',child]
        if node[0] in {'and','or','imply'}:
            return [node[0],*(replace_condition(c) for c in node[1:])]
        if node[0] in {'forall','exists'}:
            return [node[0],node[1],replace_condition(node[2])]
        return node

    def remove_effect(node):
        if node[0] in complements or (node[0]=='not' and node[1][0] in complements):return None
        if node[0]=='and':
            return ['and',*(v for c in node[1:] if (v:=remove_effect(c)) is not None)]
        return node

    def has_complement(node):
        return isinstance(node,list) and bool(node) and (
            (isinstance(node[0],str) and node[0] in complements)
            or any(has_complement(child) for child in node[1:]))
    assert not has_complement(section(problem,':goal')[1])
    for action in actions:
        f=fields(action)
        action[action.index(':precondition')+1]=replace_condition(f[':precondition'])
        action[action.index(':effect')+1]=remove_effect(f[':effect'])
    return dict(removed_complements=sorted(complements),passengers=len(passengers),
                checked_complement_writes=writes,initial_exclusive_complements=True,
                all_writes_simultaneously_update_both_sides=True,
                single_carrier='lift',all_other_operators_preserve_both_sides=True,
                native_state_and_oracle_unchanged=True)


def normalize_expression(node, counts=None):
    """Flatten and deduplicate conjunctions, retaining scope and order."""
    if counts is None:
        counts = {}
    if not isinstance(node, list) or not node:
        return copy.deepcopy(node)
    head = node[0]
    if head == 'and':
        children = []
        seen = set()
        for child in node[1:]:
            normalized = normalize_expression(child, counts)
            if isinstance(normalized, list) and normalized[:1] == ['and']:
                candidates = normalized[1:]
                counts['nested_conjunctions_removed'] = counts.get('nested_conjunctions_removed', 0) + 1
            else:
                candidates = [normalized]
            for candidate in candidates:
                # Repeating an identical literal in a conjunction has no
                # causal effect. Keep the first occurrence so generated
                # domains expose each requirement exactly once.
                key = repr(candidate)
                if key in seen:
                    counts['duplicate_conjuncts_removed'] = counts.get('duplicate_conjuncts_removed', 0) + 1
                    continue
                seen.add(key)
                children.append(candidate)
        if len(children) == 1:
            counts['singleton_conjunctions_removed'] = counts.get('singleton_conjunctions_removed', 0) + 1
            return children[0]
        return ['and', *children]
    if head in {'forall', 'exists'}:
        # Never move a clause across its quantifier or alter variable bindings.
        return [head, copy.deepcopy(node[1]), normalize_expression(node[2], counts)]
    if head in {'or', 'not', 'when', 'imply'}:
        return [head, *(normalize_expression(child, counts) for child in node[1:])]
    return copy.deepcopy(node)

def normalize_action_logic(domain):
    """Return a copied domain AST and exact counts of removed wrappers.

    Argument and action order are preserved. This pass applies associative,
    singleton and idempotent identities of conjunction within the same scope.
    """
    result = copy.deepcopy(domain)
    counts = dict(actions_changed=0, expressions_changed=0,
                  nested_conjunctions_removed=0, singleton_conjunctions_removed=0)
    for action in result:
        if not isinstance(action, list) or action[:1] != [':action']:
            continue
        changed = False
        for index in range(2, len(action), 2):
            if action[index] not in {':precondition', ':effect'}:
                continue
            previous = action[index + 1]
            action[index + 1] = normalize_expression(previous, counts)
            if action[index + 1] != previous:
                changed = True
                counts['expressions_changed'] += 1
        counts['actions_changed'] += int(changed)
    return result, counts


FUNCTIONAL={'location','position','direction','count','numb','peg','disk','ball','room','floor','passenger','hand','block','portable','ingredient','plate','dish','tool','phase','printer','table','cartridge','document','package','key','door','checkpoint','robot','food','chef','counter','vehicle','nut','bolt','slot','charge_value','stock_value','seat_count','case_count','counter_count','square','jug','amount_value','token','pancake','depth','cargo','bank','queen','color','hole','card','suit','num','cellnum','colnum'}

def mutable_predicates(actions):
 out=set()
 def effect(n):
  if n[0]=='and':
   for x in n[1:]:effect(x)
  elif n[0] in {'forall','when'}:effect(n[2])
  elif n[0]=='not':out.add(n[1][0])
  else:out.add(n[0])
 for a in actions:effect(fields(a)[':effect'])
 return out

def guard_parameters(actions,initial,family):
 """Infer only guards logically implied by an existing static relation.

 Adding such a category guard cannot remove any action reachable under the
 original static facts. A small explicit role map handles carried items.
 """
 mutable=mutable_predicates(actions);facts=defaultdict(list)
 for p,*args in initial:facts[p].append(args)
 categories={p:{a[0] for a in rows} for p,rows in facts.items() if p in FUNCTIONAL and p not in mutable and rows and len(rows[0])==1}
 for a in actions:
  f=fields(a);node=f[':precondition'];clauses=list(node[1:]) if node[:1]==['and'] else [node]
  guards=[]
  for var in f[':parameters']:
   if any(len(c)==2 and c[0] in categories and c[1]==var for c in clauses):continue
   domains=[]
   for c in clauses:
    if c[0] in mutable or c[0] not in facts:continue
    for i,arg in enumerate(c[1:]):
     if arg==var:domains.append({row[i] for row in facts[c[0]]})
   if domains:
    possible=set.intersection(*domains)
    candidates=[p for p,values in categories.items() if possible and possible<=values]
    if candidates:guards.append([min(candidates,key=lambda p:(len(categories[p]),p)),var]);continue
   if family=='briefcase' and var=='?x':guards.append(['portable',var])
  # Functional guards first, then unchanged constraints. No inequalities added.
  unary=[c for c in clauses if len(c)==2 and c[0] in categories]
  remaining=[c for c in clauses if c not in unary]
  a[a.index(':precondition')+1]=['and',*unary,*guards,*remaining]

def requirements(domain):
 required={':strips'}
 def walk(n,effect=False):
  if not isinstance(n,list) or not n:return
  head=n[0]
  if head=='=':required.add(':equality')
  if head=='not' and not effect:required.add(':negative-preconditions')
  if head in {'or','imply'}:required.add(':disjunctive-preconditions')
  if head=='forall':required.add(':universal-preconditions')
  if head=='exists':required.add(':existential-preconditions')
  if head=='when':
   required.add(':conditional-effects');walk(n[1]);walk(n[2],True);return
  if head=='forall' and effect:required.add(':conditional-effects')
  if head in {'forall','exists'}:walk(n[2],effect);return
  for child in n[1:]:
   if isinstance(child,list):walk(child,effect)
 for a in domain:
  if isinstance(a,list) and a[:1]==[':action']:
   f=fields(a);walk(f[':precondition']);walk(f[':effect'],True)
 return [':requirements',*sorted(required)]


def expression_head(expression):
    if expression and isinstance(expression[0], str):
        return expression[0].lower()
    return ""

def predicate_name(expression):
    if expression_head(expression) == "not":
        return expression_head(expression[1])
    return expression_head(expression)

def format_domain(domain, problem, action_names):
    """裁剪动作和谓词，保留原有 domain 名称。"""
    action_heads = {":action", ":durative-action"}
    domain = list(domain)
    domain = [requirements(domain) if isinstance(s,list) and s[:1]==[':requirements'] else s for s in domain]
    declarations = [
        predicate
        for section in domain[2:]
        if expression_head(section) == ":predicates"
        for predicate in section[1:]
    ]
    declared = {predicate_name(predicate) for predicate in declarations}
    selected_names = set(action_names)
    referenced = set()

    def collect(expression):
        if not isinstance(expression, list):
            return
        name = expression_head(expression)
        if name in declared:
            referenced.add(name)
        for child in expression:
            collect(child)

    for section in domain[2:]:
        name = expression_head(section)
        if name != ":predicates" and (
            name not in action_heads or section[1].lower() in selected_names
        ):
            collect(section)
    collect(problem)

    declarations = [
        declaration
        for declaration in declarations
        if predicate_name(declaration) in referenced
    ]
    actions = {
        section[1].lower(): section
        for section in domain[2:]
        if expression_head(section) in action_heads
    }
    selected_actions = [actions[name] for name in action_names]

    lines = ["(define " + sexpr(domain[1])]
    for section in domain[2:]:
        name = expression_head(section)
        if name in action_heads:
            continue
        if name == ":requirements":
            lines.append("  (:requirements " + " ".join(section[1:]) + ")")
        elif name == ":predicates":
            lines.append("  (:predicates")
            lines.extend("    " + sexpr(item) for item in declarations)
            lines.append("  )")
        else:
            lines.append("  " + sexpr(section))

    for action in selected_actions:
        if len(lines) > 1:
            lines.append("")
        if len(action[2:]) % 2:
            raise ValueError(f"动作 {action[1]} 的字段不完整")
        lines.append(f"  ({action[0]} {action[1]}")
        for key, value in zip(action[2::2], action[3::2]):
            if key in {':precondition',':effect'} and value[:1]==['and']:
                lines.append(f'    {key} (and')
                lines.extend('      '+sexpr(clause) for clause in value[1:])
                lines.append('    )')
            else:lines.append(f"    {key} {sexpr(value)}")
        lines.append("  )")
    lines.append(")")
    return "\n".join(lines)

def reorder_stacks(atoms, goal=False):
    """将相连的 (on A B) 按照从上到下的顺序排列。"""
    on_atoms = [
        (index, atom)
        for index, atom in enumerate(atoms)
        if len(atom) == 3 and expression_head(atom) == "on"
    ]
    if len(on_atoms) < 2:
        return atoms

    successors = {index: set() for index, _ in on_atoms}
    indegree = {index: 0 for index, _ in on_atoms}
    for upper_index, upper in on_atoms:
        for lower_index, lower in on_atoms:
            if upper_index != lower_index and upper[2] == lower[1]:
                successors[upper_index].add(lower_index)
                indegree[lower_index] += 1

    if not any(successors.values()):
        return atoms

    ready = sorted(index for index, degree in indegree.items() if degree == 0)
    ordered_indexes = []
    while ready:
        index = ready.pop(0)
        ordered_indexes.append(index)
        for successor in sorted(successors[index]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        ready.sort()

    if len(ordered_indexes) != len(on_atoms):
        return atoms

    atoms_by_index = dict(on_atoms)
    ordered_on = [atoms_by_index[index] for index in ordered_indexes]

    if goal:
        non_on = [
            atom
            for atom in atoms
            if expression_head(atom) != "on" or len(atom) != 3
        ]
        return ordered_on + non_on

    result = list(atoms)
    on_positions = [
        index
        for index, atom in enumerate(atoms)
        if len(atom) == 3 and expression_head(atom) == "on"
    ]
    for index, atom in zip(on_positions, ordered_on):
        result[index] = atom
    return result

def format_problem(problem):
    objects = None
    init = None
    goal = None
    domain_section = None
    extra_sections = []
    for section in problem[2:]:
        name = expression_head(section)
        if name == ":domain" and domain_section is None:
            domain_section = section
        elif name == ":objects" and objects is None:
            objects = section
        elif name == ":init":
            init = section[1:]
        elif name == ":goal":
            if len(section) != 2:
                raise ValueError(":goal 格式错误")
            goal = section[1]
        elif name not in {":domain", ":objects"}:
            extra_sections.append(section)

    if domain_section is None:
        raise ValueError("problem 缺少 :domain")
    lines = ["(define " + sexpr(problem[1]), "  " + sexpr(domain_section)]
    if objects is not None:
        lines.append("  " + sexpr(objects))
    if init is not None:
        init = reorder_stacks(init)
        lines.append("  (:init")
        lines.extend("    " + sexpr(atom) for atom in init)
        lines.append("  )")
    if goal is not None:
        atoms = goal[1:] if expression_head(goal) == "and" else [goal]
        atoms = reorder_stacks(atoms, goal=True)
        lines.extend(["  (:goal", "    (and"])
        lines.extend("      " + sexpr(atom) for atom in atoms)
        lines.extend(["    )", "  )"])
    lines.extend("  " + sexpr(section) for section in extra_sections)
    lines.append(")")
    return "\n".join(lines)
