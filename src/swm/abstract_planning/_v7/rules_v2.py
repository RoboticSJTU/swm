"""Explicit offline rule configurations; original generators keep their defaults.

The procedural transition code in rules_v2_native is independent of the PDDL
edits here. No model, network, benchmark/test input, or publication API is used.
"""
from __future__ import annotations

import copy
import random
import re
from functools import lru_cache

from . import games, grid_games, causal_games, visual_games as vg, extra_games
from .model import action, fields, instruction, make_domain, make_problem, operation_list, parse_sexpr, section, sexpr

FAMILIES = tuple(games.GAMES) + ("cooperative_kitchen", "rush_hour", "color_sort") + extra_games.FAMILIES
ALIASES = {"blocks": ("blocks", "blocks_3d"), "sliding_puzzle": ("sliding_puzzle", "numeric_sliding")}

# Three genuinely different policies per family. IDs never enter model prompts.
POLICIES = {
    "blocks": ("ordinary", "ordered_support", "two_high"),
    "hanoi": ("ordinary", "adjacent_pegs", "clockwise_pegs"),
    "sliding_puzzle": ("ordinary", "diagonal", "pair_slide"),
    "frozenlake": ("ordinary", "reinforced", "arrival_records"),
    "maze": ("ordinary", "reusable_key", "side_unlock"),
    "sokoban": ("ordinary", "pulling", "double_push"),
    "package": ("ordinary", "tape_first", "clip_first"),
    "printer": ("ordinary", "print_resets", "power_retains"),
    "kitchen": ("ordinary", "self_rinsing", "rinsing_sink"),
    "gripper": ("ordinary", "shared_load_limit", "assigned_grippers"),
    "miconic": ("ordinary", "two_seats", "adjacent_stops"),
    "lights_out": ("ordinary", "horizontal", "diagonal"),
    "floortile": ("ordinary", "walk_on_dry_paint", "four_way_brush"),
    "key_delivery": ("ordinary", "key_laid_down", "key_retained"),
    "termes": ("ordinary", "extended_placement", "level_removal"),
    "freecell": ("ordinary", "same_suit", "any_suit"),
    "briefcase": ("ordinary", "two_slots", "latched"),
    "cooperative_kitchen": ("ordinary", "prepared_transfer", "two_counter_slots"),
    "rush_hour": ("ordinary", "glide", "parking_brakes"),
    "color_sort": ("ordinary", "mixed_stacks", "mixed_buffer"),
}

# Curated, recoverable compositions of one shared controller. A charge is NOT
# also deducted by the readiness or cooling effects. No irreversible budget.
PROFILES = (
    ("plain", "none", False, False),
    ("enable", "persistent", False, False),
    ("rearm", "per_work", False, False),
    ("motion_reset", "after_motion", False, False),
    ("charge", "none", True, False),
    ("charge_enable", "persistent", True, False),
    ("charge_rearm", "per_work", True, False),
    ("charge_motion_reset", "after_motion", True, False),
    ("cool", "none", False, True),
    ("cool_enable", "persistent", False, True),
    ("cool_charge", "none", True, True),
    ("cool_charge_rearm", "per_work", True, True),
)

WORK = {
    "blocks": ("pickup", "unstack", "putdown", "stack"),
    "hanoi": ("move",),
    "sliding_puzzle": tuple("move_blank_"+d for d in grid_games.DIRECTIONS) + ("move_blank_diagonal", "slide_pair"),
    "frozenlake": ("record_checkpoint", "move_north", "move_east", "move_south", "move_west"),
    "maze": ("unlock_door",),
    "sokoban": ("push_to_goal", "push_to_nongoal", "pull_box", "push_two_boxes"),
    "package": ("cut_tape", "remove_clip", "open_package"),
    "printer": ("print_document",),
    "kitchen": tuple(m for recipe in causal_games.RECIPES.values() for food,steps in recipe.items() if food != "finish" for m in steps) + ("finish_mix", "finish_bake"),
    "gripper": ("pick", "drop"),
    "miconic": ("board", "depart"),
    "lights_out": ("press_button",),
    "floortile": ("paint_up", "paint_down", "paint_left", "paint_right"),
    "key_delivery": ("unlock", "pickup_relic", "put_down_relic"),
    "termes": ("place_block", "remove_block", "place_block_higher", "remove_block_level"),
    "freecell": ("move", "move_b", "colfromfreecell", "sendtofree", "sendtofree_b", "sendtonewcol", "newcolfromfreecell", "sendtohome", "sendtohome_b", "homefromfreecell"),
    "briefcase": ("put_in", "take_out"),
    "cooperative_kitchen": ("chop", "put_on_counter", "take_from_counter"),
    "rush_hour": ("move_vehicle", "exit_vehicle", "slide_vehicle"),
    "color_sort": ("move_to_empty", "move_onto"),
}

POLICY_TEXT = {
    "blocks": ("Any supported clear block may receive a held block.", "For a new stack, bN may support bM only when N > M. The labels define this board's load order; existing stacks may be dismantled regardless of order.", "For a new stack, the supporting block must rest directly on the table; new towers are at most two blocks high. Existing taller towers may be dismantled."),
    "hanoi": ("A disk may move between any pegs.", "A disk may move only between neighbouring pegs in the line peg1--peg2--peg3; moving within one peg is forbidden.", "A disk may move only clockwise between pegs: peg1 to peg2, peg2 to peg3, peg3 to peg1. Moving within one peg is forbidden."),
    "sliding_puzzle": ("Only the four orthogonal single-tile swaps are allowed.", "In addition to orthogonal swaps, move_blank_diagonal swaps the blank with one diagonally adjacent tile.", "In addition to single swaps, slide_pair shifts two consecutive tiles along one orthogonal line towards the blank, moving the blank two cells. All three cells must be consecutive and distinct."),
    "frozenlake": ("Cracked tiles collapse permanently after departure; recording is explicit.", "Cracked tiles have been reinforced: leaving them preserves their safety. Recording is explicit.", "Cracked tiles collapse after departure. Arrival also records every checkpoint on that cell whose earlier checkpoints were already recorded BEFORE the move. Explicit recording remains available."),
    "maze": ("keyN is a single-use lock insert: unlocking consumes it and keeps the door open.", "keyN is reusable: unlocking keeps it in inventory and keeps the door open.", "keyN is a single-use lock insert. Unlocking an adjacent matching door is allowed from any heading; movement still follows heading. Unlocking consumes the insert."),
    "sokoban": ("Boxes may only be pushed one at a time.", "pull_box additionally steps backwards into an empty cell while pulling the box in front into the player's old cell.", "push_two_boxes additionally pushes two consecutive boxes one cell along a straight line when the cell beyond both is empty; the player advances into the first box's old cell."),
    "package": ("Tape and clips may be removed in either order.", "A clip may be removed only after all tape on that package has been cut.", "Tape may be cut only after that package's clips have been removed."),
    "printer": ("Printing preserves calibration. Switching power off clears calibration.", "Each print clears calibration. Switching power off also clears it.", "Printing and switching power off preserve calibration; changing a cartridge still clears calibration."),
    "kitchen": ("All ingredient processing and recipe finishing dirty the used station.", "Every processing/finishing station automatically rinses after use and remains clean.", "Only the wash station automatically rinses and remains clean; other processing and finishing dirty their stations."),
    "gripper": ("The two grippers work independently, allowing two carried balls.", "The shared payload limit is one ball total: picking requires both grippers empty; dropping releases that load.", "The left gripper accepts even-numbered balls and the right accepts odd-numbered balls. Either may drop its held ball; capacities remain one each."),
    "miconic": ("The lift has unlimited passenger capacity and may travel directly between any floors.", "The lift has two seats. Boarding occupies one seat and departing releases it; travel between any floors is permitted.", "Capacity is unlimited. Every up/down trip must be to the immediately neighbouring floor, with height differing by one."),
    "lights_out": ("A press flips its own light and orthogonal neighbours.", "A press flips its own light and immediate LEFT/RIGHT neighbours, with no vertical effect.", "A press flips its own light and immediate DIAGONAL neighbours, with no orthogonal-neighbour effect."),
    "floortile": ("Painted tiles block entry permanently. Paint immediately above or below the robot.", "Paint dries immediately and robots may walk on painted tiles. Paint still cannot be overwritten. Only unpainted unoccupied tiles may be painted, above or below.", "Painted tiles block entry permanently. The brush may paint immediately above, below, left or right of the robot; all other paint conditions stay the same."),
    "key_delivery": ("keyN is a single-use lock insert, consumed when unlocking doorN and freeing the hand.", "keyN is reusable. Unlocking leaves it loose in the player's current room and frees the hand.", "keyN is reusable and stays in the hand after unlocking. put_down_key places a held key in the current room and frees the hand."),
    "termes": ("Placement raises an adjacent tile at the robot's current height; removal lowers a tile one level higher.", "place_block_higher additionally places a carried block on an adjacent non-depot tile one level ABOVE the robot, raising it one more level; the articulated arm has this extra reach.", "remove_block_level additionally removes the top block of an adjacent non-depot tile at the SAME positive height as the robot, lowering it one level and filling the empty carrier."),
    "freecell": ("Tableau stacks require the next higher rank of the opposite colour, and a moved card of rank at least two.", "Tableau stacks require the next higher rank of the SAME SUIT, and a moved card of rank at least two.", "Tableau stacks require only the next higher rank, with any suit, and a moved card of rank at least two."),
    "briefcase": ("Case capacity is unlimited and no latch is used.", "The case has two item slots. Insertion occupies one and removal releases one. All contained items still move together.", "Case capacity is unlimited. The case starts with the depicted latch state. open_case requires closed; close_case requires open. Insertion/removal require open and movement requires closed. Latch actions preserve contents and positions."),
    "cooperative_kitchen": ("The passing counter holds one food; raw or chopped food may be transferred.", "The passing counter holds one food and accepts only chopped food. Taking food still requires an empty hand.", "The passing counter holds two foods. Putting occupies one slot and taking frees one. Other hand, chopping and plate rules stay the same."),
    "rush_hour": ("Vehicle shifts are exactly one cell.", "slide_vehicle additionally slides a vehicle any positive number of cells along its lane in one operation, if every newly traversed cell is empty. It cannot jump vehicles and preserves its length/orientation.", "Every vehicle has a parking brake. release_brake requires its brake set and releases it without moving. Moving or exiting requires released brake; each shift sets that vehicle's brake again. Other vehicles' brakes are unchanged."),
    "color_sort": ("A nut may be placed on an empty bolt or a nut of the same colour.", "A nut may be placed on any non-full different bolt, even on a different colour. The final colour assignments remain required.", "Only the highest-numbered bolt is a mixed-colour buffer: it may receive any colour on top. Other nonempty targets require matching colours. All bolts keep the stated capacity."),
}

RATIONALE = {
    "blocks": "Explicit load order and maximum new tower height are tabletop handling policies, not guesses from block colour or drawn size.",
    "hanoi": "Directed or neighbouring-peg access is an explicit Hanoi board variant. Disk size and support conservation are unchanged.",
    "sliding_puzzle": "Diagonal slots and linked pair shifts are explicit board movement variants; every move conserves tiles and the unique blank.",
    "frozenlake": "Reinforcement prevents collapse; automatic checkpoint sensors record only dependencies satisfied before arrival.",
    "maze": "A consumed object is explicitly a disposable lock insert. Reusable keys remain present; side access affects unlocking, not walking through walls.",
    "sokoban": "Pulling grips and two-box pushing are explicit warehouse game capabilities; empty destinations and collinearity remain necessary.",
    "package": "Layered fasteners can require tape-before-clip or clip-before-tape removal. Reusable cutter/pliers never disappear and nested contents stay with their container.",
    "printer": "Printers may retain or require renewed calibration. Paper is drawn from the visible finite eight-sheet feeder, never created by printing or completion.",
    "kitchen": "Self-rinsing equipment changes cleaning dependency without changing ingredient phase order or creating ingredients.",
    "gripper": "A shared payload limit differs from independent hand capacity; gripper compatibility is an explicit equipment restriction.",
    "miconic": "Seats are occupied then released, unlike consumed energy. Neighbouring-only trips model a stop-by-stop lift controller.",
    "lights_out": "Wiring patterns are explicit board conventions. Every affected lamp reads the pre-action state before simultaneous inversion.",
    "floortile": "Dry paint can support walking without permitting repainting. Four-way brushes extend reach without creating materials or changing occupied cells.",
    "key_delivery": "Disposable inserts are distinguished from reusable metal keys. Reusable keys remain either in hand or on the floor after unlocking.",
    "termes": "An articulated placement arm and lateral top-block removal have stated height reach; blocks are transferred, not duplicated.",
    "freecell": "Same-suit or any-suit descending tableau placement is an explicit card-game variant. Foundation rank/suit rules and buffer release remain fixed.",
    "briefcase": "Finite slots and a travel latch are ordinary container constraints; contained items move simultaneously with the case.",
    "cooperative_kitchen": "A prepared-food pass prevents raw transfer; a second counter slot is occupied/released, while each chef still has one hand.",
    "rush_hour": "Glides require the whole path clear; parking brakes require releasing and re-engage on each shift. Neither changes vehicle footprints.",
    "color_sort": "Mixed stacks or one mixed buffer are declared sorting-game policies; only exposed nuts move and physical bolt capacity is conserved.",
}

POLICIES.update({g:('ordinary',) for g in extra_games.FAMILIES})
POLICY_TEXT.update({g:('The physical operations follow the complete game rules above.',) for g in extra_games.FAMILIES})
RATIONALE.update(extra_games.RATIONALE)
WORK.update(extra_games.WORK)

def catalog():
    out = []
    for family in FAMILIES:
        for pi, policy in enumerate(POLICIES[family]):
            for si, (profile, prep, energy, cool) in enumerate(PROFILES):
                # A one-operation game has no distinct travel reset. Never
                # create an inactive or duplicate controller configuration.
                if family in extra_games.FAMILIES and prep=='after_motion':continue
                # Preserve the two existing scene streams independently of the
                # user's train-only allocation. This does not change semantics.
                scene_stream = int(pi == 2 and si >= 9)
                tags = ["game_policy:"+policy] if pi else []
                if prep != "none": tags += ["preparation", "preparation:"+prep]
                if energy: tags += ["resource_consumption", "resource_recovery"]
                if cool: tags += ["shared_cooldown"]
                if policy in {"two_slots", "two_counter_slots", "two_seats", "shared_load_limit"}: tags += ["capacity_occupation", "capacity_release"]
                out.append(dict(rule_id=f"{family}.{policy}.{profile}", family=family, policy=policy,
                                policy_index=pi, profile=profile, profile_index=si, preparation=prep,
                                energy=energy, cooling=cool, split="train", scene_stream=scene_stream, quota=20,
                                structure_group=f"{family}/{policy}/{profile}", mechanisms=tags,
                                natural_language=POLICY_TEXT[family][pi], rationale=RATIONALE[family],
                                combination_review="One shared controller, one charge decrement per guarded work operation; prepare/recharge only while idle, wait always recovers cooling. Game-local resources and controller charge are separate units. No effect restores consumed paper. No prerequisite prevents waiting or charging. Preparation and cooling change only controller state; game state and held contents persist."))
    # Append new policies only after all 820 historical configurations. Their
    # indices and task IDs therefore stay stable across the additive release.
    from .policies import EXTRA_POLICIES, SUPPORTED_PROFILES
    for family in FAMILIES:
        if family not in EXTRA_POLICIES:
            continue
        policy, text, rationale = EXTRA_POLICIES[family]
        pi = len(POLICIES[family])
        POLICY_TEXT[family] = POLICY_TEXT[family][:pi] + (text,)
        for si, (profile, prep, energy, cool) in enumerate(PROFILES):
            if profile not in SUPPORTED_PROFILES:
                continue
            tags = ['game_policy:' + policy, 'physical_policy_addition_v6']
            if prep != 'none': tags += ['preparation', 'preparation:' + prep]
            if energy: tags += ['resource_consumption', 'resource_recovery']
            if cool: tags += ['shared_cooldown']
            out.append(dict(rule_id=f'{family}.{policy}.{profile}', family=family,
                policy=policy, policy_index=pi, profile=profile, profile_index=si,
                preparation=prep, energy=energy, cooling=cool, split='train',
                scene_stream=0, quota=20, structure_group=f'{family}/{policy}/{profile}',
                mechanisms=tags, natural_language=text, rationale=rationale,
                combination_review='One physical policy and one recoverable shared controller. '
                    'Each guarded operation consumes at most one energy unit; waiting and '
                    'external charging cannot be blocked by a disabled controller.'))
    return out

def profile_rules(cfg):
    work = ", ".join(dict.fromkeys(WORK[cfg['family']]))
    # Work list is intersected with the rule's stable operation set later.
    pieces = ["The bottom badge is a shared task controller used by the actors performing the allowed operations. ARM shows enabled/not enabled; DRIVE shows stored energy units (0..3); COOL shows busy/idle. Features not required below ignore their indicator, which simply persists."]
    if cfg["preparation"] != "none":
        pieces += ["Every guarded operation requires ARM enabled. prepare_controller enables it only while idle and currently disabled, without changing object positions or hand contents. This is a remote safety interlock; it can be operated from any position, including with a load held."]
        pieces += [{"persistent": "Guarded operations preserve ARM.", "per_work": "Each guarded operation disables ARM, so the next guarded operation needs enabling again.", "after_motion": RESET_TEXT.get(cfg['family'],"Position-changing travel disables ARM; turns in place and guarded work preserve it.")}[cfg["preparation"]]]
    if cfg["energy"]:
        pieces += ["Every guarded operation consumes exactly one shared DRIVE unit and requires at least one. recharge_controller, only while idle and below three, restores DRIVE to three using the external wired supply. It is available from every position and with held loads; it does not replenish game materials, seats or slots."]
    if cfg["cooling"]:
        pieces += ["Guarded operations require COOL idle and set it busy. wait_for_controller requires busy and restores idle, preserving ARM, DRIVE and all game objects. Preparation and recharging require idle; waiting does not require charge or enabling."]
    return " ".join(pieces)

def conjunct(node):
    return list(node[1:]) if node[:1] == ["and"] else [node]

def add_pre(a, *clauses):
    i=a.index(":precondition")+1; a[i]=["and", *conjunct(a[i]), *clauses]

def add_eff(a, *clauses):
    i=a.index(":effect")+1; a[i]=["and", *conjunct(a[i]), *clauses]

def omit(node, atom):
    if node == atom: return ["and"]
    return [omit(x, atom) for x in node] if isinstance(node,list) else node

def change_field(a, field, old, new):
    def walk(n):
        if n == old: return copy.deepcopy(new)
        return [walk(x) for x in n] if isinstance(n,list) else n
    i=a.index(field)+1; a[i]=walk(a[i])

RULES = {
    "blocks": {"object": ("block", 0), "arm_empty": ("hand_free", "$hand"),
               "holding": ("holding", "$hand", 0), "on_table": ("on", 0, "$table")},
    "briefcase": {"in": ("in", 0, "briefcase"), "is_at": ("at", "briefcase", 0)},
    "floortile": {"robot_at": ("at", 0, 1)},
    "frozenlake": {"checkpoint_at": ("at", 0, 1)},
    "gripper": {"gripper": ("hand", 0), "free": ("hand_free", 0),
                "carry": ("holding", 1, 0), "at_robby": ("at", "robby", 0)},
    "key_delivery": {"at_player": ("at", "player", 0), "key_at": ("at", 0, 1),
                     "relic_at": ("at", 0, 1), "hand_empty": ("hand_free", "$hand"),
                     "holding_key": ("holding", "$hand", 0),
                     "holding_relic": ("holding", "$hand", 0)},
    "kitchen": {"arm_empty": ("hand_free", "$hand"), "holding": ("holding", "$hand", 0),
                "on_plate": ("on", 0, 1), "tool_at": ("at", 0, 1)},
    "lights_out": {"button_on": ("on", 0, 1)},
    "maze": {"key_at": ("at", 0, 1), "has_key": ("in", 0, "inventory")},
    "miconic": {"boarded": ("in", 0, "lift"), "lift_at": ("at", "lift", 0)},
    "package": {"tool_at": ("at", 0, 1), "has_tool": ("in", 0, "inventory"),
                "depends_on": ("in", 0, 1)},
    "printer": {"arm_empty": ("hand_free", "$hand"), "holding": ("holding", "$hand", 0),
                "installed_cartridge": ("in", 1, 0)},
    "termes": {"at": ("at", "robot", 0)},
}

ENTITIES = {"$hand": ("hand1", "hand"), "$table": ("table", "table"),
            "briefcase": ("briefcase", "briefcase"), "robby": ("robby", "robot"),
            "player": ("player", "player"), "inventory": ("inventory", "container"),
            "lift": ("lift", "lift"), "robot": ("robot", "robot")}

VARIABLES = {"$hand": "?h", "$table": "?surface"}

MEANINGS = {"at": "{0} is at {1}", "on": "{0} is directly on {1}",
            "in": "{0} is inside {1}", "holding": "hand {0} holds {1}",
            "hand_free": "hand {0} is empty"}

def transform(node, rules, action=False):
    if not isinstance(node, list) or not node:
        return node
    if isinstance(node[0], str) and node[0] in rules:
        name, *args = rules[node[0]]
        return [name, *(node[a + 1] if isinstance(a, int) else
                       VARIABLES[a] if action and a in VARIABLES else ENTITIES[a][0]
                       for a in args)]
    return [transform(x, rules, action) for x in node]


def align(raw):
    """Only predicate/hand representation; never prune actions or read a plan."""
    d,p=parse_sexpr(raw['domain']),parse_sexpr(raw['problem'])
    rules={k:v for k,v in RULES.get(raw['game'],{}).items() if k in {q[0] for q in section(d,':predicates')[1:]}}
    keys={x for v in rules.values() for x in v[1:] if isinstance(x,str)}
    existing=set(section(p,':objects')[1:]+section(d,':constants',[':constants'])[1:])
    extra=[]
    for key in sorted(keys):
        obj,kind=ENTITIES[key];extra.append([kind,obj])
        if obj not in existing: section(p,':objects').append(obj);existing.add(obj)
    for a in d:
        if not isinstance(a,list) or a[:1] != [':action']:continue
        for field in (':precondition',':effect'):
            i=a.index(field)+1;a[i]=transform(a[i],rules,True)
        f=fields(a)
        for key,var in VARIABLES.items():
            if key in keys and var in re.findall(r'\?[a-z_0-9]+',str([f[':precondition'],f[':effect']])):
                f[':parameters'].append(var);add_pre(a,[ENTITIES[key][1],var])
    ini=transform(section(p,':init')[1:],rules)+extra
    goal=transform(section(p,':goal')[1],rules)
    return d,p,ini,goal,raw.get('vocabulary',{})|MEANINGS,rules,extra

@lru_cache(maxsize=1)
def kitchen_actions():
    ops={}
    for seed in range(6):
        r=causal_games.kitchen(100002+seed)
        d,*_=align(r)
        for a in d:
            if isinstance(a,list) and a[:1]==[':action']:ops[a[1]]=a
    return ops

def compact_sokoban(seed, scene_options=None):
    """New self-generated warehouse, no downloaded level or old test input."""
    options=scene_options or {};h,w=options.get('board',(4,5));count=options.get('boxes',2)
    rng=random.Random(seed); sc=grid_games.board(rng,h,w)
    cells=sc['cells']; boxes=[f'box{i+1}' for i in range(count)]; pos={'player':rng.choice(cells)}
    spots=rng.sample([c for c in cells if c != pos['player'] and 1 <= grid_games.rc(c)[0] < h-1 and 1 <= grid_games.rc(c)[1] < w-1],count)
    pos.update(zip(boxes,spots)); goals=rng.sample(cells,count)
    init=[['location',c] for c in cells]+[['direction',d] for d in grid_games.DIRECTIONS]+[['box',b] for b in boxes]
    init += [['at',a,b] for a,b in pos.items()]+[['empty',c] for c in cells if c not in pos.values()]
    init += [['ahead',*e] for e in sc['edges']]+[['goal_cell',c] for c in goals]+[['non_goal_cell',c] for c in cells if c not in goals]
    init += [['on_goal',b] for b in boxes if pos[b] in goals]
    ops=[action('move','?from ?to ?d','(and (location ?from) (location ?to) (direction ?d) (at player ?from) (ahead ?from ?to ?d) (empty ?to))','(and (not (at player ?from)) (at player ?to) (empty ?from) (not (empty ?to)))')]
    for name,pred in [('push_to_goal','goal_cell'),('push_to_nongoal','non_goal_cell')]:
        ops.append(action(name,'?box ?from ?boxloc ?to ?d',f'(and (box ?box) (location ?from) (location ?boxloc) (location ?to) (direction ?d) (at player ?from) (at ?box ?boxloc) (ahead ?from ?boxloc ?d) (ahead ?boxloc ?to ?d) (empty ?to) ({pred} ?to))',f'(and (not (at player ?from)) (at player ?boxloc) (not (at ?box ?boxloc)) (at ?box ?to) (empty ?from) (not (empty ?to)) {"(on_goal ?box)" if pred=="goal_cell" else "(not (on_goal ?box))"})'))
    sc['goals']=goals
    return grid_games.finish('sokoban',seed,'local_open_warehouse',cells+boxes+list(grid_games.DIRECTIONS),init,[['on_goal',b] for b in boxes],ops,{'on_goal':'box {0} is on a target','goal_cell':'cell {0} is a target','non_goal_cell':'cell {0} is not a target'},['Boxes occupy cells.'],sc)

def source(family, seed, scene_generation=None, scene_options=None):
    options=copy.deepcopy(scene_options or {})
    if options:
        from .policies import OPTIONS
        allowed={k for variant in OPTIONS[family] for k in variant}
        if family=='color_sort':allowed.add('counts')
        if family in {'blocks','sokoban'}:allowed.add('goal_selection')
        if family=='lights_out':allowed.add('scramble_presses')
        if family=='maze':allowed.add('key_placement')
        if family=='printer':allowed.add('documents_per_printer')
        if family in {'blocks','hanoi'}:allowed.update({'goal_selection','goal_minimum_distance'})
        assert not set(options)-allowed,(family,'unknown_scene_options',set(options)-allowed)
    if family in ('blocks','sliding_puzzle','cooperative_kitchen','rush_hour','color_sort'):
        game={'blocks':'blocks_3d','sliding_puzzle':'numeric_sliding'}.get(family,family)
        size={'blocks':dict(blocks=5),'sliding_puzzle':dict(board=(2,3),scramble=6),'color_sort':dict(capacity=2)}.get(family)
        if options:size=(size or {})|options
        return vg.GENERATORS[game](seed,0,size)
    if family=='sokoban': return compact_sokoban(seed,options)
    # Explicitly choose baseline source rules rather than seed-dependent rule IDs.
    sizes={'hanoi':dict(disks=3,goal_walk=5),'gripper':dict(rooms=3,balls=3),'miconic':dict(floors=3,passengers=3),
           'floortile':dict(rows=2,columns=3),'termes':dict(board=(3,3)), 'freecell':dict(rank=3),
           'briefcase':dict(rooms=3,items=3),'frozenlake':dict(board=(4,4)), 'maze':dict(board=(3,4)), 'package':dict(board=(4,4)),
           'lights_out':dict(board=(2,3)), 'key_delivery':dict(rooms=4)}
    if scene_generation is not None:
        assert family=='lights_out' and scene_generation=='lights_out_3x3_v1',scene_generation
        sizes['lights_out']=dict(board=(3,3))
    if options:sizes[family]=(sizes.get(family) or {})|options
    chosen=seed
    if family=='miconic': chosen=seed*3
    if family=='printer': chosen=seed*6+4 # one printer, two cartridges; never source print-reset
    if family=='maze': chosen=seed*3+1
    if family=='package': chosen=seed*3+1
    if family=='frozenlake': chosen=seed*6+3 # fragile + checkpoint order, explicit
    raw=games.generate(family,chosen,sizes.get(family))
    if family=='freecell':
        # The upstream random deal can produce fewer nonempty stacks than -i.
        # Derive free columns from the actual depicted stacks, not that request.
        p=parse_sexpr(raw['problem']);ini=section(p,':init')
        capacity=max(int(a[1][4:]) for a in ini[1:] if a[0]=='colnum')
        occupied=sum(a[0]=='bottomcol' for a in ini[1:])
        ini[:]=[':init',*[a for a in ini[1:] if a[0]!='colspace'],['colspace',f'coln{capacity-occupied}']]
        raw['problem']=sexpr(p)
    raw['seed']=seed
    return raw

RESET_TEXT = {
    'frozenlake':'Moving to another cell disables ARM; record_checkpoint preserves ARM.',
    'blocks':'Putting a block directly on the table disables ARM; stacking preserves it.',
    'hanoi':'Landing a disk directly on an empty peg disables ARM; landing on a disk preserves it.',
    'sliding_puzzle':'Vertical single swaps disable ARM; horizontal swaps and optional diagonal/pair moves preserve it.',
    'lights_out':'Connecting or disconnecting a row panel disables ARM; button presses preserve it.',
    'color_sort':'Placing a nut on an empty bolt disables ARM; placing it on another nut preserves it.',
    'freecell':'Tableau-to-tableau stacking disables ARM; buffer and foundation moves preserve it.',
    'rush_hour':'Shifting a non-target vehicle disables ARM; shifting or exiting the target vehicle preserves it.',
}

def reset_clause(family,name):
    if family=='frozenlake':return [] if name in {'move_north','move_east','move_south','move_west'} else None
    if family=='blocks': return [] if name=='putdown' else None
    if family=='hanoi': return ['peg','?to'] if name=='move' else None
    if family=='sliding_puzzle': return [] if name in {'move_blank_north','move_blank_south'} else None
    if family=='lights_out': return [] if name in {'connect_panel','disconnect_panel'} else None
    if family=='color_sort': return [] if name=='move_to_empty' else None
    if family=='freecell': return [] if name in {'move','move_b'} else None
    if family=='rush_hour': return ['not',['target','?vehicle']] if name in {'move_vehicle','slide_vehicle'} else None
    return [] if name in {'move','up','down','left','right','move_up','move_down','move_forward'} else None

def compile_controller(cfg, actions, initial, objects):
    prep,energy,cool=cfg['preparation'],cfg['energy'],cfg['cooling']
    for a in actions:
        name=a[1]
        if name in WORK[cfg['family']]:
            if prep!='none':add_pre(a,['controller_ready'])
            if prep=='per_work':add_eff(a,['not',['controller_ready']])
            if cool:add_pre(a,['not',['controller_busy']]);add_eff(a,['controller_busy'])
            if energy:
                fields(a)[':parameters'] += ['?charge_previous','?charge_result']
                add_pre(a,['charge_value','?charge_previous'],['charge_value','?charge_result'],['controller_charge','?charge_previous'],['charge_step','?charge_previous','?charge_result'])
                add_eff(a,['not',['controller_charge','?charge_previous']],['controller_charge','?charge_result'])
        reset=reset_clause(cfg['family'],name)
        if prep=='after_motion' and reset is not None:
            add_eff(a,['when',reset,['not',['controller_ready']]] if reset else ['not',['controller_ready']])
    if prep!='none':
        # With no cooling profile controller_busy is never initialized or
        # changed, so its negated guard is a tautology. Omit that dead
        # dependency from the published causal model.
        prepare_guard = '(and (not (controller_ready))' + (' (not (controller_busy))' if cool else '') + ')'
        actions.append(action('prepare_controller','',prepare_guard,'(controller_ready)'))
    if energy:
        recharge_guard = '(and (charge_value ?previous) (controller_charge ?previous) (not (= ?previous q3))' + (' (not (controller_busy))' if cool else '') + ')'
        actions.append(action('recharge_controller','?previous',recharge_guard,'(and (not (controller_charge ?previous)) (controller_charge q3))'))
    if cool:actions.append(action('wait_for_controller','','(controller_busy)','(not (controller_busy))'))
    initial += [['charge_value',f'q{i}'] for i in range(4)]+[['charge_step',f'q{i}',f'q{i-1}'] for i in range(1,4)]
    objects += [f'q{i}' for i in range(4)]

def compile_policy(cfg, ops, ini, objects, raw):
    """Game-specific PDDL edits. The native implementation lives separately."""
    g,pol=cfg['family'],cfg['policy']; by={a[1]:a for a in ops}
    facts=set(map(tuple,ini))
    def vals(p):return sorted(a[1] for a in ini if a[0]==p and len(a)==2)
    if g=='blocks':
        if pol in {'ordered_support','ordered_two_high'}:
            add_pre(by['stack'],['load_support','?underob','?ob'])
            ini += [['load_support',a,b] for a in vals('block') for b in vals('block') if int(a[1:])>int(b[1:])]
        if pol in {'two_high','ordered_two_high'}:add_pre(by['stack'],['on','?underob','table'])
    elif g=='hanoi':
        disks=[x for x in vals('object') if x.startswith('d')];pegs=['peg1','peg2','peg3']
        ini += [['disk',d] for d in disks]+[['peg',p] for p in pegs]
        add_pre(by['move'],['disk','?disc'])
        if pol!='ordinary':
            parent={a[1]:a[2] for a in ini if a[0]=='on'}
            def root(x):
                while x in parent:x=parent[x]
                return x
            ini += [['tower',x,root(x)] for x in disks+pegs]
            routes=[(a,b) for a in pegs for b in pegs if (abs(int(a[-1])-int(b[-1]))==1 if pol=='adjacent_pegs' else int(b[-1])==(int(a[-1])+1)%3+1 if pol=='counterclockwise_pegs' else int(b[-1])==int(a[-1])%3+1)]
            ini += [['peg_route',a,b] for a,b in routes]
            fields(by['move'])[':parameters'] += ['?source_peg','?target_peg']
            add_pre(by['move'],['peg','?source_peg'],['peg','?target_peg'],['tower','?disc','?source_peg'],['tower','?to','?target_peg'],['peg_route','?source_peg','?target_peg'])
            add_eff(by['move'],['not',['tower','?disc','?source_peg']],['tower','?disc','?target_peg'])
    elif g=='sliding_puzzle':
        cells=vals('position')
        if pol in {'diagonal','diagonal_pair'}:
            ini += [['diagonal',a,b] for a in cells for b in cells if all(abs(x-y)==1 for x,y in zip(grid_games.rc(a),grid_games.rc(b)))]
            ops.append(action('move_blank_diagonal','?tile ?from ?to','(and (tile ?tile) (position ?from) (position ?to) (at ?tile ?from) (empty ?to) (diagonal ?from ?to))','(and (not (at ?tile ?from)) (at ?tile ?to) (not (empty ?to)) (empty ?from))'))
        if pol in {'pair_slide','diagonal_pair'}:
            for a in cells:
                r,c=grid_games.rc(a)
                for dr,dc in grid_games.DIRECTIONS.values():
                    b,d=grid_games.cell(r+dr,c+dc),grid_games.cell(r+2*dr,c+2*dc)
                    if b in cells and d in cells:ini.append(['line_three',a,b,d])
            ops.append(action('slide_pair','?near ?far ?blank ?middle ?end','(and (tile ?near) (tile ?far) (position ?blank) (position ?middle) (position ?end) (line_three ?blank ?middle ?end) (empty ?blank) (at ?near ?middle) (at ?far ?end))','(and (not (empty ?blank)) (empty ?end) (not (at ?near ?middle)) (at ?near ?blank) (not (at ?far ?end)) (at ?far ?middle))'))
    elif g=='frozenlake':
        for name,a in by.items():
            if not name.startswith('move_'):continue
            if pol in {'reinforced','reinforced_arrival'}:change_field(a,':effect',['when',['fragile','?from'],['not',['safe','?from']]],['and'])
            if pol in {'arrival_records','reinforced_arrival'}:
                add_eff(a,['forall',['?mark'],['when',['and',['checkpoint','?mark'],['at','?mark','?to'],['forall',['?prev'],['imply',['before','?prev','?mark'],['marked','?prev']]]],['marked','?mark']]])
    elif g=='maze':
        if pol in {'reusable_key','reusable_side_unlock'}:change_field(by['unlock_door'],':effect',['not',['in','?key','inventory']],['and'])
        if pol in {'side_unlock','reusable_side_unlock'}:change_field(by['unlock_door'],':precondition',['facing','?d'],['and'])
    elif g=='sokoban':
        if pol in {'pulling','pulling_double_push'}:
            ops.append(action('pull_box','?box ?from ?boxloc ?to ?d','(and (box ?box) (location ?from) (location ?boxloc) (location ?to) (direction ?d) (at player ?from) (at ?box ?boxloc) (ahead ?to ?from ?d) (ahead ?from ?boxloc ?d) (empty ?to))','(and (not (at player ?from)) (at player ?to) (not (at ?box ?boxloc)) (at ?box ?from) (empty ?boxloc) (not (empty ?to)) (when (goal_cell ?from) (on_goal ?box)) (when (non_goal_cell ?from) (not (on_goal ?box))))'))
        if pol in {'double_push','pulling_double_push'}:
            ops.append(action('push_two_boxes','?first ?second ?from ?near ?far ?to ?d','(and (box ?first) (box ?second) (location ?from) (location ?near) (location ?far) (location ?to) (direction ?d) (at player ?from) (at ?first ?near) (at ?second ?far) (ahead ?from ?near ?d) (ahead ?near ?far ?d) (ahead ?far ?to ?d) (empty ?to))','(and (not (at player ?from)) (at player ?near) (not (at ?first ?near)) (at ?first ?far) (not (at ?second ?far)) (at ?second ?to) (empty ?from) (not (empty ?to)) (when (goal_cell ?far) (on_goal ?first)) (when (non_goal_cell ?far) (not (on_goal ?first))) (when (goal_cell ?to) (on_goal ?second)) (when (non_goal_cell ?to) (not (on_goal ?second))))'))
    elif g=='package':
        if pol=='tape_first':add_pre(by['remove_clip'],['not',['taped','?p']])
        if pol=='clip_first':add_pre(by['cut_tape'],['not',['clipped','?p']])
        if pol=='cutter_required':add_pre(by['open_package'],['in','cutter','inventory'])
    elif g=='printer':
        # The feeder is physically present, finite, and follows the operator.
        objects += [f'stock{i}' for i in range(9)]
        ini += [['paper_stock','stock8']]+[['stock_value',f'stock{i}'] for i in range(9)]+[['stock_step',f'stock{i}',f'stock{i-1}'] for i in range(1,9)]
        fields(by['load_paper'])[':parameters'] += ['?stock_previous','?stock_result']
        add_pre(by['load_paper'],['stock_value','?stock_previous'],['stock_value','?stock_result'],['paper_stock','?stock_previous'],['stock_step','?stock_previous','?stock_result'])
        add_eff(by['load_paper'],['not',['paper_stock','?stock_previous']],['paper_stock','?stock_result'])
        change_field(by['print_document'],':effect',['not',['calibrated','?p']],['and'])
        if pol in {'print_resets','print_resets_power_retains'}:add_eff(by['print_document'],['not',['calibrated','?p']])
        if pol in {'power_retains','print_resets_power_retains'}:change_field(by['power_off'],':effect',['not',['calibrated','?p']],['and'])
    elif g=='kitchen':
        # All recipe operators remain available, even without matching objects
        # in a particular scene. No per-plan or per-goal action pruning.
        existing={a[1] for a in ops}
        ops += [copy.deepcopy(a) for n,a in kitchen_actions().items() if n not in existing]
        if pol!='ordinary':
            for a in ops:
                if a[1] in WORK[g] and (pol=='self_rinsing' or (pol=='rinsing_sink' and a[1]=='wash') or (pol=='processing_rinses' and not a[1].startswith('finish_'))):
                    change_field(a,':effect',['not',['clean','?tool']],['and'])
    elif g=='gripper':
        # Same-room robot travel is an intentional legal no-op in the baseline.
        if pol in {'shared_load_limit','assigned_shared_load'}:add_pre(by['pick'],['forall',['?other'],['imply',['hand','?other'],['hand_free','?other']]])
        if pol in {'assigned_grippers','assigned_shared_load'}:
            ini += [['grip_fits',b,'left' if int(re.search(r'\d+$',b)[0])%2==0 else 'right'] for b in vals('ball')]
            add_pre(by['pick'],['grip_fits','?obj','?gripper'])
    elif g=='miconic':
        if pol in {'two_seats','two_seat_adjacent'}:
            objects += ['seat0','seat1','seat2'];ini += [['seat_free','seat2']]+[['seat_count',f'seat{i}'] for i in range(3)]+[['seat_step','seat2','seat1'],['seat_step','seat1','seat0']]
            for name in ('board','depart'):
                a=by[name];fields(a)[':parameters'] += ['?seat_previous','?seat_result']
                add_pre(a,['seat_count','?seat_previous'],['seat_count','?seat_result'],['seat_free','?seat_previous'],['seat_step', '?seat_previous' if name=='board' else '?seat_result', '?seat_result' if name=='board' else '?seat_previous'])
                add_eff(a,['not',['seat_free','?seat_previous']],['seat_free','?seat_result'])
        if pol in {'adjacent_stops','two_seat_adjacent'}:
            ini += [['next_floor',a,b] for a in vals('floor') for b in vals('floor') if abs(int(a[1:])-int(b[1:]))==1]
            for name in ('up','down'):add_pre(by[name],['next_floor','?f1','?f2'])
    elif g=='lights_out':
        ini[:]=[a for a in ini if a[0]!='affects']
        for b in vals('button'):
            r,c=map(int,b.removeprefix('button').split('_'))
            for l in vals('light'):
                x,y=map(int,l.removeprefix('light').split('_'));dr,dc=abs(r-x),abs(c-y)
                if (dr+dc<=1 if pol=='ordinary' else dr==0 and dc<=1 if pol=='horizontal' else dc==0 and dr<=1 if pol=='vertical' else (dr==dc==0 or dr==dc==1)):ini.append(['affects',b,l])
    elif g=='floortile':
        if pol in {'walk_on_dry_paint','dry_four_way'}:
            tiles=vals('tile');occupied={a[2] for a in ini if a[0]=='at' and a[1] in vals('robot')}
            ini += [['vacant',t] for t in tiles if t not in occupied]
            for name in ('up','down','left','right'):
                a=by[name];change_field(a,':precondition',['clear','?y'],['vacant','?y'])
                change_field(a,':effect',['clear','?x'],['when',['not',['exists',['?c'],['painted','?x','?c']]],['clear','?x']])
                add_eff(a,['vacant','?x'],['not',['vacant','?y']])
        if pol in {'four_way_brush','dry_four_way'}:
            for name,pred in [('paint_left','left'),('paint_right','right')]:
                a=copy.deepcopy(by['paint_up']);a[1]=name
                change_field(a,':precondition',['up','?y','?x'],[pred,'?y','?x']);ops.append(a)
    elif g=='key_delivery':
        if pol=='key_laid_down':add_eff(by['unlock'],['at','?key','?from'])
        if pol=='key_forwarded':add_eff(by['unlock'],['at','?key','?to'])
        if pol=='key_retained':
            change_field(by['unlock'],':effect',['not',['holding','?h','?key']],['and'])
            change_field(by['unlock'],':effect',['hand_free','?h'],['and'])
            ops.append(action('put_down_key','?key ?room ?h','(and (key ?key) (room ?room) (hand ?h) (at player ?room) (holding ?h ?key))','(and (not (holding ?h ?key)) (at ?key ?room) (hand_free ?h))'))
    elif g=='termes':
        if pol in {'extended_placement','articulated_both'}:
            ops.append(action('place_block_higher','?rpos ?bpos ?robot_height ?previous ?result','(and (position ?rpos) (position ?bpos) (numb ?robot_height) (numb ?previous) (numb ?result) (at robot ?rpos) (height ?rpos ?robot_height) (height ?bpos ?previous) (neighbor ?rpos ?bpos) (succ ?previous ?robot_height) (succ ?result ?previous) (has_block) (not (is_depot ?bpos)))','(and (not (height ?bpos ?previous)) (height ?bpos ?result) (not (has_block)))'))
        if pol in {'level_removal','articulated_both'}:
            ops.append(action('remove_block_level','?rpos ?bpos ?previous ?result','(and (position ?rpos) (position ?bpos) (numb ?previous) (numb ?result) (at robot ?rpos) (height ?rpos ?previous) (height ?bpos ?previous) (neighbor ?rpos ?bpos) (succ ?previous ?result) (not (has_block)) (not (is_depot ?bpos)))','(and (not (height ?bpos ?previous)) (height ?bpos ?result) (has_block))'))
    elif g=='freecell':
        if pol!='ordinary':
            ranks={a[1]:int(a[2][1:]) for a in ini if a[0]=='value'};suits={a[1]:a[2] for a in ini if a[0]=='hassuit'}
            ini[:]=[a for a in ini if a[0]!='canstack']
            ini += [['canstack',a,b] for a in ranks for b in ranks
                    if ranks[a]>=(1 if pol=='ace_tableau' else 2) and ranks[b]==ranks[a]+1
                    and ((suits[a] in {'c','s'}) != (suits[b] in {'c','s'})
                         if pol=='ace_tableau' else pol=='any_suit' or suits[a]==suits[b])]
    elif g=='briefcase':
        if pol in {'two_slots','two_slot_latch'}:
            contained=sum(a[0]=='in' for a in ini)
            assert contained<=2
            objects += ['slot0','slot1','slot2'];ini += [['case_free',f'slot{2-contained}']]+[['case_count',f'slot{i}'] for i in range(3)]+[['case_step','slot2','slot1'],['case_step','slot1','slot0']]
            for name in ('put_in','take_out'):
                a=by[name];fields(a)[':parameters'] += ['?slot_previous','?slot_result']
                add_pre(a,['case_count','?slot_previous'],['case_count','?slot_result'],['case_free','?slot_previous'],['case_step','?slot_previous' if name=='put_in' else '?slot_result','?slot_result' if name=='put_in' else '?slot_previous'])
                add_eff(a,['not',['case_free','?slot_previous']],['case_free','?slot_result'])
        if pol in {'latched','two_slot_latch'}:
            ini.append(['case_open'])
            for name in ('put_in','take_out'):add_pre(by[name],['case_open'])
            add_pre(by['move'],['not',['case_open']])
            ops += [action('open_case','','(not (case_open))','(case_open)'),action('close_case','','(case_open)','(not (case_open))')]
    elif g=='cooperative_kitchen':
        if pol in {'prepared_transfer','prepared_two_slots'}:add_pre(by['put_on_counter'],['prepared','?food'])
        if pol in {'two_counter_slots','prepared_two_slots'}:
            count=sum(a[0]=='on' and a[2]=='pass1' for a in ini)
            ini[:]=[a for a in ini if a[0]!='counter_free']
            objects += ['counter0','counter1','counter2'];ini += [['counter_load',f'counter{count}']]+[['counter_count',f'counter{i}'] for i in range(3)]+[['counter_step','counter0','counter1'],['counter_step','counter1','counter2']]
            for name in ('put_on_counter','take_from_counter'):
                a=by[name]
                for field in (':precondition',':effect'):
                    change_field(a,field,['not',['counter_free','?counter']],['and']);change_field(a,field,['counter_free','?counter'],['and'])
                fields(a)[':parameters'] += ['?counter_previous','?counter_result']
                add_pre(a,['counter_count','?counter_previous'],['counter_count','?counter_result'],['counter_load','?counter_previous'],['counter_step','?counter_previous' if name=='put_on_counter' else '?counter_result','?counter_result' if name=='put_on_counter' else '?counter_previous'])
                add_eff(a,['not',['counter_load','?counter_previous']],['counter_load','?counter_result'])
    elif g=='rush_hour':
        if pol in {'parking_brakes','glide_brakes'}:
            ini += [['brake_set',v] for v in vals('vehicle')]
            for name in ('move_vehicle','exit_vehicle'):
                add_pre(by[name],['not',['brake_set','?vehicle']]);add_eff(by[name],['brake_set','?vehicle'])
            ops.append(action('release_brake','?vehicle','(and (vehicle ?vehicle) (brake_set ?vehicle) (not (escaped ?vehicle)))','(not (brake_set ?vehicle))'))
        if pol in {'glide','glide_brakes'}:
            spec=raw['spec'];cells=vals('location')
            for v,car in spec['cars'].items():
                r,c=vg.rc(raw['state'][v]);anchors=[vg.cell(r,j) for j in range(spec['w']-car['length']+1)] if car['axis']=='horizontal' else [vg.cell(j,c) for j in range(spec['h']-car['length']+1)]
                for a in anchors:
                    for p in vg.footprint(car,a):ini.append(['covers',v,a,p])
                    for b in anchors:
                        if a==b:continue
                        ini.append(['lane_trip',v,a,b]);ra,ca=vg.rc(a);rb,cb=vg.rc(b)
                        swept=set()
                        for k in range(abs(rb-ra)+abs(cb-ca)+1):swept.update(vg.footprint(car,vg.cell(ra+k*((rb>ra)-(rb<ra)),ca+k*((cb>ca)-(cb<ca)))))
                        ini += [['swept',v,a,b,p] for p in sorted(swept-set(vg.footprint(car,a)))]
            ops.append(action('slide_vehicle','?vehicle ?from ?to','(and (vehicle ?vehicle) (location ?from) (location ?to) (at ?vehicle ?from) (lane_trip ?vehicle ?from ?to) (forall (?p) (imply (swept ?vehicle ?from ?to ?p) (empty ?p))))','(and (not (at ?vehicle ?from)) (at ?vehicle ?to) (forall (?p) (when (and (covers ?vehicle ?from ?p) (not (covers ?vehicle ?to ?p))) (empty ?p))) (forall (?p) (when (covers ?vehicle ?to ?p) (not (empty ?p)))))'))
            if pol=='glide_brakes':
                add_pre(ops[-1],['not',['brake_set','?vehicle']])
                add_eff(ops[-1],['brake_set','?vehicle'])
    elif g=='color_sort':
        if pol=='mixed_stacks':change_field(by['move_onto'],':precondition',['same_color','?nut','?support'],['and'])
        if pol in {'mixed_buffer','lowest_buffer'}:
            select=min if pol=='lowest_buffer' else max
            buffer=select(raw['state'],key=lambda x:int(x[1:]));ini.append(['mixed_buffer',buffer])
            change_field(by['move_onto'],':precondition',['same_color','?nut','?support'],['or',['same_color','?nut','?support'],['mixed_buffer','?target']])

def construct(cfg,seed,slot,alias=None):
    """Choose scene and goal before solving. No old task or old plan is read."""
    if cfg['family'] in extra_games.FAMILIES:return extra_games.construct(cfg,seed,slot,alias)
    from .rules_v2_native import successors
    raw=source(cfg['family'],seed,cfg.get('scene_generation'),cfg.get('scene_options'));g=cfg['family']
    if g=='printer' and slot%2==0 and cfg.get('scene_options',{}).get('documents_per_printer',2)==2:
        # Keep both cartridge objects and every operator available. Half the
        # scenes require repeated jobs with the same cartridge, so a print's
        # calibration effect changes an actual subsequent preparation need.
        p=parse_sexpr(raw['problem'])
        for a in section(p,':init')[1:]:
            if a[:2]==['document_cartridge','document1']:a[2]='cartridge0_0'
        raw['problem']=sexpr(p)
    if g=='cooperative_kitchen' and cfg['policy'] in {'prepared_transfer','prepared_two_slots'}:
        # This passing policy applies to the pictured initial counter too.
        for food,place in raw['state']['food_places'].items():
            if place==['on','pass1'] and food not in raw['state']['prepared']:raw['state']['prepared'].append(food)
        p=parse_sexpr(raw['problem']);old={tuple(a) for a in section(p,':init')[1:]}
        dynamic={a for a in old if a[0] in {'at','hand_free','holding','prepared','on','counter_free','served'}}
        section(p,':init')[:]=[':init',*(list(a) for a in sorted((old-dynamic)|vg.projection(raw,raw['state'])))];raw['problem']=sexpr(p)
    if g=='blocks' and raw['state']['held']:
        # Initial 2D rendering also supports this exact scene: start on table.
        b=raw['state']['held'];raw['state']['held']=None;raw['state']['on'][b]='table'
        p=parse_sexpr(raw['problem']);old={tuple(a) for a in section(p,':init')[1:]}
        old={a for a in old if a[0] not in {'on','holding','hand_free','clear'}}
        section(p,':init')[:]=[':init',*(list(a) for a in sorted(old|vg.projection(raw,raw['state'])))];raw['problem']=sexpr(p)
    visual=g in {'blocks','sliding_puzzle','cooperative_kitchen','rush_hour','color_sort'}
    if visual:
        d,p=parse_sexpr(raw['domain']),parse_sexpr(raw['problem']);ini=section(p,':init')[1:];goal=copy.deepcopy(section(p,':goal')[1]);vocab={};mapping={};extra=[]
        base_instruction=raw['instruction']
    else:
        d,p,ini,goal,vocab,mapping,extra=align(raw)
        if g in {'frozenlake','maze','sokoban','package','printer','kitchen'} and ['player','player'] not in ini:ini.append(['player','player'])
        base_instruction=instruction(raw['domain'],raw['problem'],raw['vocabulary'],raw['notes'],g)
    oracle_domain=copy.deepcopy(d)
    ops=[copy.deepcopy(a) for a in d if isinstance(a,list) and a[:1]==[':action']]
    base_params={a[1]:list(fields(a)[':parameters']) for a in ops}
    objects=list(section(p,':objects')[1:]);constants=section(d,':constants',[':constants'])[1:]
    compile_policy(cfg,ops,ini,objects,raw)
    if g=='printer':
        task_supply=2+slot%3
        if cfg.get('scene_options'):
            task_supply=max(task_supply,sum(a[0]=='document' for a in ini))
        for a in ini:
            if a[0]=='paper_stock':a[1]=f'stock{task_supply}'
    policy_params={a[1]:list(fields(a)[':parameters']) for a in ops}
    # Every legacy parameter is explicitly guarded; guards follow the actual
    # role instead of imposing blanket inequalities on same-category values.
    ini=list(dict.fromkeys(tuple(a) for a in ini));ini=[list(a) for a in ini]
    task=dict(family=g,cfg=cfg,seed=seed,slot=slot,raw=raw,oracle_domain=oracle_domain,base_params=base_params,
              policy_params=policy_params,base_instruction=base_instruction,mapping=mapping,alignment_extra=extra,
              visual_source=visual,goal_atoms=copy.deepcopy(goal[1:]),alias=alias or g)
    if g=='printer':task['paper_supply']=task_supply
    if g=='lights_out':
        # A native scramble under this exact wiring defines a solvable initial
        # light pattern. Goal remains independently all-off/disconnected.
        ini=[a for a in ini if a[0]!='lit'];links={b:{a[2] for a in ini if a[:2]==['affects',b]} for b in [a[1] for a in ini if a[0]=='button']}
        rng=random.Random(seed+71);on=set()
        presses=cfg.get('scene_options',{}).get('scramble_presses',4)
        for b in rng.sample(sorted(links),min(presses,len(links))):on.symmetric_difference_update(links[b])
        ini += [['lit',l] for l in sorted(on)]
        rp=parse_sexpr(raw['problem']);ri=section(rp,':init');ri[:]=[':init',*[a for a in ri[1:] if a[0] not in {'lit','affects'}],*[a for a in ini if a[0] in {'lit','affects'}]];raw['problem']=sexpr(rp)
    if g in {'blocks','sokoban'}:
        # The new task goal is a short native walk's physical destination. This
        # construction precedes PDDL solving; it never changes a failed goal.
        state=set(map(tuple,ini));rng=random.Random(seed+194);seen={frozenset(state)};candidates=[]
        for k in range(cfg.get('scene_options',{}).get('goal_walk',10 if g=='blocks' else 18)):
            choices=successors(task,state,controller=False)
            choices={a:t for a,t in choices.items() if frozenset(t) not in seen}
            if not choices:break
            preferred={a:t for a,t in choices.items() if a[0] not in {'move'}} or choices
            op,state=rng.choice(sorted(preferred.items(),key=lambda x:x[0]));seen.add(frozenset(state))
            if g=='blocks' and not any(a[0]=='holding' for a in state):candidates.append(state)
            if g=='sokoban' and {a for a in state if a[0]=='at' and a[1]!='player'}!={tuple(a) for a in ini if a[0]=='at' and a[1]!='player'}:candidates.append(state)
        if not candidates:raise ValueError('native_goal_walk_no_nontrivial_destination')
        if g=='blocks' and cfg.get('scene_options',{}).get('goal_selection')=='farthest_reachable':
            from collections import deque
            state=set(map(tuple,ini));queue=deque([(state,0)]);seen={frozenset(state)}
            minimum=cfg['scene_options']['goal_minimum_distance'];candidates=[];depth=0
            while queue:
                state,depth=queue.popleft()
                if depth>=minimum and not any(a[0]=='holding' for a in state):
                    candidates.append(state)
                    if len(candidates)>=20:break
                    continue
                for _,after in sorted(successors(task,state,controller=False).items()):
                    key=frozenset(after)
                    if key not in seen:seen.add(key);queue.append((after,depth+1))
            if not candidates:raise ValueError('native_goal_walk_not_far_enough')
            terminal=rng.choice(candidates)
        elif cfg.get('scene_options',{}).get('goal_selection')=='max_changed':
            initial=set(map(tuple,ini))
            predicate='on' if g=='blocks' else 'at'
            def changed(st):
                return sum(a[0]==predicate and a not in initial and (g=='blocks' or a[1]!='player') for a in st)
            best=max(map(changed,candidates))
            terminal=rng.choice([st for st in candidates if changed(st)==best])
        else:
            terminal=rng.choice(candidates[-4:])
        if g=='blocks':goal=['and',*[list(a) for a in sorted(terminal) if a[0]=='on'],['hand_free','hand1']];raw['spec']['goal_on']={a[1]:a[2] for a in goal[1:] if a[0]=='on'};raw['spec']['clear_goal']=[];raw['spec']['goal_mode']='full'
        else:
            targets=sorted(a[2] for a in terminal if a[0]=='at' and a[1]!='player')
            ini=[a for a in ini if a[0] not in {'goal_cell','non_goal_cell','on_goal'}]
            ini += [['goal_cell',c] for c in targets]+[['non_goal_cell',c] for c in raw['scene']['cells'] if c not in targets]
            ini += [['on_goal',a[1]] for a in ini if a[0]=='at' and a[1]!='player' and a[2] in targets]
            goal=['and',*(["on_goal",a[1]] for a in ini if a[0]=='box')]
            raw['scene']['goals']=targets
            rp=parse_sexpr(raw['problem']);section(rp,':init')[:]=[':init',*[a for a in ini if a[0]!='player']];section(rp,':goal')[1]=goal;raw['problem']=sexpr(rp)
    task['goal_atoms']=copy.deepcopy(goal[1:])
    # Badge states vary independently of appearance and rule ID. Pair slots
    # use the same states across compatible configurations.
    if slot%3!=0:ini.append(['controller_ready'])
    if cfg['cooling'] and slot>=4 and slot%4==0:ini.append(['controller_busy'])
    ini.append(['controller_charge',f'q{[0,3,1,3,0][slot%5]}'])
    task['badge']=dict(ready=slot%3!=0,busy=bool(cfg['cooling'] and slot>=4 and slot%4==0),charge=[0,3,1,3,0][slot%5])
    compile_controller(cfg,ops,ini,objects)
    from .cleanup import guard_parameters, format_domain, format_problem
    guard_parameters(ops,ini,g)
    # Categories/objects are copied from the source; new static categories and
    # bookkeeping are explicit. All operators are retained, independent of plan.
    # Constants referenced inside an action must be domain constants; a
    # problem-only object declaration is insufficient for Fast Downward.
    constants=set(constants)
    def action_constants(n):
        if not isinstance(n,list) or not n:return
        if n[0] in {'forall','exists'}:action_constants(n[2])
        elif n[0] in {'and','or','not','imply','when'}:
            for child in n[1:]:action_constants(child)
        else:
            constants.update(x for x in n[1:] if isinstance(x,str) and not x.startswith('?'))
    for a in ops:
        f=fields(a);action_constants(f[':precondition']);action_constants(f[':effect'])
    constants=sorted(constants)
    objects=sorted(set(objects)-set(constants));ini=[list(a) for a in sorted(set(map(tuple,ini)))]
    dtext=make_domain(g,ops,ini,goal[1:],constants)
    ptext=make_problem(g,objects,ini,goal[1:])
    dt,pt=parse_sexpr(dtext),parse_sexpr(ptext)
    dtext=format_domain(dt,pt,[a[1] for a in ops])+'\n'
    ptext=format_problem(pt)+'\n'
    task.update(domain=dtext,problem=ptext,initial=set(map(tuple,ini)),vocabulary=vocab,
                interfaces={a[1]:list(fields(a)[':parameters']) for a in ops})
    task['instruction']=describe(task)
    from .cleanup import clean_task
    return clean_task(task)

def describe(task):
    cfg=task['cfg'];g=task['family'];pi=cfg['policy_index']
    text=task['base_instruction']
    text=re.sub(r'(?m)^Allowed operations: .*\n?', '',text)
    text=text.replace('Use visible IDs and operation names; choose predicate names while preserving the rules. Encode the depicted state and rule-implied bookkeeping; other facts are false. Unchanged facts persist.\n','')
    if g=='blocks':
        text=re.sub(r'^[^\n]*','Arrange direct supports: '+ '; '.join(f'{a[1]} on {a[2]}' for a in task['goal_atoms'] if a[0]=='on')+'. Finish with hand1 empty.',text)
        text=text.replace('Any block can support any other block; color, size and table position impose no restrictions.','Colour and drawn size impose no constraints beyond the load policy below.')
    if g=='hanoi':pass
    if g=='maze':
        text=text.replace('The player arrow shows heading.','The player arrow shows heading. Thick solid segments are walls. A door plaque is centered on the shared edge between its two adjacent cells; it blocks that edge until unlocked.')
        text=text.replace('Unlocking consumes the key and is permanent.', 'Doors stay unlocked after unlocking.')
        if cfg['policy'] in {'side_unlock','reusable_side_unlock'}:text=text.replace('face its adjacent matching door to unlock it','operate its adjacent matching door to unlock it')
    if g=='frozenlake':
        text=text.replace('Cracked ice collapses permanently when left.','Cracked ice follows the departure policy below.')
        if cfg['policy'] in {'arrival_records','reinforced_arrival'}:
            text=text.replace('A checkpoint is recorded only by record_checkpoint while standing on it, once only.','record_checkpoint records an unrecorded checkpoint while standing on it, after its required earlier checkpoints. Arrival recording follows the policy below.')
    if g=='sliding_puzzle':
        text=text.replace('Each move swaps one tile with the orthogonally adjacent blank; no diagonal moves or jumps.','The four cardinal move_blank operations swap one tile with the orthogonally adjacent blank. Additional moves, if any, are specified below; no unlisted move or jump is allowed.')
    if g=='sokoban':text=text.replace('Pulling and pushing multiple boxes are forbidden.','Additional box capabilities are specified below.')
    if g=='printer':
        text=text.replace('power-off resets calibration.','Calibration follows the policy below.').replace('Printing consumes calibration; recalibrate for each document.','')
        text += '\nThe portable feeder holds at most eight sheets; its remaining supply is shown in the image. Loading transfers exactly one sheet from that feeder to a mounted printer, with an empty hand; printer capacity is three. Loading requires a nonempty feeder and spare printer capacity. There is no paper-replenishing action. The feeder follows the operator.\n'
    if g=='kitchen':
        text=text.replace('and dirty the tool','and apply the station-cleanliness policy below').replace('and dirtying the tool','and applying the station-cleanliness policy below').replace('and dirty the station','and apply the station-cleanliness policy below')
        text=text.replace('Cooling a ready held plate at the rack does not dirty it.', 'Only a ready held plate whose dish requires cooling may use the cooling rack; cooling does not dirty it.')
    if g=='gripper':text=text.replace('each hold one ball','each hold at most one ball, subject to the payload/compatibility policy below')
    if g=='miconic':
        text=re.sub(r'Capacity: .*?passengers\.', '',text)
        text=text.replace('The lift may travel directly to any higher floor using up or any lower floor using down.','up travels to a higher floor and down to a lower floor, subject to the travel policy below.')
        text=text.replace(' nN means N free slots.', '')
        if cfg['policy'] in {'two_seats', 'two_seat_adjacent'}:
            text += '\nseatN denotes N free seats.\n'
    if g=='lights_out':text=text.replace('Pressing buttonR_C simultaneously flips lightR_C and its orthogonal neighbours using their previous states.','Pressing buttonR_C simultaneously flips the lights wired below, using their previous states.')
    if g=='floortile':
        text=text.replace('Move a robot one orthogonal tile into a clear tile, clearing its old tile.','Move a robot one orthogonal tile into an unoccupied tile permitted by the tile-access policy below, making its old tile unoccupied.')
        text=text.replace('Paint only the clear tile immediately above or below the robot using its current colour; painting permanently makes that tile unavailable for entry or repainting.','Paint only an unpainted unoccupied tile using the current colour, with brush reach and dry-paint access specified below. A painted tile cannot be repainted.')
    if g=='key_delivery':text=text.replace('this consumes the key and frees the hand.','key retention and hand release follow the policy below.')
    if g=='freecell':text=text.replace('of the opposite colour with rank exactly one greater (only cards of rank 2 or higher can stack)','according to the tableau policy below')
    if g=='freecell':
        text=text.replace(
            'A column card may stack on an exposed card according to the tableau policy below, go to an empty freecell, start an empty column, or go to its same-suit foundation in increasing rank.',
            'A column card may stack on an exposed card according to the tableau policy below, go to an empty freecell, or go to its same-suit foundation in increasing rank. An exposed column card may start an empty column only when it currently rests on another card; the card beneath becomes exposed. A column consisting of a single card cannot be relocated to another empty column.')
    if g=='briefcase':text=text.replace('Capacity is unlimited.','Capacity and latch rules are specified below.')
    if g=='cooperative_kitchen':text=text.replace('The single-slot pass1 accepts or releases one food to a chef on an adjacent floor cell.','A chef on a floor cell adjacent to pass1 may transfer one held food onto it or take one food with an empty hand; the counter policy below sets capacity and accepted food states.')
    if g=='color_sort':text=text.replace('either empty or topped by a nut of the same color','subject to the stacking policy below')
    if g=='rush_hour':text=text.replace('Each vehicle moves one cell along its long axis,','move_vehicle shifts one cell along the vehicle\'s long axis,')
    text += '\nRules: '+POLICY_TEXT[g][pi]+'\n'
    if g=='hanoi':text+='A disk belongs to the peg reached by following its support chain. Only the moved top disk changes peg.\n'
    # List only operations actually available in the complete rule, with names.
    work=[n for n in dict.fromkeys(WORK[g]) if n in task['interfaces']]
    controller_text=profile_rules(cfg)
    text+='Guarded operations: '+', '.join(work)+'. '+controller_text+'\n'
    if g=='freecell':text=text.replace('-b actions','Actions ending in _b')
    text+='All action effects use the pre-action state, then update together. Unaffected facts persist. Unshown facts are false. Read all dynamic initial states from the picture.\n'
    symbols={
        'frozenlake':'player names the player', 'sokoban':'player names the player',
        'maze':'player names the player; inventory names the key inventory',
        'package':'player names the player; inventory names the reusable-tool inventory',
        'printer':'player names the player; hand1 names the single hand',
        'kitchen':'player names the cook; hand1 names the single hand',
        'key_delivery':'player names the player; hand1 names the single hand',
        'gripper':'robby names the robot; left and right name its two grippers',
        'miconic':'lift names the elevator', 'termes':'robot names the builder',
        'briefcase':'briefcase names the carrying case',
    }
    if g in symbols:text+='Symbols: '+symbols[g]+'.\n'
    text+=operation_list(task['domain'])
    return text
