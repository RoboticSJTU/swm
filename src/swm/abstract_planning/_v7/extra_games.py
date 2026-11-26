"""Ten small visual planning games. Fixed Python models, no rule DSL or APIs.

Scene construction and target choice precede PDDL solving. Procedural physical
transitions live in extra_games_native and do not interpret these operators.
"""
from __future__ import annotations
import random,itertools,copy
from collections import defaultdict
from functools import lru_cache
from .model import action,make_domain,make_problem,parse_sexpr,section,fields,sexpr,operation_list
from .grid_games import cell, rc
from .policies import EXTRA_POLICIES, PREDICATE_MEANINGS

FAMILIES=('peg_solitaire','water_jugs','river_crossing','frogs_and_toads','knight_tour',
          'pancake_sorting','tower_of_london','queens','reversi_solitaire','flood_it')
WORK={
 'peg_solitaire':('jump_peg',),'water_jugs':('fill_jug','empty_jug','pour_jug','pour_one_liter'),
 'river_crossing':('cross_alone','cross_with_cargo','cross_with_two'),'frogs_and_toads':('slide_token','jump_token'),
 'knight_tour':('move_knight',),'pancake_sorting':('flip_prefix','flip_suffix'),
 'tower_of_london':('move_ball',),'queens':('move_queen',),
 'reversi_solitaire':('place_black',),'flood_it':('choose_color',)}
RULES={
 'peg_solitaire':'Jump one peg orthogonally over one adjacent peg into the empty hole immediately beyond. Remove only the jumped peg; the moving peg occupies the landing hole. No diagonal or longer jumps. Pegs are indistinguishable. Leave exactly one peg in the gold-ringed hole.',
 'water_jugs':'The tap is unlimited and the drain removes water. fill_jug fills one nonfull jug to its marked capacity; empty_jug empties a nonempty jug. pour_jug transfers between different jugs until the source empties or the receiver fills, whichever happens first; require a nonempty source and spare receiver capacity. Amounts are whole liters, nN means N liters. Jug capacities never change.',
 'river_crossing':'The farmer rows between the two banks, alone or with exactly one cargo item currently on the same bank. Farmer and cargo arrive together. Linked cargo pairs cannot be left together on a bank without the farmer; a crossing is forbidden if its departure bank would contain such a pair. The red links show all unsafe pairs; unlinked pairs are safe. The boat follows the farmer and is not a separate storage place.',
 'frogs_and_toads':'Each token follows its arrow: green frogs move right, purple toads move left. slide_token moves into the adjacent blank; jump_token jumps over exactly one opposite-kind token into the blank two positions away. No backward move or jump over the same kind. The jumped token stays. Tokens never share a position. p1 is leftmost.',
 'knight_tour':'Move the knight in an L: two rows and one column, or one row and two columns. Land only on a pale active square, never on a dark square or a square already marked visited. Jumping over intervening squares is allowed. A move marks its destination visited and preserves earlier marks. The knight square is already visited; small check marks indicate other visited squares. Visit every active square; the final knight position is unrestricted.',
 'pancake_sorting':'flip_prefix selects a depth dN, N at least two, and reverses the order of the top N pancakes in one operation. Pancakes below that depth stay. Size is the numeric suffix of each pancake ID. Position p1 is the top. Finish with pancake1 at p1, pancake2 at p2, and so on, smallest to largest top-to-bottom. No pancake may be moved individually.',
 'tower_of_london':'Move exactly one top ball from one peg to the next free slot on a different peg. Require spare capacity on the destination, as marked above that peg. Other balls stay; all balls have equal load size and colors impose no placement restriction. Slots count bottom-to-top from s1. Pegs and capacities are fixed. Match only the target stacks listed in the goal.',
 'queens':'Each queen stays in its pictured row. move_queen moves it to any other column in that row; intermediate squares do not matter. Attacking positions are allowed during rearrangement. Two queens attack if they share a column or diagonal; row attacks cannot occur because each row has one queen. Finish with no attacking pair. The final columns are unrestricted.',
 'reversi_solitaire':'This is a one-player black-only Reversi puzzle; there are no opponent turns. Place a black disk on an empty square only if at least one straight row, column or diagonal from it contains one or more consecutive white disks followed by a black disk. The operation simultaneously turns all white disks in every such bounded run black, using the pre-action board. Existing black disks stay. Remove all white disks; empty squares need not be filled.',
 'flood_it':"Initially, the squares of the top-left square's color connected to it horizontally or vertically form the controlled region. choose_color selects a different color: recolor the entire controlled region, then automatically absorb every orthogonally connected region of that color. Other squares keep their colors; absorbed squares never leave. Finish with the whole board in one controlled region. Color letters identify hues as well as the palette; diagonals do not connect.",
}
RATIONALE={g:'A declared small-board version of the documented game; all changing quantities, pieces and access geometry are visible. The shared controller gates complete public operations without adding game-specific hidden state.' for g in FAMILIES}

def extra_controller_rules(cfg):
 """Only the controller conditions relevant to these board operations."""
 clauses=['The bottom badge shows ARM enabled/disabled, DRIVE energy (0..3), and COOL busy/idle. Ignore indicators not used below.']
 if cfg['preparation']!='none':
  clauses.append('Guarded operations require ARM enabled. prepare_controller requires disabled ARM and idle COOL, then enables ARM.')
  clauses.append('Each guarded operation disables ARM.' if cfg['preparation']=='per_work' else 'Guarded operations preserve ARM.')
 if cfg['energy']:
  clauses.append('Each guarded operation requires and consumes one DRIVE unit. recharge_controller requires idle COOL and DRIVE below 3, then restores DRIVE to 3 from an external supply.')
 if cfg['cooling']:
  clauses.append('Guarded operations require idle COOL and make it busy. wait_for_controller requires busy COOL and restores idle; it needs no energy or enabled ARM.')
 clauses.append('Controller operations preserve the board and every indicator they do not explicitly change.')
 return ' '.join(clauses)

def atom(p,*a):return [p,*a]
def conj(xs):return ['and',*xs]
def negate(x):return ['not',x]
def ac(name,params,pre,eff):
 return action(name,' '.join(params),sexpr(conj(pre)),sexpr(conj(eff)))

@lru_cache(maxsize=4)
def frog_solution(per_side=3):
 from collections import deque
 frogs=tuple(f'f{i+1}' for i in range(per_side));toads=tuple(f't{i+1}' for i in range(per_side))
 start=(*frogs,None,*toads);end=(*toads,None,*frogs)
 queue=deque([start]);seen={start:None}
 while queue:
  state=queue.popleft()
  if state==end:break
  blank=state.index(None)
  for i,t in enumerate(state):
   if t is None:continue
   step=1 if t.startswith('f') else -1
   if blank-i==step or (blank-i==2*step and state[i+step] is not None and state[i+step][0]!=t[0]):
    nxt=list(state);nxt[blank]=t;nxt[i]=None;nxt=tuple(nxt)
    if nxt not in seen:seen[nxt]=state;queue.append(nxt)
 assert end in seen
 path=[];s=end
 while s is not None:path.append(s);s=seen[s]
 return list(reversed(path))

@lru_cache(maxsize=8)
def frog_scenes(per_side=3,minimum_distance=5):
 """All sufficiently nontrivial boards that can reach the fixed swap goal.

 Reverse real slides/jumps from the goal, rather than sampling just one
 demonstration.  Same-kind tokens never pass each other in either direction.
 """
 from collections import deque
 end=tuple(frog_solution(per_side)[-1]);queue=deque([end]);distance={end:0}
 while queue:
  board=queue.popleft();blank=board.index(None)
  for i,token in enumerate(board):
   if token is None:continue
   step=1 if token.startswith('f') else -1
   if i-blank==step or (i-blank==2*step and board[blank+step] is not None and board[blank+step][0]!=token[0]):
    previous=list(board);previous[i]=None;previous[blank]=token;previous=tuple(previous)
    if previous not in distance:
     distance[previous]=distance[board]+1;queue.append(previous)
 return tuple(board for board,d in distance.items() if d>=minimum_distance)

def scene(family,seed,scene_options=None,policy='ordinary'):
 options=scene_options or {};rng=random.Random(seed);spec={};state={};target={}
 if family=='peg_solitaire':
  h,w=options.get('board',(5,5));holes=[cell(r,c) for r in range(h) for c in range(w)];lines=[]
  for p in holes:
   r,c=rc(p)
   directions=[(1,0),(-1,0),(0,1),(0,-1)]
   if policy=='diagonal_jumps':directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
   for dr,dc in directions:
    a,b=cell(r+dr,c+dc),cell(r+2*dr,c+2*dc)
    if a in holes and b in holes:lines.append((p,a,b))
  final=rng.choice(holes);occupied={final}
  for _ in range(options.get('reverse_steps',rng.randrange(3,7))):
   options=[(a,b,c) for a,b,c in lines if c in occupied and a not in occupied and b not in occupied]
   if not options:break
   a,b,c=rng.choice(options);occupied.remove(c);occupied.update([a,b])
  if len(occupied)<3:raise ValueError('peg_reverse_construction_too_short')
  spec=dict(h=h,w=w,holes=holes,lines=lines);state=dict(occupied=sorted(occupied));target=dict(hole=final)
 elif family=='water_jugs':
  values=options.get('capacities',rng.sample(range(3,8),3));capacities={f'jug{i+1}':n for i,n in enumerate(values)};amounts={j:rng.randrange(n+1) for j,n in capacities.items()}
  if 'initial_amounts' in options:amounts=dict(zip(capacities,options['initial_amounts']))
  chosen=rng.choice(list(capacities));quantity=rng.randrange(1,capacities[chosen])
  if amounts[chosen]==quantity:amounts[chosen]=0
  spec=dict(capacities=capacities);state=dict(amounts=amounts);target=dict(jug=chosen,liters=quantity)
  if options.get('goal_minimum_distance'):
   from collections import deque
   spec['policy']=policy;jugs=list(capacities)
   def key(s):return tuple(s['amounts'][j] for j in jugs)
   queue=deque([(state,0)]);seen={key(state)};goal_distance={}
   while queue:
    current,depth=queue.popleft()
    for j,n in current['amounts'].items():
     if 0<n<capacities[j]:goal_distance.setdefault((j,n),depth)
    for _,nxt in sorted(physical_moves(family,spec,current).items()):
     k=key(nxt)
     if k not in seen:seen.add(k);queue.append((nxt,depth+1))
   candidates=[k for k,d in goal_distance.items() if d>=options['goal_minimum_distance']]
   if not candidates:raise ValueError('native_goal_walk_not_far_enough')
   chosen,quantity=rng.choice(sorted(candidates));target=dict(jug=chosen,liters=quantity)
 elif family=='river_crossing':
  cargo=['wolf','goat','cabbage']+[f'sack{i+1}' for i in range(options.get('cargo_count',5)-3)];unsafe=[('wolf','goat'),('goat','cabbage')]
  if 'unsafe_pairs' in options:
   unsafe=[tuple(pair) for pair in options['unsafe_pairs']]
   assert all(a!=b and a in cargo and b in cargo for a,b in unsafe)
  # Draw from the true reachable graph; do not invent a safe-looking dead end.
  start=(0,)*(len(cargo)+1);queue=[start];reachable={start}
  for st in queue:
   farmer=st[0]
   for item in [None]+[i for i in range(len(cargo)) if st[i+1]==farmer]:
    nxt=list(st);nxt[0]=1-farmer
    if item is not None:nxt[item+1]=1-farmer
    if any(nxt[cargo.index(a)+1]==nxt[cargo.index(b)+1]==farmer for a,b in unsafe):continue
    nxt=tuple(nxt)
    if nxt not in reachable:reachable.add(nxt);queue.append(nxt)
  st=start if options.get('initial_layout')=='all_left' else rng.choice(sorted(x for x in reachable if x!=(1,)*(len(cargo)+1)))
  spec=dict(cargo=cargo,unsafe=unsafe);state=dict(farmer='right' if st[0] else 'left',cargo={c:'right' if v else 'left' for c,v in zip(cargo,st[1:])});target=dict(bank='right')
 elif family=='frogs_and_toads':
  count=options.get('per_side',3);distance=options.get('minimum_distance',5)
  st=rng.choice(frog_scenes(count,distance));spec=dict(tokens=[f'{kind}{i+1}' for kind in ['f','t'] for i in range(count)],positions=[f'p{i}' for i in range(1,2*count+2)])
  state=dict(board=list(st));target=dict(board=list(frog_solution(count)[-1]))
 elif family=='knight_tour':
  h,w=options.get('board',(4,4));cells=[cell(r,c) for r in range(h) for c in range(w)];path=[rng.choice(cells)]
  for _ in range(options.get('path_length',rng.randrange(7,12))-int(bool(options.get('path_length')))):
   r,c=rc(path[-1]);options=[p for p in cells if p not in path and sorted(abs(a-b) for a,b in zip((r,c),rc(p)))==[1,2]]
   if not options:break
   path.append(rng.choice(options))
  if len(path)<6:raise ValueError('knight_path_too_small')
  prefix=rng.randrange(min(3,len(path)-3));spec=dict(h=h,w=w,active=sorted(path));state=dict(knight=path[prefix],visited=path[:prefix+1]);target=dict(visit_all=True)
 elif family=='pancake_sorting':
  n=options.get('count',rng.choice([4,5]));stack=[f'pancake{i}' for i in range(1,n+1)];rng.shuffle(stack)
  if stack==sorted(stack):stack.reverse()
  spec=dict(count=n);state=dict(stack=stack);target=dict(stack=[f'pancake{i}' for i in range(1,n+1)])
 elif family=='tower_of_london':
  pegs=['peg1','peg2','peg3'];capacities=dict(zip(pegs,rng.sample([2,3,4],3)))
  if options.get('minimum_capacity'):
   capacities={p:max(n,options['minimum_capacity']) for p,n in capacities.items()}
  def deal():
   piles={p:[] for p in pegs};balls=[f'ball{i+1}' for i in range(options.get('balls',4))];rng.shuffle(balls)
   for b in balls:piles[rng.choice([p for p in pegs if len(piles[p])<capacities[p]])].append(b)
   return piles
  state=dict(piles=deal());target=dict(piles=deal());spec=dict(capacities=capacities,balls=[f'ball{i+1}' for i in range(options.get('balls',4))])
  if options.get('goal_walk'):
   # Larger adjacent-peg layouts can have disconnected reachability components.
   # Choose a target by legal physical moves before any PDDL is solved.
   spec['policy']=policy;current=copy.deepcopy(state)
   def key(s):return tuple(tuple(s['piles'][p]) for p in pegs)
   def positions(s):return {(b,p,i) for p in pegs for i,b in enumerate(s['piles'][p])}
   initial_positions=positions(state);seen={key(state)};destinations=[]
   if options.get('goal_selection')=='farthest_reachable':
    from collections import deque
    queue=deque([(state,0)]);farthest=0
    while queue and len(seen)<8000:
     current,depth=queue.popleft()
     for _,nxt in sorted(physical_moves(family,spec,current).items()):
      k=key(nxt)
      if k in seen:continue
      seen.add(k);queue.append((nxt,depth+1))
      if depth+1>farthest:farthest=depth+1;destinations=[]
      if depth+1==farthest:destinations.append(nxt)
     if farthest>=options.get('goal_minimum_distance',1):break
    if farthest<options.get('goal_minimum_distance',1):raise ValueError('native_goal_walk_not_far_enough')
    target=rng.choice(destinations)
   else:
    for _ in range(options['goal_walk']):
     choices=[s for _,s in sorted(physical_moves(family,spec,current).items()) if key(s) not in seen]
     if not choices:break
     current=rng.choice(choices);seen.add(key(current));destinations.append(current)
    if not destinations:raise ValueError('native_goal_walk_no_nontrivial_destination')
    distance=max(len(positions(s)-initial_positions) for s in destinations)
    target=rng.choice([s for s in destinations if len(positions(s)-initial_positions)==distance])
  if state==target:raise ValueError('initial_goal_already_complete')
 elif family=='queens':
  n=options.get('count',rng.choice([4,5]));spec=dict(h=n,w=n,queens=[f'q{i:02d}' for i in range(1,n+1)])
  state=dict(positions={q:cell(i,rng.randrange(n)) for i,q in enumerate(spec['queens'])});target=dict(nonattacking=True)
 elif family=='reversi_solitaire':
  h,w=options.get('board',(4,4));cells=[cell(r,c) for r in range(h) for c in range(w)]
  spec=dict(h=h,w=w);black=set(rng.sample(cells,options.get('black',rng.randrange(4,8))));white=set(rng.sample([p for p in cells if p not in black],options.get('white',rng.randrange(2,5))))
  state=dict(black=sorted(black),white=sorted(white));target=dict(no_white=True)
 elif family=='flood_it':
  hues=['red','blue','green','yellow'];colors={}
  h,w=options.get('board',(3,4))
  for r in range(h):
   for c in range(w):colors[cell(r,c)]=rng.choice(hues)
  flooded={cell(0,0)};hue=colors[cell(0,0)]
  while True:
   additions={p for p,h in colors.items() if h==hue and p not in flooded and any((max(abs(a-b) for a,b in zip(rc(p),rc(q)))==1 if policy=='eight_neighbors' else sum(abs(a-b) for a,b in zip(rc(p),rc(q)))==1) for q in flooded)}
   if not additions:break
   flooded.update(additions)
  spec=dict(h=h,w=w,original_colors=colors);state=dict(flooded=sorted(flooded),color=hue);target=dict(all_flooded=True)
 if policy!='ordinary':spec['policy']=policy
 return dict(spec=spec,state=state,target=target)

def model(family,raw):
 spec,s,target=raw['spec'],raw['state'],raw['target'];ini=[];goals=[];ops=[];objects=[];constants=[];policy=spec.get('policy','ordinary')
 def category(name,values):
  objects.extend(values);ini.extend(atom(name,x) for x in values)
 if family=='peg_solitaire':
  category('hole',spec['holes']);ini+= [atom('jump_line',*line) for line in spec['lines']]+[atom('occupied',p) for p in s['occupied']]
  goals=[atom('occupied',target['hole'])]+[negate(atom('occupied',p)) for p in spec['holes'] if p!=target['hole']]
  ops=[ac('jump_peg',['?from','?over','?to'],[atom('hole',p) for p in ['?from','?over','?to']]+[atom('jump_line','?from','?over','?to'),atom('occupied','?from'),atom('occupied','?over'),negate(atom('occupied','?to'))],[negate(atom('occupied','?from')),atom('occupied','?to'),negate(atom('occupied','?over'))])]
 elif family=='water_jugs':
  category('jug',spec['capacities']);category('amount_value',[f'n{i}' for i in range(max(7,max(spec['capacities'].values()))+1)]);constants=['n0']
  for j,n in spec['capacities'].items():ini += [atom('capacity',j,f'n{n}'),atom('amount',j,f"n{s['amounts'][j]}")]
  for a,b in itertools.permutations(spec['capacities'],2):
   for av in range(spec['capacities'][a]+1):
    for bv in range(spec['capacities'][b]):
     n=min(av,spec['capacities'][b]-bv)
     if n:ini.append(atom('pour_result',a,b,f'n{av}',f'n{bv}',f'n{av-n}',f'n{bv+n}'))
     if av and policy=='metered_pour':ini.append(atom('unit_pour_result',a,b,f'n{av}',f'n{bv}',f'n{av-1}',f'n{bv+1}'))
  common=[atom('jug','?j'),atom('amount_value','?old'),atom('amount','?j','?old')]
  ops=[ac('fill_jug',['?j','?old','?full'],common+[atom('amount_value','?full'),atom('capacity','?j','?full'),negate(atom('=','?old','?full'))],[negate(atom('amount','?j','?old')),atom('amount','?j','?full')]),
       ac('empty_jug',['?j','?old'],common+[negate(atom('=','?old','n0'))],[negate(atom('amount','?j','?old')),atom('amount','?j','n0')]),
       ac('pour_jug',['?source','?target','?a','?b','?na','?nb'],[atom('jug','?source'),atom('jug','?target')]+[atom('amount_value',x) for x in ['?a','?b','?na','?nb']]+[atom('amount','?source','?a'),atom('amount','?target','?b'),atom('pour_result','?source','?target','?a','?b','?na','?nb')],[negate(atom('amount','?source','?a')),atom('amount','?source','?na'),negate(atom('amount','?target','?b')),atom('amount','?target','?nb')])]
  if policy=='metered_pour':
   unit=copy.deepcopy(ops[-1]);unit[1]='pour_one_liter'
   pre=unit[unit.index(':precondition')+1]
   for clause in pre[1:]:
    if clause[0]=='pour_result':clause[0]='unit_pour_result'
   ops.append(unit)
  goals=[atom('amount',target['jug'],f"n{target['liters']}")]
 elif family=='river_crossing':
  category('bank',['left','right']);category('cargo',spec['cargo']);objects.append('farmer');constants=['farmer','right']
  ini += [atom('opposite','left','right'),atom('opposite','right','left'),atom('at','farmer',s['farmer'])]+[atom('at',c,b) for c,b in s['cargo'].items()]+[atom('unsafe_pair',a,b) for a,b in spec['unsafe']]
  common=[atom('bank','?from'),atom('bank','?to'),atom('opposite','?from','?to'),atom('at','farmer','?from')]
  unsafe=conj([atom('unsafe_pair','?a','?b'),atom('at','?a','?from'),atom('at','?b','?from')])
  safe=['forall',['?a','?b'],['not',unsafe]]
  safe_cargo=['forall',['?a','?b'],['imply',unsafe,['or',atom('=','?a','?cargo'),atom('=','?b','?cargo')]]]
  eff=[negate(atom('at','farmer','?from')),atom('at','farmer','?to')]
  ops=[ac('cross_alone',['?from','?to'],common+[safe],eff),ac('cross_with_cargo',['?cargo','?from','?to'],[atom('cargo','?cargo')]+common+[atom('at','?cargo','?from'),safe_cargo],eff+[negate(atom('at','?cargo','?from')),atom('at','?cargo','?to')])]
  if policy=='two_cargo_boat':
   safe_two=['forall',['?a','?b'],['imply',unsafe,['or',*[atom('=',p,c) for p in ['?a','?b'] for c in ['?first','?second']]]]]
   ops.append(ac('cross_with_two',['?first','?second','?from','?to'],[atom('cargo','?first'),atom('cargo','?second'),negate(atom('=','?first','?second'))]+common+[atom('at','?first','?from'),atom('at','?second','?from'),safe_two],eff+[e for cargo in ['?first','?second'] for e in [negate(atom('at',cargo,'?from')),atom('at',cargo,'?to')]]))
  goals=[atom('at',c,'right') for c in ['farmer']+spec['cargo']]
 elif family=='frogs_and_toads':
  category('token',spec['tokens']);category('position',spec['positions'])
  for i,t in enumerate(s['board']):ini.append(atom('at',t,f'p{i+1}') if t else atom('empty',f'p{i+1}'))
  for t in spec['tokens']:
   step=1 if t.startswith('f') else -1
   for i in range(1,len(spec['positions'])+1):
    if 1<=i+step<=len(spec['positions']):ini.append(atom('step',t,f'p{i}',f'p{i+step}'))
    if 1<=i+2*step<=len(spec['positions']):ini.append(atom('jump',t,f'p{i}',f'p{i+step}',f'p{i+2*step}'))
   ini += [atom('opposite_kind',t,u) for u in spec['tokens'] if t[0]!=u[0]]
  pre=[atom('token','?token'),atom('position','?from'),atom('position','?to'),atom('at','?token','?from'),atom('empty','?to')]
  eff=[negate(atom('at','?token','?from')),atom('at','?token','?to'),negate(atom('empty','?to')),atom('empty','?from')]
  ops=[ac('slide_token',['?token','?from','?to'],pre+[atom('step','?token','?from','?to')],eff),ac('jump_token',['?token','?other','?from','?over','?to'],pre+[atom('token','?other'),atom('position','?over'),atom('at','?other','?over'),atom('opposite_kind','?token','?other'),atom('jump','?token','?from','?over','?to')],eff)]
  if policy=='any_kind_jump':
   condition=ops[-1][ops[-1].index(':precondition')+1]
   condition[:]=[clause for clause in condition if not(isinstance(clause,list) and clause[0]=='opposite_kind')]
  goals=[atom('at',t,f'p{i+1}') if t else atom('empty',f'p{i+1}') for i,t in enumerate(target['board'])]
 elif family=='knight_tour':
  category('square',spec['active']);objects.append('knight');constants=['knight']
  ini += [atom('at','knight',s['knight'])]+[atom('visited',p) for p in s['visited']]
  ini += [atom('knight_link',a,b) for a,b in itertools.permutations(spec['active'],2) if sorted(abs(x-y) for x,y in zip(rc(a),rc(b)))==[1,2]]
  ops=[ac('move_knight',['?from','?to'],[atom('square','?from'),atom('square','?to'),atom('at','knight','?from'),atom('knight_link','?from','?to'),negate(atom('visited','?to'))],[negate(atom('at','knight','?from')),atom('at','knight','?to'),atom('visited','?to')])]
  if policy=='revisiting_walk':ops[0][ops[0].index(':precondition')+1].remove(negate(atom('visited','?to')))
  goals=[atom('visited',p) for p in spec['active']]
 elif family=='pancake_sorting':
  n=spec['count'];category('pancake',target['stack']);category('position',[f'p{i}' for i in range(1,n+1)]);category('depth',[f'd{i}' for i in range(2,n+1)])
  ini += [atom('at',t,f'p{i+1}') for i,t in enumerate(s['stack'])]
  ini += [atom('flip_index',f'd{k}',f'p{i}',f'p{k+1-i}') for k in range(2,n+1) for i in range(1,k+1)]
  ops=[ac('flip_prefix',['?depth'],[atom('depth','?depth')],[['forall',['?cake','?old','?new'],['when',conj([atom('at','?cake','?old'),atom('flip_index','?depth','?old','?new')]),conj([negate(atom('at','?cake','?old')),atom('at','?cake','?new')])]]])]
  if policy=='suffix_flips':
   ini += [atom('suffix_index',f'd{k}',f'p{i}',f'p{2*n-k+1-i}') for k in range(2,n+1) for i in range(n-k+1,n+1)]
   suffix=copy.deepcopy(ops[0]);suffix[1]='flip_suffix'
   def rename(node):
    if isinstance(node,list):return ['suffix_index' if x=='flip_index' else rename(x) for x in node]
    return node
   ops.append(rename(suffix))
  goals=[atom('at',t,f'p{i+1}') for i,t in enumerate(target['stack'])]
 elif family=='tower_of_london':
  category('peg',spec['capacities']);category('ball',spec['balls']);category('slot_value',[f's{i}' for i in range(5)])
  ini += [atom('slot_step',f's{i}',f's{i+1}') for i in range(4)]
  for peg,cap in spec['capacities'].items():
   ini += [atom('valid_level',peg,f's{i}') for i in range(cap+1)]+[atom('top',peg,f"s{len(s['piles'][peg])}")]
   ini += [atom('at',ball,peg,f's{i+1}') for i,ball in enumerate(s['piles'][peg])]
  pre=[atom('ball','?ball'),atom('peg','?source'),atom('peg','?target')]+[atom('slot_value',x) for x in ['?old','?target_old','?source_new','?target_new']]+[negate(atom('=','?source','?target')),atom('top','?source','?old'),atom('at','?ball','?source','?old'),atom('top','?target','?target_old'),atom('slot_step','?source_new','?old'),atom('slot_step','?target_old','?target_new'),atom('valid_level','?target','?target_new')]
  eff=[negate(atom('at','?ball','?source','?old')),atom('at','?ball','?target','?target_new'),negate(atom('top','?source','?old')),atom('top','?source','?source_new'),negate(atom('top','?target','?target_old')),atom('top','?target','?target_new')]
  ops=[ac('move_ball',['?ball','?source','?target','?old','?target_old','?source_new','?target_new'],pre,eff)]
  if policy=='adjacent_pegs':
   ini += [atom('peg_neighbor',a,b) for a,b in itertools.permutations(spec['capacities'],2) if abs(int(a[3:])-int(b[3:]))==1]
   ops[0][ops[0].index(':precondition')+1].append(atom('peg_neighbor','?source','?target'))
  goals=[atom('at',ball,p,f's{i+1}') for p,pile in target['piles'].items() for i,ball in enumerate(pile)]
 elif family=='queens':
  squares=[cell(r,c) for r in range(spec['h']) for c in range(spec['w'])];category('square',squares);category('queen',spec['queens'])
  ini += [atom('at',q,p) for q,p in s['positions'].items()]
  def threats(a,b):
   r,c=rc(a);x,y=rc(b);return c==y or abs(r-x)==abs(c-y)
  row_relation='row_step' if policy=='adjacent_columns' else 'same_row'
  ini += [atom(row_relation,a,b) for a,b in itertools.permutations(squares,2) if rc(a)[0]==rc(b)[0] and (policy!='adjacent_columns' or abs(rc(a)[1]-rc(b)[1])==1)]
  ini += [atom('threatens',a,b) for a,b in itertools.permutations(squares,2) if threats(a,b)]
  ini += [atom('attacks',a,b) for a,b in itertools.permutations(spec['queens'],2) if threats(s['positions'][a],s['positions'][b])]
  shared=[atom('queen','?other'),negate(atom('=','?q','?other')),atom('at','?other','?p')]
  set_attack=['forall',['?other','?p'],['when',conj(shared+[atom('threatens','?to','?p')]),conj([atom('attacks','?q','?other'),atom('attacks','?other','?q')])]]
  clear_attack=['forall',['?other','?p'],['when',conj(shared+[negate(atom('threatens','?to','?p'))]),conj([negate(atom('attacks','?q','?other')),negate(atom('attacks','?other','?q'))])]]
  ops=[ac('move_queen',['?q','?from','?to'],[atom('queen','?q'),atom('square','?from'),atom('square','?to'),atom('at','?q','?from'),atom(row_relation,'?from','?to')],[negate(atom('at','?q','?from')),atom('at','?q','?to'),set_attack,clear_attack])]
  goals=[negate(atom('attacks',a,b)) for a,b in itertools.permutations(spec['queens'],2)]
 elif family=='reversi_solitaire':
  squares=[cell(r,c) for r in range(spec['h']) for c in range(spec['w'])];category('square',squares)
  ini += [atom('black',p) for p in s['black']]+[atom('white',p) for p in s['white']]+[atom('empty',p) for p in squares if p not in s['black']+s['white']]
  for p in squares:
   r,c=rc(p)
   for dr,dc in itertools.product([-1,0,1],repeat=2):
    if dr==dc==0:continue
    if policy=='orthogonal_capture' and dr and dc:continue
    for n in range(1,max(spec['h'],spec['w'])-1):
     ray=[cell(r+dr*k,c+dc*k) for k in range(1,n+2)]
     if all(x in squares for x in ray):ini.append(atom(f'ray{n}',p,*ray))
  conditions=[];eff=[negate(atom('empty','?p')),atom('black','?p')]
  for n in range(1,max(spec['h'],spec['w'])-1):
   if n>=3:
    # Instantiate only actual static rays, rather than the full object product
    # of four or more quantified variables. Coordinates remain bound to ?p;
    # black/white guards still read the pre-action board, so all qualifying
    # directions flip together. No initial disk colors enter this expansion.
    constants=sorted(set(constants)|set(squares))
    for fact in ini:
     if fact[0]!=f'ray{n}':continue
     placement,*xs,end=fact[1:]
     condition=conj([atom('=','?p',placement),atom('black',end)]+[atom('white',x) for x in xs])
     conditions.append(condition)
     eff.append(['when',condition,conj([e for x in xs for e in [negate(atom('white',x)),atom('black',x)]])])
    continue
   xs=[f'?w{i}' for i in range(n)];variables=xs+['?end'];condition=conj([atom(f'ray{n}','?p',*variables),atom('black','?end')]+[atom('white',x) for x in xs])
   conditions.append(['exists',variables,condition]);eff.append(['forall',variables,['when',condition,conj([e for x in xs for e in [negate(atom('white',x)),atom('black',x)]])]])
  ops=[ac('place_black',['?p'],[atom('square','?p'),atom('empty','?p'),['or',*conditions]],eff)]
  goals=[negate(atom('white',p)) for p in squares]
 elif family=='flood_it':
  colors=spec['original_colors'];category('square',colors);category('color',['red','blue','green','yellow'])
  def neighbors(a,b):
   distance=[abs(x-y) for x,y in zip(rc(a),rc(b))]
   return max(distance)==1 if policy=='eight_neighbors' else sum(distance)==1
  ini += [atom('original_color',p,h) for p,h in colors.items()]+[atom('current_color',s['color'])]+[atom('flooded',p) for p in s['flooded']]
  ini += [atom('adjacent',a,b) for a,b in itertools.permutations(colors,2) if neighbors(a,b)]
  # A complete color choice absorbs whole original monochromatic components.
  # These static geometric facts come only from the visible board, not a plan.
  for p in colors:
   component={p}
   while True:
    additions={q for q,h in colors.items() if h==colors[p] and q not in component and any(neighbors(q,t) for t in component)}
    if not additions:break
    component.update(additions)
   boundary={q for q in colors if any(neighbors(q,t) for t in component)}
   ini += [atom('component_touches',p,q) for q in sorted(boundary)]
  condition=conj([atom('square','?p'),atom('original_color','?p','?new'),['exists',['?neighbor'],conj([atom('component_touches','?p','?neighbor'),atom('flooded','?neighbor')])]])
  ops=[ac('choose_color',['?new','?old'],[atom('color','?new'),atom('color','?old'),atom('current_color','?old'),negate(atom('=','?new','?old'))],[negate(atom('current_color','?old')),atom('current_color','?new'),['forall',['?p'],['when',condition,atom('flooded','?p')]]])]
  goals=[atom('flooded',p) for p in colors]
 return objects,ini,goals,ops,constants

def construct(cfg,seed,slot,alias=None):
 from .rules_v2 import compile_controller
 g=cfg['family'];raw=scene(g,seed,cfg.get('scene_options'),cfg['policy']);objects,ini,goals,ops,constants=model(g,raw)
 interfaces={a[1]:list(fields(a)[':parameters']) for a in ops}
 if slot%3:ini.append(atom('controller_ready'))
 busy=bool(cfg['cooling'] and slot>=4 and slot%4==0)
 if busy:ini.append(atom('controller_busy'))
 charge=[0,3,1,3,0][slot%5];ini.append(atom('controller_charge',f'q{charge}'))
 compile_controller(cfg,ops,ini,objects)
 if cfg['energy']:constants=sorted(set(constants)|{'q3'})
 from .cleanup import guard_parameters, format_domain, format_problem
 guard_parameters(ops,ini,g)
 initial=set(map(tuple,ini));objects=sorted(set(objects)-set(constants))
 domain=make_domain(g,ops,ini,goals,constants);problem=make_problem(g,objects,ini,goals)
 domain=format_domain(parse_sexpr(domain),parse_sexpr(problem),[a[1] for a in ops])+'\n';problem=format_problem(parse_sexpr(problem))+'\n'
 if g=='water_jugs':goaltext=f"Measure exactly {raw['target']['liters']} {'liter' if raw['target']['liters'] == 1 else 'liters'} in {raw['target']['jug']}; other jug amounts are unrestricted."
 elif g=='river_crossing':goaltext='Bring the farmer and every cargo item to the RIGHT bank.'
 elif g=='frogs_and_toads':goaltext='Target row, left-to-right: '+', '.join(token or 'blank' for token in raw['target']['board'])+'.'
 elif g=='tower_of_london':goaltext='Target stacks, bottom-to-top: '+'; '.join(p+': '+(', '.join(pile) if pile else 'empty') for p,pile in raw['target']['piles'].items())+'.'
 else:goaltext={'peg_solitaire':'Leave exactly one peg in the gold-ringed hole.','knight_tour':'Visit every pale active square.','pancake_sorting':'Sort the pancakes from smallest to largest, top-to-bottom.','queens':'Arrange all queens so that none attack another.','reversi_solitaire':'Turn every white disk black.','flood_it':'Flood the entire board from its top-left region.'}[g]
 rules=EXTRA_POLICIES[g][1] if cfg['policy']==EXTRA_POLICIES[g][0] else RULES[g]
 text=goaltext+'\nRules: '+rules+'\n'
 if g in {'peg_solitaire','knight_tour','queens','reversi_solitaire','flood_it'}:text+='Coordinates r01c01 start at top-left; rows increase down, columns right.\n'
 if g=='water_jugs':text+='The number on each jug is its current liters/capacity; exact amounts are also shown by one-liter ticks.\n'
 if g=='flood_it':text+='R=red, B=blue, G=green, Y=yellow.\n'
 text+='Guarded operations: '+', '.join(WORK[g])+'. '+extra_controller_rules(cfg)+'\nAll effects read the pre-action state and update together; other facts persist.\n'+operation_list(domain)
 task=dict(family=g,cfg=cfg,seed=seed,slot=slot,alias=alias or g,raw=raw,extra_game=True,
  domain=domain,problem=problem,instruction=text,initial=initial,goal_atoms=goals,interfaces={a[1]:list(fields(a)[':parameters']) for a in ops},
  base_params=interfaces,policy_params=interfaces,vocabulary=dict(PREDICATE_MEANINGS),mapping={},alignment_extra=[],visual_source=True,
  badge=dict(ready=slot%3!=0,busy=busy,charge=charge))
 from .cleanup import clean_task
 return clean_task(task)


def index(s):
 out=defaultdict(set)
 for p,*args in s:out[p].add(tuple(args))
 return out

def threatened(a,b):
 r,c=rc(a);x,y=rc(b);return c==y or abs(r-x)==abs(c-y)

def projection(game,spec,state):
 if game=='peg_solitaire':return {('occupied',p) for p in state['occupied']}
 if game=='water_jugs':return {('amount',j,f'n{n}') for j,n in state['amounts'].items()}
 if game=='river_crossing':return {('at','farmer',state['farmer'])}|{('at',c,b) for c,b in state['cargo'].items()}
 if game=='frogs_and_toads':return {('at',t,f'p{i+1}') if t else ('empty',f'p{i+1}') for i,t in enumerate(state['board'])}
 if game=='knight_tour':return {('at','knight',state['knight'])}|{('visited',p) for p in state['visited']}
 if game=='pancake_sorting':return {('at',t,f'p{i+1}') for i,t in enumerate(state['stack'])}
 if game=='tower_of_london':return {('at',b,p,f's{i+1}') for p,pile in state['piles'].items() for i,b in enumerate(pile)}|{('top',p,f's{len(pile)}') for p,pile in state['piles'].items()}
 if game=='queens':return {('at',q,p) for q,p in state['positions'].items()}|{('attacks',a,b) for a,b in itertools.permutations(spec['queens'],2) if threatened(state['positions'][a],state['positions'][b])}
 if game=='reversi_solitaire':
  black,white=set(state['black']),set(state['white'])
  return {('black',p) for p in black}|{('white',p) for p in white}|{('empty',cell(r,c)) for r in range(spec['h']) for c in range(spec['w']) if cell(r,c) not in black|white}
 if game=='flood_it':return {('flooded',p) for p in state['flooded']}|{('current_color',state['color'])}
 raise ValueError(game)

def physical(task,s):
 f=index(s);g=task['family'];spec=task['raw']['spec']
 if g=='peg_solitaire':return dict(occupied=sorted(p[0] for p in f['occupied']))
 if g=='water_jugs':return dict(amounts={j:int(n[1:]) for j,n in f['amount']})
 if g=='river_crossing':
  at=dict(f['at']);return dict(farmer=at['farmer'],cargo={c:at[c] for c in spec['cargo']})
 if g=='frogs_and_toads':
  board=[None]*len(spec['positions'])
  for t,p in f['at']:board[int(p[1:])-1]=t
  return dict(board=board)
 if g=='knight_tour':return dict(knight=dict(f['at'])['knight'],visited=sorted(x[0] for x in f['visited']))
 if g=='pancake_sorting':return dict(stack=[t for t,p in sorted(f['at'],key=lambda x:int(x[1][1:]))])
 if g=='tower_of_london':return dict(piles={p:[b for b,peg,i in sorted(f['at'],key=lambda x:int(x[2][1:])) if peg==p] for p in spec['capacities']})
 if g=='queens':return dict(positions=dict(f['at']))
 if g=='reversi_solitaire':return dict(black=sorted(p[0] for p in f['black']),white=sorted(p[0] for p in f['white']))
 if g=='flood_it':return dict(flooded=sorted(p[0] for p in f['flooded']),color=next(iter(f['current_color']))[0])
 raise ValueError(g)

def physical_moves(game,spec,state):
 out={};policy=spec.get('policy','ordinary')
 def emit(op,nxt):out[tuple(op)]=nxt
 if game=='peg_solitaire':
  occupied=set(state['occupied'])
  for origin in occupied:
   r,c=rc(origin)
   directions=[(1,0),(-1,0),(0,1),(0,-1)]
   if policy=='diagonal_jumps':directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
   for dr,dc in directions:
    over,to=cell(r+dr,c+dc),cell(r+2*dr,c+2*dc)
    if over in occupied and to in spec['holes'] and to not in occupied:emit(('jump_peg',origin,over,to),dict(occupied=sorted((occupied-{origin,over})|{to})))
 elif game=='water_jugs':
  amounts=state['amounts'];caps=spec['capacities']
  for j,n in amounts.items():
   if n<caps[j]:nxt=amounts|{j:caps[j]};emit(('fill_jug',j,f'n{n}',f'n{caps[j]}'),dict(amounts=nxt))
   if n>0:emit(('empty_jug',j,f'n{n}'),dict(amounts=amounts|{j:0}))
  for a,b in itertools.permutations(caps,2):
   moved=min(amounts[a],caps[b]-amounts[b])
   if moved:
    na,nb=amounts[a]-moved,amounts[b]+moved
    emit(('pour_jug',a,b,f'n{amounts[a]}',f'n{amounts[b]}',f'n{na}',f'n{nb}'),dict(amounts=amounts|{a:na,b:nb}))
    if policy=='metered_pour':
     emit(('pour_one_liter',a,b,f'n{amounts[a]}',f'n{amounts[b]}',f'n{amounts[a]-1}',f'n{amounts[b]+1}'),dict(amounts=amounts|{a:amounts[a]-1,b:amounts[b]+1}))
 elif game=='river_crossing':
  origin=state['farmer'];to='right' if origin=='left' else 'left'
  for cargo in [None]+[c for c,b in state['cargo'].items() if b==origin]:
   nxt=copy.deepcopy(state);nxt['farmer']=to
   if cargo:nxt['cargo'][cargo]=to
   if any(nxt['cargo'][a]==nxt['cargo'][b]==origin for a,b in spec['unsafe']):continue
   emit(('cross_alone',origin,to) if cargo is None else ('cross_with_cargo',cargo,origin,to),nxt)
  if policy=='two_cargo_boat':
   aboard=[cargo for cargo,bank in state['cargo'].items() if bank==origin]
   for first,second in itertools.permutations(aboard,2):
    nxt=copy.deepcopy(state);nxt['farmer']=to;nxt['cargo'].update({first:to,second:to})
    if any(nxt['cargo'][a]==nxt['cargo'][b]==origin for a,b in spec['unsafe']):continue
    emit(('cross_with_two',first,second,origin,to),nxt)
 elif game=='frogs_and_toads':
  board=state['board'];blank=board.index(None)
  for i,t in enumerate(board):
   if t is None:continue
   step=1 if t.startswith('f') else -1;distance=blank-i
   if distance==step or (distance==2*step and board[i+step] is not None and (policy=='any_kind_jump' or board[i+step][0]!=t[0])):
    nxt=list(board);nxt[i]=None;nxt[blank]=t
    emit(('slide_token',t,f'p{i+1}',f'p{blank+1}') if distance==step else ('jump_token',t,board[i+step],f'p{i+1}',f'p{i+step+1}',f'p{blank+1}'),dict(board=nxt))
 elif game=='knight_tour':
  old=state['knight'];r,c=rc(old);visited=set(state['visited'])
  for to in spec['active']:
   if (policy=='revisiting_walk' or to not in visited) and sorted(abs(a-b) for a,b in zip((r,c),rc(to)))==[1,2]:emit(('move_knight',old,to),dict(knight=to,visited=sorted(visited|{to})))
 elif game=='pancake_sorting':
  for depth in range(2,spec['count']+1):emit(('flip_prefix',f'd{depth}'),dict(stack=list(reversed(state['stack'][:depth]))+state['stack'][depth:]))
  if policy=='suffix_flips':
   for depth in range(2,spec['count']+1):emit(('flip_suffix',f'd{depth}'),dict(stack=state['stack'][:-depth]+list(reversed(state['stack'][-depth:]))))
 elif game=='tower_of_london':
  for a,b in itertools.permutations(spec['capacities'],2):
   if policy=='adjacent_pegs' and abs(int(a[3:])-int(b[3:]))!=1:continue
   left,right=state['piles'][a],state['piles'][b]
   if left and len(right)<spec['capacities'][b]:
    nxt=copy.deepcopy(state);ball=nxt['piles'][a].pop();nxt['piles'][b].append(ball)
    emit(('move_ball',ball,a,b,f's{len(left)}',f's{len(right)}',f's{len(left)-1}',f's{len(right)+1}'),nxt)
 elif game=='queens':
  for queen,origin in state['positions'].items():
   row,col=rc(origin)
   for column in range(spec['w']):
    if column!=col and (policy!='adjacent_columns' or abs(column-col)==1):
     to=cell(row,column);emit(('move_queen',queen,origin,to),dict(positions=state['positions']|{queen:to}))
 elif game=='reversi_solitaire':
  black,white=set(state['black']),set(state['white'])
  for row in range(spec['h']):
   for col in range(spec['w']):
    origin=cell(row,col)
    if origin in black|white:continue
    flips=set()
    for dr,dc in itertools.product([-1,0,1],repeat=2):
     if not(dr or dc):continue
     if policy=='orthogonal_capture' and dr and dc:continue
     trail=[];r,c=row+dr,col+dc
     while 0<=r<spec['h'] and 0<=c<spec['w']:
      p=cell(r,c)
      if p in white:trail.append(p)
      else:
       if p in black:flips.update(trail)
       break
      r+=dr;c+=dc
    if flips:emit(('place_black',origin),dict(black=sorted(black|flips|{origin}),white=sorted(white-flips)))
 elif game=='flood_it':
  colors=spec['original_colors'];current=state['color'];flooded=set(state['flooded'])
  def neighbors(a,b):
   row_gap,col_gap=abs(rc(a)[0]-rc(b)[0]),abs(rc(a)[1]-rc(b)[1])
   return max(row_gap,col_gap)==1 if policy=='eight_neighbors' else row_gap+col_gap==1
  for hue in ['red','blue','green','yellow']:
   if hue==current:continue
   joined=set(flooded)
   while True:
    additions={p for p,h in colors.items() if h==hue and p not in joined and any(neighbors(p,q) for q in joined)}
    if not additions:break
    joined.update(additions)
   emit(('choose_color',hue,current),dict(flooded=sorted(joined),color=hue))
 return out

def successors(task,s,controller=True):
 g=task['family'];cfg=task['cfg'];spec=task['raw']['spec'];state=physical(task,s)
 old=projection(g,spec,state);base={op:(s-old)|projection(g,spec,nxt) for op,nxt in physical_moves(g,spec,state).items()}
 if not controller:return base
 f=index(s);ready=() in f['controller_ready'];busy=() in f['controller_busy'];charge=next(iter(f['controller_charge']))[0];n=int(charge[1:]);out={}
 for op,nxt in base.items():
  if (cfg['preparation']!='none' and not ready) or (cfg['cooling'] and busy) or (cfg['energy'] and n==0):continue
  if cfg['preparation']=='per_work':nxt.discard(('controller_ready',))
  if cfg['cooling']:nxt.add(('controller_busy',))
  if cfg['energy']:
   nxt=(nxt-{('controller_charge',charge)})|{('controller_charge',f'q{n-1}')};op=(*op,charge,f'q{n-1}')
  out[op]=nxt
 if cfg['preparation']!='none' and not ready and not busy:out[('prepare_controller',)]=s|{('controller_ready',)}
 if cfg['energy'] and n<3 and not busy:out[('recharge_controller',charge)]=(s-{('controller_charge',charge)})|{('controller_charge','q3')}
 if cfg['cooling'] and busy:out[('wait_for_controller',)]=s-{('controller_busy',)}
 return out

def goal(task,s):
 g=task['family'];state=physical(task,s);target=task['raw']['target'];spec=task['raw']['spec']
 if g=='peg_solitaire':return set(state['occupied'])=={target['hole']}
 if g=='water_jugs':return state['amounts'][target['jug']]==target['liters']
 if g=='river_crossing':return state['farmer']=='right' and set(state['cargo'].values())=={'right'}
 if g=='frogs_and_toads':return state['board']==target['board']
 if g=='knight_tour':return set(state['visited'])==set(spec['active'])
 if g=='pancake_sorting':return state['stack']==target['stack']
 if g=='tower_of_london':return state['piles']==target['piles']
 if g=='queens':return all(not threatened(a,b) for a,b in itertools.combinations(state['positions'].values(),2))
 if g=='reversi_solitaire':return not state['white']
 if g=='flood_it':return set(state['flooded'])==set(spec['original_colors'])
 raise ValueError(g)

def invariant(task,s):
 g=task['family'];f=index(s);spec=task['raw']['spec'];state=physical(task,s)
 assert len(f['controller_charge'])==1 and next(iter(f['controller_charge']))[0] in {'q0','q1','q2','q3'}
 if g=='peg_solitaire':assert set(state['occupied'])<=set(spec['holes']) and state['occupied']
 elif g=='water_jugs':assert set(state['amounts'])==set(spec['capacities']) and all(0<=n<=spec['capacities'][j] for j,n in state['amounts'].items()) and len(f['amount'])==len(spec['capacities'])
 elif g=='river_crossing':assert len(f['at'])==len(spec['cargo'])+1 and all(b in {'left','right'} for b in [state['farmer'],*state['cargo'].values()]) and all(not(state['cargo'][a]==state['cargo'][b]!=state['farmer']) for a,b in spec['unsafe'])
 elif g=='frogs_and_toads':assert len(f['at'])==len(spec['tokens']) and len(f['empty'])==1 and sorted(t for t in state['board'] if t)==sorted(spec['tokens']) and len({p for t,p in f['at']})==len(spec['tokens'])
 elif g=='knight_tour':assert state['knight'] in state['visited'] and set(state['visited'])<=set(spec['active']) and len(f['at'])==1
 elif g=='pancake_sorting':assert len(f['at'])==spec['count'] and set(state['stack'])==set(task['raw']['target']['stack']) and len({p for t,p in f['at']})==spec['count']
 elif g=='tower_of_london':assert sorted(b for pile in state['piles'].values() for b in pile)==sorted(spec['balls']) and all(len(pile)<=spec['capacities'][p] for p,pile in state['piles'].items()) and f['top']=={(p,f's{len(pile)}') for p,pile in state['piles'].items()}
 elif g=='queens':assert len(f['at'])==spec['h'] and set(state['positions'])==set(spec['queens']) and all(rc(p)[0]==int(q[1:])-1 for q,p in state['positions'].items()) and f['attacks']=={(a,b) for a,b in itertools.permutations(spec['queens'],2) if threatened(state['positions'][a],state['positions'][b])}
 elif g=='reversi_solitaire':assert not(set(state['black'])&set(state['white'])) and len(f['black'])+len(f['white'])+len(f['empty'])==spec['h']*spec['w']
 elif g=='flood_it':
  flooded=set(state['flooded']);colors=spec['original_colors']
  def neighbors(a,b):
   dr,dc=abs(rc(a)[0]-rc(b)[0]),abs(rc(a)[1]-rc(b)[1])
   return max(dr,dc)==1 if spec.get('policy')=='eight_neighbors' else dr+dc==1
  assert cell(0,0) in flooded and flooded<=set(colors) and len(f['current_color'])==1
  connected={cell(0,0)}
  while True:
   additions={p for p in flooded-connected if any(neighbors(p,q) for q in connected)}
   if not additions:break
   connected.update(additions)
  assert connected==flooded
  assert not any(h==state['color'] and p not in flooded and any(neighbors(p,q) for q in flooded) for p,h in colors.items())
