"""Procedural game oracle. Never evaluates PDDL conditions or effects.

PDDL parameter lists are read solely to serialize a fixed action interface.
Physical conditions and updates below are ordinary Python game code.
"""
from __future__ import annotations
import copy
import re
from collections import defaultdict

from . import rule_audit, visual_games as vg, grid_games

def index(s):
    f=defaultdict(set)
    for p,*args in s:f[p].add(tuple(args))
    return f

def physical(task,s):
    f=index(s);g=task['family'];raw=task['raw']
    if g=='blocks':
        return dict(on=dict(f['on']),held=next((b for h,b in f['holding']),None))
    if g=='sliding_puzzle':return {c:next((t for t,p in f['at'] if p==c),None) for c in raw['spec']['goal_state']}
    if g=='rush_hour':return {v:next((a for x,a in f['at'] if x==v),None) for v in raw['spec']['cars']}
    if g=='color_sort':
        return {b:[n for n,p in sorted(f['at'],key=lambda a:int(a[1].split('s')[-1])) if p.startswith(b+'s')] for b in raw['spec']['bolts']}
    if g=='cooperative_kitchen':
        holding={h:next((x for k,x in f['holding'] if k==h),None) for h in raw['spec']['hands'].values()}
        places={}
        for food in raw['spec']['foods']:
            held=next((h for h,x in holding.items() if x==food),None)
            places[food]=['held',held] if held else next([rel,p] for rel in ('at','on') for x,p in f[rel] if x==food)
        plate_hand=next((h for h,x in holding.items() if x=='p1'),None)
        plate=['held',plate_hand] if plate_hand else ['at',next(p for x,p in f['at'] if x=='p1')]
        return dict(chefs={a:next(p for x,p in f['at'] if x==a) for a in raw['spec']['hands']},holding=holding,food_places=places,prepared=sorted(x[0] for x in f['prepared']),plate_place=plate,served=('p1',) in f['served'])
    raise ValueError(g)

def successors(task,s, controller=True):
    if task.get('extra_game'):
        from .extra_games import successors as extra_successors
        return extra_successors(task,s,controller)
    from .rules_v2 import WORK, reset_clause
    f=index(s);g=task['family'];pol=task['cfg']['policy'];cfg=task['cfg'];result={}
    def vals(p):return {a[0] for a in f[p] if len(a)==1}
    def loc(x):return next((b for a,b in f['at'] if a==x),None)
    def has(p,*args):return tuple(args) in f[p]
    def one(p):
        v=vals(p);assert len(v)==1,(g,p,v)
        return next(iter(v))
    def emit(name,args,adds=(),deletes=()):
        op=(name,*(args[p[1:]] for p in task['policy_params'][name]))
        result[op]=(s-set(deletes))|set(adds)
    def ordinary():
        if g in {'blocks','sliding_puzzle','rush_hour','color_sort','cooperative_kitchen'}:
            native=physical(task,s);raw=task['raw'];proj=vg.projection(raw,native)
            opts=vg.successors(raw,native)
            for a,t in opts.items():result[a]=(s-proj)|vg.projection(raw,t)
        else:result.update(rule_audit.successors(g,task['oracle_domain'],s,task['base_instruction']))
    ordinary()
    base=list(result.items());result.clear()
    for op,t in base:
        name=op[0];args=dict(zip(task['base_params'].get(name,()),op[1:]))
        # Do not use a schema's conditions to decide any of these guards.
        if g=='blocks' and name=='stack':
            if pol in {'ordered_support','ordered_two_high'} and int(args['?underob'][1:])<=int(args['?ob'][1:]):continue
            if pol in {'two_high','ordered_two_high'} and not has('on',args['?underob'],'table'):continue
        if g=='hanoi' and pol!='ordinary':
            parent=dict(f['on'])
            def root(x):
                while x in parent:x=parent[x]
                return x
            a,b=root(args['?disc']),root(args['?to'])
            if pol=='adjacent_pegs':allowed=abs(int(a[-1])-int(b[-1]))==1
            elif pol=='counterclockwise_pegs':allowed=(a,b) in {('peg1','peg3'),('peg3','peg2'),('peg2','peg1')}
            else:allowed=int(b[-1])==int(a[-1])%3+1
            if not allowed:continue
            t=(t-{('tower',args['?disc'],a)})|{('tower',args['?disc'],b)}
            op=(*op,a,b)
        if g=='frozenlake' and name.startswith('move_'):
            origin=args['?from'];target=args['?to']
            if pol in {'reinforced','reinforced_arrival'} and has('safe',origin):t.add(('safe',origin))
            if pol in {'arrival_records','reinforced_arrival'}:
                for m in vals('checkpoint'):
                    if loc(m)==target and all(has('marked',p) for p,n in f['before'] if n==m):t.add(('marked',m))
        if g=='maze' and pol in {'reusable_key','reusable_side_unlock'} and name=='unlock_door':t.add(('in',args['?key'],'inventory'))
        if g=='package':
            if pol=='tape_first' and name=='remove_clip' and has('taped',args['?p']):continue
            if pol=='clip_first' and name=='cut_tape' and has('clipped',args['?p']):continue
            if pol=='cutter_required' and name=='open_package' and not has('in','cutter','inventory'):continue
        if g=='printer':
            if name=='load_paper':
                stock=one('paper_stock');n=int(stock[5:])
                if n==0:continue
                after='stock'+str(n-1);t=(t-{('paper_stock',stock)})|{('paper_stock',after)};op=(*op,stock,after)
            if name=='print_document' and pol in {'print_resets','print_resets_power_retains'}:t.discard(('calibrated',args['?p']))
            if name=='power_off' and pol in {'power_retains','print_resets_power_retains'} and has('calibrated',args['?p']):t.add(('calibrated',args['?p']))
        if g=='kitchen' and name in WORK[g] and (pol=='self_rinsing' or (pol=='rinsing_sink' and name=='wash') or (pol=='processing_rinses' and not name.startswith('finish_'))):
            t.add(('clean',args['?tool']))
        if g=='gripper' and name=='pick':
            if pol in {'shared_load_limit','assigned_shared_load'} and f['holding']:continue
            if pol in {'assigned_grippers','assigned_shared_load'} and args['?gripper']!=('left' if int(re.search(r'\d+$',args['?obj'])[0])%2==0 else 'right'):continue
        if g=='miconic':
            if pol in {'adjacent_stops','two_seat_adjacent'} and name in {'up','down'} and abs(int(args['?f1'][1:])-int(args['?f2'][1:]))!=1:continue
            if pol in {'two_seats','two_seat_adjacent'} and name in {'board','depart'}:
                occupied=len(f['in']);before=2-occupied;after=before+(-1 if name=='board' else 1)
                if not 0<=after<=2:continue
                t=(t-{('seat_free',f'seat{before}')})|{('seat_free',f'seat{after}')};op=(*op,f'seat{before}',f'seat{after}')
        if g=='lights_out' and name=='press_button':
            b=args['?b'];r,c=map(int,b.removeprefix('button').split('_'))
            affected=[]
            for light in vals('light'):
                x,y=map(int,light.removeprefix('light').split('_'));dr,dc=abs(r-x),abs(c-y)
                if (dr+dc<=1 if pol=='ordinary' else dr==0 and dc<=1 if pol=='horizontal' else dc==0 and dr<=1 if pol=='vertical' else (dr==dc==0 or dr==dc==1)):affected.append(light)
            t=set(s)
            for light in affected:
                fact=('lit',light)
                if fact in s:t.remove(fact)
                else:t.add(fact)
        if g=='key_delivery' and name=='unlock':
            if pol=='key_laid_down':t.add(('at',args['?key'],args['?from']))
            if pol=='key_forwarded':t.add(('at',args['?key'],args['?to']))
            if pol=='key_retained':t.add(('holding',args['?h'],args['?key']));t.discard(('hand_free',args['?h']))
        if g=='briefcase':
            if pol in {'latched','two_slot_latch'} and ((name=='move' and has('case_open')) or (name in {'put_in','take_out'} and not has('case_open'))):continue
            if pol in {'two_slots','two_slot_latch'} and name in {'put_in','take_out'}:
                before=2-len(f['in']);after=before+(-1 if name=='put_in' else 1)
                if not 0<=after<=2:continue
                t=(t-{('case_free',f'slot{before}')})|{('case_free',f'slot{after}')};op=(*op,f'slot{before}',f'slot{after}')
        if g=='cooperative_kitchen':
            if pol in {'prepared_transfer','prepared_two_slots'} and name=='put_on_counter' and not has('prepared',args['?food']):continue
            if pol in {'two_counter_slots','prepared_two_slots'}:
                t={a for a in t if a[0]!='counter_free'}
                if name in {'put_on_counter','take_from_counter'}:
                    n=sum(p=='pass1' for x,p in f['on']);m=n+(1 if name=='put_on_counter' else -1)
                    t=(t-{('counter_load',f'counter{n}')})|{('counter_load',f'counter{m}')};op=(*op,f'counter{n}',f'counter{m}')
        if g=='rush_hour' and pol in {'parking_brakes','glide_brakes'} and name in {'move_vehicle','exit_vehicle'}:
            if has('brake_set',args['?vehicle']):continue
            t.add(('brake_set',args['?vehicle']))
        result[op]=t

    if g=='maze' and pol in {'side_unlock','reusable_side_unlock'}:
        p=loc('player')
        for src,dst,door in f['edge_door']:
            if src!=p or has('unlocked',door):continue
            heading=next((d for a,b,d in f['ahead'] if a==src and b==dst),None)
            if heading is None:continue
            for key in vals('key'):
                if has('in',key,'inventory') and has('fits',key,door):
                    emit('unlock_door',dict(door=door,key=key,**{'from':src,'to':dst,'d':heading}),[('unlocked',door)],[] if pol=='reusable_side_unlock' else [('in',key,'inventory')])
    if g=='sliding_puzzle' and pol in {'diagonal','pair_slide','diagonal_pair'}:
        blank=one('empty');r,c=grid_games.rc(blank);at={p:t for t,p in f['at']}
        if pol in {'diagonal','diagonal_pair'}:
            for p,tile in at.items():
                if all(abs(x-y)==1 for x,y in zip(grid_games.rc(p),(r,c))):emit('move_blank_diagonal',dict(tile=tile,**{'from':p,'to':blank}),[('at',tile,blank),('empty',p)],[('at',tile,p),('empty',blank)])
        if pol in {'pair_slide','diagonal_pair'}:
            for dr,dc in grid_games.DIRECTIONS.values():
                middle,end=grid_games.cell(r+dr,c+dc),grid_games.cell(r+2*dr,c+2*dc)
                if middle in at and end in at:
                    a,b=at[middle],at[end]
                    emit('slide_pair',dict(near=a,far=b,blank=blank,middle=middle,end=end),[('at',a,blank),('at',b,middle),('empty',end)],[('at',a,middle),('at',b,end),('empty',blank)])
    if g=='sokoban' and pol!='ordinary':
        player=loc('player');r,c=grid_games.rc(player);boxes={loc(b):b for b in vals('box')}
        for d,(dr,dc) in grid_games.DIRECTIONS.items():
            near=grid_games.cell(r+dr,c+dc);far=grid_games.cell(r+2*dr,c+2*dc);end=grid_games.cell(r+3*dr,c+3*dc);back=grid_games.cell(r-dr,c-dc)
            if near not in boxes:continue
            if pol in {'pulling','pulling_double_push'} and has('empty',back):
                box=boxes[near];adds=[('at','player',back),('at',box,player),('empty',near)];deletes=[('at','player',player),('at',box,near),('empty',back)]
                (adds if has('goal_cell',player) else deletes).append(('on_goal',box))
                emit('pull_box',dict(box=box,**{'from':player,'boxloc':near,'to':back,'d':d}),adds,deletes)
            if pol in {'double_push','pulling_double_push'} and far in boxes and has('empty',end):
                a,b=boxes[near],boxes[far];adds=[('at','player',near),('at',a,far),('at',b,end),('empty',player)];deletes=[('at','player',player),('at',a,near),('at',b,far),('empty',end)]
                for box,p in [(a,far),(b,end)]:(adds if has('goal_cell',p) else deletes).append(('on_goal',box))
                emit('push_two_boxes',dict(first=a,second=b,**{'from':player,'near':near,'far':far,'to':end,'d':d}),adds,deletes)
    if g=='floortile' and pol in {'walk_on_dry_paint','four_way_brush','dry_four_way'}:
        occupied={loc(r) for r in vals('robot')};painted={p for p,c in f['painted']}
        for robot in vals('robot'):
            origin=loc(robot);color=next(c for r,c in f['robot_has'] if r==robot)
            for target in vals('tile')-occupied:
                if not rule_audit.adjacent(origin,target):continue
                dr,dc=tuple(y-x for x,y in zip(rule_audit.coordinate(origin),rule_audit.coordinate(target)))
                name='up' if dr==1 else 'down' if dr==-1 else 'right' if dc==1 else 'left'
                if pol in {'walk_on_dry_paint','dry_four_way'}:
                    adds=[('at',robot,target),('vacant',origin)]+([] if origin in painted else [('clear',origin)])
                    deletes=[('at',robot,origin),('vacant',target),('clear',target)]
                    emit(name,dict(r=robot,x=origin,y=target),adds,deletes)
                if pol in {'four_way_brush','dry_four_way'} and dc and target not in painted:
                    emit('paint_'+name,dict(r=robot,x=origin,y=target,c=color),[('painted',target,color)],[('clear',target)])
    if g=='key_delivery' and pol=='key_retained':
        for hand,key in f['holding']:
            if key in vals('key'):emit('put_down_key',dict(key=key,room=loc('player'),h=hand),[('hand_free',hand),('at',key,loc('player'))],[('holding',hand,key)])
    if g=='termes' and pol!='ordinary':
        origin=loc('robot');heights=dict(f['height']);n=int(heights[origin][1:]);carry=has('has_block')
        for target,height in heights.items():
            if not rule_audit.adjacent(origin,target) or has('is_depot',target):continue
            h=int(height[1:])
            if pol in {'extended_placement','articulated_both'} and carry and h==n+1 and 'h'+str(h+1) in vals('numb'):
                emit('place_block_higher',dict(rpos=origin,bpos=target,robot_height='h'+str(n),previous=height,result='h'+str(h+1)),[('height',target,'h'+str(h+1))],[('height',target,height),('has_block',)])
            if pol in {'level_removal','articulated_both'} and not carry and h==n and h>0:
                emit('remove_block_level',dict(rpos=origin,bpos=target,previous=height,result='h'+str(h-1)),[('height',target,'h'+str(h-1)),('has_block',)],[('height',target,height)])
    if g=='freecell' and pol!='ordinary':
        # Reimplement card-to-card placement from rank/suit, never from canstack.
        for op in list(result):
            if op[0] in {'move','move_b','colfromfreecell'}:del result[op]
        cells=one('cellspace');cols=one('colspace');ranks={c:int(v[1:]) for c,v in f['value']};suits=dict(f['hassuit']);parent=dict(f['on'])
        for card in vals('clear')|vals('incell'):
            free=has('incell',card);bottom=has('bottomcol',card);old=parent.get(card)
            for new in vals('clear'):
                if ranks[card]<(1 if pol=='ace_tableau' else 2) or ranks[new]!=ranks[card]+1:continue
                if pol=='same_suit' and suits[card]!=suits[new]:continue
                if pol=='ace_tableau' and ((suits[card] in {'c','s'}) == (suits[new] in {'c','s'})):continue
                args=dict(card=card,oldcard=old,newcard=new,cells=cells,cols=cols)
                adds=[('on',card,new),('clear',card)];deletes=[('clear',new)]
                if free:
                    after='celln'+str(int(cells[5:])+1);args['ncells']=after;adds.append(('cellspace',after));deletes += [('incell',card),('cellspace',cells)]
                elif bottom:
                    after='coln'+str(int(cols[4:])+1);args['ncols']=after;adds.append(('colspace',after));deletes += [('bottomcol',card),('colspace',cols)]
                else:adds.append(('clear',old));deletes.append(('on',card,old))
                emit('colfromfreecell' if free else 'move_b' if bottom else 'move',args,adds,deletes)
    if g=='briefcase' and pol in {'latched','two_slot_latch'}:emit('close_case' if has('case_open') else 'open_case',{},[] if has('case_open') else [('case_open',)],[('case_open',)] if has('case_open') else [])
    if g=='cooperative_kitchen' and pol in {'two_counter_slots','prepared_two_slots'}:
        n=sum(p=='pass1' for x,p in f['on'])
        if n==1:
            for chef,hand in task['raw']['spec']['hands'].items():
                p=loc(chef)
                if not vg.adjacent(p,task['raw']['spec']['counter']):continue
                for h,food in f['holding']:
                    if h==hand and food in vals('food'):
                        if pol=='prepared_two_slots' and not has('prepared',food):continue
                        emit('put_on_counter',dict(food=food,counter='pass1',loc=p,hand=hand,chef=chef,counter_previous='counter1',counter_result='counter2'),[('hand_free',hand),('on',food,'pass1'),('counter_load','counter2')],[('holding',hand,food),('counter_load','counter1')])
    if g=='rush_hour' and pol in {'parking_brakes','glide','glide_brakes'}:
        native=physical(task,s);spec=task['raw']['spec'];occupied={p for v,a in native.items() if a for p in vg.footprint(spec['cars'][v],a)}
        for v,anchor in native.items():
            if anchor is None:continue
            if pol in {'parking_brakes','glide_brakes'}:
                if has('brake_set',v):emit('release_brake',dict(vehicle=v),[],[('brake_set',v)])
            if pol in {'glide','glide_brakes'} and not (pol=='glide_brakes' and has('brake_set',v)):
                car=spec['cars'][v];r,c=vg.rc(anchor);own=set(vg.footprint(car,anchor))
                for sign in [-1,1]:
                    for k in range(1,max(spec['h'],spec['w'])):
                        target=vg.cell(r+k*sign*(car['axis']=='vertical'),c+k*sign*(car['axis']=='horizontal'));span=set(vg.footprint(car,target))
                        if not all(0<=vg.rc(p)[0]<spec['h'] and 0<=vg.rc(p)[1]<spec['w'] for p in span) or (span-own)&occupied:break
                        adds=[('at',v,target),*(('empty',p) for p in own-span)]
                        if pol=='glide_brakes':adds.append(('brake_set',v))
                        emit('slide_vehicle',dict(vehicle=v,**{'from':anchor,'to':target}),adds,[('at',v,anchor),*(('empty',p) for p in span-own)])
    if g=='color_sort' and pol!='ordinary':
        native=physical(task,s);spec=task['raw']['spec'];select=min if pol=='lowest_buffer' else max
        buffer=select(native,key=lambda x:int(x[1:]))
        for a,pile in native.items():
            if not pile:continue
            for b,target in native.items():
                if a==b or not target or len(target)>=spec['capacity']:continue
                nut,support=pile[-1],target[-1]
                if pol in {'mixed_buffer','lowest_buffer'} and b!=buffer and spec['colors'][nut]!=spec['colors'][support]:continue
                args=dict(nut=nut,support=support,source=a,target=b,**{'from':vg.slot(a,len(pile)),'source_after':vg.slot(a,len(pile)-1),'target_before':vg.slot(b,len(target)),'to':vg.slot(b,len(target)+1)})
                emit('move_onto',args,[('at',nut,args['to']),('in',nut,b),('top',a,args['source_after']),('top',b,args['to'])],[('at',nut,args['from']),('in',nut,a),('top',a,args['from']),('top',b,args['target_before'])])
    if not controller:return result
    final={};ready=has('controller_ready');busy=has('controller_busy');charge=one('controller_charge');n=int(charge[1:])
    for op,t in result.items():
        name=op[0];guard=name in WORK[g]
        if guard and ((cfg['preparation']!='none' and not ready) or (cfg['cooling'] and busy) or (cfg['energy'] and n==0)):continue
        if guard and cfg['preparation']=='per_work':t.discard(('controller_ready',))
        if guard and cfg['cooling']:t.add(('controller_busy',))
        if guard and cfg['energy']:
            t=(t-{('controller_charge',charge)})|{('controller_charge','q'+str(n-1))};op=(*op,charge,'q'+str(n-1))
        if cfg['preparation']=='after_motion':
            reset=reset_clause(g,name)
            reset_now=reset is not None
            args=dict(zip(task['policy_params'][name],op[1:]))
            if g=='hanoi':reset_now=name=='move' and args['?to'].startswith('peg')
            elif g=='frozenlake':reset_now=name in {'move_north','move_east','move_south','move_west'}
            elif g=='rush_hour':reset_now=name in {'move_vehicle','slide_vehicle'} and args['?vehicle']!='v0'
            if reset_now:t.discard(('controller_ready',))
        final[op]=t
    if cfg['preparation']!='none' and not ready and not busy:final[('prepare_controller',)]=s|{('controller_ready',)}
    if cfg['energy'] and n<3 and not busy:final[('recharge_controller',charge)]=(s-{('controller_charge',charge)})|{('controller_charge','q3')}
    if cfg['cooling'] and busy:final[('wait_for_controller',)]=s-{('controller_busy',)}
    return final

def goal(task,s):
    """The scene's goal, fixed before the PDDL is built or solved."""
    if task.get('extra_game'):
        from .extra_games import goal as extra_goal
        return extra_goal(task,s)
    f=index(s);g=task['family']
    if g=='lights_out':return not f['lit'] and not f['connected'] and () in f['controller_free']
    if g=='sokoban':
        targets={p[0] for p in f['goal_cell']};return all(p in targets for b,p in f['at'] if b in {x[0] for x in f['box']})
    if g=='cooperative_kitchen':return vg.native_goal(task['raw'],physical(task,s))
    if g in {'rush_hour','color_sort'}:return vg.native_goal(task['raw'],physical(task,s))
    if g=='freecell':
        ranks={c:int(v[1:]) for c,v in f['value']};suits=dict(f['hassuit'])
        return all(any(suits[top]==suit and ranks[top]==max(ranks[c] for c in ranks if suits[c]==suit) for top, in f['home']) for suit in set(suits.values()))
    # Scene goals are data specifications, NOT extracted from domain/problem at
    # verification time. This direct comparison also covers negative finishes.
    for a in task['goal_atoms']:
        if a[0]=='not':
            if tuple(a[1]) in s:return False
        elif tuple(a) not in s:return False
    return True

def invariant(task,s):
    if task.get('extra_game'):
        from .extra_games import invariant as extra_invariant
        return extra_invariant(task,s)
    f=index(s);g=task['family'];pol=task['cfg']['policy']
    def vals(p):return {x[0] for x in f[p] if len(x)==1}
    def one(p):assert len(f[p])==1,(g,p,f[p])
    one('controller_charge');assert next(iter(f['controller_charge']))[0] in {'q0','q1','q2','q3'}
    locations=dict(f['at']);assert len(locations)==len(f['at']),(g,'multiple independent locations')
    holders=[x for h,x in f['holding']];assert len(holders)==len(set(holders))
    for hand in vals('hand'):
        held=[x for h,x in f['holding'] if h==hand]
        assert len(held)<=1 and ((hand,) in f['hand_free'])==(not held),(g,'hand occupancy',hand)
    if g in {'blocks','sliding_puzzle','rush_hour','color_sort','cooperative_kitchen'}:
        native=physical(task,s)
        if g!='cooperative_kitchen' or pol not in {'two_counter_slots','prepared_two_slots'}:vg.invariant(task['raw'],native)
        else:
            assert all(p in task['raw']['spec']['floors'] for p in native['chefs'].values())
            assert sum(p==['on','pass1'] for p in native['food_places'].values())<=2
            assert next(iter(f['counter_load']))[0]=='counter'+str(sum(p==['on','pass1'] for p in native['food_places'].values()))
            for food,p in native['food_places'].items():
                if p[0]=='held':assert native['holding'][p[1]]==food
                if p==['on','p1']:assert food in native['prepared']
                if pol=='prepared_two_slots' and p==['on','pass1']:assert food in native['prepared']
        return
    if g in {'sokoban','package','printer'}:
        blockers=vals('box') if g=='sokoban' else vals('package') if g=='package' else vals('printer')|vals('table')
        occupied={p for x,p in f['at'] if x in blockers|{'player'}}
        assert vals('empty')==set(task['raw']['scene']['cells'])-occupied,(g,'empty projection')
        if g=='sokoban':
            assert len([p for x,p in f['at'] if x in blockers|{'player'}])==len(occupied)
            assert vals('on_goal')=={x for x,p in f['at'] if x in blockers and (p,) in f['goal_cell']}
    if g=='hanoi':
        parent=dict(f['on']);assert len(parent)==len(f['on'])
        children=[p for p in parent.values()];assert len(children)==len(set(children))
        for d in vals('disk'):
            assert d in parent
            support=parent[d];assert support.startswith('peg') or int(support[1:])>int(d[1:])
            x=d;seen=set()
            while x in parent:assert x not in seen;seen.add(x);x=parent[x]
            assert x in vals('peg')
            if pol!='ordinary':assert (d,x) in f['tower']
        assert vals('clear')==(vals('disk')|vals('peg'))-set(children)
    if g in {'gripper','kitchen','printer','key_delivery'}:
        kinds={'gripper':('ball',),'kitchen':('ingredient','plate'),'printer':('printer','cartridge'),'key_delivery':('key','relic')}[g]
        for kind in kinds:
            for x in vals(kind):
                own=int(x in locations)+int(x in holders)
                if g=='kitchen' and kind=='ingredient':own+=sum(a==x for a,p in f['on'])
                if g=='printer' and kind=='cartridge':own+=sum(a==x for a,p in f['in'])
                if g=='key_delivery' and kind=='key' and pol=='ordinary':
                    # Consumed lock inserts have no physical location.
                    assert own in {0,1}
                else:assert own==1,(g,kind,x,'location conservation',own)
        if g=='gripper' and pol in {'shared_load_limit','assigned_shared_load'}:assert len(holders)<=1
        if g=='kitchen':
            for x in vals('ingredient'):assert sum(a==x for a,p in f['stage'])==1
        if g=='printer':
            for p in vals('printer'):
                cartridge=[x for x,y in f['in'] if y==p]
                assert len(cartridge)<=1 and ((p,) in f['installed'])==bool(cartridge)
                levels=[n for x,n in f['paper_level'] if x==p];assert len(levels)==1 and 0<=int(levels[0][1:])<=3
            one('paper_stock');assert int(next(iter(f['paper_stock']))[0][5:])+sum(int(n[1:]) for p,n in f['paper_level'])+len(f['printed'])==task['paper_supply']
    if g=='miconic':
        one('at');inside={p for p,l in f['in']}
        assert not inside & vals('served')
        if pol in {'two_seats','two_seat_adjacent'}:assert len(inside)<=2 and f['seat_free']=={('seat'+str(2-len(inside)),)}
    if g=='lights_out':assert len(f['connected'])<=1 and (() in f['controller_free'])==(not f['connected'])
    if g=='floortile':
        robots=vals('robot');positions=[p for r,p in f['at'] if r in robots];painted=[p for p,c in f['painted']]
        assert len(positions)==len(set(positions))==len(robots)
        assert len(painted)==len(set(painted))
        assert vals('clear')==vals('tile')-set(positions)-set(painted)
        if pol in {'walk_on_dry_paint','dry_four_way'}:assert vals('vacant')==vals('tile')-set(positions)
    if g=='termes':
        assert len(f['height'])==len(vals('position')) and len(dict(f['height']))==len(f['height'])
        assert all(h in vals('numb') for p,h in f['height'])
        one('at')
    if g=='briefcase':
        case=locations['briefcase']
        for item in vals('portable'):
            assert item in locations
            if (item,'briefcase') in f['in']:assert locations[item]==case
        if pol in {'two_slots','two_slot_latch'}:assert len(f['in'])<=2 and f['case_free']=={('slot'+str(2-len(f['in'])),)}
    if g=='freecell':
        one('cellspace');one('colspace')
        ranks={c:int(v[1:]) for c,v in f['value']};suits=dict(f['hassuit']);homes={suits[c]:ranks[c] for c, in f['home']}
        parent=dict(f['on']);assert len(parent)==len(f['on'])
        assert len(parent.values())==len(set(parent.values()))
        for c,rank in ranks.items():
            if rank==0:continue
            on_foundation=rank<=homes[suits[c]]
            representations=int(c in parent)+int((c,) in f['bottomcol'])+int((c,) in f['incell'])+int(on_foundation)
            assert representations==1,(g,c,'card conservation',representations)
            x=c;seen=set()
            while x in parent:assert x not in seen;seen.add(x);x=parent[x]
        assert vals('clear')==(set(parent)|vals('bottomcol'))-set(parent.values())
        init=index(task['initial']);maxcells=max(int(x[0][5:]) for x in init['cellnum']);maxcols=max(int(x[0][4:]) for x in init['colnum'])
        assert int(next(iter(f['cellspace']))[0][5:])+len(f['incell'])==maxcells
        assert int(next(iter(f['colspace']))[0][4:])+len(f['bottomcol'])==maxcols
