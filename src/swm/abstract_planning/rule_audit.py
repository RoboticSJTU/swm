"""Procedural rule oracle for the 17 original visual games.

This oracle never evaluates PDDL preconditions/effects. It enumerates physical
moves and their projected facts from scene state and natural-language rules.
Action parameter names are used only to serialize the fixed reference interface.
"""
from collections import defaultdict
import re

from .model import fields

DIRECTIONS = ("north", "east", "south", "west")
DELTAS = dict(zip(DIRECTIONS, [(-1, 0), (0, 1), (1, 0), (0, -1)]))


def coordinate(name):
    m = re.fullmatch(r"r(\d+)c(\d+)", name)
    if m: return tuple(map(int, m.groups()))
    return tuple(map(int, name.removeprefix("tile_").split("_")))


def adjacent(a, b):
    x, y = coordinate(a), coordinate(b)
    return abs(x[0]-y[0])+abs(x[1]-y[1]) == 1


def number(name):
    return int(re.search(r"\d+$", name)[0])


def audit_description(game, domain, problem, state, instruction):
    """Check explicit recipes/assignments and arithmetic rule facts, not prose IDs."""
    facts = defaultdict(set)
    for a in state: facts[a[0]].add(a[1:])
    def unary(p): return {a[0] for a in facts[p]}
    if game == 'kitchen':
        recipes = re.search(r'^Recipes: (.*)$',instruction,re.M)
        assert recipes
        expected = defaultdict(set)
        for part in recipes[1].split('; '):
            ingredient, sequence = part.split(': ',1)
            methods = re.findall(r'--(\w+)-->', sequence)
            for i,method in enumerate(methods): expected['step_'+method].add((ingredient,'phase'+str(i),'phase'+str(i+1)))
            expected['final_stage'].add((ingredient,'phase'+str(len(methods))))
        for name in set(expected)|{p for p in facts if p.startswith('step_')}: assert facts[name]==expected[name],name
        dishes = re.search(r'^Dishes: (.*)$',instruction,re.M)
        assert dishes
        assigned, required = set(),set()
        for dish,plate,foods,method in re.findall(r'(\w+) uses (\w+), requires exactly (.*?), and follows recipe \w+; finish by (\w+)',dishes[1]):
            assigned.add((plate,dish));required.update((dish,i.strip()) for i in foods.split(','))
            assert (dish,) in facts['finish_'+method+'_recipe']
        assert assigned==facts['plate_for'] and required==facts['requires']
    if game == 'printer':
        jobs = re.search(r'^Jobs: (.*)$',instruction,re.M)
        assert jobs
        parsed = re.findall(r'(\w+) uses (\w+)/(\w+)', jobs[1])
        assert {(d,p) for d,p,c in parsed}==facts['job_for']
        assert {(d,c) for d,p,c in parsed}==facts['document_cartridge']
        assert facts['compatible']=={(c,p) for c in unary('cartridge') for p in unary('printer') if c.removeprefix('cartridge').split('_')[0]==p.removeprefix('printer')}
    if game in {'maze','key_delivery'}:
        assert facts['fits']=={(k,d) for k in unary('key') for d in unary('door') if number(k)==number(d)}
    if game == 'key_delivery':
        deliveries = re.search(r'^Deliver: (.*)$',instruction,re.M)
        assert deliveries and set(re.findall(r'(\w+) to (\w+)',deliveries[1]))==facts['delivery_room']
    if game == 'lights_out':
        expected = {(b,l) for b in unary('button') for l in unary('light')
                    if sum(abs(a-c) for a,c in zip(map(int,b.removeprefix('button').split('_')),map(int,l.removeprefix('light').split('_'))))<=1}
        assert expected==facts['affects']
        assert facts['on']=={(b,'panel'+b.removeprefix('button').split('_')[0]) for b in unary('button')}
    if game == 'miconic':
        capacity = re.search(r'Capacity: (\w+) passengers',instruction)[1]
        assert (capacity=='unlimited') == (not facts['free_slots'])
        if capacity!='unlimited': assert max(map(number,unary('slotcount')))==int(capacity)
        assert facts['above']=={(a,b) for a in unary('floor') for b in unary('floor') if number(b)>number(a)}
    if game == 'freecell':
        ranks={c:number(v) for c,v in facts['value']};suits=dict(facts['hassuit'])
        # Task-relevant domains can omit every stacking operation and its
        # unused static relation. Whenever stacking is allowed, require the
        # complete relation, including moves not used by the reference plan.
        allowed={a[1] for a in domain if isinstance(a,list) and a[:1]==[':action']}
        if allowed & {'move','move_b','colfromfreecell'}:
            assert facts['canstack']=={(a,b) for a in ranks for b in ranks if ranks[a]>=2 and ranks[b]==ranks[a]+1 and (suits[a] in {'c','s'})!=(suits[b] in {'c','s'})}
    if game == 'hanoi':
        objects=unary('object');disks={x for x in objects if x.startswith('d')}
        assert facts['smaller']=={(support,disk) for disk in disks for support in objects if support.startswith('peg') or number(support)>number(disk)}


def successors(game, domain, state, instruction):
    facts = defaultdict(set)
    for a in state: facts[a[0]].add(a[1:])
    def values(p): return {a[0] for a in facts[p] if len(a) == 1}
    def has(p, *a): return tuple(a) in facts[p]
    def place(x):
        return next((b for a, b in facts['at'] if a == x), None)
    def one(p):
        v = values(p)
        assert len(v) == 1, (game, p, v)
        return next(iter(v))
    ops = {a[1]: fields(a)[':parameters'] for a in domain
           if isinstance(a, list) and a[:1] == [':action']}
    result = {}
    def emit(name, args, adds=(), deletes=()):
        if name not in ops: return
        operation = (name, *(args[p[1:]] for p in ops[name]))
        result[operation] = (state-set(deletes)) | set(adds)
    def transfer(name, args, entity, old, new, adds=(), deletes=()):
        emit(name, args, [('at', entity, new), *adds], [('at', entity, old), *deletes])
    hand = next(iter(values('hand')), None)
    player = next(iter(values('player')), None)
    location = place(player) if player else None
    base = dict(player=player, h=hand, inventory=next(iter(values('container')), 'inventory'))

    if game in {'maze', 'package', 'printer'}:
        heading = one('facing')
        index = DIRECTIONS.index(heading)
        for name, target in [('turn_left', DIRECTIONS[(index-1)%4]), ('turn_right', DIRECTIONS[(index+1)%4])]:
            emit(name, {'from':heading, 'to':target}, [('facing',target)], [('facing',heading)])
        for src, dst, d in facts['ahead']:
            assert adjacent(src, dst) and tuple(y-x for x,y in zip(coordinate(src),coordinate(dst))) == DELTAS[d]
            if src != location or d != heading: continue
            args = base | {'from':src, 'to':dst, 'd':d}
            if game == 'maze':
                if all(has('unlocked', door) for a,b,door in facts['edge_door'] if (a,b)==(src,dst)):
                    transfer('move_forward',args,player,src,dst)
            elif not any(place(x)==dst for x in values('package') | values('table') | values('printer')):
                transfer('move_forward',args,player,src,dst,[('empty',src)],[('empty',dst)])
        if game == 'maze':
            for key in values('key'):
                if place(key)==location:
                    emit('pickup_key',base | dict(key=key,loc=location), [('in',key,base['inventory'])], [('at',key,location)])
                for a,b,door in facts['edge_door']:
                    if a==location and has('ahead',a,b,heading) and has('in',key,base['inventory']) and has('fits',key,door) and not has('unlocked',door):
                        emit('unlock_door',base | {'key':key,'door':door,'from':a,'to':b,'d':heading}, [('unlocked',door)], [('in',key,base['inventory'])])
        elif game == 'package':
            for tool in values('tool'):
                if place(tool)==location:
                    emit('take_tool',base | dict(tool=tool,loc=location),[('in',tool,base['inventory'])],[('at',tool,location)])
            for p in values('package'):
                target = place(p)
                if not has('ahead',location,target,heading) or not all(has('open',parent) for child,parent in facts['in'] if child==p): continue
                args = base | {'p':p,'from':location,'to':target,'d':heading}
                for tool in values('tool'):
                    if not has('in',tool,base['inventory']): continue
                    if has('cuts_tape',tool) and has('taped',p):
                        emit('cut_tape',args | dict(tool=tool),[('tape_cut',p)],[('taped',p)])
                    if has('removes_clips',tool) and has('clipped',p):
                        emit('remove_clip',args | dict(tool=tool),[],[('clipped',p)])
                if not has('taped',p) and not has('clipped',p) and not has('open',p): emit('open_package',args,[('open',p)])
        else:
            free = has('hand_free',hand)
            for c in values('cartridge'):
                if free and place(c)==location:
                    emit('pick_cartridge',base | dict(c=c,loc=location),[('holding',hand,c)],[('hand_free',hand),('at',c,location)])
                if has('holding',hand,c):
                    emit('put_cartridge',base | dict(c=c,loc=location),[('hand_free',hand),('at',c,location)],[('holding',hand,c)])
            for table in values('table'):
                target = place(table)
                if not has('ahead',location,target,heading) or any(b==table for a,b in facts['on']): continue
                for p in values('printer'):
                    if has('holding',hand,p):
                        emit('put_on_table',base | {'p':p,'table':table,'from':location,'to':target,'d':heading},
                             [('hand_free',hand),('mounted',p),('at',p,target),('on',p,table)], [('holding',hand,p),('table_free',table)])
            for p in values('printer'):
                target = place(p)
                if target is None or not has('ahead',location,target,heading): continue
                args = base | {'p':p,'from':location,'to':target,'d':heading}
                mounted, powered = has('mounted',p),has('powered',p)
                cartridges = [c for c,holder in facts['in'] if holder==p]
                if free and not mounted:
                    emit('pick_up_printer',args,[('holding',hand,p),('empty',target)],[('hand_free',hand),('at',p,target)])
                count = next(n for obj,n in facts['paper_level'] if obj==p)
                if mounted and free and number(count)<3:
                    emit('load_paper',args | dict(previous=count,result='n'+str(number(count)+1)),
                         [('paper_level',p,'n'+str(number(count)+1))],[('paper_level',p,count)])
                for c in values('cartridge'):
                    if mounted and not powered and not cartridges and has('holding',hand,c) and has('compatible',c,p):
                        emit('install_cartridge',args | dict(c=c),[('hand_free',hand),('installed',p),('in',c,p)], [('holding',hand,c),('calibrated',p)])
                    if not powered and free and c in cartridges:
                        emit('remove_cartridge',args | dict(c=c),[('holding',hand,c)],[('hand_free',hand),('installed',p),('in',c,p),('calibrated',p)])
                if mounted and cartridges and not powered: emit('power_on',args,[('powered',p)])
                if powered: emit('power_off',args,[],[('powered',p),('calibrated',p)])
                if powered and not has('calibrated',p): emit('calibrate',args,[('calibrated',p)])
                if powered and has('calibrated',p) and number(count)>0:
                    for doc in values('document'):
                        for c in cartridges:
                            if not has('printed',doc) and has('job_for',doc,p) and has('document_cartridge',doc,c):
                                deletes = [('paper_level',p,count)]
                                if 'Printing consumes calibration' in instruction: deletes.append(('calibrated',p))
                                emit('print_document',args | dict(doc=doc,c=c,previous=count,result='n'+str(number(count)-1)),
                                     [('printed',doc),('paper_level',p,'n'+str(number(count)-1))], deletes)
    elif game == 'frozenlake':
        for target in values('safe'):
            if not adjacent(location,target): continue
            delta = tuple(y-x for x,y in zip(coordinate(location),coordinate(target)))
            name = next(d for d,v in DELTAS.items() if v==delta)
            deletes = [('safe',location)] if has('fragile',location) else []
            transfer('move_'+name,base | {'from':location,'to':target},player,location,target,[],deletes)
        for m in values('checkpoint'):
            if place(m)==location and not has('marked',m) and all(has('marked',x) for x,y in facts['before'] if y==m):
                emit('record_checkpoint',base | dict(m=m,loc=location),[('marked',m)])
    elif game == 'sokoban':
        floors = values('location'); boxes = values('box'); positions = {b:place(b) for b in boxes}
        for target in floors:
            if not adjacent(location,target): continue
            delta = tuple(y-x for x,y in zip(coordinate(location),coordinate(target)))
            d = next(d for d,v in DELTAS.items() if v==delta)
            if target not in positions.values():
                transfer('move',base | {'from':location,'to':target,'d':d},player,location,target,[('empty',location)],[('empty',target)])
            else:
                box = next(b for b,l in positions.items() if l==target)
                beyond = next((c for c in floors if tuple(y-x for x,y in zip(coordinate(target),coordinate(c)))==delta),None)
                if beyond is None or beyond in {*positions.values(),location}: continue
                goal = has('goal_cell',beyond)
                emit('push_to_goal' if goal else 'push_to_nongoal',base | {'box':box,'from':location,'boxloc':target,'to':beyond,'d':d},
                     [('at',box,beyond),('at',player,target),('empty',location),*([('on_goal',box)] if goal else [])],
                     [('at',box,target),('at',player,location),('empty',beyond),*([('on_goal',box)] if not goal else [])])
    elif game == 'kitchen':
        for target in values('location'):
            if adjacent(location,target): transfer('move',base | {'from':location,'to':target},player,location,target)
        free = has('hand_free',hand)
        for kind,name in [('ingredient','pickup_ingredient'),('plate','pickup_plate')]:
            for item in values(kind):
                if free and place(item)==location:
                    emit(name,base | dict(i=item,plate=item,loc=location),[('holding',hand,item)],[('hand_free',hand),('at',item,location)])
        for h,item in facts['holding']:
            if h==hand: emit('put_down',base | dict(item=item,loc=location),[('hand_free',hand),('at',item,location)],[('holding',hand,item)])
        for tool in values('tool'):
            if place(tool)!=location: continue
            if free and not has('clean',tool): emit('clean_tool',base | dict(tool=tool,loc=location),[('clean',tool)])
            if not has('clean',tool): continue
            for i in values('ingredient'):
                if not has('holding',hand,i): continue
                phase = next(p for obj,p in facts['stage'] if obj==i)
                recipes = re.search(r'^Recipes: (.*)$',instruction,re.M)
                recipe = next((part.split(': ',1)[1] for part in recipes[1].split('; ') if part.startswith(i+': ')),None) if recipes else None
                assert recipe is not None, ('missing recipe',i)
                methods = re.findall(r'--([a-z]+)-->',recipe)
                current = number(phase)
                if current>=len(methods): continue
                method = methods[current]
                if tool!=method+'_station': continue
                after = 'phase'+str(current+1)
                emit(method,base | dict(i=i,tool=tool,loc=location,previous=phase,result=after),
                     [('stage',i,after),*([('prepared',i)] if current+1==len(methods) else [])],[('stage',i,phase),('clean',tool)])
        for plate,dish in facts['plate_for']:
            for i in values('ingredient'):
                if has('holding',hand,i) and has('prepared',i) and place(plate)==location and has('requires',dish,i):
                    emit('put_on_plate',base | dict(i=i,plate=plate,dish=dish,loc=location),[('on',i,plate),('hand_free',hand)],[('holding',hand,i)])
            if not has('holding',hand,plate): continue
            required = [i for d,i in facts['requires'] if d==dish]
            if not has('ready',dish) and all(has('prepared',i) and has('on',i,plate) for i in required):
                for method in ['mix','bake']:
                    tool = 'finish_'+method+'_station'
                    if has('finish_'+method+'_recipe',dish) and place(tool)==location and has('clean',tool):
                        emit('finish_'+method,base | dict(dish=dish,plate=plate,tool=tool,loc=location),[('ready',dish)],[('clean',tool)])
            if has('ready',dish):
                for tool in values('cool_tool'):
                    if has('needs_cooling',dish) and place(tool)==location:
                        emit('cool',base | dict(dish=dish,plate=plate,tool=tool,loc=location),[('cooled',dish)])
                if has('delivery',location) and (not has('needs_cooling',dish) or has('cooled',dish)):
                    emit('deliver',base | dict(dish=dish,plate=plate,loc=location),[('delivered',dish),('at',plate,location),('hand_free',hand)],[('holding',hand,plate)])
    elif game == 'key_delivery':
        for src,dst in facts['road']:
            if src==location and all(has('unlocked',door) for a,b,door in facts['edge_door'] if (a,b)==(src,dst)):
                transfer('move',base | {'from':src,'to':dst},player,src,dst)
        for kind in ['key','relic']:
            for item in values(kind):
                args = base | {kind:item,'room':location}
                if has('hand_free',hand) and place(item)==location:
                    emit('pickup_'+kind,args,[('holding',hand,item)],[('hand_free',hand),('at',item,location)])
                if not has('holding',hand,item): continue
                if kind=='relic':
                    emit('put_down_relic',args,[('hand_free',hand),('at',item,location),*([('delivered',item)] if has('delivery_room',item,location) else [])],[('holding',hand,item)])
                else:
                    for src,dst,door in facts['edge_door']:
                        if src==location and has('fits',item,door) and not has('unlocked',door):
                            emit('unlock',args | {'door':door,'from':src,'to':dst},[('unlocked',door),('hand_free',hand)],[('holding',hand,item)])
    elif game == 'lights_out':
        connected = values('connected')
        for panel in values('panel'):
            if not connected:
                emit('connect_panel',dict(p=panel),[('connected',panel)],[('controller_free',)])
            if panel in connected:
                emit('disconnect_panel',dict(p=panel),[('controller_free',)],[('connected',panel)])
                for button in values('button'):
                    r,c = map(int,button.removeprefix('button').split('_'))
                    if panel!='panel'+str(r): continue
                    lights = [light for light in values('light') if sum(abs(a-b) for a,b in zip((r,c),map(int,light.removeprefix('light').split('_'))))<=1]
                    emit('press_button',dict(p=panel,b=button),[('lit',l) for l in lights if not has('lit',l)],[('lit',l) for l in lights if has('lit',l)])
    elif game == 'blocks':
        blocks = values('block'); supporting = dict(facts['on']); held = {b for h,b in facts['holding']}
        clear = {b for b in blocks if b in supporting and b not in supporting.values()}
        for b in blocks:
            if b in held:
                emit('putdown',dict(ob=b,h=hand,surface='table'),[('on',b,'table'),('hand_free',hand),('clear',b)],[('holding',hand,b)])
                for support in clear:
                    emit('stack',dict(ob=b,underob=support,h=hand),[('on',b,support),('hand_free',hand),('clear',b)],[('holding',hand,b),('clear',support)])
            if has('hand_free',hand) and b in clear:
                support = supporting[b]
                emit('pickup' if support=='table' else 'unstack',dict(ob=b,underob=support,h=hand,surface='table'),
                     [('holding',hand,b),*([('clear',support)] if support!='table' else [])],[('on',b,support),('hand_free',hand),('clear',b)])
    elif game == 'hanoi':
        supporting = dict(facts['on']); disks = set(supporting); occupied = set(supporting.values())
        supports = disks | {x for x in values('object') if x.startswith('peg')}
        for disk in disks-occupied:
            old = supporting[disk]
            for target in supports-occupied:
                if target.startswith('peg') or (target.startswith('d') and number(target)>number(disk)):
                    emit('move',{'disc':disk,'from':old,'to':target},[('on',disk,target),('clear',old)],[('on',disk,old),('clear',target)])
    elif game == 'sliding_puzzle':
        blank = one('empty')
        for tile in values('tile'):
            origin = place(tile)
            if not adjacent(blank,origin): continue
            delta = tuple(y-x for x,y in zip(coordinate(blank),coordinate(origin)))
            direction = next(d for d,v in DELTAS.items() if v==delta)
            transfer('move_blank_'+direction,{'tile':tile,'from':origin,'to':blank},tile,origin,blank,[('empty',origin)],[('empty',blank)])
    elif game == 'gripper':
        robot = one('robot'); room = place(robot)
        for target in values('room'): transfer('move',{'from':room,'to':target,'robby':robot},robot,room,target)
        for gripper in values('hand'):
            for ball in values('ball'):
                args = dict(obj=ball,room=room,gripper=gripper,robby=robot)
                if has('hand_free',gripper) and place(ball)==room:
                    emit('pick',args,[('holding',gripper,ball)],[('at',ball,room),('hand_free',gripper)])
                if has('holding',gripper,ball): emit('drop',args,[('at',ball,room),('hand_free',gripper)],[('holding',gripper,ball)])
    elif game == 'miconic':
        lift = one('lift'); floor = place(lift); counts = values('free_slots')
        for target in values('floor'):
            if target!=floor: transfer('up' if number(target)>number(floor) else 'down',dict(f1=floor,f2=target,lift=lift),lift,floor,target)
        for p in values('passenger'):
            args = dict(f=floor,p=p,lift=lift)
            if counts:
                count = next(iter(counts)); args.update(previous=count)
            if not has('served',p) and not has('in',p,lift) and has('origin',p,floor) and (not counts or number(count)>0):
                adds, deletes = [('in',p,lift)],[('not_boarded',p)]
                if counts:
                    after='n'+str(number(count)-1); args['result']=after
                    adds.append(('free_slots',after));deletes.append(('free_slots',count))
                emit('board',args,adds,deletes)
            if has('in',p,lift) and has('destin',p,floor):
                adds,deletes=[('served',p),('not_boarded',p)],[('in',p,lift),('not_served',p)]
                if counts:
                    after='n'+str(number(count)+1);args['result']=after
                    adds.append(('free_slots',after));deletes.append(('free_slots',count))
                emit('depart',args,adds,deletes)
    elif game == 'floortile':
        robots = values('robot'); tiles = values('tile'); occupied={place(r) for r in robots}; painted={t for t,c in facts['painted']}
        for r in robots:
            origin=place(r); color=next(c for robot,c in facts['robot_has'] if robot==r)
            for c in values('available_color'):
                emit('change_color',dict(r=r,c=color,c2=c),[('robot_has',r,c)],[('robot_has',r,color)])
            for target in tiles-occupied-painted:
                if not adjacent(origin,target):continue
                dr,dc=tuple(y-x for x,y in zip(coordinate(origin),coordinate(target)))
                direction = 'up' if dr==1 else 'down' if dr==-1 else 'right' if dc==1 else 'left'
                transfer(direction,dict(r=r,x=origin,y=target),r,origin,target,[('clear',origin)],[('clear',target)])
                if dc==0:emit('paint_'+direction,dict(r=r,x=origin,y=target,c=color),[('painted',target,color)],[('clear',target)])
    elif game == 'termes':
        robot = one('robot'); origin=place(robot); heights=dict(facts['height']); h=heights[origin]
        carrying=has('has_block')
        if has('is_depot',origin):
            emit('destroy_block' if carrying else 'create_block',dict(p=origin,robot=robot),[] if carrying else [('has_block',)],[('has_block',)] if carrying else [])
        for target,k in heights.items():
            if not adjacent(origin,target):continue
            difference=number(k)-number(h)
            if abs(difference)<=1:
                transfer('move' if difference==0 else 'move_up' if difference==1 else 'move_down',{'from':origin,'to':target,'h':h,'hfrom':h,'hto':k,'robot':robot},robot,origin,target)
            if carrying and difference==0 and not has('is_depot',target) and 'h'+str(number(k)+1) in values('numb'):
                after='h'+str(number(k)+1)
                emit('place_block',dict(rpos=origin,bpos=target,hbefore=k,hafter=after,robot=robot),[('height',target,after)],[('height',target,k),('has_block',)])
            if not carrying and difference==1:
                emit('remove_block',dict(rpos=origin,bpos=target,hbefore=k,hafter=h,robot=robot),[('height',target,h),('has_block',)],[('height',target,k)])
    elif game == 'briefcase':
        case=one('briefcase'); room=place(case)
        for target in values('location'):
            moving={case}|{x for x,h in facts['in'] if h==case}
            emit('move',dict(m=room,l=target,briefcase=case),[('at',x,target) for x in moving],[('at',x,room) for x in moving])
        for item in values('portable'):
            if has('in',item,case):emit('take_out',dict(x=item,briefcase=case),[],[('in',item,case)])
            elif place(item)==room:emit('put_in',dict(x=item,l=room,briefcase=case),[('in',item,case)])
    elif game == 'freecell':
        cells=one('cellspace');cols=one('colspace');bottom=values('bottomcol');incell=values('incell');clear=values('clear');supporting=dict(facts['on'])
        ranks={c:number(v) for c,v in facts['value']};suits=dict(facts['hassuit']);homes=values('home')
        for card in clear|incell:
            is_free=card in incell; is_bottom=card in bottom; old=supporting.get(card)
            args=dict(card=card,oldcard=old,cells=cells,cols=cols,vcard='n'+str(ranks[card]),suitname=suits[card])
            source_add=[] if is_free or is_bottom else [('clear',old)]
            source_del=[('incell',card)] if is_free else [('bottomcol',card)] if is_bottom else [('on',card,old)]
            if is_free:
                args['ncells']='celln'+str(number(cells)+1);source_add.append(('cellspace',args['ncells']));source_del.append(('cellspace',cells))
            elif is_bottom:
                args['ncols']='coln'+str(number(cols)+1);source_add.append(('colspace',args['ncols']));source_del.append(('colspace',cols))
            for new in clear:
                if ranks[card]>=2 and ranks[new]==ranks[card]+1 and (suits[card] in {'c','s'})!=(suits[new] in {'c','s'}):
                    emit('colfromfreecell' if is_free else 'move_b' if is_bottom else 'move',args | dict(newcard=new),
                         [*source_add,('on',card,new),('clear',card)],[*source_del,('clear',new)])
            if not is_free and number(cells)>0:
                emit('sendtofree_b' if is_bottom else 'sendtofree',args | dict(ncells='celln'+str(number(cells)-1)),
                     [*source_add,('incell',card),('cellspace','celln'+str(number(cells)-1))],[*source_del,('clear',card),('cellspace',cells)])
            if (is_free or not is_bottom) and number(cols)>0:
                emit('newcolfromfreecell' if is_free else 'sendtonewcol',args | dict(ncols='coln'+str(number(cols)-1)),
                     [*source_add,('bottomcol',card),('clear',card),('colspace','coln'+str(number(cols)-1))],[*source_del,('colspace',cols)])
            for home in homes:
                if suits[home]==suits[card] and ranks[card]==ranks[home]+1:
                    emit('homefromfreecell' if is_free else 'sendtohome_b' if is_bottom else 'sendtohome',args | dict(homecard=home,vhomecard='n'+str(ranks[home])),
                         [*source_add,('home',card)],[*source_del,('clear',card),('home',home)])
    else:
        raise ValueError('No procedural rule oracle: '+game)
    return result
