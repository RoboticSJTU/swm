"""Infer action argument roles without depending on parameter names or order.

Uses aligned initial predicate extensions and precondition/effect structure.
Singleton roles can be constants instead of parameters. Ambiguous mappings are
reported as unresolved; goal success is never used to choose a mapping.
"""
from collections import Counter, defaultdict


class AmbiguousActionMapping(ValueError):
    pass


def extensions(model, mapping):
    result = defaultdict(set)
    for predicate, *args in model['init']:
        if all(x in mapping for x in args): result[predicate].add(tuple(mapping[x] for x in args))
    return dict(result)


def atoms(node, env, predicates, context=()):
    """Role evidence retains polarity and conditional/quantified context."""
    if node[0] == 'and':
        return [a for child in node[1:] for a in atoms(child, env, predicates, context)]
    if node[0] in {'forall','exists'}:
        local = dict(env)
        local.update((p,('bound',len(context),i)) for i,p in enumerate(node[1]))
        return atoms(node[2], local, predicates, (*context,node[0]))
    if node[0] in {'not','or','imply','when'}:
        return [a for i,child in enumerate(node[1:]) for a in atoms(child,env,predicates,(*context,node[0],i))]
    predicate = predicates.get(node[0], ('unknown',len(node)-1))
    return [(context,predicate,tuple(env.get(x,('constant',x)) for x in node[1:]))]


def align_actions(candidate, oracle, mapping):
    reference_ext = extensions(oracle, {x:x for x in oracle['objects']})
    candidate_ext = extensions(candidate, mapping)
    predicates = {}
    for p, ext in candidate_ext.items():
        matches = [q for q, other in reference_ext.items() if ext == other]
        if p in matches: predicates[p] = p
        elif len(matches) == 1: predicates[p] = matches[0]
    # Identical empty/dynamic predicate names are also useful, but names of
    # parameters never provide evidence.
    reference_names = set()
    def collect(n):
        if n[0] in {'and','or','not','imply','when'}:
            for c in n[1:]:collect(c)
        elif n[0] in {'forall','exists'}:collect(n[2])
        else:reference_names.add(n[0])
    for op in oracle['ops'].values():
        collect(op[':precondition']);collect(op[':effect'])
    for p in reference_names:
        if p not in predicates and p not in candidate_ext:predicates[p]=p

    def mutable_predicates(model):
        result=set()
        def walk(n):
            if n[0]=='and':
                for child in n[1:]:walk(child)
            elif n[0] in {'forall','when'}:walk(n[2])
            elif n[0]=='not':walk(n[1])
            else:result.add(n[0])
        for op in model['ops'].values():walk(op[':effect'])
        return result
    candidate_mutable=mutable_predicates(candidate)
    reference_mutable=mutable_predicates(oracle)

    def guards(op, ext, mutable):
        result = defaultdict(list)
        def walk(n):
            if n[0]=='and':
                for child in n[1:]:walk(child)
            elif len(n)==2 and isinstance(n[1],str) and n[1].startswith('?') and n[0] in ext and n[0] not in mutable:
                result[n[1]].append({a[0] for a in ext[n[0]] if len(a)==1})
        walk(op[':precondition'])
        return {v:set.intersection(*sets) for v,sets in result.items() if sets}

    aligned = {}
    for name,op in candidate['ops'].items():
        if name not in oracle['ops']:continue
        ref=oracle['ops'][name]; cp=op[':parameters'];rp=ref[':parameters']
        cg,rg=guards(op,candidate_ext,candidate_mutable),guards(ref,reference_ext,reference_mutable)
        singletons={v:next(iter(s)) for v,s in rg.items() if len(s)==1}
        choices=[]
        for rv in rp:
            options=[i for i,cv in enumerate(cp) if rv not in rg or cv not in cg or bool(rg[rv]&cg[cv])]
            if rv in singletons:options.append(singletons[rv])
            choices.append(options)
        best_score=-1;best=[];examined=0
        ref_env={v:('constant',singletons[v]) if v in singletons else ('role',i) for i,v in enumerate(rp)}
        reference=Counter(atoms(ref[':precondition'],ref_env,{p:p for p in reference_names},('pre',))+
                          atoms(ref[':effect'],ref_env,{p:p for p in reference_names},('effect',)))
        def visit(index, route, used):
            nonlocal best_score,best,examined
            if examined>100000:raise AmbiguousActionMapping('Too many possible roles for '+name)
            if index<len(rp):
                for choice in choices[index]:
                    if isinstance(choice,int) and choice in used:continue
                    visit(index+1,(*route,choice),used|({choice} if isinstance(choice,int) else set()))
                return
            remaining=set(range(len(cp)))-used
            if any(len(cg.get(cp[i],set()))!=1 for i in remaining):return
            examined+=1
            env={x:('constant',mapping[x]) for x in candidate['objects'] if x in mapping}
            for ri,choice in enumerate(route):
                if isinstance(choice,int):env[cp[choice]]=ref_env[rp[ri]]
            extras={i:next(iter(cg[cp[i]])) for i in remaining}
            for i,x in extras.items():env[cp[i]]=('constant',x)
            observed=Counter(atoms(op[':precondition'],env,predicates,('pre',))+atoms(op[':effect'],env,predicates,('effect',)))
            score=sum((reference&observed).values())
            if score>best_score:best_score=score;best=[(route,extras)]
            elif score==best_score and (route,extras) not in best:best.append((route,extras))
        visit(0,(),set())
        # Equivalent ways of supplying the same singleton parameter collapse.
        normalized={}
        for route,extras in best:
            key=tuple(singletons[rp[i]] if rp[i] in singletons else x for i,x in enumerate(route))
            normalized.setdefault(key,(route,extras))
        if len(normalized)!=1 or best_score==0:
            raise AmbiguousActionMapping(f'Cannot uniquely infer argument roles for {name}: {len(normalized)} candidates, score {best_score}')
        route,extras=next(iter(normalized.values()))
        aligned[name]={'reference_arguments':list(route),'candidate_singletons':extras,'score':best_score,
                       'candidate_arity':len(cp),'reference_arity':len(rp)}
    return aligned


def to_reference(action, mapping, adapters):
    if action[0] not in adapters:raise AmbiguousActionMapping('Unmapped operation '+action[0])
    route=adapters[action[0]]['reference_arguments']
    return (action[0],*(mapping[action[i+1]] if isinstance(i,int) else i for i in route))


def to_candidate(action, inverse, adapters):
    if action[0] not in adapters:raise ValueError('Missing operation '+action[0])
    adapter=adapters[action[0]];args=[None]*adapter['candidate_arity']
    for value,position in zip(action[1:],adapter['reference_arguments']):
        if isinstance(position,int):args[position]=inverse[value]
        elif position!=value:raise ValueError('Singleton role mismatch')
    for i,value in adapter['candidate_singletons'].items():args[int(i)]=inverse[value]
    if any(x is None for x in args):raise ValueError('Incomplete action mapping')
    return (action[0],*args)
