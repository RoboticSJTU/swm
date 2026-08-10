# FILE: pddl_generation.txt

## MODE: common

### COMPONENT: arm_abstraction
## Arm abstraction.

- Model the available arms as interchangeable hand resources: use one hand object for single-arm and two hand objects for dual-arm.
- Do not introduce left/right-hand preference, reachability, or capability predicates in action preconditions.

### COMPONENT: rules
# RULES

## 1. Syntax.

- Use only PDDL 1.0 syntax: no `:requirements`, `:types`, or advanced constructs such as or, when, forall, exists, and conditional effects.
- In every Problem file, include `:objects` before `:init`; list object instances in `:objects`, and encode categories as unary predicates in `:init`.
- Object names must be explicit and unambiguous. Avoid arbitrary numeric suffixes; use digits only when they are part of the object’s visual or textual identity, e.g., block1. Distinguish multiple same-category objects by grounded descriptors such as color, position.

## 2. Actions.

### (1) Action selection and ordering.

- Generate only task-relevant action definitions needed to complete the instruction.
- Order action definitions by the causal execution sequence implied by the instruction and scene.

### (2) Action name construction.

Base naming pattern: VERB_OBJECT[_from_SOURCE][_PREP_TARGET][_with_TOOL].

- Use the base naming pattern as the default basis, and **follow similar PDDL action templates when naming actions instead of constructing action names based on steps.**
- Extend the base naming only when necessary to distinguish actions or express task-relevant state, role, pose, result, or condition. When extending the name, add the extra information as a modifier to the most relevant slot, while keeping the action name concise, readable, and action-distinguishing.
- Each action name must contain only one main action. Do not use `to`, `and`, `then` to merge sequential actions; split them into separate operators.
- VERB is the atomic action, such as pick, place, open, close, turn, turn_on, turn_off, pour, fill, scoop, insert, remove, wipe, wash.
- OBJECT is mandatory and denotes the category of the entity explicitly operated on or affected by the hand or tool.
- SOURCE is where the object or content comes from. TARGET is where the object or content goes.
- PREP is a generic spatial preposition such as in, on, under, into, onto.
- TOOL is the concrete tool or auxiliary device used by the action. Do not use hand as TOOL.

### (3) Naming constraints.

- Use concrete categories for OBJECT, SOURCE, and TARGET, e.g., apple, bowl, drawer; avoid generic labels like object, food, surface, container, or location.
- Do not encode non-operational attributes in action names or category predicates, such as color, material, texture, size, shape detail, or appearance, unless they change the action logic. Use the functional object category instead: use `place_block_on_table`, not `place_wooden_block_on_table`.
- Use *_top only in action names to distinguish a container’s top surface from its interior. Do not create separate *_top objects or predicates.

## 3. Predicates.

### (1) Predicate modeling.

- Minimize the number of predicates. Use only predicates necessary for the current task. Do not model irrelevant object attributes, fine-grained spatial details, or intermediate states.
- **In Domain :predicates, declare each predicate name+arity only once. Reuse generic schemas across variables, object categories, and roles. e.g., use only one generic (is_off ?x), (open ?x), (on ?o ?s), or (in ?o ?c).**

### (2) Predicate groups and order.

- Write predicates in this group order: type predicates, state predicates, spatial predicates.
  - Type predicates are unary category predicates.
  - State predicates include hand/object/device states, with hand states first.
  - Spatial predicates are binary generic relation predicates, e.g., on, in, under.
- Only include groups needed in the current context. Ordering rules must not introduce extra predicates.
- Within state predicates, keep mutually exclusive states adjacent. In action `:effect`, write each state transition as delete-old first, then add-new.

### (3) Predicate formatting.

- Use compact, consistently indented PDDL formatting: one-line `:objects`, `:parameters`, `:precondition`, and `:effect`; one predicate/fact per line inside `:predicates`, `:init`, and `:goal`.
- Order goal facts by causal achievement order; in effects, write delete-old before add-new.
- Do not add predicate-category comments inside PDDL.

## 4. Annotation.

- Each :action must have one single-line, human-readable sentence comment above it describing the action with all abstract parameters, not instantiated names.
- Correct: ; Pick up apple ?a from table ?t with hand ?h. Incorrect: ; Pick up apple from table with right hand.

### COMPONENT: action_template
## 1. Spatial Change.

### (1) Source only.

; Pick up apple ?a from table ?t with hand ?h.
(:action pick_apple_from_table
:parameters (?h ?a ?t)
:precondition (and (hand ?h) (apple ?a) (table ?t) (hand_free ?h) (on ?a ?t))
:effect (and (not (hand_free ?h)) (holding ?h ?a) (not (on ?a ?t)))
)

; Pick up apple ?a from cabinet top ?c with hand ?h.
(:action pick_apple_from_cabinet_top
:parameters (?h ?a ?c)
:precondition (and (hand ?h) (apple ?a) (cabinet ?c) (hand_free ?h) (on ?a ?c))
:effect (and (not (hand_free ?h)) (holding ?h ?a) (not (on ?a ?c)))
)

; Pick up apple ?a from pot ?p with hand ?h.
(:action pick_apple_from_pot
:parameters (?h ?a ?p)
:precondition (and (hand ?h) (apple ?a) (pot ?p) (hand_free ?h) (open ?p) (in ?a ?p))
:effect (and (not (hand_free ?h)) (holding ?h ?a) (not (in ?a ?p)))
)

; Pick up block ?b1 from block ?b2 with hand ?h.
(:action pick_block_from_block
:parameters (?h ?b1 ?b2)
:precondition (and (hand ?h) (block ?b1) (block ?b2) (hand_free ?h) (clear ?b1) (on ?b1 ?b2))
:effect (and (not (hand_free ?h)) (holding ?h ?b1) (not (clear ?b1)) (clear ?b2) (not (on ?b1 ?b2)))
)

; Pick up cup ?c from laptop ?l with hand ?h.
(:action pick_cup_from_laptop
:parameters (?h ?c ?l)
:precondition (and (hand ?h) (cup ?c) (laptop ?l) (hand_free ?h) (on ?c ?l))
:effect (and (not (hand_free ?h)) (holding ?h ?c) (clear ?l) (not (on ?c ?l)))
)

; Fill kettle ?k with water ?w from faucet ?f while holding it with hand ?h.
(:action fill_kettle_from_faucet
:parameters (?h ?k ?w ?f)
:precondition (and (hand ?h) (kettle ?k) (water ?w) (faucet ?f) (holding ?h ?k) (open ?k) (is_on ?f) (in ?w ?f))
:effect (and (in ?w ?k))
)

; Scoop soup ?sp from pot ?p with spoon ?s held by hand ?h.
(:action scoop_soup_from_pot_with_spoon
:parameters (?h ?sp ?p ?s)
:precondition (and (hand ?h) (soup ?sp) (pot ?p) (spoon ?s) (holding ?h ?s) (open ?p) (in ?sp ?p))
:effect (and (not (in ?sp ?p)) (in ?sp ?s))
)

; Insert key ?k into drawer ?d with hand ?h.
(:action insert_key_into_drawer
:parameters (?h ?k ?d)
:precondition (and (hand ?h) (key ?k) (drawer ?d) (holding ?h ?k))
:effect (and (not (holding ?h ?k)) (hand_free ?h) (inserted ?k ?d))
)

; Remove lid ?l from pot ?p to open the pot with hand ?h.
(:action remove_lid_from_pot
:parameters (?h ?l ?p)
:precondition (and (hand ?h) (lid ?l) (pot ?p) (hand_free ?h) (closed ?p) (on ?l ?p))
:effect (and (not (hand_free ?h)) (holding ?h ?l) (not (closed ?p)) (open ?p) (not (on ?l ?p)))
)

### (2) Target only.

; Place apple ?a in bowl ?b with hand ?h.
(:action place_apple_in_bowl
:parameters (?h ?a ?b)
:precondition (and (hand ?h) (apple ?a) (bowl ?b) (holding ?h ?a))
:effect (and (not (holding ?h ?a)) (hand_free ?h) (in ?a ?b))
)

; Place plate ?p flat on counter ?c with hand ?h.
(:action place_plate_flat_on_counter
:parameters (?h ?p ?c)
:precondition (and (hand ?h) (plate ?p) (counter ?c) (holding ?h ?p) (vertical ?p))
:effect (and (not (holding ?h ?p)) (hand_free ?h) (not (vertical ?p)) (flat ?p) (on ?p ?c))
)

; Place cap ?cp on bottle ?b to close the bottle with hand ?h.
(:action place_cap_on_bottle
:parameters (?h ?cp ?b)
:precondition (and (hand ?h) (cap ?cp) (bottle ?b) (holding ?h ?cp) (open ?b))
:effect (and (not (holding ?h ?cp)) (hand_free ?h) (not (open ?b)) (closed ?b) (on ?cp ?b))
)

; Place trash can ?tc on floor ?f to unblock drawer ?d with hand ?h.
(:action place_trash_can_on_floor_away_from_drawer
:parameters (?h ?tc ?f ?d)
:precondition (and (hand ?h) (trash_can ?tc) (floor ?f) (drawer ?d) (holding ?h ?tc) (blocking ?tc ?d))
:effect (and (not (holding ?h ?tc)) (hand_free ?h) (not (blocking ?tc ?d)) (unblocked ?d) (on ?tc ?f))
)

### (3) From source to target.

; Pour milk ?m from milk carton ?mc into cup ?c with hand ?h.
(:action pour_milk_from_milk_carton_into_cup
:parameters (?h ?m ?mc ?c ?t)
:precondition (and (hand ?h) (milk ?m) (milk_carton ?mc) (cup ?c) (table ?t) (holding ?h ?mc) (open ?mc) (in ?m ?mc) (on ?c ?t))
:effect (and (not (in ?m ?mc)) (in ?m ?c))
)

; Pour contents ?ct from cup ?c into bowl ?b on table ?t with hand ?h.
(:action pour_contents_from_cup_into_bowl
:parameters (?h ?ct ?c ?b ?t)
:precondition (and (hand ?h) (contents ?ct) (cup ?c) (bowl ?b) (table ?t) (holding ?h ?c) (in ?ct ?c) (on ?b ?t))
:effect (and (not (in ?ct ?c)) (in ?ct ?b))
)

## 2. State Change.

; Open drawer ?d with hand ?h.
(:action open_drawer
:parameters (?h ?d)
:precondition (and (hand ?h) (drawer ?d) (hand_free ?h) (closed ?d))
:effect (and (not (closed ?d)) (open ?d))
)

; Open detergent drawer ?d with hand ?h.
(:action open_detergent_drawer
:parameters (?h ?d)
:precondition (and (hand ?h) (detergent_drawer ?d) (hand_free ?h) (closed ?d))
:effect (and (not (closed ?d)) (open ?d))
)

; Open washing machine ?wm with hand ?h.
(:action open_washing_machine
:parameters (?h ?wm)
:precondition (and (hand ?h) (washing_machine ?wm) (hand_free ?h) (closed ?wm))
:effect (and (not (closed ?wm)) (open ?wm))
)

; Open microwave ?m with hand ?h.
(:action open_microwave
:parameters (?h ?m)
:precondition (and (hand ?h) (microwave ?m) (hand_free ?h) (closed ?m) (is_off ?m))
:effect (and (not (closed ?m)) (open ?m))
)

; Lock drawer ?d with key ?k using hand ?h.
(:action lock_drawer_with_key
:parameters (?h ?d ?k)
:precondition (and (hand ?h) (drawer ?d) (key ?k) (hand_free ?h) (closed ?d) (unlocked ?d) (inserted ?k ?d))
:effect (and (not (unlocked ?d)) (locked ?d))
)

; Turn on microwave ?m to heat apple ?a inside with hand ?h.
(:action turn_on_microwave
:parameters (?h ?m ?a)
:precondition (and (hand ?h) (microwave ?m) (apple ?a) (hand_free ?h) (closed ?m) (is_off ?m) (in ?a ?m))
:effect (and (not (is_off ?m)) (is_on ?m))
)

; Turn off microwave ?m after heating apple ?a inside with hand ?h.
(:action turn_off_microwave
:parameters (?h ?m ?a)
:precondition (and (hand ?h) (microwave ?m) (apple ?a) (hand_free ?h) (closed ?m) (is_on ?m) (in ?a ?m))
:effect (and (not (is_on ?m)) (is_off ?m) (heated ?a))
)

; Turn on faucet ?f with hand ?h.
(:action turn_on_faucet
:parameters (?h ?f)
:precondition (and (hand ?h) (faucet ?f) (hand_free ?h) (is_off ?f))
:effect (and (not (is_off ?f)) (is_on ?f))
)

; Wipe counter ?c with cloth ?cl with hand ?h.
(:action wipe_counter_with_cloth
:parameters (?h ?c ?cl)
:precondition (and (hand ?h) (counter ?c) (cloth ?cl) (holding ?h ?cl))
:effect (and (wiped ?c))
)

; Fold towel ?tw on table ?t with hand ?h.
(:action fold_towel_on_table
:parameters (?h ?tw ?t)
:precondition (and (hand ?h) (towel ?tw) (table ?t) (hand_free ?h) (unfolded ?tw) (on ?tw ?t))
:effect (and (not (unfolded ?tw)) (folded ?tw))
)

; Unfold cloth ?cl on counter ?c with hand ?h.
(:action unfold_cloth_on_counter
:parameters (?h ?cl ?c)
:precondition (and (hand ?h) (cloth ?cl) (counter ?c) (hand_free ?h) (on ?cl ?c) (folded ?cl))
:effect (and (not (folded ?cl)) (unfolded ?cl))
)

; Align rope ?r with black object ?o.
( align_rope_with_black_object
(?h ?r ?o)
(and (hand ?h) (rope ?r) (black_object ?o) (holding ?h ?r))
(and (aligned ?r ?o))
)

; Wash cup ?c with faucet ?f while holding it with hand ?h.
(:action wash_cup_with_faucet
:parameters (?h ?c ?f)
:precondition (and (hand ?h) (cup ?c) (faucet ?f) (holding ?h ?c) (is_on ?f))
:effect (and (washed ?c))
)

; Stir pot ?p with spoon ?s held by hand ?h.
(:action stir_pot_with_spoon
:parameters (?h ?s ?p)
:precondition (and (hand ?h) (spoon ?s) (pot ?p) (holding ?h ?s) (open ?p))
:effect (and (stirred ?p))
)

## MODE: initial

### COMPONENT: task_description
You are an expert in writing a PDDL domain and problem grounded in the given image and instruction for robot task planning.

### COMPONENT: reasoning_process
(a) Scene understanding: From the image, identify all task-relevant objects, their number, locations, immediate support/containment relations, and the open/closed/on/off states of relevant containers/devices.
(b) Domain construction: Based on the scene understanding and action templates, define only the necessary action definitions in the causal execution order. Before output, check that Domain `:predicates` has no duplicate predicate name+arity.
(c) Problem construction: Encode all and only task-relevant initial facts in init, including the hand state. Encode the instruction-required end states in goal, ordered by the causal order in which they become true during execution.

### COMPONENT: reasoning_schema
a three-part analysis with the following structure. (a) Scene understanding. (b) Domain construction. (c) Problem construction.

## MODE: feedback

### COMPONENT: task_description
Your task is to diagnose why the previous PDDL domain and problem failed, use the planner output and failure feedback to identify the necessary corrections, and regenerate a valid, executable, and task-correct PDDL domain and problem.

### COMPONENT: failure_context
# PREVIOUS FAILED ATTEMPT

## Domain

{failed_domain}

## Problem

{failed_problem}

## Output from PDDL planner

{failed_plan}

## Failure Feedback

{feedback}

### COMPONENT: reasoning_process
(a) Failure diagnosis: identify the key issues in the previous attempt.
(b) Scene understanding: From the image, identify all task-relevant objects, their number, locations, immediate support/containment relations, and the open/closed/on/off states of relevant containers/devices.
(c) Domain construction: Based on the scene understanding and action templates, define only the necessary action definitions in the causal execution order. Before output, check that Domain `:predicates` has no duplicate predicate name+arity.
(d) Problem construction: Encode all and only task-relevant initial facts in init, including the hand state. Encode the instruction-required end states in goal, ordered by the causal order in which they become true during execution.

### COMPONENT: reasoning_schema
a four-part analysis with the following structure: (a) Failure diagnosis. (b) Scene understanding. (c) Domain construction. (d) Problem construction.















# FILE: plan_learning.txt

## MODE: initial

### COMPONENT: task_description
You are given segmented keyframes from a full video demonstration of an instruction.

### COMPONENT: inference_workflow
Use a vision-first two-pass workflow. Instruction and history may only help disambiguate visually supported actions, not predict the next action.
(1) Pairwise pass: For each (Ki, Ki+1), infer actions only from completed visible state changes or contact changes.
(2) Global verification: Review all keyframes to ensure the pairwise-inferred actions are globally consistent with the group’s observable world-state changes.

### COMPONENT: action_reasoning_schema
\"\" OR a two-part analysis: (1) Pairwise pass. (2) Global verification.

## MODE: feedback

### COMPONENT: task_description
You are revising a previously failed action description for the current keyframe group.
Correct the previous output according to the feedback while staying strictly grounded in visual evidence.

### COMPONENT: feedback_context
- Previous failed output for the current group: {error_action}
- Feedback: {feedback}

### COMPONENT: inference_workflow
Use a vision-first three-pass workflow. Instruction and history may only help disambiguate visually supported actions, not predict the next action.
(1) Failure diagnosis: Identify the key issues in the previous failed output.
(2) Pairwise pass: For each (Ki, Ki+1), infer actions only from completed visible state changes or contact changes.
(3) Global verification: Review all keyframes to ensure the pairwise-inferred actions are globally consistent with the group’s observable world-state changes.

### COMPONENT: feedback_note
- This is a regeneration step. Correct the previous failed output based on the feedback, but keep only revisions that are visually supported.

### COMPONENT: action_reasoning_schema
\"\" OR a three-part analysis: (1) Failure diagnosis. (2) Pairwise pass. (3) Global verification.
