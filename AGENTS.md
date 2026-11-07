# Project Instructions

## Working Principles

1. **Find the main bottleneck.** Use evidence and first principles to identify the constraint whose resolution most advances the goal and eases other problems. Reassess when blocked or evidence changes; verify the goal at completion.
2. **Make the smallest effective change.** Focus effort on that bottleneck, handle secondary issues as needed, and stay within the requested scope. Prefer local edits and existing simple patterns.
3. **Keep solutions simple and general.** Keep code, pipelines, and prompts concise; add complexity only for a demonstrated need. Fix root causes without speculative abstractions or special-case patches for specific tasks, objects, or test-set scores.
4. **Explain only what matters.** Lead with the main takeaway, then the core mechanism and necessary evidence. Match depth to the request, including paper and module explanations; avoid incidental details, exhaustive lists, and repetition.
5. **Avoid unnecessary hashing.** Do not compute or compare file hashes, generate checksum manifests, or create before-and-after hash snapshots for routine changes. Use `git diff` to inspect modifications and task-specific tests to verify correctness. Only use hashing when explicitly requested or strictly necessary for data integrity.
6. **Require explicit API authorization.** Complete tasks yourself; call third-party APIs only when explicitly authorized by the user.
7. **Keep temporary scripts separate.** Store temporary and intermediate file-processing scripts in the project-root `temp/` directory, outside the main application code.

## PDDL Artifact Consistency

- After changing `domain.pddl` or `problem.pddl`, re-solve it and overwrite the corresponding `plan.txt` with the generated plan. Never manually rename actions in a stale plan.
- A PDDL change is valid and complete only when solving succeeds and the new plan is verified to accomplish the same task as the episode's `kf_actions.txt`.
- Regenerate only the requested rounds; when only the maximum round is in scope, leave older rounds untouched.

## Minimal Causal PDDL Modeling

- Follow `src/swm/prompt_templates/{pddl_generation.txt,RULES.txt,PREDICATE_TEMPLATE.txt}`. Use `eval_results/gpt-5.6-sol/human/unified_domain.pddl` as an operator-style reference. The validated 315-record baseline is `dataset/household_315_pddl_minimal_20261106/`; continue from its optimized models, not the superseded augmentation models.
- Model the smallest set of real functional units, participant bindings, locations, resources, process states and results needed for the task. Merge fixed components; retain detachable parts and independently operated compartments when their identities affect access, support or operation.
- Do not initialize unknown contents as empty or as an invented nonempty portion. For exterior-only handling that physically preserves all possible contents, record the preservation invariant instead of guessing contents. Transfers or treatments that affect them still require grounded material facts.
- Prefer direct `on`, `in`, `at`, `inserted` and `part_of` relations to mirrored connection, closure, clearance and phase flags. Remove a state or guard only with a reachable-invariant argument that preserves legal alternatives and required results; predicate/action counts are diagnostics, not optimization goals.
- Keep one operator per independently completed manipulation, use, handoff, start/stop or completion wait. Merge equivalent operations in the same causal context; do not force demonstration order or add false hardware interlocks. Preserve single-hand occupancy, sustained control, stable support, carried contents, finite transfers and access constraints.
- Preserve evidence-supported temporary tool placement and pickup when their omission would force disposal before otherwise independent final operations. Single-hand occupancy must not become an artificial task order through a missing put-down/pick-up pair.
- Establish boiling, washing, drying and other treatment results at actual completion. Update every affected tracked material in a shared vessel; distinguish device power from running/washing/drying and door-lock states. Unplugging must be allowed to stop an active appliance when physically supported.
- Use ordinary result lifecycles to enforce necessary order: use can invalidate a prior rinse, drying or aftercare result. Add a participant-bound event relation only when final physical facts and invariants cannot prove the required event/order; do not add redundant off-check or refit tickets.
- Keep the natural independent KF and record its formal alignment before the new solve. Review the goal projection and key legal/illegal alternatives independently. Solve each changed physical model once, replay both the KF alignment and the generated plan, and share that proof only across byte-identical model inputs and equivalent wording records. Re-run only models affected by a concrete later change or failure.

## Household Augmentation Quality

- Do not expand datasets through grocery/order quantity enumeration, object subsets, color/name substitutions, pickup-order permutations, adjacent frames or minor changes to the ending state. These do not establish different task logic.
- Pure pick/place sequences are basic skills even when they exceed ten steps. Retain only a few explicitly labeled representatives; do not count them as complex-process data.
- Prioritize evidence-supported kettle, microwave, washing-machine, water-dispenser and faucet processes. A complex sample needs meaningful causal dependencies, multiple operations and necessary state changes or resource/access interlocks. Step count alone is insufficient.
- Compare a candidate with existing process logic before adding it. Record the actual new dependency or mechanism, source evidence and why it is not a minor variant. Do not concatenate unrelated chores or add redundant switching, transfers, waits or preparation to increase length.
- Quantity goals never override these quality requirements. Report a shortfall rather than filling it with simple variations. Retain the ban on generated block tasks, all even-test combination boundaries, independent KF authoring, and image/instruction/KF/PDDL consistency checks.
