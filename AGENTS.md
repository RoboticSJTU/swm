# Project Working Style

Prefer minimal, direct, and easy-to-understand solutions.

- Add only what is necessary. Prefer local changes over new abstractions, helpers, configuration, parameters, files, or pipeline stages.
- Keep changes scoped to the request and preserve existing simple patterns. Do not generalize for hypothetical future needs.
- Keep code, pipelines, and prompts as simple as possible. Add complexity only when a concrete failure demonstrates the need.

## Mandatory Optimization Principles

1. **Reason from first principles.** Prioritize identifying the core bottlenecks that truly affect accuracy and addressing systemic problems.
2. **Keep the pipeline simple.** Do not keep adding modules, branches, or complex logic to fix a small number of errors.
3. **Keep prompts concise, clean, and general.** Include only necessary, broadly applicable rules.
4. **No case-by-case patches.** Do not add special-case rules for specific tasks, specific objects, or a handful of failure cases to improve test-set scores.

## PDDL Artifact Consistency

- After changing any `domain.pddl` or `problem.pddl`, solve the modified PDDL again and overwrite the corresponding `plan.txt` with the newly generated plan.
- The new `plan.txt` must be logically equivalent to the episode's `kf_actions.txt`, i.e., it must accomplish the same task. Otherwise, the PDDL modification is invalid.
- A PDDL change is complete only after solving succeeds and task equivalence is verified. Never manually rename actions in a stale plan.
- Preserve the requested round scope when regenerating plans; do not update older rounds when only the maximum round is in scope.