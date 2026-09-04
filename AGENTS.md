# Project Working Style

The user prefers code that is minimal, direct, and easy to understand.

- Follow the principle: do not add entities unless they are necessary.
- Prefer a simple local change over new abstractions, helper layers, configuration systems, parameters, or files.
- Do not generalize for hypothetical future needs.
- Keep changes scoped to the request and preserve existing simple patterns.
- Explain any unavoidable non-trivial complexity in Chinese before adding it.

## PDDL Artifact Consistency

- After changing any `domain.pddl` or `problem.pddl`, solve the modified PDDL again and overwrite the corresponding `plan.txt` with the newly generated plan.
- The new `plan.txt` must be logically equivalent to the episode's `kf_actions.txt`, i.e., it must accomplish the same task. Otherwise, the PDDL modification is invalid.
- A PDDL change is complete only after solving succeeds and task equivalence is verified. Never manually rename actions in a stale plan.
- Preserve the requested round scope when regenerating plans; do not update older rounds when only the maximum round is in scope.
