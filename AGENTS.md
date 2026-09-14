# Project Working Style

Prefer minimal, direct, and easy-to-understand solutions.

- Add only what is necessary. Prefer local changes over new abstractions, helpers, configuration, parameters, files, or pipeline stages.
- Keep changes scoped to the request and preserve existing simple patterns. Do not generalize for hypothetical future needs.
- Keep code, pipelines, and prompts as simple as possible. Add complexity only when a concrete failure demonstrates the need.

## PDDL Artifact Consistency

- After changing any `domain.pddl` or `problem.pddl`, solve the modified PDDL again and overwrite the corresponding `plan.txt` with the newly generated plan.
- The new `plan.txt` must be logically equivalent to the episode's `kf_actions.txt`, i.e., it must accomplish the same task. Otherwise, the PDDL modification is invalid.
- A PDDL change is complete only after solving succeeds and task equivalence is verified. Never manually rename actions in a stale plan.
- Preserve the requested round scope when regenerating plans; do not update older rounds when only the maximum round is in scope.

## Subagent Delegation

Use subagents only when delegation is expected to reduce primary-agent context or reasoning. Do not delegate trivial work whose coordination overhead is likely to exceed the savings.

Use the cheapest model reliably capable of the task:

* `gpt-5.6-luna` with `reasoning_effort=max`: Handle mechanical and high-volume work such as file and repository search, locating definitions, collecting logs, running known commands, extracting structured facts, simple transformations, and organizing results. Do not use it for critical judgment.
* `gpt-5.6-sol` with `reasoning_effort=high`: Handle substantive technical work such as reading and tracing code, analyzing failures, modifying code, running and interpreting experiments, root-cause analysis, solution design, and comparing alternatives.
* Primary agent (`gpt-6`): Handle task decomposition, global reasoning, architectural decisions, consequential tradeoffs, experiment strategy, integration, conflict resolution, and final decisions. Avoid doing routine exploration, implementation, experimentation, or result organization when these can be delegated reliably.

Prefer one well-scoped owner per subtask. Run subagents concurrently only for substantial independent work, and avoid redundant parallel attempts unless independent verification is decision-relevant. Subagents must not spawn further subagents unless explicitly directed by the primary agent.

Give subagents only the context necessary for their task: objective, relevant scope or files, constraints, and acceptance criteria. Require compact, decision-useful outputs containing conclusions, concrete evidence, relevant file locations or changes, test results when applicable, and unresolved uncertainties.

The primary agent should verify delegated work selectively rather than repeat it. Re-investigate only when evidence is insufficient or conflicting, the result affects a critical decision, or the change is high-risk. Deterministic evidence and successful tests normally do not require reproducing the full investigation.

For implementation and experimentation, prefer the smallest decisive change or test, broaden only when necessary, and stop once enough evidence exists to make the next decision.

Optimize for minimum total expensive-model reasoning and primary-agent context consumption while preserving correctness. Do not spend additional inference merely for parallelism, exhaustive exploration, or redundant verification.
