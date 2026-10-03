# Project Instructions

## Working Principles

1. **Find the main bottleneck.** Use evidence and first principles to identify the constraint whose resolution most advances the goal and eases other problems. Reassess when blocked or evidence changes; verify the goal at completion.
2. **Make the smallest effective change.** Focus effort on that bottleneck, handle secondary issues as needed, and stay within the requested scope. Prefer local edits and existing simple patterns.
3. **Keep solutions simple and general.** Keep code, pipelines, and prompts concise; add complexity only for a demonstrated need. Fix root causes without speculative abstractions or special-case patches for specific tasks, objects, or test-set scores.
4. **Explain only what matters.** Lead with the main takeaway, then the core mechanism and necessary evidence. Match depth to the request, including paper and module explanations; avoid incidental details, exhaustive lists, and repetition.

## PDDL Artifact Consistency

- After changing `domain.pddl` or `problem.pddl`, re-solve it and overwrite the corresponding `plan.txt` with the generated plan. Never manually rename actions in a stale plan.
- A PDDL change is valid and complete only when solving succeeds and the new plan is verified to accomplish the same task as the episode's `kf_actions.txt`.
- Regenerate only the requested rounds; when only the maximum round is in scope, leave older rounds untouched.
