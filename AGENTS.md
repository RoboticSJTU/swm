# Project Instructions

## General

- Use evidence and first principles to prioritize the bottleneck whose resolution most advances the goal and eases other problems. Reassess when blocked or evidence changes; verify completion.
- Make minimal, scoped changes; prefer local edits and simple existing patterns. Keep code, pipelines, and prompts concise and general; address secondary issues as needed and add complexity only for demonstrated needs. Fix root causes; avoid speculative abstractions and patches for specific tasks, objects, or test scores.
- Lead with the takeaway, core mechanism, and evidence. Match depth to the request, including paper and module explanations; avoid incidental detail, exhaustive lists, and repetition.
- Complete tasks yourself; third-party API calls require explicit user authorization.
- Never use file hashes, checksum manifests, or hash snapshots.
- Keep nonessential artifacts, intermediate outputs, debug logs, and test files in `temp/`. Save final deliverables in their intended locations only when explicitly requested by the user or approved after asking; otherwise use `temp/`.

## PDDL

- Default all inspection, modification, and regeneration to the maximum round. Do not model noncausal ordering in task instructions.
- After editing `domain.pddl` or `problem.pddl`, solve optimally, overwrite the corresponding `plan.txt`, and verify it against the observed scene and current task instruction before considering the change complete. Ignore counterexamples based on suboptimal plans.
- Use `kf_actions.txt` as supporting evidence; require its exact action order, counts, temporary placements, or complete final state only when explicitly required by the instruction.
