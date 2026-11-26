# v7 reference runtime

This offline runtime contains the v7 game rules and the scene-generation
revisions used by the v11 abstract planning benchmark. It reconstructs tasks
and verifies game transitions, invariants, goals, and their PDDL projection.

`v7_contract.py` loads the runtime in a separate process and reads task
specifications from the installed benchmark scenarios.

The IPC domain templates, generator sources, and notices are in
`assets/pddl_generators`. The Python FloorTile generator is in
`floortile_generator.py`.
The supplied Miconic and Freecell executables are for Linux x86_64; their C
sources and Makefiles are included for rebuilding on other platforms.
