from __future__ import annotations

import json
from pathlib import Path

from .cli import build_vlm_advisor, parse_cli
from .cross_domain import verify_cross_domain_files


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        root = Path(__file__).resolve().parents[3]
        gt = root / "eval_results/gt/swm/task_36/round1"
        candidate = root / "eval_results/test/9B_sft/swm/task_36"
        instructions = json.loads(
            (root / "tasks/instructions/instructions_swm.json").read_text(encoding="utf-8")
        )["swm"]
        argv = [
            "--gt-domain", str(gt / "domain.pddl"),
            "--gt-problem", str(gt / "problem.pddl"),
            "--candidate-domain", str(candidate / "domain.pddl"),
            "--candidate-problem", str(candidate / "problem.pddl"),
            "--candidate-plan", str(candidate / "plan.txt"),
            "--instruction", instructions["task_36"],
        ]
    _, options, flags = parse_cli(
        argv,
        0,
        {
            "gt-domain", "gt-problem", "candidate-domain", "candidate-problem",
            "candidate-plan", "instruction", "image", "vlm-model", "vlm-base-url",
            "vlm-cache", "env-file", "vlm-api-key-env", "vlm-reasoning-effort", "output",
        },
        {"vlm-mapping", "vlm-cache-only", "vlm-json-mode"},
    )
    required = (
        "gt-domain", "gt-problem", "candidate-domain", "candidate-problem",
        "candidate-plan",
    )
    missing = [name for name in required if name not in options]
    if "instruction" not in options or not options["instruction"].strip():
        missing.append("instruction")
    if missing:
        raise SystemExit("missing required options: " + ", ".join(f"--{name}" for name in missing))
    if "vlm-mapping" in flags and "image" not in options:
        raise SystemExit("--image is required with --vlm-mapping")
    advisor = build_vlm_advisor(options, flags)
    payload = verify_cross_domain_files(
        *(Path(options[name]) for name in required),
        instruction=options["instruction"],
        alignment_advisor=advisor,
        image_path=Path(options["image"]) if "image" in options else None,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if "output" in options:
        output = Path(options["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    status = payload["result"]["status"]
    return 0 if status == "PASS" else 1 if status == "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
