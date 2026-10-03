from __future__ import annotations

import time
from pathlib import Path

from swm.simulator.cli import parse_cli

from .runner import evaluate_all, write_evaluation_report


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        root = Path(__file__).resolve().parents[4]
        argv = [str(root), str(root / "temp/domain_logical_simulator/reports")]
    paths, _, flags = parse_cli(argv, 2, flag_options={"skip-determinism"})
    started = time.monotonic()
    report = evaluate_all(
        Path(paths[0]), check_determinism="skip-determinism" not in flags
    )
    write_evaluation_report(report, Path(paths[1]))
    print(f"completed in {time.monotonic() - started:.3f}s")
    print(
        {
            "reference_status": report["reference"]["status"],
            "legal_status": report["legal_controls"]["status"],
            "invalid_status": report["invalid_controls"]["status"],
            "specific_semantic_coverage": report["coverage"]["specific_semantic_coverage"],
            "deterministic": report["deterministic"],
        }
    )


if __name__ == "__main__":
    main()
