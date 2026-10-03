from __future__ import annotations

from pathlib import Path

from swm.simulator.cli import parse_cli

from .inventory import build_inventory, write_inventory_reports


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        root = Path(__file__).resolve().parents[4]
        argv = [
            str(root / "eval_results/gpt-5.6-sol/human"),
            str(root / "temp/domain_logical_simulator/reports"),
        ]
    paths, _, _ = parse_cli(argv, 2)
    report = build_inventory(Path(paths[0]))
    write_inventory_reports(report, Path(paths[1]))
    print(report["counts"])


if __name__ == "__main__":
    main()
