from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("results", "rows", "cases"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
    raise ValueError("shadow input must contain a list of result rows")


def _compare_rows(
    dag_rows: list[dict[str, Any]], simulator_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    def identity(row: dict[str, Any]) -> str:
        if row.get("task_id") is not None:
            return f"task_{row['task_id']}"
        if "case_id" in row:
            return str(row["case_id"])
        if "case" in row:
            return str(row["case"])
        raise ValueError(f"result row has no identity: {row}")

    def normalize(status: object) -> str:
        value = str(status).upper()
        return {
            "BOUNDED_PASS": "PASS",
            "COUNTEREXAMPLE": "FAIL",
            "VULNERABLE": "FAIL",
        }.get(value, value)

    dag = {identity(row): normalize(row.get("status", row.get("verdict", "UNKNOWN"))) for row in dag_rows}
    logical = {identity(row): str(row.get("status", "UNKNOWN")).upper() for row in simulator_rows}
    shared = sorted(set(dag) & set(logical))
    agreements = [case for case in shared if dag[case] == logical[case]]
    disagreements = [
        {"case": case, "dag": dag[case], "logical_simulator": logical[case]}
        for case in shared
        if dag[case] != logical[case]
    ]
    return {
        "schema_version": "domain_logical_simulator_shadow_v1",
        "shared_cases": len(shared),
        "agreements": len(agreements),
        "agreement_rate": len(agreements) / len(shared) if shared else None,
        "disagreements": disagreements,
        "dag_only": sorted(set(dag) - set(logical)),
        "simulator_only": sorted(set(logical) - set(dag)),
    }


def compare_shadow(dag_result_path: Path, simulator_report_path: Path) -> dict[str, Any]:
    dag_rows = _rows(json.loads(dag_result_path.read_text(encoding="utf-8")))
    simulator = json.loads(simulator_report_path.read_text(encoding="utf-8"))
    simulator_rows = simulator.get("reference", {}).get("rows", [])
    return _compare_rows(dag_rows, simulator_rows)


def compare_shadow_directory(
    dag_cases_dir: Path, simulator_report_path: Path
) -> dict[str, Any]:
    rows = []
    for path in sorted(dag_cases_dir.glob("*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "case_id": value.get("case_id", path.parent.name),
                "task_id": value.get("task_id"),
                "status": value.get("status", "UNKNOWN"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no result.json files below {dag_cases_dir}")
    simulator = json.loads(simulator_report_path.read_text(encoding="utf-8"))
    simulator_rows = simulator.get("reference", {}).get("rows", [])
    return _compare_rows(rows, simulator_rows)
