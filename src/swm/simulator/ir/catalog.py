from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import Counter

from swm.simulator.corpus.inventory import load_case, select_highest_rounds

from .mapping import map_ground_action

from .model import ActionFamily, CertificateCategory


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    family: ActionFamily
    activation_guard: str
    hard_conditions: tuple[str, ...]
    effects: tuple[str, ...]
    missing_evidence: str
    provenance: tuple[str, ...]
    failure_category: CertificateCategory


REVIEWED_RULES = (
    RuleSpec(
        "pick.v1",
        ActionFamily.PICK,
        "mapped family is pick",
        ("hand is free", "object is at its stated location", "object is accessible", "object has no support child"),
        ("detach old location", "set location to held(hand)"),
        "unknown capability -> UNKNOWN; contradicted state -> FAIL",
        ("generic manipulation invariant", "human300 structural observations"),
        CertificateCategory.MISSING_PRECONDITION,
    ),
    RuleSpec(
        "place_on.v1",
        ActionFamily.PLACE_ON,
        "mapped family is place_on",
        ("object is held", "target is accessible", "known single slot is free"),
        ("clear old location", "set location to on(target)"),
        "unknown capacity is ignored, not treated as unlimited",
        ("exclusive-location invariant", "human300 on-effects"),
        CertificateCategory.OCCUPIED_TARGET,
    ),
    RuleSpec(
        "place_in.v1",
        ActionFamily.PLACE_IN,
        "mapped family is place_in",
        ("object is held", "openable target is open", "known single slot is free"),
        ("clear old location", "set location to in(target)"),
        "unknown capacity is ignored; unknown closure -> UNKNOWN",
        ("exclusive-location invariant", "human300 in-effects"),
        CertificateCategory.CLOSED_OR_LOCKED_ANCESTOR,
    ),
    RuleSpec(
        "open.v1",
        ActionFamily.OPEN,
        "mapped family is open",
        ("target is closed", "target is unlocked when lock evidence exists", "opening blockers absent", "target surface is clear"),
        ("closure := open",),
        "unknown lock without lock evidence is irrelevant",
        ("state-pair invariant", "blocks_opening observations"),
        CertificateCategory.BLOCKED_NOT_CLEAR,
    ),
    RuleSpec(
        "close.v1",
        ActionFamily.CLOSE,
        "mapped family is close",
        ("target is open", "closing blockers absent"),
        ("closure := closed",),
        "missing blocker relation is irrelevant",
        ("state-pair invariant", "blocks_closing observations"),
        CertificateCategory.MISSING_PRECONDITION,
    ),
    RuleSpec(
        "pour.v1",
        ActionFamily.POUR,
        "mapped family is pour",
        ("source content is available", "receiver is accessible", "declared receiver pose is valid"),
        ("transfer explicit content markers",),
        "unknown quantity remains unknown",
        ("content conservation invariant", "human300 pour observations"),
        CertificateCategory.INVALID_POSE,
    ),
)


def write_catalog_artifacts(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "domain_logical_simulator_rule_catalog_v1",
        "rules": [
            {
                **asdict(rule),
                "family": rule.family.value,
                "failure_category": rule.failure_category.value,
            }
            for rule in REVIEWED_RULES
        ],
    }
    (report_dir / "rule_catalog.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# Reviewed Rule Catalog", ""]
    for rule in REVIEWED_RULES:
        lines.extend(
            [
                f"## `{rule.rule_id}`",
                "",
                f"- Family: `{rule.family.value}`",
                f"- Activation: {rule.activation_guard}",
                f"- Missing evidence: {rule.missing_evidence}",
                f"- Provenance: {', '.join(rule.provenance)}",
                "",
            ]
        )
    (report_dir / "RULE_CATALOG_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_mapping_audit(corpus_root: Path, report_dir: Path) -> dict[str, object]:
    family_steps: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    unsupported: list[dict[str, object]] = []
    total = 0
    for case in select_highest_rounds(corpus_root):
        _, _, plan = load_case(case)
        for step, action in enumerate(plan, 1):
            mapped = map_ground_action(action)
            family_steps[mapped.family.value] += 1
            reasons[mapped.mapping_reason] += 1
            total += 1
            if mapped.family is ActionFamily.UNSUPPORTED:
                unsupported.append(
                    {"task_id": case.task_id, "step": step, "action": action.to_line()}
                )
    generic = family_steps[ActionFamily.GENERIC_PDDL.value]
    payload = {
        "schema_version": "domain_logical_simulator_mapping_audit_v1",
        "total_steps": total,
        "mapped_steps": total - len(unsupported),
        "mapping_coverage": (total - len(unsupported)) / total,
        "canonical_family_steps": total - generic - len(unsupported),
        "canonical_family_coverage": (total - generic - len(unsupported)) / total,
        "family_steps": dict(sorted(family_steps.items())),
        "mapping_reasons": dict(sorted(reasons.items())),
        "unsupported": unsupported,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "action_mapping_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_dir / "ACTION_MAPPING_AUDIT.md").write_text(
        "\n".join(
            [
                "# Action Mapping Audit",
                "",
                f"- Total reference steps: {total}",
                f"- Mapped steps: {payload['mapped_steps']} ({payload['mapping_coverage']:.2%})",
                f"- Specific canonical family: {payload['canonical_family_steps']} ({payload['canonical_family_coverage']:.2%})",
                f"- Explicit generic PDDL transitions: {generic}",
                f"- Unsupported: {len(unsupported)}",
                "",
                "Mapping uses transition structure before action-name fallback. Generic PDDL",
                "steps retain exact PDDL preconditions/effects but do not claim a stronger",
                "physical model. No task or object identifiers appear in runtime rules.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload
