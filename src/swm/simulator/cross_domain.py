from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swm.simulator.alignment import (
    AlignmentResult,
    CanonicalStep,
    CanonicalWorld,
    CrossDomainResult,
    InstructionGoal,
    VLMAlignmentAdvisor,
    align_worlds,
    compile_instruction_goal,
    compile_step,
    compile_world,
    replay_on_gt,
)
from swm.simulator.corpus.inventory import (
    CandidateDomainRecovery,
    parse_candidate_domain_readonly,
    sha256_file,
)
from swm.pddl.strips import ground_plan, parse_domain, parse_plan, parse_problem_model

CROSS_DOMAIN_API_VERSION = "domain_logical_simulator_cross_domain_v3"


@dataclass(frozen=True)
class CrossDomainRun:
    gt_world: CanonicalWorld
    candidate_world: CanonicalWorld
    candidate_domain_recovery: CandidateDomainRecovery
    alignment: AlignmentResult
    mapping: dict[str, str]
    goal_contract: InstructionGoal
    steps: tuple[CanonicalStep, ...]
    result: CrossDomainResult
    identity_resolution: dict[str, Any] | None = None


def _live_location(
    facts: frozenset[tuple[str, ...]], obj: str
) -> tuple[str, str, str] | None:
    preferred = ("on", "in", "inserted", "against", "under")
    for predicate in preferred:
        relation = next(
            (
                item
                for item in facts
                if len(item) == 3 and item[0] == predicate and item[1] == obj
            ),
            None,
        )
        if relation is not None:
            return relation
    return None


def _resolve_initial_source_ambiguities(
    result: CrossDomainResult,
    replay_world: CanonicalWorld,
    steps: tuple[CanonicalStep, ...],
    instruction: str,
    alignment_advisor: VLMAlignmentAdvisor | None,
    image_path: Path | None,
) -> tuple[CrossDomainResult, list[dict[str, Any]]]:
    if alignment_advisor is None:
        return result, []
    overrides: set[tuple[int, str, str]] = set()
    records: list[dict[str, Any]] = []
    for _ in steps:
        issue = result.first_issue
        if (
            issue is None
            or issue.category != "source_mismatch"
            or not 1 <= issue.step <= len(steps)
        ):
            break
        step = steps[issue.step - 1]
        obj = step.role("object")
        requested = step.role("source")
        if obj is None or requested is None:
            break
        symbolic_relation = _live_location(result.final_facts, obj)
        if symbolic_relation is None or symbolic_relation not in replay_world.facts:
            break
        if symbolic_relation[0] not in {"on", "in", "inserted"}:
            break
        direct_source = symbolic_relation[2]
        indirect_outer_relations = {
            item
            for item in replay_world.facts
            if len(item) == 3
            and item[0] in {"on", "in", "inserted"}
            and item[1] == direct_source
            and item[2] == requested
        }
        if indirect_outer_relations:
            records.append(
                {
                    "step": issue.step,
                    "outcome": "FAIL",
                    "object": obj,
                    "requested_source": requested,
                    "symbolic_relation": list(symbolic_relation),
                    "reason": (
                        f"{requested} is an outer support of direct source "
                        f"{direct_source}, not the object's direct source"
                    ),
                }
            )
            break
        decision, record = alignment_advisor.resolve_initial_source_relation(
            replay_world,
            object_name=obj,
            requested_source=requested,
            symbolic_relation=symbolic_relation,
            relation_context=step.raw_action.split(maxsplit=1)[0].lstrip("("),
            image_path=image_path,
        )
        record = {"step": issue.step, **record}
        records.append(record)
        if decision is not True:
            break
        override = (issue.step, obj, requested)
        if override in overrides:
            break
        overrides.add(override)
        result = replay_on_gt(
            replay_world,
            steps,
            instruction=instruction,
            source_overrides=frozenset(overrides),
        )
    return result, records


def run_cross_domain_files(
    gt_domain_path: Path,
    gt_problem_path: Path,
    candidate_domain_path: Path,
    candidate_problem_path: Path,
    candidate_plan_path: Path,
    instruction: str,
    alignment_advisor: VLMAlignmentAdvisor | None = None,
    image_path: Path | None = None,
) -> CrossDomainRun:
    if not instruction or not instruction.strip():
        raise ValueError("instruction is required for cross-domain evaluation")
    gt_schemas = parse_domain(gt_domain_path)
    gt_problem = parse_problem_model(gt_problem_path, gt_schemas)
    gt_world = compile_world(gt_schemas, gt_problem)
    goal_contract = compile_instruction_goal(instruction, gt_world)
    if not goal_contract.positive and not goal_contract.negative:
        raise ValueError("instruction does not project to any supported GT goal")
    raw, _ = parse_plan(candidate_plan_path)
    candidate_schemas, candidate_domain_recovery = parse_candidate_domain_readonly(
        candidate_domain_path, (name for name, _ in raw)
    )
    candidate_problem = parse_problem_model(candidate_problem_path, candidate_schemas)
    candidate_world = compile_world(candidate_schemas, candidate_problem)
    grounded = ground_plan(raw, candidate_schemas, candidate_problem.objects)
    alignment = align_worlds(candidate_world, gt_world)
    if alignment_advisor:
        alignment = alignment_advisor.refine(
            candidate_world,
            gt_world,
            alignment,
            instruction=instruction,
            image_path=image_path,
        )
    mapping = alignment.mapping()
    added_objects = tuple(
        candidate_world.object(name) for name in alignment.scene_objects
    )
    if any(item is None for item in added_objects):
        raise ValueError("scene completion contains an unknown candidate object")
    replay_world = CanonicalWorld(
        gt_world.interface,
        gt_world.objects + added_objects,
        gt_world.facts | alignment.scene_facts,
        gt_world.negative_facts | alignment.scene_negative_facts,
        goal_contract.positive,
        goal_contract.negative,
        gt_world.schemas,
        gt_world.identity_facts,
    )
    ambiguous = {
        item.candidate for item in alignment.objects if item.status == "tentative"
    }
    steps = tuple(
        compile_step(
            action,
            candidate_world,
            mapping,
            reference_world=replay_world,
            ambiguous_objects=ambiguous,
        )
        for action in grounded
    )
    result = replay_on_gt(replay_world, steps, instruction=instruction)
    result, source_resolution = _resolve_initial_source_ambiguities(
        result,
        replay_world,
        steps,
        instruction,
        alignment_advisor,
        image_path,
    )
    return CrossDomainRun(
        gt_world,
        candidate_world,
        candidate_domain_recovery,
        alignment,
        mapping,
        goal_contract,
        steps,
        result,
        {"source_relations": source_resolution} if source_resolution else None,
    )


def verify_cross_domain_files(
    gt_domain_path: Path,
    gt_problem_path: Path,
    candidate_domain_path: Path,
    candidate_problem_path: Path,
    candidate_plan_path: Path,
    instruction: str,
    alignment_advisor: VLMAlignmentAdvisor | None = None,
    image_path: Path | None = None,
) -> dict[str, Any]:
    paths = {
        "gt_domain": gt_domain_path.resolve(),
        "gt_problem": gt_problem_path.resolve(),
        "candidate_domain": candidate_domain_path.resolve(),
        "candidate_problem": candidate_problem_path.resolve(),
        "candidate_plan": candidate_plan_path.resolve(),
    }
    run = run_cross_domain_files(
        paths["gt_domain"],
        paths["gt_problem"],
        paths["candidate_domain"],
        paths["candidate_problem"],
        paths["candidate_plan"],
        instruction,
        alignment_advisor,
        image_path,
    )
    return {
        "api_version": CROSS_DOMAIN_API_VERSION,
        "simulator_version": "1.7.0",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "predicate_interfaces": {
            "gt": [item.__dict__ for item in run.gt_world.interface.descriptions],
            "candidate": [item.__dict__ for item in run.candidate_world.interface.descriptions],
        },
        "candidate_domain_recovery": run.candidate_domain_recovery.to_dict(),
        "alignment": run.alignment.to_dict(),
        "identity_resolution": run.identity_resolution,
        "goal_contract": {
            "positive": [list(item) for item in sorted(run.goal_contract.positive)],
            "negative": [list(item) for item in sorted(run.goal_contract.negative)],
            "provenance": [
                {"literal": list(literal), "source": source}
                for literal, source in run.goal_contract.provenance
            ],
        },
        "canonical_plan": [
            {
                "raw_action": step.raw_action,
                "family": step.family.value,
                "roles": dict(step.roles),
                "placement_relation": step.placement_relation,
                "positive_preconditions": [list(item) for item in step.positive_preconditions],
                "negative_preconditions": [list(item) for item in step.negative_preconditions],
                "add_effects": [list(item) for item in step.add_effects],
                "delete_effects": [list(item) for item in step.delete_effects],
                "unmapped_objects": list(step.unmapped_objects),
                "ambiguous_objects": list(step.ambiguous_objects),
            }
            for step in run.steps
        ],
        "result": run.result.to_dict(),
    }
