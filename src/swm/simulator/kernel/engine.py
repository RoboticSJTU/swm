from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable, Protocol

from swm.pddl.strips import DomainSchemas, GroundAction, ProblemModel, goals_satisfied

from swm.simulator.actions import validate_physical_action
from swm.simulator.ir.mapping import map_ground_action
from swm.simulator.ir.model import (
    ActionFamily,
    CertificateCategory,
    VerificationStatus,
)
from swm.simulator.scene import SceneState, compile_scene

Literal = tuple[str, ...]


class GoalContract(Protocol):
    def evaluate(self, scene: SceneState, trace: tuple["TransitionRecord", ...]) -> tuple[VerificationStatus, str]: ...


@dataclass(frozen=True)
class Certificate:
    step: int
    action: str
    status: VerificationStatus
    category: CertificateCategory
    detail: str
    evidence: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TransitionRecord:
    step: int
    raw_action: str
    family: str
    rule_id: str
    added: tuple[Literal, ...]
    deleted: tuple[Literal, ...]


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    first_issue: Certificate | None
    trace: tuple[TransitionRecord, ...]
    final_state: SceneState | None
    coverage: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "first_issue": None
            if self.first_issue is None
            else {
                **asdict(self.first_issue),
                "status": self.first_issue.status.value,
                "category": self.first_issue.category.value,
                "evidence": dict(self.first_issue.evidence),
            },
            "trace": [asdict(item) for item in self.trace],
            "final_state": None if self.final_state is None else self.final_state.to_dict(),
            "coverage": dict(self.coverage),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _pddl_issue(state: frozenset[Literal], action: GroundAction, step: int) -> Certificate | None:
    missing = sorted(action.pre_pos - state)
    forbidden = sorted(action.pre_neg & state)
    unequal = sorted(item for item in action.equality_preconditions if item[1] != item[2])
    equal = sorted(item for item in action.inequality_preconditions if item[1] == item[2])
    if not (missing or forbidden or unequal or equal):
        return None
    evidence = (
        ("missing_positive", repr(missing)),
        ("present_negative", repr(forbidden)),
        ("unsatisfied_equality", repr(unequal)),
        ("violated_inequality", repr(equal)),
    )
    return Certificate(
        step,
        action.to_line(),
        VerificationStatus.FAIL,
        CertificateCategory.MISSING_PRECONDITION,
        "PDDL preconditions failed",
        evidence,
    )


def _remove_object_locations(facts: set[Literal], obj: str) -> None:
    removable = {
        fact
        for fact in facts
        if (fact[0] in {"on", "in", "inserted"} and len(fact) >= 3 and fact[1] == obj)
        or (fact[0] == "holding" and len(fact) >= 3 and fact[2] == obj)
    }
    facts.difference_update(removable)


def _enforce_state_pairs(facts: set[Literal], additions: set[Literal]) -> None:
    opposites = {
        "open": "closed",
        "closed": "open",
        "is_on": "is_off",
        "is_off": "is_on",
        "locked": "unlocked",
        "unlocked": "locked",
        "upright": "upside_down",
        "upside_down": "upright",
    }
    for literal in additions:
        if len(literal) == 2 and literal[0] in opposites:
            facts.discard((opposites[literal[0]], literal[1]))


def apply_canonical_transition(
    state: frozenset[Literal], action: GroundAction, family: ActionFamily
) -> frozenset[Literal]:
    next_facts = set(state)
    next_facts.difference_update(action.del_eff)
    next_facts.update(action.add_eff)
    mapped = map_ground_action(action)
    obj = mapped.role("object")
    if obj and family in {ActionFamily.PICK, ActionFamily.PLACE_ON, ActionFamily.PLACE_IN, ActionFamily.INSERT}:
        added_locations = {
            fact
            for fact in action.add_eff
            if (fact[0] in {"on", "in", "inserted"} and len(fact) >= 3 and fact[1] == obj)
            or (fact[0] == "holding" and len(fact) >= 3 and fact[2] == obj)
        }
        _remove_object_locations(next_facts, obj)
        next_facts.update(added_locations)
    _enforce_state_pairs(next_facts, action.add_eff)
    if family is ActionFamily.LOCK:
        target = mapped.role("target")
        hand = mapped.role("hand")
        if target:
            for argument in action.args:
                if argument not in {target, hand}:
                    next_facts.add(("__locked_component", target, argument))
    if family is ActionFamily.UNLOCK:
        target = mapped.role("target")
        if target:
            removable = {
                fact for fact in next_facts if fact[0] == "__locked_component" and fact[1] == target
            }
            next_facts.difference_update(removable)
    return frozenset(next_facts)


def _invariant_issue(scene: SceneState, step: int, action: GroundAction) -> Certificate | None:
    hard = [
        item
        for item in scene.diagnostics
        if item.category in {"multiple_location_parent", "multiple_location_kind", "relation_cycle", "contradictory_state"}
    ]
    if not hard:
        return None
    return Certificate(
        step,
        action.to_line(),
        VerificationStatus.FAIL,
        CertificateCategory.INCONSISTENT_TRANSITION,
        hard[0].detail,
        (("diagnostic", hard[0].category),),
    )


def verify(
    schemas: DomainSchemas,
    problem: ProblemModel,
    actions: Iterable[GroundAction],
    goal_contract: GoalContract | None = None,
) -> VerificationResult:
    initial = compile_scene(schemas, problem)
    if initial.diagnostics:
        diagnostic = initial.diagnostics[0]
        certificate = Certificate(
            0,
            "<initial-state>",
            VerificationStatus.UNKNOWN,
            CertificateCategory.UNRESOLVED_SCENE_FACT,
            diagnostic.detail,
            (("diagnostic", diagnostic.category),),
        )
        return VerificationResult(VerificationStatus.UNKNOWN, certificate, (), None, ())
    state = initial.facts
    trace: list[TransitionRecord] = []
    coverage: dict[str, int] = {}
    for step, raw_action in enumerate(actions, 1):
        pddl_issue = _pddl_issue(state, raw_action, step)
        if pddl_issue:
            return VerificationResult(VerificationStatus.FAIL, pddl_issue, tuple(trace), None, tuple(sorted(coverage.items())))
        mapped = map_ground_action(raw_action)
        guard_issue = validate_physical_action(compile_scene(schemas, problem, facts=state), raw_action, mapped, schemas)
        if guard_issue:
            certificate = Certificate(
                step,
                raw_action.to_line(),
                guard_issue.status,
                guard_issue.category,
                guard_issue.detail,
                guard_issue.evidence,
            )
            return VerificationResult(guard_issue.status, certificate, tuple(trace), None, tuple(sorted(coverage.items())))
        next_state = apply_canonical_transition(state, raw_action, mapped.family)
        next_scene = compile_scene(schemas, problem, facts=next_state)
        invariant_issue = _invariant_issue(next_scene, step, raw_action)
        if invariant_issue:
            return VerificationResult(VerificationStatus.FAIL, invariant_issue, tuple(trace), None, tuple(sorted(coverage.items())))
        rule_id = f"{mapped.family.value}.v1"
        coverage[rule_id] = coverage.get(rule_id, 0) + 1
        trace.append(
            TransitionRecord(
                step,
                raw_action.to_line(),
                mapped.family.value,
                rule_id,
                tuple(sorted(next_state - state)),
                tuple(sorted(state - next_state)),
            )
        )
        state = next_state
    final_scene = compile_scene(schemas, problem, facts=state)
    if goal_contract is not None:
        status, detail = goal_contract.evaluate(final_scene, tuple(trace))
        if status is not VerificationStatus.PASS:
            category = (
                CertificateCategory.FINAL_GOAL_NOT_ACHIEVED
                if status is VerificationStatus.FAIL
                else CertificateCategory.UNRESOLVED_SCENE_FACT
            )
            certificate = Certificate(len(trace) + 1, "<goal>", status, category, detail)
            return VerificationResult(status, certificate, tuple(trace), final_scene, tuple(sorted(coverage.items())))
    elif not goals_satisfied(set(state), problem.goal_positive, problem.goal_negative):
        certificate = Certificate(
            len(trace) + 1,
            "<goal>",
            VerificationStatus.FAIL,
            CertificateCategory.FINAL_GOAL_NOT_ACHIEVED,
            "PDDL final goal is not achieved",
        )
        return VerificationResult(VerificationStatus.FAIL, certificate, tuple(trace), final_scene, tuple(sorted(coverage.items())))
    return VerificationResult(VerificationStatus.PASS, None, tuple(trace), final_scene, tuple(sorted(coverage.items())))
