from __future__ import annotations

import re
from dataclasses import dataclass

from swm.pddl.strips import ProblemModel

from swm.simulator.ir.model import VerificationStatus
from swm.simulator.kernel.engine import TransitionRecord
from swm.simulator.scene import SceneState

Literal = tuple[str, ...]


@dataclass(frozen=True)
class GoalClause:
    literal: Literal
    negated: bool = False
    provenance: str = ""


@dataclass(frozen=True)
class OrderConstraint:
    before_family: str
    after_family: str
    provenance: str = "instruction"


@dataclass(frozen=True)
class PDDLGoalContract:
    positive: tuple[Literal, ...]
    negative: tuple[Literal, ...]

    def evaluate(
        self, scene: SceneState, trace: tuple[TransitionRecord, ...]
    ) -> tuple[VerificationStatus, str]:
        missing = sorted(set(self.positive) - scene.facts)
        violated = sorted(set(self.negative) & scene.facts)
        if missing or violated:
            return (
                VerificationStatus.FAIL,
                f"missing positive goals={missing}; violated negative goals={violated}",
            )
        return VerificationStatus.PASS, "all supported PDDL goals are satisfied"


@dataclass(frozen=True)
class TaskContract:
    final_clauses: tuple[GoalClause, ...] = ()
    required_families: tuple[str, ...] = ()
    order_constraints: tuple[OrderConstraint, ...] = ()
    unsupported_clauses: tuple[str, ...] = ()

    def evaluate(
        self, scene: SceneState, trace: tuple[TransitionRecord, ...]
    ) -> tuple[VerificationStatus, str]:
        for clause in self.final_clauses:
            present = clause.literal in scene.facts
            if present == clause.negated:
                return VerificationStatus.FAIL, f"final clause not achieved: {clause.literal}"
        families = [item.family for item in trace]
        for required in self.required_families:
            if required not in families:
                return VerificationStatus.FAIL, f"required milestone missing: {required}"
        for constraint in self.order_constraints:
            before = [index for index, family in enumerate(families) if family == constraint.before_family]
            after = [index for index, family in enumerate(families) if family == constraint.after_family]
            if not before or not after or min(before) >= max(after):
                return (
                    VerificationStatus.FAIL,
                    f"order not achieved: {constraint.before_family} before {constraint.after_family}",
                )
        if self.unsupported_clauses:
            return (
                VerificationStatus.UNKNOWN,
                f"unsupported instruction clauses: {list(self.unsupported_clauses)}",
            )
        return VerificationStatus.PASS, "all supported task-contract clauses are satisfied"


def compile_pddl_goal(problem: ProblemModel) -> PDDLGoalContract:
    return PDDLGoalContract(
        tuple(sorted(problem.goal_positive)), tuple(sorted(problem.goal_negative))
    )


def _object_in_text(text: str, objects: tuple[str, ...]) -> str | None:
    normalized = text.replace(" ", "_")
    matches = [name for name in objects if name in normalized]
    return max(matches, key=len) if matches else None


def _family(phrase: str) -> str | None:
    phrase = phrase.strip().lower()
    patterns = (
        (r"\b(open|unscrew|remove)\b", "open"),
        (r"\b(close|screw|replace)\b", "close"),
        (r"\b(pick|take|lift)\b", "pick"),
        (r"\b(place|put)\b.*\b(in|inside|into)\b", "place_in"),
        (r"\b(place|put)\b", "place_on"),
        (r"\b(pour)\b", "pour"),
        (r"\bturn on\b", "turn_on"),
        (r"\bturn off\b", "turn_off"),
    )
    return next((family for pattern, family in patterns if re.search(pattern, phrase)), None)


def compile_instruction(instruction: str, object_names: tuple[str, ...]) -> TaskContract:
    """Compile only explicit, grounded patterns; preserve everything else as unknown."""
    text = " ".join(instruction.lower().split())
    if not text:
        return TaskContract()
    segments = [item.strip(" ,.") for item in re.split(r"\bthen\b|\bfinally\b", text) if item.strip(" ,.")]
    families = [family for segment in segments if (family := _family(segment))]
    order = tuple(
        OrderConstraint(left, right, instruction)
        for left, right in zip(families, families[1:])
    )
    final: list[GoalClause] = []
    final_segment = segments[-1] if segments else text
    obj = _object_in_text(final_segment, object_names)
    if obj:
        if re.search(r"\bopen\b", final_segment):
            final.append(GoalClause(("open", obj), provenance=instruction))
        elif re.search(r"\bclose[de]?\b", final_segment):
            final.append(GoalClause(("closed", obj), provenance=instruction))
        elif re.search(r"\bturn on\b", final_segment):
            final.append(GoalClause(("is_on", obj), provenance=instruction))
        elif re.search(r"\bturn off\b", final_segment):
            final.append(GoalClause(("is_off", obj), provenance=instruction))
    unsupported = () if families or final else (instruction,)
    return TaskContract(
        final_clauses=tuple(final),
        required_families=tuple(dict.fromkeys(families)),
        order_constraints=order,
        unsupported_clauses=unsupported,
    )
