from __future__ import annotations

from dataclasses import dataclass

from swm.pddl.strips import DomainSchemas, GroundAction
from swm.simulator.alignment.predicates import PredicateInterface

from swm.simulator.ir.model import (
    ActionFamily,
    Capability,
    CanonicalAction,
    CertificateCategory,
    Closure,
    LockState,
    Pose,
    EvidenceValue,
    VerificationStatus,
)
from swm.simulator.scene import SceneState


@dataclass(frozen=True)
class GuardIssue:
    status: VerificationStatus
    category: CertificateCategory
    detail: str
    evidence: tuple[tuple[str, str], ...] = ()


def validate_physical_action(
    scene: SceneState,
    raw: GroundAction,
    action: CanonicalAction,
    schemas: DomainSchemas,
) -> GuardIssue | None:
    category_predicates = {
        item.raw_name for item in PredicateInterface.from_schemas(schemas).descriptions
        if item.static and item.arity == 1
    }

    def has_state_model(spec, predicates: set[str], *, preconditions_only=False) -> bool:
        for schema in schemas.values():
            literals = schema.pre_pos if preconditions_only else schema.pre_pos | schema.add_eff | schema.del_eff
            for literal in literals:
                if len(literal) != 2 or literal[0] not in predicates:
                    continue
                required = {
                    guard[0] for guard in schema.pre_pos
                    if len(guard) == 2 and guard[1] == literal[1] and guard[0] in category_predicates
                }
                if required and required <= set(spec.categories):
                    return True
        return False

    family = action.family
    target = action.role("target")
    obj = action.role("object")

    if family is ActionFamily.UNSUPPORTED:
        return GuardIssue(
            VerificationStatus.UNKNOWN,
            CertificateCategory.UNSUPPORTED_ACTION,
            f"no deterministic semantics for {action.raw_name}",
        )
    if family is ActionFamily.PICK and obj:
        if not scene.accessible(obj):
            return GuardIssue(
                VerificationStatus.FAIL,
                CertificateCategory.CLOSED_OR_LOCKED_ANCESTOR,
                f"{obj} is inside a closed or locked ancestor",
                (("object", obj),),
            )
        children = scene.children_on(obj)
        if children and ("clear", obj) in raw.pre_pos:
            return GuardIssue(
                VerificationStatus.FAIL,
                CertificateCategory.BLOCKED_NOT_CLEAR,
                f"{obj} supports blocking children",
                (("children", ",".join(children)),),
            )
    if family in {ActionFamily.PLACE_ON, ActionFamily.PLACE_IN, ActionFamily.INSERT} and target:
        if not scene.accessible(target):
            return GuardIssue(
                VerificationStatus.FAIL,
                CertificateCategory.CLOSED_OR_LOCKED_ANCESTOR,
                f"target {target} is inside a closed or locked ancestor",
                (("target", target),),
            )
        if family is ActionFamily.PLACE_IN:
            closure = scene.closure(target)
            if closure is Closure.CLOSED:
                return GuardIssue(
                    VerificationStatus.FAIL,
                    CertificateCategory.CLOSED_OR_LOCKED_ANCESTOR,
                    f"container {target} is closed",
                    (("target", target),),
                )
            target_spec = scene.object(target)
            if (
                target_spec is not None
                and has_state_model(target_spec, {"clear"}, preconditions_only=True)
                and scene.children_on(target)
            ):
                return GuardIssue(
                    VerificationStatus.FAIL,
                    CertificateCategory.OCCUPIED_TARGET,
                    f"target {target} is obstructed by a support child",
                    (("children", ",".join(scene.children_on(target))),),
                )
    if family is ActionFamily.OPEN and target:
        if scene.lock(target) is LockState.LOCKED:
            return GuardIssue(
                VerificationStatus.FAIL,
                CertificateCategory.CLOSED_OR_LOCKED_ANCESTOR,
                f"{target} is locked",
                (("target", target),),
            )
        locked_by = sorted(
            fact[1]
            for fact in scene.facts
            if fact[0] == "__locked_component" and fact[2] == target
        )
        if locked_by:
            return GuardIssue(
                VerificationStatus.FAIL,
                CertificateCategory.CLOSED_OR_LOCKED_ANCESTOR,
                f"{target} belongs to locked assembly {locked_by[0]}",
                (("lockable", locked_by[0]),),
            )
        blockers = sorted(
            fact[1]
            for fact in scene.facts
            if fact[0] == "blocks_opening" and len(fact) >= 3 and fact[2] == target
        )
        target_spec = scene.object(target)
        children = (
            scene.children_on(target)
            if target_spec is not None
            and target_spec.capability(Capability.OPENING_SURFACE) is EvidenceValue.KNOWN
            else ()
        )
        if blockers or children:
            return GuardIssue(
                VerificationStatus.FAIL,
                CertificateCategory.BLOCKED_NOT_CLEAR,
                f"{target} is blocked or not clear",
                (("blockers", ",".join([*blockers, *children])),),
            )
    if family is ActionFamily.CLOSE and target:
        blockers = sorted(
            fact[1]
            for fact in scene.facts
            if fact[0] == "blocks_closing" and len(fact) >= 3 and fact[2] == target
        )
        if blockers:
            return GuardIssue(
                VerificationStatus.FAIL,
                CertificateCategory.BLOCKED_NOT_CLEAR,
                f"{target} has a closing blocker",
                (("blockers", ",".join(blockers)),),
            )
    if family is ActionFamily.REMOVE_CLOSURE:
        closure = action.role("closure")
        vessel = action.role("vessel")
        blocking = []
        if closure:
            blocking.extend(scene.children_on(closure))
        if vessel:
            blocking.extend(child for child in scene.children_on(vessel) if child != closure)
        if blocking:
            return GuardIssue(
                VerificationStatus.FAIL,
                CertificateCategory.BLOCKED_NOT_CLEAR,
                "closure or vessel has an obstructing support child",
                (("blockers", ",".join(sorted(set(blocking)))),),
            )
    if family is ActionFamily.POUR:
        receiver = action.role("receiver")
        if receiver:
            if not scene.accessible(receiver):
                return GuardIssue(
                    VerificationStatus.FAIL,
                    CertificateCategory.CLOSED_OR_LOCKED_ANCESTOR,
                    f"receiver {receiver} is inside a closed or locked ancestor",
                    (("receiver", receiver),),
                )
            receiver_spec = scene.object(receiver)
            if receiver_spec and has_state_model(receiver_spec, {"upright", "upside_down"}):
                pose = scene.pose(receiver)
                if pose is Pose.UPSIDE_DOWN:
                    return GuardIssue(
                        VerificationStatus.FAIL,
                        CertificateCategory.INVALID_POSE,
                        f"receiver {receiver} is upside down",
                        (("receiver", receiver), ("pose", pose.value)),
                    )
                if pose is Pose.UNKNOWN:
                    return GuardIssue(
                        VerificationStatus.UNKNOWN,
                        CertificateCategory.UNRESOLVED_SCENE_FACT,
                        f"receiver pose is unknown for {receiver}",
                        (("receiver", receiver),),
                    )
    return None
