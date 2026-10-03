from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvidenceValue(str, Enum):
    KNOWN = "known"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class ActionFamily(str, Enum):
    PICK = "pick"
    PLACE_ON = "place_on"
    PLACE_IN = "place_in"
    OPEN = "open"
    CLOSE = "close"
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    POUR = "pour"
    REMOVE_CLOSURE = "remove_closure"
    REPLACE_CLOSURE = "replace_closure"
    INSERT = "insert"
    SCOOP = "scoop"
    LOCK = "lock"
    UNLOCK = "unlock"
    WIPE = "wipe"
    STIR = "stir"
    CUT = "cut"
    WASH = "wash"
    FOLD = "fold"
    SCRUNCH = "scrunch"
    FILL = "fill"
    WET = "wet"
    EMPTY = "empty"
    SWEEP = "sweep"
    ORIENT = "orient"
    CONFIGURE = "configure"
    ACTIVATE = "activate"
    SLIDE = "slide"
    PUSH = "push"
    UNBLOCK = "unblock"
    GENERIC_PDDL = "generic_pddl"
    UNSUPPORTED = "unsupported"


class Capability(str, Enum):
    PICKABLE = "pickable"
    SUPPORT = "support"
    CONTAINER = "container"
    OPENABLE = "openable"
    LOCKABLE = "lockable"
    DEVICE = "device"
    POURER = "pourer"
    RECEIVER = "receiver"
    CLOSURE = "closure"
    TOOL = "tool"
    SLOT_HOLDER = "slot_holder"
    OPENING_SURFACE = "opening_surface"


class LocationKind(str, Enum):
    HELD = "held"
    ON = "on"
    IN = "in"
    NONE = "none"


@dataclass(frozen=True)
class Location:
    kind: LocationKind
    parent: str | None = None


class Pose(str, Enum):
    UPRIGHT = "upright"
    UPSIDE_DOWN = "upside_down"
    FLAT = "flat"
    VERTICAL = "vertical"
    UNKNOWN = "unknown"


class Closure(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class Power(str, Enum):
    ON = "on"
    OFF = "off"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class LockState(str, Enum):
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class CertificateCategory(str, Enum):
    MISSING_PRECONDITION = "missing_precondition"
    OCCUPIED_HAND = "occupied_hand"
    BLOCKED_NOT_CLEAR = "blocked_not_clear"
    CLOSED_OR_LOCKED_ANCESTOR = "closed_or_locked_ancestor"
    OCCUPIED_TARGET = "occupied_target"
    INVALID_POSE = "invalid_pose"
    UNSUPPORTED_ACTION = "unsupported_action"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    INCONSISTENT_TRANSITION = "inconsistent_transition"
    UNRESOLVED_SCENE_FACT = "unresolved_scene_fact"
    FINAL_GOAL_NOT_ACHIEVED = "final_goal_not_achieved"


@dataclass(frozen=True)
class Provenance:
    kind: str
    source: str
    detail: str = ""


@dataclass(frozen=True)
class CanonicalAction:
    family: ActionFamily
    raw_name: str
    arguments: tuple[str, ...]
    roles: tuple[tuple[str, str], ...]
    provenance: tuple[Provenance, ...] = field(default_factory=tuple)
    mapping_reason: str = ""

    def role(self, name: str) -> str | None:
        return next((value for role, value in self.roles if role == name), None)
