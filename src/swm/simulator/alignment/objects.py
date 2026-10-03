from __future__ import annotations

from dataclasses import dataclass

from .predicates import CanonicalWorld


@dataclass(frozen=True)
class ObjectAlignment:
    candidate: str
    gt: str | None
    status: str
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlignmentResult:
    objects: tuple[ObjectAlignment, ...]
    advisor: str | None = None
    cache_key: str | None = None
    scene_facts: frozenset[tuple[str, ...]] = frozenset()
    scene_negative_facts: frozenset[tuple[str, ...]] = frozenset()
    scene_objects: tuple[str, ...] = ()

    def mapping(self, *, include_tentative: bool = False) -> dict[str, str]:
        return {
            item.candidate: item.gt
            for item in self.objects
            if item.gt is not None
            and item.status != "unmapped"
            and (include_tentative or item.status in {"confident", "vlm"})
        }

    def to_dict(self) -> dict[str, object]:
        payload = {
            "mapped": len(self.mapping()),
            "proposed": len(self.mapping(include_tentative=True)),
            "objects": [item.__dict__ for item in self.objects],
        }
        if self.advisor is not None:
            payload["advisor"] = self.advisor
            payload["cache_key"] = self.cache_key
        if self.scene_objects:
            payload["added_scene_objects"] = list(self.scene_objects)
        if self.scene_facts or self.scene_negative_facts:
            payload["added_initial_facts"] = [list(item) for item in sorted(self.scene_facts)]
            payload["added_initial_negative_facts"] = [
                list(item) for item in sorted(self.scene_negative_facts)
            ]
        return payload


def align_worlds(
    candidate: CanonicalWorld,
    gt: CanonicalWorld,
    required_objects: set[str] | None = None,
) -> AlignmentResult:
    """Match shared identifiers and the sole declared hand in each scene."""
    del required_objects
    gt_names = {item.name for item in gt.objects}
    exact = {item.name for item in candidate.objects if item.name in gt_names}
    candidate_hands = [item.name for item in candidate.objects if "hand" in item.identity_kinds]
    gt_hands = [item.name for item in gt.objects if "hand" in item.identity_kinds]
    hand_match = (
        {candidate_hands[0]: gt_hands[0]}
        if len(candidate_hands) == len(gt_hands) == 1
        and candidate_hands[0] not in exact and gt_hands[0] not in exact
        else {}
    )
    matches = {name: name for name in exact} | hand_match
    return AlignmentResult(tuple(
        ObjectAlignment(
            source.name,
            matches.get(source.name),
            "confident" if source.name in matches else "tentative",
        )
        for source in candidate.objects
    ))
