from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .objects import AlignmentResult
from .predicates import CanonicalObjectDescription, CanonicalWorld, Literal, normalized_tokens

IDENTITY_PROMPT_VERSION = "canonical-object-mapping-v12-task-equivalence"
SOURCE_RELATION_PROMPT_VERSION = "initial-source-relation-v9-unqualified-fixture"
DEFAULT_BASE_URL = "https://apicz.boyuerichdata.com/v1"


class VLMAlignmentError(RuntimeError):
    """The required VLM mapping could not be completed safely."""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _object_payload(item: CanonicalObjectDescription) -> dict[str, object]:
    return {
        "name": item.name,
        "declared_kinds": list(item.identity_kinds),
        "initial_states": list(item.state_tags),
    }


def _relation_facts(world: CanonicalWorld, names: set[str]) -> list[list[str]]:
    facts = world.identity_facts or world.facts
    return [
        list(item)
        for item in sorted(facts)
        if len(item) >= 3 and any(argument in names for argument in item[1:])
    ]


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("VLM response is not a JSON object")
    return value


class VLMAlignmentAdvisor:
    """Globally map different names and verify missing initial-scene objects."""

    def __init__(
        self,
        *,
        model: str = "qwen3.7-plus",
        api_key: str | None = None,
        api_key_env: str = "BOYUE_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Path,
        env_file: Path | None = None,
        timeout: float = 120.0,
        allow_network: bool = True,
        require_complete: bool = False,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
    ) -> None:
        if env_file is not None:
            load_env_file(env_file)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.allow_network = allow_network
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env) or ""
        if self.allow_network and not self.api_key:
            raise ValueError(f"{api_key_env} is not set")
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.require_complete = require_complete
        self._guard = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._stats: Counter[str] = Counter()

    def _increment(self, key: str, value: int = 1) -> None:
        with self._guard:
            self._stats[key] += value

    def stats(self) -> dict[str, int]:
        with self._guard:
            return dict(sorted(self._stats.items()))

    def _key_lock(self, key: str) -> threading.Lock:
        with self._guard:
            return self._key_locks.setdefault(key, threading.Lock())

    def _request(self, prompt: str, image_path: Path | None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path is not None and image_path.exists():
            mime, image_bytes = self._request_image(image_path)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                }
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 4000,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = payload["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("VLM provider returned empty content")
                result = _json_object(text)
                break
            except urllib.error.HTTPError:
                raise
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        return result

    @staticmethod
    def _request_image(image_path: Path) -> tuple[str, bytes]:
        """Bound VLM payload size while preserving the original cache identity."""
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((2048, 2048))
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=85, optimize=True)
                return "image/jpeg", output.getvalue()
        except Exception:
            mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            return mime, image_path.read_bytes()

    def _cached_request(
        self,
        prompt: str,
        image_path: Path | None,
        *,
        prompt_version: str = IDENTITY_PROMPT_VERSION,
    ) -> tuple[str, dict[str, Any]]:
        digest = hashlib.sha256()
        digest.update(prompt_version.encode("utf-8"))
        digest.update(self.model.encode("utf-8"))
        digest.update((self.reasoning_effort or "").encode("utf-8"))
        digest.update(str(self.json_mode).encode("ascii"))
        digest.update(prompt.encode("utf-8"))
        if image_path is not None and image_path.exists():
            digest.update(image_path.read_bytes())
        key = digest.hexdigest()
        path = self.cache_dir / f"{key}.json"
        with self._key_lock(key):
            if path.exists():
                self._increment("cache_hits")
                return key, json.loads(path.read_text(encoding="utf-8"))
            if not self.allow_network:
                self._increment("cache_misses")
                raise FileNotFoundError(f"VLM cache entry not found: {key}")
            self._increment("network_calls")
            response = self._request(prompt, image_path)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.cache_dir, delete=False
            ) as handle:
                json.dump(response, handle, indent=2, sort_keys=True)
                handle.write("\n")
                temporary = Path(handle.name)
            temporary.replace(path)
            return key, response

    def refine(
        self,
        candidate: CanonicalWorld,
        gt: CanonicalWorld,
        alignment: AlignmentResult,
        *,
        required_objects: set[str] | None = None,
        actions: tuple[str, ...] = (),
        instruction: str | None = None,
        image_path: Path | None = None,
        _repair_response: dict[str, Any] | None = None,
    ) -> AlignmentResult:
        del actions, required_objects  # A candidate plan cannot establish initial identity.
        by_candidate = {item.candidate: item for item in alignment.objects}
        candidate_hands = {
            item.name for item in candidate.objects if "hand" in item.identity_kinds
        }
        gt_hands = {item.name for item in gt.objects if "hand" in item.identity_kinds}
        unresolved = {
            name for name, item in by_candidate.items()
            if item.status != "confident" and name not in candidate_hands
        }
        closure_contents = [
            list(fact) for fact in sorted(gt.facts)
            if len(fact) == 3 and fact[0] == "in"
            and ("open", fact[2]) in gt.facts
            and (content := gt.object(fact[1])) is not None
            and not set(content.kinds) & {"water", "milk", "liquid", "food"}
        ]
        lock_state_owners = {
            fact[1] for fact in gt.facts
            if len(fact) == 2 and fact[0] in {"locked", "unlocked"}
        }
        lock_fixtures = [
            item.name for item in gt.objects
            if set(item.kinds) & {"cabinet", "appliance", "device"}
            and item.name not in lock_state_owners and lock_state_owners
        ]
        query_relations = bool(closure_contents or lock_fixtures)
        if not unresolved and not query_relations:
            return alignment
        if image_path is None or not image_path.exists():
            raise VLMAlignmentError("initial image is required for VLM object mapping")
        self._increment("ambiguous_alignments")

        frozen = {
            item.gt for item in alignment.objects
            if item.status == "confident" and item.gt is not None
        }
        eligible = [
            item for item in gt.objects
            if item.name not in frozen and item.name not in gt_hands
        ]
        candidate_lookup = {item.name: item for item in candidate.objects}
        gt_names = {item.name for item in eligible}
        candidate_initial = candidate.facts | candidate.negative_facts
        relevant_initial = {
            fact for fact in candidate_initial
            if set(fact[1:]) & unresolved
        }
        task = {
            "instruction": instruction,
            "query_objects": [
                _object_payload(candidate_lookup[name]) for name in sorted(unresolved)
            ],
            "eligible_gt_objects": [_object_payload(item) for item in eligible],
            "fixed_identities": {
                item.candidate: item.gt for item in alignment.objects
                if item.status == "confident" and item.gt is not None
            },
            "gt_anchor_objects": [
                _object_payload(item) for item in gt.objects if item.name in frozen
            ],
            "candidate_initial_facts": [list(fact) for fact in sorted(candidate.facts) if fact in relevant_initial],
            "candidate_initial_negative_facts": [list(fact) for fact in sorted(candidate.negative_facts) if fact in relevant_initial],
            "gt_initial_facts": [list(fact) for fact in sorted(gt.facts)],
            "gt_initial_negative_facts": [list(fact) for fact in sorted(gt.negative_facts)],
        }
        relation_prompt = ""
        if query_relations:
            task["initial_relation_queries"] = {
                "closure_contents": closure_contents,
                "lock_fixtures": lock_fixtures,
                "lock_state_owners": sorted(lock_state_owners),
            }
            relation_prompt = (
                "Also include initial_relations: a list of {\"relation\": "
                "[\"blocks_closing\" or \"blocks_opening\" or \"lock_owner\", "
                "\"gt_subject\", \"gt_target\"], \"holds\": true_or_false, "
                "\"evidence\": [\"visual reason\"]}. Use GT names only. For EACH "
                "closure_contents pair report a blocks_closing check: can the content "
                "fit entirely inside the CLOSED enclosure? Protruding beyond its "
                "interior prevents closing; merely being inside is not blockage. "
                "lock_owner links a whole fixture to the distinct component bearing "
                "its visible lock/keyhole. Do not infer relations from goals or "
                "reference actions. Other relations require positive visual evidence. "
                "Return an empty list only if there are no queried contents and no "
                "other evidenced relations. "
            )
        prompt = (
            "Map objects across two models of the SAME initial image, jointly and one-to-one. "
            "Exact-name identities are fixed. Use the instruction only to disambiguate which "
            "visible object a phrase refers to; never use a desired outcome to infer initial "
            "identity. Do not use candidate actions, plans or goals. Names, singleton types, "
            "and matching predicate strings alone do not establish identity; compare image "
            "evidence and the entire initial scene. If the instruction treats several "
            "otherwise indistinguishable objects interchangeably, assign any one-to-one "
            "pairing without inventing positional evidence. A whole object may represent "
            "its operated part when both models use the same state and instruction-level "
            "operation at different granularity; do not merge independently modeled parts. "
            "Keep contents distinct from containers. Acting hands are handled separately. "
            "For EACH query object: first look for its physical GT counterpart and return "
            "that ONE exact GT name, even if a candidate label, kind, or state assertion is "
            "wrong; the simulator checks those claims later. Otherwise, check whether the candidate "
            "really describes a distinct entity in the image and whether ALL its listed initial "
            "positive and negative facts involving it are correct in that image. A real omitted "
            "entity with correct initial facts is scene_object=true ONLY if visual/instruction "
            "evidence distinguishes it from EVERY GT object. Existing GT objects with different "
            "names are not omissions; never create duplicate objects to avoid a difficult "
            "one-to-one identity choice. Reproduce ALL listed "
            "facts verbatim under verified_facts. If the entity or any associated initial fact "
            "is demonstrably wrong and there is no GT counterpart, return scene_object=false. "
            "Absence from GT inventory alone "
            "does NOT imply false. Give concrete image/instruction evidence for every decision; "
            "do not guess by list order or label resemblance. Do not return unknown, null, "
            "multiple matches, or invented names. Unsupported decisions are invalid protocol "
            "responses, not negative scene evidence. Return JSON only: {\"candidates\": {\"query_name\": "
            "{\"matches\": [\"gt_name\"], \"evidence\": [\"reason\"]} OR "
            "{\"matches\": [], \"scene_object\": true_or_false, "
            "\"verified_facts\": {\"positive\": [[\"predicate\", \"object\", ...], ...], "
            "\"negative\": []}, \"evidence\": [\"reason\"]}}}. "
            "Only scene_object=true uses verified_facts. Include every query name exactly once."
            + (" " + relation_prompt if query_relations else "")
            + "\n\n" + json.dumps(task, indent=2, sort_keys=True)
        )
        if _repair_response is not None:
            prompt += (
                "\n\nRecheck the previous response against the image and the anchored "
                "initial relations. A generic fixture name can denote the whole unit's "
                "contact surface even if GT names the unit rather than its component, "
                "when both models locate the same anchored object on that unique surface. "
                "A candidate state object carrying locked/unlocked may denote the GT "
                "device bearing that state, while a separate candidate button maps to "
                "the GT physical button. Never merge separate physical objects or excuse a false "
                "initial relation. Correct the whole response: use only eligible GT names, "
                "a nonempty evidence list per object, and exact "
                "candidate_initial_facts plus candidate_initial_negative_facts for EVERY "
                "scene_object=true object (including facts shared with another object). "
                "For matched objects omit verified_facts. No extra keys or unknown values. "
                "Previous response:\n" + json.dumps(_repair_response, sort_keys=True)
            )
            if closure_contents:
                prompt += (
                    "\nRequired closing relation arrays (blocker first, container "
                    "second; retain holds=false checks):\n"
                    + json.dumps([["blocks_closing", *fact[1:]] for fact in closure_contents])
                )
        try:
            cache_key, response = self._cached_request(
                prompt, image_path,
                prompt_version=IDENTITY_PROMPT_VERSION + (
                    ":initial-relations-v2" if query_relations else ""
                ),
            )
        except Exception as error:
            self._increment("fallbacks")
            if self.require_complete:
                raise VLMAlignmentError(
                    "required VLM object mapping request failed: "
                    f"{type(error).__name__}: {str(error).splitlines()[0]}"
                ) from error
            return alignment

        if isinstance(response, dict):
            response = {key: value for key, value in response.items() if key not in candidate_hands}
        candidates = response.get("candidates") if isinstance(response, dict) else None
        valid = isinstance(response, dict) and isinstance(candidates, dict)
        if valid:
            candidates = {
                key: value for key, value in candidates.items() if key not in candidate_hands
            }
            # Some JSON-mode providers emit query keys beside, rather than inside,
            # the candidates wrapper. Accept only exact requested keys.
            metadata = {"candidates", "evidence", "evidence_note", "scene_object_note"}
            if query_relations:
                metadata.add("initial_relations")
            valid = not (set(response) - metadata - unresolved)
            valid = valid and not (set(candidates) & (set(response) - metadata))
            candidates = {**candidates, **{name: response[name] for name in unresolved if name in response}}
            valid = valid and set(candidates) == unresolved
        accepted: dict[str, str] = {}
        added: set[str] = set()
        rejected: set[str] = set()
        occupied = set(frozen)
        if valid:
            for name in sorted(unresolved):
                item = candidates[name]
                if not isinstance(item, dict):
                    valid = False
                    break
                matches = item.get("matches")
                evidence = item.get("evidence")
                if (
                    not isinstance(matches, list)
                    or len(matches) > 1
                    or not isinstance(evidence, list)
                    or not evidence
                    or not all(isinstance(reason, str) and reason.strip() for reason in evidence)
                ):
                    valid = False
                    break
                if matches:
                    target = matches[0]
                    if (not isinstance(target, str) or target not in gt_names
                        or target in occupied
                        or set(item) not in ({"matches", "evidence"},
                                             {"matches", "evidence", "verified_facts"})):
                        valid = False
                        break
                    accepted[name] = target
                    occupied.add(target)
                else:
                    scene_object = item.get("scene_object")
                    if not isinstance(scene_object, bool):
                        valid = False
                        break
                    if scene_object:
                        if set(item) != {"matches", "scene_object", "verified_facts", "evidence"}:
                            valid = False
                            break
                        expected = {
                            "positive": {fact for fact in candidate.facts if name in fact[1:]},
                            "negative": {fact for fact in candidate.negative_facts if name in fact[1:]},
                        }
                        verified = item.get("verified_facts")
                        if not isinstance(verified, dict) or set(verified) != set(expected):
                            valid = False
                            break
                        for polarity, facts in expected.items():
                            reported = verified[polarity]
                            if (not isinstance(reported, list)
                                or any(not isinstance(fact, list) or not all(isinstance(v, str) for v in fact) for fact in reported)
                                or len(reported) != len({tuple(fact) for fact in reported})
                                or {tuple(fact) for fact in reported} != facts):
                                valid = False
                                break
                        if not valid:
                            break
                        added.add(name)
                    else:
                        if set(item) != {"matches", "scene_object", "evidence"}:
                            valid = False
                            break
                        rejected.add(name)
        if valid:
            mapping = alignment.mapping() | accepted | {name: name for name in added}
            for name in added:
                if any(argument not in mapping for fact in relevant_initial if name in fact[1:] for argument in fact[1:]):
                    valid = False
                    break
        scene_facts: set[Literal] = set()
        scene_negative: set[Literal] = set()
        if valid and query_relations:
            relations = response.get("initial_relations")
            valid = isinstance(relations, list)
            closure_checks = {("blocks_closing", *fact[1:]) for fact in closure_contents}
            checked: set[Literal] = set()
            for item in relations if valid else ():
                relation = item.get("relation") if isinstance(item, dict) else None
                evidence = item.get("evidence") if isinstance(item, dict) else None
                holds = item.get("holds") if isinstance(item, dict) else None
                if (
                    not isinstance(relation, list) or len(relation) != 3
                    or not all(isinstance(name, str) for name in relation)
                    or relation[0] not in {"blocks_closing", "blocks_opening", "lock_owner"}
                    or gt.object(relation[1]) is None or gt.object(relation[2]) is None
                    or relation[1] == relation[2]
                    or not isinstance(holds, bool)
                    or not isinstance(evidence, list) or not evidence
                    or not all(isinstance(reason, str) and reason.strip() for reason in evidence)
                    or set(item) != {"relation", "holds", "evidence"}
                    or tuple(relation) in checked
                    or (holds and tuple(relation) in gt.negative_facts)
                    or (not holds and tuple(relation) in gt.facts)
                    or (relation[0] == "lock_owner" and (
                        relation[1] not in lock_fixtures
                        or relation[2] not in lock_state_owners
                    ))
                    or (relation[0] != "lock_owner"
                        and ["in", relation[1], relation[2]] not in closure_contents)
                ):
                    valid = False
                    break
                checked.add(tuple(relation))
                if holds:
                    scene_facts.add(tuple(relation))
            valid = valid and closure_checks <= checked
        if valid:
            for source, dest in ((candidate.facts, scene_facts), (candidate.negative_facts, scene_negative)):
                for fact in source:
                    if set(fact[1:]) & added:
                        dest.add((fact[0], *(mapping[arg] for arg in fact[1:])))
            if ((scene_facts & (gt.negative_facts | scene_negative))
                or (scene_negative & gt.facts)):
                valid = False
        if not valid:
            self._increment("invalid_responses")
            if _repair_response is None and self.allow_network:
                return self.refine(
                    candidate, gt, alignment, instruction=instruction,
                    image_path=image_path, _repair_response=response,
                )
            if self.require_complete:
                raise VLMAlignmentError("required VLM object mapping returned invalid JSON")
            return alignment

        # A uniquely anchored initial contact can identify a generic support
        # alias, but not an object with its own additional physical claims.
        mapped = alignment.mapping() | accepted
        for name in tuple(sorted(rejected)):
            contacts = [
                fact for fact in candidate.facts
                if len(fact) == 3 and fact[0] == "on"
                and fact[1] in mapped and fact[2] == name
            ]
            other_claims = [
                fact for fact in candidate.facts | candidate.negative_facts
                if name in fact[1:] and not fact[0].startswith(("kind:", "static:"))
                and fact not in contacts
            ]
            if len(contacts) != 1 or other_claims:
                continue
            support = {
                fact[2] for fact in gt.facts
                if len(fact) == 3 and fact[:2] == ("on", mapped[contacts[0][1]])
                and fact[2] in gt_names - occupied
            }
            if len(support) != 1:
                continue
            target = next(iter(support))
            if any(
                ("on", item.name, target) in gt.facts
                and set(normalized_tokens(name)) <= set(normalized_tokens(item.name))
                for item in gt.objects if item.name != mapped[contacts[0][1]]
            ):
                continue
            accepted[name] = target
            occupied.add(target)
            rejected.remove(name)

        if rejected and _repair_response is None and self.allow_network:
            plausible = any(
                any(
                    len(fact) == 2
                    and fact[0] in {"locked", "unlocked"}
                    and fact[1] == name
                    and sum(
                        (fact[0], target.name) in gt.facts
                        for target in eligible if target.name not in occupied
                    ) == 1
                    for fact in candidate.facts
                )
                for name in rejected
            )
            if plausible:
                return self.refine(
                    candidate, gt, alignment, instruction=instruction,
                    image_path=image_path, _repair_response=response,
                )

        if accepted or added:
            self._increment("accepted_alignments")
            self._increment("accepted_objects", len(accepted) + len(added))
        if added:
            self._increment("added_objects", len(added))
        if rejected:
            self._increment("rejected_objects", len(rejected))
        objects = []
        for item in alignment.objects:
            name = item.candidate
            if name in accepted:
                objects.append(replace(
                    item, gt=accepted[name], status="vlm", alternatives=(),
                ))
            elif name in added:
                objects.append(replace(item, gt=name, status="vlm", alternatives=()))
            elif name in rejected:
                objects.append(replace(item, gt=None, status="unmapped", alternatives=()))
            else:
                objects.append(item)
        return AlignmentResult(
            tuple(objects), advisor=self.model, cache_key=cache_key,
            scene_facts=frozenset(scene_facts),
            scene_negative_facts=frozenset(scene_negative),
            scene_objects=tuple(sorted(added)),
        )

    def resolve_initial_source_relation(
        self,
        gt: CanonicalWorld,
        *,
        object_name: str,
        requested_source: str,
        symbolic_relation: tuple[str, str, str],
        relation_context: str | None = None,
        image_path: Path | None,
    ) -> tuple[bool | None, dict[str, Any]]:
        """Resolve one initial direct-source ambiguity from visual evidence.

        This advisor answers only a factual scene relation. It cannot change
        object identity, action semantics, state transitions, goals, or the
        final simulator verdict.
        """
        if image_path is None or not image_path.exists():
            return None, {"outcome": "UNKNOWN", "reason": "initial image unavailable"}
        obj = gt.object(object_name)
        requested = gt.object(requested_source)
        actual_source = gt.object(symbolic_relation[2])
        if obj is None or requested is None or actual_source is None:
            return None, {"outcome": "UNKNOWN", "reason": "source object missing from GT inventory"}

        names = {object_name, requested_source, symbolic_relation[2]}
        context_tokens = re.findall(r"[a-z0-9]+", (relation_context or "").lower())
        source_phrase = None
        if "from" in context_tokens:
            source_phrase = "_".join(
                context_tokens[context_tokens.index("from") + 1:]
            ) or None
        task = {
            "object": _object_payload(obj),
            "requested_direct_source": _object_payload(requested),
            "candidate_relation_context": relation_context,
            "candidate_source_phrase": source_phrase,
            "symbolic_relation": list(symbolic_relation),
            "symbolic_source": _object_payload(actual_source),
            "relevant_initial_relations": _relation_facts(gt, names),
        }
        prompt = (
            "The GT image shows one initial scene. Decide whether the candidate's "
            "source description identifies the named object's actual direct support. "
            "The mapped GT source is an alignment choice, not an extra qualifier "
            "in the candidate description. Compare the object's actual contact region, not just "
            "whether the source inventory identifiers name the same fixture. A named "
            "component's surface and its parent can refer to the same contact region "
            "only when the image confirms it. The candidate source phrase is a lexical "
            "description of a region; verify it against the image. An unqualified "
            "fixture phrase may cover its regions even if GT splits them into "
            "separate objects; do not invent a left/right restriction from the "
            "mapped GT name. An explicit spatial or component qualifier remains "
            "binding. This requires image evidence of the same physical fixture, "
            "not merely similar types. Do not equate an "
            "outer support with a direct support, or a container with its contents, "
            "merely because they touch or belong together. Do not switch to another "
            "similar object. Return null if the image and inventory cannot establish "
            "the direct relation. Do not use the instruction, plan outcome, goal, or "
            "desired verdict. Return JSON only: {\"object\": exact_object_name, "
            "\"requested_source\": exact_requested_source_name, "
            "\"relation_supported\": true_or_false_or_null, "
            "\"evidence\": [\"short_initial_scene_reason\", ...]}. A true or false "
            "answer requires nonempty evidence.\n\n"
            + json.dumps(task, indent=2, sort_keys=True)
        )
        for attempt in range(2):
            self._increment("source_relation_queries")
            try:
                cache_key, response = self._cached_request(
                    prompt,
                    image_path,
                    prompt_version=SOURCE_RELATION_PROMPT_VERSION,
                )
            except Exception as error:
                self._increment("source_relation_fallbacks")
                return None, {
                    "outcome": "UNKNOWN",
                    "reason": f"{type(error).__name__}: {str(error).splitlines()[0]}",
                }
            if not isinstance(response, dict):
                break
            if (
                response.get("object") == object_name
                and response.get("requested_source") == requested_source
            ):
                break
            if attempt == 0:
                prompt += (
                    "\n\nThe previous response used an incorrect identifier. "
                    f'Copy these exact identifiers in the JSON: "object": "{object_name}", '
                    f'"requested_source": "{requested_source}". '
                    "Independently check the image before deciding the relation."
                )
        if not isinstance(response, dict):
            response = {}
        decision = response.get("relation_supported")
        evidence = response.get("evidence")
        if (
            response.get("object") != object_name
            or response.get("requested_source") != requested_source
            or not isinstance(decision, (bool, type(None)))
            or not isinstance(evidence, list)
            or (decision is not None and not evidence)
            or not all(isinstance(reason, str) and reason.strip() for reason in evidence)
        ):
            self._increment("source_relation_invalid")
            return None, {
                "outcome": "UNKNOWN",
                "cache_key": cache_key,
                "reason": "invalid relation response",
            }
        accepted = decision
        self._increment(
            "source_relation_true" if accepted is True
            else "source_relation_false" if accepted is False
            else "source_relation_unknown"
        )
        return accepted, {
            "outcome": "PASS" if accepted is True else "FAIL" if accepted is False else "UNKNOWN",
            "object": object_name,
            "requested_source": requested_source,
            "symbolic_relation": list(symbolic_relation),
            "evidence": evidence,
            "cache_key": cache_key,
        }
