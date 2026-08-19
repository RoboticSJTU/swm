from __future__ import annotations

import sys
from pathlib import Path

import pytest


CLEAN_OPERATOR_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CLEAN_OPERATOR_DIR.parent
sys.path.insert(0, str(CLEAN_OPERATOR_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import clean_operator_data as cleanup  # noqa: E402
import mergy  # noqa: E402


def action(
    name: str,
    params: list[str],
    pre: list[cleanup.Node],
    eff: list[cleanup.Node],
) -> cleanup.Action:
    return cleanup.Action(
        name=name,
        params=params,
        pre=cleanup.make_and(pre),
        eff=cleanup.make_and(eff),
        comment=f"; Legacy {name} action.",
    )


def literals(expr: cleanup.Node) -> set[str]:
    return {cleanup.sexp(item) for item in cleanup.conjunction(expr)}


def test_target_pose_guard_is_not_mistaken_for_place_result_pose() -> None:
    placed = action(
        "place_banana_on_plate_when_plate_flat",
        ["?h", "?banana", "?plate"],
        [
            ["hand", "?h"], ["banana", "?banana"], ["plate", "?plate"],
            ["holding", "?h", "?banana"], ["flat", "?plate"],
        ],
        [
            ["not", ["holding", "?h", "?banana"]], ["hand_free", "?h"],
            ["on", "?banana", "?plate"],
        ],
    )

    result = cleanup.orientation_transform(placed, 248, {})

    assert result is not None
    assert result.name == "place_banana_on_plate_when_plate_flat"
    assert "(flat ?banana)" not in literals(result.eff)


def test_oriented_place_clears_every_mutually_exclusive_pose() -> None:
    placed = action(
        "place_plate_flat_on_table",
        ["?h", "?plate", "?table"],
        [
            ["hand", "?h"], ["plate", "?plate"], ["table", "?table"],
            ["holding", "?h", "?plate"],
        ],
        [
            ["not", ["holding", "?h", "?plate"]], ["hand_free", "?h"],
            ["on", "?plate", "?table"], ["flat", "?plate"],
        ],
    )

    result = cleanup.canonical_place_contract(placed)

    assert result.name == "place_plate_flat_on_table"
    assert {
        "(not (vertical ?plate))", "(not (upright ?plate))",
        "(not (sideways ?plate))", "(not (on_side ?plate))",
    } <= literals(result.eff)


def test_audited_state_conflict_repairs_are_idempotent() -> None:
    problem = """
(define (problem water)
  (:domain water)
  (:objects hand hot_water_button)
  (:init
    (hand hand) (hot_water_button hot_water_button)
    (is_on hot_water_button) (is_off hot_water_button))
  (:goal (and (is_off hot_water_button))))
"""

    repaired = cleanup.repair_audited_round_anomalies(
        problem, "", tid=21, dataset="human_aug", episode=37
    )
    parsed = cleanup.parse_problem(repaired[0])
    init = cleanup.problem_facts(parsed, ":init")

    assert ["is_on", "hot_water_button"] in init
    assert ["is_off", "hot_water_button"] not in init
    assert cleanup.repair_audited_round_anomalies(
        *repaired, tid=21, dataset="human_aug", episode=37
    ) == repaired


def test_audited_missing_initial_placement_drops_impossible_first_step() -> None:
    problem = """
(define (problem microwave)
  (:domain microwave)
  (:objects hand bottle counter)
  (:init (hand hand) (bottle bottle) (counter counter) (holding hand cup))
  (:goal (and (on bottle counter))))
"""
    plan = """(place_bottle_on_counter hand bottle counter)
(place_paper_cup_in_microwave_when_microwave_open hand cup microwave)
; cost = 2 (unit cost)
"""

    repaired_problem, repaired_plan = cleanup.repair_audited_round_anomalies(
        problem, plan, tid=264, dataset="human_aug", episode=65
    )
    init = cleanup.problem_facts(cleanup.parse_problem(repaired_problem), ":init")

    assert ["on", "capped_water_bottle", "counter"] in init
    assert "place_bottle_on_counter" not in repaired_plan
    assert "; cost = 1 (unit cost)" in repaired_plan


def test_bowl_stack_migration_changes_only_bowl_to_bowl_containment() -> None:
    domain = """
(define (domain bowl_stack)
  (:predicates
    (hand ?h) (bowl ?b) (apple ?a) (hand_free ?h) (holding ?h ?o)
    (in ?o ?c) (on ?o ?s))
  (:action pick_nested_bowl_from_bowl
    :parameters (?h ?inner ?outer)
    :precondition (and
      (hand ?h) (bowl ?inner) (bowl ?outer) (hand_free ?h)
      (in ?inner ?outer))
    :effect (and
      (not (hand_free ?h)) (holding ?h ?inner)
      (not (in ?inner ?outer))))
  (:action place_bowl_in_bowl
    :parameters (?h ?inner ?outer)
    :precondition (and
      (hand ?h) (bowl ?inner) (bowl ?outer) (holding ?h ?inner))
    :effect (and
      (not (holding ?h ?inner)) (hand_free ?h) (in ?inner ?outer)))
  (:action pick_apple_from_bowl
    :parameters (?h ?a ?b)
    :precondition (and
      (hand ?h) (apple ?a) (bowl ?b) (hand_free ?h) (in ?a ?b))
    :effect (and
      (not (hand_free ?h)) (holding ?h ?a) (not (in ?a ?b)))))
"""
    problem = """
(define (problem bowl_stack_problem)
  (:domain bowl_stack)
  (:objects hand pink_bowl green_bowl apple)
  (:init
    (hand hand) (bowl pink_bowl) (bowl green_bowl) (apple apple)
    (hand_free hand)
    (in pink_bowl green_bowl)
    (in apple green_bowl))
  (:goal (and (holding hand pink_bowl))))
"""

    migrated_domain, migrated_problem = cleanup.migrate_bowl_stack_contract(
        domain, problem
    )
    actions = cleanup.domain_actions(migrated_domain)

    assert "pick_nested_bowl_from_bowl" not in actions
    assert "place_bowl_in_bowl" not in actions
    assert "pick_bowl_from_bowl" in actions
    assert "place_bowl_on_bowl" in actions
    assert "(on ?inner ?outer)" in literals(actions["pick_bowl_from_bowl"].pre)
    assert "(not (on ?inner ?outer))" in literals(
        actions["pick_bowl_from_bowl"].eff
    )
    assert "(on ?inner ?outer)" in literals(actions["place_bowl_on_bowl"].eff)

    # A real contained object is not part of the bowl-stack correction.
    apple_pick = actions["pick_apple_from_bowl"]
    assert "(in ?a ?b)" in literals(apple_pick.pre)
    assert "(not (in ?a ?b))" in literals(apple_pick.eff)

    init = cleanup.problem_facts(cleanup.parse_problem(migrated_problem), ":init")
    assert ["on", "pink_bowl", "green_bowl"] in init
    assert ["in", "pink_bowl", "green_bowl"] not in init
    assert ["in", "apple", "green_bowl"] in init

    assert cleanup.migrate_bowl_stack_contract(
        migrated_domain, migrated_problem
    ) == (migrated_domain, migrated_problem)


def test_turn_on_microwave_is_a_pure_toggle() -> None:
    legacy = action(
        "turn_on_microwave",
        ["?h", "?m", "?food", "?plate", "?bin", "?trash"],
        [
            ["hand", "?h"],
            ["microwave", "?m"],
            ["corn", "?food"],
            ["plate", "?plate"],
            ["trash_bin", "?bin"],
            ["trash", "?trash"],
            ["hand_free", "?h"],
            ["closed", "?m"],
            ["is_off", "?m"],
            ["on", "?food", "?plate"],
            ["in", "?plate", "?m"],
            ["contains", "?bin", "?trash"],
        ],
        [
            ["not", ["is_off", "?m"]],
            ["is_on", "?m"],
            ["heated", "?food"],
        ],
    )

    result = cleanup.microwave_contract_transform(
        legacy, direct_microwave_relation="in"
    )

    assert result.name == "turn_on_microwave"
    assert result.params == ["?h", "?m"]
    assert literals(result.pre) == {
        "(hand ?h)",
        "(microwave ?m)",
        "(hand_free ?h)",
        "(closed ?m)",
        "(is_off ?m)",
    }
    assert literals(result.eff) == {
        "(not (is_off ?m))",
        "(is_on ?m)",
    }


def test_open_microwave_names_an_off_guard_when_it_is_present() -> None:
    legacy = action(
        "open_microwave",
        ["?h", "?m"],
        [
            ["hand", "?h"], ["microwave", "?m"], ["hand_free", "?h"],
            ["closed", "?m"], ["is_off", "?m"],
        ],
        [["not", ["closed", "?m"]], ["open", "?m"]],
    )

    result = cleanup.microwave_open_transform(legacy)

    assert result.name == "open_microwave_when_off"


def test_turn_off_microwave_commits_heat_with_the_minimal_carrier_chain() -> None:
    legacy = action(
        "turn_off_microwave",
        ["?h", "?m", "?food", "?plate", "?counter", "?bin"],
        [
            ["hand", "?h"],
            ["microwave", "?m"],
            ["corn", "?food"],
            ["plate", "?plate"],
            ["counter", "?counter"],
            ["trash_bin", "?bin"],
            ["hand_free", "?h"],
            ["closed", "?m"],
            ["is_on", "?m"],
            ["on", "?food", "?plate"],
            ["in", "?plate", "?m"],
            ["on", "?bin", "?counter"],
        ],
        [
            ["not", ["is_on", "?m"]],
            ["is_off", "?m"],
            ["heated", "?food"],
            ["clean", "?counter"],
        ],
    )

    result = cleanup.microwave_contract_transform(legacy)

    assert result.name == (
        "turn_off_microwave_after_heating_corn_on_plate_when_plate_in_microwave"
    )
    assert result.params == ["?h", "?m", "?food", "?plate"]
    assert literals(result.pre) == {
        "(hand ?h)",
        "(microwave ?m)",
        "(corn ?food)",
        "(plate ?plate)",
        "(hand_free ?h)",
        "(closed ?m)",
        "(is_on ?m)",
        "(on ?food ?plate)",
        "(in ?plate ?m)",
    }
    assert literals(result.eff) == {
        "(not (is_on ?m))",
        "(is_off ?m)",
        "(heated ?food)",
    }


def test_microwave_heating_contract_is_textually_idempotent() -> None:
    legacy = action(
        "turn_off_microwave_after_heating_water_in_paper_cup",
        ["?h", "?m", "?water", "?cup"],
        [
            ["hand", "?h"], ["microwave", "?m"], ["paper_cup", "?cup"],
            ["water", "?water"], ["hand_free", "?h"], ["closed", "?m"],
            ["is_on", "?m"], ["in", "?water", "?cup"],
            ["in", "?cup", "?m"],
        ],
        [["not", ["is_on", "?m"]], ["is_off", "?m"], ["heated", "?water"]],
    )

    once = cleanup.microwave_contract_transform(legacy)
    twice = cleanup.microwave_contract_transform(once)

    assert once == twice
    assert list(cleanup.conjunction(once.pre))[:4] == [
        ["hand", "?h"], ["microwave", "?m"],
        ["water", "?water"], ["paper_cup", "?cup"],
    ]


def test_turn_off_microwave_supports_a_directly_heated_carrier() -> None:
    legacy = action(
        "turn_off_microwave_after_heating_bowl",
        ["?h", "?m", "?b"],
        [
            ["hand", "?h"],
            ["microwave", "?m"],
            ["bowl", "?b"],
            ["hand_free", "?h"],
            ["closed", "?m"],
            ["is_on", "?m"],
            ["in", "?b", "?m"],
        ],
        [
            ["not", ["is_on", "?m"]],
            ["is_off", "?m"],
            ["heated", "?b"],
        ],
    )

    result = cleanup.microwave_contract_transform(legacy)

    assert result.name == "turn_off_microwave_after_heating_bowl"
    assert result.params == ["?h", "?m", "?b"]
    assert "(in ?b ?m)" in literals(result.pre)
    assert literals(result.eff) == {
        "(not (is_on ?m))",
        "(is_off ?m)",
        "(heated ?b)",
    }


@pytest.mark.parametrize(
    ("legacy_name", "extra_pre", "expected_name"),
    [
        ("open_drawer", [], "open_drawer"),
        (
            "open_unlocked_drawer",
            [["unlocked", "?d"]],
            "open_drawer_when_unlocked",
        ),
        (
            "open_unblocked_drawer",
            [["unblocked", "?d"]],
            "open_drawer_when_clear_to_open",
        ),
        (
            "open_unlocked_drawer",
            [["unlocked", "?d"], ["unblocked", "?d"]],
            "open_drawer_when_unlocked_and_clear_to_open",
        ),
    ],
)
def test_drawer_name_reflects_only_present_causal_guards(
    legacy_name: str,
    extra_pre: list[cleanup.Node],
    expected_name: str,
) -> None:
    legacy = action(
        legacy_name,
        ["?h", "?d"],
        [
            ["hand", "?h"],
            ["drawer", "?d"],
            ["hand_free", "?h"],
            ["closed", "?d"],
            *extra_pre,
        ],
        [["not", ["closed", "?d"]], ["open", "?d"]],
    )

    before_params = list(legacy.params)
    before_pre = literals(legacy.pre)
    before_eff = literals(legacy.eff)

    result = cleanup.drawer_contract_transform(legacy)

    assert result.name == expected_name
    assert result.params == before_params == ["?h", "?d"]
    assert literals(result.pre) == before_pre
    assert literals(result.eff) == before_eff


@pytest.mark.parametrize(
    ("legacy_name", "params", "pre", "expected_name"),
    [
        (
            "open_middle_drawer",
            ["?h", "?middle", "?top"],
            [
                ["drawer", "?middle"],
                ["drawer", "?top"],
                ["is_middle", "?middle"],
                ["is_top", "?top"],
                ["closed", "?middle"],
                ["closed", "?top"],
            ],
            "open_middle_drawer_when_top_drawer_closed",
        ),
        (
            "open_bottom_drawer",
            ["?h", "?bottom", "?top", "?middle"],
            [
                ["drawer", "?bottom"],
                ["drawer", "?top"],
                ["drawer", "?middle"],
                ["is_bottom", "?bottom"],
                ["is_top", "?top"],
                ["is_middle", "?middle"],
                ["closed", "?bottom"],
                ["closed", "?top"],
                ["closed", "?middle"],
            ],
            "open_bottom_drawer_when_top_and_middle_drawers_closed",
        ),
    ],
)
def test_drawer_interlock_name_identifies_the_concrete_closed_drawers(
    legacy_name: str,
    params: list[str],
    pre: list[cleanup.Node],
    expected_name: str,
) -> None:
    legacy = action(
        legacy_name,
        params,
        [["hand", "?h"], ["hand_free", "?h"], *pre],
        [["not", ["closed", params[1]]], ["open", params[1]]],
    )

    before_params = list(legacy.params)
    before_pre = literals(legacy.pre)
    before_eff = literals(legacy.eff)

    result = cleanup.drawer_contract_transform(legacy)

    assert result.name == expected_name
    assert result.params == before_params == params
    assert literals(result.pre) == before_pre
    assert literals(result.eff) == before_eff


@pytest.mark.parametrize(
    ("legacy_name", "relation", "expected_name"),
    [
        ("place_mug_into_cabinet", "in", "place_mug_in_cabinet"),
        ("place_cap_onto_bottle", "on", "place_cap_on_bottle"),
    ],
)
def test_place_relation_lexemes_are_canonicalized(
    legacy_name: str,
    relation: str,
    expected_name: str,
) -> None:
    legacy = action(
        legacy_name,
        ["?h", "?item", "?target"],
        [["holding", "?h", "?item"]],
        [[relation, "?item", "?target"]],
    )

    result = cleanup.normalize_relation_lexemes(legacy)

    assert result.name == expected_name
    assert f"({relation} ?item ?target)" in literals(result.eff)


@pytest.mark.parametrize(
    "legacy",
    [
        action(
            "pick_bowl_from_table",
            ["?h", "?b", "?t"],
            [
                ["hand", "?h"],
                ["bowl", "?b"],
                ["table", "?t"],
                ["hand_free", "?h"],
                ["on", "?b", "?t"],
            ],
            [
                ["not", ["hand_free", "?h"]],
                ["holding", "?h", "?b"],
                ["not", ["on", "?b", "?t"]],
            ],
        ),
        action(
            "place_bowl_on_plate",
            ["?h", "?b", "?p"],
            [
                ["hand", "?h"],
                ["bowl", "?b"],
                ["plate", "?p"],
                ["holding", "?h", "?b"],
            ],
            [
                ["not", ["holding", "?h", "?b"]],
                ["hand_free", "?h"],
                ["on", "?b", "?p"],
            ],
        ),
    ],
    ids=["pick-from-table", "place-on-plate"],
)
def test_family_clear_contract_does_not_leak_onto_nonstack_surfaces(
    legacy: cleanup.Action,
) -> None:
    before_pre = literals(legacy.pre)
    before_eff = literals(legacy.eff)

    result = cleanup.canonical_family_clear_action(legacy, "bowl")

    assert literals(result.pre) == before_pre
    assert literals(result.eff) == before_eff
    assert not any("(clear " in literal for literal in literals(result.pre))
    assert not any("(clear " in literal for literal in literals(result.eff))


@pytest.mark.parametrize(
    ("legacy_name", "object_type", "target_type", "relation", "expected_name"),
    [
        (
            "place_ladle_in_rack",
            "ladle",
            "rack",
            "on",
            "place_ladle_on_rack",
        ),
        (
            "place_bowl_in_dish_rack",
            "bowl",
            "dish_rack",
            "on",
            "place_bowl_on_dish_rack",
        ),
    ],
)
def test_spatial_place_name_exposes_the_primary_support_relation(
    legacy_name: str,
    object_type: str,
    target_type: str,
    relation: str,
    expected_name: str,
) -> None:
    legacy = action(
        legacy_name,
        ["?h", "?item", "?target"],
        [
            ["hand", "?h"],
            [object_type, "?item"],
            [target_type, "?target"],
            ["holding", "?h", "?item"],
        ],
        [
            ["not", ["holding", "?h", "?item"]],
            ["hand_free", "?h"],
            [relation, "?item", "?target"],
        ],
    )

    result = cleanup.normalize_spatial_action_contract(legacy)

    assert result.name == expected_name


@pytest.mark.parametrize(
    ("legacy_name", "item_type", "surface_type", "reference_type", "relative", "expected_name"),
    [
        (
            "place_block_beside_bowl",
            "block",
            "table",
            "bowl",
            "beside",
            "place_block_on_table_beside_bowl",
        ),
        (
            "place_towel_beside_kettle",
            "towel",
            "counter",
            "kettle",
            "beside",
            "place_towel_on_counter_beside_kettle",
        ),
    ],
)
def test_relative_place_keeps_a_relation_that_is_really_established(
    legacy_name: str,
    item_type: str,
    surface_type: str,
    reference_type: str,
    relative: str,
    expected_name: str,
) -> None:
    legacy = action(
        legacy_name,
        ["?h", "?item", "?surface", "?reference"],
        [
            ["hand", "?h"],
            [item_type, "?item"],
            [surface_type, "?surface"],
            [reference_type, "?reference"],
            ["holding", "?h", "?item"],
        ],
        [
            ["not", ["holding", "?h", "?item"]],
            ["hand_free", "?h"],
            ["on", "?item", "?surface"],
            [relative, "?item", "?reference"],
        ],
    )

    result = cleanup.normalize_spatial_action_contract(legacy)

    assert result.name == expected_name


def test_turntable_is_an_on_support_for_paper_cup() -> None:
    legacy = action(
        "place_paper_cup_on_microwave_turntable",
        ["?h", "?cup", "?turntable"],
        [
            ["hand", "?h"],
            ["paper_cup", "?cup"],
            ["microwave_turntable", "?turntable"],
            ["holding", "?h", "?cup"],
        ],
        [
            ["not", ["holding", "?h", "?cup"]],
            ["hand_free", "?h"],
            ["in", "?cup", "?turntable"],
        ],
    )

    result = cleanup.normalize_spatial_action_contract(legacy)

    assert "(on ?cup ?turntable)" in literals(result.eff)
    assert "(in ?cup ?turntable)" not in literals(result.eff)


def test_legacy_turntable_name_is_removed_for_direct_microwave_containment() -> None:
    legacy = action(
        "place_heated_paper_cup_upright_on_microwave_turntable",
        ["?h", "?cup", "?microwave"],
        [
            ["hand", "?h"],
            ["paper_cup", "?cup"],
            ["microwave", "?microwave"],
            ["holding", "?h", "?cup"],
        ],
        [
            ["not", ["holding", "?h", "?cup"]],
            ["hand_free", "?h"],
            ["in", "?cup", "?microwave"],
        ],
    )

    result = cleanup.normalize_spatial_action_contract(legacy)

    assert result.name == "place_paper_cup_in_microwave"
    assert "(in ?cup ?microwave)" in literals(result.eff)
    assert "(on ?cup ?microwave)" not in literals(result.eff)


def test_missing_relative_effect_drops_the_claimed_relation_and_scene_object() -> None:
    legacy = action(
        "place_block_beside_bowl",
        ["?h", "?block", "?table", "?bowl"],
        [
            ["hand", "?h"],
            ["block", "?block"],
            ["table", "?table"],
            ["bowl", "?bowl"],
            ["holding", "?h", "?block"],
            ["on", "?bowl", "?table"],
        ],
        [
            ["not", ["holding", "?h", "?block"]],
            ["hand_free", "?h"],
            ["on", "?block", "?table"],
        ],
    )

    result = cleanup.normalize_spatial_action_contract(legacy)

    assert result.name == "place_block_on_table"
    assert result.params == ["?h", "?block", "?table"]
    assert not any("?bowl" in literal for literal in literals(result.pre))


def test_pick_contract_drops_contents_pose_and_scene_payload() -> None:
    legacy = action(
        "pick_paper_cup_from_counter",
        ["?h", "?cup", "?counter", "?water", "?bottle"],
        [
            ["hand", "?h"], ["paper_cup", "?cup"],
            ["counter", "?counter"], ["water", "?water"],
            ["bottle", "?bottle"], ["hand_free", "?h"],
            ["upright", "?cup"], ["in", "?water", "?cup"],
            ["on", "?cup", "?counter"], ["closed", "?bottle"],
        ],
        [
            ["not", ["hand_free", "?h"]], ["holding", "?h", "?cup"],
            ["not", ["on", "?cup", "?counter"]],
        ],
    )

    result = cleanup.canonical_pick_contract(legacy)

    assert result.name == "pick_paper_cup_from_counter"
    assert result.params == ["?h", "?cup", "?counter"]
    assert literals(result.pre) == {
        "(hand ?h)", "(paper_cup ?cup)", "(counter ?counter)",
        "(hand_free ?h)", "(on ?cup ?counter)",
    }
    assert literals(result.eff) == {
        "(not (hand_free ?h))", "(holding ?h ?cup)",
        "(not (on ?cup ?counter))",
    }


def test_stack_pick_keeps_the_family_clear_transition() -> None:
    legacy = action(
        "pick_block_from_block",
        ["?h", "?top", "?support", "?box"],
        [
            ["hand", "?h"], ["block", "?top"], ["block", "?support"],
            ["box", "?box"], ["hand_free", "?h"], ["clear", "?top"],
            ["on", "?top", "?support"], ["closed", "?box"],
        ],
        [
            ["not", ["hand_free", "?h"]], ["holding", "?h", "?top"],
            ["not", ["clear", "?top"]], ["clear", "?support"],
            ["not", ["on", "?top", "?support"]],
        ],
    )

    result = cleanup.canonical_pick_contract(legacy)

    assert result.name == "pick_block_from_block"
    assert "(clear ?top)" in literals(result.pre)
    assert "(not (clear ?top))" in literals(result.eff)
    assert "(clear ?support)" in literals(result.eff)
    assert result.params == ["?h", "?top", "?support"]


def test_place_contract_keeps_target_guard_and_pose_result_only() -> None:
    legacy = action(
        "place_corn_in_bowl",
        ["?h", "?corn", "?bowl", "?counter", "?water"],
        [
            ["hand", "?h"], ["corn", "?corn"], ["bowl", "?bowl"],
            ["counter", "?counter"], ["water", "?water"],
            ["holding", "?h", "?corn"], ["upright", "?bowl"],
            ["on", "?bowl", "?counter"], ["in", "?water", "?bowl"],
        ],
        [
            ["not", ["holding", "?h", "?corn"]], ["hand_free", "?h"],
            ["in", "?corn", "?bowl"],
        ],
    )

    result = cleanup.canonical_place_contract(legacy)

    assert result.name == "place_corn_in_bowl_when_bowl_upright"
    assert result.params == ["?h", "?corn", "?bowl"]
    assert literals(result.pre) == {
        "(hand ?h)", "(corn ?corn)", "(bowl ?bowl)",
        "(holding ?h ?corn)", "(upright ?bowl)",
    }
    assert literals(result.eff) == {
        "(not (holding ?h ?corn))", "(hand_free ?h)",
        "(in ?corn ?bowl)",
    }


def test_place_contract_names_clear_as_a_result_not_a_when_guard() -> None:
    legacy = action(
        "place_block_on_table",
        ["?h", "?block", "?table"],
        [
            ["hand", "?h"], ["block", "?block"], ["table", "?table"],
            ["holding", "?h", "?block"],
        ],
        [
            ["not", ["holding", "?h", "?block"]], ["hand_free", "?h"],
            ["on", "?block", "?table"], ["clear", "?block"],
        ],
    )

    result = cleanup.canonical_place_contract(legacy)

    assert result.name == "place_block_clear_on_table"
    assert "_when_" not in result.name
    assert "(clear ?block)" in literals(result.eff)


def test_place_contract_preserves_any_cover_induced_closure() -> None:
    legacy = action(
        "place_lid_on_glass_kettle",
        ["?h", "?lid", "?kettle"],
        [
            ["hand", "?h"], ["kettle_lid", "?lid"],
            ["glass_kettle", "?kettle"], ["holding", "?h", "?lid"],
            ["open", "?kettle"],
        ],
        [
            ["not", ["holding", "?h", "?lid"]], ["hand_free", "?h"],
            ["on", "?lid", "?kettle"], ["not", ["open", "?kettle"]],
            ["closed", "?kettle"],
        ],
    )

    result = cleanup.canonical_place_contract(legacy)

    assert result == legacy
    assert "(not (open ?kettle))" in literals(result.eff)
    assert "(closed ?kettle)" in literals(result.eff)


def test_problem_spatial_contract_mirrors_stable_category_relations() -> None:
    problem = """
(define (problem spatial_problem)
  (:domain spatial_domain)
  (:objects clothes basket cap tray cup microwave apple bowl)
  (:init
    (clothes clothes) (laundry_basket basket) (cap cap) (tray tray)
    (paper_cup cup) (microwave microwave) (apple apple) (bowl bowl)
    (on clothes basket) (in cap tray) (on cup microwave) (in apple bowl))
  (:goal (and (on cap tray) (in cup microwave))))
"""

    result = cleanup.normalize_problem_spatial_contract(problem)
    parsed = cleanup.parse_problem(result)
    init = {cleanup.sexp(item) for item in cleanup.problem_facts(parsed, ":init")}
    goals = {cleanup.sexp(item) for item in cleanup.problem_facts(parsed, ":goal")}

    assert "(in clothes basket)" in init
    assert "(on cap tray)" in init
    assert "(on cup microwave)" in init
    assert "(in apple bowl)" in init
    assert goals == {"(on cap tray)", "(in cup microwave)"}


def test_microwave_end_normalizes_direct_carrier_containment_to_in() -> None:
    legacy = action(
        "turn_off_microwave",
        ["?h", "?m", "?water", "?cup"],
        [
            ["hand", "?h"], ["microwave", "?m"], ["water", "?water"],
            ["paper_cup", "?cup"], ["hand_free", "?h"], ["closed", "?m"],
            ["is_on", "?m"], ["in", "?water", "?cup"], ["on", "?cup", "?m"],
        ],
        [
            ["not", ["is_on", "?m"]], ["is_off", "?m"],
            ["heated", "?water"],
        ],
    )

    result = cleanup.microwave_contract_transform(
        legacy, direct_microwave_relation="in"
    )

    assert result.name == (
        "turn_off_microwave_after_heating_water_in_paper_cup"
        "_when_paper_cup_in_microwave"
    )
    assert "(in ?cup ?m)" in literals(result.pre)
    assert "(on ?cup ?m)" not in literals(result.pre)


def test_wash_cycle_selection_drops_loaded_payload_and_names_real_guards() -> None:
    legacy = action(
        "turn_dial_on_washing_machine",
        ["?h", "?dial", "?wm", "?clothes", "?detergent", "?drawer"],
        [
            ["hand", "?h"], ["dial", "?dial"], ["washing_machine", "?wm"],
            ["clothes", "?clothes"], ["detergent", "?detergent"],
            ["detergent_drawer", "?drawer"], ["hand_free", "?h"],
            ["closed", "?wm"], ["closed", "?drawer"],
            ["in", "?clothes", "?wm"], ["in", "?detergent", "?drawer"],
        ],
        [["cycle_selected", "?wm"]],
    )

    result = cleanup.washing_machine_contract_transform(legacy)

    assert result.name == (
        "select_wash_cycle_with_dial_when_washing_machine_closed_and_"
        "detergent_drawer_closed"
    )
    assert result.params == ["?h", "?dial", "?wm", "?drawer"]
    assert literals(result.pre) == {
        "(hand ?h)", "(dial ?dial)", "(washing_machine ?wm)",
        "(detergent_drawer ?drawer)", "(hand_free ?h)",
        "(closed ?wm)", "(closed ?drawer)",
    }
    assert literals(result.eff) == {"(cycle_selected ?wm)"}


def test_washing_machine_start_only_commits_started() -> None:
    legacy = action(
        "push_start_button_on_washing_machine",
        ["?h", "?button", "?wm", "?clothes", "?detergent", "?drawer"],
        [
            ["hand", "?h"], ["start_button", "?button"],
            ["washing_machine", "?wm"], ["clothes", "?clothes"],
            ["detergent", "?detergent"], ["detergent_drawer", "?drawer"],
            ["hand_free", "?h"], ["cycle_selected", "?wm"],
            ["closed", "?wm"], ["closed", "?drawer"], ["off", "?wm"],
            ["in", "?clothes", "?wm"], ["in", "?detergent", "?drawer"],
        ],
        [
            ["not", ["off", "?wm"]], ["on_state", "?wm"],
            ["washing", "?wm"],
        ],
    )

    result = cleanup.washing_machine_contract_transform(legacy)

    assert result.name == (
        "push_start_button_on_washing_machine_when_cycle_selected_and_"
        "washing_machine_closed_and_detergent_drawer_closed_and_"
        "washing_machine_off"
    )
    assert result.params == ["?h", "?button", "?wm", "?drawer"]
    assert literals(result.eff) == {"(started ?wm)"}
    assert "(in ?clothes ?wm)" not in literals(result.pre)
    assert "(in ?detergent ?drawer)" not in literals(result.pre)


def test_washing_start_with_washed_result_splits_completion() -> None:
    domain = """
(define (domain wash)
  (:predicates
    (hand ?h) (start_button ?b) (washing_machine ?m) (clothes ?c)
    (hand_free ?h) (started ?m) (washed ?c) (in ?o ?x))
  (:action push_start_button_on_washing_machine
    :parameters (?h ?button ?machine ?clothes)
    :precondition (and
      (hand ?h) (start_button ?button) (washing_machine ?machine)
      (clothes ?clothes) (hand_free ?h) (in ?clothes ?machine))
    :effect (and (started ?machine) (washed ?clothes))))
"""

    result, expansions = cleanup.split_washing_start_completion(domain)
    actions = cleanup.domain_actions(result)

    assert "wash_clothes_in_washing_machine_until_washed" in actions
    assert literals(actions["push_start_button_on_washing_machine"].eff) == {
        "(started ?machine)"
    }
    completion = actions["wash_clothes_in_washing_machine_until_washed"]
    assert completion.params == ["?clothes", "?machine"]
    assert literals(completion.pre) == {
        "(clothes ?clothes)", "(washing_machine ?machine)",
        "(started ?machine)", "(in ?clothes ?machine)",
    }
    assert expansions["push_start_button_on_washing_machine"].steps[1] == (
        "wash_clothes_in_washing_machine_until_washed", ("?clothes", "?machine")
    )


def test_infinite_source_action_does_not_consume_explicit_jug_water() -> None:
    legacy = action(
        "press_water_jug_pump",
        ["?h", "?pump", "?water", "?kettle", "?jug"],
        [
            ["hand", "?h"], ["water_pump", "?pump"],
            ["water", "?water"], ["kettle", "?kettle"],
            ["water_jug", "?jug"], ["hand_free", "?h"],
            ["in", "?water", "?jug"],
        ],
        [["is_on", "?pump"], ["not", ["in", "?water", "?jug"]],
         ["in", "?water", "?kettle"]],
    )

    result, changed = cleanup.infinite_source_action(legacy)

    assert not changed
    assert result == legacy
    assert "(in ?water ?jug)" in literals(result.pre)
    assert "(not (in ?water ?jug))" in literals(result.eff)


def test_held_finite_jug_pump_transfers_water_only_on_release() -> None:
    domain = """
(define (domain fill)
  (:predicates
    (hand ?h) (water_jug_pump ?p) (pump_button ?b) (water_jug ?j)
    (water ?w) (kettle ?k) (hand_free ?h) (pressing ?h ?p)
    (is_off ?p) (is_on ?p) (open ?k) (in ?o ?c))
  (:action press_water_jug_pump
    :parameters (?h ?p ?button ?water ?kettle ?jug)
    :precondition (and
      (hand ?h) (water_jug_pump ?p) (pump_button ?button)
      (water ?water) (kettle ?kettle) (water_jug ?jug)
      (hand_free ?h) (is_off ?p) (open ?kettle) (in ?water ?jug))
    :effect (and
      (not (hand_free ?h)) (pressing ?h ?p)
      (not (is_off ?p)) (is_on ?p)
      (not (in ?water ?jug)) (in ?water ?kettle)))
  (:action stop_pressing_water_jug_pump
    :parameters (?h ?p ?water ?kettle)
    :precondition (and
      (hand ?h) (water_jug_pump ?p) (water ?water) (kettle ?kettle)
      (pressing ?h ?p) (is_on ?p) (open ?kettle) (in ?water ?kettle))
    :effect (and
      (not (pressing ?h ?p)) (hand_free ?h)
      (not (is_on ?p)) (is_off ?p))))
"""

    result, edits = cleanup.normalize_water_supply_contracts(domain)
    actions = cleanup.domain_actions(result)
    press = actions["press_and_hold_water_jug_pump"]
    release = actions[
        "release_water_jug_pump_after_filling_kettle_from_water_jug"
    ]

    assert literals(press.eff) == {
        "(not (hand_free ?h))", "(pressing ?h ?p)",
        "(not (is_off ?p))", "(is_on ?p)",
    }
    assert "(in ?water ?jug)" in literals(release.pre)
    assert "(not (in ?water ?jug))" in literals(release.eff)
    assert "(in ?water ?kettle)" in literals(release.eff)
    assert edits["press_water_jug_pump"].new_name == "press_and_hold_water_jug_pump"


def test_latched_infinite_pump_has_pure_start_and_resultful_stop() -> None:
    domain = """
(define (domain fill)
  (:predicates
    (hand ?h) (water_pump ?p) (water ?w) (kettle ?k)
    (hand_free ?h) (is_off ?p) (is_on ?p) (open ?k)
    (dispenses ?p ?w) (in ?o ?c))
  (:action press_water_pump
    :parameters (?h ?pump ?kettle)
    :precondition (and (hand ?h) (water_pump ?pump) (kettle ?kettle)
      (hand_free ?h) (is_off ?pump) (open ?kettle))
    :effect (and (not (is_off ?pump)) (is_on ?pump)))
  (:action stop_pressing_water_pump
    :parameters (?h ?pump ?kettle ?water)
    :precondition (and (hand ?h) (water_pump ?pump) (kettle ?kettle)
      (water ?water) (hand_free ?h) (is_on ?pump) (open ?kettle)
      (dispenses ?pump ?water))
    :effect (and (not (is_on ?pump)) (is_off ?pump) (in ?water ?kettle))))
"""

    result, _edits = cleanup.normalize_water_supply_contracts(domain)
    actions = cleanup.domain_actions(result)

    assert literals(actions["turn_on_water_pump"].eff) == {
        "(not (is_off ?pump))", "(is_on ?pump)"
    }
    stop = actions["turn_off_water_pump_after_filling_kettle"]
    assert "(dispenses ?pump ?water)" in literals(stop.pre)
    assert "(in ?water ?kettle)" in literals(stop.eff)


def test_momentary_fill_button_is_one_atomic_contract() -> None:
    legacy = action(
        "press_water_dispenser_button",
        ["?h", "?button", "?dispenser", "?water", "?kettle"],
        [
            ["hand", "?h"], ["dispenser_button", "?button"],
            ["water_dispenser", "?dispenser"], ["water", "?water"],
            ["kettle", "?kettle"], ["hand_free", "?h"],
            ["open", "?kettle"], ["dispenses", "?dispenser", "?water"],
        ],
        [["in", "?water", "?kettle"]],
    )

    result = cleanup.water_supply_action_contract(legacy)

    assert result.name == "press_water_dispenser_button_to_fill_kettle"
    assert "(in ?water ?kettle)" in literals(result.eff)
    assert not any("is_on" in item or "is_off" in item for item in literals(result.eff))


def test_finite_pour_projects_scene_payload_and_keeps_real_guards() -> None:
    legacy = action(
        "pour_boiled_water_from_kettle_into_cup_on_counter",
        ["?h", "?water", "?kettle", "?cup", "?counter", "?bin"],
        [
            ["hand", "?h"], ["water", "?water"], ["kettle", "?kettle"],
            ["cup", "?cup"], ["counter", "?counter"], ["trash_bin", "?bin"],
            ["holding", "?h", "?kettle"], ["open", "?kettle"],
            ["clear", "?cup"], ["boiled", "?water"],
            ["in", "?water", "?kettle"], ["on", "?cup", "?counter"],
        ],
        [["not", ["in", "?water", "?kettle"]], ["in", "?water", "?cup"]],
    )

    result = cleanup.finite_liquid_contract_transform(legacy)

    assert result.name == (
        "pour_water_from_kettle_in_cup_when_kettle_open_and_cup_clear"
    )
    assert result.params == ["?h", "?water", "?kettle", "?cup"]
    assert literals(result.pre) == {
        "(hand ?h)", "(water ?water)", "(kettle ?kettle)", "(cup ?cup)",
        "(holding ?h ?kettle)", "(in ?water ?kettle)",
        "(open ?kettle)", "(clear ?cup)",
    }
    assert literals(result.eff) == {
        "(not (in ?water ?kettle))", "(in ?water ?cup)"
    }


def test_spoonful_pour_consumes_source_carrier_state() -> None:
    legacy = action(
        "transfer_liquid_from_spoon_into_paper_cup",
        ["?h", "?spoon", "?cup", "?counter"],
        [
            ["hand", "?h"], ["spoon", "?spoon"],
            ["paper_cup", "?cup"], ["counter", "?counter"],
            ["holding", "?h", "?spoon"], ["has_water", "?spoon"],
            ["on", "?cup", "?counter"],
        ],
        [["not", ["has_water", "?spoon"]], ["has_water", "?cup"]],
    )

    result = cleanup.finite_liquid_contract_transform(legacy)

    assert result.name == "pour_spoonful_from_spoon_in_paper_cup"
    assert result.params == ["?h", "?spoon", "?cup"]
    assert literals(result.eff) == {
        "(not (has_water ?spoon))", "(has_water ?cup)"
    }


def test_empty_bowl_in_sink_consumes_has_water_and_produces_empty() -> None:
    legacy = action(
        "pour_water_from_bowl_into_sink",
        ["?h", "?bowl", "?sink"],
        [
            ["hand", "?h"], ["bowl", "?bowl"], ["sink", "?sink"],
            ["holding", "?h", "?bowl"], ["has_water", "?bowl"],
        ],
        [["not", ["has_water", "?bowl"]], ["poured", "?bowl"]],
    )

    result = cleanup.finite_liquid_contract_transform(legacy)

    assert result.name == "empty_bowl_in_sink"
    assert literals(result.eff) == {
        "(not (has_water ?bowl))", "(empty ?bowl)"
    }


def test_hot_water_button_start_keeps_only_the_child_lock_guard() -> None:
    legacy = action(
        "turn_on_hot_water_button",
        ["?h", "?button", "?lock", "?mug", "?spout"],
        [
            ["hand", "?h"], ["button", "?button"],
            ["child_lock_button", "?lock"], ["mug", "?mug"],
            ["hot_water_spout", "?spout"], ["hand_free", "?h"],
            ["is_off", "?button"], ["unlocked", "?lock"],
            ["under", "?mug", "?spout"],
        ],
        [["not", ["is_off", "?button"]], ["is_on", "?button"]],
    )

    result = cleanup.canonical_water_button_contract(legacy)

    assert result.name == "turn_on_hot_water_button_when_child_lock_unlocked"
    assert result.params == ["?h", "?button", "?lock"]
    assert literals(result.pre) == {
        "(hand ?h)", "(hot_water_button ?button)", "(hand_free ?h)",
        "(is_off ?button)", "(child_lock ?lock)", "(unlocked ?lock)",
    }


def test_cold_water_button_stop_names_and_projects_the_fill_result() -> None:
    legacy = action(
        "turn_off_cold_water_button",
        ["?h", "?button", "?liquid", "?mug", "?nozzle"],
        [
            ["hand", "?h"], ["cold_water_button", "?button"],
            ["cold_water", "?liquid"], ["mug", "?mug"],
            ["cold_water_nozzle", "?nozzle"], ["hand_free", "?h"],
            ["is_on", "?button"], ["under", "?mug", "?nozzle"],
            ["dispenses", "?nozzle", "?liquid"],
        ],
        [
            ["not", ["is_on", "?button"]], ["is_off", "?button"],
            ["in", "?liquid", "?mug"],
        ],
    )

    result = cleanup.canonical_water_button_contract(legacy)

    assert result.name == "turn_off_cold_water_button_after_filling_cup"
    assert result.params == ["?h", "?button", "?liquid", "?mug", "?nozzle"]
    assert literals(result.pre) == {
        "(hand ?h)", "(cold_water_button ?button)", "(water ?liquid)",
        "(cup ?mug)", "(water_dispenser ?nozzle)", "(hand_free ?h)",
        "(is_on ?button)", "(under ?mug ?nozzle)",
        "(dispenses ?nozzle ?liquid)",
    }


def test_water_button_plan_adds_canonical_role_types_to_problem() -> None:
    domain = """
(define (domain water)
  (:predicates
    (hand ?h) (cold_water_button ?b) (water ?w) (cup ?c)
    (water_dispenser ?d) (hand_free ?h) (is_on ?b)
    (under ?c ?d) (dispenses ?d ?w) (in ?w ?c))
  (:action turn_off_cold_water_button_after_filling_cup
    :parameters (?h ?b ?w ?c ?d)
    :precondition (and
      (hand ?h) (cold_water_button ?b) (water ?w) (cup ?c)
      (water_dispenser ?d) (hand_free ?h) (is_on ?b)
      (under ?c ?d) (dispenses ?d ?w))
    :effect (and (not (is_on ?b)) (in ?w ?c))))
"""
    problem = """
(define (problem water)
  (:domain water)
  (:objects hand button liquid mug nozzle)
  (:init
    (hand hand) (cold_water_button button) (cold_water liquid)
    (mug mug) (cold_water_nozzle nozzle) (hand_free hand)
    (is_on button) (under mug nozzle) (dispenses nozzle liquid))
  (:goal (and (in liquid mug))))
"""
    plan = """(turn_off_cold_water_button_after_filling_cup
  hand button liquid mug nozzle)
""".replace("\n  ", " ")

    result = cleanup.ensure_water_button_type_facts(domain, problem, plan)
    init = cleanup.problem_facts(cleanup.parse_problem(result), ":init")

    assert ["water", "liquid"] in init
    assert ["cup", "mug"] in init
    assert ["water_dispenser", "nozzle"] in init


def test_cold_button_names_a_real_water_mixing_result() -> None:
    legacy = action(
        "turn_off_cold_water_button",
        ["?h", "?button", "?cold", "?hot", "?mixed", "?mug", "?source"],
        [
            ["hand", "?h"], ["cold_water_button", "?button"],
            ["water", "?cold"], ["water", "?hot"], ["water", "?mixed"],
            ["mug", "?mug"], ["cold_water_dispenser", "?source"],
            ["hand_free", "?h"], ["is_on", "?button"],
            ["in", "?hot", "?mug"], ["under", "?mug", "?source"],
            ["dispenses", "?source", "?cold"],
        ],
        [
            ["not", ["is_on", "?button"]], ["is_off", "?button"],
            ["not", ["in", "?hot", "?mug"]], ["in", "?mixed", "?mug"],
        ],
    )

    result = cleanup.canonical_water_button_contract(legacy)

    assert result.name == "turn_off_cold_water_button_after_mixing_water_in_cup"
    assert "(dispenses ?source ?cold)" in literals(result.pre)
    assert "(in ?hot ?mug)" in literals(result.pre)
    assert "(not (in ?hot ?mug))" in literals(result.eff)
    assert "(in ?mixed ?mug)" in literals(result.eff)


def test_child_lock_aliases_share_one_toggle_contract() -> None:
    legacy = action(
        "unlock_child_lock",
        ["?h", "?lock"],
        [
            ["hand", "?h"], ["child_lock_button", "?lock"],
            ["hand_free", "?h"], ["locked", "?lock"],
        ],
        [["not", ["locked", "?lock"]], ["unlocked", "?lock"]],
    )

    result = cleanup.canonical_child_lock_contract(legacy)

    assert literals(result.pre) == {
        "(hand ?h)", "(child_lock ?lock)", "(hand_free ?h)",
        "(locked ?lock)",
    }


def test_power_connection_aliases_use_inserted_contract() -> None:
    legacy = action(
        "plug_power_base_cord_in_wall_outlet",
        ["?h", "?cord", "?socket"],
        [
            ["hand", "?h"], ["cord", "?cord"], ["socket", "?socket"],
            ["holding", "?h", "?cord"],
        ],
        [
            ["not", ["holding", "?h", "?cord"]], ["hand_free", "?h"],
            ["connected", "?cord", "?socket"],
        ],
    )

    result = cleanup.canonical_power_connection_contract(legacy)

    assert literals(result.pre) == {
        "(hand ?h)", "(power_base_cord ?cord)",
        "(wall_outlet ?socket)", "(holding ?h ?cord)",
    }
    assert "(inserted ?cord ?socket)" in literals(result.eff)


def test_power_connection_contract_runs_after_into_name_normalization() -> None:
    legacy = action(
        "insert_plug_into_wall_outlet",
        ["?h", "?plug", "?socket"],
        [
            ["hand", "?h"], ["plug", "?plug"], ["wall_outlet", "?socket"],
            ["holding", "?h", "?plug"],
        ],
        [
            ["not", ["holding", "?h", "?plug"]], ["hand_free", "?h"],
            ["plugged_in", "?plug", "?socket"],
        ],
    )

    result = cleanup.canonical_power_connection_contract(
        cleanup.normalize_relation_lexemes(legacy)
    )

    assert result.name == "insert_plug_in_wall_outlet"
    assert literals(result.pre) == {
        "(hand ?h)", "(plug ?plug)", "(wall_outlet ?socket)",
        "(holding ?h ?plug)",
    }
    assert literals(result.eff) == {
        "(not (holding ?h ?plug))", "(hand_free ?h)",
        "(inserted ?plug ?socket)",
    }


def test_power_connection_problem_aliases_only_rewrite_typed_connections() -> None:
    problem = """
(define (problem power)
  (:domain power)
  (:objects plug outlet water kettle)
  (:init
    (plug plug) (wall_outlet outlet) (water water) (kettle kettle)
    (plugged_in plug outlet) (in water kettle))
  (:goal (and (connected plug outlet) (in water kettle))))
"""

    result = cleanup.normalize_power_connection_problem_predicates(problem)
    parsed = cleanup.parse_problem(result)

    assert {cleanup.sexp(item) for item in cleanup.problem_facts(parsed, ":init")} == {
        "(plug plug)", "(wall_outlet outlet)", "(water water)",
        "(kettle kettle)", "(inserted plug outlet)", "(in water kettle)",
    }
    assert {cleanup.sexp(item) for item in cleanup.problem_facts(parsed, ":goal")} == {
        "(inserted plug outlet)", "(in water kettle)",
    }


def test_held_water_control_press_has_only_the_mechanical_transition() -> None:
    legacy = action(
        "press_cold_water_control",
        ["?h", "?control", "?cup", "?spout"],
        [
            ["hand", "?h"], ["cold_water_control", "?control"],
            ["paper_cup", "?cup"], ["cold_water_spout", "?spout"],
            ["hand_free", "?h"], ["unpressed", "?control"],
            ["under", "?cup", "?spout"],
        ],
        [
            ["not", ["hand_free", "?h"]], ["holding", "?h", "?control"],
            ["not", ["unpressed", "?control"]], ["pressed", "?control"],
        ],
    )

    result = cleanup.canonical_press_release_contract(legacy)

    assert result.name == "press_and_hold_cold_water_control"
    assert result.params == ["?h", "?control"]
    assert "(paper_cup ?cup)" not in literals(result.pre)
    assert "(pressing ?h ?control)" in literals(result.eff)


def test_grounded_pick_source_type_uses_problem_fact_not_action_name() -> None:
    domain = """
(define (domain pick)
  (:predicates (hand ?h) (plug ?p) (box ?b) (hand_free ?h)
               (on ?p ?b) (holding ?h ?p))
  (:action pick_plug_from_counter
    :parameters (?h ?p ?source)
    :precondition (and (hand ?h) (plug ?p) (hand_free ?h) (on ?p ?source))
    :effect (and (not (hand_free ?h)) (holding ?h ?p)
                 (not (on ?p ?source)))))
"""
    problem = """
(define (problem pick-one)
  (:domain pick)
  (:objects hand plug red_box)
  (:init (hand hand) (plug plug) (box red_box) (hand_free hand)
         (on plug red_box))
  (:goal (and (holding hand plug))))
"""

    restored = cleanup.restore_grounded_pick_source_types(
        domain, problem, "(pick_plug_from_counter hand plug red_box)\n"
    )
    schema = cleanup.domain_actions(restored)["pick_plug_from_counter"]

    assert "(box ?source)" in literals(schema.pre)
    assert "(counter ?source)" not in literals(schema.pre)


def test_water_control_release_names_result_then_real_guard() -> None:
    legacy = action(
        "release_cold_water_control",
        ["?h", "?control", "?cup"],
        [
            ["hand", "?h"], ["cold_water_control", "?control"],
            ["paper_cup", "?cup"], ["holding", "?h", "?control"],
            ["pressed", "?control"], ["hot_added", "?cup"],
        ],
        [
            ["not", ["holding", "?h", "?control"]], ["hand_free", "?h"],
            ["not", ["pressed", "?control"]], ["unpressed", "?control"],
            ["filled", "?cup"], ["warm", "?cup"],
        ],
    )

    result = cleanup.canonical_press_release_contract(
        legacy, {"cold_water_control"}
    )

    assert result.name == (
        "release_cold_water_control_after_filled_paper_cup_and_warm_paper_cup"
        "_when_paper_cup_hot_added"
    )
    assert "(pressing ?h ?control)" in literals(result.pre)
    assert "(hot_added ?cup)" in literals(result.pre)
    assert "(not (pressing ?h ?control))" in literals(result.eff)


def test_finite_water_end_consumes_source_and_marks_kettle_filled() -> None:
    legacy = action(
        "stop_pressing_water_jug_pump",
        ["?h", "?pump", "?water", "?jug", "?kettle", "?box"],
        [
            ["hand", "?h"], ["water_jug_pump", "?pump"],
            ["water", "?water"], ["water_jug", "?jug"],
            ["kettle", "?kettle"], ["box", "?box"],
            ["pump_pressed", "?pump"], ["in", "?water", "?jug"],
            ["open", "?kettle"], ["on", "?kettle", "?box"],
        ],
        [
            ["not", ["pump_pressed", "?pump"]], ["hand_free", "?h"],
            ["in", "?water", "?kettle"],
        ],
    )

    result = cleanup.canonical_finite_water_end_contract(legacy)

    assert result.name == (
        "release_water_jug_pump_after_filling_kettle_from_water_jug"
    )
    assert "(box ?box)" not in literals(result.pre)
    assert "(not (in ?water ?jug))" in literals(result.eff)
    assert "(filled ?kettle)" in literals(result.eff)
    assert "(not (empty ?kettle))" in literals(result.eff)


def test_initial_pressed_control_uses_pressing_interface() -> None:
    domain = """
(define (domain water)
  (:predicates (hand ?h) (hot_water_control ?c) (pressed ?c)
               (pressing ?h ?c) (hand_free ?h) (unpressed ?c))
  (:action release_hot_water_control
    :parameters (?h ?c)
    :precondition (and (hand ?h) (hot_water_control ?c)
                       (pressing ?h ?c) (pressed ?c))
    :effect (and (not (pressing ?h ?c)) (hand_free ?h)
                 (not (pressed ?c)) (unpressed ?c))))
"""
    problem = """
(define (problem water-one)
  (:domain water)
  (:objects hand control)
  (:init (hand hand) (hot_water_control control)
         (holding hand control) (pressed control))
  (:goal (and (unpressed control))))
"""

    result = cleanup.normalize_initial_pressed_control_interface(domain, problem)
    init = cleanup.problem_facts(cleanup.parse_problem(result), ":init")

    assert ["pressing", "hand", "control"] in init
    assert ["holding", "hand", "control"] not in init


def test_pump_pressed_hold_uses_shared_pressing_interface() -> None:
    legacy = action(
        "press_water_jug_pump_with_button",
        ["?h", "?pump", "?button", "?kettle"],
        [
            ["hand", "?h"], ["water_jug_pump", "?pump"],
            ["pump_button", "?button"], ["kettle", "?kettle"],
            ["hand_free", "?h"],
        ],
        [["not", ["hand_free", "?h"]], ["pump_pressed", "?pump"]],
    )

    result = cleanup.canonical_press_release_contract(legacy)

    assert result.name == "press_and_hold_water_jug_pump"
    assert result.params == ["?h", "?pump"]
    assert "(pressing ?h ?pump)" in literals(result.eff)
    assert "(pump_button ?button)" not in literals(result.pre)


def test_stateful_held_pump_projects_to_same_pressing_contract() -> None:
    legacy = action(
        "press_and_hold_water_jug_pump",
        ["?h", "?pump", "?button"],
        [
            ["hand", "?h"], ["water_jug_pump", "?pump"],
            ["pump_button", "?button"], ["hand_free", "?h"],
            ["is_off", "?pump"],
        ],
        [
            ["not", ["hand_free", "?h"]], ["pressing", "?h", "?pump"],
            ["not", ["is_off", "?pump"]], ["is_on", "?pump"],
        ],
    )

    result = cleanup.canonical_press_release_contract(legacy)

    assert result.params == ["?h", "?pump"]
    assert literals(result.pre) == {
        "(hand ?h)", "(water_jug_pump ?pump)", "(hand_free ?h)",
    }
    assert "(is_on ?pump)" not in literals(result.eff)


def test_direct_drawer_key_insert_names_actual_guards() -> None:
    legacy = action(
        "insert_key_in_drawer",
        ["?h", "?key", "?drawer"],
        [
            ["hand", "?h"], ["key", "?key"], ["drawer", "?drawer"],
            ["holding", "?h", "?key"], ["closed", "?drawer"],
            ["unlocked", "?drawer"],
        ],
        [
            ["not", ["holding", "?h", "?key"]], ["hand_free", "?h"],
            ["inserted", "?key", "?drawer"],
        ],
    )

    result = cleanup.canonical_key_contract(legacy)

    assert result.name == (
        "insert_key_in_drawer_when_drawer_closed_and_drawer_unlocked"
    )
    assert "(closed ?drawer)" in literals(result.pre)
    assert "(unlocked ?drawer)" in literals(result.pre)


def test_separate_drawer_lock_keeps_only_structural_owner_relation() -> None:
    legacy = action(
        "insert_key_in_drawer_lock",
        ["?h", "?key", "?lock", "?drawer", "?box"],
        [
            ["hand", "?h"], ["key", "?key"], ["drawer_lock", "?lock"],
            ["drawer", "?drawer"], ["box", "?box"],
            ["holding", "?h", "?key"], ["lock_of", "?lock", "?drawer"],
            ["closed", "?drawer"], ["on", "?box", "?drawer"],
        ],
        [["inserted", "?key", "?lock"]],
    )

    result = cleanup.canonical_key_contract(legacy)

    assert result.name == (
        "insert_held_key_in_drawer_lock_for_drawer_when_drawer_closed"
    )
    assert "(lock_of ?lock ?drawer)" in literals(result.pre)
    assert "(box ?box)" not in literals(result.pre)
    assert "(not (holding ?h ?key))" not in literals(result.eff)


def test_turn_key_name_comes_from_lock_state_effect() -> None:
    legacy = action(
        "turn_key_in_drawer_lock",
        ["?h", "?key", "?lock", "?drawer"],
        [
            ["hand", "?h"], ["key", "?key"], ["lock", "?lock"],
            ["drawer", "?drawer"], ["hand_free", "?h"],
            ["inserted", "?key", "?lock"], ["lock_of", "?lock", "?drawer"],
            ["closed", "?drawer"], ["locked", "?drawer"],
        ],
        [["not", ["locked", "?drawer"]], ["unlocked", "?drawer"]],
    )

    result = cleanup.canonical_key_contract(legacy)

    assert result.name == (
        "unlock_drawer_with_key_in_lock_when_drawer_closed"
    )
    assert "(lock_of ?lock ?drawer)" in literals(result.pre)


def test_microwave_heating_preserves_causal_carrier_containment() -> None:
    legacy = action(
        "turn_off_microwave",
        ["?h", "?m", "?water", "?cup"],
        [
            ["hand", "?h"], ["microwave", "?m"], ["water", "?water"],
            ["paper_cup", "?cup"], ["hand_free", "?h"], ["closed", "?m"],
            ["is_on", "?m"], ["in", "?water", "?cup"], ["in", "?cup", "?m"],
        ],
        [["not", ["is_on", "?m"]], ["is_off", "?m"], ["heated", "?water"]],
    )

    result = cleanup.microwave_contract_transform(legacy)

    assert result.name == (
        "turn_off_microwave_after_heating_water_in_paper_cup"
        "_when_paper_cup_in_microwave"
    )
    assert "(in ?water ?cup)" in literals(result.pre)
    assert "(in ?cup ?m)" in literals(result.pre)


def test_lower_button_is_a_pure_microwave_start() -> None:
    legacy = action(
        "press_lower_button",
        ["?h", "?button", "?m", "?cup"],
        [
            ["hand", "?h"], ["lower_button", "?button"],
            ["microwave", "?m"], ["paper_cup", "?cup"],
            ["hand_free", "?h"], ["closed", "?m"], ["is_off", "?m"],
            ["in", "?cup", "?m"],
        ],
        [["not", ["is_off", "?m"]], ["is_on", "?m"]],
    )

    result = cleanup.microwave_contract_transform(legacy)

    assert result.name == "turn_on_microwave"
    assert result.params == ["?h", "?m"]
    assert "(paper_cup ?cup)" not in literals(result.pre)


def test_composite_counter_surface_has_an_unambiguous_target_label() -> None:
    placed = action(
        "place_paper_cup_on_counter_right_of_microwave_beside_bottle",
        ["?h", "?cup", "?counter", "?bottle"],
        [
            ["hand", "?h"], ["paper_cup", "?cup"],
            ["counter_right_of_microwave", "?counter"],
            ["bottle", "?bottle"], ["holding", "?h", "?cup"],
        ],
        [
            ["not", ["holding", "?h", "?cup"]], ["hand_free", "?h"],
            ["on", "?cup", "?counter"], ["beside", "?cup", "?bottle"],
        ],
    )

    result = cleanup.canonical_place_contract(placed)

    assert result.name == (
        "place_paper_cup_on_counter_right_of_microwave_surface_beside_bottle"
    )


def test_slide_box_names_only_an_actual_blocked_guard() -> None:
    legacy = action(
        "slide_box_on_floor_right_of_cabinet",
        ["?h", "?box", "?floor", "?cabinet"],
        [
            ["hand", "?h"], ["box", "?box"], ["floor", "?floor"],
            ["cabinet", "?cabinet"], ["hand_free", "?h"],
            ["on", "?box", "?floor"], ["in_front_of", "?box", "?cabinet"],
            ["blocked", "?cabinet"],
        ],
        [
            ["not", ["in_front_of", "?box", "?cabinet"]],
            ["right_of", "?box", "?cabinet"],
            ["not", ["blocked", "?cabinet"]], ["unblocked", "?cabinet"],
        ],
    )

    result = cleanup.normalize_spatial_action_contract(legacy)

    assert result.name == (
        "slide_box_on_floor_right_of_cabinet_when_cabinet_blocked"
    )


def test_pot_lid_contract_always_consumes_temporary_upright_pose() -> None:
    legacy = action(
        "place_lid_on_pot",
        ["?h", "?lid", "?pot"],
        [
            ["hand", "?h"], ["lid", "?lid"], ["pot", "?pot"],
            ["holding", "?h", "?lid"], ["open", "?pot"],
        ],
        [
            ["not", ["holding", "?h", "?lid"]], ["hand_free", "?h"],
            ["not", ["open", "?pot"]], ["closed", "?pot"],
            ["on", "?lid", "?pot"],
        ],
    )

    result = cleanup.separable_lid_transform(legacy, 11)

    assert "(not (upright ?lid))" in literals(result.eff)


def test_kettle_start_names_only_a_real_plug_guard() -> None:
    legacy = action(
        "turn_on_kettle",
        ["?h", "?k", "?water", "?base", "?plug", "?socket", "?faucet"],
        [
            ["hand", "?h"], ["kettle", "?k"], ["water", "?water"],
            ["kettle_base", "?base"], ["plug", "?plug"], ["socket", "?socket"],
            ["faucet", "?faucet"], ["hand_free", "?h"], ["closed", "?k"],
            ["is_off", "?k"], ["in", "?water", "?k"], ["on", "?k", "?base"],
            ["inserted", "?plug", "?socket"],
        ],
        [["not", ["is_off", "?k"]], ["is_on", "?k"]],
    )

    result = cleanup.canonical_kettle_start_contract(legacy)

    assert result.name == "turn_on_kettle_when_plug_inserted"
    assert "(faucet ?faucet)" not in literals(result.pre)
    assert "(outlet ?socket)" in literals(result.pre)


def test_detergent_drawer_is_a_local_open_close_toggle() -> None:
    legacy = action(
        "push_detergent_drawer_in_washing_machine",
        ["?h", "?drawer", "?machine", "?detergent"],
        [
            ["hand", "?h"], ["detergent_drawer", "?drawer"],
            ["washing_machine", "?machine"], ["detergent", "?detergent"],
            ["hand_free", "?h"], ["open", "?drawer"],
            ["in", "?detergent", "?drawer"], ["closed", "?machine"],
        ],
        [["not", ["open", "?drawer"]], ["closed", "?drawer"]],
    )

    result = cleanup.canonical_detergent_drawer_contract(legacy)

    assert result.name == "close_detergent_drawer"
    assert result.params == ["?h", "?drawer"]
    assert literals(result.pre) == {
        "(hand ?h)", "(detergent_drawer ?drawer)",
        "(hand_free ?h)", "(open ?drawer)",
    }


def test_microwave_result_name_distinguishes_cavity_from_top() -> None:
    legacy = action(
        "turn_off_microwave",
        ["?h", "?m", "?water", "?cup"],
        [
            ["hand", "?h"], ["microwave", "?m"], ["water", "?water"],
            ["paper_cup", "?cup"], ["hand_free", "?h"], ["closed", "?m"],
            ["is_on", "?m"], ["in", "?water", "?cup"], ["on", "?cup", "?m"],
        ],
        [["not", ["is_on", "?m"]], ["is_off", "?m"], ["heated", "?water"]],
    )

    result = cleanup.microwave_contract_transform(legacy, "on")

    assert result.name == (
        "turn_off_microwave_after_heating_water_in_paper_cup_on_microwave_top"
    )


def test_turntable_pick_projects_microwave_scene_state() -> None:
    legacy = action(
        "pick_heated_paper_cup_from_microwave_turntable",
        ["?h", "?cup", "?turntable", "?microwave"],
        [
            ["hand", "?h"], ["paper_cup", "?cup"], ["turntable", "?turntable"],
            ["microwave", "?microwave"], ["hand_free", "?h"],
            ["open", "?microwave"], ["is_off", "?microwave"],
            ["on", "?cup", "?turntable"], ["in", "?turntable", "?microwave"],
        ],
        [
            ["not", ["hand_free", "?h"]], ["holding", "?h", "?cup"],
            ["not", ["on", "?cup", "?turntable"]],
        ],
    )

    result = cleanup.canonical_turntable_pick(legacy)

    assert result.params == ["?h", "?cup", "?turntable"]
    assert "(microwave ?microwave)" not in literals(result.pre)
    assert "(microwave_turntable ?turntable)" in literals(result.pre)


def test_liquid_problem_aliases_spoonful_and_empty_results() -> None:
    problem = """
(define (problem liquid)
  (:domain liquid)
  (:objects kettle spoon cup bowl)
  (:init (has_liquid kettle) (has_liquid spoon))
  (:goal (and (received_spoonful cup) (poured bowl) (drained bowl))))
"""

    result = cleanup.normalize_liquid_problem_predicates(
        problem, normalize_spoonful=True, normalize_empty=True
    )
    parsed = cleanup.parse_problem(result)

    assert {cleanup.sexp(item) for item in cleanup.problem_facts(parsed, ":init")} == {
        "(has_water kettle)", "(has_water spoon)"
    }
    assert {cleanup.sexp(item) for item in cleanup.problem_facts(parsed, ":goal")} == {
        "(has_water cup)", "(empty bowl)"
    }


def test_pick_bowl_from_sink_preserves_an_existing_faucet_guard() -> None:
    legacy = action(
        "pick_bowl_from_sink",
        ["?h", "?bowl", "?sink", "?faucet"],
        [
            ["hand", "?h"], ["faucet", "?faucet"], ["bowl", "?bowl"],
            ["sink", "?sink"], ["hand_free", "?h"],
            ["is_off", "?faucet"], ["rinsed", "?bowl"],
            ["in", "?bowl", "?sink"],
        ],
        [
            ["not", ["hand_free", "?h"]], ["holding", "?h", "?bowl"],
            ["not", ["in", "?bowl", "?sink"]],
        ],
    )

    assert cleanup.canonical_pick_contract(legacy) == legacy


def test_place_lid_away_from_kettle_keeps_opening_result() -> None:
    legacy = action(
        "place_kettle_lid_on_towel",
        ["?h", "?lid", "?towel", "?kettle"],
        [
            ["hand", "?h"], ["lid", "?lid"], ["towel", "?towel"],
            ["kettle", "?kettle"], ["holding", "?h", "?lid"],
            ["closed", "?kettle"],
        ],
        [
            ["not", ["holding", "?h", "?lid"]], ["hand_free", "?h"],
            ["on", "?lid", "?towel"], ["not", ["closed", "?kettle"]],
            ["open", "?kettle"],
        ],
    )

    result = cleanup.canonical_place_contract(legacy)

    assert result.name == "place_lid_on_towel_and_open_kettle"
    assert result.params == ["?h", "?lid", "?towel", "?kettle"]
    assert "(not (closed ?kettle))" in literals(result.eff)
    assert "(open ?kettle)" in literals(result.eff)


def test_wipe_table_drops_trash_scene_snapshot() -> None:
    legacy = action(
        "wipe_table_with_cloth",
        ["?h", "?cloth", "?table", "?bin", "?bottle", "?paper"],
        [
            ["hand", "?h"], ["cloth", "?cloth"], ["table", "?table"],
            ["trash_bin", "?bin"], ["bottle", "?bottle"],
            ["paper_ball", "?paper"], ["holding", "?h", "?cloth"],
            ["in", "?bottle", "?bin"], ["in", "?paper", "?bin"],
        ],
        [["wiped", "?table"]],
    )

    result = cleanup.wipe_contract_transform(legacy)

    assert result.name == "wipe_table_with_cloth"
    assert result.params == ["?h", "?table", "?cloth"]
    assert literals(result.pre) == {
        "(hand ?h)", "(table ?table)", "(cloth ?cloth)",
        "(holding ?h ?cloth)",
    }
    assert literals(result.eff) == {"(wiped ?table)"}


def test_cleaned_wipe_uses_a_distinct_clean_name() -> None:
    legacy = action(
        "wipe_table_with_cloth",
        ["?h", "?cloth", "?table"],
        [
            ["hand", "?h"], ["cloth", "?cloth"], ["table", "?table"],
            ["holding", "?h", "?cloth"],
        ],
        [["cleaned", "?table"]],
    )

    result = cleanup.wipe_contract_transform(legacy)

    assert result.name == "clean_table_with_cloth"
    assert literals(result.eff) == {"(cleaned ?table)"}


def test_merge_actions_numbers_same_name_with_different_contracts() -> None:
    first = mergy.ActionItem(
        name="wipe_table_with_cloth",
        param_arity=3,
        block_text="(:action wipe_table_with_cloth)",
        signature="contract-a",
        leading_comments=[],
        sources=["source-a"],
    )
    second = mergy.ActionItem(
        name="wipe_table_with_cloth",
        param_arity=3,
        block_text="(:action wipe_table_with_cloth)",
        signature="contract-b",
        leading_comments=[],
        sources=["source-b"],
    )

    merged = mergy.merge_actions([first, second])

    assert [(name, item.signature) for name, item in merged] == [
        ("wipe_table_with_cloth_1", "contract-a"),
        ("wipe_table_with_cloth_2", "contract-b"),
    ]


def test_merge_actions_combines_sources_for_the_same_contract() -> None:
    first = mergy.ActionItem(
        name="open_drawer",
        param_arity=2,
        block_text="(:action open_drawer)",
        signature="shared-contract",
        sources=["source-a"],
    )
    second = mergy.ActionItem(
        name="open_drawer",
        param_arity=2,
        block_text="(:action open_drawer)",
        signature="shared-contract",
        sources=["source-b"],
    )

    merged = mergy.merge_actions([first, second])

    assert len(merged) == 1
    assert merged[0][0] == "open_drawer"
    assert merged[0][1].sources == ["source-a", "source-b"]


def test_merge_rejects_cross_source_predicate_arity_conflicts() -> None:
    unary = mergy.PredicateItem(
        name="dispensing", arity=1, expr="(dispensing ?x)",
        sources=["source-a"],
    )
    binary = mergy.PredicateItem(
        name="dispensing", arity=2, expr="(dispensing ?x ?y)",
        sources=["source-b"],
    )

    with pytest.raises(ValueError, match="require semantic cleanup"):
        mergy.resolve_predicate_arity_collisions([unary, binary], [])


def test_unused_dispensing_declaration_is_removed_but_live_one_is_kept() -> None:
    domain = """
(define (domain water)
  (:predicates (hand ?h) (dispensing ?source ?target))
  (:action wait
    :parameters (?h)
    :precondition (and (hand ?h))
    :effect (and (hand ?h))))
"""
    problem = """
(define (problem water-one)
  (:domain water)
  (:objects hand)
  (:init (hand hand))
  (:goal (and (hand hand))))
"""

    cleaned = cleanup.remove_unused_named_predicates(
        domain, problem, {"dispensing"}
    )
    assert "(dispensing ?source ?target)" not in cleaned

    live_problem = problem.replace("(hand hand))", "(hand hand) (dispensing hand))", 1)
    live = cleanup.remove_unused_named_predicates(
        domain.replace("(dispensing ?source ?target)", "(dispensing ?cup)"),
        live_problem,
        {"dispensing"},
    )
    assert "(dispensing ?cup)" in live


def test_pick_preserves_direct_support_clear_result() -> None:
    legacy = action(
        "pick_towel_from_lid",
        ["?h", "?towel", "?lid"],
        [
            ["hand", "?h"], ["towel", "?towel"], ["lid", "?lid"],
            ["hand_free", "?h"], ["clear", "?towel"],
            ["on", "?towel", "?lid"],
        ],
        [
            ["not", ["hand_free", "?h"]], ["holding", "?h", "?towel"],
            ["not", ["clear", "?towel"]], ["clear", "?lid"],
            ["not", ["on", "?towel", "?lid"]],
        ],
    )

    result = cleanup.canonical_pick_contract(legacy)

    assert "(clear ?lid)" in literals(result.eff)


def test_place_preserves_drawer_clearance_result() -> None:
    legacy = action(
        "place_bottle_on_floor_away_from_drawer",
        ["?h", "?bottle", "?floor", "?drawer"],
        [
            ["hand", "?h"], ["bottle", "?bottle"], ["floor", "?floor"],
            ["drawer", "?drawer"], ["holding", "?h", "?bottle"],
            ["blocking", "?bottle", "?drawer"],
        ],
        [
            ["not", ["holding", "?h", "?bottle"]], ["hand_free", "?h"],
            ["on", "?bottle", "?floor"],
            ["not", ["blocking", "?bottle", "?drawer"]],
            ["clear_to_open", "?drawer"],
        ],
    )

    result = cleanup.canonical_place_contract(legacy)

    assert result.name == "place_bottle_on_floor_and_clear_drawer_to_open"
    assert "(clear_to_open ?drawer)" in literals(result.eff)
    assert result.params == ["?h", "?bottle", "?floor", "?drawer"]


def test_book_stack_repair_makes_surface_placement_clear() -> None:
    legacy = action(
        "place_book_flat_on_table",
        ["?h", "?book", "?table"],
        [
            ["hand", "?h"], ["book", "?book"], ["table", "?table"],
            ["holding", "?h", "?book"],
        ],
        [
            ["not", ["holding", "?h", "?book"]], ["hand_free", "?h"],
            ["flat", "?book"], ["on", "?book", "?table"],
        ],
    )

    result = cleanup.canonical_family_clear_action(legacy, "book")

    assert "(clear ?book)" in literals(result.eff)


def test_open_contract_names_only_a_real_local_clearance_guard() -> None:
    legacy = action(
        "open_cabinet",
        ["?h", "?cabinet", "?box", "?plug"],
        [
            ["hand", "?h"], ["cabinet", "?cabinet"],
            ["cardboard_box", "?box"], ["plug", "?plug"],
            ["hand_free", "?h"], ["closed", "?cabinet"],
            ["right_of", "?box", "?cabinet"], ["inserted", "?plug"],
        ],
        [["not", ["closed", "?cabinet"]], ["open", "?cabinet"]],
    )

    result = cleanup.canonical_open_close_contract(legacy)

    assert result.name == "open_cabinet_when_cardboard_box_right_of_cabinet"
    assert result.params == ["?h", "?cabinet", "?box"]
    assert "(inserted ?plug)" not in literals(result.pre)


@pytest.mark.parametrize(
    ("legacy_name", "expected_name"),
    [
        ("pick_block_from_table_when_clear", "pick_block_from_table"),
        ("pick_box_from_drawer_when_drawer_open", "pick_box_from_drawer"),
        (
            "place_bowl_clear_on_plate_when_plate_clear_and_plate_flat",
            "place_bowl_on_plate",
        ),
        ("place_plate_flat_clear_on_table", "place_plate_flat_on_table"),
        ("place_bowl_upright_clear_on_table", "place_bowl_upright_on_table"),
        ("pour_water_from_bottle_in_cup_when_bottle_open", "pour_water_from_bottle_in_cup"),
        ("open_drawer_when_unlocked", "open_unlocked_drawer"),
        ("open_drawer_when_clear_to_open", "open_unblocked_drawer"),
        (
            "open_drawer_when_unlocked_and_clear_to_open",
            "open_unlocked_unblocked_drawer",
        ),
        (
            "open_middle_drawer_when_top_drawer_closed",
            "open_interlocked_middle_drawer",
        ),
        ("open_microwave_when_off", "open_microwave"),
        ("close_loaded_microwave", "close_loaded_microwave"),
        (
            "turn_off_microwave_after_heating_water_in_paper_cup_when_paper_cup_in_microwave",
            "turn_off_microwave_after_heating",
        ),
        ("turn_off_faucet_after_rinsing_bowl", "turn_off_faucet_after_rinsing"),
        (
            "turn_off_hot_water_button_after_filling_cup_and_lock_child_lock",
            "turn_off_hot_water_button_after_filling",
        ),
        (
            "select_wash_cycle_with_dial_when_washing_machine_closed",
            "select_wash_cycle",
        ),
        (
            "push_start_button_on_washing_machine_when_cycle_selected",
            "start_washing_machine",
        ),
        ("turn_on_kettle_when_plug_inserted", "turn_on_kettle"),
        (
            "insert_key_from_key_set_into_drawer",
            "insert_key_from_key_set_in_drawer",
        ),
        (
            "place_lid_on_towel_and_open_kettle",
            "place_lid_on_towel_opening_kettle",
        ),
    ],
)
def test_minimal_contrast_names(legacy_name: str, expected_name: str) -> None:
    legacy = action(
        legacy_name,
        ["?h", "?o"],
        [["hand", "?h"], ["object", "?o"]],
        [["touched", "?o"]],
    )

    assert cleanup.minimal_contrast_action_name(legacy) == expected_name


def test_minimal_contrast_name_uses_persistent_device_effect() -> None:
    legacy = action(
        "press_start_button_to_turn_on_microwave",
        ["?h", "?button", "?microwave"],
        [
            ["hand", "?h"], ["start_button", "?button"],
            ["microwave", "?microwave"], ["is_off", "?microwave"],
        ],
        [["not", ["is_off", "?microwave"]], ["is_on", "?microwave"]],
    )

    assert cleanup.minimal_contrast_action_name(legacy) == "turn_on_microwave"
