"""One behavioral test per card in the 25-card pool (M1: 100% card coverage)."""

from collections.abc import Callable

import pytest

from micromagic.env import spec
from micromagic.env.spec import (
    ATTACK,
    BLOCK,
    CARD_IDS,
    EMPTY,
    PASS,
    PLAY_HAND,
    TARGET_MY_CREATURE,
    TARGET_OPP_CREATURE,
    TARGET_OPPONENT,
    Phase,
)
from micromagic.env.state import GameState


TESTED_CARDS: set[str] = set()


def covers(*names: str) -> Callable:
    """Record which cards a test exercises, for the coverage check at the bottom."""
    for name in names:
        if name not in CARD_IDS:
            raise KeyError(f"Unknown card {name!r}")
    TESTED_CARDS.update(names)
    return lambda test: test


# Scenario helpers. Player 0 is always the active player.

def blank_state(lands: int = 10) -> GameState:
    state = GameState(seed=0)
    state.hand[:] = EMPTY
    state.hand_count[:] = 0
    state.lands_total[:] = lands
    return state


def give(state: GameState, player: int, *names: str) -> None:
    for name in names:
        state.hand[player, state.hand_count[player]] = CARD_IDS[name]
        state.hand_count[player] += 1


def put(state: GameState, player: int, name: str, sick: bool = False) -> int:
    slot = int((state.creature[player] == EMPTY).argmax())
    state.creature[player, slot] = CARD_IDS[name]
    state.sick[player, slot] = sick
    return slot


def cast(state: GameState, name: str, target: int | None = None) -> None:
    slot = [int(c) for c in state.hand[0, : state.hand_count[0]]].index(CARD_IDS[name])
    state.step(PLAY_HAND.start + slot)
    if target is not None:
        assert state.phase == Phase.CHOOSE_TARGET
        state.step(target)


def attack(state: GameState, *slots: int, blocks: dict[int, int] | None = None) -> None:
    """Attack with `slots`; `blocks` maps defender slot -> attacker slot it blocks."""
    blocks = blocks or {}
    state.step(PASS)
    assert state.phase == Phase.DECLARE_ATTACKERS
    for slot in slots:
        state.step(ATTACK.start + slot)
    state.step(PASS)
    while state.phase == Phase.DECLARE_BLOCKERS:
        blocker = int(state.blocker_queue[state.blocker_cursor])
        state.step(BLOCK.start + blocks[blocker] if blocker in blocks else PASS)


@covers("Mountain", "Forest")
@pytest.mark.parametrize("land", ["Mountain", "Forest"])
def test_lands_add_one_mana_source(land: str) -> None:
    state = blank_state(lands=0)
    give(state, 0, land)
    cast(state, land)
    assert state.lands_total[0] == 1 and state._untapped_lands(0) == 1


@covers("Raging Goblin")
def test_raging_goblin_haste_attacks_the_turn_it_is_cast() -> None:
    state = blank_state()
    give(state, 0, "Raging Goblin")
    cast(state, "Raging Goblin")
    attack(state, 0)
    assert state.life[1] == spec.STARTING_LIFE - 1


@covers("Suntail Hawk", "Giant Spider")
def test_flyer_cannot_be_blocked_by_ground_creature_but_reach_can() -> None:
    state = blank_state()
    hawk = put(state, 0, "Suntail Hawk")
    put(state, 1, "Grizzly Bears")
    attack(state, hawk)
    assert state.life[1] == spec.STARTING_LIFE - 1  # bears had no legal block

    state = blank_state()
    hawk = put(state, 0, "Suntail Hawk")
    spider = put(state, 1, "Giant Spider")
    attack(state, hawk, blocks={spider: hawk})
    assert state.creature[0, hawk] == EMPTY and state.life[1] == spec.STARTING_LIFE


@covers("Goblin Piker")
def test_goblin_piker_cannot_block() -> None:
    state = blank_state()
    bears = put(state, 0, "Grizzly Bears")
    put(state, 1, "Goblin Piker")
    state.step(PASS)
    state.step(ATTACK.start + bears)
    state.step(PASS)
    assert state.phase == Phase.MAIN2  # no blockers were offered
    assert state.life[1] == spec.STARTING_LIFE - 2


@covers("Wind Drake", "Serra Angel")
def test_flyer_can_block_flyer() -> None:
    state = blank_state()
    drake = put(state, 0, "Wind Drake")
    angel = put(state, 1, "Serra Angel")
    attack(state, drake, blocks={angel: drake})
    assert state.creature[0, drake] == EMPTY and state.creature[1, angel] == CARD_IDS["Serra Angel"]


@covers("Serra Angel")
def test_serra_angel_vigilance_does_not_tap() -> None:
    state = blank_state()
    angel = put(state, 0, "Serra Angel")
    attack(state, angel)
    assert not state.tapped[0, angel] and state.life[1] == spec.STARTING_LIFE - 4


@covers("Gray Ogre", "Grizzly Bears")
def test_vanilla_two_twos_trade() -> None:
    state = blank_state(lands=3)
    give(state, 0, "Gray Ogre")
    cast(state, "Gray Ogre")
    assert state.lands_tapped[0] == 3
    state.sick[0, 0] = False
    bears = put(state, 1, "Grizzly Bears")
    attack(state, 0, blocks={bears: 0})
    assert state.creature[0, 0] == EMPTY and state.creature[1, bears] == EMPTY


@covers("Hill Giant", "Trained Armodon")
def test_three_three_kills_two_two_blocker_and_survives() -> None:
    for name in ("Hill Giant", "Trained Armodon"):
        state = blank_state()
        attacker = put(state, 0, name)
        bears = put(state, 1, "Grizzly Bears")
        attack(state, attacker, blocks={bears: attacker})
        assert state.creature[1, bears] == EMPTY
        assert state.creature[0, attacker] == CARD_IDS[name] and state.damage[0, attacker] == 2


@covers("Charging Monstrosaur")
def test_charging_monstrosaur_haste_and_trample() -> None:
    state = blank_state()
    give(state, 0, "Charging Monstrosaur")
    cast(state, "Charging Monstrosaur")
    bears = put(state, 1, "Grizzly Bears")
    attack(state, 0, blocks={bears: 0})
    assert state.creature[1, bears] == EMPTY
    assert state.life[1] == spec.STARTING_LIFE - 3  # 5 power, 2 lethal to the blocker


@covers("Shock")
def test_shock_hits_face_or_kills_a_two_toughness_creature() -> None:
    state = blank_state()
    give(state, 0, "Shock", "Shock")
    bears = put(state, 1, "Grizzly Bears")
    cast(state, "Shock", TARGET_OPPONENT)
    assert state.life[1] == spec.STARTING_LIFE - 2 and state.phase == Phase.MAIN1
    cast(state, "Shock", TARGET_OPP_CREATURE.start + bears)
    assert state.creature[1, bears] == EMPTY


@covers("Volcanic Hammer", "Trained Armodon")
def test_volcanic_hammer_kills_three_toughness_shock_does_not() -> None:
    state = blank_state()
    give(state, 0, "Shock", "Volcanic Hammer")
    armodon = put(state, 1, "Trained Armodon")
    cast(state, "Shock", TARGET_OPP_CREATURE.start + armodon)
    assert state.creature[1, armodon] == CARD_IDS["Trained Armodon"]
    cast(state, "Volcanic Hammer", TARGET_OPP_CREATURE.start + armodon)
    assert state.creature[1, armodon] == EMPTY


@covers("Lava Axe")
def test_lava_axe_targets_only_the_opponent() -> None:
    state = blank_state()
    give(state, 0, "Lava Axe")
    put(state, 1, "Grizzly Bears")
    put(state, 0, "Grizzly Bears")
    state.step(PLAY_HAND.start)
    mask = state.legal_actions()
    assert mask[TARGET_OPPONENT] and mask.sum() == 1
    state.step(TARGET_OPPONENT)
    assert state.life[1] == spec.STARTING_LIFE - 5


@covers("Giant Growth")
def test_giant_growth_saves_a_creature_and_wears_off() -> None:
    state = blank_state()
    give(state, 0, "Giant Growth")
    bears = put(state, 0, "Grizzly Bears")
    giant = put(state, 1, "Hill Giant")
    cast(state, "Giant Growth", TARGET_MY_CREATURE.start + bears)
    assert state.power(0, bears) == 5 and state.toughness(0, bears) == 5
    attack(state, bears, blocks={giant: bears})
    assert state.creature[0, bears] == CARD_IDS["Grizzly Bears"] and state.creature[1, giant] == EMPTY
    state.step(PASS)  # end turn
    assert state.power(0, bears) == 2 and state.damage[0, bears] == 0


@covers("Giant Growth")
def test_targeted_spell_needs_a_creature_on_the_battlefield() -> None:
    state = blank_state()
    give(state, 0, "Giant Growth")
    assert not state.legal_actions()[PLAY_HAND.start]


@covers("Typhoid Rats", "Colossal Dreadmaw")
def test_deathtouch_blocker_kills_a_huge_attacker() -> None:
    state = blank_state()
    dreadmaw = put(state, 0, "Colossal Dreadmaw")
    rats = put(state, 1, "Typhoid Rats")
    attack(state, dreadmaw, blocks={rats: dreadmaw})
    assert state.creature[0, dreadmaw] == EMPTY and state.creature[1, rats] == EMPTY
    assert state.life[1] == spec.STARTING_LIFE - 5  # trample: 1 toughness, 5 excess


@covers("Colossal Dreadmaw")
def test_colossal_dreadmaw_tramples_over_a_chump_blocker() -> None:
    state = blank_state()
    dreadmaw = put(state, 0, "Colossal Dreadmaw")
    bears = put(state, 1, "Grizzly Bears")
    attack(state, dreadmaw, blocks={bears: dreadmaw})
    assert state.life[1] == spec.STARTING_LIFE - 4
    assert state.creature[0, dreadmaw] == CARD_IDS["Colossal Dreadmaw"]


@covers("Healer's Hawk")
def test_healers_hawk_lifelink_on_unblocked_damage() -> None:
    state = blank_state()
    hawk = put(state, 0, "Healer's Hawk")
    attack(state, hawk)
    assert state.life.tolist() == [spec.STARTING_LIFE + 1, spec.STARTING_LIFE - 1]


@covers("Child of Night")
def test_child_of_night_lifelink_on_damage_to_a_blocker() -> None:
    state = blank_state()
    child = put(state, 0, "Child of Night")
    bears = put(state, 1, "Grizzly Bears")
    attack(state, child, blocks={bears: child})
    assert state.life[0] == spec.STARTING_LIFE + 2
    assert state.creature[0, child] == EMPTY and state.creature[1, bears] == EMPTY


@covers("Deadly Recluse", "Serra Angel")
def test_deadly_recluse_reach_and_deathtouch_kill_a_flyer() -> None:
    state = blank_state()
    angel = put(state, 0, "Serra Angel")
    recluse = put(state, 1, "Deadly Recluse")
    attack(state, angel, blocks={recluse: angel})
    assert state.creature[0, angel] == EMPTY and state.creature[1, recluse] == EMPTY


@covers("Giant Spider")
def test_giant_spider_blocks_and_survives_a_two_power_flyer() -> None:
    state = blank_state()
    drake = put(state, 0, "Wind Drake")
    spider = put(state, 1, "Giant Spider")
    attack(state, drake, blocks={spider: drake})
    assert state.creature[0, drake] == EMPTY and state.creature[1, spider] == CARD_IDS["Giant Spider"]


@covers("Eviscerate")
def test_eviscerate_destroys_any_creature() -> None:
    state = blank_state()
    give(state, 0, "Eviscerate")
    dreadmaw = put(state, 1, "Colossal Dreadmaw")
    cast(state, "Eviscerate", TARGET_OPP_CREATURE.start + dreadmaw)
    assert state.creature[1, dreadmaw] == EMPTY and state.lands_tapped[0] == 4
    give(state, 0, "Eviscerate")
    assert not state.legal_actions()[PLAY_HAND.start]  # no creatures left to target


@covers("Divination")
def test_divination_draws_two_up_to_the_hand_cap() -> None:
    state = blank_state()
    give(state, 0, "Divination")
    library_before = state._library_count(0)
    cast(state, "Divination")
    assert state.hand_count[0] == 2 and state._library_count(0) == library_before - 2

    state = blank_state()
    give(state, 0, "Divination", *["Mountain"] * 5)
    cast(state, "Divination")
    assert state.hand_count[0] == 7  # 6 after casting, only 1 draw fits


@covers("Natural Spring")
def test_natural_spring_gains_eight_life() -> None:
    state = blank_state()
    give(state, 0, "Natural Spring")
    cast(state, "Natural Spring")
    assert state.life[0] == spec.STARTING_LIFE + 8 and state.phase == Phase.MAIN1


def test_every_card_in_the_pool_has_a_test() -> None:
    missing = set(CARD_IDS) - TESTED_CARDS
    assert not missing, f"Cards without a behavioral test: {sorted(missing)}"
