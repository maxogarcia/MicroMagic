"""Tests for the abstract state, the 9 macro-actions, and MacroAgent."""

import os

import numpy as np
import pytest

from micromagic.abstraction import (
    NUM_ABSTRACT_STATES,
    NUM_MACROS,
    AbstractState,
    Context,
    Macro,
    MacroAgent,
    abstract_state,
    macro_mask,
    run_macro,
    scripted_macro_policy,
)
from micromagic.abstraction.features import COUNT_EDGES, HAND_EDGES, LIFE_EDGES, POWER_DIFF_EDGES, bucket
from micromagic.agents import RandomAgent
from micromagic.env.spec import CARD_IDS, EMPTY, PASS, Phase
from micromagic.env.state import GameState
from micromagic.tournaments.runner import TournamentRunner


ABSTRACTION_FUZZ_GAMES = int(os.environ.get("MICROMAGIC_ABSTRACTION_FUZZ_GAMES", "150"))


def blank_state(lands: int = 6) -> GameState:
    state = GameState(seed=0)
    state.hand[:] = EMPTY
    state.hand_count[:] = 0
    state.lands_total[:] = lands
    return state


def give(state: GameState, player: int, *names: str) -> None:
    for name in names:
        state.hand[player, state.hand_count[player]] = CARD_IDS[name]
        state.hand_count[player] += 1


def put(state: GameState, player: int, name: str) -> int:
    slot = int((state.creature[player] == EMPTY).argmax())
    state.creature[player, slot] = CARD_IDS[name]
    return slot


def opponent_attacks(state: GameState, *attacker_names: str) -> list[int]:
    """Give player 0 attackers, attack with all of them, and stop at player 1's block decision."""
    slots = [put(state, 0, n) for n in attacker_names]
    run_macro(state, 0, Macro.ATTACK_ALL)
    return slots


# Abstract state

@pytest.mark.parametrize("value, edges, expected", [
    (5, LIFE_EDGES, 0), (6, LIFE_EDGES, 1), (15, LIFE_EDGES, 2), (16, LIFE_EDGES, 3), (-3, LIFE_EDGES, 0),
    (-4, POWER_DIFF_EDGES, 0), (-3, POWER_DIFF_EDGES, 1), (0, POWER_DIFF_EDGES, 2), (3, POWER_DIFF_EDGES, 3),
    (4, POWER_DIFF_EDGES, 4), (0, COUNT_EDGES, 0), (2, COUNT_EDGES, 2), (5, COUNT_EDGES, 3),
    (0, HAND_EDGES, 0), (2, HAND_EDGES, 1), (3, HAND_EDGES, 2),
])
def test_buckets_match_the_team_plan(value: int, edges: tuple, expected: int) -> None:
    assert bucket(value, edges) == expected


def test_abstract_state_features_and_index_round_trip() -> None:
    state = blank_state()
    put(state, 0, "Hill Giant")
    put(state, 1, "Grizzly Bears")
    state.life[:] = [12, 4]
    give(state, 0, "Mountain", "Shock", "Shock")
    a = abstract_state(state, 0)
    assert a == AbstractState(my_life=2, opp_life=0, power_diff=3, my_creatures=1, opp_creatures=1, hand=2,
                              context=int(Context.PRECOMBAT))
    assert 0 <= a.index < NUM_ABSTRACT_STATES and AbstractState.from_index(a.index) == a
    assert abstract_state(state, 1).power_diff == 1  # -1 from the other side


def test_abstract_state_does_not_depend_on_hidden_cards() -> None:
    state = GameState(seed=3)
    other = state.clone()
    other.hand[1, : other.hand_count[1]] = other.hand[1, : other.hand_count[1]][::-1]
    other.library[0] = other.library[0][::-1].copy()
    assert abstract_state(state, 0) == abstract_state(other, 0)


# Availability

def test_masks_by_context() -> None:
    state = blank_state()
    give(state, 0, "Mountain")
    mask = macro_mask(state, 0)
    assert mask[Macro.PLAY_LAND] and mask[Macro.NO_ATTACK]
    assert not mask[Macro.BLOCK_VALUE] and not mask[Macro.NO_BLOCK] and not mask[Macro.ATTACK_ALL]
    assert not macro_mask(state, 1).any()  # not player 1's decision

    state = blank_state()
    opponent_attacks(state, "Grizzly Bears")
    assert state.phase != Phase.DECLARE_BLOCKERS  # no blockers: combat resolved immediately
    state = blank_state()
    put(state, 1, "Giant Spider")
    opponent_attacks(state, "Grizzly Bears")
    mask = macro_mask(state, 1)
    assert mask.tolist() == [m in (Macro.BLOCK_VALUE, Macro.NO_BLOCK) for m in Macro]


def test_attack_macros_only_before_combat() -> None:
    state = blank_state()
    put(state, 0, "Grizzly Bears")
    assert macro_mask(state, 0)[Macro.ATTACK_ALL]
    run_macro(state, 0, Macro.NO_ATTACK)
    assert state.phase == Phase.MAIN2
    mask = macro_mask(state, 0)
    assert not mask[Macro.ATTACK_ALL] and not mask[Macro.ATTACK_SAFE] and mask[Macro.NO_ATTACK]


# Each macro

def test_play_land() -> None:
    state = blank_state(lands=0)
    give(state, 0, "Shock", "Mountain")
    assert run_macro(state, 0, Macro.PLAY_LAND) == [2]  # hand slot 1
    assert state.lands_total[0] == 1 and not macro_mask(state, 0)[Macro.PLAY_LAND]


def test_cast_best_creature_picks_the_most_expensive_affordable() -> None:
    state = blank_state(lands=4)
    give(state, 0, "Grizzly Bears", "Hill Giant", "Charging Monstrosaur")
    run_macro(state, 0, Macro.CAST_BEST_CREATURE)
    assert state.creature[0, 0] == CARD_IDS["Hill Giant"]


def test_burn_face_uses_the_biggest_burn() -> None:
    state = blank_state(lands=6)
    give(state, 0, "Shock", "Lava Axe")
    put(state, 1, "Grizzly Bears")
    run_macro(state, 0, Macro.BURN_FACE)
    assert state.life[1] == 15 and state.hand[0, 0] == CARD_IDS["Shock"]


def test_burn_best_creature_kills_the_strongest_killable_with_the_cheapest_spell() -> None:
    state = blank_state(lands=6)
    give(state, 0, "Eviscerate", "Shock")
    bears = put(state, 1, "Grizzly Bears")
    dreadmaw = put(state, 1, "Colossal Dreadmaw")
    run_macro(state, 0, Macro.BURN_BEST_CREATURE)
    assert state.creature[1, dreadmaw] == EMPTY and state.creature[1, bears] == CARD_IDS["Grizzly Bears"]

    state = blank_state(lands=6)
    give(state, 0, "Eviscerate", "Shock")
    bears = put(state, 1, "Grizzly Bears")
    run_macro(state, 0, Macro.BURN_BEST_CREATURE)
    assert state.creature[1, bears] == EMPTY
    assert state.hand[0, 0] == CARD_IDS["Eviscerate"]  # Shock was enough


def test_burn_best_creature_unavailable_when_nothing_dies() -> None:
    state = blank_state()
    give(state, 0, "Shock")
    put(state, 1, "Trained Armodon")
    assert not macro_mask(state, 0)[Macro.BURN_BEST_CREATURE]


def test_attack_safe_skips_creatures_a_blocker_would_kill_for_free() -> None:
    state = blank_state()
    bears = put(state, 0, "Grizzly Bears")      # dies to the Spider and doesn't kill it
    hawk = put(state, 0, "Suntail Hawk")        # flies; Spider has reach but only 2 power... 1/1 dies
    giant = put(state, 0, "Hill Giant")         # 3 power vs 2/4: survives (2 < 3)
    put(state, 1, "Giant Spider")
    run_macro(state, 0, Macro.ATTACK_SAFE)
    attacked = {s for s in (bears, hawk, giant) if state.attacking[0, s]}
    assert attacked == {giant}


def test_attack_all_and_no_attack() -> None:
    state = blank_state()
    a, b = put(state, 0, "Grizzly Bears"), put(state, 0, "Hill Giant")
    run_macro(state, 0, Macro.ATTACK_ALL)
    assert state.life[1] == 15  # no blockers: 2 + 3

    state = blank_state()
    put(state, 0, "Grizzly Bears")
    run_macro(state, 0, Macro.NO_ATTACK)
    assert state.phase == Phase.MAIN2 and state.life[1] == 20


def test_no_attack_after_combat_ends_the_turn_and_casts_utility_spells() -> None:
    state = blank_state(lands=8)
    give(state, 0, "Natural Spring", "Divination")
    state.step(PASS)  # Main 1 -> Main 2 (no attackers)
    run_macro(state, 0, Macro.NO_ATTACK)
    assert state.active == 1
    assert state.life[0] == 28 and state.hand_count[0] == 2  # Divination drew 2


def test_block_value_blocks_when_the_blocker_survives() -> None:
    state = blank_state()
    spider = put(state, 1, "Giant Spider")
    opponent_attacks(state, "Grizzly Bears")
    run_macro(state, 1, Macro.BLOCK_VALUE)
    assert state.creature[1, spider] == CARD_IDS["Giant Spider"] and state.creature[0, 0] == EMPTY


def test_block_value_does_not_chump_unless_the_attack_is_lethal() -> None:
    state = blank_state()
    rats = put(state, 1, "Typhoid Rats")  # deathtouch: trading 1 mana for 6 is a value block
    bears = put(state, 1, "Grizzly Bears")
    opponent_attacks(state, "Colossal Dreadmaw")
    run_macro(state, 1, Macro.BLOCK_VALUE)
    assert state.creature[0, 0] == EMPTY and state.creature[1, rats] == EMPTY
    assert state.creature[1, bears] == CARD_IDS["Grizzly Bears"]

    state = blank_state()
    put(state, 1, "Grizzly Bears")
    opponent_attacks(state, "Hill Giant")
    assert not macro_mask(state, 1)[Macro.BLOCK_VALUE]  # losing the Bears for nothing at 20 life

    state = blank_state()
    bears = put(state, 1, "Grizzly Bears")
    state.life[1] = 3
    opponent_attacks(state, "Hill Giant")
    run_macro(state, 1, Macro.BLOCK_VALUE)
    assert not state.is_terminal() and state.life[1] == 3 and state.creature[1, bears] == EMPTY


def test_no_block_takes_the_damage() -> None:
    state = blank_state()
    put(state, 1, "Giant Spider")
    opponent_attacks(state, "Grizzly Bears")
    run_macro(state, 1, Macro.NO_BLOCK)
    assert state.life[1] == 18


def test_unavailable_macro_is_rejected() -> None:
    state = blank_state()
    with pytest.raises(ValueError):
        run_macro(state, 0, Macro.PLAY_LAND)


# MacroAgent

def test_random_macros_never_produce_illegal_actions() -> None:
    for game in range(ABSTRACTION_FUZZ_GAMES):
        decks = ("aggro", "midrange") if game % 2 == 0 else ("midrange", "aggro")
        state = GameState(seed=game, decks=decks, first_player=(game // 2) % 2)
        macro_seat = game % 2
        agent, other = MacroAgent(seed=game), RandomAgent(seed=game)
        while not state.is_terminal():
            player = state.current_player()
            if player == macro_seat:
                action = agent.act_from_state(state, player)
            else:
                action = other.act(state.observe(player), state.legal_actions())
            assert state.is_legal(action), f"illegal {action} in {state}"
            state.step(action)
        assert all(0 <= m < NUM_MACROS for _, m in agent.history)


def test_history_records_each_decision_and_resets_between_games() -> None:
    agent = MacroAgent(seed=0)
    state = GameState(seed=1)
    while not state.is_terminal() and state.current_player() == 0:
        state.step(agent.act_from_state(state, 0))
    first_game = list(agent.history)
    assert first_game and all(0 <= s < NUM_ABSTRACT_STATES for s, _ in first_game)
    agent.act_from_state(GameState(seed=2), 0)
    assert len(agent.history) == 1


def test_scripted_macro_policy_beats_random_in_the_tournament_runner() -> None:
    runner = TournamentRunner(
        lambda seed: GameState(seed=seed, decks=("aggro", "midrange") if seed % 4 < 2 else ("midrange", "aggro"))
    )
    scripted = MacroAgent(policy=scripted_macro_policy, name="scripted")
    result = runner.run_match(scripted, RandomAgent(seed=0, name="random"), num_games=40, base_seed=7000)
    assert result.summary()["wins"].get("scripted", 0) >= 28


def test_observation_api_is_refused() -> None:
    state = GameState(seed=0)
    with pytest.raises(TypeError, match="act_from_state"):
        MacroAgent().act(state.observe(0), state.legal_actions())


def test_unavailable_choice_from_a_policy_is_caught() -> None:
    agent = MacroAgent(policy=lambda abstract, mask: int(Macro.NO_BLOCK))
    state = GameState(seed=0)
    with pytest.raises(ValueError):
        agent.act_from_state(state, state.current_player())


def test_mask_never_empty_at_a_decision_point() -> None:
    rng = np.random.default_rng(0)
    state = GameState(seed=5)
    while not state.is_terminal():
        player = state.current_player()
        if state.phase != Phase.CHOOSE_TARGET:
            assert macro_mask(state, player).any()
        legal = np.flatnonzero(state.legal_actions())
        state.step(int(legal[rng.integers(legal.size)]))
