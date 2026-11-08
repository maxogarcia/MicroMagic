"""Tests for determinized MCTS: legality, tactics, perspective handling, and information hygiene."""

import numpy as np
import pytest

from micromagic.agents import RandomAgent
from micromagic.env.spec import (
    ATTACK,
    BLOCK,
    CARD_IDS,
    EMPTY,
    NUM_ACTIONS,
    PASS,
    PLAY_HAND,
    TARGET_OPPONENT,
    Phase,
)
from micromagic.env.state import GameState
from micromagic.search import MCTSAgent, RolloutEvaluator
from micromagic.tournaments.runner import TournamentRunner


def small_agent(seed: int = 0) -> MCTSAgent:
    return MCTSAgent(determinizations=2, simulations=24, seed=seed)


def blank_state() -> GameState:
    state = GameState(seed=0)
    state.hand[:] = EMPTY
    state.hand_count[:] = 0
    state.lands_total[:] = 5
    return state


def put(state: GameState, player: int, name: str) -> int:
    slot = int((state.creature[player] == EMPTY).argmax())
    state.creature[player, slot] = CARD_IDS[name]
    return slot


def test_chooses_legal_actions_and_is_deterministic() -> None:
    state = GameState(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(30):
        player = state.current_player()
        first = small_agent(1).act_from_state(state, player)
        second = small_agent(1).act_from_state(state, player)
        assert first == second and state.legal_actions()[first]
        legal = np.flatnonzero(state.legal_actions())
        state.step(int(legal[rng.integers(legal.size)]))


def test_policy_is_a_distribution_over_legal_actions() -> None:
    state = GameState(seed=2)
    probs = small_agent().policy(state, state.current_player())
    assert probs.shape == (NUM_ACTIONS,)
    assert probs.sum() == pytest.approx(1.0)
    assert (probs[~state.legal_actions()] == 0).all()


def test_attacks_for_lethal() -> None:
    state = blank_state()
    giant = put(state, 0, "Hill Giant")
    state.life[1] = 3
    state.step(PASS)  # to declare attackers
    assert state.phase == Phase.DECLARE_ATTACKERS
    assert small_agent().act_from_state(state, 0) == ATTACK.start + giant
    assert small_agent().policy(state, 0)[ATTACK.start + giant] >= 0.6


def test_burns_face_for_lethal_instead_of_a_creature() -> None:
    state = blank_state()
    state.hand[0, 0] = CARD_IDS["Shock"]
    state.hand_count[0] = 1
    put(state, 1, "Grizzly Bears")
    state.life[1] = 2
    state.step(PLAY_HAND.start)
    assert state.phase == Phase.CHOOSE_TARGET
    assert small_agent().act_from_state(state, 0) == TARGET_OPPONENT
    assert small_agent().policy(state, 0)[TARGET_OPPONENT] >= 0.6


def test_defender_blocks_a_lethal_attacker() -> None:
    # Values must be backed up from the defender's point of view during blocks.
    state = blank_state()
    giant = put(state, 0, "Hill Giant")
    put(state, 1, "Grizzly Bears")
    state.life[1] = 3
    state.step(PASS)
    state.step(ATTACK.start + giant)
    state.step(PASS)
    assert state.phase == Phase.DECLARE_BLOCKERS and state.current_player() == 1
    assert small_agent().act_from_state(state, 1) == BLOCK.start + giant
    assert small_agent().policy(state, 1)[BLOCK.start + giant] >= 0.6


def test_decisions_ignore_where_hidden_cards_really_are() -> None:
    state = GameState(seed=9)
    rng = np.random.default_rng(9)
    for _ in range(25):
        legal = np.flatnonzero(state.legal_actions())
        state.step(int(legal[rng.integers(legal.size)]))
    player, opp = state.current_player(), 1 - state.current_player()

    # Same visible game; the opponent's hand and the library orders are rearranged.
    other = state.clone()
    hidden = np.concatenate([other.hand[opp, : other.hand_count[opp]], other.library[opp][other.library_top[opp]:]])
    hidden = hidden[::-1].copy()
    n = int(other.hand_count[opp])
    other.hand[opp, :n] = hidden[:n]
    other.library[opp][other.library_top[opp]:] = hidden[n:]
    other.library[player][other.library_top[player]:] = other.library[player][other.library_top[player]:][::-1]
    assert np.array_equal(state.observe(player), other.observe(player))

    assert np.array_equal(small_agent(5).policy(state, player), small_agent(5).policy(other, player))


def test_observation_api_is_refused_with_a_clear_error() -> None:
    state = GameState(seed=0)
    with pytest.raises(TypeError, match="act_from_state"):
        small_agent().act(state.observe(0), state.legal_actions())


def test_rejects_acting_for_the_wrong_player() -> None:
    state = GameState(seed=0)
    with pytest.raises(ValueError):
        small_agent().act_from_state(state, 1 - state.current_player())


def test_rollout_evaluator_values_stay_in_range() -> None:
    evaluator = RolloutEvaluator(depth=8, seed=0)
    state = GameState(seed=1)
    for player in (0, 1):
        priors, value = evaluator(state, player)
        assert -1.0 <= value <= 1.0 and priors.sum() == pytest.approx(1.0)


def test_tournament_runner_uses_the_search_hook() -> None:
    runner = TournamentRunner(lambda seed: GameState(seed=seed))
    mcts = MCTSAgent(determinizations=1, simulations=8, seed=0)
    result = runner.run_match(mcts, RandomAgent(seed=0, name="random"), num_games=2, base_seed=0)
    assert len(result.matches) == 2
    assert all(m.winner_seat in (0, 1, None) for m in result.matches)


def test_beats_random_play() -> None:
    runner = TournamentRunner(
        lambda seed: GameState(seed=seed, decks=("aggro", "midrange") if seed % 4 < 2 else ("midrange", "aggro"))
    )
    mcts = MCTSAgent(determinizations=2, simulations=24, seed=0)
    result = runner.run_match(mcts, RandomAgent(seed=0, name="random"), num_games=8, base_seed=100)
    assert result.summary()["wins"].get("mcts", 0) >= 6
