"""Frozen contract tests for the GameState interface. Do not weaken these without team sign-off."""

import numpy as np
import pytest

from micromagic.env import spec
from micromagic.env.state import GameState


def play_random_game(seed: int, policy_seed: int = 0) -> tuple[GameState, list[np.ndarray]]:
    state = GameState(seed=seed)
    rng = np.random.default_rng(policy_seed)
    observations = []
    while not state.is_terminal():
        legal = np.flatnonzero(state.legal_actions())
        state.step(int(rng.choice(legal)))
        observations.append(state.observe(0))
    return state, observations


def test_action_ranges_partition_all_29_actions() -> None:
    indices = [
        spec.PASS,
        *spec.PLAY_HAND,
        *spec.ATTACK,
        *spec.BLOCK,
        spec.TARGET_OPPONENT,
        *spec.TARGET_OPP_CREATURE,
        *spec.TARGET_MY_CREATURE,
    ]
    assert spec.NUM_ACTIONS == 29
    assert sorted(indices) == list(range(29))
    assert len(spec.PLAY_HAND) == spec.HAND_SIZE == 7


def test_observation_shape_dtype_and_range() -> None:
    state = GameState(seed=0)
    for player in (0, 1):
        obs = state.observe(player)
        assert obs.shape == (spec.OBS_DIM,)
        assert obs.dtype == np.float32
        assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_mask_shape_and_never_empty_before_game_end() -> None:
    state = GameState(seed=1)
    rng = np.random.default_rng(1)
    while not state.is_terminal():
        mask = state.legal_actions()
        assert mask.shape == (spec.NUM_ACTIONS,) and mask.dtype == bool
        assert mask.any()
        state.step(int(rng.choice(np.flatnonzero(mask))))
    assert not state.legal_actions().any()


def test_illegal_actions_raise() -> None:
    state = GameState(seed=2)
    illegal = np.flatnonzero(~state.legal_actions())
    assert illegal.size > 0
    with pytest.raises(ValueError):
        state.step(int(illegal[0]))


@pytest.mark.parametrize("seed", range(200))
def test_random_games_terminate_without_errors(seed: int) -> None:
    state, _ = play_random_game(seed, policy_seed=seed)
    assert state.is_terminal()
    assert state.winner() in (0, 1, None)
    if state.winner() is None:
        assert state.is_truncated()


def test_same_seed_gives_identical_game() -> None:
    first, obs_a = play_random_game(seed=7)
    second, obs_b = play_random_game(seed=7)
    assert first.winner() == second.winner()
    assert len(obs_a) == len(obs_b)
    assert all(np.array_equal(a, b) for a, b in zip(obs_a, obs_b))


def test_clone_is_independent() -> None:
    state = GameState(seed=3)
    copy = state.clone()
    rng = np.random.default_rng(3)
    while not copy.is_terminal():
        copy.step(int(rng.choice(np.flatnonzero(copy.legal_actions()))))
    assert state.turn == 1 and not state.is_terminal()
    assert np.array_equal(state.observe(0), GameState(seed=3).observe(0))


def test_clone_continues_identically() -> None:
    state = GameState(seed=4)
    copy = state.clone()
    for game in (state, copy):
        rng = np.random.default_rng(4)
        while not game.is_terminal():
            game.step(int(rng.choice(np.flatnonzero(game.legal_actions()))))
    assert state.winner() == copy.winner() and state.turn == copy.turn


def test_determinize_keeps_observer_view_and_card_counts() -> None:
    state = GameState(seed=5)
    rng = np.random.default_rng(5)
    for _ in range(40):
        state.step(int(rng.choice(np.flatnonzero(state.legal_actions()))))
    sampled = state.determinize(player=0, rng=np.random.default_rng(99))

    assert np.array_equal(sampled.observe(0), state.observe(0))
    assert np.array_equal(sampled.legal_actions(), state.legal_actions())
    for player in (0, 1):
        hidden_before = np.concatenate([state.hand[player, : state.hand_count[player]],
                                        state.library[player][state.library_top[player]:]])
        hidden_after = np.concatenate([sampled.hand[player, : sampled.hand_count[player]],
                                       sampled.library[player][sampled.library_top[player]:]])
        assert sorted(hidden_before.tolist()) == sorted(hidden_after.tolist())


def test_observation_hides_opponent_hand() -> None:
    state = GameState(seed=6)
    sampled = state.determinize(player=0, rng=np.random.default_rng(0))
    sampled.hand[1, : sampled.hand_count[1]] = 0
    assert np.array_equal(sampled.observe(0), state.observe(0))
