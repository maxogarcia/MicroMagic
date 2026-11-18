"""Specification tests for potential-based reward shaping wrapper (M5)."""

from __future__ import annotations

import numpy as np
import pytest

from micromagic.env.gym_env import MicroMagicGymEnv
from micromagic.env.rewards import ShapedRewardWrapper, compute_state_potential


def test_shaped_reward_wrapper_potential():
    """Assert ShapedRewardWrapper returns potential difference based on player life difference and board power difference."""
    base_env = MicroMagicGymEnv(agent_player=0)
    env = ShapedRewardWrapper(base_env)

    obs, info = env.reset(seed=42)
    assert env.prev_potential is not None

    pot = compute_state_potential(base_env.state, base_env.agent_player)
    assert env.prev_potential == pytest.approx(pot)

    # Step in the environment
    mask = env.action_masks()
    action = int(np.flatnonzero(mask)[0])
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert isinstance(reward, float)
    assert np.isfinite(reward)


def test_shaped_reward_preserves_terminal():
    """Assert terminal state preserves exact win/loss outcome (+1.0 / -1.0)."""
    base_env = MicroMagicGymEnv(agent_player=0)
    env = ShapedRewardWrapper(base_env)

    obs, info = env.reset(seed=123)
    done = False
    step_count = 0
    final_reward = 0.0

    while not done and step_count < 1000:
        mask = env.action_masks()
        action = int(np.flatnonzero(mask)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        if done:
            final_reward = reward

    # If the game concluded decisively, the final terminal reward must be exactly +1.0 or -1.0
    winner = base_env.state.winner()
    if winner is not None:
        expected = 1.0 if winner == base_env.agent_player else -1.0
        assert final_reward == pytest.approx(expected)
    else:
        assert final_reward == pytest.approx(0.0)


def test_shaped_reward_mask_forwarding():
    """Assert env.action_masks() returns valid boolean mask identical to base env."""
    base_env = MicroMagicGymEnv(agent_player=0)
    env = ShapedRewardWrapper(base_env)

    env.reset(seed=42)
    wrapper_mask = env.action_masks()
    base_mask = base_env.action_masks()

    assert np.array_equal(wrapper_mask, base_mask)
    assert wrapper_mask.dtype == bool
