"""MCTSAgent as the opponent inside the Gymnasium wrapper (uses the act_from_state hook)."""

import numpy as np
import pytest


pytest.importorskip("gymnasium")

from micromagic.env.gym_env import MicroMagicGymEnv  # noqa: E402
from micromagic.search import MCTSAgent  # noqa: E402


def test_gym_episode_against_an_mcts_opponent() -> None:
    env = MicroMagicGymEnv(opponent_policy=MCTSAgent(determinizations=1, simulations=4, seed=0))
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        done = terminated or truncated
    assert reward in (-1.0, 0.0, 1.0)
