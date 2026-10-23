"""Wrapper tests: PettingZoo's official api_test and Gymnasium's env checker."""

import numpy as np
import pytest


pettingzoo_test = pytest.importorskip("pettingzoo.test")
gym_checker = pytest.importorskip("gymnasium.utils.env_checker")

from micromagic.env.aec import MicroMagicAECEnv  # noqa: E402
from micromagic.env.gym_env import MicroMagicGymEnv  # noqa: E402


def test_pettingzoo_api_test() -> None:
    pettingzoo_test.api_test(MicroMagicAECEnv(), num_cycles=1000)


def test_aec_full_games_with_masked_random_actions() -> None:
    env = MicroMagicAECEnv()
    rng = np.random.default_rng(0)
    for seed in range(20):
        env.reset(seed=seed)
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            action = None if terminated or truncated else int(rng.choice(np.flatnonzero(obs["action_mask"])))
            env.step(action)


def test_gymnasium_env_checker() -> None:
    # The checker samples actions without the mask, so illegal ones become PASS.
    gym_checker.check_env(MicroMagicGymEnv(on_illegal="pass"), skip_render_check=True)


def test_gym_raises_on_illegal_action_by_default() -> None:
    env = MicroMagicGymEnv()
    env.reset(seed=0)
    illegal = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(ValueError):
        env.step(illegal)


def test_gym_episode_and_action_masks() -> None:
    env = MicroMagicGymEnv()
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        assert np.array_equal(env.action_masks(), info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        done = terminated or truncated
    assert reward in (-1.0, 0.0, 1.0)
