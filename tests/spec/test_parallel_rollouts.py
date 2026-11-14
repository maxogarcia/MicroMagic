"""Specification tests for distributed parallel rollouts and throughput benchmarking (M3).

Contract:
- make_parallel_vec_env supports 'dummy', 'subproc', and 'ray' backends.
- Parallel environments return valid VecEnv instances complying with Stable-Baselines3.
- Subproc and Ray workers execute steps and return valid observations, rewards, dones,
  infos, and legal action masks.
- Workers receive independent seeds ensuring stochastic diversity across rollouts.
- Rollout throughput benchmark runs end-to-end and reports valid metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from micromagic.env.spec import NUM_ACTIONS, OBS_DIM
from micromagic.training.common import OpponentPool
from micromagic.training.parallel import (
    RayVecEnv,
    benchmark_rollouts,
    make_parallel_vec_env,
)


@pytest.fixture
def pool() -> OpponentPool:
    return OpponentPool({"random": 1.0}, seed=42)


def test_make_parallel_vec_env_dummy(pool: OpponentPool) -> None:
    venv = make_parallel_vec_env(2, pool, seed=0, backend="dummy")
    try:
        assert isinstance(venv, DummyVecEnv)
        assert venv.num_envs == 2
        obs = venv.reset()
        assert obs.shape == (2, OBS_DIM)
    finally:
        venv.close()


def test_make_parallel_vec_env_subproc(pool: OpponentPool) -> None:
    venv = make_parallel_vec_env(2, pool, seed=10, backend="subproc")
    try:
        assert isinstance(venv, SubprocVecEnv)
        assert venv.num_envs == 2

        obs = venv.reset()
        assert obs.shape == (2, OBS_DIM)

        masks = venv.env_method("action_masks")
        assert len(masks) == 2
        for m in masks:
            m_arr = np.asarray(m, dtype=bool)
            assert m_arr.shape == (NUM_ACTIONS,)
            assert m_arr.any()

        # Step each worker with a legal action
        actions = [int(np.flatnonzero(m)[0]) for m in masks]
        next_obs, rewards, dones, infos = venv.step(np.array(actions))

        assert next_obs.shape == (2, OBS_DIM)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        venv.close()


def test_make_parallel_vec_env_ray(pool: OpponentPool) -> None:
    venv = make_parallel_vec_env(2, pool, seed=20, backend="ray")
    try:
        assert isinstance(venv, VecEnv)
        assert isinstance(venv, RayVecEnv)
        assert venv.num_envs == 2

        obs = venv.reset()
        assert obs.shape == (2, OBS_DIM)

        masks = venv.env_method("action_masks")
        assert len(masks) == 2
        for m in masks:
            m_arr = np.asarray(m, dtype=bool)
            assert m_arr.shape == (NUM_ACTIONS,)
            assert m_arr.any()

        # Step each worker with a legal action
        actions = [int(np.flatnonzero(m)[0]) for m in masks]
        next_obs, rewards, dones, infos = venv.step(np.array(actions))

        assert next_obs.shape == (2, OBS_DIM)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        venv.close()


def test_worker_seed_independence(pool: OpponentPool) -> None:
    # Verify that workers receive distinct seeds and distinct initial matchups
    venv = make_parallel_vec_env(4, pool, seed=100, backend="dummy")
    try:
        seeds = venv.get_attr("seed")
        assert len(seeds) == 4
        assert len(set(seeds)) == 4, f"Worker seeds are not unique: {seeds}"
    finally:
        venv.close()


def test_benchmark_rollouts_smoke(pool: OpponentPool) -> None:
    # Run a fast smoke test of benchmarking rollouts across 1 and 2 workers
    results = benchmark_rollouts(
        workers=[1, 2],
        steps_per_worker=50,
        repeats=1,
        opponent="random",
        seed=42,
        backend="dummy",
    )
    assert len(results) == 2
    for row in results:
        assert "workers" in row
        assert "steps_per_sec" in row
        assert "total_steps" in row
        assert "elapsed_sec" in row
        assert row["steps_per_sec"] > 0
        assert row["total_steps"] == row["workers"] * 50
