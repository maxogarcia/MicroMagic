"""Specification tests for Ray-powered distributed AlphaZero self-play scaling (M4)."""

from __future__ import annotations

import numpy as np
import pytest
import ray

from micromagic.agents.alphazero import PolicyValueNet
from micromagic.env.spec import NUM_ACTIONS, OBS_DIM
from micromagic.training.parallel_alphazero import (
    RaySelfPlayPool,
    RaySelfPlayWorker,
    benchmark_alphazero_scaling,
)


@pytest.fixture(scope="module")
def ray_cluster():
    """Ensure a local Ray instance is initialized for tests."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False)
    yield
    # Keep ray running across module tests to reduce startup overhead


@pytest.fixture
def smoke_config() -> dict:
    return {
        "seed": 42,
        "search": {
            "determinizations": 1,
            "simulations": 2,
            "c_puct": 1.5,
        },
        "selfplay": {
            "temperature_moves": 2,
            "dirichlet_alpha": 0.3,
        },
    }


@pytest.fixture
def smoke_ckpt() -> dict:
    net = PolicyValueNet(hidden=[32])
    return {
        "hidden": net.hidden,
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
    }


def test_ray_selfplay_worker(ray_cluster, smoke_config, smoke_ckpt):
    """Assert RaySelfPlayWorker actor runs play_games(ckpt, seeds) and returns valid selfplay dicts."""
    worker = RaySelfPlayWorker.remote(smoke_config)
    try:
        seeds = [100, 101]
        future = worker.play_games.remote(smoke_ckpt, seeds)
        games = ray.get(future)

        assert isinstance(games, list)
        assert len(games) == len(seeds)

        required_keys = {
            "obs",
            "masks",
            "pis",
            "zs",
            "env_steps",
            "decisions",
            "search_evals",
            "winner",
        }
        for g in games:
            assert required_keys.issubset(g.keys())
            obs = g["obs"]
            masks = g["masks"]
            pis = g["pis"]
            zs = g["zs"]

            assert isinstance(obs, np.ndarray)
            assert obs.ndim == 2
            assert obs.shape[1] == OBS_DIM

            assert isinstance(masks, np.ndarray)
            assert masks.ndim == 2
            assert masks.shape[1] == NUM_ACTIONS
            assert masks.dtype == bool

            assert isinstance(pis, np.ndarray)
            assert pis.ndim == 2
            assert pis.shape[1] == NUM_ACTIONS

            assert isinstance(zs, np.ndarray)
            assert len(zs) == len(obs)

            assert isinstance(g["env_steps"], int)
            assert g["env_steps"] > 0
            assert isinstance(g["decisions"], int)
            assert isinstance(g["search_evals"], int)
    finally:
        ray.kill(worker)


def test_ray_selfplay_pool(ray_cluster, smoke_config, smoke_ckpt):
    """Assert RaySelfPlayPool(num_workers=2) executes self-play across 4 seeds and returns 4 game records."""
    pool = RaySelfPlayPool(num_workers=2, config=smoke_config)
    try:
        seeds = [200, 201, 202, 203]
        games = pool.play_games(smoke_ckpt, seeds)

        assert isinstance(games, list)
        assert len(games) == len(seeds)

        for g in games:
            assert "obs" in g
            assert "masks" in g
            assert "pis" in g
            assert "zs" in g
            assert "env_steps" in g
            assert "decisions" in g
            assert "search_evals" in g
    finally:
        pool.shutdown()


def test_benchmark_alphazero_smoke(ray_cluster, smoke_config):
    """Assert benchmark_alphazero_scaling(workers=[1, 2], games_per_worker=1) returns valid throughput metrics."""
    results = benchmark_alphazero_scaling(
        workers=[1, 2],
        games_per_worker=1,
        config=smoke_config,
    )

    assert isinstance(results, list)
    assert len(results) == 2

    required_fields = {
        "workers",
        "games",
        "elapsed_sec",
        "games_per_sec",
        "env_steps_per_sec",
        "search_evals_per_sec",
        "speedup",
        "efficiency",
    }
    for record in results:
        assert required_fields.issubset(record.keys())
        assert record["workers"] in [1, 2]
        assert record["games"] >= 1
        assert record["elapsed_sec"] > 0
        assert record["games_per_sec"] > 0
        assert record["env_steps_per_sec"] > 0
        assert record["search_evals_per_sec"] > 0
        assert record["speedup"] > 0
        assert record["efficiency"] > 0
