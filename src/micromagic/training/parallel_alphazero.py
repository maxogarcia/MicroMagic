"""Distributed Ray self-play actors and throughput benchmarking for AlphaZero-lite (M4)."""

from __future__ import annotations

import time
from typing import Any

try:
    import ray

    HAS_RAY = True
except (ImportError, ModuleNotFoundError):
    ray = None
    HAS_RAY = False

import torch as th

from micromagic.agents.alphazero import PolicyValueNet
from micromagic.training.train_alphazero import play_selfplay_game


if HAS_RAY:

    @ray.remote
    class RaySelfPlayWorker:
        """Ray remote actor executing AlphaZero self-play games."""

        def __init__(self, config: dict[str, Any] | None = None) -> None:
            th.set_num_threads(1)
            self.config = config or {}

        def play_games(
            self,
            ckpt: dict[str, Any],
            seeds: list[int],
            config: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            """Run self-play games for the given seeds using the network checkpoint."""
            active_config = config if config is not None else self.config
            net = PolicyValueNet.from_checkpoint(ckpt)
            net.eval()
            return [play_selfplay_game(net, active_config, seed) for seed in seeds]

else:

    class _DummyRaySelfPlayWorker:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "Ray is not installed. Install ray via `pip install 'micromagic[rl]'` to use RaySelfPlayWorker."
            )

    RaySelfPlayWorker = _DummyRaySelfPlayWorker  # type: ignore[assignment,misc]


class RaySelfPlayPool:
    """Manages a pool of persistent RaySelfPlayWorker actors."""

    def __init__(self, num_workers: int, config: dict[str, Any] | None = None) -> None:
        if not HAS_RAY:
            raise ImportError(
                "RaySelfPlayPool requires 'ray'. Please install ray via `pip install 'micromagic[rl]'`."
            )
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.num_workers = num_workers
        self.config = config or {}
        self.workers = [
            RaySelfPlayWorker.remote(self.config) for _ in range(num_workers)
        ]

    def play_games(
        self,
        ckpt: dict[str, Any],
        seeds: list[int],
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Distribute seeds evenly across workers and collect self-play records."""
        if not seeds:
            return []

        active_config = config if config is not None else self.config
        # Partition seeds across workers
        chunks = [seeds[i :: self.num_workers] for i in range(self.num_workers)]
        futures = []
        for i, chunk in enumerate(chunks):
            if chunk:
                futures.append(
                    self.workers[i].play_games.remote(ckpt, chunk, active_config)
                )

        results_by_worker = ray.get(futures)
        return [game for worker_games in results_by_worker for game in worker_games]

    def shutdown(self) -> None:
        """Terminate all worker actors in the pool."""
        for worker in self.workers:
            try:
                ray.kill(worker)
            except Exception:
                pass
        self.workers = []

    def close(self) -> None:
        """Alias for shutdown."""
        self.shutdown()

    def __enter__(self) -> RaySelfPlayPool:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


def ray_selfplay(
    net: PolicyValueNet,
    config: dict[str, Any],
    seeds: list[int],
    pool: RaySelfPlayPool,
) -> list[dict[str, Any]]:
    """Execute distributed self-play games through a RaySelfPlayPool."""
    ckpt = {
        "hidden": net.hidden,
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
    }
    return pool.play_games(ckpt, seeds, config)


def benchmark_alphazero_scaling(
    workers: list[int],
    games_per_worker: int,
    config: dict[str, Any] | None = None,
    net: PolicyValueNet | None = None,
) -> list[dict[str, Any]]:
    """Benchmark AlphaZero self-play throughput and speedup across worker counts."""
    active_config = config or {
        "seed": 42,
        "search": {"determinizations": 1, "simulations": 4, "c_puct": 1.5},
        "selfplay": {"temperature_moves": 4, "dirichlet_alpha": 0.3},
    }

    if net is None:
        net = PolicyValueNet(hidden=[64, 64])
    ckpt = {
        "hidden": net.hidden,
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
    }

    base_seed = int(active_config.get("seed", 42))
    results: list[dict[str, Any]] = []
    baseline_games_per_sec = 1.0

    for idx, num_w in enumerate(workers):
        pool = RaySelfPlayPool(num_workers=num_w, config=active_config)
        total_games = num_w * games_per_worker
        seeds = [base_seed + idx * 10_000 + i for i in range(total_games)]

        try:
            start_time = time.perf_counter()
            games = pool.play_games(ckpt, seeds, active_config)
            elapsed_sec = time.perf_counter() - start_time
        finally:
            pool.shutdown()

        total_env_steps = sum(g.get("env_steps", 0) for g in games)
        total_search_evals = sum(g.get("search_evals", 0) for g in games)

        games_per_sec = total_games / max(elapsed_sec, 1e-6)
        env_steps_per_sec = total_env_steps / max(elapsed_sec, 1e-6)
        search_evals_per_sec = total_search_evals / max(elapsed_sec, 1e-6)

        if idx == 0 or num_w == 1:
            baseline_games_per_sec = games_per_sec

        speedup = games_per_sec / max(baseline_games_per_sec, 1e-6)
        efficiency = speedup / num_w

        results.append(
            {
                "workers": num_w,
                "games": total_games,
                "elapsed_sec": round(elapsed_sec, 4),
                "games_per_sec": round(games_per_sec, 2),
                "env_steps_per_sec": round(env_steps_per_sec, 1),
                "search_evals_per_sec": round(search_evals_per_sec, 1),
                "speedup": round(speedup, 3),
                "efficiency": round(efficiency, 3),
            }
        )

    return results
