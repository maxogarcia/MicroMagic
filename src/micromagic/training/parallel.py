"""Distributed and multi-process parallel rollout workers (M3).

Implements vectorized environment backends:
- DummyVecEnv: Synchronous in-process execution.
- SubprocVecEnv: Multi-process worker pool using spawn on Windows.
- RayVecEnv: Distributed actor pool using Ray for cross-core / cluster rollouts.
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np
import psutil
import ray
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from micromagic.env.gym_env import MicroMagicGymEnv
from micromagic.training.common import MatchupWrapper, OpponentPool


class EnvWorkerFactory:
    """Top-level picklable callable creating a monitored MatchupWrapper gym environment."""

    def __init__(self, pool: OpponentPool, seed: int) -> None:
        self.pool = pool
        self.seed = seed

    def __call__(self) -> Monitor:
        env = MatchupWrapper(MicroMagicGymEnv(), self.pool, seed=self.seed)
        return Monitor(env)


@ray.remote
class RayEnvWorker:
    """Remote Ray actor managing a single MicroMagicGymEnv instance."""

    def __init__(self, pool: OpponentPool, seed: int) -> None:
        self.env = Monitor(MatchupWrapper(MicroMagicGymEnv(), pool, seed=seed))
        self.seed = seed

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        if done:
            terminal_obs = obs
            obs, _ = self.env.reset()
            info["terminal_observation"] = terminal_obs
        return obs, float(reward), done, info

    def env_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        target: Any = self.env
        while hasattr(target, "env"):
            if hasattr(target, method_name):
                break
            target = target.env
        method = getattr(target, method_name)
        return method(*args, **kwargs)

    def get_attr(self, attr_name: str) -> Any:
        target: Any = self.env
        while hasattr(target, "env"):
            if hasattr(target, attr_name):
                return getattr(target, attr_name)
            target = target.env
        return getattr(target, attr_name)

    def set_attr(self, attr_name: str, value: Any) -> None:
        target: Any = self.env
        while hasattr(target, "env"):
            if hasattr(target, attr_name):
                setattr(target, attr_name, value)
                return
            target = target.env
        setattr(target, attr_name, value)

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper]) -> bool:
        target: Any = self.env
        while hasattr(target, "env"):
            if isinstance(target, wrapper_class):
                return True
            target = target.env
        return isinstance(target, wrapper_class)


class RayVecEnv(VecEnv):
    """Vectorized environment backed by Ray remote actors."""

    def __init__(self, n_envs: int, pool: OpponentPool, seed: int) -> None:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)

        dummy = MicroMagicGymEnv()
        super().__init__(n_envs, dummy.observation_space, dummy.action_space)

        self.workers = [
            RayEnvWorker.remote(pool, seed=seed + 1000 + i) for i in range(n_envs)
        ]
        self._pending_step_refs: list[Any] | None = None

    def reset(self) -> np.ndarray:
        obs_refs = [w.reset.remote() for w in self.workers]
        obs_list = ray.get(obs_refs)
        return np.array(obs_list)

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_step_refs = [
            w.step.remote(int(a)) for w, a in zip(self.workers, actions)
        ]

    def step_wait(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        assert self._pending_step_refs is not None, (
            "step_async must be called before step_wait"
        )
        results = ray.get(self._pending_step_refs)
        self._pending_step_refs = None

        obs = np.array([r[0] for r in results])
        rews = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        infos = [r[3] for r in results]
        return obs, rews, dones, infos

    def close(self) -> None:
        for w in self.workers:
            ray.kill(w)
        self.workers.clear()

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Any = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        target_workers = self._get_target_workers(indices)
        refs = [
            w.env_method.remote(method_name, *method_args, **method_kwargs)
            for w in target_workers
        ]
        return ray.get(refs)

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        target_workers = self._get_target_workers(indices)
        refs = [w.get_attr.remote(attr_name) for w in target_workers]
        return ray.get(refs)

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        target_workers = self._get_target_workers(indices)
        refs = [w.set_attr.remote(attr_name, value) for w in target_workers]
        ray.get(refs)

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices: Any = None
    ) -> list[bool]:
        target_workers = self._get_target_workers(indices)
        refs = [w.env_is_wrapped.remote(wrapper_class) for w in target_workers]
        return ray.get(refs)

    def _get_target_workers(self, indices: Any) -> list[Any]:
        if indices is None:
            return self.workers
        if isinstance(indices, int):
            return [self.workers[indices]]
        return [self.workers[i] for i in indices]


def make_parallel_vec_env(
    n_envs: int,
    pool: OpponentPool,
    seed: int,
    backend: str = "auto",
) -> VecEnv:
    """Create a vectorized environment with the specified parallel backend.

    Args:
        n_envs: Number of parallel environment instances.
        pool: OpponentPool distributing opponents to envs.
        seed: Base random seed for environment and matchup generators.
        backend: "auto" (subproc if n_envs > 1 else dummy), "dummy", "subproc", or "ray".

    Returns:
        Vectorized environment implementing the SB3 VecEnv interface.
    """
    backend_key = backend.lower().strip()
    if backend_key == "auto":
        backend_key = "subproc" if n_envs > 1 else "dummy"

    if backend_key == "dummy":
        env_fns = [EnvWorkerFactory(pool, seed=seed + 1000 + i) for i in range(n_envs)]
        venv = DummyVecEnv(env_fns)
        venv.seed(seed)
        return venv

    if backend_key == "subproc":
        env_fns = [EnvWorkerFactory(pool, seed=seed + 1000 + i) for i in range(n_envs)]
        # Use spawn on Windows to ensure process isolation and clean worker startup
        venv = SubprocVecEnv(env_fns, start_method="spawn")
        venv.seed(seed)
        return venv

    if backend_key == "ray":
        return RayVecEnv(n_envs, pool, seed)

    raise ValueError(
        f"Unknown parallel backend: {backend!r}. Must be 'auto', 'dummy', 'subproc', or 'ray'."
    )


def get_system_specs() -> dict[str, Any]:
    """Collect host system hardware and operating system metadata."""
    cpu_model = platform.processor() or "Unknown"
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": sys.version.split()[0],
        "cpu_model": cpu_model,
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "logical_cores": psutil.cpu_count(logical=True) or 1,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }


def benchmark_rollouts(
    workers: list[int],
    steps_per_worker: int = 1000,
    repeats: int = 1,
    opponent: str = "random",
    seed: int = 42,
    backend: str = "subproc",
    pool: OpponentPool | None = None,
) -> list[dict[str, Any]]:
    """Benchmark environment rollout throughput across worker counts.

    Args:
        workers: List of worker counts to benchmark (e.g., [1, 2, 3, 4]).
        steps_per_worker: Number of steps executed per worker per repeat.
        repeats: Repetitions to average for each worker count.
        opponent: Baseline opponent type ("random" or "heuristic").
        seed: Random seed.
        backend: VecEnv backend to benchmark ("dummy", "subproc", or "ray").
        pool: Optional OpponentPool instance.

    Returns:
        List of dicts containing throughput metrics per worker count.
    """
    if pool is None:
        pool = OpponentPool({opponent: 1.0}, seed=seed)

    records: list[dict[str, Any]] = []
    baseline_sps: float | None = None

    for w in workers:
        run_sps: list[float] = []
        run_times: list[float] = []
        total_steps = w * steps_per_worker

        for r in range(repeats):
            # Create vectorized environment for this worker count
            run_seed = seed + (r * 10_000)
            venv = make_parallel_vec_env(w, pool, seed=run_seed, backend=backend)
            try:
                venv.reset()

                # Step through rollouts choosing random legal actions
                t0 = time.perf_counter()
                for _ in range(steps_per_worker):
                    masks = venv.env_method("action_masks")
                    actions = [int(np.random.choice(np.flatnonzero(m))) for m in masks]
                    venv.step(np.array(actions))
                t1 = time.perf_counter()

                elapsed = max(t1 - t0, 1e-6)
                sps = total_steps / elapsed
                run_sps.append(sps)
                run_times.append(elapsed)
            finally:
                venv.close()

        avg_sps = float(np.mean(run_sps))
        avg_time = float(np.mean(run_times))

        if baseline_sps is None or w == 1:
            baseline_sps = avg_sps

        speedup = avg_sps / baseline_sps if baseline_sps > 0 else 1.0
        efficiency = speedup / w if w > 0 else 1.0

        records.append(
            {
                "backend": backend,
                "workers": w,
                "total_steps": total_steps,
                "elapsed_sec": round(avg_time, 4),
                "steps_per_sec": round(avg_sps, 2),
                "speedup": round(speedup, 3),
                "efficiency": round(efficiency, 3),
            }
        )

    return records
