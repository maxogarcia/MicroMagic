"""Train AlphaZero-lite by self-play, then evaluate it with the tournament harness.

Usage:
    python -m micromagic.training.train_alphazero --config configs/alphazero_full.yaml
    python -m micromagic.training.train_alphazero --config configs/alphazero_smoke.yaml

Each iteration:
1. Self-play. `workers` processes play `games_per_iteration` games with the current
   network on both seats. At every non-forced decision the determinized MCTS visit
   distribution pi is recorded together with the observation and the legal mask. The
   first `temperature_moves` decisions of a game are sampled from pi (exploration,
   plus Dirichlet noise at the root); later ones take the most visited action.
   Decks alternate between games, so both decks are played from both seats.
2. Every recorded decision gets the final outcome z (+1 win, -1 loss, 0 draw) from the
   point of view of the seat that made it, and goes into a replay buffer holding the
   newest `buffer_size` samples.
3. Training. `train_steps` minibatches of loss = CE(pi, policy) + MSE(z, value), with
   L2 regularization through Adam's weight_decay.

The budget is counted in real env steps (`total_timesteps`), the same unit as PPO and
DQN; the simulated steps inside the search are reported separately as `search_evals`.

Writes to <out_dir>/<run_name>/: config.yaml, progress.csv (one row per iteration),
model.pt (latest network), ckpt_iter<N>.pt (every `checkpoint_every` iterations),
learning_curve.csv (if `curve` is set), and the eval CSVs written by common.evaluate.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import torch as th

from micromagic.agents.alphazero import (
    AlphaZeroAgent,
    PolicyNetAgent,
    PolicyValueNet,
    az_loss,
)
from micromagic.env.spec import NUM_ACTIONS, OBS_DIM
from micromagic.env.state import GameState
from micromagic.training import common
from micromagic.training.common import DECK_PAIRS


# Self-play


def search_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    s = config.get("search", {})
    return {
        "determinizations": int(s.get("determinizations", 2)),
        "simulations": int(s.get("simulations", 32)),
        "c_puct": float(s.get("c_puct", 1.5)),
    }


def play_selfplay_game(
    net: PolicyValueNet, config: dict[str, Any], seed: int, max_steps: int = 1000
) -> dict[str, Any]:
    """One self-play game. Returns the training samples and step counts for it."""
    sp = config.get("selfplay", {})
    temperature_moves = int(sp.get("temperature_moves", 10))
    agent = AlphaZeroAgent(
        net,
        dirichlet_alpha=float(sp.get("dirichlet_alpha", 0.3)),
        seed=seed,
        name="selfplay",
        **search_kwargs(config),
    )
    rng = np.random.default_rng(seed)
    state = GameState(seed=seed, decks=DECK_PAIRS[seed % 2])

    obs, masks, pis, players = [], [], [], []
    steps = decisions = 0
    while not state.is_terminal() and steps < max_steps:
        player = state.current_player()
        mask = state.legal_actions()
        legal = np.flatnonzero(mask)
        if legal.size == 1:
            # A forced move teaches nothing and needs no search.
            action = int(legal[0])
        else:
            pi = agent.policy(state, player)
            obs.append(state.observe(player))
            masks.append(mask.copy())
            pis.append(pi.astype(np.float32))
            players.append(player)
            if decisions < temperature_moves:
                action = int(rng.choice(NUM_ACTIONS, p=pi))
            else:
                best = np.flatnonzero(pi == pi.max())
                action = int(best[rng.integers(best.size)])
            decisions += 1
        state.step(action)
        steps += 1

    winner = state.winner() if state.is_terminal() else None
    players_arr = np.asarray(players, dtype=np.int64)
    if winner is None:
        zs = np.zeros(len(players), dtype=np.float32)
    else:
        zs = np.where(players_arr == winner, 1.0, -1.0).astype(np.float32)
    return {
        "obs": np.asarray(obs, dtype=np.float32).reshape(-1, OBS_DIM),
        "masks": np.asarray(masks, dtype=bool).reshape(-1, NUM_ACTIONS),
        "pis": np.asarray(pis, dtype=np.float32).reshape(-1, NUM_ACTIONS),
        "zs": zs,
        "env_steps": steps,
        "decisions": decisions,
        "search_evals": agent.mcts.evaluator.calls,
        "winner": winner,
    }


def _init_worker() -> None:
    # One thread per process: many small forward passes, and `workers` processes in parallel.
    th.set_num_threads(1)


def _play_games(
    ckpt: dict[str, Any], config: dict[str, Any], seeds: list[int]
) -> list[dict[str, Any]]:
    net = PolicyValueNet.from_checkpoint(ckpt)
    return [play_selfplay_game(net, config, seed) for seed in seeds]


def selfplay(
    net: PolicyValueNet,
    config: dict[str, Any],
    seeds: list[int],
    executor: ProcessPoolExecutor | None = None,
    ray_pool: Any | None = None,
) -> list[dict[str, Any]]:
    if ray_pool is not None:
        from micromagic.training.parallel_alphazero import ray_selfplay

        return ray_selfplay(net, config, seeds, ray_pool)
    if executor is None:
        return [play_selfplay_game(net, config, s) for s in seeds]
    workers = executor._max_workers  # noqa: SLF001 - the pool size we created
    chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    ckpt = {
        "hidden": net.hidden,
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
    }
    futures = [executor.submit(_play_games, ckpt, config, chunk) for chunk in chunks]
    return [game for f in futures for game in f.result()]


# Replay buffer and training


class ReplayBuffer:
    """The newest `capacity` (obs, mask, pi, z) samples."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((0, OBS_DIM), dtype=np.float32)
        self.masks = np.zeros((0, NUM_ACTIONS), dtype=bool)
        self.pis = np.zeros((0, NUM_ACTIONS), dtype=np.float32)
        self.zs = np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.zs)

    def add(self, games: list[dict[str, Any]]) -> None:
        self.obs = np.concatenate([self.obs] + [g["obs"] for g in games])[
            -self.capacity :
        ]
        self.masks = np.concatenate([self.masks] + [g["masks"] for g in games])[
            -self.capacity :
        ]
        self.pis = np.concatenate([self.pis] + [g["pis"] for g in games])[
            -self.capacity :
        ]
        self.zs = np.concatenate([self.zs] + [g["zs"] for g in games])[-self.capacity :]

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[th.Tensor, ...]:
        idx = rng.integers(len(self), size=batch_size)
        return (
            th.as_tensor(self.obs[idx]),
            th.as_tensor(self.masks[idx]),
            th.as_tensor(self.pis[idx]),
            th.as_tensor(self.zs[idx]),
        )


def train_network(
    net: PolicyValueNet,
    optimizer: th.optim.Optimizer,
    buffer: ReplayBuffer,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    net.train()
    totals = np.zeros(3)
    for _ in range(steps):
        loss, policy_loss, value_loss = az_loss(net, *buffer.sample(batch_size, rng))
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        totals += [loss.item(), policy_loss.item(), value_loss.item()]
    net.eval()
    loss, policy_loss, value_loss = totals / max(steps, 1)
    return {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss}


# Evaluation


def make_eval_agent(
    net: PolicyValueNet, config: dict[str, Any], name: str = "alphazero"
) -> AlphaZeroAgent:
    """Search agent without exploration noise; `eval_search` overrides the self-play search sizes."""
    kwargs = search_kwargs(config) | {
        k: v for k, v in config.get("eval_search", {}).items()
    }
    return AlphaZeroAgent(net, seed=int(config["seed"]), name=name, **kwargs)


def record_curve(
    net: PolicyValueNet,
    config: dict[str, Any],
    iteration: int,
    env_steps: int,
    rows: list[dict[str, Any]],
) -> None:
    curve = config["curve"]
    seed, games = int(curve.get("seed", 200_000)), int(curve["games"])
    agents = [make_eval_agent(net, config)]
    if curve.get("policy_only", True):
        agents.append(PolicyNetAgent(net, name="az_policy"))
    for agent in agents:
        for kind in curve.get("opponents", ["random", "heuristic"]):
            result = common.play_matchups(
                agent, common.make_opponent(kind, seed), games, seed
            )
            overall = common.by_deck_and_seat(result, agent.name, seed).iloc[-1]
            rows.append(
                {
                    "timesteps": env_steps,
                    "iteration": iteration,
                    "agent": agent.name,
                    "opponent": kind,
                }
                | overall[["win_rate", "ci_lower", "ci_upper"]].to_dict()
            )


# Main loop


def train(config: dict[str, Any]) -> PolicyValueNet:
    out = common.run_dir(config)
    common.save_config(config, out)

    seed = int(config["seed"])
    # Batch-1 inference during evaluation is slower with many threads, and small
    # training batches gain little past a few.
    th.set_num_threads(int(config.get("torch_threads", 4)))
    th.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = PolicyValueNet(config.get("net", {}).get("hidden", [256, 256]))
    net.eval()
    opt_cfg = config.get("optim", {})
    optimizer = th.optim.Adam(
        net.parameters(),
        lr=float(opt_cfg.get("lr", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )
    buffer = ReplayBuffer(int(config.get("buffer_size", 100_000)))

    total_timesteps = int(config["total_timesteps"])
    games_per_iteration = int(config["games_per_iteration"])
    train_steps = int(config.get("train_steps", 200))
    batch_size = int(config.get("batch_size", 256))
    min_buffer = int(config.get("min_buffer", batch_size))
    checkpoint_every = int(config.get("checkpoint_every", 0))
    curve_every = int(config["curve"]["every_iterations"]) if config.get("curve") else 0
    workers = int(config.get("workers", 1))
    backend = str(
        config.get("backend", config.get("selfplay", {}).get("backend", "process"))
    ).lower()

    executor = None
    ray_pool = None
    if workers > 1:
        if backend == "ray":
            from micromagic.training.parallel_alphazero import RaySelfPlayPool

            ray_pool = RaySelfPlayPool(workers, config)
        else:
            executor = ProcessPoolExecutor(
                workers, mp_context=mp.get_context("spawn"), initializer=_init_worker
            )

    progress: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    env_steps = search_evals = iteration = 0
    try:
        while env_steps < total_timesteps:
            iteration += 1
            start = time.perf_counter()
            seeds = [
                seed * 10_000_000 + (iteration - 1) * games_per_iteration + i
                for i in range(games_per_iteration)
            ]
            games = selfplay(net, config, seeds, executor, ray_pool=ray_pool)
            selfplay_seconds = time.perf_counter() - start
            buffer.add(games)
            steps = sum(g["env_steps"] for g in games)
            env_steps += steps
            search_evals += sum(g["search_evals"] for g in games)

            start = time.perf_counter()
            losses = {"loss": np.nan, "policy_loss": np.nan, "value_loss": np.nan}
            if len(buffer) >= min_buffer:
                losses = train_network(
                    net, optimizer, buffer, train_steps, batch_size, rng
                )
            train_seconds = time.perf_counter() - start

            winners = [g["winner"] for g in games]
            row = {
                "iteration": iteration,
                "timesteps": env_steps,
                "search_evals": search_evals,
                "games": len(games),
                "samples": sum(len(g["zs"]) for g in games),
                "buffer": len(buffer),
                "seat0_win_rate": float(np.mean([w == 0 for w in winners])),
                "draw_rate": float(np.mean([w is None for w in winners])),
                "mean_game_steps": steps / len(games),
                "selfplay_seconds": round(selfplay_seconds, 2),
                "train_seconds": round(train_seconds, 2),
                "env_steps_per_sec": round(steps / selfplay_seconds, 1),
            } | losses
            progress.append(row)
            pd.DataFrame(progress).to_csv(out / "progress.csv", index=False)
            print(
                f"iter {iteration:4d}  steps {env_steps:>9,}  loss {row['loss']:.3f} "
                f"(pi {row['policy_loss']:.3f}, v {row['value_loss']:.3f})  "
                f"{row['env_steps_per_sec']:.0f} env steps/s  seat0 {row['seat0_win_rate']:.2f}",
                flush=True,
            )

            th.save(net.checkpoint(), out / "model.pt")
            if checkpoint_every and iteration % checkpoint_every == 0:
                th.save(net.checkpoint(), out / f"ckpt_iter{iteration}.pt")
            if curve_every and iteration % curve_every == 0:
                record_curve(net, config, iteration, env_steps, curve_rows)
                pd.DataFrame(curve_rows).to_csv(out / "learning_curve.csv", index=False)
    finally:
        if executor is not None:
            executor.shutdown()
        if ray_pool is not None:
            ray_pool.shutdown()

    if curve_every and (not curve_rows or curve_rows[-1]["iteration"] != iteration):
        record_curve(net, config, iteration, env_steps, curve_rows)
        pd.DataFrame(curve_rows).to_csv(out / "learning_curve.csv", index=False)
    return net


def main(argv: list[str] | None = None) -> None:
    config = common.load_config(argv, __doc__)
    net = train(config)
    out = common.run_dir(config)
    table = common.evaluate(make_eval_agent(net, config), config, out)
    print(table.to_string(index=False))
    if config["eval"].get("policy_only", False):
        policy_out = out / "policy_only"
        policy_out.mkdir(exist_ok=True)
        table = common.evaluate(
            PolicyNetAgent(net, name="az_policy"), config, policy_out
        )
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
