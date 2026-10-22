"""Train MaskablePPO against a fixed opponent, then evaluate it with the tournament harness.

Usage:
    python -m micromagic.training.train_ppo --config configs/ppo_stub.yaml
    python -m micromagic.training.train_ppo --config configs/ppo_stub.yaml --total-timesteps 10000

Writes to <out_dir>/<run_name>/: config.yaml, progress.csv (SB3 training log),
model.zip, eval_matches.csv, eval_summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from micromagic.agents.base import Agent
from micromagic.agents.ppo import MaskablePPOAgent
from micromagic.agents.random_agent import RandomAgent
from micromagic.env.gym_env import MicroMagicGymEnv
from micromagic.env.state import GameState
from micromagic.evaluation.wilson import compute_agent_win_rates
from micromagic.tournaments.runner import TournamentRunner


def make_opponent(kind: str, seed: int) -> Agent:
    if kind == "random":
        return RandomAgent(seed=seed)
    raise ValueError(f"Unknown opponent {kind!r}; supported: 'random'")


def make_vec_env(n_envs: int, opponent: str, seed: int) -> DummyVecEnv:
    """Half the envs put the learner in seat 0 and half in seat 1, so it learns both
    going first and going second."""

    def factory(index: int):
        def _init() -> Monitor:
            env = MicroMagicGymEnv(
                opponent_policy=make_opponent(opponent, seed + 1000 + index),
                agent_player=index % 2,
            )
            return Monitor(env)

        return _init

    return DummyVecEnv([factory(i) for i in range(n_envs)])


def train(config: dict[str, Any]) -> MaskablePPO:
    out = Path(config["out_dir"]) / config["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    seed = int(config["seed"])
    env = make_vec_env(int(config["n_envs"]), config["opponent"], seed)
    ppo_kwargs = dict(config.get("ppo", {}))
    net_arch = ppo_kwargs.pop("net_arch", [256, 256])
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": net_arch},
        **ppo_kwargs,
    )
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    model.learn(total_timesteps=int(config["total_timesteps"]))
    model.save(out / "model.zip")
    env.close()
    return model


def evaluate(agent: Agent, config: dict[str, Any]) -> pd.DataFrame:
    """Play held-out games against the eval opponent, alternating seats."""
    eval_cfg = config["eval"]
    out = Path(config["out_dir"]) / config["run_name"]
    seed = int(eval_cfg["seed"])
    opponent = make_opponent(eval_cfg.get("opponent", "random"), seed)
    runner = TournamentRunner(lambda s: GameState(seed=s))
    result = runner.run_match(agent, opponent, int(eval_cfg["games"]), base_seed=seed)
    runner.save_results(result, out / "eval_matches.csv")
    summary = compute_agent_win_rates(result)
    summary.to_csv(out / "eval_summary.csv", index=False)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--total-timesteps", type=int, help="override config total_timesteps")
    parser.add_argument("--seed", type=int, help="override config seed")
    parser.add_argument("--run-name", help="override config run_name")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.run_name is not None:
        config["run_name"] = args.run_name
    config.setdefault("run_name", f"{args.config.stem}_seed{config['seed']}")

    model = train(config)
    summary = evaluate(MaskablePPOAgent(model, name="ppo"), config)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
