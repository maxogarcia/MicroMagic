"""Train MaskablePPO against an opponent pool, then evaluate it with the tournament harness.

Usage:
    python -m micromagic.training.train_ppo --config configs/ppo_full.yaml
    python -m micromagic.training.train_ppo --config configs/ppo_stub.yaml --total-timesteps 10000

The learner plays every (deck, seat) matchup. Writes to <out_dir>/<run_name>/:
config.yaml, progress.csv (SB3 training log), model.zip, learning_curve.csv
(if `curve` is set), eval_matches_<opponent>.csv, eval_summary.csv and
eval_by_deck_seat.csv.
"""

from __future__ import annotations

from typing import Any

from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from micromagic.agents.ppo import MaskablePPOAgent
from micromagic.training import common
from micromagic.training.common import OpponentPool


def make_vec_env(n_envs: int, opponent: str | dict[str, float], seed: int) -> DummyVecEnv:
    weights = {opponent: 1.0} if isinstance(opponent, str) else opponent
    return common.make_vec_env(n_envs, OpponentPool(weights, seed=seed), seed)


def train(config: dict[str, Any]) -> MaskablePPO:
    out = common.run_dir(config)
    common.save_config(config, out)

    seed = int(config["seed"])
    pool = OpponentPool(common.opponent_weights(config), seed=seed)
    env = common.make_vec_env(int(config["n_envs"]), pool, seed)
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
    model.learn(
        total_timesteps=int(config["total_timesteps"]),
        callback=common.callbacks(config, pool, lambda m: MaskablePPOAgent(m, name="ppo"), out),
    )
    model.save(out / "model.zip")
    env.close()
    return model


def main(argv: list[str] | None = None) -> None:
    config = common.load_config(argv, __doc__)
    model = train(config)
    table = common.evaluate(MaskablePPOAgent(model, name="ppo"), config, common.run_dir(config))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
