"""Train masked DQN against an opponent pool, then evaluate it with the tournament harness.

Usage:
    python -m micromagic.training.train_dqn --config configs/dqn_full.yaml

Same config keys and outputs as train_ppo, with a `dqn:` block of MaskableDQN
arguments in place of `ppo:`.
"""

from __future__ import annotations

from typing import Any

from stable_baselines3.common.logger import configure

from micromagic.agents.dqn import MaskableDQN, MaskableDQNAgent
from micromagic.training import common
from micromagic.training.common import OpponentPool


def train(config: dict[str, Any]) -> MaskableDQN:
    out = common.run_dir(config)
    common.save_config(config, out)

    seed = int(config["seed"])
    pool = OpponentPool(common.opponent_weights(config), seed=seed)
    env = common.make_vec_env(int(config["n_envs"]), pool, seed)
    dqn_kwargs = dict(config.get("dqn", {}))
    net_arch = dqn_kwargs.pop("net_arch", [256, 256])
    model = MaskableDQN(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": net_arch},
        **dqn_kwargs,
    )
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    model.learn(
        total_timesteps=int(config["total_timesteps"]),
        callback=common.callbacks(config, pool, lambda m: MaskableDQNAgent(m, name="dqn"), out),
    )
    model.save(out / "model.zip")
    env.close()
    return model


def main(argv: list[str] | None = None) -> None:
    config = common.load_config(argv, __doc__)
    model = train(config)
    table = common.evaluate(MaskableDQNAgent(model, name="dqn"), config, common.run_dir(config))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
