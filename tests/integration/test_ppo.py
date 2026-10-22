"""MaskablePPO end-to-end smoke test on the stub env: train, save/load, evaluate.

Short on purpose; it checks the wiring, not learning quality.
"""

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("sb3_contrib")
yaml = pytest.importorskip("yaml")

from micromagic.agents import RandomAgent  # noqa: E402
from micromagic.agents.ppo import MaskablePPOAgent  # noqa: E402
from micromagic.env.spec import NUM_ACTIONS, OBS_DIM  # noqa: E402
from micromagic.env.state import GameState  # noqa: E402
from micromagic.tournaments import AgentProtocol  # noqa: E402
from micromagic.training import train_ppo  # noqa: E402


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "ppo_stub.yaml"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    config = yaml.safe_load(CONFIG.read_text())
    out_dir = tmp_path_factory.mktemp("runs")
    config.update(
        out_dir=str(out_dir),
        run_name="smoke",
        total_timesteps=1024,
        n_envs=2,
        eval={"opponent": "random", "games": 10, "seed": 100000},
    )
    config["ppo"].update(n_steps=256, batch_size=128, n_epochs=2, net_arch=[32, 32])
    train_ppo.main(["--config", str(_write(out_dir / "config_in.yaml", config))])
    return out_dir / "smoke", config


def _write(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config))
    return path


def test_run_writes_all_outputs(run: tuple[Path, dict]) -> None:
    out, _ = run
    for name in ("config.yaml", "progress.csv", "model.zip", "eval_matches.csv", "eval_summary.csv"):
        assert (out / name).exists(), name


def test_loaded_agent_plays_only_legal_actions(run: tuple[Path, dict]) -> None:
    out, _ = run
    agent = MaskablePPOAgent.from_checkpoint(out / "model.zip", deterministic=False)
    assert isinstance(agent, AgentProtocol)
    opponent = RandomAgent(seed=0)
    for seed in range(5):
        state = GameState(seed=seed)
        while not state.is_terminal():
            player = state.current_player()
            mask = state.legal_actions()
            actor = agent if player == 0 else opponent
            action = actor.act(state.observe(player), mask)
            assert mask[action]
            state.step(action)


def test_save_load_roundtrip_gives_same_actions(run: tuple[Path, dict], tmp_path: Path) -> None:
    out, _ = run
    agent = MaskablePPOAgent.from_checkpoint(out / "model.zip")
    agent.save(tmp_path / "copy.zip")
    reloaded = MaskablePPOAgent.from_checkpoint(tmp_path / "copy.zip")
    state = GameState(seed=3)
    for _ in range(30):
        if state.is_terminal():
            break
        obs, mask = state.observe(state.current_player()), state.legal_actions()
        assert agent.act(obs, mask) == reloaded.act(obs, mask)
        state.step(agent.act(obs, mask))


def test_deterministic_policy_respects_mask() -> None:
    # An untrained policy still has to put zero probability on masked actions.
    from sb3_contrib import MaskablePPO

    env = train_ppo.make_vec_env(1, "random", seed=0)
    agent = MaskablePPOAgent(MaskablePPO("MlpPolicy", env, seed=0, device="cpu"))
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[0, 5]] = True
    for _ in range(20):
        assert agent.act(np.random.default_rng(0).random(OBS_DIM, dtype=np.float32), mask) in (0, 5)
