"""AlphaZero-lite: masking in the network, evaluator contract, self-play targets, and the training loop."""

import numpy as np
import pytest


th = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from micromagic.agents import RandomAgent  # noqa: E402
from micromagic.agents.alphazero import (  # noqa: E402
    AlphaZeroAgent,
    NetworkEvaluator,
    PolicyNetAgent,
    PolicyValueNet,
    az_loss,
    load_net,
    masked_log_softmax,
)
from micromagic.env.spec import NUM_ACTIONS, OBS_DIM  # noqa: E402
from micromagic.env.state import GameState  # noqa: E402
from micromagic.tournaments.runner import TournamentRunner  # noqa: E402
from micromagic.training import train_alphazero as taz  # noqa: E402


def tiny_net(seed: int = 0) -> PolicyValueNet:
    th.manual_seed(seed)
    return PolicyValueNet([32])


def tiny_config(tmp_path, **overrides) -> dict:
    config = {
        "seed": 0,
        "total_timesteps": 150,
        "games_per_iteration": 2,
        "workers": 1,
        "out_dir": str(tmp_path),
        "run_name": "az_test",
        "net": {"hidden": [32]},
        "search": {"determinizations": 1, "simulations": 4},
        "selfplay": {"temperature_moves": 4, "dirichlet_alpha": 0.3},
        "buffer_size": 1000,
        "min_buffer": 8,
        "batch_size": 8,
        "train_steps": 3,
        "curve": {"every_iterations": 1, "games": 2, "opponents": ["random"], "policy_only": True},
        "eval": {"opponents": ["random"], "games": 2, "seed": 100000},
    }
    return config | overrides


def test_priors_are_a_distribution_over_legal_actions() -> None:
    state = GameState(seed=3)
    priors, value = NetworkEvaluator(tiny_net())(state, state.current_player())
    assert priors.shape == (NUM_ACTIONS,)
    assert priors.sum() == pytest.approx(1.0)
    assert (priors[~state.legal_actions()] == 0).all()
    assert -1.0 <= value <= 1.0


def test_masked_log_softmax_is_finite_and_ignores_illegal_logits() -> None:
    logits = th.tensor([[1.0, 100.0, 2.0]])
    masks = th.tensor([[True, False, True]])
    logp = masked_log_softmax(logits, masks)
    assert th.isfinite(logp).all()
    assert logp[0, 1] == 0
    assert th.exp(logp[0, [0, 2]]).sum().item() == pytest.approx(1.0)


def test_loss_decreases_when_fitting_one_target() -> None:
    net = tiny_net()
    obs = th.rand(16, OBS_DIM)
    masks = th.zeros(16, NUM_ACTIONS, dtype=th.bool)
    masks[:, :3] = True
    pis = th.zeros(16, NUM_ACTIONS)
    pis[:, 2] = 1.0
    zs = th.ones(16)
    opt = th.optim.Adam(net.parameters(), lr=1e-2)
    first = az_loss(net, obs, masks, pis, zs)[0].item()
    for _ in range(50):
        loss = az_loss(net, obs, masks, pis, zs)[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_agents_play_legal_games_through_the_runner() -> None:
    net = tiny_net()
    runner = TournamentRunner(lambda seed: GameState(seed=seed))
    az = AlphaZeroAgent(net, determinizations=1, simulations=4, seed=0)
    result = runner.run_match(az, RandomAgent(seed=0), num_games=2, base_seed=7)
    assert len(result.matches) == 2
    result = runner.run_match(PolicyNetAgent(net), RandomAgent(seed=0), num_games=2, base_seed=7)
    assert len(result.matches) == 2


def test_search_agent_is_deterministic_given_a_seed() -> None:
    net = tiny_net()
    state = GameState(seed=5)
    while np.flatnonzero(state.legal_actions()).size < 2:
        state.step(int(np.flatnonzero(state.legal_actions())[0]))
    p = state.current_player()
    first = AlphaZeroAgent(net, determinizations=2, simulations=8, seed=1).act_from_state(state, p)
    second = AlphaZeroAgent(net, determinizations=2, simulations=8, seed=1).act_from_state(state, p)
    assert first == second and state.legal_actions()[first]


def test_save_and_load_round_trip(tmp_path) -> None:
    net = tiny_net()
    agent = AlphaZeroAgent(net, determinizations=1, simulations=4)
    agent.save(tmp_path / "net.pt")
    loaded = load_net(tmp_path / "net.pt")
    obs = th.rand(4, OBS_DIM)
    for a, b in zip(net(obs), loaded(obs)):
        assert th.allclose(a, b)
    other = PolicyNetAgent(tiny_net(seed=1))
    other.load(tmp_path / "net.pt")
    assert th.allclose(other.net(obs)[0], net(obs)[0])


def test_selfplay_targets_are_consistent_with_the_outcome() -> None:
    config = {"search": {"determinizations": 1, "simulations": 4}, "selfplay": {"temperature_moves": 4}}
    game = taz.play_selfplay_game(tiny_net(), config, seed=11)
    n = len(game["zs"])
    assert n > 0 and game["obs"].shape == (n, OBS_DIM)
    assert game["masks"].shape == game["pis"].shape == (n, NUM_ACTIONS)
    np.testing.assert_allclose(game["pis"].sum(axis=1), 1.0, rtol=1e-5)
    assert not (game["pis"][~game["masks"]] > 0).any()
    assert (game["masks"].sum(axis=1) >= 2).all()  # forced moves are not recorded
    assert set(np.unique(game["zs"])) <= {-1.0, 0.0, 1.0}
    if game["winner"] is None:
        assert (game["zs"] == 0).all()
    else:
        assert (game["zs"] == 1).any() and (game["zs"] == -1).any()
    assert game["env_steps"] >= game["decisions"] == n
    assert game["search_evals"] > 0


def test_replay_buffer_keeps_the_newest_samples() -> None:
    buffer = taz.ReplayBuffer(capacity=5)

    def fake(n: int, z: float) -> dict:
        return {
            "obs": np.zeros((n, OBS_DIM), np.float32),
            "masks": np.ones((n, NUM_ACTIONS), bool),
            "pis": np.full((n, NUM_ACTIONS), 1 / NUM_ACTIONS, np.float32),
            "zs": np.full(n, z, np.float32),
        }

    buffer.add([fake(3, -1.0)])
    buffer.add([fake(4, 1.0)])
    assert len(buffer) == 5
    assert (buffer.zs == [-1, 1, 1, 1, 1]).all()


def test_training_loop_writes_progress_curve_and_checkpoint(tmp_path) -> None:
    config = tiny_config(tmp_path)
    net = taz.train(config)
    out = tmp_path / "az_test"
    for name in ("config.yaml", "progress.csv", "model.pt", "learning_curve.csv"):
        assert (out / name).exists(), name
    import pandas as pd

    progress = pd.read_csv(out / "progress.csv")
    assert progress["timesteps"].iloc[-1] >= config["total_timesteps"]
    assert progress["timesteps"].is_monotonic_increasing
    curve = pd.read_csv(out / "learning_curve.csv")
    assert set(curve["agent"]) == {"alphazero", "az_policy"}
    loaded = load_net(out / "model.pt")
    obs = th.rand(2, OBS_DIM)
    assert th.allclose(loaded(obs)[0], net(obs)[0])
