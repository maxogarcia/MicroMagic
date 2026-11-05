"""Masked DQN: masking in action selection, in the replay buffer and in the TD target."""

import numpy as np
import pytest


pytest.importorskip("stable_baselines3")
th = pytest.importorskip("torch")

from micromagic.agents.dqn import MaskableDQN, MaskableDQNAgent  # noqa: E402
from micromagic.env.spec import NUM_ACTIONS, OBS_DIM  # noqa: E402
from micromagic.env.state import GameState  # noqa: E402
from micromagic.training.common import OpponentPool, make_vec_env  # noqa: E402


def small_model(**kwargs) -> MaskableDQN:
    env = make_vec_env(2, OpponentPool({"random": 1.0}, seed=0), seed=0)
    defaults = dict(
        learning_starts=200,
        buffer_size=2000,
        batch_size=32,
        target_update_interval=100,
        policy_kwargs={"net_arch": [32, 32]},
        seed=0,
        device="cpu",
    )
    return MaskableDQN("MlpPolicy", env, **(defaults | kwargs))


def test_greedy_action_respects_mask() -> None:
    agent = MaskableDQNAgent(small_model())
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[0, 7]] = True
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert agent.act(rng.random(OBS_DIM, dtype=np.float32), mask) in (0, 7)


def test_exploration_respects_mask() -> None:
    model = small_model()
    model.exploration_rate = 1.0
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[3, 9]] = True
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    picks = {int(model.predict(obs, action_masks=mask)[0]) for _ in range(100)}
    assert picks == {3, 9}


def test_training_stores_next_masks_and_plays_legally() -> None:
    # The env raises on an illegal action, so a finished run means warm-up,
    # epsilon-greedy and greedy steps all stayed legal.
    model = small_model()
    model.learn(total_timesteps=1500)
    buffer = model.replay_buffer
    stored = buffer.next_masks[: buffer.pos]
    # Every stored mask is a real mask: non-empty, except after the last step of a game.
    assert stored.any(axis=-1).mean() > 0.9
    terminal = buffer.dones[: buffer.pos].astype(bool)
    assert not stored[terminal].any()


def test_target_ignores_illegal_next_actions() -> None:
    model = small_model()
    model.learn(total_timesteps=300)
    buffer = model.replay_buffer
    # Make one illegal next action look hugely valuable: the masked target must ignore it.
    with th.no_grad():
        model.q_net_target.q_net[-1].bias[5] = 1e6
    buffer.next_masks[: buffer.pos, :, 5] = False
    data = buffer.sample(64)
    next_q = model.q_net_target(data.next_observations).masked_fill(~buffer.sampled_next_masks, -th.inf)
    best = next_q.max(dim=1).values
    assert th.all(best[th.isfinite(best)] < 1e5)


def test_checkpoint_roundtrip(tmp_path) -> None:
    agent = MaskableDQNAgent(small_model())
    agent.save(tmp_path / "dqn.zip")
    reloaded = MaskableDQNAgent.from_checkpoint(tmp_path / "dqn.zip")
    state = GameState(seed=1)
    for _ in range(30):
        if state.is_terminal():
            break
        obs, mask = state.observe(state.current_player()), state.legal_actions()
        assert agent.act(obs, mask) == reloaded.act(obs, mask)
        state.step(agent.act(obs, mask))
