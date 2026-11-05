"""DQN with legal-action masking (check-in 1, §1.6).

SB3's DQN has no masks, so three places are patched:
- exploration: epsilon-greedy and warm-up actions are sampled among legal actions only;
- greedy action: illegal actions get Q = -inf before the argmax;
- TD target: the replay buffer keeps the next state's mask, so the bootstrap is
  max over the legal next actions. A next state with no legal action (game
  over) bootstraps 0, which is also the value of a draw by timeout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.dqn.policies import DQNPolicy

from micromagic.agents.base import Agent


def _masked(q_values: th.Tensor, masks: th.Tensor) -> th.Tensor:
    return q_values.masked_fill(~masks, -th.inf)


class MaskableDQNPolicy(DQNPolicy):
    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Greedy action among legal ones. Q-learning's policy is greedy, so
        `deterministic` is accepted only for interface compatibility."""
        self.set_training_mode(False)
        obs_tensor, vectorized = self.obs_to_tensor(observation)
        with th.no_grad():
            q_values = self.q_net(obs_tensor)
            if action_masks is not None:
                masks = th.as_tensor(action_masks, dtype=th.bool, device=self.device).reshape(q_values.shape)
                q_values = _masked(q_values, masks)
            actions = q_values.argmax(dim=1).cpu().numpy()
        if not vectorized:
            actions = actions.squeeze(axis=0)
        return actions, state


class MaskableReplayBuffer(ReplayBuffer):
    """Also stores the legal-action mask of each next state, read from info["action_mask"]."""

    def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Discrete, *args, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, *args, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError("MaskableReplayBuffer does not support optimize_memory_usage")
        self.next_masks = np.zeros((self.buffer_size, self.n_envs, int(action_space.n)), dtype=bool)
        self.sampled_next_masks: th.Tensor | None = None

    def add(self, obs, next_obs, action, reward, done, infos: list[dict[str, Any]]) -> None:
        # Write before super().add() advances self.pos.
        self.next_masks[self.pos] = np.stack([info["action_mask"] for info in infos])
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> ReplayBufferSamples:
        # Same as ReplayBuffer._get_samples, keeping env_indices to look up the masks.
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env),
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        self.sampled_next_masks = th.as_tensor(self.next_masks[batch_inds, env_indices], device=self.device)
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class MaskableDQN(DQN):
    """Drop-in for SB3 DQN on an env exposing action_masks() (MicroMagicGymEnv does)."""

    def __init__(self, policy: str | type[DQNPolicy] = "MlpPolicy", env=None, **kwargs) -> None:
        if policy == "MlpPolicy":
            policy = MaskableDQNPolicy
        if kwargs.get("n_steps", 1) != 1:
            raise ValueError("MaskableDQN supports only 1-step targets")
        kwargs.setdefault("replay_buffer_class", MaskableReplayBuffer)
        super().__init__(policy, env, **kwargs)

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1) -> tuple[np.ndarray, np.ndarray]:
        masks = get_action_masks(self.env)
        actions, _ = self.policy.predict(self._last_obs, action_masks=masks)
        explore = np.random.rand(n_envs) < self.exploration_rate
        if self.num_timesteps < learning_starts:
            explore[:] = True
        for i in np.flatnonzero(explore):
            actions[i] = np.random.choice(np.flatnonzero(masks[i]))
        return actions, actions

    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        actions, state = self.policy.predict(observation, state, episode_start, deterministic, action_masks)
        if deterministic or np.random.rand() >= self.exploration_rate:
            return actions, state
        masks = np.ones((np.size(actions), self.action_space.n), dtype=bool) if action_masks is None else action_masks
        masks = np.asarray(masks, dtype=bool).reshape(np.size(actions), -1)
        random = np.array([np.random.choice(np.flatnonzero(m)) for m in masks])
        return random.reshape(np.shape(actions)), state

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # DQN.train with the target's max taken over legal next actions only.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            next_masks = self.replay_buffer.sampled_next_masks

            with th.no_grad():
                next_q_values = _masked(self.q_net_target(replay_data.next_observations), next_masks)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = th.where(th.isinf(next_q_values), 0.0, next_q_values).reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


class MaskableDQNAgent(Agent):
    """Greedy masked-Q agent for the tournament harness."""

    def __init__(self, model: MaskableDQN | None = None, name: str = "dqn") -> None:
        super().__init__(name)
        self.model = model

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs) -> MaskableDQNAgent:
        agent = cls(**kwargs)
        agent.load(path)
        return agent

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if self.model is None:
            raise RuntimeError("MaskableDQNAgent has no model; call load() first")
        action, _ = self.model.policy.predict(obs, action_masks=np.asarray(mask, dtype=bool))
        return int(action)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("MaskableDQNAgent has no model to save")
        self.model.save(path)

    def load(self, path: str | Path) -> None:
        self.model = MaskableDQN.load(path, device="cpu")
