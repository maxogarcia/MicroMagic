"""Potential-based reward shaping wrapper for MicroMagic Gymnasium environments (M5).

Computes potential differences between consecutive game states based on life totals
and on-board creature power, preserving optimal policy invariance (Ng, Harada, Russell 1999).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from micromagic.env.spec import CREATURE_SLOTS, EMPTY
from micromagic.env.state import GameState


def compute_state_potential(
    state: GameState,
    agent_player: int,
    life_weight: float = 0.05,
    power_weight: float = 0.02,
) -> float:
    """Compute heuristic state potential Phi(s) for the designated agent seat.

    Phi(s) = life_weight * (my_life - opp_life) + power_weight * (my_power - opp_power)
    """
    if state is None:
        return 0.0

    opp_player = 1 - agent_player
    my_life = float(state.life[agent_player])
    opp_life = float(state.life[opp_player])

    my_power = float(
        sum(
            state.power(agent_player, s)
            for s in range(CREATURE_SLOTS)
            if state.creature[agent_player, s] != EMPTY
        )
    )
    opp_power = float(
        sum(
            state.power(opp_player, s)
            for s in range(CREATURE_SLOTS)
            if state.creature[opp_player, s] != EMPTY
        )
    )

    return life_weight * (my_life - opp_life) + power_weight * (my_power - opp_power)


class ShapedRewardWrapper(gym.Wrapper):
    """Gymnasium wrapper adding potential-based reward shaping to MicroMagicGymEnv."""

    def __init__(
        self,
        env: gym.Env,
        life_weight: float = 0.05,
        power_weight: float = 0.02,
    ) -> None:
        super().__init__(env)
        self.life_weight = life_weight
        self.power_weight = power_weight
        self.prev_potential: float | None = None

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        self.prev_potential = compute_state_potential(
            state=base.state,
            agent_player=base.agent_player,
            life_weight=self.life_weight,
            power_weight=self.power_weight,
        )
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped

        if terminated or truncated:
            # Strictly preserve terminal win/loss outcome (+1.0 / -1.0 / 0.0)
            shaped_reward = float(base_reward)
        else:
            curr_potential = compute_state_potential(
                state=base.state,
                agent_player=base.agent_player,
                life_weight=self.life_weight,
                power_weight=self.power_weight,
            )
            prev = (
                self.prev_potential
                if self.prev_potential is not None
                else curr_potential
            )
            shaped_reward = float(base_reward) + (curr_potential - prev)
            self.prev_potential = curr_potential

        return obs, shaped_reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Forward action masks from the underlying environment."""
        if hasattr(self.env, "action_masks"):
            return self.env.action_masks()
        base = self.env.unwrapped
        return base.state.legal_actions()
