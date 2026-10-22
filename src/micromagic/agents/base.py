"""Agent base class shared by every MicroMagic player (baselines and learned policies).

An agent sees only what the acting seat sees: the observation from
GameState.observe(player) and the legal-action mask. That keeps every agent
usable from the tournament harness, the Gymnasium wrapper (as an opponent
policy) and the AEC env without adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Agent(ABC):
    """Picks one legal action index per call. Satisfies tournaments.AgentProtocol."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or type(self).__name__

    @abstractmethod
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """Return an action index with mask[action] True."""

    def reset(self) -> None:
        """Clear per-game state. Called between games; stateless agents ignore it."""

    def save(self, path: str | Path) -> None:
        """Write a checkpoint. Agents with nothing to learn save nothing."""

    def load(self, path: str | Path) -> None:
        """Restore a checkpoint written by save()."""

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> int:
        # Lets an agent be passed directly as MicroMagicGymEnv(opponent_policy=agent).
        return self.act(obs, mask)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
