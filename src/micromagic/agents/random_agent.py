"""Uniform-random legal-action baseline."""

from __future__ import annotations

import numpy as np

from micromagic.agents.base import Agent


class RandomAgent(Agent):
    """Picks uniformly among legal actions. The same seed gives the same choices."""

    def __init__(self, seed: int | None = None, name: str = "random") -> None:
        super().__init__(name)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise ValueError("No legal actions in mask")
        return int(self.rng.choice(legal))
