"""MaskablePPO (sb3-contrib) policy wrapped as an Agent.

Training lives in micromagic.training.train_ppo; this class only loads a
checkpoint and acts, so the tournament harness can play it like any other agent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from micromagic.agents.base import Agent


class MaskablePPOAgent(Agent):
    """deterministic=True plays the argmax legal action (evaluation default);
    False samples from the masked policy distribution."""

    def __init__(
        self,
        model: MaskablePPO | None = None,
        deterministic: bool = True,
        name: str = "ppo",
    ) -> None:
        super().__init__(name)
        self.model = model
        self.deterministic = deterministic

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs) -> MaskablePPOAgent:
        agent = cls(**kwargs)
        agent.load(path)
        return agent

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if self.model is None:
            raise RuntimeError("MaskablePPOAgent has no model; call load() first")
        action, _ = self.model.predict(
            obs, action_masks=np.asarray(mask, dtype=bool), deterministic=self.deterministic
        )
        return int(action)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("MaskablePPOAgent has no model to save")
        self.model.save(path)

    def load(self, path: str | Path) -> None:
        self.model = MaskablePPO.load(path, device="cpu")
