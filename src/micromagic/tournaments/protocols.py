"""Interface protocols for tournament game states and agents."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GameStateProtocol(Protocol):
    """Protocol defining the structural contract for game state environments."""

    def legal_actions(self) -> np.ndarray:
        """Return a boolean mask representing legal actions."""
        ...

    def step(self, action: int) -> None:
        """Advance game state by executing the selected action."""
        ...

    def current_player(self) -> int:
        """Return the 0-indexed player ID whose turn it is to act."""
        ...

    def observe(self, player: int) -> np.ndarray:
        """Return the observation vector visible to the specified player."""
        ...

    def is_terminal(self) -> bool:
        """Return True if the game has reached a terminal state, False otherwise."""
        ...

    def winner(self) -> int | None:
        """Return winning player seat ID (0 or 1), or None for draw or ongoing game."""
        ...

    def clone(self) -> GameStateProtocol:
        """Return a deep copy of the game state."""
        ...

    def determinize(self, player: int, rng: Any) -> GameStateProtocol:
        """Return a determinized copy from the specified player's perspective."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol defining the structural contract for agents."""

    name: str

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """Select a legal action index given an observation vector and action mask."""
        ...
