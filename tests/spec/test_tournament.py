"""Spec tests for tournament protocols, mock fixtures, and TournamentRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from micromagic.tournaments import (
    AgentProtocol,
    GameStateProtocol,
    TournamentRunner,
)


class ToyGameState:
    """Deterministic 5-step toy game satisfying GameStateProtocol."""

    def __init__(self, seed: int = 42, max_steps: int = 5) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self.step_idx = 0
        self._current_player = 0

    def legal_actions(self) -> np.ndarray:
        # Actions 0 and 1 are legal; action 2 is illegal
        return np.array([True, True, False], dtype=bool)

    def step(self, action: int) -> None:
        if self.is_terminal():
            msg = "Cannot step in terminal game state"
            raise RuntimeError(msg)
        self.step_idx += 1
        self._current_player = 1 - self._current_player

    def current_player(self) -> int:
        return self._current_player

    def observe(self, player: int) -> np.ndarray:
        return np.array(
            [float(self.step_idx), float(player), float(self.seed)],
            dtype=np.float32,
        )

    def is_terminal(self) -> bool:
        return self.step_idx >= self.max_steps

    def winner(self) -> int | None:
        if not self.is_terminal():
            return None
        # Deterministic winner based on seed: seat (seed % 2) wins
        return self.seed % 2

    def clone(self) -> ToyGameState:
        cloned = ToyGameState(seed=self.seed, max_steps=self.max_steps)
        cloned.step_idx = self.step_idx
        cloned._current_player = self._current_player
        return cloned

    def determinize(self, player: int, rng: Any) -> ToyGameState:
        return self.clone()


class MockAgent:
    """Mock agent implementing AgentProtocol."""

    def __init__(self, name: str, fixed_action: int = 0) -> None:
        self.name = name
        self.fixed_action = fixed_action

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        return self.fixed_action


def test_protocols_runtime_checkable() -> None:
    """Verify ToyGameState and MockAgent satisfy runtime-checkable protocols."""
    game = ToyGameState(seed=123)
    agent = MockAgent(name="BotA")

    assert isinstance(game, GameStateProtocol)
    assert isinstance(agent, AgentProtocol)


def test_seed_determinism() -> None:
    """Verify two runs with the exact same base_seed produce identical MatchResult sequences."""
    runner = TournamentRunner(env_factory=lambda s: ToyGameState(seed=s))
    agent_a = MockAgent(name="AgentA")
    agent_b = MockAgent(name="AgentB")

    res_1 = runner.run_match(agent_a, agent_b, num_games=6, base_seed=100)
    res_2 = runner.run_match(agent_a, agent_b, num_games=6, base_seed=100)

    assert res_1.matches == res_2.matches
    assert len(res_1.matches) == 6

    # Verify game seeds match base_seed + i
    for i, match in enumerate(res_1.matches):
        assert match.game_id == i
        assert match.seed == 100 + i


def test_seat_alternation() -> None:
    """Verify seat assignment alternates when swap_seats is True."""
    runner = TournamentRunner(env_factory=lambda s: ToyGameState(seed=s))
    agent_a = MockAgent(name="AgentA")
    agent_b = MockAgent(name="AgentB")

    result = runner.run_match(
        agent_a, agent_b, num_games=4, base_seed=42, swap_seats=True
    )
    matches = result.matches

    # Game 0 (even): A is player 0, B is player 1
    assert matches[0].player_0_name == "AgentA"
    assert matches[0].player_1_name == "AgentB"

    # Game 1 (odd): B is player 0, A is player 1
    assert matches[1].player_0_name == "AgentB"
    assert matches[1].player_1_name == "AgentA"

    # Game 2 (even): A is player 0, B is player 1
    assert matches[2].player_0_name == "AgentA"
    assert matches[2].player_1_name == "AgentB"

    # Game 3 (odd): B is player 0, A is player 1
    assert matches[3].player_0_name == "AgentB"
    assert matches[3].player_1_name == "AgentA"


def test_illegal_action_detection() -> None:
    """Verify ValueError is raised when an agent attempts an illegal action."""
    runner = TournamentRunner(env_factory=lambda s: ToyGameState(seed=s))
    # ToyGameState legal actions are [True, True, False] -> action 2 is illegal
    bad_agent = MockAgent(name="BadAgent", fixed_action=2)
    good_agent = MockAgent(name="GoodAgent", fixed_action=0)

    with pytest.raises(ValueError, match="attempted illegal action 2"):
        runner.run_match(bad_agent, good_agent, num_games=1, base_seed=42)


def test_csv_serialization_round_trip(tmp_path: Path) -> None:
    """Verify TournamentResult converts to DataFrame and round-trips via CSV serialization."""
    runner = TournamentRunner(env_factory=lambda s: ToyGameState(seed=s))
    agent_a = MockAgent(name="AgentA")
    agent_b = MockAgent(name="AgentB")

    results = runner.run_match(agent_a, agent_b, num_games=4, base_seed=7)
    csv_file = tmp_path / "tournament_results.csv"

    saved_path = runner.save_results(results, csv_file)
    assert saved_path.exists()

    loaded_df = pd.read_csv(saved_path)
    assert len(loaded_df) == 4
    assert list(loaded_df.columns) == [
        "game_id",
        "seed",
        "player_0_name",
        "player_1_name",
        "winner_seat",
        "winner_name",
        "step_count",
        "is_draw",
    ]
    assert loaded_df["seed"].tolist() == [7, 8, 9, 10]
