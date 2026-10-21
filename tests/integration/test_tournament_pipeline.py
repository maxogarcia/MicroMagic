"""End-to-end integration tests for tournament match pipeline, logging, and summary generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from micromagic.evaluation import ExperimentLogger
from micromagic.tournaments import TournamentRunner


class MicroToyEnvironment:
    """Multi-turn deterministic test environment simulating simplified MTG combat."""

    def __init__(self, seed: int = 42, target_damage: int = 10) -> None:
        self.seed = seed
        self.target_damage = target_damage
        self.damage = {0: 0, 1: 0}
        self._current_player = 0
        self.step_count = 0
        self._rng = np.random.RandomState(seed)

    def legal_actions(self) -> np.ndarray:
        # Action 0: Light Attack (1 damage)
        # Action 1: Heavy Attack (2 damage)
        # Action 2: Illegal pass
        return np.array([True, True, False], dtype=bool)

    def step(self, action: int) -> None:
        if self.is_terminal():
            msg = "Game already terminal"
            raise RuntimeError(msg)

        dmg = 1 if action == 0 else 2
        opponent = 1 - self._current_player
        self.damage[opponent] += dmg
        self.step_count += 1
        self._current_player = 1 - self._current_player

    def current_player(self) -> int:
        return self._current_player

    def observe(self, player: int) -> np.ndarray:
        return np.array(
            [
                float(self.damage[player]),
                float(self.damage[1 - player]),
                float(self.step_count),
                float(self.seed),
            ],
            dtype=np.float32,
        )

    def is_terminal(self) -> bool:
        return (
            self.damage[0] >= self.target_damage
            or self.damage[1] >= self.target_damage
            or self.step_count >= 100
        )

    def winner(self) -> int | None:
        if not self.is_terminal():
            return None
        # Winner is the one who dealt target damage to the opponent
        if self.damage[1] >= self.target_damage:
            return 0  # player 0 dealt enough damage to player 1
        if self.damage[0] >= self.target_damage:
            return 1  # player 1 dealt enough damage to player 0
        return None

    def clone(self) -> MicroToyEnvironment:
        env = MicroToyEnvironment(seed=self.seed, target_damage=self.target_damage)
        env.damage = dict(self.damage)
        env._current_player = self._current_player
        env.step_count = self.step_count
        return env

    def determinize(self, player: int, rng: Any) -> MicroToyEnvironment:
        return self.clone()


class PolicyAgent:
    """Deterministic agent following a configurable action policy."""

    def __init__(self, name: str, preference: int = 0) -> None:
        self.name = name
        self.preference = preference

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if mask[self.preference]:
            return self.preference
        # Fallback to first available legal action
        legal_indices = np.where(mask)[0]
        return int(legal_indices[0])


def test_10_game_tournament_pipeline_end_to_end(tmp_path: Path) -> None:
    """Verify a complete 10-game tournament pipeline with seat swapping and logging."""
    logger = ExperimentLogger(log_dir=tmp_path / "exp", experiment_name="m0_harness")
    runner = TournamentRunner(
        env_factory=lambda s: MicroToyEnvironment(seed=s, target_damage=6)
    )

    agent_aggro = PolicyAgent(name="AggroBot", preference=1)  # chooses 2 damage
    agent_control = PolicyAgent(name="ControlBot", preference=0)  # chooses 1 damage

    num_games = 10
    base_seed = 1000
    results = runner.run_match(
        agent_aggro,
        agent_control,
        num_games=num_games,
        base_seed=base_seed,
        swap_seats=True,
    )

    summary = results.summary()
    assert summary["total_games"] == 10
    assert summary["draws"] == 0
    # Aggro deals 2 dmg per turn, reaches 6 in 3 hits.
    # Because Aggro deals double damage, AggroBot wins every game regardless of seat.
    assert summary["wins"]["AggroBot"] == 10
    assert summary["win_rates"]["AggroBot"] == 1.0

    # Verify seats alternated evenly
    matches = results.matches
    assert len(matches) == 10
    for i, m in enumerate(matches):
        if i % 2 == 0:
            assert m.player_0_name == "AggroBot"
            assert m.player_1_name == "ControlBot"
        else:
            assert m.player_0_name == "ControlBot"
            assert m.player_1_name == "AggroBot"

    # Persist results and verify CSV
    results_csv = tmp_path / "results" / "tournament_run.csv"
    saved_path = runner.save_results(results, results_csv)
    assert saved_path.exists()

    # Log metrics to ExperimentLogger
    logger.log_scalar(
        "aggro_win_rate", summary["win_rates"]["AggroBot"], step=num_games
    )
    logger.log_scalar("avg_match_steps", summary["avg_steps"], step=num_games)

    scalars_df = logger.read_scalars()
    assert len(scalars_df) == 2
    assert "aggro_win_rate" in scalars_df["tag"].values
    assert "avg_match_steps" in scalars_df["tag"].values
