"""Specification tests for pairwise head-to-head outcome matrix and table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from micromagic.evaluation.matrix import compute_head_to_head, compute_pairwise_table
from micromagic.tournaments.runner import MatchResult, TournamentResult


def _create_synthetic_tournament() -> TournamentResult:
    """Generate deterministic synthetic match results for testing.

    Match distribution:
    - AgentA vs AgentB: 6 wins for A, 3 wins for B, 1 draw (10 games)
    - AgentB vs AgentC: 7 wins for B, 2 wins for C, 1 draw (10 games)
    - AgentA vs AgentC: 8 wins for A, 1 win for C, 1 draw (10 games)
    Total: 30 games across 3 agents.
    """
    matches: list[MatchResult] = []
    game_id = 0

    def add_series(
        p0: str,
        p1: str,
        wins_p0: int,
        wins_p1: int,
        draws: int,
    ) -> None:
        nonlocal game_id
        for _ in range(wins_p0):
            matches.append(
                MatchResult(
                    game_id=game_id,
                    seed=42 + game_id,
                    player_0_name=p0,
                    player_1_name=p1,
                    winner_seat=0,
                    winner_name=p0,
                    step_count=20,
                    is_draw=False,
                )
            )
            game_id += 1
        for _ in range(wins_p1):
            matches.append(
                MatchResult(
                    game_id=game_id,
                    seed=42 + game_id,
                    player_0_name=p0,
                    player_1_name=p1,
                    winner_seat=1,
                    winner_name=p1,
                    step_count=25,
                    is_draw=False,
                )
            )
            game_id += 1
        for _ in range(draws):
            matches.append(
                MatchResult(
                    game_id=game_id,
                    seed=42 + game_id,
                    player_0_name=p0,
                    player_1_name=p1,
                    winner_seat=None,
                    winner_name=None,
                    step_count=100,
                    is_draw=True,
                )
            )
            game_id += 1

    add_series("AgentA", "AgentB", wins_p0=6, wins_p1=3, draws=1)
    add_series("AgentB", "AgentC", wins_p0=7, wins_p1=2, draws=1)
    add_series("AgentA", "AgentC", wins_p0=8, wins_p1=1, draws=1)

    return TournamentResult(matches=matches)


def test_head_to_head_matrix_dimensions_and_diagonal() -> None:
    """Test that head-to-head matrix is square with matching index and columns."""
    results = _create_synthetic_tournament()
    matrix = compute_head_to_head(results)

    agents = ["AgentA", "AgentB", "AgentC"]
    assert list(matrix.index) == sorted(agents)
    assert list(matrix.columns) == sorted(agents)
    assert matrix.shape == (3, 3)

    # Diagonal should be NaN (agents do not play themselves)
    for agent in agents:
        assert np.isnan(matrix.loc[agent, agent])


def test_head_to_head_matrix_win_rate_values_and_symmetry() -> None:
    """Test win rates and complementary symmetry (W_AB + W_BA == 1.0 with 0.5 draws)."""
    results = _create_synthetic_tournament()
    matrix = compute_head_to_head(results, draw_weight=0.5)

    # AgentA vs AgentB: 6 wins, 3 losses, 1 draw -> (6 + 0.5)/10 = 0.65
    assert matrix.loc["AgentA", "AgentB"] == pytest.approx(0.65)
    assert matrix.loc["AgentB", "AgentA"] == pytest.approx(0.35)
    assert matrix.loc["AgentA", "AgentB"] + matrix.loc[
        "AgentB", "AgentA"
    ] == pytest.approx(1.0)

    # AgentB vs AgentC: 7 wins, 2 losses, 1 draw -> 0.75 vs 0.25
    assert matrix.loc["AgentB", "AgentC"] == pytest.approx(0.75)
    assert matrix.loc["AgentC", "AgentB"] == pytest.approx(0.25)
    assert matrix.loc["AgentB", "AgentC"] + matrix.loc[
        "AgentC", "AgentB"
    ] == pytest.approx(1.0)

    # AgentA vs AgentC: 8 wins, 1 loss, 1 draw -> 0.85 vs 0.15
    assert matrix.loc["AgentA", "AgentC"] == pytest.approx(0.85)
    assert matrix.loc["AgentC", "AgentA"] == pytest.approx(0.15)
    assert matrix.loc["AgentA", "AgentC"] + matrix.loc[
        "AgentC", "AgentA"
    ] == pytest.approx(1.0)


def test_head_to_head_matrix_records_format() -> None:
    """Test records format returning (wins, losses, draws) tuples and verifying symmetry."""
    results = _create_synthetic_tournament()
    matrix = compute_head_to_head(results, output_format="records")

    rec_ab = matrix.loc["AgentA", "AgentB"]
    rec_ba = matrix.loc["AgentB", "AgentA"]

    assert rec_ab == (6, 3, 1)
    assert rec_ba == (3, 6, 1)
    # Symmetry: W_AB == L_BA, L_AB == W_BA, D_AB == D_BA
    assert rec_ab[0] == rec_ba[1]
    assert rec_ab[1] == rec_ba[0]
    assert rec_ab[2] == rec_ba[2]


def test_pairwise_table_columns_and_counts() -> None:
    """Test pairwise table columns, record counts, and win rates."""
    results = _create_synthetic_tournament()
    table = compute_pairwise_table(results)

    expected_cols = [
        "agent_a",
        "agent_b",
        "wins_a",
        "wins_b",
        "draws",
        "total_games",
        "win_rate_a",
    ]
    assert list(table.columns) == expected_cols
    assert len(table) == 3

    row_ab = table[
        (table["agent_a"] == "AgentA") & (table["agent_b"] == "AgentB")
    ].iloc[0]
    assert row_ab["wins_a"] == 6
    assert row_ab["wins_b"] == 3
    assert row_ab["draws"] == 1
    assert row_ab["total_games"] == 10
    assert row_ab["win_rate_a"] == pytest.approx(0.65)


def test_dataframe_input_interchangeability() -> None:
    """Test that DataFrame input produces identical results to TournamentResult input."""
    results = _create_synthetic_tournament()
    df = results.to_dataframe()

    matrix_from_obj = compute_head_to_head(results)
    matrix_from_df = compute_head_to_head(df)

    pd.testing.assert_frame_equal(matrix_from_obj, matrix_from_df)

    table_from_obj = compute_pairwise_table(results)
    table_from_df = compute_pairwise_table(df)

    pd.testing.assert_frame_equal(table_from_obj, table_from_df)
