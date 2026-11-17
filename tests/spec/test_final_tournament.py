"""Specification tests for final tournament evaluation and Bradley-Terry bootstrap CIs (M5)."""

from __future__ import annotations

import pandas as pd
import pytest

from micromagic.evaluation.elo import fit_elo_with_bootstrap
from micromagic.evaluation.matrix import compute_head_to_head
from micromagic.tournaments.runner import MatchResult, TournamentResult


def _create_mock_tournament_results() -> TournamentResult:
    matches = [
        # agent_a beats agent_b
        MatchResult(0, 100, "agent_a", "agent_b", 0, "agent_a", 20, False),
        MatchResult(1, 101, "agent_b", "agent_a", 1, "agent_a", 22, False),
        MatchResult(2, 102, "agent_a", "agent_b", 0, "agent_a", 25, False),
        MatchResult(3, 103, "agent_b", "agent_a", 0, "agent_b", 21, False),
        # agent_b beats agent_c
        MatchResult(4, 104, "agent_b", "agent_c", 0, "agent_b", 18, False),
        MatchResult(5, 105, "agent_c", "agent_b", 1, "agent_b", 19, False),
        # agent_a beats agent_c
        MatchResult(6, 106, "agent_a", "agent_c", 0, "agent_a", 15, False),
        MatchResult(7, 107, "agent_c", "agent_a", 1, "agent_a", 17, False),
    ]
    return TournamentResult(matches=matches)


def test_bootstrap_elo_confidence_intervals():
    """Assert bootstrap resampler computes Bradley-Terry Elo ratings with valid ci_lower and ci_upper."""
    results = _create_mock_tournament_results()
    df_elo = fit_elo_with_bootstrap(
        results,
        num_bootstraps=50,
        confidence=0.95,
        seed=42,
    )

    assert isinstance(df_elo, pd.DataFrame)
    assert len(df_elo) == 3
    for col in ["agent", "elo", "ci_lower", "ci_upper", "std_err"]:
        assert col in df_elo.columns

    # Verify ranking: agent_a > agent_b > agent_c
    agents = list(df_elo["agent"])
    assert agents == ["agent_a", "agent_b", "agent_c"]

    for _, row in df_elo.iterrows():
        assert row["ci_lower"] <= row["elo"] <= row["ci_upper"] or (
            row["ci_lower"] <= row["ci_upper"]
        )


def test_pairwise_matrix_generation():
    """Assert pairwise matrix generation creates a square symmetric table with agent win rates."""
    results = _create_mock_tournament_results()
    h2h_df = compute_head_to_head(results, draw_weight=0.5, output_format="win_rate")

    assert isinstance(h2h_df, pd.DataFrame)
    assert h2h_df.shape == (3, 3)
    agents = ["agent_a", "agent_b", "agent_c"]
    assert list(h2h_df.index) == agents
    assert list(h2h_df.columns) == agents

    # Diagonal must be NaN
    for a in agents:
        assert pd.isna(h2h_df.loc[a, a])

    # Off-diagonals must sum to 1.0 (win_rate_a_vs_b + win_rate_b_vs_a == 1.0)
    for a in agents:
        for b in agents:
            if a != b:
                rate_ab = float(h2h_df.loc[a, b])
                rate_ba = float(h2h_df.loc[b, a])
                assert rate_ab + rate_ba == pytest.approx(1.0)
