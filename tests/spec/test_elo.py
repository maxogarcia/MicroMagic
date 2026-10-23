"""Specification tests for Bradley–Terry maximum likelihood Elo ratings and bootstrap CIs."""

from __future__ import annotations

import pandas as pd
import pytest

from micromagic.evaluation.elo import fit_bradley_terry, fit_elo_with_bootstrap
from micromagic.tournaments.runner import MatchResult, TournamentResult


def _build_match_df(
    pairs: list[tuple[str, str, int, int, int]],
) -> pd.DataFrame:
    """Helper to build a DataFrame of matches from (p0, p1, wins_p0, wins_p1, draws)."""
    rows = []
    game_id = 0
    for p0, p1, w0, w1, d in pairs:
        for _ in range(w0):
            rows.append(
                {
                    "game_id": game_id,
                    "player_0_name": p0,
                    "player_1_name": p1,
                    "winner_name": p0,
                    "is_draw": False,
                }
            )
            game_id += 1
        for _ in range(w1):
            rows.append(
                {
                    "game_id": game_id,
                    "player_0_name": p0,
                    "player_1_name": p1,
                    "winner_name": p1,
                    "is_draw": False,
                }
            )
            game_id += 1
        for _ in range(d):
            rows.append(
                {
                    "game_id": game_id,
                    "player_0_name": p0,
                    "player_1_name": p1,
                    "winner_name": None,
                    "is_draw": True,
                }
            )
            game_id += 1
    return pd.DataFrame(rows)


def test_equal_strength_agents_converge_to_anchor() -> None:
    """Test that two equal-strength agents (50-50 record) converge to anchor Elo (1500)."""
    df = _build_match_df([("AgentA", "AgentB", 50, 50, 0)])
    ratings = fit_bradley_terry(df, anchor_agent="AgentA", anchor_elo=1500.0)

    assert ratings["AgentA"] == pytest.approx(1500.0, abs=1e-2)
    assert ratings["AgentB"] == pytest.approx(1500.0, abs=1e-2)


def test_monotonic_transitive_dominance() -> None:
    """Test that transitive dominance A > B > C yields Elo_A > Elo_B > Elo_C."""
    # A beats B 80-20, B beats C 80-20, A beats C 95-5
    df = _build_match_df(
        [
            ("AgentA", "AgentB", 80, 20, 0),
            ("AgentB", "AgentC", 80, 20, 0),
            ("AgentA", "AgentC", 95, 5, 0),
        ]
    )
    ratings = fit_bradley_terry(df, anchor_agent="AgentB", anchor_elo=1500.0)

    assert ratings["AgentA"] > ratings["AgentB"] > ratings["AgentC"]
    assert ratings["AgentB"] == pytest.approx(1500.0, abs=1e-2)
    # Check that scale difference is approximately log(4) * (400 / ln(10)) ≈ 240.8
    assert (ratings["AgentA"] - ratings["AgentB"]) == pytest.approx(240.8, abs=25.0)


def test_draw_handling_equivalence() -> None:
    """Test that draws are weighted as 0.5 wins by default and give symmetric ratings."""
    df_draws = _build_match_df([("AgentA", "AgentB", 0, 0, 100)])
    ratings = fit_bradley_terry(df_draws, anchor_agent="AgentA", anchor_elo=1500.0)

    assert ratings["AgentA"] == pytest.approx(1500.0, abs=1e-2)
    assert ratings["AgentB"] == pytest.approx(1500.0, abs=1e-2)

    # 40 wins, 20 draws, 40 losses should produce equal ratings
    df_sym = _build_match_df([("AgentA", "AgentB", 40, 40, 20)])
    ratings_sym = fit_bradley_terry(df_sym, anchor_agent="AgentA", anchor_elo=1500.0)
    assert ratings_sym["AgentA"] == pytest.approx(1500.0, abs=1e-2)
    assert ratings_sym["AgentB"] == pytest.approx(1500.0, abs=1e-2)


def test_bootstrap_ci_generation_and_columns() -> None:
    """Test non-parametric bootstrap resampling confidence intervals and reproducibility."""
    df = _build_match_df(
        [
            ("AgentA", "AgentB", 60, 30, 10),
            ("AgentB", "AgentC", 50, 40, 10),
        ]
    )
    ci_df_1 = fit_elo_with_bootstrap(
        df,
        num_bootstraps=50,
        confidence=0.95,
        seed=42,
        anchor_agent="AgentB",
        anchor_elo=1500.0,
    )

    expected_cols = ["agent", "elo", "ci_lower", "ci_upper", "std_err"]
    assert list(ci_df_1.columns) == expected_cols
    assert len(ci_df_1) == 3

    # Point estimate falls inside or at the bootstrap bounds
    for _, row in ci_df_1.iterrows():
        assert row["ci_lower"] <= row["elo"] <= row["ci_upper"] or (
            row["ci_lower"] <= row["ci_upper"]
        )
        assert row["std_err"] >= 0.0

    # Test seed determinism
    ci_df_2 = fit_elo_with_bootstrap(
        df,
        num_bootstraps=50,
        confidence=0.95,
        seed=42,
        anchor_agent="AgentB",
        anchor_elo=1500.0,
    )
    pd.testing.assert_frame_equal(ci_df_1, ci_df_2)


def test_tournament_result_input_support() -> None:
    """Test that fit_bradley_terry accepts TournamentResult objects."""
    matches = [
        MatchResult(0, 1, "AgentA", "AgentB", 0, "AgentA", 10, False),
        MatchResult(1, 2, "AgentA", "AgentB", 1, "AgentB", 10, False),
    ]
    results = TournamentResult(matches=matches)
    ratings = fit_bradley_terry(results, anchor_agent="AgentA", anchor_elo=1500.0)
    assert "AgentA" in ratings
    assert "AgentB" in ratings
