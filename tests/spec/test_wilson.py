"""Specification tests for Wilson score intervals and agent win-rate aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from micromagic.evaluation.wilson import compute_agent_win_rates, wilson_score_interval
from micromagic.tournaments.runner import MatchResult, TournamentResult


def test_wilson_zero_total() -> None:
    """Test that zero total games returns (0.0, 0.0)."""
    assert wilson_score_interval(successes=0, total=0) == (0.0, 0.0)


def test_wilson_analytical_values_50_of_100() -> None:
    """Test analytical Wilson score interval for 50 successes in 100 trials at 95% confidence."""
    lower, upper = wilson_score_interval(successes=50, total=100, confidence=0.95)
    # Analytical: z ≈ 1.959964, lower ≈ 0.40383, upper ≈ 0.59617
    assert lower == pytest.approx(0.40383, abs=1e-4)
    assert upper == pytest.approx(0.59617, abs=1e-4)


def test_wilson_extreme_proportions_and_clamping() -> None:
    """Test 0/10 and 10/10 extreme proportion bounds clamped within [0.0, 1.0]."""
    lower_0, upper_0 = wilson_score_interval(successes=0, total=10, confidence=0.95)
    assert lower_0 == 0.0
    assert upper_0 == pytest.approx(0.27753, abs=1e-4)

    lower_10, upper_10 = wilson_score_interval(successes=10, total=10, confidence=0.95)
    assert lower_10 == pytest.approx(1.0 - 0.27753, abs=1e-4)
    assert upper_10 == 1.0


def test_wilson_monotonicity() -> None:
    """Test that increasing successes strictly yields non-decreasing interval bounds."""
    total = 20
    lowers: list[float] = []
    uppers: list[float] = []

    for k in range(total + 1):
        low, high = wilson_score_interval(successes=k, total=total, confidence=0.95)
        assert 0.0 <= low <= high <= 1.0
        lowers.append(low)
        uppers.append(high)

    for i in range(len(lowers) - 1):
        assert lowers[i] <= lowers[i + 1]
        assert uppers[i] <= uppers[i + 1]


def test_wilson_interval_narrowing_with_sample_size() -> None:
    """Test that interval width narrows as sample size increases for fixed 50% win rate."""
    cases = [(5, 10), (50, 100), (500, 1000)]
    widths = [
        wilson_score_interval(k, n, confidence=0.95)[1]
        - wilson_score_interval(k, n, confidence=0.95)[0]
        for k, n in cases
    ]
    assert widths[0] > widths[1] > widths[2]


def test_compute_agent_win_rates_aggregation() -> None:
    """Test full tournament aggregation of win rates and confidence intervals."""
    # Matches:
    # A vs B: A wins 6, B wins 3, 1 draw
    # A vs C: A wins 8, C wins 1, 1 draw
    matches = [
        MatchResult(0, 1, "AgentA", "AgentB", 0, "AgentA", 10, False),
        MatchResult(1, 2, "AgentA", "AgentB", 1, "AgentB", 10, False),
        MatchResult(2, 3, "AgentA", "AgentB", None, None, 10, True),
    ]
    results = TournamentResult(matches=matches)
    summary_df = compute_agent_win_rates(results, confidence=0.95, draw_weight=0.5)

    expected_cols = [
        "agent",
        "wins",
        "losses",
        "draws",
        "total_games",
        "effective_win_rate",
        "ci_lower",
        "ci_upper",
    ]
    assert list(summary_df.columns) == expected_cols

    agent_a = summary_df[summary_df["agent"] == "AgentA"].iloc[0]
    assert agent_a["wins"] == 1
    assert agent_a["losses"] == 1
    assert agent_a["draws"] == 1
    assert agent_a["total_games"] == 3
    assert agent_a["effective_win_rate"] == pytest.approx((1 + 0.5 * 1) / 3)
    assert 0.0 <= agent_a["ci_lower"] <= agent_a["ci_upper"] <= 1.0


def test_compute_agent_win_rates_from_dataframe() -> None:
    """Test compute_agent_win_rates operates equivalently from pandas DataFrame."""
    data = [
        {
            "player_0_name": "A",
            "player_1_name": "B",
            "winner_name": "A",
            "is_draw": False,
        },
        {
            "player_0_name": "A",
            "player_1_name": "B",
            "winner_name": "B",
            "is_draw": False,
        },
        {
            "player_0_name": "A",
            "player_1_name": "B",
            "winner_name": None,
            "is_draw": True,
        },
    ]
    df = pd.DataFrame(data)
    res = compute_agent_win_rates(df)
    assert len(res) == 2
    assert set(res["agent"]) == {"A", "B"}
