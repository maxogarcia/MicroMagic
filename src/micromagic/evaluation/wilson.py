"""Wilson score interval estimation and aggregate agent win-rate computation."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from scipy.stats import norm

from micromagic.tournaments.runner import TournamentResult


def _extract_dataframe(results: TournamentResult | pd.DataFrame) -> pd.DataFrame:
    """Normalize TournamentResult or DataFrame to a DataFrame."""
    if isinstance(results, pd.DataFrame):
        return results
    if hasattr(results, "to_dataframe"):
        return results.to_dataframe()
    msg = f"Expected TournamentResult or pd.DataFrame, got {type(results).__name__}"
    raise TypeError(msg)


def wilson_score_interval(
    successes: float | int,
    total: int,
    confidence: float = 0.95,
    draw_weight: float = 0.5,
    draws: int = 0,
) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of successful outcomes (or effective wins).
        total: Total number of trials / games.
        confidence: Confidence level in (0, 1), default 0.95.
        draw_weight: Weight applied to draws if passed via the draws parameter.
        draws: Optional count of drawn games to combine into effective successes.

    Returns:
        Tuple of (ci_lower, ci_upper) clamped to [0.0, 1.0].
    """
    if total <= 0:
        return (0.0, 0.0)

    effective_successes = float(successes) + (draws * draw_weight if draws > 0 else 0.0)
    p = max(0.0, min(1.0, effective_successes / total))

    alpha = 1.0 - confidence
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    n = float(total)

    denom = n + z2
    center = (n * p + 0.5 * z2) / denom
    radicand = max(0.0, n * p * (1.0 - p) + 0.25 * z2)
    margin = (z * math.sqrt(radicand)) / denom

    ci_lower = max(0.0, min(1.0, center - margin))
    ci_upper = max(0.0, min(1.0, center + margin))

    # Guard against float numerical noise at the boundary
    if p == 0.0:
        ci_lower = 0.0
    if p == 1.0:
        ci_upper = 1.0

    return (ci_lower, ci_upper)


def compute_agent_win_rates(
    results: TournamentResult | pd.DataFrame,
    confidence: float = 0.95,
    draw_weight: float = 0.5,
) -> pd.DataFrame:
    """Aggregate individual agent records across a tournament and compute Wilson CIs.

    Args:
        results: Match results as a TournamentResult or pandas DataFrame.
        confidence: Confidence level for Wilson intervals (default 0.95).
        draw_weight: Effective win credit assigned to each agent for a draw (default 0.5).

    Returns:
        DataFrame with columns:
        agent, wins, losses, draws, total_games, effective_win_rate, ci_lower, ci_upper.
    """
    columns = [
        "agent",
        "wins",
        "losses",
        "draws",
        "total_games",
        "effective_win_rate",
        "ci_lower",
        "ci_upper",
    ]
    df = _extract_dataframe(results)
    if df.empty:
        return pd.DataFrame(columns=columns)

    agents = sorted(set(df["player_0_name"]).union(set(df["player_1_name"])))
    agent_stats: dict[str, dict[str, int]] = {
        agent: {"wins": 0, "losses": 0, "draws": 0} for agent in agents
    }

    for _, row in df.iterrows():
        p0 = str(row["player_0_name"])
        p1 = str(row["player_1_name"])
        winner = row.get("winner_name")
        is_draw = bool(row.get("is_draw", False)) or pd.isna(winner) or (winner is None)

        if p0 not in agent_stats or p1 not in agent_stats:
            continue

        if is_draw:
            agent_stats[p0]["draws"] += 1
            agent_stats[p1]["draws"] += 1
        elif winner == p0:
            agent_stats[p0]["wins"] += 1
            agent_stats[p1]["losses"] += 1
        elif winner == p1:
            agent_stats[p1]["wins"] += 1
            agent_stats[p0]["losses"] += 1

    records: list[dict[str, Any]] = []
    for agent in agents:
        stats = agent_stats[agent]
        w = stats["wins"]
        l_cnt = stats["losses"]
        d = stats["draws"]
        total = w + l_cnt + d

        effective_successes = w + draw_weight * d
        effective_win_rate = effective_successes / total if total > 0 else 0.0

        ci_lower, ci_upper = wilson_score_interval(
            successes=effective_successes,
            total=total,
            confidence=confidence,
            draw_weight=draw_weight,
        )

        records.append(
            {
                "agent": agent,
                "wins": w,
                "losses": l_cnt,
                "draws": d,
                "total_games": total,
                "effective_win_rate": effective_win_rate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    res_df = pd.DataFrame(records, columns=columns)
    res_df.sort_values(
        by=["effective_win_rate", "wins"],
        ascending=[False, False],
        inplace=True,
    )
    res_df.reset_index(drop=True, inplace=True)
    return res_df
