"""Pairwise head-to-head outcome matrix and summary table computation."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from micromagic.tournaments.runner import TournamentResult


def _extract_dataframe(results: TournamentResult | pd.DataFrame) -> pd.DataFrame:
    """Normalize TournamentResult or DataFrame to a DataFrame."""
    if isinstance(results, pd.DataFrame):
        return results
    if hasattr(results, "to_dataframe"):
        return results.to_dataframe()
    msg = f"Expected TournamentResult or pd.DataFrame, got {type(results).__name__}"
    raise TypeError(msg)


def compute_head_to_head(
    results: TournamentResult | pd.DataFrame,
    draw_weight: float = 0.5,
    output_format: str = "win_rate",
) -> pd.DataFrame:
    """Compute a square head-to-head outcome matrix between all tournament agents.

    Args:
        results: Match results as a TournamentResult or pandas DataFrame.
        draw_weight: Effective win credit assigned to each agent for a draw.
        output_format: Output cell format: "win_rate" for float in [0.0, 1.0] (or NaN on diagonal),
            or "records" for (wins, losses, draws) integer tuples.

    Returns:
        Square pandas DataFrame indexed and columned by sorted agent names.
    """
    df = _extract_dataframe(results)
    if df.empty:
        return pd.DataFrame()

    agents = sorted(set(df["player_0_name"]).union(set(df["player_1_name"])))
    n = len(agents)
    matrix_data: dict[str, list[Any]] = {agent: [None] * n for agent in agents}

    # Aggregate counts for each directed pair (p_row, p_col)
    # pair_stats[(p_row, p_col)] = {"wins": int, "losses": int, "draws": int}
    pair_stats: dict[tuple[str, str], dict[str, int]] = {
        (a1, a2): {"wins": 0, "losses": 0, "draws": 0}
        for a1 in agents
        for a2 in agents
        if a1 != a2
    }

    for _, row in df.iterrows():
        p0 = str(row["player_0_name"])
        p1 = str(row["player_1_name"])
        winner = row.get("winner_name")
        is_draw = bool(row.get("is_draw", False)) or pd.isna(winner) or (winner is None)

        if p0 == p1:
            continue

        pair_01 = (p0, p1)
        pair_10 = (p1, p0)

        if is_draw:
            pair_stats[pair_01]["draws"] += 1
            pair_stats[pair_10]["draws"] += 1
        elif winner == p0:
            pair_stats[pair_01]["wins"] += 1
            pair_stats[pair_10]["losses"] += 1
        elif winner == p1:
            pair_stats[pair_01]["losses"] += 1
            pair_stats[pair_10]["wins"] += 1

    for row_idx, row_agent in enumerate(agents):
        for col_agent in agents:
            if row_agent == col_agent:
                cell_value: Any = np.nan if output_format == "win_rate" else (0, 0, 0)
            else:
                stats = pair_stats[(row_agent, col_agent)]
                w = stats["wins"]
                l_cnt = stats["losses"]
                d = stats["draws"]
                total = w + l_cnt + d

                if output_format == "records":
                    cell_value = (w, l_cnt, d)
                else:
                    cell_value = (w + draw_weight * d) / total if total > 0 else np.nan

            matrix_data[col_agent][row_idx] = cell_value

    return pd.DataFrame(matrix_data, index=agents)


def compute_pairwise_table(
    results: TournamentResult | pd.DataFrame,
    draw_weight: float = 0.5,
) -> pd.DataFrame:
    """Compute a canonical pairwise summary table for all unique opponent pairings.

    Args:
        results: Match results as a TournamentResult or pandas DataFrame.
        draw_weight: Effective win credit assigned to each agent for a draw.

    Returns:
        DataFrame with columns: agent_a, agent_b, wins_a, wins_b, draws, total_games, win_rate_a.
    """
    df = _extract_dataframe(results)
    columns = [
        "agent_a",
        "agent_b",
        "wins_a",
        "wins_b",
        "draws",
        "total_games",
        "win_rate_a",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    agents = sorted(set(df["player_0_name"]).union(set(df["player_1_name"])))
    pairs = list(combinations(agents, 2))

    pair_counts: dict[tuple[str, str], dict[str, int]] = {
        (a, b): {"wins_a": 0, "wins_b": 0, "draws": 0} for a, b in pairs
    }

    for _, row in df.iterrows():
        p0 = str(row["player_0_name"])
        p1 = str(row["player_1_name"])
        winner = row.get("winner_name")
        is_draw = bool(row.get("is_draw", False)) or pd.isna(winner) or (winner is None)

        if p0 == p1:
            continue

        pair_key = (p0, p1) if p0 < p1 else (p1, p0)
        if pair_key not in pair_counts:
            continue

        a, b = pair_key
        if is_draw:
            pair_counts[pair_key]["draws"] += 1
        elif winner == a:
            pair_counts[pair_key]["wins_a"] += 1
        elif winner == b:
            pair_counts[pair_key]["wins_b"] += 1

    records = []
    for a, b in pairs:
        counts = pair_counts[(a, b)]
        w_a = counts["wins_a"]
        w_b = counts["wins_b"]
        d = counts["draws"]
        total = w_a + w_b + d
        win_rate_a = (w_a + draw_weight * d) / total if total > 0 else 0.0

        records.append(
            {
                "agent_a": a,
                "agent_b": b,
                "wins_a": w_a,
                "wins_b": w_b,
                "draws": d,
                "total_games": total,
                "win_rate_a": win_rate_a,
            }
        )

    return pd.DataFrame(records, columns=columns)
