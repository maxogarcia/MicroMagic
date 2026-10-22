"""Bradley–Terry maximum likelihood Elo rating engine with bootstrap confidence intervals."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from micromagic.tournaments.runner import TournamentResult

# Scale factor: 400 / ln(10) ≈ 173.7177927613007
ELO_SCALE_FACTOR: float = 400.0 / math.log(10.0)


def _extract_dataframe(results: TournamentResult | pd.DataFrame) -> pd.DataFrame:
    """Normalize TournamentResult or DataFrame to a DataFrame."""
    if isinstance(results, pd.DataFrame):
        return results
    if hasattr(results, "to_dataframe"):
        return results.to_dataframe()
    msg = f"Expected TournamentResult or pd.DataFrame, got {type(results).__name__}"
    raise TypeError(msg)


def fit_bradley_terry(
    results: TournamentResult | pd.DataFrame,
    anchor_agent: str | None = None,
    anchor_elo: float = 1500.0,
    draw_weight: float = 0.5,
    l2_reg: float = 1e-4,
) -> dict[str, float]:
    """Fit a Bradley–Terry latent rating model via Maximum Likelihood Estimation (MLE).

    Probability agent i beats agent j: P(i beats j) = 1 / (1 + exp(-(theta_i - theta_j))).
    Elo ratings are rescaled using s = 400 / ln(10) ≈ 173.7178 such that:
    Elo_i = anchor_elo + s * (theta_i - theta_anchor).

    Args:
        results: Match results as a TournamentResult or pandas DataFrame.
        anchor_agent: Agent whose rating is fixed at anchor_elo (defaults to first sorted agent).
        anchor_elo: Fixed rating for the anchor agent (default 1500.0).
        draw_weight: Effective win credit awarded to each agent for a draw (default 0.5).
        l2_reg: L2 regularization strength on latent theta parameters to guarantee strict convexity.

    Returns:
        Dictionary mapping each agent name to its fitted Elo rating.
    """
    df = _extract_dataframe(results)
    if df.empty:
        return {}

    agents = sorted(set(df["player_0_name"]).union(set(df["player_1_name"])))
    m = len(agents)
    if m == 0:
        return {}
    if m == 1:
        return {agents[0]: float(anchor_elo)}

    agent_to_idx = {agent: idx for idx, agent in enumerate(agents)}

    # Build pairwise win weight matrix W[i, j]: effective wins of i over j
    w_matrix = np.zeros((m, m), dtype=np.float64)

    for _, row in df.iterrows():
        p0 = str(row["player_0_name"])
        p1 = str(row["player_1_name"])
        if p0 not in agent_to_idx or p1 not in agent_to_idx or p0 == p1:
            continue

        i = agent_to_idx[p0]
        j = agent_to_idx[p1]
        winner = row.get("winner_name")
        is_draw = bool(row.get("is_draw", False)) or pd.isna(winner) or (winner is None)

        if is_draw:
            w_matrix[i, j] += draw_weight
            w_matrix[j, i] += draw_weight
        elif winner == p0:
            w_matrix[i, j] += 1.0
        elif winner == p1:
            w_matrix[j, i] += 1.0

    # Optimization objective: Negative Log-Likelihood + L2 Regularization
    # NLL(theta) = - sum_{i < j} [ w_ij * ln(sigma(theta_i - theta_j)) + w_ji * ln(sigma(theta_j - theta_i)) ]
    # Using logaddexp: -ln(sigma(delta)) = logaddexp(0, -delta)
    def objective_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        # theta shape: (m,)
        diffs = theta[:, None] - theta[None, :]  # diffs[i, j] = theta[i] - theta[j]

        # Log probabilities
        neg_log_p = np.logaddexp(0.0, -diffs)  # -ln(sigma(diffs[i, j]))

        # We only consider pairs with matches
        mask = w_matrix > 0
        nll = float(np.sum(w_matrix[mask] * neg_log_p[mask]))
        l2_penalty = 0.5 * l2_reg * float(np.sum(theta**2))
        loss = nll + l2_penalty

        # Gradient:
        # grad_i = sum_{j != i} [ (w_ij + w_ji) * sigma(theta_i - theta_j) - w_ij ] + l2_reg * theta_i
        # Note: sigma(diffs[i, j]) = expit(diffs[i, j])
        p_ij = expit(diffs)
        n_ij = w_matrix + w_matrix.T
        grad = np.sum(n_ij * p_ij - w_matrix, axis=1) + l2_reg * theta

        return loss, grad

    theta_init = np.zeros(m, dtype=np.float64)
    opt_res = minimize(
        objective_and_grad,
        theta_init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )

    fitted_theta = opt_res.x

    # Determine anchor
    if anchor_agent is None or anchor_agent not in agent_to_idx:
        anchor_idx = 0
    else:
        anchor_idx = agent_to_idx[anchor_agent]

    theta_anchor = fitted_theta[anchor_idx]

    # Convert to Elo scale
    ratings: dict[str, float] = {}
    for agent, idx in agent_to_idx.items():
        ratings[agent] = float(
            anchor_elo + ELO_SCALE_FACTOR * (fitted_theta[idx] - theta_anchor)
        )

    return ratings


def fit_elo_with_bootstrap(
    results: TournamentResult | pd.DataFrame,
    num_bootstraps: int = 100,
    confidence: float = 0.95,
    seed: int = 42,
    anchor_agent: str | None = None,
    anchor_elo: float = 1500.0,
    draw_weight: float = 0.5,
) -> pd.DataFrame:
    """Compute Bradley–Terry Elo ratings with non-parametric bootstrap confidence intervals.

    Args:
        results: Match results as a TournamentResult or pandas DataFrame.
        num_bootstraps: Number of bootstrap resamples (default 100).
        confidence: Confidence level in (0, 1), default 0.95.
        seed: Random seed for reproducible bootstrap sampling.
        anchor_agent: Agent whose rating is fixed at anchor_elo (default first agent).
        anchor_elo: Fixed rating for the anchor agent (default 1500.0).
        draw_weight: Effective win credit awarded to each agent for a draw (default 0.5).

    Returns:
        DataFrame with columns: agent, elo, ci_lower, ci_upper, std_err, sorted by elo descending.
    """
    columns = ["agent", "elo", "ci_lower", "ci_upper", "std_err"]
    df = _extract_dataframe(results)
    if df.empty:
        return pd.DataFrame(columns=columns)

    # Point estimates
    point_ratings = fit_bradley_terry(
        df,
        anchor_agent=anchor_agent,
        anchor_elo=anchor_elo,
        draw_weight=draw_weight,
    )
    if not point_ratings:
        return pd.DataFrame(columns=columns)

    agents = sorted(point_ratings.keys())
    effective_anchor = (
        anchor_agent if (anchor_agent and anchor_agent in agents) else agents[0]
    )

    n_matches = len(df)
    rng = np.random.default_rng(seed)
    bootstrap_elos: dict[str, list[float]] = {agent: [] for agent in agents}

    for _ in range(num_bootstraps):
        boot_indices = rng.integers(0, n_matches, size=n_matches)
        boot_df = df.iloc[boot_indices]
        boot_ratings = fit_bradley_terry(
            boot_df,
            anchor_agent=effective_anchor,
            anchor_elo=anchor_elo,
            draw_weight=draw_weight,
        )
        for agent in agents:
            # Fall back to point estimate if agent was absent in resampled matches
            val = boot_ratings.get(agent, point_ratings[agent])
            bootstrap_elos[agent].append(val)

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    records: list[dict[str, Any]] = []
    for agent in agents:
        samples = np.array(bootstrap_elos[agent], dtype=np.float64)
        ci_lower = float(np.percentile(samples, lower_pct))
        ci_upper = float(np.percentile(samples, upper_pct))
        std_err = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0

        records.append(
            {
                "agent": agent,
                "elo": point_ratings[agent],
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "std_err": std_err,
            }
        )

    res_df = pd.DataFrame(records, columns=columns)
    res_df.sort_values(by="elo", ascending=False, inplace=True)
    res_df.reset_index(drop=True, inplace=True)
    return res_df
