"""Evaluation metrics, win-rate trackers, reward ablation benchmarks, and statistical analysis."""

from micromagic.evaluation.elo import fit_bradley_terry, fit_elo_with_bootstrap
from micromagic.evaluation.logging import ExperimentLogger
from micromagic.evaluation.matrix import compute_head_to_head, compute_pairwise_table
from micromagic.evaluation.wilson import compute_agent_win_rates, wilson_score_interval

__all__ = [
    "ExperimentLogger",
    "compute_head_to_head",
    "compute_pairwise_table",
    "wilson_score_interval",
    "compute_agent_win_rates",
    "fit_bradley_terry",
    "fit_elo_with_bootstrap",
]
