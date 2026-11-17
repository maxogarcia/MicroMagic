"""Unified tournament launcher and agent loader registry (M4).

Provides declarative tournament loading from YAML configurations and automated
checkpoint discovery across random, heuristic, tabular, search, and deep RL agents.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import torch as th

from micromagic.agents.base import Agent
from micromagic.agents.heuristic import HeuristicAgent
from micromagic.agents.random_agent import RandomAgent
from micromagic.agents.tabular import MonteCarloAgent, SarsaAgent
from micromagic.env.spec import DECK_NAMES
from micromagic.env.state import GameState
from micromagic.evaluation.elo import fit_elo_with_bootstrap
from micromagic.evaluation.matrix import compute_head_to_head, compute_pairwise_table
from micromagic.evaluation.wilson import compute_agent_win_rates
from micromagic.search.mcts import MCTSAgent
from micromagic.tournaments.runner import (
    MatchResult,
    TournamentResult,
    TournamentRunner,
)


DECK_PAIRS: tuple[tuple[str, str], ...] = (DECK_NAMES, DECK_NAMES[::-1])
REPO_ROOT: Path = Path(__file__).resolve().parents[3]


def _resolve_checkpoint_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    repo_p = REPO_ROOT / p
    if repo_p.exists():
        return repo_p
    return p


def load_agent(cfg: dict[str, Any]) -> Agent:
    """Instantiate an agent from its dictionary configuration.

    Supports:
        - "random": RandomAgent
        - "heuristic": HeuristicAgent
        - "tabular_mc" | "mc": MonteCarloAgent loaded from checkpoint
        - "tabular_sarsa" | "sarsa": SarsaAgent loaded from checkpoint
        - "mcts": MCTSAgent with rollout evaluator
        - "alphazero": AlphaZeroAgent loaded from checkpoint
        - "ppo": MaskablePPOAgent loaded from checkpoint
        - "dqn": MaskableDQNAgent loaded from checkpoint
    """
    kind = str(cfg.get("kind", "")).lower()
    name = cfg.get("name")

    if kind == "random":
        seed_val = (
            int(cfg["seed"]) if "seed" in cfg and cfg["seed"] is not None else None
        )
        return RandomAgent(name=name or "random", seed=seed_val)

    if kind == "heuristic":
        return HeuristicAgent(name=name or "heuristic")

    if kind in ("tabular_mc", "mc"):
        ckpt = _resolve_checkpoint_path(
            cfg.get("checkpoint", "results/tabular/mc_final.npz")
        )
        agent = MonteCarloAgent.load(ckpt)
        if name:
            agent.name = name
        return agent

    if kind in ("tabular_sarsa", "sarsa"):
        ckpt = _resolve_checkpoint_path(
            cfg.get("checkpoint", "results/tabular/sarsa_final.npz")
        )
        agent = SarsaAgent.load(ckpt)
        if name:
            agent.name = name
        return agent

    if kind == "mcts":
        simulations = int(cfg.get("simulations", 32))
        determinizations = int(cfg.get("determinizations", 4))
        c_puct = float(cfg.get("c_puct", 1.5))
        seed_val = (
            int(cfg["seed"]) if "seed" in cfg and cfg["seed"] is not None else None
        )
        return MCTSAgent(
            name=name or "mcts",
            determinizations=determinizations,
            simulations=simulations,
            c_puct=c_puct,
            seed=seed_val,
        )

    if kind == "alphazero":
        from micromagic.agents.alphazero import AlphaZeroAgent, PolicyValueNet

        ckpt_path = cfg.get("checkpoint")
        if not ckpt_path:
            raise ValueError("alphazero agent configuration requires 'checkpoint'")
        ckpt_resolved = _resolve_checkpoint_path(ckpt_path)
        ckpt = th.load(ckpt_resolved, map_location="cpu", weights_only=True)
        net = PolicyValueNet.from_checkpoint(ckpt)
        net.eval()
        simulations = int(cfg.get("simulations", 32))
        determinizations = int(cfg.get("determinizations", 4))
        c_puct = float(cfg.get("c_puct", 1.5))
        seed_val = (
            int(cfg["seed"]) if "seed" in cfg and cfg["seed"] is not None else None
        )
        return AlphaZeroAgent(
            net=net,
            name=name or "alphazero",
            determinizations=determinizations,
            simulations=simulations,
            c_puct=c_puct,
            seed=seed_val,
        )

    if kind == "ppo":
        from micromagic.agents.ppo import MaskablePPOAgent

        ckpt_path = cfg.get("checkpoint")
        if not ckpt_path:
            raise ValueError("ppo agent configuration requires 'checkpoint'")
        ckpt_resolved = _resolve_checkpoint_path(ckpt_path)
        agent = MaskablePPOAgent.load(ckpt_resolved)
        if name:
            agent.name = name
        return agent

    if kind == "dqn":
        from micromagic.agents.dqn import MaskableDQNAgent

        ckpt_path = cfg.get("checkpoint")
        if not ckpt_path:
            raise ValueError("dqn agent configuration requires 'checkpoint'")
        ckpt_resolved = _resolve_checkpoint_path(ckpt_path)
        agent = MaskableDQNAgent.load(ckpt_resolved)
        if name:
            agent.name = name
        return agent

    raise ValueError(f"Unknown agent kind: '{kind}'")


def run_tournament(
    config: dict[str, Any],
    games: int | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute round-robin matches across all enabled agents and compute summary metrics.

    Args:
        config: Tournament configuration dictionary.
        games: Number of games per pair (overrides config['games_per_pair']).
        out_dir: Destination directory for summary CSVs (overrides config['out_dir']).

    Returns:
        Dictionary containing matches, win_rates, head_to_head, pairwise, and tournament_result.
    """
    games_per_pair = (
        games if games is not None else int(config.get("games_per_pair", 20))
    )
    base_seed = int(config.get("seed", 42))
    swap_seats = bool(config.get("swap_seats", True))
    max_steps = int(config.get("max_steps", 1000))
    confidence = float(config.get("confidence", 0.95))
    draw_weight = float(config.get("draw_weight", 0.5))
    include_mirrors = bool(config.get("include_mirrors", False))

    if include_mirrors:
        from micromagic.training.train_tabular import ALL_DECK_PAIRS

        deck_pairs = ALL_DECK_PAIRS
    else:
        deck_pairs = DECK_PAIRS

    agent_cfgs = [c for c in config.get("agents", []) if c.get("enabled", True)]
    if len(agent_cfgs) < 2:
        raise ValueError("Tournament requires at least 2 enabled agents")

    agents: list[Agent] = [load_agent(c) for c in agent_cfgs]

    def env_factory(seed: int) -> GameState:
        pair_idx = (seed // 2) % len(deck_pairs)
        return GameState(seed=seed, decks=deck_pairs[pair_idx])

    runner = TournamentRunner(env_factory=env_factory)
    all_matches: list[MatchResult] = []
    pairs = list(combinations(agents, 2))

    for pair_idx, (agent_a, agent_b) in enumerate(pairs):
        pair_base_seed = base_seed + pair_idx * 10_000
        match_res = runner.run_match(
            agent_0=agent_a,
            agent_1=agent_b,
            num_games=games_per_pair,
            base_seed=pair_base_seed,
            swap_seats=swap_seats,
            max_steps=max_steps,
        )
        all_matches.extend(match_res.matches)

    combined_result = TournamentResult(matches=all_matches)
    matches_df = combined_result.to_dataframe()
    win_rates_df = compute_agent_win_rates(
        combined_result,
        confidence=confidence,
        draw_weight=draw_weight,
    )
    head_to_head_df = compute_head_to_head(
        combined_result,
        draw_weight=draw_weight,
        output_format="win_rate",
    )
    pairwise_df = compute_pairwise_table(
        combined_result,
        draw_weight=draw_weight,
    )

    num_bootstraps = int(config.get("num_bootstraps", 1000))
    elo_df = fit_elo_with_bootstrap(
        combined_result,
        num_bootstraps=num_bootstraps,
        confidence=confidence,
        seed=base_seed,
        draw_weight=draw_weight,
    )

    resolved_out = Path(out_dir or config.get("out_dir", "results/tournament"))
    if resolved_out:
        resolved_out.mkdir(parents=True, exist_ok=True)
        matches_df.to_csv(resolved_out / "matches.csv", index=False)
        matches_df.to_csv(resolved_out / "final_matches.csv", index=False)

        win_rates_df.to_csv(resolved_out / "win_rates.csv", index=False)
        win_rates_df.to_csv(resolved_out / "final_win_rates.csv", index=False)

        head_to_head_df.to_csv(resolved_out / "head_to_head.csv")
        head_to_head_df.to_csv(resolved_out / "final_pairwise_matrix.csv")

        pairwise_df.to_csv(resolved_out / "pairwise.csv", index=False)

        elo_df.to_csv(resolved_out / "elo_ratings.csv", index=False)
        elo_df.to_csv(resolved_out / "final_elo_ratings.csv", index=False)

    return {
        "matches": matches_df,
        "win_rates": win_rates_df,
        "head_to_head": head_to_head_df,
        "pairwise": pairwise_df,
        "elo_ratings": elo_df,
        "tournament_result": combined_result,
    }
