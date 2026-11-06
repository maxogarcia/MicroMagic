"""Tabular RL training harness for First-Visit MC and 4-Step SARSA (M2).

Rotates the learner through all 4 (deck, seat) cross and mirror matchups
against baselines (RandomAgent, HeuristicAgent) and evaluates performance with Wilson CIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from micromagic.agents.base import Agent
from micromagic.agents.tabular import MonteCarloAgent, SarsaAgent, TabularAgent
from micromagic.env.spec import DECK_NAMES
from micromagic.env.state import GameState
from micromagic.evaluation.wilson import compute_agent_win_rates, wilson_score_interval
from micromagic.tabular.macros import execute_macro
from micromagic.tournaments.runner import TournamentResult, TournamentRunner
from micromagic.training.common import (
    make_opponent,
)

DECK_PAIRS: tuple[tuple[str, str], ...] = (DECK_NAMES, DECK_NAMES[::-1])

ALL_DECK_PAIRS: tuple[tuple[str, str], ...] = (
    ("aggro", "midrange"),
    ("midrange", "aggro"),
    ("aggro", "aggro"),
    ("midrange", "midrange"),
)


def tabular_env_factory(base_seed: int, include_mirrors: bool = True):
    """Env factory for TournamentRunner cycling through deck pairs and seats."""
    deck_pairs = ALL_DECK_PAIRS if include_mirrors else DECK_PAIRS

    def factory(seed: int) -> GameState:
        game_id = seed - base_seed
        pair_idx = (game_id // 2) % len(deck_pairs)
        return GameState(seed=seed, decks=deck_pairs[pair_idx])

    return factory


def play_tabular_matchups(
    agent: Agent,
    opponent: Agent,
    games: int,
    base_seed: int,
    include_mirrors: bool = True,
) -> TournamentResult:
    """Run `games` tournament matches rotating through deck pairs and alternating seats."""
    runner = TournamentRunner(
        tabular_env_factory(base_seed, include_mirrors=include_mirrors)
    )
    return runner.run_match(agent, opponent, games, base_seed=base_seed)


def by_deck_and_seat_tabular(
    result: TournamentResult,
    agent_name: str,
    base_seed: int,
    include_mirrors: bool = True,
) -> pd.DataFrame:
    """Win rate of `agent_name` per deck, seat, and matchup with 95% Wilson CIs."""
    deck_pairs = ALL_DECK_PAIRS if include_mirrors else DECK_PAIRS
    df = result.to_dataframe()
    seat = np.where(df["player_0_name"] == agent_name, 0, 1)
    pair = ((df["seed"] - base_seed) // 2) % len(deck_pairs)
    my_decks = [deck_pairs[p][s] for p, s in zip(pair, seat)]
    opp_decks = [deck_pairs[p][1 - s] for p, s in zip(pair, seat)]
    matchups = [f"{m}_vs_{o}" for m, o in zip(my_decks, opp_decks)]
    df = df.assign(
        seat=seat,
        deck=my_decks,
        opp_deck=opp_decks,
        matchup=matchups,
        win=(df["winner_name"] == agent_name).astype(int),
        draw=df["is_draw"].astype(int),
    )

    def row(
        group: pd.DataFrame, deck: str, seat_label: str, matchup: str = "all"
    ) -> dict[str, Any]:
        n = len(group)
        wins = int(group["win"].sum())
        draws = int(group["draw"].sum())
        lo, hi = wilson_score_interval(wins, n, draws=draws) if n > 0 else (0.0, 0.0)
        return {
            "matchup": matchup,
            "deck": deck,
            "seat": seat_label,
            "games": n,
            "wins": wins,
            "draws": draws,
            "win_rate": (wins + 0.5 * draws) / n if n else 0.0,
            "ci_lower": lo,
            "ci_upper": hi,
        }

    rows: list[dict[str, Any]] = []

    # 1. Matchup x seat breakdown
    for (m, s), g in df.groupby(["matchup", "seat"], sort=True):
        d = g["deck"].iloc[0]
        rows.append(row(g, d, "first" if s == 0 else "second", matchup=m))

    # 2. Matchup combined (both seats)
    for m, g in df.groupby("matchup", sort=True):
        d = g["deck"].iloc[0]
        rows.append(row(g, d, "both", matchup=m))

    # 3. Deck x seat breakdown across all matchups
    for (deck, s), g in df.groupby(["deck", "seat"], sort=True):
        rows.append(row(g, deck, "first" if s == 0 else "second", matchup="all"))

    # 4. Deck combined (both seats) across all matchups
    for deck, g in df.groupby("deck", sort=True):
        rows.append(row(g, deck, "both", matchup="all"))

    # 5. Overall total row
    rows.append(row(df, "all", "both", matchup="all"))
    return pd.DataFrame(rows)


def train_tabular(
    algo: str = "mc",
    episodes: int = 1000,
    opponent: Agent | str = "random",
    seed: int = 42,
    include_mirrors: bool = True,
    epsilon: float = 0.2,
    epsilon_decay: float = 0.9995,
    min_epsilon: float = 0.05,
    gamma: float = 0.99,
    alpha: float = 0.1,
    n_steps: int = 4,
    eval_every: int = 0,
    eval_games: int = 100,
    out_dir: str | Path | None = None,
) -> TabularAgent:
    """Train a tabular agent (MC or 4-step SARSA) over self-play or fixed opponent.

    Args:
        algo: "mc" for First-Visit Monte Carlo or "sarsa" for 4-Step SARSA.
        episodes: Number of training games.
        opponent: Baseline opponent instance or name ("random", "heuristic").
        seed: Random seed for initialization and environment generation.
        include_mirrors: Whether to include mirror deck pairs in rotation.
        epsilon: Initial exploration probability.
        epsilon_decay: Per-episode epsilon multiplier.
        min_epsilon: Minimum exploration floor.
        gamma: Discount factor.
        alpha: Step size (SARSA only).
        n_steps: Return horizon (SARSA only).
        eval_every: Evaluate against opponent every K episodes (0 = disabled).
        eval_games: Number of evaluation games per checkpoint.
        out_dir: Directory to save checkpoint models and metrics.

    Returns:
        Trained TabularAgent instance (MonteCarloAgent or SarsaAgent).
    """
    algo_key = algo.lower().strip()
    agent: TabularAgent
    if algo_key in ("mc", "monte_carlo", "montecarlo"):
        agent = MonteCarloAgent(
            name=f"MC_seed{seed}",
            epsilon=epsilon,
            gamma=gamma,
            seed=seed,
        )
    elif algo_key in ("sarsa", "4step_sarsa", "nstep_sarsa"):
        agent = SarsaAgent(
            name=f"SARSA_seed{seed}",
            n_steps=n_steps,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unsupported tabular algorithm: {algo!r}. Must be 'mc' or 'sarsa'."
        )

    # Opponent setup
    fixed_opp: Agent | None = None
    if isinstance(opponent, Agent):
        fixed_opp = opponent
    elif isinstance(opponent, str):
        fixed_opp = make_opponent(opponent, seed=seed + 500_000)
    else:
        raise TypeError(f"Opponent must be Agent or str, got {type(opponent)}")

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    deck_pairs = ALL_DECK_PAIRS if include_mirrors else DECK_PAIRS
    curve_rows: list[dict[str, Any]] = []

    # Training loop
    for ep in range(episodes):
        learner_seat = ep % 2
        pair_idx = (ep // 2) % len(deck_pairs)
        decks = deck_pairs[pair_idx]
        game_seed = seed + ep

        state = GameState(seed=game_seed, decks=decks)
        agent.reset()
        fixed_opp.reset()

        if isinstance(agent, MonteCarloAgent):
            transitions: list[tuple[int, int, float]] = []
            while not state.is_terminal():
                curr = state.current_player()
                obs = state.observe(curr)
                mask = state.legal_actions()

                if curr == learner_seat:
                    s, macro = agent.select_macro(obs, mask)
                    transitions.append((s, macro, 0.0))
                    action = execute_macro(macro, obs, mask)
                else:
                    action = fixed_opp.act(obs, mask)

                state.step(action)

            winner = state.winner()
            if winner == learner_seat:
                terminal_reward = 1.0
            elif state.is_truncated() or winner is None:
                terminal_reward = 0.0
            else:
                terminal_reward = -1.0

            agent.update_episode(transitions, terminal_reward)

        elif isinstance(agent, SarsaAgent):
            prev_s: int | None = None
            prev_macro: int | None = None

            while not state.is_terminal():
                curr = state.current_player()
                obs = state.observe(curr)
                mask = state.legal_actions()

                if curr == learner_seat:
                    s, macro = agent.select_macro(obs, mask)
                    if prev_s is not None and prev_macro is not None:
                        agent.step_update(
                            s=prev_s,
                            a=prev_macro,
                            r=0.0,
                            next_s=s,
                            next_a=macro,
                            done=False,
                        )
                    prev_s = s
                    prev_macro = macro
                    action = execute_macro(macro, obs, mask)
                else:
                    action = fixed_opp.act(obs, mask)

                state.step(action)

            winner = state.winner()
            if winner == learner_seat:
                terminal_reward = 1.0
            elif state.is_truncated() or winner is None:
                terminal_reward = 0.0
            else:
                terminal_reward = -1.0

            if prev_s is not None and prev_macro is not None:
                agent.step_update(
                    s=prev_s,
                    a=prev_macro,
                    r=terminal_reward,
                    next_s=0,
                    next_a=None,
                    done=True,
                )

        # Epsilon decay
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

        # Periodic evaluation
        if eval_every > 0 and (ep + 1) % eval_every == 0 and out_path is not None:
            eval_opp = (
                make_opponent(opponent, seed=seed + 900_000 + ep)
                if isinstance(opponent, str)
                else fixed_opp
            )
            prev_det = agent.deterministic
            agent.deterministic = True
            try:
                res = play_tabular_matchups(
                    agent,
                    eval_opp,
                    games=eval_games,
                    base_seed=seed + 900_000 + ep,
                    include_mirrors=include_mirrors,
                )
                df = by_deck_and_seat_tabular(
                    res,
                    agent.name,
                    base_seed=seed + 900_000 + ep,
                    include_mirrors=include_mirrors,
                )
                df.to_csv(out_path / f"eval_ep{ep + 1}.csv", index=False)
                overall = df[(df["deck"] == "all") & (df["seat"] == "both")].iloc[0]
                curve_rows.append(
                    {
                        "episodes": ep + 1,
                        "win_rate": float(overall["win_rate"]),
                        "ci_lower": float(overall["ci_lower"]),
                        "ci_upper": float(overall["ci_upper"]),
                        "wins": int(overall["wins"]),
                        "draws": int(overall["draws"]),
                        "games": int(overall["games"]),
                    }
                )
                pd.DataFrame(curve_rows).to_csv(
                    out_path / "learning_curve.csv", index=False
                )
            finally:
                agent.deterministic = prev_det

    if out_path is not None:
        agent.save(out_path / "final_agent.npz")

    return agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train tabular RL agent for MicroMagic"
    )
    parser.add_argument(
        "--algo", choices=["mc", "sarsa"], default="mc", help="Algorithm"
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "heuristic"],
        default="random",
        help="Opponent type",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--include-mirrors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include mirror matchups in training and evaluation",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.2, help="Initial exploration rate"
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=0.9995, help="Epsilon decay rate"
    )
    parser.add_argument(
        "--min-epsilon", type=float, default=0.05, help="Minimum exploration rate"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.1, help="SARSA step size")
    parser.add_argument("--n-steps", type=int, default=4, help="SARSA n-step horizon")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate every K episodes during training",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=500,
        help="Number of evaluation games per opponent",
    )
    parser.add_argument(
        "--eval-opponents",
        nargs="+",
        default=["random"],
        help="Opponent types for post-training evaluation",
    )
    parser.add_argument(
        "--out-dir", type=str, default="runs/tabular", help="Output directory"
    )
    args = parser.parse_args()

    agent = train_tabular(
        algo=args.algo,
        episodes=args.episodes,
        opponent=args.opponent,
        seed=args.seed,
        include_mirrors=args.include_mirrors,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        gamma=args.gamma,
        alpha=args.alpha,
        n_steps=args.n_steps,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        out_dir=args.out_dir,
    )
    print(f"Training completed for {agent.name}.")

    out_path = Path(args.out_dir) if args.out_dir else None
    if out_path is not None and args.eval_games > 0:
        prev_det = agent.deterministic
        agent.deterministic = True
        eval_seed = args.seed + 1_000_000
        summaries: list[pd.DataFrame] = []
        tables: list[pd.DataFrame] = []
        try:
            for opp_kind in args.eval_opponents:
                eval_opp = make_opponent(opp_kind, seed=eval_seed)
                res = play_tabular_matchups(
                    agent,
                    eval_opp,
                    games=args.eval_games,
                    base_seed=eval_seed,
                    include_mirrors=args.include_mirrors,
                )
                res.to_dataframe().to_csv(
                    out_path / f"eval_matches_{opp_kind}.csv", index=False
                )
                summaries.append(compute_agent_win_rates(res).assign(opponent=opp_kind))
                tables.append(
                    by_deck_and_seat_tabular(
                        res,
                        agent.name,
                        base_seed=eval_seed,
                        include_mirrors=args.include_mirrors,
                    ).assign(opponent=opp_kind)
                )

            summary_df = pd.concat(summaries, ignore_index=True)
            summary_df.to_csv(out_path / "eval_summary.csv", index=False)
            table_df = pd.concat(tables, ignore_index=True)
            table_df.to_csv(out_path / "eval_by_deck_seat.csv", index=False)
            print("\n--- Evaluation Summary ---")
            print(summary_df.to_string(index=False))
            print("\n--- By Deck & Seat (with Mirrors) ---")
            print(table_df.to_string(index=False))
        finally:
            agent.deterministic = prev_det


if __name__ == "__main__":
    main()
