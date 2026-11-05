"""Tabular RL training harness for First-Visit MC and 4-Step SARSA (M2).

Rotates the learner through all 4 (deck, seat) matchups against baselines
(RandomAgent, HeuristicAgent) and evaluates performance with Wilson CIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from micromagic.agents.base import Agent
from micromagic.agents.tabular import MonteCarloAgent, SarsaAgent, TabularAgent
from micromagic.env.spec import DECK_NAMES
from micromagic.env.state import GameState
from micromagic.tabular.macros import execute_macro
from micromagic.training.common import (
    by_deck_and_seat,
    make_opponent,
    play_matchups,
)

DECK_PAIRS: tuple[tuple[str, str], ...] = (DECK_NAMES, DECK_NAMES[::-1])


def train_tabular(
    algo: str = "mc",
    episodes: int = 1000,
    opponent: Agent | str = "random",
    seed: int = 42,
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

    # Training loop
    for ep in range(episodes):
        learner_seat = ep % 2
        pair_idx = (ep // 2) % len(DECK_PAIRS)
        decks = DECK_PAIRS[pair_idx]
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
            res = play_matchups(
                agent, eval_opp, games=eval_games, base_seed=seed + 900_000 + ep
            )
            df = by_deck_and_seat(res, agent.name, base_seed=seed + 900_000 + ep)
            df.to_csv(out_path / f"eval_ep{ep + 1}.csv", index=False)

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
        "--out-dir", type=str, default="runs/tabular", help="Output directory"
    )
    args = parser.parse_args()

    agent = train_tabular(
        algo=args.algo,
        episodes=args.episodes,
        opponent=args.opponent,
        seed=args.seed,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        gamma=args.gamma,
        alpha=args.alpha,
        n_steps=args.n_steps,
        out_dir=args.out_dir,
    )
    print(f"Training completed for {agent.name}.")


if __name__ == "__main__":
    main()
