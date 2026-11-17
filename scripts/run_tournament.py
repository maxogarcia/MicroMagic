"""One-command unified tournament runner script (M4).

Executes round-robin tournament matches between declared agents across baselines,
tabular Q-learning, MCTS, and deep RL agents. Computes win rates, Wilson score
confidence intervals, and head-to-head outcome matrices.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from micromagic.tournaments.launcher import run_tournament


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tournament.yaml",
        help="Path to YAML tournament configuration",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Override games per pairing",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Destination directory for tournament output CSVs",
    )
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    out_dir = Path(args.out_dir or config.get("out_dir", "results/tournament"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MicroMagic Unified Tournament Runner")
    print("=" * 70)
    print(f"Config:         {args.config}")
    print(
        f"Games / Pair:   {args.games if args.games is not None else config.get('games_per_pair', 20)}"
    )
    print(f"Output Dir:     {out_dir}")
    print("=" * 70)

    results = run_tournament(config, games=args.games, out_dir=out_dir)

    print("\n" + "=" * 70)
    print("Tournament Standings (95% Wilson Score Intervals):")
    print("=" * 70)
    print(results["win_rates"].to_string(index=False))

    print("\n" + "=" * 70)
    print("Pairwise Head-to-Head Win Rates (Row vs Column):")
    print("=" * 70)
    print(results["head_to_head"].to_string())

    print("\n" + "=" * 70)
    print(f"Saved artifacts to {out_dir}/:")
    print("  - matches.csv")
    print("  - win_rates.csv")
    print("  - head_to_head.csv")
    print("  - pairwise.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
