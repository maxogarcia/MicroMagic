"""Reward shaping ablation benchmark script (M5).

Compares agent learning performance under potential-based reward shaping versus
sparse terminal-only reward across matched random seeds. Records learning progression
and Wilson confidence intervals to results/benchmarks/reward_ablation.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from micromagic.training.train_tabular import train_tabular


def run_reward_ablation(
    seeds: list[int] | None = None,
    episodes: int = 500,
    eval_every: int = 50,
    eval_games: int = 100,
    out_dir: str | Path = "results/benchmarks",
) -> pd.DataFrame:
    """Execute matched-seed reward shaping ablation benchmark."""
    if seeds is None:
        seeds = [42, 123]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    print("=" * 70)
    print("MicroMagic Reward Shaping Ablation Study (M5)")
    print("=" * 70)
    print(f"Seeds:          {seeds}")
    print(f"Episodes:       {episodes}")
    print(f"Eval Every:     {eval_every}")
    print(f"Eval Games:     {eval_games}")
    print(f"Output Dir:     {out_path}")
    print("=" * 70)

    conditions = [
        ("terminal_only", False),
        ("shaped", True),
    ]

    for condition_name, use_shaping in conditions:
        for seed in seeds:
            print(
                f"\n>>> Running condition: {condition_name.upper()} | Seed: {seed} <<<"
            )
            tmp_run_dir = out_path / f"tmp_{condition_name}_seed{seed}"
            tmp_run_dir.mkdir(parents=True, exist_ok=True)

            train_tabular(
                algo="sarsa",
                episodes=episodes,
                opponent="random",
                seed=seed,
                include_mirrors=True,
                eval_every=eval_every,
                eval_games=eval_games,
                reward_shaping=use_shaping,
                out_dir=tmp_run_dir,
            )

            # Read the generated learning curve
            curve_file = tmp_run_dir / "learning_curve.csv"
            if curve_file.exists():
                df_curve = pd.read_csv(curve_file)
                for _, r in df_curve.iterrows():
                    all_rows.append(
                        {
                            "condition": condition_name,
                            "seed": seed,
                            "episodes": int(r["episodes"]),
                            "win_rate": float(r["win_rate"]),
                            "ci_lower": float(r["ci_lower"]),
                            "ci_upper": float(r["ci_upper"]),
                            "wins": int(r["wins"]),
                            "draws": int(r["draws"]),
                            "games": int(r["games"]),
                        }
                    )

    df_results = pd.DataFrame(all_rows)
    csv_file = out_path / "reward_ablation.csv"
    df_results.to_csv(csv_file, index=False)
    print(f"\nSaved reward ablation results to: {csv_file}")

    print("\n" + "=" * 70)
    print("Reward Ablation Summary (Final Milestone Win Rates):")
    print("=" * 70)
    summary = (
        df_results[df_results["episodes"] == episodes]
        .groupby("condition")[["win_rate", "ci_lower", "ci_upper"]]
        .mean()
    )
    print(summary.to_string())
    print("=" * 70)

    return df_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=int, default=500, help="Training episodes per run"
    )
    parser.add_argument(
        "--eval-every", type=int, default=50, help="Evaluation interval"
    )
    parser.add_argument(
        "--eval-games", type=int, default=100, help="Games per evaluation"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123], help="Random seeds"
    )
    parser.add_argument(
        "--out-dir", type=str, default="results/benchmarks", help="Output directory"
    )
    args = parser.parse_args(argv)

    run_reward_ablation(
        seeds=args.seeds,
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
