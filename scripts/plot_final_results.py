"""Publication-grade visualization script for MicroMagic final results (M5).

Generates high-resolution 300 DPI figures for:
1. results/figures/final_elo_ratings.png: Bradley-Terry Elo ratings with bootstrap 95% error bars.
2. results/figures/final_h2h_matrix.png: Pairwise head-to-head win-rate heatmap.
3. results/figures/reward_ablation.png: Shaped vs Terminal-Only reward learning curves.
4. results/figures/scaling_comparison.png: Multi-process rollout throughput alongside AlphaZero Ray scaling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Clean styling defaults
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e0e0e0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7


def plot_elo_ratings(tournament_dir: Path, out_dir: Path) -> None:
    """Plot Bradley-Terry Elo ratings with 95% bootstrap confidence intervals."""
    elo_file = tournament_dir / "final_elo_ratings.csv"
    if not elo_file.exists():
        elo_file = tournament_dir / "elo_ratings.csv"
    if not elo_file.exists():
        print(f"Skipping Elo plot: {elo_file} not found")
        return

    df = pd.read_csv(elo_file).sort_values("elo", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    agents = df["agent"].tolist()
    elos = df["elo"].to_numpy()
    ci_lower = df["ci_lower"].to_numpy()
    ci_upper = df["ci_upper"].to_numpy()

    xerr = np.array([elos - ci_lower, ci_upper - elos])

    colors = ["#2b5c8f" if "mcts" in a or "alpha" in a else "#3b82f6" for a in agents]

    bars = ax.barh(
        agents,
        elos,
        xerr=xerr,
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="#1e3a8a",
    )
    ax.axvline(
        1500,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label="Anchor Baseline (1500)",
    )

    for bar, elo in zip(bars, elos):
        ax.text(
            bar.get_width() + 15,
            bar.get_y() + bar.get_height() / 2,
            f"{elo:.1f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#1e293b",
        )

    ax.set_xlabel(
        "Bradley–Terry Elo Rating (MLE, 1,000 Bootstrap 95% CI)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        "MicroMagic Final Agent Standings (Tournament Bradley–Terry Elo)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(
        left=min(1200, float(ci_lower.min()) - 100),
        right=max(1800, float(ci_upper.max()) + 150),
    )
    ax.grid(True, axis="x", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    out_file = out_dir / "final_elo_ratings.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_h2h_matrix(tournament_dir: Path, out_dir: Path) -> None:
    """Plot pairwise head-to-head win-rate heatmap."""
    h2h_file = tournament_dir / "final_pairwise_matrix.csv"
    if not h2h_file.exists():
        h2h_file = tournament_dir / "head_to_head.csv"
    if not h2h_file.exists():
        print(f"Skipping H2H plot: {h2h_file} not found")
        return

    df = pd.read_csv(h2h_file, index_col=0)
    agents = df.index.tolist()
    matrix = df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cmap = plt.cm.RdYlGn
    cmap.set_bad("#e2e8f0")

    masked_matrix = np.ma.masked_invalid(matrix)
    cax = ax.imshow(masked_matrix, cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(agents)))
    ax.set_yticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=35, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(agents, fontsize=10, fontweight="bold")

    for i in range(len(agents)):
        for j in range(len(agents)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "—"
                color = "#64748b"
            else:
                text = f"{val:.1%}"
                color = "black" if 0.2 < val < 0.8 else "white"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row Win Rate vs Column", fontsize=10, fontweight="bold")

    ax.set_title(
        "Pairwise Head-to-Head Win Rate Matrix", fontsize=12, fontweight="bold", pad=12
    )
    plt.tight_layout()

    out_file = out_dir / "final_h2h_matrix.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_reward_ablation(benchmarks_dir: Path, out_dir: Path) -> None:
    """Plot learning curves comparing Shaped vs Terminal-Only reward."""
    ablation_file = benchmarks_dir / "reward_ablation.csv"
    if not ablation_file.exists():
        print(f"Skipping Reward Ablation plot: {ablation_file} not found")
        return

    df = pd.read_csv(ablation_file)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = {"terminal_only": "#dc2626", "shaped": "#16a34a"}
    labels = {
        "terminal_only": "Terminal-Only Reward (Sparse)",
        "shaped": "Potential-Based Shaped Reward (Dense)",
    }

    for cond in ["terminal_only", "shaped"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        grouped = (
            sub.groupby("episodes")
            .agg(
                {
                    "win_rate": "mean",
                    "ci_lower": "mean",
                    "ci_upper": "mean",
                }
            )
            .reset_index()
        )

        episodes = grouped["episodes"]
        win_rate = grouped["win_rate"]
        ci_lower = grouped["ci_lower"]
        ci_upper = grouped["ci_upper"]

        ax.plot(
            episodes,
            win_rate,
            label=labels[cond],
            color=colors[cond],
            marker="o",
            linewidth=2.0,
        )
        ax.fill_between(episodes, ci_lower, ci_upper, color=colors[cond], alpha=0.18)

    ax.axhline(
        0.50,
        color="#6b7280",
        linestyle="--",
        linewidth=1.0,
        label="Random Parity (0.50)",
    )
    ax.axhline(
        0.70,
        color="#2563eb",
        linestyle=":",
        linewidth=1.2,
        label="Milestone Threshold (0.70)",
    )

    ax.set_xlabel("Training Episodes", fontsize=11, fontweight="bold")
    ax.set_ylabel("Win Rate vs Random (95% Wilson CI)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Reward Ablation Study: Potential-Based Shaping vs Terminal Reward",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)
    plt.tight_layout()

    out_file = out_dir / "reward_ablation.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_scaling_comparison(benchmarks_dir: Path, out_dir: Path) -> None:
    """Plot rollout throughput scaling (M3) alongside AlphaZero Ray scaling (M4)."""
    rollouts_file = benchmarks_dir / "benchmark_rollouts.csv"
    az_file = benchmarks_dir / "alphazero_scaling.csv"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: SB3 Rollout Throughput (M3)
    if rollouts_file.exists():
        df_roll = pd.read_csv(rollouts_file)
        for backend, group in df_roll.groupby("backend"):
            color = "#2563eb" if backend == "subproc" else "#7c3aed"
            label = (
                f"SubprocVecEnv ({group['workers'].max()} workers)"
                if backend == "subproc"
                else f"RayVecEnv ({group['workers'].max()} workers)"
            )
            ax1.plot(
                group["workers"],
                group["steps_per_sec"],
                marker="s",
                linewidth=2.0,
                color=color,
                label=f"{label} Steps/s",
            )

        ax1.set_xlabel("Worker Processes", fontsize=11, fontweight="bold")
        ax1.set_ylabel(
            "Throughput (Environment Steps / Second)", fontsize=11, fontweight="bold"
        )
        ax1.set_title(
            "Milestone M3: Vectorized Rollout Worker Throughput",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax1.set_xticks([1, 2, 3, 4])
        ax1.grid(True, alpha=0.6)
        ax1.legend(loc="upper left", framealpha=0.9)

    # Subplot 2: AlphaZero Ray Self-Play Speedup (M4)
    if az_file.exists():
        df_az = pd.read_csv(az_file)
        workers = df_az["workers"]
        speedup = df_az["speedup"]

        ax2.plot(
            workers,
            speedup,
            marker="o",
            linewidth=2.2,
            color="#059669",
            label="Measured Speedup",
        )
        ax2.plot(
            workers,
            workers,
            linestyle="--",
            color="#6b7280",
            linewidth=1.2,
            label="Ideal Linear Speedup",
        )

        for w, s in zip(workers, speedup):
            ax2.annotate(
                f"{s:.2f}x",
                (w, s),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#065f46",
            )

        ax2.set_xlabel("Ray Self-Play Workers", fontsize=11, fontweight="bold")
        ax2.set_ylabel(
            "Speedup Factor ($S_K = T_1 / T_K$)", fontsize=11, fontweight="bold"
        )
        ax2.set_title(
            "Milestone M4: AlphaZero Ray Distributed Scaling (up to 16 Workers)",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax2.set_xticks(workers)
        ax2.set_ylim(bottom=0.5, top=18.0)
        ax2.grid(True, alpha=0.6)
        ax2.legend(loc="upper left", framealpha=0.9)

    plt.suptitle(
        "MicroMagic Compute Scaling: Vectorized Rollouts (M3) & AlphaZero Self-Play (M4)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_file = out_dir / "scaling_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main() -> None:
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    tournament_dir = Path("results/tournament")
    benchmarks_dir = Path("results/benchmarks")

    print("=" * 70)
    print("Generating MicroMagic Final Publication Figures (M5)")
    print("=" * 70)
    plot_elo_ratings(tournament_dir, out_dir)
    plot_h2h_matrix(tournament_dir, out_dir)
    plot_reward_ablation(benchmarks_dir, out_dir)
    plot_scaling_comparison(benchmarks_dir, out_dir)
    print("=" * 70)
    print("All figures successfully rendered in results/figures/ at 300 DPI.")


if __name__ == "__main__":
    main()
