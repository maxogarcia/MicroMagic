"""Generate visual training curves and environment bias plots for Milestone M2."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig_dir = Path("results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)
tab_dir = Path("results/tabular")

# ---------------------------------------------------------
# Figure 1: Learning Curves (MC vs 4-Step SARSA)
# ---------------------------------------------------------
mc_curve_path = tab_dir / "mc_learning_curve.csv"
sarsa_curve_path = tab_dir / "sarsa_learning_curve.csv"

if mc_curve_path.exists() and sarsa_curve_path.exists():
    df_mc = pd.read_csv(mc_curve_path)
    df_sarsa = pd.read_csv(sarsa_curve_path)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # MC Curve + Shading
    ax.plot(
        df_mc["episodes"],
        df_mc["win_rate"],
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="First-Visit MC Control",
    )
    ax.fill_between(
        df_mc["episodes"],
        df_mc["ci_lower"],
        df_mc["ci_upper"],
        color="#1f77b4",
        alpha=0.18,
        label="MC 95% Wilson CI",
    )

    # SARSA Curve + Shading
    ax.plot(
        df_sarsa["episodes"],
        df_sarsa["win_rate"],
        color="#ff7f0e",
        linewidth=2.5,
        marker="s",
        markersize=5,
        label="4-Step SARSA",
    )
    ax.fill_between(
        df_sarsa["episodes"],
        df_sarsa["ci_lower"],
        df_sarsa["ci_upper"],
        color="#ff7f0e",
        alpha=0.18,
        label="SARSA 95% Wilson CI",
    )

    # Reference lines
    ax.axhline(
        0.50,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.5,
        label="Random Baseline (50%)",
    )
    ax.axhline(
        0.70,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label="M2 Milestone Target (70%)",
    )

    ax.set_xlabel("Training Episodes", fontsize=12, fontweight="bold")
    ax.set_ylabel("Win Rate vs RandomAgent", fontsize=12, fontweight="bold")
    ax.set_title(
        "MicroMagic Tabular RL Learning Curves (M2)\n1,000 Episodes Across All 4 Deck Pairings (with Mirrors)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0.25, 0.90)
    ax.set_xlim(0, 1050)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

    plt.tight_layout()
    fig_path = fig_dir / "learning_curves.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved {fig_path}")

# ---------------------------------------------------------
# Figure 2: Environment Bias (Random vs Random & Heuristic vs Heuristic)
# ---------------------------------------------------------
bias_rand_path = tab_dir / "bias_random_v_random.csv"
bias_heur_path = tab_dir / "bias_heuristic_v_heuristic.csv"

if bias_rand_path.exists() and bias_heur_path.exists():
    df_rand = pd.read_csv(bias_rand_path)
    df_heur = pd.read_csv(bias_heur_path)

    quadrants = [
        ("aggro_vs_aggro", "Aggro vs Aggro\n(Mirror)"),
        ("aggro_vs_midrange", "Aggro vs Midrange\n(Cross)"),
        ("midrange_vs_aggro", "Midrange vs Aggro\n(Cross)"),
        ("midrange_vs_midrange", "Midrange vs Midrange\n(Mirror)"),
    ]

    r_rates, r_err_low, r_err_high = [], [], []
    h_rates, h_err_low, h_err_high = [], [], []

    for m_key, _ in quadrants:
        row_r = df_rand[
            (df_rand["matchup"] == m_key) & (df_rand["seat"] == "both")
        ].iloc[0]
        r_rates.append(float(row_r["win_rate"]))
        r_err_low.append(float(row_r["win_rate"]) - float(row_r["ci_lower"]))
        r_err_high.append(float(row_r["ci_upper"]) - float(row_r["win_rate"]))

        row_h = df_heur[
            (df_heur["matchup"] == m_key) & (df_heur["seat"] == "both")
        ].iloc[0]
        h_rates.append(float(row_h["win_rate"]))
        h_err_low.append(float(row_h["win_rate"]) - float(row_h["ci_lower"]))
        h_err_high.append(float(row_h["ci_upper"]) - float(row_h["win_rate"]))

    x = np.arange(len(quadrants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    rects1 = ax.bar(
        x - width / 2,
        r_rates,
        width,
        yerr=[r_err_low, r_err_high],
        capsize=5,
        color="#4a7bb0",
        label="Random vs Random (1,000 games)",
        edgecolor="black",
        linewidth=0.8,
    )

    rects2 = ax.bar(
        x + width / 2,
        h_rates,
        width,
        yerr=[h_err_low, h_err_high],
        capsize=5,
        color="#e66c37",
        label="Heuristic vs Heuristic (1,000 games)",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.axhline(
        0.50, color="#555555", linestyle="--", linewidth=1.5, label="Parity (50%)"
    )
    ax.axhline(
        0.70, color="#2ca02c", linestyle=":", linewidth=1.5, label="70% Threshold"
    )

    ax.set_ylabel("Win Rate (Seat-Balanced)", fontsize=12, fontweight="bold")
    ax.set_title(
        "MicroMagic Environment Bias Diagnostics Across 4 Matchup Quadrants\n1,000 Games Per Baseline Pairing with 95% Wilson CIs",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in quadrants], fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

    # Annotate exact win rates on bars
    for rect, rate in zip(rects1, r_rates):
        height = rect.get_height()
        ax.annotate(
            f"{rate * 100:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1c3d61",
        )

    for rect, rate in zip(rects2, h_rates):
        height = rect.get_height()
        ax.annotate(
            f"{rate * 100:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#87320a",
        )

    plt.tight_layout()
    fig_path = fig_dir / "environment_bias.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved {fig_path}")
