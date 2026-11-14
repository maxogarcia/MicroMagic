"""Run environment bias diagnostics for Random vs Random and Heuristic vs Heuristic."""

from pathlib import Path
from micromagic.agents import RandomAgent
from micromagic.agents.heuristic import HeuristicAgent
from micromagic.training.train_tabular import (
    by_deck_and_seat_tabular,
    play_tabular_matchups,
)

out_dir = Path("results/tabular")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Random vs Random over 1,000 games
print("Running Random vs Random over 1,000 games across all 4 deck pairs...")
r0 = RandomAgent(seed=42, name="random_0")
r1 = RandomAgent(seed=43, name="random_1")
res_random = play_tabular_matchups(
    r0, r1, games=1000, base_seed=100_000, include_mirrors=True
)
df_random = by_deck_and_seat_tabular(
    res_random, "random_0", base_seed=100_000, include_mirrors=True
)
df_random.to_csv(out_dir / "bias_random_v_random.csv", index=False)
print("Saved results/tabular/bias_random_v_random.csv")

# 2. Heuristic vs Heuristic over 1,000 games
print("Running Heuristic vs Heuristic over 1,000 games across all 4 deck pairs...")
h0 = HeuristicAgent(name="heuristic_0")
h1 = HeuristicAgent(name="heuristic_1")
res_heuristic = play_tabular_matchups(
    h0, h1, games=1000, base_seed=200_000, include_mirrors=True
)
df_heuristic = by_deck_and_seat_tabular(
    res_heuristic, "heuristic_0", base_seed=200_000, include_mirrors=True
)
df_heuristic.to_csv(out_dir / "bias_heuristic_v_heuristic.csv", index=False)
print("Saved results/tabular/bias_heuristic_v_heuristic.csv")

print("\n--- Random vs Random Matchup Summary ---")
print(
    df_random[df_random["seat"] == "both"][
        ["matchup", "deck", "games", "wins", "win_rate", "ci_lower", "ci_upper"]
    ].to_string(index=False)
)

print("\n--- Heuristic vs Heuristic Matchup Summary ---")
print(
    df_heuristic[df_heuristic["seat"] == "both"][
        ["matchup", "deck", "games", "wins", "win_rate", "ci_lower", "ci_upper"]
    ].to_string(index=False)
)

print("\n--- Heuristic vs Heuristic Full Breakdown ---")
print(df_heuristic.to_string(index=False))
