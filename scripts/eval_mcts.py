"""Evaluate MCTSAgent against the Random baseline with the tournament runner.

Usage: python scripts/eval_mcts.py --config configs/mcts_vs_random.yaml
Writes the per-game CSV and a JSON summary (win rate with a 95% Wilson interval,
broken down by the deck and seat MCTS played) to the config's out_dir.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from micromagic.agents import RandomAgent
from micromagic.env.benchmark import machine_info
from micromagic.env.spec import DECK_NAMES
from micromagic.env.state import GameState
from micromagic.evaluation.wilson import wilson_score_interval
from micromagic.search import MCTSAgent, RolloutEvaluator
from micromagic.tournaments.runner import TournamentRunner


REPO_ROOT = Path(__file__).resolve().parent.parent


def decks_for(seed: int) -> tuple[str, str]:
    """Decks by seat. Paired with the runner's seat swap, MCTS plays each deck from each seat."""
    return DECK_NAMES if seed % 4 < 2 else DECK_NAMES[::-1]


def rate(wins: int, games: int) -> dict:
    low, high = wilson_score_interval(wins, games) if games else (0.0, 0.0)
    return {"wins": wins, "games": games, "win_rate": round(wins / games, 4) if games else None,
            "wilson_95": [round(low, 4), round(high, 4)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "mcts_vs_random.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    m = config["mcts"]

    mcts = MCTSAgent(
        determinizations=m["determinizations"], simulations=m["simulations"], c_puct=m["c_puct"],
        evaluator=RolloutEvaluator(depth=m["rollout_depth"], seed=config["seed"]), seed=config["seed"], name="mcts",
    )
    runner = TournamentRunner(lambda seed: GameState(seed=seed, decks=decks_for(seed)))
    start = time.perf_counter()
    result = runner.run_match(mcts, RandomAgent(seed=config["seed"], name="random"),
                              num_games=config["games"], base_seed=config["base_seed"])
    seconds = time.perf_counter() - start

    by_deck: dict[str, list[int]] = {d: [0, 0] for d in DECK_NAMES}
    by_seat = {0: [0, 0], 1: [0, 0]}
    for match in result.matches:
        seat = 0 if match.player_0_name == "mcts" else 1
        won = int(match.winner_name == "mcts")
        deck = decks_for(match.seed)[seat]
        by_deck[deck][0] += won
        by_deck[deck][1] += 1
        by_seat[seat][0] += won
        by_seat[seat][1] += 1

    wins = sum(int(match.winner_name == "mcts") for match in result.matches)
    draws = sum(int(match.is_draw) for match in result.matches)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = REPO_ROOT / config["out_dir"]
    csv_path = runner.save_results(result, out_dir / f"mcts_vs_random_{stamp}.csv")
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "machine": machine_info(),
        "overall": {**rate(wins, len(result.matches)), "draws": draws},
        "by_mcts_deck": {d: rate(*v) for d, v in by_deck.items()},
        "by_mcts_seat": {f"seat_{s}": rate(*v) for s, v in by_seat.items()},
        "seconds_total": round(seconds, 1),
        "seconds_per_game": round(seconds / len(result.matches), 2),
        "matches_csv": str(csv_path.relative_to(REPO_ROOT)),
    }
    json_path = out_dir / f"mcts_vs_random_{stamp}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    o = summary["overall"]
    print(f"MCTS vs Random: {o['wins']}/{o['games']} wins ({o['win_rate']:.1%}, 95% Wilson "
          f"[{o['wilson_95'][0]:.1%}, {o['wilson_95'][1]:.1%}]), {draws} draws, {summary['seconds_per_game']} s/game")
    for deck, r in summary["by_mcts_deck"].items():
        print(f"  as {deck:9s} {r['wins']}/{r['games']} ({r['win_rate']:.1%})")
    for seat, r in summary["by_mcts_seat"].items():
        print(f"  {seat}       {r['wins']}/{r['games']} ({r['win_rate']:.1%})")
    print(f"Saved {json_path.relative_to(REPO_ROOT)} and {summary['matches_csv']}")


if __name__ == "__main__":
    main()
