"""Record one MicroMagic game and write it as an HTML replay you can open in a browser.

Usage:
    python scripts/replay_game.py --seed 7
    python scripts/replay_game.py --seed 7 --decks midrange aggro --first-player 1 --out replays/demo.html

Both seats play uniformly random legal moves. To replay trained agents, call
micromagic.env.replay.record_game with policies=(agent_a.act, agent_b.act).
"""

import argparse
from pathlib import Path

from micromagic.env.replay import record_game, write_html
from micromagic.env.spec import DECK_NAMES


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decks", nargs=2, choices=DECK_NAMES, default=list(DECK_NAMES), metavar="DECK")
    parser.add_argument("--first-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--out", type=Path, default=None, help="default: replays/game_seed<SEED>.html")
    args = parser.parse_args()

    out = args.out or REPO_ROOT / "replays" / f"game_seed{args.seed}.html"
    replay = record_game(seed=args.seed, decks=tuple(args.decks), first_player=args.first_player)
    write_html(replay, out)
    print(f"{replay['result']} ({len(replay['frames']) - 1} actions) -> {out}")


if __name__ == "__main__":
    main()
