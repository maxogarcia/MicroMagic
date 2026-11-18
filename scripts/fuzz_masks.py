"""M1 acceptance run for the mask fuzzer.

Usage: python scripts/fuzz_masks.py --games 10000 --seed 0
Any illegal action or broken invariant raises with the offending state.
"""

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.spec.test_fuzz import run_fuzz  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    start = time.perf_counter()
    summary = run_fuzz(args.games, seed=args.seed)
    elapsed = time.perf_counter() - start
    print(f"seed={args.seed} games={summary['games']} steps={summary['steps']} "
          f"illegal_actions=0 invariant_failures=0 seconds={elapsed:.1f}")
    print(f"random-vs-random: aggro_wins={summary['aggro_wins']} "
          f"midrange_wins={summary['midrange_wins']} draws={summary['draws']}")


if __name__ == "__main__":
    main()
