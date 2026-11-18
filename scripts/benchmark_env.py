"""Run the environment speed benchmark and save the result to results/benchmarks/.

Usage: python scripts/benchmark_env.py --config configs/benchmark_env.yaml
The headline number is the agent loop (legal_actions + observe + step) on one pinned core.
"""

import argparse
import json
import re
from pathlib import Path

import yaml

from micromagic.env.benchmark import run_benchmark


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "benchmark_env.yaml")
    args = parser.parse_args()

    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Benchmark config not found: {args.config}") from err

    result = run_benchmark(config)

    print(f"Machine: {result['machine']['cpu']}, Python {result['machine']['python']}, "
          f"pinned to CPU {result['machine']['pinned_cpu']}")
    for name, loop in result["loops"].items():
        print(f"  {name:12s} {loop['steps_per_sec_median']:>9,.0f} steps/s "
              f"(min {loop['steps_per_sec_min']:,.0f}, max {loop['steps_per_sec_max']:,.0f})")
    print(f"  clone {result['clone_us']} us, determinize {result['determinize_us']} us")
    verdict = "meets" if result["meets_target"] else "MISSES"
    print(f"Agent loop {result['headline_steps_per_sec']:,.0f} steps/s {verdict} the "
          f"{result['target_steps_per_sec']:,.0f} steps/s target")

    cpu_slug = re.sub(r"[^a-z0-9]+", "-", result["machine"]["cpu"].lower()).strip("-")[:40]
    out_dir = REPO_ROOT / config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"env_speed_{result['timestamp_utc'][:10]}_{cpu_slug}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
