"""Throughput benchmarking script for AlphaZero-lite Ray distributed self-play scaling (M4).

Measures wall-clock time, games/sec, env_steps/sec, search_evals/sec, speedup,
and scaling efficiency across worker counts [1, 2, 4, 8, 16] using Ray actors.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from micromagic.agents.alphazero import PolicyValueNet
from micromagic.training.parallel import get_system_specs
from micromagic.training.parallel_alphazero import benchmark_alphazero_scaling


def run_benchmark(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute AlphaZero scaling benchmark suite per configuration."""
    workers: list[int] = [int(w) for w in config.get("workers", [1, 2, 4, 8, 16])]
    games_per_worker: int = int(config.get("games_per_worker", 2))
    out_dir = Path(config.get("out_dir", "results/benchmarks"))
    out_dir.mkdir(parents=True, exist_ok=True)

    system_specs = get_system_specs()
    timestamp = datetime.now(timezone.utc).isoformat()

    hidden = config.get("net", {}).get("hidden", [128, 128])
    net = PolicyValueNet(hidden=hidden)

    print("=" * 70)
    print("MicroMagic AlphaZero Ray Distributed Self-Play Scaling Benchmark (M4)")
    print("=" * 70)
    print(f"Platform:       {system_specs['platform']}")
    print(f"Python:         {system_specs['python_version']}")
    print(f"CPU Model:      {system_specs['cpu_model']}")
    print(f"Physical Cores: {system_specs['physical_cores']}")
    print(f"Logical Cores:  {system_specs['logical_cores']}")
    print(f"RAM:            {system_specs['ram_gb']} GB")
    print(f"Workers:        {workers}")
    print(f"Games / Worker: {games_per_worker}")
    print(f"Network Trunk:  {hidden}")
    print(f"Search Config:  {config.get('search', {})}")
    print("=" * 70)

    records = benchmark_alphazero_scaling(
        workers=workers,
        games_per_worker=games_per_worker,
        config=config,
        net=net,
    )

    for r in records:
        print(
            f"  Workers: {r['workers']:2d} | "
            f"Games: {r['games']:3d} | "
            f"Time: {r['elapsed_sec']:6.2f}s | "
            f"Throughput: {r['games_per_sec']:6.2f} games/s | "
            f"Env Steps: {r['env_steps_per_sec']:7.1f} steps/s | "
            f"Search Evals: {r['search_evals_per_sec']:8.1f}/s | "
            f"Speedup: {r['speedup']:5.2f}x | "
            f"Efficiency: {r['efficiency'] * 100:5.1f}%"
        )

    df = pd.DataFrame(records)
    csv_path = out_dir / "alphazero_scaling.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    output_payload = {
        "timestamp": timestamp,
        "config": config,
        "system_specs": system_specs,
        "results": records,
    }

    json_path = out_dir / "alphazero_scaling.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Saved JSON: {json_path}")

    print("\n" + "=" * 70)
    print("AlphaZero Scaling Summary Table:")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_alphazero.yaml",
        help="Path to YAML benchmark configuration",
    )
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_benchmark(config)


if __name__ == "__main__":
    main()
