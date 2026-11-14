"""Throughput benchmarking script for parallel rollout workers (M3).

Measures steps/sec, speedup, and scaling efficiency across worker counts
and vectorized backends (DummyVecEnv, SubprocVecEnv, RayVecEnv).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from micromagic.training.parallel import (
    benchmark_rollouts,
    get_system_specs,
)


def run_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Execute throughput benchmark suite per configuration."""
    workers: list[int] = [int(w) for w in config.get("workers", [1, 2, 3, 4])]
    steps_per_worker: int = int(config.get("steps_per_worker", 5000))
    repeats: int = int(config.get("repeats", 3))
    opponent: str = str(config.get("opponent", "random"))
    seed: int = int(config.get("seed", 42))
    backends: list[str] = list(config.get("backends", ["subproc", "ray"]))
    out_dir = Path(config.get("out_dir", "results/benchmarks"))
    out_dir.mkdir(parents=True, exist_ok=True)

    system_specs = get_system_specs()
    timestamp = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("MicroMagic Parallel Rollouts Throughput Benchmark (M3)")
    print("=" * 70)
    print(f"Platform:       {system_specs['platform']}")
    print(f"Python:         {system_specs['python_version']}")
    print(f"CPU Model:      {system_specs['cpu_model']}")
    print(f"Physical Cores: {system_specs['physical_cores']}")
    print(f"Logical Cores:  {system_specs['logical_cores']}")
    print(f"RAM:            {system_specs['ram_gb']} GB")
    print(f"Workers:        {workers}")
    print(f"Steps / Worker: {steps_per_worker}")
    print(f"Repeats:        {repeats}")
    print(f"Backends:       {backends}")
    print("=" * 70)

    all_records: list[dict[str, Any]] = []

    for backend in backends:
        print(f"\n>>> Benchmarking Backend: {backend.upper()} <<<")
        backend_records = benchmark_rollouts(
            workers=workers,
            steps_per_worker=steps_per_worker,
            repeats=repeats,
            opponent=opponent,
            seed=seed,
            backend=backend,
        )
        for r in backend_records:
            print(
                f"  Workers: {r['workers']:2d} | "
                f"Total Steps: {r['total_steps']:6d} | "
                f"Time: {r['elapsed_sec']:6.2f}s | "
                f"Throughput: {r['steps_per_sec']:8.1f} steps/s | "
                f"Speedup: {r['speedup']:5.2f}x | "
                f"Efficiency: {r['efficiency'] * 100:5.1f}%"
            )
        all_records.extend(backend_records)

    df = pd.DataFrame(all_records)
    csv_path = out_dir / "benchmark_rollouts.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    output_payload = {
        "timestamp": timestamp,
        "config": config,
        "system_specs": system_specs,
        "results": all_records,
    }

    json_path = out_dir / "benchmark_rollouts.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Saved JSON: {json_path}")

    print("\n" + "=" * 70)
    print("Benchmark Summary Table:")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="MicroMagic Rollouts Benchmark")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/benchmark_rollouts.yaml")
    )
    parser.add_argument("--workers", nargs="+", type=int, help="Override worker counts")
    parser.add_argument(
        "--steps-per-worker", type=int, help="Override steps per worker"
    )
    parser.add_argument("--repeats", type=int, help="Override repeats")
    parser.add_argument("--backends", nargs="+", help="Override backends")
    parser.add_argument("--out-dir", type=Path, help="Override output directory")
    args = parser.parse_args()

    config: dict[str, Any] = {}
    if args.config.exists():
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    if args.workers is not None:
        config["workers"] = args.workers
    if args.steps_per_worker is not None:
        config["steps_per_worker"] = args.steps_per_worker
    if args.repeats is not None:
        config["repeats"] = args.repeats
    if args.backends is not None:
        config["backends"] = args.backends
    if args.out_dir is not None:
        config["out_dir"] = str(args.out_dir)

    run_benchmark(config)


if __name__ == "__main__":
    main()
