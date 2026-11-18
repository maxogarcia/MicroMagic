"""Smoke test for the speed benchmark (it checks the wiring; it doesn't time anything meaningful)."""

from micromagic.env.benchmark import run_benchmark


def test_benchmark_reports_every_loop_and_the_target() -> None:
    result = run_benchmark({
        "seed": 0, "games_per_repeat": 2, "repeats": 2, "pin_to_one_core": False,
        "target_steps_per_sec": 2000, "state_samples": 3, "calls_per_state": 2, "out_dir": "unused",
    })
    for loop in ("engine", "agent_loop"):
        stats = result["loops"][loop]
        assert stats["steps_per_sec_min"] > 0
        assert stats["steps_per_sec_min"] <= stats["steps_per_sec_median"] <= stats["steps_per_sec_max"]
    assert result["clone_us"] > 0 and result["determinize_us"] > 0
    assert result["meets_target"] == (result["headline_steps_per_sec"] >= 2000)
    assert result["machine"]["python"]
