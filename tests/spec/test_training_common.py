"""Training infrastructure: matchup rotation, opponent pool, per-deck/seat evaluation."""

import numpy as np
import pytest


pytest.importorskip("stable_baselines3")

from micromagic.agents import RandomAgent  # noqa: E402
from micromagic.agents.heuristic import HeuristicAgent  # noqa: E402
from micromagic.env import cards as card_db  # noqa: E402
from micromagic.training.common import (  # noqa: E402
    DECK_PAIRS,
    OpponentPool,
    by_deck_and_seat,
    make_vec_env,
    matchup_env_factory,
    play_matchups,
)


def test_eval_cycles_through_all_four_matchups() -> None:
    result = play_matchups(RandomAgent(seed=0, name="a"), RandomAgent(seed=1, name="b"), 8, base_seed=100)
    table = by_deck_and_seat(result, "a", base_seed=100)
    per_matchup = table[table["seat"] != "both"]
    assert sorted(zip(per_matchup["deck"], per_matchup["seat"])) == [
        ("aggro", "first"),
        ("aggro", "second"),
        ("midrange", "first"),
        ("midrange", "second"),
    ]
    assert (per_matchup["games"] == 2).all()
    overall = table.iloc[-1]
    assert overall["deck"] == "all" and overall["games"] == 8


def test_matchup_factory_assigns_decks_by_game_index() -> None:
    factory = matchup_env_factory(base_seed=0)
    for game_id in range(8):
        state = factory(game_id)
        # Library arrays hold the whole shuffled deck (draws only move library_top).
        for seat, deck in enumerate(DECK_PAIRS[(game_id // 2) % 2]):
            assert sorted(state.library[seat]) == sorted(card_db.DECKS[deck])


def test_pool_falls_back_until_first_snapshot() -> None:
    pool = OpponentPool({"random": 0.0, "heuristic": 1.0, "self": 3.0}, seed=0)
    assert all(isinstance(pool.sample(), HeuristicAgent) for _ in range(20))
    snapshot = RandomAgent(seed=9, name="snap")
    pool.add_snapshot(snapshot)
    picks = [pool.sample() for _ in range(400)]
    share = sum(p is snapshot for p in picks) / len(picks)
    assert 0.65 < share < 0.85


def test_pool_keeps_newest_snapshots() -> None:
    pool = OpponentPool({"self": 1.0}, seed=0, max_snapshots=3)
    for i in range(5):
        pool.add_snapshot(RandomAgent(name=f"s{i}"))
    assert [s.name for s in pool.snapshots] == ["s2", "s3", "s4"]


def test_pool_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        OpponentPool({"expert": 1.0})


def test_training_envs_cover_every_deck_and_seat() -> None:
    venv = make_vec_env(4, OpponentPool({"random": 1.0}, seed=0), seed=0)
    seen = set()
    for _ in range(10):
        venv.reset()
        for env in venv.envs:
            base = env.unwrapped
            seen.add((base.decks[base.agent_player], base.agent_player))
    assert seen == {("aggro", 0), ("aggro", 1), ("midrange", 0), ("midrange", 1)}
    assert all(np.asarray(m).any() for m in venv.env_method("action_masks"))
