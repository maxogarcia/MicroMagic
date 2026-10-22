"""Mask fuzzer: random legal play must never crash or break a state invariant.

The default run is sized for CI. For the M1 acceptance run use
    python scripts/fuzz_masks.py --games 10000
"""

import os

import numpy as np

from micromagic.env.spec import CREATURE_SLOTS, EMPTY, HAND_SIZE, OBS_DIM, Phase
from micromagic.env.state import GameState


FUZZ_GAMES = int(os.environ.get("MICROMAGIC_FUZZ_GAMES", "300"))
COMBAT_PHASES = (Phase.DECLARE_ATTACKERS, Phase.DECLARE_BLOCKERS)


def check_invariants(state: GameState) -> None:
    mask = state.legal_actions()
    assert state.is_terminal() or mask.any(), f"Empty mask in a live game: {state}"
    assert ((0 <= state.hand_count) & (state.hand_count <= HAND_SIZE)).all()
    for player in (0, 1):
        assert (state.hand[player, state.hand_count[player] :] == EMPTY).all()
        assert (state.hand[player, : state.hand_count[player]] != EMPTY).all()
    assert ((0 <= state.lands_tapped) & (state.lands_tapped <= state.lands_total)).all()
    assert all(0 <= state.library_top[p] <= len(state.library[p]) for p in (0, 1))
    empty = state.creature == EMPTY
    assert not (empty & (state.tapped | state.sick | state.attacking | (state.pump != 0))).any()
    assert not (empty & (state.blocking != EMPTY)).any()
    if state.phase not in COMBAT_PHASES:
        assert not state.attacking.any() and (state.blocking == EMPTY).all()
    if not state.is_terminal():
        assert (state.phase == Phase.CHOOSE_TARGET) == (state.pending_card != EMPTY)
    assert state.blocking.shape == (2, CREATURE_SLOTS)
    obs = state.observe(state.current_player())
    assert obs.shape == (OBS_DIM,) and 0.0 <= obs.min() and obs.max() <= 1.0


def run_fuzz(games: int, seed: int = 0, check_every_action_every: int = 25) -> dict[str, int]:
    """Play `games` random games, alternating decks and seats. Returns summary counts."""
    rng = np.random.default_rng(seed)
    steps = 0
    results = {"aggro_wins": 0, "midrange_wins": 0, "draws": 0}
    for game in range(games):
        decks = ("aggro", "midrange") if game % 2 == 0 else ("midrange", "aggro")
        state = GameState(seed=seed * 1_000_003 + game, decks=decks, first_player=(game // 2) % 2)
        while not state.is_terminal():
            legal = np.flatnonzero(state.legal_actions())
            # Every action the mask allows must be executable, not just the one we pick.
            if steps % check_every_action_every == 0:
                for action in legal:
                    state.clone().step(int(action))
            state.step(int(rng.choice(legal)))
            steps += 1
            check_invariants(state)
        winner = state.winner()
        if winner is None:
            results["draws"] += 1
        else:
            results[f"{decks[winner]}_wins"] += 1
    results["games"] = games
    results["steps"] = steps
    return results


def test_mask_fuzzer_finds_no_illegal_actions() -> None:
    summary = run_fuzz(FUZZ_GAMES)
    assert summary["games"] == FUZZ_GAMES
