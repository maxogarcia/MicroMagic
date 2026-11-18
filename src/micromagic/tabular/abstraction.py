"""Tabular state featurizer and state index encoder/decoder (M1/M2).

Compresses the continuous/one-hot observation into 7 discrete feature buckets
totaling 19,200 states, matching ADR-008 and docs/Aditya_Implementation_Plan.md §4.
"""

from __future__ import annotations

import numpy as np

from micromagic.env.spec import Phase


# Feature dimensions: (own_life, opp_life, power_diff, my_creatures, opp_creatures, hand, phase)
STATE_DIMS: tuple[int, ...] = (4, 4, 5, 4, 4, 3, 5)
NUM_STATES: int = 19200


def _bucket_life(life: int) -> int:
    """Bucket player life: <=5 (0), 6-10 (1), 11-15 (2), 16+ (3)."""
    if life <= 5:
        return 0
    if life <= 10:
        return 1
    if life <= 15:
        return 2
    return 3


def _bucket_power_diff(diff: int) -> int:
    """Bucket creature power difference (mine - theirs):
    <=-4 (0), -3..-1 (1), 0 (2), 1..3 (3), >=4 (4).
    """
    if diff <= -4:
        return 0
    if diff <= -1:
        return 1
    if diff == 0:
        return 2
    if diff <= 3:
        return 3
    return 4


def _bucket_creature_count(count: int) -> int:
    """Bucket creature count: 0 (0), 1 (1), 2 (2), 3+ (3)."""
    if count <= 0:
        return 0
    if count == 1:
        return 1
    if count == 2:
        return 2
    return 3


def _bucket_hand_count(count: int) -> int:
    """Bucket cards in hand: 0 (0), 1-2 (1), 3+ (2)."""
    if count <= 0:
        return 0
    if count <= 2:
        return 1
    return 2


def _bucket_phase(phase: Phase | int) -> int:
    """Bucket phase/decision context:
    MAIN1 (0), DECLARE_ATTACKERS (1), DECLARE_BLOCKERS (2), MAIN2 (3), CHOOSE_TARGET (4).
    """
    return min(max(int(phase), 0), 4)


def extract_abstract_state(obs: np.ndarray) -> tuple[int, int, int, int, int, int, int]:
    """Extract a 7-tuple of discrete feature indices from an observation vector.

    Features:
        0: Own life bucket (0..3)
        1: Opponent life bucket (0..3)
        2: Board power difference bucket (0..4)
        3: Own creature count bucket (0..3)
        4: Opponent creature count bucket (0..3)
        5: Cards in hand bucket (0..2)
        6: Phase/decision context bucket (0..4)
    """
    from micromagic.agents.heuristic import decode

    view = decode(obs)
    own_power = sum(c.power for c in view.mine)
    opp_power = sum(c.power for c in view.theirs)

    return (
        _bucket_life(view.my_life),
        _bucket_life(view.opp_life),
        _bucket_power_diff(own_power - opp_power),
        _bucket_creature_count(len(view.mine)),
        _bucket_creature_count(len(view.theirs)),
        _bucket_hand_count(len(view.hand)),
        _bucket_phase(view.phase),
    )


def encode_state(features: tuple[int, ...]) -> int:
    """Bijection from 7-feature discrete tuple to single integer state_id in [0, 19200)."""
    idx = 0
    for val, dim in zip(features, STATE_DIMS):
        idx = idx * dim + val
    return idx


def decode_state(state_id: int) -> tuple[int, ...]:
    """Inverse mapping from integer state_id to 7-feature discrete tuple."""
    if not (0 <= state_id < NUM_STATES):
        raise ValueError(f"state_id {state_id} outside valid range [0, {NUM_STATES})")
    res: list[int] = []
    rem = state_id
    for dim in reversed(STATE_DIMS):
        res.append(rem % dim)
        rem //= dim
    return tuple(reversed(res))
