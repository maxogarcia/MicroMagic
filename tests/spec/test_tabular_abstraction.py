"""Specification tests for tabular state featurizer and state index encoder/decoder.

Contract:
- 7 discrete feature buckets:
    1. Own life: <=5 (0), 6-10 (1), 11-15 (2), 16+ (3)
    2. Opponent life: <=5 (0), 6-10 (1), 11-15 (2), 16+ (3)
    3. Board power difference (my power - opp power):
       <=-4 (0), -3..-1 (1), 0 (2), 1..3 (3), >=4 (4)
    4. Own creature count: 0 (0), 1 (1), 2 (2), 3+ (3)
    5. Opponent creature count: 0 (0), 1 (1), 2 (2), 3+ (3)
    6. Cards in hand: 0 (0), 1-2 (1), 3+ (2)
    7. Decision context (phase):
       MAIN1 (0), DECLARE_ATTACKERS (1), DECLARE_BLOCKERS (2), MAIN2 (3), CHOOSE_TARGET (4)
- Total discrete states: 4 * 4 * 5 * 4 * 4 * 3 * 5 = 19,200
- Encoding / decoding is bijective across [0, 19200).
"""

from __future__ import annotations


import numpy as np
import pytest

from micromagic.env.spec import (
    CREATURE_SLOT_FEATURES,
    CREATURE_SLOTS,
    GLOBAL_FEATURE_NAMES,
    HAND_SIZE,
    HAND_SLOT_FEATURES,
    OBS_DIM,
    OBS_GLOBAL,
    OBS_HAND,
    OBS_MY_CREATURES,
    OBS_OPP_CREATURES,
    SCALE_LIFE,
    SCALE_STAT,
    Phase,
)
from micromagic.env.state import GameState
from micromagic.tabular.abstraction import (
    NUM_STATES,
    STATE_DIMS,
    decode_state,
    encode_state,
    extract_abstract_state,
)


_G = {name: i for i, name in enumerate(GLOBAL_FEATURE_NAMES)}
_STATUS = 1 + 45  # presence flag + CARD_FEATURES (45)


def _make_synthetic_obs(
    my_life: int = 20,
    opp_life: int = 20,
    my_power: int = 0,
    opp_power: int = 0,
    my_creatures: int = 0,
    opp_creatures: int = 0,
    hand_count: int = 0,
    phase: Phase = Phase.MAIN1,
) -> np.ndarray:
    """Build a synthetic observation vector with exact values for testing bucketing boundaries."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # Globals
    g = obs[OBS_GLOBAL]
    g[_G["my_life"]] = float(my_life) / SCALE_LIFE
    g[_G["opp_life"]] = float(opp_life) / SCALE_LIFE
    g[_G[f"phase_{phase.name.lower()}"]] = 1.0

    # Hand
    hand_block = obs[OBS_HAND].reshape(HAND_SIZE, HAND_SLOT_FEATURES)
    for i in range(min(hand_count, HAND_SIZE)):
        hand_block[i, 0] = 1.0  # presence flag

    # My creatures
    my_creatures_block = obs[OBS_MY_CREATURES].reshape(
        CREATURE_SLOTS, CREATURE_SLOT_FEATURES
    )
    for i in range(min(my_creatures, CREATURE_SLOTS)):
        my_creatures_block[i, 0] = 1.0
        # If this is creature 0, give it all the power, else 0 power
        p = my_power if i == 0 else 0
        my_creatures_block[i, _STATUS + 6] = float(p) / SCALE_STAT
        my_creatures_block[i, _STATUS + 7] = float(max(p, 1)) / SCALE_STAT

    # Opp creatures
    opp_creatures_block = obs[OBS_OPP_CREATURES].reshape(
        CREATURE_SLOTS, CREATURE_SLOT_FEATURES
    )
    for i in range(min(opp_creatures, CREATURE_SLOTS)):
        opp_creatures_block[i, 0] = 1.0
        p = opp_power if i == 0 else 0
        opp_creatures_block[i, _STATUS + 6] = float(p) / SCALE_STAT
        opp_creatures_block[i, _STATUS + 7] = float(max(p, 1)) / SCALE_STAT

    return obs


def test_num_states_constant() -> None:
    assert NUM_STATES == 19200
    assert STATE_DIMS == (4, 4, 5, 4, 4, 3, 5)
    prod = 1
    for d in STATE_DIMS:
        prod *= d
    assert prod == 19200


@pytest.mark.parametrize(
    ("life", "expected_bucket"),
    [
        (0, 0),
        (1, 0),
        (5, 0),
        (6, 1),
        (10, 1),
        (11, 2),
        (15, 2),
        (16, 3),
        (20, 3),
        (25, 3),
    ],
)
def test_own_life_bucketing(life: int, expected_bucket: int) -> None:
    obs = _make_synthetic_obs(my_life=life)
    features = extract_abstract_state(obs)
    assert features[0] == expected_bucket


@pytest.mark.parametrize(
    ("life", "expected_bucket"),
    [
        (0, 0),
        (1, 0),
        (5, 0),
        (6, 1),
        (10, 1),
        (11, 2),
        (15, 2),
        (16, 3),
        (20, 3),
        (30, 3),
    ],
)
def test_opp_life_bucketing(life: int, expected_bucket: int) -> None:
    obs = _make_synthetic_obs(opp_life=life)
    features = extract_abstract_state(obs)
    assert features[1] == expected_bucket


@pytest.mark.parametrize(
    ("my_power", "opp_power", "expected_bucket"),
    [
        (0, 8, 0),  # diff = -8 <= -4
        (0, 4, 0),  # diff = -4 <= -4
        (0, 3, 1),  # diff = -3 (-3..-1)
        (1, 2, 1),  # diff = -1 (-3..-1)
        (3, 3, 2),  # diff = 0
        (0, 0, 2),  # diff = 0
        (2, 1, 3),  # diff = 1 (1..3)
        (5, 2, 3),  # diff = 3 (1..3)
        (4, 0, 4),  # diff = 4 >= 4
        (9, 1, 4),  # diff = 8 >= 4
    ],
)
def test_board_power_diff_bucketing(
    my_power: int, opp_power: int, expected_bucket: int
) -> None:
    obs = _make_synthetic_obs(
        my_power=my_power,
        opp_power=opp_power,
        my_creatures=1 if my_power > 0 else 0,
        opp_creatures=1 if opp_power > 0 else 0,
    )
    features = extract_abstract_state(obs)
    assert features[2] == expected_bucket


@pytest.mark.parametrize(
    ("count", "expected_bucket"),
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 3),
        (5, 3),
    ],
)
def test_creature_count_bucketing(count: int, expected_bucket: int) -> None:
    # Own creature count
    obs_my = _make_synthetic_obs(my_creatures=count)
    features_my = extract_abstract_state(obs_my)
    assert features_my[3] == expected_bucket

    # Opponent creature count
    obs_opp = _make_synthetic_obs(opp_creatures=count)
    features_opp = extract_abstract_state(obs_opp)
    assert features_opp[4] == expected_bucket


@pytest.mark.parametrize(
    ("count", "expected_bucket"),
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 2),
        (7, 2),
    ],
)
def test_hand_count_bucketing(count: int, expected_bucket: int) -> None:
    obs = _make_synthetic_obs(hand_count=count)
    features = extract_abstract_state(obs)
    assert features[5] == expected_bucket


@pytest.mark.parametrize(
    ("phase", "expected_bucket"),
    [
        (Phase.MAIN1, 0),
        (Phase.DECLARE_ATTACKERS, 1),
        (Phase.DECLARE_BLOCKERS, 2),
        (Phase.MAIN2, 3),
        (Phase.CHOOSE_TARGET, 4),
    ],
)
def test_phase_decision_context(phase: Phase, expected_bucket: int) -> None:
    obs = _make_synthetic_obs(phase=phase)
    features = extract_abstract_state(obs)
    assert features[6] == expected_bucket


def test_encode_decode_round_trip() -> None:
    # Test boundary combinations
    boundaries = [
        (0, 0, 0, 0, 0, 0, 0),
        (3, 3, 4, 3, 3, 2, 4),
        (0, 3, 2, 1, 3, 1, 4),
        (2, 1, 4, 0, 2, 2, 1),
    ]
    for b in boundaries:
        s_id = encode_state(b)
        assert 0 <= s_id < NUM_STATES
        assert decode_state(s_id) == b

    # Test random sampling
    rng = np.random.default_rng(42)
    for _ in range(500):
        f = tuple(int(rng.integers(0, d)) for d in STATE_DIMS)
        s_id = encode_state(f)
        assert 0 <= s_id < NUM_STATES
        assert decode_state(s_id) == f


def test_extract_state_from_real_game_states() -> None:
    state = GameState(seed=123)
    for _ in range(100):
        if state.is_terminal():
            break
        player = state.current_player()
        obs = state.observe(player)
        features = extract_abstract_state(obs)
        assert len(features) == 7
        for val, dim in zip(features, STATE_DIMS):
            assert 0 <= val < dim
        s_id = encode_state(features)
        assert 0 <= s_id < NUM_STATES
        assert decode_state(s_id) == features

        mask = state.legal_actions()
        action = int(np.flatnonzero(mask)[0])
        state.step(action)
