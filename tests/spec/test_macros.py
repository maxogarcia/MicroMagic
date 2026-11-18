"""Specification tests for MacroAction enum and macro-action executor.

Contract:
- 9 macro-actions:
    0: PLAY_LAND
    1: CAST_BEST_CREATURE
    2: BURN_FACE
    3: BURN_BEST_CREATURE
    4: ATTACK_ALL
    5: ATTACK_SAFE
    6: NO_ATTACK
    7: BLOCK_VALUE
    8: NO_BLOCK
- compute_macro_mask returns a boolean mask of shape (9,).
- compute_macro_mask guarantees at least one legal macro whenever primitive_mask has legal actions.
- execute_macro returns an action a in [0, 28] guaranteed to satisfy primitive_mask[a] == True.
"""

from __future__ import annotations

import numpy as np

from micromagic.env.spec import NUM_ACTIONS
from micromagic.env.state import GameState
from micromagic.tabular.macros import (
    NUM_MACROS,
    MacroAction,
    compute_macro_mask,
    execute_macro,
)


def test_macro_action_enum() -> None:
    assert MacroAction.PLAY_LAND == 0
    assert MacroAction.CAST_BEST_CREATURE == 1
    assert MacroAction.BURN_FACE == 2
    assert MacroAction.BURN_BEST_CREATURE == 3
    assert MacroAction.ATTACK_ALL == 4
    assert MacroAction.ATTACK_SAFE == 5
    assert MacroAction.NO_ATTACK == 6
    assert MacroAction.BLOCK_VALUE == 7
    assert MacroAction.NO_BLOCK == 8
    assert NUM_MACROS == 9
    assert len(MacroAction) == 9


def test_compute_macro_mask_empty_primitive_mask() -> None:
    obs = np.zeros(925, dtype=np.float32)
    empty_primitive = np.zeros(NUM_ACTIONS, dtype=bool)
    mask = compute_macro_mask(obs, empty_primitive)
    assert mask.shape == (NUM_MACROS,)
    assert mask.dtype == bool
    assert not mask.any()


def test_compute_macro_mask_always_has_legal_macro_when_actions_available() -> None:
    # Test across games and turns
    for seed in (1, 42, 99):
        state = GameState(seed=seed)
        for _ in range(150):
            if state.is_terminal():
                break
            player = state.current_player()
            obs = state.observe(player)
            primitive_mask = state.legal_actions()

            if primitive_mask.any():
                macro_mask = compute_macro_mask(obs, primitive_mask)
                assert macro_mask.shape == (NUM_MACROS,)
                assert macro_mask.dtype == bool
                assert macro_mask.any(), (
                    f"Expected at least one legal macro in phase {state.phase.name}, "
                    f"legal primitive actions: {np.flatnonzero(primitive_mask)}"
                )

            # Advance state using a legal action
            action = int(np.flatnonzero(primitive_mask)[0])
            state.step(action)


def test_execute_macro_guarantees_legal_primitive_action() -> None:
    """Every macro (even unmasked or arbitrary) must return a legal primitive action."""
    for seed in (7, 13, 101):
        state = GameState(seed=seed)
        for _ in range(120):
            if state.is_terminal():
                break
            player = state.current_player()
            obs = state.observe(player)
            primitive_mask = state.legal_actions()
            macro_mask = compute_macro_mask(obs, primitive_mask)

            # Test ALL 9 macros, whether available or not: none should ever return an illegal primitive action!
            for m in range(NUM_MACROS):
                action = execute_macro(m, obs, primitive_mask)
                assert 0 <= action < NUM_ACTIONS
                assert primitive_mask[action], (
                    f"execute_macro({MacroAction(m).name}) returned illegal action {action} "
                    f"in phase {state.phase.name} (seed={seed})"
                )

            # Test step forward by executing an available macro
            legal_macros = np.flatnonzero(macro_mask)
            chosen_macro = int(legal_macros[0])
            primitive_action = execute_macro(chosen_macro, obs, primitive_mask)
            assert primitive_mask[primitive_action]
            state.step(primitive_action)


def test_macro_legality_specific_phases() -> None:
    # Seed 0: start in MAIN1 with Mountain in hand
    state = GameState(seed=0)
    obs = state.observe(state.current_player())
    prim_mask = state.legal_actions()
    macro_mask = compute_macro_mask(obs, prim_mask)

    # In opening turn of seed 0, hand has Mountain and PASS is legal
    assert macro_mask[MacroAction.PLAY_LAND]
    assert macro_mask[MacroAction.NO_ATTACK]  # Pass is legal fallback

    # Executing PLAY_LAND returns a legal action in PLAY_HAND
    a = execute_macro(MacroAction.PLAY_LAND, obs, prim_mask)
    assert prim_mask[a]
