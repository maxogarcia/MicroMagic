"""MicroMagic Tabular State Abstraction and Macro-Action Execution (M1/M2)."""

from micromagic.tabular.abstraction import (
    NUM_STATES,
    STATE_DIMS,
    decode_state,
    encode_state,
    extract_abstract_state,
)
from micromagic.tabular.macros import (
    NUM_MACROS,
    MacroAction,
    compute_macro_mask,
    execute_macro,
)


__all__ = [
    "NUM_MACROS",
    "NUM_STATES",
    "STATE_DIMS",
    "MacroAction",
    "compute_macro_mask",
    "decode_state",
    "encode_state",
    "execute_macro",
    "extract_abstract_state",
]
