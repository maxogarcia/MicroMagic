"""Abstract state for the tabular agents: a few bucketed features instead of the 925-number observation.

Buckets follow the team plan (docs/Aditya_Implementation_Plan.md, section 4). The plan left
"decision context" open; here it is the kind of decision the macro-agent faces:
pre-combat main phase, post-combat main phase, or choosing blocks.
"""

from enum import IntEnum
from typing import NamedTuple

import numpy as np

from micromagic.env.spec import CREATURE_SLOTS, EMPTY, Phase
from micromagic.env.state import GameState


class Context(IntEnum):
    PRECOMBAT = 0   # Main 1 (and the attack step, which the attack macros run)
    POSTCOMBAT = 1  # Main 2: the last plays of the turn, then end it
    BLOCK = 2       # opponent is attacking: choose blocks


LIFE_EDGES = (5, 10, 15)            # <=5 | 6-10 | 11-15 | 16+
POWER_DIFF_EDGES = (-4, -1, 0, 3)   # <=-4 | -3..-1 | 0 | 1..3 | >=4
COUNT_EDGES = (0, 1, 2)             # 0 | 1 | 2 | 3+
HAND_EDGES = (0, 2)                 # 0 | 1-2 | 3+

# Number of buckets per feature, in AbstractState field order.
STATE_SHAPE = (4, 4, 5, 4, 4, 3, len(Context))
NUM_ABSTRACT_STATES = int(np.prod(STATE_SHAPE))  # 11,520


class AbstractState(NamedTuple):
    my_life: int
    opp_life: int
    power_diff: int
    my_creatures: int
    opp_creatures: int
    hand: int
    context: int

    @property
    def index(self) -> int:
        """A single integer in [0, NUM_ABSTRACT_STATES), for indexing Q-tables."""
        return int(np.ravel_multi_index(tuple(self), STATE_SHAPE))

    @classmethod
    def from_index(cls, index: int) -> "AbstractState":
        return cls(*(int(i) for i in np.unravel_index(index, STATE_SHAPE)))


def bucket(value: int, edges: tuple[int, ...]) -> int:
    """Index of the first edge that `value` is <= to; values above every edge get the last bucket."""
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def decision_context(state: GameState) -> Context:
    if state.phase == Phase.DECLARE_BLOCKERS:
        return Context.BLOCK
    if state.phase == Phase.MAIN2:
        return Context.POSTCOMBAT
    return Context.PRECOMBAT


def board_power(state: GameState, player: int) -> int:
    return sum(state.power(player, s) for s in range(CREATURE_SLOTS) if state.creature[player, s] != EMPTY)


def abstract_state(state: GameState, player: int) -> AbstractState:
    """Bucketed features of `state` from `player`'s point of view (only information `player` can see)."""
    opp = 1 - player
    return AbstractState(
        my_life=bucket(int(state.life[player]), LIFE_EDGES),
        opp_life=bucket(int(state.life[opp]), LIFE_EDGES),
        power_diff=bucket(board_power(state, player) - board_power(state, opp), POWER_DIFF_EDGES),
        my_creatures=bucket(int((state.creature[player] != EMPTY).sum()), COUNT_EDGES),
        opp_creatures=bucket(int((state.creature[opp] != EMPTY).sum()), COUNT_EDGES),
        hand=bucket(int(state.hand_count[player]), HAND_EDGES),
        context=int(decision_context(state)),
    )
