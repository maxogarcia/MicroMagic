"""MacroAgent: plays through macro-actions, so a tabular policy only picks among 9 macros.

Subclass it (or pass `policy`) and implement choose_macro(abstract_state, macro_mask).
The agent records (abstract state index, macro) for every decision of the current game in
`history`, which is what Monte Carlo control needs for its episode returns.
"""

from collections.abc import Callable

import numpy as np

from micromagic.abstraction.features import AbstractState, Context, abstract_state
from micromagic.abstraction.macros import Macro, MacroExecution, macro_mask
from micromagic.agents.base import Agent
from micromagic.env.state import GameState


MacroPolicy = Callable[[AbstractState, np.ndarray], int]


class MacroAgent(Agent):
    def __init__(self, policy: MacroPolicy | None = None, seed: int | None = None, name: str = "macro") -> None:
        super().__init__(name)
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.history: list[tuple[int, int]] = []
        self._execution: MacroExecution | None = None
        self._game: GameState | None = None

    def choose_macro(self, abstract: AbstractState, mask: np.ndarray) -> int:
        """Pick a macro with mask[macro] True. Default: the policy, or uniformly random."""
        if self.policy is not None:
            return int(self.policy(abstract, mask))
        return int(self.rng.choice(np.flatnonzero(mask)))

    def act_from_state(self, state: GameState, player: int) -> int:
        if state is not self._game:  # a new game: forget the previous one
            self.reset()
            self._game = state
        if self._execution is not None:
            action = self._execution.next_action(state)
            if action is not None:
                return action

        mask = macro_mask(state, player)
        if not mask.any():
            # Not a macro decision point (only possible if another controller left the game
            # mid-macro); fall back to the first legal primitive action.
            return int(np.flatnonzero(state.legal_actions())[0])
        abstract = abstract_state(state, player)
        macro = self.choose_macro(abstract, mask)
        if not mask[macro]:
            raise ValueError(f"{self.name} chose unavailable macro {Macro(macro).name}")
        self.history.append((abstract.index, macro))
        self._execution = MacroExecution(state, player, Macro(macro))
        action = self._execution.next_action(state)
        if action is None:
            raise RuntimeError(f"{Macro(macro).name} produced no action")
        return action

    def reset(self) -> None:
        self.history = []
        self._execution = None
        self._game = None

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        raise TypeError("MacroAgent plans on the game state: call act_from_state(state, player).")


# Highest priority first. Removal before creatures so blockers are cleared; safe attacks only.
SCRIPTED_PRIORITY = {
    int(Context.PRECOMBAT): (Macro.PLAY_LAND, Macro.BURN_BEST_CREATURE, Macro.CAST_BEST_CREATURE,
                             Macro.ATTACK_SAFE, Macro.NO_ATTACK),
    int(Context.POSTCOMBAT): (Macro.PLAY_LAND, Macro.CAST_BEST_CREATURE, Macro.BURN_FACE, Macro.NO_ATTACK),
    int(Context.BLOCK): (Macro.BLOCK_VALUE, Macro.NO_BLOCK),
}


def scripted_macro_policy(abstract: AbstractState, mask: np.ndarray) -> int:
    """A fixed priority list over macros: a simple, deterministic baseline."""
    for macro in SCRIPTED_PRIORITY[abstract.context]:
        if mask[macro]:
            return int(macro)
    return int(np.flatnonzero(mask)[0])
