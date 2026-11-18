"""Leaf evaluators for MCTS: given a state, return move priors and a value estimate.

An evaluator is any callable `evaluate(state, player) -> (priors, value)`:
- priors: float array of length NUM_ACTIONS, zero on illegal actions, summing to 1
- value: expected outcome in [-1, 1] from `player`'s point of view (+1 = `player` wins)

AlphaZero-lite plugs in here: wrap the policy/value network so it reads
state.observe(player) and returns (softmax over legal actions, value head).
"""

from typing import Protocol

import numpy as np

from micromagic.env.spec import CREATURE_SLOTS, EMPTY, HAND_SIZE, STARTING_LIFE
from micromagic.env.state import GameState


class Evaluator(Protocol):
    def __call__(self, state: GameState, player: int) -> tuple[np.ndarray, float]: ...


def terminal_value(state: GameState, player: int) -> float:
    winner = state.winner()
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def uniform_priors(state: GameState) -> np.ndarray:
    mask = state.legal_actions()
    priors = mask.astype(np.float64)
    total = priors.sum()
    return priors / total if total else priors


# A board score is a guess, so it stays well inside the +/-1 of a finished game; otherwise
# search cannot tell "winning now" from "looking good" and drifts away from lethal lines.
BOARD_VALUE_CAP = 0.6


def board_value(state: GameState, player: int) -> float:
    """A hand-tuned score of the position: life, board, cards in hand, and lands.

    Finished games score exactly -1, 0 or +1; unfinished ones stay within +/-BOARD_VALUE_CAP.
    """
    if state.is_terminal():
        return terminal_value(state, player)
    opp = 1 - player

    def board(p: int) -> float:
        total = 0.0
        for slot in range(CREATURE_SLOTS):
            if state.creature[p, slot] != EMPTY:
                total += state.power(p, slot) + state.toughness(p, slot)
        return total

    life = (int(state.life[player]) - int(state.life[opp])) / STARTING_LIFE
    creatures = (board(player) - board(opp)) / 20.0
    cards = (int(state.hand_count[player]) - int(state.hand_count[opp])) / HAND_SIZE
    lands = (int(state.lands_total[player]) - int(state.lands_total[opp])) / 10.0
    return BOARD_VALUE_CAP * float(np.tanh(1.5 * life + 1.0 * creatures + 0.3 * cards + 0.2 * lands))


class HeuristicEvaluator:
    """Uniform priors and the board score. No simulation, so it is the fastest evaluator."""

    def __call__(self, state: GameState, player: int) -> tuple[np.ndarray, float]:
        return uniform_priors(state), board_value(state, player)


class RolloutEvaluator:
    """Uniform priors; value from a short random playout followed by the board score.

    Random playouts are noisy in a 25-turn game, so they are cut off after `depth`
    actions and the board score judges the position reached.
    """

    def __init__(self, depth: int = 10, seed: int | None = None) -> None:
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        self.depth = depth
        self.rng = np.random.default_rng(seed)

    def __call__(self, state: GameState, player: int) -> tuple[np.ndarray, float]:
        priors = uniform_priors(state)
        if self.depth == 0 or state.is_terminal():
            return priors, board_value(state, player)
        sim = state.clone()
        for _ in range(self.depth):
            if sim.is_terminal():
                break
            legal = np.flatnonzero(sim.legal_actions())
            sim.step(int(legal[self.rng.integers(legal.size)]))
        return priors, board_value(sim, player)

