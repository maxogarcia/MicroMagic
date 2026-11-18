"""Perfect-information Monte Carlo tree search (PIMC) with PUCT selection.

MicroMagic hides the opponent's hand and both library orders, so a search tree over the
true state would let the agent peek. Instead, each decision samples K "worlds" with
GameState.determinize(player, rng), which keeps everything the player can see and
reshuffles everything they cannot. A PUCT search runs in each world, and the root visit
counts are summed across worlds (Cowling, Ward and Powley 2012 call this ensemble
determinization). Within one world the game is deterministic, because draws come from
the sampled library order.

Two players act in a non-alternating order (the defender chooses blocks during the
attacker's turn), so every node records whose decision it is, and values are backed up
from that player's point of view.
"""

from dataclasses import dataclass, field

import numpy as np

from micromagic.agents.base import Agent
from micromagic.env.spec import NUM_ACTIONS
from micromagic.env.state import GameState
from micromagic.search.evaluators import Evaluator, RolloutEvaluator, terminal_value


@dataclass(slots=True)
class Node:
    state: GameState
    player: int                  # who chooses at this node
    actions: np.ndarray          # legal action indices
    priors: np.ndarray           # prior per legal action
    visits: np.ndarray           # N(s, a)
    value_sum: np.ndarray        # W(s, a), from `player`'s point of view
    children: list = field(default_factory=list)


class MCTS:
    """PUCT search in one fully observed (determinized) world."""

    def __init__(
        self,
        evaluator: Evaluator,
        simulations: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: float | None = None,
        noise_fraction: float = 0.25,
        rng: np.random.Generator | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError(f"simulations must be >= 1, got {simulations}")
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.rng = rng or np.random.default_rng()

    def search(self, state: GameState) -> np.ndarray:
        """Run the simulations from `state` (which the search may modify) and return root visit counts."""
        if state.is_terminal():
            raise ValueError("Cannot search from a finished game")
        root, _, _ = self._expand(state)
        if self.dirichlet_alpha is not None and root.actions.size > 1:
            # AlphaZero-style exploration noise at the root, for self-play training.
            noise = self.rng.dirichlet([self.dirichlet_alpha] * root.actions.size)
            root.priors = (1 - self.noise_fraction) * root.priors + self.noise_fraction * noise

        for _ in range(self.simulations):
            node, path = root, []
            while True:
                i = self._select(node)
                path.append((node, i))
                child = node.children[i]
                if child is None:
                    next_state = node.state.clone()
                    next_state.step(int(node.actions[i]))
                    child, value, perspective = self._expand(next_state)
                    node.children[i] = child
                    break
                if isinstance(child, tuple):  # a finished game: (value, perspective)
                    value, perspective = child
                    break
                node = child
            for parent, i in path:
                parent.visits[i] += 1
                parent.value_sum[i] += value if parent.player == perspective else -value

        counts = np.zeros(NUM_ACTIONS, dtype=np.float64)
        counts[root.actions] = root.visits
        return counts

    def _expand(self, state: GameState) -> tuple["Node | tuple", float, int]:
        """Create a node for `state`; return it with the leaf value and whose view that value is from."""
        if state.is_terminal():
            value = terminal_value(state, 0)
            return (value, 0), value, 0
        player = state.current_player()
        priors, value = self.evaluator(state, player)
        actions = np.flatnonzero(state.legal_actions())
        node = Node(
            state=state,
            player=player,
            actions=actions,
            priors=np.asarray(priors, dtype=np.float64)[actions],
            visits=np.zeros(actions.size, dtype=np.int64),
            value_sum=np.zeros(actions.size, dtype=np.float64),
            children=[None] * actions.size,
        )
        return node, float(value), player

    def _select(self, node: Node) -> int:
        if node.actions.size == 1:
            return 0
        total = node.visits.sum()
        q = np.divide(node.value_sum, node.visits, out=np.zeros_like(node.value_sum), where=node.visits > 0)
        u = self.c_puct * node.priors * np.sqrt(total + 1) / (1 + node.visits)
        scores = q + u
        best = np.flatnonzero(scores == scores.max())
        return int(best[0]) if best.size == 1 else int(self.rng.choice(best))


class MCTSAgent(Agent):
    """Determinized PUCT search over K sampled worlds per decision.

    Needs the game state, so callers use act_from_state(state, player); the tournament
    runner and the Gymnasium wrapper do this automatically for agents that define it.
    The search only ever sees determinize()d copies, so it cannot use hidden information.
    """

    def __init__(
        self,
        determinizations: int = 4,
        simulations: int = 64,
        c_puct: float = 1.5,
        evaluator: Evaluator | None = None,
        dirichlet_alpha: float | None = None,
        seed: int | None = None,
        name: str = "mcts",
    ) -> None:
        super().__init__(name)
        if determinizations < 1:
            raise ValueError(f"determinizations must be >= 1, got {determinizations}")
        self.determinizations = determinizations
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mcts = MCTS(
            evaluator if evaluator is not None else RolloutEvaluator(seed=seed),
            simulations=simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            rng=self.rng,
        )

    def policy(self, state: GameState, player: int) -> np.ndarray:
        """Visit distribution over the 29 actions, summed over K worlds (an AlphaZero training target)."""
        counts = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for _ in range(self.determinizations):
            counts += self.mcts.search(state.determinize(player, self.rng))
        return counts / counts.sum()

    def act_from_state(self, state: GameState, player: int) -> int:
        if state.current_player() != player:
            raise ValueError(f"It is player {state.current_player()}'s decision, not player {player}'s")
        legal = np.flatnonzero(state.legal_actions())
        if legal.size == 1:
            return int(legal[0])  # forced move: about a quarter of all decisions
        probs = self.policy(state, player)
        best = np.flatnonzero(probs == probs.max())
        return int(best[0]) if best.size == 1 else int(self.rng.choice(best))

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        raise TypeError(
            "MCTSAgent searches the game state, not the observation vector: "
            "call act_from_state(state, player)."
        )
