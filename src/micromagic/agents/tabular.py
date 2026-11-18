"""Tabular RL agents: First-Visit Monte Carlo Control and 4-Step SARSA (M2).

Implements on-policy First-Visit MC and n-step SARSA operating over the 19,200
discrete abstract state space and 9 macro-actions developed in M1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from micromagic.agents.base import Agent
from micromagic.tabular.abstraction import (
    NUM_STATES,
    encode_state,
    extract_abstract_state,
)
from micromagic.tabular.macros import (
    NUM_MACROS,
    MacroAction,
    compute_macro_mask,
    execute_macro,
)


class TabularAgent(Agent):
    """Base class for tabular RL agents operating on abstract states and macro actions."""

    def __init__(
        self,
        name: str | None = None,
        epsilon: float = 0.1,
        gamma: float = 0.99,
        deterministic: bool = False,
        seed: int | None = None,
        q_table: np.ndarray | None = None,
    ) -> None:
        super().__init__(name=name)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.deterministic = bool(deterministic)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if q_table is not None:
            if q_table.shape != (NUM_STATES, NUM_MACROS):
                raise ValueError(
                    f"Invalid q_table shape {q_table.shape}; expected {(NUM_STATES, NUM_MACROS)}"
                )
            self.q_table = np.asarray(q_table, dtype=np.float32)
        else:
            self.q_table = np.zeros((NUM_STATES, NUM_MACROS), dtype=np.float32)

    def select_macro(self, obs: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
        """Select a legal macro action using epsilon-soft policy over legal macros.

        Returns (abstract_state_id, chosen_macro_action).
        """
        s = encode_state(extract_abstract_state(obs))
        macro_mask = compute_macro_mask(obs, mask)
        legal_macros = np.flatnonzero(macro_mask)

        if len(legal_macros) == 0:
            return s, int(MacroAction.NO_ATTACK)

        if not self.deterministic and self.rng.random() < self.epsilon:
            chosen_macro = int(self.rng.choice(legal_macros))
        else:
            q_vals = self.q_table[s, legal_macros]
            max_q = np.max(q_vals)
            best_indices = legal_macros[np.isclose(q_vals, max_q)]
            if len(best_indices) == 1:
                chosen_macro = int(best_indices[0])
            else:
                chosen_macro = int(self.rng.choice(best_indices))

        return s, chosen_macro

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """Map observation and legal action mask to an executable legal primitive action."""
        if not mask.any():
            return 0
        _, chosen_macro = self.select_macro(obs, mask)
        return execute_macro(chosen_macro, obs, mask)

    def save(self, path: str | Path) -> None:
        """Persist Q-table and metadata to .npz checkpoint."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            algo=type(self).__name__,
            name=self.name,
            q_table=self.q_table,
            epsilon=self.epsilon,
            gamma=self.gamma,
            seed=self.seed if self.seed is not None else -1,
        )

    @classmethod
    def load(cls, path: str | Path) -> TabularAgent:
        """Load tabular checkpoint from .npz file."""
        data = np.load(path, allow_pickle=True)
        algo = str(data["algo"]) if "algo" in data else ""
        if (cls is TabularAgent and algo in ("MonteCarloAgent", "mc")) or issubclass(
            cls, MonteCarloAgent
        ):
            return MonteCarloAgent.load(path)
        if (cls is TabularAgent and algo in ("SarsaAgent", "sarsa")) or issubclass(
            cls, SarsaAgent
        ):
            return SarsaAgent.load(path)

        seed_val = (
            int(data["seed"]) if "seed" in data and int(data["seed"]) != -1 else None
        )
        return cls(
            name=str(data["name"]) if "name" in data else None,
            epsilon=float(data["epsilon"]) if "epsilon" in data else 0.1,
            gamma=float(data["gamma"]) if "gamma" in data else 0.99,
            seed=seed_val,
            q_table=data["q_table"],
        )


class MonteCarloAgent(TabularAgent):
    """On-policy First-Visit Monte Carlo Control agent (Sutton & Barto §5.4)."""

    def __init__(
        self,
        name: str | None = None,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        deterministic: bool = False,
        seed: int | None = None,
        q_table: np.ndarray | None = None,
        n_table: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            name=name,
            epsilon=epsilon,
            gamma=gamma,
            deterministic=deterministic,
            seed=seed,
            q_table=q_table,
        )
        if n_table is not None:
            if n_table.shape != (NUM_STATES, NUM_MACROS):
                raise ValueError(f"Invalid n_table shape {n_table.shape}")
            self.n_table = np.asarray(n_table, dtype=np.int32)
        else:
            self.n_table = np.zeros((NUM_STATES, NUM_MACROS), dtype=np.int32)

    def update_episode(
        self,
        transitions: list[tuple[int, int, float]],
        terminal_reward: float,
    ) -> None:
        """Update Q-values for first visits of (state, macro) in the episode."""
        if not transitions:
            return

        # Backwards sweep to calculate discounted returns G_t
        returns = [0.0] * len(transitions)
        G = float(terminal_reward)
        for t in range(len(transitions) - 1, -1, -1):
            s, a, r = transitions[t]
            G = r + (self.gamma * G if t < len(transitions) - 1 else G)
            returns[t] = G

        # Identify first visit index for each (state, macro) pair
        first_visit_indices: dict[tuple[int, int], int] = {}
        for t, (s, a, _) in enumerate(transitions):
            if (s, a) not in first_visit_indices:
                first_visit_indices[(s, a)] = t

        # Update Q(s, a) incrementally via running average
        for (s, a), t in first_visit_indices.items():
            g_ret = returns[t]
            self.n_table[s, a] += 1
            self.q_table[s, a] += (g_ret - self.q_table[s, a]) / self.n_table[s, a]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            algo="mc",
            name=self.name,
            q_table=self.q_table,
            n_table=self.n_table,
            epsilon=self.epsilon,
            gamma=self.gamma,
            seed=self.seed if self.seed is not None else -1,
        )

    @classmethod
    def load(cls, path: str | Path) -> MonteCarloAgent:
        data = np.load(path, allow_pickle=True)
        seed_val = (
            int(data["seed"]) if "seed" in data and int(data["seed"]) != -1 else None
        )
        return cls(
            name=str(data["name"]) if "name" in data else None,
            epsilon=float(data["epsilon"]) if "epsilon" in data else 0.1,
            gamma=float(data["gamma"]) if "gamma" in data else 1.0,
            seed=seed_val,
            q_table=data["q_table"],
            n_table=data["n_table"] if "n_table" in data else None,
        )


class SarsaAgent(TabularAgent):
    """4-step (or n-step) on-policy SARSA agent (Sutton & Barto §7.2)."""

    def __init__(
        self,
        name: str | None = None,
        n_steps: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        deterministic: bool = False,
        seed: int | None = None,
        q_table: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            name=name,
            epsilon=epsilon,
            gamma=gamma,
            deterministic=deterministic,
            seed=seed,
            q_table=q_table,
        )
        self.n_steps = int(n_steps)
        self.alpha = float(alpha)
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self.history.clear()

    def step_update(
        self,
        s: int,
        a: int,
        r: float,
        next_s: int,
        next_a: int | None,
        done: bool,
    ) -> None:
        """Perform n-step SARSA update for time tau = t - n + 1."""
        self.history.append({"s": int(s), "a": int(a), "r": float(r)})
        t = len(self.history) - 1
        tau = t - self.n_steps + 1

        if tau >= 0 and not done:
            g = 0.0
            for i in range(self.n_steps):
                g += (self.gamma**i) * self.history[tau + i]["r"]
            if next_a is not None:
                g += (self.gamma**self.n_steps) * self.q_table[next_s, next_a]

            s_tau = self.history[tau]["s"]
            a_tau = self.history[tau]["a"]
            self.q_table[s_tau, a_tau] += self.alpha * (g - self.q_table[s_tau, a_tau])

        if done:
            T = len(self.history)
            start_tau = max(0, T - self.n_steps)
            for tau_idx in range(start_tau, T):
                g = 0.0
                for i in range(tau_idx, T):
                    g += (self.gamma ** (i - tau_idx)) * self.history[i]["r"]
                s_tau = self.history[tau_idx]["s"]
                a_tau = self.history[tau_idx]["a"]
                self.q_table[s_tau, a_tau] += self.alpha * (
                    g - self.q_table[s_tau, a_tau]
                )
            self.reset()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            algo="sarsa",
            name=self.name,
            q_table=self.q_table,
            n_steps=self.n_steps,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            seed=self.seed if self.seed is not None else -1,
        )

    @classmethod
    def load(cls, path: str | Path) -> SarsaAgent:
        data = np.load(path, allow_pickle=True)
        seed_val = (
            int(data["seed"]) if "seed" in data and int(data["seed"]) != -1 else None
        )
        return cls(
            name=str(data["name"]) if "name" in data else None,
            n_steps=int(data["n_steps"]) if "n_steps" in data else 4,
            alpha=float(data["alpha"]) if "alpha" in data else 0.1,
            gamma=float(data["gamma"]) if "gamma" in data else 0.99,
            epsilon=float(data["epsilon"]) if "epsilon" in data else 0.1,
            seed=seed_val,
            q_table=data["q_table"],
        )
