"""AlphaZero-lite: a policy/value network that guides determinized MCTS.

The network reads what the acting seat can see, GameState.observe(player), and returns
- policy logits over the 29 actions (illegal ones are masked before the softmax), and
- a value in [-1, 1]: the expected outcome for that seat (+1 = win).

NetworkEvaluator wraps it in the Evaluator interface from micromagic.search, so the
same PIMC + PUCT search Scott built for the rollout agent now takes its priors and leaf
values from the network instead of from random playouts. Because MCTS only ever sees
determinize()d worlds and the network only sees observe(player), neither can use the
opponent's hidden hand.

Two agents are built on it:
- AlphaZeroAgent: the full agent, network-guided search (needs the game state).
- PolicyNetAgent: the raw policy head with no search. It is cheap to run and shows how
  much of AlphaZero-lite's strength comes from the search versus the network alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from torch import nn

from micromagic.agents.base import Agent
from micromagic.env.spec import NUM_ACTIONS, OBS_DIM
from micromagic.env.state import GameState
from micromagic.search.mcts import MCTSAgent


class PolicyValueNet(nn.Module):
    """A shared MLP trunk with a policy head (logits) and a value head (tanh)."""

    def __init__(self, hidden: tuple[int, ...] | list[int] = (256, 256)) -> None:
        super().__init__()
        self.hidden = [int(h) for h in hidden]
        layers: list[nn.Module] = []
        width = OBS_DIM
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(width, NUM_ACTIONS)
        self.value_head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    @th.inference_mode()
    def predict(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Masked softmax priors (zero on illegal actions) and the value, for one observation."""
        logits, value = self(th.as_tensor(obs, dtype=th.float32).unsqueeze(0))
        logits = logits[0].double().numpy()
        logits = np.where(mask, logits, -np.inf)
        priors = np.exp(logits - logits.max())
        return priors / priors.sum(), float(value[0])

    def checkpoint(self) -> dict[str, Any]:
        return {"hidden": self.hidden, "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict[str, Any]) -> PolicyValueNet:
        net = cls(ckpt["hidden"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net


def masked_log_softmax(logits: th.Tensor, masks: th.Tensor) -> th.Tensor:
    """log pi over legal actions; illegal entries are 0 (not -inf) so pi * log pi stays finite."""
    logp = th.log_softmax(logits.masked_fill(~masks, -th.inf), dim=-1)
    return th.where(masks, logp, th.zeros_like(logp))


def az_loss(
    net: PolicyValueNet, obs: th.Tensor, masks: th.Tensor, pis: th.Tensor, zs: th.Tensor
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """AlphaZero loss: cross-entropy to the search policy plus MSE to the game outcome.
    (L2 regularization comes from the optimizer's weight_decay.)"""
    logits, values = net(obs)
    policy_loss = -(pis * masked_log_softmax(logits, masks)).sum(dim=-1).mean()
    value_loss = th.mean((values - zs) ** 2)
    return policy_loss + value_loss, policy_loss, value_loss


class NetworkEvaluator:
    """Evaluator for MCTS: priors and value from the network, read from `player`'s view."""

    def __init__(self, net: PolicyValueNet) -> None:
        self.net = net
        self.calls = 0

    def __call__(self, state: GameState, player: int) -> tuple[np.ndarray, float]:
        self.calls += 1
        return self.net.predict(state.observe(player), state.legal_actions())


class AlphaZeroAgent(MCTSAgent):
    """Network-guided determinized PUCT search. Set dirichlet_alpha only for self-play."""

    def __init__(
        self,
        net: PolicyValueNet | None = None,
        determinizations: int = 2,
        simulations: int = 32,
        c_puct: float = 1.5,
        dirichlet_alpha: float | None = None,
        seed: int | None = None,
        name: str = "alphazero",
    ) -> None:
        self.net = net if net is not None else PolicyValueNet()
        self.net.eval()
        super().__init__(
            determinizations=determinizations,
            simulations=simulations,
            c_puct=c_puct,
            evaluator=NetworkEvaluator(self.net),
            dirichlet_alpha=dirichlet_alpha,
            seed=seed,
            name=name,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs) -> AlphaZeroAgent:
        return cls(net=load_net(path), **kwargs)

    def save(self, path: str | Path) -> None:
        th.save(self.net.checkpoint(), path)

    def load(self, path: str | Path) -> None:
        self.net.load_state_dict(load_net(path).state_dict())
        self.net.eval()


class PolicyNetAgent(Agent):
    """The network's policy head alone: greedy (or sampled) over legal actions, no search."""

    def __init__(
        self, net: PolicyValueNet | None = None, deterministic: bool = True, seed: int | None = None,
        name: str = "az_policy",
    ) -> None:
        super().__init__(name)
        self.net = net if net is not None else PolicyValueNet()
        self.net.eval()
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        mask = np.asarray(mask, dtype=bool)
        priors, _ = self.net.predict(obs, mask)
        if self.deterministic:
            return int(np.argmax(priors))
        return int(self.rng.choice(NUM_ACTIONS, p=priors))

    def save(self, path: str | Path) -> None:
        th.save(self.net.checkpoint(), path)

    def load(self, path: str | Path) -> None:
        self.net.load_state_dict(load_net(path).state_dict())
        self.net.eval()


def load_net(path: str | Path) -> PolicyValueNet:
    return PolicyValueNet.from_checkpoint(th.load(path, map_location="cpu", weights_only=False))
