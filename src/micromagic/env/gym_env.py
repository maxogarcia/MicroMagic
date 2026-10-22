"""Single-agent Gymnasium wrapper: one learning seat versus a fixed opponent policy."""

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from micromagic.env.spec import NUM_ACTIONS, OBS_DIM, PASS
from micromagic.env.state import GameState


# An opponent policy maps (observation, legal-action mask) to an action index.
OpponentPolicy = Callable[[np.ndarray, np.ndarray], int]


class MicroMagicGymEnv(gym.Env):
    """The legal-action mask is in info["action_mask"] and from action_masks(),
    which is what sb3-contrib's MaskablePPO calls. The default opponent picks
    uniformly among legal actions.

    on_illegal="raise" (default) raises on an illegal action so masking bugs
    surface immediately. on_illegal="pass" plays PASS instead and sets
    info["illegal_action"], for unmasked agents and Gymnasium's env checker.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_policy: OpponentPolicy | None = None,
        agent_player: int = 0,
        decks: tuple[np.ndarray, np.ndarray] | None = None,
        render_mode: str | None = None,
        on_illegal: str = "raise",
    ) -> None:
        super().__init__()
        if on_illegal not in ("raise", "pass"):
            raise ValueError(f"on_illegal must be 'raise' or 'pass', got {on_illegal!r}")
        self.on_illegal = on_illegal
        self.opponent_policy = opponent_policy
        self.agent_player = agent_player
        self.decks = decks
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.state: GameState | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(2**31))
        self.state = GameState(seed=game_seed, decks=self.decks)
        self._play_opponent()
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        is_illegal = not self.state.legal_actions()[action]
        if is_illegal and self.on_illegal == "pass":
            action = PASS
        self.state.step(action)
        self._play_opponent()

        reward = 0.0
        terminated = truncated = False
        if self.state.is_terminal():
            truncated = self.state.is_truncated()
            terminated = not truncated
            winner = self.state.winner()
            if winner is not None:
                reward = 1.0 if winner == self.agent_player else -1.0
        info = self._info()
        info["illegal_action"] = is_illegal
        return self._obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.state.legal_actions()

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return repr(self.state)
        return None

    def _play_opponent(self) -> None:
        opponent = 1 - self.agent_player
        while not self.state.is_terminal() and self.state.current_player() == opponent:
            mask = self.state.legal_actions()
            if self.opponent_policy is None:
                action = int(self.np_random.choice(np.flatnonzero(mask)))
            else:
                action = int(self.opponent_policy(self.state.observe(opponent), mask))
            self.state.step(action)

    def _obs(self) -> np.ndarray:
        return self.state.observe(self.agent_player)

    def _info(self) -> dict:
        return {"action_mask": self.state.legal_actions()}
