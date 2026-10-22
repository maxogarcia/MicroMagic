"""Agent base class and Random baseline: legality, determinism, harness compatibility."""

import numpy as np
import pytest

from micromagic.agents import Agent, RandomAgent
from micromagic.env.gym_env import MicroMagicGymEnv
from micromagic.env.spec import NUM_ACTIONS
from micromagic.env.state import GameState
from micromagic.tournaments import AgentProtocol, TournamentRunner


def play_game(agents: dict[int, Agent], seed: int) -> list[int]:
    state = GameState(seed=seed)
    actions = []
    while not state.is_terminal():
        player = state.current_player()
        mask = state.legal_actions()
        action = agents[player].act(state.observe(player), mask)
        assert mask[action], f"illegal action {action}"
        actions.append(action)
        state.step(action)
    return actions


def test_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]


def test_random_agent_satisfies_harness_protocol() -> None:
    assert isinstance(RandomAgent(seed=0), AgentProtocol)


def test_random_agent_only_picks_legal_actions() -> None:
    for seed in range(50):
        play_game({0: RandomAgent(seed=seed), 1: RandomAgent(seed=seed + 1)}, seed)


def test_random_agent_same_seed_same_game() -> None:
    first = play_game({0: RandomAgent(seed=1), 1: RandomAgent(seed=2)}, seed=7)
    second = play_game({0: RandomAgent(seed=1), 1: RandomAgent(seed=2)}, seed=7)
    assert first == second


def test_random_agent_covers_all_legal_actions() -> None:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[0, 3, 9]] = True
    agent = RandomAgent(seed=0)
    picks = {agent.act(np.zeros(1), mask) for _ in range(200)}
    assert picks == {0, 3, 9}


def test_random_agent_rejects_empty_mask() -> None:
    with pytest.raises(ValueError):
        RandomAgent(seed=0).act(np.zeros(1), np.zeros(NUM_ACTIONS, dtype=bool))


def test_agent_works_as_gym_opponent_policy() -> None:
    env = MicroMagicGymEnv(opponent_policy=RandomAgent(seed=0), agent_player=1)
    env.reset(seed=0)
    learner = RandomAgent(seed=1)
    done = False
    while not done:
        mask = env.action_masks()
        assert mask.any()
        _, _, terminated, truncated, _ = env.step(learner.act(np.zeros(1), mask))
        done = terminated or truncated


def test_random_vs_random_through_tournament_runner() -> None:
    runner = TournamentRunner(lambda s: GameState(seed=s))
    result = runner.run_match(RandomAgent(seed=0, name="a"), RandomAgent(seed=1, name="b"), num_games=20)
    summary = result.summary()
    assert summary["total_games"] == 20
    assert sum(summary["wins"].values()) + summary["draws"] == 20
