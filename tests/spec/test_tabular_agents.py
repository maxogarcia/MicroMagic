"""Specification tests for Tabular RL Agents (M2).

Contract:
- MonteCarloAgent and SarsaAgent inherit from Agent and satisfy AgentProtocol.
- Agent act(obs, mask) strictly guarantees mask[action] == True.
- First-Visit Monte Carlo Control correctly computes sample-average returns
  for first visits of (state, macro) within each episode.
- 4-Step SARSA correctly calculates n-step discounted bootstrapped returns
  and updates Q-table with step-size alpha.
- Checkpoints save and restore Q-table and hyperparameters identically.
- Tabular training harness runs end-to-end and updates Q-tables against baseline opponents.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from micromagic.agents import Agent, MonteCarloAgent, SarsaAgent, TabularAgent
from micromagic.env.spec import NUM_ACTIONS
from micromagic.env.state import GameState
from micromagic.tabular.abstraction import NUM_STATES
from micromagic.tabular.macros import NUM_MACROS
from micromagic.tournaments import AgentProtocol
from micromagic.training.train_tabular import train_tabular


def test_tabular_agent_interface() -> None:
    mc = MonteCarloAgent(name="mc_test")
    sarsa = SarsaAgent(name="sarsa_test")

    assert isinstance(mc, Agent)
    assert isinstance(mc, TabularAgent)
    assert isinstance(mc, AgentProtocol)
    assert mc.name == "mc_test"
    assert mc.q_table.shape == (NUM_STATES, NUM_MACROS)

    assert isinstance(sarsa, Agent)
    assert isinstance(sarsa, TabularAgent)
    assert isinstance(sarsa, AgentProtocol)
    assert sarsa.name == "sarsa_test"
    assert sarsa.q_table.shape == (NUM_STATES, NUM_MACROS)


def test_tabular_agent_act_guarantees_legal_primitive() -> None:
    for AgentCls in (MonteCarloAgent, SarsaAgent):
        agent = AgentCls(seed=42)
        # Across 50 randomized game states and legal masks, verify action legality
        for seed in range(50):
            state = GameState(seed=seed)
            step_count = 0
            while not state.is_terminal():
                player = state.current_player()
                obs = state.observe(player)
                mask = state.legal_actions()
                action = agent.act(obs, mask)
                assert 0 <= action < NUM_ACTIONS, (
                    f"Action {action} outside [0, {NUM_ACTIONS})"
                )
                assert mask[action], (
                    f"Illegal action {action} chosen by {AgentCls.__name__}"
                )
                state.step(action)
                step_count += 1
                if step_count > 20:
                    break


def test_first_visit_mc_update() -> None:
    agent = MonteCarloAgent(epsilon=0.0)
    assert agent.q_table.shape == (NUM_STATES, NUM_MACROS)

    s1, a1 = 10, 2
    s2, a2 = 20, 3
    # Episode with repeat visit of (s1, a1):
    # (s1, a1, r=0), (s2, a2, r=0), (s1, a1, r=0), terminal reward = +1.0
    # First-visit MC must update (s1, a1) once with G=1.0 and (s2, a2) once with G=1.0
    transitions = [
        (s1, a1, 0.0),
        (s2, a2, 0.0),
        (s1, a1, 0.0),
    ]
    terminal_reward = 1.0

    agent.update_episode(transitions, terminal_reward)

    # Q(s, a) = G = 1.0 for first visits
    assert np.isclose(agent.q_table[s1, a1], 1.0)
    assert np.isclose(agent.q_table[s2, a2], 1.0)
    assert agent.n_table[s1, a1] == 1
    assert agent.n_table[s2, a2] == 1

    # Second episode: (s1, a1) visited again, terminal reward = 0.0
    # New Q(s1, a1) = (1.0 * 1 + 0.0) / 2 = 0.5
    agent.update_episode([(s1, a1, 0.0)], terminal_reward=0.0)
    assert np.isclose(agent.q_table[s1, a1], 0.5)
    assert agent.n_table[s1, a1] == 2


def test_four_step_sarsa_update() -> None:
    # 4-step SARSA: n=4, gamma=0.9, alpha=0.5
    agent = SarsaAgent(n_steps=4, gamma=0.9, alpha=0.5, epsilon=0.0)

    # Hand-craft 5 steps of transitions:
    # t=0: s0=1, a0=0, r1=1.0, s1=2, a1=1
    # t=1: s1=2, a1=1, r2=2.0, s2=3, a2=2
    # t=2: s2=3, a2=2, r3=0.0, s3=4, a3=0
    # t=3: s3=4, a3=0, r4=1.0, s4=5, a4=1
    # At t=3 (step 4), the 4-step target for (s0, a0) is ready:
    # G_0 = r1 + gamma*r2 + gamma^2*r3 + gamma^3*r4 + gamma^4 * Q(s4, a4)
    # Set Q(s4, a4) = 4.0 initially
    agent.q_table[5, 1] = 4.0

    # Step t=0
    agent.step_update(s=1, a=0, r=1.0, next_s=2, next_a=1, done=False)
    assert agent.q_table[1, 0] == 0.0  # not updated yet (buffer length < 4)

    # Step t=1
    agent.step_update(s=2, a=1, r=2.0, next_s=3, next_a=2, done=False)
    # Step t=2
    agent.step_update(s=3, a=2, r=0.0, next_s=4, next_a=0, done=False)
    # Step t=3 (tau = 0 update occurs)
    agent.step_update(s=4, a=0, r=1.0, next_s=5, next_a=1, done=False)

    expected_g0 = 1.0 + 0.9 * 2.0 + (0.9**2) * 0.0 + (0.9**3) * 1.0 + (0.9**4) * 4.0
    expected_q0 = 0.0 + 0.5 * (expected_g0 - 0.0)
    assert np.isclose(agent.q_table[1, 0], expected_q0)

    # Now terminate episode at t=4: next_s=6, done=True, r5=3.0
    agent.step_update(s=5, a=1, r=3.0, next_s=6, next_a=None, done=True)
    # End of episode should flush remaining steps tau=1, 2, 3, 4
    # Check that (s1, a1)=(2, 1) has been updated
    assert agent.q_table[2, 1] != 0.0


def test_tabular_agent_save_and_load(tmp_path: Path) -> None:
    save_path = tmp_path / "test_mc_agent.npz"
    agent = MonteCarloAgent(epsilon=0.15, gamma=0.95, seed=123)
    agent.q_table[100, 3] = 42.5
    agent.q_table[200, 5] = -3.14
    agent.n_table[100, 3] = 17

    agent.save(save_path)
    assert save_path.exists()

    loaded_agent = MonteCarloAgent.load(save_path)
    assert isinstance(loaded_agent, MonteCarloAgent)
    assert np.array_equal(loaded_agent.q_table, agent.q_table)
    assert np.array_equal(loaded_agent.n_table, agent.n_table)
    assert loaded_agent.epsilon == agent.epsilon
    assert loaded_agent.gamma == agent.gamma

    # Also test SarsaAgent save/load
    sarsa_path = tmp_path / "test_sarsa_agent.npz"
    sarsa = SarsaAgent(n_steps=4, alpha=0.05, gamma=0.99, epsilon=0.2, seed=456)
    sarsa.q_table[50, 1] = 9.99
    sarsa.save(sarsa_path)

    loaded_sarsa = SarsaAgent.load(sarsa_path)
    assert isinstance(loaded_sarsa, SarsaAgent)
    assert np.array_equal(loaded_sarsa.q_table, sarsa.q_table)
    assert loaded_sarsa.alpha == sarsa.alpha
    assert loaded_sarsa.n_steps == sarsa.n_steps
    assert loaded_sarsa.gamma == sarsa.gamma


def test_train_tabular_smoke() -> None:
    # 5 episodes of MC against RandomAgent
    mc_agent = train_tabular(
        algo="mc",
        episodes=5,
        opponent="random",
        seed=42,
    )
    assert isinstance(mc_agent, MonteCarloAgent)
    assert np.count_nonzero(mc_agent.q_table) > 0

    # 5 episodes of SARSA against RandomAgent
    sarsa_agent = train_tabular(
        algo="sarsa",
        episodes=5,
        opponent="random",
        seed=42,
    )
    assert isinstance(sarsa_agent, SarsaAgent)
    assert np.count_nonzero(sarsa_agent.q_table) > 0
