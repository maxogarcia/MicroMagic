"""Specification tests for the unified one-command tournament launcher (M4)."""

from __future__ import annotations

import pandas as pd

from micromagic.agents.heuristic import HeuristicAgent
from micromagic.agents.random_agent import RandomAgent
from micromagic.agents.tabular import MonteCarloAgent, SarsaAgent
from micromagic.search.mcts import MCTSAgent
from micromagic.tournaments.launcher import load_agent, run_tournament


def test_agent_loader_registry():
    """Verify load_agent(cfg) instantiates RandomAgent, HeuristicAgent, MonteCarloAgent, SarsaAgent, and MCTSAgent."""
    # 1. RandomAgent
    agent_random = load_agent({"name": "test_random", "kind": "random", "seed": 42})
    assert isinstance(agent_random, RandomAgent)
    assert agent_random.name == "test_random"

    # 2. HeuristicAgent
    agent_heuristic = load_agent({"name": "test_heuristic", "kind": "heuristic"})
    assert isinstance(agent_heuristic, HeuristicAgent)
    assert agent_heuristic.name == "test_heuristic"

    # 3. MonteCarloAgent (from committed checkpoint)
    agent_mc = load_agent(
        {
            "name": "test_mc",
            "kind": "tabular_mc",
            "checkpoint": "results/tabular/mc_final.npz",
        }
    )
    assert isinstance(agent_mc, MonteCarloAgent)
    assert agent_mc.name == "test_mc"
    assert agent_mc.q_table.shape == (19200, 9)

    # 4. SarsaAgent (from committed checkpoint)
    agent_sarsa = load_agent(
        {
            "name": "test_sarsa",
            "kind": "tabular_sarsa",
            "checkpoint": "results/tabular/sarsa_final.npz",
        }
    )
    assert isinstance(agent_sarsa, SarsaAgent)
    assert agent_sarsa.name == "test_sarsa"
    assert agent_sarsa.q_table.shape == (19200, 9)

    # 5. MCTSAgent (rollout evaluator)
    agent_mcts = load_agent(
        {
            "name": "test_mcts",
            "kind": "mcts",
            "determinizations": 1,
            "simulations": 2,
            "seed": 42,
        }
    )
    assert isinstance(agent_mcts, MCTSAgent)
    assert agent_mcts.name == "test_mcts"


def test_run_tournament_smoke():
    """Verify run_tournament(config, games=2) executes round-robin matches and returns win rates and matrices."""
    config = {
        "seed": 42,
        "games_per_pair": 2,
        "agents": [
            {"name": "random_a", "kind": "random", "seed": 10},
            {"name": "random_b", "kind": "random", "seed": 20},
        ],
    }
    results = run_tournament(config, games=2)

    assert isinstance(results, dict)
    assert "matches" in results
    assert "win_rates" in results
    assert "head_to_head" in results
    assert "pairwise" in results

    matches_df = results["matches"]
    assert isinstance(matches_df, pd.DataFrame)
    assert len(matches_df) == 2

    win_rates_df = results["win_rates"]
    assert isinstance(win_rates_df, pd.DataFrame)
    assert len(win_rates_df) == 2
    for col in [
        "agent",
        "wins",
        "losses",
        "draws",
        "total_games",
        "effective_win_rate",
        "ci_lower",
        "ci_upper",
    ]:
        assert col in win_rates_df.columns

    h2h_df = results["head_to_head"]
    assert isinstance(h2h_df, pd.DataFrame)
    assert h2h_df.shape == (2, 2)
    assert list(h2h_df.index) == ["random_a", "random_b"]

    pairwise_df = results["pairwise"]
    assert isinstance(pairwise_df, pd.DataFrame)
    assert len(pairwise_df) == 1
    assert "win_rate_a" in pairwise_df.columns
