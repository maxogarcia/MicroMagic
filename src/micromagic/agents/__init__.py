"""Agents: the Agent base class, baselines, and wrappers around learned policies.

MaskablePPOAgent is imported from micromagic.agents.ppo directly so that the
baselines do not require torch / sb3-contrib.
"""

from micromagic.agents.base import Agent
from micromagic.agents.random_agent import RandomAgent
from micromagic.agents.tabular import MonteCarloAgent, SarsaAgent, TabularAgent

__all__ = [
    "Agent",
    "MonteCarloAgent",
    "RandomAgent",
    "SarsaAgent",
    "TabularAgent",
]
