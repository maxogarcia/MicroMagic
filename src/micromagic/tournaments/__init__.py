"""Tournament evaluation harnesses, round-robin, Swiss, and Elo rating systems."""

from micromagic.tournaments.protocols import AgentProtocol, GameStateProtocol
from micromagic.tournaments.runner import (
    MatchResult,
    TournamentResult,
    TournamentRunner,
)

__all__ = [
    "AgentProtocol",
    "GameStateProtocol",
    "MatchResult",
    "TournamentResult",
    "TournamentRunner",
]
