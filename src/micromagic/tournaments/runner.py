"""Tournament runner and match recording infrastructure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from micromagic.tournaments.protocols import AgentProtocol, GameStateProtocol


@dataclass(frozen=True)
class MatchResult:
    """Immutable record of an individual tournament game."""

    game_id: int
    seed: int
    player_0_name: str
    player_1_name: str
    winner_seat: int | None
    winner_name: str | None
    step_count: int
    is_draw: bool


@dataclass
class TournamentResult:
    """Aggregate result of a series of tournament matches."""

    matches: list[MatchResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert match records into a pandas DataFrame."""
        return pd.DataFrame([asdict(match) for match in self.matches])

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics for the tournament."""
        total_games = len(self.matches)
        if total_games == 0:
            return {
                "total_games": 0,
                "wins": {},
                "win_rates": {},
                "draws": 0,
                "draw_rate": 0.0,
                "seat_wins": {0: 0, 1: 0},
                "avg_steps": 0.0,
            }

        wins: dict[str, int] = {}
        draw_count = 0
        seat_wins: dict[int, int] = {0: 0, 1: 0}
        total_steps = 0

        for match in self.matches:
            total_steps += match.step_count
            if match.is_draw or match.winner_name is None:
                draw_count += 1
            else:
                wins[match.winner_name] = wins.get(match.winner_name, 0) + 1
                if match.winner_seat in seat_wins:
                    seat_wins[match.winner_seat] += 1

        win_rates = {
            agent: win_count / total_games for agent, win_count in wins.items()
        }

        return {
            "total_games": total_games,
            "wins": wins,
            "win_rates": win_rates,
            "draws": draw_count,
            "draw_rate": draw_count / total_games,
            "seat_wins": seat_wins,
            "avg_steps": total_steps / total_games,
        }


class TournamentRunner:
    """Executes matches between agents with alternating seat allocation and deterministic seeding."""

    def __init__(self, env_factory: Callable[[int], GameStateProtocol]) -> None:
        """Initialize runner with an environment factory taking an integer seed."""
        self.env_factory = env_factory

    def run_match(
        self,
        agent_0: AgentProtocol,
        agent_1: AgentProtocol,
        num_games: int,
        base_seed: int = 42,
        swap_seats: bool = True,
        max_steps: int = 1000,
    ) -> TournamentResult:
        """Run N games between agent_0 and agent_1 with seat alternation and deterministic seeding.

        Args:
            agent_0: First agent.
            agent_1: Second agent.
            num_games: Number of games to execute.
            base_seed: Starting seed for pseudo-random number generator.
            swap_seats: If True, swap seat allocations on odd game indices.
            max_steps: Maximum step cutoff per game to prevent infinite loops.

        Returns:
            TournamentResult containing individual MatchResult records.

        Raises:
            ValueError: If an agent selects an illegal action according to legal_actions mask.
        """
        matches: list[MatchResult] = []

        for game_id in range(num_games):
            seed = base_seed + game_id
            if swap_seats and (game_id % 2 == 1):
                seat_0_agent = agent_1
                seat_1_agent = agent_0
            else:
                seat_0_agent = agent_0
                seat_1_agent = agent_1

            agents = {0: seat_0_agent, 1: seat_1_agent}
            env = self.env_factory(seed)
            step_count = 0

            while not env.is_terminal() and step_count < max_steps:
                curr_player = env.current_player()
                if curr_player not in agents:
                    msg = f"Unknown player seat {curr_player}; expected 0 or 1"
                    raise ValueError(msg)

                active_agent = agents[curr_player]
                obs = env.observe(curr_player)
                mask = env.legal_actions()

                action = active_agent.act(obs, mask)

                if len(mask) > 0:
                    if action < 0 or action >= len(mask) or not bool(mask[action]):
                        msg = (
                            f"Agent '{active_agent.name}' attempted illegal action {action} "
                            f"at step {step_count} (seat {curr_player})"
                        )
                        raise ValueError(msg)

                env.step(action)
                step_count += 1

            winner_seat = env.winner()
            if winner_seat is not None and winner_seat in agents:
                winner_name: str | None = agents[winner_seat].name
                is_draw = False
            else:
                winner_seat = None
                winner_name = None
                is_draw = True

            match_res = MatchResult(
                game_id=game_id,
                seed=seed,
                player_0_name=seat_0_agent.name,
                player_1_name=seat_1_agent.name,
                winner_seat=winner_seat,
                winner_name=winner_name,
                step_count=step_count,
                is_draw=is_draw,
            )
            matches.append(match_res)

        return TournamentResult(matches=matches)

    def save_results(
        self,
        results: TournamentResult,
        filepath: Path | str,
    ) -> Path:
        """Persist tournament match results to CSV.

        Args:
            results: TournamentResult to save.
            filepath: Destination CSV path.

        Returns:
            Resolved Path object of the saved file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = results.to_dataframe()
        df.to_csv(path, index=False)
        return path
