"""Which agents can sit at the table, and how to build them.

Built-in bots always work. Learned agents appear only when their checkpoint is on disk and
their libraries are installed: PPO/DQN under runs/{ppo,dqn}/<run>/model.zip (needs the `rl`
extra), tabular MC/SARSA Q-tables under results/tabular/*_final.npz.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BotInfo:
    id: str
    label: str
    description: str


BUILT_IN = (
    BotInfo("random", "Random", "Picks uniformly among legal actions."),
    BotInfo("heuristic", "Heuristic", "Max's rule-based baseline: curves out, attacks and blocks when favorable."),
    BotInfo("mcts", "MCTS", "Determinized tree search: 4 sampled worlds x 64 simulations per decision (about 0.1 s)."),
    BotInfo("mcts_quick", "MCTS (quick)", "Lighter search: 2 worlds x 24 simulations, for fast demos."),
)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def available_bots(root: Path | None = None) -> list[BotInfo]:
    root = Path(root or Path.cwd())
    bots = list(BUILT_IN)
    if _has("sb3_contrib"):
        for algo, label in (("ppo", "PPO"), ("dqn", "DQN")):
            for model in sorted((root / "runs" / algo).glob("*/model.zip")):
                bots.append(BotInfo(f"{algo}:{model.parent.name}", f"{label} ({model.parent.name})",
                                    f"Trained {label} checkpoint at {model.relative_to(root).as_posix()}."))
    if _has("micromagic.agents.tabular"):
        for algo, label in (("mc", "First-Visit MC"), ("sarsa", "4-step SARSA")):
            table = root / "results" / "tabular" / f"{algo}_final.npz"
            if table.exists():
                bots.append(BotInfo(f"{algo}:final", label, f"Tabular Q-table at {table.relative_to(root).as_posix()}."))
    return bots


def make_bot(bot_id: str, seed: int = 0, root: Path | None = None) -> tuple[BotInfo, object]:
    """Build a listed bot. Only ids from available_bots() are accepted (no arbitrary file paths)."""
    root = Path(root or Path.cwd())
    info = next((b for b in available_bots(root) if b.id == bot_id), None)
    if info is None:
        raise ValueError(f"Unknown or unavailable bot {bot_id!r}")

    if bot_id == "random":
        from micromagic.agents import RandomAgent
        return info, RandomAgent(seed=seed, name=info.label)
    if bot_id == "heuristic":
        from micromagic.agents.heuristic import HeuristicAgent
        return info, HeuristicAgent(name=info.label)
    if bot_id in ("mcts", "mcts_quick"):
        from micromagic.search import MCTSAgent
        worlds, sims = (4, 64) if bot_id == "mcts" else (2, 24)
        return info, MCTSAgent(determinizations=worlds, simulations=sims, seed=seed, name=info.label)

    algo, run = bot_id.split(":", 1)
    if algo == "ppo":
        from micromagic.agents.ppo import MaskablePPOAgent
        return info, MaskablePPOAgent.from_checkpoint(root / "runs" / "ppo" / run / "model.zip", name=info.label)
    if algo == "dqn":
        from micromagic.agents.dqn import MaskableDQNAgent
        return info, MaskableDQNAgent.from_checkpoint(root / "runs" / "dqn" / run / "model.zip", name=info.label)
    if algo in ("mc", "sarsa"):
        from micromagic.agents import tabular
        cls = tabular.MonteCarloAgent if algo == "mc" else tabular.SarsaAgent
        agent = cls.load(root / "results" / "tabular" / f"{algo}_final.npz")
        agent.deterministic = True
        agent.name = info.label
        return info, agent
    raise ValueError(f"Unknown bot {bot_id!r}")
