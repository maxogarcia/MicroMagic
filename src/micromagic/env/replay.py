"""Record MicroMagic games and render them as a self-contained HTML replay.

    replay = record_game(seed=7)                       # two random agents
    replay = record_game(policies=(my_agent.act, None)) # any (obs, mask) -> int policy
    write_html(replay, Path("replays/game_7.html"))

A replay stores a full snapshot of the board after every action, a plain-English
description of the action, and the events it caused (damage, deaths, life gain),
so it doubles as a game log for qualitative analysis in the report.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from micromagic.env import cards as card_db
from micromagic.env.spec import (
    ATTACK,
    BLOCK,
    CREATURE_SLOTS,
    EMPTY,
    PASS,
    PLAY_HAND,
    TARGET_MY_CREATURE,
    TARGET_OPP_CREATURE,
    TARGET_OPPONENT,
    CardType,
    Keyword,
    Phase,
    SpellTemplate,
    TargetScope,
)
from micromagic.env.state import GameState


Policy = Callable[[np.ndarray, np.ndarray], int]

VIEWER_TEMPLATE = Path(__file__).resolve().parent.parent / "data" / "replay_viewer.html"
DATA_PLACEHOLDER = "/*__REPLAY_DATA__*/null"
PHASE_LABELS = {
    Phase.MAIN1: "Main 1",
    Phase.DECLARE_ATTACKERS: "Attack",
    Phase.DECLARE_BLOCKERS: "Block",
    Phase.MAIN2: "Main 2",
    Phase.CHOOSE_TARGET: "Target",
}


def rules_text(card: card_db.Card) -> str:
    """One line of rules text, in the style printed on a card."""
    if card.type == CardType.LAND:
        return "Adds one mana"
    if card.type == CardType.CREATURE:
        words = {"cant_block": "Can't block"}
        names = [words.get(k.name.lower(), k.name.lower().replace("_", " ")) for k in sorted(card.keywords)]
        return ", ".join(names).capitalize()
    n = card.amount
    target = "target opponent" if card.target == TargetScope.PLAYER else "any target"
    return {
        SpellTemplate.BURN: f"Deal {n} damage to {target}",
        SpellTemplate.DESTROY: "Destroy target creature",
        SpellTemplate.PUMP: f"Target creature gets +{n}/+{n} until end of turn",
        SpellTemplate.DRAW: f"Draw {n} cards",
        SpellTemplate.GAIN_LIFE: f"Gain {n} life",
    }[card.template]


def card_catalog() -> list[dict]:
    """Static card data the viewer needs, indexed by card id."""
    deck_of = {}
    for deck_name, deck in card_db.DECKS.items():
        for card_id in deck:
            deck_of.setdefault(int(card_id), deck_name)
    catalog = []
    for card_id in range(len(card_db.CARDS)):
        card = card_db.CARDS[card_id]
        catalog.append(
            {
                "deck": deck_of.get(card_id),
                "rules": rules_text(card),
                "id": card.id,
                "name": card.name,
                "type": card.type.name.lower(),
                "cost": card.cost,
                "power": card.power,
                "toughness": card.toughness,
                "keywords": sorted(k.name.lower() for k in card.keywords),
                "template": card.template.name.lower() if card.template is not None else None,
                "amount": card.amount,
                "target": card.target.name.lower(),
            }
        )
    return catalog


def snapshot(state: GameState) -> dict:
    """Everything on the table, including hidden zones (the viewer can hide them)."""
    players = []
    for p in (0, 1):
        creatures = []
        for slot in range(CREATURE_SLOTS):
            card_id = int(state.creature[p, slot])
            if card_id == EMPTY:
                creatures.append(None)
                continue
            creatures.append(
                {
                    "card": card_id,
                    "power": state.power(p, slot),
                    "toughness": state.toughness(p, slot),
                    "damage": int(state.damage[p, slot]),
                    "tapped": bool(state.tapped[p, slot]),
                    "sick": bool(state.sick[p, slot]),
                    "attacking": bool(state.attacking[p, slot]),
                    "blocking": int(state.blocking[p, slot]),
                    "pumped": int(state.pump[p, slot]),
                }
            )
        players.append(
            {
                "life": int(state.life[p]),
                "hand": [int(c) for c in state.hand[p, : state.hand_count[p]]],
                "library": state._library_count(p),
                "lands": int(state.lands_total[p]),
                "lands_untapped": int(state.lands_total[p] - state.lands_tapped[p]),
                "creatures": creatures,
            }
        )
    return {
        "turn": state.turn,
        "active": state.active,
        "phase": PHASE_LABELS[state.phase],
        "pending": int(state.pending_card) if state.pending_card != EMPTY else None,
        "blocker": (
            int(state.blocker_queue[state.blocker_cursor])
            if state.phase == Phase.DECLARE_BLOCKERS and not state.is_terminal()
            else None
        ),
        "players": players,
        "terminal": state.is_terminal(),
        "winner": state.winner(),
        "truncated": state.is_truncated(),
    }


def _name(card_id: int) -> str:
    return card_db.CARDS[int(card_id)].name


def describe_action(state: GameState, action: int) -> str:
    """A sentence describing `action`, read from the state *before* it is taken."""
    actor = state.current_player()
    me, opp = state.active, 1 - state.active

    if state.phase == Phase.CHOOSE_TARGET:
        spell = _name(state.pending_card)
        if action == TARGET_OPPONENT:
            return f"{spell} targets the opponent"
        if action in TARGET_OPP_CREATURE:
            return f"{spell} targets {_name(state.creature[opp, action - TARGET_OPP_CREATURE.start])}"
        return f"{spell} targets own {_name(state.creature[me, action - TARGET_MY_CREATURE.start])}"

    if action == PASS:
        if state.phase == Phase.MAIN1:
            can_attack = any(state._can_attack(me, s) for s in range(CREATURE_SLOTS))
            return "Moves to combat" if can_attack else "Skips combat"
        if state.phase == Phase.DECLARE_ATTACKERS:
            return "Done attacking" if state.attacking[me].any() else "Declines to attack"
        if state.phase == Phase.DECLARE_BLOCKERS:
            blocker = int(state.blocker_queue[state.blocker_cursor])
            return f"{_name(state.creature[actor, blocker])} doesn't block"
        return "Ends turn"

    if action in PLAY_HAND:
        card_id = int(state.hand[actor, action - PLAY_HAND.start])
        verb = "Plays" if card_db.TYPE[card_id] == CardType.LAND else "Casts"
        return f"{verb} {_name(card_id)}"
    if action in ATTACK:
        return f"Attacks with {_name(state.creature[actor, action - ATTACK.start])}"
    if action in BLOCK:
        blocker = int(state.blocker_queue[state.blocker_cursor])
        attacker = action - BLOCK.start
        return f"{_name(state.creature[actor, blocker])} blocks {_name(state.creature[1 - actor, attacker])}"
    return f"Action {action}"


def _why_no_blocks(state: GameState, action: int) -> str | None:
    """If finishing attacks with `action` leaves the defender no legal block, say why (else None)."""
    me, opp = state.active, 1 - state.active
    if state.phase != Phase.DECLARE_ATTACKERS or action != PASS or not state.attacking[me].any():
        return None
    mine = [s for s in range(CREATURE_SLOTS) if state.creature[opp, s] != EMPTY]
    if not mine:
        return "{defender} has no creatures, so there are no blocks"
    untapped = [s for s in mine if not state.tapped[opp, s]]
    if not untapped:
        return "{defender}'s creatures are all tapped, so none can block"
    able = [s for s in untapped if not state.has_keyword(opp, s, Keyword.CANT_BLOCK)]
    if not able:
        names = sorted({_name(state.creature[opp, s]) for s in untapped})
        return " and ".join(names) + " can't block"
    attackers = np.flatnonzero(state.attacking[me])
    if all(state.has_keyword(me, int(a), Keyword.FLYING) for a in attackers) and not any(
        state.has_keyword(opp, s, Keyword.FLYING) or state.has_keyword(opp, s, Keyword.REACH) for s in able
    ):
        return "Only creatures with flying or reach can block flyers, so {defender} can't block"
    return None


def _events(before: dict, after: dict, names: tuple[str, str]) -> list[str]:
    """Life changes, creature deaths, and draws between two snapshots."""
    events = []
    for p in (0, 1):
        b, a = before["players"][p], after["players"][p]
        delta = a["life"] - b["life"]
        if delta < 0:
            events.append(f"{names[p]} takes {-delta} damage ({a['life']} life)")
        elif delta > 0:
            events.append(f"{names[p]} gains {delta} life ({a['life']} life)")
        for slot in range(CREATURE_SLOTS):
            was, now = b["creatures"][slot], a["creatures"][slot]
            if was is not None and (now is None or now["card"] != was["card"]):
                events.append(f"{names[p]}'s {_name(was['card'])} dies")
        drawn = b["library"] - a["library"]
        if drawn > 0:
            events.append(f"{names[p]} draws {drawn} card{'s' if drawn > 1 else ''}")
    if after["turn"] != before["turn"] and not after["truncated"]:
        events.append(f"Turn {after['turn']}: {names[after['active']]}")
    return events


def _random_policy(rng: np.random.Generator) -> Policy:
    return lambda obs, mask: int(rng.choice(np.flatnonzero(mask)))


class GameRecorder:
    """Records a game frame by frame as it is played; used by record_game and the play server."""

    def __init__(
        self,
        state: GameState,
        decks: tuple[str, str],
        agent_names: tuple[str, str],
        seed: int,
        first_player: int,
    ) -> None:
        self.decks = tuple(decks)
        self.names = tuple(d.title() for d in decks)
        self.agent_names = tuple(agent_names)
        self.seed = seed
        self.first_player = first_player
        self.frames = [{"actor": None, "action": None, "turn": state.turn, "phase": PHASE_LABELS[state.phase],
                        "text": "Both players draw 7 cards", "events": [], "state": snapshot(state)}]

    def step(self, state: GameState, action: int) -> dict:
        """Describe `action`, apply it to `state`, and record the resulting frame."""
        actor = state.current_player()
        text = describe_action(state, action)
        turn, phase = state.turn, PHASE_LABELS[state.phase]
        before = self.frames[-1]["state"]
        no_blocks = _why_no_blocks(state, action)
        state.step(action)
        after = snapshot(state)
        events = _events(before, after, self.names)
        if no_blocks and state.phase != Phase.DECLARE_BLOCKERS:
            events.insert(0, no_blocks.format(defender=self.names[1 - actor]))
        frame = {"actor": actor, "action": int(action), "turn": turn, "phase": phase, "text": text,
                 "events": events, "state": after}
        self.frames.append(frame)
        return frame

    def result(self, state: GameState) -> str:
        if not state.is_terminal():
            return "In progress"
        winner = state.winner()
        if winner is None:
            return "Draw (turn limit)" if state.is_truncated() else "Draw"
        return f"{self.names[winner]} wins on turn {state.turn}"

    def to_replay(self, state: GameState) -> dict:
        return {
            "version": 1,
            "seed": self.seed,
            "decks": list(self.decks),
            "players": list(self.names),
            "agents": list(self.agent_names),
            "first_player": self.first_player,
            "result": self.result(state),
            "cards": card_catalog(),
            "frames": self.frames,
        }


def choose_action(policy, state: GameState, player: int) -> int:
    """Ask a policy for an action: search agents get the state, everything else (obs, mask)."""
    if hasattr(policy, "act_from_state"):
        return int(policy.act_from_state(state, player))
    return int(policy(state.observe(player), state.legal_actions()))


def record_game(
    policies: tuple[Policy | None, Policy | None] = (None, None),
    seed: int = 0,
    decks: tuple[str, str] = ("aggro", "midrange"),
    first_player: int = 0,
    agent_names: tuple[str, str] | None = None,
) -> dict:
    """Play one game and record every decision. `None` means a random legal policy.

    A policy is any (obs, mask) -> int callable, or an agent with act_from_state (e.g. MCTSAgent).
    """
    rng = np.random.default_rng(seed + 1)
    actors = [p if p is not None else _random_policy(rng) for p in policies]
    agents = agent_names or tuple("Random" if p is None else getattr(p, "name", "Agent") for p in policies)

    state = GameState(seed=seed, decks=decks, first_player=first_player)
    recorder = GameRecorder(state, decks, agents, seed, first_player)
    while not state.is_terminal():
        recorder.step(state, choose_action(actors[state.current_player()], state, state.current_player()))
    return recorder.to_replay(state)


def render_html(replay: dict) -> str:
    """The standalone replay page for `replay`, as a string."""
    try:
        template = VIEWER_TEMPLATE.read_text(encoding="utf-8")
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Replay viewer template not found at {VIEWER_TEMPLATE}") from err
    if DATA_PLACEHOLDER not in template:
        raise ValueError(f"{VIEWER_TEMPLATE} is missing the data placeholder {DATA_PLACEHOLDER!r}")
    # "</" is escaped so card text can never close the <script> tag early.
    payload = json.dumps(replay, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(DATA_PLACEHOLDER, payload)


def write_html(replay: dict, path: Path) -> Path:
    """Write a standalone HTML replay that opens in any browser (no server needed)."""
    html = render_html(replay)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
