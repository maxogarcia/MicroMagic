"""A live game between any mix of humans and bots, on the same engine the agents train on.

The session owns the GameState and decides what each seat may see: a seat's view never
contains the opponent's hand or either library, so a browser cannot peek. Humans submit
primitive actions (validated against the same legal-action mask the agents use); bots move
one action per bot_step() call, which lets the client animate their moves.
"""

import secrets
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from micromagic.env.replay import GameRecorder, card_catalog, choose_action, describe_action, snapshot
from micromagic.env.spec import (
    ATTACK,
    BLOCK,
    DECK_NAMES,
    PASS,
    PLAY_HAND,
    TARGET_MY_CREATURE,
    TARGET_OPP_CREATURE,
    TARGET_OPPONENT,
)
from micromagic.env.state import GameState
from micromagic.play.bots import BotInfo, make_bot


@dataclass
class Seat:
    kind: str                      # "human" or "bot"
    name: str                      # shown in the UI, e.g. "You", "MCTS"
    token: str | None = None       # humans prove their seat with it
    bot: BotInfo | None = None
    agent: object = None


@dataclass
class GameSession:
    id: str
    seats: list[Seat]
    decks: tuple[str, str]
    seed: int
    first_player: int
    state: GameState = field(init=False)
    recorder: GameRecorder = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock)
    created: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.state = GameState(seed=self.seed, decks=self.decks, first_player=self.first_player)
        self.recorder = GameRecorder(self.state, self.decks, tuple(s.name for s in self.seats), self.seed,
                                     self.first_player)

    # Who may see and do what

    @property
    def humans(self) -> list[int]:
        return [i for i, s in enumerate(self.seats) if s.kind == "human"]

    def check_token(self, seat: int, token: str | None) -> bool:
        return 0 <= seat < 2 and self.seats[seat].kind == "human" and token is not None and \
            secrets.compare_digest(self.seats[seat].token or "", token)

    def reveal_allowed(self, viewer: int | None) -> bool:
        """Hands may be shown face up only when no human opponent would be exposed."""
        others = [h for h in self.humans if h != viewer]
        return not others

    def waiting_for(self) -> str:
        if self.state.is_terminal():
            return "nobody"
        return self.seats[self.state.current_player()].kind

    # Views

    def view(self, viewer: int | None, reveal: bool = False) -> dict:
        """The game as `viewer` (a seat, or None for a spectator) is allowed to see it."""
        with self.lock:
            state = self.state
            # Bot vs bot is shown face up; otherwise a human must opt in, and only when
            # no other human's hand would be exposed.
            show_all = self.reveal_allowed(viewer) and (reveal or not self.humans or state.is_terminal())
            snap = snapshot(state)
            for p, player in enumerate(snap["players"]):
                if not (show_all or p == viewer):
                    player["hand"] = None
                player["hand_count"] = int(state.hand_count[p])
            to_move = None if state.is_terminal() else state.current_player()
            legal = []
            if viewer is not None and to_move == viewer and self.seats[viewer].kind == "human":
                legal = self._annotate_legal(viewer)
            return {
                "id": self.id,
                "viewer": viewer,
                "revealed": show_all,
                "seats": [{"kind": s.kind, "name": s.name, "deck": d} for s, d in zip(self.seats, self.decks)],
                "players": [d.title() for d in self.decks],
                "first_player": self.first_player,
                "seed": self.seed,
                "to_move": to_move,
                "waiting_for": self.waiting_for(),
                "result": self.recorder.result(state),
                "state": snap,
                "legal": legal,
                "log": [self._log_entry(f, i) for i, f in enumerate(self.recorder.frames)],
                "frame": len(self.recorder.frames) - 1,
            }

    def _log_entry(self, frame: dict, index: int) -> dict:
        return {"i": index, "actor": frame["actor"], "turn": frame["turn"], "text": frame["text"],
                "events": frame["events"]}

    def _annotate_legal(self, seat: int) -> list[dict]:
        """Each legal action with what the UI needs to make the right thing clickable."""
        state = self.state
        out = []
        for action in np.flatnonzero(state.legal_actions()):
            action = int(action)
            entry = {"action": action, "label": describe_action(state, action)}
            if action == PASS:
                entry.update(kind="pass", button=_pass_button(state))
            elif action in PLAY_HAND:
                entry.update(kind="play", hand_slot=action - PLAY_HAND.start)
            elif action in ATTACK:
                entry.update(kind="attack", slot=action - ATTACK.start)
            elif action in BLOCK:
                entry.update(kind="block", attacker_slot=action - BLOCK.start,
                             blocker_slot=int(state.blocker_queue[state.blocker_cursor]))
            elif action == TARGET_OPPONENT:
                entry.update(kind="target", owner=1 - seat, slot=None)
            elif action in TARGET_OPP_CREATURE:
                entry.update(kind="target", owner=1 - seat, slot=action - TARGET_OPP_CREATURE.start)
            elif action in TARGET_MY_CREATURE:
                entry.update(kind="target", owner=seat, slot=action - TARGET_MY_CREATURE.start)
            out.append(entry)
        return out

    # Moves

    def act(self, seat: int, token: str | None, action: int) -> None:
        with self.lock:
            if not self.check_token(seat, token):
                raise PermissionError("This seat belongs to someone else")
            if self.state.is_terminal():
                raise ValueError("The game is over")
            if self.state.current_player() != seat:
                raise ValueError("It isn't your decision")
            if not self.state.is_legal(int(action)):
                raise ValueError(f"Action {action} isn't legal right now")
            self.recorder.step(self.state, int(action))

    def bot_step(self) -> bool:
        """Play one action for the bot whose decision it is. Returns False if it's not a bot's turn."""
        with self.lock:
            if self.state.is_terminal():
                return False
            seat = self.seats[self.state.current_player()]
            if seat.kind != "bot":
                return False
            player = self.state.current_player()
            action = choose_action(seat.agent, self.state, player)
            if not self.state.is_legal(action):
                raise RuntimeError(f"{seat.name} chose illegal action {action}")
            self.recorder.step(self.state, action)
            return True

    def replay(self) -> dict:
        with self.lock:
            return self.recorder.to_replay(self.state)


def new_session(
    seat_specs: list[dict],
    decks: tuple[str, str] = DECK_NAMES,
    seed: int | None = None,
    first_player: int = 0,
) -> GameSession:
    """seat_specs: two dicts, {"kind": "human", "name": ...} or {"kind": "bot", "bot": "<bot id>"}."""
    if len(seat_specs) != 2:
        raise ValueError("A game needs exactly two seats")
    for deck in decks:
        if deck not in DECK_NAMES:
            raise ValueError(f"Unknown deck {deck!r}; choose from {DECK_NAMES}")
    if first_player not in (0, 1):
        raise ValueError("first_player must be 0 or 1")
    seed = int(seed) if seed is not None else secrets.randbelow(2**31)
    seats = []
    for i, spec in enumerate(seat_specs):
        if spec.get("kind") == "human":
            seats.append(Seat(kind="human", name=str(spec.get("name") or f"Player {i + 1}")[:24],
                              token=secrets.token_urlsafe(16)))
        elif spec.get("kind") == "bot":
            info, agent = make_bot(str(spec.get("bot")), seed=seed + 1000 * (i + 1))
            seats.append(Seat(kind="bot", name=info.label, bot=info, agent=agent))
        else:
            raise ValueError(f"Seat {i}: kind must be 'human' or 'bot'")
    return GameSession(id=secrets.token_urlsafe(8), seats=seats, decks=tuple(decks), seed=seed,
                       first_player=first_player)


__all__ = ["GameSession", "Seat", "new_session", "card_catalog"]


def _pass_button(state) -> str:
    """Imperative wording for the pass button (the log uses third person: "Skips combat")."""
    text = describe_action(state, PASS)
    suffix = " doesn't block"
    if text.endswith(suffix):
        return "Don't block with " + text[: -len(suffix)]
    return {
        "Moves to combat": "Go to combat",
        "Skips combat": "Skip combat",
        "Done attacking": "Done attacking",
        "Declines to attack": "Don't attack",
        "Ends turn": "End turn",
    }.get(text, text)
