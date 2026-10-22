"""Data-driven card pool: cards and decks are loaded from data/cards.json."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from micromagic.env.spec import CARD_FEATURES, CARD_IDS, NUM_CARDS, SCALE_STAT, CardType


CARDS_PATH = Path(__file__).resolve().parent.parent / "data" / "cards.json"


@dataclass(frozen=True, slots=True)
class Card:
    id: int
    name: str
    type: CardType
    cost: int
    power: int = 0
    toughness: int = 0


def load_card_data(path: Path = CARDS_PATH) -> tuple[dict[int, Card], dict[str, np.ndarray]]:
    """Return (cards by id, decks as arrays of card ids)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Card data not found at {path}") from err

    cards: dict[int, Card] = {}
    for entry in raw["cards"]:
        card = Card(
            id=int(entry["id"]),
            name=entry["name"],
            type=CardType[entry["type"].upper()],
            cost=int(entry.get("cost", 0)),
            power=int(entry.get("power", 0)),
            toughness=int(entry.get("toughness", 0)),
        )
        if not 0 <= card.id < NUM_CARDS:
            raise ValueError(f"Card id {card.id} for {card.name!r} is outside 0..{NUM_CARDS - 1}")
        # The card-id map in spec.py is the source of truth for ids.
        if CARD_IDS.get(card.name, card.id) != card.id:
            raise ValueError(f"{card.name!r} has id {card.id} but spec.CARD_IDS says {CARD_IDS[card.name]}")
        cards[card.id] = card

    ids_by_name = {card.name: card.id for card in cards.values()}
    decks: dict[str, np.ndarray] = {}
    for deck_name, counts in raw["decks"].items():
        try:
            ids = [ids_by_name[name] for name, n in counts.items() for _ in range(n)]
        except KeyError as err:
            raise ValueError(f"Deck {deck_name!r} lists unknown card {err.args[0]!r}") from err
        decks[deck_name] = np.array(ids, dtype=np.int16)
    return cards, decks


def build_feature_table(cards: dict[int, Card]) -> np.ndarray:
    """Per-card observation features, shape (NUM_CARDS, CARD_FEATURES)."""
    table = np.zeros((NUM_CARDS, CARD_FEATURES), dtype=np.float32)
    for card in cards.values():
        row = table[card.id]
        row[card.id] = 1.0
        row[NUM_CARDS:] = (
            card.cost / SCALE_STAT,
            card.power / SCALE_STAT,
            card.toughness / SCALE_STAT,
            card.type == CardType.LAND,
            card.type == CardType.CREATURE,
            card.type == CardType.SPELL,
        )
    return table


CARDS, DECKS = load_card_data()
FEATURE_TABLE = build_feature_table(CARDS)
# Flat lookups so the engine's hot path avoids attribute access on Card objects.
COST = np.array([CARDS[i].cost if i in CARDS else 0 for i in range(NUM_CARDS)], dtype=np.int16)
POWER = np.array([CARDS[i].power if i in CARDS else 0 for i in range(NUM_CARDS)], dtype=np.int16)
TOUGHNESS = np.array([CARDS[i].toughness if i in CARDS else 0 for i in range(NUM_CARDS)], dtype=np.int16)
TYPE = np.array([CARDS[i].type if i in CARDS else -1 for i in range(NUM_CARDS)], dtype=np.int8)
