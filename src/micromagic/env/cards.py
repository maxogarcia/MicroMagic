"""Data-driven card pool: cards and decks are loaded from data/cards.json."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from micromagic.env.spec import (
    CARD_FEATURES,
    CARD_IDS,
    DECK_SIZE,
    NUM_CARDS,
    NUM_KEYWORDS,
    SCALE_STAT,
    CardType,
    Keyword,
    SpellTemplate,
    TargetScope,
)


CARDS_PATH = Path(__file__).resolve().parent.parent / "data" / "cards.json"


@dataclass(frozen=True, slots=True)
class Card:
    id: int
    name: str
    type: CardType
    cost: int = 0
    power: int = 0
    toughness: int = 0
    keywords: frozenset[Keyword] = field(default_factory=frozenset)
    template: SpellTemplate | None = None
    amount: int = 0
    target: TargetScope = TargetScope.NONE


def _parse_card(entry: dict) -> Card:
    card_type = CardType[entry["type"].upper()]
    template = SpellTemplate[entry["template"].upper()] if "template" in entry else None
    if (card_type == CardType.SPELL) != (template is not None):
        raise ValueError(f"{entry['name']!r}: spells need a template and only spells may have one")
    return Card(
        id=int(entry["id"]),
        name=entry["name"],
        type=card_type,
        cost=int(entry.get("cost", 0)),
        power=int(entry.get("power", 0)),
        toughness=int(entry.get("toughness", 0)),
        keywords=frozenset(Keyword[k.upper()] for k in entry.get("keywords", [])),
        template=template,
        amount=int(entry.get("amount", 0)),
        target=TargetScope[entry.get("target", "none").upper()],
    )


def load_card_data(path: Path = CARDS_PATH) -> tuple[dict[int, Card], dict[str, np.ndarray]]:
    """Return (cards by id, decks as arrays of card ids)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Card data not found at {path}") from err

    cards: dict[int, Card] = {}
    for entry in raw["cards"]:
        card = _parse_card(entry)
        if not 0 <= card.id < NUM_CARDS:
            raise ValueError(f"Card id {card.id} for {card.name!r} is outside 0..{NUM_CARDS - 1}")
        # The card-id map in spec.py is the source of truth for ids.
        if CARD_IDS.get(card.name) != card.id:
            raise ValueError(f"{card.name!r} has id {card.id} but spec.CARD_IDS says {CARD_IDS.get(card.name)}")
        cards[card.id] = card

    decks: dict[str, np.ndarray] = {}
    for deck_name, counts in raw["decks"].items():
        try:
            ids = [CARD_IDS[name] for name, n in counts.items() for _ in range(n)]
        except KeyError as err:
            raise ValueError(f"Deck {deck_name!r} lists unknown card {err.args[0]!r}") from err
        if len(ids) != DECK_SIZE:
            raise ValueError(f"Deck {deck_name!r} has {len(ids)} cards, expected {DECK_SIZE}")
        decks[deck_name] = np.array(ids, dtype=np.int16)
    return cards, decks


def build_feature_table(cards: dict[int, Card]) -> np.ndarray:
    """Per-card observation features, shape (NUM_CARDS, CARD_FEATURES)."""
    table = np.zeros((NUM_CARDS, CARD_FEATURES), dtype=np.float32)
    keyword_start = NUM_CARDS + 6
    template_start = keyword_start + NUM_KEYWORDS
    for card in cards.values():
        row = table[card.id]
        row[card.id] = 1.0
        row[NUM_CARDS : NUM_CARDS + 6] = (
            card.cost / SCALE_STAT,
            card.power / SCALE_STAT,
            card.toughness / SCALE_STAT,
            card.type == CardType.LAND,
            card.type == CardType.CREATURE,
            card.type == CardType.SPELL,
        )
        for keyword in card.keywords:
            row[keyword_start + keyword] = 1.0
        if card.template is not None:
            row[template_start + card.template] = 1.0
        row[-1] = card.amount / SCALE_STAT
    return table


CARDS, DECKS = load_card_data()
FEATURE_TABLE = build_feature_table(CARDS)


def _lookup(attr: str, dtype: type, missing: int = 0) -> np.ndarray:
    return np.array([getattr(CARDS[i], attr) if i in CARDS else missing for i in range(NUM_CARDS)], dtype=dtype)


# Flat lookups so the engine's hot path avoids attribute access on Card objects.
COST = _lookup("cost", np.int16)
POWER = _lookup("power", np.int16)
TOUGHNESS = _lookup("toughness", np.int16)
TYPE = _lookup("type", np.int8, missing=-1)
AMOUNT = _lookup("amount", np.int16)
TARGET = _lookup("target", np.int8)
TEMPLATE = np.array(
    [CARDS[i].template if i in CARDS and CARDS[i].template is not None else -1 for i in range(NUM_CARDS)],
    dtype=np.int8,
)
# HAS_KEYWORD[card_id, keyword] -> bool
HAS_KEYWORD = np.zeros((NUM_CARDS, NUM_KEYWORDS), dtype=bool)
for _card in CARDS.values():
    for _keyword in _card.keywords:
        HAS_KEYWORD[_card.id, _keyword] = True
