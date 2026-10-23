"""Frozen interface constants for MicroMagic.

Max's networks and Aditya's harness index into these values directly, so
changing anything in this file requires a PR reviewed by all three of us.
Every action index and observation offset is relative to the player acting
or observing ("my" = that player, "opp" = the other one).
"""

from enum import IntEnum


NUM_PLAYERS = 2
STARTING_LIFE = 20
OPENING_HAND = 7
# Hand is capped at 7: a player holding 7 cards skips their draw.
HAND_SIZE = 7
CREATURE_SLOTS = 5
DECK_SIZE = 30
# Card ids 0..NUM_CARDS-1. Fixed at 25 so OBS_DIM never changes as the
# final card pool is filled in.
NUM_CARDS = 25
# A game still running after this many turns ends as a draw (truncation).
MAX_TURNS = 50
EMPTY = -1


# Action space: 29 discrete actions.
PASS = 0                             # end phase / done attacking / don't block / end turn
PLAY_HAND = range(1, 8)              # play the card in my hand slot 0-6
ATTACK = range(8, 13)                # declare my creature slot 0-4 as an attacker
BLOCK = range(13, 18)                # current blocker blocks the attacker in opp slot 0-4
TARGET_OPPONENT = 18                 # target the opponent player
TARGET_OPP_CREATURE = range(19, 24)  # target opp creature slot 0-4
TARGET_MY_CREATURE = range(24, 29)   # target my creature slot 0-4
NUM_ACTIONS = 29


class Phase(IntEnum):
    MAIN1 = 0
    DECLARE_ATTACKERS = 1
    DECLARE_BLOCKERS = 2
    MAIN2 = 3
    CHOOSE_TARGET = 4


NUM_PHASES = len(Phase)


class CardType(IntEnum):
    LAND = 0
    CREATURE = 1
    SPELL = 2


# Card-id map. Ids 0-4 are the toy-mode cards used by the stub; ids 5-24 are
# reserved for the final 25-card pool and are filled in by data/cards.json.
CARD_IDS: dict[str, int] = {
    "Basic Land": 0,
    "Toy Creature 1/1": 1,
    "Toy Creature 2/2": 2,
    "Toy Creature 3/3": 3,
    "Toy Creature 4/4": 4,
}


# Observation layout (float32, every value scaled into [0, 1]).
GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "my_life",
    "opp_life",
    "turn",
    *(f"phase_{p.name.lower()}" for p in Phase),
    "is_my_turn",
    "my_lands_total",
    "my_lands_untapped",
    "land_played_this_turn",
    "opp_lands_total",
    "opp_lands_untapped",
    "my_hand_count",
    "opp_hand_count",
    "my_library_count",
    "opp_library_count",
)
GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)

# Per card: one-hot id, then cost, power, toughness, is_land, is_creature, is_spell.
CARD_FEATURES = NUM_CARDS + 6
HAND_SLOT_FEATURES = 1 + CARD_FEATURES
CREATURE_STATUS_NAMES: tuple[str, ...] = (
    "damage",
    "tapped",
    "summoning_sick",
    "attacking",
    "blocking",
    "blocked",
)
CREATURE_SLOT_FEATURES = 1 + CARD_FEATURES + len(CREATURE_STATUS_NAMES)

OBS_GLOBAL = slice(0, GLOBAL_FEATURES)
OBS_HAND = slice(OBS_GLOBAL.stop, OBS_GLOBAL.stop + HAND_SIZE * HAND_SLOT_FEATURES)
OBS_MY_CREATURES = slice(OBS_HAND.stop, OBS_HAND.stop + CREATURE_SLOTS * CREATURE_SLOT_FEATURES)
OBS_OPP_CREATURES = slice(
    OBS_MY_CREATURES.stop, OBS_MY_CREATURES.stop + CREATURE_SLOTS * CREATURE_SLOT_FEATURES
)
OBS_DIM = OBS_OPP_CREATURES.stop  # 622 = 18 + 7*32 + 2*5*38

# Divisors used to scale raw counts into [0, 1].
SCALE_LIFE = float(STARTING_LIFE)
SCALE_STAT = 10.0
SCALE_ZONE = float(DECK_SIZE)
