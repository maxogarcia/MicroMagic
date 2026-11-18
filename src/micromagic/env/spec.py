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


class Keyword(IntEnum):
    FLYING = 0
    REACH = 1
    HASTE = 2
    TRAMPLE = 3
    VIGILANCE = 4
    DEATHTOUCH = 5
    LIFELINK = 6
    CANT_BLOCK = 7


NUM_KEYWORDS = len(Keyword)


class SpellTemplate(IntEnum):
    BURN = 0       # deal `amount` damage to a target
    DESTROY = 1    # destroy target creature
    PUMP = 2       # target creature gets +amount/+amount until end of turn
    DRAW = 3       # caster draws `amount` cards
    GAIN_LIFE = 4  # caster gains `amount` life


NUM_TEMPLATES = len(SpellTemplate)


class TargetScope(IntEnum):
    NONE = 0      # untargeted (draw, gain life)
    ANY = 1       # the opponent or any creature
    PLAYER = 2    # the opponent only
    CREATURE = 3  # any creature


# Card-id map: the 25-card pool. data/cards.json must match these ids.
CARD_IDS: dict[str, int] = {
    "Mountain": 0,
    "Forest": 1,
    "Raging Goblin": 2,
    "Suntail Hawk": 3,
    "Goblin Piker": 4,
    "Wind Drake": 5,
    "Gray Ogre": 6,
    "Hill Giant": 7,
    "Charging Monstrosaur": 8,
    "Shock": 9,
    "Volcanic Hammer": 10,
    "Lava Axe": 11,
    "Giant Growth": 12,
    "Typhoid Rats": 13,
    "Healer's Hawk": 14,
    "Grizzly Bears": 15,
    "Deadly Recluse": 16,
    "Child of Night": 17,
    "Trained Armodon": 18,
    "Giant Spider": 19,
    "Eviscerate": 20,
    "Divination": 21,
    "Serra Angel": 22,
    "Natural Spring": 23,
    "Colossal Dreadmaw": 24,
}
DECK_NAMES: tuple[str, ...] = ("aggro", "midrange")


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

# Per card: one-hot id (25), cost, power, toughness, is_land, is_creature,
# is_spell, keyword flags (8), spell-template one-hot (5), spell amount.
CARD_FEATURES = NUM_CARDS + 6 + NUM_KEYWORDS + NUM_TEMPLATES + 1
HAND_SLOT_FEATURES = 1 + CARD_FEATURES
CREATURE_STATUS_NAMES: tuple[str, ...] = (
    "damage",
    "tapped",
    "summoning_sick",
    "attacking",
    "blocking",
    "blocked",
    "current_power",
    "current_toughness",
)
CREATURE_SLOT_FEATURES = 1 + CARD_FEATURES + len(CREATURE_STATUS_NAMES)

OBS_GLOBAL = slice(0, GLOBAL_FEATURES)
# The spell waiting for a target (all zeros outside CHOOSE_TARGET).
OBS_PENDING_SPELL = slice(OBS_GLOBAL.stop, OBS_GLOBAL.stop + CARD_FEATURES)
OBS_HAND = slice(OBS_PENDING_SPELL.stop, OBS_PENDING_SPELL.stop + HAND_SIZE * HAND_SLOT_FEATURES)
OBS_MY_CREATURES = slice(OBS_HAND.stop, OBS_HAND.stop + CREATURE_SLOTS * CREATURE_SLOT_FEATURES)
OBS_OPP_CREATURES = slice(
    OBS_MY_CREATURES.stop, OBS_MY_CREATURES.stop + CREATURE_SLOTS * CREATURE_SLOT_FEATURES
)
OBS_DIM = OBS_OPP_CREATURES.stop  # 925 = 18 + 45 + 7*46 + 2*5*54

# Divisors used to scale raw counts into [0, 1].
SCALE_LIFE = float(STARTING_LIFE)
SCALE_STAT = 10.0
SCALE_ZONE = float(DECK_SIZE)
