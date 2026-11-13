"""The 9 macro-actions and the scripted executor that turns each into primitive game actions.

A macro is chosen at a decision point (pre-combat main, post-combat main, or blocks).
MacroExecution then produces one legal primitive action per call until the macro is
finished or control passes to the other player. Rules for "best", "safe" and "value"
are documented in docs/abstraction.md and are meant to be agreed with the team.
"""

from enum import IntEnum

import numpy as np

from micromagic.abstraction.features import Context, decision_context
from micromagic.env import cards as card_db
from micromagic.env.spec import (
    ATTACK,
    BLOCK,
    CREATURE_SLOTS,
    EMPTY,
    HAND_SIZE,
    PASS,
    PLAY_HAND,
    TARGET_OPP_CREATURE,
    TARGET_OPPONENT,
    CardType,
    Keyword,
    Phase,
    SpellTemplate,
    TargetScope,
)
from micromagic.env.state import GameState


class Macro(IntEnum):
    PLAY_LAND = 0
    CAST_BEST_CREATURE = 1
    BURN_FACE = 2
    BURN_BEST_CREATURE = 3   # also uses Eviscerate (destroy)
    ATTACK_ALL = 4
    ATTACK_SAFE = 5
    NO_ATTACK = 6            # pre-combat: skip combat; post-combat: end the turn
    BLOCK_VALUE = 7
    NO_BLOCK = 8


NUM_MACROS = len(Macro)
MAIN_MACROS = (Macro.PLAY_LAND, Macro.CAST_BEST_CREATURE, Macro.BURN_FACE, Macro.BURN_BEST_CREATURE)

# Cards no macro chooses directly; ending the turn casts them when sensible (docs/abstraction.md).
NATURAL_SPRING = next(c.id for c in card_db.CARDS.values() if c.template == SpellTemplate.GAIN_LIFE)
DIVINATION = next(c.id for c in card_db.CARDS.values() if c.template == SpellTemplate.DRAW)


# Card and combat helpers

def _hand(state: GameState, player: int) -> list[int]:
    return [int(c) for c in state.hand[player, : state.hand_count[player]]]


def _affordable(state: GameState, player: int, card_id: int) -> bool:
    return card_db.COST[card_id] <= state._untapped_lands(player)


def _remaining_toughness(state: GameState, player: int, slot: int) -> int:
    return state.toughness(player, slot) - int(state.damage[player, slot])


def _kills(state: GameState, source: tuple[int, int], target: tuple[int, int]) -> bool:
    """Whether creature `source` deals lethal damage to creature `target` in a fight."""
    power = state.power(*source)
    if power <= 0:
        return False
    return state.has_keyword(*source, Keyword.DEATHTOUCH) or power >= _remaining_toughness(state, *target)


def _can_block(state: GameState, blocker: tuple[int, int], attacker: tuple[int, int]) -> bool:
    if not state.has_keyword(*attacker, Keyword.FLYING):
        return True
    return state.has_keyword(*blocker, Keyword.FLYING) or state.has_keyword(*blocker, Keyword.REACH)


def _potential_blockers(state: GameState, defender: int) -> list[int]:
    return [
        s for s in range(CREATURE_SLOTS)
        if state.creature[defender, s] != EMPTY
        and not state.tapped[defender, s]
        and not state.has_keyword(defender, s, Keyword.CANT_BLOCK)
    ]


def _eligible_attackers(state: GameState, player: int) -> list[int]:
    return [s for s in range(CREATURE_SLOTS) if state._can_attack(player, s)]


def safe_attackers(state: GameState, player: int) -> list[int]:
    """Attackers that no untapped blocker can kill without dying itself."""
    opp = 1 - player
    blockers = _potential_blockers(state, opp)
    safe = []
    for a in _eligible_attackers(state, player):
        threatened = any(
            _can_block(state, (opp, b), (player, a))
            and _kills(state, (opp, b), (player, a))
            and not _kills(state, (player, a), (opp, b))
            for b in blockers
        )
        if not threatened:
            safe.append(a)
    return safe


def best_creature(state: GameState, player: int) -> int | None:
    """Highest mana value creature in hand that can be cast now (ties: bigger body)."""
    if state.phase not in (Phase.MAIN1, Phase.MAIN2) or not (state.creature[player] == EMPTY).any():
        return None
    options = [
        c for c in _hand(state, player)
        if card_db.TYPE[c] == CardType.CREATURE and _affordable(state, player, c)
    ]
    if not options:
        return None
    return max(options, key=lambda c: (card_db.COST[c], card_db.POWER[c] + card_db.TOUGHNESS[c]))


def face_burn(state: GameState, player: int) -> int | None:
    """The biggest castable burn spell that can hit the opponent."""
    if state.phase not in (Phase.MAIN1, Phase.MAIN2):
        return None
    options = [
        c for c in _hand(state, player)
        if card_db.TEMPLATE[c] == SpellTemplate.BURN
        and card_db.TARGET[c] in (TargetScope.ANY, TargetScope.PLAYER)
        and _affordable(state, player, c)
    ]
    return max(options, key=lambda c: card_db.AMOUNT[c]) if options else None


def removal_plan(state: GameState, player: int) -> tuple[int, int] | None:
    """(spell, opponent slot): kill the opponent's strongest creature that a castable spell can kill.

    Among spells that kill that creature, the cheapest is used, so Eviscerate is saved
    for creatures burn cannot handle.
    """
    if state.phase not in (Phase.MAIN1, Phase.MAIN2):
        return None
    opp = 1 - player
    spells = [
        c for c in _hand(state, player)
        if _affordable(state, player, c) and (
            card_db.TEMPLATE[c] == SpellTemplate.DESTROY
            or (card_db.TEMPLATE[c] == SpellTemplate.BURN and card_db.TARGET[c] == TargetScope.ANY)
        )
    ]
    best = None
    for slot in range(CREATURE_SLOTS):
        card = state.creature[opp, slot]
        if card == EMPTY:
            continue
        killers = [
            s for s in spells
            if card_db.TEMPLATE[s] == SpellTemplate.DESTROY or card_db.AMOUNT[s] >= _remaining_toughness(state, opp, slot)
        ]
        if not killers:
            continue
        spell = min(killers, key=lambda s: (card_db.COST[s], card_db.TEMPLATE[s] == SpellTemplate.DESTROY))
        value = (state.power(opp, slot), int(card_db.COST[card]))
        if best is None or value > best[0]:
            best = (value, spell, slot)
    return (best[1], best[2]) if best else None


def block_plan(state: GameState, defender: int) -> dict[int, int]:
    """Blocker slot -> attacker slot. Value blocks first; chump blocks only to survive lethal damage."""
    attacker_player = 1 - defender
    attackers = sorted(
        (a for a in range(CREATURE_SLOTS) if state.attacking[attacker_player, a]),
        key=lambda a: -state.power(attacker_player, a),
    )
    free = set(_potential_blockers(state, defender))
    plan: dict[int, int] = {}

    for a in attackers:
        att = (attacker_player, a)
        candidates = []
        for b in free:
            blk = (defender, b)
            if not _can_block(state, blk, att):
                continue
            survives = not _kills(state, att, blk)
            kills = _kills(state, blk, att)
            even_trade = kills and card_db.COST[state.creature[att]] >= card_db.COST[state.creature[blk]]
            if survives or even_trade:
                # Prefer blocks that kill and survive, then survive, then the cheapest trade.
                candidates.append(((survives and kills), survives, -int(card_db.COST[state.creature[blk]]), b))
        if candidates:
            b = max(candidates)[-1]
            plan[b] = a
            free.discard(b)

    def unblocked_damage() -> int:
        return sum(max(state.power(attacker_player, a), 0) for a in attackers if a not in plan.values())

    for a in attackers:
        if unblocked_damage() < state.life[defender]:
            break
        if a in plan.values():
            continue
        chumps = [b for b in free if _can_block(state, (defender, b), (attacker_player, a))]
        if chumps:
            b = min(chumps, key=lambda s: card_db.COST[state.creature[defender, s]])
            plan[b] = a
            free.discard(b)
    return plan


def _hand_slot(state: GameState, player: int, card_id: int) -> int:
    return _hand(state, player).index(card_id)


# Availability

def macro_mask(state: GameState, player: int) -> np.ndarray:
    """Which of the 9 macros `player` can choose now. All False when it is not `player`'s decision."""
    mask = np.zeros(NUM_MACROS, dtype=bool)
    if state.is_terminal() or state.current_player() != player:
        return mask
    context = decision_context(state)

    if context == Context.BLOCK:
        mask[Macro.BLOCK_VALUE] = bool(block_plan(state, player))
        mask[Macro.NO_BLOCK] = True
        return mask
    if state.phase == Phase.CHOOSE_TARGET:
        return mask  # only reached inside a macro

    in_main = state.phase in (Phase.MAIN1, Phase.MAIN2)
    mask[Macro.PLAY_LAND] = in_main and not state.land_played and any(
        card_db.TYPE[c] == CardType.LAND for c in _hand(state, player)
    )
    mask[Macro.CAST_BEST_CREATURE] = best_creature(state, player) is not None
    mask[Macro.BURN_FACE] = face_burn(state, player) is not None
    mask[Macro.BURN_BEST_CREATURE] = removal_plan(state, player) is not None
    if context == Context.PRECOMBAT:
        mask[Macro.ATTACK_ALL] = bool(_eligible_attackers(state, player))
        mask[Macro.ATTACK_SAFE] = bool(safe_attackers(state, player))
    mask[Macro.NO_ATTACK] = True
    return mask


# Execution

class MacroExecution:
    """Produces the primitive actions for one macro, one per next_action() call.

    next_action() returns None once the macro is finished or the decision has passed to
    the other player; the agent then chooses a new macro at its next decision point.
    """

    def __init__(self, state: GameState, player: int, macro: Macro) -> None:
        macro = Macro(macro)
        if not macro_mask(state, player)[macro]:
            raise ValueError(f"{macro.name} is not available to player {player} in phase {state.phase.name}")
        self.player = player
        self.macro = macro
        self.start_context = decision_context(state)
        self.steps = 0
        self.done = False
        self.card: int | None = None
        self.target_slot: int | None = None
        self.attackers: set[int] = set()
        self.blocks: dict[int, int] = {}

        if macro == Macro.PLAY_LAND:
            self.card = next(c for c in _hand(state, player) if card_db.TYPE[c] == CardType.LAND)
        elif macro == Macro.CAST_BEST_CREATURE:
            self.card = best_creature(state, player)
        elif macro == Macro.BURN_FACE:
            self.card = face_burn(state, player)
        elif macro == Macro.BURN_BEST_CREATURE:
            self.card, self.target_slot = removal_plan(state, player)
        elif macro == Macro.ATTACK_ALL:
            self.attackers = set(_eligible_attackers(state, player))
        elif macro == Macro.ATTACK_SAFE:
            self.attackers = set(safe_attackers(state, player))
        elif macro == Macro.BLOCK_VALUE:
            self.blocks = block_plan(state, player)

    def next_action(self, state: GameState) -> int | None:
        if self.done or state.is_terminal() or state.current_player() != self.player:
            self.done = True
            return None
        action = self._next(state)
        if action is None:
            self.done = True
            return None
        self.steps += 1
        return action

    def _next(self, state: GameState) -> int | None:
        m, me = self.macro, self.player

        if m in (Macro.PLAY_LAND, Macro.CAST_BEST_CREATURE):
            return PLAY_HAND.start + _hand_slot(state, me, self.card) if self.steps == 0 else None

        if m in (Macro.BURN_FACE, Macro.BURN_BEST_CREATURE):
            if self.steps == 0:
                return PLAY_HAND.start + _hand_slot(state, me, self.card)
            if self.steps == 1 and state.phase == Phase.CHOOSE_TARGET:
                return TARGET_OPPONENT if m == Macro.BURN_FACE else TARGET_OPP_CREATURE.start + self.target_slot
            return None

        if m in (Macro.ATTACK_ALL, Macro.ATTACK_SAFE):
            if state.phase == Phase.MAIN1 and self.steps == 0:
                return PASS  # to declare attackers
            if state.phase == Phase.DECLARE_ATTACKERS:
                for slot in sorted(self.attackers):
                    if state.is_legal(ATTACK.start + slot):
                        return ATTACK.start + slot
                return PASS  # done declaring; blocks are the opponent's decision
            return None

        if m == Macro.NO_ATTACK:
            if self.start_context == Context.PRECOMBAT:
                return PASS if state.phase in (Phase.MAIN1, Phase.DECLARE_ATTACKERS) else None
            # Post-combat: cast the cards no macro covers when they help, then end the turn.
            for card in (NATURAL_SPRING, DIVINATION):
                if self._worth_casting(state, card):
                    return PLAY_HAND.start + _hand_slot(state, me, card)
            return PASS if state.phase == Phase.MAIN2 else None

        if m in (Macro.BLOCK_VALUE, Macro.NO_BLOCK):
            if state.phase != Phase.DECLARE_BLOCKERS:
                return None
            blocker = int(state.blocker_queue[state.blocker_cursor])
            attacker = self.blocks.get(blocker)
            if attacker is not None and state.is_legal(BLOCK.start + attacker):
                return BLOCK.start + attacker
            return PASS
        return None

    def _worth_casting(self, state: GameState, card: int) -> bool:
        me = self.player
        if state.phase != Phase.MAIN2 or card not in _hand(state, me) or not _affordable(state, me, card):
            return False
        if card == DIVINATION:
            # Drawing at the hand cap draws nothing; leave room for both cards.
            return state.hand_count[me] - 1 + card_db.AMOUNT[card] <= HAND_SIZE
        return True


def run_macro(state: GameState, player: int, macro: Macro) -> list[int]:
    """Apply `macro` to `state` in place; return the primitive actions taken."""
    execution = MacroExecution(state, player, macro)
    actions = []
    while (action := execution.next_action(state)) is not None:
        state.step(action)
        actions.append(action)
    if not actions:
        raise RuntimeError(f"{Macro(macro).name} produced no actions")
    return actions
