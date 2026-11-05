"""Macro-action enumeration, legality masking, and scripted execution (M1/M2).

Implements 9 high-level macro actions mapped down to legal primitive actions,
satisfying ADR-009 and docs/Aditya_Implementation_Plan.md §4.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from micromagic.agents.heuristic import Creature, decode
from micromagic.env import cards as card_db
from micromagic.env.spec import (
    ATTACK,
    BLOCK,
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


class MacroAction(IntEnum):
    PLAY_LAND = 0
    CAST_BEST_CREATURE = 1
    BURN_FACE = 2
    BURN_BEST_CREATURE = 3
    ATTACK_ALL = 4
    ATTACK_SAFE = 5
    NO_ATTACK = 6
    BLOCK_VALUE = 7
    NO_BLOCK = 8


NUM_MACROS: int = len(MacroAction)


def _kills(power: int, deathtouch: bool, target: Creature) -> bool:
    return power > 0 and (deathtouch or power >= target.health)


def _can_block(blocker: Creature, attacker: Creature) -> bool:
    if blocker.has(Keyword.CANT_BLOCK):
        return False
    return (
        not attacker.has(Keyword.FLYING)
        or blocker.has(Keyword.FLYING)
        or blocker.has(Keyword.REACH)
    )


def _safe(attacker: Creature, blockers: list[Creature]) -> bool:
    for b in blockers:
        if not _can_block(b, attacker):
            continue
        if _kills(b.power, b.has(Keyword.DEATHTOUCH), attacker) and not _kills(
            attacker.power, attacker.has(Keyword.DEATHTOUCH), b
        ):
            return False
    return True


def compute_macro_mask(obs: np.ndarray, primitive_mask: np.ndarray) -> np.ndarray:
    """Compute boolean availability mask of shape (9,) for macro actions.

    Guarantees at least one macro action is True whenever primitive_mask has legal actions.
    """
    macro_mask = np.zeros(NUM_MACROS, dtype=bool)
    if not primitive_mask.any():
        return macro_mask

    view = decode(obs)

    # 1. Main phases (MAIN1 / MAIN2)
    if view.phase in (Phase.MAIN1, Phase.MAIN2):
        # PLAY_LAND
        for slot, card in enumerate(view.hand):
            if (
                primitive_mask[PLAY_HAND.start + slot]
                and card_db.TYPE[card] == CardType.LAND
            ):
                macro_mask[MacroAction.PLAY_LAND] = True
                break

        # CAST_BEST_CREATURE
        for slot, card in enumerate(view.hand):
            if (
                primitive_mask[PLAY_HAND.start + slot]
                and card_db.TYPE[card] == CardType.CREATURE
            ):
                macro_mask[MacroAction.CAST_BEST_CREATURE] = True
                break

        # BURN_FACE
        for slot, card in enumerate(view.hand):
            if (
                primitive_mask[PLAY_HAND.start + slot]
                and card_db.TEMPLATE[card] == SpellTemplate.BURN
                and card_db.TARGET[card] in (TargetScope.PLAYER, TargetScope.ANY)
            ):
                macro_mask[MacroAction.BURN_FACE] = True
                break

        # BURN_BEST_CREATURE
        if view.theirs:
            for slot, card in enumerate(view.hand):
                if (
                    primitive_mask[PLAY_HAND.start + slot]
                    and card_db.TEMPLATE[card]
                    in (SpellTemplate.BURN, SpellTemplate.DESTROY)
                    and card_db.TARGET[card] in (TargetScope.CREATURE, TargetScope.ANY)
                ):
                    macro_mask[MacroAction.BURN_BEST_CREATURE] = True
                    break

        # NO_ATTACK acts as pass / transition to next phase in main phases
        if primitive_mask[PASS]:
            macro_mask[MacroAction.NO_ATTACK] = True

    # 2. Combat: Declare Attackers
    elif view.phase == Phase.DECLARE_ATTACKERS:
        can_attack = any(
            primitive_mask[ATTACK.start + c.slot] for c in view.mine if not c.attacking
        )
        if can_attack or primitive_mask[PASS]:
            macro_mask[MacroAction.ATTACK_ALL] = True
            macro_mask[MacroAction.ATTACK_SAFE] = True
        if primitive_mask[PASS]:
            macro_mask[MacroAction.NO_ATTACK] = True

    # 3. Combat: Declare Blockers
    elif view.phase == Phase.DECLARE_BLOCKERS:
        can_block = any(
            primitive_mask[BLOCK.start + a.slot] for a in view.theirs if a.attacking
        )
        if can_block or primitive_mask[PASS]:
            macro_mask[MacroAction.BLOCK_VALUE] = True
        if primitive_mask[PASS]:
            macro_mask[MacroAction.NO_BLOCK] = True

    # 4. Spell resolution: Choose Target
    elif view.phase == Phase.CHOOSE_TARGET:
        if primitive_mask[TARGET_OPPONENT]:
            macro_mask[MacroAction.BURN_FACE] = True
        if any(primitive_mask[TARGET_OPP_CREATURE.start + c.slot] for c in view.theirs):
            macro_mask[MacroAction.BURN_BEST_CREATURE] = True

    # Fallback guarantee: if primitive_mask has legal actions, at least one macro must be available
    if not macro_mask.any():
        if primitive_mask[PASS]:
            if view.phase == Phase.DECLARE_BLOCKERS:
                macro_mask[MacroAction.NO_BLOCK] = True
            else:
                macro_mask[MacroAction.NO_ATTACK] = True
        else:
            # If PASS is not available (e.g. forced targeting of friendly creature),
            # provide a fallback macro that will map down safely to the first legal primitive action
            if view.phase == Phase.CHOOSE_TARGET:
                macro_mask[MacroAction.BURN_BEST_CREATURE] = True
            else:
                macro_mask[MacroAction.NO_ATTACK] = True

    return macro_mask


def execute_macro(macro: int, obs: np.ndarray, primitive_mask: np.ndarray) -> int:
    """Execute a macro action and return a legal primitive action index.

    Guarantees the returned action satisfies primitive_mask[action] == True.
    """
    if not primitive_mask.any():
        return PASS

    view = decode(obs)
    chosen_action: int | None = None

    if macro == MacroAction.PLAY_LAND:
        playable_lands = [
            slot
            for slot, card in enumerate(view.hand)
            if primitive_mask[PLAY_HAND.start + slot]
            and card_db.TYPE[card] == CardType.LAND
        ]
        if playable_lands:
            chosen_action = PLAY_HAND.start + playable_lands[0]

    elif macro == MacroAction.CAST_BEST_CREATURE:
        playable_creatures = [
            (slot, card)
            for slot, card in enumerate(view.hand)
            if primitive_mask[PLAY_HAND.start + slot]
            and card_db.TYPE[card] == CardType.CREATURE
        ]
        if playable_creatures:
            best_slot, _ = max(
                playable_creatures,
                key=lambda sc: (card_db.COST[sc[1]], card_db.POWER[sc[1]]),
            )
            chosen_action = PLAY_HAND.start + best_slot

    elif macro == MacroAction.BURN_FACE:
        if view.phase == Phase.CHOOSE_TARGET:
            if primitive_mask[TARGET_OPPONENT]:
                chosen_action = TARGET_OPPONENT
        elif view.phase in (Phase.MAIN1, Phase.MAIN2):
            burn_spells = [
                (slot, card)
                for slot, card in enumerate(view.hand)
                if primitive_mask[PLAY_HAND.start + slot]
                and card_db.TEMPLATE[card] == SpellTemplate.BURN
                and card_db.TARGET[card] in (TargetScope.PLAYER, TargetScope.ANY)
            ]
            if burn_spells:
                best_slot, _ = max(
                    burn_spells,
                    key=lambda sc: (card_db.AMOUNT[sc[1]], card_db.COST[sc[1]]),
                )
                chosen_action = PLAY_HAND.start + best_slot

    elif macro == MacroAction.BURN_BEST_CREATURE:
        if view.phase == Phase.CHOOSE_TARGET:
            targetable = [
                c
                for c in view.theirs
                if primitive_mask[TARGET_OPP_CREATURE.start + c.slot]
            ]
            if targetable:
                best_target = max(targetable, key=lambda c: (c.value, c.power))
                chosen_action = TARGET_OPP_CREATURE.start + best_target.slot
        elif view.phase in (Phase.MAIN1, Phase.MAIN2):
            removal_spells = [
                (slot, card)
                for slot, card in enumerate(view.hand)
                if primitive_mask[PLAY_HAND.start + slot]
                and card_db.TEMPLATE[card]
                in (SpellTemplate.BURN, SpellTemplate.DESTROY)
                and card_db.TARGET[card] in (TargetScope.CREATURE, TargetScope.ANY)
            ]
            if removal_spells and view.theirs:
                best_slot, _ = max(removal_spells, key=lambda sc: card_db.COST[sc[1]])
                chosen_action = PLAY_HAND.start + best_slot

    elif macro == MacroAction.ATTACK_ALL:
        if view.phase == Phase.DECLARE_ATTACKERS:
            ready = [
                c
                for c in view.mine
                if not c.attacking and primitive_mask[ATTACK.start + c.slot]
            ]
            if ready:
                chosen_action = ATTACK.start + ready[0].slot
            elif primitive_mask[PASS]:
                chosen_action = PASS

    elif macro == MacroAction.ATTACK_SAFE:
        if view.phase == Phase.DECLARE_ATTACKERS:
            blockers = [
                c for c in view.theirs if not c.tapped and not c.has(Keyword.CANT_BLOCK)
            ]
            ready = [
                c
                for c in view.mine
                if not c.attacking and primitive_mask[ATTACK.start + c.slot]
            ]
            safe_attackers = [c for c in ready if _safe(c, blockers)]
            if safe_attackers:
                chosen_action = ATTACK.start + safe_attackers[0].slot
            elif primitive_mask[PASS]:
                chosen_action = PASS

    elif macro == MacroAction.NO_ATTACK:
        if primitive_mask[PASS]:
            chosen_action = PASS

    elif macro == MacroAction.BLOCK_VALUE:
        if view.phase == Phase.DECLARE_BLOCKERS:
            attackers = [c for c in view.theirs if c.attacking]
            options = [a for a in attackers if primitive_mask[BLOCK.start + a.slot]]
            blocker = next(
                (
                    c
                    for c in view.mine
                    if not c.tapped
                    and not c.blocking
                    and any(_can_block(c, a) for a in attackers if not a.blocked)
                ),
                None,
            )
            if blocker is not None and options:
                incoming = sum(a.power for a in attackers if not a.blocked)
                facing_lethal = incoming >= view.my_life
                best, best_score = PASS, 0.0
                for a in options:
                    if not _can_block(blocker, a):
                        continue
                    kills = _kills(blocker.power, blocker.has(Keyword.DEATHTOUCH), a)
                    dies = _kills(a.power, a.has(Keyword.DEATHTOUCH), blocker)
                    if kills and not dies:
                        score = 3.0 + a.value
                    elif not dies:
                        score = 1.0 + a.power
                    elif kills and a.value >= blocker.value:
                        score = 2.0 + a.value - blocker.value
                    elif facing_lethal:
                        score = 0.5 + a.power
                    else:
                        continue
                    if score > best_score:
                        best, best_score = BLOCK.start + a.slot, score
                if best != PASS and primitive_mask[best]:
                    chosen_action = best
            if chosen_action is None and primitive_mask[PASS]:
                chosen_action = PASS

    elif macro == MacroAction.NO_BLOCK:
        if primitive_mask[PASS]:
            chosen_action = PASS

    # Strict fallback guarantee: always produce a valid action satisfying primitive_mask[action]
    if (
        chosen_action is None
        or not (0 <= chosen_action < len(primitive_mask))
        or not primitive_mask[chosen_action]
    ):
        if primitive_mask[PASS]:
            return PASS
        return int(np.flatnonzero(primitive_mask)[0])

    return chosen_action
