"""Rule-based baseline (check-in 1, §1.5).

Plays a land, sends burn and removal at the largest threat (or the face when it
is lethal), casts the most expensive affordable spell, attacks when the attack
is safe or lethal, and blocks for value, chump-blocking only when facing lethal.

It reads only its own observation and the mask, decoded through the frozen
layout in spec.py, so it plays through the tournament harness, the AEC env and
as a Gymnasium opponent like any other agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from micromagic.agents.base import Agent
from micromagic.env import cards as card_db
from micromagic.env.spec import (
    ATTACK,
    BLOCK,
    CARD_FEATURES,
    CREATURE_SLOT_FEATURES,
    CREATURE_SLOTS,
    GLOBAL_FEATURE_NAMES,
    HAND_SIZE,
    HAND_SLOT_FEATURES,
    MAX_TURNS,
    NUM_CARDS,
    OBS_GLOBAL,
    OBS_HAND,
    OBS_MY_CREATURES,
    OBS_OPP_CREATURES,
    OBS_PENDING_SPELL,
    PASS,
    PLAY_HAND,
    SCALE_LIFE,
    SCALE_STAT,
    SCALE_ZONE,
    TARGET_MY_CREATURE,
    TARGET_OPP_CREATURE,
    TARGET_OPPONENT,
    CardType,
    Keyword,
    Phase,
    SpellTemplate,
    TargetScope,
)


_G = {name: i for i, name in enumerate(GLOBAL_FEATURE_NAMES)}
# Status columns follow the presence flag and card features in a creature slot.
_STATUS = 1 + CARD_FEATURES


@dataclass
class Creature:
    slot: int
    card: int
    damage: int
    tapped: bool
    sick: bool
    attacking: bool
    blocking: bool
    blocked: bool
    power: int
    toughness: int

    def has(self, keyword: Keyword) -> bool:
        return bool(card_db.HAS_KEYWORD[self.card, keyword])

    @property
    def health(self) -> int:
        return self.toughness - self.damage

    @property
    def value(self) -> float:
        bonus = sum(self.has(k) for k in (Keyword.FLYING, Keyword.DEATHTOUCH, Keyword.LIFELINK, Keyword.TRAMPLE))
        return self.power + 0.5 * self.toughness + bonus


@dataclass
class View:
    my_life: int
    opp_life: int
    turn: int
    phase: Phase
    hand: list[int]
    mine: list[Creature]
    theirs: list[Creature]
    pending: int | None


def _card_id(features: np.ndarray) -> int:
    return int(np.argmax(features[:NUM_CARDS]))


def _creatures(block: np.ndarray) -> list[Creature]:
    out = []
    for slot, row in enumerate(block.reshape(CREATURE_SLOTS, CREATURE_SLOT_FEATURES)):
        if row[0] < 0.5:
            continue
        s = row[_STATUS:]
        out.append(
            Creature(
                slot=slot,
                card=_card_id(row[1:]),
                damage=round(s[0] * SCALE_STAT),
                tapped=s[1] > 0.5,
                sick=s[2] > 0.5,
                attacking=s[3] > 0.5,
                blocking=s[4] > 0.5,
                blocked=s[5] > 0.5,
                power=round(s[6] * SCALE_STAT),
                toughness=round(s[7] * SCALE_STAT),
            )
        )
    return out


def decode(obs: np.ndarray) -> View:
    """Turn an observation back into the facts the heuristic reasons about."""
    g = obs[OBS_GLOBAL]
    hand = obs[OBS_HAND].reshape(HAND_SIZE, HAND_SLOT_FEATURES)
    pending = obs[OBS_PENDING_SPELL]
    return View(
        my_life=round(g[_G["my_life"]] * SCALE_LIFE),
        opp_life=round(g[_G["opp_life"]] * SCALE_LIFE),
        turn=round(g[_G["turn"]] * MAX_TURNS),
        phase=Phase(int(np.argmax(g[_G["phase_main1"] : _G["phase_main1"] + len(Phase)]))),
        hand=[_card_id(row[1:]) for row in hand if row[0] > 0.5],
        mine=_creatures(obs[OBS_MY_CREATURES]),
        theirs=_creatures(obs[OBS_OPP_CREATURES]),
        pending=_card_id(pending) if pending.any() else None,
    )


def _kills(power: int, deathtouch: bool, target: Creature) -> bool:
    return power > 0 and (deathtouch or power >= target.health)


def _can_block(blocker: Creature, attacker: Creature) -> bool:
    if blocker.has(Keyword.CANT_BLOCK):
        return False
    return not attacker.has(Keyword.FLYING) or blocker.has(Keyword.FLYING) or blocker.has(Keyword.REACH)


class HeuristicAgent(Agent):
    """Deterministic rule policy: the same observation always gives the same action,
    except during blocks, where it remembers which of its creatures it already
    decided for this combat (the observation does not say which blocker is up)."""

    def __init__(self, name: str = "heuristic") -> None:
        super().__init__(name)
        self.reset()

    def reset(self) -> None:
        self._combat: tuple | None = None
        self._decided: set[int] = set()

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        view = decode(obs)
        if view.phase == Phase.CHOOSE_TARGET:
            action = self._target(view, mask)
        elif view.phase in (Phase.MAIN1, Phase.MAIN2):
            action = self._main(view, mask)
        elif view.phase == Phase.DECLARE_ATTACKERS:
            action = self._attack(view, mask)
        else:
            action = self._block(view, mask)
        # Safety net: never return an illegal action, whatever the rules above decide.
        return action if mask[action] else int(np.flatnonzero(mask)[0])

    # Main phase

    def _main(self, view: View, mask: np.ndarray) -> int:
        playable = [(slot, card) for slot, card in enumerate(view.hand) if mask[PLAY_HAND.start + slot]]
        if not playable:
            return PASS

        def first(predicate) -> int | None:
            candidates = [(slot, card) for slot, card in playable if predicate(card)]
            if not candidates:
                return None
            # The most expensive card first: it uses the most mana this turn.
            slot, _ = max(candidates, key=lambda sc: card_db.COST[sc[1]])
            return PLAY_HAND.start + slot

        is_type = lambda t: lambda c: card_db.TYPE[c] == t  # noqa: E731
        is_template = lambda t: lambda c: card_db.TEMPLATE[c] == t  # noqa: E731

        land = first(is_type(CardType.LAND))
        if land is not None:
            return land

        burn = first(is_template(SpellTemplate.BURN))
        if burn is not None:
            return burn
        removal = first(lambda c: is_template(SpellTemplate.DESTROY)(c) and bool(view.theirs))
        if removal is not None:
            return removal

        creature = first(is_type(CardType.CREATURE))
        if creature is not None:
            return creature
        draw = first(is_template(SpellTemplate.DRAW))
        if draw is not None:
            return draw
        life = first(lambda c: is_template(SpellTemplate.GAIN_LIFE)(c) and view.my_life <= 14)
        if life is not None:
            return life
        # Pump only before combat, on a creature that can attack.
        if view.phase == Phase.MAIN1 and any(not c.tapped and not c.sick for c in view.mine):
            pump = first(is_template(SpellTemplate.PUMP))
            if pump is not None:
                return pump
        return PASS

    def _burn_target(self, view: View, card: int) -> int:
        """Face when lethal, else the most valuable creature it kills, else face."""
        amount = int(card_db.AMOUNT[card])
        if amount >= view.opp_life:
            return TARGET_OPPONENT
        if card_db.TARGET[card] != TargetScope.PLAYER:
            killable = [c for c in view.theirs if amount >= c.health]
            if killable:
                best = max(killable, key=lambda c: c.value)
                # Spend burn on a creature only if it is worth at least a 2-drop.
                if best.value >= 2.5:
                    return TARGET_OPP_CREATURE.start + best.slot
        return TARGET_OPPONENT

    def _target(self, view: View, mask: np.ndarray) -> int:
        card = view.pending
        if card is None:
            return int(np.flatnonzero(mask)[0])
        template = card_db.TEMPLATE[card]
        if template == SpellTemplate.BURN:
            action = self._burn_target(view, card)
            if mask[action]:
                return action
        theirs = [c for c in view.theirs if mask[TARGET_OPP_CREATURE.start + c.slot]]
        mine = [c for c in view.mine if mask[TARGET_MY_CREATURE.start + c.slot]]
        if template == SpellTemplate.DESTROY and theirs:
            return TARGET_OPP_CREATURE.start + max(theirs, key=lambda c: c.value).slot
        if template == SpellTemplate.PUMP and mine:
            ready = [c for c in mine if not c.tapped and not c.sick] or mine
            return TARGET_MY_CREATURE.start + max(ready, key=lambda c: c.power).slot
        # Forced to target something bad: hurt the opponent's weakest or our own least valuable.
        if theirs:
            return TARGET_OPP_CREATURE.start + min(theirs, key=lambda c: c.value).slot
        if mask[TARGET_OPPONENT]:
            return TARGET_OPPONENT
        return TARGET_MY_CREATURE.start + min(mine, key=lambda c: c.value).slot

    # Combat

    def _attack(self, view: View, mask: np.ndarray) -> int:
        ready = [c for c in view.mine if c.attacking or mask[ATTACK.start + c.slot]]
        blockers = [c for c in view.theirs if not c.tapped and not c.has(Keyword.CANT_BLOCK)]

        # Damage that gets through if they block our biggest attackers.
        powers = sorted((c.power for c in ready), reverse=True)
        through = sum(powers[len(blockers) :])
        if through >= view.opp_life or self._winning_race(view, through):
            chosen = ready
        else:
            chosen = [c for c in ready if self._safe(c, blockers)]

        for c in chosen:
            if not c.attacking:
                return ATTACK.start + c.slot
        return PASS

    @staticmethod
    def _winning_race(view: View, through: int) -> bool:
        """Our clock, attacking with everything, is no slower than theirs."""
        their_power = sum(c.power for c in view.theirs)
        if through <= 0:
            return False
        if their_power <= 0:
            return True
        ours = -(-view.opp_life // through)
        theirs = -(-view.my_life // their_power)
        return ours <= theirs

    @staticmethod
    def _safe(attacker: Creature, blockers: list[Creature]) -> bool:
        """No blocker can kill it and survive."""
        for b in blockers:
            if not _can_block(b, attacker):
                continue
            # A block that kills it is fine as long as the blocker dies too.
            if _kills(b.power, b.has(Keyword.DEATHTOUCH), attacker) and not _kills(
                attacker.power, attacker.has(Keyword.DEATHTOUCH), b
            ):
                return False
        return True

    def _block(self, view: View, mask: np.ndarray) -> int:
        attackers = [c for c in view.theirs if c.attacking]
        # Identifies this combat, also across games when the harness reuses the agent.
        key = (
            view.turn,
            view.my_life,
            view.opp_life,
            tuple((c.slot, c.card) for c in attackers),
            tuple((c.slot, c.card, c.tapped) for c in view.mine),
        )
        if key != self._combat:
            self._combat, self._decided = key, set()

        open_attackers = [a for a in attackers if not a.blocked]
        options = [a for a in attackers if mask[BLOCK.start + a.slot]]
        # The engine asks each untapped creature in slot order and skips any that
        # can't block an open attacker, so the one asking is the first such creature
        # we haven't decided for yet.
        blocker = next(
            (
                c
                for c in view.mine
                if c.slot not in self._decided
                and not c.tapped
                and not c.blocking
                and any(_can_block(c, a) for a in open_attackers)
            ),
            None,
        )
        if blocker is None or not options:
            return PASS
        self._decided.add(blocker.slot)

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
                score = 0.5 + a.power  # chump the biggest hitter
            else:
                continue
            if score > best_score:
                best, best_score = BLOCK.start + a.slot, score
        return best
