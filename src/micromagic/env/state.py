"""MicroMagic game state and rules engine.

Simplifications versus full Magic, all deliberate:
- Sorcery speed only: no instants, no responses, no stack.
- Mana is generic: each untapped land pays 1 toward any cost, tapped on payment.
- At most one blocker per attacker, so no damage-assignment ordering.
- No first strike, so combat damage is a single simultaneous step.
- No mulligans. The player going first skips their first draw.
- A player holding 7 cards skips their draw. Drawing from an empty library loses.
- Targeted spells pick their target right after being cast (CHOOSE_TARGET);
  a spell is only castable when it has at least one legal target.
- After MAX_TURNS turns the game ends as a draw (truncated).
"""

import numpy as np

from micromagic.env import cards as card_db
from micromagic.env.spec import (
    ATTACK,
    BLOCK,
    CARD_FEATURES,
    CREATURE_SLOT_FEATURES,
    CREATURE_SLOTS,
    EMPTY,
    GLOBAL_FEATURES,
    HAND_SIZE,
    HAND_SLOT_FEATURES,
    MAX_TURNS,
    NUM_ACTIONS,
    OBS_DIM,
    OBS_HAND,
    OBS_MY_CREATURES,
    OBS_OPP_CREATURES,
    OBS_PENDING_SPELL,
    OPENING_HAND,
    PASS,
    PLAY_HAND,
    SCALE_LIFE,
    SCALE_STAT,
    SCALE_ZONE,
    STARTING_LIFE,
    TARGET_MY_CREATURE,
    TARGET_OPP_CREATURE,
    TARGET_OPPONENT,
    CardType,
    Keyword,
    Phase,
    SpellTemplate,
    TargetScope,
)


DeckSpec = np.ndarray | str


class GameState:
    __slots__ = (
        "rng",
        "library",
        "library_top",
        "hand",
        "hand_count",
        "lands_total",
        "lands_tapped",
        "creature",
        "damage",
        "tapped",
        "sick",
        "attacking",
        "blocking",
        "pump",
        "deathtouched",
        "life",
        "active",
        "turn",
        "phase",
        "land_played",
        "blocker_queue",
        "blocker_cursor",
        "pending_card",
        "return_phase",
        "_terminal",
        "_truncated",
        "_winner",
    )

    def __init__(
        self,
        seed: int | None = None,
        decks: tuple[DeckSpec, DeckSpec] | None = None,
        first_player: int = 0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        if decks is None:
            decks = ("aggro", "midrange")
        deck_arrays = [card_db.DECKS[d] if isinstance(d, str) else np.asarray(d, dtype=np.int16) for d in decks]
        self.library = [self.rng.permutation(d) for d in deck_arrays]
        # Library arrays are never shrunk; library_top[p] indexes the next card to draw.
        self.library_top = np.zeros(2, dtype=np.int16)

        self.hand = np.full((2, HAND_SIZE), EMPTY, dtype=np.int16)
        self.hand_count = np.zeros(2, dtype=np.int16)
        self.lands_total = np.zeros(2, dtype=np.int16)
        self.lands_tapped = np.zeros(2, dtype=np.int16)

        self.creature = np.full((2, CREATURE_SLOTS), EMPTY, dtype=np.int16)
        self.damage = np.zeros((2, CREATURE_SLOTS), dtype=np.int16)
        self.tapped = np.zeros((2, CREATURE_SLOTS), dtype=bool)
        self.sick = np.zeros((2, CREATURE_SLOTS), dtype=bool)
        self.attacking = np.zeros((2, CREATURE_SLOTS), dtype=bool)
        # blocking[defender, slot] = the attacker slot that creature blocks, or EMPTY.
        self.blocking = np.full((2, CREATURE_SLOTS), EMPTY, dtype=np.int16)
        # Until-end-of-turn +N/+N from pump spells.
        self.pump = np.zeros((2, CREATURE_SLOTS), dtype=np.int16)
        # Dealt damage by a deathtouch source this turn.
        self.deathtouched = np.zeros((2, CREATURE_SLOTS), dtype=bool)

        self.life = np.full(2, STARTING_LIFE, dtype=np.int16)
        self.active = first_player
        self.turn = 0
        self.phase = Phase.MAIN1
        self.land_played = False
        self.blocker_queue = np.full(CREATURE_SLOTS, EMPTY, dtype=np.int16)
        self.blocker_cursor = 0
        self.pending_card = EMPTY
        self.return_phase = Phase.MAIN1
        self._terminal = False
        self._truncated = False
        self._winner: int | None = None

        for player in (0, 1):
            for _ in range(OPENING_HAND):
                self._draw(player)
        self._begin_turn(first_player, skip_draw=True)

    # Public interface (frozen contract)

    def legal_actions(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if self._terminal:
            return mask

        me = self.active
        if self.phase == Phase.CHOOSE_TARGET:
            self._mark_targets(mask, int(card_db.TARGET[self.pending_card]))
            return mask

        mask[PASS] = True
        if self.phase in (Phase.MAIN1, Phase.MAIN2):
            for slot in range(self.hand_count[me]):
                mask[PLAY_HAND.start + slot] = self._can_play(me, int(self.hand[me, slot]))
        elif self.phase == Phase.DECLARE_ATTACKERS:
            for slot in range(CREATURE_SLOTS):
                mask[ATTACK.start + slot] = self._can_attack(me, slot)
        elif self.phase == Phase.DECLARE_BLOCKERS:
            blocker = int(self.blocker_queue[self.blocker_cursor])
            for slot in self._blockable_attackers(blocker):
                mask[BLOCK.start + slot] = True
        return mask

    def step(self, action: int) -> None:
        if self._terminal:
            raise ValueError("Game is over; no actions are legal.")
        if not 0 <= action < NUM_ACTIONS or not self.legal_actions()[action]:
            raise ValueError(f"Illegal action {action} in phase {self.phase.name}")

        if self.phase == Phase.CHOOSE_TARGET:
            self._resolve_targeted_spell(action)
        elif action == PASS:
            self._pass()
        elif action in PLAY_HAND:
            self._play_card(action - PLAY_HAND.start)
        elif action in ATTACK:
            self.attacking[self.active, action - ATTACK.start] = True
        elif action in BLOCK:
            self._assign_block(action - BLOCK.start)

    def current_player(self) -> int:
        if self.phase == Phase.DECLARE_BLOCKERS:
            return 1 - self.active
        return self.active

    def observe(self, player: int) -> np.ndarray:
        opp = 1 - player
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        g = obs[:GLOBAL_FEATURES]
        g[0] = max(int(self.life[player]), 0) / SCALE_LIFE
        g[1] = max(int(self.life[opp]), 0) / SCALE_LIFE
        g[2] = self.turn / MAX_TURNS
        g[3 + int(self.phase)] = 1.0
        offset = 3 + len(Phase)
        g[offset] = float(self.active == player)
        g[offset + 1] = self.lands_total[player] / SCALE_ZONE
        g[offset + 2] = (self.lands_total[player] - self.lands_tapped[player]) / SCALE_ZONE
        g[offset + 3] = float(self.land_played and self.active == player)
        g[offset + 4] = self.lands_total[opp] / SCALE_ZONE
        g[offset + 5] = (self.lands_total[opp] - self.lands_tapped[opp]) / SCALE_ZONE
        g[offset + 6] = self.hand_count[player] / HAND_SIZE
        g[offset + 7] = self.hand_count[opp] / HAND_SIZE
        g[offset + 8] = self._library_count(player) / SCALE_ZONE
        g[offset + 9] = self._library_count(opp) / SCALE_ZONE

        # A cast spell is public, so both players see what is waiting for a target.
        if self.phase == Phase.CHOOSE_TARGET:
            obs[OBS_PENDING_SPELL] = card_db.FEATURE_TABLE[self.pending_card]

        # Only the observer's own hand is visible; the opponent's is just a count.
        hand_obs = obs[OBS_HAND].reshape(HAND_SIZE, HAND_SLOT_FEATURES)
        for slot in range(self.hand_count[player]):
            hand_obs[slot, 0] = 1.0
            hand_obs[slot, 1:] = card_db.FEATURE_TABLE[self.hand[player, slot]]

        self._observe_creatures(player, obs[OBS_MY_CREATURES])
        self._observe_creatures(opp, obs[OBS_OPP_CREATURES])
        return np.clip(obs, 0.0, 1.0, out=obs)

    def is_terminal(self) -> bool:
        return self._terminal

    def is_truncated(self) -> bool:
        """True when the game ended by hitting MAX_TURNS rather than by a win."""
        return self._truncated

    def winner(self) -> int | None:
        return self._winner

    def clone(self) -> "GameState":
        new = object.__new__(GameState)
        for name in GameState.__slots__:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(new, name, value)
        new.library = [lib.copy() for lib in self.library]
        new.rng = np.random.default_rng()
        new.rng.bit_generator.state = self.rng.bit_generator.state
        return new

    def determinize(self, player: int, rng: np.random.Generator) -> "GameState":
        """Resample everything `player` cannot see: the opponent's hand and both library orders."""
        new = self.clone()
        opp = 1 - player

        opp_hand = new.hand[opp, : new.hand_count[opp]]
        opp_remaining = new.library[opp][new.library_top[opp] :]
        pool = rng.permutation(np.concatenate([opp_hand, opp_remaining]))
        n = int(new.hand_count[opp])
        new.hand[opp, :n] = pool[:n]
        new.library[opp][new.library_top[opp] :] = pool[n:]

        mine = new.library[player][new.library_top[player] :]
        new.library[player][new.library_top[player] :] = rng.permutation(mine)
        new.rng = np.random.default_rng(rng.integers(2**63))
        return new

    # Creature stats

    def power(self, player: int, slot: int) -> int:
        return int(card_db.POWER[self.creature[player, slot]] + self.pump[player, slot])

    def toughness(self, player: int, slot: int) -> int:
        return int(card_db.TOUGHNESS[self.creature[player, slot]] + self.pump[player, slot])

    def has_keyword(self, player: int, slot: int, keyword: Keyword) -> bool:
        card_id = self.creature[player, slot]
        return card_id != EMPTY and bool(card_db.HAS_KEYWORD[card_id, keyword])

    # Turn structure

    def _begin_turn(self, player: int, skip_draw: bool = False) -> None:
        self.turn += 1
        if self.turn > MAX_TURNS:
            self._end_game(winner=None, truncated=True)
            return

        self.active = player
        self.lands_tapped[player] = 0
        self.tapped[player] = False
        self.sick[player] = False
        self.land_played = False
        self.phase = Phase.MAIN1
        if not skip_draw:
            self._draw(player)

    def _pass(self) -> None:
        if self.phase == Phase.MAIN1:
            has_attacker = any(self._can_attack(self.active, s) for s in range(CREATURE_SLOTS))
            self.phase = Phase.DECLARE_ATTACKERS if has_attacker else Phase.MAIN2
        elif self.phase == Phase.DECLARE_ATTACKERS:
            self._finish_declaring_attackers()
        elif self.phase == Phase.DECLARE_BLOCKERS:
            self.blocker_cursor += 1
            self._seek_next_blocker()
        elif self.phase == Phase.MAIN2:
            self._end_turn()

    def _end_turn(self) -> None:
        # Damage and pump effects wear off at end of turn.
        self.damage[:] = 0
        self.pump[:] = 0
        self.deathtouched[:] = False
        self._begin_turn(1 - self.active)

    # Main phase

    def _can_play(self, player: int, card_id: int) -> bool:
        card_type = card_db.TYPE[card_id]
        if card_type == CardType.LAND:
            return not self.land_played
        if card_db.COST[card_id] > self._untapped_lands(player):
            return False
        if card_type == CardType.CREATURE:
            return bool((self.creature[player] == EMPTY).any())
        scope = int(card_db.TARGET[card_id])
        if scope in (TargetScope.NONE, TargetScope.ANY, TargetScope.PLAYER):
            return True
        return bool((self.creature != EMPTY).any())

    def _play_card(self, slot: int) -> None:
        me = self.active
        card_id = int(self.hand[me, slot])
        self._remove_from_hand(me, slot)

        card_type = card_db.TYPE[card_id]
        if card_type == CardType.LAND:
            self.lands_total[me] += 1
            self.land_played = True
            return

        self.lands_tapped[me] += card_db.COST[card_id]
        if card_type == CardType.CREATURE:
            free_slot = int(np.flatnonzero(self.creature[me] == EMPTY)[0])
            self._clear_slot(me, free_slot)
            self.creature[me, free_slot] = card_id
            self.sick[me, free_slot] = True
            return

        if card_db.TARGET[card_id] == TargetScope.NONE:
            self._resolve_untargeted_spell(card_id)
            return
        self.pending_card = card_id
        self.return_phase = self.phase
        self.phase = Phase.CHOOSE_TARGET

    # Spells

    def _mark_targets(self, mask: np.ndarray, scope: int) -> None:
        me, opp = self.active, 1 - self.active
        if scope in (TargetScope.ANY, TargetScope.PLAYER):
            mask[TARGET_OPPONENT] = True
        if scope in (TargetScope.ANY, TargetScope.CREATURE):
            mask[TARGET_OPP_CREATURE.start : TARGET_OPP_CREATURE.stop] = self.creature[opp] != EMPTY
            mask[TARGET_MY_CREATURE.start : TARGET_MY_CREATURE.stop] = self.creature[me] != EMPTY

    def _resolve_untargeted_spell(self, card_id: int) -> None:
        me = self.active
        amount = int(card_db.AMOUNT[card_id])
        template = card_db.TEMPLATE[card_id]
        if template == SpellTemplate.DRAW:
            for _ in range(amount):
                if not self._terminal:
                    self._draw(me)
        elif template == SpellTemplate.GAIN_LIFE:
            self.life[me] += amount

    def _resolve_targeted_spell(self, action: int) -> None:
        card_id = self.pending_card
        me, opp = self.active, 1 - self.active
        amount = int(card_db.AMOUNT[card_id])
        template = card_db.TEMPLATE[card_id]

        if action == TARGET_OPPONENT:
            # Only burn can target a player.
            self.life[opp] -= amount
        else:
            if action in TARGET_OPP_CREATURE:
                owner, slot = opp, action - TARGET_OPP_CREATURE.start
            else:
                owner, slot = me, action - TARGET_MY_CREATURE.start
            if template == SpellTemplate.BURN:
                self.damage[owner, slot] += amount
            elif template == SpellTemplate.DESTROY:
                self._clear_slot(owner, slot)
            elif template == SpellTemplate.PUMP:
                self.pump[owner, slot] += amount

        self.pending_card = EMPTY
        self.phase = self.return_phase
        self._destroy_lethally_damaged()
        self._check_life_totals()

    # Combat

    def _can_attack(self, player: int, slot: int) -> bool:
        if self.creature[player, slot] == EMPTY or self.tapped[player, slot] or self.attacking[player, slot]:
            return False
        return not self.sick[player, slot] or self.has_keyword(player, slot, Keyword.HASTE)

    def _can_block(self, blocker_slot: int, attacker_slot: int) -> bool:
        defender, attacker = 1 - self.active, self.active
        if not self.has_keyword(attacker, attacker_slot, Keyword.FLYING):
            return True
        return self.has_keyword(defender, blocker_slot, Keyword.FLYING) or self.has_keyword(
            defender, blocker_slot, Keyword.REACH
        )

    def _blockable_attackers(self, blocker_slot: int) -> list[int]:
        already_blocked = self._blocked_attackers()
        return [
            a
            for a in np.flatnonzero(self.attacking[self.active])
            if not already_blocked[a] and self._can_block(blocker_slot, int(a))
        ]

    def _finish_declaring_attackers(self) -> None:
        me, opp = self.active, 1 - self.active
        if not self.attacking[me].any():
            self.phase = Phase.MAIN2
            return

        for slot in np.flatnonzero(self.attacking[me]):
            if not self.has_keyword(me, int(slot), Keyword.VIGILANCE):
                self.tapped[me, slot] = True

        can_block = (self.creature[opp] != EMPTY) & ~self.tapped[opp]
        can_block &= ~card_db.HAS_KEYWORD[np.where(can_block, self.creature[opp], 0), Keyword.CANT_BLOCK]
        blockers = np.flatnonzero(can_block)
        self.blocker_queue[:] = EMPTY
        self.blocker_queue[: blockers.size] = blockers
        self.blocker_cursor = 0
        self.phase = Phase.DECLARE_BLOCKERS
        self._seek_next_blocker()

    def _assign_block(self, attacker_slot: int) -> None:
        defender = 1 - self.active
        blocker_slot = int(self.blocker_queue[self.blocker_cursor])
        self.blocking[defender, blocker_slot] = attacker_slot
        self.blocker_cursor += 1
        self._seek_next_blocker()

    def _seek_next_blocker(self) -> None:
        """Move the cursor to the next creature with a legal block, or resolve combat."""
        while self.blocker_cursor < CREATURE_SLOTS:
            blocker = int(self.blocker_queue[self.blocker_cursor])
            if blocker == EMPTY:
                break
            if self._blockable_attackers(blocker):
                return
            self.blocker_cursor += 1
        self._resolve_combat()

    def _blocked_attackers(self) -> np.ndarray:
        blocked = np.zeros(CREATURE_SLOTS, dtype=bool)
        targets = self.blocking[1 - self.active]
        blocked[targets[targets != EMPTY]] = True
        return blocked

    def _deal_creature_damage(self, owner: int, slot: int, amount: int, deathtouch: bool) -> None:
        if amount <= 0:
            return
        self.damage[owner, slot] += amount
        if deathtouch:
            self.deathtouched[owner, slot] = True

    def _resolve_combat(self) -> None:
        me, opp = self.active, 1 - self.active
        # Read every combat stat before applying damage: damage is simultaneous.
        for attacker in map(int, np.flatnonzero(self.attacking[me])):
            power = self.power(me, attacker)
            deathtouch = self.has_keyword(me, attacker, Keyword.DEATHTOUCH)
            blockers = np.flatnonzero(self.blocking[opp] == attacker)

            if blockers.size == 0:
                self.life[opp] -= max(power, 0)
            else:
                blocker = int(blockers[0])
                to_blocker = power
                if self.has_keyword(me, attacker, Keyword.TRAMPLE):
                    remaining = self.toughness(opp, blocker) - int(self.damage[opp, blocker])
                    lethal = min(1, remaining) if deathtouch else remaining
                    to_blocker = min(power, max(lethal, 0))
                    self.life[opp] -= max(power - to_blocker, 0)
                self._deal_creature_damage(opp, blocker, to_blocker, deathtouch)

                blocker_power = self.power(opp, blocker)
                self._deal_creature_damage(
                    me, attacker, blocker_power, self.has_keyword(opp, blocker, Keyword.DEATHTOUCH)
                )
                if self.has_keyword(opp, blocker, Keyword.LIFELINK):
                    self.life[opp] += max(blocker_power, 0)

            if self.has_keyword(me, attacker, Keyword.LIFELINK):
                self.life[me] += max(power, 0)

        self.attacking[:] = False
        self.blocking[:] = EMPTY
        self.phase = Phase.MAIN2
        self._destroy_lethally_damaged()
        self._check_life_totals()

    # State-based checks

    def _destroy_lethally_damaged(self) -> None:
        for player in (0, 1):
            for slot in range(CREATURE_SLOTS):
                if self.creature[player, slot] == EMPTY:
                    continue
                lethal = self.damage[player, slot] >= self.toughness(player, slot)
                if lethal or self.deathtouched[player, slot]:
                    self._clear_slot(player, slot)

    def _check_life_totals(self) -> None:
        dead = self.life <= 0
        if dead.all():
            self._end_game(winner=None)
        elif dead.any():
            self._end_game(winner=int(np.flatnonzero(~dead)[0]))

    # Zones and helpers

    def _clear_slot(self, player: int, slot: int) -> None:
        self.creature[player, slot] = EMPTY
        self.damage[player, slot] = 0
        self.tapped[player, slot] = False
        self.sick[player, slot] = False
        self.attacking[player, slot] = False
        self.blocking[player, slot] = EMPTY
        self.pump[player, slot] = 0
        self.deathtouched[player, slot] = False

    def _draw(self, player: int) -> None:
        if self.hand_count[player] >= HAND_SIZE:
            return
        if self._library_count(player) == 0:
            self._end_game(winner=1 - player)
            return
        self.hand[player, self.hand_count[player]] = self.library[player][self.library_top[player]]
        self.library_top[player] += 1
        self.hand_count[player] += 1

    def _remove_from_hand(self, player: int, slot: int) -> None:
        # Shift later cards left so occupied slots stay contiguous from slot 0.
        count = int(self.hand_count[player])
        self.hand[player, slot : count - 1] = self.hand[player, slot + 1 : count]
        self.hand[player, count - 1] = EMPTY
        self.hand_count[player] -= 1

    def _library_count(self, player: int) -> int:
        return len(self.library[player]) - int(self.library_top[player])

    def _untapped_lands(self, player: int) -> int:
        return int(self.lands_total[player] - self.lands_tapped[player])

    def _observe_creatures(self, owner: int, out: np.ndarray) -> None:
        slots = out.reshape(CREATURE_SLOTS, CREATURE_SLOT_FEATURES)
        blocked = self._blocked_attackers() if self.phase == Phase.DECLARE_BLOCKERS else None
        for slot in range(CREATURE_SLOTS):
            card_id = self.creature[owner, slot]
            if card_id == EMPTY:
                continue
            row = slots[slot]
            row[0] = 1.0
            row[1 : 1 + CARD_FEATURES] = card_db.FEATURE_TABLE[card_id]
            status = row[1 + CARD_FEATURES :]
            status[0] = self.damage[owner, slot] / SCALE_STAT
            status[1] = self.tapped[owner, slot]
            status[2] = self.sick[owner, slot]
            status[3] = self.attacking[owner, slot]
            status[4] = self.blocking[owner, slot] != EMPTY
            status[5] = blocked is not None and owner == self.active and blocked[slot]
            status[6] = self.power(owner, slot) / SCALE_STAT
            status[7] = self.toughness(owner, slot) / SCALE_STAT

    def _end_game(self, winner: int | None, truncated: bool = False) -> None:
        if self._terminal:
            return
        self._terminal = True
        self._truncated = truncated
        self._winner = winner

    def __repr__(self) -> str:
        return (
            f"GameState(turn={self.turn}, active={self.active}, phase={self.phase.name}, "
            f"life={self.life.tolist()}, hands={self.hand_count.tolist()}, "
            f"lands={self.lands_total.tolist()}, creatures={self.creature.tolist()})"
        )
