"""MicroMagic game state and rules engine (toy mode: lands and vanilla creatures).

Simplifications versus full Magic, all deliberate:
- Sorcery speed only: no instants, no responses, no stack.
- Mana is generic: each untapped land pays 1 toward any cost, tapped on payment.
- At most one blocker per attacker, so no damage-assignment ordering.
- No mulligans. The player going first skips their first draw.
- A player holding 7 cards skips their draw. Drawing from an empty library loses.
- After MAX_TURNS turns the game ends as a draw (truncated).
Spells (CardType.SPELL) and targeting are not castable yet; the CHOOSE_TARGET
phase and TARGET_* actions are reserved for them and always masked off.
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
    OPENING_HAND,
    PASS,
    PLAY_HAND,
    SCALE_LIFE,
    SCALE_STAT,
    SCALE_ZONE,
    STARTING_LIFE,
    CardType,
    Phase,
)


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
        "life",
        "active",
        "turn",
        "phase",
        "land_played",
        "blocker_queue",
        "blocker_cursor",
        "_terminal",
        "_truncated",
        "_winner",
    )

    def __init__(
        self,
        seed: int | None = None,
        decks: tuple[np.ndarray, np.ndarray] | None = None,
        first_player: int = 0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        if decks is None:
            decks = (card_db.DECKS["toy"], card_db.DECKS["toy"])
        self.library = [self.rng.permutation(np.asarray(d, dtype=np.int16)) for d in decks]
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

        self.life = np.full(2, STARTING_LIFE, dtype=np.int16)
        self.active = first_player
        self.turn = 0
        self.phase = Phase.MAIN1
        self.land_played = False
        self.blocker_queue = np.full(CREATURE_SLOTS, EMPTY, dtype=np.int16)
        self.blocker_cursor = 0
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

        mask[PASS] = True
        me = self.active
        if self.phase in (Phase.MAIN1, Phase.MAIN2):
            for slot in range(self.hand_count[me]):
                mask[PLAY_HAND.start + slot] = self._can_play(me, int(self.hand[me, slot]))
        elif self.phase == Phase.DECLARE_ATTACKERS:
            for slot in range(CREATURE_SLOTS):
                mask[ATTACK.start + slot] = self._can_attack(me, slot)
        elif self.phase == Phase.DECLARE_BLOCKERS:
            already_blocked = self._blocked_attackers()
            for slot in range(CREATURE_SLOTS):
                mask[BLOCK.start + slot] = bool(self.attacking[me, slot]) and not already_blocked[slot]
        return mask

    def step(self, action: int) -> None:
        if self._terminal:
            raise ValueError("Game is over; no actions are legal.")
        if not 0 <= action < NUM_ACTIONS or not self.legal_actions()[action]:
            raise ValueError(f"Illegal action {action} in phase {self.phase.name}")

        if action == PASS:
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

        # Only the observer's own hand is visible; the opponent's is just a count.
        hand_obs = obs[OBS_HAND].reshape(HAND_SIZE, HAND_SLOT_FEATURES)
        for slot in range(self.hand_count[player]):
            hand_obs[slot, 0] = 1.0
            hand_obs[slot, 1:] = card_db.FEATURE_TABLE[self.hand[player, slot]]

        self._observe_creatures(player, obs[OBS_MY_CREATURES], player)
        self._observe_creatures(opp, obs[OBS_OPP_CREATURES], player)
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
            self._advance_blocker()
        elif self.phase == Phase.MAIN2:
            self._end_turn()

    def _end_turn(self) -> None:
        # Damage wears off at end of turn.
        self.damage[:] = 0
        self._begin_turn(1 - self.active)

    # Main phase

    def _can_play(self, player: int, card_id: int) -> bool:
        card_type = card_db.TYPE[card_id]
        if card_type == CardType.LAND:
            return not self.land_played
        if card_type == CardType.CREATURE:
            has_space = bool((self.creature[player] == EMPTY).any())
            return has_space and card_db.COST[card_id] <= self._untapped_lands(player)
        # Spells need targeting, which is not implemented yet.
        return False

    def _play_card(self, slot: int) -> None:
        me = self.active
        card_id = int(self.hand[me, slot])
        self._remove_from_hand(me, slot)

        if card_db.TYPE[card_id] == CardType.LAND:
            self.lands_total[me] += 1
            self.land_played = True
            return

        self.lands_tapped[me] += card_db.COST[card_id]
        free_slot = int(np.flatnonzero(self.creature[me] == EMPTY)[0])
        self.creature[me, free_slot] = card_id
        self.damage[me, free_slot] = 0
        self.tapped[me, free_slot] = False
        self.sick[me, free_slot] = True

    # Combat

    def _can_attack(self, player: int, slot: int) -> bool:
        return bool(
            self.creature[player, slot] != EMPTY
            and not self.tapped[player, slot]
            and not self.sick[player, slot]
            and not self.attacking[player, slot]
        )

    def _finish_declaring_attackers(self) -> None:
        me, opp = self.active, 1 - self.active
        if not self.attacking[me].any():
            self.phase = Phase.MAIN2
            return

        # No vigilance yet: attacking taps the creature.
        self.tapped[me] |= self.attacking[me]
        blockers = np.flatnonzero((self.creature[opp] != EMPTY) & ~self.tapped[opp])
        if blockers.size == 0:
            self._resolve_combat()
            return

        self.blocker_queue[:] = EMPTY
        self.blocker_queue[: blockers.size] = blockers
        self.blocker_cursor = 0
        self.phase = Phase.DECLARE_BLOCKERS

    def _assign_block(self, attacker_slot: int) -> None:
        defender = 1 - self.active
        blocker_slot = int(self.blocker_queue[self.blocker_cursor])
        self.blocking[defender, blocker_slot] = attacker_slot
        self._advance_blocker()

    def _advance_blocker(self) -> None:
        self.blocker_cursor += 1
        no_more_blockers = (
            self.blocker_cursor >= CREATURE_SLOTS
            or self.blocker_queue[self.blocker_cursor] == EMPTY
        )
        all_attackers_blocked = not (self.attacking[self.active] & ~self._blocked_attackers()).any()
        if no_more_blockers or all_attackers_blocked:
            self._resolve_combat()

    def _blocked_attackers(self) -> np.ndarray:
        blocked = np.zeros(CREATURE_SLOTS, dtype=bool)
        targets = self.blocking[1 - self.active]
        blocked[targets[targets != EMPTY]] = True
        return blocked

    def _resolve_combat(self) -> None:
        me, opp = self.active, 1 - self.active
        for attacker in np.flatnonzero(self.attacking[me]):
            attacker_power = card_db.POWER[self.creature[me, attacker]]
            blockers = np.flatnonzero(self.blocking[opp] == attacker)
            if blockers.size == 0:
                self.life[opp] -= attacker_power
                continue
            blocker = blockers[0]
            self.damage[opp, blocker] += attacker_power
            self.damage[me, attacker] += card_db.POWER[self.creature[opp, blocker]]

        self.attacking[:] = False
        self.blocking[:] = EMPTY
        self._destroy_lethally_damaged()
        if self.life[opp] <= 0:
            self._end_game(winner=me)
            return
        self.phase = Phase.MAIN2

    def _destroy_lethally_damaged(self) -> None:
        occupied = self.creature != EMPTY
        toughness = card_db.TOUGHNESS[np.where(occupied, self.creature, 0)]
        dead = occupied & (self.damage >= toughness)
        self.creature[dead] = EMPTY
        self.damage[dead] = 0
        self.tapped[dead] = False
        self.sick[dead] = False

    # Zones and helpers

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

    def _observe_creatures(self, owner: int, out: np.ndarray, observer: int) -> None:
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

    def _end_game(self, winner: int | None, truncated: bool = False) -> None:
        self._terminal = True
        self._truncated = truncated
        self._winner = winner

    def __repr__(self) -> str:
        return (
            f"GameState(turn={self.turn}, active={self.active}, phase={self.phase.name}, "
            f"life={self.life.tolist()}, hands={self.hand_count.tolist()}, "
            f"lands={self.lands_total.tolist()}, creatures={self.creature.tolist()})"
        )

