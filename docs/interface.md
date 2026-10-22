# MicroMagic Environment Interface (v1, M1)

Constants live in `src/micromagic/env/spec.py`. Changing them needs a PR reviewed by all three of us.

**Changed since v0.1:** `OBS_DIM` is now 925 (was 622). Card features gained keyword flags, a spell-template one-hot, and the spell amount; creature slots gained current power/toughness; a pending-spell block was added. The action space is unchanged.

## Action space (29 discrete actions)

| Index | Constant | Meaning | Legal when |
| --- | --- | --- | --- |
| 0 | `PASS` | End main phase / done declaring attackers / don't block with this creature / end turn | Every phase except `CHOOSE_TARGET` |
| 1–7 | `PLAY_HAND` | Play the card in my hand slot 0–6 | Main phase; a land if none played this turn, a creature I can pay for with an open slot, or a spell I can pay for that has a legal target |
| 8–12 | `ATTACK` | Declare my creature slot 0–4 as an attacker | Declare attackers; untapped, not already attacking, and not summoning-sick unless it has haste |
| 13–17 | `BLOCK` | The current blocker blocks the attacker in opponent slot 0–4 | Declare blockers (defender acts); that attacker is unblocked and this blocker can block it (flying needs flying or reach) |
| 18 | `TARGET_OPPONENT` | Target the opponent | `CHOOSE_TARGET`, for burn spells |
| 19–23 | `TARGET_OPP_CREATURE` | Target opponent creature slot 0–4 | `CHOOSE_TARGET`, for burn, destroy, and pump |
| 24–28 | `TARGET_MY_CREATURE` | Target my creature slot 0–4 | `CHOOSE_TARGET`, for burn, destroy, and pump |

Multi-part choices are sequences of single actions:
- **Casting a targeted spell** is `PLAY_HAND[i]` (mana is paid), then one `TARGET_*` action. The spell can't be cancelled, and it is only castable when a legal target exists.
- **Blocking** walks through the defender's creatures that can legally block something, one at a time. For each, the defender picks `BLOCK[k]` or `PASS`. During blocks, `current_player()` is the defender, so turns don't strictly alternate.

## Turn structure

Untap and draw (automatic) → `MAIN1` → `DECLARE_ATTACKERS` → `DECLARE_BLOCKERS` → combat damage (automatic) → `MAIN2` → end turn.
`CHOOSE_TARGET` interrupts either main phase and returns to it.
Combat is skipped when no creature can attack, and blocks are skipped when no defender can legally block.

## Rules

- 20 life, two fixed 30-card decks (`aggro`, `midrange`), opening hand of 7, no mulligans. The first player skips their first draw.
- Hand capped at 7: a player holding 7 skips their draw (Divination draws only what fits). Drawing from an empty library loses.
- One land per turn. Mana is generic: each untapped land pays 1 of a card's mana value.
- Sorcery speed only, no first strike, at most one blocker per attacker, combat damage is simultaneous.
- Creatures with damage ≥ toughness die, or with any damage from a deathtouch source. Damage and pump effects wear off at end of turn.
- Win when the opponent's life is ≤ 0 (both at once is a draw). After 50 turns the game is a draw (truncation).

### Keywords

| Keyword | Rule |
| --- | --- |
| Flying | Can only be blocked by creatures with flying or reach |
| Reach | Can block flyers |
| Haste | Can attack the turn it enters |
| Trample | Assigns lethal damage to its blocker (1 with deathtouch), the rest to the player |
| Vigilance | Doesn't tap to attack |
| Deathtouch | Any damage it deals to a creature is lethal |
| Lifelink | Damage it deals gains its controller that much life |
| Can't block | Never offered as a blocker |

### Spell templates

| Template | Effect | Targets |
| --- | --- | --- |
| Burn | Deal `amount` damage | Opponent or any creature (`any`), or opponent only (`player`) |
| Destroy | Destroy the creature | Any creature |
| Pump | +`amount`/+`amount` until end of turn | Any creature |
| Draw | Draw `amount` cards | None |
| Gain life | Gain `amount` life | None |

## Card pool and decks

Defined in `src/micromagic/data/cards.json`; ids match `spec.CARD_IDS`.

| Id | Card | Mana value | Type | Stats / effect | Aggro | Midrange |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Mountain | 0 | Land | | 12 | |
| 1 | Forest | 0 | Land | | | 13 |
| 2 | Raging Goblin | 1 | Creature | 1/1 haste | 3 | |
| 3 | Suntail Hawk | 1 | Creature | 1/1 flying | 2 | |
| 4 | Goblin Piker | 2 | Creature | 2/1 can't block | 3 | |
| 5 | Wind Drake | 3 | Creature | 2/2 flying | 1 | |
| 6 | Gray Ogre | 3 | Creature | 2/2 | 1 | |
| 7 | Hill Giant | 4 | Creature | 3/3 | 2 | |
| 8 | Charging Monstrosaur | 5 | Creature | 5/5 trample, haste | 1 | |
| 9 | Shock | 1 | Burn | 2 damage, any target | 2 | |
| 10 | Volcanic Hammer | 2 | Burn | 3 damage, any target | 1 | |
| 11 | Lava Axe | 5 | Burn | 5 damage, opponent only | 1 | |
| 12 | Giant Growth | 1 | Pump | +3/+3 | 1 | |
| 13 | Typhoid Rats | 1 | Creature | 1/1 deathtouch | | 1 |
| 14 | Healer's Hawk | 1 | Creature | 1/1 flying, lifelink | | 1 |
| 15 | Grizzly Bears | 2 | Creature | 2/2 | | 2 |
| 16 | Deadly Recluse | 2 | Creature | 1/2 reach, deathtouch | | 2 |
| 17 | Child of Night | 2 | Creature | 2/1 lifelink | | 1 |
| 18 | Trained Armodon | 3 | Creature | 3/3 | | 2 |
| 19 | Giant Spider | 4 | Creature | 2/4 reach | | 2 |
| 20 | Eviscerate | 4 | Destroy | Destroy target creature | | 2 |
| 21 | Divination | 3 | Draw | Draw 2 | | 1 |
| 22 | Serra Angel | 5 | Creature | 4/4 flying, vigilance | | 1 |
| 23 | Natural Spring | 5 | Gain life | Gain 8 | | 1 |
| 24 | Colossal Dreadmaw | 6 | Creature | 6/6 trample | | 1 |

`GameState(seed, decks=("aggro", "midrange"), first_player=0)`: decks are deck names or arrays of card ids, indexed by seat.

## Observation (`float32[925]`, all values in [0, 1], relative to the observer)

| Slice | Size | Contents |
| --- | --- | --- |
| `OBS_GLOBAL` | 18 | Life (both), turn, phase one-hot (5), is-my-turn, lands total/untapped (both), land played this turn, hand counts (both), library counts (both) |
| `OBS_PENDING_SPELL` | 45 | Card features of the spell waiting for a target (zeros outside `CHOOSE_TARGET`) |
| `OBS_HAND` | 7 × 46 | Per hand slot: present flag + card features |
| `OBS_MY_CREATURES` | 5 × 54 | Per slot: present flag + card features + damage, tapped, summoning-sick, attacking, blocking, blocked, current power, current toughness |
| `OBS_OPP_CREATURES` | 5 × 54 | Same, for the opponent's creatures |

Card features (45) = card-id one-hot (25) + mana value, power, toughness (÷10) + is-land, is-creature, is-spell + keyword flags (8, in `spec.Keyword` order) + spell-template one-hot (5) + spell amount (÷10). Life above 20 is clipped to 1.0. The opponent's hand is only visible as a count.

## Wrappers

- `micromagic.env.aec.MicroMagicAECEnv`: PettingZoo AEC env; observations are `{"observation", "action_mask"}`; rewards ±1 at game end, 0 for a draw.
- `micromagic.env.gym_env.MicroMagicGymEnv(opponent_policy, agent_player, on_illegal="raise")`: Gymnasium single-agent env; the mask is in `info["action_mask"]` and `action_masks()` (for sb3-contrib MaskablePPO). The default opponent plays uniformly random legal actions. `on_illegal="pass"` replaces an illegal action with `PASS`, or with the first legal action during `CHOOSE_TARGET`.

## Testing

- `pytest` runs the contract, rules, per-card, and wrapper tests, plus a 300-game mask fuzz (set `MICROMAGIC_FUZZ_GAMES` to change it).
- `python scripts/fuzz_masks.py --games 10000 --seed 0` is the M1 acceptance run.
