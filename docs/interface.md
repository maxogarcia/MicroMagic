# MicroMagic Environment Interface (v0.1, frozen Oct 21)

Constants live in `micromagic/env/spec.py`. Changing them needs a PR reviewed by all three of us.

## Action space (29 discrete actions)

| Index | Constant | Meaning | Legal when |
| --- | --- | --- | --- |
| 0 | `PASS` | End main phase / done declaring attackers / don't block with this creature / end turn | Always (until the game ends) |
| 1–7 | `PLAY_HAND` | Play the card in my hand slot 0–6 | Main phase; a land if none played this turn, or a creature I can pay for with an open creature slot |
| 8–12 | `ATTACK` | Declare my creature slot 0–4 as an attacker | Declare attackers; creature untapped, not summoning-sick, not already attacking |
| 13–17 | `BLOCK` | The current blocker blocks the attacker in opponent slot 0–4 | Declare blockers (defender acts); that attacker is attacking and not yet blocked |
| 18 | `TARGET_OPPONENT` | Target the opponent | Reserved for spells, always masked in v0.1 |
| 19–23 | `TARGET_OPP_CREATURE` | Target opponent creature slot 0–4 | Reserved for spells, always masked in v0.1 |
| 24–28 | `TARGET_MY_CREATURE` | Target my creature slot 0–4 | Reserved for spells, always masked in v0.1 |

Multi-part choices are sequences of single actions. Blocking walks through the defender's untapped creatures one at a time; for each, the defender picks `BLOCK[k]` or `PASS`. During blocks, `current_player()` is the defender, so turns do not strictly alternate.

## Turn structure

Untap and draw (automatic) → `MAIN1` → `DECLARE_ATTACKERS` → `DECLARE_BLOCKERS` → combat damage (automatic) → `MAIN2` → end turn.
Combat is skipped when the active player has no creature able to attack, and blocks are skipped when the defender has no untapped creature.

## Rules (toy mode)

- 20 life, 30-card decks, opening hand of 7, no mulligans. Player 0 goes first and skips the first draw.
- Hand capped at 7: a player holding 7 skips their draw. Drawing from an empty library loses.
- One land per turn. Mana is generic: each untapped land pays 1.
- Summoning sickness; attacking taps the creature; at most one blocker per attacker.
- Creatures with damage ≥ toughness die; damage wears off at end of turn.
- Win at opponent life ≤ 0. After 50 turns the game is a draw (truncation).

## Observation (`float32[622]`, all values in [0, 1], relative to the observer)

| Slice | Size | Contents |
| --- | --- | --- |
| `OBS_GLOBAL` | 18 | Life (both), turn, phase one-hot (5), is-my-turn, lands total/untapped (both), land played this turn, hand counts (both), library counts (both) |
| `OBS_HAND` | 7 × 32 | Per hand slot: present flag + card features |
| `OBS_MY_CREATURES` | 5 × 38 | Per slot: present flag + card features + damage, tapped, summoning-sick, attacking, blocking, blocked |
| `OBS_OPP_CREATURES` | 5 × 38 | Same, for the opponent's creatures |

Card features (31) = card-id one-hot (25) + cost, power, toughness (÷10) + is-land, is-creature, is-spell. The opponent's hand is only visible as a count.

## Wrappers

- `micromagic.env.aec.MicroMagicAECEnv`: PettingZoo AEC env, observations are `{"observation", "action_mask"}`, rewards ±1 at game end, 0 for a draw.
- `micromagic.env.gym_env.MicroMagicGymEnv(opponent_policy, agent_player)`: Gymnasium single-agent env; mask in `info["action_mask"]` and `action_masks()` (for sb3-contrib MaskablePPO). The default opponent plays uniformly random legal actions.

## Agents (`micromagic/agents/`)

- `Agent` (`agents/base.py`): `name`, `act(obs, mask) -> int`, `reset()`, `save(path)`, `load(path)`. `obs` is `GameState.observe(player)` for the acting seat and `mask` is `legal_actions()`; `act` must return an index with `mask[action]` true. An `Agent` is callable, so it can be passed directly as `MicroMagicGymEnv(opponent_policy=agent)`. It satisfies the harness's `tournaments.AgentProtocol`.
- `RandomAgent(seed)`: uniform over legal actions; the same seed gives the same choices.
- `MaskablePPOAgent` (`agents/ppo.py`, needs the `rl` extra): wraps an sb3-contrib `MaskablePPO` model. `MaskablePPOAgent.from_checkpoint("runs/ppo/<run>/model.zip")` loads a trained policy; `deterministic=True` (default) plays the most likely legal action.

Training: `python -m micromagic.training.train_ppo --config configs/ppo_stub.yaml`. Half the training envs seat the learner first and half second. After training it plays `eval.games` games against the eval opponent through `TournamentRunner` (alternating seats) and writes `eval_matches.csv` and `eval_summary.csv` (Wilson CIs) next to `model.zip` and SB3's `progress.csv`.
