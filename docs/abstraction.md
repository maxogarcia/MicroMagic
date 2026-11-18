# Tabular abstraction: abstract state and macro-actions

Code: `src/micromagic/abstraction/`. Tests: `tests/spec/test_abstraction.py`.
The plan (`docs/Aditya_Implementation_Plan.md`, §4) fixed the state buckets and the 9 macros and left the rules below open. **These are proposed defaults for Aditya to confirm or change.**

## Abstract state

`abstract_state(state, player) -> AbstractState`, using only what `player` can see. `.index` is an integer in `[0, NUM_ABSTRACT_STATES)` (11,520) for a Q-table; `AbstractState.from_index` reverses it.

| Feature | Buckets |
| --- | --- |
| Own life, opponent life | ≤5, 6–10, 11–15, 16+ |
| Board power difference (own − opponent, current power) | ≤−4, −3..−1, 0, 1..3, ≥4 |
| Own creatures, opponent creatures | 0, 1, 2, 3+ |
| Cards in hand | 0, 1–2, 3+ |
| **Decision context** (proposed) | pre-combat main, post-combat main, blocking |

## Macro-actions

`macro_mask(state, player)` gives the 9 available macros (all False when it isn't `player`'s decision). The agent decides only in the three contexts above: it never makes a decision inside the attack step or spell targeting, because the macros run those.

| Macro | Available when | What the executor does (proposed rule) |
| --- | --- | --- |
| `PLAY_LAND` | Main phase, land in hand, no land played this turn | Plays it |
| `CAST_BEST_CREATURE` | Main phase, an affordable creature, open slot | Casts the highest mana value creature (ties: bigger power + toughness) |
| `BURN_FACE` | Main phase, an affordable burn that can hit players | Casts the biggest such burn at the opponent |
| `BURN_BEST_CREATURE` | Main phase, an affordable spell kills some opponent creature | Kills the opponent's highest-power killable creature with the cheapest spell that kills it; includes Eviscerate, which is saved for what burn can't kill |
| `ATTACK_ALL` | Pre-combat, any creature can attack | Declares every eligible attacker |
| `ATTACK_SAFE` | Pre-combat, at least one safe attacker | Declares only attackers that no untapped blocker can kill without dying itself (flying, reach, deathtouch, and can't-block counted) |
| `NO_ATTACK` | Always in a main phase | Pre-combat: skips combat. Post-combat: ends the turn, first casting Natural Spring and Divination if affordable (Divination only if both cards fit under the hand cap) |
| `BLOCK_VALUE` | Blocking, and a value or necessary block exists | Strongest attackers first: blocks where the blocker survives, or trades evenly or better by mana value; chump-blocks (cheapest blocker) only while unblocked damage would be lethal |
| `NO_BLOCK` | Always while blocking | No blocks |

Giant Growth is never cast by macro agents. Unblocked-damage estimates ignore trample overflow.

## Using it

`MacroAgent` plays through macros and works anywhere search agents do (`TournamentRunner`, Gym opponent) through `act_from_state`. Override `choose_macro`, or pass `policy=`:

```python
from micromagic.abstraction import MacroAgent, NUM_ABSTRACT_STATES, NUM_MACROS
import numpy as np

class EpsilonGreedyMacroAgent(MacroAgent):
    def __init__(self, epsilon: float, seed: int | None = None) -> None:
        super().__init__(seed=seed, name="mc")
        self.q = np.zeros((NUM_ABSTRACT_STATES, NUM_MACROS))
        self.epsilon = epsilon

    def choose_macro(self, abstract, mask):
        legal = np.flatnonzero(mask)
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        return int(legal[np.argmax(self.q[abstract.index, legal])])
```

After each game, `agent.history` is the list of `(abstract_state_index, macro)` decisions in order, ready for first-visit Monte Carlo returns using the game's ±1 / 0 result. For SARSA, update at each `choose_macro` call from the previous `(state, macro)` pair. `history` resets automatically when the agent sees a new game.

`run_macro(state, player, macro)` applies a whole macro in place and returns the primitive actions taken, for custom training loops.

## Baseline

`MacroAgent(policy=scripted_macro_policy)` follows a fixed priority list: land, removal, creature, safe attack, then pass; blocks by value. Against Random it won 165 of 200 games (seeds 300000–300199, decks and seats alternated; Linux, Python 3.13). A uniformly random choice among available macros won 92 of 200, so learning has room to improve on random play.
