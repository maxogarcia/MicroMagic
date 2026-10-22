# MicroMagic Team Plan

**Team:** Maximilian Garcia (RL Algorithms) · Scott Kennedy (Environment & Rules Engine) · Aditya Hegde (Systems & Evaluation)

**Repo:** https://github.com/maxogarcia/MicroMagic

**Assumed final deadline:** report and presentation due the week of Dec 7. Shift every date if the real deadline differs.

## 1. Review of Deliverable 1 (Check-in 1)

### Fix before submitting (or in the next check-in)

| # | Issue | Fix |
|---|---|---|
| 1 | "Started exploring the data" is a Required item, but the PDF shows no exploration. It lists sources only. | Add the deck-statistics table from `docs/data_exploration.md`: lands/creatures/spells, average CMC, mean P/T, keywords, opening-hand land odds. Add 2–3 takeaways. |
| 2 | No finalized specifications. Rules, decks, action/observation space, and reward are not stated anywhere in the PDF. | Add a short "Finalized Specifications" section or link `docs/checkpoint.md` §1 in the repo. |
| 3 | Abstract says a "20-30 card pool." The spec is fixed at 25 cards, two 30-card decks. | Change to "a 25-card pool and two fixed 30-card archetype decks." |
| 4 | New scope: the abstract adds First-Visit Monte Carlo control and 4-step SARSA. Tabular methods cannot work on the raw ~550-dim observation. | Keep them (they connect the project to course material), but they run on an **abstracted state and macro-action space** (§4.3). Note this in the methods section. |
| 5 | Source "MTG-Casual-RL" has the wrong name. | The paper is **MTG-Causal-RL: Causal Reinforcement Learning for Complex Card Games: A Magic The Gathering Benchmark**, da Costa Cunha, Mian, French, Liu, arXiv:2605.06066 (May 2026). |
| 6 | MTGJSON is described as "Build scripts" and links to the build-scripts repo. | Use: "Open-source card database in JSON; used to cross-check Scryfall data." Link: https://mtgjson.com |
| 7 | 17Lands is listed as if it is used, but its datasets cover recent Arena sets, and most of our 25 cards are not in them. | Mark it "considered, not used," with that reason, or remove it. |
| 8 | The outline has no References section. | Add "12. References". You could also rename "Analysis" to "Analysis & Limitations". |
| 9 | The GitHub repo contains only a README. | Push the local scaffold: card data, fetch and exploration scripts, docs, package layout. |

### Minor

- Source-list formatting is inconsistent ("Link :", missing bullets on PettingZoo/SB3/Ray, http:// for RLCard). Use one format throughout.
- MTG-Causal-RL (masked PPO on MTG) and MageZero (AlphaZero-style on the XMage engine) are our closest prior work. In Related Work, say how we differ: a controlled comparison across tabular, value, policy, and search families on a deliberately small, fully specified game, plus a scaling study.
- The final-report template is headings only, which is fine for now. Add the References heading to it as well.

## 2. Work split

Ownership follows the roles in the proposal. Each person owns their area end to end: code, tests, experiments, and report sections. Every PR is reviewed by one teammate.

| | Scott: Environment & Rules Engine | Max: RL Algorithms | Aditya: Systems & Evaluation |
|---|---|---|---|
| **Code** | Game state, rules, card effects, keywords, combat; PettingZoo AEC env and Gym wrapper; action masking; seeding; `clone()` / `determinize()`; state/macro-action abstraction for the tabular agents | Heuristic baseline agent; masked DQN (SB3 subclass); MaskablePPO; AlphaZero-lite network, training loop, and MCTS integration; self-play opponent pool; hyperparameter tuning | Tournament harness; Elo (Bradley-Terry), Wilson CIs; logging (TensorBoard/CSV); First-Visit MC control and 4-step SARSA; Ray distributed self-play; scaling experiments; reward ablation; all plotting |
| **Tests** | Keyword/spell unit tests; combat edge cases; mask-validity fuzzing (10k random games); determinism; speed benchmark | Heuristic beats Random ≥ 90%; smoke tests: each agent beats Random on a toy env; masking never selects illegal actions | Harness reproducibility (same seeds → same results); Elo sanity checks on synthetic data |
| **Report sections** | 4 Environment of MicroMagic; Appendix A–B (rules, card pool) | 2 Introduction; 3 Background & Related Work; 5 Methodology (deep RL + AZ-lite) | 6 Distributed Self-Play; 7 Experimental Setup; 8 Results (figures); 5 Methodology (tabular subsection) |
| **Shared** | 1 Abstract, 9 Analysis, 10 Conclusion, presentation slides: all three, final edit by Max | | |

**Why tabular MC/SARSA goes to Aditya:** these agents are simple and can be trained as soon as the env exists. That makes them the first real agents to run through the tournament harness, so they validate the harness before the deep-RL agents arrive. They also keep Max's load manageable, since Max already owns three algorithms.

**Why Scott helps with AlphaZero-lite:** MCTS depends on fast `clone()` and on sampling the hidden state (`determinize()`). Both are engine internals.

**Why the heuristic baseline goes to Max:** it is a rule-based agent, so it sits with the other agents. Max also needs to know exactly how strong it is, since it is the main benchmark the deep-RL agents train against. Moving it (and the reward ablation, to Aditya) evens out the workload; see §2.1.

### 2.1 Workload estimate

Rough hour estimates for a team new to these libraries. They are for balancing the work, not for tracking time. Total is about 286 hours, or roughly 13–14 hours per person per week over 7 weeks.

| Scott (Env) | hrs | Max (Algorithms) | hrs | Aditya (Systems/Eval) | hrs |
|---|---:|---|---:|---|---:|
| Interface contract + stub env | 6 | Agent base, Random agent, PPO wiring | 6 | CI, issue board, logging | 6 |
| Engine core (turns, mana, combat) | 20 | Heuristic baseline agent | 8 | Tournament harness | 6 |
| 7 keywords + 5 spell templates | 14 | Masked DQN subclass | 10 | Elo, Wilson CIs, head-to-head matrix | 8 |
| Unit tests, mask fuzzer, determinism | 10 | PPO/DQN training + opponent pool | 16 | Tabular state/macro-action abstraction | 8 |
| PettingZoo/Gym wrappers + masking | 8 | AlphaZero-lite network + self-play loop | 16 | First-Visit MC control + 4-step SARSA | 12 |
| Speed work + `clone()`/`determinize()` | 10 | MCTS integration (shared with Scott) | 6 | Environment bias check | 3 |
| MCTS core (shared with Max) | 10 | Hyperparameter tuning | 10 | Ray rollout workers | 12 |
| Game logger/replayer | 4 | Final 3-seed runs | 6 | Ray AZ actors + scaling runs | 12 |
| Help with tabular abstraction | 3 | | | Reward ablation | 4 |
| | | | | Final tournament + figures | 10 |
| Report §4, appendix, share of joint sections | 10 | Report §2, §3, §5, final edit, share of joint sections | 18 | Report §6–8, share of joint sections | 14 |
| **Total** | **95 (33%)** | **Total** | **96 (34%)** | **Total** | **95 (33%)** |

Before rebalancing (heuristic baseline with Scott, reward ablation with Scott), the split was about 37% / 31% / 32%. Moving those two tasks evened it out.

**Caveats:**

- Uncertainty is not equal. Scott's engine work is the most predictable: the rules are known and each piece can be unit-tested. Max's hours are the most likely to grow, because RL training often fails for unclear reasons (DQN on a sparse reward especially). That is why §6 caps DQN debugging at one week. Treat Max's 34% as a floor.
- The load shifts over time. Scott is heaviest in the first three weeks, while the engine blocks everyone. Max and Aditya are heaviest Nov 9–29. If Scott finishes early, Scott picks up engine bugs found during training and helps with the MCTS work.
- Re-check at M2 (Nov 8). Compare actual progress against this table and move tasks if one person is more than a week behind.

## 3. Interface contract (agree by Wed Oct 21)

This removes the main bottleneck: Max and Aditya can build against the interface before the engine is finished.

```python
# micromagic/env/state.py (Scott)
class GameState:
    def legal_actions(self) -> np.ndarray:        # bool[29] mask
    def step(self, action: int) -> None:
    def current_player(self) -> int:              # 0/1, the player who must act
    def observe(self, player: int) -> np.ndarray:  # float32[OBS_DIM], hides opponent's hand
    def is_terminal(self) -> bool:
    def winner(self) -> int | None:                # 0, 1, or None for a draw
    def clone(self) -> "GameState":                # fast copy, for MCTS
    def determinize(self, player: int, rng) -> "GameState":  # resample hidden info consistent with `player`'s view
```

```text
# micromagic/env/aec.py          -> PettingZoo AEC env (obs dict: {"observation", "action_mask"})
# micromagic/env/gym_env.py      -> Gymnasium wrapper(opponent_policy) (info["action_mask"])
```

```python
# micromagic/agents/base.py (Max)
class Agent:
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int: ...
    def save(self, path): ...
    def load(self, path): ...
```

- **Stub env by Oct 21 (Scott):** an env that satisfies the API but plays a trivial game, e.g. "first to deal 20 damage, lands and vanilla creatures only." Max and Aditya develop against it.
- Constants such as `OBS_DIM`, action indices, and the card-id map go in `micromagic/env/spec.py`. Changing them requires a PR reviewed by all three.

## 4. Build plan

### 4.1 Timeline

| Week of | Scott (Env) | Max (Algorithms) | Aditya (Systems/Eval) | Team milestone |
|---|---|---|---|---|
| **Oct 19** | Interface contract + stub env (Wed). Engine core: zones, turn structure, mana, lands, vanilla creatures, combat without keywords | Agent base class; Random agent; wire MaskablePPO to the stub env | Repo hygiene: CI with pytest via GitHub Actions, issue board, logging utils; harness skeleton (play N games between two agents, record results) | **M0:** API frozen, CI green, PPO runs end to end on the stub |
| **Oct 26** | All 7 keywords + can't-block; 5 spell templates; unit tests; mask fuzzer; determinism test | Masked-DQN subclass (mask in ε-greedy and in the target max); heuristic baseline v1 (rules from §1.5 of the check-in; Scott reviews) | Elo fit + bootstrap CIs, Wilson CIs, head-to-head matrix; tabular state/macro-action abstraction (with Scott) | **M1 (Nov 1):** full engine v1 passes tests; Random/Heuristic tournament runs |
| **Nov 2** | Speed benchmark and optimization (target ≥ 2k steps/s per core); `clone()`/`determinize()` | Heuristic baseline final; PPO and DQN training vs. Random → Heuristic; first learning curves; opponent pool (snapshot every K updates) | First-Visit MC control + 4-step SARSA on the abstracted space; environment bias check (Random-vs-Random, Heuristic-vs-Heuristic per deck/seat) | **M2 (Nov 8):** tabular, PPO, and DQN agents all beat Random; bias numbers known |
| **Nov 9** | MCTS core (PUCT, PIMC over K determinizations), with Max | AlphaZero-lite network + self-play training loop; PPO/DQN hyperparameter sweeps (small) | Ray: parallel rollout workers for PPO/DQN; throughput logging | **M3 (Nov 15):** AZ-lite trains; Ray speedup measured on 1–4 workers |
| **Nov 16** | Engine bug fixes; example-game logger/replayer for qualitative analysis | AZ-lite tuning (N sims, K determinizations); final configs; freeze agents Nov 22 | Ray self-play actors for AZ-lite; scaling runs 1/2/4/8/16 workers; final full tournament script | **M4 (Nov 22):** code freeze for experiments |
| **Nov 23 (Thanksgiving)** | Engine support for final runs; start drafting §4 | Final 3-seed training runs at the equal budget | Run the final tournament (1,000 games/pair); reward ablation (shaped vs. terminal); generate all figures and tables | **M5 (Nov 29):** all results in |
| **Nov 30** | Write §4 + appendix | Write §2, §3, §5; final edit | Write §6–8; figures | **M6:** full draft by Wed Dec 2; slides Dec 4; submit |

**Slack:** Thanksgiving week is lightly loaded, so that it can absorb a slipped week.

### 4.2 Definition of done per milestone

- **M1:** 100% of the card pool has unit tests. The mask fuzzer finds 0 illegal actions in 10k games. The same seed gives the same game.
- **M2:** each learning agent beats Random at ≥ 70% over 500 games with fixed seeds.
- **M4:** every agent loads from a checkpoint with a single config file. The tournament runs from one command.

### 4.3 Tabular agents (First-Visit MC control, 4-step SARSA)

- **State abstraction (owner: Aditya, with Scott):**
  - Own life bucket {≤5, 6–10, 11–15, 16+} and opponent life bucket (same buckets)
  - Board power difference {≤−4, −3..−1, 0, 1..3, ≥4}
  - Own and opponent creature counts {0, 1, 2, 3+}
  - Cards in hand {0, 1–2, 3+}
  - Decision context
- **Macro-actions (~8):** `play_land`, `cast_best_creature`, `burn_face`, `burn_best_creature`, `attack_all`, `attack_safe`, `no_attack`, `block_value`, `no_block`. A scripted executor expands each one into primitive actions. Unavailable macros are masked.
- **First-Visit MC control:** on-policy, ε-soft (Sutton & Barto §5.4).
- **4-step SARSA:** on-policy n-step SARSA (S&B §7.2), with ε decaying.
- **Opponents:** Random, then Heuristic (fixed opponents, no self-play). These count as baselines, not competitors. They are included in the tournament and the Elo table.

### 4.4 Compute budget

- **Target:** 5M agent steps × 3 seeds for PPO, DQN, and AZ-lite. AZ-lite gets the same number of real env steps; simulated MCTS steps are reported separately as compute.
- **If it does not fit:** drop to 2M steps before dropping seeds. Report the actual budget used.
- **Hardware:** team laptops, plus the SJSU HPC if we get access. Request it now (week of Oct 19).

## 5. Working agreement

- **Branches:** `main` is protected. Use feature branches named `scott/combat-keywords` and so on. PRs need 1 review and green CI.
- **Issues:** one GitHub issue per task in §4.1, labeled by milestone (M0–M6) and owner.
- **Sync:** 30 minutes twice a week (suggest Mon and Thu). Each person covers what they did, what's next, and what's blocking them. Notes go in Drive → CMPE 260 / Project / Meeting Notes.
- **Experiments:** every run is launched from a committed config (`configs/*.yaml`) with a seed. Results go to `runs/` (gitignored); final CSVs are copied to `results/` and committed.
- **Report:** one Google Doc in the Project folder. Each section owner drafts their part in it from week Nov 23 onward. Keep figures as PNG + source CSV.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Engine slips → everyone blocked | Stub env + frozen API by Oct 21; toy mode (lands + vanilla creatures) is useful on its own |
| Python engine too slow for MCTS | Benchmark in week of Nov 2; custom `clone()` (no deepcopy); cut N sims/K determinizations; report cost honestly |
| DQN fails to learn with a sparse reward | Use dueling + n-step returns. If it is still flat, report it as a finding (value-based methods struggle here). Do not spend more than a week on it. |
| Ray complexity eats time | Fall back to SubprocVecEnv / multiprocessing for training. Ray is used only for the scaling study. |
| One deck/seat dominates | Bias check in M2; report results per deck/seat; adjust a deck list once, before M4, if win rate is > 70% for Heuristic-vs-Heuristic |
