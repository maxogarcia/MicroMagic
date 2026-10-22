# MicroMagic — Aditya's Systems & Evaluation Handoff

> **Purpose:** Implementation-oriented handoff for coding agents working on Aditya Hegde's portion of the CMPE 260 MicroMagic project. This document reorganizes the _MicroMagic Team Plan_ (5-page source PDF); it is **not** a claim that any items have already been implemented.
>
> **Repository:** https://github.com/maxogarcia/MicroMagic  
> **Owner:** Aditya Hegde — Systems & Evaluation  
> **Teammates:** Scott Kennedy — Environment & Rules Engine; Maximilian Garcia (Max) — RL Algorithms  
> **Planning baseline:** October 19–December 4, 2026; final report/presentation assumed due in the week of December 7. **Verify actual deadlines.**

## 0. Instructions for a coding agent

1. **Inspect the repository before editing anything.** Map existing code, tests, configs, and branches against the responsibilities and checklist in `docs/Aditya_Implementation_Plan.md`. Do not assume a feature is missing or complete solely because it is mentioned here.
2. **Implement only work in Aditya's ownership area** unless explicitly asked to contribute elsewhere. Preserve Scott's environment API and Max's agent interface. For shared interfaces, coordinate before changing signatures or constants.
3. **Work incrementally:** choose the earliest uncompleted, unblocked milestone; write tests; implement; run tests; document results and blockers.
4. **Do not fabricate measurements.** Any Elo scores, confidence intervals, throughput, speedups, win rates, learning curves, or figures must be derived from actual experiments. Record seeds, configs, sample counts, machine details when relevant, and output paths.
5. **Prefer reproducibility:** run every experiment from a committed `configs/*.yaml` configuration with explicit seed(s). Treat `runs/` as gitignored local output; commit final CSVs into `results/`.
6. **Before claiming done**, report changed files, commands executed, test results, generated artifacts, known issues, and next blocking dependency. Do not merge or push without instruction.
7. **Treat dates as the original plan, not verified present-day status.** The schedule below is a baseline and should be revised against repository state and teammate updates.

## 1. Scope and ownership

### Aditya owns end to end

- Continuous integration / repository issue-board setup and logging utilities (TensorBoard/CSV).
- Reproducible **tournament harness** to run N matches between two agents and record outcomes.
- Evaluation and statistics: **Elo via Bradley–Terry**, Elo fit with bootstrap confidence intervals, **Wilson confidence intervals** for win proportions, and pairwise head-to-head results matrix.
- **Tabular state abstraction** and **macro-action abstraction** (in collaboration with Scott).
- **First-Visit Monte Carlo control** and **4-step SARSA** on that abstracted representation.
- **Environment bias check** by deck and seat.
- **Ray distributed rollout workers** for PPO/DQN; throughput instrumentation.
- **Ray self-play actors** for AlphaZero-lite; 1/2/4/8/16-worker scaling experiments.
- **Reward ablation**: shaped versus terminal reward.
- Final full tournament, experiment summaries, tables and figures.
- Final report: **Section 5 tabular methodology subsection**, **Section 6 Distributed Self-Play**, **Section 7 Experimental Setup**, **Section 8 Results**, plus shared sections and slides.

### Other owners — interfaces/dependencies, not Aditya's primary tasks

| Owner     | Owns                                                                                                                                                                                                              | Needed by Aditya                                                                                                                    |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Scott** | Game state/rules, card effects, keywords, combat; PettingZoo AEC and Gymnasium wrappers; action masking; seeding; `clone()` and `determinize()`; engine speed and engine tests; support for abstract state/macros | Stub/full environment, stable observation/action schema, mask correctness, abstract-state feature support and executable primitives |
| **Max**   | `Agent` base; Random and Heuristic agents; MaskablePPO; masked DQN; AlphaZero-lite, MCTS integration with Scott; opponent pool, training/tuning, final trained checkpoints                                        | Standard agent API, Random/Heuristic opponents, training integration points, reproducible checkpoints and configs                   |

**Shared team deliverables:** Abstract (Section 1), Analysis (Section 9), Conclusion (Section 10), and presentation slides (final slide edit by Max). Every pull request requires one teammate review and passing CI.

### Original workload estimate (planning, not time tracking)

| Aditya task                             | Estimated hours |
| --------------------------------------- | --------------: |
| CI, issue board, logging                |               6 |
| Tournament harness                      |               6 |
| Elo, Wilson CIs, head-to-head matrix    |               8 |
| Tabular state/macro-action abstraction  |               8 |
| First-Visit MC control and 4-step SARSA |              12 |
| Environment bias check                  |               3 |
| Ray rollout workers                     |              12 |
| Ray AZ actors and scaling runs          |              12 |
| Reward ablation                         |               4 |
| Final tournament and figures            |              10 |
| Report Sections 6–8 and shared writing  |              14 |
| **Total**                               |          **95** |

## 2. Interface contract: build against this before engine completion

Source plan targets interface agreement and stub environment by **Wednesday, October 21**. Confirm its actual implementation before relying on it.

```python
# micromagic/env/state.py — owned by Scott
class GameState:
    def legal_actions(self) -> np.ndarray: ...  # bool[29] mask
    def step(self, action: int) -> None: ...
    def current_player(self) -> int: ...          # 0/1; player whose turn it is
    def observe(self, player: int) -> np.ndarray: ...  # float32[OBS_DIM]; hide opposing hand
    def is_terminal(self) -> bool: ...
    def winner(self) -> int | None: ...           # 0, 1, or None for a draw
    def clone(self) -> "GameState": ...           # fast copy for MCTS
    def determinize(self, player: int, rng) -> "GameState": ...

# micromagic/env/aec.py — owned by Scott
# PettingZoo AEC env with observation dict:
# {"observation": ..., "action_mask": ...}

# micromagic/env/gym_env.py — owned by Scott
# Gymnasium wrapper(opponent_policy); info["action_mask"]

# micromagic/agents/base.py — owned by Max
class Agent:
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int: ...
    def save(self, path): ...
    def load(self, path): ...
```

- Constants, observation dimensions, action indices, and card-ID mapping belong in `micromagic/env/spec.py` and **require review from all three teammates** if changed.
- Scott's stub should satisfy the API even if it implements a trivial toy game (lands, vanilla creatures; first to 20 damage). Aditya should develop the harness against this stub rather than wait for the full engine.
- **Integration clarification needed:** The plan's base `Agent.act(obs, mask)` interface does not include explicit opponent state, current player, or recurrent agent state. Investigate the repository's actual adapters before defining tournament and macro-action behavior.

## 3. Execution plan — Aditya's tasks in dependency order

### M0 — Week of October 19: engineering foundation

**Deliverables**

- [ ] Review repository structure, existing test suite, workflows, and `configs/` conventions.
- [ ] Configure GitHub Actions to run `pytest` on pull requests; verify a green run.
- [ ] Set up GitHub issue board: issue per task, tagged by milestone `M0`–`M6` and owner.
- [ ] Create standardized experiment logging utilities for CSV/TensorBoard.
- [ ] Implement an initial tournament harness that runs **N games between two agents** and records individual results (initially using the stub environment).
- [ ] Add deterministic seed handling and a replayable, machine-readable result schema.
- [ ] Add tests demonstrating **same seeds + same agents/configs produce the same results** (subject to determinism of environment and agents).

**Acceptance:** CI green; harness can run a toy match end-to-end, persist outcomes and repeat results under fixed seeds. Team M0 also targets a frozen API and PPO functioning against the stub.

### M1 — Week of October 26; milestone November 1: evaluation methods and abstraction

**Deliverables**

- [ ] Compute head-to-head match results and pairwise win/loss/draw matrix.
- [ ] Implement Bradley–Terry-based Elo fitting and **bootstrap confidence intervals**.
- [ ] Implement **Wilson binomial proportion confidence intervals**; define how draws are represented and whether they are excluded or assigned fractional scores before reporting intervals.
- [ ] Add tests for Elo behavior on synthetic comparison data (e.g., clearly stronger vs weaker agent, reversed ordering, symmetric results).
- [ ] Implement Aditya's part of tabular abstract state extraction and available macro-action masks **with Scott**.
- [ ] Validate Random-versus-Heuristic tournament integration once Max's baseline is ready.

**Acceptance:** Statistical output generated from recorded games and tested on synthetic cases; abstract-state and macro APIs agreed with Scott. Team M1 expects full engine v1 tested and a Random/Heuristic tournament.

### M2 — Week of November 2; milestone November 8: tabular agents and bias measurement

**Deliverables**

- [ ] Implement **First-Visit MC control** (on-policy, epsilon-soft; Sutton & Barto §5.4) in the _abstract_ state/action space.
- [ ] Implement **4-step SARSA** (on-policy n-step SARSA with epsilon decay; Sutton & Barto §7.2), again in abstract space.
- [ ] Train against fixed **Random**, then fixed **Heuristic**, **not self-play**.
- [ ] Expose agents through the tournament harness; include them in the same evaluation/Elo tables as other methods, while describing them as **baselines** rather than primary competitors.
- [ ] Run environment-bias checks: Random-vs-Random and Heuristic-vs-Heuristic, separated by **deck and seat**.
- [ ] Record bias diagnostics; if Heuristic-vs-Heuristic reveals a win rate over 70% for one deck/seat, the plan allows adjusting a deck list **once**, before M4, coordinated with the team.

**Acceptance:** Team M2: **each learning agent beats Random at least 70% over 500 fixed-seed games**; bias numbers known. Do not claim this target is met without actual runs. Reassess task allocation with team on November 8.

### M3 — Week of November 9; milestone November 15: distributed PPO/DQN rollouts

**Deliverables**

- [ ] Add Ray-powered parallel rollout workers compatible with Max's PPO/DQN pipeline.
- [ ] Collect throughput and elapsed-time metrics with 1–4 workers.
- [ ] Check seed independence and deterministic reproduction where feasible.
- [ ] Log machine, worker counts, total transitions, wall-clock time, failures and effective throughput.
- [ ] Provide a non-Ray fallback integration where feasible: `SubprocVecEnv` or Python multiprocessing. If Ray is too complex, plan permits using Ray only for the scaling study.

**Acceptance:** Reproducible throughput measurements on 1–4 workers. Team M3 includes AZ-lite training.

### M4 — Week of November 16; milestone November 22: AZ-lite scaling and freeze

**Deliverables**

- [ ] Add distributed **Ray self-play actors for AlphaZero-lite**, working with Max's model/learner and Scott's MCTS engine as delivered.
- [ ] Benchmark **1 / 2 / 4 / 8 / 16** worker configurations where resources permit.
- [ ] Measure throughput and scaling speedup (with hardware and run definitions recorded).
- [ ] Complete one-command final-tournament runner, checkpoint discovery/loading and config handling.
- [ ] Agree and freeze experimental configurations by **November 22**.

**Acceptance:** Team M4: each agent loads from a checkpoint with **a single config file** and the tournament launches from **one command**. Code freeze for experiments.

### M5 — Week of November 23; milestone November 29: final experimental results

**Deliverables**

- [ ] Run final round-robin/head-to-head tournament at **1,000 games per pair**.
- [ ] Run **shaped reward vs terminal-only reward** ablation, holding the comparison setup as consistent as possible; coordinate with Max if agents require retraining.
- [ ] Aggregate win rates, Wilson CIs, Elo estimates/bootstrap CIs, head-to-head matrix and deck/seat-specific outcomes.
- [ ] Produce finalized tables and plots from committed source CSVs. Include distributed throughput/speedup figures.
- [ ] Check that all reported results have seeds, config references, sample sizes and error/uncertainty measures as applicable.

**Acceptance:** All experimental results present by November 29; figures can be regenerated from recorded outputs.

### M6 — Week of November 30: report and slides

**Deliverables**

- [ ] Write Section **5** subsection on tabular algorithms and abstraction.
- [ ] Write Section **6 — Distributed Self-Play**.
- [ ] Write Section **7 — Experimental Setup**.
- [ ] Write Section **8 — Results**, with reproducible figures.
- [ ] Contribute to Section **1 Abstract**, Section **9 Analysis**, Section **10 Conclusion**, and presentation slides.
- [ ] Keep each figure as a PNG and its underlying source CSV.

**Deadlines from plan:** complete draft **Wednesday December 2**; slides **December 4**; assumed report/presentation due week of December 7 (verify official date).

## 4. Tabular state abstraction specification

The original plan explicitly rejects applying tabular MC/SARSA directly to the raw observation (approximately 550 dimensions). Aditya and Scott need an abstract state plus scripted macro-action executor.

### Features named in the source

| Feature                 | Buckets / possible values                                                             |
| ----------------------- | ------------------------------------------------------------------------------------- |
| Own life                | `<=5`, `6–10`, `11–15`, `16+`                                                         |
| Opponent life           | same as own life                                                                      |
| Board power difference  | `<=-4`, `-3..-1`, `0`, `1..3`, `>=4`                                                  |
| Own creature count      | `0`, `1`, `2`, `3+`                                                                   |
| Opponent creature count | `0`, `1`, `2`, `3+`                                                                   |
| Cards in hand           | `0`, `1–2`, `3+`                                                                      |
| Decision context        | **Listed in plan but not defined further; resolve with Scott before locking schema.** |

### Macro-actions

The PDF describes “~8” macro-actions but **lists nine**:

1. `play_land`
2. `cast_best_creature`
3. `burn_face`
4. `burn_best_creature`
5. `attack_all`
6. `attack_safe`
7. `no_attack`
8. `block_value`
9. `no_block`

**Contract:** scripted executor maps the selected macro-action to legal primitive game actions. Macros unavailable in the current state are masked. The exact tie-breaking/evaluation heuristics, turn-phase mapping, macro termination conditions and decision-context buckets **are unspecified in the PDF**; document choices and obtain teammate agreement rather than inventing official rules.

### Learning/evaluation requirements

- First-Visit MC: on-policy, epsilon-soft control, per Sutton & Barto §5.4.
- 4-step SARSA: on-policy 4-step returns with decaying epsilon, per Sutton & Barto §7.2.
- Fixed opponents: Random followed by Heuristic. **No self-play** for these tabular methods.
- Evaluate alongside other agents; label the tabular methods as baselines, as the plan directs.

## 5. Statistical and experimental quality gates

| Area                | Required check                                                                                                                         |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Harness determinism | Same seeds and configurations reproduce games/results; test intentionally rather than assuming.                                        |
| Elo                 | Bradley–Terry fit behaves sensibly on synthetic results; provide bootstrap confidence intervals.                                       |
| Win rates           | Wilson confidence intervals, game counts and draw handling documented.                                                                 |
| Bias                | Random/Random and Heuristic/Heuristic results split by archetype deck and seat.                                                        |
| Tournament          | One-command runner; final comparison **1,000 games per pair**.                                                                         |
| Training            | M2 target: each learning agent at least **70%** wins against Random in **500** fixed-seed games.                                       |
| Parallelism         | 1–4-worker rollout benchmarking; AZ-lite scaling at 1/2/4/8/16 workers where feasible.                                                 |
| Reward ablation     | Shaped vs terminal reward; do not confound reward type with changes in compute, opponent or evaluation protocol without disclosing it. |
| Outputs             | `configs/*.yaml` committed; `runs/` gitignored; final CSVs in `results/`; figures PNG + source CSV.                                    |

### Compute budget (team-wide; coordinate with Max)

- Target **5 million real agent/environment steps × 3 seeds** for each of PPO, DQN and AZ-lite.
- AZ-lite gets the same count of **real environment steps**; report simulated MCTS steps separately as computation.
- If budget is infeasible, reduce to **2 million steps before reducing the number of seeds**. Always report the actual budget used.
- Available/potential hardware: team laptops and SJSU HPC **if access is granted**; source plan says request HPC in week of October 19.

## 6. Collaboration and code-change boundaries

- Protected `main`; develop on feature branches (plan example: `scott/combat-keywords`; use an equivalent `aditya/<task>` convention for own work).
- PRs need **one teammate review and green CI**.
- Any edits to `micromagic/env/spec.py` constants, action indices, `OBS_DIM`, or card-ID map **must be reviewed by all three**.
- Issue per timeline task, labeled by milestone `M0`–`M6` and owner.
- Two ~30-minute syncs weekly (suggested Monday and Thursday); report completed work, next work and blockers. Meeting notes: Drive → `CMPE 260 / Project / Meeting Notes`.
- Team report: shared Google Doc in Project folder; each owner drafts assigned sections from November 23 onward.

## 7. Project risks and prescribed fallbacks

| Risk                           | Plan's response                                                                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Engine delayed                 | Build against Scott's frozen API and stub; toy engine still useful.                                                                         |
| Engine too slow for MCTS       | Scott benchmarks/optimizes; custom `clone()` rather than `deepcopy`; reduce MCTS sims/determinizations; report true cost.                   |
| DQN fails on sparse rewards    | Max tries dueling + n-step returns; **cap debugging at one week**, then report failure as finding.                                          |
| Ray complexity delays progress | Use `SubprocVecEnv`/multiprocessing for training; restrict Ray to scaling experiments.                                                      |
| Deck or seat dominates         | Investigate at M2, present stratified numbers; one deck-list adjustment permitted before M4 if Heuristic-vs-Heuristic win rate exceeds 70%. |
| Uneven workload                | Reassess at M2 (November 8), redistribute if a teammate is over one week behind.                                                            |

## 8. Open questions and agent blockers

Before irreversible design decisions, determine or ask:

1. **Repo reality:** Which modules, tests, CI pipelines, scripts and trained checkpoints already exist?
2. **API availability:** Has Scott delivered the full/stub environment? Is the observation accessible in the defined format? Are action masks and determinization supported?
3. **Agent integration:** Is Max's `Agent` base usable by the harness, including per-game reset and checkpoint load conventions? Are Random and Heuristic available?
4. **Abstract states:** What precisely is the **decision context** feature? Which observable game features should be used to calculate board power and action viability?
5. **Macro behavior:** The plan says “~8” but names nine macros. Keep nine as enumerated unless the team intentionally consolidates them. Who implements the primitive macro executor and tests?
6. **Evaluation policy:** How are draws, seat/deck alternation and any paired common-random-seed design handled for Elo, Wilson CIs and reporting?
7. **Ray compatibility:** Is Ray installed and compatible with actual training loops? What hardware access/worker counts are realistic?
8. **Reward ablation:** Who trains the matched shaped-vs-terminal checkpoints, and where is reward configuration defined?
9. **Schedule:** Does the assumed week-of-December-7 deadline match official course requirements?

> **Important:** These questions are deliberately highlighted because the source plan does **not** give complete implementation detail. Coding agents may propose designs, but should clearly mark assumptions and avoid treating them as settled team contracts.

## 9. Suggested next-agent prompt

```text
You are working on the MicroMagic project as Aditya's coding agent (Systems & Evaluation).
Read MicroMagic_Aditya_Agent_Handoff.md, then inspect the repository.

First provide:
1. A brief audit of which Aditya-owned deliverables already exist, are incomplete, or are blocked.
2. The earliest sensible unblocked task, including its dependencies and acceptance tests.
3. An implementation plan limited to Aditya's ownership area, with any interface assumptions explicitly named.

Then implement that task, add tests, run the appropriate checks, and report changed files,
commands/results, remaining blockers, and next steps.

Do not fabricate experimental results. Do not modify Scott's environment spec or Max's
agent contracts without coordination. Do not push or merge without my approval.
```

---

## 10. Source and document limitations

Adapted from **MicroMagic Team Plan**, pages 1–5, especially: ownership/workload (page 2), interface contract (page 3), timeline and tabular abstraction (pages 4–5), and compute/working agreement/risks (page 5). Original repository: https://github.com/maxogarcia/MicroMagic.

This is a **role-focused reorganization**, not a literal page-by-page transcription. The plan's Check-in 1 corrections are primarily report/team maintenance rather than assigned Aditya work; coordinate on them as shared team tasks. The PDF lists a decision-context abstract-state feature without defining its bins, and labels nine listed macros as “~8”; those discrepancies are preserved and flagged above.
