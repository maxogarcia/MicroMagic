# MaskablePPO: An Intuitive Guide for MicroMagic

## The 60-Second Explanation

MaskablePPO is an agent that looks at the current situation, assigns a probability to every legal move, chooses one, and later adjusts those probabilities based on whether the move worked out better or worse than expected.

The name has two parts:

- **PPO** means the agent improves its policy in cautious steps instead of rewriting its strategy after every surprising game.
- **Maskable** means the environment tells it which actions are legal right now. Illegal actions are removed before it chooses.

For MicroMagic, that could mean the full action catalog contains `play a land`, `cast each card`, `attack with each creature`, `block each attacker`, and `pass`. On a particular turn, perhaps only `play Mountain`, `cast Goblin Piker`, and `pass` are legal. The mask hides everything else from the choice.

## A Concrete MicroMagic Turn

Imagine this position:

- You have 6 life. Your opponent has 5.
- You have two untapped Mountains and a Goblin Piker in hand.
- You control a Hill Giant that can attack.
- Your opponent controls one untapped Gray Ogre.

Suppose the fixed action catalog has 20 possible actions. Only four are legal now:

1. Cast Goblin Piker.
2. Attack with Hill Giant.
3. Keep Hill Giant back and move to the second main phase.
4. End the turn.

The action mask removes the other 16 actions. PPO might currently assign the legal actions probabilities of 35%, 40%, 15%, and 10%.

It chooses an action—say, attack with Hill Giant. Much later, the game ends in a win. That does not automatically mean the attack was brilliant. The agent asks a more useful question: **was this outcome better than I expected from that position?**

If the win was better than expected, PPO raises the probability of similar choices in similar situations. If the agent was already almost certain to win, the update is smaller. If the attack led to a worse outcome than expected across many games, its probability falls.

## The Two Parts Inside PPO

PPO is commonly described as an **actor-critic** method. Those names sound more complicated than the idea.

### The actor: "What should I do?"

The actor is the policy. It produces the action probabilities. In the example above, it is the part that says there is a 40% chance of attacking and a 35% chance of casting the creature.

Keeping probabilities instead of always selecting one fixed move matters in a hidden-information game. If a policy becomes perfectly predictable, an opponent can sometimes exploit it. Randomness also helps the agent try alternatives during training.

### The critic: "How promising is this situation?"

The critic estimates the likely future result from the current observation. It does not choose the move. It gives the actor a baseline for judging whether the actual result was surprisingly good or surprisingly bad.

If the critic expects a weak position to lose but the agent wins, the actions leading to that result receive a strong positive signal. If the critic already expects an easy win, the same win is less surprising.

The difference between the result and the critic's expectation is called the **advantage**. For this project, it is enough to read that word as "better or worse than expected."

## Why PPO Is "Proximal"

Early policy-gradient methods could overreact. One lucky game might make an action far more likely; one unlucky game might nearly erase a useful move.

PPO puts a guardrail around each update. It lets the policy move toward actions that looked better, but limits how far the new policy can move away from the policy that generated the training games.

Think of coaching after a match:

> "Attacking there worked. Try it somewhat more often. Do not rebuild the entire strategy around one game."

PPO collects a batch of recent games, learns from that batch for several passes, and then gathers fresh experience using the updated policy. This is called **on-policy** learning: recent play matters most, and old games are not reused indefinitely as if they came from the current strategy.

## What the Mask Changes

Plain PPO can assign probability to every action number in the action space. That is awkward in a card game because legal actions change constantly.

Without a mask, the agent may initially spend much of its exploration trying nonsense:

- casting a card that is not in its hand;
- spending more mana than it has;
- attacking with a creature that cannot attack; or
- blocking when it is not the defending player.

The environment could punish those moves, but then the agent wastes training time learning basic legality. Worse, a penalty-based design makes "do nothing" attractive when the action space contains many illegal choices.

MaskablePPO instead receives a Boolean list: legal or illegal. Illegal actions get no probability, and the remaining probability is redistributed across legal choices. The official SB3-Contrib implementation is PPO with invalid-action masking; it also requires mask-aware evaluation utilities.[^maskableppo]

The mask is not strategic advice. It can say that attacking is legal. It cannot say whether attacking is wise.

## Monte Carlo, TD, SARSA, and PPO Are Not Four Versions of the Same Thing

This distinction prevents a common misunderstanding.

**Monte Carlo and temporal-difference learning describe how an agent builds a learning target from experience. PPO describes how a policy is optimized.** PPO's critic normally uses a TD-like mixture of short and longer estimates, so PPO is not simply the opposite of TD.

| Approach | Plain-language learning signal | When it learns | What it primarily changes |
|---|---|---|---|
| First-visit Monte Carlo control | "The game ended in a win. Which visited state-action choices tend to appear in winning games?" | After the episode ends | A table of action values and the policy derived from it |
| Four-step SARSA | "Add the next few rewards, then use our current estimate of where we landed." | During and after play | A table of action values for the policy actually followed |
| MaskablePPO | "Was this legal action better or worse than the critic expected, and how cautiously should its probability change?" | After a batch of recent experience | A neural policy directly, plus its value critic |
| MCTS | "From this position, simulate many possible continuations before choosing this move." | At decision time | A temporary search tree; learning is optional |

### Same example, different learning

Suppose an early attack eventually helps win on turn 12.

- **Monte Carlo:** waits for turn 12, then sends the final win signal back to that earlier state-action visit.
- **Four-step SARSA:** after four transitions, combines any rewards seen so far with its current estimate of the position reached at that point.
- **MaskablePPO:** gathers the attack and many other decisions into a batch. Its critic estimates how promising each position was, and the actor changes the probability of each action according to whether the outcome was better or worse than expected.
- **MCTS:** before making the attack, simulates future branches from the current position. In a hidden-information game, ordinary MCTS risks using cards the player should not know, which is why it is deferred here.

## Why MaskablePPO Fits MicroMagic

### 1. The legal action set changes every turn

Card games have many conditional actions. Masking handles legality directly while leaving strategy to the policy. PettingZoo's current AEC documentation treats action masks as a standard way to mark valid and invalid actions in sequential multi-agent environments.[^pettingzoo]

### 2. Exact state repetition will be rare

A lookup table works best when the same key appears many times. In MicroMagic, different hands, card orders, boards, life totals, and combat choices create many combinations. PPO's neural network can learn that two nonidentical situations share useful features—for example, that attacking into a larger untapped blocker is often dangerous.

This is a reason to try function approximation, not proof that the network will generalize correctly.

### 3. A stochastic policy makes sense under hidden information

The player cannot see the opponent's hand or future draws. A probability distribution can represent several reasonable actions instead of pretending one move is certainly best. Recent ICLR 2026 evidence found that tuned generic policy-gradient methods were competitive with more specialized imperfect-information approaches across the games studied.[^iigppo] That result supports PPO as a credible baseline; it does not establish that PPO will solve MicroMagic.

### 4. The implementation already exists

SB3-Contrib provides MaskablePPO, mask-aware evaluation, seeded training options, checkpoint support, and standard logging hooks. The team can spend its time on the game contract and evaluation instead of writing a policy optimizer from scratch.

## Where MaskablePPO Can Still Fail

1. **Bad observations:** if the vector omits summoning sickness or available mana, the policy cannot infer legal tactics reliably.
2. **Hidden-information leakage:** if the observation accidentally includes the opponent's hand, training results become invalid even if win rate rises.
3. **Weak rewards:** terminal-only rewards may make early learning slow; careless damage rewards can teach reckless play.
4. **Unstable self-play:** two copies of a changing policy can chase each other's weaknesses or forget how to beat older strategies.
5. **Bad evaluation:** ordinary SB3 evaluation helpers do not automatically apply MaskablePPO's masks. The mask-aware evaluator is required.[^maskableppo]

MaskablePPO also does not provide a recurrent policy in the current SB3-Contrib support table. If important information exists only in past public actions, the initial environment must encode the necessary public history in the observation rather than assuming the policy remembers it.

For the proposed life-difference ablation, the calculation must always use the learner's viewpoint: `my life minus opponent life`. It must include changes caused during the opponent's reply before PPO receives its next observation. Otherwise the same event can accidentally have the wrong reward sign or fail to remain zero-sum.

## What Training Would Look Like in This Project

Stable Baselines3 expects one learning agent, while PettingZoo alternates between both players. A small adapter therefore presents the game from one learner's viewpoint. After the learner acts, the adapter lets the opponent complete its replies and returns control only when the learner must decide again. This prevents PPO from accidentally treating the opponent's observation as its own next state.

1. The adapter creates a seeded game, chooses the learner's seat, and returns an ego-centered observation plus legal-action mask.
2. MaskablePPO assigns probabilities only to legal actions and samples one.
3. The adapter applies that action, runs opponent actions until the learner acts again, and accumulates the learner's reward.
4. After a rollout batch, the critic estimates what was expected, and the actor makes bounded probability updates based on what went better or worse.
5. Evaluation freezes the policy and plays held-out games across both seats against random, heuristic, and frozen-policy opponents.

Before any long run, a 10,000-step smoke test must complete the entire path: game creation, masked action selection, learning update, checkpoint save/load, and held-out evaluation.

## Questions to Check Your Understanding

1. If `cast Lava Axe` is legal but strategically poor, should the action mask remove it? Why not?
2. Why might rewarding every point of damage teach a policy that loses games aggressively?
3. What does the critic add that a final win/loss signal alone does not provide?
4. Why should PPO increase a good action's probability gradually rather than jump straight to 100%?
5. Why is PPO not cleanly described as "different from TD" even though it directly learns a policy?

Short answers: the mask enforces rules, not judgment; raw damage can conflict with winning; the critic supplies an expectation baseline; bounded updates reduce overreaction; and PPO commonly uses TD-style estimates inside its actor-critic learning loop.

## Evidence Notes

**Source claim:** PettingZoo states, "AEC environments often include action masks, in order to mark valid/invalid actions for the agent."[^pettingzoo]

**Source claim:** The MaskablePPO documentation requires its mask-aware callback and evaluator "to properly evaluate a model with action masks."[^maskableppo]

**Source claim:** Rudolph et al. report, "Over 7000 training runs, we find that FP-, DO-, and CFR-based approaches fail to outperform generic policy gradient methods."[^iigppo]

**Project inference:** MaskablePPO matches MicroMagic's variable legal actions and large observation space better than a purely tabular primary agent. Whether it produces sensible play remains an experiment, not an established fact.

[^pettingzoo]: Farama Foundation, "AEC API," current documentation, accessed October 11, 2026, https://pettingzoo.farama.org/main/api/aec/.
[^maskableppo]: Stable Baselines3-Contrib 2.9.0, "Maskable PPO," accessed October 11, 2026, https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html.
[^iigppo]: Max Rudolph et al., "Reevaluating Policy Gradient Methods for Imperfect-Information Games," ICLR 2026, https://proceedings.iclr.cc/paper_files/paper/2026/hash/fe1c4991d57f37dfef62d01b3901ca54-Abstract-Conference.html.
