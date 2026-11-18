# Playing MicroMagic in a browser

The play app runs the same `GameState`, decks, card pool and rules the agents train on.
Nothing is simplified for the UI: every click is one action from the 29-action space, and
the server checks it with `is_legal` before stepping the engine.

## Start it

From the repository root (with the package installed, `pip install -e .`):

```
python -m micromagic.play            # opens http://127.0.0.1:8765
python -m micromagic.play --lan      # also prints a link other computers on your Wi-Fi can open
python -m micromagic.play --port 9000 --no-browser
```

Card art loads from Scryfall when the browser is online. Offline, cards are drawn from
the card data instead; the game works either way.

## Three modes

| Mode | What it does |
| --- | --- |
| **Play a bot** | You against any listed agent. Pick your deck and who goes first. |
| **Watch bots** | Two agents play each other with both hands face up. Pause, and choose a speed from Slow to Instant. |
| **Play a friend** | Two people. You get an invite link for the second seat; open it in another window, or on another computer when the server runs with `--lan`. |

Every game can be downloaded as a standalone replay page (the same viewer as
`micromagic.env.replay`). Against a human opponent the replay unlocks when the game ends,
since it shows both hands.

## How to play

- Cards you can use glow gold: click a card in your hand to play it, a creature to attack with it.
- When you block, your untapped creatures glow; click one of the opponent's attackers to block it.
- When a spell needs a target, valid targets glow red (including the opponent's name for burn to the face).
- The gold button moves on: **Go to combat**, **Skip combat**, **Done attacking**, **Don't block with ...**, **End turn**.
- Hover a card to see it full size.

## Which bots are listed

Always: Random, Heuristic (Max's baseline), MCTS (4 worlds x 64 simulations) and MCTS (quick).

Learned agents appear automatically when their files exist under the folder you start the server from:

- PPO and DQN: `runs/ppo/<run>/model.zip`, `runs/dqn/<run>/model.zip` (needs `pip install -e .[rl]`)
- Tabular First-Visit MC and 4-step SARSA: `results/tabular/mc_final.npz`, `results/tabular/sarsa_final.npz`
  (needs `micromagic.agents.tabular`)

Only listed bots can be chosen; the API never loads an arbitrary path.

## For the demo

1. `python -m micromagic.play --lan`
2. **Watch bots**: the trained agent vs Heuristic at Normal speed shows the learned policy in action.
3. **Play a bot**: let someone from the audience play the trained agent.
4. **Play a friend**: send the invite link to a second laptop on the same network.

## API

`src/micromagic/play/server.py` is standard library only. Each human seat has a secret token in its
link; the server hides the opponent's hand from every view unless no other human would be exposed.
See the module docstring for the endpoints. Tests: `tests/integration/test_play.py`.
