# MicroMagic

A project for experimenting with various reinforcement learning algorithms and methodologies. Comparing their metrics over a simplified verious of Magic: The Gathering

## Play against the agents

```
pip install -e .
python -m micromagic.play
```

Opens a browser table where you can play any agent, watch two agents play each other, or
play a friend on another computer (`--lan`). It uses the same decks, cards and rules as
training. See [docs/play.md](docs/play.md).

## Train the agents

```
pip install -e .[test,rl]
python -m micromagic.training.train_ppo       --config configs/ppo_full.yaml
python -m micromagic.training.train_dqn       --config configs/dqn_full.yaml
python -m micromagic.training.train_alphazero --config configs/alphazero_full.yaml
python -m micromagic.training.train_tabular   --algo mc      # or --algo sarsa; see --help
```

The PPO, DQN and AlphaZero-lite runs write to `runs/<algo>/<run_name>/`: the config, a
training log, checkpoints, `learning_curve.csv`, and evaluation tables per deck and seat.
`--total-timesteps`, `--seed` and `--run-name` override the config.
`configs/alphazero_smoke.yaml` checks the AlphaZero-lite loop end to end in under a minute.
