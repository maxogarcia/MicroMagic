"""Heuristic baseline: observation decoding, legality, rule behavior, and strength vs. Random."""

import numpy as np
import pytest

from micromagic.agents import RandomAgent
from micromagic.agents.heuristic import HeuristicAgent, decode
from micromagic.env.spec import ATTACK, CARD_IDS, EMPTY, PASS, TARGET_OPPONENT, Phase
from micromagic.env.state import GameState
from micromagic.tournaments import AgentProtocol


DECK_PAIRS = (("aggro", "midrange"), ("midrange", "aggro"))


def play(decks, heuristic_seat: int, seed: int) -> int | None:
    state = GameState(seed=seed, decks=decks)
    agents = {heuristic_seat: HeuristicAgent(), 1 - heuristic_seat: RandomAgent(seed=seed)}
    while not state.is_terminal():
        player = state.current_player()
        mask = state.legal_actions()
        action = agents[player].act(state.observe(player), mask)
        assert mask[action], f"illegal action {action} in {state.phase.name}"
        state.step(action)
    return state.winner()


def test_satisfies_harness_protocol() -> None:
    assert isinstance(HeuristicAgent(), AgentProtocol)


def test_decode_matches_engine_state() -> None:
    state = GameState(seed=5)
    rng = np.random.default_rng(0)
    for _ in range(300):
        if state.is_terminal():
            break
        player = state.current_player()
        view = decode(state.observe(player))
        assert view.my_life == max(int(state.life[player]), 0)
        assert view.opp_life == max(int(state.life[1 - player]), 0)
        assert view.phase == state.phase
        assert view.hand == [int(c) for c in state.hand[player, : state.hand_count[player]]]
        assert [(c.slot, c.card) for c in view.mine] == [
            (s, int(c)) for s, c in enumerate(state.creature[player]) if c != EMPTY
        ]
        for c in view.theirs:
            assert c.power == state.power(1 - player, c.slot)
            assert c.tapped == bool(state.tapped[1 - player, c.slot])
        state.step(int(rng.choice(np.flatnonzero(state.legal_actions()))))


@pytest.mark.parametrize("decks", DECK_PAIRS)
@pytest.mark.parametrize("seat", [0, 1])
def test_only_plays_legal_actions(decks, seat) -> None:
    for seed in range(15):
        play(decks, seat, seed)


def test_tracks_which_creature_is_asked_to_block() -> None:
    # The observation does not say which creature the engine is asking to
    # block; the agent infers it. Compare with the engine's blocker cursor.
    checked = 0
    for seed in range(40):
        state = GameState(seed=seed, decks=DECK_PAIRS[seed % 2])
        heuristic, opponent = HeuristicAgent(), RandomAgent(seed=seed)
        while not state.is_terminal():
            player = state.current_player()
            obs, mask = state.observe(player), state.legal_actions()
            if player == 1 and state.phase == Phase.DECLARE_BLOCKERS:
                asking = int(state.blocker_queue[state.blocker_cursor])
                same_combat = heuristic._combat is not None and heuristic._combat[0] == decode(obs).turn
                before = set(heuristic._decided) if same_combat else set()
                action = heuristic.act(obs, mask)
                assert set(heuristic._decided) - before == {asking}
                checked += 1
            else:
                action = (heuristic if player == 1 else opponent).act(obs, mask)
            state.step(action)
    assert checked > 50


def test_casts_lethal_burn_at_the_face() -> None:
    state = GameState(seed=0)
    me = state.active
    state.life[1 - me] = 2
    state.hand[me, 0] = CARD_IDS["Shock"]
    state.lands_total[me] = 1
    state.land_played = True
    agent = HeuristicAgent()
    state.step(agent.act(state.observe(me), state.legal_actions()))
    assert state.phase == Phase.CHOOSE_TARGET
    assert agent.act(state.observe(me), state.legal_actions()) == TARGET_OPPONENT


def test_attacks_with_everything_when_lethal() -> None:
    state = GameState(seed=0)
    me, opp = state.active, 1 - state.active
    state.creature[me, :2] = CARD_IDS["Gray Ogre"]
    state.creature[opp, 0] = CARD_IDS["Giant Spider"]
    state.life[opp] = 2
    state.phase = Phase.DECLARE_ATTACKERS
    agent = HeuristicAgent()
    first = agent.act(state.observe(me), state.legal_actions())
    assert first in ATTACK
    state.step(first)
    assert agent.act(state.observe(me), state.legal_actions()) in ATTACK


def test_does_not_attack_into_a_bigger_blocker() -> None:
    state = GameState(seed=0)
    me, opp = state.active, 1 - state.active
    state.creature[me, 0] = CARD_IDS["Grizzly Bears"]
    state.creature[opp, 0] = CARD_IDS["Giant Spider"]
    state.phase = Phase.DECLARE_ATTACKERS
    assert HeuristicAgent().act(state.observe(me), state.legal_actions()) == PASS


def test_beats_random_on_every_matchup() -> None:
    # M2 asks for >= 90% overall; aggro is the weaker deck (it wins 18% of
    # random-vs-random games), so its bar here is lower than midrange's.
    games = 60
    for decks in DECK_PAIRS:
        for seat in (0, 1):
            wins = sum(play(decks, seat, 50_000 + g) == seat for g in range(games))
            floor = 0.5 if decks[seat] == "aggro" else 0.85
            assert wins / games >= floor, (decks[seat], seat, wins)
