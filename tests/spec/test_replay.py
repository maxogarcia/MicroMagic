"""Tests for the game recorder and HTML replay writer."""

import json
from pathlib import Path

import numpy as np

from micromagic.env.replay import DATA_PLACEHOLDER, describe_action, record_game, snapshot, write_html
from micromagic.env.spec import ATTACK, BLOCK, CARD_IDS, EMPTY, PASS, PLAY_HAND, TARGET_OPP_CREATURE
from micromagic.env.state import GameState


def test_recording_is_deterministic_and_complete() -> None:
    first, second = record_game(seed=3), record_game(seed=3)
    assert json.dumps(first) == json.dumps(second)
    assert first["frames"][0]["actor"] is None
    assert first["frames"][-1]["state"]["terminal"]
    assert len(first["cards"]) == len(CARD_IDS)
    for frame in first["frames"][1:]:
        assert frame["actor"] in (0, 1) and frame["text"]


def test_recorded_actions_replay_to_the_same_final_state() -> None:
    replay = record_game(seed=11, decks=("midrange", "aggro"), first_player=1)
    state = GameState(seed=11, decks=("midrange", "aggro"), first_player=1)
    for frame in replay["frames"][1:]:
        state.step(frame["action"])
    assert snapshot(state) == replay["frames"][-1]["state"]


def test_custom_policy_is_used() -> None:
    def always_pass_if_possible(obs: np.ndarray, mask: np.ndarray) -> int:
        return 0 if mask[0] else int(np.flatnonzero(mask)[0])

    replay = record_game(policies=(always_pass_if_possible, always_pass_if_possible), seed=0)
    acted = [f for f in replay["frames"][1:]]
    assert all(f["action"] == PASS for f in acted)
    assert replay["result"].startswith("Draw") or "wins" in replay["result"]


def test_descriptions_read_like_a_game_log() -> None:
    state = GameState(seed=0)
    state.hand[0] = EMPTY
    state.hand[0, :2] = [CARD_IDS["Mountain"], CARD_IDS["Shock"]]
    state.hand_count[0] = 2
    state.lands_total[0] = 3
    state.creature[1, 0] = CARD_IDS["Grizzly Bears"]
    assert describe_action(state, PLAY_HAND.start) == "Plays Mountain"
    assert describe_action(state, PLAY_HAND.start + 1) == "Casts Shock"
    state.step(PLAY_HAND.start + 1)
    assert describe_action(state, TARGET_OPP_CREATURE.start) == "Shock targets Grizzly Bears"

    state = GameState(seed=0)
    state.creature[0, 0] = CARD_IDS["Hill Giant"]
    state.creature[1, 0] = CARD_IDS["Giant Spider"]
    state.step(PASS)
    assert describe_action(state, ATTACK.start) == "Attacks with Hill Giant"
    state.step(ATTACK.start)
    state.step(PASS)
    assert describe_action(state, BLOCK.start) == "Giant Spider blocks Hill Giant"
    assert describe_action(state, PASS) == "Giant Spider doesn't block"


def test_combat_events_are_logged() -> None:
    replay = record_game(seed=7)
    events = [e for f in replay["frames"] for e in f["events"]]
    assert any("dies" in e for e in events)
    assert any("damage" in e for e in events)


def test_write_html_embeds_the_replay(tmp_path: Path) -> None:
    replay = record_game(seed=2)
    replay["players"][0] = "</script><b>"  # must not break out of the data block
    out = write_html(replay, tmp_path / "nested" / "game.html")
    html = out.read_text(encoding="utf-8")
    assert DATA_PLACEHOLDER not in html
    assert "</script><b>" not in html
    start = html.index('type="application/json">') + len('type="application/json">')
    data = json.loads(html[start : html.index("</script>", start)])
    assert data["players"][0] == "</script><b>" and len(data["frames"]) == len(replay["frames"])
