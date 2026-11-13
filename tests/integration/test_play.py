"""Tests for the browser play app: sessions, hidden information, and the HTTP API."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from micromagic.env.spec import PASS
from micromagic.env.state import GameState
from micromagic.play import available_bots, make_bot, new_session
from micromagic.play.server import make_server


def _human_vs_human(seed: int = 5):
    return new_session([{"kind": "human", "name": "A"}, {"kind": "human", "name": "B"}], seed=seed, first_player=0)


def _first_legal(view: dict) -> int:
    return view["legal"][0]["action"]


def test_built_in_bots_are_listed_and_unknown_ids_rejected() -> None:
    ids = {b.id for b in available_bots()}
    assert {"random", "heuristic", "mcts", "mcts_quick"} <= ids
    with pytest.raises(ValueError):
        make_bot("../../etc/passwd")


def test_bot_vs_bot_session_matches_the_training_engine() -> None:
    game = new_session([{"kind": "bot", "bot": "random"}, {"kind": "bot", "bot": "random"}],
                       decks=("aggro", "midrange"), seed=21, first_player=1)
    while game.bot_step():
        pass
    assert game.state.is_terminal()
    # The recorded actions replay to the same result on a fresh GameState.
    state = GameState(seed=21, decks=("aggro", "midrange"), first_player=1)
    for frame in game.recorder.frames[1:]:
        state.step(frame["action"])
    assert state.winner() == game.state.winner()


def test_hidden_information_and_tokens() -> None:
    game = _human_vs_human()
    mine = game.view(0)
    assert mine["state"]["players"][0]["hand"] is not None
    assert mine["state"]["players"][1]["hand"] is None
    assert mine["state"]["players"][1]["hand_count"] == 7
    # A human can't reveal another human's hand, and a spectator sees neither.
    assert game.view(0, reveal=True)["state"]["players"][1]["hand"] is None
    spectator = game.view(None, reveal=True)
    assert all(p["hand"] is None for p in spectator["state"]["players"])
    assert spectator["legal"] == []

    action = _first_legal(mine)
    with pytest.raises(PermissionError):
        game.act(0, "wrong-token", action)
    with pytest.raises(ValueError):
        game.act(1, game.seats[1].token, action)  # not seat 1's decision
    illegal = int(np.flatnonzero(~game.state.legal_actions())[0])
    with pytest.raises(ValueError):
        game.act(0, game.seats[0].token, illegal)
    game.act(0, game.seats[0].token, action)


def test_bot_hand_can_be_revealed_to_its_only_human_opponent() -> None:
    game = new_session([{"kind": "human"}, {"kind": "bot", "bot": "random"}], seed=2)
    assert game.view(0)["state"]["players"][1]["hand"] is None
    assert game.view(0, reveal=True)["state"]["players"][1]["hand"] is not None


def test_watching_bots_shows_both_hands() -> None:
    game = new_session([{"kind": "bot", "bot": "random"}, {"kind": "bot", "bot": "heuristic"}], seed=4)
    assert all(p["hand"] is not None for p in game.view(None)["state"]["players"])


def test_pass_button_wording_is_imperative() -> None:
    game = _human_vs_human(seed=9)
    view = game.view(0)
    pass_entry = next(e for e in view["legal"] if e["action"] == PASS)
    assert pass_entry["button"] in ("Go to combat", "Skip combat")


def test_humans_can_finish_a_game() -> None:
    game = _human_vs_human(seed=13)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        if game.state.is_terminal():
            break
        seat = game.state.current_player()
        legal = game.view(seat)["legal"]
        game.act(seat, game.seats[seat].token, legal[int(rng.integers(len(legal)))]["action"])
    assert game.state.is_terminal()
    assert game.view(0)["result"]


@pytest.fixture()
def server():
    srv = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _call(url: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def test_http_api_round_trip(server) -> None:
    status, page = _call(f"{server}/")
    assert status == 200 and b"/*__VIEWER_CSS__*/" not in page and b"<style>" in page

    status, raw = _call(f"{server}/api/options")
    options = json.loads(raw)
    assert status == 200 and "heuristic" in {b["id"] for b in options["bots"]}

    status, raw = _call(f"{server}/api/games", {"seats": [{"kind": "human"}, {"kind": "bot", "bot": "random"}], "seed": 1})
    created = json.loads(raw)
    gid, token = created["id"], created["seats"]["0"]["token"]

    status, _ = _call(f"{server}/api/games/{gid}?seat=0&token=nope")
    assert status == 403
    status, raw = _call(f"{server}/api/games/{gid}?seat=0&token={token}")
    view = json.loads(raw)
    assert status == 200 and view["waiting_for"] == "human"

    # Against a bot the replay (both hands) is available mid-game; vs a human it is not (next test).
    status, _ = _call(f"{server}/api/games/{gid}/replay?seat=0&token={token}")
    assert status == 200

    for _ in range(5000):
        status, raw = _call(f"{server}/api/games/{gid}?seat=0&token={token}")
        view = json.loads(raw)
        if view["result"]:
            break
        if view["waiting_for"] == "bot":
            _call(f"{server}/api/games/{gid}/bot_step", {})
        else:
            status, _ = _call(f"{server}/api/games/{gid}/act", {"seat": 0, "token": token, "action": _first_legal(view)})
            assert status == 200
    assert view["result"]

    status, _ = _call(f"{server}/api/games/doesnotexist")
    assert status == 404


def test_replay_locked_while_two_humans_are_playing(server) -> None:
    _, raw = _call(f"{server}/api/games", {"seats": [{"kind": "human"}, {"kind": "human"}], "seed": 1})
    created = json.loads(raw)
    status, _ = _call(f"{server}/api/games/{created['id']}/replay?seat=0&token={created['seats']['0']['token']}")
    assert status == 403


def test_skipped_blocks_are_explained_in_the_log() -> None:
    """When attackers face no legal blocker, the log says why instead of silently skipping blocks."""
    from micromagic.env.spec import Phase

    explained = 0
    for seed in range(40):
        game = new_session([{"kind": "bot", "bot": "heuristic"}, {"kind": "bot", "bot": "heuristic"}], seed=seed)
        while not game.state.is_terminal():
            st = game.state
            finishing_attack = st.phase == Phase.DECLARE_ATTACKERS and st.attacking[st.active].any()
            game.bot_step()
            frame = game.recorder.frames[-1]
            if finishing_attack and frame["action"] == PASS and game.state.phase != Phase.DECLARE_BLOCKERS:
                assert any("block" in e for e in frame["events"]), frame
                explained += 1
    assert explained > 0
