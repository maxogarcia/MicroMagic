"""Local web server for playing MicroMagic in a browser (standard library only).

    python -m micromagic.play            # http://127.0.0.1:8765, opens your browser
    python -m micromagic.play --lan      # also reachable from other computers on your network

API (JSON):
    GET  /api/options                         bots, decks, card data
    POST /api/games                           {"seats": [spec, spec], "decks": [..], "seed": .., "first_player": ..}
    GET  /api/games/<id>?seat=&token=&reveal= the game as that seat (or a spectator) may see it
    POST /api/games/<id>/act                  {"seat": .., "token": .., "action": ..}
    POST /api/games/<id>/bot_step             let the bot whose turn it is make one move
    GET  /api/games/<id>/replay?seat=&token=  the game as a standalone replay page
"""

import argparse
import json
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from micromagic.env.replay import card_catalog, render_html
from micromagic.env.spec import DECK_NAMES
from micromagic.play.bots import available_bots
from micromagic.play.session import GameSession, new_session


DATA = Path(__file__).resolve().parent.parent / "data"
PAGE = DATA / "play.html"
VIEWER = DATA / "replay_viewer.html"
CSS_PLACEHOLDER = "/*__VIEWER_CSS__*/"


def page_html() -> bytes:
    """The play page, with the replay viewer's stylesheet inlined so both pages share one look."""
    viewer = VIEWER.read_text(encoding="utf-8")
    css = viewer[viewer.index("<style>") + len("<style>"):viewer.index("</style>")]
    return PAGE.read_text(encoding="utf-8").replace(CSS_PLACEHOLDER, css).encode("utf-8")
MAX_GAMES = 200
MAX_BODY = 64 * 1024


class GameStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.games: dict[str, GameSession] = {}
        self.lock = threading.Lock()

    def add(self, session: GameSession) -> None:
        with self.lock:
            if len(self.games) >= MAX_GAMES:
                oldest = min(self.games.values(), key=lambda g: g.created)
                del self.games[oldest.id]
            self.games[session.id] = session

    def get(self, game_id: str) -> GameSession:
        with self.lock:
            if game_id not in self.games:
                raise KeyError(game_id)
            return self.games[game_id]


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def make_handler(store: GameStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MicroMagic"

        def log_message(self, fmt: str, *args) -> None:  # keep the terminal quiet during play
            pass

        # Plumbing

        def _send(self, status: HTTPStatus, body: bytes, content_type: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request too large")
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as err:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {err}") from err

        def _route(self, method: str) -> None:
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = [p for p in url.path.split("/") if p]
            try:
                self._dispatch(method, parts, query)
            except ApiError as err:
                self._json({"error": str(err)}, err.status)
            except KeyError:
                self._json({"error": "No such game (the server may have restarted)"}, HTTPStatus.NOT_FOUND)
            except PermissionError as err:
                self._json({"error": str(err)}, HTTPStatus.FORBIDDEN)
            except ValueError as err:
                self._json({"error": str(err)}, HTTPStatus.BAD_REQUEST)

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

        # Routes

        def _viewer(self, game: GameSession, query: dict) -> int | None:
            if "seat" not in query:
                return None
            seat = int(query["seat"])
            if not game.check_token(seat, query.get("token")):
                raise PermissionError("This seat belongs to someone else")
            return seat

        def _dispatch(self, method: str, parts: list[str], query: dict) -> None:
            if method == "GET" and parts in ([], ["play"]):
                self._send(HTTPStatus.OK, page_html(), "text/html; charset=utf-8")
                return
            if parts[:1] != ["api"]:
                raise ApiError(HTTPStatus.NOT_FOUND, "Not found")

            if method == "GET" and parts == ["api", "options"]:
                self._json({
                    "bots": [b.__dict__ for b in available_bots(store.root)],
                    "decks": list(DECK_NAMES),
                    "cards": card_catalog(),
                })
                return

            if method == "POST" and parts == ["api", "games"]:
                body = self._body()
                game = new_session(
                    body.get("seats", []),
                    decks=tuple(body.get("decks") or DECK_NAMES),
                    seed=body.get("seed"),
                    first_player=int(body.get("first_player", 0)),
                )
                store.add(game)
                base = f"http://{self.headers.get('Host', 'localhost')}/"
                seats = {i: {"token": game.seats[i].token,
                             "link": f"{base}?game={game.id}&seat={i}&token={game.seats[i].token}"}
                         for i in game.humans}
                self._json({"id": game.id, "seats": seats, "spectate": f"{base}?game={game.id}"})
                return

            if len(parts) >= 3 and parts[1] == "games":
                game = store.get(parts[2])
                action = parts[3] if len(parts) > 3 else None
                if method == "GET" and action is None:
                    viewer = self._viewer(game, query)
                    self._json(game.view(viewer, reveal=query.get("reveal") == "1"))
                    return
                if method == "POST" and action == "act":
                    body = self._body()
                    game.act(int(body["seat"]), body.get("token"), int(body["action"]))
                    self._json({"ok": True})
                    return
                if method == "POST" and action == "bot_step":
                    self._json({"moved": game.bot_step()})
                    return
                if method == "GET" and action == "replay":
                    viewer = self._viewer(game, query)
                    if not (game.state.is_terminal() or game.reveal_allowed(viewer)):
                        raise PermissionError("The replay shows both hands; it is available when the game ends")
                    html = render_html(game.replay()).encode("utf-8")
                    self._send(HTTPStatus.OK, html, "text/html; charset=utf-8",
                               {"Content-Disposition": f'attachment; filename="micromagic_{game.id}.html"'})
                    return
            raise ApiError(HTTPStatus.NOT_FOUND, "Not found")

    return Handler


def lan_address() -> str:
    """This computer's address on the local network (no packets are sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def make_server(host: str = "127.0.0.1", port: int = 8765, root: Path | None = None) -> ThreadingHTTPServer:
    store = GameStore(Path(root or Path.cwd()))
    server = ThreadingHTTPServer((host, port), make_handler(store))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play MicroMagic in your browser.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--lan", action="store_true", help="let other computers on your network join games")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    args = parser.parse_args(argv)

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    server = make_server(host, args.port)
    local = f"http://127.0.0.1:{args.port}/"
    print(f"MicroMagic is running at {local}")
    if args.lan:
        print(f"Other players on your network can open http://{lan_address()}:{args.port}/")
    print("Bots available:", ", ".join(b.label for b in available_bots()))
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(local,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
