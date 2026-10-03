"""Review HTTP server - stdlib only, bound to localhost (private audio never leaves).

Same shape as ``dash`` and ``studio``: the handler routes, :mod:`.identity` decides.
"""

from __future__ import annotations

import json
import re
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import identity, singers

MAX_BODY_BYTES = 1 << 20
_AUDIO = re.compile(r"^/audio/(sng_\d+)$")
_FULL = re.compile(r"^/audio/(sng_\d+)/full$")


class _Handler(BaseHTTPRequestHandler):
    server: ReviewServer  # type: ignore[assignment]

    def do_GET(self) -> None:  # noqa: N802 (stdlib API name)
        url = urlsplit(self.path)
        if url.path in ("/", "/identity"):
            body = Path(__file__).with_name("identity.html").read_bytes()
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", body)
        elif url.path == "/singers":
            body = Path(__file__).with_name("singers.html").read_bytes()
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", body)
        elif url.path == "/api/singers":
            try:
                payload = singers.build_cards(self.server.data_root)
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {
                    "error": "no voice match yet - run `signalml review singers match`"})
                return
            self.server.singer_ids = {m["id"] for c in payload["cards"] for m in c["members"]}
            self._json(HTTPStatus.OK, payload)
        elif url.path == "/api/identity":
            payload = identity.build_cards(self.server.data_root)
            self.server.known_ids = {t["id"] for c in payload["cards"] for t in c["tracks"]}
            self._json(HTTPStatus.OK, payload)
        elif (m := _AUDIO.match(url.path)) is not None:
            q = parse_qs(url.query)
            try:
                wav = identity.snippet_wav(self.server.data_root, m.group(1),
                                           float(q.get("s", ["0"])[0]), float(q.get("d", ["6"])[0]))
            except (ValueError, OSError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send(HTTPStatus.OK, "audio/wav", wav)
        elif (m := _FULL.match(url.path)) is not None:
            try:
                wav = identity.full_track_wav(self.server.data_root, m.group(1))
            except (ValueError, OSError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send(HTTPStatus.OK, "audio/wav", wav)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 (stdlib API name)
        path = urlsplit(self.path).path
        if path not in ("/api/identity/labels", "/api/singers/labels"):
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY_BYTES:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "empty or oversized body"})
            return
        try:
            posted = json.loads(self.rfile.read(length)).get("labels") or []
            if path == "/api/singers/labels":
                if not self.server.singer_ids:  # a POST before any GET
                    self.server.singer_ids = singers.known_ids(self.server.data_root)
                labels = [singers.validate_label(x, self.server.singer_ids) for x in posted]
                store = singers.label_store(self.server.data_root)
            else:
                if not self.server.known_ids:  # a POST before any GET (tests, restarts)
                    self.server.known_ids = {t["id"] for c in identity.build_cards(
                        self.server.data_root)["cards"] for t in c["tracks"]}
                labels = [identity.validate_label(x, self.server.known_ids) for x in posted]
                store = identity.LabelStore(self.server.data_root)
        except (ValueError, AttributeError, TypeError, FileNotFoundError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        store.append(labels)
        self._json(HTTPStatus.OK, {"saved": len(labels)})

    def _json(self, code: int, payload) -> None:
        self._send(code, "application/json; charset=utf-8",
                   json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:  # keep the console quiet
        pass


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, data_root: str | Path, host: str = "127.0.0.1", port: int = 8775):
        self.data_root = Path(data_root)
        self.known_ids: set[str] = set()
        self.singer_ids: set[str] = set()
        super().__init__((host, port), _Handler)


def serve(data_root: str | Path, *, port: int = 8775, open_browser: bool = True,
          page: str = "identity") -> None:
    server = ReviewServer(data_root, "127.0.0.1", port)
    url = f"http://127.0.0.1:{server.server_address[1]}/{'singers' if page == 'singers' else ''}"
    print(f"signalml review: {url} (Ctrl+C to stop; answers save as you go)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nsignalml review: stopped - run `signalml review apply` to write the answers")
    finally:
        server.server_close()
