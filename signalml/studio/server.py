"""Studio HTTP server — a router over :mod:`signalml.studio.api`, stdlib only.

**Deliberately not FastAPI yet.** ``STUDIO.md`` §2 picks FastAPI, and that decision
stands: it is the right answer once the render queue needs WebSocket progress, which
lands with the renderer in P8. Everything servable *today* is read-only JSON over a
handful of routes, and this repo already runs two stdlib servers (``dash``, ``ship``),
so paying a dependency now would buy nothing. The logic lives in ``api.py`` as pure
functions, so swapping the transport later is a file, not a rewrite.

The handler owns no logic: every route is one call into ``api`` and one
``model_dump``. If a route ever needs a decision made here, it belongs in ``api.py``.
"""

from __future__ import annotations

import json
import re
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from ..synth import VarianceTrack
from . import api

MAX_BODY_BYTES = 4 << 20  # a variance track is a few hundred floats; 4 MiB is generous

_VARIANCE_ROUTE = re.compile(r"^/api/render/([0-9a-f]{16,64})/variance$")


class _Handler(BaseHTTPRequestHandler):
    server: StudioServer  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"
    server_version = "signalml-studio/1"

    # ---- routes

    def do_GET(self) -> None:  # noqa: N802 (stdlib API name)
        root = self.server.data_root
        profile = self.server.profile_name
        path, query = self._split_path()

        if path == "/api/context":
            self._ok(api.build_context(root, profile_name=profile).model_dump())
        elif path == "/api/scores":
            self._ok([s.model_dump() for s in api.list_scores(root)])
        elif path == "/api/voices":
            self._ok([v.model_dump() for v in api.list_voices(root)])
        elif path == "/api/queue":
            # Shaped now, served by P8: a queue with no renderer behind it would be a
            # list that is always empty, which is worse than an honest "not yet".
            self._error(
                HTTPStatus.NOT_IMPLEMENTED,
                "the render queue lands with the renderer in P8 (docs/STUDIO_UI.md §13)",
            )
        elif (match := _VARIANCE_ROUTE.match(path)) is not None:
            self._get_variance(match.group(1))
        elif path == "/api/segments/plan":
            self._plan_from_query(query)
        else:
            self._error(HTTPStatus.NOT_FOUND, f"no route {path}")

    def do_POST(self) -> None:  # noqa: N802 (stdlib API name)
        path, _ = self._split_path()
        if path != "/api/segments/plan":
            self._error(HTTPStatus.NOT_FOUND, f"no route {path}")
            return
        body = self._read_json()
        if body is None:
            return
        self._plan(
            score=body.get("score"),
            compare=body.get("compare"),
            voice=body.get("voice"),
            seed=body.get("seed", 0),
            min_rest=body.get("min_rest_sec"),
        )

    def do_PUT(self) -> None:  # noqa: N802 (stdlib API name)
        path, _ = self._split_path()
        match = _VARIANCE_ROUTE.match(path)
        if match is None:
            self._error(HTTPStatus.NOT_FOUND, f"no route {path}")
            return
        body = self._read_json()
        if body is None:
            return
        try:
            track = VarianceTrack.model_validate(body.get("variance", body))
        except ValueError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, f"invalid variance track: {exc}")
            return
        try:
            written = api.write_variance(
                self.server.data_root, match.group(1), track,
                notes=str(body.get("notes", "")),
            )
        except (FileNotFoundError, OSError) as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except ValueError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            return
        self._ok(written.model_dump(), status=HTTPStatus.CREATED if written.changed
                 else HTTPStatus.OK)

    # ---- handlers

    def _get_variance(self, key: str) -> None:
        try:
            track = api.read_variance(self.server.data_root, key)
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except (OSError, ValueError) as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            return
        self._ok(track.model_dump())

    def _plan_from_query(self, query: str) -> None:
        params = {k: v[0] for k, v in parse_qs(query, keep_blank_values=True).items()}
        self._plan(
            score=params.get("score"),
            compare=params.get("compare"),
            voice=params.get("voice"),
            seed=params.get("seed", 0),
            min_rest=params.get("min_rest_sec"),
        )

    def _plan(self, *, score, compare, voice, seed, min_rest) -> None:
        if not score:
            self._error(HTTPStatus.BAD_REQUEST, "missing 'score' (a path to score.json)")
            return
        try:
            seed_int = int(seed)
            min_rest_f = (
                api.DEFAULT_MIN_REST_SEC if min_rest in (None, "") else float(min_rest)
            )
        except (TypeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"bad numeric parameter: {exc}")
            return
        try:
            plan = api.build_render_plan(
                self.server.data_root,
                score,
                compare=compare or None,
                voice=voice or None,
                seed=seed_int,
                profile_name=self.server.profile_name,
                min_rest_sec=min_rest_f,
            )
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except ValueError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
            return
        self._ok(plan.model_dump())

    # ---- plumbing

    def _split_path(self) -> tuple[str, str]:
        path, _, query = self.path.partition("?")
        return path.rstrip("/") or "/", query

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "bad Content-Length")
            return None
        if length > MAX_BODY_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        f"body over {MAX_BODY_BYTES} bytes")
            return None
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}")
            return None
        if not isinstance(body, dict):
            self._error(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
            return None
        return body

    def _ok(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send(status, json.dumps({"error": message}).encode("utf-8"))

    def _send(self, status: HTTPStatus, body: bytes) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:  # keep the console quiet
        pass


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        data_root: str | Path,
        host: str = "127.0.0.1",
        port: int = 8770,
        *,
        profile_name: str | None = None,
    ):
        self.data_root = Path(data_root)
        self.profile_name = profile_name
        super().__init__((host, port), _Handler)


def serve(
    data_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8770,
    profile_name: str | None = None,
    open_browser: bool = False,
) -> None:
    server = StudioServer(data_root, host, port, profile_name=profile_name)
    context = api.build_context(data_root, profile_name=profile_name)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"signalml studio: {data_root} at {url} (Ctrl+C to stop)")
    print(
        f"  profile {context.audio_profile} ({context.sample_rate} Hz)"
        f"{'  ** DEV — renders are throwaway **' if context.is_dev_profile else ''}"
    )
    print(
        f"  checkpoint {context.checkpoint or '(none yet)'} · "
        f"{context.counts['scores']} scores · {context.counts['voices']} voices · "
        f"{context.counts['renders']} cached renders"
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nsignalml studio: stopped")
    finally:
        server.server_close()
