"""Dashboard server: one JSON endpoint + one self-contained page, stdlib only.

``build_status`` is a pure function of DATA_ROOT (manifest.jsonl + per-song
analysis.json) so it's contract-testable offline like every stage. The page
polls ``/api/status``; nothing here writes to the data root.
"""

from __future__ import annotations

import json
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..manifest import Manifest, StatusFlags

_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/netviz": ("netviz_demo.html", "text/html; charset=utf-8"),
    "/netviz.js": ("netviz.js", "text/javascript; charset=utf-8"),
}

STAGE_FLAGS = list(StatusFlags.model_fields)  # separated, cleaned, aligned, featurized


def _analysis(data_root: Path, song_id: str) -> dict:
    path = data_root / "songs" / song_id / "analysis.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def build_status(data_root: str | Path) -> dict:
    data_root = Path(data_root)
    manifest = Manifest.for_data_root(data_root)
    recs = manifest.records

    songs = []
    singers: dict[str, dict] = {}
    align_scores: list[float] = []
    profiles: set[str] = set()
    for rec in recs:
        analysis = _analysis(data_root, rec.id)
        features = analysis.get("features", {})
        profile = analysis.get("clean", {}).get("profile")
        if profile:
            profiles.add(profile)
        if rec.quality.align_score is not None:
            align_scores.append(rec.quality.align_score)

        songs.append({
            "id": rec.id,
            "singer": rec.meta.singer,
            "song": rec.meta.song,
            "duration_sec": rec.file.duration_sec,
            "source_quality": rec.meta.source_quality,
            "has_lyrics": rec.meta.has_lyrics,
            "stages": {flag: getattr(rec.status, flag) for flag in STAGE_FLAGS},
            "align_score": rec.quality.align_score,
            "bpm": features.get("bpm"),
            "key": features.get("key"),
        })

        s = singers.setdefault(rec.meta.singer or "(untagged)",
                               {"songs": 0, "hours": 0.0})
        s["songs"] += 1
        s["hours"] += (rec.file.duration_sec or 0.0) / 3600

    total_hours = sum(r.file.duration_sec or 0.0 for r in recs) / 3600
    stages = [{"name": "scanned", "done": len(recs), "total": len(recs)}] + [
        {"name": flag, "done": sum(1 for r in recs if getattr(r.status, flag)),
         "total": len(recs)}
        for flag in STAGE_FLAGS
    ]

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "profiles": sorted(profiles),
        "totals": {
            "songs": len(recs),
            "hours": round(total_hours, 2),
            "singers": len({r.meta.singer for r in recs if r.meta.singer}),
            "lyrics_pct": round(
                100 * sum(1 for r in recs if r.meta.has_lyrics) / len(recs), 1
            ) if recs else 0.0,
        },
        "stages": stages,
        "singers": [
            {"name": name, "songs": v["songs"], "hours": round(v["hours"], 2)}
            for name, v in sorted(singers.items(), key=lambda kv: -kv[1]["hours"])
        ],
        "align_scores": align_scores,
        "songs": songs,
    }


class _Handler(BaseHTTPRequestHandler):
    server: DashServer  # type: ignore[assignment]

    def do_GET(self) -> None:  # noqa: N802 (stdlib API name)
        if self.path in _ASSETS:
            filename, content_type = _ASSETS[self.path]
            body = Path(__file__).with_name(filename).read_bytes()
            self._send(200, content_type, body)
        elif self.path == "/api/status":
            payload = build_status(self.server.data_root)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:  # keep the console quiet
        pass


class DashServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, data_root: str | Path, host: str = "127.0.0.1", port: int = 8765):
        self.data_root = Path(data_root)
        super().__init__((host, port), _Handler)


def serve(data_root: str | Path, *, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> None:
    server = DashServer(data_root, host, port)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"signalml dash: watching {data_root} at {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nsignalml dash: stopped")
    finally:
        server.server_close()
