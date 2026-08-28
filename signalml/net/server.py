"""Sender side of ``signalml ship`` — a token-gated, read-only HTTP server.

Deliberately stdlib-only: one port, one firewall rule, no new dependency, and it keeps
working unchanged if the receiver later becomes a Linux box or a cloud instance.

Threat model is a home LAN, and the surface is sized to match: the server serves
*only* files the plan lists, addressed by an opaque key (never by path, so traversal is
not expressible), read-only, over an ephemeral process that dies with Ctrl+C. The token
keeps a curious device on the same subnet from stumbling into the corpus; it is not a
substitute for TLS and this is not for the open internet.
"""

from __future__ import annotations

import json
import secrets
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .plan import PLAN_NAME, ShipPlan, resolve_source, stage_dir

CHUNK = 1 << 20  # 1 MiB — saturates 1 GbE without pinning a core on syscalls


def lan_address() -> str:
    """Best-guess LAN IPv4. Opens no connection (UDP connect just picks a route), so
    it survives a machine with VPN/WSL adapters as long as the default route is real."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "signalml-ship/1"

    # injected by serve()
    plan: ShipPlan
    token: str
    data_root: Path
    stage: Path
    repo_root: Path
    quiet: bool = False

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib hook
        if not self.quiet:
            print(f"  {self.address_string()} {fmt % args}", flush=True)

    # ---- helpers

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self) -> tuple[str, str] | None:
        """``/<token>/<verb>[/<arg>]`` -> (verb, arg), or None if the token is wrong."""
        parts = [p for p in self.path.split("?", 1)[0].split("/") if p]
        if len(parts) < 2 or not secrets.compare_digest(parts[0], self.token):
            return None
        return parts[1], (parts[2] if len(parts) > 2 else "")

    def _range(self, size: int) -> tuple[int, int] | None:
        header = self.headers.get("Range")
        if not header or not header.startswith("bytes="):
            return None
        spec = header[len("bytes="):].split(",")[0].strip()
        start_s, _, end_s = spec.partition("-")
        try:
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
        except ValueError:
            return None
        if start >= size or start < 0 or end < start:
            return None
        return start, min(end, size - 1)

    # ---- verbs

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib hook
        self.do_GET(head_only=True)

    def do_GET(self, head_only: bool = False) -> None:  # noqa: N802 - stdlib hook
        route = self._route()
        if route is None:
            # identical response for a bad token and a bad path: no probing signal
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        verb, arg = route

        if verb == "ping":
            self._json(HTTPStatus.OK, {
                "name": self.plan.name, "what": self.plan.what,
                "items": len(self.plan.items), "bytes": self.plan.total_bytes,
                "host": self.plan.source_host, "created": self.plan.created,
            })
            return

        if verb == "plan":
            body = self.plan.model_dump_json(indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(body)
            return

        if verb != "blob":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        item = self.plan.by_key(arg)
        if item is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "no such item"})
            return
        src = resolve_source(item, data_root=self.data_root, stage=self.stage,
                             repo_root=self.repo_root)
        if not src.exists():
            self._json(HTTPStatus.GONE,
                       {"error": f"{item.path} vanished since planning"})
            return

        size = src.stat().st_size
        span = self._range(size)
        start, end = span if span else (0, size - 1)
        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if span else HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Signalml-Sha256", item.sha256)
        if span:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        with open(src, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                block = fh.read(min(CHUNK, remaining))
                if not block:
                    break
                try:
                    self.wfile.write(block)
                except (BrokenPipeError, ConnectionResetError):
                    return  # receiver hung up; it will resume with a Range request
                remaining -= len(block)


def serve(
    plan: ShipPlan,
    *,
    data_root: str | Path,
    repo_root: str | Path,
    host: str = "0.0.0.0",
    port: int = 8770,
    token: str | None = None,
    quiet: bool = False,
    ready: threading.Event | None = None,
) -> ThreadingHTTPServer:
    """Start serving ``plan``. Blocks until Ctrl+C unless ``ready`` is passed (tests)."""
    data_root = Path(data_root)
    token = token or secrets.token_urlsafe(16)

    handler = type("ShipHandler", (_Handler,), {
        "plan": plan, "token": token, "data_root": data_root,
        "stage": stage_dir(data_root, plan.name), "repo_root": Path(repo_root),
        "quiet": quiet,
    })
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True

    advertised = host if host not in ("0.0.0.0", "") else lan_address()
    url = f"http://{advertised}:{port}/{token}"
    gb = plan.total_bytes / (1 << 30)
    if not quiet:
        print(f"serving {plan.name}: {len(plan.items)} item(s), {gb:.2f} GB")
        print(f"  plan: {stage_dir(data_root, plan.name) / PLAN_NAME}")
        print()
        print("On the training rig, run:")
        print(f"  signalml ship pull {url} --data-root <RIG_DATA_ROOT>")
        print()
        print("If it cannot connect, on THIS machine (admin PowerShell, once):")
        print(f'  New-NetFirewallRule -DisplayName "signalml ship" -Direction Inbound '
              f"-Protocol TCP -LocalPort {port} -Profile Private -Action Allow")
        print("  ...and disconnect any VPN with a 'block LAN traffic' setting.")
        print()
        print("Ctrl+C when the pull reports done.", flush=True)

    if ready is not None:
        ready.set()
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
    return httpd
