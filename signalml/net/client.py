"""Receiver side of ``signalml ship`` — pull, resume, verify, apply.

The rig pulls (rather than the laptop pushing) for two reasons: it is the machine you
are sitting at when you start a run, and resume/verify logic belongs where the bytes
land. The same asymmetry works in reverse later — the rig serves, the laptop pulls
checkpoints back.

Every transfer is idempotent. An item whose destination already matches size + sha256
is skipped; a partial ``.part`` resumes with an HTTP ``Range`` request; a completed
file is hashed before it is renamed into place. Re-running a half-finished pull is
always the right move.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..manifest import MANIFEST_NAME, Manifest, ManifestRecord, sha256_file
from .plan import (
    BUNDLE_NAME,
    CODE_SUBDIR,
    PLAN_NAME,
    RECEIPTS_NAME,
    ShipItem,
    ShipPlan,
    stage_dir,
)

CHUNK = 1 << 20
TIMEOUT = 30.0


@dataclass
class PullSummary:
    plan: ShipPlan
    fetched: int = 0
    skipped: int = 0
    bytes_fetched: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    manifest_merged: int = 0
    code_action: str = ""
    notes: list[str] = field(default_factory=list)


class Receipts:
    """path -> sha256 of what we last landed, so a re-run does not re-hash 35 GB."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def matches(self, key: str, digest: str) -> bool:
        return self._data.get(key) == digest

    def record(self, key: str, digest: str) -> None:
        self._data[key] = digest

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=0), encoding="utf-8")
        os.replace(tmp, self.path)


def _get(url: str, *, start: int = 0):
    req = urllib.request.Request(url)
    if start:
        req.add_header("Range", f"bytes={start}-")
    return urllib.request.urlopen(req, timeout=TIMEOUT)  # noqa: S310 - user-supplied LAN URL


def fetch_plan(base_url: str) -> ShipPlan:
    with _get(f"{base_url.rstrip('/')}/plan") as resp:
        return ShipPlan.model_validate_json(resp.read().decode("utf-8"))


def ping(base_url: str) -> dict:
    with _get(f"{base_url.rstrip('/')}/ping") as resp:
        return json.loads(resp.read().decode("utf-8"))


def _landing(item: ShipItem, *, data_root: Path, incoming: Path) -> Path:
    """Where an item's bytes land. Only plain data files go straight to their final
    home; anything that gets merged or applied stages first."""
    if item.kind == "file" and item.dest == "data":
        return data_root / item.path
    return incoming / item.dest / item.path


def _download(url: str, dest: Path, item: ShipItem, *, retries: int = 3,
              on_bytes=None) -> None:
    """Resumable, hash-verified single-file download (temp + atomic rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > item.size:
            part.unlink()
            have = 0
        hasher = hashlib.sha256()
        if have:
            with open(part, "rb") as fh:
                while block := fh.read(CHUNK):
                    hasher.update(block)
        try:
            if have == item.size:
                pass  # nothing left to pull; fall through to the hash check
            else:
                with _get(url, start=have) as resp, open(part, "ab") as out:
                    while block := resp.read(CHUNK):
                        out.write(block)
                        hasher.update(block)
                        if on_bytes:
                            on_bytes(len(block))
            if hasher.hexdigest() == item.sha256:
                os.replace(part, dest)
                return
            part.unlink(missing_ok=True)  # corrupt: start clean rather than resume
            raise OSError(f"sha256 mismatch for {item.path}")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 10))
            _ = exc


def _merge_manifest(incoming_file: Path, data_root: Path) -> int:
    """Upsert by id, so successive shipments accumulate instead of clobbering."""
    manifest = Manifest.for_data_root(data_root)
    merged = 0
    for line in incoming_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        manifest.upsert(ManifestRecord.model_validate_json(line))
        merged += 1
    manifest.save()
    return merged


def _git(repo: Path | None, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    out = subprocess.run(
        ["git", *(["-C", str(repo)] if repo else []), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out


def _apply_code(plan: ShipPlan, incoming: Path, repo_root: Path) -> str:
    """Clone the bundle on first arrival, fetch + fast-forward on every one after."""
    bundle = incoming / "code" / CODE_SUBDIR / BUNDLE_NAME
    if not bundle.exists():
        return "no code in this shipment"
    branch = plan.git.branch if plan.git else "main"
    commit = plan.git.commit if plan.git else None

    if not (repo_root / ".git").exists():
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        if repo_root.exists() and any(repo_root.iterdir()):
            raise RuntimeError(
                f"{repo_root} exists, is not a git repo, and is not empty — point "
                f"--repo-dir at a new directory or move the old one aside")
        _git(None, "clone", "--branch", branch, str(bundle), str(repo_root))
        action = f"cloned {branch} @ {commit[:8] if commit else '?'} -> {repo_root}"
    else:
        _git(repo_root, "fetch", str(bundle), "+refs/heads/*:refs/remotes/ship/*")
        # -uno: untracked files (our own .ship/ side files included) never block a
        # checkout, only locally modified tracked files do
        dirty = bool(
            _git(repo_root, "status", "--porcelain", "-uno").stdout.strip())
        if dirty:
            action = (f"fetched into refs/remotes/ship/* but did NOT check out — the "
                      f"rig worktree has local changes. Resolve, then: "
                      f"git checkout -B {branch} ship/{branch}")
        else:
            _git(repo_root, "checkout", "-B", branch, f"ship/{branch}")
            action = f"updated {branch} -> {commit[:8] if commit else '?'}"

    # side files (provenance, and the diff if this shipped dirty) live in the repo
    side = repo_root / CODE_SUBDIR
    side.mkdir(parents=True, exist_ok=True)
    for extra in (incoming / "code" / CODE_SUBDIR).glob("*"):
        if extra.name != BUNDLE_NAME:
            shutil.copy2(extra, side / extra.name)

    sub = _git(repo_root, "submodule", "update", "--init", "--recursive", check=False)
    if sub.returncode != 0:
        action += " (submodule init failed — bootstrap_rig.ps1 retries it)"
    return action


def pull(
    base_url: str,
    *,
    data_root: str | Path,
    repo_root: str | Path | None = None,
    apply_code: bool = True,
    progress: bool = True,
) -> PullSummary:
    """Fetch every item in the served plan into ``data_root`` (and ``repo_root``)."""
    data_root = Path(data_root)
    base_url = base_url.rstrip("/")
    plan = fetch_plan(base_url)
    stage = stage_dir(data_root, plan.name)
    incoming = stage / "incoming"
    stage.mkdir(parents=True, exist_ok=True)
    plan.save(stage / PLAN_NAME)
    receipts = Receipts(stage / RECEIPTS_NAME)
    summary = PullSummary(plan=plan)

    todo: list[tuple[ShipItem, Path]] = []
    for item in plan.items:
        if item.dest == "code" and (not apply_code or repo_root is None):
            continue
        dest = _landing(item, data_root=data_root, incoming=incoming)
        if dest.exists() and dest.stat().st_size == item.size and (
            receipts.matches(item.path, item.sha256) or sha256_file(dest) == item.sha256
        ):
            receipts.record(item.path, item.sha256)
            summary.skipped += 1
            continue
        todo.append((item, dest))

    bar = None
    if progress and todo:
        from tqdm import tqdm
        bar = tqdm(total=sum(i.size for i, _ in todo), unit="B", unit_scale=True,
                   unit_divisor=1024, desc="pulling")

    for item, dest in todo:
        try:
            _download(f"{base_url}/blob/{item.key}", dest, item,
                      on_bytes=(bar.update if bar else None))
        except Exception as exc:  # network/disk: keep going, report at the end
            summary.failed[item.path] = str(exc)
            continue
        receipts.record(item.path, item.sha256)
        summary.fetched += 1
        summary.bytes_fetched += item.size
    if bar:
        bar.close()
    receipts.save()

    manifest_item = next((i for i in plan.items if i.kind == "manifest"), None)
    if manifest_item and manifest_item.path not in summary.failed:
        landed = _landing(manifest_item, data_root=data_root, incoming=incoming)
        if landed.exists():
            summary.manifest_merged = _merge_manifest(landed, data_root)

    if apply_code and repo_root is not None and plan.git is not None:
        if summary.failed:
            summary.code_action = "skipped — data items failed; re-run the pull first"
        else:
            summary.code_action = _apply_code(plan, incoming, Path(repo_root))

    for note in plan.notes:
        summary.notes.append(note)
    return summary


def load_local_plan(data_root: str | Path, name: str) -> ShipPlan:
    """The copy of the plan this receiver stored, for a later ``ship verify``."""
    path = stage_dir(data_root, name) / PLAN_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no shipment named {name!r} under {Path(data_root) / 'ship'} "
            f"(expected {path}); pass --plan explicitly")
    return ShipPlan.load(path)


__all__ = [
    "MANIFEST_NAME",
    "PullSummary",
    "fetch_plan",
    "load_local_plan",
    "ping",
    "pull",
]
