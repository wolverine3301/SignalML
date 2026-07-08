"""Shared helpers for stage jobs (per-song directories, analysis.json)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def song_dir(data_root: str | Path, song_id: str) -> Path:
    return Path(data_root) / "songs" / song_id


def update_analysis(directory: str | Path, section: str, payload: dict) -> Path:
    """Merge one stage's section into ``<song_dir>/analysis.json`` (atomic write)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "analysis.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[section] = payload
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path
