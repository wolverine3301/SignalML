"""scripts/repoint_sources.py - offline, synthetic manifest."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from signalml.manifest import Manifest, scan_directory
from signalml.stages.common import song_dir, update_analysis

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repoint_sources.py"


def _load():
    spec = importlib.util.spec_from_file_location("repoint_sources", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_repoint_swaps_source_resets_audio_chain_and_refuses_duplicates(tmp_path, make_wav):
    root = tmp_path / "dr"
    make_wav(root / "raw" / "a.wav", seconds=0.5, hz=220)
    make_wav(root / "raw" / "b.wav", seconds=0.5, hz=330)
    manifest, recs = scan_directory(root)
    for r in recs:
        r.status.separated = r.status.cleaned = r.status.aligned = True
        manifest.upsert(r)
    manifest.save()
    a, b = sorted(recs, key=lambda r: r.id)
    sdir = song_dir(root, a.id)
    (sdir / "align").mkdir(parents=True)
    (sdir / "align" / "quality.json").write_text("{}", encoding="utf-8")
    update_analysis(sdir, "clean", {"profile": "dev"})
    better = make_wav(tmp_path / "originals" / "a_full.wav", seconds=0.5, hz=440)
    dup = tmp_path / "originals" / "b_copy.wav"
    dup.write_bytes((root / a.file.path).read_bytes())  # a's song again: a duplicate
    with open(tmp_path / "map.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerows([["id", "path", "status"], [a.id, str(better), "ok"],
                     [b.id, str(dup), "ok"], ["sng_9999", str(better), "NO ORIGINAL"]])
    rc = _load().main(["--data-root", str(root), "--map", str(tmp_path / "map.csv"),
                       "--batch", "orig"])
    assert rc == 1  # one problem: the duplicate
    m = Manifest.for_data_root(root)
    ra, rb = m.get(a.id), m.get(b.id)
    assert ra.file.path == "RAW/orig/a_full.wav" and not ra.status.separated
    assert not ra.status.cleaned and ra.status.aligned  # timeline unchanged
    assert rb.file.path == b.file.path and rb.status.separated  # refused, untouched
    assert (sdir / "align" / "quality.json.stale").exists()
    hist = json.loads((sdir / "analysis.json").read_text(encoding="utf-8"))["source_history"]
    assert hist[-1]["file"]["path"] == a.file.path
    assert list(root.glob("manifest.jsonl.bak-*-pre-repoint"))
