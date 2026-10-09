"""Point manifest records at a better source file, so S3/S4 rebuild them from it.

Use case (2026-10-09): songs whose source in the manifest is an old separation (e.g. a
2020 Spleeter vocal, cut at 11 kHz) while the original full mix still exists. The
original is sample-identical in time to the old source, so alignments, lyrics and
transcripts stay valid; only the audio chain is redone.

For each ``id,path`` row of ``--map`` (CSV; rows whose ``status`` column is not "ok"
are skipped):

- copies the file to ``DATA_ROOT/RAW/<batch>/<file name>`` (kept if already there),
- refuses it when another record already has that checksum (it would duplicate a song),
- updates ``file`` (path, sha256, duration, rate, channels) and appends the old source
  to ``analysis.json`` ``source_history``,
- resets ``status.separated`` and ``status.cleaned`` (aligned / transcribed stay),
- renames artifacts measured on the old audio to ``*.stale`` (align/quality*.json,
  align/dereverb*.json, clean/vocals.dereverb.wav) so no gate reads them.

Writes a manifest backup first (``manifest.jsonl.bak-<time>-pre-repoint``). ``--dry-run``
reports without touching anything. Then: ``signalml separate`` + ``signalml clean
--profile prod`` (they pick up exactly these records), ``signalml quality --force``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

from signalml.manifest import Manifest, probe_audio, resolve_data_root, sha256_file
from signalml.stages.common import song_dir, update_analysis

STALE = ("align/quality*.json", "align/dereverb*.json", "clean/vocals.dereverb.wav")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--map", required=True, help="CSV with columns id,path[,status]")
    ap.add_argument("--path-column", default="path")
    ap.add_argument("--batch", required=True, help="RAW/<batch>/ receives the files")
    ap.add_argument("--reason", default="better source")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = resolve_data_root(args.data_root)
    manifest = Manifest.for_data_root(root)
    rows = list(csv.DictReader(open(args.map, encoding="utf-8")))
    rows = [r for r in rows if r.get("status", "ok") == "ok" and r.get(args.path_column)]
    dest_dir = root / "RAW" / args.batch
    by_sha = {r.file.sha256: r.id for r in manifest.records}
    done, problems = [], {}
    if not args.dry_run:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(root / "manifest.jsonl",
                        root / f"manifest.jsonl.bak-{stamp}-pre-repoint")
        dest_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        rid, src = row["id"], Path(row[args.path_column])
        rec = manifest.get(rid)
        if rec is None:
            problems[rid] = "not in the manifest"
            continue
        if not src.exists():
            problems[rid] = f"source missing: {src}"
            continue
        dest = dest_dir / src.name
        if args.dry_run:
            done.append(rid)
            continue
        if not dest.exists():
            shutil.copyfile(src, dest)
        digest = sha256_file(dest)
        owner = by_sha.get(digest)
        if owner and owner != rid:
            problems[rid] = f"checksum already belongs to {owner} (duplicate song)"
            continue
        old = rec.file.model_dump()
        duration, rate, channels = probe_audio(dest)
        rec.file.path = dest.relative_to(root).as_posix()
        rec.file.sha256 = digest
        rec.file.duration_sec, rec.file.sample_rate, rec.file.channels = duration, rate, channels
        rec.status.separated = False
        rec.status.cleaned = False
        sdir = song_dir(root, rid)
        stale = []
        for pattern in STALE:
            for p in sdir.glob(pattern):
                p.rename(p.with_name(p.name + ".stale"))
                stale.append(p.relative_to(sdir).as_posix())
        analysis = json.loads((sdir / "analysis.json").read_text(encoding="utf-8")) \
            if (sdir / "analysis.json").exists() else {}
        history = analysis.get("source_history") or []
        history.append({"file": old, "replaced": _dt.date.today().isoformat(),
                        "reason": args.reason, "stale_renamed": stale})
        update_analysis(sdir, "source_history", history)
        manifest.upsert(rec)
        by_sha[digest] = rid
        done.append(rid)
    if not args.dry_run:
        manifest.save()
    print(f"repoint{' (dry run)' if args.dry_run else ''}: {len(done)} records, "
          f"{len(problems)} problems -> RAW/{args.batch}")
    for rid, why in problems.items():
        print(f"  {rid}: {why}", file=sys.stderr)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
