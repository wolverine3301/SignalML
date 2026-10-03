"""Speaker embeddings of sung vocal tracks with ECAPA-TDNN — the external half of the
multitrack singer match (``signalml review singers``).

Runs in its OWN venv (speechbrain + torch), never imported by signalml, the same posture
as whisper_lyrics.py, MFA and SOME. One invocation handles a whole batch so the model
loads once:

    python ecapa_embed.py --jobs jobs.json [--device cuda] [--savedir <model cache>]

jobs.json is ``[{"wav": "...", "out": "...", "spans": [[start_sec, end_sec], ...]}, ...]``;
``spans`` are the sung stretches to embed (omitted = the whole file). Each span is cut
into WINDOW_SEC windows; each ``out`` receives JSON ``{"model", "window_sec", "windows":
[[t0, t1], ...], "embeddings": [[192 floats], ...]}`` - one L2-normalised vector per
window, so the caller can average them or look at their spread. A failed track writes
``{"error": ...}`` instead, so one bad file never costs the batch.

The model is speechbrain/spkrec-ecapa-voxceleb (Apache-2.0), trained on VoxCeleb speech:
it measures voice identity, and is weaker on singing than on talking - which is why the
caller compares tracks of the SAME song (same studio, same mics) rather than trusting
absolute scores across the corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SR = 16000  # the model's input rate
WINDOW_SEC = 3.0
MIN_WINDOW_SEC = 1.5  # shorter tails carry too little voice to embed
MAX_WINDOWS = 60  # spread evenly over the track, so a long take is not just its first minute
BATCH = 32


def windows(spans: list[list[float]] | None, duration: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in spans or [[0.0, duration]]:
        t, end = max(0.0, start), min(duration, end)
        while end - t >= MIN_WINDOW_SEC:
            out.append((t, min(t + WINDOW_SEC, end)))
            t += WINDOW_SEC
    if len(out) > MAX_WINDOWS:
        step = (len(out) - 1) / (MAX_WINDOWS - 1)
        out = [out[round(i * step)] for i in range(MAX_WINDOWS)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--source", default=MODEL)
    ap.add_argument("--savedir", default=None,
                    help="local copy of the model (default: next to this script)")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = "cuda:0"  # speechbrain wants the index spelled out
    savedir = args.savedir or str(Path(__file__).resolve().parent / "pretrained_ecapa")
    # COPY, not symlinks: Windows refuses symlinks without developer mode
    model = EncoderClassifier.from_hparams(source=args.source, savedir=savedir,
                                           run_opts={"device": device},
                                           local_strategy=LocalStrategy.COPY)
    model.eval()

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    for i, job in enumerate(jobs):
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            y, sr = sf.read(job["wav"], dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            wins = windows(job.get("spans"), len(y) / sr)
            if not wins:
                raise RuntimeError("no sung stretch long enough to embed")
            g = np.gcd(SR, sr)
            clips = [resample_poly(y[int(a * sr): int(b * sr)], SR // g, sr // g).astype(np.float32)
                     for a, b in wins]
            embs = []
            for k in range(0, len(clips), BATCH):
                part = clips[k: k + BATCH]
                n = max(len(c) for c in part)
                wav = torch.zeros(len(part), n)
                for j, c in enumerate(part):
                    wav[j, : len(c)] = torch.from_numpy(c)
                lens = torch.tensor([len(c) / n for c in part])
                with torch.no_grad():
                    e = model.encode_batch(wav.to(device), lens.to(device)).squeeze(1)
                embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
            emb = np.concatenate(embs)
            payload = {"model": args.source, "window_sec": WINDOW_SEC,
                       "windows": [[round(a, 3), round(b, 3)] for a, b in wins],
                       "embeddings": np.round(emb, 5).tolist()}
        except Exception as exc:  # one bad file must not cost the batch
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        out.write_text(json.dumps(payload), encoding="utf-8")
        print(f"[{i + 1}/{len(jobs)}] {out.name}: "
              f"{payload.get('error') or len(payload['windows'])} windows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
