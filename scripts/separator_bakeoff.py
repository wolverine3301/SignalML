"""Which separator should S3 use? Ground truth from MedleyDB: its mixes really are the
sum of its stems, so the vocal stems summed are the answer key for each mix.

Caveat found on the first run (2026-10-09): in some MedleyDB V2 tracks the stems sit
1-3 ms off the mix (65-117 samples). That collapses SI-SDR while barely touching the
spectral metrics, so every waveform comparison here first aligns the two signals by
cross-correlation (``best_lag``), and ``prepare`` aligns the truth to the mix.

Three steps, so the 44 GB corpus never has to travel:

``prepare`` (wherever MedleyDB is): picks vocal tracks without stem bleed (metadata),
cuts the ``--seconds`` window with the most vocal energy from each, and writes a small
kit: ``<kit>/<track>/{mix.wav, vocals.wav}`` + ``kit.json``.

``run`` (the GPU box): separates every kit mix with each S3 config (``separate.yaml``'s
Demucs and the ``separate.roformer_*.yaml`` candidates) and scores the vocal estimate
against the truth: SI-SDR, log-mel L1 (prod mel), multi-resolution log-STFT L1, and
**bleed**: the estimate's level where the true vocal is silent, relative to its level
where the vocal sings (instruments leaking into the vocal stem, the corpus's main
damage). Writes ``results.json``, ``results.md`` and the estimates for listening.

    python scripts/separator_bakeoff.py prepare --medleydb <V2 dir> --metadata <dir> --kit <dir>
    python scripts/separator_bakeoff.py run --kit <dir> --data-root <DATA_ROOT>
    python scripts/separator_bakeoff.py score --kit <dir>   # re-score saved estimates
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

VOCAL_WORDS = ("vocalist", "singer", "vocal", "choir", "rapper", "beatbox")
SR = 44100
MAX_LAG = 2048


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = MAX_LAG) -> int:
    """Samples to shift ``b`` by (positive = later) to line it up with ``a``."""
    from scipy.signal import correlate

    n = min(len(a), len(b))
    c = correlate(a[:n], b[:n], mode="full", method="fft")
    mid = n - 1
    window = c[mid - max_lag: mid + max_lag + 1]
    return int(np.argmax(np.abs(window))) - max_lag


def shift(x: np.ndarray, lag: int) -> np.ndarray:
    """``x`` delayed by ``lag`` samples (advanced if negative), same length."""
    out = np.zeros_like(x)
    if lag >= 0:
        out[lag:] = x[: len(x) - lag]
    else:
        out[:lag] = x[-lag:]
    return out


# ---------------------------------------------------------------- prepare


def vocal_stems(meta: dict) -> list[str]:
    out = []
    for stem in (meta.get("stems") or {}).values():
        inst = stem.get("instrument")
        names = inst if isinstance(inst, list) else [inst]
        if any(w in str(n).lower() for n in names for w in VOCAL_WORDS):
            out.append(stem["filename"])
    return out


def prepare(args) -> int:
    root, meta_dir, kit = Path(args.medleydb), Path(args.metadata), Path(args.kit)
    kit.mkdir(parents=True, exist_ok=True)
    picked = []
    for track_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = track_dir.name
        meta_path = meta_dir / f"{name}_METADATA.yaml"
        mix = track_dir / f"{name}_MIX.wav"
        if not (meta_path.exists() and mix.exists()):
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("has_bleed", "")).lower() in ("yes", "true"):
            continue
        stems = vocal_stems(meta)
        if not stems:
            continue
        files = [track_dir / f"{name}_STEMS" / f for f in stems]
        if not all(f.exists() for f in files):
            continue
        y, sr = sf.read(str(mix), dtype="float32", always_2d=True)
        voc = sum(sf.read(str(f), dtype="float32", always_2d=True)[0] for f in files)
        n = min(len(y), len(voc))
        y, voc = y[:n], voc[:n]
        # stems can sit a few ms off the mix: line the vocal truth up with it, using all
        # stems summed (the closest thing to the mix) over the middle of the song
        everything = sum(sf.read(str(f), dtype="float32", always_2d=True)[0]
                         for f in sorted((track_dir / f"{name}_STEMS").glob("*.wav")))
        mid = slice(n // 2 - 10 * sr, n // 2 + 10 * sr) if n > 20 * sr else slice(0, n)
        lag = best_lag(y[mid].mean(axis=1), everything[:n][mid].mean(axis=1))
        voc = shift(voc, lag)
        w = int(args.seconds * sr)
        if n < w:
            continue
        e = np.concatenate([[0.0], np.cumsum((voc.astype(np.float64) ** 2).sum(axis=1))])
        start = int(np.argmax(e[w:] - e[:-w]))
        out = kit / name
        out.mkdir(exist_ok=True)
        sf.write(str(out / "mix.wav"), y[start:start + w], sr, subtype="FLOAT")
        sf.write(str(out / "vocals.wav"), voc[start:start + w], sr, subtype="FLOAT")
        picked.append({"track": name, "start_sec": round(start / sr, 2), "stems": stems,
                       "genre": meta.get("genre"), "stem_lag_samples": lag})
        print("picked", name, meta.get("genre"), len(stems), "vocal stem(s)", flush=True)
        if len(picked) >= args.n:
            break
    (kit / "kit.json").write_text(json.dumps({"seconds": args.seconds, "tracks": picked},
                                             indent=1), encoding="utf-8")
    print(len(picked), "tracks ->", kit)
    return 0


# ---------------------------------------------------------------- run


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est, ref = est - est.mean(), ref - ref.mean()
    s = (est @ ref) / max(ref @ ref, 1e-12) * ref
    return float(10 * np.log10(max(s @ s, 1e-12) / max((est - s) @ (est - s), 1e-12)))


def logmel_l1(est, ref) -> float:
    import librosa

    from signalml.config import active_profile

    p = active_profile("prod")

    def m(y):
        return np.log10(np.maximum(librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=p.n_fft, hop_length=p.hop_length, win_length=p.win_length,
            n_mels=p.n_mels, fmax=p.fmax or SR / 2), 1e-5))
    return float(np.abs(m(est) - m(ref)).mean())


def mrstft_l1(est, ref) -> float:
    import librosa

    out = []
    for n in (512, 1024, 2048):
        a = np.abs(librosa.stft(est, n_fft=n, hop_length=n // 4))
        b = np.abs(librosa.stft(ref, n_fft=n, hop_length=n // 4))
        out.append(np.abs(np.log10(a + 1e-5) - np.log10(b + 1e-5)).mean())
    return float(np.mean(out))


def bleed_db(est, ref) -> float | None:
    """Estimate level where the truth is silent (< -50 dB of its peak frame) relative to
    its level where the truth sings: how much of the band leaks into the vocal stem."""
    hop = 1024
    n = len(ref) // hop
    if n < 10:
        return None
    rf = (ref[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
    ef = (est[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
    rdb = 10 * np.log10(rf + 1e-12)
    silent, sung = rdb < rdb.max() - 50, rdb > rdb.max() - 20
    if silent.sum() < 5 or sung.sum() < 5:
        return None
    return float(10 * np.log10(ef[silent].mean() + 1e-12) - 10 * np.log10(ef[sung].mean() + 1e-12))


def score_pair(est: np.ndarray, ref: np.ndarray) -> dict:
    """Mono estimate vs mono truth, after lining the estimate up with the truth."""
    lag = best_lag(ref, est)
    e = shift(est, lag)
    return {"si_sdr": round(si_sdr(e, ref), 2), "logmel_l1": round(logmel_l1(e, ref), 4),
            "mrstft_l1": round(mrstft_l1(e, ref), 4),
            "bleed_db": None if (b := bleed_db(e, ref)) is None else round(b, 1),
            "lag": lag}


def summarize(kit: Path, results: dict) -> None:
    def mean(per, k):
        vals = [v[k] for v in per.values() if v[k] is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    for res in results.values():
        for k in ("si_sdr", "logmel_l1", "mrstft_l1", "bleed_db"):
            res[k] = mean(res["tracks"], k)
    (kit / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lines = ["| separator | SI-SDR dB | log-mel L1 | MR-STFT L1 | bleed dB | x realtime |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['si_sdr']} | {v['logmel_l1']} | {v['mrstft_l1']} | "
              f"{v['bleed_db']} | {v.get('realtime_x')} |" for k, v in results.items()]
    (kit / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def score(args) -> int:
    """Re-score the estimates a run saved (``_estimates/<separator>/<track>.wav``)."""
    kit = Path(args.kit)
    tracks = json.loads((kit / "kit.json").read_text(encoding="utf-8"))["tracks"]
    old = json.loads((kit / "results.json").read_text(encoding="utf-8")) \
        if (kit / "results.json").exists() else {}
    results = {}
    for est_dir in sorted(p for p in (kit / "_estimates").iterdir() if p.is_dir()):
        per = {}
        for t in tracks:
            f = est_dir / f"{t['track']}.wav"
            if not f.exists():
                continue
            est = sf.read(str(f), dtype="float32", always_2d=True)[0].mean(axis=1)
            ref = sf.read(str(kit / t["track"] / "vocals.wav"), dtype="float32",
                          always_2d=True)[0].mean(axis=1)
            n = min(len(est), len(ref))
            per[t["track"]] = score_pair(est[:n], ref[:n])
            print(est_dir.name, t["track"], per[t["track"]], flush=True)
        results[est_dir.name] = {**{k: v for k, v in old.get(est_dir.name, {}).items()
                                    if k in ("config", "load_sec", "realtime_x")},
                                 "tracks": per}
    summarize(kit, results)
    return 0


def run(args) -> int:
    from signalml.config import CONFIGS_DIR
    from signalml.stages.separate import (
        build_demucs_separator,
        build_roformer_separator,
        load_separate_config,
    )

    kit = Path(args.kit)
    tracks = json.loads((kit / "kit.json").read_text(encoding="utf-8"))["tracks"]
    configs = args.configs or ["separate.yaml", "separate.roformer_bs.yaml",
                               "separate.roformer_mel.yaml"]
    results: dict[str, dict] = {}
    for name in configs:
        cfg = load_separate_config(CONFIGS_DIR / name if not Path(name).is_absolute() else name)
        if args.device:
            cfg.device = args.device
        label = cfg.model
        t0 = time.time()
        sep = (build_roformer_separator(cfg, args.data_root) if cfg.backend == "roformer"
               else build_demucs_separator(cfg))
        load_sec = time.time() - t0
        per, audio_sec, model_sec = {}, 0.0, 0.0
        for t in tracks:
            d = kit / t["track"]
            t1 = time.time()
            stems, sr, _ = sep(d / "mix.wav")
            model_sec += time.time() - t1
            est = np.asarray(stems["vocals"], dtype=np.float32)
            est = est.T if est.shape[0] <= 8 else est   # (C, T) -> (T, C)
            ref, _ = sf.read(str(d / "vocals.wav"), dtype="float32", always_2d=True)
            n = min(len(est), len(ref))
            e, r = est[:n].mean(axis=1), ref[:n].mean(axis=1)
            audio_sec += n / sr
            out = kit / "_estimates" / label
            out.mkdir(parents=True, exist_ok=True)
            sf.write(str(out / f"{t['track']}.wav"), est[:n], sr)
            per[t["track"]] = score_pair(e, r)
            print(label, t["track"], per[t["track"]], flush=True)
        results[label] = {"config": name, "load_sec": round(load_sec, 1),
                          "realtime_x": round(audio_sec / max(model_sec, 1e-6), 1),
                          "tracks": per}
        del sep
    summarize(kit, results)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--medleydb", required=True, help="MedleyDB V2 audio dir (track folders)")
    p.add_argument("--metadata", required=True, help="MedleyDB Metadata dir")
    p.add_argument("--kit", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seconds", type=float, default=45.0)
    r = sub.add_parser("run")
    r.add_argument("--kit", required=True)
    r.add_argument("--data-root", required=True, help="checkpoints resolve against this")
    r.add_argument("--configs", nargs="*", default=None)
    r.add_argument("--device", default=None)
    sc = sub.add_parser("score")
    sc.add_argument("--kit", required=True)
    args = ap.parse_args(argv)
    return {"prepare": prepare, "run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
