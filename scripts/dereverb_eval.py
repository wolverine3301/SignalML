"""Is de-reverb worth running on every separated phrase before a rig build? (S4b eval)

Three measurements, all on phrase excerpts (the phrase window plus ``--context`` seconds
either side, so the model sees the tail and the judge can measure it):

1. **Synthetic, with an answer key.** Dry studio phrases get a known room, hall or echo
   (seeded), or nothing at all (the do-no-harm arm). They go through the model and are
   scored against the dry original: SI-SDR, log-mel L1 (prod mel), multi-resolution
   log-STFT L1, pitch error. "Removed" = the share of the reverb's distance to the dry
   signal that the model took away.
2. **Labelled real phrases** (``--phrases``, JSONL of ``{"id", "utts", "i", "tags"}``,
   e.g. a blind listening test's answers). Keep / revert verdicts and before / after
   HNR, tail and gap, grouped by tag: a useful model keeps the reverb-tagged ones and
   leaves the clean ones alone.
3. **Random separated phrases** (``--random``): what the gate would do corpus-wide. That
   is, the share kept, why the others were reverted, and how many phrases under the HNR
   cutoff the de-reverbed audio lifts over it (``rescued``).

Writes ``results.jsonl``, ``summary.json``, ``report.md`` and a blind A/B listening page
(``listen/index.html``, answers kept in the browser and downloadable) under ``--out``.
Needs the ``train`` + ``restore`` extras and the checkpoint in ``configs/dereverb.yaml``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from signalml.manifest import Manifest, resolve_data_root
from signalml.stages.common import song_dir
from signalml.stages.dereverb import (
    judge_phrases,
    load_dereverb_config,
    make_backend,
    stem_input,
)

HNR_CUTOFF = 13.0


# ---------------------------------------------------------------- synthetic damage


def _ir(sr: int, rt60: float, predelay: float, damping_hz: float | None,
        rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(rt60 * 1.2 * sr)) / sr
    ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    if damping_hz:  # darker as it decays: one-pole low-pass, cutoff falling over time
        from scipy.signal import lfilter
        a = np.exp(-2 * np.pi * damping_hz / sr)
        ir = lfilter([1 - a], [1, -a], ir)
    return np.concatenate([np.zeros(int(predelay * sr)), ir])


def add_reverb(y: np.ndarray, sr: int, kind: str, rng: np.random.Generator) -> tuple:
    """Dry mono ``y`` -> (stereo (2, T): dry in the middle + a wide wet signal, params).

    Room and hall use an independent impulse response per channel (decorrelated, as a
    reverb plugin's output is); echo is a ping-pong delay (repeats alternate sides)."""
    from scipy.signal import fftconvolve, lfilter

    if kind == "none":
        return np.stack([y, y]).astype(np.float32), {}
    wet = np.zeros((2, len(y)))
    if kind == "echo":
        delay, fb = rng.uniform(0.25, 0.45), 0.4
        rep, a = y.copy(), np.exp(-2 * np.pi * 4000 / sr)
        for k in range(1, 4):
            rep = lfilter([1 - a], [1, -a], rep)  # each repeat a little darker
            d = int(k * delay * sr)
            if d < len(y):
                wet[k % 2, d:] += fb ** (k - 1) * rep[: len(y) - d]
        params = {"delay_s": round(delay, 3)}
        wet_db = rng.uniform(-10, -6)
    else:
        rt60 = rng.uniform(0.4, 0.9) if kind == "room" else rng.uniform(1.2, 2.5)
        for ch in range(2):
            ir = _ir(sr, rt60, 0.01 if kind == "room" else 0.025,
                     None if kind == "room" else 6000.0, rng)
            wet[ch] = fftconvolve(y, ir)[: len(y)]
        params = {"rt60_s": round(rt60, 2)}
        wet_db = rng.uniform(-12, -8) if kind == "room" else rng.uniform(-8, -4)
    gain = 10 ** (wet_db / 20) * np.sqrt(
        2 * (y ** 2).sum() / max((wet ** 2).sum(), 1e-12))
    params["wet_db"] = round(float(wet_db), 1)
    return (y[None, :] + gain * wet).astype(np.float32), params


# ---------------------------------------------------------------- metrics


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    s = (est @ ref) / max(ref @ ref, 1e-12) * ref
    return float(10 * np.log10(max(s @ s, 1e-12) / max((est - s) @ (est - s), 1e-12)))


def _logmel(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    from signalml.config import active_profile

    p = active_profile("prod")
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=p.n_fft, hop_length=p.hop_length,
                                       win_length=p.win_length, n_mels=p.n_mels,
                                       fmax=p.fmax or sr / 2)
    return np.log10(np.maximum(m, 1e-5))


def logmel_l1(est: np.ndarray, ref: np.ndarray, sr: int) -> float:
    return float(np.abs(_logmel(est, sr) - _logmel(ref, sr)).mean())


def mrstft_l1(est: np.ndarray, ref: np.ndarray) -> float:
    import librosa

    out = []
    for n in (512, 1024, 2048):
        a = np.abs(librosa.stft(est, n_fft=n, hop_length=n // 4))
        b = np.abs(librosa.stft(ref, n_fft=n, hop_length=n // 4))
        out.append(np.abs(np.log10(a + 1e-5) - np.log10(b + 1e-5)).mean())
    return float(np.mean(out))


def f0_cents(est: np.ndarray, ref: np.ndarray, sr: int) -> float | None:
    import parselmouth

    def track(y):
        return parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr).to_pitch(
            time_step=0.01, pitch_floor=75.0, pitch_ceiling=1100.0).selected_array["frequency"]

    a, b = track(est), track(ref)
    n = min(len(a), len(b))
    both = (a[:n] > 0) & (b[:n] > 0)
    return round(float(np.median(1200 * np.abs(np.log2(a[:n][both] / b[:n][both])))), 1) \
        if both.sum() >= 5 else None


def scores(est, ref, sr) -> dict:
    return {"si_sdr": round(si_sdr(est, ref), 2), "logmel_l1": round(logmel_l1(est, ref, sr), 4),
            "mrstft_l1": round(mrstft_l1(est, ref), 4), "f0_cents": f0_cents(est, ref, sr)}


# ---------------------------------------------------------------- excerpts


def _alignment(sdir: Path, utts_name: str):
    adir = sdir / "align"
    utts = json.loads((adir / utts_name).read_text(encoding="utf-8"))
    phones = json.loads((adir / ("phones" + utts_name[len("utterances"):])).read_text(
        encoding="utf-8"))["phones"]
    return utts, phones


def excerpt(y, sr, utts, phones, idx, ctx):
    """The phrase window +/- ctx, with the overlapping utterances and phones shifted."""
    u = utts[idx]
    a = max(0.0, u["start"] - ctx)
    b = min(len(y) / sr, u["end"] + ctx)
    seg = y[int(a * sr):int(b * sr)].astype(np.float32)
    local, li = [], None
    for j, v in enumerate(utts):
        if v["end"] > a and v["start"] < b:
            if j == idx:
                li = len(local)
            local.append({"start": round(max(v["start"], a) - a, 4),
                          "end": round(min(v["end"], b) - a, 4)})
    lp = [dict(p, start=p["start"] - a, end=p["end"] - a) for p in phones
          if p["start"] >= a and p["end"] <= b]
    return seg, local, lp, li, (a, b)


def _clip(seg, sr, local_u, pad=0.3):
    s = max(0, int((local_u["start"] - pad) * sr))
    e = min(len(seg), int((local_u["end"] + pad) * sr))
    return seg[s:e]


# ---------------------------------------------------------------- selection


def pick_synthetic(data_root: Path, manifest: Manifest, n: int, rng: random.Random):
    pool = []
    for rec in manifest.records:
        m = rec.meta
        if (m.excluded or m.source_quality != "studio" or m.processing not in (None, "dry")
                or not rec.status.cleaned
                or not (song_dir(data_root, rec.id) / "clean" / "vocals.wav").exists()):
            continue
        adir = song_dir(data_root, rec.id) / "align"
        names = sorted(p.name for p in adir.glob("utterances*.json")) if adir.is_dir() else []
        if names:
            pool.append((rec.id, names[0]))
    rng.shuffle(pool)
    out = []
    for sid, name in pool:
        if len(out) >= n:
            break
        utts, phones = _alignment(song_dir(data_root, sid), name)
        q = song_dir(data_root, sid) / "align" / ("quality" + name[len("utterances"):])
        hnr = json.loads(q.read_text(encoding="utf-8"))["phrases"] if q.exists() else None
        good = [i for i, u in enumerate(utts)
                if 2.0 <= u["end"] - u["start"] <= 9.0
                and (hnr is None or (hnr[i].get("hnr_db") or 0) >= 18.0)]
        if good:
            out.append((sid, name, rng.choice(good)))
    return out


def pick_random(data_root: Path, manifest: Manifest, n: int, rng: random.Random,
                exclude: set[str]):
    pool = []
    for rec in manifest.records:
        if (rec.meta.excluded or rec.meta.source_quality != "separated" or rec.id in exclude
                or not rec.status.cleaned
                or not (song_dir(data_root, rec.id) / "clean" / "vocals.wav").exists()):
            continue
        adir = song_dir(data_root, rec.id) / "align"
        names = sorted(p.name for p in adir.glob("utterances*.json")) if adir.is_dir() else []
        if names:
            pool.append((rec.id, names[-1]))  # the phrase variant sorts after the main file
    rng.shuffle(pool)
    out = []
    for sid, name in pool[: n]:
        utts, _ = _alignment(song_dir(data_root, sid), name)
        good = [i for i, u in enumerate(utts) if 1.5 <= u["end"] - u["start"] <= 12.0]
        if good:
            out.append((sid, name, rng.choice(good)))
    return out


# ---------------------------------------------------------------- listening page


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>De-reverb A/B</title><style>
body{font:15px system-ui;margin:16px;max-width:760px;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}
button{margin:4px 4px 0 0;padding:6px 10px;border:1px solid #888;border-radius:6px;background:#fff}
button.on{background:#222;color:#fff}audio{width:100%;margin:4px 0}textarea{width:100%}
.muted{color:#777;font-size:13px}</style></head><body>
<h2>De-reverb blind A/B</h2><p class="muted">Each pair is one phrase, original and
de-reverbed, in random order. Pick the one you'd rather train on. Answers stay in this
browser; press Download when done.</p><div id="list"></div>
<button onclick="dl()">Download answers</button><script>
const PAIRS=__PAIRS__;const KEY="dereverb_ab___STAMP__";
let ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
const L=document.getElementById("list");
PAIRS.forEach((p,i)=>{const d=document.createElement("div");d.className="card";
d.innerHTML=`<b>${i+1}.</b> <span class="muted">${p.label||""}</span>
<div>A</div><audio controls preload="none" src="${p.a}"></audio>
<div>B</div><audio controls preload="none" src="${p.b}"></audio>
<div>${["A better","B better","same","both bad"]
  .map(c=>`<button data-c="${c}">${c}</button>`).join("")}</div>
<textarea rows="1" placeholder="note"></textarea>`;
const bs=d.querySelectorAll("button"),ta=d.querySelector("textarea");
const cur=ans[p.key]||{};bs.forEach(b=>{if(b.dataset.c===cur.c)b.classList.add("on");
b.onclick=()=>{bs.forEach(x=>x.classList.remove("on"));b.classList.add("on");
ans[p.key]=Object.assign(ans[p.key]||{},{c:b.dataset.c});save()}});
ta.value=cur.note||"";ta.oninput=()=>{ans[p.key]=Object.assign(ans[p.key]||{},{note:ta.value});save()};
L.appendChild(d)});
function dl(){const b=new Blob([JSON.stringify(ans,null,1)],{type:"application/json"});
const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=KEY+".json";a.click()}
</script></body></html>"""


def write_page(out: Path, pairs: list[dict], stamp: str) -> None:
    (out / "listen" / "index.html").write_text(
        PAGE.replace("__PAIRS__", json.dumps(pairs)).replace("__STAMP__", stamp),
        encoding="utf-8")


# ---------------------------------------------------------------- main


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 3) if xs else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None, help="default DATA_ROOT/work/dereverb_eval_<date>")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--synthetic", type=int, default=30, help="dry studio phrases")
    ap.add_argument("--phrases", default=None, help="labelled phrases (JSONL)")
    ap.add_argument("--random", type=int, default=60, help="random separated phrases")
    ap.add_argument("--exclude-ids", default=None,
                    help="file with song ids to leave out of --random (one per line)")
    ap.add_argument("--context", type=float, default=1.5)
    ap.add_argument("--input", choices=["stems", "clean"], default=None,
                    help="what the model hears (default: the config's source)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=["synthetic", "labelled", "random"],
                    choices=["synthetic", "labelled", "random"])
    args = ap.parse_args(argv)

    data_root = resolve_data_root(args.data_root)
    out = Path(args.out) if args.out else \
        data_root / "work" / f"dereverb_eval_{_dt.date.today().isoformat()}"
    (out / "listen" / "clips").mkdir(parents=True, exist_ok=True)
    cfg = load_dereverb_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.input:
        cfg.source = args.input
    manifest = Manifest.for_data_root(data_root)
    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    exclude = set(Path(args.exclude_ids).read_text(encoding="utf-8").split()) \
        if args.exclude_ids else set()

    t0 = time.time()
    fn, info = make_backend(cfg, data_root)
    print(f"model loaded in {time.time() - t0:.1f}s: {info.get('device')}", flush=True)
    audio_cache: dict[str, tuple] = {}

    def song_audio(sid):
        """(clean mono, sr, stereo stem (C, T) or None, gain clean = gain * stem mix)."""
        if sid not in audio_cache:
            audio_cache.clear()  # one song at a time keeps memory flat
            sdir = song_dir(data_root, sid)
            y, sr = sf.read(str(sdir / "clean" / "vocals.wav"), dtype="float32")
            y = y.mean(axis=1) if y.ndim > 1 else y
            stem, gain = None, 1.0
            if cfg.source == "stems" and (sdir / "stems" / "vocals.wav").exists():
                st, ssr = sf.read(str(sdir / "stems" / "vocals.wav"), dtype="float32",
                                  always_2d=True)
                got = stem_input(y, st) if ssr == sr else None
                if got is not None:
                    stem, gain, _ = got
            audio_cache[sid] = (y, sr, stem, gain)
        return audio_cache[sid]

    rows, pairs, model_sec, audio_sec = [], [], 0.0, 0.0
    results = (out / "results.jsonl").open("w", encoding="utf-8")

    def run_case(arm, sid, name, idx, transform=None, extra=None):
        nonlocal model_sec, audio_sec
        y, sr, stem, gain = song_audio(sid)
        utts, phones = _alignment(song_dir(data_root, sid), name)
        seg, lu, lp, li, (a, b) = excerpt(y, sr, utts, phones, idx, args.context)
        if li is None or not lp:
            return None
        ref = seg
        params = {}
        if transform:  # synthetic: a dry studio phrase made wet, in stereo
            wide, params = transform(seg, sr)
            seg = wide.mean(axis=0)
            model_in = wide if cfg.source == "stems" else seg
            gain_in = 1.0
        elif stem is not None:
            model_in = stem[:, int(a * sr):int(a * sr) + len(seg)]
            gain_in = gain
        else:
            model_in, gain_in = seg, 1.0
        t = time.time()
        derev = gain_in * np.asarray(fn(model_in, sr), dtype=np.float32)[: len(seg)]
        model_sec += time.time() - t
        audio_sec += len(seg) / sr
        verdict = judge_phrases(seg, derev, sr, lu, lp, cfg.gate)[li]
        row = {"arm": arm, "id": sid, "utts": name, "i": idx, "window": [a, b],
               "input": "stereo" if np.ndim(model_in) == 2 else "mono",
               "verdict": verdict, **(extra or {})}
        if transform:
            w = slice(int(lu[li]["start"] * sr), int(lu[li]["end"] * sr))
            row["params"] = params
            row["before"] = scores(seg[w], ref[w], sr)
            row["after"] = scores(derev[w], ref[w], sr)
        rows.append(row)
        results.write(json.dumps(row, ensure_ascii=False) + "\n")
        results.flush()
        return seg, derev, lu[li], sr

    # 1. synthetic
    synthetic = pick_synthetic(data_root, manifest, args.synthetic, rng) \
        if "synthetic" in args.arms else []
    for k, (sid, name, idx) in enumerate(synthetic):
        for kind in ("none", "room", "hall", "echo"):
            run_case(f"synthetic:{kind}", sid, name, idx,
                     transform=lambda s, sr, kind=kind: add_reverb(s, sr, kind, nrng))
        print(f"synthetic {k + 1}/{args.synthetic}", flush=True)

    # 2. labelled phrases
    if args.phrases and "labelled" in args.arms:
        for line in Path(args.phrases).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            res = run_case("labelled", p["id"], p["utts"], int(p["i"]),
                           extra={"tags": p.get("tags") or [], "label": p.get("label")})
            if res:
                seg, derev, u, sr = res
                key = f"{p['id']}:{p['i']}"
                files = []
                for tag, sig in (("o", seg), ("d", derev)):
                    f = out / "listen" / "clips" / f"{key.replace(':', '_')}_{tag}.flac"
                    sf.write(str(f), _clip(sig, sr, u), sr)
                    files.append(f"clips/{f.name}")
                flip = rng.random() < 0.5
                pairs.append({"key": key, "a": files[1] if flip else files[0],
                              "b": files[0] if flip else files[1],
                              "dereverbed": "A" if flip else "B"})
        print(f"labelled: {sum(r['arm'] == 'labelled' for r in rows)}", flush=True)

    # 3. random separated phrases
    randoms = pick_random(data_root, manifest, args.random, rng, exclude) \
        if "random" in args.arms else []
    for sid, name, idx in randoms:
        run_case("random", sid, name, idx)
    print(f"random: {sum(r['arm'] == 'random' for r in rows)}", flush=True)
    results.close()

    # ---- summary
    summary = {"model": info, "gate": cfg.gate.model_dump(), "source": cfg.source,
               "inputs": {k: sum(r["input"] == k for r in rows) for k in ("stereo", "mono")},
               "audio_sec": round(audio_sec, 1), "model_sec": round(model_sec, 1),
               "realtime_x": round(audio_sec / max(model_sec, 1e-6), 1)}
    for kind in ("none", "room", "hall", "echo"):
        rs = [r for r in rows if r["arm"] == f"synthetic:{kind}"]
        if not rs:
            continue
        d = {"n": len(rs), "kept": round(np.mean([r["verdict"]["keep"] for r in rs]), 3)}
        for m in ("si_sdr", "logmel_l1", "mrstft_l1", "f0_cents"):
            d[m] = {"in": _mean([r["before"][m] for r in rs]),
                    "out": _mean([r["after"][m] for r in rs])}
        if kind != "none" and d["logmel_l1"]["in"]:
            d["removed_logmel"] = round(1 - d["logmel_l1"]["out"] / d["logmel_l1"]["in"], 3)
        summary[f"synthetic:{kind}"] = d
    for arm in ("labelled", "random"):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        v = [r["verdict"] for r in rs]
        reasons: dict[str, int] = {}
        for x in v:
            reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
        under = [x for x in v if (x["before"]["hnr_db"] or 99) < HNR_CUTOFF]
        rescued = [x for x in under if x["keep"] and (x["after"]["hnr_db"] or 0) >= HNR_CUTOFF]
        d = {"n": len(v), "kept": round(np.mean([x["keep"] for x in v]), 3),
             "reasons": reasons, "under_cutoff": len(under), "rescued": len(rescued),
             "hnr_change_kept": _mean([x["after"]["hnr_db"] - x["before"]["hnr_db"]
                                       for x in v if x["keep"]]),
             "tail_drop_kept": _mean([x["before"]["tail_db"] - x["after"]["tail_db"]
                                      for x in v if x["keep"] and x["before"]["tail_db"]
                                      is not None and x["after"]["tail_db"] is not None])}
        if arm == "labelled":
            by_tag: dict[str, list] = {}
            for r in rs:
                for t in r.get("tags") or ["(untagged)"]:
                    by_tag.setdefault(t, []).append(r["verdict"]["keep"])
            d["kept_by_tag"] = {t: f"{sum(k)}/{len(k)}" for t, k in sorted(by_tag.items())}
        summary[arm] = d
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    if pairs:
        write_page(out, [{k: p[k] for k in ("key", "a", "b")} for p in pairs], stamp)
        (out / "listen" / "key.json").write_text(
            json.dumps({p["key"]: p["dereverbed"] for p in pairs}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
