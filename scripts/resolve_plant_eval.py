"""How well does ``lyrics --resolve`` repair pasted lyrics? A planted-error test.

Ground truth: songs whose lyrics are hand-corrected to the performance (the original
corpus), restricted to ones Whisper hears well. Each trial copies the true lyrics,
plants ONE departure of the kind official lyrics have against a cover, resolves the
copy against the song's real Whisper transcript, and checks the result against truth:

  unsung_echo   "(...)" appended to a line, never sung          -> should be dropped
  sung_echo     sung words wrapped in parentheses               -> should be kept
  cut_section   2-4 lines from another song inserted (unsung)   -> should be dropped
  omitted_line  a sung line deleted from the text               -> (re-insert: candidate)
  swapped_word  a word replaced (boy -> girl), Whisper hears it -> (substitute: candidate)
  clean         nothing planted                                 -> nothing should change

Two candidate repairs are scored alongside the current resolver, so the report says
whether they are worth building: ``reinsert`` (a run of heard words that matches a line
of the song closely, where the text has nothing, is re-inserted from the text) and
``substitute`` (a single word Whisper heard confidently, flanked by matching words, and
spelled as a dictionary word, replaces the text's word).

Score: word edit distance to truth before and after (lower is better); a trial is
"fixed" when the resolved text equals the truth word for word, "damaged" when it ends
further from the truth than the planted copy was.

    python scripts/resolve_plant_eval.py --data-root <DATA_ROOT> --out <report.json>
        [--dictionary <english_mfa.dict>] [--trials 3] [--min-agreement 0.8]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signalml.manifest import Manifest  # noqa: E402
from signalml.stages.common import song_dir  # noqa: E402
from signalml.stages.lyrics_check import align_words, norm_words  # noqa: E402
from signalml.stages.lyrics_resolve import (  # noqa: E402
    _song_voiced,
    normalize_for_alignment,
    resolve,
)

SWAPS = {"boy": "girl", "girl": "boy", "he": "she", "she": "he", "him": "her",
         "her": "him", "his": "her", "baby": "darling", "love": "need", "never": "always",
         "always": "never", "night": "day", "day": "night", "heart": "soul", "man": "woman"}


def toks(text: str) -> list[str]:
    """Words as the aligner will read them (so "2" and "two" compare equal)."""
    return [w for ln in text.splitlines() for w in norm_words(normalize_for_alignment(ln))]


def edit(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def heard(segments: list[dict]) -> list[tuple[str, float]]:
    out = []
    for s in segments:
        for w in s.get("words") or []:
            out += [(t, float(w.get("p", 1.0))) for t in norm_words(w["w"])]
    return out


# ---- candidate repairs (text in, text out) ---------------------------------------

def reinsert(text: str, segments: list[dict], min_words: int = 4,
             min_sim: float = 0.8) -> str:
    """Where the heard words hold a run the text lacks, and that run matches one of the
    song's own lines closely, put that line (the text's spelling) back in place."""
    lines = [ln for ln in text.splitlines() if norm_words(ln)]
    ref, line_of = [], []
    for li, ln in enumerate(lines):
        toks = norm_words(ln)
        ref += toks
        line_of += [li] * len(toks)
    hyp = [w for w, _ in heard(segments)]
    ops = align_words(ref, hyp)
    inserts: dict[int, list[str]] = defaultdict(list)  # insert before line index
    run: list[int] = []
    last_ref = -1
    for op, i, j in [*ops, ("=", len(ref), -1)]:
        if op == "ins":
            run.append(j)
            continue
        if len(run) >= min_words:
            words = [hyp[k] for k in run]
            best, best_sim = None, 0.0
            for ln in lines:
                t = norm_words(ln)
                sim = 1 - edit(words, t) / max(len(words), len(t))
                if sim > best_sim:
                    best, best_sim = ln, sim
            if best is not None and best_sim >= min_sim:
                at = line_of[last_ref] + 1 if last_ref >= 0 else 0
                inserts[at].append(best)
        run = []
        if op in ("=", "sub", "del") and i < len(ref):
            last_ref = i
    out = []
    for li, ln in enumerate(lines):
        out += inserts.get(li, [])
        out.append(ln)
    out += inserts.get(len(lines), [])
    return "\n".join(out)


def substitute(text: str, segments: list[dict], vocab: set[str] | None,
               min_p: float = 0.9) -> str:
    """Replace a text word by the heard one where exactly that word differs, both
    neighbours matched, Whisper was confident, and the heard word is a real word."""
    lines = [ln for ln in text.splitlines() if norm_words(ln)]
    ref, pos = [], []
    for li, ln in enumerate(lines):
        for wi, t in enumerate(norm_words(ln)):
            ref.append(t)
            pos.append((li, wi))
    hw = heard(segments)
    ops = align_words(ref, [w for w, _ in hw])
    status = {i: (op, j) for op, i, j in ops if op in ("=", "sub", "del")}
    toks = [norm_words(ln) for ln in lines]
    for i, (op, j) in status.items():
        if op != "sub":
            continue
        if status.get(i - 1, ("",))[0] != "=" or status.get(i + 1, ("",))[0] != "=":
            continue
        word, p = hw[j]
        if p < min_p or len(word) < 2 or (vocab is not None and word not in vocab):
            continue
        li, wi = pos[i]
        toks[li][wi] = word
    return "\n".join(" ".join(t) for t in toks)


# ---- plants -----------------------------------------------------------------------

def plant(kind: str, lines: list[str], rnd: random.Random, donor: list[str]):
    """-> (planted text, truth text) or None when this song cannot take the plant."""
    truth = "\n".join(lines)
    n = len(lines)
    if kind == "clean":
        return truth, truth
    if kind == "unsung_echo":
        cands = [i for i, ln in enumerate(lines) if len(norm_words(ln)) >= 4]
        if not cands:
            return None
        i = rnd.choice(cands)
        tail = norm_words(lines[i])[-rnd.choice((2, 3)):]
        # an echo of words that are NOT next sung: skip lines followed by the same words
        nxt = norm_words(lines[i + 1])[:len(tail)] if i + 1 < n else []
        if nxt == tail:
            return None
        out = lines[:i] + [f"{lines[i]} ({' '.join(tail)})"] + lines[i + 1:]
        return "\n".join(out), truth
    if kind == "sung_echo":
        cands = [i for i, ln in enumerate(lines) if len(ln.split()) >= 5]
        if not cands:
            return None
        i = rnd.choice(cands)
        w = lines[i].split()
        k = rnd.choice((2, 3))
        out = lines[:i] + [" ".join(w[:-k]) + " (" + " ".join(w[-k:]) + ")"] + lines[i + 1:]
        return "\n".join(out), truth
    if kind == "cut_section":
        if len(donor) < 4:
            return None
        k = rnd.choice((2, 3, 4))
        s = rnd.randrange(0, len(donor) - k + 1)
        at = rnd.randrange(1, n)
        out = lines[:at] + donor[s:s + k] + lines[at:]
        return "\n".join(out), truth
    if kind == "omitted_line":
        # only lines that occur elsewhere in the song (a repeated chorus line) - the
        # realistic case, and the only one re-insertion could restore
        norm = [" ".join(norm_words(ln)) for ln in lines]
        cands = [i for i in range(n) if len(norm[i].split()) >= 4 and norm.count(norm[i]) >= 2]
        if not cands:
            return None
        i = rnd.choice(cands)
        return "\n".join(lines[:i] + lines[i + 1:]), truth
    if kind == "swapped_word":
        cands = [(i, wi) for i, ln in enumerate(lines)
                 for wi, w in enumerate(ln.split()) if w.lower().strip(",.!?") in SWAPS]
        if not cands:
            return None
        i, wi = rnd.choice(cands)
        w = lines[i].split()
        core = w[wi].lower().strip(",.!?")
        w[wi] = SWAPS[core]
        return "\n".join(lines[:i] + [" ".join(w)] + lines[i + 1:]), truth
    raise ValueError(kind)


KINDS = ["clean", "unsung_echo", "sung_echo", "cut_section", "omitted_line", "swapped_word"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dictionary", default=None, help="MFA .dict: real-word check")
    ap.add_argument("--trials", type=int, default=3, help="trials per song per kind")
    ap.add_argument("--min-agreement", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audio", action="store_true",
                    help="give resolve the cleaned vocal (voiced-stretch evidence)")
    args = ap.parse_args()
    root = Path(args.data_root)
    vocab = None
    if args.dictionary:
        vocab = {ln.split("\t", 1)[0].lower() for ln in
                 Path(args.dictionary).read_text(encoding="utf-8").splitlines() if ln}

    songs = []
    for rec in Manifest.for_data_root(root).records:
        asr = song_dir(root, rec.id) / "lyrics" / "asr.json"
        human = rec.meta.has_lyrics and rec.meta.lyrics_path and \
            not (rec.meta.lyrics_source or "").startswith("asr:")
        # the hand-corrected corpus; import-batch songs carry pasted official lyrics
        if not (human and asr.exists()) or "/RAW/20" in f"/{rec.meta.lyrics_path}":
            continue
        payload = json.loads(asr.read_text(encoding="utf-8"))
        if "segments" not in payload:
            continue
        text = (root / rec.meta.lyrics_path).read_text(encoding="utf-8", errors="replace")
        if "(" in text or "[" in text:
            continue  # truth must be plain: the plants add the brackets
        lines = [ln.strip() for ln in text.splitlines() if norm_words(ln)]
        segs = payload["segments"]
        ref = [w for ln in lines for w in norm_words(ln)]
        hyp = [w for w, _ in heard(segs)]
        agree = sum(op == "=" for op, _, _ in align_words(ref, hyp)) / max(1, len(ref))
        if agree >= args.min_agreement and len(lines) >= 8:
            voiced = _song_voiced(song_dir(root, rec.id)) if args.audio else None
            songs.append((rec.id, lines, segs, voiced))
    print(f"{len(songs)} ground-truth songs", flush=True)

    rnd = random.Random(args.seed)
    methods = {
        "resolve": lambda t, s, v: resolve(t, s, v).text,
        "resolve+reinsert": lambda t, s, v: reinsert(resolve(t, s, v).text, s),
        "resolve+substitute": lambda t, s, v: substitute(resolve(t, s, v).text, s, vocab),
    }
    trials = []
    for sid, lines, segs, voiced in songs:
        donors = [d for d in songs if d[0] != sid]
        for kind in KINDS:
            for _ in range(args.trials if kind != "clean" else 1):
                donor = rnd.choice(donors)[1] if donors else []
                p = plant(kind, lines, rnd, donor)
                if p is None:
                    continue
                planted, truth = p
                tw, pw = toks(truth), toks(planted)
                row = {"song": sid, "kind": kind, "planted_err": edit(pw, tw)}
                for name, fn in methods.items():
                    row[name] = edit(toks(fn(planted, segs, voiced)), tw)
                trials.append(row)

    report: dict = {"songs": len(songs), "trials": len(trials), "by_kind": {}}
    for kind in KINDS:
        rows = [r for r in trials if r["kind"] == kind]
        if not rows:
            continue
        k = {"trials": len(rows),
             "planted_err_mean": round(sum(r["planted_err"] for r in rows) / len(rows), 3)}
        for name in methods:
            k[name] = {
                "err_mean": round(sum(r[name] for r in rows) / len(rows), 3),
                "fixed": sum(r[name] == 0 for r in rows),
                "damaged": sum(r[name] > r["planted_err"] for r in rows),
            }
        report["by_kind"][kind] = k
    Path(args.out).write_text(json.dumps({"report": report, "trials": trials}, indent=1),
                              encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
