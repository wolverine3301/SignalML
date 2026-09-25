"""Transcribe sung vocals to lyrics with faster-whisper — the external half of S5a.

Runs in its OWN venv (faster-whisper + ctranslate2), never imported by signalml, the
same posture as MFA, SOME and the trainer. One invocation handles a whole batch so the
~3 GB model loads once:

    python whisper_lyrics.py --model <dir-or-name> --jobs jobs.json [--device cuda]

jobs.json is ``[{"wav": "...", "out": "..."}, ...]``; each ``out`` receives JSON:
``{"model", "language", "segments": [{"start","end","text","avg_logprob",
"compression_ratio","no_speech_prob"}]}``. A failed song writes ``{"error": ...}``
instead, so one bad file never costs the batch.

Settings are the ones measured on Logan's hand-corrected lyrics (2026-09-24): a vocable
prompt (Whisper otherwise drops ooh/ah), no conditioning on previous text, and the
anti-loop guards — repetitive output and text hallucinated over long silences are the
failure mode on music, not misheard words.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROMPT = ("Song lyrics, sung. Transcribe every sung sound including ad-libs and vocables: "
          "ooh, oh, ah, yeah, mm, hmm, la la la, na na, whoa.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default=None,
                    help="default: float16 on cuda, int8 on cpu")
    ap.add_argument("--language", default="en")
    ap.add_argument("--cpu-threads", type=int, default=0)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    device = args.device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    compute = args.compute_type or ("float16" if device == "cuda" else "int8")
    model = WhisperModel(args.model, device=device, compute_type=compute,
                         cpu_threads=args.cpu_threads)
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    print(f"whisper_lyrics: {len(jobs)} job(s) on {device}/{compute}", flush=True)
    failures = 0
    for i, job in enumerate(jobs, 1):
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            segs, info = model.transcribe(
                job["wav"], language=args.language, beam_size=5, vad_filter=False,
                condition_on_previous_text=False, initial_prompt=PROMPT,
                compression_ratio_threshold=2.0, word_timestamps=True,
                hallucination_silence_threshold=2.0)
            payload = {
                "model": Path(args.model).name, "language": info.language,
                "duration": round(info.duration, 3),
                "segments": [{"start": round(s.start, 3), "end": round(s.end, 3),
                              "text": s.text.strip(), "avg_logprob": round(s.avg_logprob, 4),
                              "compression_ratio": round(s.compression_ratio, 3),
                              "no_speech_prob": round(s.no_speech_prob, 4)} for s in segs],
            }
        except Exception as exc:  # noqa: BLE001 - reported per song, batch continues
            failures += 1
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(jobs)}] {'ERR' if 'error' in payload else 'ok '} {job['wav']}",
              flush=True)
    return 1 if failures == len(jobs) and jobs else 0


if __name__ == "__main__":
    sys.exit(main())
