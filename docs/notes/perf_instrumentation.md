# Pipeline performance instrumentation

Lightweight timing across every stage, so "where does the time go" is answered by
recorded data instead of a stopwatch and a guess. Deliberately small: wall-clock
phase timers, no profiler, no sampling, no new dependency.

Motivating cases from the 2026-09 corpus run:

- The July `align` run died mid-step and left **nothing** — no partial timings, no
  indication of which song it was on. A run record written incrementally would have
  said exactly where it stopped.
- The 2026-09-08 align took ~4 h for 21.4 h of audio. Nobody could say whether that
  was reasonable until it was over, because there was no throughput baseline.
- 57% of utterances hit MFA's retry beam, which is the single biggest cost driver in
  that stage, and it was only visible by grepping MFA's own debug logs.

## Two levels, both on existing conventions

**Per song, per stage → `songs/<id>/analysis.json`.** That file already carries one
section per stage via `update_analysis()`. Timings become a `timing` key inside the
section that produced them — no new files, already atomic, and it travels with the
artifact it describes.

```json
"clean": {
  "profile": "prod",
  "resampled": true,
  "timing": { "total": 1.82, "read": 0.31, "resample": 0.44,
              "loudness": 0.68, "write": 0.39 }
}
```

**Per run, aggregate → `DATA_ROOT/runs/<stage>_<utc-timestamp>.json`.** One record per
stage invocation: machine context, resolved config, counts, wall time, and every
song's total. Written **incrementally** (appended as songs complete, flushed every N)
so a crashed or killed run still leaves a usable trail — the July failure mode.

```json
{
  "stage": "align", "started": "2026-09-07T22:32:28Z", "finished": null,
  "machine": { "host": "...", "cpu_count": 24, "gpu": "RTX 2080 SUPER",
               "torch": "2.11.0+cu128", "data_root_volume": "network" },
  "config": { "profile": "prod", "beam": 200, "retry_beam": 1000, "num_jobs": 12 },
  "counts": { "attempted": 300, "ok": 281, "failed": 19 },
  "phases": { "stage_corpus": 88.4, "mfa_subprocess": 13910.2, "import": 41.7 },
  "songs": [ { "id": "sng_0115", "sec": 44.1, "audio_sec": 202.8, "ok": true } ]
}
```

## Mechanism — `signalml/perf.py`

One module, one class. A nested context manager that accumulates repeated phases, so
a per-clip write inside a loop sums into a single `clip_write` number.

```python
with Timer("dataset") as t:
    with t.phase("select"):
        selected = _select(...)
    for rec in selected:
        with t.phase("segment"):
            clips, _ = segment_phones(...)
        with t.phase("audio_read"):
            audio, sr = sf.read(...)
        for clip in clips:
            with t.phase("clip_write"):
                sf.write(...)
    run.record_song(rec.id, t.split(), audio_sec=rec.file.duration_sec)
```

Non-negotiables, so this never becomes the thing that breaks a stage:

- **Off switch**: `SIGNALML_PERF=0` makes every timer a no-op context manager.
- **Never raises**: the recorder swallows its own errors — a telemetry bug must not
  fail an 8-hour job.
- **No signature changes**: stages keep returning their existing summary objects, so
  the contract tests in `tests/` are untouched.
- `time.perf_counter()` only. Overhead is ~1 µs per phase; at 6,784 clips that is
  under 10 ms total.

## Phases worth naming

| stage | phases |
|---|---|
| S1 acquire | `download`, `probe`, `checksum` |
| S3 separate | `model_load`, `inference`, `stem_write` |
| S4 clean | `read`, `resample`, `loudness`, `silence`, `write` |
| S5 align | `stage_corpus`, `mfa_subprocess`, `textgrid_parse`, `manifest_import` |
| S6 features | `f0`, `mel`, `bpm_key`, `npz_write` |
| S6b dataset | `select`, `segment`, `audio_read`, `clip_write`, `csv`, `config_card` |

Machine context is captured once per run, because these numbers are only comparable
across the laptop, the work PC, and the 5090 rig if you know which one produced them —
and because `DATA_ROOT` being a network volume is itself a major variable (`Y:`
benchmarks at 351 MB/s read, 154 MB/s write, 5 ms per small file).

## Reporting — `signalml perf report`

Reads `runs/*.json` and prints, per stage:

- **Realtime factor** — audio-seconds processed per wall-second. The right normalizer
  for an audio pipeline: comparable across corpora, machines, and run sizes. Align on
  2026-09-08 was ~5.4× realtime across 12 workers; clean was ~180×.
- Phase breakdown as a percentage of stage wall time — the "where do I optimize" view.
- Per-song median / p95, plus the **10 slowest songs**. That is how pathological items
  surface: MFA's retry cases would have shown up immediately as a bimodal distribution
  rather than needing a grep through kaldi debug logs.
- Run-over-run comparison for the same stage, so a config change (beam width, Demucs
  shifts, `num_jobs`) can be judged rather than guessed.

## Build order

1. `perf.py` + wire S6b dataset — smallest stage, fastest feedback, and it is the one
   being run repeatedly right now while recipes get tuned.
2. S4 clean and S6 features — cheap, and both get re-run with `--force` often.
3. S5 align — highest value (it dominates wall time) but needs the incremental-write
   path working first, since that is the run most likely to die.
4. S3 separate — matters when the Spleeter→Demucs re-separation question gets tested.

`signalml perf report` last: the JSON is useful with `jq` long before a formatter
exists.
