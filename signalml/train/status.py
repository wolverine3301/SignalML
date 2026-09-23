"""``signalml train status`` — what a run is doing, and which checkpoint to keep.

A training run's only honest outputs are its TensorBoard event files and the
checkpoints beside them. Reading them by hand means an SSH session, a Python
one-liner and knowing which tag the trainer writes losses under; the 2026-09-20 rig
session did exactly that from a phone, from a scratch script that died with it.

Two questions this answers that TensorBoard-in-a-browser does not answer well over a
LAN link:

- *where is it now* — latest step, latest losses, checkpoints on disk;
- *which checkpoint is the good one* — the step where validation actually bottomed,
  and whether that checkpoint still exists. ``num_ckpt_keep`` is a rolling window, so
  a run that peaks early deletes its own best result and says nothing about it.

The event files are read by the **trainer's** interpreter (it has tensorboard; our
pipeline venv does not), the same way ``runner`` shells out to their scripts. The
helper below is handed over on stdin and answers in JSON, so nothing here depends on
quoting surviving a trip through cmd.exe.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from .runner import Runner, TrainConfig, load_train_config

# Lower is better for every one of these; the first present wins as "the" metric.
VAL_METRICS = ("validation/total_loss", "validation/mel_loss")

# Runs in the trainer venv. Prints one JSON object. Keep it dependency-free beyond
# tensorboard, and keep it quiet on stdout — the JSON is the protocol.
_PROBE = r'''
import glob, json, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

exp_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ["SIGNALML_EXP_DIR"]
audio_out = os.environ.get("SIGNALML_AUDIO_OUT") or ""
out = {"exp_dir": exp_dir, "event_file": None, "scalars": {}, "audio": [],
       "checkpoints": [], "error": None}
try:
    ckpts = []
    if os.path.isdir(exp_dir):
        for name in os.listdir(exp_dir):
            if name.endswith(".ckpt"):
                path = os.path.join(exp_dir, name)
                step = "".join(c for c in name if c.isdigit())
                ckpts.append({"name": name, "step": int(step) if step else None,
                              "bytes": os.path.getsize(path)})
    out["checkpoints"] = sorted(ckpts, key=lambda c: (c["step"] is None, c["step"]))

    evs = sorted(glob.glob(os.path.join(exp_dir, "**", "events.out.tfevents.*"),
                           recursive=True), key=os.path.getmtime)
    if not evs:
        out["error"] = "no event files yet"
        print(json.dumps(out)); raise SystemExit
    out["event_file"] = evs[-1]
    out["event_files"] = len(evs)
    out["event_age_sec"] = round(__import__("time").time() - os.path.getmtime(evs[-1]))

    # EVERY event file, not just the newest: each resumed leg opens a new one, so
    # reading the last file alone reports step 0 on a run that is mid-flight.
    merged = {}
    audio_latest = {}
    for ev in evs:
        ea = EventAccumulator(ev, size_guidance={"scalars": 0, "audio": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            per_step = merged.setdefault(tag, {})
            for s in ea.Scalars(tag):
                # a re-run of the same steps overwrites: last writer wins
                per_step[int(s.step)] = (float(s.value), float(s.wall_time))
        if audio_out:
            for tag in ea.Tags().get("audio", []):
                items = ea.Audio(tag)
                if items and (tag not in audio_latest
                              or items[-1].step >= audio_latest[tag].step):
                    audio_latest[tag] = items[-1]
    for tag, per_step in merged.items():
        out["scalars"][tag] = [[step, per_step[step][0], per_step[step][1]]
                               for step in sorted(per_step)]
    if audio_out:
        os.makedirs(audio_out, exist_ok=True)
        for tag, last in audio_latest.items():
            name = tag.replace("/", "_").replace("\\", "_") + "_step%d.wav" % last.step
            path = os.path.join(audio_out, name)
            with open(path, "wb") as fh:
                fh.write(last.encoded_audio_string)
            out["audio"].append({"tag": tag, "step": int(last.step), "path": path})
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
'''


def probe_command(cfg: TrainConfig, exp_dir: Path) -> list[str]:
    return [str(cfg.resolved_trainer_python()), "-", str(exp_dir)]


def collect(
    exp_name: str,
    *,
    cfg: TrainConfig | None = None,
    audio_out: str | Path | None = None,
    runner: Runner | None = None,
) -> dict:
    """Read one experiment's event files. Returns the probe's JSON plus a summary."""
    cfg = cfg or load_train_config()
    runner = runner or subprocess.run
    exp_dir = cfg.resolved_trainer_dir() / "checkpoints" / exp_name
    env = {"SIGNALML_AUDIO_OUT": str(audio_out)} if audio_out else {}

    probe = _run(runner, probe_command(cfg, exp_dir), env, cfg)
    return summarize(probe, exp_name=exp_name, exp_dir=exp_dir)


def _run(runner: Callable, cmd: list[str], env_overrides: dict, cfg: TrainConfig) -> dict:
    env = {**os.environ, **cfg.env, **env_overrides}
    proc = runner(cmd, input=_PROBE, capture_output=True, text=True, env=env,
                  encoding="utf-8", errors="replace")
    stdout = (getattr(proc, "stdout", "") or "").strip()
    if getattr(proc, "returncode", 0) != 0 or not stdout:
        return {"error": f"probe failed (rc={getattr(proc, 'returncode', '?')}): "
                         f"{(getattr(proc, 'stderr', '') or stdout or '').strip()[-400:]}"}
    try:
        return json.loads(stdout.splitlines()[-1])
    except ValueError:
        return {"error": f"probe emitted no JSON: {stdout[-400:]}"}


def summarize(probe: dict, *, exp_name: str, exp_dir: Path | None = None) -> dict:
    """Turn raw scalar series into the two facts a running job is asked for."""
    summary: dict = {"exp_name": exp_name, "exp_dir": str(exp_dir or probe.get("exp_dir")),
                     "error": probe.get("error"), "event_file": probe.get("event_file"),
                     "event_age_sec": probe.get("event_age_sec"),
                     "checkpoints": probe.get("checkpoints", []),
                     "event_files": probe.get("event_files"),
                     "audio": probe.get("audio", []), "latest": {}, "best": None,
                     "steps_per_sec": None, "step": 0}
    scalars: dict[str, list] = probe.get("scalars") or {}
    for tag, pts in scalars.items():
        if pts:
            summary["latest"][tag] = {"step": pts[-1][0], "value": pts[-1][1]}
            summary["step"] = max(summary["step"], pts[-1][0])

    train_tags = [t for t in scalars if t.startswith("training/")]
    if train_tags:
        pts = scalars[sorted(train_tags)[0]]
        # the last two points only: a merged series spans the gap between legs of a
        # resumed run, and averaging across a coffee break is not a step rate
        if len(pts) > 1 and pts[-1][2] > pts[-2][2]:
            summary["steps_per_sec"] = round(
                (pts[-1][0] - pts[-2][0]) / (pts[-1][2] - pts[-2][2]), 2)

    for metric in VAL_METRICS:
        pts = scalars.get(metric)
        if pts:
            step, value, _ = min(pts, key=lambda p: p[1])
            kept = any(c.get("step") == step for c in summary["checkpoints"])
            summary["best"] = {"metric": metric, "step": step, "value": value,
                               "checkpoint_present": kept,
                               "latest_value": pts[-1][1]}
            break
    return summary


def format_status(s: dict) -> str:
    if s.get("error") and not s.get("step"):
        return f"{s['exp_name']}: {s['error']}"
    lines = [f"{s['exp_name']}  step {s['step']}"
             + (f"  ({s['steps_per_sec']} steps/s)" if s.get("steps_per_sec") else "")]
    if s.get("event_age_sec") is not None:
        lines[0] += f"  last write {s['event_age_sec']}s ago"
    for tag in sorted(s.get("latest", {})):
        got = s["latest"][tag]
        lines.append(f"  {tag:28s} step {got['step']:>7}  {got['value']:.4f}")
    best = s.get("best")
    if best:
        lines.append("")
        lines.append(f"  best {best['metric']} at step {best['step']}: "
                     f"{best['value']:.4f}  (latest {best['latest_value']:.4f})")
        # Only meaningful once something has been saved: before the first checkpoint
        # interval every step is "missing", which is not the failure we mean.
        if not best["checkpoint_present"] and s.get("checkpoints"):
            lines.append("  WARNING: no checkpoint at that step — num_ckpt_keep is a "
                         "rolling window. Set trainer_opts.permanent_ckpt_start / "
                         "_interval in the recipe and rebuild before the next run.")
    ckpts = s.get("checkpoints") or []
    if ckpts:
        lines.append("")
        lines.append(f"  {len(ckpts)} checkpoint(s): "
                     + ", ".join(str(c["step"]) for c in ckpts if c["step"] is not None))
        lines.append(f"  {sum(c['bytes'] for c in ckpts) / 2 ** 30:.1f} GB in {s['exp_dir']}")
    for item in s.get("audio", []):
        lines.append(f"  wrote {item['path']}")
    if s.get("error"):
        lines.append(f"  note: {s['error']}")
    return "\n".join(lines)
