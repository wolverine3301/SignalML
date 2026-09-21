# Rig session 1 — 2026-09-20 (first training run)

What happened the first time this pipeline met a training machine, what is parked
where, and what the next session should do differently. Environment details live in
`docs/notes/transfer.md` (shipping) and `vendor_diffsinger.md` (the trainer).

## The machine

**Not a 5090.** `DESKTOP-P21F7SI` at `10.0.0.64` — a borrowed PC (Cade's), Windows 11
Pro, **RTX 4090 24 GB** (driver 616.92, sm_89), Ryzen 9 7900X3D, 63 GB RAM. Driven
entirely over SSH from the work PC; its default shell is `cmd.exe`, so non-trivial
commands go through `powershell -NoProfile -Command`, or a script piped over stdin.

Because it is borrowed, everything of ours lives under `G:\SIGNAL_AI` — repo,
`DATA_ROOT`, and the uv wheel/interpreter caches (`UV_CACHE_DIR`,
`UV_PYTHON_INSTALL_DIR`), so nothing accumulates on his C: drive. Installed by us:
Python 3.12.10 (only 3.9 and the Store stub were present), uv, both venvs on
torch 2.11.0+cu128.

**Detaching a long run:** `Start-Process` dies when the SSH session closes (two empty
log files and no process). Use `Invoke-CimMethod Win32_Process Create` pointing at a
`.cmd` file that does its own `> log 2>&1`; a redirect written inside a PowerShell
`-Command` string is eaten by PowerShell before the process ever sees it.

## The run

`overfit_v1` — 3 singers, N clips, 1.N h, prod profile. Rebuilt on the rig from a
`rebuildable` shipment (0.77 GB, 49 songs, hash-verified) and it produced *exactly* the
clip count built on the work PC, which is the dataset-purity claim actually holding
across two machines.

20,000 steps in ~64 minutes (~5.9 steps/s average at 20000 frames / batch 32).
`Trainer.fit stopped: max_steps=20000 reached`, exit 0.

| step | val total | val mel | train mel |
|---|---|---|---|
| 0 | 1.195 | 1.118 | 0.969 |
| 500 | 0.337 | 0.282 | — |
| **3000** | **0.170** | **0.122** | 0.254 |
| 20000 | 0.218 | 0.175 | 0.063 |

Validation bottomed around step 3000 and rose while training loss kept falling: the
model memorized three singers, which is what an overfit proof is *for* — but it means
**the last checkpoint is not the best one**, and `num_ckpt_keep: 3` had already deleted
step 3000 by the end. Next run should keep best-validation checkpoints, not the last N.

Artifacts: `model_ckpt_steps_20000.ckpt` (847 MB) on the rig at
`G:\SIGNAL_AI\SignalML\third_party\DiffSinger\checkpoints\overfit_v1\`, copied back to
`DATA_ROOT\runs\overfit_v1\rig_2026-09-20\` with its `config.yaml`. Validation audio
(`gt`/`aux`/`diff` per clip) is in the TensorBoard event files and extractable without
TensorBoard — `EventAccumulator(...).Audio(tag)[-1].encoded_audio_string` is a WAV.

Everything rendered here went through the **CC BY-NC community vocoder**: dev preview
only, nothing from this run ships (Q4).

## Bugs this session found

1. `scripts/bootstrap_rig.ps1` **never parsed** under PowerShell 5.1 — an em dash in a
   BOM-less `.ps1` decodes as cp1252, where 0x94 is a closing smart quote that ends the
   string. Fixed, with `tests/test_scripts.py` guarding ASCII + CRLF.
2. `doctor` failed a working 4090: it matched `sm_89` against the wheel's arch list as a
   string, but CUDA cubins are binary-compatible upward within a generation (the sm_86
   kernels run fine — verified with a real matmul before changing the check). It had
   made `bootstrap_rig.ps1` exit non-zero on a correctly installed rig.
3. Batch sizing was ~4x too large: `max_batch_frames` is *frames*, and 80000 of them at
   prod is ~15 minutes of audio per step (0.26 steps/s, 24.0 of 24.5 GB used, one epoch
   per 8 steps). Recipes now carry measured numbers.

## Next session

- Keep the best-val checkpoint (see above) before another long run.
- Ship the rest of the corpus (`corpus`, 6.42 GB planned and hash-cached) and run
  `full_acoustic_v2` — N singers, N h, the first real multi-singer run.
- P7.4 still has step zero outstanding: `openvpi/SingingVocoders` is not vendored, and
  until it is, every render is NC-licensed.
- Studio stems and VocalSet are still not cleaned/onboarded — both are vocoder data.
