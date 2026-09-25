# Rig session 2 — runbook (drafted 2026-09-24 night)

Goal: grow the English singing corpus with curated channel songs, then start the long
`full_acoustic_v2` run. Rig facts (host, SSH, detaching) are in `rig_session_2026-09-20.md`
and the `training-rig-environment` memory; shipping is `transfer.md`.

What changed since session 1, all on branch **`rig-day`** (merge to `main` + push first):

- `full_v2` trains on **every** speaker (floor off: N singers / ~14.7 h instead of
  49 / N h), minus five tags that are male leads or duets (`exclude_singers`).
- `signalml harvest plan|inbox` — curated channels -> checklist -> hand
  download -> corpus layout + manifest, with the licence note in `license_note`.
- `signalml lyrics` — Whisper lyrics for songs without a `lyrics.txt`.
- `signalml ship plan --what unprocessed --prefix RAW/harvest` — raw new songs to the rig.
- `segmentation.ph_num_mode: vowel_onset` — **decide before SOME labels full_v2.**
  The variance model's "word" is the phones within one *note* (confirmed in their
  binarizer + inference code); S6b grouped by dictionary word, which only matches for
  one-syllable words. The acoustic model ignores ph_num, so switching costs the
  acoustic run nothing. Recommendation: set it in `dataset.full_v2.yaml` before the
  build. (overfit_v1's existing SOME labels were made with word grouping.)
- `signalml score to-ds song.json` — a score -> `.ds` for variance + acoustic inference,
  the way to hear the model sing something it never heard (grouping must match above).

## 1. Morning, work PC (Logan)

```powershell
$env:SIGNALML_DATA_ROOT = 'Y:\SignalAI\DATA_ROOT'   # NOT DATA_ROOT - unset = silently ./data
# one plan per channel (metadata only; existing plans are in DATA_ROOTcquire_lists):
python -m uv run signalml harvest plan --channel <url> --singer "<name>" --license "<licence note>"
# download each checklist (DATA_ROOT\acquire_lists\*.md) into DATA_ROOT\inbox, then:
python -m uv run signalml harvest inbox --inbox Y:\SignalAI\DATA_ROOT\inbox
```

`inbox` is safe to re-run as downloads trickle in; it prints what is still missing.
Automated download is blocked here (YouTube 403) — the download itself is manual.

Then land the code: `git checkout main && git merge rig-day && git push`.

## 2. Noon, rig: code + tools (GPU idle while this runs)

```powershell
cd G:\SIGNAL_AI\SignalML; git pull
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_rig.ps1 -DataRoot G:\SIGNAL_AI\DATA_ROOT
```

Tools the rig does not have yet — all under `G:\SIGNAL_AI`, nothing on C:.

- **Whisper** (S5a): venv + faster-whisper; copy `Y:\SignalAI\tools\whisper\large-v3`
  (2.9 GB) over rather than re-downloading. ctranslate2 needs cuBLAS 12 + cuDNN 9 DLLs:
  torch cu128 already ships them, so prepend
  `G:\SIGNAL_AI\SignalML\.venv\Lib\site-packages\torch\lib` to PATH for the lyrics run.
  If CUDA still fails, `--device cpu` works (~2x realtime — too slow for 100+ songs).
  Wire `configs/lyrics.local.yaml` (template in `configs/lyrics.yaml`).
- **MFA** (S5 align — the new songs must align on the rig): Miniforge silent install to
  `G:\SIGNAL_AI\miniforge3`, `conda create -n aligner -c conda-forge montreal-forced-aligner`,
  set `MFA_ROOT_DIR=G:\SIGNAL_AI\MFA` *before* `mfa model download acoustic english_mfa`
  and `mfa model download dictionary english_mfa`. Wire `configs/align.local.yaml` with
  the absolute conda.exe path (see the work PC's copy).
- **SOME** (variance labels for full_v2): copy `Y:\SignalAI\tools\SOME` + the
  `0119_continuous256_5spk` checkpoint; its venv needs rebuilding on the rig.

## 3. Ship (work PC serves, rig pulls)

```powershell
# work PC
signalml ship plan --what rebuildable --recipe configs\dataset.full_v2.yaml --name full_v2
signalml ship plan --what unprocessed --prefix RAW/harvest --name harvest --no-code
signalml ship serve --name full_v2      # then --name harvest; serve prints the pull line
# rig
signalml ship pull <printed url> --data-root G:\SIGNAL_AI\DATA_ROOT --no-code
signalml ship verify --data-root G:\SIGNAL_AI\DATA_ROOT --name full_v2
```

**Always pass `--prefix` with `unprocessed`:** without it every unaligned record ships,
including 1,619 VocalSet clips.

## 4. Rig: front half for the new songs

```powershell
signalml separate   # GPU, seconds per song
signalml clean
signalml lyrics     # Whisper; hand-written lyrics are never touched
signalml align
signalml manifest report
```

**Machine lyrics are not trustworthy for timing yet** (measured overnight on 15 songs
with hand-corrected lyrics): Whisper finds ~90% of the words (median word error ~19%),
but MFA alignments made from its lyrics agree with the hand-lyrics alignments on only
**~58% of sung time** (control: re-aligning the hand lyrics reproduces 99.8%, so the
metric is sound). `align_score` cannot see this - it reads ~1.0 either way. Part of the
gap is likely incomplete old references (a chorus written once), unquantified. Tried and
rejected: anti-loop guards (no gain), aligning per Whisper segment (21% - Whisper's
timestamps drift on singing).

So: the dataset recipe has `filters.machine_lyrics` (default true). Options, best first:
1. Skim-correct the Whisper drafts (`songs/<id>/lyrics/lyrics.txt`) for the harvested
   songs - correcting a ~90%-right draft is far faster than typing - then re-align.
   A corrected file is still tagged asr; fine, or clear `lyrics_source` by hand.
2. Train full_v2 with `machine_lyrics: false` (hand-lyrics songs only) and treat the
   harvest as the next run's data.
3. Admit them unreviewed - only as a deliberate A/B against option 2.

## 5. Build + train

```powershell
signalml dataset build --recipe configs\dataset.full_v2.yaml
signalml train acoustic --dataset full_acoustic_v2 --dry-run
signalml train acoustic --dataset full_acoustic_v2    # detach: Win32_Process Create + .cmd
signalml train status --exp full_acoustic_v2
```

Check speed in the first 10 minutes: `max_batch_frames: 50000` was never measured on the
4090 (overfit ran ~5.9 steps/s at 20000). 160k steps must fit in the weekend. Permanent
checkpoints every 10k keep the best step recoverable.

While acoustic trains (CPU work, GPU busy): SOME `batch_infer.py` over each
`datasets/full_acoustic_v2/<singer>-en/` -> `signalml dataset variance-config --dataset
full_acoustic_v2 --recipe configs\dataset.full_v2.yaml` -> variance training after the
acoustic run (or overfit_v1 variance first, as a quick end-to-end proof — its path was
verified on CPU 2026-09-24: binarize + 30 steps clean).

## 6. Hear it sing a new score

```powershell
signalml score from-midi melody.mid --lyrics lyrics.txt --out song.json  # or hand-write
signalml score to-ds song.json                                         # -> song.ds
cd third_party\DiffSinger
python scripts\infer.py variance ..\..\song.ds --exp full_variance_v2 --predict dur --predict pitch
python scripts\infer.py acoustic <variance output .ds> --exp full_acoustic_v2 --spk <singer>
```

(Check `infer.py --help` for the exact flags of the vendored version; the vocoder is
the NC community one — dev preview only.)

## 7. Before leaving the rig

Ship results home (the rig is borrowed; the work PC holds the canonical DATA_ROOT):
the new songs' `songs/<id>/` (stems, clean, lyrics, align) and the best checkpoints.
`ship` works in either direction — plan + serve on the rig, pull on the work PC.
