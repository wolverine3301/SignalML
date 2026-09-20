# Vendored trainer: openvpi/DiffSinger — D2 spike result (2026-07-12)

**Decision: adopt-their-world.** Vendored as a git submodule at `third_party/DiffSinger`,
tag **v2.5.1** (commit `323a569`, Apache-2.0, released 2026-01). Our manifest + score
JSON remain the system of record; `signalml dataset build` (S6b) emits *their* raw
dataset format and `signalml train` wraps *their* scripts in *their own venv*. Local
patches: **zero** (policy: if a patch feels needed, first ask whether S6b should emit
different data instead).

## Why v2.5.1 / why adopt-their-world

- v2.5.x brings universal multilingual support (per-dataset `language`, per-language
  `dictionaries`, `merged_phoneme_groups`, `use_lang_id`) — this is the Gaelic wave-2
  mechanism handed to us for free, and the "universal phoneme set" ambition maps onto
  their language-prefixed phoneme system.
- v2.4+ has rectified flow (ARCHITECTURE §3.2's expectation); v2.3+ uses mel_base 'e'.
- Owning the boundary (our own binarizer to their tensor contract) buys nothing until
  their conventions actually distort an upstream stage — revisit only on that evidence.

## The integration contract (what S6b must emit)

Per singer/language source, a dataset folder:

```
datasets/<name>/<speaker>-<lang>/
  ├── wavs/*.wav            # mono, clip-length segments (use S4 silence map to cut)
  └── transcriptions.csv    # name, ph_seq, ph_dur           (acoustic)
                            # + ph_num, note_seq, note_dur   (variance: dur + pitch)
```

Wired into a generated trainer config:

```yaml
datasets:                    # one entry per (singer, language) folder
  - raw_data_dir: ...
    speaker: <singer>        # manifest meta.singer — the voice-bank identity
    spk_id: <int>            # stable map kept in dataset_card.md
    language: en
dictionaries: {en: <generated>}   # from score/phoneset.py — see D3
num_spk / num_lang / use_spk_id: true / use_lang_id (wave-2)
```

Key spike finding: **their binarizer computes training mels from the wavs itself** —
S6b hands over *audio + labels*, not spectra. Our features NPZs stay analysis/dashboard
artifacts; the mel contract (D5) is owned end-to-end by the vendored stack and matches
our `prod` profile **exactly**: 44100 Hz / fft+win 2048 / hop 512 / 128 mels /
fmax 16000 / `mel_base: 'e'` (their `fmin: 40` is theirs to own too). Nothing in our
P0–P6 output needs to change.

## Environment (their pins conflict with ours — isolate)

`requirements.txt` pins `librosa<0.10`, `numpy<2`, `lightning~=2.3` — incompatible
with the pipeline venv (librosa ≥0.10, numpy ≥2). Same pattern as MFA-in-conda:

```powershell
# one-time, inside third_party/DiffSinger (done on the work PC 2026-07-12):
python -m uv venv .venv --python 3.10   # 3.12 fails: pyworld==0.3.4 has no cp312 wheel
python -m uv pip install --python .venv\Scripts\python.exe "torch==2.11.0+cu128" torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Install notes: uv's *managed* CPython download prints a "Missing expected target
directory for Python minor version link" error — it's cosmetic; the interpreter lands
and `uv venv --python 3.10` finds it. Avoid the Windows-Store Python 3.10 stub (uv
can't use it). Verified: `torch 2.11.0+cu128`, `cuda.is_available() == True` (2080S),
`scripts/train.py --help` runs.

`signalml train` (P7.3) invokes `third_party/DiffSinger/.venv/Scripts/python.exe
scripts/{binarize,train,infer}.py` — the interpreter path is a config value
(`trainer_python` in `configs/train/*.yaml`), injectable like `mfa_command`.

## Consequences for the open decision points

- **D1 (note annotation):** their variance pipeline expects `note_seq`/`note_dur` in
  transcriptions.csv; the ecosystem's own tools are MakeDiffSinger
  `variance-temp-solution` + `midi-recognition` and openvpi **SOME**, with SlurCutter
  for manual fixes. D1 = evaluate SOME on 5–10 of our separated vocals; output lands
  as an S5b stage writing `note_seq` next to `phones.json`.
- **D3 (phone mapping):** the trainer dictionary is a tab-separated
  `syllable<TAB>ph ph ...` text file — *generated output* from `score/phoneset.py`,
  as planned. ⚠ Their docs say phoneme names "ASCII preferred", separators
  (`/ - + @ # & | < >`) forbidden: IPA symbols (ʃ, ɫ̩) must be smoke-tested in the
  overfit run; fallback is a deterministic ASCII transliteration table in
  phoneset.py (X-SAMPA-style), which changes nothing upstream (phones.json stays IPA).
- **D5 (mel contract):** resolved by adoption (above).
- **D6 (vocoder):** own-vocoder training lives in a separate repo —
  **openvpi/SingingVocoders** — vendored at P7.4's start; their default acoustic
  config points at the community `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`
  checkpoint (CC BY-NC — dev preview only, per Q4; record in run configs).
- **D7 (voice bank):** conditioning hook is `use_spk_id: true` + the `spk_id`
  embedding table; profile pins checkpoint hash as designed. Embedding extraction
  reads the table out of their checkpoint format.

## Checklist to first sound (overfit run, P7.5)

1. Trainer venv created (recipe above) — GPU smoke: their `scripts/train.py --help`.
   `scripts/bootstrap_rig.ps1` does this, including the cu128 torch swap *inside*
   this venv (their `requirements.txt` leaves torch unpinned on purpose, so without
   the swap the trainer gets a CPU wheel and dies at the first kernel launch).
2. S6b: manifest query → clip cutting (S4 silence map) → transcriptions.csv
   (acoustic columns first) → generated config + dictionary.
3. Overfit set: 2–3 songs, hand/SOME note labels for the variance step (or
   acoustic-only first: resynthesize with ground-truth durations — hears sound
   before D1 is solved). `configs/dataset.overfit.yaml` is that set, at the **prod**
   profile (MIGRATION_PLAN P7.5, amended 2026-09-20).
4. Community vocoder checkpoint downloaded into their `checkpoints/` (NC, dev only).

## How a run is driven (P7.3, 2026-09-20)

`signalml train acoustic --dataset <name>` (`signalml/train/runner.py`, wiring in
`configs/train.yaml`) runs steps 2–4's output through their `binarize.py` then
`train.py`, with `cwd = third_party/DiffSinger` — their `base_config` and
`vocoder_ckpt` paths are relative to it — and `PYTHONUTF8=1`, without which Windows'
cp1252 mangles IPA phoneme names somewhere between the dictionary and the phone set.
Zero patches still holds: we add a preflight and a run record around their scripts,
never inside them.

Consequence of adopting their world: **checkpoints live in
`third_party/DiffSinger/checkpoints/<exp_name>/`**, because `work_dir` is derived from
`--exp_name` relative to the process CWD and moving it would mean patching
`utils/hparams.py`. Our run record under `<DATA_ROOT>/runs/<exp>/<timestamp>/` points
at that directory and carries the provenance (git hash, dataset-card hash, config +
dictionary snapshot, torch/CUDA probe). Watch free space on the repo's drive —
`signalml train` warns below `min_free_gb`.
