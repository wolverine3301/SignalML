# MedleyDB onboarding (2026-08-27)

MedleyDB is the first *external, labelled* corpus in the pipeline. It matters here for
one reason: it annotates **`female singer` / `male singer` per stem**, so gender and
singer identity enter the manifest as data rather than as a blanket CLI flag — which is
what turns `dataset build`'s `filters.gender: F` from a hopeful setting into an
auditable guard.

## What is on disk

- Audio: `DATA_ROOT/MedleyDB_V2.tar/MedleyDB_V2/V2/` — **74 V2 tracks**
  (`<Track>/<Track>_STEMS/*.wav` processed submixes, `<Track>_RAW/*.wav` untouched
  mic/DI feeds, plus `<Track>_MIX.wav`). Stems are 44.1 kHz stereo. MedleyDB V1 is not
  present.
- Metadata: `DATA_ROOT/medleydb/Metadata/*_METADATA.yaml` — **all 330** tracks
  (harmless: unmatched entries report "audio not on this machine"). Not shipped in the
  audio tarball; fetched from github.com/marl/medleydb:

  ```bash
  git clone --filter=blob:none --sparse --depth 1 https://github.com/marl/medleydb.git
  cd medleydb && git sparse-checkout set medleydb/data/Metadata
  ```

  The repo's `Annotations/` (melody f0, pitch, activation confidence, source id) is
  **not** onboarded — S6 computes its own F0 and we have no use for the rest yet. Pull
  it the same way if a vocal-activity gate turns out to beat our silence map.

## What the corpus holds for us (female singers, V2 subset)

12 tracks / **20 female-singer stems / 1.70 h** of full-length stem audio across
**11 artist-level identities**; actual *singing* time is well under that (stems run the
length of the song). Four tracks contribute multiple stems (lead + doubles/harmonies of
the same singer): CatMartino ×4, Torres ×4, Cayetana ×2, FilthyBird ×2. Four are flagged
`has_bleed: yes` (SongYiJeon, both TleilaxEnsemble, Verdi).

## Import

```bash
signalml manifest import-medleydb --overrides configs/medleydb_overrides.yaml --dry-run
```

Defaults to `--instruments "female singer"`. Each selected stem becomes a manifest
record with `gender` from the instrument label, `source_quality=studio`,
`processing=produced` (stem level = the engineer's submix; `--level raw` imports the
untouched takes as `dry`), `license_note` = CC BY-NC-SA 4.0, and `separated=true` — a
MedleyDB stem *is* the isolated source, so Demucs is skipped and the stem is copied to
`songs/<id>/stems/vocals.wav`. Provenance (track, stem key, instrument, component,
bleed) lands in `analysis.json`. Idempotent by checksum; `--dry-run` lists the exact
tags that would be written.

Useful narrowing: `--melody-only` (lead vocal only — drops harmony/double stems),
`--exclude-bleed`, `--allow-mixed` (stems whose `instrument` is a *list*, e.g.
`[male singer, vocalists]`; refused by default because the vocal is not isolated).

## The two gaps, and the curation file

- **No lyrics.** MedleyDB ships no transcripts, so records import with
  `has_lyrics=false` and `align` refuses them. They are immediately useful as vocoder
  training material and as the ECAPA reference set for the voice-bank novelty guard;
  they cannot enter an *acoustic* dataset until transcripts exist. The wiring is
  already there: drop `<Track>_STEM_NN.txt` next to the source stem WAV (ASR draft +
  human fix) and re-import/retag.
- **No singer identity.** The finest label is `artist` — a *band*, and for classical
  entries the *composer*. `configs/medleydb_overrides.yaml` corrects those per track or
  per stem (`singer`, `gender`, `language`, `processing`, `song`, `exclude`); it already
  renames the two uncredited classical soloists and tags `Verdi_IlTrovatore` as
  Italian so it cannot leak into an `en` dataset.

## Licence

MedleyDB is **CC BY-NC-SA 4.0 — non-commercial**, recorded on every record and rolled
up in the dataset card's licence section. Same discipline as the community NSF-HiFiGAN
checkpoint (Q4): fine for research and for proving the pipeline, but a model trained on
it inherits the non-commercial claim, and ShareAlike is a further complication if
anything derived is distributed. Keep MedleyDB-derived checkpoints out of anything that
ships until that is resolved (own-recorded or permissively licensed data for the
production run).
