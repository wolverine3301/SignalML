# VocalSet onboarding + corpus scoping (2026-08-27)

Two things landed together: the VocalSet adapter, and the `corpus` tag that lets a
training run be scoped to one corpus, a combination, or everything-but.

## Corpus scoping (the general mechanism)

`meta.corpus` is a normalised slug on every manifest record (`own`, `medleydb`,
`vocalset`, ...). Ingest adapters set it; `META.txt` can carry `CORPUS:`; and
`signalml manifest set-corpus <name>` backfills records that predate the field (it only
touches untagged records unless you pass `--force`).

Recipes then scope a build:

```yaml
filters:
  corpora: []                   # empty = every corpus; e.g. [vocalset, own]
  exclude_corpora: []           # e.g. [medleydb] to keep a corpus out of this run
```

`corpora` whitelists, `exclude_corpora` blacklists, and both are reported per skipped
record. The dataset card gains a **Corpora** roll-up next to the licence roll-up, so a
finished dataset states which corpora it drew from. The obvious use beyond experiment
scoping: `exclude_corpora: [medleydb]` (and later GTSinger/MoisesDB) produces a run
whose entire lineage is permissively licensed.

## VocalSet

10.1 h, 20 professional singers (**9 female / 11 male**), 17 vocal techniques across
four contexts (scales, arpeggios, long tones, excerpts), each sung on the five vowels —
~3.5k short a cappella files. Downloaded from
[Zenodo record 1193957](https://zenodo.org/records/1193957) (2.1 GB zip) into
`DATA_ROOT/vocalset/`.

**Licence: CC BY 4.0** — attribution only, commercial use permitted. It is the only
permissively licensed corpus in the collection, which is what makes it strategically
useful: VocalSet plus own-recorded material is the only lineage a shippable checkpoint
can currently claim.

### Import

```bash
signalml manifest import-vocalset --dry-run     # census first
signalml manifest import-vocalset               # female singers by default
```

Every label comes from the filename, which VocalSet defines as the unique identifier
(`f2_arpeggios_f_slow_forte_e.wav` → singer `f2`, female, context `arpeggios`, technique
`f_slow_forte`, vowel `e`). Gender is therefore in the corpus by construction — no flag,
no guessing. Directory layout is only a fallback for the context, so `FULL/` and
`train`/`test` splits both parse.

Records land `separated=true` (a cappella — nothing to separate), `source_quality=studio`,
`processing=dry` (no production on the voice at all), `corpus=vocalset`, singer namespaced
as `vocalset-f2` so ids cannot collide with another corpus. Technique/context/vowel go to
`analysis.json` and `quality.notes`. Defaults to `--genders F`, since importing the male
half that a female-only recipe refuses anyway just burns disk; `--contexts` and
`--techniques` narrow further. Idempotent by checksum.

### What actually landed (2026-08-27)

`FULL/` extracted to `DATA_ROOT/vocalset/` (2.7 GB, 3613 WAVs, 44.1 kHz mono).
**1619 female records / 3.9 h** across 9 singers; 3 files were byte-identical
duplicates and deduped on checksum. The manifest went 428 -> 2047 records.

Filenames are hand-made and messy — the parser normalises them: abbreviated and
typo'd contexts (`arps`, `arepggios`), a misspelt `f_sow_forte`, stray leading spaces
and underscores, and `(1)` duplicate markers. That takes the raw label set from 49
strings down to 26 real techniques. The `excerpts` section is the only material sung on
**words** — three short pieces (`caro` = Caro mio ben, Italian; `row` = Row row row your
boat, English; `dona` = Dona nobis pacem, Latin) — so the excerpt name and its text are
recorded in `analysis.json`; those 139 files are the ones that could take lyrics
sidecars later and become alignable.

### Deliberate: `language` is null

The material is sung on isolated vowels, so there are no lyrics and no language. Records
import with `language=None`, which *keeps them out of `en` acoustic datasets* — training
an acoustic model on vowel-only "words" would be actively harmful. They are fully
available to the vocoder and to voice-bank/ECAPA work, which is the point of having them.

The `spoken` technique in the excerpts section imports as `domain=spoken`, lining up with
the wave-3 speech plan (D10).

### Fit, honestly

Nine female singers of clean dry studio audio is a better *timbre* spread than MedleyDB's
eleven band-level identities, and the technique coverage (belt, vibrato, breathy, straight,
lip trill, ...) is unlike anything else in the corpus. But every file is a few seconds of
one vowel: it is vocoder and speaker-embedding material, not acoustic-model material.
