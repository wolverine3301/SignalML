# Open Questions

> Kept current throughout the planning session. Each entry has enough context to answer
> quickly, plus my recommended default. Resolved questions move to the bottom with the
> decision and date.
>
> Status legend: 🔴 hard fork (design depends on it) · 🟡 shapes details, not structure ·
> 🟢 nice-to-know / can be decided later

---

**No open questions right now.** Implementation has started (Migration Plan P0,
2026-07-07). New questions will appear here as they arise.

## Future direction (endorsed, revisit when data/compute grows)

- **Universal phoneme set as a "pronunciation guide" for any language** (Logan's idea,
  refined from the Q2 note): typed lyrics → transcriber → project-owned universal
  (IPA-subset) phoneme set → the trained model sings any language *roughly*, even with
  scarce data for that language.

  **Verdict: keep it — this will work.** It is essentially cross-lingual phoneme
  transfer, the standard mechanism behind multilingual TTS/SVS: a shared IPA inventory
  lets the model reuse phones learned from *any* training language. Known limits to
  expect: phones never seen in training get approximated by their nearest trained
  neighbors, and accent/phonotactics will sound non-native for low-data languages —
  "roughly sing any language" is the accurate promise. The already-made decisions point
  straight at this: MFA IPA phone set (Q2), versioned `phone_set` field
  (`mfa_ipa/en_v1`), language tag in the manifest and score. When a second language
  family lands, add the mapping table (`signalml/score/phoneset.py`) and bump the
  phone-set version — no architectural change needed.

---

# Resolved

## 2026-07-12

| Q | Decision | Where it landed |
|---|---|---|
| D2 Trainer boundary (DECISION_POINTS.md) | **openvpi/DiffSinger v2.5.1 vendored (submodule, Apache-2.0), adopt-their-world, zero patches, own trainer venv** | `docs/notes/vendor_diffsinger.md`; DECISION_POINTS D2 ✅ (D5 mostly dissolved, D1/D3 sharpened); MIGRATION P7.1 done |

## 2026-07-07 (second batch)

| Q | Decision | Where it landed |
|---|---|---|
| Q13 Gaelic | **Both Irish and Scottish Gaelic** (~12 h total, separate folders, **all with lyrics**). **Wave-2 confirmed**: alignment/custom-aligner work waits until the English model proves out. Meanwhile the Gaelic audio is *not* idle — vocoder training is alignment-free, so all 12 h join vocoder training from day one; the manifest carries `language: ga`/`gd` per folder from the start | ARCHITECTURE §7 + risk register; MIGRATION P5.7 (per-language aligner config, wave-2) and P7.4 (Gaelic audio in vocoder corpus) |
| Q14 Lyrics | **Full coverage expected**: every song has a simple `.txt` lyrics file alongside the raw audio. P5's coverage scan becomes a verification pass, not a backfill hunt. A lyrics-*acquisition* step (fetch/Whisper-assist + human verify) stays on the roadmap as an S1 enhancement for future data | MIGRATION P5.2 (verify-not-backfill); S1 future enhancement noted |

## 2026-07-07 (first batch)

| Q | Decision | Where it landed |
|---|---|---|
| Q1 Framework | **PyTorch** confirmed | Already assumed everywhere; no doc changes needed |
| Q2 Phonemization | **MFA, MFA IPA phone set** (not ARPAbet as I'd assumed); custom universal set later | ARCHITECTURE §3.4/§6, CONTRACTS §S5/§4 (IPA examples), MIGRATION P5/P6 (G2P is now IPA-based — espeak-ng/`phonemizer` or MFA G2P, **not** `g2p_en`) |
| Q3 Score format | **JSON score file** confirmed | No changes; `phone_set` field now carries IPA set name |
| Q4 Language | **Language-robust pipeline; focus English + Gaelic (own curated data)** | ARCHITECTURE §7 data section; spawned Q13 |
| Q4 Vocoder/licensing | **Train our own vocoder** (community NC checkpoint only as temporary dev preview); commercial door kept open, dataset already curated with that in mind | ARCHITECTURE §3.3, MIGRATION P7 (own-vocoder training promoted from optional to milestone) |
| Q4 Corpus size | **~78 h female vocals** — substantial; changes the data outlook from "scarce" to "alignment quality is the bottleneck" | ARCHITECTURE §7 |
| Q5 Environment | **Windows-first for now** (WSL2 demoted to fallback/upgrade path) | ARCHITECTURE §2 rewritten; MIGRATION P5 (MFA via conda on native Windows, SOFA as escape hatch); CLAUDE.md |
| Q6 Separation | **Demucs confirmed** | No changes |
| Q7 Config | No preference → **YAML + pydantic** stands | No changes |
| Q8 Instrumental | **Symbolic-first, sequenced last** confirmed | No changes |
| Q9 Manifest | **JSONL** confirmed | No changes |
| Q10 Packaging | **`signalml`** confirmed | No changes |
| Q11 Sample rate | **Adjustable parameter**: 22.05 kHz *dev profile* for fast pipeline testing, 44.1 kHz *production profile* for real training | CONTRACTS §1/§6 (audio.yaml now defines named profiles), ARCHITECTURE §3.3 note (models/vocoders do **not** transfer across sample rates — a production run means production-profile features end-to-end), CLAUDE.md conventions |
| Q12 Masking track | **Keep as separate tool** (`signalml/tasks/masking/`) | No changes |
