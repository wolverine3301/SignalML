# Agent Queue — what an unattended session works on

> **Purpose.** Logan's scarce resource is *his time*, not tokens. This file removes the
> "what should I point Claude at?" step, and — more importantly — stops unattended
> sessions from spending a week's tokens on work that isn't on the critical path.
>
> **Read this file first in any unattended session** (scheduled Friday run, or `/nextup`).
> It is the single source of truth for priority. `MIGRATION_PLAN.md` says what the phases
> are; this file says what to *do next given what's blocked*.

---

## The one rule that matters

**The project is a singing-voice AI and music studio.** Everything else — emotional
speech (D10), wave-3 talk/act/sing, instrumental generation (P9) — is downstream of that
and has *no consequence* until the singing MVP exists.

A session that burns its budget on Tier C while Tier A items are unblocked has failed,
even if the code is good. When in doubt, pick the lower-numbered tier.

### Why this file exists

2026-09-18: a session with spare tokens and no direction built out the emotion corpora
(D10) — 4,847 lines, correct work, wrong priority. Worse, it branched from
`aligner-eval-harness` (unreviewed P5.4 work on the critical path) instead of from `main`,
stacking off-path code on top of work Logan still needed to review. Both failures are
addressed by the protocol below.

---

## Session protocol

1. **Branch from `main`.** Never from another unreviewed branch. Name it for the queue id:
   `a2-mel-contract`, not `misc-fixes`.
2. **One queue item per session.** Finish it or leave it clearly half-done with a note —
   do not start a second item to fill leftover budget. Spare budget goes to depth on the
   same item (tests, edge cases, docs), not breadth.
3. **Never push. Never merge. Never touch `main`.** Commit on the branch and stop.
4. **The dev loop gates the commit:** `python -m uv sync`, `python -m uv run pytest`,
   `python -m uv run ruff check .` — all green, or the commit message says plainly what
   is failing and why.
5. **Update this file in the same commit:** move the item's status, and add a line to
   *Review queue* below. That is how the next session knows what happened, and how Logan
   reviews cheaply.
6. **If every Tier A and B item is blocked**, say so in the Review queue and stop. Do not
   fall through to Tier C to look busy — an honest "nothing unblocked this week" is worth
   more than off-path code.

### What unattended sessions may NOT do

- Download corpora, run GPU work, or anything needing the 5090 rig or `DATA_ROOT` audio.
- Resolve a 🔴 decision point in `DECISION_POINTS.md`. Sessions may *prepare* a decision
  (lay out options, verify maintenance state of a dependency, write the memo) — Logan makes
  the call.
- Change a stage contract, schema version, or `configs/audio.yaml` without flagging it at
  the top of the Review queue entry in capitals.

---

## Review queue — branches waiting on Logan

Newest last. One line each: what it is, whether it's critical path, and the verdict needed.

| Branch | Tier | What | Needs |
|---|---|---|---|
| `emotion-corpora-survey` | C (D10) | EARS + CREMA-D adapters, emotion contract, labeller calibration, 4,847 lines | Low urgency, and deliberately **left out of `main`** on rig day 2026-09-20. Rebase onto `main` when it is wanted |

**Merged to `main` 2026-09-20** (rig day — Logan's call to land everything but the
emotion branch): `rig-prep` (P7.3 `signalml train` wrapper, preflight, run records,
`trainer_opts`, `overfit_v1` / `full_acoustic_v2` recipes, bootstrap-script fixes —
`scripts/bootstrap_rig.ps1` had never parsed under PowerShell 5.1), `aligner-eval-harness`
(P5.4 eval + drift, landed unreviewed), `agent-queue` (this file).

---

## Tier A — critical path, laptop-doable

The singing MVP and the studio around it. These need no GPU and no corpus.

### A1. Studio front-end (D11)

`docs/STUDIO_UI.md` is a ~355-line interaction design with **zero implementation**.
`signalml/studio/api.py` (619 lines) already exposes the model-free half.
Build the UI against that API with stubbed model responses: shell, left rail, transport,
queue widget, then Screen 1 (Voice Lab), Screen 2 (Performance), 2½ (targeted repair, D12),
Screen 3 (Bank manager).
**Split across sessions — one screen per session.** Biggest single chunk of unblocked
critical-path work in the repo.
**Done when:** each screen renders and drives the real API for every model-free verb.

### A2. D5 — lock the mel contract

🟠 *silent-failure zone.* Mel parameters that disagree between stages corrupt training with
no error. CLAUDE.md already names hardcoded 22050s as a known bug class.
**Do:** single source of truth in `configs/audio.yaml`, a validator that fails loudly on
mismatch, artifacts recording their profile, a test that catches a deliberately mismatched
profile.
**Done when:** a wrong-profile artifact cannot reach a dataset without an explicit error.

### A3. D3 — phone-set reconciliation (MFA IPA ↔ trainer dictionary)

🔴 decision, but the *mapping work* is delegable. DiffSinger's dictionary and `mfa_ipa/en_v1`
must agree, including slur/stress representation.
**Do:** build the mapping table in `signalml/score/phoneset.py`, enumerate the phones that
don't map cleanly, write the memo. **Logan decides the unmappable cases.**
**Done when:** every phone in the corpus maps or appears on a short exceptions list.

### A4. P8 plumbing — synth CLI + mixdown

Delegable per `DECISION_POINTS.md`, nominally "once P7 emits any checkpoint" — but the CLI,
the mixdown path and the golden-path test can be built against a **stub checkpoint** now,
so that P7's first real checkpoint meets finished plumbing.
**Done when:** one command turns (score.json, voice, backing track) into a mixed WAV, with
the model call behind an injectable backend.

### A5. S5b "transcribe" stage skeleton (D1) ✅ DONE 2026-09-22

Built, plus more than the item asked for: `signalml transcribe` (manifest-driven,
injectable backend, 15 offline tests), `ph_num` in S6b, `signalml dataset
variance-config`, and the first bake-off — SOME over five real songs on CPU, mean note
coverage 0.950. Maintenance state checked as asked, and it had moved: SOME is superseded
by **GAME**, and SOME's *weights* are CC BY-NC-SA 4.0 while its code is MIT. Logan's call
recorded. Full write-up: `docs/notes/note_transcription.md`. D1 is now 🟡, and Logan's own
note transcriber is an endorsed direction in OPEN_QUESTIONS.

**What is left for variance training:** notes over a whole dataset (SOME's `batch_infer.py`,
~2 clips/s on CPU) then `signalml train variance`.

### A6. Pipeline performance instrumentation

`docs/notes/perf_instrumentation.md` has the design; no implementation exists.
**Done when:** stage timings land in `analysis.json` and `manifest report` can show them.

---

## Tier B — makes rig time and review time cheaper

Do these when Tier A is blocked, or when a session has budget left over and A-depth is done.

- **B1. Rig runbooks.** Exact command sequences for corpus onboarding, MFA install, P2–P5
  batch runs. Rig time should be spent *running*, not deciding. Draft from README + plan.
- **B2. D6 — vocoder training recipe.** We decided *that* we train our own, not *how*. Write
  the recipe: data requirements, schedule, eval gates, the NC-weights firewall.
- **B3. D9 — dataset curation & QC policy** plus the sampling scripts to enforce it.
- **B4. Test and fixture coverage gaps.** 59 modules, 25 test files. Find contract tests that
  don't exist yet; CLAUDE.md requires one per stage.

---

## Tier C — wave-3 and beyond (only when A and B are exhausted)

Not "someday never" — genuinely endorsed direction, just *after* the singing MVP. Emotional
speech (D10) is the eventual payoff of the singing work: "emotional music notes" controlling
tone and delivery in TTS. It cannot be evaluated until the singing model exists.

- Continue D10 emotion corpora (branch from `main`).
- P9 symbolic-first instrumental generation.
- Universal-phoneme-set cross-lingual work (Gaelic wave-2).

---

## Blocked — do not attempt unattended

| Item | Blocked on |
|---|---|
| P7 training (acoustic / variance / vocoder) | 5090 rig |
| P5 corpus runs, MFA install, SOFA eval | rig + dataset |
| D1 note-transcription bake-off | 5–10 real separated vocals on the rig |
| Corpus onboarding (`manifest scan` over real folders) | dataset on the rig |
