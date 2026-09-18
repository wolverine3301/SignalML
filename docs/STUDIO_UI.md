# Studio UI — the interaction and visual layer

> Designed 2026-09-17. **Companion to `STUDIO.md`, not a replacement.** That doc
> decides *what the screens do and why*; this one decides *what they look like and
> how they behave* — layout, component anatomy, states, keyboard, and the server
> API the frontend talks to. Where the two disagree, `STUDIO.md` wins on scope and
> this one wins on presentation. Same gate: **starts after P8**.
>
> Visual reference (5 artboards, the three working screens plus bank and the
> system card): <https://claude.ai/artifact/3T3QyeP4pUvahoCvMYg7Rf>

## 0. What this adds

`STUDIO.md` left four things unspecified that decide whether the tool is pleasant:

1. **The shell** — the two-machine setup (laptop drives, 5090 renders) means
   *latency and queue state are the dominant UX fact*, and nothing owned them.
2. **What a card/region looks like while it is being made** — a render is not
   instant, and "spinner until audio" wastes the fact that the variance models
   finish first.
3. **How the reproducibility discipline surfaces** — the hard rule is "every action
   is CLI-reproducible"; that is only load-bearing if the UI shows you the command.
4. **Which verb is offered first** when a line is bad. D12 says this decides whether
   the tool feels like an editor or a slot machine. That is a layout decision.

Everything below follows from those four, plus the existing contracts.

## 1. The shell

Three fixed chrome elements on every screen. Content lives between them.

```
┌──────────────────────────────────────────────────────────────── 52px ──┐
│ SIGNAL·ML STUDIO   ●rig-5090  D:\signalml-data   ckpt a91f4c2e   [DEV] │  top bar
├────┬───────────────────────────────────────────────────────────────────┤
│LAB │                                                                   │
│PERF│                    screen content                                 │  64px rail
│EDIT│                                                                   │
│BANK│                                                                   │
├────┴───────────────────────────────────────────────────────────────────┤
│ ▶  c-07 · legato   ▁▂▃▅▃▂▁▂▅▇▅▂▁   A/B [1][2][3]   ◆ Queue · 7 waiting │  72px transport
└────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Top bar — the context you must never be wrong about

Left to right: brand, **rig identity + `DATA_ROOT`**, **checkpoint pin**, **audio
profile badge**.

The profile badge is the one piece of chrome with real teeth. Q11 says profiles never
mix and dev-profile checkpoints are throwaway; the failure it prevents is silent
(a 22.05 kHz render landing in a 44.1 kHz song). So:

- `prod` → a quiet green pill, `PROD · 44.1 kHz`.
- `dev` → an amber pill reading **`DEV · 22.05 kHz — renders are throwaway`**, *and*
  a 3px amber inset rule down the left edge of the top bar.

The rig identity matters for the same reason: a render that is "slow" because the rig
is unreachable should never look like a render that is slow because it is working.

### 1.2 Left rail — 64px, four destinations

`LAB` · `PERFORM` · `EDIT` · `BANK`. Icon over a 9px label; active item gets a
2px inset accent and a tinted ground. Four items is too few to earn a 200px labelled
sidebar and too many to leave unlabelled.

`EDIT` is one destination covering both screen 2½ (targeted repair) and screen 4 (the
general editor) — they are the same timeline component in two modes (§6, §8).

### 1.3 Transport — one audio context for the whole app

There is exactly one "now playing" in the Studio, and it lives here. That constraint
is what makes the compare loop work:

- **Position-preserving A/B.** The `[1][2][3]` chips swap the *source* without moving
  the playhead. Pressing `2` at 3.4 s into c-07 puts you at 3.4 s into c-08. This is
  the single most valuable interaction in the Voice Lab and it is impossible if each
  card owns its own player.
- Loop is on by default for regions and demo phrases (they are 5–8 s).
- Scrubber doubles as the region map on screen 2½: neighbouring segments render dimmed
  either side of the looped region, so you can hear the seam in context.

### 1.4 Queue widget — bottom right, always mounted

The queue is **server-side, on the rig**. The browser attaches to it; it does not own
it. Close the laptop mid-batch and the renders continue; reopening reattaches and
back-fills the tray. States:

| Dot | Label | Meaning |
|---|---|---|
| green | `idle · GPU free` | nothing running |
| blue | `working · N waiting · ~T left` | one GPU consumer, as designed |
| amber | `paused · training holds the GPU` | S7 has the device; CPU jobs still run |
| red | `rig unreachable · reconnecting` | queue survives, UI is stale |

Clicking expands a job list with per-job cancel. Batch sampling ("queue 32 overnight,
triage in the morning") is just this list with a long tail.

## 2. Design language

**Tokens are inherited verbatim from `signalml/dash/index.html`.** The Mission Control
dashboard already established a palette; a second one would be a bug.

```
--page #0d0d0d   --surface #1a1a19  --surface-2 #232322  --grid #2c2c2a
--ink #ffffff    --ink-2 #c3c2b7    --ink-3 #898781      --border rgba(255,255,255,.10)
--series #3987e5 --good #0ca30c     --warning #fab219    --critical #d03b3b
```

One addition: **`--critical-ink #e86a6a`** for red *text*. `#d03b3b` is 3.7:1 on
`--surface` — fine as a fill or border, under 4.5:1 as body copy.

**Colour semantics are fixed and never decorative:**

| | |
|---|---|
| blue | live, selected, re-renderable |
| amber | this will cost you something — dirty segment, dev profile, edited curve |
| green | passes a gate |
| red | blocked by a gate, or licence-encumbered |
| grey | frozen, cached, inert |

**Type.** Space Grotesk (display/brand) · IBM Plex Sans (UI) · IBM Plex Mono.
Everything that is a hash, seed, path, cosine, or sample rate is mono and is
selectable text — the provenance discipline is only useful if you can copy it.

## 3. Cross-cutting patterns

These four appear on every screen and are the substance of the design.

### 3.1 Price before you commit

Because `RenderInputs` makes a render a pure function of its inputs, the client can
always ask "what would this cost" *before* rendering. So **every render control states
its price in its own label**:

- `Render 3 phrases` (with `3 of 14 phrases re-render · 11 cached · ≈48 s` above it)
- `Render · all 14 cached` (disabled-looking, secondary — nothing to do)
- `Queue 8 · ≈ 2 min 40 s`

This is `signalml score segments SCORE --compare EDITED` — which already exists — given
a face. It is also the honest answer to "why is this taking so long": you were told.

### 3.2 The CLI mirror

Every inspector panel ends with a mono block containing **the exact command that
reproduces what you are looking at**, with a copy button. Keyboard `i`.

This is the "zero synthesis logic in the UI" rule made visible rather than merely
asserted. It is also the cheapest possible test of that rule: if a panel cannot write
its own command line, the UI has grown logic the CLI does not have, and that is a bug
to fix in the API, not in the frontend.

### 3.3 Progressive render — the variance track is the loading state

A candidate card is never a spinner. It moves through four states, each of which shows
something real:

| | State | What is on the card |
|---|---|---|
| 1 | queued | seed, sampler params, position in queue |
| 2 | variance in | **the F0 contour, drawn** — you can judge the melody before you hear it |
| 3 | audible | waveform, guard chip, demo-phrase pills, triage keys live |
| 4 | cached | as 3, plus a `⚡` marker; replay and re-sort never re-render |

State 2 falls straight out of the architecture: the variance models run before the
acoustic model and their output is already a persisted artifact (`VarianceTrack`).
Getting a free, meaningful progress display out of it is the payoff for that decision.

### 3.4 The guard chip

Every rendered candidate carries a similarity chip. It shows **the number, an icon,
and the singer it is nearest** — never colour alone.

```
✓ 0.62  nearest singer_014      < 0.72   green
⚠ 0.76  near singer_003         0.72–0.79 amber
✕ 0.83  blocked · singer_022    ≥ 0.79   red (threshold from profile.json)
```

"0.83 to singer_022" tells you to steer away on an axis. "Red" tells you nothing. The
inspector expands this to the top-3 nearest singers as bars with the guard line drawn
at its threshold, so "how much room do I have" is a glance.

## 4. Screen 1 — Voice Lab

Three columns: **sampler 300px · tray (flex) · inspector 340px**.

### Sampler (left)

Top to bottom: temperature (with a `safe/averaged ←→ distinctive/risky` scale under
it, because the number alone means nothing), PCA axis sliders (centre-detented, showing
signed values, pencil-to-rename with labels persisted to `studio/pca_labels.json`),
blend A/B slots, then **seed + reroll + batch count + the queue button** pinned to the
bottom. Unnamed axes render in italic grey — an honest "we haven't heard what this one
does yet" rather than a fake label.

### Tray (centre)

- **Header**: count, "scratch session, unsaved", blind-A/B toggle, sort.
- **Shelf** — a strip above the grid holding pinned candidates, with the breed action
  operating on the shelf selection (`Breed c-04 × c-07 → 4 children`). Pinning is the
  only persistent act in the tray, so it deserves its own place on screen rather than
  being a flag hidden on a card.
- **Grid**, 3 across, cards ~204px. Card anatomy top to bottom: id + rating dots +
  pin · guard chip · F0 sparkline · three demo-phrase pills (legato / syllabic /
  range) · footer with seed, temperature and the reroll/tweak/discard icons.
- **Blind A/B** (`shift+b`) replaces ids with `A`/`B`/`C`, hides seeds, temperature
  and the sparkline, and randomises order. Reveal on rate.

### Inspector (right)

Nearest-singers bars with the guard line · full provenance (seed, temperature, axes,
checkpoint, profile, render key) · the CLI mirror · `Save as voice…` gated on the guard
state, with one line saying what saving actually does (renders the validation set,
re-runs the guard as a hard gate, writes `voices/<name>/`).

## 5. Screen 2 — Performance

**Voice gallery 250px · centre (flex) · take controls + history 320px.**

### Voice gallery

Compact cards: play button (plays `ref/`), name, guard margin. A stale card gets an
amber border, a `STALE` badge, one line naming how far behind the pin is, and a
`Reproject →` button that opens the flow in §7.

### Centre — score source, then the timeline

The score-source tabs are `Corpus score` · `Import MIDI + lyrics` · `Retext a melody`.
**There is no bare lyrics box anywhere in the Studio.** Words are always retexted onto
a melody you have already chosen; free-form "type words, get a song" is melody
generation, which is P9. The tab labels carry that honesty; the UI must not imply
otherwise by offering an empty text field with a Render button next to it.

Below it, the score header (`signalml-score/0.2 · 96 bpm · G:major · en ·
mfa_ipa/en_v1 · 62 notes`) and then the **segment timeline** — the centrepiece:

- A row of proportional **phrase blocks**, each showing its index and lyric, coloured
  by state (§6.1). This is `score/segment.py`'s output, directly.
- Under it a **piano-roll-lite**: notes as small rects at pitch, with **dashed vertical
  rules at the phrase rests** — the only legal splice points, so the constraint is
  visible rather than a surprise later.
- Under that, the **cost bar** (§3.1) with the render button.

### Right — take controls and history

Transpose (with an auto-fit hint once a voice has an observed comfortable range),
tempo scale, seed, backing track + mix gain. Then render history: one row per render
with voice, transpose, seed, checkpoint, each replayable and each with
`open in editor →`. The footer states why it exists: an accidentally great take stays
recoverable.

## 6. Screen 2½ — targeted repair (D12)

The convergence of Performance and the Editor. One selected **live** region.

### 6.1 Region states

| State | Look | Meaning |
|---|---|---|
| clean | `--surface-2`, hairline border | cache hit, nothing to do |
| dirty | amber tint + amber border, `✎` | inputs changed, will re-render |
| live · selected | blue tint + blue border + focus ring | score-backed, carries provenance |
| frozen | 45° hatch, grey border, `❄` | plain audio, no longer regenerable |

**Freeze / unfreeze** is the verb, with the consequence stated in the UI next to the
button ("freezing drops the score link — audio only, no longer re-renderable").
Boundaries snap to phrase rests; the segmenter only cuts there anyway.

### 6.2 The repair canvas — F0 over mel

The main panel is the mel spectrogram with three overlaid layers:

1. **Note rails** — dashed horizontals at each note's pitch across its own span, with
   note names on the right gutter. The score's intent, drawn.
2. **Predicted F0** — a grey dashed line. What the variance models said.
3. **Edited F0** — a solid amber line with draggable handles. What you changed it to.

The deviation region is boxed in amber with a callout giving the actual error
(`−34 cents under D5, 1.9 s`) and, crucially, what the edit *preserved*: "the take
keeps its timing, breath and consonants — only the curve changed." That sentence is
the whole argument for intent (b) over rerolling.

Tools: `draw` · `smooth` · `pull to note` · `vibrato` · `Reset to predicted`.
The syllable/IPA ruler runs along the bottom.

### 6.3 Verb order is the design

The right panel lists the three fixes **ordered by how much of the take each keeps**,
numbered, with reroll last and its cost stated in plain words:

1. **Repair the pitch** — edit the curve, keep the seed. *(active by default)*
2. **Comp from another take** — take 2 already sings this phrase well. Always works.
3. **Reroll the seed** — cheap (~6 s), but it throws away the 90% that works.

The panel says why out loud: *a tool whose only answer is "regenerate" teaches you to
reroll thirty times chasing one detail.*

The **vary-strength slider** (D12 method 3, partial denoising) is rendered **visibly
locked**, with a padlock and the reason: "stays locked until D12 gate E3 passes by ear.
Nothing in the UI depends on it." Showing a gated feature as gated is better than
hiding it — it tells you the shape of the tool and keeps the gate honest.

### 6.4 Comp lanes

Sibling takes stack as rows under the canvas, each a mini-waveform; clicking a lane
comps it in for this region. A lane that is notably better on this phrase can be
flagged (`best "shine" →`). The footer states the mechanism: splice at the mel, vocode
the whole song in one pass, re-apply effects wholesale from the EDL.

### 6.5 Cost panel

Mirrors §3.1 at region scale and shows the *diff of the cache key*, which is the part
worth seeing:

```
phrases affected   1 of 14
segment hash       4c9e11b0  unchanged
variance           null → 8d17ac54
new render key     b02f7e41c9d3…
estimate           ≈ 9 s
```

## 7. Screen 3 — Bank manager

A table, deliberately: name · reference-phrase waveform with play · checkpoint pin with
staleness dot · guard margin as number + bar · provenance/licence · actions.

Two things earn their place here:

- **Licence is a column.** A voice built on the community NC vocoder renders at 55%
  opacity with a red `CC BY-NC / dev preview only` cell and `never ships` in place of
  its actions. Licensing hygiene is a contract; making it a badge you cannot miss is
  cheaper than a policy document.
- **Reproject is a modal with an A/B, not a button that just does it.** Before/after
  columns of the same `ref/` phrases, plus self-similarity and the guard-margin delta,
  plus a plain-language threshold ("0.91 is a good projection; below ~0.85, sample a
  fresh voice instead"), plus the equivalent CLI line, then
  `Keep the old pin` / `Accept reprojection`. Reprojection is an approximation and the
  UI should say so.

## 8. Screen 4 — the general editor (phase E)

Not designed in detail here, deliberately. The important finding from designing 2½ is
that **screen 4's tier-1 timeline and screen 2½'s timeline are the same component** —
live/frozen regions, rest-snapped boundaries, comp lanes, an EDL underneath. Build 2½
first and the editor is largely layout and a tool palette.

The design constraint that follows: keep tier-2 effects (EQ / comp / de-ess / reverb)
behind an **FX drawer** on the track, never inline on the timeline. Tier 1 plus
comping is the daily driver; the chain is occasional. Letting the chain onto the main
surface is how this screen turns into Audacity.

## 9. Keyboard

Triage is the inner loop; it has to work one-handed.

| Key | Action |
|---|---|
| `space` | play / pause the focused thing |
| `1`–`9` | swap to candidate/take N, **playhead held** |
| `←` `→` | move focus in the tray |
| `p` / `x` | pin to shelf / discard |
| `r` / `b` | reroll same settings / breed the shelf |
| `i` | inspector + CLI mirror |
| `l` | loop the selected region |
| `shift+b` | blind A/B on / off |

Digits are *swap*, not *rate*, because position-preserving comparison is worth more
than a rating shortcut; rating is the dot row on the card (click) and `1`–`9` would
otherwise collide with it.

## 10. Server API

The API-first pre-commitment (`STUDIO.md` §1, D11) made concrete. `signalml/studio/`
holds the server with **no model code**; every handler is a thin call into
`signalml.voices` / `signalml.synth` / `signalml.score`.

**Built (2026-09-18, model-free — runs on the laptop today):**

```
GET  /api/context                  rig id, DATA_ROOT, active checkpoint, audio profile
GET  /api/scores                   corpus scores (manifest-driven, never a dir scan)
GET  /api/voices                   bank list + staleness against the active checkpoint
GET  /api/segments/plan?score=...  the cost bar (same answer as POST; GET is linkable)
POST /api/segments/plan            score (+ edited score) → segments, hashes, cache
                                   hits, estimate   ← this is the cost bar
GET  /api/render/{key}/variance    the VarianceTrack
PUT  /api/render/{key}/variance    hand-edited curve → new key (the repair path)
GET  /api/queue                    501 until P8 — shaped, deliberately not faked
```

**Pending — each needs the renderer, or a stage that is not written yet:**

```
POST /api/voices                   promote a candidate (full `voice new` guard path)
POST /api/voices/{name}/reproject  → before/after ref renders, pending acceptance
POST /api/scores/import            MIDI + lyrics → score.json (P6 importer)
POST /api/scores/retext            D4 lyric-fitting onto a fixed melody
POST /api/render                   RenderInputs[] → job id (enqueue; cache-aware)
GET  /api/render/{key}/audio       cached artifact
WS   /api/events                   queue progress, stage transitions, tray updates
```

**Transport: stdlib `http.server` for now, FastAPI when the queue lands.** §2's
FastAPI choice is not reversed — it is the right answer for the WebSocket the job
queue needs, and that queue arrives with the renderer in P8. Everything servable today
is read-only JSON over eight routes, this repo already runs two stdlib servers
(`dash`, `ship`), and the logic all lives in `studio/api.py` as pure functions, so the
transport swap is a file rather than a rewrite. Taking the dependency now would buy
nothing and would have to be justified on the rig's install too.

Three rules for this surface:

- **`/api/segments/plan` is the only thing the cost bar needs**, and it is model-free:
  segment hashes and render cache keys already exist, so pricing an edit never needs a
  GPU. That is why the bar could ship before the renderer.
- **Every mutating endpoint returns the equivalent CLI invocation** in its response
  body. That is where the CLI mirror's text comes from — the frontend never composes
  a command string itself, because a frontend-composed command is a lie waiting to
  happen.
- **Unknown is a first-class answer.** With no voice or no checkpoint, per-segment
  cache state is `"unknown"`, not `"dirty"` — and the estimate is `null` with
  `calibrated: false` rather than a made-up number. A cost bar that guesses once is a
  cost bar nobody reads again.

### Where the cache and the diff disagree

`plan_rerender` calls a phrase new because it was not in the old score; content
addressing means an identical phrase rendered for some *other* song already has audio.
**The cache wins** — reporting it as a re-render would overcharge an edit that is in
fact free. `tests/test_studio.py::test_cache_beats_the_diff` pins this.

### Calibrating the estimate

`RenderRecord` carries `elapsed_sec` and `audio_sec` (measurements, deliberately
outside the cache key, so adding them invalidated nothing). `measure_render_rate`
aggregates them total-over-total across the cache, so the “≈48 s” in the cost bar is
measured on *this* rig rather than guessed. Until P8 writes the first timed record the
rate is `None` and the UI says so.

### State ownership

| Lives on the rig | Lives in the browser |
|---|---|
| job queue, render cache, tray scratch dir, pca labels, voices, EDLs | focus, selection, blind-mode flag, playhead, zoom |

Nothing the user would mourn lives in the tab. That is what makes "close the laptop
mid-batch" work.

## 11. Empty, slow, and broken

- **First run** (zero voices, zero renders): one centred card, one sentence, one
  button — `Sample 8 candidates` — plus the checkpoint/profile/seed line underneath.
  Not a tour.
- **Empty tray after discarding everything**: the sampler stays; the tray shows the
  last-used settings and a reroll, not an illustration.
- **Rig unreachable**: the queue chip goes red, the top bar rig dot goes red, render
  buttons disable with `rig unreachable` in the label. Already-cached audio still
  plays — the cache is on disk and the UI should not pretend otherwise.
- **Guard blocks a save**: never a modal. The card's chip already said so; `Save` is
  disabled with the reason inline and a nudge toward the axis to steer on.

## 12. What this design does not decide

- **The frontend framework.** D11 fork 1 stays open. Nothing above needs a framework;
  the candidate tray and the timeline are the two components that would justify one.
- **The demo score set** (D11 fork 2) — still to be chosen jointly with D8's benchmark
  set. The design assumes exactly three phrases named *legato / syllabic / range* and
  the card layout depends on that count.
- **Whether the vary-strength slider ever exists** — gated on D12 E3, shown locked.
- **Screen 4's tier 2/3 tools** — deferred with the rest of phase E.

## 13. Build order

Maps onto `STUDIO.md` §7 without changing it.

| | Build | Status |
|---|---|---|
| A′ | `signalml/studio/` API core: context, scores, voices, `/api/segments/plan`, variance read/write, and `signalml studio serve / plan / variance` | **done 2026-09-18** — model-free, 25 contract tests |
| A | shell + transport + queue + Voice Lab (temperature, seed, tray, save) | after P8. The shell is a prerequisite, not polish — the queue widget is what makes remote rendering bearable |
| B | shelf/breed, PCA axes, blind A/B, batch | after A |
| C | Performance: score picker, timeline, history | timeline component lands here and is reused by 2½ and 4 |
| F | Screen 2½: F0 editing, comp lanes, freeze | the highest-value screen; needs C's timeline. Its *data* layer shipped in A′ |
| E | Screen 4 on the same timeline | tier 1 + comping only |

**No frontend has been built, on purpose.** D11 fork 1 (Gradio spike vs custom SPA) is
still open, and writing a page now would close it by accident. `signalml studio plan`
prints the cost bar in the terminal instead, which proves the API end to end without
committing to a framework:

```
songs/sng_0001/score.json -> edited.json  [prod, seed 7, voice aurora]
  1 of 3 phrases re-render, 2 cached — 5 s
  [  0]   0.000-  1.000s cached  46a00dd7441d  when the  <- old 0
  [  1]   2.000-  3.000s cached  67ac344debba  eve ning  <- old 1
  [  2]   4.000-  5.500s RENDER  c02eb64ed556  burn
  dropped from the old score: [2]
```
