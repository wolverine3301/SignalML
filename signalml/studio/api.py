"""Studio API — the model-free half of ``signalml studio`` (docs/STUDIO_UI.md §10).

**Nothing here imports a model, touches a GPU, or needs a checkpoint.** That is the
point: the Studio's most useful screen element — the cost bar, "3 of 14 phrases
re-render, 11 cached" — is pure arithmetic over ``score/segment.py`` hashes and
``synth/render.py`` cache keys, both of which already exist. So it is buildable,
testable and *usable* before P8 lands a renderer, and P8 changes none of its shapes.

Every function here is a pure function of ``DATA_ROOT`` plus its arguments, in the same
spirit as ``dash.server.build_status``: the HTTP layer in :mod:`signalml.studio.server`
is a router over these and owns no logic of its own.

Two conventions the whole module keeps:

- **Never directory-scan for work.** Scores are found by querying the manifest, like
  every stage (docs/PIPELINE_AND_CONTRACTS.md §2). The one exception is the render
  cache, which *is* a content-addressed directory — existence is the cache.
- **Every answer carries the command line that reproduces it** (``cli``). The UI never
  composes a command itself; a frontend-composed command is a lie waiting to happen,
  and this is the cheapest possible check that the UI holds no logic the CLI lacks.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..config import AudioProfile, active_profile
from ..hashing import sha256_json
from ..manifest import Manifest
from ..score import (
    DEFAULT_MIN_REST_SEC,
    load_score,
    plan_rerender,
    split_segments,
)
from ..stages.common import song_dir
from ..synth import (
    RenderInputs,
    RenderRecord,
    VarianceTrack,
    find_cached,
    iter_records,
    load_record,
    load_variance,
    new_record,
    render_dir,
    save_record,
    save_variance,
)
from ..synth.render import RECORD_FILENAME, VARIANCE_FILENAME

STUDIO_API_FORMAT = "signalml-studio/0.1"

SCORE_FILENAME = "score.json"
VOICES_DIRNAME = "voices"
PROFILE_FILENAME = "profile.json"
EMBEDDING_FILENAME = "embedding.npy"
REF_DIRNAME = "ref"

# Where the active checkpoint is declared until P7 grows a real registry.
CHECKPOINT_POINTER = "checkpoints/ACTIVE"
ENV_CHECKPOINT = "SIGNALML_CHECKPOINT"

# A plan whose voice/checkpoint cannot be resolved still reports segments and hashes;
# it just cannot say whether their audio is cached. These name that distinction.
CacheState = Literal["clean", "dirty", "unknown"]


# --------------------------------------------------------------------------- context


class StudioContext(BaseModel):
    """What the top bar shows. Getting any of this wrong is a silent-damage class of
    bug (a dev-profile render landing in a prod song), so it is a first-class answer
    rather than something the UI infers."""

    format: str = STUDIO_API_FORMAT
    rig: str
    data_root: str
    data_root_exists: bool
    audio_profile: str
    sample_rate: int
    is_dev_profile: bool  # the UI tints the whole top bar on this
    checkpoint: str | None
    checkpoint_source: str | None
    counts: dict[str, int]


def resolve_checkpoint(data_root: str | Path) -> tuple[str | None, str | None]:
    """The active checkpoint hash and where it came from, or ``(None, None)``.

    Pointer file first, then env var — a per-run override should beat a machine
    default, same precedence the audio profile uses. There is no checkpoint at all
    before P7 trains one, and ``None`` is the honest answer rather than a placeholder
    that would quietly key the render cache.
    """
    pointer = Path(data_root) / CHECKPOINT_POINTER
    if pointer.is_file():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            return value, CHECKPOINT_POINTER
    value = os.environ.get(ENV_CHECKPOINT, "").strip()
    if value:
        return value, ENV_CHECKPOINT
    return None, None


def build_context(
    data_root: str | Path, *, profile_name: str | None = None
) -> StudioContext:
    data_root = Path(data_root)
    profile = active_profile(profile_name)
    checkpoint, source = resolve_checkpoint(data_root)
    return StudioContext(
        rig=socket.gethostname(),
        data_root=str(data_root),
        data_root_exists=data_root.is_dir(),
        audio_profile=profile.name,
        sample_rate=profile.sample_rate,
        is_dev_profile=profile.name == "dev",
        checkpoint=checkpoint,
        checkpoint_source=source,
        counts={
            "scores": len(list_scores(data_root)),
            "voices": len(list_voices(data_root)),
            "renders": sum(1 for _ in iter_records(data_root)),
        },
    )


# ---------------------------------------------------------------------------- scores


class ScoreSummary(BaseModel):
    song_id: str
    path: str
    singer: str | None = None
    song: str | None = None
    bpm: float
    key: str | None = None
    language: str
    phone_set: str
    note_count: int
    segments: int
    duration_sec: float


def list_scores(
    data_root: str | Path, *, min_rest_sec: float = DEFAULT_MIN_REST_SEC
) -> list[ScoreSummary]:
    """Corpus scores, found by querying the manifest — never by scanning ``songs/``.

    A song whose ``score.json`` is missing or unreadable is simply absent from the
    listing: the Studio's score picker is a view over work that exists, and a malformed
    file is a job for ``signalml score validate``, not a reason to fail the screen.
    """
    data_root = Path(data_root)
    out: list[ScoreSummary] = []
    for rec in Manifest.for_data_root(data_root).records:
        path = song_dir(data_root, rec.id) / SCORE_FILENAME
        if not path.is_file():
            continue
        try:
            score = load_score(path)
        except (OSError, ValueError):
            continue
        segments = split_segments(score, min_rest_sec=min_rest_sec)
        out.append(
            ScoreSummary(
                song_id=rec.id,
                path=path.as_posix(),
                singer=rec.meta.singer,
                song=rec.meta.song,
                bpm=score.bpm,
                key=score.key,
                language=score.language,
                phone_set=score.phone_set,
                note_count=len(score.notes),
                segments=len(segments),
                duration_sec=round(score.notes[-1].end - score.notes[0].start, 3),
            )
        )
    return out


# ---------------------------------------------------------------------------- voices


class VoiceSummary(BaseModel):
    """One row of the bank manager (docs/STUDIO_UI.md §7)."""

    name: str
    path: str
    created: str | None = None
    checkpoint: str | None = None
    embedding_dim: int | None = None
    sampler: str | None = None
    max_cos_train: float | None = None
    verifier: str | None = None
    license_note: str | None = None
    notes: str = ""
    ref_phrases: list[str] = Field(default_factory=list)
    embedding_sha256: str | None = None
    # None = no active checkpoint to compare against, which is not the same as "fresh".
    stale: bool | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_voices(
    data_root: str | Path, *, checkpoint: str | None = None
) -> list[VoiceSummary]:
    """The voice bank. Empty before P7/P8 — which is a normal state, not an error.

    The render cache aside, this is the one place a directory listing is right: a voice
    bank is a directory of voices, and no manifest describes it.
    """
    data_root = Path(data_root)
    root = data_root / VOICES_DIRNAME
    if not root.is_dir():
        return []
    if checkpoint is None:
        checkpoint, _ = resolve_checkpoint(data_root)

    out: list[VoiceSummary] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        profile_path = directory / PROFILE_FILENAME
        if not profile_path.is_file():
            continue
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(profile, dict):
            continue

        embedding = directory / EMBEDDING_FILENAME
        guard = profile.get("similarity_guard")
        guard = guard if isinstance(guard, dict) else {}
        pinned = profile.get("checkpoint")
        ref_dir = directory / REF_DIRNAME
        out.append(
            VoiceSummary(
                name=str(profile.get("name") or directory.name),
                path=directory.as_posix(),
                created=profile.get("created"),
                checkpoint=pinned,
                embedding_dim=profile.get("embedding_dim"),
                sampler=profile.get("sampler"),
                max_cos_train=guard.get("max_cos_train"),
                verifier=guard.get("verifier"),
                license_note=profile.get("license_note"),
                notes=str(profile.get("notes") or ""),
                ref_phrases=sorted(p.name for p in ref_dir.glob("*.wav"))
                if ref_dir.is_dir()
                else [],
                embedding_sha256=_sha256_file(embedding) if embedding.is_file() else None,
                stale=None if (checkpoint is None or pinned is None) else pinned != checkpoint,
            )
        )
    return out


def find_voice(
    data_root: str | Path, name: str, *, checkpoint: str | None = None
) -> VoiceSummary | None:
    for voice in list_voices(data_root, checkpoint=checkpoint):
        if voice.name == name:
            return voice
    return None


# ------------------------------------------------------------------------- the plan


def config_fingerprint(profile: AudioProfile) -> str:
    """Hash of the resolved config that feeds ``RenderInputs.config_sha256``.

    Pre-P8 the audio profile *is* all the resolved render config there is, so that is
    what this hashes. When S8 lands its own config it widens here, which changes every
    key — hence the version field, and hence ``RENDER_KEY_VERSION`` will need a bump at
    that point. Nothing is cached yet, so that costs nothing today; doing it silently
    later would cost correctness.
    """
    return sha256_json({"v": 1, "audio_profile": profile.model_dump()})


class SegmentPlanRow(BaseModel):
    """One phrase block on the timeline (docs/STUDIO_UI.md §5, §6.1)."""

    index: int
    start: float
    end: float
    duration: float
    text: str
    note_count: int
    note_ids: list[str] = Field(default_factory=list)
    content_hash: str
    state: CacheState
    cached: bool | None = None
    cache_key: str | None = None
    # When comparing two scores: the old index this phrase's content came from.
    reused_from: int | None = None


class RenderEstimate(BaseModel):
    """How long the plan will take, and whether that number means anything.

    ``calibrated`` is the honest half. Before any render has been timed there is no
    rate to apply and ``seconds`` is ``None`` — the UI says "unknown", which is true,
    instead of showing a made-up number that would teach the user to distrust the bar.
    """

    audio_sec: float
    seconds: float | None = None
    calibrated: bool = False
    rate_sec_per_audio_sec: float | None = None
    samples: int = 0


class RenderPlan(BaseModel):
    """The cost bar, as data. The whole of D12's "price an edit before rendering"."""

    format: str = STUDIO_API_FORMAT
    score_path: str
    compare_path: str | None = None
    voice: str | None = None
    checkpoint: str | None = None
    audio_profile: str
    seed: int
    min_rest_sec: float
    # False = voice/checkpoint unresolved, so per-segment cache state is "unknown".
    resolved: bool
    segments: list[SegmentPlanRow]
    dropped: list[int] = Field(default_factory=list)
    totals: dict[str, int]
    estimate: RenderEstimate
    cli: str
    # Human-readable caveats for the UI to show under the bar (why something is
    # unknown, which voice was missing). Not a note count - see `note_count`.
    notes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        """One line for the cost bar. Never claims to know what it does not.

        With no voice or checkpoint the cache cannot be probed, so "3 of 3 re-render"
        would be a guess dressed as a count — and the per-phrase markers would all say
        ``?`` underneath it, which is how a user learns to stop reading the bar.
        """
        t = self.totals
        if not self.resolved:
            head = f"{t['total']} phrases, cache state unknown"
            tail = "at most" if self.estimate.seconds is not None else None
        else:
            head = f"{t['render']} of {t['total']} phrases re-render, {t['cached']} cached"
            tail = ""
        if self.estimate.seconds is None:
            return f"{head} — duration unknown (no timed renders yet)"
        prefix = f"{tail} " if tail else ""
        return f"{head} — {prefix}{self.estimate.seconds:.0f} s"


def measure_render_rate(data_root: str | Path) -> tuple[float | None, int]:
    """Seconds of wall clock per second of rendered audio, measured from the cache.

    Uses every record that carries both ``elapsed_sec`` and ``audio_sec``, aggregated
    as total/total rather than a mean of ratios so long phrases weigh more than short
    ones. Returns ``(None, 0)`` until something has actually been timed on this machine
    — an estimate is worth having only when it came from this GPU, not a guess.
    """
    elapsed = audio = 0.0
    samples = 0
    for record in iter_records(data_root):
        if record.elapsed_sec is None or not record.audio_sec:
            continue
        elapsed += record.elapsed_sec
        audio += record.audio_sec
        samples += 1
    if samples == 0 or audio <= 0:
        return None, 0
    return elapsed / audio, samples


def build_render_plan(
    data_root: str | Path,
    score_path: str | Path,
    *,
    compare: str | Path | None = None,
    voice: str | None = None,
    seed: int = 0,
    profile_name: str | None = None,
    min_rest_sec: float = DEFAULT_MIN_REST_SEC,
) -> RenderPlan:
    """Price a render before committing to it.

    Without ``compare``, the plan describes ``score_path`` and dirtiness comes purely
    from the render cache. With ``compare``, the plan describes the *edited* score
    (``compare``) and additionally reports which phrases survive the edit — the same
    diff ``signalml score segments --compare`` prints, with cache state layered on.

    Where the two disagree, **the cache wins**: ``plan_rerender`` calls a phrase new
    because it was not in the old score, but content addressing means an identical
    phrase rendered in some other song is already on disk. Reporting it as a re-render
    would overcharge the user for an edit that is in fact free.
    """
    data_root = Path(data_root)
    profile = active_profile(profile_name)
    checkpoint, _ = resolve_checkpoint(data_root)

    base = load_score(score_path)
    target = load_score(compare) if compare is not None else base
    segments = split_segments(target, min_rest_sec=min_rest_sec)

    notes: list[str] = []
    reused_from: dict[int, int] = {}
    dropped: list[int] = []
    diff_dirty: set[int] = set()
    if compare is not None:
        diff = plan_rerender(base, target, min_rest_sec=min_rest_sec)
        reused_from = dict(diff.reuse)
        dropped = list(diff.dropped)
        diff_dirty = set(diff.render)

    voice_ref = find_voice(data_root, voice, checkpoint=checkpoint) if voice else None
    if voice and voice_ref is None:
        notes.append(f"voice {voice!r} is not in the bank — cache state is unknown")
    resolved = bool(
        voice_ref
        and voice_ref.embedding_sha256
        and (voice_ref.checkpoint or checkpoint)
    )
    if not resolved and not notes:
        if checkpoint is None:
            notes.append("no active checkpoint — cache state is unknown (expected before P7)")
        elif voice is None:
            notes.append("no voice given — pass one to resolve cache state")
        else:
            notes.append(f"voice {voice!r} has no embedding.npy — cache state is unknown")

    config_sha256 = config_fingerprint(profile)
    rows: list[SegmentPlanRow] = []
    for seg in segments:
        cache_key: str | None = None
        cached: bool | None = None
        if resolved and voice_ref is not None:
            inputs = RenderInputs(
                segment_hash=seg.content_hash,
                voice=voice_ref.name,
                embedding_sha256=voice_ref.embedding_sha256 or "",
                checkpoint_sha256=voice_ref.checkpoint or checkpoint or "",
                audio_profile=profile.name,
                seed=seed,
                config_sha256=config_sha256,
            )
            cache_key = inputs.cache_key()
            cached = find_cached(data_root, inputs) is not None

        if cached is None:
            state: CacheState = "dirty" if seg.index in diff_dirty else "unknown"
            if compare is None:
                state = "unknown"
        else:
            state = "clean" if cached else "dirty"

        rows.append(
            SegmentPlanRow(
                index=seg.index,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                duration=round(seg.duration, 3),
                text=seg.text,
                note_count=len(seg.notes),
                note_ids=[n.id for n in seg.notes if n.id is not None],
                content_hash=seg.content_hash,
                state=state,
                cached=cached,
                cache_key=cache_key,
                reused_from=reused_from.get(seg.index),
            )
        )

    to_render = [r for r in rows if r.state != "clean"]
    audio_sec = round(sum(r.duration for r in to_render), 3)
    rate, samples = measure_render_rate(data_root)
    estimate = RenderEstimate(
        audio_sec=audio_sec,
        seconds=round(audio_sec * rate, 1) if rate is not None else None,
        calibrated=rate is not None,
        rate_sec_per_audio_sec=round(rate, 4) if rate is not None else None,
        samples=samples,
    )

    return RenderPlan(
        score_path=Path(score_path).as_posix(),
        compare_path=Path(compare).as_posix() if compare is not None else None,
        voice=voice,
        checkpoint=(voice_ref.checkpoint if voice_ref else None) or checkpoint,
        audio_profile=profile.name,
        seed=seed,
        min_rest_sec=min_rest_sec,
        resolved=resolved,
        segments=rows,
        dropped=dropped,
        totals={
            "total": len(rows),
            "cached": sum(1 for r in rows if r.state == "clean"),
            "render": len(to_render),
            "unknown": sum(1 for r in rows if r.state == "unknown"),
            "dropped": len(dropped),
        },
        estimate=estimate,
        cli=_plan_cli(score_path, compare, voice, seed, profile.name, min_rest_sec),
        notes=notes,
    )


def _plan_cli(
    score_path: str | Path,
    compare: str | Path | None,
    voice: str | None,
    seed: int,
    profile: str,
    min_rest_sec: float,
) -> str:
    parts = ["signalml studio plan", Path(score_path).as_posix()]
    if compare is not None:
        parts += ["--compare", Path(compare).as_posix()]
    if voice:
        parts += ["--voice", voice]
    parts += ["--seed", str(seed), "--profile", profile]
    if min_rest_sec != DEFAULT_MIN_REST_SEC:
        parts += ["--min-rest", str(min_rest_sec)]
    return " ".join(parts)


# ------------------------------------------------------------------------- variance


class VarianceWrite(BaseModel):
    """Result of persisting a hand-edited curve: a *new* render, not a mutated one."""

    format: str = STUDIO_API_FORMAT
    source_key: str
    key: str
    changed: bool
    variance_sha256: str
    record: RenderRecord
    cli: str


def read_variance(data_root: str | Path, key: str) -> VarianceTrack:
    """The persisted variance track for a cached render.

    Raises ``FileNotFoundError`` when the render exists but was never given one —
    which is the normal state until P8's variance models write them.
    """
    path = render_dir(data_root, key) / VARIANCE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no variance track for render {key[:16]}...")
    return load_variance(path)


def write_variance(
    data_root: str | Path, key: str, track: VarianceTrack, *, notes: str = ""
) -> VarianceWrite:
    """Persist an edited curve against ``key`` — the repair path's data layer.

    The edit never overwrites the render it came from. ``variance_sha256`` is a cache
    key input, so an edited track *addresses a different render*: the original stays on
    disk, the repaired one caches like any other, and A/B-ing the two is just playing
    two keys. That is what makes repair reversible rather than destructive.

    Returns with ``changed=False`` when the curve hashes to what the source already
    had — a no-op edit should not mint a second identical render.
    """
    source = load_record(render_dir(data_root, key) / RECORD_FILENAME)
    inputs = source.inputs.with_variance(track)
    new_key = inputs.cache_key()
    if new_key == source.key:
        return VarianceWrite(
            source_key=key,
            key=new_key,
            changed=False,
            variance_sha256=track.content_hash(),
            record=source,
            cli=f"signalml studio variance {key} --show",
        )

    save_variance(data_root, new_key, track)
    record = new_record(
        inputs,
        score_id=source.score_id,
        segment_index=source.segment_index,
        note_ids=list(source.note_ids),
        notes=notes or f"variance edited from {key[:16]}",
        audio_sec=round(track.duration_sec, 3) or None,
    )
    save_record(data_root, record)
    return VarianceWrite(
        source_key=key,
        key=new_key,
        changed=True,
        variance_sha256=track.content_hash(),
        record=record,
        cli=f"signalml studio variance {key} --apply <edited.json>",
    )
