"""Studio API contract tests — CPU-only, offline, no model and no checkpoint.

That constraint is the point of the module under test: the cost bar is arithmetic over
segment hashes and render cache keys, so all of it runs on a laptop today. Anything
here that needed a GPU would mean logic had leaked into the wrong layer.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from signalml.manifest import scan_directory
from signalml.score import Score, load_score, save_score, split_segments
from signalml.stages.common import song_dir
from signalml.studio import (
    build_context,
    build_render_plan,
    config_fingerprint,
    list_scores,
    list_voices,
    measure_render_rate,
    read_variance,
    resolve_checkpoint,
    write_variance,
)
from signalml.studio.api import CHECKPOINT_POINTER, ENV_CHECKPOINT
from signalml.studio.server import StudioServer
from signalml.synth import (
    RenderInputs,
    VarianceTrack,
    new_record,
    save_record,
    save_variance,
)

CHECKPOINT = "a91f4c2e" * 8


def _note(start, end, midi, syllable, phonemes, **kw):
    return {
        "start": start, "end": end, "midi": midi,
        "syllable": syllable, "phonemes": phonemes, **kw,
    }


def _score(*, tail: str = "shine") -> Score:
    """Three phrases separated by rests well over the 0.30 s split threshold."""
    return Score.model_validate({
        "bpm": 96.0,
        "key": "G:major",
        "language": "en",
        "phone_set": "mfa_ipa/en_v1",
        "notes": [
            _note(0.0, 0.5, 67, "when", ["w", "ɛ", "n"]),
            _note(0.5, 1.0, 69, "the", ["ð", "ə"]),
            _note(2.0, 2.5, 71, "eve", ["i", "v"]),
            _note(2.5, 3.0, 72, "ning", ["n", "ɪ", "ŋ"]),
            _note(4.0, 5.5, 74, tail, ["ʃ", "aɪ", "n"]),
        ],
    })


@pytest.fixture()
def data_root(tmp_path, make_wav) -> Path:
    """A data root with one manifest record that has a score.json beside it."""
    root = tmp_path / "dr"
    make_wav(root / "raw" / "song.wav", seconds=1.0)
    manifest, _ = scan_directory(root, language="en", gender="F", singer="alice",
                                 source_quality="separated")
    rec = manifest.records[0]
    manifest.save()
    save_score(_score(), song_dir(root, rec.id) / "score.json")
    return root


@pytest.fixture()
def voiced_root(data_root: Path) -> Path:
    """The same root, plus an `aurora` voice pinned to CHECKPOINT and an ACTIVE file."""
    voice = data_root / "voices" / "aurora"
    (voice / "ref").mkdir(parents=True)
    np.save(voice / "embedding.npy", np.zeros(256, dtype=np.float32))
    (voice / "ref" / "legato.wav").write_bytes(b"RIFF")  # presence only; never decoded
    (voice / "profile.json").write_text(json.dumps({
        "name": "aurora",
        "created": "2026-07-03",
        "embedding_dim": 256,
        "checkpoint": CHECKPOINT,
        "sampler": "gaussian_v1",
        "similarity_guard": {"max_cos_train": 0.79, "verifier": "ecapa"},
        "notes": "sampled 2026-07-03, seed 1234",
    }), encoding="utf-8")
    pointer = data_root / CHECKPOINT_POINTER
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(CHECKPOINT, encoding="utf-8")
    return data_root


def _score_path(root: Path) -> Path:
    return next((root / "songs").glob("*/score.json"))


# --------------------------------------------------------------------------- context


def test_checkpoint_pointer_beats_env(data_root, monkeypatch):
    monkeypatch.setenv(ENV_CHECKPOINT, "from-env")
    assert resolve_checkpoint(data_root) == ("from-env", ENV_CHECKPOINT)

    pointer = data_root / CHECKPOINT_POINTER
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(f"  {CHECKPOINT}\n", encoding="utf-8")
    assert resolve_checkpoint(data_root) == (CHECKPOINT, CHECKPOINT_POINTER)


def test_context_reports_profile_and_counts(data_root):
    ctx = build_context(data_root, profile_name="dev")
    assert ctx.audio_profile == "dev"
    assert ctx.is_dev_profile is True  # the UI tints the top bar on exactly this
    assert ctx.sample_rate == 22050
    assert ctx.checkpoint is None  # no checkpoint before P7, and that is not an error
    assert ctx.counts == {"scores": 1, "voices": 0, "renders": 0}


def test_context_prod_profile_is_not_flagged_dev(voiced_root):
    ctx = build_context(voiced_root, profile_name="prod")
    assert ctx.is_dev_profile is False
    assert ctx.sample_rate == 44100
    assert ctx.checkpoint == CHECKPOINT
    assert ctx.counts["voices"] == 1


# ---------------------------------------------------------------------------- scores


def test_list_scores_is_manifest_driven(data_root):
    scores = list_scores(data_root)
    assert len(scores) == 1
    assert scores[0].singer == "alice"
    assert scores[0].note_count == 5 and scores[0].segments == 3
    assert scores[0].phone_set == "mfa_ipa/en_v1"


def test_score_not_in_the_manifest_is_invisible(data_root):
    """A stray songs/ directory is not work: stages query the manifest, never the disk."""
    stray = data_root / "songs" / "sng_9999"
    stray.mkdir(parents=True)
    save_score(_score(), stray / "score.json")
    assert all(s.song_id != "sng_9999" for s in list_scores(data_root))
    assert len(list_scores(data_root)) == 1


def test_unreadable_score_is_skipped_not_raised(data_root):
    _score_path(data_root).write_text("{ not json", encoding="utf-8")
    assert list_scores(data_root) == []


# ---------------------------------------------------------------------------- voices


def test_voices_empty_root(data_root):
    assert list_voices(data_root) == []


def test_voice_summary_and_staleness(voiced_root):
    voice = list_voices(voiced_root)[0]
    assert voice.name == "aurora"
    assert voice.max_cos_train == 0.79 and voice.verifier == "ecapa"
    assert voice.ref_phrases == ["legato.wav"]
    assert voice.embedding_sha256 is not None
    assert voice.stale is False

    assert list_voices(voiced_root, checkpoint="something-else")[0].stale is True
    # No active checkpoint is "unknown", not "fresh" — the bank must not claim currency.
    assert list_voices(voiced_root / "nope", checkpoint=None) == []


def test_voice_without_checkpoint_to_compare_is_unknown(voiced_root):
    (voiced_root / CHECKPOINT_POINTER).unlink()
    assert list_voices(voiced_root)[0].stale is None


# ------------------------------------------------------------------------- the plan


def test_plan_without_a_voice_is_honest_about_not_knowing(data_root):
    plan = build_render_plan(data_root, _score_path(data_root), profile_name="dev")
    assert plan.resolved is False
    assert plan.totals["total"] == 3
    assert all(row.state == "unknown" for row in plan.segments)
    assert all(row.cache_key is None for row in plan.segments)
    assert plan.notes and "unknown" in plan.notes[0]
    assert plan.estimate.seconds is None and plan.estimate.calibrated is False
    # The headline must not read "3 of 3 re-render" over three "?" markers.
    assert plan.summary().startswith("3 phrases, cache state unknown")
    assert "re-render" not in plan.summary()


def test_plan_resolves_and_reports_a_cold_cache(voiced_root):
    plan = build_render_plan(voiced_root, _score_path(voiced_root),
                             voice="aurora", seed=7, profile_name="prod")
    assert plan.resolved is True
    assert plan.checkpoint == CHECKPOINT
    assert plan.totals == {"total": 3, "cached": 0, "render": 3, "unknown": 0,
                           "dropped": 0}
    assert all(row.cached is False and row.state == "dirty" for row in plan.segments)
    assert len({row.cache_key for row in plan.segments}) == 3  # keys are per-phrase


def test_plan_sees_cache_hits(voiced_root):
    """Seed a cache entry for phrase 1 only; the plan must charge for the other two."""
    path = _score_path(voiced_root)
    score = load_score(path)
    segments = split_segments(score)
    voice = list_voices(voiced_root)[0]
    from signalml.config import active_profile

    inputs = RenderInputs(
        segment_hash=segments[1].content_hash,
        voice="aurora",
        embedding_sha256=voice.embedding_sha256,
        checkpoint_sha256=CHECKPOINT,
        audio_profile="prod",
        seed=7,
        config_sha256=config_fingerprint(active_profile("prod")),
    )
    save_record(voiced_root, new_record(inputs, segment_index=1))

    plan = build_render_plan(voiced_root, path, voice="aurora", seed=7,
                             profile_name="prod")
    assert [row.state for row in plan.segments] == ["dirty", "clean", "dirty"]
    assert plan.totals["cached"] == 1 and plan.totals["render"] == 2
    assert "2 of 3 phrases re-render, 1 cached" in plan.summary()


def test_plan_seed_and_profile_are_in_the_key(voiced_root):
    path = _score_path(voiced_root)
    base = build_render_plan(voiced_root, path, voice="aurora", seed=7,
                             profile_name="prod")
    other_seed = build_render_plan(voiced_root, path, voice="aurora", seed=8,
                                   profile_name="prod")
    other_profile = build_render_plan(voiced_root, path, voice="aurora", seed=7,
                                      profile_name="dev")
    keys = [p.segments[0].cache_key for p in (base, other_seed, other_profile)]
    assert len(set(keys)) == 3, "seed and audio profile must both change the cache key"


def test_plan_compare_reports_the_edit(voiced_root, tmp_path):
    """Editing the last phrase's lyric must cost exactly that phrase."""
    path = _score_path(voiced_root)
    edited_path = tmp_path / "edited.json"
    save_score(_score(tail="burn"), edited_path)

    plan = build_render_plan(voiced_root, path, compare=edited_path,
                             voice="aurora", seed=7, profile_name="prod")
    assert plan.compare_path is not None
    assert [row.reused_from for row in plan.segments] == [0, 1, None]
    assert plan.dropped == [2]
    assert plan.totals["render"] == 3  # nothing is cached yet, so everything is charged


def test_cache_beats_the_diff(voiced_root, tmp_path):
    """A phrase the diff calls new is free if its content is already on disk.

    Content addressing means an identical phrase rendered for some other song already
    has audio. Charging for it because it is new *here* would overcharge the edit.
    """
    path = _score_path(voiced_root)
    edited_path = tmp_path / "edited.json"
    edited = _score(tail="burn")
    save_score(edited, edited_path)

    voice = list_voices(voiced_root)[0]
    from signalml.config import active_profile

    new_phrase = split_segments(edited)[2]
    inputs = RenderInputs(
        segment_hash=new_phrase.content_hash,
        voice="aurora",
        embedding_sha256=voice.embedding_sha256,
        checkpoint_sha256=CHECKPOINT,
        audio_profile="prod",
        seed=7,
        config_sha256=config_fingerprint(active_profile("prod")),
    )
    save_record(voiced_root, new_record(inputs, segment_index=2))

    plan = build_render_plan(voiced_root, path, compare=edited_path,
                             voice="aurora", seed=7, profile_name="prod")
    row = plan.segments[2]
    assert row.reused_from is None, "the diff still reports it as a new phrase"
    assert row.state == "clean" and row.cached is True, "but the cache already has it"


def test_estimate_calibrates_from_timed_renders(voiced_root):
    """The estimate must come from measured renders, never from a built-in guess."""
    assert measure_render_rate(voiced_root) == (None, 0)

    voice = list_voices(voiced_root)[0]
    from signalml.config import active_profile

    for i, seg in enumerate(split_segments(load_score(_score_path(voiced_root)))):
        inputs = RenderInputs(
            segment_hash=seg.content_hash,
            voice="aurora",
            embedding_sha256=voice.embedding_sha256,
            checkpoint_sha256=CHECKPOINT,
            audio_profile="prod",
            seed=99,  # a seed the plan below does not use, so nothing reads as cached
            config_sha256=config_fingerprint(active_profile("prod")),
        )
        save_record(voiced_root, new_record(inputs, segment_index=i,
                                            elapsed_sec=4.0, audio_sec=2.0))

    rate, samples = measure_render_rate(voiced_root)
    assert rate == pytest.approx(2.0) and samples == 3

    plan = build_render_plan(voiced_root, _score_path(voiced_root), voice="aurora",
                             seed=7, profile_name="prod")
    assert plan.estimate.calibrated is True
    assert plan.estimate.seconds == pytest.approx(2 * plan.estimate.audio_sec)
    assert "s" in plan.summary()


def test_unresolved_estimate_is_labelled_an_upper_bound(voiced_root):
    """Timed renders exist, but with no voice the plan cannot say which are cached,
    so its duration covers every phrase and must be presented as a ceiling."""
    from signalml.config import active_profile

    voice = list_voices(voiced_root)[0]
    for i, seg in enumerate(split_segments(load_score(_score_path(voiced_root)))):
        inputs = RenderInputs(
            segment_hash=seg.content_hash, voice="aurora",
            embedding_sha256=voice.embedding_sha256, checkpoint_sha256=CHECKPOINT,
            audio_profile="prod", seed=3,
            config_sha256=config_fingerprint(active_profile("prod")),
        )
        save_record(voiced_root, new_record(inputs, segment_index=i,
                                            elapsed_sec=1.0, audio_sec=1.0))

    plan = build_render_plan(voiced_root, _score_path(voiced_root),
                             profile_name="prod")  # no voice
    assert plan.resolved is False
    assert plan.estimate.calibrated is True
    assert "cache state unknown" in plan.summary()
    assert "at most" in plan.summary()


# ------------------------------------------------------------------------- variance


def _track(origin="predicted", f0=440.0) -> VarianceTrack:
    return VarianceTrack(
        origin=origin, sample_rate=44100, frame_hop=512,
        phonemes=["ʃ", "aɪ", "n"], durations_sec=[0.1, 0.9, 0.2],
        f0_hz=[f0, f0, f0, f0], voiced=[True, True, True, False],
    )


@pytest.fixture()
def cached_render(voiced_root):
    from signalml.config import active_profile

    voice = list_voices(voiced_root)[0]
    inputs = RenderInputs(
        segment_hash=split_segments(load_score(_score_path(voiced_root)))[2].content_hash,
        voice="aurora",
        embedding_sha256=voice.embedding_sha256,
        checkpoint_sha256=CHECKPOINT,
        audio_profile="prod",
        seed=7,
        config_sha256=config_fingerprint(active_profile("prod")),
    )
    record = new_record(inputs, segment_index=2, note_ids=["n0005"])
    save_record(voiced_root, record)
    save_variance(voiced_root, record.key, _track())
    return voiced_root, record


def test_read_variance(cached_render):
    root, record = cached_render
    assert read_variance(root, record.key).phonemes == ["ʃ", "aɪ", "n"]


def test_read_variance_missing(voiced_root):
    from signalml.config import active_profile

    voice = list_voices(voiced_root)[0]
    inputs = RenderInputs(
        segment_hash="deadbeef", voice="aurora",
        embedding_sha256=voice.embedding_sha256, checkpoint_sha256=CHECKPOINT,
        audio_profile="prod", seed=1,
        config_sha256=config_fingerprint(active_profile("prod")),
    )
    record = new_record(inputs)
    save_record(voiced_root, record)
    with pytest.raises(FileNotFoundError):
        read_variance(voiced_root, record.key)


def test_edited_variance_mints_a_new_render_and_leaves_the_original(cached_render):
    """Repair must be non-destructive: A/B is just playing two keys."""
    root, record = cached_render
    written = write_variance(root, record.key, _track(origin="edited", f0=466.0),
                             notes="pulled 'shine' onto the rail")

    assert written.changed is True
    assert written.key != record.key
    assert written.record.inputs.variance_sha256 is not None
    assert written.record.inputs.segment_hash == record.inputs.segment_hash
    assert written.record.note_ids == ["n0005"]  # provenance carries across
    assert read_variance(root, record.key).f0_hz[0] == 440.0  # original untouched
    assert read_variance(root, written.key).f0_hz[0] == 466.0


def test_no_op_variance_edit_mints_nothing(cached_render):
    """The source render predicted its curve, so re-applying that curve is not a no-op:
    it moves variance_sha256 from null to a hash. Re-applying it a second time is."""
    root, record = cached_render
    first = write_variance(root, record.key, _track())
    assert first.changed is True

    second = write_variance(root, first.key, _track())
    assert second.changed is False and second.key == first.key


# ---------------------------------------------------------------------------- server


@pytest.fixture()
def server(voiced_root):
    srv = StudioServer(voiced_root, port=0, profile_name="prod")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _get(base, path):
    with urllib.request.urlopen(f"{base}{path}", timeout=5) as resp:
        return json.loads(resp.read())


def test_server_read_routes(server, voiced_root):
    _, base = server
    assert _get(base, "/api/context")["audio_profile"] == "prod"
    assert _get(base, "/api/scores")[0]["singer"] == "alice"
    assert _get(base, "/api/voices")[0]["name"] == "aurora"


def test_server_plan_get_and_post(server, voiced_root):
    _, base = server
    score = _score_path(voiced_root).as_posix()

    from urllib.parse import urlencode

    query = urlencode({"score": score, "voice": "aurora", "seed": 7})
    via_get = _get(base, f"/api/segments/plan?{query}")

    request = urllib.request.Request(
        f"{base}/api/segments/plan",
        data=json.dumps({"score": score, "voice": "aurora", "seed": 7}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as resp:
        via_post = json.loads(resp.read())

    assert via_get == via_post, "GET and POST must price a render identically"
    assert via_get["totals"]["total"] == 3
    assert via_get["cli"].startswith("signalml studio plan ")


def test_server_errors(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as missing_score:
        _get(base, "/api/segments/plan")
    assert missing_score.value.code == 400

    with pytest.raises(urllib.error.HTTPError) as no_route:
        _get(base, "/api/nope")
    assert no_route.value.code == 404

    with pytest.raises(urllib.error.HTTPError) as queue:
        _get(base, "/api/queue")
    assert queue.value.code == 501, "the queue is shaped but lands with P8"


def test_server_variance_round_trip(server, cached_render):
    _, base = server
    _, record = cached_render

    track = _get(base, f"/api/render/{record.key}/variance")
    assert track["phonemes"] == ["ʃ", "aɪ", "n"]

    edited = dict(track, origin="edited", f0_hz=[466.0] * 4)
    request = urllib.request.Request(
        f"{base}/api/render/{record.key}/variance",
        data=json.dumps({"variance": edited, "notes": "flat vowel"}).encode(),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(request, timeout=5) as resp:
        assert resp.status == 201
        written = json.loads(resp.read())
    assert written["changed"] is True and written["key"] != record.key


def test_server_rejects_a_malformed_variance(server, cached_render):
    _, base = server
    _, record = cached_render
    request = urllib.request.Request(
        f"{base}/api/render/{record.key}/variance",
        data=json.dumps({"variance": {"sample_rate": 44100}}).encode(),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=5)
    assert exc.value.code == 422
