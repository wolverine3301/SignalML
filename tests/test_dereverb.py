"""S4b dereverb - CPU only, offline, no torch: the model is a stub, the audio synthetic."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
import yaml
from scipy.signal import fftconvolve

from signalml.config import active_profile
from signalml.manifest import Manifest, scan_directory
from signalml.restore.roformer import chunked_apply, load_msst_config
from signalml.stages.common import song_dir, update_analysis
from signalml.stages.dereverb import (
    DEREVERB_VERSION,
    OUT_NAME,
    DereverbConfig,
    compose,
    dereverb_name,
    dereverb_songs,
    is_current,
    judge_phrases,
    stem_input,
)

SR = active_profile("dev").sample_rate
UTTS = [{"start": 0.3, "end": 2.2}, {"start": 2.2, "end": 4.6}]
PHONES = ([{"ph": "a", "start": 0.5 + 0.3 * i, "end": 0.8 + 0.3 * i} for i in range(4)]
          + [{"ph": "a", "start": 2.6 + 0.3 * i, "end": 2.9 + 0.3 * i} for i in range(4)])


def _tone(seconds: float, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    env = np.minimum(1.0, np.minimum(t, seconds - t) / 0.05)  # 50 ms fades, no clicks
    return env * sum(0.3 / k * np.sin(2 * np.pi * hz * k * t) for k in range(1, 6))


def _dry(hz: float = 220.0, gain: float = 1.0) -> np.ndarray:
    """5 s: sung 0.5-1.7 s and 2.6-3.8 s, silence elsewhere (gap after each >= 0.8 s)."""
    y = np.zeros(int(5.0 * SR))
    for start in (0.5, 2.6):
        seg = gain * _tone(1.2, hz)
        y[int(start * SR):int(start * SR) + len(seg)] = seg
    return y


def _reverb(y: np.ndarray, rt60: float = 1.2, wet: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(rt60 * SR)) / SR
    ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    tail = fftconvolve(y, ir)[: len(y)]
    return y + wet * tail * np.sqrt((y ** 2).sum() / (tail ** 2).sum())


# ---------------------------------------------------------------- overlap-add


@pytest.mark.parametrize("length", [9_000, 50_000, 123_457])
@pytest.mark.parametrize("batch", [1, 3])
def test_chunked_apply_reassembles_exactly(length, batch):
    x = np.random.default_rng(1).standard_normal((2, length)).astype(np.float32)
    out = chunked_apply(x, lambda b: b, chunk=8_000, overlap=2, batch_size=batch)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, x, atol=1e-5)
    half = chunked_apply(x, lambda b: 0.5 * b, chunk=8_000, overlap=4, batch_size=batch)
    np.testing.assert_allclose(half, 0.5 * x, atol=1e-5)


def test_chunked_apply_feeds_full_chunks():
    seen = []

    def fn(batch):
        seen.append(batch.shape)
        return batch

    chunked_apply(np.ones((1, 30_000), dtype=np.float32), fn, chunk=8_000, batch_size=2)
    assert all(shape[1:] == (1, 8_000) for shape in seen)


def test_msst_config_reads_tuples_but_stays_safe(tmp_path):
    good = tmp_path / "ok.yaml"
    good.write_text("model:\n  freqs_per_bands: !!python/tuple [2, 4, 8]\n", encoding="utf-8")
    assert load_msst_config(good)["model"]["freqs_per_bands"] == (2, 4, 8)
    bad = tmp_path / "bad.yaml"
    bad.write_text("x: !!python/object/apply:os.system ['echo hi']\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_msst_config(bad)


# ---------------------------------------------------------------- judging


def test_removing_reverb_is_kept():
    dry = _dry()
    first, second = judge_phrases(_reverb(dry), dry, SR, UTTS, PHONES)
    for v in (first, second):
        assert v["keep"] and v["reason"] == "improved"
        assert v["before"]["tail_db"] - v["after"]["tail_db"] > 10  # the tail went
        assert v["f0_cents"] < 5 and v["voiced_overlap"] > 0.95


@pytest.mark.parametrize("derev, reason", [
    (_dry(hz=233.1), "f0"),        # ~ +100 cents: the model moved the pitch
    (_dry(gain=0.2), "level"),     # -14 dB: the model ate the singing
])
def test_damage_is_reverted(derev, reason):
    verdicts = judge_phrases(_reverb(_dry()), derev, SR, UTTS, PHONES)
    assert [v["keep"] for v in verdicts] == [False, False]
    assert {v["reason"] for v in verdicts} == {reason}


def test_nothing_gained_is_reverted():
    wet = _reverb(_dry())
    verdicts = judge_phrases(wet, wet.copy(), SR, UTTS, PHONES)
    assert [(v["keep"], v["reason"]) for v in verdicts] == [(False, "no_gain")] * 2


def test_phrase_without_phones_is_unscored():
    dry = _dry()
    *_, empty = judge_phrases(_reverb(dry), dry, SR, UTTS + [{"start": 4.6, "end": 5.0}],
                              PHONES)
    assert (empty["keep"], empty["reason"]) == (False, "unscored")


def test_verdicts_track_alignment_and_model():
    v = {"version": DEREVERB_VERSION, "model": "m1",
         "phrases": [dict(u, keep=True) for u in UTTS]}
    assert is_current(v, UTTS, "m1") and is_current(v, UTTS)
    assert not is_current(v, UTTS, "m2")
    assert not is_current(v, [UTTS[0], dict(UTTS[1], end=4.7)])
    assert not is_current(dict(v, version=DEREVERB_VERSION + 1), UTTS)
    assert dereverb_name("utterances.phrase.json") == "dereverb.phrase.json"


# ---------------------------------------------------------------- composing


def test_compose_swaps_only_kept_phrases():
    orig = np.full(int(5.0 * SR), 1.0, dtype=np.float32)
    derev = np.full_like(orig, -1.0)
    verdicts = [dict(UTTS[0], keep=False), dict(UTTS[1], keep=True)]
    out = compose(orig, derev, SR, verdicts, fade_sec=0.02)
    assert out[int(1.0 * SR)] == pytest.approx(1.0) and out[int(3.0 * SR)] == pytest.approx(-1.0)
    edge = int(2.2 * SR)
    assert abs(out[edge]) < 0.1  # mid-crossfade at the boundary
    assert np.all(np.abs(np.diff(out)) < 0.01)  # no step anywhere
    assert np.array_equal(compose(orig, derev, SR, verdicts[:1]), orig)


# ---------------------------------------------------------------- stage


def _song(root, make_wav, singer: str, source_quality: str) -> str:
    make_wav(root / "raw" / f"{singer}.wav", seconds=1.0, hz=200 + sum(map(ord, singer)) % 300)
    manifest, new = scan_directory(root, language="en", gender="F", singer=singer)
    rec = new[0]
    rec.status.separated = rec.status.cleaned = rec.status.aligned = True
    rec.meta.source_quality = source_quality
    manifest.upsert(rec)
    manifest.save()
    sdir = song_dir(root, rec.id)
    (sdir / "clean").mkdir(parents=True, exist_ok=True)
    sf.write(str(sdir / "clean" / "vocals.wav"), _reverb(_dry()).astype(np.float32), SR)
    update_analysis(sdir, "clean", {"profile": "dev"})
    adir = sdir / "align"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "utterances.json").write_text(json.dumps(UTTS), encoding="utf-8")
    (adir / "phones.json").write_text(json.dumps({"phones": PHONES}), encoding="utf-8")
    return rec.id


class _Stub:
    """De-reverb that returns the dry signal it was built from (an oracle)."""

    def __init__(self, scale: float = 1.0):
        self.calls, self.shapes, self.scale = 0, [], scale

    def __call__(self, y, sr):
        self.calls += 1
        self.shapes.append(np.shape(y))
        return (self.scale * _dry()[: np.shape(y)[-1]]).astype(np.float32)


def test_stage_scope_outputs_and_idempotence(tmp_path, make_wav):
    root = tmp_path / "dr"
    sep = _song(root, make_wav, "wet", "separated")
    stu = _song(root, make_wav, "dry", "studio")
    stub = _Stub()
    cfg = DereverbConfig(model="stub-1")
    s = dereverb_songs(root, config=cfg, backend=(stub, {"device": "cpu"}))
    assert s.processed == [sep] and stu in s.skipped and stub.calls == 1
    assert s.judged == [f"{sep}/dereverb.json"] and (s.kept, s.phrases) == (2, 2)
    sdir = song_dir(root, sep)
    out, sr = sf.read(str(sdir / "clean" / OUT_NAME))
    assert sr == SR and len(out) == len(_dry())
    info = json.loads((sdir / "analysis.json").read_text(encoding="utf-8"))["dereverb"]
    assert info["model"] == "stub-1" and info["output"] == f"clean/{OUT_NAME}"
    assert not (song_dir(root, stu) / "clean" / OUT_NAME).exists()  # studio untouched

    again = dereverb_songs(root, config=cfg, backend=(stub, {}))
    assert again.processed == [] and again.judged == [] and stub.calls == 1
    # a re-alignment makes the verdicts stale; only they are redone
    (sdir / "align" / "utterances.json").write_text(
        json.dumps([UTTS[0], dict(UTTS[1], end=4.7)]), encoding="utf-8")
    stale = dereverb_songs(root, config=cfg, backend=(stub, {}))
    assert stale.processed == [] and stale.judged == [f"{sep}/dereverb.json"]
    # another model: audio and verdicts are remade
    other = dereverb_songs(root, config=DereverbConfig(model="stub-2"), backend=(stub, {}))
    assert other.processed == [sep] and stub.calls == 2
    assert dereverb_songs(root, config=cfg, backend=(stub, {}), force=True,
                          limit=0).processed == []  # limit caps remade audio


def test_stage_never_loads_a_model_with_nothing_to_do(tmp_path, make_wav):
    root = tmp_path / "dr"
    _song(root, make_wav, "dry", "studio")
    # backend=None would import torch and the real checkpoint; nothing is in scope
    assert dereverb_songs(root, config=DereverbConfig()).processed == []
    m = Manifest.for_data_root(root)
    assert len(m.records) == 1


def test_stem_input_finds_the_clean_gain():
    rng = np.random.default_rng(3)
    stem = rng.standard_normal((5_000, 2)).astype(np.float32)
    got = stem_input(0.6 * stem.mean(axis=1), stem)
    assert got is not None
    wide, gain, resid = got
    assert wide.shape == (2, 5_000) and gain == pytest.approx(0.6) and resid < -60
    assert stem_input(rng.standard_normal(5_000), stem) is None        # not its source
    assert stem_input(0.6 * stem.mean(axis=1), stem[:, :1]) is None   # mono stem
    assert stem_input(0.6 * stem.mean(axis=1)[:4_000], stem) is None  # other length


def test_stage_feeds_the_stereo_stem(tmp_path, make_wav):
    root = tmp_path / "dr"
    sid = _song(root, make_wav, "wide", "separated")
    sdir = song_dir(root, sid)
    wet = sf.read(str(sdir / "clean" / "vocals.wav"), dtype="float32")[0]
    (sdir / "stems").mkdir()
    sf.write(str(sdir / "stems" / "vocals.wav"), np.stack([2 * wet, 2 * wet], axis=1), SR,
             subtype="FLOAT")
    stub = _Stub(scale=2.0)  # stem level; the stage scales it back by the fitted gain
    s = dereverb_songs(root, config=DereverbConfig(), backend=(stub, {}))
    assert s.processed == [sid] and stub.shapes == [(2, len(wet))]
    info = json.loads((sdir / "analysis.json").read_text(encoding="utf-8"))["dereverb"]
    assert info["source"] == "stems/vocals.wav" and info["gain"] == pytest.approx(0.5)
    out, _ = sf.read(str(sdir / "clean" / OUT_NAME), dtype="float32")
    np.testing.assert_allclose(out, _dry(), atol=1e-4)  # back at the clean vocal's level
    assert s.kept == 2
    mono = DereverbConfig(model="m-clean", source="clean")
    dereverb_songs(root, config=mono, backend=(stub, {}))
    assert stub.shapes[-1] == (len(wet),)
    info = json.loads((sdir / "analysis.json").read_text(encoding="utf-8"))["dereverb"]
    assert info["source"] == "clean/vocals.wav"


def test_target_stem_takes_either_output_layout():
    from signalml.restore.roformer import target_stem

    bs = np.arange(2 * 3 * 2 * 10, dtype=np.float32).reshape(2, 3, 2, 10)  # (B, stems, C, T)
    np.testing.assert_array_equal(target_stem(bs, 1, 10), bs[:, 1])
    mel = np.ones((2, 2, 8), dtype=np.float32)  # (B, C, T), iSTFT came back 2 short
    out = target_stem(mel, 0, 10)
    assert out.shape == (2, 2, 10) and np.all(out[..., 8:] == 0)
