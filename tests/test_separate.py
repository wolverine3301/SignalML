"""S3 separate contract tests — offline, fake separator injected (no torch/demucs)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from signalml.manifest import Manifest, scan_directory
from signalml.stages.separate import (
    STEM_NAMES,
    SeparateConfig,
    load_separate_config,
    separate,
)

SR = 22050


@pytest.fixture()
def data_root(tmp_path, make_wav):
    root = tmp_path / "dr"
    make_wav(root / "raw" / "song1.wav", seconds=0.5)
    make_wav(root / "raw" / "song2.wav", seconds=0.5, hz=330)
    manifest, _ = scan_directory(root)
    manifest.save()
    return root


class FakeSeparator:
    def __init__(self, stems=STEM_NAMES):
        self.stem_names = stems
        self.calls = 0
        self.factory_builds = 0

    def factory(self):
        self.factory_builds += 1
        return self

    def __call__(self, path):
        self.calls += 1
        t = int(0.5 * SR)
        stems = {name: np.zeros((2, t), dtype=np.float32) for name in self.stem_names}
        info = {"model": "fake", "device": "cpu", "sample_rate": SR}
        return stems, SR, info


def test_separate_end_to_end(data_root):
    fake = FakeSeparator()
    summary = separate(data_root, separator_factory=fake.factory)

    assert sorted(summary.separated) == ["sng_0001", "sng_0002"]
    assert not summary.failed and not summary.skipped

    for rid in summary.separated:
        stems_dir = data_root / "songs" / rid / "stems"
        for name in STEM_NAMES:
            wav = stems_dir / f"{name}.wav"
            assert wav.exists()
            info = sf.info(str(wav))
            assert info.samplerate == SR
            assert info.channels == 2  # (C,T) transposed to (T,C) on write

        analysis = json.loads((data_root / "songs" / rid / "analysis.json").read_text())
        assert analysis["separate"]["model"] == "fake"
        assert analysis["separate"]["date"]

    manifest = Manifest.for_data_root(data_root)
    assert all(r.status.separated for r in manifest.records)


def test_second_run_skips_and_never_builds_separator(data_root):
    fake = FakeSeparator()
    separate(data_root, separator_factory=fake.factory)
    summary2 = separate(data_root, separator_factory=fake.factory)

    assert summary2.separated == []
    assert sorted(summary2.skipped) == ["sng_0001", "sng_0002"]
    assert fake.factory_builds == 1  # idle run built nothing


def test_force_reprocesses(data_root):
    fake = FakeSeparator()
    separate(data_root, separator_factory=fake.factory)
    summary = separate(data_root, separator_factory=fake.factory, force=True)
    assert sorted(summary.separated) == ["sng_0001", "sng_0002"]
    assert fake.calls == 4


def test_limit(data_root):
    fake = FakeSeparator()
    summary = separate(data_root, separator_factory=fake.factory, limit=1)
    assert len(summary.separated) == 1
    assert fake.calls == 1


def test_missing_stem_fails_that_song_only(data_root):
    bad = FakeSeparator(stems=("drums", "bass", "other"))  # no vocals
    summary = separate(data_root, separator_factory=bad.factory)
    assert len(summary.failed) == 2
    assert all("vocals" in err for err in summary.failed.values())
    manifest = Manifest.for_data_root(data_root)
    assert not any(r.status.separated for r in manifest.records)


def test_missing_source_file_recorded(data_root):
    (data_root / "raw" / "song1.wav").unlink()
    fake = FakeSeparator()
    summary = separate(data_root, separator_factory=fake.factory)
    assert "sng_0001" in summary.failed
    assert "sng_0002" in summary.separated
    assert fake.calls == 1  # only the existing song reached the separator


def test_load_separate_config_defaults_and_file(tmp_path):
    assert load_separate_config(tmp_path / "missing.yaml") == SeparateConfig()
    custom = tmp_path / "sep.yaml"
    custom.write_text("model: htdemucs\ndevice: cpu\n", encoding="utf-8")
    cfg = load_separate_config(custom)
    assert cfg.model == "htdemucs"
    assert cfg.device == "cpu"
    assert cfg.shifts == 1  # default preserved
