from __future__ import annotations

import pytest

from signalml.cli import STAGES, main


def test_stub_stage_reports_migration_phase(capsys):
    rc = main(["separate"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "not implemented" in out
    assert "P2" in out


def test_unknown_stage_rejected():
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_stage_table_covers_pipeline():
    assert {"acquire", "separate", "clean", "align", "features", "train", "sing"} <= set(STAGES)


def test_manifest_scan_cli(tmp_path, make_wav, capsys):
    make_wav(tmp_path / "raw" / "song.wav")
    (tmp_path / "raw" / "song.txt").write_text("words", encoding="utf-8")

    rc = main(["manifest", "scan", "--data-root", str(tmp_path), "--language", "en"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 new record(s)" in out
    assert (tmp_path / "manifest.jsonl").exists()
    assert "WARNING" not in out  # lyrics sidecar present


def test_manifest_scan_cli_warns_on_missing_lyrics(tmp_path, make_wav, capsys):
    make_wav(tmp_path / "raw" / "nolyrics.wav")
    rc = main(["manifest", "scan", "--data-root", str(tmp_path)])
    assert rc == 0
    assert "WARNING" in capsys.readouterr().out


def test_acquire_cli_requires_urls(tmp_path, capsys):
    rc = main(["acquire", "--data-root", str(tmp_path)])
    assert rc == 1
    assert "no URLs" in capsys.readouterr().err
