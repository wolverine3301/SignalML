from __future__ import annotations

import pytest

from signalml.cli import STAGES, main


def test_stub_stage_reports_migration_phase(capsys):
    rc = main(["train"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "not implemented" in out
    assert "P7" in out


def test_align_cli_idle(tmp_path, capsys):
    rc = main(["align", "--data-root", str(tmp_path)])
    assert rc == 0
    assert "0 aligned" in capsys.readouterr().out


def test_manifest_report_cli(tmp_path, make_wav, capsys):
    make_wav(tmp_path / "raw" / "song.wav")
    (tmp_path / "raw" / "song.txt").write_text("words", encoding="utf-8")
    main(["manifest", "scan", "--data-root", str(tmp_path), "--language", "en",
          "--gender", "F", "--singer", "alice", "--source-quality", "studio"])
    rc = main(["manifest", "report", "--data-root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "distinct singers: 1" in out
    assert "studio: 1 song(s)" in out
    assert "lyrics coverage: 1/1" in out


def test_clean_cli_idle(tmp_path, capsys):
    rc = main(["clean", "--data-root", str(tmp_path)])
    assert rc == 0
    assert "0 cleaned" in capsys.readouterr().out


def test_separate_cli_idle_without_demucs(tmp_path, capsys):
    # no manifest records -> no work -> must return 0 WITHOUT importing demucs/torch
    rc = main(["separate", "--data-root", str(tmp_path)])
    assert rc == 0
    assert "0 separated" in capsys.readouterr().out


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
