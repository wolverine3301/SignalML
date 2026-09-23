from __future__ import annotations

import pytest

from signalml.cli import STAGES, main


def test_stub_stage_reports_migration_phase(capsys):
    rc = main(["sing"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "not implemented" in out
    assert "P8" in out


def test_train_dry_run_blocks_on_an_unbuilt_dataset(tmp_path, capsys):
    rc = main(["train", "acoustic", "--dataset", "nope", "--data-root",
               str(tmp_path), "--dry-run"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "BLOCKED" in err and "dataset build" in err


def test_train_variance_asks_for_its_config(tmp_path, capsys):
    """D1 landed 2026-09-22: variance is no longer refused, it needs a config."""
    rc = main(["train", "variance", "--dataset", "nope", "--data-root",
               str(tmp_path), "--dry-run"])
    assert rc == 1
    assert "config_variance.yaml" in capsys.readouterr().err


def test_train_vocoder_names_its_blocker(tmp_path, capsys):
    rc = main(["train", "vocoder", "--dataset", "nope", "--data-root", str(tmp_path)])
    assert rc == 2
    assert "SingingVocoders" in capsys.readouterr().err


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


def test_ship_plan_cli(tmp_path, make_wav, capsys):
    (tmp_path / "datasets" / "d1" / "alice-en" / "wavs").mkdir(parents=True)
    (tmp_path / "datasets" / "d1" / "alice-en" / "wavs" / "sng_0001_000.wav").write_bytes(
        b"RIFF" + b"\0" * 64)
    (tmp_path / "datasets" / "d1" / "alice-en" / "transcriptions.csv").write_text(
        "name,ph_seq,ph_dur\nsng_0001_000,SP aj SP,0.1 0.4 0.1\n", encoding="utf-8")
    make_wav(tmp_path / "raw" / "song.wav")
    main(["manifest", "scan", "--data-root", str(tmp_path), "--language", "en",
          "--gender", "F", "--singer", "alice"])
    capsys.readouterr()
    rc = main(["ship", "plan", "--data-root", str(tmp_path), "--what", "dataset",
               "--dataset-name", "d1", "--no-code"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "plan d1 (dataset)" in out
    assert "next: signalml ship serve --name d1" in out
    assert (tmp_path / "ship" / "d1" / "SHIP.json").exists()


def test_ship_plan_cli_reports_a_missing_dataset(tmp_path, make_wav, capsys):
    make_wav(tmp_path / "raw" / "song.wav")
    main(["manifest", "scan", "--data-root", str(tmp_path), "--language", "en",
          "--gender", "F", "--singer", "alice"])
    capsys.readouterr()
    rc = main(["ship", "plan", "--data-root", str(tmp_path), "--what", "dataset",
               "--dataset-name", "nope", "--no-code"])
    assert rc == 1
    assert "does not exist" in capsys.readouterr().err


def test_ship_serve_without_a_plan_is_an_error(tmp_path, capsys):
    rc = main(["ship", "serve", "--data-root", str(tmp_path), "--name", "nope"])
    assert rc == 1
    assert "ship plan" in capsys.readouterr().err


def test_ship_verify_missing_plan(tmp_path, capsys):
    rc = main(["ship", "verify", "--data-root", str(tmp_path), "--name", "nope"])
    assert rc == 1
    assert "no plan at" in capsys.readouterr().err


def test_doctor_cli_runs_and_reports(tmp_path, capsys):
    rc = main(["doctor", "--data-root", str(tmp_path)])
    assert rc in (0, 1)  # depends on the machine's torch/CUDA state
    out = capsys.readouterr().out
    assert "check(s):" in out
    assert "DATA_ROOT" in out
