"""``signalml doctor`` contract tests — offline, and deliberately environment-agnostic.

We assert on the checks whose verdict this machine controls (data root, disk, report
formatting, exit semantics). Torch/CUDA verdicts depend on the box the suite runs on —
a laptop with CPU wheels *should* report FAIL there — so those are not pinned.
"""

from __future__ import annotations

from signalml.doctor import (
    Check,
    check_data_root,
    check_python,
    format_report,
    has_failures,
    run_checks,
)


def test_python_check_passes_on_a_supported_interpreter():
    assert check_python().status == "ok"


def test_missing_data_root_fails_with_a_fix(tmp_path):
    checks = check_data_root(tmp_path / "nope")
    assert [c.status for c in checks] == ["fail"]
    assert "SIGNALML_DATA_ROOT" in checks[0].fix


def test_data_root_reports_manifest_and_space(tmp_path, make_wav):
    from signalml.manifest import scan_directory

    make_wav(tmp_path / "raw" / "song.wav")
    manifest, _ = scan_directory(tmp_path, language="en", gender="F", singer="alice")
    manifest.save()
    by_name = {c.name: c for c in check_data_root(tmp_path)}
    assert by_name["DATA_ROOT"].status == "ok"
    assert "1 record(s)" in by_name["manifest"].detail
    assert by_name["disk space"].status == "ok"


def test_shipment_larger_than_the_disk_fails(tmp_path):
    by_name = {c.name: c for c in check_data_root(tmp_path, need_bytes=1 << 60)}
    assert by_name["disk space"].status == "fail"
    assert "needs" in by_name["disk space"].detail


def test_report_lists_fixes_only_for_problems():
    checks = [
        Check("fine", "ok", "all good", "unused fix"),
        Check("broken", "fail", "it is broken", "do the thing"),
    ]
    report = format_report(checks)
    assert "do the thing" in report
    assert "unused fix" not in report
    assert "NOT ready" in report
    assert has_failures(checks)


def test_report_is_green_when_only_warnings():
    checks = [Check("fine", "ok", "x"), Check("meh", "warn", "y", "optional fix")]
    assert not has_failures(checks)
    assert "Ready." in format_report(checks)


def test_run_checks_covers_the_documented_surface(tmp_path):
    names = {c.name for c in run_checks(data_root=tmp_path)}
    for expected in ("python", "uv", "git", "torch", "audio profile", "DATA_ROOT",
                     "disk space", "DiffSinger submodule"):
        assert expected in names
