from __future__ import annotations

import pytest

from signalml.cli import STAGES, main


def test_known_stage_reports_migration_phase(capsys):
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
