"""Guards for the PowerShell scripts in ``scripts/``.

These run on machines we cannot test from here (the rig, a fresh work PC), so the
failure mode is a script that does not even parse — which is exactly what happened:
``bootstrap_rig.ps1`` shipped with an em dash inside a double-quoted string, and
Windows PowerShell 5.1 reads a .ps1 with no BOM as cp1252, where the dash's third
byte (0x94) is a closing smart quote. The string ended early and the whole file
became a syntax error. ASCII + CRLF sidesteps both halves of that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.ps1"))


def test_there_are_scripts_to_check():
    assert SCRIPTS, "scripts/*.ps1 vanished — update this guard"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_powershell_scripts_are_ascii(script: Path):
    raw = script.read_bytes()
    offenders = sorted({b for b in raw if b > 0x7F})
    assert not offenders, (
        f"{script.name} has non-ASCII bytes {[hex(b) for b in offenders]}. "
        f"PowerShell 5.1 decodes a BOM-less .ps1 as cp1252, where curly quotes and "
        f"dashes turn into string delimiters. Use plain ASCII."
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_powershell_scripts_use_crlf(script: Path):
    raw = script.read_bytes()
    lone_lf = raw.replace(b"\r\n", b"").count(b"\n")
    assert lone_lf == 0, (
        f"{script.name} has {lone_lf} LF-only line ending(s); PowerShell 5.1 "
        f"mis-parses here-strings in LF-only files"
    )
