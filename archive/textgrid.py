# -*- coding: utf-8 -*-
"""MAUS-style TextGrid parsing (legacy, dies in Migration P5).

Known bugs preserved for legacy parity — do not fix in place, replace with praatio in
P5 (docs/CODE_SURVEY.md): the ``tier = +1`` below *assigns* 1 rather than incrementing,
and the ``pop(0)`` heuristic can silently misalign.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhoneAlignment:
    phones: list[str]
    starts: list[float]
    ends: list[float]


def find_textgrid_pair(sample_dir: str | Path) -> tuple[Path, Path]:
    """Finds the .wav and .TextGrid in a folder."""
    sample_dir = Path(sample_dir)
    wav = None
    tg = None
    for f in sample_dir.iterdir():
        if f.suffix.lower() == ".wav":
            wav = f
        elif f.suffix.lower() == ".textgrid":
            tg = f
    if wav is None or tg is None:
        raise FileNotFoundError(f"Missing wav/textgrid in {sample_dir}")
    return wav, tg


def parse_textgrid_phones(path: str | Path) -> PhoneAlignment:
    """Parses the phoneme tier from a MAUS-style TextGrid."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    tier = -1  # 0 = word tier, 1 = phone tier
    phones: list[str] = []
    starts: list[float] = []
    ends: list[float] = []

    for line in lines:
        parts = [p.strip('"') for p in line.split()]
        if len(parts) == 3:
            if parts[2] == "ORT-MAU":
                tier += 1
            elif parts[2] == "MAU":
                tier = +1  # legacy bug kept: assigns 1, does not increment

        if tier == 1 and len(parts) == 3:
            key, _, val = parts
            if key == "xmin":
                starts.append(float(val))
            elif key == "xmax":
                ends.append(float(val))
            elif key == "text":
                phones.append(val)

    # Original popped first entries for min/max
    if starts:
        starts.pop(0)
    if ends:
        ends.pop(0)

    return PhoneAlignment(phones=phones, starts=starts, ends=ends)
