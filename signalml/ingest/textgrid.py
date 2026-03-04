# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:17:11 2026

@author: Owner
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class PhoneAlignment:
    phones: List[str]
    starts: List[float]
    ends: List[float]


def find_textgrid_pair(sample_dir: str | Path) -> Tuple[Path, Path]:
    """
    Finds the .wav and .TextGrid in a folder.
    Replacement for loadFilePairs(sampleN) but without hard-coded 'dataset/'.
    """
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
    """
    Parses phoneme tier from a TextGrid produced by MAUS-style alignments.
    Mirrors your mapPhoneElements logic but returns a dataclass.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    tier = -1  # 0 = word tier, 1 = phone tier (based on your counter)
    phones: List[str] = []
    starts: List[float] = []
    ends: List[float] = []

    for line in lines:
        parts = [p.strip('"') for p in line.split()]
        if len(parts) == 3:
            if parts[2] == "ORT-MAU":
                tier += 1
            elif parts[2] == "MAU":
                tier = +1  # keeping your behavior, but note: this sets tier to 1 always

        if tier == 1 and len(parts) == 3:
            key, _, val = parts
            if key == "xmin":
                starts.append(float(val))
            elif key == "xmax":
                ends.append(float(val))
            elif key == "text":
                phones.append(val)

    # Your original popped first entries for min/max
    if starts:
        starts.pop(0)
    if ends:
        ends.pop(0)

    return PhoneAlignment(phones=phones, starts=starts, ends=ends)