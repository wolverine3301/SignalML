# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:34:10 2026

@author: Owner
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    mono: bool = True


@dataclass(frozen=True)
class SpectrogramConfig:
    n_fft: int = 2048
    hop_length: int = 256
    win_length: int = 2048
    n_mels: int = 256