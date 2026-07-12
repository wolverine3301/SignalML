"""Smoke: every package module imports (catches broken restructure wiring)."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "signalml",
    "signalml.config",
    "signalml.cli",
    "signalml.audio",
    "signalml.audio.io",
    "signalml.audio.segment",
    "signalml.pipeline",
    "signalml.pipeline.features",
    "signalml.stages",
    "signalml.stages.align",
    "signalml.score",
    "signalml.score.schema",
    "signalml.score.phoneset",
    "signalml.score.g2p",
    "signalml.score.from_midi",
    "signalml.score.from_musicxml",
    "signalml.voices",
    "signalml.train",
    "signalml.synth",
    "signalml.tasks",
    "signalml.tasks.masking",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
