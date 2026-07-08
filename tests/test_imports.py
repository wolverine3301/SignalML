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
    "signalml.ingest",
    "signalml.ingest.textgrid",
    "signalml.ingest.phonemes",
    "signalml.pipeline",
    "signalml.pipeline.features",
    "signalml.pipeline.phoneme_features",
    "signalml.pipeline.phoneme_jobs",
    "signalml.stages",
    "signalml.score",
    "signalml.voices",
    "signalml.train",
    "signalml.synth",
    "signalml.tasks",
    "signalml.tasks.masking",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
