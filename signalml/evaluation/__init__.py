"""Evaluation harnesses. Pure measurement — no stage writes, no manifest mutation.

``alignment`` implements the P5.4 aligner comparison (docs/notes/aligner_eval.md):
MFA vs SOFA against external ground truth, plus the separation-drift ablation and the
``quality.align_score`` calibration. Metrics only; the aligners themselves are run by
S5 (``signalml align``).
"""

from .alignment import (
    BoundaryMetrics,
    OnsetMetrics,
    ScoreCalibration,
    WordOnset,
    boundary_metrics,
    calibrate_align_score,
    onset_metrics,
    phone_boundaries,
    spearman,
    word_onsets,
)

__all__ = [
    "BoundaryMetrics",
    "OnsetMetrics",
    "ScoreCalibration",
    "WordOnset",
    "boundary_metrics",
    "calibrate_align_score",
    "onset_metrics",
    "phone_boundaries",
    "spearman",
    "word_onsets",
]
