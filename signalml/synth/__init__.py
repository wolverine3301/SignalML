"""S8 inference: score.json + voice profile -> vocal WAV (+ mixdown).

Contract: docs/PIPELINE_AND_CONTRACTS.md §S8. Implemented in Migration P8.

What exists ahead of P8 is :mod:`signalml.synth.render` — render addressing, provenance
and the segment cache (D12). It is model-free on purpose: the unit S8 renders is a
*segment*, not a song, so targeted re-rendering stays possible whatever method P8 ends
up using.
"""

from .render import (
    RENDER_KEY_VERSION,
    RENDER_RECORD_FORMAT,
    VARIANCE_FORMAT,
    RenderInputs,
    RenderRecord,
    VarianceTrack,
    artifact_path,
    find_cached,
    iter_records,
    load_record,
    load_variance,
    new_record,
    render_dir,
    renders_root,
    save_record,
    save_variance,
)

__all__ = [
    "RENDER_KEY_VERSION",
    "RENDER_RECORD_FORMAT",
    "VARIANCE_FORMAT",
    "RenderInputs",
    "RenderRecord",
    "VarianceTrack",
    "artifact_path",
    "find_cached",
    "iter_records",
    "load_record",
    "load_variance",
    "new_record",
    "render_dir",
    "renders_root",
    "save_record",
    "save_variance",
]
