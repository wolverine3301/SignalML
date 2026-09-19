"""Studio — the local front-end over S8 (docs/STUDIO.md, docs/STUDIO_UI.md).

**No model code lives in this package**, by rule: every action is a call into
``signalml.voices`` / ``signalml.synth`` / ``signalml.score``, so anything done in the
UI is reproducible from the command line and the UI layer stays disposable.

What is built ahead of P8 is the model-free half — the cost bar, the score and voice
listings, and the variance read/write path — because all of it is arithmetic over
hashes that already exist (D12's pre-commitments). The renderer, the job queue and its
WebSocket land with P8; see docs/STUDIO_UI.md §13 for the build order.
"""

from .api import (
    STUDIO_API_FORMAT,
    RenderEstimate,
    RenderPlan,
    ScoreSummary,
    SegmentPlanRow,
    StudioContext,
    VoiceSummary,
    build_context,
    build_render_plan,
    config_fingerprint,
    find_voice,
    list_scores,
    list_voices,
    measure_render_rate,
    read_variance,
    resolve_checkpoint,
    write_variance,
)

__all__ = [
    "STUDIO_API_FORMAT",
    "RenderEstimate",
    "RenderPlan",
    "ScoreSummary",
    "SegmentPlanRow",
    "StudioContext",
    "VoiceSummary",
    "build_context",
    "build_render_plan",
    "config_fingerprint",
    "find_voice",
    "list_scores",
    "list_voices",
    "measure_render_rate",
    "read_variance",
    "resolve_checkpoint",
    "write_variance",
]
