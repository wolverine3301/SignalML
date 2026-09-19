"""Canonical JSON hashing — the determinism contract behind every cache key.

Segment content hashes (``score/segment.py``) and render cache keys (``synth/render.py``)
must agree byte-for-byte across machines, Python versions, and runs, or the render cache
silently serves the wrong audio. Both go through here so there is exactly one place where
"how do we turn a dict into bytes" is decided.

Rules (changing any of them is a cache-invalidating format change, so bump the caller's
``*_VERSION`` constant if you do): keys sorted, no insignificant whitespace, non-ASCII
(IPA!) written through as UTF-8 rather than escaped, floats rounded by the
*caller* before they get here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic UTF-8 serialization of ``obj``."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    """Hex sha256 of :func:`canonical_json_bytes`."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
