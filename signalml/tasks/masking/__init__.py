"""Masking/denoising dataset factory (the legacy birds/vehicles/voice experiments).

Kept as a separate tool per OPEN_QUESTIONS Q12 — shares audio primitives with the core
pipeline but never shapes its design. Augmentation (pitch shift, mixing) lives here or
in dataset recipes only, never in ingest.
"""

from .folder import build_class_chunks, flatten_class_dict
from .masking import generate_masking_dataset
from .mixing import (
    augment_with_pitch,
    mix_aligned_subset,
    mix_average,
    mix_mel_npz_roundtrip,
    mix_random_background,
    pitch_shift_random,
)

__all__ = [
    "build_class_chunks",
    "flatten_class_dict",
    "generate_masking_dataset",
    "augment_with_pitch",
    "mix_aligned_subset",
    "mix_average",
    "mix_mel_npz_roundtrip",
    "mix_random_background",
    "pitch_shift_random",
]
