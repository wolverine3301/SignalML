"""Legacy feature pipeline — absorbed into ``stages/features.py`` in Migration P4.

Mixing/masking moved to ``signalml.tasks.masking`` (Q12).
"""

from .features import mels_from_chunks, stft_from_chunks

__all__ = ["mels_from_chunks", "stft_from_chunks"]
