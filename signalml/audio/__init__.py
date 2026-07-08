"""Shared audio primitives: loading, saving, segmentation."""

from .io import audio_duration_sec, load_audio, load_npz_arrays, save_npz_arrays, save_wav
from .segment import ChunkResult, chunk_audio_file

__all__ = [
    "load_audio",
    "audio_duration_sec",
    "save_wav",
    "save_npz_arrays",
    "load_npz_arrays",
    "chunk_audio_file",
    "ChunkResult",
]
