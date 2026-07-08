"""signalml CLI — stage entrypoints land phase by phase (docs/MIGRATION_PLAN.md).

Every stage takes ``--profile`` (audio profile per Q11) once implemented.
"""

from __future__ import annotations

import argparse
import sys

# stage name -> (migration phase, one-line description)
STAGES: dict[str, tuple[str, str]] = {
    "acquire": ("P1", "download audio via yt-dlp into raw/ + manifest records"),
    "manifest": ("P1", "manifest utilities (scan/backfill)"),
    "separate": ("P2", "stem separation with Demucs htdemucs_ft"),
    "clean": ("P3", "resample/loudness-normalize/filter vocal stems"),
    "align": ("P5", "MFA alignment -> align/phones.json (MFA IPA)"),
    "features": ("P4", "mel/F0/BPM/key feature extraction"),
    "dataset": ("P7", "build binarized training datasets from the manifest"),
    "train": ("P7", "train acoustic/variance/vocoder models"),
    "voice": ("P7", "voice bank: sample/validate/persist voice profiles"),
    "sing": ("P8", "score.json + voice profile -> vocal WAV (+ mixdown)"),
    "render": ("P9", "render MIDI backing tracks (symbolic-first instrumental)"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="signalml",
        description="Parametric singing/audio synthesis pipeline.",
        epilog="Stages are implemented phase by phase; see docs/MIGRATION_PLAN.md.",
    )
    parser.add_argument("stage", choices=sorted(STAGES), help="pipeline stage to run")
    args, _rest = parser.parse_known_args(argv)

    phase, desc = STAGES[args.stage]
    print(
        f"signalml {args.stage}: not implemented yet ({desc}). "
        f"Lands in Migration Plan {phase} - see docs/MIGRATION_PLAN.md."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
