"""signalml CLI — stage entrypoints land phase by phase (docs/MIGRATION_PLAN.md).

Implemented: ``manifest scan`` and ``acquire`` (P1). Everything else is a stub that
reports which migration phase it lands in.
"""

from __future__ import annotations

import argparse
import sys

# stub stage name -> (migration phase, one-line description)
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

_IMPLEMENTED = {"manifest", "acquire", "separate", "clean", "features"}


def _cmd_manifest_scan(args: argparse.Namespace) -> int:
    from .manifest import resolve_data_root, scan_directory

    data_root = resolve_data_root(args.data_root)
    manifest, new_records = scan_directory(
        data_root,
        subpath=args.path,
        language=args.language,
        gender=args.gender,
        singer=args.singer,
    )
    manifest.save()
    print(f"Scanned {data_root / args.path}: {len(new_records)} new record(s), "
          f"{len(manifest)} total in {manifest.path}")
    missing_lyrics = [r.id for r in new_records if not r.meta.has_lyrics]
    if missing_lyrics:
        print(f"WARNING: {len(missing_lyrics)} new record(s) without a lyrics .txt sidecar: "
              f"{', '.join(missing_lyrics[:10])}{' ...' if len(missing_lyrics) > 10 else ''}")
    return 0


def _cmd_acquire(args: argparse.Namespace) -> int:
    from .manifest import resolve_data_root
    from .stages.acquire import acquire, read_url_list

    urls = list(args.url)
    if args.urls:
        urls.extend(read_url_list(args.urls))
    if not urls:
        print("acquire: no URLs given (pass URLs or --urls FILE)", file=sys.stderr)
        return 1

    summary = acquire(urls, resolve_data_root(args.data_root), language=args.language)
    print(f"acquire: {len(summary.added)} added, "
          f"{len(summary.skipped_known_url) + len(summary.skipped_known_checksum)} skipped, "
          f"{len(summary.failed)} failed")
    for url, err in summary.failed.items():
        print(f"  FAILED {url}: {err}", file=sys.stderr)
    return 0 if not summary.failed else 1


def _cmd_separate(args: argparse.Namespace) -> int:
    from .manifest import resolve_data_root
    from .stages.separate import load_separate_config, separate

    cfg = load_separate_config(args.config)
    if args.model:
        cfg = cfg.model_copy(update={"model": args.model})
    if args.device:
        cfg = cfg.model_copy(update={"device": args.device})

    summary = separate(
        resolve_data_root(args.data_root), cfg=cfg, force=args.force, limit=args.limit
    )
    print(f"separate: {len(summary.separated)} separated, "
          f"{len(summary.skipped)} already done, {len(summary.failed)} failed")
    for rid, err in summary.failed.items():
        print(f"  FAILED {rid}: {err}", file=sys.stderr)
    return 0 if not summary.failed else 1


def _cmd_clean(args: argparse.Namespace) -> int:
    from .config import active_profile
    from .manifest import resolve_data_root
    from .stages.clean import clean, load_clean_config

    summary = clean(
        resolve_data_root(args.data_root),
        cfg=load_clean_config(args.config),
        profile=active_profile(args.profile),
        force=args.force,
        limit=args.limit,
    )
    print(f"clean: {len(summary.cleaned)} cleaned, "
          f"{len(summary.skipped)} skipped, {len(summary.failed)} failed")
    for rid, err in summary.failed.items():
        print(f"  FAILED {rid}: {err}", file=sys.stderr)
    return 0 if not summary.failed else 1


def _cmd_features(args: argparse.Namespace) -> int:
    from .config import active_profile
    from .manifest import resolve_data_root
    from .stages.features import features, load_features_config

    cfg = load_features_config(args.config)
    if args.f0_method:
        cfg = cfg.model_copy(update={"f0_method": args.f0_method})

    summary = features(
        resolve_data_root(args.data_root),
        cfg=cfg,
        profile=active_profile(args.profile),
        force=args.force,
        limit=args.limit,
    )
    print(f"features: {len(summary.featurized)} featurized, "
          f"{len(summary.skipped)} skipped, {len(summary.failed)} failed")
    for rid, err in summary.failed.items():
        print(f"  FAILED {rid}: {err}", file=sys.stderr)
    return 0 if not summary.failed else 1


def _add_stub(subparsers: argparse._SubParsersAction, name: str) -> None:
    phase, desc = STAGES[name]
    p = subparsers.add_parser(name, help=f"[{phase}] {desc}")
    p.set_defaults(func=lambda _args, _n=name, _p=phase, _d=desc: _stub(_n, _p, _d))


def _stub(name: str, phase: str, desc: str) -> int:
    print(f"signalml {name}: not implemented yet ({desc}). "
          f"Lands in Migration Plan {phase} - see docs/MIGRATION_PLAN.md.")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="signalml",
        description="Parametric singing/audio synthesis pipeline.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    # manifest scan
    manifest_p = subparsers.add_parser("manifest", help="[P1] manifest utilities")
    manifest_sub = manifest_p.add_subparsers(dest="command", required=True)
    scan_p = manifest_sub.add_parser(
        "scan", help="backfill manifest records for audio already under DATA_ROOT"
    )
    scan_p.add_argument("--data-root", default=None,
                        help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    scan_p.add_argument("--path", default="raw", help="subpath to scan (default: raw)")
    scan_p.add_argument("--language", default=None, help="tag new records, e.g. en/ga/gd (Q13)")
    scan_p.add_argument("--gender", default=None, choices=["F", "M"], help="tag new records")
    scan_p.add_argument("--singer", default=None, help="tag new records")
    scan_p.set_defaults(func=_cmd_manifest_scan)

    # acquire
    acquire_p = subparsers.add_parser("acquire", help="[P1] download audio via yt-dlp")
    acquire_p.add_argument("url", nargs="*", help="URLs to download")
    acquire_p.add_argument("--urls", default=None, help="file with one URL per line")
    acquire_p.add_argument("--data-root", default=None,
                           help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    acquire_p.add_argument("--language", default=None, help="tag new records, e.g. en/ga/gd")
    acquire_p.set_defaults(func=_cmd_acquire)

    # separate
    separate_p = subparsers.add_parser("separate", help="[P2] Demucs stem separation")
    separate_p.add_argument("--data-root", default=None,
                            help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    separate_p.add_argument("--config", default=None, help="separate.yaml override path")
    separate_p.add_argument("--model", default=None, help="demucs model (default from config)")
    separate_p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    separate_p.add_argument("--force", action="store_true", help="re-separate finished songs")
    separate_p.add_argument("--limit", type=int, default=None, help="max songs this run")
    separate_p.set_defaults(func=_cmd_separate)

    # clean
    clean_p = subparsers.add_parser("clean", help="[P3] resample/normalize/filter stems")
    clean_p.add_argument("--data-root", default=None,
                         help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    clean_p.add_argument("--config", default=None, help="clean.yaml override path")
    clean_p.add_argument("--profile", default=None,
                         help="audio profile (default: $SIGNALML_AUDIO_PROFILE or yaml default)")
    clean_p.add_argument("--force", action="store_true", help="re-clean finished songs")
    clean_p.add_argument("--limit", type=int, default=None, help="max songs this run")
    clean_p.set_defaults(func=_cmd_clean)

    # features
    features_p = subparsers.add_parser("features", help="[P4] mel/F0/BPM/key extraction")
    features_p.add_argument("--data-root", default=None,
                            help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    features_p.add_argument("--config", default=None, help="features.yaml override path")
    features_p.add_argument("--profile", default=None,
                            help="audio profile (default: $SIGNALML_AUDIO_PROFILE or yaml default)")
    features_p.add_argument("--f0-method", default=None,
                            choices=["pyin", "torchcrepe", "rmvpe"])
    features_p.add_argument("--force", action="store_true", help="re-extract finished songs")
    features_p.add_argument("--limit", type=int, default=None, help="max songs this run")
    features_p.set_defaults(func=_cmd_features)

    for name in sorted(STAGES):
        if name not in _IMPLEMENTED:
            _add_stub(subparsers, name)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
