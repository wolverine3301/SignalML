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
    "score": ("P6", "score JSON tools: validate / from-midi / phoneset"),
    "features": ("P4", "mel/F0/BPM/key feature extraction"),
    "dataset": ("P7", "build binarized training datasets from the manifest"),
    "train": ("P7", "train acoustic/variance/vocoder models"),
    "voice": ("P7", "voice bank: sample/validate/persist voice profiles"),
    "sing": ("P8", "score.json + voice profile -> vocal WAV (+ mixdown)"),
    "render": ("P9", "render MIDI backing tracks (symbolic-first instrumental)"),
}

_IMPLEMENTED = {"manifest", "acquire", "separate", "clean", "features", "align", "score",
                "dataset"}


def _cmd_manifest_scan(args: argparse.Namespace) -> int:
    from .manifest import resolve_data_root, scan_directory

    data_root = resolve_data_root(args.data_root)
    manifest, new_records = scan_directory(
        data_root,
        subpath=args.path,
        language=args.language,
        gender=args.gender,
        singer=args.singer,
        source_quality=args.source_quality,
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


def _cmd_manifest_import_stems(args: argparse.Namespace) -> int:
    from .manifest import import_stem_folders, resolve_data_root

    manifest, new_records, skipped = import_stem_folders(
        resolve_data_root(args.data_root),
        subpath=args.path,
        language=args.language,
        gender=args.gender,
    )
    manifest.save()
    dupes = sum(1 for r in skipped.values() if "duplicate" in r)
    print(f"import-stems: {len(new_records)} imported, {len(skipped)} skipped "
          f"({dupes} duplicates of existing records)")
    for folder, reason in sorted(skipped.items()):
        if "duplicate" not in reason:  # duplicates are expected; keep the noise down
            print(f"  SKIPPED {folder}: {reason}", file=sys.stderr)
    missing_singer = [r.id for r in new_records if not r.meta.singer]
    if missing_singer:
        print(f"WARNING: {len(missing_singer)} imported record(s) without SINGER in "
              f"META.txt: {', '.join(missing_singer[:10])}"
              f"{' ...' if len(missing_singer) > 10 else ''}")
    return 0


def _cmd_manifest_retag(args: argparse.Namespace) -> int:
    from .manifest import RETAG_SAFE_FIELDS, resolve_data_root, retag_from_sidecars

    fields = tuple(f.strip() for f in args.fields.split(",")) if args.fields \
        else RETAG_SAFE_FIELDS
    manifest, changes, warnings = retag_from_sidecars(
        resolve_data_root(args.data_root), fields=fields)
    manifest.save()
    for rid, field, old, new in changes:
        print(f"  {rid}: {field} {old!r} -> {new!r}")
    for rid, msg in warnings:
        print(f"  WARNING {rid}: {msg}", file=sys.stderr)
    print(f"retag: {len(changes)} change(s) across fields {list(fields)}, "
          f"{len(warnings)} warning(s)")
    return 0


def _cmd_manifest_report(args: argparse.Namespace) -> int:
    from .manifest import Manifest, manifest_report, resolve_data_root

    manifest = Manifest.for_data_root(resolve_data_root(args.data_root))
    print(manifest_report(manifest))
    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    from .manifest import resolve_data_root
    from .stages.align import align, load_align_config

    summary = align(
        resolve_data_root(args.data_root),
        cfg=load_align_config(args.config),
        force=args.force,
        limit=args.limit,
    )
    print(f"align: {len(summary.aligned)} aligned, "
          f"{len(summary.skipped)} skipped, {len(summary.failed)} failed")
    for rid, err in summary.failed.items():
        print(f"  FAILED {rid}: {err}", file=sys.stderr)
    return 0 if not summary.failed else 1


def _cmd_score_validate(args: argparse.Namespace) -> int:
    from .score import validate_score_file

    bad = 0
    for path in args.score:
        problems = validate_score_file(path)
        if problems:
            bad += 1
            print(f"INVALID {path}:")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"ok {path}")
    return 0 if not bad else 1


def _cmd_score_from_midi(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .score import ChainG2P, EspeakG2P, LexiconG2P, MfaG2P, save_score, score_from_midi

    backends = []
    if args.lexicon:
        backends.append(LexiconG2P(path=args.lexicon))
    if args.g2p == "mfa":
        backends.append(MfaG2P(args.g2p_model, phone_set=args.phone_set))
    elif args.g2p == "espeak":
        backends.append(EspeakG2P(phone_set=args.phone_set))
    if not backends:
        print("score from-midi: need --lexicon and/or --g2p mfa|espeak", file=sys.stderr)
        return 1

    score = score_from_midi(
        args.midi,
        Path(args.lyrics).read_text(encoding="utf-8"),
        g2p=ChainG2P(*backends),
        language=args.language,
        phone_set=args.phone_set,
        track=args.track,
    )
    out = Path(args.out) if args.out else Path(args.midi).with_name("score.json")
    save_score(score, out)
    print(f"wrote {out} ({len(score.notes)} notes, bpm {score.bpm}, key {score.key})")
    return 0


def _cmd_score_phoneset(args: argparse.Namespace) -> int:
    from .score.phoneset import diff_against_mfa_dictionary, get_phone_set

    ps = get_phone_set(args.name)
    if not args.dict:
        print(f"{ps.name} ({ps.language}): {len(ps.phones)} phones")
        print(" ".join(sorted(ps.phones)))
        print(f"note: {ps.notes}")
        return 0
    diff = diff_against_mfa_dictionary(ps, args.dict)
    print(f"dictionary phones: {len(diff.dictionary_phones)}")
    if diff.missing_from_set:
        print(f"MISSING from {ps.name} (aligner output would be rejected!): "
              f"{' '.join(diff.missing_from_set)}")
    if diff.unused_by_dict:
        print(f"in {ps.name} but unused by this dictionary: {' '.join(diff.unused_by_dict)}")
    print("clean" if diff.clean else "MISMATCH — update phoneset.py and bump the version")
    return 0 if diff.clean else 1


def _cmd_dataset_build(args: argparse.Namespace) -> int:
    from .manifest import resolve_data_root
    from .stages.dataset import build, load_dataset_recipe

    summary = build(
        resolve_data_root(args.data_root),
        recipe=load_dataset_recipe(args.recipe),
        force=args.force,
    )
    print(f"dataset build: {summary.clips} clip(s) / {summary.seconds / 3600:.2f} h "
          f"from {len(summary.songs_used)} song(s) -> {summary.out_dir}")
    if summary.dropped_clips:
        print(f"  {summary.dropped_clips} clip(s) dropped (noise/too short)")
    for rid, reason in sorted(summary.skipped.items()):
        print(f"  SKIPPED {rid}: {reason}", file=sys.stderr)
    return 0 if summary.clips else 1


def _cmd_dash(args: argparse.Namespace) -> int:
    from .dash.server import serve
    from .manifest import resolve_data_root

    serve(resolve_data_root(args.data_root), host=args.host, port=args.port,
          open_browser=not args.no_open)
    return 0


def _add_stub(subparsers: argparse._SubParsersAction, name: str) -> None:
    phase, desc = STAGES[name]
    p = subparsers.add_parser(name, help=f"[{phase}] {desc}")
    p.set_defaults(func=lambda _args, _n=name, _p=phase, _d=desc: _stub(_n, _p, _d))


def _stub(name: str, phase: str, desc: str) -> int:
    print(f"signalml {name}: not implemented yet ({desc}). "
          f"Lands in Migration Plan {phase} - see docs/MIGRATION_PLAN.md.")
    return 2


def main(argv: list[str] | None = None) -> int:
    # IPA phones must survive Windows' legacy cp1252 console (score/phoneset output)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
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
    scan_p.add_argument("--source-quality", default=None, choices=["studio", "separated"],
                        help="tag new records: studio stems vs to-be-Demucs'd mixes")
    scan_p.set_defaults(func=_cmd_manifest_scan)
    report_p = manifest_sub.add_parser(
        "report", help="corpus census: singers/hours/languages/lyrics coverage/status"
    )
    report_p.add_argument("--data-root", default=None,
                          help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    report_p.set_defaults(func=_cmd_manifest_report)
    import_p = manifest_sub.add_parser(
        "import-stems",
        help="onboard pre-separated song folders (vocals.wav) without Demucs",
    )
    import_p.add_argument("--data-root", default=None,
                          help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    import_p.add_argument("--path", required=True,
                          help="subpath with one-song-per-folder stems, e.g. RAW/legacy_stems")
    import_p.add_argument("--language", default=None, help="tag new records")
    import_p.add_argument("--gender", default=None, choices=["F", "M"])
    import_p.set_defaults(func=_cmd_manifest_import_stems)
    retag_p = manifest_sub.add_parser(
        "retag", help="refresh tag fields on existing records from META.txt sidecars"
    )
    retag_p.add_argument("--data-root", default=None,
                         help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    retag_p.add_argument("--fields", default=None,
                         help="comma list (default: processing,domain,genre — singer/"
                              "song excluded so manifest fixes aren't clobbered)")
    retag_p.set_defaults(func=_cmd_manifest_retag)

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

    # align
    align_p = subparsers.add_parser("align", help="[P5] MFA alignment -> align/phones.json")
    align_p.add_argument("--data-root", default=None,
                         help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    align_p.add_argument("--config", default=None, help="align.yaml override path")
    align_p.add_argument("--force", action="store_true", help="re-align finished songs")
    align_p.add_argument("--limit", type=int, default=None, help="max songs this run")
    align_p.set_defaults(func=_cmd_align)

    # score
    score_p = subparsers.add_parser("score", help="[P6] score JSON tools")
    score_sub = score_p.add_subparsers(dest="command", required=True)
    validate_p = score_sub.add_parser("validate", help="validate score.json files")
    validate_p.add_argument("score", nargs="+", help="score.json path(s)")
    validate_p.set_defaults(func=_cmd_score_validate)
    from_midi_p = score_sub.add_parser(
        "from-midi", help="MIDI + syllabified lyrics -> score.json"
    )
    from_midi_p.add_argument("midi", help="MIDI file (monophonic melody track)")
    from_midi_p.add_argument("--lyrics", required=True,
                             help="lyrics text file (hyphenate syllables; '-' = melisma)")
    from_midi_p.add_argument("--out", default=None, help="output path (default: score.json)")
    from_midi_p.add_argument("--lexicon", default=None,
                             help="JSON lexicon path (checked before --g2p backend)")
    from_midi_p.add_argument("--g2p", default="mfa", choices=["mfa", "espeak", "none"],
                             help="automatic G2P backend (default: mfa)")
    from_midi_p.add_argument("--g2p-model", default="english_us_mfa",
                             help="MFA G2P model name (default: english_us_mfa)")
    from_midi_p.add_argument("--language", default="en")
    from_midi_p.add_argument("--phone-set", default="mfa_ipa/en_v1")
    from_midi_p.add_argument("--track", type=int, default=None,
                             help="melody track index (default: first with notes)")
    from_midi_p.set_defaults(func=_cmd_score_from_midi)
    phoneset_p = score_sub.add_parser(
        "phoneset", help="show a phone set / diff it against an MFA dictionary"
    )
    phoneset_p.add_argument("--name", default="mfa_ipa/en_v1")
    phoneset_p.add_argument("--dict", default=None,
                            help="installed MFA .dict file to verify against")
    phoneset_p.set_defaults(func=_cmd_score_phoneset)

    # dataset
    dataset_p = subparsers.add_parser("dataset", help="[P7] build training datasets")
    dataset_sub = dataset_p.add_subparsers(dest="command", required=True)
    build_p = dataset_sub.add_parser(
        "build", help="manifest query -> vendored-trainer raw dataset + card"
    )
    build_p.add_argument("--data-root", default=None,
                         help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    build_p.add_argument("--recipe", default=None,
                         help="recipe yaml (default: configs/dataset.yaml)")
    build_p.add_argument("--force", action="store_true",
                         help="rebuild an existing dataset directory")
    build_p.set_defaults(func=_cmd_dataset_build)

    # dash
    dash_p = subparsers.add_parser("dash", help="live pipeline monitoring dashboard")
    dash_p.add_argument("--data-root", default=None,
                        help="data root (default: $SIGNALML_DATA_ROOT or ./data)")
    dash_p.add_argument("--host", default="127.0.0.1")
    dash_p.add_argument("--port", type=int, default=8765)
    dash_p.add_argument("--no-open", action="store_true",
                        help="don't open the browser automatically")
    dash_p.set_defaults(func=_cmd_dash)

    for name in sorted(STAGES):
        if name not in _IMPLEMENTED:
            _add_stub(subparsers, name)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
