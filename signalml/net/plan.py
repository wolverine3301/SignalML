"""Shipment planning — the contract half of ``signalml ship`` (docs/notes/transfer.md).

A shipment is a **plan** (``SHIP.json``: every file with its size + sha256, plus git
provenance) and a **transport** (``server.py`` / ``client.py``). The plan is the
contract: ``ship verify`` on the receiving side is the gate you run *before* starting a
multi-day training run, so a truncated wav fails in seconds instead of at hour eight.

Selection modes mirror how the data is derived, so we never move bytes the rig can
regenerate for free:

``dataset``
    ``datasets/<name>/`` only — exactly what the vendored binarizer eats.
``rebuildable``
    ``clean/vocals.wav`` + ``align/`` + ``analysis.json`` for the songs a recipe
    selects, so the rig can rebuild datasets under *new* recipes without another haul.
``full``
    the above plus ``stems/`` (and the source audio with ``--with-raw``) — only worth
    it if you intend to re-separate or re-align on the rig.

``--with-code`` adds a git bundle: the clone/update path for a rig that cannot reach
the GitHub remote (clone on first pull, fetch + checkout on every one after). A rig
with network access clones from GitHub instead and pulls data with ``--no-code`` —
either way the code lands first, since ``ship pull`` runs through signalml itself.
Planning refuses a dirty worktree unless ``--allow-dirty`` (which captures
``git diff HEAD`` alongside it): checkpoints record a git hash, so the rig must never
run code that does not correspond to a commit.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

from ..manifest import MANIFEST_NAME, Manifest, sha256_file

SHIP_DIRNAME = "ship"
PLAN_NAME = "SHIP.json"
HASHCACHE_NAME = "hashcache.json"
RECEIPTS_NAME = "RECEIPTS.json"
BUNDLE_NAME = "signalml.bundle"
PATCH_NAME = "dirty.patch"
PROVENANCE_NAME = "PROVENANCE.txt"
CODE_SUBDIR = ".ship"  # where code items land inside the receiving repo

REPO_ROOT = Path(__file__).resolve().parents[2]

What = Literal["dataset", "rebuildable", "full", "unprocessed", "code"]
Dest = Literal["data", "code"]

# clip names in transcriptions.csv are "<song_id>_<NNN>"
_CLIP_RE = re.compile(r"^(sng_\d+)_\d+$")


def item_key(dest: str, path: str) -> str:
    """Opaque, deterministic server key. The wire never carries a filesystem path, so
    the server can only ever serve files the plan already lists."""
    return hashlib.sha256(f"{dest}:{path}".encode("utf-8")).hexdigest()[:24]


class ShipItem(BaseModel):
    key: str
    dest: Dest = "data"
    path: str  # posix, relative to the destination root
    size: int
    sha256: str
    # file     = copied verbatim to <root>/<path>
    # manifest = JSONL merged (upsert by id) into the receiver's manifest
    # bundle   = git bundle, cloned/fetched into the receiver's repo
    # patch    = uncommitted diff, written next to the bundle, never auto-applied
    kind: Literal["file", "manifest", "bundle", "patch"] = "file"
    staged: bool = False  # source lives in the sender's stage dir, not under a root


class GitProvenance(BaseModel):
    commit: str
    branch: str
    dirty: bool = False
    describe: str | None = None
    submodules: list[str] = Field(default_factory=list)


class ShipPlan(BaseModel):
    version: int = 1
    name: str
    created: str
    what: What
    source_host: str
    audio_profile: str | None = None
    dataset_name: str | None = None
    song_ids: list[str] = Field(default_factory=list)
    git: GitProvenance | None = None
    items: list[ShipItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(i.size for i in self.items)

    def by_key(self, key: str) -> ShipItem | None:
        return next((i for i in self.items if i.key == key), None)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ShipPlan":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class HashCache:
    """(size, mtime_ns) -> sha256, so re-planning a 35 GB corpus is instant.

    Cheap but honest: fooling it needs a rewrite with an identical size *and* mtime,
    and every pipeline stage writes through a temp + ``os.replace``.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, list] = {}
        self._dirty = False
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def digest(self, path: Path) -> str:
        st = path.stat()
        key = str(path.resolve())
        hit = self._data.get(key)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            return hit[2]
        value = sha256_file(path)
        self._data[key] = [st.st_size, st.st_mtime_ns, value]
        self._dirty = True
        return value

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data), encoding="utf-8")
        os.replace(tmp, self.path)
        self._dirty = False


def stage_dir(data_root: str | Path, name: str) -> Path:
    return Path(data_root) / SHIP_DIRNAME / name


# --------------------------------------------------------------------------- git


def _git(repo: Path, *args: str, check: bool = True) -> str:
    out = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def git_provenance(repo: str | Path) -> GitProvenance:
    repo = Path(repo)
    return GitProvenance(
        commit=_git(repo, "rev-parse", "HEAD"),
        branch=_git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        dirty=bool(_git(repo, "status", "--porcelain")),
        describe=_git(repo, "describe", "--always", "--dirty", check=False) or None,
        submodules=[ln.strip() for ln
                    in _git(repo, "submodule", "status", check=False).splitlines()
                    if ln.strip()],
    )


def _write_code_stage(
    repo: Path, out: Path, *, allow_dirty: bool
) -> tuple[GitProvenance, list[Path]]:
    """Bundle the repo into the stage's ``code/``. Returns (provenance, files)."""
    prov = git_provenance(repo)
    if prov.dirty and not allow_dirty:
        raise RuntimeError(
            "working tree is dirty — checkpoints record a git hash, so the rig must "
            "not run uncommitted code. Commit first, or re-plan with --allow-dirty "
            "(captures `git diff HEAD` as dirty.patch, applied by hand on the rig)"
        )
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / BUNDLE_NAME
    if bundle.exists():
        bundle.unlink()
    # --all alone omits HEAD, and `git clone <bundle>` then refuses
    _git(repo, "bundle", "create", str(bundle), "--all", "HEAD")
    files = [bundle]

    if prov.dirty:
        patch = out / PATCH_NAME
        patch.write_text(_git(repo, "diff", "HEAD", check=False), encoding="utf-8")
        files.append(patch)

    lines = [
        f"commit    {prov.commit}",
        f"branch    {prov.branch}",
        f"describe  {prov.describe or '(none)'}",
        f"dirty     {prov.dirty}",
        f"shipped   {_dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"from      {socket.gethostname()}",
        "",
        "submodules (cloned from their own remotes by scripts/bootstrap_rig.ps1):",
        *[f"  {s}" for s in prov.submodules],
        "",
        "On the rig:  scripts/bootstrap_rig.ps1  ->  signalml doctor",
    ]
    prov_file = out / PROVENANCE_NAME
    prov_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.append(prov_file)
    return prov, files


# ---------------------------------------------------------------------- selection


def _iter_files(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.endswith(".part"):
            yield p


def _dataset_song_ids(dataset_dir: Path) -> list[str]:
    """Song ids that actually contributed clips, read back out of the shipped CSVs."""
    ids: set[str] = set()
    for csv_path in dataset_dir.rglob("transcriptions.csv"):
        with open(csv_path, encoding="utf-8") as fh:
            next(fh, None)  # header
            for line in fh:
                match = _CLIP_RE.match(line.split(",", 1)[0].strip().strip('"'))
                if match:
                    ids.add(match.group(1))
    return sorted(ids)


def _selected_records(data_root: Path, recipe) -> tuple[list, dict[str, str]]:
    """The exact record set ``dataset build`` would use for this recipe — shipping and
    training must never disagree about what "the corpus" means."""
    from ..stages.dataset import BuildSummary, _select

    manifest = Manifest.for_data_root(data_root)
    summary = BuildSummary(out_dir=data_root)
    return _select(manifest, recipe, summary, data_root), summary.skipped


def _write_manifest_subset(records, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")
    return out


# ------------------------------------------------------------------------- build


def build_plan(
    data_root: str | Path,
    *,
    what: What = "rebuildable",
    name: str | None = None,
    recipe=None,
    dataset_name: str | None = None,
    with_code: bool = True,
    with_features: bool = False,
    with_raw: bool = False,
    prefix: str | None = None,
    allow_dirty: bool = False,
    repo_root: str | Path = REPO_ROOT,
    progress: bool = True,
) -> ShipPlan:
    """Resolve a selection into a hashed, transferable plan.

    Nothing in the corpus is copied — only generated files (the manifest subset, the
    git bundle) are staged under ``<data_root>/ship/<name>/``.
    """
    data_root = Path(data_root)
    repo_root = Path(repo_root)
    name = name or dataset_name or (recipe.name if recipe is not None else what)
    stage = stage_dir(data_root, name)
    stage.mkdir(parents=True, exist_ok=True)
    cache = HashCache(data_root / SHIP_DIRNAME / HASHCACHE_NAME)

    plan = ShipPlan(
        name=name,
        created=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        what=what,
        source_host=socket.gethostname(),
        dataset_name=dataset_name,
    )

    # (dest, relpath, source, kind, staged)
    sources: list[tuple[str, str, Path, str, bool]] = []

    def add(dest: str, rel: str, src: Path, kind: str = "file", staged: bool = False):
        sources.append((dest, rel, src, kind, staged))

    if what == "unprocessed":
        # Songs that still need the front half of the pipeline (separate -> clean ->
        # lyrics -> align) on the rig: the as-provided audio, its sidecars and the
        # records. ids are minted HERE, so the rig never invents one that collides.
        manifest = Manifest.for_data_root(data_root)
        subset = [r for r in manifest.records
                  if not r.status.aligned and (not prefix or r.file.path.startswith(prefix))]
        if not subset:
            raise RuntimeError(
                f"no unaligned songs{f' under {prefix}' if prefix else ''} - nothing to ship")
        plan.song_ids = [r.id for r in subset]
        for rec in subset:
            audio = data_root / rec.file.path
            if audio.exists():
                add("data", rec.file.path, audio)
            for side in (audio.with_suffix(".txt"), audio.parent / "lyrics.txt",
                         audio.parent / "META.txt"):
                if side.exists():
                    add("data", side.relative_to(data_root).as_posix(), side)
        add("data", MANIFEST_NAME,
            _write_manifest_subset(subset, stage / "manifest.subset.jsonl"),
            kind="manifest", staged=True)

    if what in ("dataset", "rebuildable", "full"):
        if not (data_root / MANIFEST_NAME).exists():
            raise FileNotFoundError(
                f"no manifest at {data_root / MANIFEST_NAME} — nothing to ship")

        if what == "dataset":
            dsname = dataset_name or (recipe.name if recipe is not None else None)
            if not dsname:
                raise ValueError("--what dataset needs --dataset-name (or a recipe)")
            dsdir = data_root / "datasets" / dsname
            if not dsdir.is_dir():
                raise FileNotFoundError(
                    f"{dsdir} does not exist — run `signalml dataset build` first")
            plan.dataset_name = dsname
            for p in _iter_files(dsdir):
                add("data", p.relative_to(data_root).as_posix(), p)
            plan.song_ids = _dataset_song_ids(dsdir)
        else:
            if recipe is None:
                from ..stages.dataset import load_dataset_recipe
                recipe = load_dataset_recipe()
            records, skipped = _selected_records(data_root, recipe)
            if not records:
                raise RuntimeError(
                    f"recipe {recipe.name!r} selects 0 songs ({len(skipped)} skipped) "
                    f"— nothing to ship")
            plan.song_ids = [r.id for r in records]
            plan.audio_profile = recipe.profile
            wanted = ["clean/vocals.wav", "align/phones.json", "align/vocals.TextGrid",
                      "analysis.json", "score.json"]
            if with_features:
                wanted.append("features/vocals.npz")
            if what == "full":
                wanted += [f"stems/{s}.wav" for s in ("vocals", "drums", "bass", "other")]
            for rec in records:
                sdir = data_root / "songs" / rec.id
                for rel in wanted:
                    p = sdir / rel
                    if p.exists():
                        add("data", p.relative_to(data_root).as_posix(), p)
            if what == "full" and (data_root / "datasets").is_dir():
                for p in _iter_files(data_root / "datasets"):
                    add("data", p.relative_to(data_root).as_posix(), p)

        manifest = Manifest.for_data_root(data_root)
        keep = set(plan.song_ids)
        subset = [r for r in manifest.records if r.id in keep] if keep else manifest.records

        if with_raw:
            for rec in subset:
                p = data_root / rec.file.path
                if p.exists():
                    add("data", rec.file.path, p)

        # the manifest travels as a subset: the receiver must not learn about songs
        # whose audio did not come with them (their status flags would be lies)
        add("data", MANIFEST_NAME,
            _write_manifest_subset(subset, stage / "manifest.subset.jsonl"),
            kind="manifest", staged=True)

    if with_code or what == "code":
        prov, files = _write_code_stage(repo_root, stage / "code", allow_dirty=allow_dirty)
        plan.git = prov
        kinds = {BUNDLE_NAME: "bundle", PATCH_NAME: "patch"}
        for f in files:
            add("code", f"{CODE_SUBDIR}/{f.name}", f,
                kind=kinds.get(f.name, "file"), staged=True)

    items: list = sources
    if progress and sources:
        from tqdm import tqdm
        items = tqdm(sources, desc="hashing", unit="file")

    seen: set[str] = set()
    for dest, rel, src, kind, staged in items:
        if rel in seen:
            continue
        seen.add(rel)
        plan.items.append(ShipItem(
            key=item_key(dest, rel), dest=dest, path=rel, size=src.stat().st_size,
            sha256=cache.digest(src), kind=kind, staged=staged,
        ))
    cache.save()

    if plan.git and plan.git.dirty:
        plan.notes.append(
            f"shipped from a DIRTY worktree — {CODE_SUBDIR}/{PATCH_NAME} holds the "
            f"diff and is never applied automatically")
    plan.save(stage / PLAN_NAME)
    return plan


def resolve_source(item: ShipItem, *, data_root: Path, stage: Path,
                   repo_root: Path) -> Path:
    """Sender side: where an item's bytes actually live on this machine."""
    if item.staged:
        if item.dest == "code":
            return stage / "code" / Path(item.path).name
        return stage / "manifest.subset.jsonl"
    return (data_root if item.dest == "data" else repo_root) / item.path


# ------------------------------------------------------------------------ verify


class VerifyResult(BaseModel):
    ok: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    corrupt: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.missing and not self.corrupt


def verify(plan: ShipPlan, *, data_root: str | Path,
           repo_root: str | Path | None = None, progress: bool = True) -> VerifyResult:
    """Receiver-side gate: recompute every hash at the destination.

    Run this before a training run. Items that are merged or applied on arrival
    (manifest, bundle, patch) have no byte-for-byte destination and are skipped.
    """
    data_root = Path(data_root)
    repo_root = Path(repo_root) if repo_root else None
    result = VerifyResult()
    items: list = plan.items
    if progress and items:
        from tqdm import tqdm
        items = tqdm(items, desc="verifying", unit="file")
    for item in items:
        if item.kind != "file":
            result.skipped.append(item.path)
            continue
        root = data_root if item.dest == "data" else repo_root
        if root is None:
            result.skipped.append(item.path)
            continue
        dest = root / item.path
        if not dest.exists():
            result.missing.append(item.path)
        elif dest.stat().st_size != item.size or sha256_file(dest) != item.sha256:
            result.corrupt.append(item.path)
        else:
            result.ok.append(item.path)
    return result
