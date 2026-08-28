"""``signalml ship`` contract tests — CPU-only, offline (loopback only).

The claims under test are the ones the tool exists to make: a plan is a complete,
hashed description of a selection; the transport is resumable and idempotent; the
receiver ends up byte-identical or loudly not; and the code half really does clone and
then update a repo that started with nothing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from signalml.manifest import MANIFEST_NAME, Manifest, scan_directory
from signalml.net import client
from signalml.net import plan as planmod
from signalml.net.server import serve
from signalml.stages.common import song_dir, update_analysis
from signalml.stages.dataset import DatasetRecipe, SegmentationCfg

HAS_GIT = shutil.which("git") is not None

PHONES = [
    {"ph": "ʃ", "start": 0.5, "end": 0.8, "word": "shine", "stress": None},
    {"ph": "aj", "start": 0.8, "end": 1.2, "word": "shine", "stress": None},
    {"ph": "n", "start": 1.25, "end": 1.6, "word": "shine", "stress": None},
    {"ph": "s", "start": 2.6, "end": 3.0, "word": "stay", "stress": None},
    {"ph": "t", "start": 3.0, "end": 3.4, "word": "stay", "stress": None},
]


def recipe(**overrides) -> DatasetRecipe:
    base = {
        "name": "t1", "trainer": "acoustic", "profile": "dev",
        "filters": {"min_align_score": 0.8},
        "segmentation": SegmentationCfg(min_clip_sec=0.5).model_dump(),
    }
    base.update(overrides)
    return DatasetRecipe.model_validate(base)


def ready_song(root, make_wav, *, singer="alice", profile="dev"):
    """A record that `dataset build` would accept: cleaned + aligned at `profile`."""
    make_wav(root / "raw" / f"{singer}_song.wav", seconds=1.0)
    manifest, new = scan_directory(root, language="en", gender="F", singer=singer)
    rec = new[0]
    rec.status.separated = rec.status.cleaned = rec.status.aligned = True
    rec.quality.align_score = 0.9
    rec.meta.license_note = "personal research use"
    manifest.upsert(rec)
    manifest.save()

    sdir = song_dir(root, rec.id)
    make_wav(sdir / "clean" / "vocals.wav", seconds=4.0)
    update_analysis(sdir, "clean", {"profile": profile})
    (sdir / "align").mkdir(parents=True, exist_ok=True)
    (sdir / "align" / "phones.json").write_text(
        json.dumps({"phone_set": "mfa_ipa/en_v1", "phones": PHONES}, ensure_ascii=False),
        encoding="utf-8")
    return rec.id


def make_dataset(root, name="t1", song_id="sng_0001"):
    """A minimal S6b output tree (wavs + transcriptions.csv + card)."""
    folder = root / "datasets" / name / "alice-en"
    (folder / "wavs").mkdir(parents=True, exist_ok=True)
    (folder / "wavs" / f"{song_id}_000.wav").write_bytes(b"RIFF" + b"\0" * 2048)
    (folder / "transcriptions.csv").write_text(
        f"name,ph_seq,ph_dur\n{song_id}_000,SP aj SP,0.1 0.4 0.1\n", encoding="utf-8")
    (root / "datasets" / name / "dataset_card.md").write_text("# card\n", encoding="utf-8")
    return root / "datasets" / name


@pytest.fixture()
def sender(tmp_path, make_wav):
    """A data root with one shippable song and a built dataset."""
    root = tmp_path / "send"
    root.mkdir()
    song_id = ready_song(root, make_wav)
    make_dataset(root, "t1", song_id)
    return root


@pytest.fixture()
def git_repo(tmp_path):
    if not HAS_GIT:
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    (repo / "signalml").mkdir(parents=True)
    (repo / "signalml" / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("# fixture repo\n", encoding="utf-8")
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,  # noqa: E731
                                    capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True,
                   capture_output=True)
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "test")
    run("add", "-A")
    run("commit", "-m", "fixture")
    return repo


def start_server(plan, data_root, repo_root, token="tok"):
    ready = threading.Event()
    httpd = serve(plan, data_root=data_root, repo_root=repo_root, host="127.0.0.1",
                  port=0, token=token, quiet=True, ready=ready)
    ready.wait(5)
    host, port = httpd.server_address[:2]
    return httpd, f"http://{host}:{port}/{token}"


class TestPlan:
    def test_dataset_selection_hashes_every_file(self, sender):
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  with_code=False, progress=False)
        paths = {i.path for i in plan.items}
        assert "datasets/t1/alice-en/wavs/sng_0001_000.wav" in paths
        assert "datasets/t1/dataset_card.md" in paths
        assert all(len(i.sha256) == 64 and i.size >= 0 for i in plan.items)
        assert plan.song_ids == ["sng_0001"]  # read back out of transcriptions.csv

    def test_manifest_ships_as_a_merged_subset(self, sender):
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  with_code=False, progress=False)
        item = next(i for i in plan.items if i.kind == "manifest")
        assert item.path == MANIFEST_NAME and item.staged
        staged = planmod.stage_dir(sender, "t1") / "manifest.subset.jsonl"
        assert [json.loads(ln)["id"] for ln in
                staged.read_text(encoding="utf-8").splitlines()] == ["sng_0001"]

    def test_rebuildable_matches_what_the_recipe_selects(self, sender):
        plan = planmod.build_plan(sender, what="rebuildable", recipe=recipe(),
                                  with_code=False, progress=False)
        paths = {i.path for i in plan.items}
        assert "songs/sng_0001/clean/vocals.wav" in paths
        assert "songs/sng_0001/align/phones.json" in paths
        # features are recomputable, so they stay home unless asked for
        assert not any(p.endswith("vocals.npz") for p in paths)

    def test_rebuildable_refuses_when_the_recipe_selects_nothing(self, sender):
        with pytest.raises(RuntimeError, match="0 songs"):
            planmod.build_plan(sender, what="rebuildable",
                               recipe=recipe(filters={"gender": "M"}),
                               with_code=False, progress=False)

    def test_hash_cache_survives_a_replan(self, sender):
        first = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                   with_code=False, progress=False)
        second = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                    with_code=False, progress=False)
        assert {i.path: i.sha256 for i in first.items} == \
               {i.path: i.sha256 for i in second.items}

    @pytest.mark.skipif(not HAS_GIT, reason="git not available")
    def test_dirty_worktree_is_refused_but_capturable(self, sender, git_repo):
        (git_repo / "README.md").write_text("# edited\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="dirty"):
            planmod.build_plan(sender, what="dataset", dataset_name="t1",
                               repo_root=git_repo, progress=False)
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  repo_root=git_repo, allow_dirty=True, progress=False)
        assert plan.git.dirty and plan.notes
        assert any(i.kind == "patch" for i in plan.items)


class TestTransport:
    def test_round_trip_lands_identical_bytes(self, sender, tmp_path):
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  with_code=False, progress=False)
        httpd, url = start_server(plan, sender, tmp_path)
        rig = tmp_path / "rig"
        try:
            summary = client.pull(url, data_root=rig, apply_code=False, progress=False)
        finally:
            httpd.shutdown()
        assert not summary.failed and summary.fetched == len(plan.items)
        assert summary.manifest_merged == 1
        landed = rig / "datasets/t1/alice-en/wavs/sng_0001_000.wav"
        original = sender / "datasets/t1/alice-en/wavs/sng_0001_000.wav"
        assert landed.read_bytes() == original.read_bytes()
        assert Manifest.for_data_root(rig).get("sng_0001") is not None
        assert planmod.verify(plan, data_root=rig, progress=False).clean

    def test_second_pull_is_a_no_op(self, sender, tmp_path):
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  with_code=False, progress=False)
        httpd, url = start_server(plan, sender, tmp_path)
        rig = tmp_path / "rig"
        try:
            client.pull(url, data_root=rig, apply_code=False, progress=False)
            again = client.pull(url, data_root=rig, apply_code=False, progress=False)
        finally:
            httpd.shutdown()
        assert again.fetched == 0 and again.skipped == len(plan.items)

    def test_partial_file_resumes(self, sender, tmp_path):
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  with_code=False, progress=False)
        item = next(i for i in plan.items
                    if i.path.endswith("sng_0001_000.wav"))
        rig = tmp_path / "rig"
        dest = rig / item.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        source = (sender / item.path).read_bytes()
        dest.with_name(dest.name + ".part").write_bytes(source[:100])  # interrupted

        httpd, url = start_server(plan, sender, tmp_path)
        try:
            summary = client.pull(url, data_root=rig, apply_code=False, progress=False)
        finally:
            httpd.shutdown()
        assert not summary.failed
        assert dest.read_bytes() == source
        assert not dest.with_name(dest.name + ".part").exists()

    def test_corruption_is_caught_and_repaired_by_a_re_pull(self, sender, tmp_path):
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  with_code=False, progress=False)
        httpd, url = start_server(plan, sender, tmp_path)
        rig = tmp_path / "rig"
        try:
            client.pull(url, data_root=rig, apply_code=False, progress=False)
            victim = rig / "datasets/t1/alice-en/wavs/sng_0001_000.wav"
            victim.write_bytes(b"truncated")
            result = planmod.verify(plan, data_root=rig, progress=False)
            assert not result.clean and result.corrupt
            client.pull(url, data_root=rig, apply_code=False, progress=False)
        finally:
            httpd.shutdown()
        assert planmod.verify(plan, data_root=rig, progress=False).clean

    def test_verify_reports_missing_files(self, sender, tmp_path):
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  with_code=False, progress=False)
        result = planmod.verify(plan, data_root=tmp_path / "empty", progress=False)
        assert not result.clean and result.missing and not result.ok


class TestServerSurface:
    def test_wrong_token_and_unknown_key_are_both_404(self, sender, tmp_path):
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  with_code=False, progress=False)
        httpd, url = start_server(plan, sender, tmp_path)
        base = url.rsplit("/", 1)[0]
        try:
            for bad in (f"{base}/wrong/plan", f"{url}/blob/deadbeef",
                        f"{url}/blob/{'../' * 6}etc/passwd"):
                with pytest.raises(urllib.error.HTTPError) as exc:
                    urllib.request.urlopen(bad, timeout=5)
                assert exc.value.code == 404
        finally:
            httpd.shutdown()

    def test_ping_describes_the_shipment(self, sender, tmp_path):
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  with_code=False, progress=False)
        httpd, url = start_server(plan, sender, tmp_path)
        try:
            info = client.ping(url)
        finally:
            httpd.shutdown()
        assert info["name"] == "t1" and info["items"] == len(plan.items)


@pytest.mark.skipif(not HAS_GIT, reason="git not available")
class TestCodeShipping:
    def _ship(self, sender, git_repo, tmp_path, rig_repo, rig_data):
        plan = planmod.build_plan(sender, what="dataset", dataset_name="t1",
                                  repo_root=git_repo, progress=False)
        httpd, url = start_server(plan, sender, git_repo)
        try:
            return plan, client.pull(url, data_root=rig_data, repo_root=rig_repo,
                                     progress=False)
        finally:
            httpd.shutdown()

    def test_bundle_clones_a_repo_that_did_not_exist(self, sender, git_repo, tmp_path):
        rig_repo, rig_data = tmp_path / "rig_repo", tmp_path / "rig_data"
        plan, summary = self._ship(sender, git_repo, tmp_path, rig_repo, rig_data)
        assert "cloned" in summary.code_action
        assert (rig_repo / "signalml" / "__init__.py").read_text() == "VERSION = 1\n"
        head = subprocess.run(["git", "-C", str(rig_repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert head == plan.git.commit
        assert (rig_repo / ".ship" / "PROVENANCE.txt").exists()

    def test_second_shipment_updates_the_clone(self, sender, git_repo, tmp_path):
        rig_repo, rig_data = tmp_path / "rig_repo", tmp_path / "rig_data"
        self._ship(sender, git_repo, tmp_path, rig_repo, rig_data)

        (git_repo / "signalml" / "__init__.py").write_text("VERSION = 2\n",
                                                           encoding="utf-8")
        for args in (["add", "-A"], ["commit", "-m", "bump"]):
            subprocess.run(["git", "-C", str(git_repo), *args], check=True,
                           capture_output=True)
        plan, summary = self._ship(sender, git_repo, tmp_path, rig_repo, rig_data)
        assert "updated" in summary.code_action
        assert (rig_repo / "signalml" / "__init__.py").read_text() == "VERSION = 2\n"

    def test_local_changes_on_the_rig_block_checkout_but_not_fetch(
            self, sender, git_repo, tmp_path):
        rig_repo, rig_data = tmp_path / "rig_repo", tmp_path / "rig_data"
        self._ship(sender, git_repo, tmp_path, rig_repo, rig_data)
        (rig_repo / "signalml" / "__init__.py").write_text("LOCAL = 1\n",
                                                           encoding="utf-8")
        (git_repo / "README.md").write_text("# v2\n", encoding="utf-8")
        for args in (["add", "-A"], ["commit", "-m", "v2"]):
            subprocess.run(["git", "-C", str(git_repo), *args], check=True,
                           capture_output=True)
        _, summary = self._ship(sender, git_repo, tmp_path, rig_repo, rig_data)
        assert "did NOT check out" in summary.code_action
        assert (rig_repo / "signalml" / "__init__.py").read_text() == "LOCAL = 1\n"
