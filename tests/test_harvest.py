"""Channel harvest contract tests — offline: channel listings are fixtures, the converter
is a copy, nothing touches YouTube."""

from __future__ import annotations

import shutil
from pathlib import Path

from signalml.ingest.harvest import (
    Entry,
    corpus_youtube_ids,
    ingest_inbox,
    load_plans,
    plan_channel,
    save_plan,
    skip_reason,
    song_key,
)
from signalml.manifest import Manifest

S = "nova reyes"


def _e(i: int, title: str, dur: int = 200) -> Entry:
    return Entry(id=f"vid{i:08d}", title=title, duration=dur)


def _copy(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


class TestPlan:
    def test_skip_reasons(self):
        def r(title, dur=200):
            return skip_reason(_e(1, title, dur), S, have_ids=set())

        assert r("Nova Reyes - Glass Orchard (Acoustic / Audio)") is None
        assert r("Kitchen Sessions: Paper Boats (Wren Hale Cover)") is None
        assert r("Nova Reyes - Low Tide (Track by Track)") == "talking / not a song"
        assert r("Nova Reyes, June Lark - Undertow (PORCH Version)") == "duet / feature"
        assert r("Nova and Pim - Paper Moon (Wren Hale) cover") == "duet / feature"
        assert r("Kestrel - Harbor ft. Nova Reyes") == "duet / feature"
        assert r("Nova Reyes - Lantern Road (Live From Jimmy Kimmel Live!)") == \
            "TV / arena performance"
        assert r("Nova Reyes - Wildfire Hymn (XY Remix / Audio)") == "remix / instrumental"
        assert r("Wren Hale Medley (Cover) - Paper Boats") == "medley / compilation"
        assert r("Nova Reyes - Quiet Hours (Audio)") == "studio release"
        assert r("Nova Reyes - Ember Lane") == "unclear (probably a music video)"
        assert r("Nova Reyes: Big Porch Concert", 1479) == "too short or too long"
        assert r("嘘/シド【カバー】") == "non-Latin title"

    def test_prefer_keeps_and_ranks_first(self):
        """A channel's own name for its acoustic series is not in any generic list."""
        entries = [_e(1, "Nova Reyes - Harbor Lights (Vevo Acoustic)"),
                   _e(2, "Nova Reyes - Harbor (PORCH Version)")]
        plain = plan_channel(entries, S)
        assert [e.id for e in plain.picked] == ["vid00000001"]
        preferred = plan_channel(entries, S, prefer=["PORCH"])
        assert [e.id for e in preferred.picked] == ["vid00000002", "vid00000001"]

    def test_one_version_per_song_and_cap(self):
        entries = [_e(1, "Nova Reyes - Harbor (Acoustic)"),
                   _e(2, "Harbor (Live Acoustic)"),
                   _e(3, "Nova Reyes - Harbor Lights (Acoustic)"),
                   _e(4, "Nova Reyes - Slow Burn (Acoustic)")]
        plan = plan_channel(entries, S, cap=2)
        assert [e.id for e in plan.picked] == ["vid00000001", "vid00000003"]
        assert plan.skipped["vid00000002"] == "another version of the same song"
        assert plan.skipped["vid00000004"] == "over the cap of 2"
        assert len(plan_channel(entries, S, versions=2).picked) == 4

    def test_song_key(self):
        assert song_key("Nova Reyes - Harbor (PORCH Version)", S) == "harbor"
        assert song_key("Kitchen Sessions: Low Tide", S) == "lowtide"
        assert song_key("Mara Quinn - Glass Orchard (NOVA COVER)", S) == "glassorchard"

    def test_already_in_corpus(self):
        plan = plan_channel([_e(1, "Nova Reyes - Harbor Lights (Acoustic)")], S,
                            have_ids={"vid00000001"})
        assert plan.picked == [] and plan.skipped["vid00000001"] == "already in corpus"

    def test_save_and_load_roundtrip(self, tmp_path):
        plan = plan_channel([_e(1, "Nova Reyes - Harbor Lights (Acoustic)")], S,
                            license="licence note")
        jpath, mpath = save_plan(plan, tmp_path)
        (back,) = load_plans([jpath])
        assert back.picked == plan.picked and back.license == "licence note"
        assert "watch?v=vid00000001" in mpath.read_text(encoding="utf-8")


class TestInbox:
    def _plan(self, tmp_path):
        plan = plan_channel([_e(1, "Nova Reyes - Harbor Lights (Acoustic)"),
                             _e(2, "Nova Reyes - Slow Burn (Acoustic)"),
                             _e(3, "Nova Reyes - Harbor (Acoustic)")],
                            S, license="licence note")
        jpath, _ = save_plan(plan, tmp_path / "lists")
        return load_plans([jpath])

    def test_ingest_places_tags_and_reports_missing(self, tmp_path, make_wav):
        root, inbox = tmp_path / "dr", tmp_path / "inbox"
        plans = self._plan(tmp_path)
        make_wav(inbox / "whatever [vid00000001].wav", hz=220)          # matched by id
        make_wav(inbox / "Nova Reyes - Slow Burn (Acoustic).wav", hz=330)  # matched by title
        make_wav(inbox / "some other song.wav", hz=440)                  # no match

        s = ingest_inbox(root, inbox, plans, batch="b1", converter=_copy)
        assert sorted(i for i, _ in s.placed) == ["vid00000001", "vid00000002"]
        assert [f.name for f in s.unmatched] == ["some other song.wav"]
        assert [e.id for e in s.missing[S]] == ["vid00000003"]
        assert len(s.new_records) == 2

        wav = dict(s.placed)["vid00000001"]
        assert wav.parent.parent == root / "RAW" / "b1" / S
        assert wav.stem == wav.parent.name and wav.stem.endswith("vid00000001")
        recs = {r.meta.song: r for r in Manifest.for_data_root(root).records}
        rec = recs["Nova Reyes - Harbor Lights (Acoustic)"]
        assert rec.meta.singer == S and rec.meta.gender == "F" and rec.meta.language == "en"
        assert rec.meta.genre == "acoustic" and rec.meta.license_note == "licence note"
        assert rec.source.kind == "youtube" and rec.source.url.endswith("vid00000001")
        # processed files leave the inbox; the unmatched one stays for a human
        assert sorted(p.name for p in inbox.glob("*.wav")) == ["some other song.wav"]
        assert "vid00000001" in corpus_youtube_ids(Manifest.for_data_root(root))

    def test_rerun_is_idempotent(self, tmp_path, make_wav):
        root, inbox = tmp_path / "dr", tmp_path / "inbox"
        plans = self._plan(tmp_path)
        make_wav(inbox / "a vid00000001.wav")
        ingest_inbox(root, inbox, plans, batch="b1", converter=_copy)
        make_wav(inbox / "again vid00000001.wav")  # same song downloaded twice
        s = ingest_inbox(root, inbox, plans, batch="b1", converter=_copy)
        assert s.placed == [] and s.already_present == ["vid00000001"]
        assert len(Manifest.for_data_root(root).records) == 1
