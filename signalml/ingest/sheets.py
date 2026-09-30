"""Session documents that ship with some multitracks: composition sheet, cue sheet, meta.

* **Composition sheet** (``*_CompSheet.pdf``): the title line, then the lyrics under
  upper-case section headers (``VERSE 1``, ``CHORUS 2``, ``MIDDLE SECTION``,
  ``OUTRO (CHORUS 3)``). Official lyrics: they go to ``lyrics.txt`` (lyric lines only,
  so the aligner never sings a header) and the section list to the song structure.
* **Cue sheet** (``*_CueSheet.pdf``): ``(G minor, 4/4, 96bpm)`` then one row per
  section - start time, bars, section name, notes. Seen in the wild: ``approx 87bpm``,
  a ``186-216bpm`` range, the literal word ``key`` for an unknown key, and typo'd
  times (``2.16``, ``1:53:``, ``0.:17``).
* **meta.txt**: ``Style: Smooth Acoustic Rock`` - the song's genre label.

PDF text comes out with curly quotes and apostrophes as U+FFFD; inside a word that
is an apostrophe (``I\\ufffdm`` -> ``I'm``), elsewhere it is dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

SECTION = re.compile(
    r"^(?:(?:PRE-?\s?)?CHORUS|VERSE|BRIDGE|MIDDLE(?:\s+(?:SECTION|8|EIGHT))?|OUTRO|INTRO|"
    r"REFRAINS?|HOOK|BREAKDOWN|SOLO|TAG|CODA|INTERLUDE|POST-?\s?CHORUS)\b[A-Z0-9 ()&'\-/.]*$")
KEY = re.compile(r"^[A-G](?:#|b)?\s+(?:major|minor)$", re.I)
TIME = re.compile(r"^(\d{1,2})\D{1,2}(\d{2})\b")


def pdf_text(path: str | Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def fix_quotes(line: str) -> str:
    line = re.sub(r"(?<=\w)�(?=\w)", "'", line)
    return line.replace("�", "").strip()


def _lines(text: str) -> list[str]:
    return [fix_quotes(x) for x in text.splitlines() if fix_quotes(x)]


def parse_comp_sheet(text: str) -> dict:
    """``{"title", "sections": [{"name", "lines"}]}``; lines before any header land in
    an unnamed section."""
    lines = _lines(text)
    title, body = (lines[0], lines[1:]) if lines else ("", [])
    sections: list[dict] = []
    for line in body:
        if SECTION.match(line) and not re.search(r"[a-z]", line):
            sections.append({"name": line, "lines": []})
        else:
            if not sections:
                sections.append({"name": "", "lines": []})
            sections[-1]["lines"].append(line)
    return {"title": title, "sections": sections}


def lyrics_text(comp: dict) -> str:
    """Plain lyrics for the aligner: every lyric line, headers dropped."""
    return "\n".join(line for s in comp["sections"] for line in s["lines"]) + "\n"


def parse_cue_sheet(text: str) -> dict:
    """``{"title", "key", "meter", "bpm": [..], "sections": [{"start_sec", "bars",
    "name", "notes"}]}``. Rows that cannot be read are kept under ``unparsed``."""
    lines = _lines(text)
    out: dict = {"title": lines[0] if lines else "", "key": None, "meter": None,
                 "bpm": [], "sections": [], "unparsed": []}
    header = next((x for x in lines[1:3] if x.startswith("(")), "")
    for part in (p.strip() for p in header.strip("()").split(",")):
        if KEY.match(part):
            out["key"] = part
        elif re.fullmatch(r"\d+/\d+", part):
            out["meter"] = part
        elif "bpm" in part.lower():
            out["bpm"] = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", part)]
    for line in lines:
        m = TIME.match(line)
        if not m:
            continue
        rest = line[m.end():].lstrip(" :.").split(None, 1)
        if not rest:
            out["unparsed"].append(line)
            continue
        bars = int(rest[0]) if rest[0].isdigit() else None
        name = rest[1] if len(rest) > 1 else ""
        out["sections"].append({"start_sec": int(m.group(1)) * 60 + int(m.group(2)),
                                "bars": bars, "name": name.strip(), "notes": ""})
    return out


def parse_meta_txt(text: str) -> dict:
    """``Key: value`` lines, keys lower-cased (``{"style": "smooth acoustic rock"}``)."""
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() and value.strip():
            out[key.strip().lower()] = value.strip()
    return out


def read_session_docs(paths: list[str | Path]) -> dict:
    """Everything the documents of one session say: ``{"comp", "cue", "meta"}`` for
    whichever of them exist."""
    out: dict = {}
    for p in map(Path, paths):
        name = p.name.lower()
        try:
            if name.endswith("compsheet.pdf"):
                out["comp"] = parse_comp_sheet(pdf_text(p))
            elif name.endswith("cuesheet.pdf"):
                out["cue"] = parse_cue_sheet(pdf_text(p))
            elif name == "meta.txt":
                out["meta"] = parse_meta_txt(p.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # a broken document must not stop an import
            out.setdefault("errors", []).append(f"{p.name}: {exc}")
    return out
