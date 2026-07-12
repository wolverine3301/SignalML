"""G2P — lyrics words -> MFA IPA syllables (Q2; docs/MIGRATION_PLAN.md P6.3).

Emits the *same* phone inventory the aligner uses (``mfa_ipa/en_v1``) so train- and
inference-time phones stay identical. Stress is carried on the syllable as a separate
field, never as a phone suffix.

Backends (mirroring the F0-backend pattern from P4):

- ``LexiconG2P`` — project-owned JSON lexicon with explicit syllables + stress. The
  test backend, and the override layer for words the automatic backends get wrong.
- ``MfaG2P`` — the preferred route (ARCHITECTURE §6): shells out to ``mfa g2p`` in the
  aligner conda env with a trained G2P model. MFA emits a flat phone string; we
  syllabify with a max-onset heuristic and carry no stress (MFA IPA has none).
- ``EspeakG2P`` — fallback via ``phonemizer`` + espeak-ng, normalized into the target
  set; experimental, requires those extras installed.

``ChainG2P`` tries backends in order (lexicon first), so hand-fixes always win.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from .phoneset import PhoneSet, get_phone_set


class SyllablePron(BaseModel):
    phones: list[str] = Field(min_length=1)
    stress: int | None = Field(default=None, ge=0, le=2)


class WordPron(BaseModel):
    word: str
    syllables: list[SyllablePron] = Field(min_length=1)


class G2PError(KeyError):
    """Word could not be pronounced by any backend."""


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def normalize_word(word: str) -> str:
    """Lookup normalization: lowercase, strip surrounding punctuation, keep apostrophes."""
    return word.strip().strip("\".,;:!?()[]").lower()


def syllabify(phones: Sequence[str], phone_set: PhoneSet) -> list[list[str]]:
    """Split a flat phone sequence into syllables: every nucleus starts a rime, and all
    consonants between two nuclei attach to the *following* syllable (max-onset,
    simplified — no phonotactic legality check; documented limitation)."""
    nuclei_idx = [i for i, ph in enumerate(phones) if phone_set.is_nucleus(ph)]
    if not nuclei_idx:
        return [list(phones)]  # no vowel (e.g. "hmm") — one syllable, best effort
    syllables: list[list[str]] = []
    start = 0
    for pos, nucleus in enumerate(nuclei_idx):
        if pos + 1 < len(nuclei_idx):
            end = nucleus + 1  # coda consonants move to the next onset
        else:
            end = len(phones)  # last syllable takes the true coda
        syllables.append(list(phones[start:end]))
        start = end
    return syllables


class LexiconG2P:
    """JSON lexicon: ``{"word": [{"phones": [...], "stress": 1}, ...]}`` (one entry per
    syllable, in order). The file is project-owned and versioned with the phone set."""

    def __init__(self, lexicon: dict[str, list[dict]] | None = None,
                 path: str | Path | None = None):
        if (lexicon is None) == (path is None):
            raise ValueError("pass exactly one of lexicon= or path=")
        if path is not None:
            lexicon = json.loads(Path(path).read_text(encoding="utf-8"))
        assert lexicon is not None
        self._entries = {
            normalize_word(w): [SyllablePron.model_validate(s) for s in syls]
            for w, syls in lexicon.items()
        }

    def pronounce(self, word: str) -> WordPron:
        key = normalize_word(word)
        if key not in self._entries:
            raise G2PError(f"{word!r} not in lexicon")
        return WordPron(word=key, syllables=self._entries[key])


class MfaG2P:
    """Drives ``mfa g2p`` (aligner conda env). Results are cached per instance; feed
    whole word lists via ``pronounce_words`` to amortize MFA's startup cost."""

    def __init__(
        self,
        g2p_model: str = "english_us_mfa",
        *,
        mfa_command: Sequence[str] = ("conda", "run", "-n", "aligner",
                                      "--no-capture-output", "mfa"),
        phone_set: str = "mfa_ipa/en_v1",
        runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    ):
        self.g2p_model = g2p_model
        self.mfa_command = list(mfa_command)
        self.phone_set = get_phone_set(phone_set)
        self._runner = runner or (
            lambda cmd: subprocess.run(cmd, check=True, capture_output=True, text=True)
        )
        self._cache: dict[str, WordPron] = {}

    def pronounce_words(self, words: Sequence[str]) -> None:
        """Batch-pronounce into the cache (one MFA invocation for the whole list)."""
        todo = sorted({normalize_word(w) for w in words} - set(self._cache))
        if not todo:
            return
        with tempfile.TemporaryDirectory(prefix="signalml_g2p_") as tmp:
            wordlist = Path(tmp) / "words.txt"
            out_path = Path(tmp) / "pronunciations.txt"
            wordlist.write_text("\n".join(todo) + "\n", encoding="utf-8")
            self._runner(
                [*self.mfa_command, "g2p", str(wordlist), self.g2p_model, str(out_path),
                 "--num_pronunciations", "1"]
            )
            for line in out_path.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t") if "\t" in line else line.split(maxsplit=1)
                if len(parts) == 3 and _is_float(parts[1]):
                    parts = [parts[0], parts[2]]  # some MFA versions add a score column
                if len(parts) != 2:
                    continue
                word, pron = parts[0], parts[1].split()
                unknown = self.phone_set.unknown(pron)
                if unknown:
                    raise ValueError(
                        f"mfa g2p emitted phones outside {self.phone_set.name} for "
                        f"{word!r}: {unknown} — phone set and G2P model disagree"
                    )
                self._cache[word] = WordPron(
                    word=word,
                    syllables=[SyllablePron(phones=syl)
                               for syl in syllabify(pron, self.phone_set)],
                )

    def pronounce(self, word: str) -> WordPron:
        key = normalize_word(word)
        if key not in self._cache:
            self.pronounce_words([key])
        if key not in self._cache:
            raise G2PError(f"mfa g2p produced no pronunciation for {word!r}")
        return self._cache[key]


# espeak IPA -> mfa_ipa/en_v1 normalization (kept deliberately small; grow with evidence)
_ESPEAK_TO_MFA = {
    "r": "ɹ", "ɜː": "ɝ", "ɜ": "ɝ", "ᵻ": "ɪ", "ɐ̃": "ɐ",
    "u": "ʉ", "uː": "ʉː", "eɪ": "ej", "oʊ": "ow", "aɪ": "aj",
    "aʊ": "aw", "ɔɪ": "ɔj", "əl": "ɫ̩",
}


class EspeakG2P:
    """Fallback backend via ``phonemizer`` (espeak-ng). Experimental: syllabification is
    the same max-onset heuristic and stress comes from espeak's ˈ/ˌ marks."""

    def __init__(self, phone_set: str = "mfa_ipa/en_v1", language: str = "en-us"):
        try:
            from phonemizer.backend import EspeakBackend
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "EspeakG2P needs `phonemizer` + espeak-ng installed "
                "(pip install phonemizer; espeak-ng from its Windows installer)"
            ) from exc
        self.phone_set = get_phone_set(phone_set)
        self._backend = EspeakBackend(language, with_stress=True)

    def pronounce(self, word: str) -> WordPron:
        key = normalize_word(word)
        ipa = self._backend.phonemize([key], strip=True)[0].strip()
        if not ipa:
            raise G2PError(f"espeak produced nothing for {word!r}")
        phones, stresses = self._parse_espeak(ipa)
        syllables = syllabify(phones, self.phone_set)
        # attach stress to syllables by nucleus order (espeak marks precede syllables)
        prons = [SyllablePron(phones=syl, stress=stresses[i] if i < len(stresses) else None)
                 for i, syl in enumerate(syllables)]
        return WordPron(word=key, syllables=prons)

    def _parse_espeak(self, ipa: str) -> tuple[list[str], list[int | None]]:
        """Tokenize espeak IPA output into target-set phones + per-syllable stress."""
        stresses: list[int | None] = []
        pending: int | None = None
        phones: list[str] = []
        # longest-match tokenization against known symbols (target set + mapping keys)
        symbols = sorted(set(self.phone_set.phones) | set(_ESPEAK_TO_MFA), key=len,
                         reverse=True)
        i = 0
        while i < len(ipa):
            ch = ipa[i]
            if ch == "ˈ":
                pending = 1
                i += 1
                continue
            if ch == "ˌ":
                pending = 2
                i += 1
                continue
            for sym in symbols:
                if ipa.startswith(sym, i):
                    ph = _ESPEAK_TO_MFA.get(sym, sym)
                    phones.append(ph)
                    if self.phone_set.is_nucleus(ph):
                        stresses.append(pending if pending is not None else 0)
                        pending = None
                    i += len(sym)
                    break
            else:
                raise ValueError(
                    f"cannot map espeak symbol at {ipa[i:]!r} (word IPA {ipa!r}) into "
                    f"{self.phone_set.name} — extend _ESPEAK_TO_MFA"
                )
        unknown = self.phone_set.unknown(phones)
        if unknown:
            raise ValueError(f"unmapped espeak phones {unknown} for IPA {ipa!r}")
        return phones, stresses


class ChainG2P:
    """Try backends in order; first hit wins (put the lexicon first so hand-curated
    pronunciations always override automatic ones)."""

    def __init__(self, *backends):
        if not backends:
            raise ValueError("ChainG2P needs at least one backend")
        self.backends = backends

    def pronounce(self, word: str) -> WordPron:
        errors = []
        for backend in self.backends:
            try:
                return backend.pronounce(word)
            except G2PError as exc:
                errors.append(f"{type(backend).__name__}: {exc}")
        raise G2PError(f"no backend pronounced {word!r}: {'; '.join(errors)}")


_WORD_RE = re.compile(r"[^\W\d_]+[''']?[^\W\d_]*", re.UNICODE)


def words_in_text(text: str) -> list[str]:
    """Unique normalized words in lyrics text, in first-appearance order (for batch
    G2P warm-up)."""
    seen: dict[str, None] = {}
    for match in _WORD_RE.finditer(text):
        seen.setdefault(normalize_word(match.group()), None)
    return list(seen)
