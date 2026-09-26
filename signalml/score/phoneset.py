"""Project-owned versioned phone sets (Q2: MFA IPA; OPEN_QUESTIONS *Future direction*).

Every phoneme field in the pipeline (``phones.json``, ``score.json``) declares which
phone set it uses (e.g. ``mfa_ipa/en_v1``). Validators reject phones outside the
declared set so a bad aligner run or G2P mapping fails loudly instead of poisoning
training data. Stress is a separate field, never a phone suffix.

``mfa_ipa/en_v1`` is seeded from the MFA ``english_us_mfa`` v3 dictionary inventory
(mfa-models docs). After installing MFA, verify the seed against the actual installed
dictionary with ``diff_against_mfa_dictionary`` (exposed as
``signalml score phoneset --dict``) and bump the version if the inventory changed.

Adding a language later (Gaelic wave-2, Q13) = a new ``PhoneSet`` entry here plus a
mapping table; nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Symbols MFA emits that are *not* phones. Empty/sil/sp intervals are silence (dropped
# from phones.json); "spn" is spoken-noise/OOV (kept, flagged — it marks alignment gaps).
SILENCE_MARKS = frozenset({"", "sil", "sp", "<eps>"})
NOISE_MARKS = frozenset({"spn"})


@dataclass(frozen=True)
class PhoneSet:
    """One versioned phone inventory. ``nuclei`` are syllable-nucleus phones
    (vowels + syllabic consonants) — used by G2P syllabification."""

    name: str
    language: str
    phones: frozenset[str]
    nuclei: frozenset[str]
    notes: str = ""

    def unknown(self, phones: list[str] | tuple[str, ...]) -> list[str]:
        """Phones not in this set (noise marks excluded), order-preserving, deduped."""
        seen: dict[str, None] = {}
        for ph in phones:
            if ph not in self.phones and ph not in NOISE_MARKS and ph not in SILENCE_MARKS:
                seen.setdefault(ph, None)
        return list(seen)

    def is_nucleus(self, ph: str) -> bool:
        return ph in self.nuclei


# Union of the english_us_mfa v3 inventory (US G2P output) and the installed general
# english_mfa v3 dictionary (aligner output; adds pan-dialect phones: a/aː eː oː u/uː
# ɔ ɛː ɜ/ɜː əw, retroflex ʈ/ʈʲ/ʈʷ/ɖ, kp ʋ ʎ) — verified 2026-07-12 against the
# installed dict via `signalml score phoneset --dict`.
_EN_V1_VOWELS = (
    "a aː aj aw e eː ej i iː ɪ o oː ow ɔ ɔj u uː ə əw ɚ ɛ ɛː ɜ ɜː ɝ æ ɐ ɑ ɑː ɒ ɒː "
    "ʉ ʉː ʊ".split()
)
_EN_V1_SYLLABIC_CONSONANTS = "m̩ n̩ ɫ̩".split()
_EN_V1_CONSONANTS = (
    "b bʲ c cʰ cʷ d dʒ dʲ d̪ f fʲ h j k kp kʰ kʷ l m mʲ n p pʰ pʲ pʷ s t tʃ tʰ tʲ tʷ "
    "t̪ v w z ç ð ŋ ɖ ɟ ɟʷ ɡ ɡʷ ɫ ɱ ɲ ɹ ɾ ɾʲ ɾ̃ vʲ ʃ ʈ ʈʲ ʈʷ ʋ ʒ ʔ θ ʎ".split()
)

MFA_IPA_EN_V1 = PhoneSet(
    name="mfa_ipa/en_v1",
    language="en",
    phones=frozenset(_EN_V1_VOWELS + _EN_V1_SYLLABIC_CONSONANTS + _EN_V1_CONSONANTS),
    nuclei=frozenset(_EN_V1_VOWELS + _EN_V1_SYLLABIC_CONSONANTS),
    notes="Union of english_us_mfa v3 (G2P side) and the general english_mfa v3 "
    "dictionary (aligner side); re-verify after any MFA model update with "
    "`signalml score phoneset --dict <path>` and bump the version once artifacts "
    "depend on it.",
)

_REGISTRY: dict[str, PhoneSet] = {MFA_IPA_EN_V1.name: MFA_IPA_EN_V1}


def get_phone_set(name: str) -> PhoneSet:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown phone set {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def known_phone_sets() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class PhoneSetDiff:
    """Result of comparing a PhoneSet against an MFA pronunciation dictionary."""

    dictionary_phones: frozenset[str]
    missing_from_set: list[str] = field(default_factory=list)  # in dict, not in set
    unused_by_dict: list[str] = field(default_factory=list)  # in set, not in dict

    @property
    def clean(self) -> bool:
        return not self.missing_from_set


def parse_mfa_dictionary_phones(dict_path: str | Path) -> frozenset[str]:
    """Extract the phone inventory from an MFA pronunciation-dictionary text file.

    MFA dict lines are ``word [prob [silence_probs...]] phone phone ...`` — between the
    word and the phones there may be up to four numeric columns; everything after the
    numeric run is phones.
    """
    phones: set[str] = set()
    for line in Path(dict_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        rest = parts[1:]
        # skip the (optional) leading run of numeric probability columns
        i = 0
        while i < len(rest) and i < 4 and _is_number(rest[i]):
            i += 1
        phones.update(rest[i:])
    return frozenset(phones)


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def diff_against_mfa_dictionary(phone_set: PhoneSet, dict_path: str | Path) -> PhoneSetDiff:
    """Compare a phone set with an installed MFA dictionary. ``missing_from_set`` is the
    dangerous direction — the aligner would emit phones the pipeline rejects."""
    dict_phones = parse_mfa_dictionary_phones(dict_path)
    return PhoneSetDiff(
        dictionary_phones=dict_phones,
        missing_from_set=sorted(dict_phones - phone_set.phones - SILENCE_MARKS - NOISE_MARKS),
        unused_by_dict=sorted(phone_set.phones - dict_phones),
    )


# Consonant allophones in mfa_ipa/en_v1 folded into their base phoneme. The general
# english_mfa dictionary spans many dialects, and on sung English it spreads one
# consonant over several tokens per word (measured 2026-09-25: 38% of /t/
# landed on t_retroflex/t_dental/t_palatalized/..., 37% of /l/ on dark or palatal l,
# /w/ almost entirely on the labiodental approximant), so each is learned from a
# fraction of the data. Vowels are left alone: accent is part of a singer's sound.
# Syllabic consonants (m, n, l with the syllabic mark) stay: they are nuclei.
CONSONANT_CORE_EN_V1: dict[str, str] = {
    **{ph: "t" for ph in ("tʰ", "tʲ", "tʷ", "ʈ", "ʈʲ", "t̪", "ɾ")},
    **{ph: "d" for ph in ("dʲ", "d̪", "ɖ")},
    **{ph: "k" for ph in ("kʰ", "kʷ", "c", "cʰ", "cʷ")},
    **{ph: "ɡ" for ph in ("ɡʷ", "ɟ", "ɟʷ")},
    **{ph: "p" for ph in ("pʰ", "pʲ")},
    "bʲ": "b", "fʲ": "f", "vʲ": "v", "mʲ": "m", "ɲ": "n",
    "ɫ": "l", "ʎ": "l", "ʋ": "w",
}
PHONE_MERGES: dict[str, dict[str, str]] = {"none": {}, "consonant_core": CONSONANT_CORE_EN_V1}

