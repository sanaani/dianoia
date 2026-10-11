"""Turn raw Enron messages into authored paragraphs and sentences.

An Enron body carries a great deal that its sender did not write: quoted replies, forwarded
chains, header blocks, disclaimers. Noema's lesson (`"no goals"` alone scored AUC 0.702 on random
vectors) is that shared boilerplate fakes geometric signal, so everything below is about keeping
only what the sender wrote, once.
"""

from __future__ import annotations

import hashlib
import re

# A line at which the authored part of a body ends: everything from here down is quoted,
# forwarded or header material written by someone else.
_CUT = re.compile(
    r"""^\s*(?:
        -{2,}\s*(?:original\s+message|forwarded\s+by|forwarded\s+message)
      | (?:from|to|sent|subject|cc|date)\s*:\s
      | >
      | _{5,}
      | \*{5,}
      | =+\s*$
      | .{0,80}\bon\s+\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}
      | .{0,60}\bwrote:\s*$
      | sent\s+by:
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_JUNK = re.compile(r"(@|https?://|www\.|=20|=09|\.(?:doc|xls|pdf|ppt)\b|\t)", re.IGNORECASE)


def authored(body: str) -> str:
    """The part of ``body`` above the first quoted, forwarded or header line."""
    kept = []
    for line in body.replace("\r", "").split("\n"):
        if _CUT.match(line):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks, with hard line wraps joined back up."""
    blocks = re.split(r"\n\s*\n", text)
    return [_WS.sub(" ", block).strip() for block in blocks if block.strip()]


def normalise(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def digest(text: str) -> str:
    return hashlib.sha1(normalise(text).encode()).hexdigest()


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def is_prose(text: str, *, min_words: int, max_words: int) -> bool:
    """Ordinary English prose: the right length, mostly letters, no addresses, links or encodings."""
    if _JUNK.search(text):
        return False
    n = len(words(text))
    if not min_words <= n <= max_words:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters / max(len(text), 1) < 0.7:
        return False
    upper = sum(ch.isupper() for ch in text)
    return upper / max(letters, 1) < 0.3


def thread_key(subject: str | None) -> str:
    """The subject with any run of Re:/Fw:/Fwd: prefixes removed, lowercased."""
    s = subject or ""
    while True:
        stripped = re.sub(r"^\s*(re|fw|fwd)\s*:\s*", "", s, flags=re.IGNORECASE)
        if stripped == s:
            break
        s = stripped
    return normalise(s)
