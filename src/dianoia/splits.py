"""One conversation-level train / validation / test split for every source (10/04/2026).

Every data builder takes a unit's split from here instead of deriving its own exclusions. A unit belongs to a
**group**, the largest thing whose parts must never be divided between splits:

- meeting: the whole meeting (QMSum file stem);
- chat: the whole IRC day file (threads of one day share people and topics);
- wiki: the whole Wikipedia talk-page conversation (ConvoKit ``wiki-corpus``);
- fomc: the whole Federal Open Market Committee meeting (ConvoKit ``fomc-corpus``);
- wikiconv: the whole talk *page* (ConvoKit ``wikiconv-<year>``): threads on one page share topic and editors;
- email: the excerpt itself (the splitter's email training units do not record their mailbox: a known limitation).

A group's split is a fixed function of its name (sha256 into 100 buckets: 80 train, 10 val, 10 test), so a new source
or a rebuilt dataset gets the same assignment without coordination. ``write_manifest`` records the groups a dataset
uses and the manifest's own checksum, and ``audit-data.py`` refuses a dataset that disagrees with it.

The check sources (``ext/qmsum``, ``ext/irc`` and the phase-1 email test items) are never in any of these splits;
they stay the encoder's and splitter's check sets. Known limitation: splitter q3 and encoder S1 were trained on
excerpts of the meeting and chat groups that now land in val and test, so only the next-idea predictor is clean on
those; the wiki groups were seen by nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SPLITS = (("test", 10), ("val", 10), ("train", 80))

# License class of each source, as its dataset states it (10/04/2026). A build may be published (for example to
# Hugging Face) only if every source it uses is "open"; check the dataset's own terms again before publishing.
LICENSES = {
    "meeting": ("open", "QMSum (AMI and ICSI meetings: CC BY 4.0)"),
    "chat": ("open", "Ubuntu IRC disentanglement (Kummerfeld et al.): CC BY 4.0"),
    "wiki": ("open", "ConvoKit wiki-corpus, Wikipedia talk pages: CC BY-SA"),
    "wikiconv": ("open", "ConvoKit WikiConv, Wikipedia talk pages: CC BY-SA"),
    "fomc": ("open", "FOMC meeting transcripts (federalreserve.gov): US government work, public domain"),
    "email": ("research", "Enron email: public record, no license granted"),
}


def group_of(uid: str) -> tuple[str, str]:
    """The (source, group) a unit belongs to, from its uid in any of the formats the builders write."""
    u = re.sub(r"^train-", "", uid).split("@")[0].split("/")[0]
    if u.startswith("meeting-"):
        stem = u[len("meeting-") :]
        return "meeting", re.sub(r"-\d+$", "", stem) if "/" not in uid and "@" not in uid else stem
    if u.startswith("chat-"):
        m = re.match(r"chat-(\d{4}-\d{2}-\d{2})", u)
        return "chat", m.group(1) if m else u
    if u.startswith("fomc-"):
        return "fomc", u[len("fomc-") :]
    if u.startswith("wc-"):  # wc-<page id>.<thread id>
        return "wikiconv", u[len("wc-") :].split(".")[0]
    if u.startswith("wiki-"):
        return "wiki", u[len("wiki-") :]
    if u.startswith("email-"):
        return "email", u
    raise ValueError(f"unknown uid format: {uid}")


def split_of(source: str, group: str) -> str:
    b = int(hashlib.sha256(f"{source}:{group}".encode()).hexdigest(), 16) % 100
    edge = 0
    for name, share in SPLITS:
        edge += share
        if b < edge:
            return name
    return "train"


def unit_split(uid: str) -> str:
    return split_of(*group_of(uid))


def write_manifest(uids, path: Path) -> str:
    """Record every group the given units use and its split; returns the manifest's sha256."""
    groups = sorted({group_of(u) for u in uids})
    body = {"rule": "sha256('<source>:<group>') % 100: 0-9 test, 10-19 val, 20-99 train",
            "groups": {f"{s}:{g}": split_of(s, g) for s, g in groups}}
    sources = sorted({s for s, _ in groups})
    body["licenses"] = {s: LICENSES[s][1] for s in sources}
    body["publishable"] = all(LICENSES[s][0] == "open" for s in sources)
    text = json.dumps(body, indent=0, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()
