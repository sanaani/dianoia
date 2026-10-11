"""#ubuntu conversations from Kummerfeld et al.'s automatic disentanglement of the whole channel, 2004-2019 (10/05/2026).

Source: ``acl19-irc-disentanglement_auto-data-full.txt.bz2`` (github.com/jkkummerfeld/irc-disentanglement, data CC BY
4.0): 496,469 conversations, one per blank-line-separated block, lines ``<day>-ubuntu.txt:[hh:mm] <nick> text``. The same
release as ``ext/irc-train``'s annotated threads, so every day already used (the check set ``ext/irc`` and the
annotated ``ext/irc-train``) and its two neighbors are skipped whole: the check set stays unseen and no annotated thread is repeated.
Windowing as ``jepa-units.py`` chats: bots out, nicknames to speaker letters (also in the text), 8-message windows with
2+ speakers and 15-200 words. Conversations are kept by a sha256 sample of their uid (``--keep`` percent), streamed, so
memory stays flat. uid ``chat-<day>-a<n>/<k>``: the split group is the day (``splits.py``), as for every chat source.

    python scripts/irc-auto-windows.py --keep 25 --out irc-auto-windows.jsonl
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re

from dianoia.paths import DATA, work_path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BOTS = {"ubottu", "ubotu", "ubot2", "ubot3", "ubot4", "ubot5"}
LINE = re.compile(r"^(\d{4}-\d\d-\d\d)-ubuntu\.txt:\[\d\d:\d\d\] <\s*([^>]+?)\s*> (.*)$")


def used_days() -> set[str]:
    """Every day already used, and the day on each side of it: the auto release files a conversation under a day that can
    differ from the annotated sample's by one (J10's audit, 10/05/2026: held-out 2008-07-03 threads were in the auto file
    2008-07-04)."""
    import datetime

    days = {f.name[:10] for d in ("irc", "irc-train") for f in (DATA / "ext" / d).glob("*.raw.txt")}
    one = datetime.timedelta(days=1)
    return {(datetime.date.fromisoformat(d) + k * one).isoformat() for d in days for k in (-1, 0, 1)}


def windows(day: str, n: int, msgs: list[tuple[str, str]]) -> list[dict]:
    msgs = [(s, t) for s, t in msgs if t and s.lower() not in BOTS]
    if len(msgs) < 4:
        return []
    lt = {}
    for s, _ in msgs:
        lt.setdefault(s, LETTERS[min(len(lt), len(LETTERS) - 1)])
    turns = []
    for s, t in msgs:
        for nick, letter in lt.items():
            t = re.sub(rf"(?<![\w-]){re.escape(nick)}(?![\w-])", letter, t)
        turns.append([lt[s], t])
    out = []
    for k, s in enumerate(range(0, len(turns), 8)):
        win = turns[s : s + 8]
        words = sum(len(t.split()) for _, t in win)
        if len(win) >= 3 and len({x[0] for x in win}) >= 2 and 15 <= words <= 200:
            out.append({"uid": f"chat-{day}-a{n}/{k}", "kind": "original", "medium": "chat", "turns": win})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "ext" / "irc-auto" / "auto.txt.bz2"))
    ap.add_argument("--keep", type=float, default=25.0, help="percent of conversations kept (sha256 of uid)")
    ap.add_argument("--out", default="irc-auto-windows.jsonl")
    args = ap.parse_args()
    skip = used_days()
    stats = {"conversations": 0, "skipped_used_day": 0, "sampled_out": 0, "kept_conversations": 0, "windows": 0}
    with bz2.open(args.src, "rt", encoding="utf-8", errors="replace") as src, \
            open(work_path("jepa", args.out), "w", encoding="utf-8") as out:
        block: list[str] = []

        def flush() -> None:
            if not block:
                return
            stats["conversations"] += 1
            m0 = LINE.match(block[0])
            if not m0:
                return
            day, n = m0.group(1), stats["conversations"]
            if day in skip:
                stats["skipped_used_day"] += 1
                return
            if int(hashlib.sha256(f"chat-{day}-a{n}".encode()).hexdigest(), 16) % 10000 >= args.keep * 100:
                stats["sampled_out"] += 1
                return
            msgs = [(m.group(2), m.group(3).strip()) for m in map(LINE.match, block) if m]
            ws = windows(day, n, msgs)
            stats["kept_conversations"] += bool(ws)
            stats["windows"] += len(ws)
            for w in ws:
                out.write(json.dumps(w, ensure_ascii=False) + "\n")

        for line in src:
            line = line.rstrip("\n")
            if line.strip():
                block.append(line)
            else:
                flush()
                block = []
        flush()
    print(json.dumps({**stats, "used_days": len(skip)}))


if __name__ == "__main__":
    main()
