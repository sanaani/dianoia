"""The next-idea predictor's dataset with conversation-level splits (10/04/2026): ``jepa/units-s.jsonl``.

Takes the units of one or more files (``jepa-prepare.py``'s excerpts and paraphrases, ``jepa-chain.py``'s chained
windows) and sets each unit's split from ``dianoia.splits``: a whole meeting, chat day or wiki conversation is in
exactly one of train, val and test. Paraphrased units (Sonnet rewordings of a training excerpt) stay in training
only. Chat excerpts holding an IRC bot's canned text are dropped. Writes the split manifest (``jepa/<out>.manifest.json``)
and prints its checksum; ``audit-data.py`` must pass on the output before anything trains on it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from dianoia.clean import heldout_repeats
from dianoia.paths import DATA, work_path
from dianoia.splits import group_of, unit_split, write_manifest

BOTS = {"ubottu", "ubotu", "ubot2", "ubot3", "ubot4", "ubot5"}


def bot_text() -> set[str]:
    """Normalized texts the IRC channel bots posted (their canned factoids), addressee prefix removed."""
    line = re.compile(r"^\[\d\d:\d\d\] <([^>]+)> (.*)$")
    out = set()
    for f in (DATA / "ext" / "irc-train").glob("*.raw.txt"):
        for l in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = line.match(l)
            if m and m.group(1).lower() in BOTS:
                n = norm(re.sub(r"^\S+:\s*", "", m.group(2)))
                if len(n) >= 30:
                    out.add(n)
    return out


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="unit files under jepa/")
    ap.add_argument("--out", default="units-s.jsonl")
    args = ap.parse_args()
    # The splitter's chat excerpts (split/train.jsonl) predate the bot filter of jepa-units.py: drop the ones with a
    # bot's canned text.
    bots = bot_text()
    excerpt_turns = {json.loads(l)["uid"]: json.loads(l)["turns"] for l in open(work_path("split", "train.jsonl"), encoding="utf-8")}
    has_bot = {uid for uid, turns in excerpt_turns.items()
               if any(norm(re.sub(r"^\S+:\s*", "", t)) in bots for _, t in turns)}
    seen, out, dropped_bot = set(), [], 0
    for name in args.inputs:
        for l in open(work_path("jepa", name), encoding="utf-8"):
            u = json.loads(l)
            key = (u["uid"], u.get("kind", "original"))
            if key in seen:
                continue
            seen.add(key)
            if u["uid"] in has_bot:
                dropped_bot += 1
                continue
            s = unit_split(u["uid"])
            if s != "train" and u.get("kind") == "paraphrase":
                continue
            out.append({**u, "split": s, "group": ":".join(group_of(u["uid"]))})
    # Held-out units repeating a training group once names are placeholders (templated notices): dropped (clean.py).
    repeats = heldout_repeats(out, key=lambda uid: uid)
    out = [u for u in out if u["uid"] not in repeats]
    with open(work_path("jepa", args.out), "w", encoding="utf-8") as f:
        for u in out:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    sha = write_manifest([u["uid"] for u in out], work_path("jepa", args.out + ".manifest.json"))
    c = Counter((u["split"], u["medium"]) for u in out)
    print(json.dumps({"units": len(out), "dropped_bot_excerpts": dropped_bot, "heldout_repeats_dropped": len(repeats), "manifest_sha256": sha, "by_split_medium": {f"{a}/{b}": n for (a, b), n in sorted(c.items())}}))


if __name__ == "__main__":
    main()
