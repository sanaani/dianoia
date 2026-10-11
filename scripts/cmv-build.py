"""C1 data: ChangeMyView paired arguments, cut into splitter windows (10/05/2026).

The first outcome test. Winning Arguments (Tan et al., WWW 2016; ConvoKit ``winning-args-corpus``) pairs, under one
original post, a first reply whose thread won the poster's delta with a topically similar one whose thread did not:
4,263 pairs (3,456 train, 807 held out, the authors' split). Only the first reply of each side is used, never the rest
of its thread: the poster's later messages announce the delta.

License: the data is distributed with the paper for research, with no stated license. Class ``research``: not
publishable, not for training anything shipped (same class as Enron).

Leaks removed before splitting, measured on the root replies:
- ``EDIT``/``UPDATE`` sections are cut (from the first such line to the end). Winners carry them about twice as often
  (193 vs 95), and an original post's edit often reports what changed the poster's mind.
- Lines mentioning a delta are dropped (21 replies).

Outputs, in ``--out``: ``windows.jsonl`` (splitter rows: uid ``<text id>#<k>``, kind original, medium cmv, one speaker;
paragraphs merged up to about 120 words a window, so no window overflows the splitter's 700 output tokens) and
``pairs.jsonl`` (pair id, split, original post id, winning and losing reply ids), plus ``texts.jsonl`` (the cleaned
text of each post and reply, for the text model).

    python scripts/cmv-build.py --corpus ~/dianoia-data/work/cmv/winning-args-corpus --out ~/dianoia-data/work/cmv
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

EDIT = re.compile(r"(?im)^\W*(edit|update)\b")
DELTA = re.compile(r"∆|&#8710;|\bdelta\b", re.I)
WORDS = 120


def clean(text: str, drop_delta: bool) -> str:
    text = html.unescape(text)
    m = EDIT.search(text)
    if m:
        text = text[: m.start()]
    lines = [l.strip() for l in text.split("\n")]
    if drop_delta:
        lines = [l for l in lines if not DELTA.search(l)]
    return "\n".join(lines).strip()


def windows(text: str) -> list[str]:
    """Paragraphs merged in order up to about WORDS words; a longer paragraph is its own window."""
    out, cur = [], []
    for p in (" ".join(p.split()) for p in re.split(r"\n\s*\n|\n", text)):
        if not p:
            continue
        if cur and len(" ".join(cur).split()) + len(p.split()) > WORDS:
            out.append(" ".join(cur))
            cur = []
        cur.append(p)
    if cur:
        out.append(" ".join(cur))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    corpus, out = Path(args.corpus), Path(args.out)
    conv = json.loads((corpus / "conversations.json").read_text(encoding="utf-8"))
    pairs: dict[str, dict] = {}
    replies = {}
    for line in open(corpus / "utterances.jsonl", encoding="utf-8"):
        u = json.loads(line)
        s = u["meta"].get("success")
        if s is None or u["reply-to"] != u["root"]:
            continue
        replies[u["id"]] = clean(u["text"], drop_delta=True)
        for p in u["meta"]["pair_ids"]:
            pairs.setdefault(p, {"pair": p, "op": u["root"], "split": "train" if conv[u["root"]]["train"] else "test"})[
                "win" if s == 1 else "lose"] = u["id"]
    assert all("win" in p and "lose" in p for p in pairs.values())
    texts = {}
    for p in pairs.values():
        c = conv[p["op"]]
        texts[p["op"]] = clean(c["op-title"].strip() + "\n\n" + c["op-text-body"], drop_delta=False)
    texts.update(replies)
    empty = {t for t, x in texts.items() if not x.strip()}
    kept = [p for p in pairs.values() if not {p["op"], p["win"], p["lose"]} & empty]
    used = {t for p in kept for t in (p["op"], p["win"], p["lose"])}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in sorted(kept, key=lambda p: p["pair"]):
            f.write(json.dumps(p) + "\n")
    n_win = 0
    with open(out / "windows.jsonl", "w", encoding="utf-8") as fw, open(out / "texts.jsonl", "w", encoding="utf-8") as ft:
        for t in sorted(used):
            ft.write(json.dumps({"id": t, "text": texts[t]}, ensure_ascii=False) + "\n")
            for k, w in enumerate(windows(texts[t])):
                fw.write(json.dumps({"uid": f"{t}#{k}", "kind": "original", "medium": "cmv", "turns": [["A", w]]},
                                    ensure_ascii=False) + "\n")
                n_win += 1
    print(json.dumps({"pairs": len(kept), "dropped_empty": len(pairs) - len(kept),
                      "by_split": Counter(p["split"] for p in kept), "texts": len(used), "windows": n_win}))


if __name__ == "__main__":
    main()
