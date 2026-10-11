"""WikiConv talk-page threads of one year as splitter input (runs on the GPU instance; 10/04/2026).

Source: ConvoKit ``wikiconv-<year>`` (English Wikipedia talk pages; CC BY-SA), unzipped under ``--src``. Only comments
posted in that year are in its file, so years 2016 onward cannot repeat ``wiki-corpus`` (2011 dumps) and two years never
repeat each other. Same cleaning, speaker letters and 6-comment windows as ``remote-wiki-windows.py``. Comments by bots
(a speaker name ending in "bot" or containing "bot " or "Bot ") are dropped before windowing: InternetArchiveBot alone
posted 307,717 of 2018's 3.9 million comments. uid ``wc-<page id>.<thread id>/<k>``: the split group is the whole page
(``splits.py``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location("wiki_windows", Path(__file__).with_name("remote-wiki-windows.py"))
ww = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ww)

BOT = re.compile(r"bot(?:\s|$)", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped corpus directory (utterances.jsonl, conversations.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = Path(args.src)
    pages = {c: (m.get("meta") or {}).get("page_id") for c, m in json.load(open(src / "conversations.json")).items()}
    threads, bots, short = defaultdict(list), 0, 0
    for l in open(src / "utterances.jsonl", encoding="utf-8"):
        r = json.loads(l)
        sp = r.get("speaker") or ""
        if BOT.search(sp):
            bots += 1
            continue
        t = ww.clean(r.get("text") or "")
        if len(t.split()) < 3:
            short += 1
            continue
        threads[r["conversation_id"]].append((r.get("timestamp") or 0, r["id"], sp, " ".join(t.split()[:120])))
    out, used = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for cid in sorted(threads):
            seen, turns = set(), []
            for _, _, sp, t in sorted(threads[cid]):
                if t not in seen:  # a comment restored or re-posted verbatim counts once
                    seen.add(t)
                    turns.append((sp, t))
            if len(turns) < 3 or len({s for s, _ in turns}) < 2:
                continue
            lt = {}
            for s, _ in turns:
                lt.setdefault(s, ww.LETTERS[min(len(lt), len(ww.LETTERS) - 1)])
            for name, letter in lt.items():
                if name and len(name) >= 3:
                    turns = [(s, re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])", letter, t)) for s, t in turns]
            page = pages.get(cid) or cid.split(".")[0]
            n0 = out
            for k, s in enumerate(range(0, len(turns), 6)):
                win = turns[s : s + 6]
                words = sum(len(t.split()) for _, t in win)
                if len(win) >= 2 and len({x for x, _ in win}) >= 2 and 20 <= words <= 200:
                    f.write(json.dumps({"uid": f"wc-{page}.{cid}/{k}", "kind": "original", "medium": "wiki",
                                        "turns": [[lt[x], t] for x, t in win]}, ensure_ascii=False) + "\n")
                    out += 1
            used += out > n0
    print(json.dumps({"windows": out, "threads": used, "bot_comments_dropped": bots, "short_dropped": short}))


if __name__ == "__main__":
    main()
