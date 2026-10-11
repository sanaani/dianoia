"""Wikipedia talk-page conversations as splitter input for the next-idea predictor (runs on the GPU instance;
10/04/2026). Output ``jepa/wiki-windows.jsonl`` in ``jepa-units.py``'s format, so ``jepa-chain.py`` can chain it.

Source: ConvoKit's Wikipedia Talk Pages Corpus (``wiki-corpus``: 125,292 conversations, 391,294 comments; CC BY-SA
4.0). Written group discussion: editors argue, decide and coordinate. Each conversation's comments in time order,
wiki markup removed (links keep their label, templates and signatures go, URLs become "Link 1"), one speaker-letter
mapping per conversation, cut into consecutive windows of 6 comments. A window needs 2 speakers and 20-200 words;
comments over 120 words are cut to their first 120. Conversations are sampled with a seed up to ``--max`` windows.
"""

from __future__ import annotations

import argparse
import json
import random
import re

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def clean(t: str) -> str:
    t = t.replace("\\/", "/")  # links stored with escaped slashes slipped past the URL rules (10/04)
    t = re.sub(r"\{\{[^{}]*\}\}", " ", t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t)
    t = re.sub(r"https?://\S+", "Link 1", t)
    t = re.sub(r"\d\d:\d\d, \d+ \w+ \d{4} \(UTC\)", " ", t)
    t = re.sub(r"'{2,}|={2,}|<[^>]+>|~{3,}", " ", t)
    t = re.sub(r"^[:*#]+", " ", t, flags=re.M)
    return re.sub(r"\s+", " ", t).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=20000)
    ap.add_argument("--out", default="jepa/wiki-windows.jsonl")
    args = ap.parse_args()
    from convokit import Corpus, download

    corpus = Corpus(filename=download("wiki-corpus"))
    convs = list(corpus.iter_conversations())
    random.Random(21).shuffle(convs)
    out = []
    for c in convs:
        utts = sorted(c.iter_utterances(), key=lambda u: (u.timestamp or 0, u.id))
        turns = []
        for u in utts:
            t = clean(u.text or "")
            if len(t.split()) >= 3:
                turns.append((u.speaker.id, " ".join(t.split()[:120])))
        if len(turns) < 3 or len({s for s, _ in turns}) < 2:
            continue
        lt = {}
        for s, _ in turns:
            lt.setdefault(s, LETTERS[min(len(lt), len(LETTERS) - 1)])
        # A username in the text becomes its speaker letter, as nicknames do in the IRC windows.
        for name, letter in lt.items():
            if name and len(name) >= 3:
                turns = [(s, re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])", letter, t)) for s, t in turns]
        for k, s in enumerate(range(0, len(turns), 6)):
            win = turns[s : s + 6]
            words = sum(len(t.split()) for _, t in win)
            if len(win) >= 2 and len({x for x, _ in win}) >= 2 and 20 <= words <= 200:
                out.append({"uid": f"wiki-{c.id}/{k}", "kind": "original", "medium": "wiki", "turns": [[lt[x], t] for x, t in win]})
        if len(out) >= args.max:
            break
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out[: args.max]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"windows": min(len(out), args.max), "conversations": len({r["uid"].split("/")[0] for r in out[: args.max]})}))


if __name__ == "__main__":
    main()
