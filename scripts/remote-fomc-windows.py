"""FOMC meeting transcripts as splitter input (runs on the GPU instance; 10/05/2026).

Source: ConvoKit ``fomc-corpus`` (Federal Open Market Committee transcripts, 1977-2008; US government work, public
domain): 268 meetings, about 108,000 utterances. Each meeting's utterances in order, speakers to letters (names in the
text too), cut into consecutive 6-turn windows with 2+ speakers and 20-200 words; a turn over 120 words is cut to its
first 120 (staff briefings run to pages). uid ``fomc-<meeting>/<k>``: the split group is the whole meeting.
"""

from __future__ import annotations

import argparse
import json
import re

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="jepa/fomc-windows.jsonl")
    args = ap.parse_args()
    from convokit import Corpus, download

    corpus = Corpus(filename=download("fomc-corpus"))
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for c in sorted(corpus.iter_conversations(), key=lambda c: c.id):
            utts = sorted(c.iter_utterances(), key=lambda u: (u.timestamp or 0, u.id))
            turns = [(u.speaker.id, " ".join((u.text or "").split()[:120])) for u in utts]
            turns = [(s, t) for s, t in turns if len(t.split()) >= 3]
            lt = {}
            for s, _ in turns:
                lt.setdefault(s, LETTERS[min(len(lt), len(LETTERS) - 1)])
            # Speakers are recorded as "CHAIRMAN GREENSPAN", "MR. KOHN": their surnames in the text become letters too.
            names = {s: s.split()[-1].strip(".").title() for s in lt}
            for s, letter in lt.items():
                if len(names[s]) >= 3:
                    turns = [(x, re.sub(rf"(?<![\w-]){re.escape(names[s])}(?![\w-])", letter, t, flags=re.I)) for x, t in turns]
            mid = re.sub(r"\W+", "", c.id)
            for k, s in enumerate(range(0, len(turns), 6)):
                win = turns[s : s + 6]
                words = sum(len(t.split()) for _, t in win)
                if len(win) >= 2 and len({x for x, _ in win}) >= 2 and 20 <= words <= 200:
                    f.write(json.dumps({"uid": f"fomc-{mid}/{k}", "kind": "original", "medium": "meeting",
                                        "turns": [[lt[x], t] for x, t in win]}, ensure_ascii=False) + "\n")
                    n += 1
    print(json.dumps({"windows": n, "meetings": len(list(corpus.iter_conversations()))}))


if __name__ == "__main__":
    main()
