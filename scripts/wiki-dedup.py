"""One copy of each Wikipedia talk-page discussion (10/04/2026).

ConvoKit's wiki corpus holds the same exchange under several conversation ids (editors copy a discussion onto both
users' talk pages), so splitting by conversation id put copies of one discussion in train and test: the J7 audit found
192 held-out windows duplicated in training. Conversations that share at least 2 long turns (30+ characters,
normalized; the audit's duplicate rule) or at least 3 long ideas (near-copies edited by a word) are joined into one
cluster, and only the cluster's lowest-id conversation is kept. Windows holding a template message (a long line found
in 5+ conversations) are dropped. Writes ``jepa/wiki-windows-dedup.jsonl`` and ``jepa/pred-wiki-dedup.jsonl``.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from dianoia.paths import work_path


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def main() -> None:
    wins = [json.loads(l) for l in open(work_path("jepa", "wiki-windows.jsonl"), encoding="utf-8")]
    conv = lambda uid: uid.split("/")[0]
    turns_of = defaultdict(set)
    for w in wins:
        turns_of[conv(w["uid"])] |= {norm(t) for _, t in w["turns"] if len(norm(t)) >= 30}
    by_turn = defaultdict(set)
    for c, ts in turns_of.items():
        for t in ts:
            by_turn[t].add(c)
    parent = {c: c for c in turns_of}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    pairs = Counter()
    for t, cs in by_turn.items():
        cs = sorted(cs)
        if 1 < len(cs) <= 50:  # a line shared by 50+ conversations is boilerplate, not a copy
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    pairs[(cs[i], cs[j])] += 1
    # Near-copies (a discussion copied with small edits) share ideas even where their turns differ by a word: the
    # J7 audit's second pass found 22. Count shared long ideas (8+ words) per conversation pair too.
    ideas_of = defaultdict(set)
    for l in open(work_path("jepa", "pred-wiki.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        ideas_of[conv(r["uid"])] |= {norm(i) for i in r["ideas"] if len(i.split()) >= 8}
    by_idea = defaultdict(set)
    for c, ids in ideas_of.items():
        for i in ids:
            by_idea[i].add(c)
    idea_pairs = Counter()
    for i, cs in by_idea.items():
        cs = sorted(cs)
        if 1 < len(cs) <= 50:
            for x in range(len(cs)):
                for y in range(x + 1, len(cs)):
                    idea_pairs[(cs[x], cs[y])] += 1
    for (a, b), k in list(pairs.items()) + [((a, b), 99) for (a, b), k in idea_pairs.items() if k >= 3]:
        if k >= 2 and a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb, key=lambda x: int(x.split("-")[1]))] = min(ra, rb, key=lambda x: int(x.split("-")[1]))
    keep = {c for c in turns_of if find(c) == c}
    clusters = Counter(find(c) for c in turns_of)
    # Template messages (a proposed-deletion notice, a welcome message) are posted word for word in many unrelated
    # conversations: boilerplate a predictor could learn by heart, like the IRC bots' factoids. A long line found in 5+
    # conversations is a template, and a window holding one is dropped.
    templates = {t for t, cs in by_turn.items() if len(cs) >= 5}
    bad_windows = {w["uid"] for w in wins if any(norm(t) in templates for _, t in w["turns"])}
    for name, src in (("wiki-windows", "wiki-windows.jsonl"), ("pred-wiki", "pred-wiki.jsonl")):
        with open(work_path("jepa", f"{name}-dedup.jsonl"), "w", encoding="utf-8") as f:
            for l in open(work_path("jepa", src), encoding="utf-8"):
                uid = json.loads(l)["uid"]
                if conv(uid) in keep and uid not in bad_windows:
                    f.write(l)
    print(json.dumps({"conversations": len(turns_of), "kept": len(keep), "dropped_copies": len(turns_of) - len(keep),
                      "largest_cluster": max(clusters.values()), "template_lines": len(templates),
                      "windows_dropped_for_templates": len(bad_windows)}))


if __name__ == "__main__":
    main()
