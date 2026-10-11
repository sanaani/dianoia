"""Phase 10: judge the exclusion cascade on Claude's hand-labeled pool pairs (10/06/2026).

    python scripts/p10-cascade-check.py --stages p10/stage1/pool-scores.jsonl,p10/stage2/pool-scores.jsonl

Hand-labeled sets, none trained on (sonnetC50 and fresh50 are val-year pool pairs):
  sonnetC50 - gold2.jsonl: 50 random pool pairs Sonnet called real (any borrow score), read by hand (48% real)
  gold140 - gold.jsonl: 140 random filtered borrowings (2013-2017, excluded from training), with --gold-scores
  fresh50   - review-final-val2.txt: 50 pairs Sonnet kept (real, borrow >= 4), read with scores hidden
For each stage alone and for the cascade (a pair survives if every stage keeps it), it prints how many of each hand
class survive. A stage is good if it removes its target class and keeps nearly every real (C) pair. Hand labels only;
no model output is graded.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

W = Path.home() / "dianoia-data/work/p10"
FRESH_A, FRESH_B = {4, 11, 29, 34, 35, 36}, {18, 23, 37}  # DATA-REVIEW.md, fresh check of 10/06/2026


def hand_sets():
    raw = {r["id"]: r["label"] for r in map(json.loads, open(W / "label/gold2.jsonl", encoding="utf-8"))}
    ids = re.findall(r"^#(\d+) (\S+)", (W / "review-final-val2.txt").read_text(encoding="utf-8"), re.M)
    fresh = {pid: "A" if int(n) in FRESH_A else "B" if int(n) in FRESH_B else "C" for n, pid in ids}
    return {"sonnetC50": raw, "fresh50": fresh}


def row(name, labels, keep):
    k = [l for i, l in labels.items() if keep(i)]
    tot = {c: sum(l == c for l in labels.values()) for c in "ABC"}
    share = f"{k.count('C') / len(k):.0%} real, {k.count('A') / len(k):.0%} same-tech" if k else "-"
    return (f"  {name:26} kept {len(k):2}/{len(labels)}  C {k.count('C')}/{tot['C']}  A {k.count('A')}/{tot['A']}"
            f"  B {k.count('B')}/{tot['B']}   ({share})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True, help="comma-separated pool-scores.jsonl, in cascade order")
    ap.add_argument("--gold-scores", default="", help="each stage's scores of gold-all.jsonl (P10D6), same order")
    a = ap.parse_args()

    def load(files):
        out = []
        for f in files.split(","):
            p = Path(f) if Path(f).is_absolute() else W.parent / f
            out.append((p.parent.name, {r["id"]: r["p"] for r in map(json.loads, open(p, encoding="utf-8"))}))
        return out

    stages = load(a.stages)
    sets = hand_sets()
    if a.gold_scores:
        # The 140 raw hand labels; two adjudicated to C (DATA-REVIEW.md, 10/05/2026). Scored by every stage, so the
        # cascade line is the AND of the stages.
        adj = {"9105785-6538129": "C", "8877500-4349026": "C"}
        g = {r["id"]: adj.get(r["id"], r["label"]) for r in map(json.loads, open(W / "label/gold.jsonl", encoding="utf-8"))}
        gs = load(a.gold_scores)
        stages = [(n, {**s, **{i: gs[k][1][i] for i in g if i in gs[k][1]}}) for k, (n, s) in enumerate(stages)]
        sets = {"gold140 (raw)": g, **sets}
    for name, labels in sets.items():
        missing = [i for i in labels if any(i not in s for _, s in stages[:1])]
        print(f"{name} (unscored by the first stage: {len(missing)})")
        print(row("no filter", labels, lambda i: True))
        for sn, s in stages:
            print(row(f"{sn} alone", labels, lambda i, s=s: s.get(i, 0) >= 0.5))
        # A pair missing from a later stage's scores was removed by an earlier stage (it scores only survivors).
        print(row("cascade (all stages)", labels, lambda i: all(s.get(i, 0) >= 0.5 for _, s in stages)))


if __name__ == "__main__":
    main()
