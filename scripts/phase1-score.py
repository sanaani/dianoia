"""Phase 1 score: does each encoder put the paraphrase nearer than the lexical twin?

For item i with source x, paraphrase p and twin t of one kind, a *win* is cos(x, p) > cos(x, t)
(a tie counts one half). The win rate over items is a paired AUC. Intervals are a 2,000-draw
bootstrap over items. Word Jaccard is scored the same way as the lexical control: the materials
were built so that it loses, and it must.

Verdict bars, fixed before scoring:
- holds ideas  : win rate >= 0.80 and the interval's lower end > 0.5
- partial      : lower end > 0.5, rate < 0.80
- tracks words : lower end <= 0.5
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from dianoia import corpus, encoders
from dianoia.paths import result_path, work_path

KINDS = ["negation", "role", "quantity", "substitution"]
B = 2000


def jaccard(a: str, b: str) -> float:
    wa, wb = {w.lower() for w in corpus.words(a)}, {w.lower() for w in corpus.words(b)}
    return len(wa & wb) / max(len(wa | wb), 1)


def win_rate(pos: np.ndarray, neg: np.ndarray, rng) -> dict:
    wins = (pos > neg) + 0.5 * (pos == neg)
    boots = [wins[rng.integers(0, len(wins), len(wins))].mean() for _ in range(B)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rate = float(wins.mean())
    verdict = "holds ideas" if rate >= 0.8 and lo > 0.5 else "partial" if lo > 0.5 else "tracks words"
    return {"n": int(len(wins)), "win_rate": round(rate, 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
            "margin": round(float((pos - neg).mean()), 4), "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=encoders.ENCODERS)
    args = ap.parse_args()
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    rng = np.random.default_rng(0)
    out: dict = {"n_items": len(items), "scores": {}}

    lsa_pool = pd.read_parquet(work_path("corpus", "sentences.parquet"))
    encoders.fit_lsa(lsa_pool.loc[~lsa_pool["boilerplate"], "text"].tolist())

    scores = {"jaccard": {}}
    for k in KINDS:
        sub = items[items[k].notna()]
        pos = np.array([jaccard(a, b) for a, b in zip(sub["text"], sub["paraphrase"])])
        neg = np.array([jaccard(a, b) for a, b in zip(sub["text"], sub[k])])
        scores["jaccard"][k] = win_rate(pos, neg, rng)

    for name in args.encoders:
        texts = items["text"].tolist()
        vx = encoders.encode(name, texts)
        vp = encoders.encode(name, items["paraphrase"].tolist())
        pos_all = (vx * vp).sum(1)
        scores[name] = {"paraphrase_cos_median": round(float(np.median(pos_all)), 4)}
        for k in KINDS:
            mask = items[k].notna().to_numpy()
            vt = encoders.encode(name, items.loc[mask, k].tolist())
            neg = (vx[mask] * vt).sum(1)
            scores[name][k] = win_rate(pos_all[mask], neg, rng)
        np.save(work_path("phase1", f"vec-{name}.npy"), vx)
        print(name, {k: scores[name][k]["win_rate"] for k in KINDS}, flush=True)

    out["scores"] = scores
    result_path("phase-1-ideas-not-words", "scores.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = [f"{'encoder':10s} " + " ".join(f"{k:>24s}" for k in KINDS)]
    for name, s in scores.items():
        cells = [f"{s[k]['win_rate']:.3f} [{s[k]['ci'][0]:.2f},{s[k]['ci'][1]:.2f}]" for k in KINDS]
        lines.append(f"{name:10s} " + " ".join(f"{c:>24s}" for c in cells))
    table = "\n".join(lines)
    result_path("phase-1-ideas-not-words", "scores.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
