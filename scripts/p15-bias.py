"""Phase 15 step 2 follow-up (descriptive, asked for 10/09/2026 after S1 failed): is Fastino a fair ranker?

    python p15-bias.py --work <work> --out <work/p15/out> --results <results/phase-15-curve-screen> [--top 0.2]

Reading only Fastino's top 20% finds about 71% of the strict positives. This asks whether the positives it keeps are
nearer analogies than the ones it misses (the risk: a ranker that favors obvious pairs would push the training set
toward near analogies), and whether a free ranker does as well. Nearness is phase 14's own pair measures: word
overlap and the off-the-shelf cosine (higher = nearer). Writes BIAS.md and bias.json; no verdict, nothing was
pre-registered.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def boot_diff(a, b, n=10000, seed=15):
    rng = random.Random(seed)
    m = lambda x: sum(x) / len(x)
    ds = sorted(m(rng.choices(a, k=len(a))) - m(rng.choices(b, k=len(b))) for _ in range(n))
    return {"diff": round(m(a) - m(b), 4), "lo": round(ds[int(0.025 * n)], 4), "hi": round(ds[int(0.975 * n)], 4)}


def main():
    ap = argparse.ArgumentParser()
    for x in ("work", "out", "results"):
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--top", type=float, default=0.2)
    a = ap.parse_args()
    s = pd.DataFrame(map(json.loads, open(Path(a.out) / "screen/scores.jsonl", encoding="utf-8")))
    meta = pd.read_parquet(Path(a.work) / "p14/far.parquet", columns=["id", "cited_field", "cited_domain", "overlap", "cosine"])
    d = s.merge(meta, on="id", how="left")
    assert d["overlap"].notna().all(), "test pairs missing from far.parquet"
    N, k = len(d), int(a.top * len(d))
    rank = lambda col: set(d.sort_values([col, "id"], ascending=[False, True]).head(k)["id"])
    rankers = {"Fastino": rank("p"), "word overlap (high first)": rank("overlap"), "cosine (high first)": rank("cosine")}
    d["neg_overlap"], d["neg_cosine"] = -d["overlap"], -d["cosine"]
    rankers |= {"word overlap (low first)": rank("neg_overlap"), "cosine (low first)": rank("neg_cosine")}
    res = {"test_pairs": N, "top_share": a.top, "read": k, "labels": {}}
    lines = ["# Phase 15 step 2 follow-up: is Fastino a fair ranker? (descriptive)", "",
             f"Asked on 10/09/2026 after S1 failed; nothing here was pre-registered, so there is no verdict. "
             f"{N:,} held-out test pairs; Sonnet reads the top {a.top:.0%} ({k:,}) by each ranker. Nearness is phase 14's "
             f"word overlap and off-the-shelf cosine between the two abstracts (higher = nearer).", ""]
    for y, name in (("y", "strict"), ("y3", "v3p gate")):
        pos = d[d[y] == 1]
        kept = pos[pos["id"].isin(rankers["Fastino"])]
        miss = pos[~pos["id"].isin(rankers["Fastino"])]
        r = {"positives": len(pos), "kept": len(kept),
             "found_by_ranker": {n: int(pos["id"].isin(ids).sum()) for n, ids in rankers.items()},
             "kept_minus_missed": {c: boot_diff(kept[c].tolist(), miss[c].tolist()) for c in ("overlap", "cosine")},
             "means": {g: {c: round(float(x[c].mean()), 4) for c in ("overlap", "cosine")}
                       for g, x in (("kept", kept), ("missed", miss), ("all_pairs", d))},
             "domains": {g: dict(Counter(x["cited_domain"])) for g, x in (("kept", kept), ("missed", miss))}}
        res["labels"][name] = r
        lines += [f"## {name.capitalize()} positives ({len(pos)})", "",
                  "| Ranker | Positives in its top %s | Share |" % f"{a.top:.0%}", "|---|---|---|"]
        lines += [f"| {n} | {v} | {v / len(pos):.1%} |" for n, v in r["found_by_ranker"].items()]
        lines += ["", "| | Kept by Fastino (%d) | Missed (%d) | Kept minus missed (95%% interval) | All test pairs |"
                  % (len(kept), len(miss)), "|---|---|---|---|---|"]
        for c in ("overlap", "cosine"):
            b = r["kept_minus_missed"][c]
            lines.append(f"| {c} | {r['means']['kept'][c]:.4f} | {r['means']['missed'][c]:.4f} | "
                         f"{b['diff']:+.4f} ({b['lo']:+.4f} to {b['hi']:+.4f}) | {r['means']['all_pairs'][c]:.4f} |")
        lines += ["", f"Cited domain, kept: {r['domains']['kept']}; missed: {r['domains']['missed']}.", ""]
    rho = {c: round(float(spearmanr(d["p"], d[c]).statistic), 4) for c in ("overlap", "cosine")}
    res["spearman_p_vs"] = rho
    lines.append(f"Rank correlation of Fastino's score with nearness over all test pairs: overlap {rho['overlap']:+.3f}, "
                 f"cosine {rho['cosine']:+.3f}.")
    (Path(a.results) / "bias.json").write_text(json.dumps(res, indent=1))
    (Path(a.results) / "BIAS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({n: {"kept": v["kept"], "of": v["positives"], "diff": v["kept_minus_missed"]} for n, v in res["labels"].items()}))


if __name__ == "__main__":
    main()
