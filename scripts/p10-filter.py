"""Phase 10 borrowing filters, and the human review of what they keep (10/05/2026).

    p10-filter.py features --raw R --patents P --borrowings B --out borrowings_feat.parquet
    p10-filter.py review   --feat F --raw R --max-lift X --max-cites D --n 50 --seed S --out review.txt

The first review (results/phase-10-analogy/DATA-REVIEW.md) found 1 real cross-domain borrowing in 5. Most of the rest
were the same technology filed under two sections (G-H software and electronics), or citation dumps. Two filters,
both from the citation network and the classification only, never from the abstracts' text. A text filter would
build a test that raw text loses by construction.

- **Field distance:** citation lift between the two patents' first CPC subclasses, over all 146 million citations
  (any category): lift = P(cited subclass | citing subclass) / P(cited subclass), the larger of the two directions.
  Routine pairings (G06F with H04L, A61K with C07D) cite each other far more than chance; real borrowings come from
  subclasses that rarely do.
- **Citation dumps:** the citing patent's total number of applicant citations (any section).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import textwrap
from pathlib import Path

import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location("p10data", Path(__file__).with_name("p10-data.py"))
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def cmd_features(a):
    import pyarrow.compute as pc
    raw = Path(a.raw)
    p = pl.read_parquet(a.patents)
    hi = int(p["patent_id"].max())
    # First (inventional) subclass per patent, as an index into SUB.
    sub_of, SUB = np.full(hi + 1, -1, dtype=np.int16), {}
    for b in D.batches(raw, "g_cpc_current", ["patent_id", "cpc_sequence", "cpc_subclass"], 1_000_000):
        ids, ok = D.ids_of(b.column(0))
        ok &= (ids <= hi) & (pc.cast(b.column(1), "int32").to_numpy(zero_copy_only=False) == 0)
        subs = b.column(2).to_pylist()
        for i in np.nonzero(ok)[0]:
            sub_of[ids[i]] = SUB.setdefault(subs[i], len(SUB))
    n = len(SUB)
    print(json.dumps({"subclasses": n}), flush=True)
    flow = np.zeros((n, n), dtype=np.int64)
    app_count = np.zeros(hi + 1, dtype=np.int32)
    rows = 0
    for b in D.batches(raw, "g_us_patent_citation", ["patent_id", "citation_patent_id", "citation_category"], 1_000_000):
        c1, ok1 = D.ids_of(b.column(0))
        c2, ok2 = D.ids_of(b.column(1))
        ok = ok1 & ok2 & (c1 <= hi) & (c2 <= hi)
        app = pc.fill_null(pc.equal(b.column(2), "cited by applicant"), False).to_numpy(zero_copy_only=False) & ok1 & (c1 <= hi)
        np.add.at(app_count, c1[app], 1)
        s1, s2 = sub_of[c1[ok]], sub_of[c2[ok]]
        k = (s1 >= 0) & (s2 >= 0)
        np.add.at(flow, (s1[k], s2[k]), 1)
        rows += b.num_rows
    print(json.dumps({"citation_rows": rows, "flows": int(flow.sum())}), flush=True)
    tot = flow.sum()
    row, col = flow.sum(1, keepdims=True), flow.sum(0, keepdims=True)
    lift = (flow + 1) * tot / ((row + n) * (col + n))  # add-one smoothing: an unseen pairing has lift near 1/n-ish, not 0
    names = sorted(SUB, key=SUB.get)
    b = pl.read_parquet(a.borrowings).unique(["citing", "cited"])
    # Citation dumps never pass review (round 1), so they are dropped before the subclass-pair join, which grows with
    # every pair kept: about 5 million borrowings times every subclass pairing would not fit in the laptop's memory.
    b = b.filter(pl.Series(app_count[b["citing"].to_numpy()] <= 100))
    print(json.dumps({"borrowings_without_dumps": b.height}), flush=True)

    # Round 2 (10/05/2026): round 1's first-subclass lift left 21 of 50 same-technology pairs, often filed under an odd
    # first subclass, and more random citations. Two more signals:
    # - all_lift: the largest lift over every pair of subclasses the two patents carry (any one routine pairing
    #   means the same technology);
    # - bridge: how many other patents in the citing patent's first subclass also cite the cited patent (a real
    #   borrowing recurs in the borrowing field; a stray citation does not).
    citing_ids, cited_ids = b["citing"].to_numpy(), b["cited"].to_numpy()
    need = np.zeros(hi + 1, dtype=bool)
    need[citing_ids] = True
    need[cited_ids] = True
    pid_l, sub_l = [], []
    for bt in D.batches(raw, "g_cpc_current", ["patent_id", "cpc_subclass"], 1_000_000):
        ids, ok = D.ids_of(bt.column(0))
        ok &= ids <= hi
        ok[ok] = need[ids[ok]]
        subs = bt.column(1).to_pylist()
        for i in np.nonzero(ok)[0]:
            if subs[i] in SUB:
                pid_l.append(ids[i])
                sub_l.append(SUB[subs[i]])
    allsub = pl.DataFrame({"p": np.array(pid_l, dtype=np.int64), "s": np.array(sub_l, dtype=np.int16)}).unique()
    del pid_l, sub_l
    pairs = (b.select("citing", "cited").join(allsub.rename({"p": "citing", "s": "s1"}), on="citing")
             .join(allsub.rename({"p": "cited", "s": "s2"}), on="cited"))
    s1a, s2a = pairs["s1"].to_numpy(), pairs["s2"].to_numpy()
    pairs = pairs.with_columns(l=pl.Series(np.maximum(lift[s1a, s2a], lift[s2a, s1a])))
    all_lift = pairs.group_by("citing", "cited").agg(pl.col("l").max().alias("all_lift"))
    del pairs
    is_cited = np.zeros(hi + 1, dtype=bool)
    is_cited[cited_ids] = True
    keys = []
    for bt in D.batches(raw, "g_us_patent_citation", ["patent_id", "citation_patent_id"], 1_000_000):
        c1, ok1 = D.ids_of(bt.column(0))
        c2, ok2 = D.ids_of(bt.column(1))
        ok = ok1 & ok2 & (c1 <= hi) & (c2 <= hi)
        c1, c2 = c1[ok], c2[ok]
        k = is_cited[c2] & (sub_of[c1] >= 0)
        u, n = np.unique(c2[k] * 1024 + sub_of[c1[k]], return_counts=True)
        keys.append(pl.DataFrame({"key": u, "n": n}))
    bridge = pl.concat(keys).group_by("key").agg(pl.col("n").sum())
    del keys
    b = b.join(all_lift, on=["citing", "cited"], how="left")
    b = (b.with_columns(key=pl.Series(cited_ids.astype(np.int64)) * 1024 + pl.Series(sub_of[citing_ids].astype(np.int64)))
         .join(bridge, on="key", how="left").with_columns((pl.col("n").fill_null(1) - 1).alias("bridge")).drop("key", "n"))
    s1, s2 = sub_of[b["citing"].to_numpy()], sub_of[b["cited"].to_numpy()]
    ok = (s1 >= 0) & (s2 >= 0)
    b = b.filter(pl.Series(ok))
    s1, s2 = s1[ok], s2[ok]
    out = b.with_columns(citing_sub=pl.Series([names[i] for i in s1]), cited_sub=pl.Series([names[i] for i in s2]),
                         lift=pl.Series(np.maximum(lift[s1, s2], lift[s2, s1])),
                         citing_app_cites=pl.Series(app_count[b["citing"].to_numpy()]))
    out.write_parquet(a.out)
    q = out["lift"].quantile
    print(json.dumps({"borrowings": out.height, "lift_quantiles": {x: round(q(x), 2) for x in (0.1, 0.25, 0.5, 0.75, 0.9)},
                      "all_lift_quantiles": {x: round(out["all_lift"].quantile(x), 2) for x in (0.1, 0.25, 0.5, 0.75, 0.9)},
                      "bridge_quantiles": {x: int(out["bridge"].quantile(x)) for x in (0.25, 0.5, 0.75, 0.9)},
                      "app_cites_quantiles": {x: int(out["citing_app_cites"].quantile(x)) for x in (0.5, 0.75, 0.9)}}), flush=True)


def abstracts(raw, ids):
    want, out = {str(i) for i in ids}, {}
    for b in D.batches(raw, "g_patent_abstract", ["patent_id", "patent_abstract"], 0):
        for pid, t in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            if pid in want:
                out[int(pid)] = t
    return out


def cmd_review(a):
    f = pl.read_parquet(a.feat)
    cond = ((pl.col("lift") <= a.max_lift) & (pl.col("citing_app_cites") <= a.max_cites)
            & (pl.col("all_lift") <= a.max_all_lift) & (pl.col("bridge") >= a.min_bridge))
    keep = f.filter(cond & pl.col("citing_date").dt.year().is_between(2013, 2017))
    # One pair per citing patent, so one prolific patent cannot fill the sample.
    keep = keep.sample(fraction=1.0, shuffle=True, seed=a.seed).unique("citing", keep="first", maintain_order=True)
    s = keep.head(a.n)
    A = abstracts(Path(a.raw), set(s["citing"].to_list()) | set(s["cited"].to_list()))
    yrs = f.filter(pl.col("citing_date").dt.year() >= 2018)
    kept_yrs = yrs.filter(cond)
    head = {"max_lift": a.max_lift, "max_cites": a.max_cites, "max_all_lift": a.max_all_lift, "min_bridge": a.min_bridge, "seed": a.seed,
            "kept_share": round(kept_yrs.height / yrs.height, 4), "kept_citing_2018_on": kept_yrs["citing"].n_unique(),
            "kept_citing_val_2018_19": kept_yrs.filter(pl.col("citing_date").dt.year().is_between(2018, 2019))["citing"].n_unique(),
            "kept_citing_test_2021_24": kept_yrs.filter(pl.col("citing_date").dt.year() >= 2021)["citing"].n_unique()}
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(head) + "\n\n")
        for k, r in enumerate(s.iter_rows(named=True), 1):
            fh.write(f"#{k}  {r['citing']} ({r['citing_date'].year}, {r['citing_sub']})  cites  {r['cited']} ({r['cited_date'].year}, "
                     f"{r['cited_sub']})  lift {r['lift']:.2f}  applicant cites {r['citing_app_cites']}\n")
            for lab, pid in (("CITING", r["citing"]), ("CITED ", r["cited"])):
                t = " ".join((A.get(pid) or "<no abstract>").split()[:80])
                fh.write(textwrap.fill(f"{lab}: {t}", 160) + "\n")
            fh.write("\n")
    print(json.dumps(head), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    for x in ("raw", "patents", "borrowings", "feat", "out"):
        ap.add_argument(f"--{x}")
    ap.add_argument("--max-lift", type=float, default=1e9)
    ap.add_argument("--max-cites", type=int, default=10**9)
    ap.add_argument("--max-all-lift", type=float, default=1e9)
    ap.add_argument("--min-bridge", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    {"features": cmd_features, "review": cmd_review}[a.cmd](a)


if __name__ == "__main__":
    main()
