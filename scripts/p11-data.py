"""Phase 11 data (results/phase-11-analogy-training/HYPOTHESES.md, 10/07/2026).

    python p11-data.py build --data <dianoia-data> --config <config> --out <work/p11> [--smoke]
    python p11-data.py audit --data <dianoia-data> --config <config> --work <work/p11> --out <audit.json>

build writes, for the GPU job (which has no polars):
  train.jsonl  the training pairs (p10-label.py train), shuffled once with the config seed; a size-k model trains on
               the first k lines, so the 1k set is inside the 3k set, which is inside N
  eval.jsonl   phase 10's queries (queries.parquet: the same queries, true cited patents and 999 candidates), with the
               citing abstract and the pair's word overlap (kept-final.parquet)
  docs.jsonl   the abstract of every cited patent and candidate
audit checks what HYPOTHESES.md fixes about the split and exits 1 on any violation; training does not start without
a passing audit of these exact files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import polars as pl
import yaml


def _paths(a):
    cfg = yaml.safe_load(open(a.config))
    d = Path(a.data) / "work"
    return cfg, {k: d / v for k, v in cfg["inputs"].items()}


def cmd_build(a):
    cfg, p = _paths(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train = [json.loads(l) for l in open(p["train"], encoding="utf-8")]
    random.Random(cfg["seed"]).shuffle(train)
    q = pl.read_parquet(p["queries"])
    keep = pl.read_parquet(p["keep"])
    overlap = {(r["citing"], r["cited"]): r["overlap"] for r in keep.iter_rows(named=True)}
    if a.smoke:  # code paths only: a few pairs and queries, few candidates
        sm = cfg["smoke"]
        train = train[:sm["train"]]
        q = pl.concat([q.filter(pl.col("split") == s).head(sm["queries"]) for s in ("val", "test")])
        q = q.with_columns(pl.col("candidates").list.head(sm["candidates"]))
    ab = pl.read_parquet(p["abstracts"])
    need = set(q["query"].to_list()) | set(q["cited"].to_list()) | {c for cs in q["candidates"].to_list() for c in cs}
    text = dict(ab.filter(pl.col("patent_id").is_in(list(need))).select("patent_id", "abstract").iter_rows())
    with open(out / "train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in train:
            f.write(json.dumps({"id": r["id"], "citing": r["citing"], "cited": r["cited"], "year": r["year"],
                                "citing_text": r["citing_text"], "cited_text": r["cited_text"]}) + "\n")
    with open(out / "eval.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in q.iter_rows(named=True):
            f.write(json.dumps({"split": r["split"], "query": r["query"], "cited": r["cited"], "candidates": r["candidates"],
                                "query_text": text[r["query"]], "overlap": overlap.get((r["query"], r["cited"]))}) + "\n")
    # Every cited patent and candidate, including one that is also a query: one query's patent is another's cited
    # source, and leaving out all query patents dropped it (the audit caught it, 10/07/2026).
    docs = set(q["cited"].to_list()) | {c for cs in q["candidates"].to_list() for c in cs}
    with open(out / "docs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in sorted(docs):
            f.write(json.dumps({"id": i, "text": text[i]}) + "\n")
    print(json.dumps({"train": len(train), "eval": dict(q.group_by("split").len().iter_rows()), "docs": len(docs)}), flush=True)


def _sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def cmd_audit(a):
    cfg, p = _paths(a)
    w = Path(a.work)
    train = [json.loads(l) for l in open(w / "train.jsonl", encoding="utf-8")]
    ev = [json.loads(l) for l in open(w / "eval.jsonl", encoding="utf-8")]
    docs = {r["id"]: r["text"] for r in map(json.loads, open(w / "docs.jsonl", encoding="utf-8"))}
    held = set()
    for r in pl.read_parquet(p["keep"]).iter_rows(named=True):
        held |= {r["citing"], r["cited"]}
    for r in ev:  # the queries' own patents, in case a query is not in kept-final
        held |= {r["query"], r["cited"]}
    years = set(cfg["train_years"])
    bad = {
        "train_shares_patent_with_eval": sum(int(r["citing"]) in held or int(r["cited"]) in held for r in train),
        "train_year_not_allowed": sum(r["year"] not in years for r in train),
        "train_duplicate_pair": len(train) - len({r["id"] for r in train}),
        "train_empty_text": sum(not r["citing_text"].strip() or not r["cited_text"].strip() for r in train),
        "eval_missing_doc": sum(c not in docs for r in ev for c in [r["cited"], *r["candidates"]]),
        "eval_empty_query": sum(not r["query_text"].strip() for r in ev),
        "eval_cited_among_candidates": sum(r["cited"] in r["candidates"] for r in ev),
        "eval_wrong_split": sum(r["split"] not in ("val", "test") for r in ev),
    }
    if not a.smoke:
        bad["train_too_small"] = int(len(train) < max(s for s in cfg["sizes"] if isinstance(s, int)))
    res = {"pass": all(v == 0 for v in bad.values()), "violations": bad, "train": len(train),
           "eval": {s: sum(r["split"] == s for r in ev) for s in ("val", "test")},
           "sha256": {f: _sha(w / f) for f in ("train.jsonl", "eval.jsonl", "docs.jsonl")}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    sys.exit(0 if res["pass"] else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "audit"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--work")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"build": cmd_build, "audit": cmd_audit}[a.cmd](a)


if __name__ == "__main__":
    main()
