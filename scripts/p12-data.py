"""Phase 12 data (results/phase-12-far-analogy/HYPOTHESES.md, 10/07/2026). Run by pipelines/phase12.

    python p12-data.py measure --data <dianoia-data> --config <config> --out <work/p12>
    python p12-data.py build   --data <dianoia-data> --config <config> --work <work/p12> [--smoke]
    python p12-data.py audit   --data <dianoia-data> --config <config> --work <work/p12> --out <audit.json> [--smoke]
    python p12-data.py handcheck --data <dianoia-data> --config <config> --work <work/p12> --out <handcheck.json>

measure scores every confirmed borrowing (phase 10's kept pairs and phase 11's training pairs) on two measures that are
not the encoder being trained: IDF-weighted word overlap (p10-label.py overlap_scorer) and cosine under a second
encoder (config measure_encoder). A pair below the pool median on both is far, above on both is near, else mixed.
Writes pool.parquet, and for the hand check review-pairs.parquet (far test and far training pairs, with a split
column) and review-text.jsonl.

build writes, for the GPU job (which has no polars):
  eval.jsonl         phase 10's queries whose pair is far (split dev from val, test from test) or near (split
                     test_near), with phase 10's 999 candidates and the full query abstract
  docs.jsonl         the full abstract of every cited patent and candidate in eval.jsonl
  train-far.jsonl    far pairs that share no patent with any phase 10 query, true source or candidate; each with 7 hard
  train-near.jsonl   negatives (phase 10's candidate rule, from the cited patent's own section); near is sampled to
                     the far set's size
  train-docs.jsonl   the text of every hard negative
Training text is cut to 180 words, as the pairs' own text was (p10-label.py WORDS), so a negative is never told apart
by its length. audit checks what HYPOTHESES.md fixes and exits 1 on any violation; training does not start without a
passing audit of these exact files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import polars as pl
import yaml


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L = _load("p10label", "p10-label.py")
D = L.D  # p10-data.py, loaded by p10-label.py
WORDS = L.WORDS


def _cut(t):
    return " ".join(t.split()[:WORDS])


def _key(t):
    """An abstract's identity by text: continuations carry the same abstract under a new patent number, and 50
    training pairs matched an evaluation query or true source this way though no patent number did (10/07/2026)."""
    return " ".join(t.lower().split()[:60])


def _keys_expr():
    return pl.col("abstract").str.to_lowercase().str.split(" ").list.head(60).list.join(" ").alias("key")


def _paths(a):
    cfg = yaml.safe_load(open(a.config))
    d = Path(a.data) / "work"
    return cfg, {k: d / v for k, v in cfg["inputs"].items()}


def _text(abstracts, ids):
    return dict(pl.read_parquet(abstracts).filter(pl.col("patent_id").is_in(list(ids))).select("patent_id", "abstract").iter_rows())


def cmd_measure(a):
    cfg, p = _paths(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = pl.read_parquet(p["keep"])
    ab = _text(p["abstracts"], set(kept["citing"].to_list()) | set(kept["cited"].to_list()))
    rows, missing = [], 0
    for r in kept.iter_rows(named=True):
        if r["citing"] in ab and r["cited"] in ab:
            rows.append({"id": f'{r["citing"]}-{r["cited"]}', "citing": r["citing"], "cited": r["cited"], "source": "p10",
                         "split": r["split"], "citing_text": _cut(ab[r["citing"]]), "cited_text": _cut(ab[r["cited"]])})
        else:
            missing += 1
    for r in map(json.loads, open(p["train"], encoding="utf-8")):
        rows.append({"id": r["id"], "citing": int(r["citing"]), "cited": int(r["cited"]), "source": "p11", "split": "train",
                     "citing_text": _cut(r["citing_text"]), "cited_text": _cut(r["cited_text"])})
    if len({r["id"] for r in rows}) != len(rows):
        sys.exit("a pair is in both phase 10's kept pairs and phase 11's training pairs")
    ov = L.overlap_scorer(rows)

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(cfg["measure_encoder"], device="cpu")
    texts = sorted({r[k] for r in rows for k in ("citing_text", "cited_text")})
    vec = dict(zip(texts, m.encode(texts, batch_size=a.batch, normalize_embeddings=True, show_progress_bar=False)))
    for r in rows:
        r["overlap"] = ov(r["citing"], r["cited"])
        r["cosine"] = round(float(vec[r["citing_text"]] @ vec[r["cited_text"]]), 4)
    pool = pl.DataFrame(rows)
    mo, mc = pool["overlap"].median(), pool["cosine"].median()
    lo_o, lo_c = pl.col("overlap") < mo, pl.col("cosine") < mc
    pool = pool.with_columns(pl.when(lo_o & lo_c).then(pl.lit("far")).when(~lo_o & ~lo_c).then(pl.lit("near"))
                             .otherwise(pl.lit("mixed")).alias("label"))
    pool.write_parquet(out / "pool.parquet")
    (out / "medians.json").write_text(json.dumps({"overlap": mo, "cosine": mc, "pairs": pool.height,
                                                  "missing_abstract": missing}, indent=1))

    # The hand check's inputs, in p10-label.py review's format: far test pairs (split "test") and far training pairs
    # (split "train"), drawn from by `review --split`.
    q = pl.read_parquet(p["queries"])
    qpairs = {(r["query"], r["cited"]): r["split"] for r in q.select("query", "cited", "split").iter_rows(named=True)}
    far = pool.filter(pl.col("label") == "far")
    rv = [{"citing": r["citing"], "cited": r["cited"],
           "split": "test" if qpairs.get((r["citing"], r["cited"])) == "test" else ("train" if r["split"] == "train" else "other")}
          for r in far.iter_rows(named=True)]
    pl.DataFrame(rv).write_parquet(out / "review-pairs.parquet")
    with open(out / "review-text.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in far.iter_rows(named=True):
            f.write(json.dumps({k: r[k] for k in ("id", "citing", "cited", "citing_text", "cited_text")}) + "\n")
    print(json.dumps({"pairs": pool.height, "missing_abstract": missing, "median_overlap": mo, "median_cosine": mc,
                      "label": dict(pool.group_by("label").len().iter_rows()),
                      "by_split": {f"{s}/{l}": n for s, l, n in pool.group_by("split", "label").len().sort("split", "label").iter_rows()},
                      "spearman_overlap_cosine": round(pool.select(pl.corr("overlap", "cosine", method="spearman")).item(), 3)}), flush=True)


def _held(q):
    return set(q["query"].to_list()) | set(q["cited"].to_list()) | {c for cs in q["candidates"].to_list() for c in cs}


def cmd_build(a):
    cfg, p = _paths(a)
    w = Path(a.work)
    out = w / "smoke-in" if a.smoke else w
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])
    pool = pl.read_parquet(w / "pool.parquet")
    label = {(r["citing"], r["cited"]): r["label"] for r in pool.select("citing", "cited", "label").iter_rows(named=True)}
    q = pl.read_parquet(p["queries"])
    held = _held(q)  # every phase 10 query, its true source and its candidates, far or not

    # Eval: phase 10's queries, by their pair's label.
    ev = []
    for r in q.iter_rows(named=True):
        lab = label.get((r["query"], r["cited"]))
        split = {("val", "far"): "dev", ("test", "far"): "test", ("test", "near"): "test_near"}.get((r["split"], lab))
        if split:
            ev.append({"split": split, "query": r["query"], "cited": r["cited"], "candidates": r["candidates"]})
    n_test = sum(r["split"] == "test" for r in ev)
    if n_test < cfg["min_far_test"] and not a.smoke:
        sys.exit(f"STOP (HYPOTHESES.md): {n_test} far test queries, fewer than {cfg['min_far_test']}; ask Shaun before changing the rule")

    # Training pairs: every far or near pair sharing no patent with a phase 10 query, true source or candidate, by
    # number or by text.
    ab_keys = pl.read_parquet(p["abstracts"]).select("patent_id", _keys_expr())
    held_keys = set(ab_keys.filter(pl.col("patent_id").is_in(list(held)))["key"].to_list())
    ok = pool.filter(pl.col("label").is_in(["far", "near"]) & ~pl.col("citing").is_in(list(held)) & ~pl.col("cited").is_in(list(held)))
    n_num = ok.height
    ok = ok.filter(~pl.col("citing_text").map_elements(_key, return_dtype=pl.String).is_in(list(held_keys))
                   & ~pl.col("cited_text").map_elements(_key, return_dtype=pl.String).is_in(list(held_keys)))
    dropped_text = n_num - ok.height
    far = sorted(ok.filter(pl.col("label") == "far").iter_rows(named=True), key=lambda r: r["id"])
    near = sorted(ok.filter(pl.col("label") == "near").iter_rows(named=True), key=lambda r: r["id"])
    if len(near) < len(far):
        sys.exit(f"{len(near)} near pairs cannot match {len(far)} far ones (HYPOTHESES.md: equal sizes)")
    near = rng.sample(near, len(far))
    rng.shuffle(far)
    if a.smoke:  # code paths only
        sm = cfg["smoke"]
        far, near = far[:sm["train"]], near[:sm["train"]]
        ev = [r for s in ("dev", "test", "test_near") for r in [x for x in ev if x["split"] == s][:sm["queries"]]]
        ev = [{**r, "candidates": r["candidates"][:sm["candidates"]]} for r in ev]

    # Hard negatives (phase 10's candidate rule, p10-data.py finalize): the cited patent's first section, granted
    # before the citing patent, sharing no section with it, not cited by it in any category; abstract on disk. Never a
    # phase 10 eval patent and never one end of a training pair.
    pats = pl.read_parquet(p["patents"])
    train = far + near
    ends = {r["citing"] for r in train} | {r["cited"] for r in train}
    info = {r["patent_id"]: r for r in pats.filter(pl.col("patent_id").is_in(list(ends))).iter_rows(named=True)}
    citing = {r["citing"] for r in train}
    cites = (D.scan(Path(a.data) / "raw/patentsview", "g_us_patent_citation")
             .select(D.num("patent_id").alias("citing"), D.num("citation_patent_id").alias("c"))
             .filter(pl.col("citing").is_in(list(citing))).drop_nulls().collect(engine="streaming"))
    excl = {}
    for ci, c in cites.iter_rows():
        excl.setdefault(ci, set()).add(c)
    end_keys = {_key(r[k]) for r in train for k in ("citing_text", "cited_text")}
    have = ab_keys.filter(~pl.col("key").is_in(list(held_keys | end_keys)))["patent_id"]
    cand = (pats.join(pl.DataFrame({"patent_id": have}), on="patent_id")
            .filter(~pl.col("patent_id").is_in(list(held | ends)) & pl.col("section").is_in(list(D.LETTERS[:8])))
            .sort("patent_id"))
    by_sec = {s: list(zip(g["patent_id"].to_list(), g["date"].to_list(), g["mask"].to_list()))
              for (s,), g in cand.group_by("section")}
    k = cfg["negatives"]
    for r in train:
        ci, cd = info[r["citing"]], info[r["cited"]]
        r["year"] = ci["date"].year
        sec, ex, got, tries = by_sec[cd["section"]], excl.get(r["citing"], set()), [], 0
        while len(got) < k:
            tries += 1
            if tries > 100000:
                sys.exit(f"no {k} negatives for {r['id']}")
            pid, dt, mk = sec[rng.randrange(len(sec))]
            if dt < ci["date"] and (mk & ci["mask"] & D.DISTANT) == 0 and pid not in ex and pid not in got:
                got.append(pid)
        r["negatives"] = got
    negs = {n for r in train for n in r["negatives"]}
    need = {r["query"] for r in ev} | {r["cited"] for r in ev} | {c for r in ev for c in r["candidates"]} | negs
    text = _text(p["abstracts"], need)

    for name, rows in (("train-far", far), ("train-near", near)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps({x: r[x] for x in ("id", "citing", "cited", "year", "label", "overlap", "cosine",
                                                       "citing_text", "cited_text", "negatives")}) + "\n")
    with open(out / "train-docs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in sorted(negs):
            f.write(json.dumps({"id": i, "text": _cut(text[i])}) + "\n")
    meas = {(r["citing"], r["cited"]): (r["overlap"], r["cosine"]) for r in pool.iter_rows(named=True)}
    with open(out / "eval.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in ev:
            o, c = meas[(r["query"], r["cited"])]
            f.write(json.dumps({**r, "query_text": text[r["query"]], "overlap": o, "cosine": c}) + "\n")
    docs = {r["cited"] for r in ev} | {c for r in ev for c in r["candidates"]}
    with open(out / "docs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in sorted(docs):
            f.write(json.dumps({"id": i, "text": text[i]}) + "\n")
    print(json.dumps({"train_far": len(far), "train_near": len(near), "negatives": len(negs), "docs": len(docs),
                      "eval": {s: sum(r["split"] == s for r in ev) for s in ("dev", "test", "test_near")},
                      "far_dropped_shared_patent": pool.filter(pl.col("label") == "far").height - len(far) if not a.smoke else None,
                      "far_or_near_dropped_shared_text": dropped_text}), flush=True)


def _sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def cmd_audit(a):
    cfg, p = _paths(a)
    w = Path(a.work)
    src = w / "smoke-in" if a.smoke else w
    rd = lambda n: [json.loads(l) for l in open(src / n, encoding="utf-8")]
    far, near, ev = rd("train-far.jsonl"), rd("train-near.jsonl"), rd("eval.jsonl")
    tdocs = {r["id"]: r["text"] for r in rd("train-docs.jsonl")}
    docs = {r["id"]: r["text"] for r in rd("docs.jsonl")}
    med = json.loads((w / "medians.json").read_text())
    q = pl.read_parquet(p["queries"])
    held = _held(q)
    pats = {r["patent_id"]: r for r in pl.read_parquet(p["patents"]).filter(pl.col("patent_id").is_in(
        list({x for r in far + near for x in (r["citing"], r["cited"], *r["negatives"])}))).iter_rows(named=True)}
    rule = {"far": lambda r: r["overlap"] < med["overlap"] and r["cosine"] < med["cosine"],
            "near": lambda r: r["overlap"] >= med["overlap"] and r["cosine"] >= med["cosine"]}
    train = far + near
    k = cfg["negatives"]

    def neg_bad(r):
        ci, cd = pats[r["citing"]], pats[r["cited"]]
        return sum(n in held or n in (r["citing"], r["cited"]) or n not in tdocs or pats[n]["section"] != cd["section"]
                   or pats[n]["date"] >= ci["date"] or (pats[n]["mask"] & ci["mask"] & D.DISTANT) != 0 for n in r["negatives"])

    ekeys = {_key(r["query_text"]) for r in ev} | {_key(t) for t in docs.values()}
    bad = {
        "train_shares_patent_with_eval": sum(r["citing"] in held or r["cited"] in held for r in train),
        "train_shares_text_with_eval": sum(_key(r["citing_text"]) in ekeys or _key(r["cited_text"]) in ekeys for r in train)
                                       + sum(_key(t) in ekeys for t in tdocs.values()),
        "negative_breaks_rule_or_is_held": sum(neg_bad(r) for r in train),
        "negative_count_wrong": sum(len(set(r["negatives"])) != k for r in train),
        "train_duplicate_pair": len(train) - len({r["id"] for r in train}),
        "far_and_near_overlap": len({r["id"] for r in far} & {r["id"] for r in near}),
        "train_empty_text": sum(not r["citing_text"].strip() or not r["cited_text"].strip() for r in train)
                            + sum(not t.strip() for t in tdocs.values()),
        "far_rule_broken": sum(not (r["label"] == "far" and rule["far"](r)) for r in far),
        "near_rule_broken": sum(not (r["label"] == "near" and rule["near"](r)) for r in near),
        "near_size_differs": int(len(near) != len(far)),
        "eval_label_rule_broken": sum(not rule["near" if r["split"] == "test_near" else "far"](r) for r in ev),
        "eval_missing_doc": sum(c not in docs or not docs[c].strip() for r in ev for c in [r["cited"], *r["candidates"]]),
        "eval_empty_query": sum(not r["query_text"].strip() for r in ev),
        "eval_cited_among_candidates": sum(r["cited"] in r["candidates"] for r in ev),
        "eval_wrong_split": sum(r["split"] not in ("dev", "test", "test_near") for r in ev),
    }
    if not a.smoke:
        bad["far_test_too_few"] = int(sum(r["split"] == "test" for r in ev) < cfg["min_far_test"])
    files = ("train-far.jsonl", "train-near.jsonl", "train-docs.jsonl", "eval.jsonl", "docs.jsonl")
    res = {"pass": all(v == 0 for v in bad.values()), "violations": bad, "train_far": len(far), "train_near": len(near),
           "eval": {s: sum(r["split"] == s for r in ev) for s in ("dev", "test", "test_near")},
           "medians": med, "sha256": {f: _sha(src / f) for f in files}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    sys.exit(0 if res["pass"] else 1)


def cmd_handcheck(a):
    """The gate on the hand check (HYPOTHESES.md): reads p10-label.py review-score's output for the far test pairs
    (<work>/review-test-score.json) and the far training pairs (review-train-score.json). Exits 1 unless the test
    reading counts (rater agreement at least 9/10) and at least min_real of its pairs are real borrowings (C or G)."""
    cfg, _ = _paths(a)
    w = Path(a.work)
    res = {s: json.loads((w / f"review-{s}-score.json").read_text()) for s in ("test", "train")}
    t = res["test"]
    real = t["kept"] / t["new"] if "kept" in t else None
    res["test_real_share"] = None if real is None else round(real, 3)
    res["pass"] = real is not None and real >= cfg["review"]["min_real"]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    sys.exit(0 if res["pass"] else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("measure", "build", "audit", "handcheck"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--work")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"measure": cmd_measure, "build": cmd_build, "audit": cmd_audit, "handcheck": cmd_handcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
