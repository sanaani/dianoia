"""Phase 10 data: cross-domain borrowings in US patents, from PatentsView (10/05/2026). Run by pipelines/phase10.

    p10-data.py patents     --raw R --out patents.parquet
    p10-data.py borrowings  --raw R --patents P --batch N --out borrowings.parquet
    p10-data.py sample      --raw R --patents P --borrowings B --config C --out DIR      (draft.parquet, needed.parquet)
    p10-data.py abstracts   --raw R --needed N --batch N --out abstracts.parquet
    p10-data.py finalize    --draft D --abstracts A --config C --out queries.parquet
    p10-data.py audit       --queries Q --needed N --abstracts A --patents P --out audit.json

A borrowing is an applicant citation (not an examiner's) from a patent to one that shares none of its CPC sections. A
query is a citing patent; its answer is the cited one, hidden among 999 candidates from the cited patent's own first
section, all granted before the query and before the pool cutoff, none sharing a section with the query, none cited by
it. Queries are split by grant date (val 2018-2019, test 2021-2024).

Each table is converted once to parquet (table()) and then read lazily by polars, so no table is ever held whole in
memory (the citation table is about 150 million rows). Patent ids are integers (utility patents only; design and reissue ids carry letters and are dropped).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

import polars as pl
import yaml

LETTERS = "ABCDEFGHY"
DISTANT = 0xFF  # bits for A-H; Y (a cross-sectional tag) never makes two patents "near"


# The columns each table is converted with (all kept as strings; scripts cast what they use).
COLUMNS = {
    "g_patent": ["patent_id", "patent_type", "patent_date", "withdrawn"],
    "g_cpc_current": ["patent_id", "cpc_sequence", "cpc_section", "cpc_subclass"],
    "g_us_patent_citation": ["patent_id", "citation_patent_id", "citation_category"],
    "g_patent_abstract": ["patent_id", "patent_abstract"],
}


def table(raw: Path, name: str) -> Path:
    """<name>.tsv.zip as parquet, converted once, streamed straight from the zip by pyarrow in 16 MB blocks (written
    via .part, so a killed conversion is never mistaken for a whole one).

    Not polars' CSV reader: on Windows it allocated over 4 GB on the 3 GB CPC table even when only streaming it to
    parquet, and memcap killed it every time; pyarrow's reader peaked at 2 GB (10/05/2026)."""
    import pyarrow as pa
    import pyarrow.csv as pc
    import pyarrow.parquet as pq
    out = raw / "parquet" / f"{name}.parquet"
    if out.exists():
        return out
    out.parent.mkdir(exist_ok=True)
    part = out.with_suffix(".part")
    cols = COLUMNS[name]
    n = 0
    with zipfile.ZipFile(raw / f"{name}.tsv.zip") as zf, zf.open(zf.infolist()[0]) as src:
        r = pc.open_csv(src, read_options=pc.ReadOptions(block_size=16 << 20, use_threads=False),
                        parse_options=pc.ParseOptions(delimiter="\t", quote_char='"', newlines_in_values=True),
                        convert_options=pc.ConvertOptions(include_columns=cols, column_types={c: pa.string() for c in cols}))
        with pq.ParquetWriter(part, r.schema) as w:
            for batch in r:
                w.write_batch(batch)
                n += batch.num_rows
    part.replace(out)
    print(json.dumps({"converted": name, "rows": n}), flush=True)
    return out


def scan(raw: Path, name: str) -> pl.LazyFrame:
    return pl.scan_parquet(table(raw, name), low_memory=True)


def num(col: str) -> pl.Expr:
    """Patent ids as integers (utility patents are all numeric); non-numeric ids become null."""
    return pl.col(col).cast(pl.Int64, strict=False)


def batches(raw: Path, name: str, cols, size):
    """One row group at a time (16 MB each). iter_batches buffered the whole file: 3 GB on the abstracts table, against
    563 MB this way (10/05/2026). <size> is kept for the callers; row groups already bound memory."""
    import pyarrow.parquet as pq
    f = pq.ParquetFile(table(raw, name), pre_buffer=False)
    for i in range(f.metadata.num_row_groups):
        yield from f.read_row_group(i, columns=cols).to_batches()


def ids_of(col):
    """Numeric patent ids as int64 and the mask of rows that had one (design and reissue ids carry letters)."""
    import numpy as np
    import pyarrow.compute as pc
    ok = pc.fill_null(pc.match_substring_regex(col, r"^\d+$"), False)
    out = np.zeros(len(col), dtype=np.int64)
    okn = ok.to_numpy(zero_copy_only=False)
    out[okn] = pc.cast(pc.filter(col, ok), "int64").to_numpy()
    return out, okn


def cmd_patents(a) -> None:
    """Patent number as array index: utility numbers are dense integers below 13 million, so per-patent date, first
    section and section mask are three small arrays filled batch by batch. polars' join and group_by on the 57
    million CPC rows allocated over 5 GB and memcap killed it at every batch size (10/05/2026); this stays under 1 GB."""
    import numpy as np
    import pyarrow.compute as pc
    raw = Path(a.raw)
    hi = 0
    for b in batches(raw, "g_patent", ["patent_id"], a.batch):
        ids, ok = ids_of(b.column(0))
        hi = max(hi, int(ids[ok].max(initial=0)))
    date = np.zeros(hi + 1, dtype=np.int32)          # days since 1970; 0 = not a granted utility patent
    for b in batches(raw, "g_patent", ["patent_id", "patent_type", "patent_date", "withdrawn"], a.batch):
        ids, ok = ids_of(b.column(0))
        keep = ok & pc.and_(pc.equal(b.column(1), "utility"), pc.equal(b.column(3), "0")).to_numpy(zero_copy_only=False)
        d = pc.cast(pc.strptime(b.column(2), format="%Y-%m-%d", unit="s"), "date32").cast("int32").to_numpy(zero_copy_only=False)
        date[ids[keep]] = d[keep]
    first = np.full(hi + 1, -1, dtype=np.int8)
    mask = np.zeros(hi + 1, dtype=np.int16)
    code = {c: k for k, c in enumerate(LETTERS)}
    for b in batches(raw, "g_cpc_current", ["patent_id", "cpc_sequence", "cpc_section"], a.batch):
        ids, ok = ids_of(b.column(0))
        ok &= ids <= hi
        sec = np.array([code.get(x, -1) for x in b.column(2).to_pylist()], dtype=np.int8)
        ok &= sec >= 0
        np.bitwise_or.at(mask, ids[ok], (1 << sec[ok].astype(np.int16)).astype(np.int16))
        f = ok & (pc.cast(b.column(1), "int32").to_numpy(zero_copy_only=False) == 0)
        first[ids[f]] = sec[f]
    pid = np.nonzero((date > 0) & (first >= 0))[0]
    out = pl.DataFrame({"patent_id": pid.astype(np.int64), "date": pl.Series(date[pid]).cast(pl.Date),
                        "section": [LETTERS[k] for k in first[pid]], "mask": mask[pid].astype(np.int32)})
    out.write_parquet(a.out)
    print(json.dumps({"patents": out.height, "by_section": dict(out.group_by("section").len().sort("section").iter_rows()),
                      "dates": [str(out["date"].min()), str(out["date"].max())]}), flush=True)


def cmd_borrowings(a) -> None:
    """The citation table (about 150 million rows) streamed in batches against per-patent arrays (as in patents)."""
    import numpy as np
    import pyarrow.compute as pc
    raw = Path(a.raw)
    p = pl.read_parquet(a.patents)
    hi = int(p["patent_id"].max())
    date = np.zeros(hi + 1, dtype=np.int32)
    sec = np.full(hi + 1, -1, dtype=np.int8)
    mask = np.zeros(hi + 1, dtype=np.int32)
    ix = p["patent_id"].to_numpy()
    date[ix] = p["date"].cast(pl.Int32).to_numpy()
    sec[ix] = np.array([LETTERS.index(s) for s in p["section"].to_list()], dtype=np.int8)
    mask[ix] = p["mask"].to_numpy()
    del p
    cats, parts, n = set(), [], 0
    for b in batches(raw, "g_us_patent_citation", ["patent_id", "citation_patent_id", "citation_category"], a.batch):
        n += b.num_rows
        cats.update(pc.unique(b.column(2)).to_pylist())
        app = pc.fill_null(pc.equal(b.column(2), "cited by applicant"), False).to_numpy(zero_copy_only=False)
        c1, ok1 = ids_of(b.column(0))
        c2, ok2 = ids_of(b.column(1))
        ok = app & ok1 & ok2 & (c1 <= hi) & (c2 <= hi)
        c1, c2 = c1[ok], c2[ok]
        keep = ((date[c1] > 0) & (date[c2] > 0) & (sec[c1] >= 0) & (sec[c1] < 8) & (sec[c2] >= 0) & (sec[c2] < 8)
                & ((mask[c1] & mask[c2] & DISTANT) == 0) & (date[c2] < date[c1]))
        c1, c2 = c1[keep], c2[keep]
        parts.append(pl.DataFrame({"citing": c1, "cited": c2}))
    print(json.dumps({"citation_rows": n, "citation_categories": sorted(x for x in cats if x)}), flush=True)
    j = pl.concat(parts)
    c1, c2 = j["citing"].to_numpy(), j["cited"].to_numpy()
    out = j.with_columns(citing_date=pl.Series(date[c1]).cast(pl.Date), cited_date=pl.Series(date[c2]).cast(pl.Date),
                         citing_section=pl.Series([LETTERS[k] for k in sec[c1]]), cited_section=pl.Series([LETTERS[k] for k in sec[c2]]),
                         citing_mask=pl.Series(mask[c1]), cited_mask=pl.Series(mask[c2]))
    out.write_parquet(a.out)
    print(json.dumps({"borrowings": out.height, "citing_patents": out["citing"].n_unique()}), flush=True)


def cmd_sample(a) -> None:
    cfg = yaml.safe_load(open(a.config))
    rng = random.Random(cfg["seed"])
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cut = pl.lit(cfg["split"]["pool_before"]).str.to_date()
    b = pl.read_parquet(a.borrowings)
    if a.keep:
        # Only pairs that passed the labeling gate (p10-label.py kept; DATA-REVIEW.md): raw borrowings were about one
        # real analogy in five.
        k = pl.read_parquet(a.keep).select("citing", "cited")
        b = b.join(k, on=["citing", "cited"])
        print(json.dumps({"kept_borrowings": b.height}), flush=True)
    over = int(cfg["recovery"]["queries"] * 1.5)  # oversampled: some patents have no abstract (finalize trims)
    draft = []
    for split in ("val", "test"):
        lo, hi = cfg["split"][split]
        s = b.filter(pl.col("citing_date").is_between(pl.lit(lo).str.to_date(), pl.lit(hi).str.to_date()) & (pl.col("cited_date") < cut))
        citing = sorted(s["citing"].unique().to_list())
        rng.shuffle(citing)
        pick = set(citing[:over])
        rows = s.filter(pl.col("citing").is_in(list(pick))).sort("citing", "cited")
        by = {}
        for r in rows.iter_rows(named=True):
            by.setdefault(r["citing"], []).append(r)
        for q in sorted(by):
            r = rng.choice(by[q])
            draft.append({"split": split, "query": q, "cited": r["cited"], "query_date": r["citing_date"], "query_mask": r["citing_mask"],
                          "query_section": r["citing_section"], "cited_section": r["cited_section"]})
    draft = pl.DataFrame(draft)
    qids = set(draft["query"].to_list())
    # Everything each query cites, any category: none of it may be a candidate.
    cited_any = (scan(raw, "g_us_patent_citation").select(num("patent_id").alias("query"), num("citation_patent_id").alias("c"))
                 .filter(pl.col("query").is_in(list(qids))).drop_nulls().collect(engine="streaming"))
    excl = cited_any.group_by("query").agg(pl.col("c").alias("excluded"))
    draft = draft.join(excl, on="query", how="left")
    draft.write_parquet(out / "draft.parquet")

    pats = pl.read_parquet(a.patents).filter(pl.col("date") < cut)
    taken = qids | set(draft["cited"].to_list()) | set(cited_any["c"].to_list())
    pats = pats.filter(~pl.col("patent_id").is_in(list(taken)) & pl.col("section").is_in(list(LETTERS[:8])))
    need = [{"patent_id": p, "role": "query"} for p in sorted(qids)] + [{"patent_id": p, "role": "cited"} for p in sorted(set(draft["cited"].to_list()))]
    per_pool, per_spread = int(cfg["recovery"]["pool_per_section"] * 1.25), cfg["words"]["spread_per_section"]
    for sec in LETTERS[:8]:
        ids = sorted(pats.filter(pl.col("section") == sec)["patent_id"].to_list())
        rng.shuffle(ids)
        need += [{"patent_id": p, "role": "pool"} for p in ids[:per_pool]]
        need += [{"patent_id": p, "role": "spread"} for p in ids[per_pool:per_pool + per_spread]]
    need = pl.DataFrame(need).join(pl.read_parquet(a.patents).select("patent_id", "date", "section", "mask"), on="patent_id", how="left")
    need.write_parquet(out / "needed.parquet")
    print(json.dumps({"draft": dict(draft.group_by("split").len().iter_rows()), "needed": dict(need.group_by("role").len().iter_rows())}), flush=True)


def cmd_abstracts(a) -> None:
    # Row groups, not a polars scan: a scan of this table held every abstract and memcap killed it at 3.9 GB (review,
    # 10/05/2026).
    # Filtered inside Arrow, on the system allocator: collecting 244k abstracts as Python strings on Arrow's default
    # pool was killed at 3.4 GB three times (10/07/2026), though the kept text is about 250 MB.
    import pyarrow as pa
    import pyarrow.compute as pc
    pa.set_memory_pool(pa.system_memory_pool())
    ids = pl.read_parquet(a.needed)["patent_id"].unique()
    want = pa.array([str(i) for i in ids.to_list()])
    parts = []
    for b in batches(Path(a.raw), "g_patent_abstract", ["patent_id", "patent_abstract"], a.batch):
        k = b.filter(pc.is_in(b.column(0), value_set=want))
        if k.num_rows:
            parts.append(pl.from_arrow(pa.Table.from_batches([k])))
    ab = (pl.concat(parts).rename({"patent_abstract": "abstract"})
          .filter(pl.col("abstract").is_not_null())
          .with_columns(pl.col("patent_id").cast(pl.Int64), pl.col("abstract").str.replace_all(r"\s+", " ").str.strip_chars())
          .filter(pl.col("abstract") != "")
          .filter(pl.col("abstract").str.split(" ").list.len() >= 30))
    ab.write_parquet(a.out)
    print(json.dumps({"needed": len(ids), "abstracts": ab.height}), flush=True)


def cmd_finalize(a) -> None:
    cfg = yaml.safe_load(open(a.config))
    rng = random.Random(cfg["seed"] + 1)
    have = set(pl.read_parquet(a.abstracts)["patent_id"].to_list())
    need = pl.read_parquet(a.needed)
    pool = need.filter((pl.col("role") == "pool") & pl.col("patent_id").is_in(have)).sort("patent_id")
    by_sec = {s: list(g.iter_rows(named=True)) for (s,), g in pool.group_by("section")}
    k = cfg["recovery"]["candidates"] - 1
    rows = []
    draft = pl.read_parquet(a.draft).sort("split", "query")
    for split in ("val", "test"):
        d = [r for r in draft.filter(pl.col("split") == split).iter_rows(named=True) if r["query"] in have and r["cited"] in have]
        rng.shuffle(d)
        for r in d:
            excl = set(r["excluded"] or [])
            ok = [p["patent_id"] for p in by_sec[r["cited_section"]]
                  if p["date"] < r["query_date"] and (p["mask"] & r["query_mask"] & DISTANT) == 0 and p["patent_id"] not in excl]
            if len(ok) < k:
                continue
            rows.append({**{x: r[x] for x in ("split", "query", "cited", "query_section", "cited_section")}, "candidates": rng.sample(ok, k)})
            if sum(x["split"] == split for x in rows) == cfg["recovery"]["queries"]:
                break
    q = pl.DataFrame(rows, schema={"split": pl.String, "query": pl.Int64, "cited": pl.Int64, "query_section": pl.String,
                                   "cited_section": pl.String, "candidates": pl.List(pl.Int64)})
    # "queries" is a ceiling: the labeled val split has only 923 pairs (10/07/2026). Below min_queries (OPERATIONALIZATION:
    # under 500 test queries is "insufficient data") the run stops.
    floor = cfg["recovery"].get("min_queries", cfg["recovery"]["queries"])
    short = {s: int((q["split"] == s).sum()) for s in ("val", "test") if (q["split"] == s).sum() < floor}
    if short:  # the smoke fixture first produced none, and the failure surfaced two calls later
        sys.exit(f"too few queries with {k} eligible candidates: {short} (want at least {floor} per split)")
    q.write_parquet(a.out)
    print(json.dumps({"queries": dict(q.group_by("split").len().iter_rows()),
                      "cited_sections": dict(q.group_by("cited_section").len().sort("cited_section").iter_rows())}), flush=True)


def cmd_audit(a) -> None:
    q = pl.read_parquet(a.queries)
    need = pl.read_parquet(a.needed)
    ab = pl.read_parquet(a.abstracts)
    pats = pl.read_parquet(a.patents).select("patent_id", "date", "section", "mask")
    info = {r["patent_id"]: r for r in pats.filter(pl.col("patent_id").is_in(set(need["patent_id"].to_list()))).iter_rows(named=True)}
    have = set(ab.filter(pl.col("abstract").str.len_chars() > 0)["patent_id"].to_list())
    queries = set(q["query"].to_list())
    spread = set(need.filter(pl.col("role") == "spread")["patent_id"].to_list())
    bad = {"query_is_candidate": 0, "query_in_spread": len(queries & spread), "cited_in_spread": len(set(q["cited"].to_list()) & spread),
           "candidate_not_before_query": 0, "candidate_wrong_section": 0, "candidate_shares_section_with_query": 0,
           "cited_not_before_query": 0, "missing_abstract": 0, "wrong_candidate_count": 0, "duplicate_query": q.height - len(queries)}
    for r in q.iter_rows(named=True):
        qi, ci = info[r["query"]], info[r["cited"]]
        bad["cited_not_before_query"] += ci["date"] >= qi["date"]
        bad["wrong_candidate_count"] += len(set(r["candidates"]) | {r["cited"]}) != len(r["candidates"]) + 1
        for c in [r["query"], r["cited"], *r["candidates"]]:
            bad["missing_abstract"] += c not in have
        for c in r["candidates"]:
            x = info[c]
            bad["query_is_candidate"] += c in queries
            bad["candidate_not_before_query"] += x["date"] >= qi["date"]
            bad["candidate_wrong_section"] += x["section"] != ci["section"]
            bad["candidate_shares_section_with_query"] += (x["mask"] & qi["mask"] & DISTANT) != 0
    res = {"pass": all(v == 0 for v in bad.values()), "violations": bad, "queries": dict(q.group_by("split").len().iter_rows())}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    sys.exit(0 if res["pass"] else 1)


def cmd_export(a) -> None:
    """queries.parquet as JSON lines for the GPU instance, which has no polars."""
    q = pl.read_parquet(a.queries)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in q.iter_rows(named=True):
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"exported": q.height}), flush=True)


def cmd_convert(a) -> None:
    table(Path(a.raw), a.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    for f in ("raw", "name", "patents", "borrowings", "keep", "needed", "draft", "abstracts", "queries", "config", "out"):
        ap.add_argument(f"--{f}")
    ap.add_argument("--batch", type=int, default=1_000_000)
    a = ap.parse_args()
    {"convert": cmd_convert, "patents": cmd_patents, "borrowings": cmd_borrowings, "sample": cmd_sample, "abstracts": cmd_abstracts,
     "finalize": cmd_finalize, "audit": cmd_audit, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
