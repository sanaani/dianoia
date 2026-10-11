"""Phase 13 data (results/phase-13-scale-far/HYPOTHESES.md, approved 10/07/2026). Run by pipelines/phase13.

    python p13-data.py far     --data <dianoia-data> --config <config> --out <work/p13>
    python p13-data.py review  --data <dianoia-data> --config <config> --work <work/p13> --source corpus|fastino
    python p13-data.py decide  --data <dianoia-data> --config <config> --work <work/p13> --out <filter.json>
    python p13-data.py queries --data <dianoia-data> --config <config> --out <work/p13/short.jsonl>

far (step 2) applies phase 12's far rule, unchanged, to every cross-field applicant citation of phase 10's corpus
(2013-2024): word overlap with IDF from phase 12's pool (p10-label.py overlap_scorer, basis), BAAI/bge-small-en-v1.5
cosine, and phase 12's medians; far = below both. A pair touching a phase 10 query patent (query, true source or
candidate), by number or by its abstract's first 60 words, is dropped: the test questions stay out of training
permanently. Each far pair carries its field combination (citing CPC class -> cited CPC class, first listed CPC, e.g.
A61>G01; the subclass pair is kept too), whether Fastino stages 1-2 kept it, and its loss weight per source:
sqrt(median combination count / its combination's count), clipped to [1, 10]. The median is over the source's pairs
(the combination size the typical pair sits in). Revised 10/08/2026 with Shaun, before any test: at subclass level
with the median over combinations, the typical combination held 2 pairs and no pair reached 5x. Writes far.parquet
and far-text.jsonl.

review writes the blind reading file for one source: 25 unread far pairs from combinations weighted 5x or more, 25
from the rest, and 10 earlier hand-labeled pairs, shuffled, numbers only (p10-label.py review's format, so
p10-label.py review-score scores it). decide reads both scores and applies the decision rule.

queries (step 1) writes the short queries for phase 12's far test questions: the title (g_patent) and the Background
opening (g_brf_sum_text), cut by the rule in background_opening.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import polars as pl
import yaml


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L = _load("p10label", "p10-label.py")
WORDS = L.WORDS


def _key(t):
    """As p12-data.py: an abstract's identity by text, since continuations reuse abstracts under new numbers."""
    return " ".join(t.lower().split()[:60])


def _cfg(a):
    cfg = yaml.safe_load(open(a.config))
    return cfg, Path(a.data) / "work"


def _held(w, cfg):
    """Every patent phase 10's queries touch (query, true source, candidate), and their abstracts' text keys."""
    q = pl.read_parquet(w / cfg["inputs"]["queries"])
    ids = set(q["query"].to_list()) | set(q["cited"].to_list()) | {c for cs in q["candidates"].to_list() for c in cs}
    ab = pl.read_parquet(w / cfg["inputs"]["abstracts"]).filter(pl.col("patent_id").is_in(list(ids)))
    return ids, {_key(t) for t in ab["abstract"].to_list() if t}


def _subclass(data, ids):
    """First listed CPC subclass per patent (cpc_sequence 0)."""
    f = Path(data) / "raw/patentsview/parquet/g_cpc_current.parquet"
    d = (pl.scan_parquet(f).filter(pl.col("cpc_sequence") == "0")
         .filter(pl.col("patent_id").is_in([str(i) for i in ids])).select("patent_id", "cpc_subclass").collect())
    return {int(p): s for p, s in d.iter_rows()}


def weights(combos, lo, hi):
    """Loss weight per pair from its combination: sqrt(median combination count / count), clipped to [lo, hi]. The
    median is over pairs: the size of the combination the typical pair sits in."""
    cnt = dict(pl.Series(combos).value_counts().iter_rows())
    med = float(pl.Series([cnt[c] for c in combos]).median())
    return [min(hi, max(lo, math.sqrt(med / cnt[c]))) for c in combos], med


def cmd_far(a):
    cfg, w = _cfg(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = cfg["far"]
    med = json.loads((w / c["medians"]).read_text())
    # Streamed to disk and resumable (Shaun, 10/07/2026: "stream the results and discard as you go"; the first run was
    # memory-killed at 140k of 150k pairs after 7 hours). No pair's text is held in memory: the overlap pass writes
    # low.jsonl, the cosine pass appends far rows to far-part.jsonl and records how many low rows it has scored, and a
    # restart skips what is already on disk.
    tmp = out / "far-tmp"
    tmp.mkdir(exist_ok=True)
    low_f, low_done = tmp / "low.jsonl", tmp / "low.done.json"
    if not low_done.exists():
        basis = [{"citing": r["citing"], "cited": r["cited"], "citing_text": r["citing_text"], "cited_text": r["cited_text"]}
                 for r in pl.read_parquet(w / c["pool"]).iter_rows(named=True)]
        held, keys = _held(w, cfg)
        ov = L.overlap_scorer([], basis=basis).texts
        n, nlow, drop = 0, 0, {"held_patent": 0, "held_text": 0, "no_text": 0}
        with open(low_f, "w", encoding="utf-8", newline="\n") as g:
            for f in sorted((w / c["corpus"]).glob("corpus-*.jsonl")):
                for r in map(json.loads, open(f, encoding="utf-8")):
                    if a.smoke and n >= 400:
                        break
                    n += 1
                    if r["citing"] in held or r["cited"] in held:
                        drop["held_patent"] += 1
                    elif _key(r["citing_text"]) in keys or _key(r["cited_text"]) in keys:
                        drop["held_text"] += 1
                    else:
                        o = ov(r["citing_text"], r["cited_text"])
                        if o is None:
                            drop["no_text"] += 1
                        elif o < med["overlap"]:
                            g.write(json.dumps({**{k: r[k] for k in ("id", "citing", "cited", "year", "citing_text",
                                                                      "cited_text")}, "overlap": o}) + "\n")
                            nlow += 1
        del basis, held, keys, ov
        low_done.write_text(json.dumps({"stage": "overlap", "pairs": n, "below_overlap_median": nlow, **drop}))
    stats0 = json.loads(low_done.read_text())
    print(json.dumps(stats0), flush=True)
    if a.until == "overlap":
        return

    part_f, prog_f = tmp / "far-part.jsonl", tmp / "cosine.progress.json"
    done = json.loads(prog_f.read_text())["done"] if prog_f.exists() else 0
    # Drop any far rows appended after the last recorded checkpoint (a kill between the append and the record).
    if part_f.exists():
        trim = part_f.with_suffix(".trim")
        with open(trim, "w", encoding="utf-8", newline="\n") as g:
            g.writelines(l for l in open(part_f, encoding="utf-8") if json.loads(l)["low_row"] < done)
        trim.replace(part_f)
    total = stats0["below_overlap_median"]
    if done < total:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(cfg["measure_encoder"], device=a.device)
        step = 4 * a.batch

        def flush(chunk):
            e = m.encode([r[k] for _, r in chunk for k in ("citing_text", "cited_text")], batch_size=a.batch,
                         normalize_embeddings=True, show_progress_bar=False)
            with open(part_f, "a", encoding="utf-8", newline="\n") as g:
                for j, (i, r) in enumerate(chunk):
                    cos = round(float(e[2 * j] @ e[2 * j + 1]), 4)
                    if cos < med["cosine"]:
                        g.write(json.dumps({**r, "cosine": cos, "low_row": i}) + "\n")
            prog_f.write_text(json.dumps({"done": chunk[-1][0] + 1}))

        chunk, k = [], 0
        for i, l in enumerate(open(low_f, encoding="utf-8")):
            if i < done:
                continue
            chunk.append((i, json.loads(l)))
            if len(chunk) == step:
                flush(chunk)
                chunk, k = [], k + 1
                if k % 50 == 0:
                    print(json.dumps({"stage": "cosine", "done": i + 1, "of": total}), flush=True)
        if chunk:
            flush(chunk)
        del m
    if a.until == "cosine":
        print(json.dumps({"stage": "cosine", "done": json.loads(prog_f.read_text())["done"], "of": total}), flush=True)
        return

    # Final assembly holds no text: metadata first, then far-text.jsonl is streamed from far-part.jsonl.
    meta_cols = ("id", "citing", "cited", "year", "overlap", "cosine")
    far = [{k: r[k] for k in meta_cols} for r in map(json.loads, open(part_f, encoding="utf-8"))]
    print(json.dumps({"stage": "cosine", "done": total, "of": total, "far": len(far)}), flush=True)
    fastino = set()
    for f in sorted((w / c["survivors"]).glob("survivors-*.jsonl")):
        fastino |= {json.loads(l)["id"] for l in open(f, encoding="utf-8")}
    sub = _subclass(a.data, {r[k] for r in far for k in ("citing", "cited")})
    far = [r for r in far if r["citing"] in sub and r["cited"] in sub]
    for r in far:
        r["subclass_pair"] = f'{sub[r["citing"]]}>{sub[r["cited"]]}'
        r["combo"] = f'{sub[r["citing"]][:3]}>{sub[r["cited"]][:3]}'
        r["section_pair"] = f'{sub[r["citing"]][0]}>{sub[r["cited"]][0]}'
        r["fastino"] = r["id"] in fastino
    lo, hi = c["weight_clip"]
    stats = {}
    for src, sel in (("corpus", lambda r: True), ("fastino", lambda r: r["fastino"])):
        part = [r for r in far if sel(r)]
        ws, mc = weights([r["combo"] for r in part], lo, hi)
        for r, x in zip(part, ws):
            r[f"w_{src}"] = round(x, 3)
        stats[src] = {"far_pairs": len(part), "combinations": len({r["combo"] for r in part}), "median_combo_count": mc,
                      "weight_ge5": sum(x >= 5 for x in ws), "weight_eq1": sum(x == 1 for x in ws),
                      "by_year": dict(sorted(pl.Series([r["year"] for r in part]).value_counts().iter_rows()))}
    for r in far:
        r.setdefault("w_fastino", None)
    ids = {r["id"] for r in far}
    with open(out / "far-text.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in map(json.loads, open(part_f, encoding="utf-8")):
            if r["id"] in ids:
                f.write(json.dumps({k: r[k] for k in ("id", "citing", "cited", "citing_text", "cited_text")}) + "\n")
    pl.DataFrame(far).write_parquet(out / "far.parquet")
    print(json.dumps({"stage": "far", "below_both": len(far), **stats}), flush=True)


def cmd_far_merge(a):
    """Fold a GPU run's cosine pass (--work, a copy of far-tmp started at this run's checkpoint) into --out's far-tmp."""
    here, gpu = Path(a.out) / "far-tmp", Path(a.work) / "far-tmp"
    start = json.loads((here / "cosine.progress.json").read_text())["done"] if (here / "cosine.progress.json").exists() else 0
    first = json.loads((gpu / "cosine.start.json").read_text())["done"]
    end = json.loads((gpu / "cosine.progress.json").read_text())["done"]
    total = json.loads((here / "low.done.json").read_text())["below_overlap_median"]
    if first > start:
        raise SystemExit(f"GPU run started at row {first}, after this run's checkpoint {start}: a gap")
    # Rows this run wrote after its last checkpoint are dropped; the GPU run's rows from that checkpoint on are added.
    part, trim, n = here / "far-part.jsonl", here / "far-part.trim", 0
    with open(trim, "w", encoding="utf-8", newline="\n") as g:
        if part.exists():
            g.writelines(l for l in open(part, encoding="utf-8") if json.loads(l)["low_row"] < start)
        for l in open(gpu / "far-part.jsonl", encoding="utf-8"):
            if start <= json.loads(l)["low_row"] < end:
                g.write(l)
                n += 1
    trim.replace(part)
    (here / "cosine.progress.json").write_text(json.dumps({"done": max(start, end)}))
    print(json.dumps({"merged_far_rows": n, "from_row": start, "to_row": end, "of": total}), flush=True)


def cmd_review(a):
    """Blind reading file for one source; draws exclude every pair Claude has read before (p10, p12, p13 files)."""
    cfg, w = _cfg(a)
    rv = cfg["review"]
    work = Path(a.work)
    far = pl.read_parquet(work / "far.parquet")
    if a.source == "fastino":
        far = far.filter(pl.col("fastino"))
    elif a.source == "rejected":
        far = far.filter(~pl.col("fastino"))
    elif a.source == "s2reject":  # step 2c: stage 1 said yes, stage 2 said no
        far = far.filter(~pl.col("fastino") & pl.col("id").is_in(list(_stage_yes(w, 1))))
    # Rare combinations are judged over the whole far set (w_corpus) for both sources: within Fastino's 10k pairs the
    # typical combination holds 11 pairs, so no Fastino weight reaches 5x (10/08/2026).
    wcol = "w_corpus"
    read = set()
    for d in [work, *(w / x for x in rv["read_dirs"])]:
        read |= L._read_ids(d)
    far = far.filter(~pl.col("id").is_in(list(read)))
    rng = random.Random(rv["seed"])
    heavy = sorted(far.filter(pl.col(wcol) >= rv["heavy_weight"])["id"].to_list())
    rest = sorted(far.filter(pl.col(wcol) < rv["heavy_weight"])["id"].to_list())
    half = rv["n"] // 2
    if len(heavy) < half:
        sys.exit(f"only {len(heavy)} unread pairs weighted {rv['heavy_weight']}x or more; ask Shaun")
    new = [(i, "heavy") for i in rng.sample(heavy, half)] + [(i, "rest") for i in rng.sample(rest, rv["n"] - half)]
    texts = {r["id"]: r for r in map(json.loads, open(work / "far-text.jsonl", encoding="utf-8"))}
    gold = [r for r in map(json.loads, open(w / rv["gold"], encoding="utf-8"))]
    gtexts = {}
    for f in rv["gold_texts"]:
        gtexts.update({r["id"]: r for r in map(json.loads, open(w / f, encoding="utf-8"))})
    old = rng.sample([r for r in gold if r["id"] in gtexts], 10)
    texts.update(gtexts)
    items = [(i, "new", None, s) for i, s in new] + [(r["id"], "old", r["label"], None) for r in old]
    rng.shuffle(items)
    stem = work / f"review-{a.source}"
    with open(f"{stem}.txt", "w", encoding="utf-8", newline="\n") as f, \
            open(f"{stem}-key.jsonl", "w", encoding="utf-8", newline="\n") as k:
        for n, (i, src, lab, stratum) in enumerate(items, 1):
            r = texts[i]
            f.write(f"#{n}\nCITING: {r['citing_text']}\nCITED : {r['cited_text']}\n\n")
            k.write(json.dumps({"n": n, "id": i, "source": src, "earlier": lab, "stratum": stratum}) + "\n")
    print(json.dumps({"source": a.source, "new": len(new), "heavy_pool": len(heavy), "rest_pool": len(rest),
                      "rejudge": len(old), "out": f"{stem}.txt"}), flush=True)


def _score(work, source):
    """A reading counts only with at least 9/10 agreement on the re-judged pairs (C and G count as one)."""
    key = [json.loads(l) for l in open(work / f"review-{source}-key.jsonl", encoding="utf-8")]
    lab = {int(n): l for n, l in (x.split() for x in open(work / f"review-{source}-labels.txt") if x.strip())}
    if set(lab) != {k["n"] for k in key}:
        sys.exit(f"{source}: labels cover {len(lab)} of {len(key)} pairs")
    real = lambda l: l in ("C", "G")
    old = [k for k in key if k["source"] == "old"]
    agree = sum(real(k["earlier"]) == real(lab[k["n"]]) for k in old)
    new = [k for k in key if k["source"] == "new"]
    by = lambda s: [lab[k["n"]] for k in new if k["stratum"] == s]
    share = lambda ls: round(sum(map(real, ls)) / len(ls), 3)
    return {"agreement": f"{agree}/10", "counted": agree >= 9, "real_share": share([lab[k["n"]] for k in new]),
            "heavy_real_share": share(by("heavy")), "rest_real_share": share(by("rest")),
            "labels": {x: [lab[k["n"]] for k in new].count(x) for x in "CGAB"}}


def cmd_decide(a):
    """HYPOTHESES.md step 2's rule: unfiltered far pairs at >= min_real real -> scale on them with no labeling; else
    Fastino-only far pairs at >= min_real -> those; else the Sonnet stage stays. A reading under 9/10 agreement
    counts as not passing."""
    cfg, _ = _cfg(a)
    work, m = Path(a.work), cfg["review"]["min_real"]
    s = {src: _score(work, src) for src in ("corpus", "fastino")}
    ok = lambda x: x["counted"] and x["real_share"] >= m
    choice = "unfiltered" if ok(s["corpus"]) else "fastino" if ok(s["fastino"]) else "sonnet"
    Path(a.out).write_text(json.dumps({"min_real": m, **s, "choice": choice}, indent=1))
    print(json.dumps({"choice": choice, **{k: v["real_share"] for k, v in s.items()}}), flush=True)


def _stage_yes(w, k):
    """Ids Fastino stage <k> accepts in any year (p10-label.py _fastino_yes: scores are effectively 0 or 1)."""
    yes = set()
    for f in sorted((w / "p10/scale").glob(f"*/s{k}/pool-scores.jsonl")):
        yes |= L._fastino_yes(f)
    return yes


def _sonnet(w, version):
    """Sonnet labels already on disk: phase 10's per-year files and phase 13's own."""
    lab = {}
    for f in [*sorted((w / "p10/survivors").glob(f"sonnet-{version}-*.jsonl")), w / f"p13/sonnet/{version}.jsonl"]:
        if f.exists():
            lab.update({r["id"]: r for r in map(json.loads, open(f, encoding="utf-8"))})
    return lab


def cmd_sonnet_todo(a):
    """Far pairs Fastino kept that Sonnet has not labeled yet, with their text, for p10-label.py sonnet. v3: every
    such pair. v4: pairs whose v3 label is C with borrow >= min_borrow (phase 11's rule)."""
    cfg, w = _cfg(a)
    far = pl.read_parquet(Path(a.work) / "far.parquet").filter(pl.col("fastino"))
    have = _sonnet(w, a.prompt)
    want = set(far["id"].to_list()) - set(have)
    if a.prompt == "v4":
        v3 = _sonnet(w, "v3")
        want = {i for i in want if i in v3 and v3[i]["label"] == "C" and (v3[i]["borrow"] or 0) >= cfg["sonnet"]["min_borrow"]}
    n = 0
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in map(json.loads, open(Path(a.work) / "far-text.jsonl", encoding="utf-8")):
            if r["id"] in want:
                f.write(json.dumps(r) + "\n")
                n += 1
    print(json.dumps({"prompt": a.prompt, "kept_far": far.height, "already": len(set(far["id"].to_list()) & set(have)),
                      "todo": n}), flush=True)


def cmd_reject_sample(a):
    """Step 2d labels: a uniform random sample (seed from config) of far pairs Fastino rejected, n1 that stage 1
    rejected and n2 that stage 1 accepted but stage 2 rejected, excluding every pair already read (by Claude or
    Sonnet). Writes the pairs with their text and a 'stage' field."""
    cfg, w = _cfg(a)
    sd = cfg["filter"]
    far = pl.read_parquet(Path(a.work) / "far.parquet").filter(~pl.col("fastino"))
    s1 = _stage_yes(w, 1)
    read = set(_sonnet(w, "v3"))
    for d in [Path(a.work), *(w / x for x in cfg["review"]["read_dirs"])]:
        read |= L._read_ids(d)
    ids = [i for i in far["id"].to_list() if i not in read]
    rng = random.Random(sd["seed"])
    pick = {i: "s1" for i in rng.sample(sorted(i for i in ids if i not in s1), sd["n_stage1"])}
    pick.update({i: "s2" for i in rng.sample(sorted(i for i in ids if i in s1), sd["n_stage2"])})
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in map(json.loads, open(Path(a.work) / "far-text.jsonl", encoding="utf-8")):
            if r["id"] in pick:
                f.write(json.dumps({**r, "stage": pick[r["id"]]}) + "\n")
    print(json.dumps({"stage1": sum(v == "s1" for v in pick.values()), "stage2": sum(v == "s2" for v in pick.values()),
                      "excluded_read": len(far) - len(ids)}), flush=True)


def cmd_v4_todo(a):
    """Pairs of --pairs whose v3 label in --labels is C with borrow >= min_borrow: the strict rule's second read."""
    cfg, _ = _cfg(a)
    v3 = {r["id"]: r for r in map(json.loads, open(a.labels, encoding="utf-8"))}
    ok = lambda i: i in v3 and v3[i]["label"] == "C" and (v3[i]["borrow"] or 0) >= cfg["sonnet"]["min_borrow"]
    n = 0
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in map(json.loads, open(a.pairs, encoding="utf-8")):
            if ok(r["id"]):
                f.write(json.dumps(r) + "\n")
                n += 1
    print(json.dumps({"v3_labeled": len(v3), "todo": n}), flush=True)


def _strict(v3, v4, i, min_borrow):
    """Phase 11's rule: Sonnet v3 C with borrow >= min_borrow, then v4 C or G. None when v3 never labeled the pair."""
    if i not in v3:
        return None
    r = v3[i]
    return int(r["label"] == "C" and (r["borrow"] or 0) >= min_borrow and i in v4 and v4[i]["label"] in ("C", "G"))


def cmd_filter_build(a):
    """Step 2d training file, filter/train.jsonl: every Sonnet-labeled far pair (Fastino's keeps, and the reject
    sample), y = strict rule. Splits: 30% of the reject sample is the holdout (random, seed), never trained on; the rest
    is 90/10 train/dev. A training or dev pair sharing a patent with a holdout pair is dropped."""
    cfg, w = _cfg(a)
    fc, mb = cfg["filter"], cfg["sonnet"]["min_borrow"]
    fw = Path(a.work) / "filter"
    v3, v4 = _sonnet(w, "v3"), _sonnet(w, "v4")
    rv3 = {r["id"]: r for r in map(json.loads, open(fw / "v3.jsonl", encoding="utf-8"))}
    rv4 = {r["id"]: r for r in map(json.loads, open(fw / "v4.jsonl", encoding="utf-8"))}
    sample = {r["id"]: r for r in map(json.loads, open(fw / "sample.jsonl", encoding="utf-8"))}
    kept = set(pl.read_parquet(Path(a.work) / "far.parquet").filter(pl.col("fastino"))["id"].to_list())
    rng = random.Random(fc["seed"])
    sids = sorted(i for i in sample if i in rv3)
    hold = set(rng.sample(sids, round(fc["holdout"] * len(sids))))
    hpat = {sample[i][k] for i in hold for k in ("citing", "cited")}
    rows, drop = [], 0
    for r in map(json.loads, open(Path(a.work) / "far-text.jsonl", encoding="utf-8")):
        i = r["id"]
        if i in sample:
            y, src = _strict(rv3, rv4, i, mb), "reject"
        elif i in kept:
            y, src = _strict(v3, v4, i, mb), "kept"
        else:
            continue
        if y is None:
            continue
        if i in hold:
            split = "holdout"
        elif r["citing"] in hpat or r["cited"] in hpat:
            drop += 1
            continue
        else:
            split = "dev" if rng.random() < fc["dev"] else "train"
        rows.append({"id": i, "citing": r["citing"], "cited": r["cited"], "citing_text": r["citing_text"],
                     "cited_text": r["cited_text"], "y": y, "split": split, "source": src,
                     "stage": sample[i]["stage"] if i in sample else "kept"})
    with open(fw / "train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    c = Counter((r["split"], r["source"], r["y"]) for r in rows)
    print(json.dumps({"rows": len(rows), "dropped_shares_patent_with_holdout": drop,
                      "counts": {f"{s}/{src}/y{y}": n for (s, src, y), n in sorted(c.items())}}), flush=True)


def cmd_filter_audit(a):
    """Checks filter/train.jsonl against step 2d's rules; exits 1 on any violation. The GPU job refuses to train
    without a passing audit of these exact files (sha256)."""
    import hashlib
    cfg, w = _cfg(a)
    fw = Path(a.work) / "filter"
    rows = [json.loads(l) for l in open(fw / "train.jsonl", encoding="utf-8")]
    sample = {json.loads(l)["id"] for l in open(fw / "sample.jsonl", encoding="utf-8")}
    held, keys = _held(w, cfg)
    hold = [r for r in rows if r["split"] == "holdout"]
    hpat = {r[k] for r in hold for k in ("citing", "cited")}
    fit = [r for r in rows if r["split"] != "holdout"]
    bad = {
        "holdout_not_from_reject_sample": sum(r["id"] not in sample for r in hold),
        "fit_shares_patent_with_holdout": sum(r["citing"] in hpat or r["cited"] in hpat for r in fit),
        "touches_test_question": sum(r["citing"] in held or r["cited"] in held for r in rows),
        "shares_text_with_test": sum(_key(r["citing_text"]) in keys or _key(r["cited_text"]) in keys for r in rows),
        "duplicate_pair": len(rows) - len({r["id"] for r in rows}),
        "empty_text": sum(not r["citing_text"].strip() or not r["cited_text"].strip() for r in rows),
        "bad_label": sum(r["y"] not in (0, 1) for r in rows),
        "bad_split": sum(r["split"] not in ("train", "dev", "holdout") for r in rows),
        "no_holdout_positive": int(not any(r["y"] for r in hold)),
    }
    out = {"pass": not any(bad.values()), "violations": {k: v for k, v in bad.items() if v},
           "sha256": {"train.jsonl": hashlib.sha256((fw / "train.jsonl").read_bytes()).hexdigest()},
           "holdout": len(hold), "holdout_positive": sum(r["y"] for r in hold)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("pass", "violations", "holdout", "holdout_positive")}), flush=True)
    sys.exit(0 if out["pass"] else 1)


def cmd_filter_verdict(a):
    """F1 (HYPOTHESES.md step 2d): on the held-out rejects, the share of strict positives in the top-scored top_share,
    with a bootstrap interval over holdout pairs; passes if it is at least min_recall. Also the recall curve and
    positives per pair read at each depth, to size F2. Writes <out> (json) and STEP2D.md beside it."""
    cfg, _ = _cfg(a)
    fc = cfg["filter"]
    fw = Path(a.work) / "filter"
    y = {r["id"]: r["y"] for r in map(json.loads, open(fw / "train.jsonl", encoding="utf-8")) if r["split"] == "holdout"}
    p = {r["id"]: r["p"] for r in map(json.loads, open(fw / "out/holdout-scores.jsonl", encoding="utf-8"))}
    if set(p) != set(y):
        sys.exit(f"holdout scores cover {len(set(p) & set(y))} of {len(y)} pairs")

    def recall(ids, share):
        ranked = sorted(ids, key=lambda i: -p[i])
        top = ranked[:max(1, round(share * len(ranked)))]
        pos = sum(y[i] for i in ids)
        return sum(y[i] for i in top) / pos if pos else float("nan")

    ids = sorted(y)
    rec = recall(ids, fc["top_share"])
    rng = random.Random(fc["seed"])
    bs = sorted(x for x in (recall([rng.choice(ids) for _ in ids], fc["top_share"]) for _ in range(2000)) if x == x)
    lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]
    curve = {f"{s:.0%}": round(recall(ids, s), 3) for s in (0.05, 0.1, 0.2, 0.3, 0.5)}
    base = sum(y.values()) / len(y)
    per_read = {k: round(v * base / float(k.rstrip("%")) * 100, 3) for k, v in curve.items()}
    out = {"F1": {"recall_top": round(rec, 3), "lo": round(lo, 3), "hi": round(hi, 3), "pass": rec >= fc["min_recall"]},
           "holdout": len(y), "positives": sum(y.values()), "base_rate": round(base, 4), "recall_curve": curve,
           "positives_per_read": per_read}
    Path(a.out).write_text(json.dumps(out, indent=1))
    pct = lambda x: f"{100 * x:.0f}%"
    md = ["# Phase 13 step 2d: the scoring filter", "",
          f"**F1: {'PASS' if out['F1']['pass'] else 'FAIL'}.** The top-scored {pct(fc['top_share'])} of {len(y)} held-out "
          f"rejects hold {pct(rec)} of their {out['positives']} strict positives (95% interval {pct(lo)} to {pct(hi)}; "
          f"bar {pct(fc['min_recall'])}).", "",
          "| Top share read | Recall | Strict positives per pair read |", "|---|---|---|"]
    md += [f"| {k} | {pct(v)} | {per_read[k]:.3f} |" for k, v in curve.items()]
    md += ["", f"Reading at random finds {base:.3f} strict positives per pair."]
    Path(a.out).with_name("STEP2D.md").write_text("\n".join(md) + "\n")
    print(json.dumps(out["F1"]), flush=True)


def cmd_rejects(a):
    """Step 2b (HYPOTHESES.md, 10/08/2026): the share real among far pairs Fastino rejected, and Fastino's estimated
    recall, real kept / (real kept + real rejected), each reading's share scaled by its pool's size. Reported only."""
    work = Path(a.work)
    far = pl.read_parquet(work / "far.parquet")
    n_kept, n_rej = int(far["fastino"].sum()), int((~far["fastino"]).sum())
    kept, rej = _score(work, "fastino"), _score(work, "rejected")
    real_kept, real_rej = kept["real_share"] * n_kept, rej["real_share"] * n_rej
    out = {"kept": {**kept, "pool": n_kept, "real_est": round(real_kept)},
           "rejected": {**rej, "pool": n_rej, "real_est": round(real_rej)},
           "recall_est": round(real_kept / (real_kept + real_rej), 3) if real_kept + real_rej else None}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"rejected_real_share": rej["real_share"], "agreement": rej["agreement"],
                      "recall_est": out["recall_est"]}), flush=True)


# ---- step 1: short queries ---------------------------------------------------------------------------------------

HEAD = re.compile(r"^\s*(?:technical\s+)?(?:field|background|description of (?:the )?related art|related art|prior art)", re.I)


def background_opening(text, max_sentences=2):
    """The fixed cut rule (written 10/07/2026, before any short query was scored).

    1. Take the Background section: the text after the first heading containing "BACKGROUND" (all capitals, as
       PatentsView keeps headings), up to the next all-capitals heading (a line or run of 3+ capitalized words such as
       "SUMMARY", "BRIEF SUMMARY", "SUMMARY OF THE INVENTION"). No Background heading: no query (counted, reported).
    2. Drop a leading "Field" paragraph: sentences that only place the invention ("The present disclosure relates to
       ...", "This invention relates generally to ...", "... is directed to ...", "... pertains to ...").
    3. Keep sentences in order up to max_sentences, stopping before the first sentence that names the invention's own
       solution: one containing "present invention", "present disclosure", "the invention", "this disclosure",
       "embodiments", "disclosed herein" or "according to".
    4. Fewer than 8 words left: no query (counted, reported).
    """
    # Parsing details fitted to PatentsView's layout (headings on their own lines, paragraphs split by blank lines),
    # checked on unscored 2022 rows before any short query was scored.
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    start = next((i for i, x in enumerate(lines) if "BACKGROUND" in x and x == x.upper() and len(x.split()) <= 8), None)
    if start is None:
        return None
    body = []
    for x in lines[start + 1:]:
        if x == x.upper() and re.search(r"[A-Z]{3}", x) and len(x.split()) <= 10:
            break  # the next all-capitals heading (SUMMARY, BRIEF DESCRIPTION OF THE DRAWINGS, ...)
        if len(x.split()) <= 8 and not x.endswith((".", ":", ";")):
            continue  # a sub-heading: "Background Information", "1. Field of the Invention", "2. Description of ..."
        body.append(x)
    flat = " ".join(" ".join(body).split())
    # A period ends a sentence unless it closes an abbreviation ("U.S. Pat. No. 7,877,972", "FIG. 1", "Deere & Co.")
    # or a single letter (an initial); a lower-case continuation rejoins ("Moline, Ill. relates to ...").
    abbr = {"u.s", "no", "nos", "pat", "fig", "figs", "e.g", "i.e", "al", "ser", "vs", "inc", "co", "corp", "ltd", "approx"}
    sents, cur = [], []
    for tok in flat.split(" "):
        cur.append(tok)
        stem = tok.rstrip(".").lower()
        if tok.endswith((".", "!", "?")) and stem not in abbr and not re.fullmatch(r"[a-z]", stem):
            sents.append(" ".join(cur))
            cur = []
    sents += [" ".join(cur)] if cur else []
    merged = []
    for s in sents:
        if merged and not re.match(r"[A-Z(\d\"']", s):
            merged[-1] += " " + s
        else:
            merged.append(s)
    sents = [s for s in merged if s.strip()]
    place = re.compile(r"\b(relates?|relating|related|pertains?|directed|is in the field|field of)\b", re.I)
    own = re.compile(r"present invention|present disclosure|the invention|this disclosure|embodiments?|disclosed herein"
                     r"|according to", re.I)
    while sents and place.search(sents[0]) and len(sents[0].split()) < 60:
        sents.pop(0)
    keep = []
    for s in sents:
        if own.search(s) or len(keep) == max_sentences:
            break
        keep.append(s)
    q = " ".join(keep)
    return q if len(q.split()) >= 8 else None


def _tsv_rows(path, want, cols):
    """Rows of a PatentsView tsv.zip whose patent_id is in <want>, streamed (the files are larger than memory)."""
    import csv
    import io
    csv.field_size_limit(1 << 30)
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as raw:
        rd = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""), delimiter="\t")
        for r in rd:
            if r["patent_id"] in want:
                yield {c: r[c] for c in ("patent_id", *cols)}


def cmd_queries(a):
    cfg, w = _cfg(a)
    ev = [json.loads(l) for l in open(w / cfg["step1"]["eval"], encoding="utf-8")]
    ev = [r for r in ev if r["split"] in cfg["step1"]["splits"]]
    want = {str(r["query"]) for r in ev}
    raw = Path(a.data) / "raw/patentsview"
    title = {r["patent_id"]: r["patent_title"] for r in _tsv_rows(raw / "g_patent.tsv.zip", want, ["patent_title"])}
    bg, found = {}, set()
    for f in sorted(raw.glob("g_brf_sum_text_*.tsv.zip")):
        for r in _tsv_rows(f, want, ["summary_text"]):
            found.add(r["patent_id"])
            bg[r["patent_id"]] = background_opening(r["summary_text"], cfg["step1"]["max_sentences"])
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in ev:
            p = str(r["query"])
            f.write(json.dumps({"split": r["split"], "query": r["query"], "title": title.get(p),
                                "background": bg.get(p)}) + "\n")
    n = len(ev)
    print(json.dumps({"queries": n, "title": sum(p in title for p in want), "summary_found": len(found),
                      "background": sum(1 for v in bg.values() if v)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("far", "far-merge", "review", "decide", "rejects", "sonnet-todo", "reject-sample", "v4-todo", "filter-build", "filter-audit", "filter-verdict", "queries"):
        p = sp.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--out")
        p.add_argument("--work")
        p.add_argument("--source", choices=("corpus", "fastino", "rejected", "s2reject"))
        p.add_argument("--prompt", choices=("v3", "v4"))
        p.add_argument("--pairs")
        p.add_argument("--labels")
        p.add_argument("--batch", type=int, default=64)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--device", default="cpu")
        p.add_argument("--until", choices=("overlap", "cosine"), help="far: stop after this pass")
    a = ap.parse_args()
    {"filter-build": cmd_filter_build, "filter-audit": cmd_filter_audit, "filter-verdict": cmd_filter_verdict, "reject-sample": cmd_reject_sample, "v4-todo": cmd_v4_todo, "sonnet-todo": cmd_sonnet_todo, "rejects": cmd_rejects, "far-merge": cmd_far_merge, "far": cmd_far, "review": cmd_review, "decide": cmd_decide, "queries": cmd_queries}[a.cmd](a)


if __name__ == "__main__":
    main()
