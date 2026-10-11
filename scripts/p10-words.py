"""Phase 10 word steps: tags, frequency and domain spread, H1, H2 (lexical), role-replacement inputs, H3 words
(10/05/2026). Run by pipelines/phase10; the measures and verdict rules are pipelines/phase10/OPERATIONALIZATION.md.

    p10-words.py tag        --abstracts A --workers N --chunk N --out tagged.parquet
    p10-words.py stats      --tagged T --needed N --config C --out word_stats.parquet
    p10-words.py h1         --stats S --config C --out h1.json
    p10-words.py h2         --tagged T --queries Q --config C --out h2_lexical.json
    p10-words.py candidates --tagged T --stats S --queries Q --config C --out cands.jsonl
    p10-words.py h3words    --stats S --config C --out h3_words.jsonl

A word is its WordNet lemma and coarse part of speech: "n" (NN*) or "v" (VB*). Patent boilerplate is dropped before
any count ("said" is tagged a verb, "wherein" a noun): the STOP lists, fixed before the data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SECTIONS = "ABCDEFGH"
TOKEN = re.compile(r"[A-Za-z][A-Za-z\-]*|\d+(?:\.\d+)?|[^\sA-Za-z\d]")
WORD = re.compile(r"^[a-z]{3,}$")
STOP = {
    "v": {"be", "have", "do", "may", "can", "will", "shall", "say", "comprise", "include", "provide", "use", "wherein",
          "base", "configure", "dispose", "associate", "accord"},
    "n": {"wherein", "whereby", "thereof", "therein", "invention", "method", "system", "apparatus", "device", "embodiment",
          "example", "least", "plurality", "first", "second", "one", "portion", "member", "unit", "means", "present"},
}


def boot(xs, reps, seed=0, groups=None):
    """Mean and 95% interval; with groups, whole groups are resampled (H1 steps share their specific word)."""
    rng = random.Random(seed)
    if groups is None:
        groups = list(range(len(xs)))
    by = defaultdict(list)
    for g, x in zip(groups, xs):
        by[g].append(x)
    keys = list(by)
    sums = [(sum(by[k]), len(by[k])) for k in keys]
    ms = []
    for _ in range(reps):
        s = n = 0
        for _ in keys:
            a, b = sums[rng.randrange(len(keys))]
            s, n = s + a, n + b
        ms.append(s / max(n, 1))
    ms.sort()
    return [round(sum(xs) / max(len(xs), 1), 4), round(ms[int(0.025 * reps)], 4), round(ms[int(0.975 * reps)], 4)]


# ---- tag ---------------------------------------------------------------------------------------------------------

def _tag(rows):
    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    out = []
    for pid, text in rows:
        toks = TOKEN.findall(text)
        tags = pos_tag(toks)
        pos, lemmas = [], []
        for t, tg in tags:
            p = "n" if tg.startswith("NN") and not tg.startswith("NNP") else "v" if tg.startswith("VB") else "o"
            l = lem.lemmatize(t.lower(), p) if p != "o" else t.lower()
            if p != "o" and (not WORD.match(l) or l in STOP[p]):
                p = "o"
            pos.append(p)
            lemmas.append(l)
        out.append((pid, toks, pos, lemmas))
    return out


def cmd_tag(a):
    ab = pl.read_parquet(a.abstracts)
    rows = list(zip(ab["patent_id"].to_list(), ab["abstract"].to_list()))
    del ab
    chunks = [rows[i:i + a.chunk] for i in range(0, len(rows), a.chunk)]
    schema = pa.schema([("patent_id", pa.int64()), ("tokens", pa.list_(pa.string())), ("pos", pa.list_(pa.string())), ("lemmas", pa.list_(pa.string()))])
    n = 0
    with Pool(a.workers) as pool, pq.ParquetWriter(a.out + ".part", schema) as w:
        for res in pool.imap(_tag, chunks):
            w.write_table(pa.table(list(zip(*res)), schema=schema))
            n += len(res)
            if n % (a.chunk * 20) < a.chunk:
                print(json.dumps({"tagged": n, "of": len(rows)}), flush=True)
    Path(a.out + ".part").replace(a.out)
    print(json.dumps({"tagged": n}), flush=True)


# ---- stats -------------------------------------------------------------------------------------------------------

def cmd_stats(a):
    from wordfreq import zipf_frequency
    cfg = yaml.safe_load(open(a.config))
    need = pl.read_parquet(a.needed).filter(pl.col("role") == "spread").select("patent_id", "section")
    # Counted per row group (tagged_chunks): the tagged lists as Python objects went past 3 GB (10/07/2026).
    counts = defaultdict(lambda: [0] * 8)
    totals = [0] * 8
    for c in tagged_chunks(a.tagged):
        t = (c.join(need, on="patent_id").explode(["pos", "lemmas"]).filter(pl.col("pos") != "o")
             .group_by("lemmas", "pos", "section").len())
        for l, p, sec, n in t.iter_rows():
            k = SECTIONS.index(sec)
            counts[(l, p)][k] += n
            totals[k] += n
    rows = []
    for (l, p), c in counts.items():
        n = sum(c)
        rate = [x / totals[k] for k, x in enumerate(c)]
        s = sum(rate)
        probs = [r / s for r in rate]
        ent = -sum(q * math.log2(q) for q in probs if q > 0)
        home = max(range(8), key=lambda k: probs[k])
        rows.append({"lemma": l, "pos": p, "n": n, **{f"c_{SECTIONS[k]}": c[k] for k in range(8)}, "spread": ent,
                     "home": SECTIONS[home], "home_share": probs[home], "zipf": zipf_frequency(l, "en")})
    df = pl.DataFrame(rows)
    # Generality: z(Zipf) + z(spread), per part of speech, over words with enough patent occurrences to have a spread.
    ok = df.filter(pl.col("n") >= cfg["words"]["min_count"])
    parts = []
    for p in ("n", "v"):
        d = ok.filter(pl.col("pos") == p)
        d = d.with_columns((((pl.col("zipf") - d["zipf"].mean()) / d["zipf"].std()) + ((pl.col("spread") - d["spread"].mean()) / d["spread"].std())).alias("generality"))
        parts.append(d)
    out = pl.concat(parts).sort("pos", "lemma")
    out.write_parquet(a.out)
    print(json.dumps({"words": df.height, "with_spread": out.height, "by_pos": dict(out.group_by("pos").len().iter_rows()),
                      "tokens_per_section": dict(zip(SECTIONS, totals))}), flush=True)


def stats_maps(path):
    s = pl.read_parquet(path)
    return {(r["lemma"], r["pos"]): r for r in s.iter_rows(named=True)}


# ---- H1 ----------------------------------------------------------------------------------------------------------

def hypernym_steps(pos):
    from nltk.corpus import wordnet as wn
    steps = set()
    for syn in wn.all_synsets(pos):
        lo = [l.name().lower() for l in syn.lemmas() if WORD.match(l.name().lower())]
        for h in syn.hypernyms():
            hi = [l.name().lower() for l in h.lemmas() if WORD.match(l.name().lower())]
            steps.update((x, y) for x in lo for y in hi if x != y)
    return sorted(steps)


def cmd_h1(a):
    from wordfreq import zipf_frequency
    cfg = yaml.safe_load(open(a.config))
    st = stats_maps(a.stats)
    v = cfg["verdicts"]
    res = {}
    for p in ("n", "v"):
        steps = hypernym_steps(p)
        f = [(x, y) for x, y in steps if zipf_frequency(x, "en") > 0 and zipf_frequency(y, "en") > 0]
        both = [(x, y) for x, y in f if (x, p) in st and (y, p) in st]
        freq_up = [float(zipf_frequency(y, "en") > zipf_frequency(x, "en")) for x, y in f]
        spread_up = [float(st[(y, p)]["spread"] > st[(x, p)]["spread"]) for x, y in both]
        joint = [float(zipf_frequency(y, "en") > zipf_frequency(x, "en") and st[(y, p)]["spread"] > st[(x, p)]["spread"]) for x, y in both]
        g = [x for x, _ in both]
        share = boot(joint, v["bootstrap"], groups=g)
        verdict = "supported" if share[1] > v["h1_pass"] else "falsified" if share[0] <= v["h1_fail"] else "inconclusive"
        res[p] = {"wordnet_steps": len(steps), "steps_with_frequency": len(f), "steps_with_spread": len(both),
                  "coverage_spread": round(len(both) / max(len(steps), 1), 4),
                  "frequency_up": boot(freq_up, v["bootstrap"], groups=[x for x, _ in f]),
                  "spread_up": boot(spread_up, v["bootstrap"], groups=g), "both_up": share, "verdict": verdict}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


# ---- H2 (lexical) ------------------------------------------------------------------------------------------------

# The tagged file one row group (about 2,000 abstracts) at a time: as Python lists the 237,817 abstracts went past
# 3 GB, and a whole-file polars explode reached 3.6 GB in a second (10/07/2026).
def tagged_chunks(tagged, cols=("patent_id", "pos", "lemmas")):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(tagged, pre_buffer=False)
    for i in range(f.metadata.num_row_groups):
        yield pl.from_arrow(f.read_row_group(i, columns=list(cols)))


def _kept(chunk, keep):
    return chunk.select("patent_id", "pos", "lemmas").explode(["pos", "lemmas"]).filter(pl.col("pos").is_in(list(keep)))


def bags(tagged, keep, ids):
    """Lemma counts (parts of speech in <keep>) for the documents in <ids>."""
    out = {i: Counter() for i in ids}
    for c in tagged_chunks(tagged):
        c = c.filter(pl.col("patent_id").is_in(list(ids)))
        for pid, l, n in _kept(c, keep).group_by("patent_id", "lemmas").len().iter_rows():
            out[pid][l] = n
    return out


def doc_freq(tagged, keep):
    """Document frequency of each lemma over every tagged abstract, and the number of abstracts."""
    df, n = Counter(), 0
    for c in tagged_chunks(tagged):
        n += c.height
        for l, k in _kept(c, keep).select("patent_id", "lemmas").unique().group_by("lemmas").len().iter_rows():
            df[l] += k
    return df, n


def recovery(queries, vecs, k):
    """Top-k hit and reciprocal rank of the cited patent among its candidates, per query (ties count against)."""
    hits, rr = [], []
    for r in queries:
        q = vecs.get(r["query"], {})
        def cos(d):
            v = vecs.get(d, {})
            return sum(w * v.get(t, 0.0) for t, w in q.items())
        true = cos(r["cited"])
        rank = 1 + sum(cos(c) >= true for c in r["candidates"])
        hits.append(float(rank <= k))
        rr.append(1.0 / rank)
    return hits, rr


def tfidf(bag_map, df, n_docs):
    out = {}
    for pid, b in bag_map.items():
        v = {t: (1 + math.log(c)) * math.log(n_docs / (1 + df[t])) for t, c in b.items() if t in df}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out[pid] = {t: x / norm for t, x in v.items()}
    return out


def cmd_h2(a):
    cfg = yaml.safe_load(open(a.config))
    q = pl.read_parquet(a.queries)
    reps, k = cfg["verdicts"]["bootstrap"], cfg["recovery"]["top_k"]
    res = {}
    hits = {}
    ids = set(q["query"].to_list()) | set(q["cited"].to_list()) | {c for cs in q["candidates"].to_list() for c in cs}
    for name, keep in (("nouns", {"n"}), ("verbs", {"v"}), ("both", {"n", "v"})):
        # Vectors only for the documents retrieval touches; IDF over every tagged abstract, as before.
        df, n_docs = doc_freq(a.tagged, keep)
        vecs = tfidf(bags(a.tagged, keep, ids), df, n_docs)
        for split in ("val", "test"):
            rows = list(q.filter(pl.col("split") == split).iter_rows(named=True))
            h, rr = recovery(rows, vecs, k)
            hits[(name, split)] = h
            res[f"{name}_{split}"] = {"top10": boot(h, reps), "mrr": round(sum(rr) / len(rr), 4), "queries": len(rows)}
        print(json.dumps({name: {s: res[f"{name}_{s}"]["top10"] for s in ("val", "test")}}), flush=True)
    for split in ("val", "test"):
        d = [x - y for x, y in zip(hits[("verbs", split)], hits[("nouns", split)])]
        res[f"verbs_minus_nouns_{split}"] = boot(d, reps)
        res[f"chance_{split}"] = k / cfg["recovery"]["candidates"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({x: res[x] for x in res if "minus" in x}), flush=True)


# ---- role-replacement inputs -------------------------------------------------------------------------------------

def hypernym_cands(lemma, pos, st, max_steps):
    """Every single-word lemma up to max_steps hypernym steps above any sense of the word, with its depth and
    generality; only words with a measured generality qualify."""
    from nltk.corpus import wordnet as wn
    best = {}
    frontier = [(s, 0) for s in wn.synsets(lemma, pos)]
    seen = set()
    while frontier:
        syn, d = frontier.pop()
        if d >= max_steps:
            continue
        for h in syn.hypernyms():
            for l in h.lemmas():
                c = l.name().lower()
                if c != lemma and (c, pos) in st and (c not in best or best[c][0] > d + 1):
                    best[c] = (d + 1, st[(c, pos)]["generality"])
            if (h, d + 1) not in seen:
                seen.add((h, d + 1))
                frontier.append((h, d + 1))
    return [[c, dg[0], round(dg[1], 4)] for c, dg in sorted(best.items())]


def cmd_candidates(a):
    cfg = yaml.safe_load(open(a.config))
    st = stats_maps(a.stats)
    q = pl.read_parquet(a.queries)
    ids = set(q["query"].to_list()) | set(q["cited"].to_list()) | {c for cs in q["candidates"].to_list() for c in cs}
    t = pl.concat([c.filter(pl.col("patent_id").is_in(list(ids))) for c in tagged_chunks(a.tagged, ("patent_id", "tokens", "pos", "lemmas"))])
    qs = cfg["dials"]["generality_quantiles"]
    thresholds = {}
    gen = pl.read_parquet(a.stats)
    for p in ("n", "v"):
        g = gen.filter(pl.col("pos") == p)["generality"]
        thresholds[p] = [round(float(g.quantile(x)), 4) for x in qs]
    lex = {}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        docs = []
        for pid, toks, pos, lemmas in zip(t["patent_id"].to_list(), t["tokens"].to_list(), t["pos"].to_list(), t["lemmas"].to_list()):
            for p, l in zip(pos, lemmas):
                if p != "o" and f"{l}|{p}" not in lex:
                    lex[f"{l}|{p}"] = {"generality": round(st[(l, p)]["generality"], 4) if (l, p) in st else None,
                                       "zipf": st[(l, p)]["zipf"] if (l, p) in st else None,
                                       "spread": round(st[(l, p)]["spread"], 4) if (l, p) in st else None,
                                       "cands": hypernym_cands(l, p, st, cfg["words"]["max_hypernym_steps"])}
            docs.append({"id": pid, "tokens": toks, "pos": pos, "lemmas": lemmas})
        # Frequency and spread of every substitute too, for H6(b): how far verbs and nouns were moved.
        subs = {(c, k.split("|")[1]) for k, v in lex.items() for c, _, _ in v["cands"]}
        info = {f"{c}|{p}": [st[(c, p)]["zipf"], round(st[(c, p)]["spread"], 4)] for c, p in subs}
        f.write(json.dumps({"thresholds": thresholds, "lexicon": lex, "substitutes": info}) + "\n")
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with_c = sum(bool(v["cands"]) for v in lex.values())
    print(json.dumps({"docs": len(docs), "lexicon": len(lex), "with_candidates": with_c, "thresholds": thresholds}), flush=True)


# ---- H3 words ----------------------------------------------------------------------------------------------------

def cmd_h3words(a):
    from nltk.corpus import wordnet as wn
    cfg = yaml.safe_load(open(a.config))
    st = stats_maps(a.stats)
    rng = random.Random(cfg["seed"])
    held = sorted(rng.sample(SECTIONS, 3))
    pairs = []
    for (l, p), r in sorted(st.items()):
        if r["n"] < 50 or r["home_share"] < 0.4:
            continue
        syns = wn.synsets(l, p)
        if not syns:
            continue
        hyp = [x.name().lower() for h in syns[0].hypernyms() for x in h.lemmas() if (x.name().lower(), p) in st and x.name().lower() != l]
        if hyp:
            pairs.append({"word": l, "pos": p, "hypernym": hyp[0], "home": r["home"], "split": "test" if r["home"] in held else "train",
                          "zipf_hypernym": st[(hyp[0], p)]["zipf"]})
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"held_out_sections": held}) + "\n")
        for x in pairs:
            f.write(json.dumps(x) + "\n")
    print(json.dumps({"pairs": len(pairs), "held_out": held, "by_split": Counter(x["split"] for x in pairs)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    for f in ("abstracts", "tagged", "needed", "stats", "queries", "config", "out"):
        ap.add_argument(f"--{f}")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()
    {"tag": cmd_tag, "stats": cmd_stats, "h1": cmd_h1, "h2": cmd_h2, "candidates": cmd_candidates, "h3words": cmd_h3words}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
