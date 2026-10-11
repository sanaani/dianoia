"""Phase 14 data (results/phase-14-openalex/HYPOTHESES.md, approved 10/08/2026). Run by pipelines/phase14.

    python p14-data.py fetch       --config <config> --work <work/p14> [--smoke]
    python p14-data.py pairs       --config <config> --work <work/p14>
    python p14-data.py overlap     --data <dianoia-data> --config <config> --work <work/p14>
    python p14-data.py far         --config <config> --work <work/p14>
    python p14-data.py gold-sheet  --config <config> --work <work/p14>
    python p14-data.py gold        --config <config> --work <work/p14>
    python p14-data.py review      --config <config> --work <work/p14> --source far|strict
    python p14-data.py sonnet-todo --config <config> --work <work/p14> --prompt v3p|v4p --out <todo.jsonl>
    python p14-data.py decide      --config <config> --work <work/p14> --out <step2.json>

fetch draws the sample (three seeded OpenAlex samples of 10,000, duplicates dropped) and, per batch of 100 sampled
articles, the works they cite that sit outside the Social Sciences and have an abstract. OpenAlex allows 1,000
requests a day without a key (10/08/2026), so every finished page or batch is written to disk at once and a rerun
skips it; when the day's budget runs out it exits 3 and the next run carries on. A free key (OPENALEX_API_KEY, or
<dianoia-data>/openalex-key.txt) gives 10 times the budget.

pairs matches each article's reference list against what fetch found and writes pairs.jsonl (abstracts rebuilt from
OpenAlex's word index, cut to 180 words; both in English with at least min_words words). overlap applies the first
half of phase 12's far rule with phase 12's patent pool as its word-weight basis and writes far-tmp/low.jsonl in
p13-data.py far's layout, so its cosine pass (on the GPU, p14-far-gpu.sh) runs unchanged. far assembles far.parquet
and far-text.jsonl from the pairs below both medians.

gold-sheet writes 10 far pairs to read blind (Claude's gold labels, gold-labels.txt); gold turns them into gold.jsonl.
review writes a blind reading file: 50 new pairs (far, or strict by Sonnet) and the 10 gold pairs, shuffled, numbers
only, and refuses to build until the gold labels are min_gold_hours old. decide applies step 2's rule.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
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
API = "https://api.openalex.org/works"


def _cfg(a):
    return yaml.safe_load(open(a.config)), Path(a.work)


def _wid(u):
    return u.rsplit("/", 1)[-1]


def _text(inv):
    """An abstract from OpenAlex's inverted index ({word: [positions]})."""
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def _cut(t, n):
    return " ".join(t.split()[:n])


def _write(path, rows):
    """Atomic: a page or batch is on disk whole or not at all."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    tmp.replace(path)


def _rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


class Budget(Exception):
    pass


def _get(params, oc):
    import requests
    key = os.environ.get("OPENALEX_API_KEY")
    kf = Path(os.environ.get("DIANOIA_DATA", Path.home() / "dianoia-data")) / "openalex-key.txt"
    if not key and kf.exists():  # Shaun's free key, kept outside the repo (10/08/2026)
        key = kf.read_text().strip()
    if key:
        params = {**params, "api_key": key}
    for attempt in range(5):
        time.sleep(1 / oc["max_per_second"])
        try:
            r = requests.get(API, params=params, timeout=60)
        except requests.RequestException as e:
            print(json.dumps({"retry": attempt, "error": str(e)[:200]}), flush=True)
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            raise Budget(r.text[:300])
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        _get.left = r.headers.get("X-RateLimit-Remaining")
        return r.json()
    raise RuntimeError(f"OpenAlex failed 5 times: {params.get('filter', '')[:120]}")


def cmd_fetch(a):
    cfg, work = _cfg(a)
    oc = cfg["openalex"]
    root = work / ("fetch-smoke" if a.smoke else "fetch")
    (root / "sample").mkdir(parents=True, exist_ok=True)
    (root / "refs").mkdir(parents=True, exist_ok=True)
    seeds = oc["seeds"][:1] if a.smoke else oc["seeds"]
    pages = 2 if a.smoke else oc["sample_size"] // oc["per_page"]
    calls = 0
    try:
        for s in seeds:
            for p in range(1, pages + 1):
                f = root / "sample" / f"s{s}-p{p:03d}.jsonl"
                if f.exists():
                    continue
                j = _get({"filter": oc["filter"], "sample": oc["sample_size"], "seed": s, "per-page": oc["per_page"],
                          "page": p, "select": "id,publication_year,language,referenced_works,abstract_inverted_index"}, oc)
                calls += 1
                _write(f, [{"id": _wid(w["id"]), "year": w["publication_year"], "language": w["language"],
                            "refs": [_wid(x) for x in w["referenced_works"]], "text": _text(w["abstract_inverted_index"])}
                           for w in j["results"]])
        ids = _sample_ids(root)
        batches = [ids[i:i + 100] for i in range(0, len(ids), 100)]
        if a.smoke:
            batches = batches[:2]
        for b, ch in enumerate(batches):
            f = root / "refs" / f"b{b:04d}.jsonl"
            if f.exists():
                continue
            out, cur = [], "*"
            while cur:
                j = _get({"filter": f"cited_by:{'|'.join(ch)},{oc['ref_filter']}", "per-page": oc["per_page"],
                          "cursor": cur, "select": "id,language,primary_topic,abstract_inverted_index"}, oc)
                calls += 1
                for w in j["results"]:
                    t = w["primary_topic"] or {}
                    out.append({"id": _wid(w["id"]), "language": w["language"], "text": _text(w["abstract_inverted_index"]),
                                "field": (t.get("field") or {}).get("display_name"),
                                "domain": (t.get("domain") or {}).get("display_name")})
                cur = j["meta"].get("next_cursor") if j["results"] else None
            _write(f, out)
            if b % 20 == 0:
                print(json.dumps({"batch": b, "of": len(batches), "calls": calls, "left": _get.left}), flush=True)
    except Budget as e:
        print(json.dumps({"stopped": "OpenAlex daily budget used; rerun tomorrow (or set OPENALEX_API_KEY)",
                          "calls": calls, "detail": str(e)[:200]}), flush=True)
        sys.exit(3)
    done = {"sample_pages": len(list((root / "sample").glob("*.jsonl"))), "sample_articles": len(_sample_ids(root)),
            "ref_batches": len(list((root / "refs").glob("*.jsonl"))), "calls_this_run": calls}
    (root / "done.json").write_text(json.dumps(done))
    print(json.dumps(done), flush=True)


def _sample_ids(root):
    """Sampled articles in draw order (seed, page), duplicates across draws dropped."""
    seen, ids = set(), []
    for f in sorted((root / "sample").glob("*.jsonl")):
        for r in map(json.loads, open(f, encoding="utf-8")):
            if r["id"] not in seen:
                seen.add(r["id"])
                ids.append(r["id"])
    return ids


def cmd_pairs(a):
    cfg, work = _cfg(a)
    oc, n = cfg["openalex"], cfg["pairs"]["min_words"]
    root = work / "fetch"
    words = L.WORDS
    cited = {}
    for f in sorted((root / "refs").glob("*.jsonl")):
        cited.update({r["id"]: r for r in map(json.loads, open(f, encoding="utf-8"))})
    seen, drop, out = set(), Counter(), []
    for f in sorted((root / "sample").glob("*.jsonl")):
        for r in map(json.loads, open(f, encoding="utf-8")):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            if r["language"] != "en" or len(r["text"].split()) < n:
                drop["citing_text"] += 1
                continue
            for c in r["refs"]:
                d = cited.get(c)
                if d is None:
                    continue
                if d["language"] != "en" or len(d["text"].split()) < n:
                    drop["cited_text"] += 1
                    continue
                out.append({"id": f"{r['id']}>{c}", "citing": r["id"], "cited": c, "year": r["year"],
                            "cited_field": d["field"], "cited_domain": d["domain"],
                            "citing_text": _cut(r["text"], words), "cited_text": _cut(d["text"], words)})
    _write(work / "pairs.jsonl", out)
    print(json.dumps({"articles": len(seen), "cited_outside": len(cited), "pairs": len(out),
                      "citing_with_pair": len({r["citing"] for r in out}), "dropped": drop,
                      "by_field": Counter(r["cited_field"] for r in out).most_common()}), flush=True)


def cmd_overlap(a):
    """Phase 12's word-overlap half of the far rule, basis = phase 12's patent pool, in p13-data.py far's layout."""
    cfg, work = _cfg(a)
    w = Path(a.data) / "work"
    med = json.loads((w / cfg["far"]["medians"]).read_text())
    basis = [{k: r[k] for k in ("citing", "cited", "citing_text", "cited_text")}
             for r in pl.read_parquet(w / cfg["far"]["pool"]).iter_rows(named=True)]
    ov = L.overlap_scorer([], basis=basis).texts
    del basis
    tmp = work / "far-tmp"
    tmp.mkdir(exist_ok=True)
    n, low, none = 0, 0, 0
    with open(tmp / "low.jsonl", "w", encoding="utf-8", newline="\n") as g:
        for r in map(json.loads, open(work / "pairs.jsonl", encoding="utf-8")):
            n += 1
            o = ov(r["citing_text"], r["cited_text"])
            if o is None:
                none += 1
            elif o < med["overlap"]:
                g.write(json.dumps({**r, "overlap": o}) + "\n")
                low += 1
    (tmp / "low.done.json").write_text(json.dumps({"stage": "overlap", "pairs": n, "below_overlap_median": low,
                                                   "no_text": none}))
    print(json.dumps({"pairs": n, "below_overlap_median": low, "no_text": none, "median": med["overlap"]}), flush=True)


def cmd_far(a):
    cfg, work = _cfg(a)
    part = work / "far-tmp/far-part.jsonl"
    total = json.loads((work / "far-tmp/low.done.json").read_text())
    prog = json.loads((work / "far-tmp/cosine.progress.json").read_text())["done"]
    if prog < total["below_overlap_median"]:
        sys.exit(f"cosine pass unfinished: {prog} of {total['below_overlap_median']}")
    meta = ("id", "citing", "cited", "year", "cited_field", "cited_domain", "overlap", "cosine")
    far = []
    with open(work / "far-text.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in map(json.loads, open(part, encoding="utf-8")):
            far.append({k: r[k] for k in meta})
            f.write(json.dumps({k: r[k] for k in ("id", "citing", "cited", "citing_text", "cited_text")}) + "\n")
    df = pl.DataFrame(far)
    df.write_parquet(work / "far.parquet")
    print(json.dumps({"pairs": total["pairs"], "below_overlap": total["below_overlap_median"], "far": len(far),
                      "far_share": round(len(far) / total["pairs"], 4), "citing_articles": df["citing"].n_unique(),
                      "by_field": dict(Counter(df["cited_field"].to_list()).most_common())}), flush=True)


def _sheet(path, key, items, texts):
    """A blind reading file: numbers only; ids and strata only in the key."""
    with open(path, "w", encoding="utf-8", newline="\n") as f, open(key, "w", encoding="utf-8", newline="\n") as k:
        for n, it in enumerate(items, 1):
            r = texts[it["id"]]
            f.write(f"#{n}\nCITING: {r['citing_text']}\nCITED : {r['cited_text']}\n\n")
            k.write(json.dumps({"n": n, **it}) + "\n")


def _labels(path, n):
    lab = {int(x): l.strip().upper() for x, l in (s.split(None, 1) for s in open(path) if s.strip())}
    if set(lab) != set(range(1, n + 1)) or not set(lab.values()) <= set("CGAB"):
        sys.exit(f"{path}: need one C/G/A/B label for each of 1..{n}")
    return lab


def cmd_gold_sheet(a):
    cfg, work = _cfg(a)
    rv = cfg["review"]
    ids = sorted(pl.read_parquet(work / "far.parquet")["id"].to_list())
    pick = random.Random(rv["gold_seed"]).sample(ids, rv["n_gold"])
    texts = {r["id"]: r for r in map(json.loads, open(work / "far-text.jsonl", encoding="utf-8")) if r["id"] in pick}
    _sheet(work / "gold.txt", work / "gold-key.jsonl", [{"id": i} for i in pick], texts)
    print(json.dumps({"gold": len(pick), "read": str(work / "gold.txt"), "write": str(work / "gold-labels.txt")}))


def cmd_gold(a):
    cfg, work = _cfg(a)
    key = _rows(work / "gold-key.jsonl")
    lab = _labels(work / "gold-labels.txt", len(key))
    _write(work / "gold.jsonl", [{"id": k["id"], "label": lab[k["n"]]} for k in key])
    print(json.dumps({"gold": Counter(lab.values())}))


def _sonnet(work, v):
    f = work / "sonnet" / f"{v}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(f, encoding="utf-8"))} if f.exists() else {}


def _strict_ids(work, mb):
    v3, v4 = _sonnet(work, "v3p"), _sonnet(work, "v4p")
    return sorted(i for i, r in v3.items() if r["label"] == "C" and (r["borrow"] or 0) >= mb
                  and i in v4 and v4[i]["label"] in ("C", "G"))


def cmd_review(a):
    cfg, work = _cfg(a)
    rv = cfg["review"]
    age = (time.time() - (work / "gold-labels.txt").stat().st_mtime) / 3600
    if age < rv["min_gold_hours"]:
        sys.exit(f"gold labels are {age:.1f} hours old; the blind re-read waits {rv['min_gold_hours']} hours")
    gold = _rows(work / "gold.jsonl")
    read = {g["id"] for g in gold}
    for f in work.glob("review-*-key.jsonl"):
        if f.name != f"review-{a.source}-key.jsonl":
            read |= {r["id"] for r in _rows(f)}
    if a.source == "far":
        pool = pl.read_parquet(work / "far.parquet")["id"].to_list()
    else:
        pool = _strict_ids(work, cfg["sonnet"]["min_borrow"])
    pool = sorted(i for i in pool if i not in read)
    if len(pool) < rv["n"]:
        sys.exit(f"only {len(pool)} unread {a.source} pairs")
    rng = random.Random(rv["seed"] + (a.source == "strict"))
    items = [{"id": i, "source": "new"} for i in rng.sample(pool, rv["n"])] + \
            [{"id": g["id"], "source": "gold", "earlier": g["label"]} for g in gold]
    rng.shuffle(items)
    want = {it["id"] for it in items}
    texts = {r["id"]: r for r in map(json.loads, open(work / "far-text.jsonl", encoding="utf-8")) if r["id"] in want}
    _sheet(work / f"review-{a.source}.txt", work / f"review-{a.source}-key.jsonl", items, texts)
    print(json.dumps({"source": a.source, "new": rv["n"], "pool": len(pool), "gold": len(gold)}))


def cmd_sonnet_todo(a):
    """v3p: a seeded random n far pairs. v4p: those whose v3p label is C with borrow >= min_borrow (phase 11's rule)."""
    cfg, work = _cfg(a)
    sc = cfg["sonnet"]
    if a.prompt == "v3p":
        ids = sorted(pl.read_parquet(work / "far.parquet")["id"].to_list())
        gold = {g["id"] for g in _rows(work / "gold-key.jsonl")} if (work / "gold-key.jsonl").exists() else set()
        ids = [i for i in ids if i not in gold]
        want = set(random.Random(sc["seed"]).sample(ids, min(sc["n"], len(ids))))
    else:
        v3 = _sonnet(work, "v3p")
        want = {i for i, r in v3.items() if r["label"] == "C" and (r["borrow"] or 0) >= sc["min_borrow"]}
    n = 0
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in map(json.loads, open(work / "far-text.jsonl", encoding="utf-8")):
            if r["id"] in want:
                f.write(json.dumps(r) + "\n")
                n += 1
    print(json.dumps({"prompt": a.prompt, "todo": n}))


def cmd_rest_todo(a):
    """Step 3 (10/09/2026): the far pairs step 2 left unlabeled, for 300 strict test pairs. v3p: every far pair not in
    sonnet/v3p.jsonl and not a gold pair. v4p: those whose v3p-rest label is C with borrow >= min_borrow."""
    cfg, work = _cfg(a)
    if a.prompt == "v3p":
        done = set(_sonnet(work, "v3p")) | {g["id"] for g in _rows(work / "gold-key.jsonl")}
        want = set(pl.read_parquet(work / "far.parquet")["id"].to_list()) - done
    else:
        want = {i for i, r in _sonnet(work, "v3p-rest").items()
                if r["label"] == "C" and (r["borrow"] or 0) >= cfg["sonnet"]["min_borrow"]}
    n = 0
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in map(json.loads, open(work / "far-text.jsonl", encoding="utf-8")):
            if r["id"] in want:
                f.write(json.dumps(r) + "\n")
                n += 1
    print(json.dumps({"prompt": a.prompt, "todo": n}))


def cmd_step3_build(a):
    """Step 3's test (HYPOTHESES.md, step 3 details): strict pairs from both Sonnet passes, one per citing article,
    then n_test at random; 999 candidates per pair from the true source's OpenAlex field (filled from its domain),
    never a work the same article cites. Writes step3/eval.jsonl and step3/docs.jsonl, texts cut to 180 words."""
    cfg, work = _cfg(a)
    sc, mb = cfg["step3"], cfg["sonnet"]["min_borrow"]
    v4 = {**_sonnet(work, "v4p"), **_sonnet(work, "v4p-rest")}
    strict = sorted(set(_strict_ids(work, mb)) | {
        i for i, r in _sonnet(work, "v3p-rest").items()
        if r["label"] == "C" and (r["borrow"] or 0) >= mb and i in v4 and v4[i]["label"] in ("C", "G")})
    rng = random.Random(sc["seed"])
    by_citing = {}
    for i in strict:
        by_citing.setdefault(i.split(">")[0], []).append(i)
    one = sorted(rng.choice(v) for _, v in sorted(by_citing.items()))
    test = sorted(rng.sample(one, min(sc["n_test"], len(one))))
    n_min, words = cfg["pairs"]["min_words"], L.WORDS
    pool = {}
    for f in sorted((work / "fetch/refs").glob("*.jsonl")):
        for r in map(json.loads, open(f, encoding="utf-8")):
            if r["language"] == "en" and len(r["text"].split()) >= n_min:
                pool[r["id"]] = r
    refs = {}
    for f in sorted((work / "fetch/sample").glob("*.jsonl")):
        for r in map(json.loads, open(f, encoding="utf-8")):
            refs.setdefault(r["id"], set(r["refs"]))
    want = set(test)
    texts = {r["id"]: r for r in map(json.loads, open(work / "far-text.jsonl", encoding="utf-8")) if r["id"] in want}
    by_field, by_domain = {}, {}
    for i, r in sorted(pool.items()):
        by_field.setdefault(r["field"], []).append(i)
        by_domain.setdefault(r["domain"], []).append(i)
    out = work / "step3"
    out.mkdir(exist_ok=True)
    ev, need, filled = [], set(), 0
    for i in test:
        citing, cited = i.split(">")
        cr = pool[cited]
        skip = refs[citing] | {cited}
        same = [x for x in by_field.get(cr["field"], []) if x not in skip]
        cand = rng.sample(same, min(sc["n_candidates"], len(same)))
        if len(cand) < sc["n_candidates"]:
            filled += 1
            have = set(cand)
            more = [x for x in by_domain.get(cr["domain"], []) if x not in skip and x not in have]
            cand += rng.sample(more, sc["n_candidates"] - len(cand))
        ev.append({"split": "test", "query": i, "query_text": texts[i]["citing_text"], "cited": cited,
                   "candidates": cand, "field": cr["field"], "label": v4[i]["label"]})
        need |= {cited, *cand}
    _write(out / "eval.jsonl", ev)
    _write(out / "docs.jsonl", [{"id": d, "text": _cut(pool[d]["text"], words)} for d in sorted(need)])
    print(json.dumps({"strict": len(strict), "citing_articles": len(one), "test": len(ev), "docs": len(need),
                      "filled_from_domain": filled, "labels": Counter(r["label"] for r in ev),
                      "fields": Counter(r["field"] for r in ev).most_common()}), flush=True)


def _score(work, source):
    """As phase 13: a reading counts only at >= 9/10 agreement with the gold labels (C and G count as one)."""
    key = _rows(work / f"review-{source}-key.jsonl")
    lab = _labels(work / f"review-{source}-labels.txt", len(key))
    real = lambda l: l in ("C", "G")
    gold = [k for k in key if k["source"] == "gold"]
    agree = sum(real(k["earlier"]) == real(lab[k["n"]]) for k in gold)
    new = [lab[k["n"]] for k in key if k["source"] == "new"]
    return {"agreement": f"{agree}/{len(gold)}", "counted": agree >= 9, "real_share": round(sum(map(real, new)) / len(new), 3),
            "c_share": round(new.count("C") / len(new), 3), "labels": {x: new.count(x) for x in "CGAB"}}


def cmd_decide(a):
    """Step 2: the strict reading, counted, at >= min_real real -> step 3. The far reading and C's share are reported."""
    cfg, work = _cfg(a)
    m = cfg["review"]["min_real"]
    s = {src: _score(work, src) for src in ("far", "strict")}
    v3, v4 = _sonnet(work, "v3p"), _sonnet(work, "v4p")
    strict = _strict_ids(work, cfg["sonnet"]["min_borrow"])
    passed = s["strict"]["counted"] and s["strict"]["real_share"] >= m
    out = {"min_real": m, **s, "sonnet": {"v3p": len(v3), "v3p_labels": Counter(r["label"] for r in v3.values()),
                                          "v4p": len(v4), "strict": len(strict),
                                          "strict_share": round(len(strict) / len(v3), 4) if v3 else None},
           "step2_pass": passed}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"step2_pass": passed, "far_real": s["far"]["real_share"], "strict_real": s["strict"]["real_share"],
                      "strict": len(strict)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    cmds = {"step3-build": cmd_step3_build, "rest-todo": cmd_rest_todo, "fetch": cmd_fetch, "pairs": cmd_pairs, "overlap": cmd_overlap, "far": cmd_far, "gold-sheet": cmd_gold_sheet,
            "gold": cmd_gold, "review": cmd_review, "sonnet-todo": cmd_sonnet_todo, "decide": cmd_decide}
    for name in cmds:
        p = sp.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--work", required=True)
        p.add_argument("--data")
        p.add_argument("--out")
        p.add_argument("--source", choices=("far", "strict"))
        p.add_argument("--prompt", choices=("v3p", "v4p"))
        p.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cmds[a.cmd](a)


if __name__ == "__main__":
    main()
