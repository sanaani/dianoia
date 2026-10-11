"""Label patent borrowings as real cross-domain borrowings or not: Sonnet seeds, Fastino's GLiNER2.5-Decide scales
(10/05/2026).

    p10-label.py gold      --work W --raw R --out gold.jsonl           Claude's 140 hand labels, with abstracts
    p10-label.py seed      --feat F --raw R --n 3000 --seed S --out seed.jsonl    a fresh pool sample, with abstracts
    p10-label.py sonnet    --pairs P --out labels.jsonl [--workers 4]  Sonnet labels, blind, 10 pairs per call
    p10-label.py calibrate --gold G --labels L                         Sonnet against the hand labels

Shaun (10/05/2026): citation and classification filters reached at most 28% real borrowings, and no labeled dataset
fits, so the text must be read. Sonnet (his subscription) labels a seed set; Decide is fine-tuned on it and labels the
pool; only Decide's most confident "real" pairs are kept. Each stage is gated: Sonnet must agree with the hand labels,
Decide must hold 80% precision on held-out labels, and a fresh 50 of what Decide keeps is read by hand.

This labels input data. It grades no model output (the evaluation rules forbid that).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

import polars as pl

spec = importlib.util.spec_from_file_location("p10data", Path(__file__).with_name("p10-data.py"))
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

WORDS = 180  # per abstract: both fit GLiNER2.5-Decide's 512 tokens, and the labeler sees the same text
LABELS = {"C": "real borrowing", "A": "same technology", "B": "unrelated", "G": "generic technology reused (v4)"}

# The examples are invented: the first draft used pairs from the hand-labeled set, which would have shown Sonnet the
# calibration answers (10/05/2026). v2 (same day) decides by purpose: v1 called car seat -> bed "same technology"
# because the mechanism matched, and found only 15 of the 35 hand-labeled borrowings. Tuned on review-40 and r1;
# r2 is the held-out check.
PROMPT = """You are labeling pairs of US patents for a study of cross-domain analogy: an invention borrowing a mechanism
from a distant field. In each pair, the CITING patent cites the CITED patent, and the two sit in different patent
classification sections. Classification is often misleading, so judge only from the two abstracts.

Label each pair with exactly one letter:
C = real cross-domain borrowing. The citing invention serves a clearly different purpose or field than the cited one,
    and it reuses a recognizable mechanism, structure, material, method or principle from it.
    Examples: a fishing-reel drag brake -> tension control for a surgical suture; aircraft honeycomb panels ->
    lightweight furniture boards; a lawn-sprinkler rotor -> a dishwasher spray arm.
A = same technology. Both are about the same kind of product, technology or application, even if filed under
    different classifications. Examples: a phone camera module -> a phone lens actuator; a pesticide compound ->
    a spray formulation of it; a hearing-aid circuit -> a hearing-aid battery holder.
B = unrelated. No shared mechanism or purpose is visible from the abstracts (a stray or boilerplate citation).

Decide by PURPOSE first. If the two inventions serve different purposes or fields, and the citing one reuses
something specific from the cited one, the answer is C, even when the reused mechanism is identical: a borrowing is
exactly the same mechanism put to a new purpose. Choose A only when both serve the same purpose or product area.
Shared generic words alone ("controller", "sensor", "layer") are not a borrowing: if nothing specific is reused,
choose B.

Also give "borrow": how clearly the citing invention borrows a specific mechanism from a different field, 1 to 5
(1 = not at all, 3 = plausible but weak or generic, 5 = unmistakable: you could name the borrowed mechanism and both
purposes).

Reply with only a JSON list, one object per pair, in order: [{{"id": "<id>", "label": "C|A|B", "borrow": 1-5}}, ...]

{pairs}"""


# v4 (10/06/2026): results/phase-10-analogy/RUBRIC.md. v3 said "the same mechanism put to a new purpose" is C, so a
# generic technology moved into a new product (a UV lamp, a bearing wheel) counted as a borrowing; the evening hand
# check put 30% of the kept set there. v4 gives that case its own label, G, excluded like A. Examples are invented, not
# taken from the hand checks (the v1 lesson: examples copied from gold leaked the answers).
PROMPT_V4 = """You are labeling pairs of US patents for a dataset that will teach a language model to think by analogy:
to see that a problem in one field is like a problem in another, and carry the other field's solution across. In each
pair, the CITING patent cites the CITED patent, and the two sit in different patent classification sections.
Classification is often misleading, so judge only from the two abstracts.

Label each pair with exactly one letter:
C = analogical borrowing. The cited field's own way of solving its own problem is carried into a different problem,
    and seeing it required noticing that the two problems are alike; the roles map onto different objects.
    Examples: a fishing-reel drag brake -> tension control for a surgical suture; the way a beehive vents heat ->
    passive cooling for a battery cabinet; a lawn-sprinkler's water-driven rotor -> a dishwasher spray arm.
G = generic technology reused. The citing invention uses a technology, component or material that is generic and
    named the same way in both, doing the same job, so the cited field is incidental: the inventor could have taken it
    from a supplier's catalog or a textbook without looking at the cited field.
    Examples: a UV lamp disinfecting aquarium water -> a UV lamp disinfecting toothbrushes; a ball-bearing caster on
    an office chair -> a ball-bearing wheel on a hospital cart; a non-stick fluoropolymer coating on a pan -> the same
    coating on a printer roller.
A = same technology. Both are about the same kind of product or application, even under different classifications.
    Examples: a phone camera module -> a phone lens actuator; a pesticide compound -> a spray formulation of it.
B = unrelated. Nothing specific is reused (a stray or boilerplate citation, or only generic words like "sensor").

Ask: could the citing inventor have picked up the reused thing from a catalog or a general textbook, without looking
at the cited field? If yes, G (or A if the products are the same kind). If the idea is the cited field's particular
solution, C.

Also give "borrow": how clearly the citing invention carries over the cited field's particular solution, 1 to 5
(1 = not at all, 3 = plausible but weak, 5 = unmistakable: you could name the solution and both problems).

Reply with only a JSON list, one object per pair, in order: [{{"id": "<id>", "label": "C|G|A|B", "borrow": 1-5}}, ...]

{pairs}"""


# Phase 14 (10/08/2026): v3 and v4 reworded for research papers (results/phase-14-openalex/HYPOTHESES.md). The CITING
# paper is a management, business or accounting paper and the CITED one is outside the social sciences. Same labels,
# same decision rules, same borrow scale; the examples are invented, none taken from any reading.
PROMPT_V3P = """You are labeling pairs of research papers for a study of cross-domain analogy: a management idea
borrowing from a distant field. In each pair, the CITING paper (business, management or accounting) cites the CITED
paper, which is outside the social sciences. Topic labels are often wrong, so judge only from the two abstracts.

Label each pair with exactly one letter:
C = real cross-domain borrowing. The citing paper addresses a clearly different problem or field than the cited one,
    and it reuses a recognizable mechanism, model, principle or way of working from it.
    Examples: how ant colonies find short paths -> routing delivery trucks; immune-system memory -> how firms
    remember past supplier failures; how forests recover after fire -> how industries recover after a crisis.
A = same topic. Both study the same kind of thing (for example both about hospital operations, or both about the
    energy efficiency of buildings), even though they are filed under different fields.
B = unrelated. No shared mechanism or problem is visible from the abstracts (a stray or background citation).

Decide by PROBLEM first. If the two papers address different problems or fields, and the citing one reuses something
specific from the cited one, the answer is C, even when the reused mechanism is identical. Choose A only when both
address the same kind of problem. Shared generic words alone ("model", "network", "performance") are not a
borrowing: if nothing specific is reused, choose B.

Also give "borrow": how clearly the citing paper borrows a specific mechanism from a different field, 1 to 5
(1 = not at all, 3 = plausible but weak or generic, 5 = unmistakable: you could name the borrowed mechanism and both
problems).

Reply with only a JSON list, one object per pair, in order: [{{"id": "<id>", "label": "C|A|B", "borrow": 1-5}}, ...]

{pairs}"""

PROMPT_V4P = """You are labeling pairs of research papers for a dataset that will teach a language model to think by
analogy: to see that a problem in one field is like a problem in another, and carry the other field's solution
across. In each pair, the CITING paper (business, management or accounting) cites the CITED paper, which is outside
the social sciences. Topic labels are often wrong, so judge only from the two abstracts.

Label each pair with exactly one letter:
C = analogical borrowing. The cited field's own way of handling its own problem is carried into a management problem,
    and seeing it required noticing that the two problems are alike; the roles map onto different things.
    Examples: how ant colonies find short paths -> routing delivery trucks; immune-system memory -> how firms
    remember past supplier failures; how forests recover after fire -> how industries recover after a crisis.
G = general method used as is. The citing paper uses a statistical test, a standard algorithm, a measurement
    instrument or a technology that is generic and named the same way in both, doing the same job, so the cited
    field is incidental: the author could have taken it from a textbook or a software manual.
    Examples: a regression method from a medical study -> the same regression on firm data; a standard optimization
    algorithm from engineering -> the same algorithm scheduling staff; a sensor network from environmental
    monitoring -> the same sensors tracking warehouse stock.
A = same topic. Both study the same kind of thing, even though they are filed under different fields.
B = unrelated. Nothing specific is reused (a stray or background citation, or only generic words like "model").

Ask: could the citing author have picked up the reused thing from a general textbook or a software manual, without
looking at the cited field? If yes, G (or A if the two study the same kind of thing). If the idea is the cited field's
particular solution, C.

Also give "borrow": how clearly the citing paper carries over the cited field's particular solution, 1 to 5
(1 = not at all, 3 = plausible but weak, 5 = unmistakable: you could name the solution and both problems).

Reply with only a JSON list, one object per pair, in order: [{{"id": "<id>", "label": "C|G|A|B", "borrow": 1-5}}, ...]

{pairs}"""

PROMPTS = {"v3": PROMPT, "v4": PROMPT_V4, "v3p": PROMPT_V3P, "v4p": PROMPT_V4P}

def abstracts(raw, ids):
    want, out = {str(i) for i in ids}, {}
    for b in D.batches(Path(raw), "g_patent_abstract", ["patent_id", "patent_abstract"], 0):
        for pid, t in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            if pid in want and t:
                out[int(pid)] = " ".join(t.split()[:WORDS])
    return out


def cmd_gold(a):
    hand = json.loads(Path(__file__).parent.parent.joinpath("results/phase-10-analogy/hand-labels.json").read_text())
    rows = []
    for fname, labs in hand["rounds"].items():
        lab_of = {k: lab for lab, ks in labs.items() for k in ks}
        for line in open(Path(a.work) / fname, encoding="utf-8"):
            m = re.match(r"#(\d+)\s+(\d+) \(.*?\)\s+cites\s+(\d+) \(", line)
            if m:
                k, citing, cited = int(m.group(1)), int(m.group(2)), int(m.group(3))
                rows.append({"id": f"{citing}-{cited}", "citing": citing, "cited": cited, "label": lab_of[k], "round": fname})
    ab = abstracts(a.raw, {r["citing"] for r in rows} | {r["cited"] for r in rows})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "citing_text": ab[r["citing"]], "cited_text": ab[r["cited"]]}) + "\n")
    print(json.dumps({"gold": len(rows), "labels": Counter(r["label"] for r in rows)}), flush=True)


def cmd_seed(a):
    """A seed sample from the round-1 pool (lift <= 0.5, <= 50 applicant citations), citing years 2013-2017 only, so
    no val or test query (2018 on) is ever labeled by Sonnet or trained on by Decide; one pair per citing patent."""
    gold = {json.loads(l)["id"] for l in open(a.gold, encoding="utf-8")} if a.gold else set()
    # A second seed round (10/06/2026) skips the first one's pairs, too.
    gold |= {json.loads(l)["id"] for f in (a.exclude or "").split(",") if f for l in open(f, encoding="utf-8")}
    f = pl.read_parquet(a.feat).filter((pl.col("lift") <= 0.5) & (pl.col("citing_app_cites") <= 50)
                                       & pl.col("citing_date").dt.year().is_between(2013, 2017))
    f = f.sample(fraction=1.0, shuffle=True, seed=a.seed).unique("citing", keep="first", maintain_order=True)
    f = f.filter(~(pl.col("citing").cast(pl.String) + "-" + pl.col("cited").cast(pl.String)).is_in(list(gold))).head(a.n)
    ab = abstracts(a.raw, set(f["citing"].to_list()) | set(f["cited"].to_list()))
    n = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in f.iter_rows(named=True):
            if r["citing"] in ab and r["cited"] in ab:
                fh.write(json.dumps({"id": f"{r['citing']}-{r['cited']}", "citing": r["citing"], "cited": r["cited"],
                                     "citing_text": ab[r["citing"]], "cited_text": ab[r["cited"]]}) + "\n")
                n += 1
    print(json.dumps({"seed": n}), flush=True)


def cmd_pool(a):
    """Everything Decide scores: the round-1 pool in the val (2018-2019) and test (2021-2024) citing years, cited before
    2021, one pair per citing patent (the one pipelines/phase10 would sample from). Disjoint from the seed by year."""
    import yaml
    cfg = yaml.safe_load(open(a.config))
    yr = pl.col("citing_date")
    f = pl.read_parquet(a.feat).filter((pl.col("lift") <= 0.5) & (pl.col("citing_app_cites") <= 50)
                                       & (pl.col("cited_date") < pl.lit(cfg["split"]["pool_before"]).str.to_date()))
    f = f.filter(yr.is_between(pl.lit(cfg["split"]["val"][0]).str.to_date(), pl.lit(cfg["split"]["val"][1]).str.to_date())
                 | yr.is_between(pl.lit(cfg["split"]["test"][0]).str.to_date(), pl.lit(cfg["split"]["test"][1]).str.to_date()))
    f = f.sample(fraction=1.0, shuffle=True, seed=a.seed).unique("citing", keep="first", maintain_order=True)
    ab = abstracts(a.raw, set(f["citing"].to_list()) | set(f["cited"].to_list()))
    n = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in f.iter_rows(named=True):
            if r["citing"] in ab and r["cited"] in ab:
                fh.write(json.dumps({"id": f"{r['citing']}-{r['cited']}", "citing": r["citing"], "cited": r["cited"],
                                     "citing_text": ab[r["citing"]], "cited_text": ab[r["cited"]]}) + "\n")
                n += 1
    print(json.dumps({"pool": n}), flush=True)


def _read_ids(work):
    """Every pair id Claude has read in any review file or hand-label file, so a fresh check is really fresh."""
    seen = set()
    for f in Path(work).glob("review*.txt"):
        seen |= set(re.findall(r"^#\d+ (\S+)", f.read_text(encoding="utf-8"), re.M))
    for f in list(Path(work).glob("*-key.jsonl")) + list(Path(work, "label").glob("rubric-*.jsonl")) + \
            [Path(work, "label/gold.jsonl"), Path(work, "label/gold2.jsonl")]:
        if f.exists():
            seen |= {json.loads(l)["id"] for l in open(f, encoding="utf-8")}
    return seen


def cmd_review(a):
    """A hand check that measures its own rater (10/06/2026). Claude's morning and evening readings of the same 50 pairs
    agreed on 31; that was found only by accident. Draws <n> unread pairs from <labels> (a kept parquet; --split),
    shuffles in 10 earlier hand-labeled pairs from <gold> (jsonl with id and label), and writes <out>.txt (numbers
    only, no ids or scores) and <out>-key.jsonl. Judge with RUBRIC.md, then run review-score."""
    import pyarrow.parquet as pq
    texts = {}
    for f in a.pairs.split(","):
        for r in map(json.loads, open(f, encoding="utf-8")):
            texts[r["id"]] = r
    read = _read_ids(a.work)
    # A jsonl <labels> (the phase 11 training set, 10/06/2026) is a pair list with no split: every pair is a candidate.
    if a.labels.endswith(".jsonl"):
        kept = [json.loads(l)["id"] for l in open(a.labels, encoding="utf-8")]
    else:
        kept = [f'{r["citing"]}-{r["cited"]}' for r in pq.read_table(a.labels).to_pylist() if r["split"] == a.split]
    rng = random.Random(a.seed)
    new = rng.sample([i for i in kept if i not in read and i in texts], a.n)
    old = rng.sample([r for r in map(json.loads, open(a.gold, encoding="utf-8")) if r["id"] in texts], 10)
    items = [(i, "new", None) for i in new] + [(r["id"], "old", r["label"]) for r in old]
    rng.shuffle(items)
    with open(f"{a.out}.txt", "w", encoding="utf-8", newline="\n") as f, \
            open(f"{a.out}-key.jsonl", "w", encoding="utf-8", newline="\n") as k:
        for n, (i, src, lab) in enumerate(items, 1):
            r = texts[i]
            f.write(f"#{n}\nCITING: {r['citing_text']}\nCITED : {r['cited_text']}\n\n")
            k.write(json.dumps({"n": n, "id": i, "source": src, "earlier": lab}) + "\n")
    print(json.dumps({"new": len(new), "rejudge": len(old), "out": f"{a.out}.txt"}), flush=True)


def cmd_review_score(a):
    """Scores a review: <labels> is a text file of "<n> <C|G|A|B>" lines. Prints the gate rates on the new pairs only if
    the 10 re-judged pairs agree with their earlier labels (C and G count as one, the gate's kept class) at least 9
    times; otherwise it prints the disagreements and stops."""
    key = [json.loads(l) for l in open(f"{a.out}-key.jsonl", encoding="utf-8")]
    lab = {int(n): l for n, l in (line.split() for line in open(a.labels) if line.strip())}
    if set(lab) != {k["n"] for k in key}:
        raise SystemExit(f"labels cover {len(lab)} of {len(key)} pairs")
    kept = lambda l: "K" if l in ("C", "G", "K") else l
    old = [(k["n"], k["earlier"], lab[k["n"]]) for k in key if k["source"] == "old"]
    agree = sum(kept(e) == kept(l) for _, e, l in old)
    if agree < 9:
        print(json.dumps({"rater_agreement": f"{agree}/10", "disagreements": [o for o in old if kept(o[1]) != kept(o[2])],
                          "gate": "NOT READ: fix the rubric or its application first"}))
        return
    new = [kept(lab[k["n"]]) for k in key if k["source"] == "new"]
    with open(f"{a.out}-labels.jsonl", "w", newline="\n") as f:
        for k in key:
            f.write(json.dumps({**k, "label": lab[k["n"]]}) + "\n")
    print(json.dumps({"rater_agreement": f"{agree}/10", "new": len(new), "kept": new.count("K"),
                      "same_product": new.count("A"), "unrelated": new.count("B"),
                      "gate_pass": new.count("K") >= 0.8 * len(new) and new.count("A") <= 0.1 * len(new)}))


def cmd_corpus(a):
    """The scale-up corpus (10/06/2026): every borrowing in the population the cascade was trained and measured on
    (the pool's filters: lift <= 0.5, citing_app_cites <= 50; any citing year, every pair, not one per citing patent),
    minus pairs already in <exclude> files (the pool). One jsonl shard per citing year in <out>/, with "year", so any
    time split can be drawn later."""
    f = pl.read_parquet(a.feat, columns=["citing", "cited", "citing_date", "lift", "citing_app_cites"])
    f = f.filter((pl.col("lift") <= 0.5) & (pl.col("citing_app_cites") <= 50))
    done = {json.loads(l)["id"] for x in (a.exclude or "").split(",") if x for l in open(x, encoding="utf-8")}
    f = f.with_columns((pl.col("citing").cast(pl.String) + "-" + pl.col("cited").cast(pl.String)).alias("id"),
                       pl.col("citing_date").dt.year().alias("year")).filter(~pl.col("id").is_in(list(done)))
    ab = abstracts(a.raw, set(f["citing"].to_list()) | set(f["cited"].to_list()))
    out, n = Path(a.out), Counter()
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for r in f.iter_rows(named=True):
        if r["citing"] in ab and r["cited"] in ab:
            y = r["year"]
            if y not in files:
                files[y] = open(out / f"corpus-{y}.jsonl", "w", encoding="utf-8", newline="\n")
            files[y].write(json.dumps({"id": r["id"], "citing": r["citing"], "cited": r["cited"], "year": y,
                                       "citing_text": ab[r["citing"]], "cited_text": ab[r["cited"]]}) + "\n")
            n[y] += 1
    for fh in files.values():
        fh.close()
    print(json.dumps({"pairs": sum(n.values()), "by_year": dict(sorted(n.items())), "excluded": len(done)}), flush=True)


def cmd_survivors(a):
    """Corpus pairs that exclusion stages 1 and 2 (P10S) both keep, for Sonnet: one jsonl per citing year in <out>/.
    <work> is the work dir holding p10/corpus and p10/scale; --exclude takes years, comma-separated, to skip."""
    w, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = set((a.exclude or "").split(","))
    n = {}
    for f in sorted((w / "p10/corpus").glob("corpus-*.jsonl")):
        y = f.stem.split("-")[1]
        s = [w / f"p10/scale/{y}/s{k}/pool-scores.jsonl" for k in (1, 2)]
        if y in skip or not all(p.exists() for p in s):
            continue
        yes = _fastino_yes(s[0]) & _fastino_yes(s[1])
        with open(out / f"survivors-{y}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for l in open(f, encoding="utf-8"):
                if json.loads(l)["id"] in yes:
                    fh.write(l if l.endswith("\n") else l + "\n")
        n[y] = len(yes)
    print(json.dumps({"survivors": n, "total": sum(n.values())}), flush=True)


def cmd_train(a):
    """Phase 11 training pairs (results/phase-11-analogy-training/HYPOTHESES.md): corpus survivors of stages 1-2 with
    Sonnet v3 C and borrow >= <min_borrow> and Sonnet v4 C or G, from --years (comma list). <work>/p10/survivors
    holds survivors-, sonnet-v3- and sonnet-v4-<year>.jsonl. A pair sharing a patent (either end) with a val or test
    pair of <pairs> (kept-final.parquet) is dropped."""
    held = pl.read_parquet(a.pairs)
    banned = set(held["citing"].to_list()) | set(held["cited"].to_list())
    sv, n, rows = Path(a.work) / "p10/survivors", Counter(), []
    for y in a.years.split(","):
        v3 = {r["id"]: r for r in map(json.loads, open(sv / f"sonnet-v3-{y}.jsonl", encoding="utf-8"))}
        v4 = {r["id"]: r["label"] for r in map(json.loads, open(sv / f"sonnet-v4-{y}.jsonl", encoding="utf-8"))}
        for r in map(json.loads, open(sv / f"survivors-{y}.jsonl", encoding="utf-8")):
            l3 = v3.get(r["id"])
            if not (l3 and l3["label"] == "C" and (l3.get("borrow") or 0) >= a.min_borrow and v4.get(r["id"]) in ("C", "G")):
                continue
            if int(r["citing"]) in banned or int(r["cited"]) in banned:
                n[f"{y} dropped"] += 1
                continue
            rows.append({**r, "borrow": l3["borrow"], "v4": v4[r["id"]]})
            n[y] += 1
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"pairs": len(rows), "citing_patents": len({r["citing"] for r in rows}), "by_year": dict(sorted(n.items()))}), flush=True)


def cmd_subset(a):
    """A random <n> pool pairs per split (val, test) for Sonnet to label directly. Fallback of 10/05/2026: Decide,
    fine-tuned on 2,392 Sonnet labels, kept "real" pairs at only 50% precision, with no usable confidence dial."""
    import yaml
    cfg = yaml.safe_load(open(a.config))
    year = dict(pl.read_parquet(a.feat).select((pl.col("citing").cast(pl.String) + "-" + pl.col("cited").cast(pl.String)).alias("id"),
                                               pl.col("citing_date").dt.year().alias("y")).iter_rows())
    val = range(int(cfg["split"]["val"][0][:4]), int(cfg["split"]["val"][1][:4]) + 1)
    test = range(int(cfg["split"]["test"][0][:4]), int(cfg["split"]["test"][1][:4]) + 1)
    rows = list(map(json.loads, open(a.pairs, encoding="utf-8")))
    # Pairs already labeled in an earlier subset are skipped, so a second round adds only new pairs.
    done = {json.loads(l)["id"] for f in (a.exclude or "").split(",") if f for l in open(f, encoding="utf-8")}
    rows = [r for r in rows if r["id"] not in done]
    # Fastino screen (10/06/2026): with --scores, only pairs Fastino accepts go to Sonnet; a kept pair needs both.
    if a.scores:
        yes = _fastino_yes(a.scores)
        rows = [r for r in rows if r["id"] in yes]
    random.Random(a.seed).shuffle(rows)
    out = {"val": [], "test": []}
    for r in rows:
        s = "val" if year[r["id"]] in val else "test" if year[r["id"]] in test else None
        if s and len(out[s]) < a.n:
            out[s].append({**r, "split": s})
    with open(a.out, "w", encoding="utf-8") as f:
        for s in ("val", "test"):
            for r in out[s]:
                f.write(json.dumps(r) + "\n")
    print(json.dumps({s: len(v) for s, v in out.items()}), flush=True)


def cmd_kept(a):
    """The pairs the recovery test may use: Sonnet "real" with borrow >= <min_borrow> (DATA-REVIEW.md: borrow 4-5 was
    86% real in the first hand check, against 44% at 3). <pairs> and <labels> are matching comma-separated lists."""
    rows = []
    # With --scores, Fastino must also accept the pair: Sonnet borrow >= 4 alone was 12% same technology in the fresh
    # check of 10/06/2026; with Fastino's vote, 1 of 18 (DATA-REVIEW.md).
    yes = _fastino_yes(a.scores) if a.scores else None
    for pf, lf in zip(a.pairs.split(","), a.labels.split(",")):
        lab = {r["id"]: r for r in map(json.loads, open(lf, encoding="utf-8"))}
        for r in map(json.loads, open(pf, encoding="utf-8")):
            l = lab.get(r["id"])
            if yes is not None and r["id"] not in yes:
                continue
            if l and l["label"] == "C" and (l.get("borrow") or 0) >= a.min_borrow:
                rows.append({"citing": r["citing"], "cited": r["cited"], "split": r["split"], "borrow": l["borrow"]})
    df = pl.DataFrame(rows).unique(["citing"])
    df.write_parquet(a.out)
    print(json.dumps({"kept": dict(df.group_by("split").len().iter_rows())}), flush=True)


def cmd_sonnet(a):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from dianoia import llm
    rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8")]
    rng = random.Random(0)
    rng.shuffle(rows)  # mixed order: the hand-labeled rounds are not grouped
    # Resumable (10/08/2026): each call's labels are appended to <out> as it returns, and a restart skips labeled ids.
    done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")} if Path(a.out).exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    # Pairs per call: 10 first; Shaun (10/05/2026) asked for few large calls, so it is a setting, checked on the gold set.
    chunks = [rows[i:i + a.per_call] for i in range(0, len(rows), a.per_call)]

    def prompt(ch):
        return PROMPTS[a.prompt].format(pairs="\n\n".join(f"PAIR id={r['id']}\nCITING: {r['citing_text']}\nCITED: {r['cited_text']}" for r in ch))

    # Each worker runs a claude process of up to about 300 MB: 10 workers (2 GB+) drew the memory reaper (10/05/2026).
    import psutil
    fits = max(1, int((psutil.virtual_memory().available / 2**20 - 1500) // 300))
    if a.workers > fits:
        print(json.dumps({"workers": a.workers, "lowered_to": fits, "reason": "free memory"}), flush=True)
        a.workers = fits
    out, bad = {}, 0
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.workers) as pool, open(a.out, "a", encoding="utf-8", newline="\n") as f:
        for k, (ch, res) in enumerate(zip(chunks, pool.map(lambda ch: _safe(llm.ask_json, prompt(ch), model=a.model, effort=a.effort, timeout=3600), chunks)), 1):
            got = {str(x.get("id")): (x.get("label"), x.get("borrow")) for x in (res or []) if isinstance(x, dict)}
            for r in ch:
                lab, borrow = got.get(r["id"], (None, None))
                if lab in LABELS:
                    out[r["id"]] = (lab, borrow)
                    f.write(json.dumps({"id": r["id"], "label": lab, "borrow": borrow}) + "\n")
                else:
                    bad += 1
            f.flush()
            if k % 10 == 0:
                print(json.dumps({"calls": k, "of": len(chunks), "labeled": len(out), "unparsed": bad}), flush=True)
    print(json.dumps({"labeled": len(out), "earlier": len(done), "unparsed": bad,
                      "labels": Counter(v[0] for v in out.values()), "model": a.model}), flush=True)


def overlap_scorer(rows, basis=None):
    """IDF-weighted word overlap of a pair's two abstracts (RUBRIC.md, 10/06/2026): the measured stand-in for
    "generic technology reused", which no rater applied consistently. 0 = no shared distinctive words; 1 = every
    distinctive word of the shorter abstract appears in the other. IDF comes from every abstract in <rows> (dicts with
    citing, cited, citing_text, cited_text). Returns ov(citing, cited), None when either abstract is missing. Phase 12
    measures its pool with this same function (10/07/2026). With <basis> (rows of the same shape), IDF comes from the
    basis abstracts only and a word the basis never saw gets the basis's largest IDF, so phase 12's far medians apply
    unchanged to new pairs (phase 13, 10/07/2026)."""
    import math
    stop = {"the", "and", "for", "with", "that", "this", "from", "are", "which", "has", "have", "can", "may", "said",
            "one", "two", "least", "plurality", "comprising", "comprises", "includes", "including", "first", "second",
            "provided", "method", "apparatus", "device", "system", "wherein", "such", "each", "being", "into", "its",
            "between", "also", "other", "more", "least", "having", "used", "use", "thereof", "further", "portion"}
    words = lambda t: {w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in stop}
    def read(rs):
        text, df = {}, Counter()
        for r in rs:
            for k in ("citing", "cited"):
                if r[k] not in text:
                    text[r[k]] = words(r[f"{k}_text"])
                    df.update(text[r[k]])
        return text, df

    text, df = read(rows)
    if basis is not None:
        btext, df = read(basis)
        n = len(btext)
    else:
        n = len(text)
    idf = {w: math.log(n / c) for w, c in df.items()}
    unseen = math.log(n)
    w_ = lambda w: idf.get(w, unseen)

    def ov(ci, cd):
        x, y = text.get(ci), text.get(cd)
        if not x or not y:
            return None
        sx, sy = sum(map(w_, x)), sum(map(w_, y))
        return round(sum(map(w_, x & y)) / min(sx, sy), 4)

    def texts(xt, yt):  # the same score from two texts, for streaming pairs past a fixed basis
        x, y = words(xt), words(yt)
        if not x or not y:
            return None
        return round(sum(map(w_, x & y)) / min(sum(map(w_, x)), sum(map(w_, y))), 4)

    ov.texts = texts

    return ov


def cmd_overlap(a):
    """Adds an "overlap" column (overlap_scorer, IDF from every abstract in <pairs>) to <labels> (a kept parquet) and
    writes <out>."""
    ov = overlap_scorer([r for f in a.pairs.split(",") for r in map(json.loads, open(f, encoding="utf-8"))])
    kept = pl.read_parquet(a.labels)
    kept = kept.with_columns(pl.Series("overlap", [ov(c, d) for c, d in zip(kept["citing"], kept["cited"])]))
    kept.write_parquet(a.out)
    q = kept["overlap"].drop_nulls().quantile
    print(json.dumps({"pairs": kept.height, "missing": kept["overlap"].null_count(),
                      "quartiles": [round(q(x), 3) for x in (0.25, 0.5, 0.75)]}), flush=True)


def _fastino_yes(path):
    """Ids Fastino accepts. Its scores are effectively 0 or 1 (P10D2), so 0.5 is the only cut."""
    return {r["id"] for r in map(json.loads, open(path, encoding="utf-8")) if r["p"] >= 0.5}


def _safe(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except Exception as e:  # one bad call loses its 10 pairs, not the run
        print(json.dumps({"error": str(e)[:200]}), flush=True)
        return None


def cmd_calibrate(a):
    gold = {r["id"]: r["label"] for r in map(json.loads, open(a.gold, encoding="utf-8"))}
    lab = {r["id"]: r["label"] for r in map(json.loads, open(a.labels, encoding="utf-8"))}
    both = [k for k in gold if k in lab]
    conf = Counter((gold[k], lab[k]) for k in both)
    said_c = [k for k in both if lab[k] == "C"]
    prec = sum(gold[k] == "C" for k in said_c) / max(len(said_c), 1)
    rec = sum(lab[k] == "C" for k in both if gold[k] == "C") / max(sum(gold[k] == "C" for k in both), 1)
    agree = sum(gold[k] == lab[k] for k in both) / max(len(both), 1)
    res = {"pairs": len(both), "agreement": round(agree, 3), "precision_C": round(prec, 3), "recall_C": round(rec, 3),
           "said_C": len(said_c), "confusion (hand, sonnet)": {f"{h}{s}": n for (h, s), n in sorted(conf.items())},
           "gate_precision_C>=0.85": prec >= 0.85}
    # The dial: keep only pairs labeled C with borrow >= k. Tuning rounds choose k; review-r2 is the held-out check.
    rows = {r["id"]: r for r in map(json.loads, open(a.labels, encoding="utf-8"))}
    rnd = {r["id"]: r["round"] for r in map(json.loads, open(a.gold, encoding="utf-8"))}
    for part, rounds in (("tune", ("review-40.txt", "review-r1.txt")), ("heldout", ("review-r2.txt",))):
        ks = [k for k in both if rnd[k] in rounds]
        for t in (3, 4, 5):
            kept = [k for k in ks if lab[k] == "C" and (rows[k].get("borrow") or 0) >= t]
            res[f"{part}_borrow>={t}"] = {"kept": len(kept), "precision_C": round(sum(gold[k] == "C" for k in kept) / max(len(kept), 1), 3),
                                          "same_tech": sum(gold[k] == "A" for k in kept)}
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    for x in ("work", "raw", "feat", "gold", "pairs", "labels", "config", "out"):
        ap.add_argument(f"--{x}")
    ap.add_argument("--years", help="train: citing years, comma-separated")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--split", default="val", help="review: which split of the kept parquet to draw from")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-call", type=int, default=10)
    ap.add_argument("--min-borrow", type=int, default=4)
    ap.add_argument("--effort", help="claude -p --effort (e.g. medium)")
    ap.add_argument("--exclude", help="comma-separated jsonl files whose ids are skipped")
    ap.add_argument("--scores", help="Fastino pool-scores.jsonl: subset and kept use only pairs it accepts")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--prompt", choices=tuple(PROMPTS), default="v3", help="v4: RUBRIC.md, with label G; v3p/v4p: papers")
    a = ap.parse_args()
    {"gold": cmd_gold, "seed": cmd_seed, "pool": cmd_pool, "subset": cmd_subset, "kept": cmd_kept, "sonnet": cmd_sonnet, "calibrate": cmd_calibrate,
     "overlap": cmd_overlap, "corpus": cmd_corpus, "survivors": cmd_survivors, "train": cmd_train, "review": cmd_review, "review-score": cmd_review_score}[a.cmd](a)


if __name__ == "__main__":
    main()
