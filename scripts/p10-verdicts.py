"""Phase 10 verdicts: pipelines/phase10/OPERATIONALIZATION.md's rules applied to the metrics (10/05/2026).

    p10-verdicts.py --metrics work/p10/metrics --audit work/p10/audit.json --config C --out results/phase-10-analogy

Writes REPORT.md (verdict per hypothesis, plain numbers) and metrics.json (everything, minus per-query arrays). Every
rule here is the table in OPERATIONALIZATION.md; nothing is decided after seeing a number.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

MIN_TEST_QUERIES = 500  # half the planned 1,000; fewer and a recovery verdict is not given


def boot(xs, reps, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return [round(sum(xs) / n, 4), round(ms[int(0.025 * reps)], 4), round(ms[int(0.975 * reps)], 4)]


def diff(a, b, reps):
    return boot([x - y for x, y in zip(a, b)], reps)


def pct(x):
    return f"{100 * x:.1f}%"


def ci(v):
    return f"{pct(v[0])} [{pct(v[1])}, {pct(v[2])}]"


def judge(lower, point):
    return "supported" if lower > 0 else "falsified" if point <= 0 else "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--gpu", help="gpu.json (default: <metrics>/gpu.json)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    reps = cfg["verdicts"]["bootstrap"]
    m = Path(a.metrics)
    h1 = json.loads((m / "h1.json").read_text())
    h2 = json.loads((m / "h2_lexical.json").read_text())
    g = json.loads(Path(a.gpu or m / "gpu.json").read_text())
    audit = json.loads(Path(a.audit).read_text())
    V, lines = {}, []

    # H1
    parts = {}
    for p in ("n", "v"):
        r = h1[p]
        parts[p] = "insufficient data" if r["steps_with_spread"] < 200 else r["verdict"]
    if "insufficient data" in parts.values():
        V["H1"] = "insufficient data"
    elif "falsified" in parts.values():
        V["H1"] = "falsified"
    elif all(x == "supported" for x in parts.values()):
        V["H1"] = "supported"
    else:
        V["H1"] = "inconclusive"
    lines.append(f"**H1, abstraction climbs frequency and spread: {V['H1']}.** Share of WordNet steps where the more general "
                 f"word is both more common and more spread across fields: nouns {ci(h1['n']['both_up'])} of "
                 f"{h1['n']['steps_with_spread']:,} steps, verbs {ci(h1['v']['both_up'])} of {h1['v']['steps_with_spread']:,} "
                 f"(frequency alone: nouns {pct(h1['n']['frequency_up'][0])}, verbs {pct(h1['v']['frequency_up'][0])}; "
                 f"spread alone: nouns {pct(h1['n']['spread_up'][0])}, verbs {pct(h1['v']['spread_up'][0])}).")

    # H2
    d = h2["verbs_minus_nouns_test"]
    V["H2"] = judge(d[1], d[0])
    nv = g["variants"]
    neural = diff(nv["verbs_only"]["test"]["hits"], nv["nouns_only"]["test"]["hits"], reps) if "verbs_only" in nv else None
    lines.append(f"**H2, verbs carry the analogy: {V['H2']}.** Test recovery from verbs only {ci(h2['verbs_test']['top10'])} "
                 f"against nouns only {ci(h2['nouns_test']['top10'])}; difference {ci(d)} (chance {pct(h2['chance_test'])})."
                 + (f" With the neural encoder: difference {ci(neural)} (reported, not judged)." if neural else ""))

    # H3
    h3 = g["h3"]
    if "skipped" in h3:
        V["H3"] = "insufficient data"
        lines.append(f"**H3, a shared abstraction direction: insufficient data** ({h3['skipped']}).")
    else:
        d3 = diff(h3["map_hits"], h3["frequency_hits"], reps)
        V["H3"] = judge(d3[1], d3[0])
        lines.append(f"**H3, a shared abstraction direction: {V['H3']}.** On held-out sections {', '.join(h3['held_out_sections'])} "
                     f"({h3['pairs']['test']} words), the map finds the hypernym in its top 10 of {h3['candidates']} "
                     f"{ci(boot(h3['map_hits'], reps))}; frequency only {ci(boot(h3['frequency_hits'], reps))}; the word's own "
                     f"vector {ci(boot(h3['identity_hits'], reps))}. Map minus frequency {ci(d3)}.")

    # Grid: choose on val, judge on test.
    cells = {k: v for k, v in nv.items() if k.startswith("n") and "v" in k[1:]}
    lv = lambda k: (int(k[1:k.index("v")]), int(k[k.index("v") + 1:]))
    key = lambda k: (cells[k]["val"]["top10"], cells[k]["val"]["mrr"])
    best = max((k for k in cells if k != "n0v0"), key=key)
    bn, bv = lv(best)
    base = cells["n0v0"]
    gain = diff(cells[best]["test"]["hits"], base["test"]["hits"], reps)
    less_leak = cells[best]["leakage"] < base["leakage"]
    V["H4"] = "supported" if gain[1] > 0 and less_leak else "falsified" if gain[1] <= 0 else "inconclusive"
    lines.append(f"**H4, role replacement beats raw text and leaks less: {V['H4']}.** The best setting on val, nouns at level {bn} "
                 f"and verbs at level {bv}, recovers {ci(boot(cells[best]['test']['hits'], reps))} on test against "
                 f"{ci(boot(base['test']['hits'], reps))} for the abstracts as written: gain {ci(gain)}. Leakage "
                 f"{pct(cells[best]['leakage'])} against {pct(base['leakage'])} (chance 12.5%).")

    # H5: the noun dial at the val-best verb level.
    nls = sorted({lv(k)[0] for k in cells})
    top = nls[-1]
    row = {n: cells[f"n{n}v{bv}"] for n in nls if f"n{n}v{bv}" in cells}
    bn5 = max(row, key=lambda n: (row[n]["val"]["top10"], row[n]["val"]["mrr"]))
    test_best_lower = max(row[n]["test"]["top10"] for n in row if n < top)
    if bn5 < top:
        d5 = diff(row[bn5]["test"]["hits"], row[top]["test"]["hits"], reps)
        V["H5"] = "supported" if d5[1] > 0 else "falsified" if row[top]["test"]["top10"] >= test_best_lower else "inconclusive"
    else:
        d5 = None
        V["H5"] = "falsified" if row[top]["test"]["top10"] >= test_best_lower else "inconclusive"
    curve = ", ".join(f"level {n}: {pct(row[n]['test']['top10'])}" for n in row)
    lines.append(f"**H5, a best level of abstraction: {V['H5']}.** Test recovery as nouns are abstracted further (verbs at "
                 f"level {bv}): {curve}." + (f" Val-best level {bn5} minus level {top}: {ci(d5)}." if d5 else f" Val chose the top level ({top})."))

    # H6
    col = {v: cells[f"n{bn}v{v}"] for v in sorted({lv(k)[1] for k in cells}) if f"n{bn}v{v}" in cells}
    bv6 = max((v for v in col if v > 0), key=lambda v: (col[v]["val"]["top10"], col[v]["val"]["mrr"]))
    d6 = diff(col[bv6]["test"]["hits"], col[0]["test"]["hits"], reps)
    rs = g["rewrite_stats"][best]
    moved_less = (rs["v"]["mean_depth"] < rs["n"]["mean_depth"] and rs["v"]["mean_d_zipf"] < rs["n"]["mean_d_zipf"]
                  and rs["v"]["mean_d_spread"] < rs["n"]["mean_d_spread"])
    if d6[0] <= 0 or bv >= bn:
        V["H6"] = "falsified"
    elif d6[1] > 0 and moved_less:
        V["H6"] = "supported"
    else:
        V["H6"] = "inconclusive"
    lines.append(f"**H6, verbs need abstracting, but less than nouns: {V['H6']}.** (a) With nouns at level {bn}, verbs at level {bv6} "
                 f"against verbs as written: {ci(d6)}. (b) In the best setting, replaced words moved up WordNet by "
                 f"{rs['v']['mean_depth']} steps for verbs against {rs['n']['mean_depth']} for nouns; Zipf rose "
                 f"{rs['v']['mean_d_zipf']} against {rs['n']['mean_d_zipf']}; spread rose {rs['v']['mean_d_spread']} against "
                 f"{rs['n']['mean_d_spread']} bits; share of words replaced: verbs {pct(rs['v']['replaced_share'])}, nouns "
                 f"{pct(rs['n']['replaced_share'])}. (c) Not tested (no per-verb test was designed in advance).")

    # Recovery verdicts need the planned sample: the dry run "falsified" H2 and H4-H6 on 2 queries (10/05/2026).
    n_test = len(base["test"]["hits"])
    if n_test < MIN_TEST_QUERIES:
        for h in ("H2", "H4", "H5", "H6"):
            V[h] = f"insufficient data ({n_test} test queries; {MIN_TEST_QUERIES} needed)"

    grid = ["| nouns \\ verbs | " + " | ".join(str(v) for v in sorted({lv(k)[1] for k in cells})) + " |",
            "|---|" + "---|" * len({lv(k)[1] for k in cells})]
    for n in nls:
        grid.append(f"| {n} | " + " | ".join(f"{pct(cells[f'n{n}v{v}']['val']['top10'])} / {pct(cells[f'n{n}v{v}']['test']['top10'])}"
                                             for v in sorted({lv(k)[1] for k in cells})) + " |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = ["# Phase 10 report: historical recovery of cross-domain borrowings in patents", "",
              "Written by `scripts/p10-verdicts.py` from the pipeline's metrics, under the rules in "
              "`pipelines/phase10/OPERATIONALIZATION.md`, fixed before the data. Hypotheses: `HYPOTHESES.md`.", "",
              f"Data audit: {'passed' if audit['pass'] else 'FAILED'}; queries {audit['queries']}. Chance recovery: {pct(cfg['recovery']['top_k'] / cfg['recovery']['candidates'])} (top {cfg['recovery']['top_k']} of {cfg['recovery']['candidates']:,}; the dry run trims candidates).", "",
              "## Verdicts", "", "| Hypothesis | Verdict |", "|---|---|"]
    report += [f"| {h} | {v} |" for h, v in V.items()]
    report += ["", "## What was found", ""] + [l + "\n" for l in lines]
    report += ["## Recovery by setting (val / test, top 10 of 1,000)", "", *grid, "",
               f"Encoder {g['models']['embedder']}; substitutes chosen by {g['models']['substitute_lm']}."]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    strip = lambda x: {k: strip(v) for k, v in x.items() if k != "hits" and not k.endswith("_hits")} if isinstance(x, dict) else x
    (out / "metrics.json").write_text(json.dumps({"verdicts": V, "h1": h1, "h2": h2, "gpu": strip(g), "audit": audit}, indent=1))
    print(json.dumps(V))


if __name__ == "__main__":
    main()
