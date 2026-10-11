"""Phase 14 step 3 verdict (results/phase-14-openalex/HYPOTHESES.md, step 3 details, 10/09/2026).

    python p14-verdicts.py --scores <work/p14/out/scores> --eval <work/p14/step3/eval.jsonl> --config <config>
                           --out <results/phase-14-openalex>

O1: far-trained minus off the shelf in top-10, paired bootstrap over the test pairs; passes if the 95% lower bound is
above 0. Reported, not tested: near-trained; top-10 by cited field and by Sonnet v4p's label (C or G).
Writes STEP3.md and step3.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import yaml

spec = importlib.util.spec_from_file_location("v13", Path(__file__).with_name("p13-verdicts.py"))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

NAMES = {"base": "off the shelf", "far": "far-trained", "near": "near-trained"}


def main():
    ap = argparse.ArgumentParser()
    for x in ("scores", "eval", "config", "out"):
        ap.add_argument(f"--{x}", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    n = cfg["step3"]["bootstrap"]
    ev = {r["query"]: r for r in map(json.loads, open(a.eval, encoding="utf-8"))}
    sc = {t: json.loads((Path(a.scores) / f"{t}.json").read_text()) for t in NAMES}
    hit = {t: {q: v["hit"] for q, v in sc[t].items()} for t in NAMES}
    mean = lambda d: round(sum(d.values()) / len(d), 4) if d else None
    table = {t: {"n": len(hit[t]), "top10": mean(hit[t]), "mrr": mean({q: v["rr"] for q, v in sc[t].items()})}
             for t in NAMES}
    o1 = V.boot(hit["far"], hit["base"], n)
    near = V.boot(hit["near"], hit["base"], n)
    groups = defaultdict(lambda: defaultdict(list))
    for q, r in ev.items():
        groups["label"][r["label"]].append(q)
        groups["field"][r["field"] or "none"].append(q)
    split = {g: {k: {"n": len(qs), **{t: mean({q: hit[t][q] for q in qs}) for t in NAMES}}
                 for k, qs in sorted(v.items(), key=lambda x: -len(x[1]))} for g, v in groups.items()}
    res = {"O1": {**o1, "pass": o1["lo"] > 0}, "near_minus_base": near, "table": table, "split": split}
    out = Path(a.out)
    (out / "step3.json").write_text(json.dumps(res, indent=1))
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    pts = lambda d: f"{100 * d['diff']:+.1f} points (95% interval {100 * d['lo']:+.1f} to {100 * d['hi']:+.1f}, {d['n']} pairs)"
    lines = ["# Phase 14 step 3: does the patent-trained far search carry over to management papers?", "",
             f"{len(ev)} strict far pairs (one per citing article), each citing abstract asked against its true source "
             "and 999 other works from the same field. Top-10 = the true source is among the 10 best of 1,000.", "",
             "| Encoder | Top-10 | Mean reciprocal rank |", "|---|---|---|"]
    lines += [f"| {NAMES[t]} | {pct(table[t]['top10'])} | {table[t]['mrr']:.3f} |" for t in NAMES]
    lines += ["", f"**O1 (far-trained beats off the shelf): {'PASS' if res['O1']['pass'] else 'FAIL'}.** "
              f"Far-trained minus off the shelf: {pts(o1)}.", "",
              f"- Near-trained minus off the shelf (reported): {pts(near)}.", "",
              "Reported, not tested: top-10 by Sonnet v4p's label and by cited field.", "",
              "| Group | n | " + " | ".join(NAMES.values()) + " |", "|---|---|---|---|---|"]
    for g in ("label", "field"):
        for k, v in split[g].items():
            lines.append(f"| {g}: {k} | {v['n']} | " + " | ".join(pct(v[t]) for t in NAMES) + " |")
    (out / "STEP3.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"O1": res["O1"], "table": {t: table[t]["top10"] for t in NAMES}}))


if __name__ == "__main__":
    main()
