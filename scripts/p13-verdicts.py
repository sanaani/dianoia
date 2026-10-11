"""Phase 13 step 1 verdict (results/phase-13-scale-far/HYPOTHESES.md, approved 10/07/2026).

    python p13-verdicts.py --scores <work/p13/out/scores> --p12 <work/p12/out/scores> --short <short.jsonl>
                           --config <config> --out <results/phase-13-scale-far>

S1: on Background queries, far-trained minus off the shelf in far test top-10, paired bootstrap over the questions
that have a Background opening; passes if the 95% lower bound is above 0. Reported, not tested: the same on titles,
and each encoder's top-10 by query form. Check: the abstract form must reproduce phase 12's per-question hits (same
queries, candidates and adapters); a mismatch on more than 2% of questions stops the report.
Writes STEP1.md and step1.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

FORMS = ("abstract", "background", "title")
NAMES = {"base": "off the shelf", "far": "far-trained", "near": "near-trained"}


def boot(a, b, n, seed=0):
    """Mean of a - b and its 95% interval, resampling questions."""
    ids = sorted(set(a) & set(b))
    d = [a[i] - b[i] for i in ids]
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return {"n": len(d), "diff": round(sum(d) / len(d), 4), "lo": round(ms[int(0.025 * n)], 4),
            "hi": round(ms[int(0.975 * n) - 1], 4)}


def main():
    ap = argparse.ArgumentParser()
    for x in ("scores", "p12", "short", "config", "out"):
        ap.add_argument(f"--{x}", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    n = cfg["step1"]["bootstrap"]
    sc = {t: json.loads((Path(a.scores) / f"{t}.json").read_text()) for t in NAMES}
    hit = {t: {f: {q: v["hit"] for q, v in sc[t][f].items()} for f in FORMS} for t in NAMES}
    rr = {t: {f: {q: v["rr"] for q, v in sc[t][f].items()} for f in FORMS} for t in NAMES}

    # Reproduction check against phase 12's GPU scores (test split, same question order as eval.jsonl).
    p12 = {"base": "base", "far": Path(cfg["step1"]["adapters"]["far"]).name, "near": Path(cfg["step1"]["adapters"]["near"]).name}
    order = [str(r["query"]) for r in map(json.loads, open(a.short, encoding="utf-8")) if r["split"] == "test"]
    repro = {}
    for t, f in p12.items():
        old = json.loads((Path(a.p12) / f"{f}.json").read_text())["test"]["hits"]
        new = [hit[t]["abstract"][q] for q in order]
        repro[t] = sum(x != y for x, y in zip(old, new))
    if max(repro.values()) > 0.02 * len(order):
        raise SystemExit(f"abstract scores do not reproduce phase 12: {repro} questions differ")

    table = {t: {f: {"n": len(hit[t][f]), "top10": round(sum(hit[t][f].values()) / len(hit[t][f]), 4),
                     "mrr": round(sum(rr[t][f].values()) / len(rr[t][f]), 4)} for f in FORMS} for t in NAMES}
    s1 = boot(hit["far"]["background"], hit["base"]["background"], n)
    title = boot(hit["far"]["title"], hit["base"]["title"], n)
    far_near_bg = boot(hit["far"]["background"], hit["near"]["background"], n)
    res = {"S1": {**s1, "pass": s1["lo"] > 0}, "title_far_minus_base": title, "background_far_minus_near": far_near_bg,
           "table": table, "repro_mismatches": repro}
    out = Path(a.out)
    (out / "step1.json").write_text(json.dumps(res, indent=1))
    pct = lambda x: f"{100 * x:.1f}%"
    pts = lambda d: f"{100 * d['diff']:+.1f} points (95% interval {100 * d['lo']:+.1f} to {100 * d['hi']:+.1f}, {d['n']} questions)"
    lines = ["# Phase 13 step 1: short questions", "",
             "Phase 12's 291 far test questions, asked three ways, against the same 1,000 candidates each. Top-10 = the "
             "true source is among the 10 best of 1,000.", "",
             "| Encoder | Abstract | Background opening | Title |", "|---|---|---|---|"]
    for t in NAMES:
        lines.append(f"| {NAMES[t]} | " + " | ".join(f"{pct(table[t][f]['top10'])} (n={table[t][f]['n']})" for f in FORMS) + " |")
    lines += ["", f"**S1 (far-trained beats off the shelf on Background queries): {'PASS' if res['S1']['pass'] else 'FAIL'}.** "
              f"Far-trained minus off the shelf: {pts(s1)}.", "",
              f"- Titles, far-trained minus off the shelf (reported, not tested): {pts(title)}.",
              f"- Background, far-trained minus near-trained (reported): {pts(far_near_bg)}.",
              f"- Abstract scores reproduce phase 12's: {repro} questions differ (GPU rounding).", ""]
    (out / "STEP1.md").write_text("\n".join(lines))
    print(json.dumps({"S1": res["S1"], "table": {t: {f: table[t][f]["top10"] for f in FORMS} for t in NAMES}}))


if __name__ == "__main__":
    main()
