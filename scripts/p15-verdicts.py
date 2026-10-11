"""Phase 15 verdicts (results/phase-15-curve-screen/HYPOTHESES.md, approved 10/09/2026).

    python p15-verdicts.py --work <work> --out <work/p15/out> --config <config> --results <results/phase-15-curve-screen>

Step 1 (L1): full (phase 12's far-s0-e2) minus half on top-10 over the pooled patent and paper test queries, paired
bootstrap (p13-verdicts.boot). Step 2 (S1): the share of held-out strict positives kept when the lowest-scored half
is discarded, with its exact (Clopper-Pearson, two-sided 95%) lower bound. Writes STEP1.md, step1.json, STEP2.md,
step2.json; either step is skipped while its GPU output is missing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import yaml
from scipy.stats import beta

spec = importlib.util.spec_from_file_location("v13", Path(__file__).with_name("p13-verdicts.py"))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

pct = lambda x: f"{100 * x:.1f}%"
pts = lambda d: f"{100 * d['diff']:+.1f} points (95% interval {100 * d['lo']:+.1f} to {100 * d['hi']:+.1f}, {d['n']} queries)"


def step1(cfg, work, out, res_dir):
    c = cfg["curve"]
    rd = lambda p: json.loads((work / p).read_text())
    q12 = [r["query"] for r in map(json.loads, open(work / "p12/eval.jsonl", encoding="utf-8")) if r["split"] == "test"]
    p12 = lambda p: {f"patent:{q}": h for q, h in zip(q12, rd(p)["test"]["hits"])}
    p14 = lambda p: {f"paper:{q}": v["hit"] for q, v in rd(p).items()}
    sub = lambda n, s: {f"{s}:{q}": v["hit"] for q, v in json.loads((out / "scores" / f"far{n}-s0-{s}.json").read_text()).items()}
    if not all((out / "scores" / f"far{n}-s0-{s}.json").exists() for n in c["sizes"] for s in ("patent", "paper")):
        return None
    full = c["full"]
    models = {0: {**p12(full["patent"]["base"]), **p14(full["paper"]["base"])},
              **{n: {**sub(n, "patent"), **sub(n, "paper")} for n in c["sizes"]},
              1676: {**p12(full["patent"]["far"]), **p14(full["paper"]["far"])}}
    mean = lambda d, src=None: sum(v for k, v in d.items() if src is None or k.startswith(src)) / \
        max(1, sum(1 for k in d if src is None or k.startswith(src)))
    table = {n: {"pooled": round(mean(m), 4), "patent": round(mean(m, "patent:"), 4), "paper": round(mean(m, "paper:"), 4)}
             for n, m in models.items()}
    half, quarter = sorted(c["sizes"])[-1], sorted(c["sizes"])[0]
    n = c["bootstrap"]
    l1 = V.boot(models[1676], models[half], n)
    verdict = "still climbing" if l1["lo"] > c["climbing_lo"] else "flat" if l1["hi"] < c["flat_hi"] else "unclear"
    by_src = {s: V.boot({k: v for k, v in models[1676].items() if k.startswith(s)}, models[half], n) for s in ("patent:", "paper:")}
    qh = V.boot(models[half], models[quarter], n)
    seeds = [round(mean(p12(p)), 4) for p in [full["patent"]["far"], *full["patent"]["seeds"]]]
    gpu = json.loads((out / "p15.json").read_text()) if (out / "p15.json").exists() else {}
    res = {"L1": {**l1, "verdict": verdict}, "by_source": by_src, "half_minus_quarter": qh, "table": table,
           "patent_seed_spread_full": seeds, "chosen_epochs": gpu.get("curve", {}).get("chosen_epochs")}
    (res_dir / "step1.json").write_text(json.dumps(res, indent=1))
    lines = ["# Phase 15 step 1: is the far search still improving with more data?", "",
             "Top-10 (true source among the 10 best of 1,000) for the encoder trained on nested shares of phase 12's 1,676 "
             "far patent pairs. 0 is the encoder off the shelf; 1,676 is phase 12's far-s0-e2.", "",
             "| Training pairs | Pooled (546) | Patent test (291) | Paper test (255) |", "|---|---|---|---|"]
    lines += [f"| {k:,} | {pct(v['pooled'])} | {pct(v['patent'])} | {pct(v['paper'])} |" for k, v in sorted(table.items())]
    lines += ["", f"**L1: {verdict.upper()}.** Full minus half (pooled): {pts(l1)}. "
              f"Rule: still climbing if the lower bound is above 0, flat if the upper bound is below +3 points.", "",
              f"- Patent test only: {pts(by_src['patent:'])}.",
              f"- Paper test only: {pts(by_src['paper:'])}.",
              f"- Half minus quarter (pooled, reported): {pts(qh)}.",
              f"- Phase 12's three seeds at full size, patent test: {', '.join(pct(x) for x in seeds)} (the subsets use one seed).",
              f"- Epochs chosen on patent dev: {res['chosen_epochs']}."]
    (res_dir / "STEP1.md").write_text("\n".join(lines) + "\n")
    return res


def cp_lo(k, n):
    return 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))


def step2(cfg, out, res_dir):
    s = cfg["screen"]
    f = out / "screen/scores.jsonl"
    if not f.exists():
        return None
    rows = sorted(map(json.loads, open(f, encoding="utf-8")), key=lambda r: (r["p"], r["id"]))
    N = len(rows)

    def kept(frac, key):
        cut = int(frac * N)
        tot = sum(r[key] for r in rows)
        k = sum(r[key] for r in rows[cut:])
        return {"discard": frac, "kept": k, "of": tot, "share": round(k / max(1, tot), 4), "lo": round(cp_lo(k, tot), 4)}

    s1 = kept(s["discard"], "y")
    s1["pass"] = s1["lo"] >= s["min_recall_lo"]
    curve = [kept(x / 10, "y") for x in range(1, 10)]
    curve3 = [kept(x / 10, "y3") for x in range(1, 10)]
    best = max((x / 100 for x in range(0, 100) if kept(x / 100, "y")["lo"] >= s["min_recall_lo"]), default=0.0)
    pos = [i for i, r in enumerate(rows) if r["y"]]
    auc = sum(i - j for j, i in enumerate(pos)) / max(1, len(pos) * (N - len(pos)))
    meta = json.loads((out / "screen/screen.json").read_text())
    res = {"S1": s1, "curve": curve, "curve_v3_gate": curve3, "largest_discard_passing": best, "auc": round(auc, 4),
           "test_pairs": N, **meta}
    (res_dir / "step2.json").write_text(json.dumps(res, indent=1))
    lines = ["# Phase 15 step 2: can Fastino throw away the worst half before Sonnet reads?", "",
             f"Fastino GLiNER2.5-Decide fine-tuned on {meta['rows']['fit']:,} examples ({meta['rows']['fit_pos']} positive, "
             f"positives repeated {s['pos_repeat']} times), scored on {N:,} held-out phase 14 far pairs from articles it "
             f"never saw, {s1['of']} of them strict positives.", "",
             f"**S1: {'PASS' if s1['pass'] else 'FAIL'}.** Discarding the lowest-scored {pct(s['discard'])} keeps "
             f"{s1['kept']} of {s1['of']} strict positives ({pct(s1['share'])}; exact 95% lower bound {pct(s1['lo'])}; "
             f"bar {pct(s['min_recall_lo'])}).", "",
             "| Discarded | Strict positives kept | Lower bound | v3p gate kept | At random |", "|---|---|---|---|---|"]
    lines += [f"| {pct(a['discard'])} | {pct(a['share'])} ({a['kept']}/{a['of']}) | {pct(a['lo'])} | {pct(b['share'])} | "
              f"{pct(1 - a['discard'])} |" for a, b in zip(curve, curve3)]
    lines += ["", f"- Largest discard whose lower bound stays at or above {pct(s['min_recall_lo'])}: {pct(best)}.",
              f"- Area under the ROC curve: {auc:.3f} (0.5 is chance).",
              f"- GPU: training {meta.get('train_minutes')} min, scoring {meta.get('score_minutes')} min."]
    (res_dir / "STEP2.md").write_text("\n".join(lines) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser()
    for x in ("work", "out", "config", "results"):
        ap.add_argument(f"--{x}", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    res_dir = Path(a.results)
    r1 = step1(cfg, Path(a.work), Path(a.out), res_dir)
    r2 = step2(cfg, Path(a.out), res_dir)
    print(json.dumps({"L1": r1 and r1["L1"], "S1": r2 and r2["S1"]}))


if __name__ == "__main__":
    main()
