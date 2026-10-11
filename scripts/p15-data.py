"""Phase 15 data (results/phase-15-curve-screen/HYPOTHESES.md, approved 10/09/2026).

    python p15-data.py curve  --config <config.used.yaml> --work <work>   nested subsets of phase 12's far training pairs
    python p15-data.py screen --config <config.used.yaml> --work <work>   phase 14's Sonnet-labeled far pairs for Fastino

curve:  p15/curve/train-<n>.jsonl for each size, nested (the first n of one shuffle, seed 15), rows unchanged.
screen: p15/screen/pairs.jsonl, one row per far pair Sonnet v3p labeled: split (fit, val or test, by citing article),
        y (passes the strict rule), y3 (v3p C with borrow >= 4) and the input text Fastino reads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

import yaml

spec = importlib.util.spec_from_file_location("p14", Path(__file__).with_name("p14-data.py"))
P14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P14)


def _cfg(a):
    return yaml.safe_load(open(a.config)), Path(a.work)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_curve(a):
    cfg, work = _cfg(a)
    c = cfg["curve"]
    src = work / c["train"]
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    order = list(range(len(rows)))
    random.Random(c["seed"]).shuffle(order)
    out = work / "p15/curve"
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for n in c["sizes"]:
        keep = set(order[:n])
        P14._write(out / f"train-{n}.jsonl", [r for i, r in enumerate(rows) if i in keep])
        made[n] = _sha(out / f"train-{n}.jsonl")
    (out / "manifest.json").write_text(json.dumps({"source": c["train"], "source_sha256": _sha(src),
                                                   "rows": len(rows), "seed": c["seed"], "sha256": made}, indent=1))
    print(json.dumps({"rows": len(rows), "sizes": c["sizes"]}))


def cmd_screen(a):
    cfg, work = _cfg(a)
    s = cfg["screen"]
    p14 = work / "p14"
    mb = s["min_borrow"]
    v3 = {**P14._sonnet(p14, "v3p"), **P14._sonnet(p14, "v3p-rest")}
    v4 = {**P14._sonnet(p14, "v4p"), **P14._sonnet(p14, "v4p-rest")}
    gate = lambda r: r["label"] == "C" and (r["borrow"] or 0) >= mb
    texts = {r["id"]: r for r in map(json.loads, open(p14 / "far-text.jsonl", encoding="utf-8")) if r["id"] in v3}
    arts = sorted({i.split(">")[0] for i in texts})
    rng = random.Random(s["seed"])
    rng.shuffle(arts)
    n_test = round(len(arts) * s["test_share"])
    rest = arts[n_test:]
    n_val = round(len(rest) * s["val_share"])
    split = {**{x: "test" for x in arts[:n_test]}, **{x: "val" for x in rest[:n_val]}, **{x: "fit" for x in rest[n_val:]}}
    w = s["words"]
    rows = []
    for i in sorted(texts):
        r = texts[i]
        y3 = gate(v3[i])
        rows.append({"id": i, "citing": r["citing"], "split": split[r["citing"]], "y3": int(y3),
                     "y": int(y3 and i in v4 and v4[i]["label"] in ("C", "G")),
                     "text": f"Citing paper: {P14._cut(r['citing_text'], w)}\n\nCited work: {P14._cut(r['cited_text'], w)}"})
    out = work / "p15/screen"
    out.mkdir(parents=True, exist_ok=True)
    P14._write(out / "pairs.jsonl", rows)
    n = Counter((r["split"], r["y"]) for r in rows)
    print(json.dumps({"pairs": len(rows), "articles": len(arts), "positives": sum(r["y"] for r in rows),
                      "by_split": {k: {"n": n[(k, 0)] + n[(k, 1)], "pos": n[(k, 1)]} for k in ("fit", "val", "test")}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("curve", "screen"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    {"curve": cmd_curve, "screen": cmd_screen}[a.cmd](a)


if __name__ == "__main__":
    main()
