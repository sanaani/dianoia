"""The idea-object check: is the structure Sonnet extracts the same however a unit is worded, punctuated or split?

Every version of every unit (``work/ideas/versions.jsonl``, from ``idea-data.py``) goes into one shuffled list,
so the extractor never sees which texts belong together or that variants exist. Then, mechanically, each
version's object is aligned to its original's (``ideas.Aligner``) and compared layer by layer
(``ideas.compare``); move versions are also tested for a change at the edited turn (``ideas.changed_at``).
Bars are in PREREGISTRATION.md (10/01/2026 addendum, "idea objects").
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

import numpy as np

from dianoia import ideas
from dianoia.paths import result_path, work_path

KEEP = ["copy", "paraphrase", "stripped", "disfluent", "split"]
LAYERS = ["event_f1", "count_equal", "act", "act_coarse", "neg", "mod", "pred", "content_cos", "rel_f1", "rel_label",
          "rel_coarse", "center"]
BARS = {"event_f1": 0.85, "act_coarse": 0.85, "neg": 0.90, "mod": 0.80, "rel_f1": 0.70, "rel_coarse": 0.75, "center": 0.70}


def main() -> None:
    versions = [json.loads(line) for line in open(work_path("ideas", "versions.jsonl"), encoding="utf-8")]
    order = list(range(len(versions)))
    random.Random(20261001).shuffle(order)
    got = ideas.extract([versions[i]["turns"] for i in order])
    objs = [None] * len(versions)
    for i, o in zip(order, got):
        objs[i] = o
    with open(work_path("ideas", "objects.jsonl"), "w", encoding="utf-8") as fh:
        for v, o in zip(versions, objs):
            fh.write(json.dumps({**v, "object": o}, ensure_ascii=False) + "\n")

    by = defaultdict(dict)
    for v, o in zip(versions, objs):
        by[v["uid"]][v["kind"]] = (v, o)
    al = ideas.Aligner()
    cells = defaultdict(lambda: defaultdict(list))
    sens = defaultdict(lambda: defaultdict(list))
    for uid, kinds in by.items():
        if "original" not in kinds or kinds["original"][1] is None:
            continue
        ov, oo = kinds["original"]
        medium = ov["medium"]
        cells[(medium, "original")]["multi_event"].append(len(oo["events"]) > 1)
        cells[(medium, "original")]["n_events"].append(len(oo["events"]))
        for kind, (v, o) in kinds.items():
            if kind == "original" or o is None:
                continue
            pairs = al.align(oo, o, v["turn_map"])
            for k, x in ideas.compare(oo, o, pairs).items():
                cells[(medium, kind)][k].append(float(x))
            if kind == "move":
                t = v["changed_turn"]
                sens[medium]["move"].append(ideas.changed_at(oo, o, pairs, t, v["turn_map"]))
                for base in ("copy", "paraphrase"):
                    if base in kinds and kinds[base][1] is not None:
                        bv, bo = kinds[base]
                        sens[medium][base].append(ideas.changed_at(oo, bo, al.align(oo, bo, bv["turn_map"]), t, bv["turn_map"]))

    report = {"extracted": sum(o is not None for o in objs), "versions": len(versions), "bars": BARS, "cells": {}, "sensitivity": {}}
    for (medium, kind), d in sorted(cells.items()):
        report["cells"][f"{medium}/{kind}"] = {"n": len(next(iter(d.values()))), **{k: round(float(np.mean(x)), 3) for k, x in d.items()}}
    for medium, d in sens.items():
        report["sensitivity"][medium] = {k: {"n": len(x), "changed": round(float(np.mean(x)), 3)} for k, x in d.items()}
    verdict = {}
    for name, c in report["cells"].items():
        kind = name.split("/")[1]
        if kind in KEEP[1:]:
            verdict[name] = {k: ("pass" if c.get(k, 0) >= bar else "FAIL") for k, bar in BARS.items()}
    report["verdict"] = verdict
    report["sensitivity_pass"] = {m: d["move"]["changed"] >= 0.80 for m, d in report["sensitivity"].items()}
    result_path("phase-4-idea-objects", "ceiling.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, c in report["cells"].items():
        print(name, {k: c.get(k) for k in ["n", *LAYERS, "multi_event", "n_events"] if k in c})
    print(json.dumps(report["sensitivity"], indent=1))
    fails = [f"{n}:{k}" for n, d in verdict.items() for k, s in d.items() if s == "FAIL"]
    print("FAILS", len(fails), fails)


if __name__ == "__main__":
    main()
