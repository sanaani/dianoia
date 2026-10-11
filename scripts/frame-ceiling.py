"""The frame ceiling: how consistent are Sonnet's own frames across rewordings of one sentence?

Every test item's source and its paraphrase, rename, register, negation and verb/object variants are
framed in one shuffled list (Sonnet is never shown which texts belong together). Then, mechanically:

- keep variants (paraphrase, rename, register) should give the source's frame: agreement is the best
  invariance any model trained on these targets could reach;
- move variants (negation, predicate_change) should give a different core frame: if they agree, the
  frames have collapsed distinctions that matter.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from dianoia import frames
from dianoia.paths import result_path, work_path

KINDS = ["paraphrase", "rename", "register", "negation", "predicate_change"]


def main() -> None:
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    test = items[items["split"] == "test"]
    literal = [e["entity"] for e in json.loads(result_path("phase-0-corpus", "literal-entities.json").read_text())["literal_entities"] if e["label"] != "PERSON"]
    rows = []
    for i, r in test.iterrows():
        rows.append({"item": int(i), "kind": "source", "text": r["text"]})
        for k in KINDS:
            if isinstance(r.get(k), str) and r[k]:
                rows.append({"item": int(i), "kind": k, "text": r[k]})
    random.Random(20261001).shuffle(rows)
    got = frames.frames_for([r["text"] for r in rows], literal)
    for r, f in zip(rows, got):
        r["frame"] = f
    df = pd.DataFrame(rows)
    df.to_parquet(work_path("ae", "frame-ceiling.parquet"))

    src = df[df["kind"] == "source"].set_index("item")["frame"]
    report = {"framed": int(df["frame"].notna().sum()), "texts": len(df), "kinds": {}}
    for k in KINDS:
        sub = df[(df["kind"] == k) & df["frame"].notna() & df["item"].map(src).notna()]
        ag = [frames.agreement(f, src[i]) for i, f in zip(sub["item"], sub["frame"])]
        if not ag:
            continue
        report["kinds"][k] = {"n": len(ag), **{s: round(float(np.mean([a[s] for a in ag])), 3) for s in [*frames.SLOTS, "core", "all"]}}
    result_path("phase-3-own-ae", "frame-ceiling.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {"n": v["n"], "core": v["core"], "all": v["all"], "action": v["action"], "act": v["act"]} for k, v in report["kinds"].items()}, indent=1))
    for i in list(src.index[:3]):
        print("SOURCE", src[i])
        for _, r in df[(df["item"] == i) & (df["kind"] != "source")].iterrows():
            print(f"  {r['kind']:16s}", r["frame"])


if __name__ == "__main__":
    main()
