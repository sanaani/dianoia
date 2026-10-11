"""Tier 2 inputs: the SONAR vectors to decode for every test item, plus the nearest-neighbor floor.

Arms, each one row per test item:
- ``lift64`` / ``lift128`` / ``lift256``: the item's ft_sonar intent vector cut to that prefix, lifted
  into SONAR space. These are the round trip under test.
- ``sonar_self``: the source's own frozen SONAR vector. SONAR's decoder on it is the content-complete
  ceiling: the words come back too, so it shows what *not* discarding anything looks like.
The floor needs no decoding: the nearest admissible pool sentence by ft_sonar (source paragraph, exact
duplicates and Jaccard >= 0.5 near-copies excluded), written straight to the decoded table.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from dianoia import decode
from dianoia.paths import work_path

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("train_lift", Path(__file__).parent / "train-lift.py")
train_lift = importlib.util.module_from_spec(spec)
sys.modules["train_lift"] = train_lift
spec.loader.exec_module(train_lift)


def main() -> None:
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    index = pd.read_parquet(work_path("phase1", "index.parquet"))
    lines = index.pivot_table(index="item", columns="field", values="line", aggfunc="first")
    test = items.index[items["split"] == "test"].to_numpy()
    src = lines.loc[test, "text"].astype(int).to_numpy()
    z = torch.from_numpy(np.load(work_path("phase1", "vec", "ft_sonar.npy")).astype(np.float32)[src])
    sonar = np.load(work_path("phase1", "vec", "sonar.npy")).astype(np.float32)

    lift = train_lift.Lift()
    lift.load_state_dict(torch.load(work_path("phase1", "lift-ft_sonar-pool.pt")))
    lift.eval()
    rows, vecs = [], []
    with torch.no_grad():
        for d in (64, 128, 256):
            v = lift(train_lift.cut(z, d)).numpy()
            vecs.append(v)
            rows += [{"item": int(i), "arm": f"lift{d}"} for i in test]
    vecs.append(sonar[src])
    rows += [{"item": int(i), "arm": "sonar_self"} for i in test]
    np.save(work_path("tier2", "to-decode.npy"), np.concatenate(vecs).astype(np.float32))
    pd.DataFrame(rows).to_parquet(work_path("tier2", "to-decode.parquet"))

    pool = pd.read_parquet(work_path("pool", "pool.parquet"))
    pv = np.load(work_path("pool", "out", "ft_sonar.npy")).astype(np.float32)
    pv /= np.linalg.norm(pv, axis=1, keepdims=True)
    sents = pd.read_parquet(work_path("corpus", "sentences.parquet")).set_index("sid")
    zn = (z / z.norm(dim=1, keepdim=True)).numpy()
    nearest = []
    ptexts, ppids = pool["text"].tolist(), pool["pid"].tolist()
    for r, i in enumerate(test):
        top = np.argsort(-(pv @ zn[r]))[:200]
        adm = decode.exclusion(items.at[i, "text"], sents.at[items.at[i, "sid"], "pid"], [ptexts[j] for j in top], [ppids[j] for j in top])
        j = int(top[np.flatnonzero(adm)[0]])
        nearest.append({"item": int(i), "arm": "nearest", "text": ptexts[j]})
    pd.DataFrame(nearest).to_parquet(work_path("tier2", "nearest.parquet"))
    print(json.dumps({"to_decode": len(rows), "nearest": len(nearest)}))


if __name__ == "__main__":
    main()
