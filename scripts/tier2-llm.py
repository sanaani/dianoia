"""Tier 2, exploratory decoder: Haiku turns each intent coordinate back into words.

Not the pre-registered decoder (that is lift + SONAR, scored in ``score-tier2.py``). Haiku sees only
the target's ten nearest admissible pool sentences by ft_sonar, with their cosines, never the source.
Round 0 proposes five reconstructions; rounds 1-3 see every candidate so far with its cosine to the
target and propose better ones. Candidates are scored by ft_sonar on the GPU, one batch per round.

Arms written for scoring: ``llm_first`` (Haiku's first guess, no encoder feedback at all) and
``llm_search`` (the best-scoring candidate after the rounds). Because the search climbs ft_sonar, its
landing is also judged by ft_bge and by the structured intent check, which it never saw.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from dianoia import decode
from dianoia.paths import work_path

IP = (Path.home() / "dianoia-data" / "instance-ip").read_text().strip()
KEY = str(Path.home() / ".ssh" / "dianoia.pem")
SSH = ["-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30"]


def remote_encode(texts: list[str]) -> np.ndarray:
    local = work_path("tier2", "llm-round.jsonl")
    with open(local, "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t or "."}, ensure_ascii=False) + "\n")
    subprocess.run(["scp", *SSH, str(local), f"ubuntu@{IP}:~/work/llm-round.jsonl"], check=True)
    subprocess.run(["ssh", *SSH, f"ubuntu@{IP}", "cd ~/work && source /opt/pytorch/bin/activate && python remote-reencode.py llm-round.jsonl out/llm-round --models sonar > llm-round.log 2>&1"], check=True)
    out = work_path("tier2", "llm-round-ft_sonar.npy")
    subprocess.run(["scp", *SSH, f"ubuntu@{IP}:~/work/out/llm-round/ft_sonar.npy", str(out)], check=True)
    return np.load(out)


def main() -> None:
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    index = pd.read_parquet(work_path("phase1", "index.parquet"))
    lines = index.pivot_table(index="item", columns="field", values="line", aggfunc="first")
    test = items.index[items["split"] == "test"].to_numpy()
    z = np.load(work_path("phase1", "vec", "ft_sonar.npy")).astype(np.float32)[lines.loc[test, "text"].astype(int).to_numpy()]
    zn = z / np.linalg.norm(z, axis=1, keepdims=True)
    pool = pd.read_parquet(work_path("pool", "pool.parquet"))
    pv = np.load(work_path("pool", "out", "ft_sonar.npy")).astype(np.float32)
    pv /= np.linalg.norm(pv, axis=1, keepdims=True)
    sents = pd.read_parquet(work_path("corpus", "sentences.parquet")).set_index("sid")
    ptexts, ppids = pool["text"].tolist(), pool["pid"].tolist()

    neighs = []
    for r, i in enumerate(test):
        top = np.argsort(-(pv @ zn[r]))[:300]
        adm = decode.exclusion(items.at[i, "text"], sents.at[items.at[i, "sid"], "pid"], [ptexts[j] for j in top], [ppids[j] for j in top])
        keep = top[np.flatnonzero(adm)[:10]]
        neighs.append([(ptexts[j], float(pv[j] @ zn[r])) for j in keep])

    out = decode.llm_decode("ft_sonar", zn, neighs, rounds=3, encode_fn=remote_encode)
    rows = []
    for i, d in zip(test, out):
        rows.append({"item": int(i), "arm": "llm_first", "text": d["first"]})
        rows.append({"item": int(i), "arm": "llm_search", "text": d["search_best"]})
    pd.DataFrame(rows).to_parquet(work_path("tier2", "llm.parquet"))
    with open(work_path("tier2", "llm-trace.jsonl"), "w", encoding="utf-8") as fh:
        for i, d, n in zip(test, out, neighs):
            fh.write(json.dumps({"item": int(i), "source": items.at[i, "text"], "neighbours": n, "trace": d["trace"]}, ensure_ascii=False) + "\n")
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
