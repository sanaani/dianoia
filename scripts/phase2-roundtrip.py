"""Phase 2: encode a sentence, decode the vector to words blind, re-encode, and measure the landing.

Targets are phase 1's items (each carries a paraphrase, which sets the tolerance). For a target x
decoded to x̂ under encoder E:

- cos_E(x, x̂), and the paraphrase tolerance tau_E(x) = cos_E(x, paraphrase(x)). A round trip
  *passes* if cos_E(x, x̂) >= tau_E(x): it landed as close as a faithful rewording does.
- The same two numbers under every other encoder E'. A decoder that searches against E can learn
  E's blind spots; landing close under an encoder it never saw is the check that it recovered the
  idea rather than gamed the metric.
- Retrieval: the rank of x among the pool plus x when the pool is queried with E(x̂).
- Word Jaccard(x, x̂): how many of the words came back. The hypothesis is that the idea returns
  while the words do not.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from dianoia import corpus, decode, encoders
from dianoia.paths import result_path, work_path

OUT = "phase-2-round-trip"


def pool_vectors(name: str, texts: list[str]) -> np.ndarray:
    path = work_path("pool", f"vec-{name}.npy")
    if path.exists():
        v = np.load(path)
        if len(v) == len(texts):
            return v
    v = encoders.encode(name, texts)
    np.save(path, v)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="the encoder whose vectors are decoded")
    ap.add_argument("--judges", nargs="*", default=["minilm", "bge", "e5", "gtr"], help="encoders that score the landing")
    ap.add_argument("--targets", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model", default="haiku")
    args = ap.parse_args()

    pool = pd.read_parquet(work_path("pool", "pool.parquet"))
    items = pd.read_parquet(work_path("phase1", "items.parquet")).iloc[: args.targets].reset_index(drop=True)
    sents = pd.read_parquet(work_path("corpus", "sentences.parquet")).set_index("sid")
    items["pid"] = items["sid"].map(sents["pid"])

    ptexts, ppids = pool["text"].tolist(), pool["pid"].tolist()
    pv = pool_vectors(args.encoder, ptexts)
    tv = encoders.encode(args.encoder, items["text"].tolist())

    neighs, nearest = [], []
    for i, row in items.iterrows():
        adm = decode.exclusion(row["text"], row["pid"], ptexts, ppids)
        idx, sims = decode.neighbours(tv[i], pv, adm, args.k)
        neighs.append([(ptexts[j], float(s)) for j, s in zip(idx, sims)])
        nearest.append(ptexts[idx[0]])

    dec = decode.llm_decode(args.encoder, tv, neighs, rounds=args.rounds, model=args.model)
    outputs = {
        "nearest": nearest,
        "llm_first": [d["first"] for d in dec],
        "llm_one_shot_best": [d["one_shot_best"] for d in dec],
        "llm_search": [d["search_best"] for d in dec],
        "paraphrase": items["paraphrase"].tolist(),
    }

    report: dict = {"encoder": args.encoder, "targets": len(items), "k": args.k, "rounds": args.rounds, "model": args.model, "judges": {}}
    for judge in args.judges:
        jv = encoders.encode(judge, items["text"].tolist())
        jpool = pool_vectors(judge, ptexts)
        tau = (jv * encoders.encode(judge, items["paraphrase"].tolist())).sum(1)
        rows = {}
        for name, texts in outputs.items():
            safe = [t if t else "." for t in texts]
            hv = encoders.encode(judge, safe)
            cos = (jv * hv).sum(1)
            # rank of the true source among pool + source, queried with the decoded vector
            q_pool = jpool @ hv.T  # (pool, targets)
            q_src = cos
            rank = 1 + (q_pool > q_src[None, :]).sum(0)
            rows[name] = {
                "cos_median": round(float(np.median(cos)), 4),
                "pass_rate": round(float((cos >= tau).mean()), 3),
                "top1": round(float((rank == 1).mean()), 3),
                "top10": round(float((rank <= 10).mean()), 3),
                "rank_median": int(np.median(rank)),
            }
        rows["tau_median"] = round(float(np.median(tau)), 4)
        report["judges"][judge] = rows
    report["word_jaccard_median"] = {
        name: round(float(np.median([decode_jaccard(a, b) for a, b in zip(items["text"], texts)])), 3)
        for name, texts in outputs.items()
    }

    tag = f"{args.encoder}-{args.model}"
    result_path(OUT, f"report-{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    with open(result_path(OUT, f"decodes-{tag}.jsonl"), "w", encoding="utf-8") as f:
        for i, row in items.iterrows():
            f.write(decode.dumps({"sid": row["sid"], "source": row["text"], **{k: v[i] for k, v in outputs.items()}, "trace": dec[i]["trace"]}) + "\n")
    print(json.dumps(report, indent=2))


def decode_jaccard(a: str, b: str) -> float:
    wa, wb = {w.lower() for w in corpus.words(a)}, {w.lower() for w in corpus.words(b)}
    return len(wa & wb) / max(len(wa | wb), 1)


if __name__ == "__main__":
    main()
