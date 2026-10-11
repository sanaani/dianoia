"""What does a compact idea vector cost in retrieval quality? (10/05/2026)

The storage figures (1,024 float32 numbers per idea, 256-d int8 ~10x smaller, binary ~100x smaller) were measured;
their quality cost was not. On J7's test pool (every held-out idea) this ranks with each compact form and compares
with full float32:

- recall@10: share of the full-vector top 10 neighbors (self excluded) that the compact form also returns
- next_median_rank: rank of a conversation's true next idea when the query is the current idea (the "last idea"
  baseline), the same pool-ranking task as remote-jepa.py, so a change shows directly in the metric we report

PCA is fit on training ideas, never the pool. int8 is symmetric per dimension; binary is the sign of each number,
ranked by Hamming distance.

    python remote-quant.py --units jepa/units-j7.jsonl --vec-cache jepa/vec-cache.pt --out jepa/Q1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def int8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x / scale).round().clamp(-127, 127) * scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--vec-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--queries", type=int, default=5000)
    ap.add_argument("--fit", type=int, default=100_000, help="training ideas the PCA and int8 scales are fit on")
    args = ap.parse_args()
    units = [json.loads(l) for l in open(args.units, encoding="utf-8")]
    test = [u for u in units if u["split"] == "test" and u.get("kind") != "paraphrase"]
    pool_t = list(dict.fromkeys(t for u in test for t in u["ideas"]))
    train_t = list(dict.fromkeys(t for u in units if u["split"] == "train" for t in u["ideas"]))
    rng = random.Random(0)
    fit_t = rng.sample(train_t, min(args.fit, len(train_t)))

    cache = torch.load(args.vec_cache, mmap=True)
    pos = {t: k for k, t in enumerate(cache["texts"])}

    def rows(ts):
        idx = torch.tensor([pos[t] for t in ts])
        order = torch.argsort(idx)
        out = torch.empty(len(ts), cache["raw"].shape[1])
        out[order] = cache["raw"][idx[order]]
        return F.normalize(out, dim=-1).cuda()

    P, Fit = rows(pool_t), rows(fit_t)
    index = {t: k for k, t in enumerate(pool_t)}
    q = torch.tensor(rng.sample(range(len(pool_t)), min(args.queries, len(pool_t)))).cuda()
    pairs = [(index[a], index[b]) for u in test for a, b in zip(u["ideas"], u["ideas"][1:]) if a != b]
    cur = torch.tensor([a for a, _ in pairs]).cuda()
    nxt = torch.tensor([b for _, b in pairs]).cuda()

    mean = Fit.mean(0)
    _, _, V = torch.linalg.svd(Fit - mean, full_matrices=False)
    proj = V[:256].T

    def pca(x):
        return F.normalize((x - mean) @ proj, dim=-1)

    s_full = Fit.abs().amax(0) / 127
    s_pca = pca(Fit).abs().amax(0) / 127
    forms = {
        "float32_1024": (P, "cos"),
        "int8_1024": (int8(P, s_full), "cos"),
        "float32_pca256": (pca(P), "cos"),
        "int8_pca256": (int8(pca(P), s_pca), "cos"),
        "binary_1024": ((P - mean > 0).float(), "ham"),
        "binary_pca256": ((pca(P) > 0).float(), "ham"),
    }

    def scores(X, kind, qi):
        if kind == "cos":
            return X[qi] @ X.T
        # Hamming similarity on 0/1 vectors: matches = x.y + (1-x).(1-y)
        A = X[qi]
        return A @ X.T + (1 - A) @ (1 - X).T

    # Ties are common in Hamming scores (whole numbers); a tiny random jitter breaks them without favoring index order.
    # Only there: the same jitter on cosine scores reshuffled close neighbors, so full float32 recalled itself at
    # 0.993 (Q1, 10/05/2026).
    def jitter(sc, kind):
        return sc + torch.rand_like(sc) * 1e-3 if kind == "ham" else sc

    def topk(X, kind, qi, k=10):
        out = []
        for s in range(0, len(qi), 1000):
            sc = scores(X, kind, qi[s : s + 1000]).float()
            sc[torch.arange(sc.shape[0]), qi[s : s + 1000]] = -1e9
            out.append(jitter(sc, kind).topk(k).indices)
        return torch.cat(out)

    def next_rank(X, kind):
        ranks = []
        for s in range(0, len(cur), 1000):
            c, n = cur[s : s + 1000], nxt[s : s + 1000]
            sc = jitter(scores(X, kind, c).float(), kind)
            sc[torch.arange(len(c)), c] = -1e9
            ranks.append((sc > sc.gather(1, n[:, None])).sum(1) + 1)
        return float(torch.cat(ranks).float().median())

    ref = topk(*forms["float32_1024"], q)
    result = {"pool": len(pool_t), "queries": len(q), "next_pairs": len(pairs), "fit": len(fit_t)}
    for name, (X, kind) in forms.items():
        got = topk(X, kind, q)
        recall = float(sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ref, got)) / (10 * len(q)))
        result[name] = {"recall@10": round(recall, 3), "next_median_rank": next_rank(X, kind)}
        print(name, result[name], flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "quant.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
