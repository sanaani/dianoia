"""First JEPA test: predict the next idea's vector from the ideas before it (GPU; 10/04/2026).

Question: does the idea space carry structure a small predictor can use to anticipate where a conversation goes
next, beyond trivial baselines? Data: ``jepa/units.jsonl`` (``jepa-prepare.py``), each unit's reference ideas in
order. Every idea is encoded once with the idea encoder (frozen); the predictor sees only vectors.

Predictor: a small causal transformer over the normalized idea vectors of a unit (plus a learned position
embedding); at every position it outputs the next idea's vector, as a residual step from the current idea (zero at
initialization, so it starts as the "last" baseline). Loss: InfoNCE against every target idea in the batch
(temperature 0.05), so it must point at the right idea, not at the average idea, plus a cosine term.
J1 (no residual, no cosine term, 3,000 steps) underfit and lost to every baseline.

Evaluation on held-out units, for every position after the first: the rank of the true next idea among all
held-out ideas (pool), top-1 / top-10, median rank, and mean cosine, against baselines that need no training:
- **last**: the previous idea itself;
- **context mean**: the mean of all previous ideas;
- **same unit** (an oracle-ish ceiling on locality): the best of the unit's other ideas.
A few predictions are decoded with the decoder for reading. Output: ``<out>/jepa.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
import random
import statistics as st
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Predictor(torch.nn.Module):
    def __init__(self, d: int = 1024, layers: int = 4, heads: int = 8, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.pos = torch.nn.Embedding(max_len, d)
        layer = torch.nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.tf = torch.nn.TransformerEncoder(layer, layers)
        self.out = torch.nn.Linear(d, d)
        # Residual from the last idea: at initialization the prediction is the previous idea itself (the strongest
        # baseline), and training learns the step from it.
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        h = x + self.pos(torch.arange(n, device=x.device))
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
        return F.normalize(x + self.out(self.tf(h, mask=causal, src_key_padding_mask=pad)), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="jepa/units.jsonl")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cos-weight", type=float, default=1.0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-windows", action="store_true", help="validate on whole held-out chained conversations")
    ap.add_argument("--require-audit", action="store_true", help="refuse to train without a passing audit of --units")
    ap.add_argument("--eval-only", default="", help="score this predictor.pt on --units' test split; no training")
    ap.add_argument("--mix-alpha", type=float, default=1.0,
                    help="source mix: each unit drawn with weight (its medium's unit count)^(alpha-1); 1 = uniform over units")
    ap.add_argument("--dump-retrieved", type=int, default=0,
                    help="also write N sampled test predictions with the nearest pool idea per guess (retrieved.jsonl)")
    ap.add_argument("--vec-cache", default="", help="float32 vector cache (jepa/vec-cache.pt), keyed by the adapter sha256")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    enc = load("remote_encoder", "remote-encoder.py")
    if args.require_audit:
        # The data gate (audit-data.py): a passing audit of exactly this file's content, or no training.
        import hashlib

        audit = json.loads(Path(args.units + ".audit.json").read_text(encoding="utf-8"))
        sha = hashlib.sha256(Path(args.units).read_bytes()).hexdigest()
        if not audit["pass"] or audit["units_sha256"] != sha:
            raise SystemExit(f"audit gate: pass={audit['pass']} fails={audit['fails']} sha match={audit['units_sha256'] == sha}")
    units = [json.loads(l) for l in open(args.units, encoding="utf-8")]
    for u in units:
        u["ideas"] = u["ideas"][: args.max_len]
    ideas = sorted({i for u in units for i in u["ideas"]}, key=len)

    model = enc.SonarAE(16).cuda()
    model.load_state_dict(torch.load(args.adapter, map_location="cuda"), strict=False)
    model.eval()
    # Vector cache (10/04/2026): every run used to re-encode every idea (4-10 minutes). Vectors are cached in float32,
    # so a cached vector is exactly what a fresh encoding gives, keyed by the adapter's sha256 so a new encoder never
    # reuses stale vectors.
    import hashlib

    adapter_sha = hashlib.sha256(Path(args.adapter).read_bytes()).hexdigest()
    cache_path = Path(args.vec_cache) if args.vec_cache else None
    # Memory-mapped, and only this dataset's rows go to the GPU: two runs each reading the whole 4.4 GB cache ran the
    # 16 GB instance out of memory (J9, 10/04/2026).
    cache = torch.load(cache_path, mmap=True) if cache_path and cache_path.exists() else None
    if cache is not None and cache["adapter_sha256"] != adapter_sha:
        cache = None
    raw = {}
    if cache:
        pos = {t: k for k, t in enumerate(cache["texts"])}
        have = [t for t in dict.fromkeys(ideas) if t in pos]
        if have:
            # Read the rows in file order, so the disk streams them: a million scattered lookups on a cold cache held
            # each J9 run in disk wait for minutes (10/04/2026).
            # Streamed to the GPU 100k rows at a time: one CPU table plus the indexing copy of 1.6M rows ran the 15 GB
            # instance out of memory (J10, 10/05/2026).
            rows = torch.tensor([pos[t] for t in have])
            order = torch.argsort(rows)
            srt = rows[order]
            got = torch.empty(len(have), cache["raw"].shape[1], device="cuda")
            for s in range(0, len(have), 100_000):
                got[order[s : s + 100_000].cuda()] = cache["raw"][srt[s : s + 100_000]].cuda()
            raw = dict(zip(have, got))
    todo = [t for t in ideas if t not in raw]
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in enc.batched(todo, 128):
            for t, v in zip(b, model.encode(b).float()):
                raw[t] = v
    print(json.dumps({"ideas": len(ideas), "encoded_now": len(todo), "from_cache": len(ideas) - len(todo),
                      "encode_min": round((time.time() - t0) / 60, 1)}), flush=True)
    if cache_path and todo:
        # Append the new vectors to every cached one (not only this dataset's), then replace the file in one rename.
        new = list(dict.fromkeys(todo))
        old_texts, old_raw = (cache["texts"], cache["raw"]) if cache else ([], torch.empty(0, raw[new[0]].shape[0]))
        tmp = cache_path.with_suffix(".tmp")
        # One preallocated table filled in chunks from the GPU: stacking 1.2M new vectors and then concatenating made
        # two full copies and ran the 15 GB instance out of memory (J10, 10/05/2026).
        out = torch.empty(len(old_texts) + len(new), old_raw.shape[1])
        out[: len(old_texts)] = old_raw
        for s in range(0, len(new), 100_000):
            out[len(old_texts) + s : len(old_texts) + s + 100_000] = torch.stack([raw[t] for t in new[s : s + 100_000]]).cpu()
        torch.save({"adapter_sha256": adapter_sha, "texts": list(old_texts) + new, "raw": out}, tmp)
        del out
        tmp.replace(cache_path)
    # GPU-resident data (10/04/2026): one normalized table of every idea, and each unit as a row of indices, so a batch
    # is a gather, not a Python loop over ideas. Filled in chunks: stacking all 1.6M vectors, then normalizing, held
    # three full copies and ran the 22 GB GPU out of memory (J10, 10/05/2026).
    index = {t: k for k, t in enumerate(ideas)}
    V = torch.empty(len(ideas), raw[ideas[0]].shape[0], device="cuda")
    norm_sum = 0.0
    for s in range(0, len(ideas), 100_000):
        chunk = torch.stack([raw[t] for t in ideas[s : s + 100_000]])
        norm_sum += float(chunk.norm(dim=-1).sum())
        V[s : s + 100_000] = F.normalize(chunk, dim=-1)
        del chunk
    norm_scale = norm_sum / len(ideas)
    vec = {t: V[k] for t, k in index.items()}
    del raw

    # Every 10th training conversation (with its paraphrase) is kept for early stopping.
    tr_uids = sorted({u["uid"] for u in units if u["split"] == "train"})
    val_uids = set(tr_uids[::10])
    train = [u for u in units if u["split"] == "train" and u["uid"] not in val_uids]
    val = [u for u in units if u["split"] == "train" and u["uid"] in val_uids and u.get("kind", "original") == "original"]
    if any(u["split"] == "val" for u in units):
        # Conversation-level splits from dianoia.splits (jepa-build.py): whole meetings, chat days and wiki
        # conversations are held out; paraphrases are training-only.
        val = [u for u in units if u["split"] == "val" and u.get("kind") != "paraphrase"]
        train = [u for u in units if u["split"] == "train"]
    elif args.val_windows:
        # Whole held-out conversations: every 20th chained conversation, none of whose windows is trained on. (The
        # excerpt validation above leaks: other windows of the same conversations are in training; J5, J6.)
        convs = sorted({u["uid"].split("@")[0] for u in units if u.get("kind") == "window"})
        held = set(convs[::20])
        conv_of = lambda u: u["uid"].split("@")[0]
        val = [u for u in units if u.get("kind") == "window" and conv_of(u) in held]
        train = [u for u in units if u["split"] == "train" and not (u.get("kind") == "window" and conv_of(u) in held)]
    test = [u for u in units if u["split"] == "test"]

    def rows_of(us):
        """Each unit's ideas as indices into V, padded with 0 (masked), and the lengths."""
        n = max(len(u["ideas"]) for u in us)
        idx = torch.zeros(len(us), n, dtype=torch.long)
        for k, u in enumerate(us):
            idx[k, : len(u["ideas"])] = torch.tensor([index[i] for i in u["ideas"]])
        return idx.cuda(), torch.tensor([len(u["ideas"]) for u in us], device="cuda")

    train_idx, train_len = rows_of(train)
    val_idx, val_len = rows_of(val)
    test_idx, test_len = rows_of(test)

    def gather(idx, lens):
        n = int(lens.max())
        pad = torch.arange(n, device="cuda")[None, :] >= lens[:, None]
        x = V[idx[:, :n]] * (~pad).unsqueeze(-1)  # padded positions are zero, as before
        return x, pad

    def batchify(us):
        """Kept for the list-of-units callers: same tensors as gather."""
        idx, lens = rows_of(us)
        return gather(idx, lens)

    pred = Predictor(layers=args.layers, max_len=args.max_len, dropout=args.dropout).cuda()
    opt = torch.optim.AdamW(pred.parameters(), lr=args.lr, weight_decay=args.wd)

    def step_loss(x, pad):
        p = pred(x, pad)[:, :-1]
        y, keep = x[:, 1:], ~pad[:, 1:]
        p, y = p[keep], y[keep]
        # InfoNCE picks the right idea out of the batch; the cosine term keeps the prediction where the decoder reads.
        nce = F.cross_entropy(p @ y.T / 0.05, torch.arange(len(y), device="cuda"))
        cos = (p * y).sum(-1).mean()
        return nce + args.cos_weight * (1 - cos), cos

    val_pool = torch.stack([vec[i] for i in sorted({i for u in val for i in u["ideas"]})])

    def ranks_of(p, y, pool):
        """For each prediction, how many pool ideas score above the true next idea (one matrix product)."""
        true = (p * y).sum(-1, keepdim=True)
        return ((p @ pool.T) > true).sum(1)

    def val_rank() -> float:
        """Median rank of each true next idea among every validation idea, as the test is scored."""
        ranks = []
        for s in range(0, len(val), args.batch):
            x, pad = gather(val_idx[s : s + args.batch], val_len[s : s + args.batch])
            p = pred(x, pad)[:, :-1]
            keep = ~pad[:, 1:]
            ranks += ranks_of(p[keep], x[:, 1:][keep], val_pool).tolist()
        return float(st.median(ranks))

    best, best_state, curve = float("inf"), None, []
    if args.eval_only:
        # Score a trained predictor on this dataset's test split (e.g. two predictors on one pool).
        pred.load_state_dict(torch.load(args.eval_only, map_location="cuda"))
        best_state, args.steps = pred.state_dict(), 0
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05) if args.steps else None
    if args.mix_alpha != 1.0:
        # J8 (10/04/2026): WikiConv made wiki 81% of training units, and chat and meeting got worse on J7's pool.
        per = Counter(u["medium"] for u in train)
        mix_w = torch.tensor([per[u["medium"]] ** (args.mix_alpha - 1) for u in train], device="cuda")
        mix_g = torch.Generator(device="cuda").manual_seed(args.seed)
        share = {m: round(n ** args.mix_alpha / sum(k ** args.mix_alpha for k in per.values()), 3) for m, n in per.items()}
        print(json.dumps({"mix_alpha": args.mix_alpha, "units": dict(per), "batch_share": share}), flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        pred.train()
        if args.mix_alpha != 1.0:
            rows = torch.multinomial(mix_w, args.batch, replacement=False, generator=mix_g)
        else:
            # rng.sample over indices draws exactly what rng.sample(train, ...) drew (it depends only on the length).
            rows = torch.tensor(rng.sample(range(len(train)), args.batch), device="cuda")
        loss, cos = step_loss(*gather(train_idx[rows], train_len[rows]))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            pred.eval()
            with torch.no_grad():
                vs = [step_loss(*gather(val_idx[s : s + args.batch], val_len[s : s + args.batch])) for s in range(0, len(val), args.batch)]
                vr = val_rank()
            vl, vc = st.fmean(float(a) for a, _ in vs), st.fmean(float(c) for _, c in vs)
            curve.append({"step": step, "loss": round(float(loss), 4), "cos": round(float(cos), 4), "val_loss": round(vl, 4),
                          "val_cos": round(vc, 4), "val_rank": vr, "minutes": round((time.time() - t0) / 60, 1)})
            print(json.dumps(curve[-1]), flush=True)
            # Selected by the validation median rank over the whole validation pool, scored as the test is: J5's
            # in-batch validation loss kept falling while its pool rank got worse.
            if vr < best:
                best, best_state = vr, {k: v.detach().clone() for k, v in pred.state_dict().items()}
    pred.load_state_dict(best_state)

    # Evaluation on held-out units.
    pred.eval()
    pool_t = sorted({i for u in test for i in u["ideas"]})
    pool = torch.stack([vec[i] for i in pool_t])
    idx = {t: k for k, t in enumerate(pool_t)}
    res = {m: {"rank": [], "cos": []} for m in ("predictor", "last", "context_mean", "same_unit_best")}
    by_med = {}
    examples = []
    # Plausibility sample (10/05/2026): for N seeded test positions outside email, the pool idea nearest each guess,
    # skipping the unit's own context (the "last" guess would otherwise retrieve itself), plus one random pool idea as
    # the floor; graded blind by grade-plausible.py.
    dump, dump_rng = [], random.Random(args.seed + 2)
    dump_p = args.dump_retrieved / max(1, sum(len(u["ideas"]) - 1 for u in test if u["medium"] != "email"))
    with torch.no_grad():
        # Vectorized (10/04/2026): every prediction of a batch is scored against the pool in one matrix product, for
        # the predictor and both baselines; the ranks are the same as the per-prediction loop it replaced.
        ex_rng = random.Random(args.seed + 1)
        for s in range(0, len(test), 64):
            us = test[s : s + 64]
            x, pad = gather(test_idx[s : s + 64], test_len[s : s + 64])
            p = pred(x, pad)[:, :-1]
            keep = ~pad[:, 1:]
            y = x[:, 1:][keep]
            csum = x.cumsum(1)[:, :-1] / torch.arange(1, x.shape[1], device="cuda")[None, :, None]
            guesses = {"predictor": p[keep], "last": x[:, :-1][keep], "context_mean": F.normalize(csum[keep], dim=-1)}
            meds = [u["medium"] for k, u in enumerate(us) for _ in range(1, len(u["ideas"]))]
            for m, g in guesses.items():
                r = ranks_of(g, y, pool).tolist()
                res[m]["rank"] += r
                res[m]["cos"] += (g * y).sum(-1).tolist()
                for med, rk in zip(meds, r):
                    by_med.setdefault((med, m), []).append(rk)
            # Nearest other idea of the same unit (the locality ceiling).
            sim = x @ x.transpose(1, 2)
            eye = torch.eye(x.shape[1], dtype=torch.bool, device="cuda")[None]
            sim = sim.masked_fill(eye | pad[:, None, :], float("-inf"))
            res["same_unit_best"]["cos"] += sim.max(-1).values[:, 1:][keep].tolist()
            flat = [(k, t) for k, u in enumerate(us) for t in range(1, len(u["ideas"]))]
            pk = p[keep]
            if args.dump_retrieved:
                for j, (k, t) in enumerate(flat):
                    u = us[k]
                    if u["medium"] == "email" or dump_rng.random() >= dump_p:
                        continue
                    ctx = {idx[i] for i in u["ideas"][:t]}
                    row = {"uid": u["uid"], "medium": u["medium"], "context": u["ideas"][max(0, t - 4) : t],
                           "true": u["ideas"][t], "random": pool_t[dump_rng.randrange(len(pool_t))]}
                    for m, g in guesses.items():
                        top = (g[j] @ pool.T).topk(len(ctx) + 1).indices.tolist()
                        row[m] = pool_t[next(i for i in top if i not in ctx)]
                    dump.append(row)
            for j, (k, t) in enumerate(flat):
                if len(examples) < 40 and ex_rng.random() < 0.05:
                    u = us[k]
                    examples.append({"context": u["ideas"][max(0, t - 3) : t], "true": u["ideas"][t], "vec": pk[j]})
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # The decoder reads a raw vector: the prediction is a direction, scaled to the encoder's mean norm.
            dec = model.generate(torch.stack([e["vec"] for e in examples]) * norm_scale) if examples else []
    for e, d in zip(examples, dec):
        e["predicted"] = d
        del e["vec"]

    def summ(r):
        o = {"cos": round(st.fmean(r["cos"]), 4)}
        if r["rank"]:
            o.update({"top1": round(st.fmean(x == 0 for x in r["rank"]), 4), "top10": round(st.fmean(x < 10 for x in r["rank"]), 4),
                      "median_rank": st.median(r["rank"]), "n": len(r["rank"])})
        return o

    summary = {"pool": len(pool_t), "train_units": len(train), "val_units": len(val), "test_units": len(test),
               "best_val_rank": best, "curve": curve,
               **{m: summ(r) for m, r in res.items()},
               "by_medium": {f"{med}/{m}": {"top10": round(st.fmean(x < 10 for x in r), 4), "median_rank": st.median(r)}
                             for (med, m), r in sorted(by_med.items())},
               "examples": examples}
    if args.dump_retrieved:
        with open(out / "retrieved.jsonl", "w", encoding="utf-8") as f:
            for row in dump:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "jepa.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    torch.save(pred.state_dict(), out / "predictor.pt")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("examples", "by_medium")}), flush=True)


if __name__ == "__main__":
    main()
