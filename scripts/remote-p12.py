"""Phase 12 GPU stage: contrastive LoRA training of the search encoder on far (and, as the control, near) patent
borrowings, then ranking phase 10's candidates (results/phase-12-far-analogy/HYPOTHESES.md, 10/07/2026).

    python remote-p12.py --work p12 --config p12/config.used.yaml --out p12/out [--smoke]

Inputs in --work (scripts/p12-data.py build, audited): train-far.jsonl, train-near.jsonl, train-docs.jsonl,
eval.jsonl, docs.jsonl, audit.json. Every stage is saved under --out and skipped when its file exists, so a restart
resumes where it stopped:
  scores/base.json           the encoder off the shelf
  adapters/<run>-e<k>/       LoRA adapter after epoch k; runs far-s0 and near-s0 (3 epochs each), then far-s1 and
                             far-s2 for the epoch count dev chose for far-s0
  scores/<run>-e<k>.json     per split (dev, test, test_near), top-10 hit and reciprocal rank per query
Writes p12.json. No verdicts here: scripts/p12-verdicts.py judges from the per-query arrays.

Loss: InfoNCE over the batch. Each citing abstract (with phase 10's query instruction) is pulled toward its cited
abstract and away from every other document in the batch: its own 7 hard negatives, the other pairs' cited abstracts
and their hard negatives. A document that is another query's positive with the same patent id is masked, never a
negative. The batch of 32 pairs (288 documents) is exact: embeddings are first computed without gradients, the loss
gradient with respect to them is taken, then each chunk is re-run with gradients and given its share (gradient
caching, Gao et al. 2021), with each chunk's random state replayed so dropout matches between the two passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

DTYPE = {"cuda": torch.bfloat16, "cpu": torch.float32}
SPLITS = ("dev", "test", "test_near")


def log(**kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **kw}), flush=True)


def require_audit(work):
    """Training reads exactly the files the audit passed (dianoia data procedure, 10/04/2026)."""
    au = json.loads((work / "audit.json").read_text())
    assert au["pass"], f"audit failed: {au['violations']}"
    for f, h in au["sha256"].items():
        got = hashlib.sha256((work / f).read_bytes()).hexdigest()
        assert got == h, f"{f} changed since the audit"


def embed(model, tok, texts, dev, max_len, grad=False):
    """Last-token pooling with left padding (Qwen3-Embedding's own, as phase 10's search), L2-normalized, float32."""
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
    with torch.set_grad_enabled(grad):
        h = model(**enc).last_hidden_state[:, -1]
    return F.normalize(h.float(), dim=-1)


def encode_all(model, tok, texts, dev, bs=64, max_len=512):
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vec = [None] * len(texts)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        for i, v in zip(idx, embed(model, tok, [texts[i] for i in idx], dev, max_len).half().cpu()):
            vec[i] = v
    return torch.stack(vec)


# ---- training ----------------------------------------------------------------------------------------------------

def _rng_state(dev):
    return (torch.get_rng_state(), torch.cuda.get_rng_state() if dev == "cuda" else None)


def _set_rng(st, dev):
    torch.set_rng_state(st[0])
    if dev == "cuda":
        torch.cuda.set_rng_state(st[1])


def cached_step(model, tok, qs, ds, pos, mask, cfg, dev):
    """One exact large-batch InfoNCE step by gradient caching; returns the loss."""
    t = cfg["train"]
    n = t["chunk"]

    def chunks(texts):
        return [texts[i:i + n] for i in range(0, len(texts), n)]

    states, embs = [], []
    for part in (qs, ds):
        e, s = [], []
        for c in chunks(part):
            s.append(_rng_state(dev))
            e.append(embed(model, tok, c, dev, t["max_tokens"]))
        states.append(s)
        embs.append(torch.cat(e))
    q, d = (x.detach().requires_grad_() for x in embs)
    logits = (q @ d.T) / t["temperature"]
    logits = logits.masked_fill(mask, float("-inf"))
    loss = F.cross_entropy(logits, pos)
    loss.backward()
    for part, grads, st in ((qs, q.grad, states[0]), (ds, d.grad, states[1])):
        for k, c in enumerate(chunks(part)):
            _set_rng(st[k], dev)
            e = embed(model, tok, c, dev, t["max_tokens"], grad=True)
            e.backward(grads[k * n:k * n + len(c)])
    return float(loss)


def train(base, tok, pairs, ndocs, cfg, dev, out, run, epochs, seed, smoke_steps=0):
    """One LoRA run; adapter saved to out/adapters/<run>-e<k> after epoch k. Yields (k, model) after each epoch."""
    from peft import LoraConfig, get_peft_model
    t = cfg["train"]
    torch.manual_seed(seed)
    model = get_peft_model(base, LoraConfig(r=t["lora_r"], lora_alpha=t["lora_alpha"], lora_dropout=t["lora_dropout"],
                                            target_modules="all-linear", task_type="FEATURE_EXTRACTION"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights and AdamW state in float32 over the bfloat16 base
    params = [p for p in model.parameters() if p.requires_grad]
    per_epoch = smoke_steps or max(1, math.ceil(len(pairs) / t["batch"]))
    total = per_epoch * epochs
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=0.0)
    warm = max(1, int(total * t["warmup_ratio"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    rng = random.Random(seed)
    k = cfg["negatives"]
    step, t0, losses = 0, time.time(), []
    for e in range(1, epochs + 1):
        model.train()
        order = list(range(len(pairs)))
        rng.shuffle(order)
        batches = [order[i:i + t["batch"]] for i in range(0, len(order), t["batch"])][:per_epoch]
        for b in batches:
            rows = [pairs[i] for i in b]
            qs = [cfg["instruct"] + r["citing_text"] for r in rows]
            ids, ds = [], []
            for r in rows:
                ids += [r["cited"], *r["negatives"]]
                ds += [r["cited_text"], *[ndocs[x] for x in r["negatives"]]]
            pos = torch.arange(len(rows), device=dev) * (k + 1)
            idt = torch.tensor(ids, device=dev)
            cited = torch.tensor([r["cited"] for r in rows], device=dev)
            # Another pair's copy of this query's own cited patent is not a negative.
            mask = (idt[None, :] == cited[:, None])
            mask[torch.arange(len(rows), device=dev), pos] = False
            losses.append(cached_step(model, tok, qs, ds, pos, mask, cfg, dev))
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or smoke_steps:
                log(stage="train", run=run, epoch=e, step=step, of=total, loss=round(sum(losses[-20:]) / len(losses[-20:]), 4),
                    minutes=round((time.time() - t0) / 60, 1))
        model.save_pretrained(out / "adapters" / f"{run}-e{e}")
        (out / "adapters" / f"{run}-e{e}" / "losses.json").write_text(json.dumps(losses))
        log(stage="saved", adapter=f"{run}-e{e}", pairs=len(pairs))
        yield e, model


# ---- scoring -----------------------------------------------------------------------------------------------------

def score(ev, qv, dv, k):
    m = {}
    for s in SPLITS:
        hits, rr = [], []
        for i, r in enumerate(ev):
            if r["split"] != s:
                continue
            q = qv[i].float()
            true = float(q @ dv[r["cited"]].float())
            c = torch.stack([dv[x] for x in r["candidates"]]).float() @ q
            rank = 1 + int((c >= true).sum())
            hits.append(float(rank <= k))
            rr.append(1.0 / rank)
        m[s] = {"hits": hits, "rr": rr, "top10": round(sum(hits) / max(1, len(hits)), 4), "mrr": round(sum(rr) / max(1, len(rr)), 4)}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    work, out = Path(a.work), Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    require_audit(work)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sm = cfg["smoke"] if a.smoke else None
    name = sm["embedder"] if a.smoke and dev == "cpu" else cfg["embedder"]
    rd = lambda f: [json.loads(l) for l in open(work / f, encoding="utf-8")]
    data = {"far": rd("train-far.jsonl"), "near": rd("train-near.jsonl")}
    ndocs = {r["id"]: r["text"] for r in rd("train-docs.jsonl")}
    ev = rd("eval.jsonl")
    if a.smoke:
        if dev == "cpu":  # code paths only, light on the laptop
            cfg["train"].update(batch=2, chunk=4, max_tokens=64)
        data = {x: v[:sm["train"]] for x, v in data.items()}
        ev = [r for s in SPLITS for r in [x for x in ev if x["split"] == s][:sm["queries"]]]
        for r in ev:
            r["candidates"] = r["candidates"][:sm["candidates"]]
    need = {c for r in ev for c in [r["cited"], *r["candidates"]]}
    docs = {r["id"]: r["text"] for r in rd("docs.jsonl") if r["id"] in need}
    eval_len = 64 if a.smoke and dev == "cpu" else 512
    log(stage="start", device=dev, model=name, far=len(data["far"]), near=len(data["near"]), eval=len(ev), docs=len(docs), smoke=a.smoke)
    k = cfg["recovery"]["top_k"]

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    fresh = lambda: AutoModel.from_pretrained(name, torch_dtype=DTYPE[dev]).to(dev)
    doc_ids = sorted(docs)

    def scored(tag, model):
        f = out / "scores" / f"{tag}.json"
        if not f.exists():
            t0 = time.time()
            dv = dict(zip(doc_ids, encode_all(model, tok, [docs[i] for i in doc_ids], dev, max_len=eval_len)))
            qv = encode_all(model, tok, [cfg["instruct"] + r["query_text"] for r in ev], dev, max_len=eval_len)
            f.write_text(json.dumps(score(ev, qv, dv, k)))
            m = json.loads(f.read_text())
            log(stage="scored", model=tag, minutes=round((time.time() - t0) / 60, 1), **{s: m[s]["top10"] for s in SPLITS})
        return json.loads(f.read_text())

    def run(tag, which, epochs, seed, every=True):
        """Trains (or reloads) one run and scores every epoch (only the last when every is False: the extra seeds
        need no epoch choice); skips what a restart already has."""
        tags = [f"{tag}-e{e}" for e in range(1 if every else epochs, epochs + 1)]
        if all((out / "scores" / f"{x}.json").exists() for x in tags):
            return
        from peft import PeftModel
        if all((out / "adapters" / x).exists() for x in tags):
            for x in tags:
                scored(x, PeftModel.from_pretrained(fresh(), out / "adapters" / x).to(dev))
            return
        for e, m in train(fresh(), tok, data[which], ndocs, cfg, dev, out, tag, epochs, seed, sm["steps"] if sm else 0):
            if f"{tag}-e{e}" in tags:
                scored(f"{tag}-e{e}", m)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    base = fresh()
    scored("base", base)
    del base
    seeds, E = cfg["train"]["seeds"], max(cfg["train"]["epochs"])
    run(f"far-s{seeds[0]}", "far", E, seeds[0])
    run(f"near-s{seeds[0]}", "near", E, seeds[0])

    def choose(tag):  # dev far top-10, then MRR (OPERATIONALIZATION.md)
        ms = {e: json.loads((out / "scores" / f"{tag}-e{e}.json").read_text())["dev"] for e in cfg["train"]["epochs"]}
        return max(cfg["train"]["epochs"], key=lambda e: (ms[e]["top10"], ms[e]["mrr"]))

    ef, en = choose(f"far-s{seeds[0]}"), choose(f"near-s{seeds[0]}")
    log(stage="chosen", far=ef, near=en)
    for s in seeds[1:]:
        run(f"far-s{s}", "far", ef, s, every=False)

    load = lambda tag: json.loads((out / "scores" / f"{tag}.json").read_text())
    named = {"base": "base", "far": f"far-s{seeds[0]}-e{ef}", "near": f"near-s{seeds[0]}-e{en}",
             **{f"far_s{s}": f"far-s{s}-e{ef}" for s in seeds[1:]}}
    metrics = {"named": named, "chosen_epochs": {"far": ef, "near": en},
               "models": {p.stem: load(p.stem) for p in sorted((out / "scores").glob("*.json"))},
               "queries": {s: [r["query"] for r in ev if r["split"] == s] for s in SPLITS},
               "overlap": {s: [r["overlap"] for r in ev if r["split"] == s] for s in SPLITS},
               "cosine": {s: [r["cosine"] for r in ev if r["split"] == s] for s in SPLITS},
               "losses": {t: json.loads((out / "adapters" / t / "losses.json").read_text())
                          for t in named.values() if (out / "adapters" / t / "losses.json").exists()}}
    (out / "p12.json").write_text(json.dumps(metrics))
    log(stage="done", models=list(metrics["models"]))


if __name__ == "__main__":
    main()
