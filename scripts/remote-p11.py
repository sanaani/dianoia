"""Phase 11 GPU stage: train Qwen3 LoRA on analogy pairs, write source descriptions, rank candidates (10/07/2026).

    python remote-p11.py --work p11 --config p11/config.used.yaml --out p11/out [--smoke]

Inputs in --work (scripts/p11-data.py build, audited): train.jsonl, eval.jsonl, docs.jsonl, audit.json.
Stages, each saved under --out and skipped when its file exists (a restart resumes at the stage it lost):
  emb_docs.pt          Qwen3-Embedding vectors of every cited patent and candidate (no instruction, as in phase 10)
  gen/<model>.jsonl    for every val and test query, the description the model writes from the citing abstract
                       (greedy, cut at the first blank line). Models: untrained; k3000-e1 and k3000-e2 (one run,
                       saved after each epoch); then k1000 and kall at the epoch count val chose at 3k.
  adapters/<model>/    each trained LoRA adapter
Writes p11.json: per model and split, top-10 hit and reciprocal rank per query, in eval.jsonl's order, plus the raw
abstract baseline. No verdicts here: scripts/p11-verdicts.py judges from the per-query arrays.
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


def log(**kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **kw}), flush=True)


def require_audit(work):
    """Training reads exactly the files the audit passed (dianoia data procedure, 10/04/2026)."""
    au = json.loads((work / "audit.json").read_text())
    assert au["pass"], f"audit failed: {au['violations']}"
    for f, h in au["sha256"].items():
        got = hashlib.sha256((work / f).read_bytes()).hexdigest()
        assert got == h, f"{f} changed since the audit"


# ---- training ----------------------------------------------------------------------------------------------------

def encode_pairs(tok, pairs, cfg):
    t = cfg["train"]
    out = []
    for r in pairs:
        a = tok(r["citing_text"], add_special_tokens=False)["input_ids"][:t["max_prompt_tokens"]]
        prompt = tok(cfg["prompt"].format(abstract=tok.decode(a)), add_special_tokens=False)["input_ids"]
        target = tok(r["cited_text"], add_special_tokens=False)["input_ids"][:t["max_target_tokens"]] + [tok.eos_token_id]
        out.append((prompt + target, [-100] * len(prompt) + target))
    return out


def train(base, tok, pairs, cfg, dev, out, names, smoke_steps=0):
    """One LoRA run of len(names) epochs; adapter saved to out/adapters/<names[e]> after epoch e+1."""
    from peft import LoraConfig, get_peft_model
    t = cfg["train"]
    model = get_peft_model(base, LoraConfig(r=t["lora_r"], lora_alpha=t["lora_alpha"], lora_dropout=t["lora_dropout"],
                                            target_modules="all-linear", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights and AdamW state in float32 over the bfloat16 base
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    data = encode_pairs(tok, pairs, cfg)
    acc = t["batch"] // t["micro_batch"]
    steps_per_epoch = max(1, math.ceil(len(data) / t["batch"]))
    total = smoke_steps * len(names) if smoke_steps else steps_per_epoch * len(names)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=t["lr"], weight_decay=0.0)
    warm = max(1, int(total * t["warmup_ratio"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    rng = random.Random(cfg["seed"])
    step, t0, losses = 0, time.time(), []
    model.train()
    for e, name in enumerate(names):
        order = list(range(len(data)))
        rng.shuffle(order)
        batches = [order[i:i + t["batch"]] for i in range(0, len(order), t["batch"])]
        for bi, b in enumerate(batches[:smoke_steps] if smoke_steps else batches):
            for mi in range(0, len(b), t["micro_batch"]):
                mb = [data[j] for j in b[mi:mi + t["micro_batch"]]]
                L = max(len(x) for x, _ in mb)
                ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in mb], device=dev)
                lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in mb], device=dev)
                att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in mb], device=dev)
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                (loss * len(mb) / len(b)).backward()
                losses.append(float(loss))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or smoke_steps:
                log(stage="train", run=names[-1], epoch=e + 1, step=step, of=total, loss=round(sum(losses[-50:]) / len(losses[-50:]), 4),
                    minutes=round((time.time() - t0) / 60, 1))
        model.save_pretrained(out / "adapters" / name)
        log(stage="saved", adapter=name, pairs=len(pairs), epoch=e + 1)
    return model


# ---- generation and retrieval -------------------------------------------------------------------------------------

def generate(model, tok, ev, cfg, dev, path, max_new):
    if path.exists():
        return
    g = cfg["generate"]
    tok.padding_side = "left"
    model.eval()
    if getattr(model, "is_gradient_checkpointing", False):  # checkpointing turns the KV cache off: slow generation
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    rows, t0 = [], time.time()
    prompts = []
    for r in ev:
        a = tok(r["query_text"], add_special_tokens=False)["input_ids"][:cfg["train"]["max_prompt_tokens"]]
        prompts.append(cfg["prompt"].format(abstract=tok.decode(a)))
    order = sorted(range(len(ev)), key=lambda i: len(prompts[i]))
    out = {}
    for b in range(0, len(order), g["batch"]):
        idx = order[b:b + g["batch"]]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, seq in zip(idx, o[:, enc["input_ids"].shape[1]:]):
            full = tok.decode(seq, skip_special_tokens=True)
            out[i] = full.strip().split("\n\n")[0].strip() or full.strip() or "none"
    tok.padding_side = "right"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(ev):
            f.write(json.dumps({"split": r["split"], "query": r["query"], "text": out[i]}) + "\n")
    log(stage="generated", file=path.name, n=len(ev), minutes=round((time.time() - t0) / 60, 1))


class Embedder:
    def __init__(self, name, dev, bs):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.m = AutoModel.from_pretrained(name, torch_dtype=DTYPE[dev]).to(dev).eval()
        self.dev, self.bs = dev, bs

    def __call__(self, texts):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vec = [None] * len(texts)
        for b in range(0, len(order), self.bs):
            idx = order[b:b + self.bs]
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
            with torch.no_grad():
                h = self.m(**enc).last_hidden_state[:, -1]
            for i, v in zip(idx, F.normalize(h.float(), dim=-1).half().cpu()):
                vec[i] = v
        return torch.stack(vec)


def recovery(ev, qv, dv, k):
    hits, rr = [], []
    for r, q in zip(ev, qv):
        q = q.float()
        true = float(q @ dv[r["cited"]].float())
        c = torch.stack([dv[x] for x in r["candidates"]]).float() @ q
        rank = 1 + int((c >= true).sum())
        hits.append(float(rank <= k))
        rr.append(1.0 / rank)
    return hits, rr


def score(ev, qv, dv, k):
    m = {}
    for s in ("val", "test"):
        idx = [i for i, r in enumerate(ev) if r["split"] == s]
        h, rr = recovery([ev[i] for i in idx], qv[idx], dv, k)
        m[s] = {"hits": h, "rr": rr, "top10": round(sum(h) / max(1, len(h)), 4), "mrr": round(sum(rr) / max(1, len(rr)), 4)}
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
    out.mkdir(parents=True, exist_ok=True)
    require_audit(work)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sm = cfg["smoke"] if a.smoke else None
    # The laptop dry run uses the small model; on the GPU the smoke runs the real one, so memory problems show early.
    name = sm["model"] if a.smoke and dev == "cpu" else cfg["model"]
    if a.smoke and dev == "cpu":
        cfg["embedder"] = sm["embedder"]
    max_new = 16 if a.smoke else cfg["generate"]["max_new_tokens"]
    pairs = [json.loads(l) for l in open(work / "train.jsonl", encoding="utf-8")]
    ev = [json.loads(l) for l in open(work / "eval.jsonl", encoding="utf-8")]
    if a.smoke:
        # Lighter on the laptop: full-vocabulary logits for a batch of long prompts took the CPU run past 3.8 GB
        # (10/07/2026). Code paths only, so small batches and short texts.
        if dev == "cpu":
            cfg["generate"]["batch"] = 2
            cfg["train"].update(micro_batch=1, batch=2, max_prompt_tokens=64, max_target_tokens=64)
        pairs = pairs[:sm["train"]]
        ev = [r for s in ("val", "test") for r in [x for x in ev if x["split"] == s][:sm["queries"]]]
        for r in ev:
            r["candidates"] = r["candidates"][:sm["candidates"]]
    need = {c for r in ev for c in [r["cited"], *r["candidates"]]}
    docs = {r["id"]: r["text"] for r in map(json.loads, open(work / "docs.jsonl", encoding="utf-8")) if r["id"] in need}
    sizes = [len(pairs) if s == "all" else s for s in cfg["sizes"]]
    if a.smoke:
        sizes = [2, 4, len(pairs)]
    log(stage="start", device=dev, model=name, train=len(pairs), eval=len(ev), docs=len(docs), sizes=sizes, smoke=a.smoke)
    k = cfg["recovery"]["top_k"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def fresh():
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=DTYPE[dev]).to(dev)

    def gen_with(model_name, model):
        generate(model, tok, ev, cfg, dev, out / "gen" / f"{model_name}.jsonl", max_new)

    def run(names, n, gen):
        """Train on the first n pairs for len(names) epochs (adapter saved after each), then write descriptions with
        the adapters named in gen. Skipped when those descriptions exist; reuses saved adapters after a restart."""
        if all((out / "gen" / f"{x}.jsonl").exists() for x in gen):
            return
        from peft import PeftModel
        if all((out / "adapters" / x).exists() for x in gen):
            for x in gen:
                gen_with(x, PeftModel.from_pretrained(fresh(), out / "adapters" / x).to(dev))
            return
        m = train(fresh(), tok, pairs[:n], cfg, dev, out, names, sm["steps"] if sm else 0)
        if names[-1] in gen:
            gen_with(names[-1], m)
        del m
        for x in gen:
            if x != names[-1]:  # earlier epochs: reload their saved adapters
                gen_with(x, PeftModel.from_pretrained(fresh(), out / "adapters" / x).to(dev))

    gen_with("untrained", fresh().eval())
    epochs = cfg["train"]["epochs"]
    mid = sizes[1]
    run([f"k{mid}-e{e}" for e in range(1, max(epochs) + 1)], mid, [f"k{mid}-e{e}" for e in epochs])

    # Retrieval: documents once, then each set of queries.
    emb = Embedder(cfg["embedder"], dev, 64)
    f = out / "emb_docs.pt"
    if f.exists():
        dv = torch.load(f)
    else:
        ids = sorted(docs)
        dv = dict(zip(ids, emb([docs[i] for i in ids])))
        torch.save(dv, f)
    metrics = {"models": {}, "queries": {s: [r["query"] for r in ev if r["split"] == s] for s in ("val", "test")},
               "overlap": {s: [r["overlap"] for r in ev if r["split"] == s] for s in ("val", "test")}}
    metrics["models"]["abstract"] = score(ev, emb([cfg["instruct_abstract"] + r["query_text"] for r in ev]), dv, k)

    def score_gen(x):
        g = [json.loads(l) for l in open(out / "gen" / f"{x}.jsonl", encoding="utf-8")]
        assert [r["query"] for r in g] == [r["query"] for r in ev]
        metrics["models"][x] = score(ev, emb([cfg["instruct_description"] + r["text"] for r in g]), dv, k)
        log(stage="scored", model=x, val=metrics["models"][x]["val"]["top10"], test=metrics["models"][x]["test"]["top10"])

    for x in ["untrained"] + [f"k{mid}-e{e}" for e in epochs]:
        score_gen(x)
    # Epoch count chosen on val at the middle size: top-10, then MRR (OPERATIONALIZATION.md).
    best = max(epochs, key=lambda e: (metrics["models"][f"k{mid}-e{e}"]["val"]["top10"], metrics["models"][f"k{mid}-e{e}"]["val"]["mrr"]))
    metrics["chosen_epochs"] = best
    log(stage="chosen", epochs=best)
    del emb
    torch.cuda.empty_cache() if dev == "cuda" else None
    for n in (sizes[0], sizes[2]):
        run([f"k{n}-e{e}" for e in range(1, best + 1)], n, [f"k{n}-e{best}"])
    emb = Embedder(cfg["embedder"], dev, 64)
    for n in (sizes[0], sizes[2]):
        score_gen(f"k{n}-e{best}")
    metrics["named"] = {"untrained": "untrained", "abstract": "abstract", "k_small": f"k{sizes[0]}-e{best}",
                        "k_mid": f"k{mid}-e{best}", "k_all": f"k{sizes[2]}-e{best}", "sizes": sizes}
    metrics["samples"] = {x: [json.loads(l) for l in open(out / "gen" / f"{x}.jsonl", encoding="utf-8")][:5]
                          for x in ("untrained", metrics["named"]["k_all"])}
    (out / "p11.json").write_text(json.dumps(metrics))
    log(stage="done", models=list(metrics["models"]))


if __name__ == "__main__":
    main()
