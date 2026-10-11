"""T1: a text model writes the next idea in words; scored exactly as the JEPA predictor is (10/05/2026).

The bar for "predicting in idea space is good" (phase 8 README section 6) includes matching a text model that predicts
the next idea in words. Qwen3-0.6B-Base (about 33x the 2-layer predictor's ~18M parameters) is fine-tuned on J10's
training units, each written one idea per line, so like the predictor it learns every next-idea position of a
conversation at once, with the same source mix (alpha 0.5). On J7's test units it writes the next line (greedy, vLLM)
at every position; the line is encoded with the same S1 encoder and ranked against J7's pool exactly as
remote-jepa.py ranks the predictor. For the blind grading, the rows of a remote-jepa.py ``--dump-retrieved`` file get
two candidates more: ``text_gen`` (the generated line itself) and ``text`` (the pool idea nearest its encoding, as
the predictor's candidate is chosen). Generation runs in the vLLM environment, scoring in the SONAR one; the pool is
encoded fresh (about 40,000 ideas), so no vector cache is needed.

    python remote-textnext.py train --units jepa/units-j10.jsonl --out jepa/T1/model
    python remote-textnext.py generate --units jepa/units-j7.jsonl --model jepa/T1/model --out jepa/T1
    python remote-textnext.py score --units jepa/units-j7.jsonl --adapter ablation/S1/lora.pt \
        --retrieved jepa/J10-P/retrieved.jsonl --out jepa/T1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics as st
import time
from collections import Counter
from pathlib import Path

BASE = "Qwen/Qwen3-0.6B-Base"
MAX_IDEAS = 32  # remote-jepa.py --max-len


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean(t: str) -> str:
    return " ".join(t.split())  # one idea per line: no idea may hold a newline


def train(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    units = [json.loads(l) for l in open(args.units, encoding="utf-8")]
    train_u = [[clean(i) for i in u["ideas"][:MAX_IDEAS]] for u in units if u["split"] == "train" and len(u["ideas"]) > 1]
    media = [u["medium"] for u in units if u["split"] == "train" and len(u["ideas"]) > 1]
    per = Counter(media)
    w = [per[m] ** (args.mix_alpha - 1) for m in media]  # remote-jepa.py --mix-alpha
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = random.Random(0)
    nl = tok("\n", add_special_tokens=False).input_ids
    print(json.dumps({"units": len(train_u), "by_medium": dict(per), "steps": args.steps, "batch": args.batch}), flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        b = rng.choices(train_u, weights=w, k=args.batch)
        for s0 in range(0, len(b), args.micro):
            ids, labs = [], []
            for ideas in b[s0 : s0 + args.micro]:
                x, y = [], []
                for k, idea in enumerate(ideas):
                    t = tok(idea, add_special_tokens=False).input_ids + nl
                    x += t
                    # The first idea has no context: it is given, never predicted (as in the predictor).
                    y += [-100] * len(t) if k == 0 else t
                ids.append(x[: args.max_tokens])
                labs.append(y[: args.max_tokens])
            n = max(map(len, ids))
            x = torch.tensor([i + [tok.pad_token_id] * (n - len(i)) for i in ids], device="cuda")
            m = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i in ids], device="cuda")
            y = torch.tensor([l + [-100] * (n - len(l)) for l in labs], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, attention_mask=m, labels=y).loss
            (loss * len(ids) / len(b)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": round(float(loss), 4), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    model.to(torch.bfloat16).save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(json.dumps({"trained_minutes": round((time.time() - t0) / 60, 1)}), flush=True)


def test_positions(units_path):
    units = [json.loads(l) for l in open(units_path, encoding="utf-8")]
    test = [u for u in units if u["split"] == "test" and u.get("kind") != "paraphrase"]
    for u in test:
        u["ideas"] = u["ideas"][:MAX_IDEAS]
    return test, [(k, t) for k, u in enumerate(test) for t in range(1, len(u["ideas"]))]


def generate(args) -> None:
    """vLLM (its own environment): the next line at every test position, greedy."""
    from vllm import LLM, SamplingParams

    test, pos = test_positions(args.units)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
    prompts = ["".join(clean(i) + "\n" for i in test[k]["ideas"][:t]) for k, t in pos]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=96, stop=["\n"]), use_tqdm=False)
    minutes = (time.time() - t0) / 60
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "generated.jsonl", "w", encoding="utf-8") as f:
        for (k, t), o in zip(pos, outs):
            f.write(json.dumps({"k": k, "t": t, "text": clean(o.outputs[0].text), "tokens": len(o.outputs[0].token_ids)},
                               ensure_ascii=False) + "\n")
    print(json.dumps({"positions": len(pos), "gen_minutes": round(minutes, 2)}), flush=True)


def score(args) -> None:
    """S1-encode the generated lines and J7's pool, rank as remote-jepa.py does, and add the plausibility candidates."""
    import torch
    import torch.nn.functional as F

    test, pos = test_positions(args.units)
    gen_rows = [json.loads(l) for l in open(Path(args.out) / "generated.jsonl", encoding="utf-8")]
    assert [(g["k"], g["t"]) for g in gen_rows] == pos, "generated.jsonl does not match these units"
    gen = [g["text"] or "." for g in gen_rows]
    enc = load("remote_encoder", "remote-encoder.py")
    model = enc.SonarAE(16).cuda()
    model.load_state_dict(torch.load(args.adapter, map_location="cuda"), strict=False)
    model.eval()

    def encode(texts):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        flat = [None] * len(texts)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(order), 128):
                ix = order[s : s + 128]
                for i, v in zip(ix, model.encode([texts[i] for i in ix]).float()):
                    flat[i] = v
        return F.normalize(torch.stack(flat), dim=-1)

    pool_t = sorted({i for u in test for i in u["ideas"]})
    P, G = encode(pool_t), encode(gen)
    idx = {t: k for k, t in enumerate(pool_t)}
    Y = P[torch.tensor([idx[test[k]["ideas"][t]] for k, t in pos], device="cuda")]
    ranks = []
    for s in range(0, len(pos), 1000):
        g, y = G[s : s + 1000], Y[s : s + 1000]
        ranks += ((g @ P.T) > (g * y).sum(-1, keepdim=True)).sum(1).tolist()
    by = {}
    for (k, _), r in zip(pos, ranks):
        by.setdefault(test[k]["medium"], []).append(r)
    summ = lambda r: {"top1": round(st.fmean(x == 0 for x in r), 4), "top10": round(st.fmean(x < 10 for x in r), 4),
                      "median_rank": st.median(r), "n": len(r)}
    out = Path(args.out)
    res = {"pool": len(pool_t), "positions": len(pos), "text": summ(ranks), "by_medium": {m: summ(r) for m, r in sorted(by.items())},
           "tokens_per_idea": round(st.fmean(g["tokens"] for g in gen_rows), 1),
           "examples": [{"context": test[k]["ideas"][max(0, t - 3) : t], "true": test[k]["ideas"][t], "generated": gen[j]}
                        for j, (k, t) in enumerate(pos) if j % max(1, len(pos) // 40) == 0]}
    (out / "textnext.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}), flush=True)

    if args.retrieved:
        # The rows of a --dump-retrieved file, located by uid, true idea and the 4 ideas before it.
        at = {}
        for j, (k, t) in enumerate(pos):
            ideas = test[k]["ideas"]
            at[(test[k]["uid"], ideas[t], tuple(ideas[max(0, t - 4) : t]))] = (j, k, t)
        kept = 0
        with open(out / "retrieved.jsonl", "w", encoding="utf-8") as f:
            for r in map(json.loads, open(args.retrieved, encoding="utf-8")):
                hit = at.get((r["uid"], r["true"], tuple(r["context"])))
                if hit is None:
                    continue
                j, k, t = hit
                ctx = {idx[i] for i in test[k]["ideas"][:t]}
                top = (G[j] @ P.T).topk(len(ctx) + 1).indices.tolist()
                r["text_gen"] = gen[j]
                r["text"] = pool_t[next(i for i in top if i not in ctx)]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
        print(json.dumps({"retrieved_rows": kept}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--units", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--micro", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--max-tokens", type=int, default=1024)
    t.add_argument("--mix-alpha", type=float, default=0.5)
    g = sub.add_parser("generate")
    g.add_argument("--units", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--units", required=True)
    s.add_argument("--adapter", required=True)
    s.add_argument("--retrieved", default="")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"train": train, "generate": generate, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
