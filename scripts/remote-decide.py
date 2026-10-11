"""D1: the production yardstick. Fastino's GLiNER2.5-Decide, fine-tuned on the same outcomes, against idea space (10/05/2026).

Shaun (10/05/2026): judge idea space against a model we would actually ship, on held-out outcomes, never by an LLM
grader. GLiNER2.5-Decide (Apache-2.0, a 340M-parameter DeBERTa-v3-large encoder, runs locally) takes text and a label
set and returns a label with a confidence. It is fine-tuned here on the training side of a pair file
(remote-cmv.py's format: every pair's ``win`` and ``lose`` text becomes one example) and scored on the same held-out
pairs as the idea model: the side it is more confident is the good one wins the pair.

Reported, on the held-out pairs: zero-shot (before fine-tuning) and fine-tuned accuracy; the paired difference with the
idea model (its held-out scores from remote-cmv.py's features.jsonl); and Decide plus idea scores, a logistic model on
the pair differences (Decide, idea, surprise, relevance, length) scored by 5-fold cross-fitting over the held-out
pairs, so every pair is scored by a model that never saw it.

    python remote-decide.py --pairs gh/g1/pairs.jsonl --texts gh/g1/texts.jsonl --features gh/G1/features.jsonl \
        --labels "clean,reverted" --out gh/D1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = "fastino/GLiNER2.5-Decide"
WORDS = 300  # DeBERTa reads 512 tokens; longer texts are cut to their first 300 words


def boot(xs, reps=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor(xs, dtype=torch.float64)
    m = x[torch.randint(len(x), (reps, len(x)), generator=g)].mean(1)
    return [round(float(x.mean()), 4), round(float(m.quantile(0.025)), 4), round(float(m.quantile(0.975)), 4)]


def logit_fit(X):
    Xb, y = torch.cat([X, -X]), torch.cat([torch.ones(len(X)), torch.zeros(len(X))]).double()
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xb @ w, y) + 1e-3 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--features", required=True, help="remote-cmv.py features.jsonl (idea scores on held-out pairs)")
    ap.add_argument("--labels", required=True, help="good,bad label names, e.g. clean,reverted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    # A dry run on a few pairs before the real one (10/05/2026: G1 crashed twice on paths no small run had exercised).
    ap.add_argument("--smoke", type=int, default=0, help="use only this many pairs of each split, one epoch")
    args = ap.parse_args()
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    good, bad = args.labels.split(",")
    task = "outcome"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = {r["id"]: " ".join(r["text"].split()[:WORDS]) for r in map(json.loads, open(args.texts, encoding="utf-8"))}
    pairs = [json.loads(l) for l in open(args.pairs)]
    feats = {f["pair"]: f for f in map(json.loads, open(args.features, encoding="utf-8"))}
    train = [p for p in pairs if p["split"] == "train"]
    test = [p for p in pairs if p["split"] == "test" and p["pair"] in feats and feats[p["pair"]]["idea_score"][0] is not None]
    # Validation as remote-cmv.py draws it: 10% of training posts (for GitHub, of pairs).
    ops = sorted({p["op"] for p in train})
    val_ops = set(random.Random(0).sample(ops, len(ops) // 10))
    fit = [p for p in train if p["op"] not in val_ops]
    val = [p for p in train if p["op"] in val_ops]
    if args.smoke:
        fit, val, test, args.epochs = fit[: args.smoke], val[: args.smoke], test[: args.smoke], 1
    res = {"base": BASE, "pairs": {"fit": len(fit), "val": len(val), "test": len(test)}, "labels": [good, bad]}
    print(json.dumps(res), flush=True)

    def write(ps, path):
        rng = random.Random(0)
        rows = []
        for p in ps:
            for side, lab in (("win", good), ("lose", bad)):
                rows.append({"input": texts[p[side]], "output": {"classifications": [{"task": task, "labels": [good, bad], "true_label": [lab]}]}})
        rng.shuffle(rows)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    write(fit, out / "train.jsonl")
    write(val, out / "val.jsonl")
    model = AutoExtractor.from_pretrained(BASE).to("cuda")  # loads on the CPU: 2 s a text there (smoke, 10/05/2026)

    def p_good(m, ids):
        m.eval()
        sc = {}
        with torch.no_grad():
            for t in ids:
                r = m.classify_text(texts[t], {task: [good, bad]}, include_confidence=True)[task]
                sc[t] = r["confidence"] if r["label"] == good else 1 - r["confidence"]
        return sc

    def acc(sc, ps):
        return [float(sc[p["win"]] > sc[p["lose"]]) + 0.5 * (sc[p["win"]] == sc[p["lose"]]) for p in ps]

    ids = sorted({p[s] for p in test for s in ("win", "lose")})
    t0 = time.time()
    res["zero_shot"] = boot(acc(p_good(model, ids), test))
    print(json.dumps({"zero_shot": res["zero_shot"], "minutes": round((time.time() - t0) / 60, 1)}), flush=True)

    steps_per_epoch = max(1, 2 * len(fit) // 8)
    cfg = TrainingConfig(output_dir=str(out / "model"), num_epochs=args.epochs, batch_size=8, encoder_lr=1e-5, task_lr=5e-4,
                         eval_strategy="steps", eval_steps=steps_per_epoch, save_best=True, logging_steps=50, fp16=False, bf16=True, seed=0)
    t0 = time.time()
    log = ExtractorTrainer(model, cfg).train(train_data=str(out / "train.jsonl"), eval_data=str(out / "val.jsonl"))
    res["train_minutes"] = round((time.time() - t0) / 60, 1)
    res["train_log"] = {k: v for k, v in (log or {}).items() if isinstance(v, (int, float, str))}
    val_ids = sorted({p[s] for p in val for s in ("win", "lose")})
    res["val_acc"] = boot(acc(p_good(model, val_ids), val))
    sc = p_good(model, ids)
    dec = acc(sc, test)
    res["decide"] = boot(dec)
    idea = [float(feats[p["pair"]]["idea_score"][0] > feats[p["pair"]]["idea_score"][1]) for p in test]
    res["idea"] = boot(idea)
    res["decide_minus_idea"] = boot([a - b for a, b in zip(dec, idea)])
    if "size" in test[0]:
        res["smaller"] = boot([float(p["size"][0] < p["size"][1]) + 0.5 * (p["size"][0] == p["size"][1]) for p in test])

    # Decide plus idea scores, cross-fitted over the held-out pairs.
    def row(p):
        f = feats[p["pair"]]
        w, l = f["win_f"], f["lose_f"]
        return [sc[p["win"]] - sc[p["lose"]], f["idea_score"][0] - f["idea_score"][1], w["surprise"] - l["surprise"],
                w["relevance"] - l["relevance"], float(torch.log(torch.tensor(w["words"] + 1.0)) - torch.log(torch.tensor(l["words"] + 1.0)))]

    X = torch.tensor([row(p) for p in test], dtype=torch.float64)
    folds = [k % 5 for k in range(len(test))]
    random.Random(1).shuffle(folds)
    comb = [0.0] * len(test)
    for k in range(5):
        tr = [i for i in range(len(test)) if folds[i] != k]
        sd = X[tr].std(0).clamp(min=1e-9)
        w = logit_fit(X[tr] / sd)
        for i in range(len(test)):
            if folds[i] == k:
                comb[i] = float(float((X[i] / sd) @ w) > 0)
    res["decide_plus_idea"] = boot(comb)
    res["decide_plus_idea_minus_decide"] = boot([a - b for a, b in zip(comb, dec)])
    sd = X.std(0).clamp(min=1e-9)
    res["combined_weights_all"] = dict(zip(["decide", "idea", "surprise", "relevance", "log_words"], [round(float(x), 3) for x in logit_fit(X / sd)]))
    (out / "decide.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(out / "scores.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"pair": p["pair"], "decide": [sc[p["win"]], sc[p["lose"]]]}) + "\n" for p in test)
    print(json.dumps({k: v for k, v in res.items() if k != "train_log"}), flush=True)


if __name__ == "__main__":
    main()
