"""Phase 10: fine-tune Fastino's GLiNER2.5-Decide on Sonnet's borrowing scores, then score pool pairs (10/06/2026).

    python remote-p10-decide.py --seed s1.jsonl,s2.jsonl --labels l1.jsonl,l2.jsonl --gold gold-all.jsonl \
        --pool pool.jsonl --out p10/decide2 [--smoke 10]

Round 1 (10/05/2026) trained on Sonnet's plain C label and kept "real" pairs at 50% precision, with a confidence that
was effectively 0 or 1. Shaun (10/06/2026): train on the Sonnet scores instead. A pair is positive only if Sonnet
called it real with borrow >= 4 (86% real in Claude's hand check), negative if Sonnet called it same technology or
unrelated; Sonnet's weaker "real" calls (borrow <= 3) are left out as ambiguous. All seed pairs are from 2013-2017,
never a val or test year.

Judged on Claude's 190 hand labels (140 adjudicated, plus the 50 val-year pairs of the first pool check) and on
held-out seed pairs, by precision in the top 5/10/20/30% by score (ranks, so a saturated confidence cannot hide
signal) and by AUC. The pool file gets a score per pair. This labels input data; no model output is graded.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

BASE = "fastino/GLiNER2.5-Decide"
TASK, YES, NO = "borrowing", "real borrowing", "not a borrowing"
# Exclusion stages (Shaun, 10/06/2026: "We're not trying to prove inclusion in the set. We're trying to prove exclusion
# through a series of steps."). The include model learned mostly real vs unrelated, two thirds of its negatives, and
# kept only 16 of 41 real pairs in the fresh check. Each stage now removes one kind of non-borrowing and is trained on
# that distinction only; a pair survives if every stage keeps it. Per target: (task, remove label, keep label, rule
# from a Sonnet label to keep=True / remove=False / unused=None, hand label counted as keep in the gold check).
TARGETS = {
    "include": ("borrowing", "not a borrowing", "real borrowing",
                lambda l, b, mb: (b >= mb or None) if l == "C" else False, lambda h: h == "C"),
    "unrelated": ("unrelated", "unrelated", "related",
                  lambda l, b, mb: l != "B", lambda h: h != "B"),
    "same-tech": ("same technology", "same technology", "real borrowing",
                  lambda l, b, mb: (b >= mb or None) if l == "C" else (False if l == "A" else None), lambda h: h == "C"),
    # Stage 3 (Shaun, 10/06/2026: a Fastino pass that emulates what Sonnet screened): trained on stage 1-2 survivors,
    # K = Sonnet v3 real with borrow >= 4 and Sonnet v4 C or G, R = anything else (labels built locally).
    "screen": ("analogy screen", "remove", "keep", lambda l, b, mb: l == "K", lambda h: h == "C"),
}
ADJUDICATED = {"9105785-6538129": "C", "8877500-4349026": "C"}  # DATA-REVIEW.md, 10/05/2026


def text(r):
    return f"CITING: {r['citing_text']}\nCITED: {r['cited_text']}"


def ranked(p, y):
    """Precision in the top fraction by score, and AUC (ties count half)."""
    order = sorted(range(len(p)), key=lambda i: -p[i])
    out = {}
    for f in (0.05, 0.1, 0.2, 0.3):
        top = order[: max(1, round(f * len(p)))]
        out[f"top{int(f * 100)}%"] = {"kept": len(top), "precision": round(sum(y[i] for i in top) / len(top), 3)}
    pos = [p[i] for i in range(len(p)) if y[i]]
    neg = [p[i] for i in range(len(p)) if not y[i]]
    auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(len(pos) * len(neg), 1)
    out["auc"] = round(auc, 3)
    out["base_rate"] = round(sum(y) / max(len(y), 1), 3)
    out["distinct_scores"] = len({round(x, 4) for x in p})
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ("seed", "labels", "gold", "pool", "out"):
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--min-borrow", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--score-only", default="", help="a trained model dir: skip training, only score --pool")
    ap.add_argument("--target", choices=tuple(TARGETS), default="include")
    ap.add_argument("--max-fit", type=int, default=0, help="cap on training pairs before balancing (time)")
    ap.add_argument("--pool-keep", default="", help="earlier stages' pool-scores files: score only pairs all kept")
    ap.add_argument("--batch-size", type=int, default=1, help=">1: score with batch_classify_text")
    ap.add_argument("--speed-test", type=int, default=0, help="score N pool pairs both ways, compare, and stop")
    ap.add_argument("--bf16", action="store_true", help="score in bfloat16 (with --score-only)")
    a = ap.parse_args()
    global TASK, YES, NO
    TASK, NO, YES, rule, hand_keep = TARGETS[a.target]  # p_yes is P(keep) for every target: callers cut at 0.5 alike
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lab = {}
    for f in a.labels.split(","):
        lab.update({r["id"]: r for r in map(json.loads, open(f, encoding="utf-8"))})

    def target(r):
        l = lab.get(r["id"])
        if not l:
            return None
        return rule(l["label"], l.get("borrow") or 0, a.min_borrow)

    seed = []
    for f in a.seed.split(","):
        for r in map(json.loads, open(f, encoding="utf-8")):
            t = target(r)
            if t is not None:
                seed.append({**r, "y": t})
    gold = [{**r, "y": hand_keep(ADJUDICATED.get(r["id"], r["label"]))} for r in map(json.loads, open(a.gold, encoding="utf-8"))]
    citing = sorted({r["citing"] for r in seed})
    held = set(random.Random(0).sample(citing, len(citing) // 10))
    fit = [r for r in seed if r["citing"] not in held]
    if a.max_fit and len(fit) > a.max_fit:
        fit = random.Random(2).sample(fit, a.max_fit)
    hold = [r for r in seed if r["citing"] in held]
    if a.smoke:
        fit, hold, gold, a.epochs = fit[: a.smoke * 4], hold[: a.smoke], gold[: a.smoke], 1
    res = {"base": BASE, "fit": len(fit), "holdout": len(hold), "gold": len(gold),
           "fit_positive_share": round(sum(r["y"] for r in fit) / max(len(fit), 1), 3), "min_borrow": a.min_borrow}
    print(json.dumps(res), flush=True)

    def write(rows, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"input": text(r), "output": {"classifications": [{"task": TASK, "labels": [YES, NO], "true_label": [YES if r["y"] else NO]}]}}) + "\n")

    # Balance the two classes by repeating the smaller one (include: real is 11%; remove: real is about 28%).
    pos = [r for r in fit if r["y"]]
    bal = fit + pos * max(0, round((len(fit) - len(pos)) / max(len(pos), 1)) - 1)
    random.Random(1).shuffle(bal)
    write(bal, out / "train.jsonl")
    write(hold, out / "val.jsonl")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoExtractor.from_pretrained(a.score_only or BASE).to(dev)
    if a.bf16 and a.score_only:  # scoring only (10/06/2026 speed probe); training keeps its own precision settings
        model = model.to(torch.bfloat16)

    def p_one(rows):
        model.eval()
        ps = []
        with torch.no_grad():
            for r in rows:
                o = model.classify_text(text(r), {TASK: [YES, NO]}, include_confidence=True)[TASK]
                ps.append(o["confidence"] if o["label"] == YES else 1 - o["confidence"])
        return ps

    # Batched scoring (10/06/2026): one pair at a time ran 30 pairs/s, so the 5.08M-pair corpus would need ~66 GPU
    # hours. --speed-test checks that batch_classify_text gives the same scores before any run relies on it.
    def p_batch(rows):
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(rows), 512):
                outs = model.batch_classify_text([text(r) for r in rows[i:i + 512]], {TASK: [YES, NO]},
                                                 batch_size=a.batch_size, include_confidence=True)
                ps += [o[TASK]["confidence"] if o[TASK]["label"] == YES else 1 - o[TASK]["confidence"] for o in outs]
        return ps

    p_yes = p_batch if a.batch_size > 1 else p_one
    if a.speed_test:
        rows = [json.loads(l) for _, l in zip(range(a.speed_test), open(a.pool.split(",")[0], encoding="utf-8"))]
        t0 = time.time(); one = p_one(rows); t1 = time.time(); bat = p_batch(rows); t2 = time.time()
        diff = max(abs(x - y) for x, y in zip(one, bat))
        res["speed_test"] = {"pairs": len(rows), "one_per_s": round(len(rows) / (t1 - t0), 1),
                             "batch_per_s": round(len(rows) / (t2 - t1), 1), "batch_size": a.batch_size,
                             "max_abs_diff": round(diff, 5), "same_side_of_0.5": sum((x >= .5) == (y >= .5) for x, y in zip(one, bat))}
        print(json.dumps({"speed_test": res["speed_test"]}), flush=True)
        (out / "decide.json").write_text(json.dumps(res, indent=1))
        return
    if a.score_only:
        return score_pool(a, out, p_yes, res)
    res["zero_shot_gold"] = ranked(p_yes(gold), [r["y"] for r in gold])
    print(json.dumps({"zero_shot_gold": res["zero_shot_gold"]}), flush=True)
    cfg = TrainingConfig(output_dir=str(out / "model"), num_epochs=a.epochs, batch_size=8, encoder_lr=1e-5, task_lr=5e-4,
                         eval_strategy="steps", eval_steps=max(1, len(bal) // 8), save_best=True, logging_steps=50,
                         fp16=False, bf16=dev == "cuda", seed=0)
    t0 = time.time()
    ExtractorTrainer(model, cfg).train(train_data=str(out / "train.jsonl"), eval_data=str(out / "val.jsonl"))
    res["train_minutes"] = round((time.time() - t0) / 60, 1)
    ph = p_yes(hold)
    res["holdout_vs_sonnet"] = ranked(ph, [r["y"] for r in hold])
    # A remover is set from the removal side (Shaun, 10/06/2026): at each cut, the share of pairs it removes and the
    # share of Sonnet-kept pairs it would wrongly remove. The cut used is the one losing at most 5% of keeps.
    keeps = [p for p, r in zip(ph, hold) if r["y"]]
    res["removal_curve"] = [{"cut": c, "removed": round(sum(p < c for p in ph) / max(len(ph), 1), 3),
                             "keeps_lost": round(sum(p < c for p in keeps) / max(len(keeps), 1), 3)}
                            for c in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)]
    print(json.dumps({"removal_curve": res["removal_curve"]}), flush=True)
    res["gold_vs_hand"] = ranked(p_yes(gold), [r["y"] for r in gold])
    print(json.dumps({k: res[k] for k in ("holdout_vs_sonnet", "gold_vs_hand")}), flush=True)
    score_pool(a, out, p_yes, res)


def score_pool(a, out, p_yes, res):
    t0, n = time.time(), 0
    # A later stage scores only what the earlier stages kept (--pool-keep: their pool-scores files), plus any extra
    # pool files given after the first (e.g. the hand-checked pairs, so each stage is measured on them on its own).
    keep = None
    for kf in filter(None, (a.pool_keep or "").split(",")):
        ok = {r["id"] for r in map(json.loads, open(kf, encoding="utf-8")) if r["p"] >= 0.5}
        keep = ok if keep is None else keep & ok
    seen, buf = set(), []
    with open(out / "pool-scores.jsonl", "w", encoding="utf-8") as f:
        def flush():
            nonlocal n
            for r, p in zip(buf, p_yes(buf)):  # chunks of 2,048, so batched scoring pays off
                f.write(json.dumps({"id": r["id"], "p": round(p, 5)}) + "\n")
            n += len(buf)
            buf.clear()
            print(json.dumps({"scored": n, "per_second": round(n / (time.time() - t0), 1)}), flush=True)

        for i, line in ((i, l) for i, pf in enumerate(a.pool.split(",")) for l in open(pf, encoding="utf-8")):
            r = json.loads(line)
            if r["id"] in seen or (i == 0 and keep is not None and r["id"] not in keep):
                continue
            seen.add(r["id"])
            buf.append(r)
            if a.smoke and n + len(buf) >= a.smoke:
                break
            if len(buf) == 2048:
                flush()
        if buf:
            flush()
    res["pool_scored"] = n
    res["pool_per_second"] = round(n / max(time.time() - t0, 1e-9), 1)
    (out / "decide.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"done": True, "pool_scored": n}), flush=True)


if __name__ == "__main__":
    main()
