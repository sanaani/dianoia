"""Phase 15 GPU stage (results/phase-15-curve-screen/HYPOTHESES.md, approved 10/09/2026).

    python remote-p15.py --work . --config p15/config.used.yaml --out p15/out [--part curve|screen|both] [--smoke]

curve (step 1): phase 12's far training recipe (remote-p12.train, unchanged) on each nested subset in p15/curve, 3
  epochs, each epoch scored on phase 12's patent dev queries; the epoch dev chooses (top-10, then MRR) is then scored
  on phase 12's far patent test and phase 14's 255 paper pairs, exactly as remote-p12 / remote-p14 score them.
  Writes scores/<run>-e<k>-dev.json, scores/<run>-patent.json, scores/<run>-paper.json (per query: hit, rr).
screen (step 2): fastino/GLiNER2.5-Decide fine-tuned on p15/screen/pairs.jsonl's fit split (positives repeated,
  negatives sampled), best checkpoint by validation loss, then P(real borrowing) for every test pair.
  Writes screen/scores.jsonl and screen/screen.json.
Every finished stage is skipped on a rerun. Writes p15.json at the end. Verdicts: scripts/p15-verdicts.py.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import shutil
import time
from pathlib import Path

import torch
import yaml


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load("r12", "remote-p12.py")
R14 = _load("r14", "remote-p14.py")
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]


def curve(cfg, work, out, dev, smoke):
    R.require_audit(work / "p12")
    c, sm = cfg["curve"], cfg["smoke"]
    name = sm["embedder"] if smoke and dev == "cpu" else cfg["embedder"]
    if smoke and dev == "cpu":
        cfg["train"].update(batch=2, chunk=4, max_tokens=64)
    eval_len = 64 if smoke and dev == "cpu" else 512
    ev12 = rd(work / "p12/eval.jsonl")
    sets = {"dev": [r for r in ev12 if r["split"] == "dev"], "patent": [r for r in ev12 if r["split"] == "test"],
            "paper": rd(work / "p14/step3/eval.jsonl")}
    if smoke:
        sets = {k: v[:sm["queries"]] for k, v in sets.items()}
        for v in sets.values():
            for r in v:
                r["candidates"] = r["candidates"][:sm["candidates"]]
    need = {k: {x for r in v for x in [r["cited"], *r["candidates"]]} for k, v in sets.items()}
    pat = need["dev"] | need["patent"]
    docs = {r["id"]: r["text"] for r in rd(work / "p12/docs.jsonl") if r["id"] in pat}
    docs.update({r["id"]: r["text"] for r in rd(work / "p14/step3/docs.jsonl") if r["id"] in need["paper"]})
    ndocs = {r["id"]: r["text"] for r in rd(work / "p12/train-docs.jsonl")}
    R.log(stage="curve-start", device=dev, model=name, **{k: len(v) for k, v in sets.items()}, docs=len(docs))

    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    fresh = lambda: AutoModel.from_pretrained(name, dtype=R.DTYPE[dev]).to(dev)
    k = cfg["recovery"]["top_k"]

    def scored(tag, split, model):
        f = out / "scores" / f"{tag}-{split}.json"
        if not f.exists():
            t0 = time.time()
            ids = sorted(need[split])
            dv = dict(zip(ids, R.encode_all(model, tok, [docs[i] for i in ids], dev, max_len=eval_len)))
            qv = R.encode_all(model, tok, [cfg["instruct"] + r["query_text"] for r in sets[split]], dev, max_len=eval_len)
            res = R14.ranks(sets[split], qv, dv, k)
            f.write_text(json.dumps(res))
            R.log(stage="scored", model=tag, split=split, minutes=round((time.time() - t0) / 60, 1),
                  top10=round(sum(x["hit"] for x in res.values()) / max(1, len(res)), 4))
        return json.loads(f.read_text())

    top = lambda m: (sum(x["hit"] for x in m.values()) / max(1, len(m)), sum(x["rr"] for x in m.values()) / max(1, len(m)))
    chosen = {}
    E = max(c["epochs"])
    for n in c["sizes"]:
        run = f"far{n}-s0"
        pairs = rd(work / f"p15/curve/train-{n}.jsonl")
        if smoke:
            pairs = pairs[:sm["train"]]
        tags = [f"{run}-e{e}" for e in c["epochs"]]
        if not all((out / "scores" / f"{t}-dev.json").exists() for t in tags):
            if all((out / "adapters" / t).exists() for t in tags):
                for t in tags:
                    scored(t, "dev", PeftModel.from_pretrained(fresh(), out / "adapters" / t).to(dev))
            else:
                for e, m in R.train(fresh(), tok, pairs, ndocs, cfg, dev, out, run, E, 0, sm["steps"] if smoke else 0):
                    scored(f"{run}-e{e}", "dev", m)
                del m
        ep = max(c["epochs"], key=lambda e: top(json.loads((out / "scores" / f"{run}-e{e}-dev.json").read_text())))
        chosen[n] = ep
        R.log(stage="chosen", run=run, epoch=ep)
        if not all((out / "scores" / f"{run}-{s}.json").exists() for s in ("patent", "paper")):
            m = PeftModel.from_pretrained(fresh(), out / "adapters" / f"{run}-e{ep}").to(dev)
            for s in ("patent", "paper"):
                r = scored(f"{run}-e{ep}", s, m)
                (out / "scores" / f"{run}-{s}.json").write_text(json.dumps(r))
            del m
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    return {"chosen_epochs": chosen}


def screen(cfg, work, out, dev, smoke):
    s = cfg["screen"]
    o = out / "screen"
    o.mkdir(parents=True, exist_ok=True)
    if (o / "scores.jsonl").exists():
        return json.loads((o / "screen.json").read_text())
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
    good, bad = s["labels"]
    task = "borrowing"
    rows = rd(work / "p15/screen/pairs.jsonl")
    by = {k: [r for r in rows if r["split"] == k] for k in ("fit", "val", "test")}
    rng = random.Random(s["seed"])
    pos = [r for r in by["fit"] if r["y"]]
    neg = [r for r in by["fit"] if not r["y"]]
    fit = pos * s["pos_repeat"] + rng.sample(neg, min(s["n_neg"], len(neg)))
    rng.shuffle(fit)
    val, test, epochs = by["val"], by["test"], s["epochs"]
    if smoke:
        n = cfg["smoke"]["screen"]
        fit, val, epochs = fit[:n], val[:n], 1
        test = [r for r in test if r["y"]][:n // 2] + [r for r in test if not r["y"]][:n // 2]
    res = {"base": s["base"], "rows": {"fit": len(fit), "fit_pos": sum(r["y"] for r in fit), "val": len(val),
                                       "test": len(test), "test_pos": sum(r["y"] for r in test)}}
    R.log(stage="screen-start", device=dev, **res["rows"])

    def write(rs, path):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps({"input": r["text"], "output": {"classifications": [
                {"task": task, "labels": [good, bad], "true_label": [good if r["y"] else bad]}]}}, ensure_ascii=False) + "\n"
                for r in rs)

    write(fit, o / "train.jsonl")
    write(val, o / "val.jsonl")
    model = AutoExtractor.from_pretrained(s["base"]).to(dev)
    cfgT = TrainingConfig(output_dir=str(o / "model"), num_epochs=epochs, batch_size=8, encoder_lr=1e-5, task_lr=5e-4,
                          eval_strategy="epoch", save_best=True, logging_steps=50, fp16=False, bf16=dev == "cuda", seed=0)
    t0 = time.time()
    ExtractorTrainer(model, cfgT).train(train_data=str(o / "train.jsonl"), eval_data=str(o / "val.jsonl"))
    res["train_minutes"] = round((time.time() - t0) / 60, 1)
    del model
    gc.collect()
    model = AutoExtractor.from_pretrained(str(o / "model" / "best")).to(dev)
    model.eval()
    t0 = time.time()
    with torch.no_grad(), open(o / "scores.part.jsonl", "w", encoding="utf-8") as f:
        for r in test:
            c = model.classify_text(r["text"], {task: [good, bad]}, include_confidence=True)[task]
            p = c["confidence"] if c["label"] == good else 1 - c["confidence"]
            f.write(json.dumps({"id": r["id"], "p": round(float(p), 6), "y": r["y"], "y3": r["y3"]}) + "\n")
    res["score_minutes"] = round((time.time() - t0) / 60, 1)
    (o / "scores.part.jsonl").replace(o / "scores.jsonl")
    (o / "screen.json").write_text(json.dumps(res, indent=1))
    shutil.rmtree(o / "model", ignore_errors=True)  # checkpoints of 1.4 GB each; the scores are the result
    R.log(stage="screen-done", **res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=("curve", "screen", "both"), default="both")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    work, out = Path(a.work), Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    if a.part in ("curve", "both"):
        res["curve"] = curve(cfg, work, out, dev, a.smoke)
    if a.part in ("screen", "both"):
        res["screen"] = screen(cfg, work, out, dev, a.smoke)
    if a.part == "both":
        (out / "p15.json").write_text(json.dumps(res, indent=1))
    R.log(stage="done", **{k: list(v) for k, v in res.items()})


if __name__ == "__main__":
    main()
