"""Phase 13 step 2d GPU stage: a scoring filter for Fastino's rejects (results/phase-13-scale-far/HYPOTHESES.md).

    python remote-p13-filter.py --work p13/filter --config p13/config.used.yaml --out p13/filter/out [--smoke]

Inputs in --work: train.jsonl and audit.json (p13-data.py filter-build and filter-audit; training refuses to start
unless the audit passed on this exact file); ../far-text.jsonl (every far pair, to score). Fine-tunes the config's
cross-encoder (filter.base, one logit) on the train split with binary cross-entropy, one epoch at a time, saving a
checkpoint after each; the epoch kept is the one with the highest dev recall in the top filter.top_share (the F1
measure, on dev). The holdout is never read until that model is chosen. Writes holdout-scores.jsonl and
far-scores.jsonl (id, p) and train-log.json. Resumable: finished epochs and score files are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import torch
import yaml


def log(**kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **kw}), flush=True)


def require_audit(work):
    au = json.loads((work / "audit.json").read_text())
    assert au["pass"], f"audit failed: {au['violations']}"
    for f, h in au["sha256"].items():
        assert hashlib.sha256((work / f).read_bytes()).hexdigest() == h, f"{f} changed since the audit"


def recall_top(ys, ps, share):
    order = sorted(range(len(ys)), key=lambda i: -ps[i])
    top = order[:max(1, round(share * len(order)))]
    pos = sum(ys)
    return sum(ys[i] for i in top) / pos if pos else 0.0


@torch.no_grad()
def score(model, tok, rows, dev, bs, max_len):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        ch = rows[i:i + bs]
        x = tok([r["citing_text"] for r in ch], [r["cited_text"] for r in ch], truncation=True, max_length=max_len,
                padding=True, return_tensors="pt").to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            out += torch.sigmoid(model(**x).logits.float().view(-1)).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    fc = cfg["filter"]
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not a.smoke:
        require_audit(work)
    rows = [json.loads(l) for l in open(work / "train.jsonl", encoding="utf-8")]
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "holdout")}
    name, max_len, bs, epochs = fc["base"], 512, 16, fc.get("epochs", 3)
    if a.smoke:
        by = {s: v[:64] for s, v in by.items()}
        max_len, epochs = 128, 1
        if dev == "cpu":
            name = fc.get("smoke_base", name)
    log(stage="start", device=dev, model=name, **{s: len(v) for s, v in by.items()},
        positives={s: sum(r["y"] for r in v) for s, v in by.items()}, smoke=a.smoke)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    ck = out / "ckpt"
    hist_f = out / "train-log.json"
    hist = json.loads(hist_f.read_text()) if hist_f.exists() else []
    start = len(hist)
    model = AutoModelForSequenceClassification.from_pretrained(str(ck / "last") if start else name, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=fc.get("lr", 2e-5), weight_decay=0.01)
    if start and (ck / "opt.pt").exists():
        opt.load_state_dict(torch.load(ck / "opt.pt", map_location=dev))
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(start, epochs):
        t0 = time.time()
        model.train()
        tr = by["train"][:]
        random.Random(fc["seed"] + ep).shuffle(tr)
        tot = 0.0
        for i in range(0, len(tr), bs):
            ch = tr[i:i + bs]
            x = tok([r["citing_text"] for r in ch], [r["cited_text"] for r in ch], truncation=True,
                    max_length=max_len, padding=True, return_tensors="pt").to(dev)
            y = torch.tensor([float(r["y"]) for r in ch], device=dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logit = model(**x).logits.float().view(-1)
            loss = lossf(logit, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(ch)
        dp = score(model, tok, by["dev"], dev, 64, max_len)
        dr = recall_top([r["y"] for r in by["dev"]], dp, fc["top_share"])
        # A checkpoint per epoch (Shaun, 10/08/2026: long steps save progress); "best" is the dev choice so far.
        shutil.rmtree(ck / "last", ignore_errors=True)
        model.save_pretrained(ck / "last")
        torch.save(opt.state_dict(), ck / "opt.pt")
        if not hist or dr > max(h["dev_recall"] for h in hist):
            shutil.rmtree(ck / "best", ignore_errors=True)
            shutil.copytree(ck / "last", ck / "best")
        hist.append({"epoch": ep + 1, "train_loss": round(tot / len(tr), 4), "dev_recall": round(dr, 4),
                     "minutes": round((time.time() - t0) / 60, 1)})
        hist_f.write_text(json.dumps(hist, indent=1))
        log(stage="epoch", **hist[-1])
    best = max(hist, key=lambda h: h["dev_recall"])
    log(stage="chosen", **best)
    model = AutoModelForSequenceClassification.from_pretrained(str(ck / "best"), num_labels=1).to(dev)
    tok.save_pretrained(ck / "best")
    hf = out / "holdout-scores.jsonl"
    if not hf.exists():
        ps = score(model, tok, by["holdout"], dev, 64, max_len)
        hf.write_text("".join(json.dumps({"id": r["id"], "p": round(p, 5)}) + "\n" for r, p in zip(by["holdout"], ps)))
    # Every far pair, streamed in chunks and appended; a restart skips what is on disk.
    ff = out / "far-scores.jsonl"
    done = {json.loads(l)["id"] for l in open(ff, encoding="utf-8")} if ff.exists() else set()
    chunk, n = [], len(done)
    with open(ff, "a", encoding="utf-8", newline="\n") as f:
        for l in open(work.parent / "far-text.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in done:
                continue
            chunk.append(r)
            if len(chunk) == 4096 or (a.smoke and len(chunk) == 64):
                f.writelines(json.dumps({"id": x["id"], "p": round(p, 5)}) + "\n"
                             for x, p in zip(chunk, score(model, tok, chunk, dev, 128, max_len)))
                f.flush()
                n += len(chunk)
                chunk = []
                log(stage="far", scored=n)
                if a.smoke:
                    break
        if chunk and not a.smoke:
            f.writelines(json.dumps({"id": x["id"], "p": round(p, 5)}) + "\n"
                         for x, p in zip(chunk, score(model, tok, chunk, dev, 128, max_len)))
            n += len(chunk)
    log(stage="done", far_scored=n, chosen_epoch=best["epoch"])


if __name__ == "__main__":
    main()
