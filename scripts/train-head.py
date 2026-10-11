"""Train the intent head on train-mailbox items, choosing the epoch on a validation slice.

Validation is 10% of the *train* mailboxes, drawn by seed; test mailboxes are never loaded here.
The checkpoint kept is the epoch with the best validation win rate at 256 dims.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from dianoia import head, tier1
from dianoia.paths import result_path, work_path

SEED = 20260930
KEEP_TRAIN = ["paraphrase", "rename", "register", "drop_modifiers", "canonical"]


def load(split: str, inputs: tuple[str, ...] = ("sonar",)):
    """The head's input is the concatenation of the named encoders' vectors, each unit-normalised."""
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    index = pd.read_parquet(work_path("phase1", "index.parquet"))
    parts = []
    for name in inputs:
        v = np.load(work_path("phase1", "vec", f"{name}.npy")).astype(np.float32)
        parts.append(v / np.linalg.norm(v, axis=1, keepdims=True) if len(inputs) > 1 else v)
    vecs = np.concatenate(parts, axis=1)
    keep = items.index[items["split"] == split].to_numpy()
    return items, index, vecs, keep


def tensors(items_idx: np.ndarray, index: pd.DataFrame, vecs: np.ndarray):
    """Per field, an (n, 1024) array (zeros where absent) and an (n,) mask, rows in items_idx order."""
    pos = {i: r for r, i in enumerate(items_idx)}
    fields = ["text", *KEEP_TRAIN, *tier1.MOVE]
    arr = {f: np.zeros((len(items_idx), vecs.shape[1]), dtype=np.float32) for f in fields}
    mask = {f: np.zeros(len(items_idx), dtype=bool) for f in fields}
    for row in index.itertuples(index=False):
        if row.item in pos and row.field in arr:
            arr[row.field][pos[row.item]] = vecs[row.line]
            mask[row.field][pos[row.item]] = True
    return arr, mask


def val_rate(model, arr, mask, d: int = 256) -> float:
    model.eval()
    with torch.no_grad():
        z = {f: head.prefix(model(torch.from_numpy(a)), d).numpy() for f, a in arr.items()}
    sim = {f: np.where(mask[f], (z["text"] * z[f]).sum(1), np.nan) for f in z if f != "text"}
    w, c = tier1.pair_wins(sim)["all"]
    return float(w.sum() / max(c.sum(), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--inputs", nargs="*", default=["sonar"])
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of fit items used (learning curve)")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    items, index, vecs, train_idx = load("train", tuple(args.inputs))
    boxes = sorted(items.loc[train_idx, "mailbox"].unique())
    val_boxes = set(rng.choice(boxes, size=max(1, round(0.1 * len(boxes))), replace=False))
    is_val = items.loc[train_idx, "mailbox"].isin(val_boxes).to_numpy()
    fit_idx, val_idx = train_idx[~is_val], train_idx[is_val]
    if args.fraction < 1.0:
        fit_idx = np.sort(rng.choice(fit_idx, size=round(args.fraction * len(fit_idx)), replace=False))
    fit, fit_m = tensors(fit_idx, index, vecs)
    val, val_m = tensors(val_idx, index, vecs)
    ft = {f: torch.from_numpy(a) for f, a in fit.items()}
    fm = {f: torch.from_numpy(m) for f, m in fit_m.items()}

    model = head.IntentHead(d_in=vecs.shape[1], d_hidden=args.hidden, dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, log = (-1.0, -1), []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(fit_idx))
        losses = []
        for s in range(0, len(perm), args.batch):
            b = perm[s : s + args.batch]
            z = {f: model(ft[f][b]) for f in ft}
            neg = torch.stack([z[m] for m in tier1.MOVE], dim=1)
            neg_mask = torch.stack([fm[m][b] for m in tier1.MOVE], dim=1)
            pos = {k: z[k] for k in KEEP_TRAIN}
            pos_mask = {k: fm[k][b] for k in KEEP_TRAIN}
            l = head.loss(z["text"], pos, pos_mask, neg, neg_mask, tau=args.tau)
            opt.zero_grad()
            l.backward()
            opt.step()
            losses.append(float(l.detach()))
        sched.step()
        rate = val_rate(model, val, val_m)
        log.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "val_win_256": round(rate, 4)})
        print(log[-1], flush=True)
        if rate > best[0]:
            best = (rate, epoch)
            torch.save(model.state_dict(), work_path("phase1", f"head{args.tag}.pt"))
    summary = {"fit_items": len(fit_idx), "val_items": len(val_idx), "val_mailboxes": sorted(val_boxes),
               "best_epoch": best[1], "best_val_win_256": round(best[0], 4), "args": vars(args), "log": log}
    result_path("phase-1-intent", f"train-head{args.tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("best", best)


if __name__ == "__main__":
    main()
