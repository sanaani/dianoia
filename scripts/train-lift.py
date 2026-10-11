"""Train the lift: an intent vector (any prefix of it) to the SONAR vector of the canonical sentence.

SONAR's decoder reads raw SONAR vectors, so the lift regresses the canonical sentence's raw vector
(cosine plus scaled MSE). During training the intent vector is cut to a random prefix of 64, 128 or
256 dims (the rest zeroed), so one lift decodes every truncation. Same fit/validation split as the
head; test mailboxes are never loaded.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from dianoia import head
from dianoia.paths import result_path, work_path

SEED = 20260930


class Lift(nn.Module):
    def __init__(self, d_in: int = 256, d_hidden: int = 2048, d_out: int = 1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def cut(z: torch.Tensor, d: int) -> torch.Tensor:
    """The prefix of length d, normalised, zero-padded back to full width."""
    out = torch.zeros_like(z)
    out[:, :d] = F.normalize(z[:, :d], dim=-1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--intent", default="ft_sonar", help="phase1/vec/<name>.npy holding the intent vectors (256-d)")
    ap.add_argument("--pool", action="store_true", help="also fit pool sentences from train mailboxes to their own SONAR vectors")
    args = ap.parse_args()
    torch.manual_seed(SEED)

    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    index = pd.read_parquet(work_path("phase1", "index.parquet"))
    sonar = np.load(work_path("phase1", "vec", "sonar.npy")).astype(np.float32)
    val_boxes = set(json.loads(result_path("phase-1-intent", "train-head.json").read_text())["val_mailboxes"])
    lines = index.pivot_table(index="item", columns="field", values="line", aggfunc="first")
    train = items[(items["split"] == "train")].index.intersection(lines.index[lines["canonical"].notna()])
    src_line = lines.loc[train, "text"].astype(int).to_numpy()
    can_line = lines.loc[train, "canonical"].astype(int).to_numpy()
    is_val = items.loc[train, "mailbox"].isin(val_boxes).to_numpy()

    z = torch.from_numpy(np.load(work_path("phase1", "vec", f"{args.intent}.npy")).astype(np.float32)[src_line])
    target = torch.from_numpy(sonar[can_line])
    scale = target.norm(dim=1).mean()
    if args.pool:
        gen = json.loads(result_path("phase-1-intent", "generation.json").read_text())
        pool = pd.read_parquet(work_path("pool", "pool.parquet"))
        keep = (~pool["mailbox"].isin(set(gen["test_mailboxes"]) | val_boxes)).to_numpy()
        pz = torch.from_numpy(np.load(work_path("pool", "out", f"{args.intent}.npy"), mmap_mode="r")[keep].astype(np.float32))
        pt = torch.from_numpy(np.load(work_path("pool", "out", "sonar.npy"), mmap_mode="r")[keep].astype(np.float32))
        z = torch.cat([z, pz])
        target = torch.cat([target, pt])
        is_val = np.concatenate([is_val, np.zeros(len(pz), dtype=bool)])
        print("pool pairs added:", len(pz), flush=True)

    lift = Lift()
    opt = torch.optim.AdamW(lift.parameters(), lr=args.lr, weight_decay=0.01)
    fit, val = torch.from_numpy(~is_val), torch.from_numpy(is_val)
    best, log = (-1.0, -1), []
    for epoch in range(args.epochs):
        lift.train()
        perm = torch.nonzero(fit).squeeze(1)[torch.randperm(int(fit.sum()))]
        for s in range(0, len(perm), 512):
            b = perm[s : s + 512]
            d = int(np.random.choice(head.PREFIXES))
            y = lift(cut(z[b], d))
            l = (1 - F.cosine_similarity(y, target[b])).mean() + F.mse_loss(y / scale, target[b] / scale)
            opt.zero_grad()
            l.backward()
            opt.step()
        lift.eval()
        with torch.no_grad():
            cos = {d: float(F.cosine_similarity(lift(cut(z[val], d)), target[val]).mean()) for d in head.PREFIXES}
        log.append({"epoch": epoch, **{f"val_cos_{d}": round(c, 4) for d, c in cos.items()}})
        print(log[-1], flush=True)
        if cos[256] > best[0]:
            best = (cos[256], epoch)
            torch.save(lift.state_dict(), work_path("phase1", f"lift-{args.intent}{'-pool' if args.pool else ''}.pt"))
    result_path("phase-2-round-trip", f"train-lift-{args.intent}{'-pool' if args.pool else ''}.json").write_text(json.dumps({"best_epoch": best[1], "best_val_cos_256": round(best[0], 4), "log": log}, indent=2) + "\n")


if __name__ == "__main__":
    main()
