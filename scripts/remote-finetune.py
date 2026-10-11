"""GPU worker: fine-tune an encoder end to end for intent, then encode every phase-1 and pool text.

The same objective as the frozen head (``dianoia.head.loss``: InfoNCE at nested 64/128/256 prefixes,
every move variant a hard negative), but the encoder itself is trained, so it sees tokens rather than a
pooled vector. Two bases:

- ``bge``   BAAI/bge-base-en-v1.5, all weights trained, CLS pooling, then a 768 -> 256 projection.
- ``sonar`` SONAR's text encoder with LoRA adapters (r=16) on attention, mean pooling, then a
            1024 -> 256 projection.

Fit and validation items come from the laptop's split (train mailboxes only; validation = the head's
validation mailboxes). The epoch kept is the best validation win rate at 256 dims. Test items are
encoded only after training ends, with the kept checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from head import PREFIXES, loss as intent_loss, prefix

KEEP = ["paraphrase", "rename", "register", "drop_modifiers", "canonical"]
MOVE = ["act_change", "role_swap", "negation", "modality", "predicate_change", "type_change", "quantity"]
FIELDS = ["text", *KEEP, *MOVE]


class Encoder(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        self.base = base
        if base == "bge":
            from transformers import AutoModel, AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
            self.model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
            width = 768
        else:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoTokenizer
            from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

            rev = "29f2eac73636d1548c2cb227dbfb2772b31ebc4f"
            self.tok = AutoTokenizer.from_pretrained("cointegrated/SONAR_200_text_encoder", revision=rev)
            self.tok.src_lang = "eng_Latn"
            enc = M2M100Encoder.from_pretrained("cointegrated/SONAR_200_text_encoder", revision=rev, torch_dtype=torch.float16)
            enc.gradient_checkpointing_enable()
            enc.enable_input_require_grads()
            self.model = get_peft_model(enc, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
            width = 1024
        self.proj = nn.Linear(width, 256)

    def forward(self, texts: list[str]) -> torch.Tensor:
        b = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=96).to("cuda")
        h = self.model(**b).last_hidden_state
        if self.base == "bge":
            pooled = h[:, 0]
        else:
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1)
        return self.proj(pooled.float())


def batch_tensors(enc: Encoder, rows: list[dict]):
    texts, where = [], {}
    for f in FIELDS:
        for i, r in enumerate(rows):
            if r.get(f):
                where[(f, i)] = len(texts)
                texts.append(r[f])
    z = enc(texts).float()
    d = z.shape[1]
    out, mask = {}, {}
    for f in FIELDS:
        rows_z = [z[where[(f, i)]] if (f, i) in where else torch.zeros(d, device=z.device) for i in range(len(rows))]
        out[f] = torch.stack(rows_z)
        mask[f] = torch.tensor([(f, i) in where for i in range(len(rows))], device=z.device)
    return out, mask


@torch.no_grad()
def encode_texts(enc: Encoder, texts: list[str], batch: int = 256) -> np.ndarray:
    enc.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for s in range(0, len(texts), batch):
        idx = order[s : s + batch]
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = enc([texts[i] for i in idx]).float().cpu().numpy()
    return out


def val_rate(enc: Encoder, rows: list[dict]) -> float:
    texts, where = [], []
    for i, r in enumerate(rows):
        for f in FIELDS:
            if r.get(f):
                where.append((i, f, len(texts)))
                texts.append(r[f])
    z = encode_texts(enc, texts)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    vec = {(i, f): z[k] for i, f, k in where}
    wins = pairs = 0.0
    for i in range(len(rows)):
        x = vec[(i, "text")]
        for k in KEEP:
            for m in MOVE:
                if (i, k) in vec and (i, m) in vec:
                    sk, sm = float(x @ vec[(i, k)]), float(x @ vec[(i, m)])
                    wins += (sk > sm) + 0.5 * (sk == sm)
                    pairs += 1
    return wins / max(pairs, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["bge", "sonar"], required=True)
    ap.add_argument("--items", default="phase1/items.jsonl")
    ap.add_argument("--texts", nargs="*", default=["phase1/texts.jsonl", "pool/texts.jsonl"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default="out/ft")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(20260930)
    rows = [json.loads(line) for line in open(args.items, encoding="utf-8")]
    fit = [r for r in rows if r["role"] == "fit"]
    val = [r for r in rows if r["role"] == "val"]
    enc = Encoder(args.base).cuda()
    params = [{"params": [p for p in enc.model.parameters() if p.requires_grad], "lr": args.lr if args.base == "bge" else 2e-4},
              {"params": enc.proj.parameters(), "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=args.base == "bge")
    out = Path(args.out) / (args.base + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    best, log = (-1.0, -1), []
    rng = np.random.default_rng(20260930)
    for epoch in range(args.epochs):
        enc.train()
        t = time.time()
        perm = rng.permutation(len(fit))
        losses = []
        for s in range(0, len(perm), args.batch):
            b = [fit[i] for i in perm[s : s + args.batch]]
            with torch.autocast("cuda", dtype=torch.float16):
                z, m = batch_tensors(enc, b)
            neg = torch.stack([z[k] for k in MOVE], 1)
            neg_mask = torch.stack([m[k] for k in MOVE], 1)
            l = intent_loss(z["text"], {k: z[k] for k in KEEP}, {k: m[k] for k in KEEP}, neg, neg_mask)
            opt.zero_grad()
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(l.detach()))
        rate = val_rate(enc, val)
        log.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "val_win_256": round(rate, 4), "seconds": round(time.time() - t)})
        print(json.dumps(log[-1]), flush=True)
        if rate > best[0]:
            best = (rate, epoch)
            torch.save(enc.state_dict(), out / "best.pt")
    enc.load_state_dict(torch.load(out / "best.pt"))
    for path in args.texts:
        texts = [json.loads(line)["text"] for line in open(path, encoding="utf-8")]
        np.save(out / (Path(path).parent.name + ".npy"), encode_texts(enc, texts).astype(np.float16))
    (out / "train.json").write_text(json.dumps({"base": args.base, "best_epoch": best[1], "best_val_win_256": round(best[0], 4), "args": vars(args), "log": log}, indent=2))
    print("best", best, flush=True)


if __name__ == "__main__":
    main()
