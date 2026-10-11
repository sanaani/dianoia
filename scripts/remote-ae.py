"""GPU worker: our own encoder-decoder (stage 0 plumbing, stage 1 bottleneck).

Encoder: BAAI/bge-base-en-v1.5, all weights trained, CLS -> Linear(768, 256) -> the idea vector z.
Decoder: Qwen/Qwen3-0.6B with LoRA (r=16) reading z as K soft prefix tokens through
Linear(256, K * hidden). Loss: token cross entropy on the target sentence only.

Every batch cuts z to a random prefix of 64, 128 or 256 dims (half the time 256), zeroing the rest,
so one model decodes every truncation.

Stage 0 (``--stage 0``): target = the input sentence itself, on pool sentences from fit mailboxes.
Stage 1 (``--stage 1``): built on a stage-0 checkpoint; see ``remote-ae.py --help``.

After training, decodes held-out sentences greedily at 64 / 128 / 256 dims, plus a random-vector
noise floor, and writes ``decoded.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PREFIXES = (64, 128, 256)
K = 8
ENC_ID = "BAAI/bge-base-en-v1.5"
DEC_ID = "Qwen/Qwen3-0.6B"


class IdeaAE(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        self.etok = AutoTokenizer.from_pretrained(ENC_ID)
        self.enc = AutoModel.from_pretrained(ENC_ID)
        self.to_z = nn.Linear(768, 256)
        self.dtok = AutoTokenizer.from_pretrained(DEC_ID)
        dec = AutoModelForCausalLM.from_pretrained(DEC_ID, torch_dtype=dtype)
        self.dec = get_peft_model(dec, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]))
        hidden = dec.config.hidden_size
        self.to_prefix = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, K * hidden))
        self.hidden = hidden

    def encode(self, texts: list[str]) -> torch.Tensor:
        b = self.etok(texts, return_tensors="pt", padding=True, truncation=True, max_length=96).to("cuda")
        return self.to_z(self.enc(**b).last_hidden_state[:, 0])

    @staticmethod
    def cut(z: torch.Tensor, d: int) -> torch.Tensor:
        out = torch.zeros_like(z)
        out[:, :d] = F.normalize(z[:, :d], dim=-1) * (d ** 0.5)
        return out

    def prefix(self, z: torch.Tensor) -> torch.Tensor:
        emb = self.dec.get_input_embeddings()
        return self.to_prefix(z).view(z.shape[0], K, self.hidden).to(emb.weight.dtype)

    def loss(self, z: torch.Tensor, targets: list[str]) -> torch.Tensor:
        t = self.dtok([s + self.dtok.eos_token for s in targets], return_tensors="pt", padding=True,
                      truncation=True, max_length=56, add_special_tokens=False).to("cuda")
        emb = self.dec.get_input_embeddings()(t["input_ids"])
        inputs = torch.cat([self.prefix(z), emb], dim=1)
        mask = torch.cat([torch.ones(z.shape[0], K, device="cuda", dtype=t["attention_mask"].dtype), t["attention_mask"]], dim=1)
        labels = torch.cat([torch.full((z.shape[0], K), -100, device="cuda"), t["input_ids"].masked_fill(t["attention_mask"] == 0, -100)], dim=1)
        return self.dec(inputs_embeds=inputs, attention_mask=mask, labels=labels).loss

    @torch.no_grad()
    def generate(self, z: torch.Tensor, max_new: int = 64) -> list[str]:
        p = self.prefix(z)
        mask = torch.ones(z.shape[0], K, device="cuda", dtype=torch.long)
        out = self.dec.generate(inputs_embeds=p, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                                eos_token_id=self.dtok.eos_token_id, pad_token_id=self.dtok.pad_token_id or self.dtok.eos_token_id)
        return [s.strip() for s in self.dtok.batch_decode(out, skip_special_tokens=True)]


def batches(rows, size, rng):
    order = list(range(len(rows)))
    rng.shuffle(order)
    for s in range(0, len(order), size):
        yield [rows[i] for i in order[s : s + size]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--data", default="ae/pool.jsonl", help="rows with text and role (fit/val/test)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (effective batch = batch * accum)")
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--out", default="out/ae0")
    args = ap.parse_args()
    random.seed(20260930)
    torch.manual_seed(20260930)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    fit = [r for r in rows if r["role"] == "fit"]
    val = [r for r in rows if r["role"] == "val"][:256]
    test = [r for r in rows if r["role"] == "test"]
    random.Random(1).shuffle(test)
    test = test[: args.eval_n]

    model = IdeaAE(dtype).cuda()
    params = [
        {"params": model.enc.parameters(), "lr": 2e-5},
        {"params": list(model.to_z.parameters()) + list(model.to_prefix.parameters()), "lr": 5e-4},
        {"params": [p for p in model.dec.parameters() if p.requires_grad], "lr": 2e-4},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 5e-4, 2e-4], total_steps=args.steps, pct_start=max(0.05, 5 / args.steps))
    rng = random.Random(7)
    step, t0, log = 0, time.time(), []
    stream = (b for _ in iter(int, 1) for b in batches(fit, args.batch, rng))
    while step < args.steps:
        for _ in [0]:
            model.train()
            opt.zero_grad()
            d = 256 if rng.random() < 0.5 else rng.choice(PREFIXES)
            for _ in range(args.accum):
                b = next(stream)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                    z = model.encode([r["text"] for r in b])
                    l = model.loss(model.cut(z, d), [r["text"] for r in b]) / args.accum
                l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 1000 == 0:
                torch.save({k: v for k, v in model.state_dict().items() if "lora" in k or not k.startswith("dec.")}, out / "model.pt")
            if step % 250 == 0:
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                    vz = model.encode([r["text"] for r in val[:128]])
                    vl = float(model.loss(model.cut(vz, 256), [r["text"] for r in val[:128]]))
                log.append({"step": step, "train_loss": round(float(l) * args.accum, 4), "val_loss_256": round(vl, 4), "minutes": round((time.time() - t0) / 60, 1)})
                print(json.dumps(log[-1]), flush=True)
            if step >= args.steps:
                break
    torch.save({k: v for k, v in model.state_dict().items() if "lora" in k or not k.startswith("dec.")}, out / "model.pt")

    model.eval()
    results = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        for s in range(0, len(test), 32):
            b = test[s : s + 32]
            z = model.encode([r["text"] for r in b])
            decoded = {f"ae{d}": model.generate(model.cut(z, d)) for d in PREFIXES}
            noise = torch.randn_like(z)
            decoded["noise"] = model.generate(model.cut(noise, 256))
            for k, r in enumerate(b):
                results.append({"source": r["text"], **{arm: v[k] for arm, v in decoded.items()}})
    with open(out / "decoded.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "train.json").write_text(json.dumps({"args": vars(args), "dtype": str(dtype), "log": log}, indent=2))
    print("done", len(results), flush=True)


if __name__ == "__main__":
    main()
