"""GPU worker: our own encoder-decoder, stage 0 at scale (rebuild the sentence through the idea vector).

The first stage-0 run (``remote-ae.py``: bge-base, Qwen3-0.6B, 8 soft tokens, 256-d, 121k sentences)
failed its gate at 25%. This is the scaled rerun the 10/01/2026 addendum records.

Encoder (``--enc``, default bge-large): all weights trained; CLS -> Linear(hidden, ``--dim``) = z.
Decoder (``--dec``, default Qwen3-1.7B): LoRA r=32, reads z as ``--k`` soft prefix tokens through an MLP.
Loss: token cross entropy on the target sentence only.

Matryoshka: every batch cuts z to a random nested prefix (64 ... ``--dim``, the full width 40% of the
time) and zeroes the rest, so one model decodes every truncation.

Data: each batch is half Enron (``--enron``) and half general English (``--general``), so Enron is not
drowned out. Held-out evaluation sentences are ``--heldout`` rows with role "test".

Every ``--eval-every`` steps: save a checkpoint and decode ``--eval-n`` held-out sentences at 256 and at
the full width (``decoded-step<N>.jsonl``); the laptop grades these while training continues. The last
step decodes ``--final-n`` sentences at every prefix plus a random-vector noise floor.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ALL_PREFIXES = (64, 128, 256, 512, 1024)


def read_jsonl(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


class IdeaAE(nn.Module):
    def __init__(self, enc_id: str, dec_id: str, dim: int, k: int, dtype, lora_r: int = 32):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        self.dim, self.k = dim, k
        self.prefixes = tuple(p for p in ALL_PREFIXES if p <= dim)
        self.llm_enc = "qwen" in enc_id.lower()
        self.etok = AutoTokenizer.from_pretrained(enc_id)
        if self.llm_enc:
            # ICAE-style: the decoder's own language model reads the sentence (its own LoRA) and
            # summarises it at the final token, so encoder and decoder share one representation.
            self.etok.padding_side = "left"
            enc = AutoModel.from_pretrained(enc_id, torch_dtype=dtype)
            enc.gradient_checkpointing_enable()
            enc.enable_input_require_grads()
            self.enc = get_peft_model(enc, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]))
        else:
            self.enc = AutoModel.from_pretrained(enc_id)
        self.to_z = nn.Linear(self.enc.config.hidden_size, dim)
        self.dtok = AutoTokenizer.from_pretrained(dec_id)
        dec = AutoModelForCausalLM.from_pretrained(dec_id, torch_dtype=dtype)
        dec.gradient_checkpointing_enable()
        dec.enable_input_require_grads()
        self.dec = get_peft_model(dec, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]))
        self.hidden = dec.config.hidden_size
        self.to_prefix = nn.Sequential(nn.Linear(dim, 2048), nn.GELU(), nn.Linear(2048, k * self.hidden))

    def encode(self, texts: list[str]) -> torch.Tensor:
        if self.llm_enc:
            b = self.etok([t + self.etok.eos_token for t in texts], return_tensors="pt", padding=True, truncation=True,
                          max_length=96, add_special_tokens=False).to("cuda")
            return self.to_z(self.enc(**b).last_hidden_state[:, -1].float())
        b = self.etok(texts, return_tensors="pt", padding=True, truncation=True, max_length=96).to("cuda")
        return self.to_z(self.enc(**b).last_hidden_state[:, 0])

    def cut(self, z: torch.Tensor, d: int) -> torch.Tensor:
        out = torch.zeros_like(z)
        out[:, :d] = F.normalize(z[:, :d], dim=-1) * (d ** 0.5)
        return out

    def prefix(self, z: torch.Tensor) -> torch.Tensor:
        emb = self.dec.get_input_embeddings()
        return self.to_prefix(z).view(z.shape[0], self.k, self.hidden).to(emb.weight.dtype)

    def loss(self, z: torch.Tensor, targets: list[str]) -> torch.Tensor:
        t = self.dtok([s + self.dtok.eos_token for s in targets], return_tensors="pt", padding=True,
                      truncation=True, max_length=64, add_special_tokens=False).to("cuda")
        emb = self.dec.get_input_embeddings()(t["input_ids"])
        n = z.shape[0]
        inputs = torch.cat([self.prefix(z), emb], dim=1)
        mask = torch.cat([torch.ones(n, self.k, device="cuda", dtype=t["attention_mask"].dtype), t["attention_mask"]], dim=1)
        labels = torch.cat([torch.full((n, self.k), -100, device="cuda"), t["input_ids"].masked_fill(t["attention_mask"] == 0, -100)], dim=1)
        return self.dec(inputs_embeds=inputs, attention_mask=mask, labels=labels).loss

    @torch.no_grad()
    def generate(self, z: torch.Tensor, max_new: int = 64) -> list[str]:
        p = self.prefix(z)
        mask = torch.ones(z.shape[0], self.k, device="cuda", dtype=torch.long)
        out = self.dec.generate(inputs_embeds=p, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                                eos_token_id=self.dtok.eos_token_id, pad_token_id=self.dtok.pad_token_id or self.dtok.eos_token_id)
        return [s.strip() for s in self.dtok.batch_decode(out, skip_special_tokens=True)]

    def trainable_state(self) -> dict:
        return {k: v for k, v in self.state_dict().items()
                if "lora" in k or not (k.startswith("dec.") or (self.llm_enc and k.startswith("enc.")))}


def evaluate(model: IdeaAE, rows: list[dict], dims: list[int], dtype, noise: bool) -> list[dict]:
    model.eval()
    results = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        for s in range(0, len(rows), 32):
            b = rows[s : s + 32]
            z = model.encode([r["text"] for r in b])
            arms = {f"ae{d}": model.generate(model.cut(z, d)) for d in dims}
            if noise:
                arms["noise"] = model.generate(model.cut(torch.randn_like(z), model.dim))
            for k, r in enumerate(b):
                results.append({"source": r["text"], **{a: v[k] for a, v in arms.items()}})
    model.train()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--dec", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--enron", default="ae/enron-fit.jsonl.gz")
    ap.add_argument("--general", default="ae/wiki.jsonl.gz")
    ap.add_argument("--heldout", default="ae/pool.jsonl")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--final-n", type=int, default=500)
    ap.add_argument("--resume", default="")
    ap.add_argument("--out", default="out/ae0x")
    args = ap.parse_args()
    random.seed(20261001)
    torch.manual_seed(20261001)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    enron = [r["text"] for r in read_jsonl(args.enron)]
    general = [r["text"] for r in read_jsonl(args.general)] if args.general and Path(args.general).exists() else []
    held = [r for r in read_jsonl(args.heldout) if r["role"] == "test"]
    random.Random(1).shuffle(held)
    eval_rows, final_rows = held[: args.eval_n], held[: args.final_n]
    print(json.dumps({"enron": len(enron), "general": len(general), "heldout": len(held)}), flush=True)

    model = IdeaAE(args.enc, args.dec, args.dim, args.k, dtype).cuda()
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(ck["model"], strict=False)
        start = ck["step"]
    params = [
        {"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": 2e-4 if model.llm_enc else 1e-5},
        {"params": list(model.to_z.parameters()) + list(model.to_prefix.parameters()), "lr": 3e-4},
        {"params": [p for p in model.dec.parameters() if p.requires_grad], "lr": 2e-4},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-4 if model.llm_enc else 1e-5, 3e-4, 2e-4], total_steps=args.steps, pct_start=0.03)
    for _ in range(start):
        sched.step()
    rng = random.Random(7 + start)
    log, t0 = [], time.time()
    model.train()
    for step in range(start + 1, args.steps + 1):
        half = args.batch // 2 if general else args.batch
        texts = rng.sample(enron, half) + (rng.sample(general, args.batch - half) if general else [])
        d = model.dim if rng.random() < 0.4 else rng.choice(model.prefixes)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            l = model.loss(model.cut(model.encode(texts), d), texts)
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            log.append({"step": step, "loss": round(float(l), 4), "d": d, "minutes": round((time.time() - t0) / 60, 1)})
            print(json.dumps(log[-1]), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            torch.save({"model": model.trainable_state(), "step": step, "args": vars(args)}, out / "checkpoint.pt")
            last = step == args.steps
            dims = list(model.prefixes) if last else sorted({256, model.dim})
            rows = evaluate(model, final_rows if last else eval_rows, dims, dtype, noise=last)
            with open(out / f"decoded-step{step}.jsonl", "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(json.dumps({"eval_written": step}), flush=True)
    (out / "train.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
