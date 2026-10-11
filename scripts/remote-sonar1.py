"""GPU worker, stage 1 on SONAR: teach a working sentence autoencoder to keep intent and drop wording.

SONAR's text encoder and decoder already rebuild sentences (phase 2: 99% within tolerance decoding their
own vectors), so stage 0 is passed by construction. Both get LoRA adapters (r=16) and are trained so that:

- every input (a canonicalized pool sentence, a phase-1 fit item's source, or any of its keep
  variants) decodes to that sentence's **canonical** form, through the 1024-d SONAR vector;
- the phase-1 contrastive loss (``head.loss`` on the first 256 dims of the normalised vector, weight
  ``--lam``) keeps move variants (negation, modality, act, roles, verb/object, type, quantity) apart.

Evaluation on phase-1 test items: decode the source's vector greedily (beam 4), restore names from the
source's bindings, re-encode the decode, and record its cosine to the source's vector beside the
paraphrase tolerance tau = cos(v(source), v(paraphrase)).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, M2M100ForConditionalGeneration, NllbTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

from head import loss as contrastive_loss

ENC = ("cointegrated/SONAR_200_text_encoder", "29f2eac73636d1548c2cb227dbfb2772b31ebc4f")
DEC = ("raxtemur/SONAR_200_text_decoder", "278f6ad02ad7f737dfe120d2ccad9a5e6dfad491")
KEEP = ["paraphrase", "rename", "register", "drop_modifiers"]
MOVE = ["act_change", "role_swap", "negation", "modality", "predicate_change", "type_change", "quantity"]
PLACEHOLDER = re.compile(r"\[[A-Z]+_\d+\]")
ATT = ["q_proj", "k_proj", "v_proj", "out_proj"]


class SonarAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.etok = AutoTokenizer.from_pretrained(ENC[0], revision=ENC[1])
        self.etok.src_lang = "eng_Latn"
        enc = M2M100Encoder.from_pretrained(ENC[0], revision=ENC[1])
        enc.gradient_checkpointing_enable()
        enc.enable_input_require_grads()
        self.enc = get_peft_model(enc, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=ATT + ["fc1", "fc2"]))
        self.dtok = NllbTokenizer.from_pretrained(DEC[0], revision=DEC[1])
        dec = M2M100ForConditionalGeneration.from_pretrained(DEC[0], revision=DEC[1])
        dec.gradient_checkpointing_enable()
        self.dec = get_peft_model(dec, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=ATT + ["fc1", "fc2"]))
        self.bos = self.dtok.convert_tokens_to_ids("eng_Latn")

    def encode(self, texts: list[str]) -> torch.Tensor:
        b = self.etok(texts, return_tensors="pt", padding=True, truncation=True, max_length=80).to("cuda")
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)

    def loss(self, v: torch.Tensor, targets: list[str]) -> torch.Tensor:
        self.dtok.tgt_lang = "eng_Latn"
        t = self.dtok(text_target=targets, return_tensors="pt", padding=True, truncation=True, max_length=64).to("cuda")
        labels = t["input_ids"].masked_fill(t["attention_mask"] == 0, -100)
        return self.dec(encoder_outputs=BaseModelOutput(last_hidden_state=v.unsqueeze(1)), labels=labels).loss

    @torch.no_grad()
    def generate(self, v: torch.Tensor) -> list[str]:
        g = self.dec.generate(encoder_outputs=BaseModelOutput(last_hidden_state=v.unsqueeze(1)), forced_bos_token_id=self.bos,
                              max_length=96, num_beams=4)
        return self.dtok.batch_decode(g, skip_special_tokens=True)


def restore(text: str, bindings: dict) -> tuple[str, int]:
    missing = 0

    def sub(m):
        nonlocal missing
        if m.group(0) in bindings:
            return bindings[m.group(0)]
        missing += 1
        return m.group(0)

    return PLACEHOLDER.sub(sub, text), missing


def evaluate(model: SonarAE, items: list[dict], out: Path, tag: str) -> None:
    model.eval()
    rows = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(items), 32):
            b = items[s : s + 32]
            v = model.encode([r["text"] for r in b]).float()
            vp = model.encode([r.get("paraphrase") or r["text"] for r in b]).float()
            dec = model.generate(v)
            restored = [restore(d, json.loads(r.get("bindings") or "{}")) for d, r in zip(dec, b)]
            vd = model.encode([x or "." for x, _ in restored]).float()
            for k, r in enumerate(b):
                rows.append({"source": r["text"], "canonical": r.get("canonical"), "ae1024": restored[k][0], "ae1024_raw": dec[k],
                             "ae1024_unbound": restored[k][1], "ae1024_cos": float(F.cosine_similarity(v[k : k + 1], vd[k : k + 1])),
                             "tau": float(F.cosine_similarity(v[k : k + 1], vp[k : k + 1])) if r.get("paraphrase") else None})
    with open(out / f"decoded-{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Invariance: a keep variant (different words, same idea) should decode to what the source decodes to.
    # Written as source = the source's decode, ae1024 = the variant's decode, so the same Sonnet intent check
    # scores whether the two decodes say the same thing.
    inv = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for kind in ("paraphrase", "rename", "register"):
            have = [(r, row) for r, row in zip(items, rows) if r.get(kind)]
            for s0 in range(0, len(have), 32):
                b = have[s0 : s0 + 32]
                dec = model.generate(model.encode([r[kind] for r, _ in b]).float())
                for (r, row), d in zip(b, dec):
                    inv.append({"kind": kind, "input": r[kind], "original": r["text"], "source": row["ae1024_raw"], "ae1024": d})
    with open(out / f"invariance-{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in inv:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default="ae/canonical.jsonl")
    ap.add_argument("--items", default="ae/items-v2.jsonl", help="phase-1 items with Sonnet canonical forms")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--con-items", type=int, default=6)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--out", default="out/sonar1")
    args = ap.parse_args()
    random.seed(20261001)
    torch.manual_seed(20261001)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    canon = [json.loads(line) for line in open(args.canonical, encoding="utf-8")]
    items = [json.loads(line) for line in open(args.items, encoding="utf-8")]
    fit = [r for r in items if r["role"] == "fit" and r.get("canonical")]
    test = [r for r in items if r["role"] == "test" and r.get("canonical")]
    pairs = [(r["text"], r["canonical"]) for r in canon]
    for r in fit:
        pairs += [(r[k], r["canonical"]) for k in ["text", *KEEP] if r.get(k)]
    print(json.dumps({"pairs": len(pairs), "fit_items": len(fit), "test_items": len(test)}), flush=True)

    model = SonarAE().cuda()
    evaluate(model, test, out, "step0")
    print(json.dumps({"eval_written": 0}), flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=args.steps, pct_start=0.05)
    rng = random.Random(11)
    t0 = time.time()
    fields = ["text", *KEEP, *MOVE]
    for step in range(1, args.steps + 1):
        b = rng.sample(pairs, args.batch)
        ib = rng.sample(fit, args.con_items)
        texts, where = [], {}
        for f in fields:
            for i, r in enumerate(ib):
                if r.get(f):
                    where[(f, i)] = len(texts)
                    texts.append(r[f])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l_dec = model.loss(model.encode([x for x, _ in b]), [y for _, y in b])
            zz = F.normalize(model.encode(texts).float()[:, :256], dim=-1)
            get = {f: torch.stack([zz[where[(f, i)]] if (f, i) in where else torch.zeros(256, device="cuda") for i in range(len(ib))]) for f in fields}
            msk = {f: torch.tensor([(f, i) in where for i in range(len(ib))], device="cuda") for f in fields}
            l_con = contrastive_loss(get["text"], {k: get[k] for k in KEEP}, {k: msk[k] for k in KEEP},
                                     torch.stack([get[m] for m in MOVE], 1), torch.stack([msk[m] for m in MOVE], 1))
            l = l_dec + args.lam * l_con
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            print(json.dumps({"step": step, "dec_loss": round(float(l_dec), 4), "con_loss": round(float(l_con), 4), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
        if step % args.eval_every == 0:
            evaluate(model, test, out, f"step{step}")
            torch.save({k: v for k, v in model.state_dict().items() if "lora" in k}, out / "lora.pt")
            print(json.dumps({"eval_written": step}), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
