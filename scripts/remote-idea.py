"""GPU worker, stage 1 on idea objects: SONAR reads any wording of a unit and writes its idea object.

SONAR's encoder and decoder get LoRA adapters (r=16), as in ``remote-sonar1.py``. The target is no longer a
sentence but the unit's idea object, serialized (``serialize``), extracted once by Sonnet from the original
text (``idea-targets.py``). Every training step draws, for each unit, one input version: the original, or a
mechanical variant made here (stripped of punctuation and capitals, disfluent, split across turns), all
mapped to the same target through the 1024-d vector. That is how invariance is taught: the decoder cannot
know which version it was given.

Evaluation on the check's units (``objects.jsonl``, never trained on): decode from the original and from each
of its versions (paraphrase, stripped, disfluent, split, move); the laptop parses the decodes and compares
them with Sonnet's objects (``score-idea.py``).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, M2M100ForConditionalGeneration, NllbTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

ENC = ("cointegrated/SONAR_200_text_encoder", "29f2eac73636d1548c2cb227dbfb2772b31ebc4f")
DEC = ("raxtemur/SONAR_200_text_decoder", "278f6ad02ad7f737dfe120d2ccad9a5e6dfad491")
ATT = ["q_proj", "k_proj", "v_proj", "out_proj"]
FILLERS = ["um", "uh", "like", "you know", "i mean", "so", "kind of"]
EVAL_KINDS = ["original", "paraphrase", "stripped", "disfluent", "split", "move"]


MOD_COARSE = {"none": "none", "certain": "none", "possible": "uncertain", "likely": "uncertain", "necessary": "must",
              "obligated": "must", "able": "can", "permitted": "can", "intend": "will", "want": "will"}


def serialize(obj: dict) -> str:
    """Events in turn order, renumbered 1..n: ``t<turn> act pred [arg0 ; arg1 ; arg2] neg mod time``, then
    ``* center``. Relations are not included and modality is coarsened (PREREGISTRATION, 10/01/2026, after
    the idea-object check)."""
    obj = {**obj, "relations": [], "events": [{**e, "mod": MOD_COARSE.get(e["mod"], e["mod"])} for e in obj["events"]]}
    evs = sorted(obj["events"], key=lambda e: e["turn"])
    num = {e["id"]: k + 1 for k, e in enumerate(evs)}
    parts = [f"t{e['turn'] + 1} {e['act']} {e['pred']} [{e['arg0']} ; {e['arg1']} ; {e['arg2']}] {'not' if e['neg'] else 'yes'} "
             f"{e['mod']} {e['time']}" for e in evs]
    parts += [f">> {num[r['from']]} {num[r['to']]} {r['label']}" for r in obj["relations"]]
    if obj.get("center") in num:
        parts.append(f"* {num[obj['center']]}")
    return " | ".join(parts)


def render(turns) -> str:
    return "\n".join(f"{s}: {t}" for s, t in turns)


def stripped(turns):
    return [[s, re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip() or "mm"] for s, t in turns]


def disfluent(turns, rng):
    out = []
    for s, t in turns:
        ws, new = t.split(), []
        if len(ws) >= 6 and rng.random() < 0.4:
            new += ws[:2] + ["--"]
        for w in ws:
            if rng.random() < 0.15:
                new.append(rng.choice(FILLERS) + ",")
            new.append(w)
            if rng.random() < 0.05:
                new.append(w)
        out.append([s, " ".join(new)])
    return out


def split(turns, rng):
    k = rng.randrange(len(turns))
    ws = turns[k][1].split()
    if len(ws) < 4:
        return turns
    h = rng.randrange(2, len(ws) - 1)
    return turns[:k] + [[turns[k][0], " ".join(ws[:h])], [turns[k][0], " ".join(ws[h:])]] + turns[k + 1 :]


def version(turns, rng):
    r = rng.random()
    if r < 0.4:
        return turns
    if r < 0.6:
        return stripped(turns)
    if r < 0.8:
        return disfluent(turns, rng)
    if r < 0.9:
        return split(turns, rng)
    return stripped(disfluent(split(turns, rng), rng))


class SonarAE(torch.nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.etok = AutoTokenizer.from_pretrained(ENC[0], revision=ENC[1])
        self.etok.src_lang = "eng_Latn"
        enc = M2M100Encoder.from_pretrained(ENC[0], revision=ENC[1])
        enc.gradient_checkpointing_enable()
        enc.enable_input_require_grads()
        self.enc = get_peft_model(enc, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=ATT + ["fc1", "fc2"]))
        self.dtok = NllbTokenizer.from_pretrained(DEC[0], revision=DEC[1])
        dec = M2M100ForConditionalGeneration.from_pretrained(DEC[0], revision=DEC[1])
        dec.gradient_checkpointing_enable()
        self.dec = get_peft_model(dec, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=ATT + ["fc1", "fc2"]))
        self.bos = self.dtok.convert_tokens_to_ids("eng_Latn")

    def encode(self, texts: list[str], max_len: int) -> torch.Tensor:
        b = self.etok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)

    def loss(self, v: torch.Tensor, targets: list[str], max_len: int) -> torch.Tensor:
        self.dtok.tgt_lang = "eng_Latn"
        t = self.dtok(text_target=targets, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
        labels = t["input_ids"].masked_fill(t["attention_mask"] == 0, -100)
        return self.dec(encoder_outputs=BaseModelOutput(last_hidden_state=v.unsqueeze(1)), labels=labels).loss

    @torch.no_grad()
    def generate(self, v: torch.Tensor, max_len: int) -> list[str]:
        g = self.dec.generate(encoder_outputs=BaseModelOutput(last_hidden_state=v.unsqueeze(1)), forced_bos_token_id=self.bos,
                              max_length=max_len, num_beams=2)
        return self.dtok.batch_decode(g, skip_special_tokens=True)


def evaluate(model, rows, out: Path, tag: str, args) -> None:
    model.eval()
    res = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(rows), 16):
            b = rows[s : s + 16]
            dec = model.generate(model.encode([render(r["turns"]) for r in b], args.enc_len).float(), args.dec_len)
            for r, d in zip(b, dec):
                res.append({"uid": r["uid"], "medium": r["medium"], "kind": r["kind"], "decoded": d})
    with open(out / f"decoded-{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="ideas/targets.jsonl")
    ap.add_argument("--heldout", default="ideas/objects.jsonl")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--enc-len", type=int, default=256)
    ap.add_argument("--dec-len", type=int, default=320)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--eval-n", type=int, default=0, help="held-out units per eval (0 = all)")
    ap.add_argument("--out", default="out/idea1")
    args = ap.parse_args()
    random.seed(20261002)
    torch.manual_seed(20261002)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = [json.loads(line) for line in open(args.targets, encoding="utf-8")]
    train = [(r["turns"], serialize(r["object"])) for r in train if r.get("object")]
    held = [json.loads(line) for line in open(args.heldout, encoding="utf-8")]
    uids = sorted({r["uid"] for r in held})
    random.Random(3).shuffle(uids)
    keep = set(uids[: args.eval_n]) if args.eval_n else set(uids)
    held = [r for r in held if r["uid"] in keep and r["kind"] in EVAL_KINDS]
    model = SonarAE(args.rank).cuda()
    lens = [len(model.dtok(t)["input_ids"]) for _, t in train[:500]]
    print(json.dumps({"train": len(train), "heldout_versions": len(held), "target_tokens_p50": sorted(lens)[250],
                      "target_tokens_p95": sorted(lens)[475], "over_dec_len": sum(x > args.dec_len for x in lens) / len(lens)}), flush=True)
    evaluate(model, [r for r in held if r["kind"] == "original"][:64], out, "step0", args)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = random.Random(11)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        tot = 0.0
        for _ in range(args.accum):
            b = rng.sample(train, args.batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = model.loss(model.encode([render(version(t, rng)) for t, _ in b], args.enc_len), [y for _, y in b], args.dec_len)
            (l / args.accum).backward()
            tot += float(l) / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": round(tot, 4), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            torch.save({k: v for k, v in model.state_dict().items() if "lora" in k}, out / "lora.pt")
            evaluate(model, held, out, f"step{step}", args)
            print(json.dumps({"eval_written": step}), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
