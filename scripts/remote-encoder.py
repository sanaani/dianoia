"""GPU worker: train the idea encoder and decoder (SONAR as released, frozen in bf16, + fresh LoRA), then run the round trip.

Training data: ``pairs/arm-<L|B>.jsonl`` from ``build-pairs.py``, grouped by anchor text ``a``. Each step samples
anchor groups; for each, one positive ("same") and one hard negative ("diff") when the group has them. Loss:
- **reconstruction**: the decoder writes each text back from its vector. A stripped or disfluent version is
  decoded to its clean anchor (denoising); everything else is decoded to itself;
- **contrastive** (InfoNCE, temperature 0.05): each anchor must pick its own positive out of every positive and
  hard negative in the batch, and the reverse.

Evaluation (``--eval``, also run with no adapter as the baseline): for every test text, encode, decode greedily,
re-encode. Written per item to ``<out>/eval.jsonl`` for bootstrap intervals on the laptop:
- own-idea rank of the round trip among all test anchors, and against the anchor's own siblings;
- cosine of anchor to each positive and negative (for separation);
- the decoded text, plus NLI by a second model unrelated to the filter (``FacebookAI/roberta-large-mnli``):
  decoded entails original (detail retention) and original entails decoded (meaning kept);
- per-token loss of the decoded text under a small separate LM (``Qwen/Qwen2.5-0.5B``) for fluency.
Geometry ideas (``geometry/ideas.json``): idea, 12 rewordings and negation are encoded for cloud statistics.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer,
                          M2M100ForConditionalGeneration, NllbTokenizer)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder, shift_tokens_right

ENC = ("cointegrated/SONAR_200_text_encoder", "29f2eac73636d1548c2cb227dbfb2772b31ebc4f")
DEC = ("raxtemur/SONAR_200_text_decoder", "278f6ad02ad7f737dfe120d2ccad9a5e6dfad491")
JUDGE = "FacebookAI/roberta-large-mnli"
FLUENCY = "Qwen/Qwen2.5-0.5B"
ATT = ["q_proj", "k_proj", "v_proj", "out_proj"]
DENOISE = {"stripped", "disfluent"}
NATURAL = {"paraphrase", "register", "entail"}  # Sonnet and MRPC paraphrases, Sonnet register shifts, MultiNLI entailments


class SonarAE(torch.nn.Module):
    def __init__(self, r: int, adapters: bool = True):
        super().__init__()
        self.etok = AutoTokenizer.from_pretrained(ENC[0], revision=ENC[1])
        self.etok.src_lang = "eng_Latn"
        enc = M2M100Encoder.from_pretrained(ENC[0], revision=ENC[1], torch_dtype=torch.bfloat16)
        self.dtok = NllbTokenizer.from_pretrained(DEC[0], revision=DEC[1])
        self.dtok.tgt_lang = "eng_Latn"
        dec = M2M100ForConditionalGeneration.from_pretrained(DEC[0], revision=DEC[1], torch_dtype=torch.bfloat16)
        if adapters:
            enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            enc.enable_input_require_grads()
            dec.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            cfg = dict(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=ATT + ["fc1", "fc2"])
            enc, dec = get_peft_model(enc, LoraConfig(**cfg)), get_peft_model(dec, LoraConfig(**cfg))
        self.enc, self.dec = enc, dec
        self.bos = self.dtok.convert_tokens_to_ids("eng_Latn")

    def encode(self, texts: list[str], max_len: int = 96) -> torch.Tensor:
        b = self.etok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)

    def recon(self, v: torch.Tensor, targets: list[str], max_len: int = 96) -> torch.Tensor:
        t = self.dtok(text_target=targets, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
        labels = t["input_ids"].masked_fill(t["attention_mask"] == 0, -100)
        base = self.dec.get_base_model() if hasattr(self.dec, "get_base_model") else self.dec
        start = base.config.decoder_start_token_id
        h = base.model(encoder_outputs=BaseModelOutput(last_hidden_state=v.unsqueeze(1)),
                       decoder_input_ids=shift_tokens_right(labels, self.dtok.pad_token_id, start)).last_hidden_state
        # Project onto the 256k vocabulary only at real tokens, 256 at a time, recomputed in the backward pass, so
        # the full logit table is never held in memory.
        keep = labels != -100
        h, y = h[keep], labels[keep]
        ce = lambda hc, yc: F.cross_entropy(base.lm_head(hc).float(), yc, reduction="sum")
        total = sum(checkpoint(ce, h[s : s + 256], y[s : s + 256], use_reentrant=False) for s in range(0, len(y), 256))
        return total / len(y)

    @torch.no_grad()
    def generate(self, v: torch.Tensor, max_len: int = 96) -> list[str]:
        g = self.dec.generate(encoder_outputs=BaseModelOutput(last_hidden_state=v.unsqueeze(1)), forced_bos_token_id=self.bos,
                              max_length=max_len, num_beams=1, do_sample=False)
        return self.dtok.batch_decode(g, skip_special_tokens=True)


def groups(path: str) -> list[dict]:
    g: dict[str, dict] = defaultdict(lambda: {"pos": [], "neg": [], "source": ""})
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        g[r["a"]]["pos" if r["rel"] == "same" else "neg"].append((r["b"], r["kind"]))
        g[r["a"]]["source"] = r["source"]
    return [{"a": a, **v} for a, v in g.items()]


def train(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = groups(args.data)
    model = SonarAE(args.rank).cuda().train()  # from_pretrained leaves eval mode, which disables checkpointing
    if args.init:  # continue from a trained adapter
        missing = model.load_state_dict(torch.load(args.init, map_location="cuda"), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    print(json.dumps({"groups": len(data), "with_pos": sum(bool(g["pos"]) for g in data), "with_neg": sum(bool(g["neg"]) for g in data)}), flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = random.Random(args.seed)
    t0 = time.time()
    # Natural paraphrases split evenly between Sonnet/MRPC rewordings and MultiNLI entailments, so MultiNLI's
    # larger pool does not crowd out the email paraphrases.
    # With --contrast-n, the attitude-contrast groups get their own slots and are kept out of the paraphrase pool,
    # so the phase-1 and MRPC paraphrases keep the share they had without them.
    is_con = (lambda g: g["source"].startswith("contrast-")) if args.contrast_n else (lambda g: False)
    para = [g for g in data if any(k in ("paraphrase", "register") for _, k in g["pos"]) and not is_con(g)]
    ent = [g for g in data if any(k == "entail" for _, k in g["pos"])]
    has_nat = lambda g: any(k in NATURAL for _, k in g["pos"])
    con = [g for g in data if is_con(g) and has_nat(g)]
    # --lpara-n: slots for literal-idea paraphrase groups (``lpara-*``): different words, same idea, with
    # minimal-edit hard negatives (nearly the same words, a different idea).
    lpara = [g for g in data if g["source"].startswith("lpara") and has_nat(g)] if args.lpara_n else []
    # --spara-n: slots for splitter-paraphrase groups (``spara-*``): the splitter's ideas from an email and from
    # Sonnet's paraphrase of it, with the ideas split from slanted versions as hard negatives.
    spara = [g for g in data if g["source"].startswith("spara") and has_nat(g)] if args.spara_n else []
    # --merge-n: slots for merged-idea groups (``merge-*``): one idea stating two things, positive to each of the two.
    merge = [g for g in data if g["source"].startswith("merge") and has_nat(g)] if args.merge_n else []
    if args.lpara_n or args.spara_n or args.merge_n:
        para = [g for g in para if not g["source"].startswith(("lpara", "spara", "merge"))]
    n_nat = round(args.batch * args.natural_frac)
    print(json.dumps({"paraphrase_groups": len(para), "entail_groups": len(ent), "contrast_groups": len(con),
                      "natural_per_batch": n_nat, "contrast_per_batch": args.contrast_n, "lpara_groups": len(lpara),
                      "lpara_per_batch": args.lpara_n, "spara_groups": len(spara), "spara_per_batch": args.spara_n,
                      "merge_groups": len(merge), "merge_per_batch": args.merge_n}), flush=True)
    for step in range(1, args.steps + 1):
        # A share of each batch comes from ideas with a natural paraphrase (different words, same meaning), which
        # then serves as the positive; the rest is drawn uniformly from all ideas.
        nat = (rng.sample(para, n_nat - n_nat // 2) + rng.sample(ent, n_nat // 2)) if n_nat else []
        nat += rng.sample(con, args.contrast_n) if args.contrast_n else []
        nat += rng.sample(lpara, args.lpara_n) if args.lpara_n else []
        nat += rng.sample(spara, args.spara_n) if args.spara_n else []
        nat += rng.sample(merge, args.merge_n) if args.merge_n else []
        nat = list({g["a"]: g for g in nat}.values())
        n_take = len(nat) if args.contrast_n or args.lpara_n or args.spara_n or args.merge_n else n_nat
        seen = {g["a"] for g in nat}
        rest = [g for g in rng.sample(data, args.batch + n_take) if g["a"] not in seen][: max(args.batch - n_take, 0)]
        b = [(g, True) for g in nat] + [(g, False) for g in rest]
        anchors, pos, pos_t, neg = [], [], [], []
        for g, is_nat in b:
            anchors.append(g["a"])
            if is_nat:
                p, k = rng.choice([x for x in g["pos"] if x[1] in NATURAL])
            elif g["pos"]:
                p, k = rng.choice(g["pos"])
            else:  # no human positive: the anchor's own stripped form
                p, k = " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in g["a"].lower()).split()), "stripped"
            pos.append(p)
            pos_t.append(g["a"] if k in DENOISE else p)
            if g["neg"]:
                neg.append(rng.choice(g["neg"])[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            va = model.encode(anchors)
            vp = model.encode(pos)
            vn = model.encode(neg) if neg else va[:0]
            rec = (model.recon(va, anchors) + model.recon(vp, pos_t) + (model.recon(vn, neg) if neg else 0)) / (3 if neg else 2)
        za, zp, zn = (F.normalize(x.float(), dim=-1) for x in (va, vp, vn))
        cand = torch.cat([zp, zn])
        lab = torch.arange(len(za), device="cuda")
        nce = (F.cross_entropy(za @ cand.T / args.temp, lab) + F.cross_entropy(zp @ za.T / args.temp, lab)) / 2
        loss = rec + args.nce_weight * nce
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "rec": round(float(rec), 4), "nce": round(float(nce), 4),
                              "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    torch.save({k: v for k, v in model.state_dict().items() if "lora" in k}, out / "lora.pt")
    print(json.dumps({"trained": args.steps, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)


def batched(xs, n):
    for s in range(0, len(xs), n):
        yield xs[s : s + n]


def evaluate(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapters = args.adapter is not None
    model = SonarAE(args.rank, adapters=adapters).cuda()
    if adapters:
        sd = torch.load(args.adapter, map_location="cuda")
        missing = model.load_state_dict(sd, strict=False)
        assert not [k for k in missing.unexpected_keys], missing.unexpected_keys
    model.eval()
    test = groups(args.test)
    rng = random.Random(5)
    by_src: dict[str, list] = defaultdict(list)
    for line in open(args.test, encoding="utf-8"):
        r = json.loads(line)
        by_src[r["source"]].append(r["a"])
    keep = set()
    for src, anchors in by_src.items():
        u = sorted(set(anchors))
        keep |= set(rng.sample(u, min(args.per_source, len(u))))
    test = [g for g in test if g["a"] in keep]
    texts = sorted({g["a"] for g in test} | {t for g in test for t, _ in g["pos"] + g["neg"]})
    geo = json.load(open(args.geometry, encoding="utf-8"))["ideas"]
    geo_texts = sorted({t for o in geo for t in [o["idea"], *(o["rewordings"] or []), o["negation"] or ""] if t})
    vec: dict[str, torch.Tensor] = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in batched(sorted(set(texts) | set(geo_texts), key=len), 128):
            for t, v in zip(b, model.encode(b).float()):
                vec[t] = v
        anchors = sorted({g["a"] for g in test}, key=len)
        # Round trips start from every anchor, every other wording of it, and every geometry rewording.
        starts = sorted(set(anchors) | {t for g in test for t, _ in g["pos"]}
                        | {t for o in geo for t in (o["rewordings"] or [])}, key=len)
        dec = {}
        for b in batched(starts, 64):
            for t, d in zip(b, model.generate(torch.stack([vec[t] for t in b]))):
                dec[t] = d
        back = {}
        for b in batched(starts, 128):
            for t, v in zip(b, model.encode([dec[t] for t in b]).float()):
                back[t] = v
    A = F.normalize(torch.stack([vec[t] for t in anchors]), dim=-1)
    idx = {t: i for i, t in enumerate(anchors)}
    del model
    torch.cuda.empty_cache()
    jtok = AutoTokenizer.from_pretrained(JUDGE)
    judge = AutoModelForSequenceClassification.from_pretrained(JUDGE, torch_dtype=torch.float16).cuda().eval()
    ent = {v.lower(): int(k) for k, v in judge.config.id2label.items()}["entailment"]

    def nli(pairs):
        res = []
        with torch.no_grad():
            for b in batched(pairs, 64):
                x = jtok([p.lower() for p, _ in b], [h.lower() for _, h in b], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
                res += judge(**x).logits.float().softmax(-1)[:, ent].tolist()
        return res

    retain = nli([(dec[t], t) for t in anchors])
    kept = nli([(t, dec[t]) for t in anchors])
    cross = [(g["a"], p, k) for g in test for p, k in g["pos"]]
    c_retain = nli([(dec[p], a) for a, p, _ in cross])
    c_kept = nli([(a, dec[p]) for a, p, _ in cross])
    cross_nli = {(a, p): (r_, k_) for (a, p, _), r_, k_ in zip(cross, c_retain, c_kept)}
    del judge
    torch.cuda.empty_cache()
    ftok = AutoTokenizer.from_pretrained(FLUENCY)
    flm = AutoModelForCausalLM.from_pretrained(FLUENCY, torch_dtype=torch.bfloat16).cuda().eval()

    def nll(ts):
        res = []
        with torch.no_grad():
            for t in ts:
                x = ftok(t or ".", return_tensors="pt").to("cuda")
                res.append(float(flm(**x, labels=x["input_ids"]).loss) if x["input_ids"].shape[1] > 1 else 10.0)
        return res

    flu_dec = nll([dec[t] for t in anchors])
    flu_orig = nll(anchors)
    flu_cross = dict(zip([p for _, p, _ in cross], nll([dec[p] for _, p, _ in cross])))
    src = {}
    for line in open(args.test, encoding="utf-8"):
        r = json.loads(line)
        src.setdefault(r["a"], r["source"])
    with open(out / "eval.jsonl", "w", encoding="utf-8") as f:
        for g in test:
            t = g["a"]
            i = idx[t]
            zb = F.normalize(back[t], dim=-1)
            sims = A @ zb
            rank = int((sims > sims[i]).sum())
            za = A[i]
            sib = [(s, k, float(F.normalize(vec[s], dim=-1) @ zb), float(F.normalize(vec[s], dim=-1) @ za)) for s, k in g["neg"]]
            pos = [(s, k, float(F.normalize(vec[s], dim=-1) @ za)) for s, k in g["pos"]]
            # Round trip from another wording p: does it come back nearer this idea than any sibling and any
            # other test anchor, and does its decoded text still carry the idea?
            crs = []
            for p, k in g["pos"]:
                zp = F.normalize(back[p], dim=-1)
                s_all = A @ zp
                crs.append({"kind": k, "text": p, "decoded": dec[p], "rank_all": int((s_all > s_all[i]).sum()),
                            "to_anchor": float(za @ zp),
                            "to_siblings": [float(F.normalize(vec[s], dim=-1) @ zp) for s, _ in g["neg"]],
                            "retain": cross_nli[(t, p)][0], "kept": cross_nli[(t, p)][1], "nll_dec": flu_cross[p]})
            f.write(json.dumps({"a": t, "source": src[t], "decoded": dec[t], "rank_all": rank, "pool": len(anchors),
                                "self_back": float(za @ zb), "siblings": [{"kind": k, "back": b_, "anchor": a_} for _, k, b_, a_ in sib],
                                "positives": [{"kind": k, "anchor": a_} for _, k, a_ in pos], "cross": crs,
                                "retain": retain[i], "kept": kept[i], "nll_dec": flu_dec[i], "nll_orig": flu_orig[i]},
                               ensure_ascii=False) + "\n")
    G = []
    for o in geo:
        if not o["rewordings"]:
            continue
        z = lambda t: F.normalize(vec[t], dim=-1)
        rw = torch.stack([z(t) for t in o["rewordings"]])
        c = F.normalize(rw.mean(0), dim=-1)
        G.append({"role": o["role"], "radius": float((1 - rw @ c).median()), "idea_to_centre": float(1 - z(o["idea"]) @ c),
                  "negation_to_centre": float(1 - z(o["negation"]) @ c) if o["negation"] else None,
                  "rewordings_closer_than_negation": float(((1 - rw @ c) < (1 - z(o["negation"]) @ c)).float().mean()) if o["negation"] else None,
                  # round trip from each rewording comes back nearer the idea than its negation
                  "roundtrip_beats_negation": float(sum(float(F.normalize(back[t], dim=-1) @ z(o["idea"])) > float(F.normalize(back[t], dim=-1) @ z(o["negation"]))
                                                        for t in o["rewordings"]) / len(o["rewordings"])) if o["negation"] else None})
    (out / "geometry.json").write_text(json.dumps(G, indent=1), encoding="utf-8")
    print(json.dumps({"evaluated": len(test), "geometry": len(G)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--data")
    ap.add_argument("--test", default="pairs/test.jsonl")
    ap.add_argument("--geometry", default="geometry/ideas.json")
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--nce-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--natural-frac", type=float, default=0.0, help="share of each batch drawn from ideas with a natural paraphrase")
    ap.add_argument("--contrast-n", type=int, default=0, help="slots per batch for attitude-contrast groups, kept out of the paraphrase pool")
    ap.add_argument("--lpara-n", type=int, default=0, help="slots per batch for literal-idea paraphrase groups")
    ap.add_argument("--spara-n", type=int, default=0, help="slots per batch for splitter-paraphrase groups")
    ap.add_argument("--merge-n", type=int, default=0, help="slots per batch for merged-idea groups")
    ap.add_argument("--init", help="adapter to continue training from")
    ap.add_argument("--per-source", type=int, default=1500)
    args = ap.parse_args()
    train(args) if args.mode == "train" else evaluate(args)


if __name__ == "__main__":
    main()
