"""GPU worker for the idea splitter: predict, fine-tune and score.

    predict  --model M --out preds.jsonl      split every eval version (original, stripped, disfluent)
    train    --out DIR                        fine-tune Propositionizer on Sonnet's splits (any version -> the
                                              original's ideas, so the splitter learns to ignore wording noise)
    score    --preds preds.jsonl --out s.json  the accuracy function below

The accuracy function, after Hosseini et al. 2024 (abstractive proposition segmentation), with an NLI model
(DeBERTa-v3-large, MNLI/FEVER/ANLI/LING/WANLI, MIT) giving P(entailment):

- faithfulness: mean over the splitter's ideas of P(source text entails idea);
- coverage:     mean over Sonnet's reference ideas of P(the splitter's ideas, together, entail the reference idea);
- invariance:   content agreement between the ideas split from the original and from each of its stripped and
                disfluent versions: each side's ideas, together, entail every idea of the other, averaged both
                ways (amended 10/02/2026, see PREREGISTRATION.md); the stricter one-to-one match (BiNLI F1, min
                of both directions) is reported as unit invariance. All text is lowercased before NLI;
- rewording (10/03/2026, reported only): the same content agreement for a paraphrase (inv_paraphrase), the
                same text split into different turns (inv_split), and, as the control that must score LOWER, a
                version whose meaning was changed (inv_move);
- segmentation (10/04/2026, reported only): count_<version>, the share of units where the version is cut into
                as many ideas as the original;
- literalness:  share of ideas with no unresolved pronoun (constraint: >= 0.95);
- **accuracy = harmonic mean of faithfulness, coverage and invariance.**
Atomic agreement with Sonnet's references (reference-based BiNLI F1) is reported beside it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

PROP = "chentong00/propositionizer-wiki-flan-t5-large"
NLI = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
PRONOUN = re.compile(r"\b(he|she|it|its|they|them|their|theirs|his|her|hers|him|this|these|those)\b", re.I)
# Not references (amended 10/02/2026): time determiners ("this morning", "these days") and dummy "it" ("it rains",
# "it is possible", "whether it is ok to").
TIME = (r"(morning|afternoon|evening|night|week|weekend|month|year|quarter|time|days|weeks|months|years|moment|"
        r"summer|winter|spring|fall|season|monday|tuesday|wednesday|thursday|friday|saturday|sunday)")
NOT_REF = re.compile(rf"\b(this|these|those) (past |coming |next |last )?{TIME}\b|"
                     r"\bit (rains|snows|seems|appears|takes|is (possible|ok|okay|likely|unlikely|necessary|important|easier|"
                     r"harder|better|worse|true|time|too late|hard|easy)|would be (possible|ok|okay|great|good|helpful|best))\b", re.I)


def unresolved(idea: str) -> bool:
    return bool(PRONOUN.search(NOT_REF.sub(" ", re.sub(r"\b[A-H]\b", "", idea))))


def render(turns) -> str:
    return " ".join(f"{s}: {t}" for s, t in turns)


def prompt(turns) -> str:
    return f"Title: . Section: . Content: {render(turns)}"


def parse(text: str) -> list[str]:
    try:
        out = json.loads(text)
        return [str(x) for x in out if str(x).strip()] if isinstance(out, list) else []
    except Exception:
        return [x.strip(' "') for x in re.findall(r'"([^"]+)"', text)]


FILL = ["um", "uh", "like", "you know", "i mean", "so", "kind of"]


def noisy(turns, rng):
    """Half the time the input as written; otherwise stripped of punctuation and capitals, or made disfluent."""
    r = rng.random()
    if r < 0.5:
        return turns
    out = []
    for s, t in turns:
        if r < 0.75:
            t = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip() or "mm"
        else:
            ws, new = t.split(), []
            if len(ws) >= 6 and rng.random() < 0.4:
                new += ws[:2] + ["--"]
            for w in ws:
                if rng.random() < 0.15:
                    new.append(rng.choice(FILL) + ",")
                new.append(w)
            t = " ".join(new)
        out.append([s, t])
    return out


def load_seq2seq(path):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    return tok, model


def predict(args) -> None:
    rows = [json.loads(line) for line in open(args.eval, encoding="utf-8")]
    tok, model = load_seq2seq(args.model)
    rows.sort(key=lambda r: len(render(r["turns"])))
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), 16):
            b = rows[s : s + 16]
            enc = tok([prompt(r["turns"]) for r in b], return_tensors="pt", padding=True, truncation=True, max_length=768).to("cuda")
            g = model.generate(**enc, max_new_tokens=640, num_beams=1, do_sample=False)
            for r, t in zip(b, tok.batch_decode(g, skip_special_tokens=True)):
                out.append({"uid": r["uid"], "kind": r["kind"], "medium": r["medium"], "ideas": parse(t), "raw": t})
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"predicted": len(out), "empty": sum(not r["ideas"] for r in out)}), flush=True)


def train(args) -> None:
    from transformers import Adafactor

    rows = [json.loads(line) for line in open(args.train, encoding="utf-8")]
    pairs = [(r["turns"], json.dumps(r["ideas"], ensure_ascii=False)) for r in rows if r.get("ideas")]
    tok, model = load_seq2seq(PROP)
    model = model.float().train()
    model.gradient_checkpointing_enable()
    opt = Adafactor(model.parameters(), lr=args.lr, relative_step=False, scale_parameter=False, warmup_init=False)
    rng = random.Random(7)
    steps = args.epochs * len(pairs) // args.batch
    t0 = time.time()
    print(json.dumps({"pairs": len(pairs), "steps": steps}), flush=True)
    for step in range(1, steps + 1):
        b = rng.sample(pairs, args.batch)
        enc = tok([prompt(noisy(t, rng)) for t, _ in b], return_tensors="pt", padding=True, truncation=True, max_length=768).to("cuda")
        lab = tok([y for _, y in b], return_tensors="pt", padding=True, truncation=True, max_length=640).input_ids.to("cuda")
        lab[lab == tok.pad_token_id] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**enc, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": round(float(loss), 4), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("trained", flush=True)


QWEN = "Qwen/Qwen3-1.7B"


def qprompt(turns) -> str:
    body = "\n".join(f"{sp}: {t}" for sp, t in turns)
    return f"Rewrite as hyper-literal, self-contained ideas, one per line.\n\n{body}\n\nIdeas:\n"


def norm_target(turns, ideas: list[str]) -> list[str]:
    """One convention for the recipient of a single-speaker message ("the addressee", never an invented "B"),
    and no repeated ideas."""
    if len({sp for sp, _ in turns}) == 1:
        ideas = [re.sub(r"\bB\b", "the addressee", i) for i in ideas]
    return list(dict.fromkeys(i.strip() for i in ideas if i.strip()))


def train_qwen(args) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --train may name several files (comma-separated), e.g. the units and their paraphrases (split-para-data.py).
    rows = [json.loads(line) for f in args.train.split(",") for line in open(f, encoding="utf-8")]
    pairs = [(r["turns"], "\n".join(norm_target(r["turns"], r["ideas"]))) for r in rows if r.get("ideas")]
    tok = AutoTokenizer.from_pretrained(QWEN)
    model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(pairs) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    rng = random.Random(7)
    t0 = time.time()
    print(json.dumps({"pairs": len(pairs), "steps": steps}), flush=True)
    micro = 2
    for step in range(1, steps + 1):
        b = rng.sample(pairs, args.batch)
        for s0 in range(0, len(b), micro):
            ids, labs = [], []
            for t, y in b[s0 : s0 + micro]:
                p = tok(qprompt(noisy(t, rng)), add_special_tokens=False).input_ids[-900:]
                a = tok(y + tok.eos_token, add_special_tokens=False).input_ids[:700]
                ids.append(p + a)
                labs.append([-100] * len(p) + a)
            n = max(map(len, ids))
            x = torch.tensor([i + [tok.pad_token_id] * (n - len(i)) for i in ids], device="cuda")
            m = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i in ids], device="cuda")
            y = torch.tensor([l + [-100] * (n - len(l)) for l in labs], device="cuda")
            loss = model(input_ids=x, attention_mask=m, labels=y).loss
            (loss * len(ids) / len(b)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": round(float(loss), 4), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("trained", flush=True)


def predict_qwen(args) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in open(args.eval, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(model, args.model).merge_and_unload().eval()
    rows.sort(key=lambda r: len(render(r["turns"])))
    # Written batch by batch and resumable: a rerun skips the rows already in --out.
    done = {(o["uid"], o["kind"]) for o in map(json.loads, open(args.out, encoding="utf-8"))} if Path(args.out).exists() else set()
    rows = [r for r in rows if (r["uid"], r["kind"]) not in done]
    n, empty, t0 = 0, 0, time.time()

    def gen(b):
        """Generate for a batch; on CUDA out-of-memory, retry it in halves (the longest rows come last)."""
        try:
            enc = tok([qprompt(r["turns"]) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = model.generate(**enc, max_new_tokens=700, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            return tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(b) == 1:
                raise
            h = len(b) // 2
            print(json.dumps({"oom_split": len(b)}), flush=True)
            return gen(b[:h]) + gen(b[h:])

    with torch.no_grad(), open(args.out, "a", encoding="utf-8") as fh:
        for s in range(0, len(rows), args.batch):
            b = rows[s : s + args.batch]
            for r, t in zip(b, gen(b)):
                ideas = list(dict.fromkeys(x.strip() for x in t.split("\n") if x.strip()))
                fh.write(json.dumps({"uid": r["uid"], "kind": r["kind"], "medium": r["medium"], "ideas": ideas, "raw": t},
                                    ensure_ascii=False) + "\n")
                n, empty = n + 1, empty + (not ideas)
            fh.flush()
            if (s // args.batch) % 20 == 0:
                print(json.dumps({"rows": n, "of": len(rows), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    print(json.dumps({"predicted": n, "resumed_from": len(done), "empty": empty}), flush=True)


class Entail:
    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(NLI)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI, torch_dtype=torch.float16).cuda().eval()
        self.idx = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}["entailment"]
        self.cache = {}

    @torch.no_grad()
    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        pairs = [(p.lower(), h.lower()) for p, h in pairs]  # stripped inputs give lowercase ideas; case is not meaning
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        todo.sort(key=lambda p: len(p[0]) + len(p[1]))
        for s in range(0, len(todo), 64):
            b = todo[s : s + 64]
            enc = self.tok([p for p, _ in b], [h for _, h in b], return_tensors="pt", padding=True, truncation="only_first",
                           max_length=512).to("cuda")
            pr = self.model(**enc).logits.float().softmax(-1)[:, self.idx].tolist()
            self.cache.update(zip(b, pr))
        return [self.cache[p] for p in pairs]


def windows(ideas: list[str], size: int = 10) -> list[str]:
    """Premises made of consecutive ideas, so a long split still fits the NLI model's 512 tokens."""
    if len(ideas) <= size:
        return [" ".join(ideas)]
    return [" ".join(ideas[s : s + size]) for s in range(0, len(ideas) - size // 2, size // 2)]


def covered(nli, premise_ideas, hyps) -> list[float]:
    ws = windows(premise_ideas)
    sc = nli([(w, h) for h in hyps for w in ws])
    return [max(sc[i * len(ws) : (i + 1) * len(ws)]) for i in range(len(hyps))]


def binli_f1(nli, a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    fwd = nli([(x, y) for x in a for y in b])
    bwd = nli([(y, x) for x in a for y in b])
    m = [[min(fwd[i * len(b) + j], bwd[i * len(b) + j]) for j in range(len(b))] for i in range(len(a))]
    p = sum(max(r) for r in m) / len(a)
    r = sum(max(m[i][j] for i in range(len(a))) for j in range(len(b))) / len(b)
    return 2 * p * r / (p + r) if p + r else 0.0


def hm(xs):
    return len(xs) / sum(1 / max(x, 1e-6) for x in xs)


def score(args) -> None:
    preds = {(r["uid"], r["kind"]): r for r in map(json.loads, open(args.preds, encoding="utf-8"))}
    refs = {(r["uid"], r["kind"]): r for r in map(json.loads, open(args.eval, encoding="utf-8"))}
    nli = Entail()
    per = defaultdict(lambda: defaultdict(list))
    for (uid, kind), r in refs.items():
        if kind != "original" or (uid, "original") not in preds:
            continue
        med = r["medium"]
        ideas = preds[(uid, "original")]["ideas"]
        src = render(r["turns"])
        per[med]["n_ideas"].append(len(ideas))
        per[med]["n_ref"].append(len(r["ideas"]))
        if not ideas:
            for k in ("faithfulness", "coverage", "ref_f1"):
                per[med][k].append(0.0)
            continue
        per[med]["faithfulness"].append(sum(nli([(src, i) for i in ideas])) / len(ideas))
        per[med]["coverage"].append(sum(covered(nli, ideas, r["ideas"])) / len(r["ideas"]))
        per[med]["literal"].append(sum(not unresolved(i) for i in ideas) / len(ideas))
        per[med]["ref_f1"].append(binli_f1(nli, ideas, r["ideas"]))
        for v in ("stripped", "disfluent", "paraphrase", "split", "move"):
            other = preds.get((uid, v), {}).get("ideas")
            if other is not None:
                per[med][f"unit_inv_{v}"].append(binli_f1(nli, ideas, other))
                both = (sum(covered(nli, other, ideas)) / len(ideas) + sum(covered(nli, ideas, other)) / len(other)) / 2 if other else 0.0
                per[med][f"inv_{v}"].append(both)
                per[med][f"count_{v}"].append(float(len(other) == len(ideas)))
    report = {}
    for med, d in per.items():
        m = {k: round(sum(x) / len(x), 3) for k, x in d.items() if x}
        m["invariance"] = round((m.get("inv_stripped", 0) + m.get("inv_disfluent", 0)) / 2, 3)
        m["unit_invariance"] = round((m.get("unit_inv_stripped", 0) + m.get("unit_inv_disfluent", 0)) / 2, 3)
        m["accuracy"] = round(hm([m["faithfulness"], m["coverage"], m["invariance"]]), 3)
        m["n"] = len(d["faithfulness"])
        report[med] = m
    allm = {k: round(sum(report[m][k] * report[m]["n"] for m in report) / sum(report[m]["n"] for m in report), 3)
            for k in ("faithfulness", "coverage", "invariance", "unit_invariance", "literal", "ref_f1", "accuracy",
                      "inv_paraphrase", "inv_split", "inv_move", "unit_inv_paraphrase", "unit_inv_move",
                      "count_stripped", "count_disfluent", "count_paraphrase", "count_split")
            if all(k in report[m] for m in report)}
    report["all"] = allm
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "train", "score", "train-qwen", "predict-qwen"])
    ap.add_argument("--model", default=PROP)
    ap.add_argument("--eval", default="split/eval.jsonl")
    ap.add_argument("--train", default="split/train.jsonl")
    ap.add_argument("--preds", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    {"predict": predict, "train": train, "score": score, "train-qwen": train_qwen, "predict-qwen": predict_qwen}[args.cmd](args)


if __name__ == "__main__":
    main()
