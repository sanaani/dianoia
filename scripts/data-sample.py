"""Quality check of the public paraphrase sources before any of them is built into training data.

For each source, a seeded sample of its human-labeled same-meaning pairs is scored by the NLI model in BOTH
directions (lowercased, as in ``remote-split.py``). A pair counts as the same idea only if each side entails the
other at >= 0.9. Contradiction pairs (the planned hard negatives) are checked for P(contradiction) >= 0.9.

Also reported: word overlap (how much rewording a surviving pair actually teaches), length, and for the
entailments taken as lossy positives, how much detail each drops: the share of the longer side's content words
missing from the shorter side, and the share of extreme generalizations (shorter side <= 4 words or > 80% dropped). Writes
``results/phase-6-data/sample.json`` plus every scored pair to ``work/data-sample/pairs.jsonl`` for reading.
No LLM is used.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from dianoia.paths import work_path
except ImportError:  # GPU worker: no package, write next to the script
    def work_path(*a):
        return Path("work").joinpath(*a)

NLI = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
N_POS, N_NEG, BAR = 500, 200, 0.9
OUT = Path(__file__).resolve().parents[1] / "results" / "phase-6-data" if Path(__file__).resolve().parents[1].joinpath("results").exists() else Path("results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STOP = set("a an the is are was were be been being of in on at to for with and or but by from as it its this that these those there here".split())


def sources(rng: random.Random) -> dict[str, dict[str, list[tuple[str, str]]]]:
    def pick(rows, n):
        rows = list(rows)
        return rng.sample(rows, min(n, len(rows)))

    snli = load_dataset("stanfordnlp/snli", split="train")
    mnli = load_dataset("nyu-mll/multi_nli", split="train")
    mrpc = load_dataset("nyu-mll/glue", "mrpc", split="train")
    stsb = load_dataset("sentence-transformers/stsb", split="train")
    out = {}
    for name, ds in (("snli", snli), ("mnli", mnli)):
        lab = ds["label"]
        idx_e = [i for i, l in enumerate(lab) if l == 0]
        idx_c = [i for i, l in enumerate(lab) if l == 2]
        out[name] = {"pos": [(ds[i]["premise"], ds[i]["hypothesis"]) for i in rng.sample(idx_e, N_POS)],
                     "neg": [(ds[i]["premise"], ds[i]["hypothesis"]) for i in rng.sample(idx_c, N_NEG)]}
    out["mrpc"] = {"pos": pick(((r["sentence1"], r["sentence2"]) for r in mrpc if r["label"] == 1), N_POS), "neg": []}
    out["stsb"] = {"pos": pick(((r["sentence1"], r["sentence2"]) for r in stsb if r["score"] >= 0.9), N_POS), "neg": []}
    return out


class Nli:
    def __init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(NLI)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI).to(DEV).eval()
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.e, self.c = lab["entailment"], lab["contradiction"]

    @torch.no_grad()
    def probs(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        out = []
        for s in range(0, len(pairs), 16):
            b = pairs[s : s + 16]
            x = self.tok([p.lower() for p, _ in b], [h.lower() for _, h in b], return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEV)
            p = self.model(**x).logits.softmax(-1)
            out += [(float(r[self.e]), float(r[self.c])) for r in p]
        return out


def words(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", t.lower()))


def dropped(a: str, b: str) -> float:
    """Share of the longer side's content words absent from the shorter side."""
    long_, short = (a, b) if len(a.split()) >= len(b.split()) else (b, a)
    lw, sw = words(long_) - STOP, words(short) - STOP
    return len(lw - sw) / max(1, len(lw))


def main() -> None:
    rng = random.Random(6)
    data = sources(rng)
    import gc; gc.collect()
    nli = Nli()
    rows, stats = [], {}
    for name, d in data.items():
        t = time.time()
        pos = d["pos"]
        fwd = nli.probs(pos)
        bwd = nli.probs([(b, a) for a, b in pos])
        kept = []
        for (a, b), f, r in zip(pos, fwd, bwd):
            ok = f[0] >= BAR and r[0] >= BAR
            wa, wb = words(a), words(b)
            ov = len(wa & wb) / max(1, len(wa | wb))
            dr = dropped(a, b)
            short = min(len(a.split()), len(b.split()))
            rows.append({"source": name, "kind": "pos", "a": a, "b": b, "fwd": round(f[0], 3), "bwd": round(r[0], 3), "kept": ok,
                         "overlap": round(ov, 3), "dropped": round(dr, 3), "extreme": short <= 4 or dr > 0.8})
            if ok:
                kept.append(ov)
        mine = [x for x in rows if x["source"] == name and x["kind"] == "pos"]
        s = {"lossy_mean_dropped": round(sum(x["dropped"] for x in mine) / len(mine), 3),
             "lossy_extreme_share": round(sum(x["extreme"] for x in mine) / len(mine), 3),
             "pos_sampled": len(pos), "pos_kept": len(kept), "pos_rate": round(len(kept) / len(pos), 3),
             "fwd_pass": round(sum(f[0] >= BAR for f in fwd) / len(pos), 3),
             "kept_mean_overlap": round(sum(kept) / max(1, len(kept)), 3),
             "kept_low_overlap_share": round(sum(o < 0.5 for o in kept) / max(1, len(kept)), 3),
             "mean_words": round(sum(len(a.split()) + len(b.split()) for a, b in pos) / (2 * len(pos)), 1)}
        if d["neg"]:
            con = nli.probs(d["neg"])
            for (a, b), c in zip(d["neg"], con):
                rows.append({"source": name, "kind": "neg", "a": a, "b": b, "contra": round(c[1], 3), "kept": c[1] >= BAR})
            s |= {"neg_sampled": len(con), "neg_rate": round(sum(c[1] >= BAR for c in con) / len(con), 3)}
        s["seconds"] = round(time.time() - t)
        stats[name] = s
        print(name, json.dumps(s), flush=True)
    work_path("data-sample").mkdir(parents=True, exist_ok=True)
    with open(work_path("data-sample", "pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.json").write_text(json.dumps({"model": NLI, "bar": BAR, "seed": 6, "sources": stats}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
