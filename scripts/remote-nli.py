"""GPU worker: score every MultiNLI training pair with the NLI model (lowercased, as everywhere else).

Entailments get P(entail) in both directions; contradictions get P(contradiction). Pairs outside 4-60 words a
side are dropped. Output ``pairs/mnli-scored.jsonl``: premise, hypothesis, label, genre, scores. No LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def ok(t: str) -> bool:
    return 4 <= len(t.split()) <= 60


def main() -> None:
    ds = load_dataset("nyu-mll/multi_nli", split="train")
    rows = [{"p": r["premise"].strip(), "h": r["hypothesis"].strip(), "label": r["label"], "genre": r["genre"]}
            for r in ds if r["label"] in (0, 2) and ok(r["premise"]) and ok(r["hypothesis"])]
    tok = AutoTokenizer.from_pretrained(NLI)
    model = AutoModelForSequenceClassification.from_pretrained(NLI, torch_dtype=torch.float16).cuda().eval()
    lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    jobs = []  # (row index, field, premise, hypothesis)
    for i, r in enumerate(rows):
        if r["label"] == 0:
            jobs += [(i, "fwd", r["p"], r["h"]), (i, "bwd", r["h"], r["p"])]
        else:
            jobs.append((i, "contra", r["p"], r["h"]))
    jobs.sort(key=lambda j: len(j[2]) + len(j[3]))
    with torch.no_grad():
        for s in range(0, len(jobs), 128):
            b = jobs[s : s + 128]
            x = tok([j[2].lower() for j in b], [j[3].lower() for j in b], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to("cuda")
            p = model(**x).logits.float().softmax(-1).cpu()
            for j, q in zip(b, p):
                rows[j[0]][j[1]] = round(float(q[lab["contradiction"] if j[1] == "contra" else lab["entailment"]]), 4)
            if s % (128 * 200) == 0:
                print(json.dumps({"done": s, "of": len(jobs)}), flush=True)
    Path("pairs").mkdir(exist_ok=True)
    with open("pairs/mnli-scored.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    e = [r for r in rows if r["label"] == 0]
    c = [r for r in rows if r["label"] == 2]
    print(json.dumps({"entail": len(e), "both_ways": sum(r["fwd"] >= 0.9 and r["bwd"] >= 0.9 for r in e),
                      "fwd": sum(r["fwd"] >= 0.9 for r in e), "contra": len(c), "contra_kept": sum(r["contra"] >= 0.9 for r in c)}))


if __name__ == "__main__":
    main()
