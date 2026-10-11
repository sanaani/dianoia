"""GPU worker: align the splitter's ideas from an email with its ideas from each paraphrase of it (10/03/2026).

Input ``spara/preds.jsonl`` (``remote-split.py predict-qwen`` on ``spara/inputs.jsonl``): per contrast email item,
the splitter's ideas for the original (``orig``), two Sonnet paraphrases (``para0``, ``para1``) and two approving
and two disapproving versions (``appr*``, ``dis*``). For each original idea and each version, the score of a
candidate is min(P(orig idea entails it), P(it entails orig idea)) under the DeBERTa NLI model (lowercased, as
everywhere else), so the alignment never uses the encoder being trained. A paraphrase idea is kept as a positive
when the two are each other's best match and the score is >= ``--bar``. For slanted versions the best match by
fwd entailment of the shared facts is kept as a hard negative (it states the same facts with another judgment).
Output ``spara/aligned.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="spara/preds.jsonl")
    ap.add_argument("--out", default="spara/aligned.jsonl")
    ap.add_argument("--bar", type=float, default=0.5)
    args = ap.parse_args()
    items = defaultdict(dict)
    for l in open(args.preds, encoding="utf-8"):
        r = json.loads(l)
        items[r["uid"]][r["kind"]] = r["ideas"]
    tok = AutoTokenizer.from_pretrained(NLI)
    model = AutoModelForSequenceClassification.from_pretrained(NLI, torch_dtype=torch.float16).cuda().eval()
    ent = {v.lower(): int(k) for k, v in model.config.id2label.items()}["entailment"]
    pairs = sorted({(o, v) for d in items.values() for k, vs in d.items() if k != "orig" for o in d.get("orig", []) for v in vs}
                   | {(v, o) for d in items.values() for k, vs in d.items() if k != "orig" for o in d.get("orig", []) for v in vs},
                   key=lambda p: len(p[0]) + len(p[1]))
    score = {}
    with torch.no_grad():
        for s in range(0, len(pairs), 128):
            b = pairs[s : s + 128]
            x = tok([p.lower() for p, _ in b], [h.lower() for _, h in b], return_tensors="pt", padding=True,
                    truncation=True, max_length=256).to("cuda")
            score.update(zip(b, model(**x).logits.float().softmax(-1)[:, ent].tolist()))
            if s % (128 * 200) == 0:
                print(json.dumps({"done": s, "of": len(pairs)}), flush=True)
    n_pos = n_neg = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for uid, d in items.items():
            orig = d.get("orig", [])
            rec = {"uid": uid, "orig": orig, "pos": [], "neg": []}
            for kind, vs in d.items():
                if kind == "orig" or not vs or not orig:
                    continue
                both = {(o, v): min(score[(o, v)], score[(v, o)]) for o in orig for v in vs}
                if kind.startswith("para"):
                    for o in orig:
                        v = max(vs, key=lambda v: both[(o, v)])
                        if both[(o, v)] >= args.bar and max(orig, key=lambda o2: both[(o2, v)]) == o:
                            rec["pos"].append([o, v, kind, round(both[(o, v)], 4)])
                else:
                    # A slanted idea shares the facts but not the judgment: match by the larger one-way score. A
                    # slanted version can split into the bare fact plus a separate judgment; a match that entails
                    # both ways is that bare fact, the same idea, so it is never a negative.
                    for o in orig:
                        cand = [v for v in vs if both[(o, v)] < args.bar]
                        if cand:
                            v = max(cand, key=lambda v: max(score[(o, v)], score[(v, o)]))
                            rec["neg"].append([o, v, kind, round(both[(o, v)], 4)])
            n_pos += len(rec["pos"])
            n_neg += len(rec["neg"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(items), "positives": n_pos, "negatives": n_neg}), flush=True)


if __name__ == "__main__":
    main()
