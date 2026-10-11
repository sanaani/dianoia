"""GPU worker: the whole pipeline end to end, text -> splitter ideas -> idea encoder -> decoder (10/03/2026).

Input ``e2e/units.jsonl`` from ``e2e-prepare.py``: per check unit and version (original, stripped, disfluent,
paraphrase, split, move), the source text, the splitter's ideas, and Sonnet's reference ideas for the original,
all with one placeholder mapping per unit. Every idea is encoded (``--adapter``: a trained LoRA from
``remote-encoder.py``; none: SONAR as released), decoded greedily and re-encoded.

Per unit, written to ``<out>/e2e.jsonl``:
- **pipeline coverage**: Sonnet's reference ideas entailed by the decoded ideas taken together (DeBERTa NLI, as in
  ``remote-split.py``), beside the same measure on the splitter's ideas before the encoder (the splitter's share
  of the loss);
- **pipeline faithfulness**: the source text entails each decoded idea;
- **idea round trip**: decoded idea entails the splitter idea and back (second NLI judge, RoBERTa-large MNLI);
- **fluency gap**: per-token loss of decoded ideas minus the splitter's ideas (Qwen2.5-0.5B);
- **version match**, in the encoder's space: for each version, every idea's best cosine to the original's ideas
  (mean, and the worst idea), and the rank of the unit's own original among all units' originals by the mean
  best-match both ways. A paraphrase should match well and retrieve its own unit; a meaning-changed version
  (move) should have a worse worst-idea match than the paraphrase of the same unit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

here = Path(__file__).parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, here / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="e2e/units.jsonl")
    ap.add_argument("--adapter")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    enc = load("remote_encoder", "remote-encoder.py")
    rows = [json.loads(l) for l in open(args.units, encoding="utf-8")]
    ideas = sorted({i for r in rows for i in r["ideas"]}, key=len)

    model = enc.SonarAE(args.rank, adapters=args.adapter is not None).cuda()
    if args.adapter:
        missing = model.load_state_dict(torch.load(args.adapter, map_location="cuda"), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    model.eval()
    raw, dec = {}, {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in enc.batched(ideas, 128):
            for t, v in zip(b, model.encode(b).float()):
                raw[t] = v
        # The decoder reads the raw vector; only the cosines below use the normalized one.
        for b in enc.batched(ideas, 64):
            for t, d in zip(b, model.generate(torch.stack([raw[t] for t in b]))):
                dec[t] = d
    vec = {t: F.normalize(v, dim=-1) for t, v in raw.items()}
    del model
    torch.cuda.empty_cache()

    # Second judge for the per-idea round trip, then the fluency LM, then DeBERTa for coverage and faithfulness.
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    jtok = AutoTokenizer.from_pretrained(enc.JUDGE)
    judge = AutoModelForSequenceClassification.from_pretrained(enc.JUDGE, torch_dtype=torch.float16).cuda().eval()
    ent = {v.lower(): int(k) for k, v in judge.config.id2label.items()}["entailment"]

    def rnli(pairs):
        res = []
        with torch.no_grad():
            for b in enc.batched(pairs, 64):
                x = jtok([p.lower() for p, _ in b], [h.lower() for _, h in b], return_tensors="pt", padding=True,
                         truncation=True, max_length=256).to("cuda")
                res += judge(**x).logits.float().softmax(-1)[:, ent].tolist()
        return res

    retain = dict(zip(ideas, rnli([(dec[i], i) for i in ideas])))
    kept = dict(zip(ideas, rnli([(i, dec[i]) for i in ideas])))
    del judge
    torch.cuda.empty_cache()
    ftok = AutoTokenizer.from_pretrained(enc.FLUENCY)
    flm = AutoModelForCausalLM.from_pretrained(enc.FLUENCY, torch_dtype=torch.bfloat16).cuda().eval()

    def nll(t):
        x = ftok(t or ".", return_tensors="pt").to("cuda")
        with torch.no_grad():
            return float(flm(**x, labels=x["input_ids"]).loss) if x["input_ids"].shape[1] > 1 else 10.0

    flu = {i: (nll(dec[i]), nll(i)) for i in ideas}
    del flm
    torch.cuda.empty_cache()
    split = load("remote_split", "remote-split.py")
    nli = split.Entail()

    by_unit = defaultdict(dict)
    for r in rows:
        by_unit[r["uid"]][r["kind"]] = r
    uids = sorted(u for u in by_unit if "original" in by_unit[u] and by_unit[u]["original"]["ideas"])
    orig = {u: torch.stack([vec[i] for i in by_unit[u]["original"]["ideas"]]) for u in uids}

    def match(a, b):
        """Each idea of a: best cosine to b's ideas."""
        return (a @ b.T).max(1).values

    with open(Path(args.out) / "e2e.jsonl", "w", encoding="utf-8") as f:
        for u in uids:
            o = by_unit[u]["original"]
            ids, ref = o["ideas"], o["ref_ideas"]
            decs = [dec[i] for i in ids]
            rec = {"uid": u, "medium": o["medium"], "n_ideas": len(ids),
                   "coverage_split": sum(split.covered(nli, ids, ref)) / len(ref),
                   "coverage_pipeline": sum(split.covered(nli, decs, ref)) / len(ref),
                   "faithful_split": sum(nli([(o["source"], i) for i in ids])) / len(ids),
                   "faithful_pipeline": sum(nli([(o["source"], d) for d in decs])) / len(ids),
                   "idea_retain": sum(retain[i] for i in ids) / len(ids),
                   "idea_kept": sum(kept[i] for i in ids) / len(ids),
                   "fluency_gap": sum(flu[i][0] - flu[i][1] for i in ids) / len(ids),
                   "examples": [[i, dec[i]] for i in ids[:3]], "versions": {}}
            for kind, v in by_unit[u].items():
                if kind == "original" or not v["ideas"]:
                    continue
                V = torch.stack([vec[i] for i in v["ideas"]])
                fwd, bwd = match(V, orig[u]), match(orig[u], V)
                sims = {w: float((match(V, orig[w]).mean() + match(orig[w], V).mean()) / 2) for w in uids}
                own = sims[u]
                rec["versions"][kind] = {"mean_match": float((fwd.mean() + bwd.mean()) / 2), "worst_match": float(min(fwd.min(), bwd.min())),
                                         "rank": sum(s > own for s in sims.values()), "pool": len(uids)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"units": len(uids), "ideas": len(ideas)}), flush=True)


if __name__ == "__main__":
    main()
