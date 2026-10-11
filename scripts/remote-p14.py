"""Phase 14 step 3 GPU stage: does phase 12's far-trained encoder find what management papers borrow?
(results/phase-14-openalex/HYPOTHESES.md, step 3 details, 10/09/2026)

    python remote-p14.py --work p14 --config p14/config.used.yaml --out p14/out [--smoke]

Inputs in --work: eval.jsonl and docs.jsonl (p14-data.py step3-build) and phase 12's adapters under adapters/<name>.
As remote-p13.py, with one query form (the citing abstract): for each encoder (off the shelf, far-trained,
near-trained) the candidates are encoded once and each query, carrying phase 12's instruction unchanged, is ranked
against its 1,000. Writes scores/<encoder>.json (per query: top-10 hit and reciprocal rank) and p14-test.json. One
encoder at a time; a finished encoder's scores are skipped on a rerun. Verdicts: scripts/p14-verdicts.py.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import torch
import yaml

spec = importlib.util.spec_from_file_location("r12", Path(__file__).with_name("remote-p12.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def ranks(rows, qv, dv, k):
    out = {}
    for r, q in zip(rows, qv):
        q = q.float()
        true = float(q @ dv[r["cited"]].float())
        c = torch.stack([dv[x] for x in r["candidates"]]).float() @ q
        rank = 1 + int((c >= true).sum())
        out[r["query"]] = {"hit": float(rank <= k), "rr": 1.0 / rank}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    work, out = Path(a.work), Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rd = lambda f: [json.loads(l) for l in open(work / f, encoding="utf-8")]
    ev = rd("eval.jsonl")
    name = cfg["embedder"]
    if a.smoke:
        ev = ev[:3]
        for r in ev:
            r["candidates"] = r["candidates"][:15]
        if dev == "cpu":
            name = cfg["smoke"]["embedder"]
    need = {c for r in ev for c in [r["cited"], *r["candidates"]]}
    docs = {r["id"]: r["text"] for r in rd("docs.jsonl") if r["id"] in need}
    max_len = 64 if a.smoke and dev == "cpu" else 512
    k = cfg["step3"]["top_k"]
    R.log(stage="start", device=dev, model=name, queries=len(ev), docs=len(docs), smoke=a.smoke)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    doc_ids = sorted(docs)
    encoders = [("base", None)] + ([] if a.smoke and dev == "cpu" else
                                   [(t, work / "adapters" / Path(p).name) for t, p in cfg["step3"]["adapters"].items()])
    summary = {}
    for tag, adapter in encoders:
        f = out / "scores" / f"{tag}.json"
        if not f.exists():
            t0 = time.time()
            model = AutoModel.from_pretrained(name, dtype=R.DTYPE[dev]).to(dev)
            if adapter is not None:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
            dv = dict(zip(doc_ids, R.encode_all(model, tok, [docs[i] for i in doc_ids], dev, max_len=max_len)))
            qv = R.encode_all(model, tok, [cfg["instruct"] + r["query_text"] for r in ev], dev, max_len=max_len)
            f.write_text(json.dumps(ranks(ev, qv, dv, k)))
            del model, dv
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()
            R.log(stage="scored", model=tag, minutes=round((time.time() - t0) / 60, 1))
        res = json.loads(f.read_text())
        summary[tag] = {"n": len(res), "top10": round(sum(x["hit"] for x in res.values()) / max(1, len(res)), 4)}
        R.log(stage="summary", model=tag, **summary[tag])
    (out / "p14-test.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
