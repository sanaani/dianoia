"""Phase 13 step 1 GPU stage: do phase 12's encoders still find far sources from a short query?
(results/phase-13-scale-far/HYPOTHESES.md, approved 10/07/2026)

    python remote-p13.py --work p13 --config p13/config.used.yaml --out p13/out [--smoke]

Inputs in --work: eval.jsonl and docs.jsonl (phase 12's, copied unchanged), short.jsonl (p13-data.py queries), and
phase 12's adapters under adapters/<name>. For each encoder (off the shelf, far-trained, near-trained) the far test
questions' candidates are encoded once, then each question is asked three ways: its abstract (phase 12's query, a
check that this run reproduces phase 12's numbers), its Background opening and its title. Every query carries phase
12's instruction, unchanged, since the adapters were trained with it. Writes scores/<encoder>.json (per form, per
question: top-10 hit and reciprocal rank) and p13-short.json. One encoder at a time: its model and document vectors
are dropped before the next loads. No verdicts here: scripts/p13-verdicts.py judges from the per-query arrays.
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

FORMS = ("abstract", "background", "title")


def ranks(rows, qv, dv, k):
    out = {}
    for r, q in zip(rows, qv):
        q = q.float()
        true = float(q @ dv[r["cited"]].float())
        c = torch.stack([dv[x] for x in r["candidates"]]).float() @ q
        rank = 1 + int((c >= true).sum())
        out[str(r["query"])] = {"hit": float(rank <= k), "rr": 1.0 / rank}
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
    ev = [r for r in rd("eval.jsonl") if r["split"] in cfg["step1"]["splits"]]
    short = {r["query"]: r for r in rd("short.jsonl")}
    for r in ev:
        s = short[r["query"]]
        r["abstract"], r["background"], r["title"] = r["query_text"], s["background"], s["title"]
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
    k = cfg["step1"]["top_k"]
    R.log(stage="start", device=dev, model=name, queries=len(ev), docs=len(docs), smoke=a.smoke)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    doc_ids = sorted(docs)
    encoders = [("base", None)] + ([] if a.smoke and dev == "cpu" else
                                   [(t, work / "adapters" / Path(p).name) for t, p in cfg["step1"]["adapters"].items()])
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
            res = {}
            for form in FORMS:
                rows = [r for r in ev if r[form]]
                qv = R.encode_all(model, tok, [cfg["instruct"] + r[form] for r in rows], dev, max_len=max_len)
                res[form] = ranks(rows, qv, dv, k)
            f.write_text(json.dumps(res))
            del model, dv
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()
            R.log(stage="scored", model=tag, minutes=round((time.time() - t0) / 60, 1))
        res = json.loads(f.read_text())
        summary[tag] = {form: {"n": len(v), "top10": round(sum(x["hit"] for x in v.values()) / max(1, len(v)), 4)}
                        for form, v in res.items()}
        R.log(stage="summary", model=tag, **{form: summary[tag][form]["top10"] for form in FORMS})
    (out / "p13-short.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
