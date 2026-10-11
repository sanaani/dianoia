"""Score an autoencoder stage's decodes: word Jaccard and the field-by-field intent check.

Input: a ``decoded.jsonl`` from ``remote-ae.py`` (source plus one decode per arm). The intent check is
``score-tier2.CHECK`` (same act, same action and object, same roles, same quantities), run by Haiku on
every arm. Stage 0's gate is the ae256 arm passing all four fields on >= 80% of held-out sentences.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from dianoia import corpus, llm
from dianoia.paths import result_path

spec = importlib.util.spec_from_file_location("score_tier2", Path(__file__).parent / "score-tier2.py")
score_tier2 = importlib.util.module_from_spec(spec)
sys.modules["score_tier2"] = score_tier2
spec.loader.exec_module(score_tier2)


def jaccard(a: str, b: str) -> float:
    wa, wb = {w.lower() for w in corpus.words(a)}, {w.lower() for w in corpus.words(b)}
    return len(wa & wb) / max(len(wa | wb), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("decoded")
    ap.add_argument("--out", required=True, help="result file stem under results/")
    ap.add_argument("--target", default="source", help="the field the decode should reproduce")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--model", default="sonnet", help="headless Claude model that grades (sonnet from 10/01/2026)")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.decoded, encoding="utf-8")]
    arms = [k for k in ("ae64", "ae128", "ae256", "ae512", "ae1024", "noise") if k in rows[0]]
    long = pd.DataFrame([{"i": i, "arm": a, "source": r["source"], "target": r.get(args.target, r["source"]), "text": r[a]} for i, r in enumerate(rows) for a in arms])
    long["jaccard"] = [jaccard(s, t) for s, t in zip(long["source"], long["text"])]

    prompts, chunks = [], []
    for s in range(0, len(long), 15):
        ch = long.iloc[s : s + 15]
        body = "\n\n".join(f"PAIR {k}\nORIGINAL: {r['source']}\nRECONSTRUCTION: {r['text']}" for k, (_, r) in enumerate(ch.iterrows()))
        prompts.append(score_tier2.CHECK.format(pairs=body))
        chunks.append(ch)

    def safe(q):
        try:
            return llm.ask_json(q, model=args.model)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        answers = list(ex.map(safe, prompts))
    fields = ["act", "action", "roles", "quantities"]
    for ch, ans in zip(chunks, answers):
        got = {int(a["p"]): a for a in ans if isinstance(a, dict) and str(a.get("p", "")).isdigit()} if isinstance(ans, list) else {}
        for k, idx in enumerate(ch.index):
            a = got.get(k)
            if a:
                for f in fields:
                    long.loc[idx, f] = bool(a.get(f))
    ok = long[fields].notna().all(axis=1)
    long["all"] = long[fields].fillna(False).all(axis=1).where(ok)

    report = {"grader": args.model, "decoded": args.decoded, "n_sources": len(rows), "arms": {}}
    for arm, g in long.groupby("arm"):
        c = g[g["all"].notna()]
        report["arms"][arm] = {"n_checked": len(c), "intent_check_all": round(float(c["all"].astype(float).mean()), 3),
                               **{f: round(float(c[f].astype(float).mean()), 3) for f in fields},
                               "word_jaccard_median": round(float(g["jaccard"].median()), 3),
                               "exact_copy": round(float((g["source"].str.strip().str.lower() == g["text"].str.strip().str.lower()).mean()), 3)}
        cos = [r.get(f"{arm}_cos") for r in rows]
        tau = [r.get("tau") for r in rows]
        have = [(c, t) for c, t in zip(cos, tau) if c is not None and t is not None]
        if have:
            report["arms"][arm]["pass_within_tolerance"] = round(float(np.mean([c >= t for c, t in have])), 3)
            report["arms"][arm]["cos_median"] = round(float(np.median([c for c, _ in have])), 3)
            report["arms"][arm]["tau_median"] = round(float(np.median([t for _, t in have])), 3)
        unbound = [r.get(f"{arm}_unbound") for r in rows if r.get(f"{arm}_unbound") is not None]
        if unbound:
            report["arms"][arm]["decodes_with_unbound_placeholder"] = round(float(np.mean([u > 0 for u in unbound])), 3)
    result_path(*args.out.split("/")).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    long.to_json(result_path(*args.out.split("/")).with_suffix(".jsonl"), orient="records", lines=True, force_ascii=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
