"""Inputs for the end-to-end check (``remote-e2e.py``): the splitter's ideas for every version of every check unit,
the source text, and Sonnet's reference ideas for the original, with one placeholder mapping per unit (the idea
encoder was trained on placeholders, never on names). Writes ``e2e/units.jsonl``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from dianoia.paths import work_path

spec = importlib.util.spec_from_file_location("build_pairs", Path(__file__).with_name("build-pairs.py"))
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(work_path("split", "pred-versions.jsonl")))
    ap.add_argument("--out", default=str(work_path("e2e", "units.jsonl")))
    args = ap.parse_args()
    ph = bp.Placeholders()
    refs = {(r["uid"], r["kind"]): r for r in map(json.loads, open(work_path("split", "eval.jsonl"), encoding="utf-8"))}
    units = [json.loads(l) for l in open(work_path("split", "versions-all.jsonl"), encoding="utf-8")]
    preds = {(r["uid"], r["kind"]): r["ideas"] for r in map(json.loads, open(args.preds, encoding="utf-8"))}
    by_uid: dict[str, list] = {}
    for u in units:
        by_uid.setdefault(u["uid"], []).append(u)
    work_path("e2e").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for uid, vs in by_uid.items():
            ref = refs[(uid, "original")]["ideas"]
            texts = [*ref]
            spans = []
            for v in vs:
                src = " ".join(f"{s}: {t}" for s, t in v["turns"])
                ideas = preds.get((uid, v["kind"]), [])
                spans.append((v, len(texts), len(ideas)))
                texts += [src, *ideas]
            out = ph.apply(texts)
            for v, s, k in spans:
                f.write(json.dumps({"uid": uid, "kind": v["kind"], "medium": v["medium"], "source": out[s],
                                    "ideas": out[s + 1 : s + 1 + k], "ref_ideas": out[: len(ref)]}, ensure_ascii=False) + "\n")
                n += 1
    print(json.dumps({"units": len(by_uid), "rows": n}))


if __name__ == "__main__":
    main()
