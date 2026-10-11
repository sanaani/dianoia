"""Summarize ``remote-e2e.py`` runs by medium: name=path pairs, e.g. ``base=e2e/base R1=e2e/R1``.

Per version kind the triple is (mean best-match cosine to the original's ideas, worst idea's best match, share of
units whose own original is the nearest of all units). ``para_beats_move`` is the share of units where the
paraphrase's worst idea matches the original better than the meaning-changed version's worst idea.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

KEYS = ("coverage_split", "coverage_pipeline", "faithful_split", "faithful_pipeline", "idea_retain", "idea_kept", "fluency_gap")
KINDS = ("stripped", "disfluent", "paraphrase", "split", "move")


def summary(rows: list[dict]) -> dict:
    m = {k: round(st.fmean(r[k] for r in rows), 3) for k in KEYS}
    for kind in KINDS:
        vs = [r["versions"][kind] for r in rows if kind in r["versions"]]
        m[kind] = (round(st.fmean(v["mean_match"] for v in vs), 3), round(st.fmean(v["worst_match"] for v in vs), 3),
                   round(st.fmean(v["rank"] == 0 for v in vs), 3))
    both = [r for r in rows if "paraphrase" in r["versions"] and "move" in r["versions"]]
    m["para_beats_move"] = round(st.fmean(r["versions"]["paraphrase"]["worst_match"] > r["versions"]["move"]["worst_match"] for r in both), 3)
    m["n"] = len(rows)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {}
    for name, p in (x.split("=", 1) for x in args.runs):
        rows = [json.loads(l) for l in open(Path(p) / "e2e.jsonl", encoding="utf-8")]
        res[name] = {med: summary([r for r in rows if med == "all" or r["medium"] == med]) for med in ("all", "email", "meeting", "chat")}
        print(name, json.dumps(res[name]["all"]))
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
