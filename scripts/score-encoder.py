"""Score the idea encoder's round trip per arm, with bootstrap 95% intervals (pre-registered 10/02/2026).

Reads ``<run>/eval.jsonl`` and ``<run>/geometry.json`` from ``remote-encoder.py eval`` for each run given. Metrics
use ranks and text judgments only, never the trained encoder's raw distances:
- own_idea_all: the round trip's nearest test anchor is its own;
- own_idea_siblings: the round trip is nearer its own anchor than each rule-edited sibling (negation, role swap,
  modality, number, contradiction, phase-1 meaning changes); per sibling, resampled by anchor;
- detail_retention: P(decoded entails original), second NLI model;
- meaning_kept: P(original entails decoded), second NLI model;
- negation_separation: the anchor is nearer every positive than its negation;
- fluency_gap: per-token loss of the decoded text minus the original's, small separate LM (lower is better);
- geometry: share of the 80 ideas' rewordings closer to their centre than the negation is.
Amendment (10/02/2026, after the baseline showed the self round trip at ceiling, before any trained arm was
scored): the decision metrics are the round trips from ANOTHER wording of the idea (cross_*):
- cross_own_siblings: the round trip from a stripped, disfluent, paraphrased, register-shifted, MRPC or both-ways
  wording comes back nearer the original idea than every meaning-changed sibling;
- cross_retention: its decoded text entails the original (second NLI model).
Decision (arms L vs B): keep L unless cross_own_siblings or cross_retention is lower by > 0.02 with
non-overlapping intervals. The self round-trip metrics are reported only.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

NEG_KINDS = {"negation", "role_swap", "modality", "number", "contradiction", "quantity", "act_change", "type_change", "slant", "contrast", "edit"}


def per_item(rows: list[dict]) -> dict[str, list[float]]:
    m: dict[str, list] = {k: [] for k in ("cross_own_siblings", "cross_retention", "cross_meaning_kept", "cross_own_all",
                                          "cross_fluency_gap", "own_idea_all", "own_idea_siblings", "detail_retention",
                                          "meaning_kept", "negation_separation", "fluency_gap")}
    for r in rows:
        # Decision metrics (amendment 10/02/2026): round trips from another wording of the idea. MultiNLI's lossy
        # hypotheses are excluded: they are not expected to carry all the detail back.
        cr = [c for c in r.get("cross", []) if c["kind"] != "lossy"]
        sib_kinds = [s["kind"] for s in r["siblings"]]
        with_sib = [c for c in cr if any(k in NEG_KINDS for k in sib_kinds)]
        m["cross_own_siblings"].append(statistics.fmean(
            float(c["to_anchor"] > max(x for x, k in zip(c["to_siblings"], sib_kinds) if k in NEG_KINDS)) for c in with_sib) if with_sib else None)
        m["cross_retention"].append(statistics.fmean(c["retain"] for c in cr) if cr else None)
        m["cross_meaning_kept"].append(statistics.fmean(c["kept"] for c in cr) if cr else None)
        m["cross_own_all"].append(statistics.fmean(float(c["rank_all"] == 0) for c in cr) if cr else None)
        m["cross_fluency_gap"].append(statistics.fmean(c["nll_dec"] - r["nll_orig"] for c in cr) if cr else None)
        m["own_idea_all"].append(float(r["rank_all"] == 0))
        sib = [s for s in r["siblings"] if s["kind"] in NEG_KINDS]
        m["own_idea_siblings"].append(sum(r["self_back"] > s["back"] for s in sib) / len(sib) if sib else None)
        m["detail_retention"].append(r["retain"])
        m["meaning_kept"].append(r["kept"])
        neg = [s["anchor"] for s in r["siblings"] if s["kind"] == "negation"]
        pos = [p["anchor"] for p in r["positives"]]
        m["negation_separation"].append(float(min(pos) > max(neg)) if neg and pos else None)
        m["fluency_gap"].append(r["nll_dec"] - r["nll_orig"])
    return m


def boot(xs: list, n: int = 1000, seed: int = 0) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None}
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return {"mean": round(statistics.fmean(xs), 4), "lo": round(means[int(0.025 * n)], 4), "hi": round(means[int(0.975 * n)], 4), "n": len(xs)}


def score(run: Path) -> dict:
    rows = [json.loads(l) for l in open(run / "eval.jsonl", encoding="utf-8")]
    out = {"all": {k: boot(v) for k, v in per_item(rows).items()}}
    for src in sorted({r["source"] for r in rows}):
        out[src] = {k: boot(v) for k, v in per_item([r for r in rows if r["source"] == src]).items()}
    geo = json.load(open(run / "geometry.json", encoding="utf-8"))
    out["geometry"] = {"rewordings_closer_than_negation": boot([g["rewordings_closer_than_negation"] for g in geo]),
                       "roundtrip_beats_negation": boot([g.get("roundtrip_beats_negation") for g in geo]),
                       "radius_median": round(statistics.median(g["radius"] for g in geo), 4),
                       "negation_median": round(statistics.median(g["negation_to_centre"] for g in geo if g["negation_to_centre"] is not None), 4)}
    return out


def worse(l: dict, b: dict) -> bool:
    """Arm L is worse than arm B by more than 0.02 with non-overlapping intervals."""
    return b["mean"] - l["mean"] > 0.02 and l["hi"] < b["lo"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path pairs, e.g. base=out/base L1=out/L1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {name: score(Path(p)) for name, p in (x.split("=", 1) for x in args.runs)}
    for name, r in res.items():
        print(name, json.dumps({k: (v["mean"], v.get("lo"), v.get("hi")) for k, v in r["all"].items()}), json.dumps(r["geometry"]))
    Ls = [n for n in res if n.startswith("L")]
    Bs = [n for n in res if n.startswith("B")]
    if Ls and Bs:
        verdict = {}
        for metric in ("cross_own_siblings", "cross_retention"):
            l = [res[n]["all"][metric] for n in Ls]
            b = [res[n]["all"][metric] for n in Bs]
            verdict[metric] = {"L": l, "B": b, "L_worse_in_every_seed_pair": all(worse(x, y) for x in l for y in b)}
        verdict["keep"] = "B" if any(v["L_worse_in_every_seed_pair"] for v in verdict.values() if isinstance(v, dict)) else "L"
        res["decision"] = verdict
        print("decision", json.dumps(verdict))
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
