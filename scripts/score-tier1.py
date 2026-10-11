"""Tier 1 on the test mailboxes: every encoder, every move kind, against the pre-registered bars.

Encoders scored: word Jaccard (the control), raw SONAR, the intent head at 64 / 128 / 256 dims,
the off-the-shelf encoders, the two word-only controls, and a shuffled-vector control (the head's
vectors assigned to the wrong texts, which must land near 0.5).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from dianoia import corpus, encoders, head, tier1
from dianoia.paths import result_path, work_path

OUT = "phase-1-intent"
OTS = ["minilm", "bge", "e5", "gtr", "qwen3"]


def jaccard(a: str, b: str) -> float:
    wa, wb = {w.lower() for w in corpus.words(a)}, {w.lower() for w in corpus.words(b)}
    return len(wa & wb) / max(len(wa | wb), 1)


def by_field(index: pd.DataFrame, vecs: np.ndarray, test_idx: np.ndarray) -> dict[str, np.ndarray]:
    pos = {i: r for r, i in enumerate(test_idx)}
    out: dict[str, np.ndarray] = {}
    for row in index.itertuples(index=False):
        if row.item in pos:
            a = out.setdefault(row.field, np.full((len(test_idx), vecs.shape[1]), np.nan, dtype=np.float32))
            a[pos[row.item]] = vecs[row.line]
    return out


def cos_sims(z: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = {f: v / np.linalg.norm(v, axis=1, keepdims=True) for f, v in z.items()}
    return {f: (n["text"] * v).sum(1) for f, v in n.items() if f != "text"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chosen", default="head256", help="the model the bars apply to, chosen on validation before this runs")
    ap.add_argument("--materials", choices=["original", "judged"], default="original",
                    help="judged drops every variant the blind judge disputed (PREREGISTRATION addendum)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    index = pd.read_parquet(work_path("phase1", "index.parquet"))
    texts = [json.loads(line)["text"] for line in open(work_path("phase1", "texts.jsonl"), encoding="utf-8")]
    test_idx = items.index[items["split"] == "test"].to_numpy()
    if args.materials == "judged":
        judged = pd.read_parquet(work_path("phase1", "judged.parquet"))
        bad = judged[judged["disputed"] & ~judged["field"].isin(["rename", "drop_modifiers"])]
        drop = set(zip(bad["item"], bad["field"]))
        index = index[[(i, f) not in drop for i, f in zip(index["item"], index["field"])]]
        for i, f in drop:
            items.at[i, f] = None
    test = items.loc[test_idx].reset_index(drop=True)

    sims: dict[str, dict[str, np.ndarray]] = {}
    sims["jaccard"] = {f: np.array([jaccard(s, t) if isinstance(t, str) else np.nan for s, t in zip(test["text"], test[f])])
                       for f in tier1.KEEP + tier1.MOVE}

    sonar = np.load(work_path("phase1", "vec", "sonar.npy")).astype(np.float32)
    sims["sonar"] = cos_sims(by_field(index, sonar, test_idx))

    model = head.IntentHead()
    model.load_state_dict(torch.load(work_path("phase1", "head.pt")))
    model.eval()
    with torch.no_grad():
        z_all = model(torch.from_numpy(sonar)).numpy()
    for d in head.PREFIXES:
        sims[f"head{d}"] = cos_sims(by_field(index, z_all[:, :d], test_idx))
    shuffled = z_all[rng.permutation(len(z_all))]
    sims["shuffled"] = cos_sims(by_field(index, shuffled, test_idx))

    for name in OTS:
        path = work_path("phase1", "vec", f"{name}.npy")
        if path.exists():
            sims[name] = cos_sims(by_field(index, np.load(path).astype(np.float32), test_idx))

    for path in sorted(work_path("phase1", "vec").glob("ft_*.npy")):
        v = np.load(path).astype(np.float32)
        for d in head.PREFIXES:
            sims[f"{path.stem}{d}"] = cos_sims(by_field(index, v[:, :d], test_idx))

    pool = pd.read_parquet(work_path("pool", "pool.parquet"))
    encoders.fit_lsa(pool["text"].tolist())
    for name in ("wordavg", "lsa"):
        v = encoders.encode(name, texts)
        sims[name] = cos_sims(by_field(index, v, test_idx))

    results, wins = {}, {}
    for name, sim in sims.items():
        pw = tier1.pair_wins(sim)
        wins[name] = pw
        results[name] = {key: tier1.summarise(w, c, rng) for key, (w, c) in pw.items()}
        worst = min((v["win_rate"], k) for k, v in results[name].items() if k.startswith("move:") and v.get("n_pairs"))
        results[name]["worst_move"] = {"kind": worst[1], "win_rate": worst[0]}

    c = args.chosen
    diffs = {f"head{d}-sonar": tier1.paired_diff(*wins[f"head{d}"]["all"], *wins["sonar"]["all"], rng) for d in head.PREFIXES}
    diffs[f"{c}-sonar"] = tier1.paired_diff(*wins[c]["all"], *wins["sonar"]["all"], rng)
    best_ots = max((results[n]["all"]["win_rate"], n) for n in sims if n in OTS)[1] if any(n in sims for n in OTS) else None
    if best_ots:
        diffs[f"{c}-{best_ots}"] = tier1.paired_diff(*wins[c]["all"], *wins[best_ots]["all"], rng)

    h = results[c]
    verdict = {
        "jaccard_below_0.5": results["jaccard"]["all"]["win_rate"] < 0.5,
        "head_overall_ge_0.90": h["all"]["win_rate"] >= 0.90,
        "head_worst_move_ge_0.80": h["worst_move"]["win_rate"] >= 0.80,
        "chosen": c,
        "head_minus_sonar_ge_0.10_ci_above_0": diffs[f"{c}-sonar"]["diff"] >= 0.10 and diffs[f"{c}-sonar"]["ci"][0] > 0,
        "natural_pairs_ge_0.80": "pending (scored separately)",
    }
    by_domain = {}
    for dom in sorted(test["domain"].unique()):
        mask = (test["domain"] == dom).to_numpy()
        by_domain[dom] = {n: tier1.summarise(wins[n]["all"][0] * mask, wins[n]["all"][1] * mask, rng)["win_rate"]
                          for n in ("jaccard", "sonar", "head256", c, best_ots) if n}
        by_domain[dom]["n_sources"] = int(mask.sum())

    report = {"materials": args.materials, "test_sources": len(test), "results": results, "diffs": diffs, "verdict": verdict, "by_domain": by_domain}
    result_path(OUT, f"tier1-{args.materials}.json").write_text(json.dumps(report, indent=2) + "\n")

    kinds = ["all", *[f"move:{m}" for m in tier1.MOVE]]
    lines = ["encoder     " + " ".join(f"{k.replace('move:', ''):>17s}" for k in kinds)]
    for name, r in results.items():
        cells = [f"{r[k]['win_rate']:.3f}" if r[k].get("n_pairs") else "-" for k in kinds]
        lines.append(f"{name:11s} " + " ".join(f"{c:>17s}" for c in cells))
    lines += ["", json.dumps(diffs), json.dumps(verdict), json.dumps(by_domain)]
    result_path(OUT, f"tier1-{args.materials}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
