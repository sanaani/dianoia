"""Tier 1's last bar: natural pairs, two real Enron sentences, no Haiku rewriting.

Anchors are sampled by seed from test-mailbox pool sentences. Each anchor's candidates are its nearest
test-mailbox sentences under three encoders of different kinds (SONAR, bge, LSA), so no one encoder
chooses the test on its own; the source paragraph, exact duplicates and near-copies (Jaccard >= 0.8)
are excluded. Haiku then labels each (anchor, candidate) pair, blind to which encoder found it:

- same_intent : same speech act and same proposition, up to names and wording
- related     : same topic, different intent
- twin        : shares most of its words, different intent
- unrelated

Score: for an anchor with at least one same_intent candidate and one related-or-twin candidate, a win
is the same_intent candidate nearer than the other. Candidates were found by nearness, so every pair
is already near under some encoder; the test is the ranking among near sentences, not nearness itself.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from dianoia import corpus, encoders, head, llm, tier1
from dianoia.paths import result_path, work_path

SEED = 20260930
PROMPT = """Label each pair of sentences from business email by how their INTENT relates. Intent = the \
speech act (request, commitment, report, question, proposal, approval, refusal, thanks) plus the claim or \
action (what is done, to what, by whom, how much). Names and wording do not matter.

- "same_intent": same speech act and same claim or action; a reader would act on them the same way.
- "related": same topic or situation, but a different act or claim.
- "twin": they share most of their words, but the act or claim differs.
- "unrelated": neither.

Return only a JSON array of objects {{"p": <pair number>, "label": <one of the four>}}.

{pairs}"""


def jaccard(a: str, b: str) -> float:
    wa, wb = {w.lower() for w in corpus.words(a)}, {w.lower() for w in corpus.words(b)}
    return len(wa & wb) / max(len(wa | wb), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=600)
    ap.add_argument("--per-encoder", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    gen = json.loads(result_path("phase-1-intent", "generation.json").read_text())
    test_boxes = set(gen["test_mailboxes"])
    pool = pd.read_parquet(work_path("pool", "pool.parquet"))
    in_test = pool["mailbox"].isin(test_boxes).to_numpy()
    tpool = pool[in_test].reset_index(drop=True)
    vec = {n: np.load(work_path("pool", "out", f"{n}.npy"), mmap_mode="r")[in_test].astype(np.float32) for n in ("sonar", "bge")}
    encoders.fit_lsa(pool["text"].tolist())
    vec["lsa"] = encoders.encode("lsa", tpool["text"].tolist())
    unit = {n: v / np.linalg.norm(v, axis=1, keepdims=True) for n, v in vec.items()}

    n_words = tpool["text"].map(lambda t: len(corpus.words(t)))
    eligible = np.flatnonzero(n_words.between(8, 35).to_numpy())
    rng = np.random.default_rng(SEED)
    anchors = np.sort(rng.choice(eligible, size=min(args.anchors, len(eligible)), replace=False))

    pairs = []
    for a in anchors:
        found: dict[int, set] = {}
        for name, u in unit.items():
            sims = u @ u[a]
            for j in np.argsort(-sims)[:50]:
                if j == a or tpool.at[j, "pid"] == tpool.at[a, "pid"] or jaccard(tpool.at[a, "text"], tpool.at[j, "text"]) >= 0.8:
                    continue
                found.setdefault(int(j), set()).add(name)
                if sum(name in s for s in found.values()) >= args.per_encoder:
                    break
        for j, by in found.items():
            pairs.append({"a": int(a), "b": j, "found_by": sorted(by)})

    prompts, chunks = [], []
    for s in range(0, len(pairs), 20):
        chunk = pairs[s : s + 20]
        body = "\n\n".join(f"PAIR {k}\nA: {tpool.at[p['a'], 'text']}\nB: {tpool.at[p['b'], 'text']}" for k, p in enumerate(chunk))
        prompts.append(PROMPT.format(pairs=body))
        chunks.append(chunk)

    def safe(p):
        try:
            return llm.ask_json(p)
        except Exception:
            return []

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        answers = list(ex.map(safe, prompts))
    for chunk, ans in zip(chunks, answers):
        got = {int(x["p"]): x["label"] for x in ans if isinstance(x, dict) and "p" in x}
        for k, p in enumerate(chunk):
            p["label"] = got.get(k)
    labelled = pd.DataFrame([p for p in pairs if p.get("label")])
    labelled["a_text"] = labelled["a"].map(tpool["text"])
    labelled["b_text"] = labelled["b"].map(tpool["text"])
    labelled.to_parquet(work_path("phase1", "natural-pairs.parquet"))

    # Score every encoder that has pool vectors, plus the head on SONAR.
    model = head.IntentHead()
    model.load_state_dict(torch.load(work_path("phase1", "head.pt")))
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(vec["sonar"])).numpy()
    spaces = {f"head{d}": z[:, :d] for d in head.PREFIXES}
    spaces["sonar"] = vec["sonar"]
    for n in ("minilm", "bge", "e5", "gtr", "qwen3"):
        path = work_path("pool", "out", f"{n}.npy")
        if path.exists():
            spaces[n] = np.load(path, mmap_mode="r")[in_test].astype(np.float32)
    for path in sorted(work_path("pool", "out").glob("ft_*.npy")):
        v = np.load(path, mmap_mode="r")[in_test].astype(np.float32)
        for d in head.PREFIXES:
            spaces[f"{path.stem}{d}"] = v[:, :d]
    spaces["lsa"] = vec["lsa"]
    for n, v in spaces.items():
        u = v / np.linalg.norm(v, axis=1, keepdims=True)
        labelled[n] = (u[labelled["a"]] * u[labelled["b"]]).sum(1)
    labelled["jaccard"] = [jaccard(a, b) for a, b in zip(labelled["a_text"], labelled["b_text"])]

    rng = np.random.default_rng(0)
    report = {"anchors": len(anchors), "pairs_labelled": len(labelled), "labels": labelled["label"].value_counts().to_dict(), "scores": {}}
    for n in [*spaces, "jaccard"]:
        wins, counts = [], []
        for _, g in labelled.groupby("a"):
            same = g.loc[g["label"] == "same_intent", n].to_numpy()
            other = g.loc[g["label"].isin(["related", "twin"]), n].to_numpy()
            if len(same) and len(other):
                w = (same[:, None] > other[None, :]) + 0.5 * (same[:, None] == other[None, :])
                wins.append(w.sum())
                counts.append(w.size)
        report["scores"][n] = tier1.summarise(np.array(wins, float), np.array(counts, float), rng)
    result_path("phase-1-intent", "natural-pairs.json").write_text(json.dumps(report, indent=2) + "\n")
    labelled.drop(columns=["a", "b"]).to_json(result_path("phase-1-intent", "natural-pairs.jsonl"), orient="records", lines=True, force_ascii=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
