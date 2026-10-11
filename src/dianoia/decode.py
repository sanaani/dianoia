"""Decoders: a vector in, words out, without ever seeing the sentence the vector came from.

The only things a decoder may use are the target vector, the encoder, and a pool of *other* corpus
sentences with their vectors. ``exclusion`` removes from the pool anything that would let a decoder
copy the source instead of reading the vector: the source itself, any sentence with the same
normalised text, any sentence from the same paragraph, and any sentence sharing half its words.

- ``nearest``  the nearest admissible pool sentence. The floor: no generation at all.
- ``llm_knn``  Haiku sees the k nearest admissible sentences and their cosines to the target and
               writes the sentence it judges to sit at the target. One shot.
- ``llm_search`` ``llm_knn`` followed by rounds of propose, re-encode, score, revise: Haiku sees its
               own candidates with their cosines and improves them. Black-box inversion by search,
               the way vec2text's corrector works, with the model as the proposer.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dianoia import corpus, encoders, llm

KNN_PROMPT = """A sentence from a business email was turned into a meaning vector. You cannot see it. \
You can see other sentences from the same email corpus that sit NEAR it in meaning space, with their \
similarity to it (1.0 = identical meaning). Higher-similarity neighbours matter more, but no neighbour \
is the hidden sentence, and each may share only part of its idea.

For each target below, infer what the hidden sentence most likely says: its topic, its claim, its \
speech act (request, report, promise...). Then write {m} candidate sentences, each a plausible \
reconstruction, most likely first. Write ordinary email prose, one sentence each.

Return a JSON array with one object per target: {{"target": <id>, "candidates": [..{m} strings..]}}.

{blocks}"""

SEARCH_PROMPT = """You are reconstructing hidden sentences from their meaning vectors by search. For each \
target you see the nearest corpus sentences (neighbours, which are NOT the hidden sentence) and every \
candidate tried so far with its similarity to the hidden sentence (1.0 = identical meaning). \
Similarity is the only feedback: study which candidates scored higher and why, and move toward them.

For each target write {m} NEW candidates that you expect to score higher than every candidate so far. \
Vary them: change the claim, the actors, the speech act, the specificity. One sentence each.

Return a JSON array with one object per target: {{"target": <id>, "candidates": [..{m} strings..]}}.

{blocks}"""


def exclusion(target_text: str, target_pid: str, pool_texts: list[str], pool_pids: list[str]) -> np.ndarray:
    """Boolean mask over the pool: True where a sentence may be shown to the decoder."""
    norm = corpus.normalise(target_text)
    words = {w.lower() for w in corpus.words(target_text)}
    keep = np.ones(len(pool_texts), dtype=bool)
    for i, (t, pid) in enumerate(zip(pool_texts, pool_pids)):
        if pid == target_pid or corpus.normalise(t) == norm:
            keep[i] = False
            continue
        w = {x.lower() for x in corpus.words(t)}
        if words and len(words & w) / max(len(words | w), 1) >= 0.5:
            keep[i] = False
    return keep


def neighbours(target: np.ndarray, pool_vecs: np.ndarray, admissible: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sims = pool_vecs @ target
    sims = np.where(admissible, sims, -np.inf)
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def _block(tid: int, neigh: list[tuple[str, float]], tried: list[tuple[str, float]] | None = None) -> str:
    lines = [f"TARGET {tid}", "neighbours:"]
    lines += [f"  {s:.3f}  {t}" for t, s in neigh]
    if tried is not None:
        lines.append("candidates so far (best first):")
        lines += [f"  {s:.3f}  {t}" for t, s in sorted(tried, key=lambda x: -x[1])]
    return "\n".join(lines)


def _parse(answer, ids: list[int]) -> dict[int, list[str]]:
    got = {int(a["target"]): [c for c in a.get("candidates", []) if isinstance(c, str) and c.strip()]
           for a in answer if isinstance(a, dict) and str(a.get("target", "")).lstrip("-").isdigit()}
    return {i: got.get(i, []) for i in ids}


def llm_decode(
    enc: str,
    targets: np.ndarray,
    neighs: list[list[tuple[str, float]]],
    *,
    rounds: int = 3,
    m: int = 5,
    batch: int = 8,
    model: str = "haiku",
    encode_fn=None,
) -> list[dict]:
    """Return, per target, the one-shot best, the searched best and the full trace."""
    n = len(targets)
    tried: list[list[tuple[str, float]]] = [[] for _ in range(n)]
    trace: list[list[dict]] = [[] for _ in range(n)]

    encode = encode_fn or (lambda texts: encoders.encode(enc, texts))

    def score_all(pending: list[tuple[int, list[str]]], rnd: int) -> None:
        """Encode every candidate of a round in one call, then record each one's cosine to its target."""
        flat = [c for _, cands in pending for c in cands]
        if not flat:
            return
        v = encode(flat)
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        k = 0
        for i, cands in pending:
            t = targets[i] / max(float(np.linalg.norm(targets[i])), 1e-9)
            for c in cands:
                s = float(v[k] @ t)
                k += 1
                tried[i].append((c, s))
                trace[i].append({"round": rnd, "text": c, "cos": round(s, 4)})

    for rnd in range(rounds + 1):
        prompts, groups = [], []
        for start in range(0, n, batch):
            ids = list(range(start, min(start + batch, n)))
            if rnd == 0:
                blocks = "\n\n".join(_block(i, neighs[i]) for i in ids)
                prompts.append(KNN_PROMPT.format(m=m, blocks=blocks))
            else:
                blocks = "\n\n".join(_block(i, neighs[i][:5], tried[i]) for i in ids)
                prompts.append(SEARCH_PROMPT.format(m=m, blocks=blocks))
            groups.append(ids)
        def safe(p):
            try:
                return llm.ask_json(p, model=model)
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=4) as ex:
            answers = list(ex.map(safe, prompts))
        pending = []
        for ids, answer in zip(groups, answers):
            pending += [(i, cands[:m]) for i, cands in _parse(answer, ids).items()]
        score_all(pending, rnd)
        if rnd == 0:
            one_shot = [max(t, key=lambda x: x[1]) if t else ("", 0.0) for t in tried]
            first = [trace[i][0]["text"] if trace[i] else "" for i in range(n)]

    out = []
    for i in range(n):
        best = max(tried[i], key=lambda x: x[1]) if tried[i] else ("", 0.0)
        out.append({"first": first[i], "one_shot_best": one_shot[i][0], "search_best": best[0], "trace": trace[i]})
    return out


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
