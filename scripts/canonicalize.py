"""Stage 1 targets: the canonical sentence for pool sentences from fit mailboxes.

The canonical sentence is the idea with the wording removed: names of particular people, organizations,
places, products, deals and documents become typed, numbered placeholders (corpus-level names stay
literal, never people); decoration and non-essential modifiers are dropped; the speech act, the action,
what it acts on, who does it to whom, and every material number, amount and date are kept.

Checked mechanically as in phase 1 (placeholder format, bindings occur in the source and not in the
canonical). Sampled by seed from fit mailboxes only.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from dianoia import corpus, llm
from dianoia.paths import result_path, work_path

SEED = 20261001
PLACEHOLDER = re.compile(r"\[([A-Z]+)_(\d+)\]")
PROMPT = """Rewrite each sentence from business email as its CANONICAL form: the idea with the wording removed.

- Replace every NAME of a particular person, organization, place, product, deal or document with a typed, \
numbered placeholder, numbered by first appearance: [PERSON_1], [ORG_1], [PLACE_1], [PRODUCT_1], [DEAL_1], \
[DOC_1]. The same name always gets the same placeholder. Keep as written: I, we, you, common nouns, and \
these corpus-level names: {literal}.
- Keep the speech act (statement, request, question, commitment...), the action, what it acts on, who does \
it to whom, negation and certainty (not, might, must), and every number, amount and date exactly.
- Drop decoration: politeness, hedges, filler, non-essential place and manner phrases.
- Write plain, short, standard English. One sentence.

Return only a JSON array: [{{"i": <number>, "canonical": "...", "bindings": {{"[PERSON_1]": "exact text replaced", ...}}}}, ...]

{items}"""


def check(src: str, gen: dict) -> tuple[str, dict] | None:
    canon = gen.get("canonical")
    if not isinstance(canon, str) or not canon.strip():
        return None
    bindings = {k if k.startswith("[") else f"[{k}]": v for k, v in (gen.get("bindings") or {}).items()}
    if not all(PLACEHOLDER.fullmatch(k) for k in bindings):
        return None
    if set(PLACEHOLDER.findall(canon)) != {PLACEHOLDER.fullmatch(k).groups() for k in bindings}:
        return None
    if not all(isinstance(v, str) and v and v in src and v not in canon for v in bindings.values()):
        return None
    return canon.strip(), bindings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--phase1-items", action="store_true",
                    help="re-canonicalize every phase-1 item (all splits) instead of sampling pool sentences")
    args = ap.parse_args()
    pool = pd.read_parquet(work_path("pool", "pool.parquet"))
    gen = json.loads(result_path("phase-1-intent", "generation.json").read_text())
    val = set(json.loads(result_path("phase-1-intent", "train-head.json").read_text())["val_mailboxes"])
    fit = pool[~pool["mailbox"].isin(set(gen["test_mailboxes"]) | val)].copy()
    fit["n_words"] = fit["text"].map(lambda t: len(corpus.words(t)))
    fit = fit[fit["n_words"].between(5, 40)]
    rng = np.random.default_rng(SEED)
    sample = fit.iloc[np.sort(rng.choice(len(fit), size=min(args.n, len(fit)), replace=False))].reset_index(drop=True)
    if args.phase1_items:
        sample = pd.read_parquet(work_path("phase1", "items.parquet"))[["sid", "text", "mailbox", "split"]].reset_index()
    literal = [e["entity"] for e in json.loads(result_path("phase-0-corpus", "literal-entities.json").read_text())["literal_entities"] if e["label"] != "PERSON"]

    prompts, chunks = [], []
    for s in range(0, len(sample), args.batch):
        ch = sample.iloc[s : s + args.batch]
        present = sorted({e for t in ch["text"] for e in literal if re.search(rf"\b{re.escape(e)}\b", t)})
        prompts.append(PROMPT.format(literal=present or "none", items="\n".join(f"{k}. {t}" for k, t in enumerate(ch["text"]))))
        chunks.append(ch)

    def safe(p):
        try:
            return llm.ask_json(p, model=args.model)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        answers = list(ex.map(safe, prompts))
    rows, failed = [], 0
    for ch, ans in zip(chunks, answers):
        got = {int(a["i"]): a for a in ans if isinstance(a, dict) and str(a.get("i", "")).isdigit()} if isinstance(ans, list) else {}
        for k, (_, r) in enumerate(ch.iterrows()):
            ok = check(r["text"], got.get(k, {}))
            if ok:
                rows.append({"text": r["text"], "canonical": ok[0], "bindings": json.dumps(ok[1], ensure_ascii=False), "mailbox": r["mailbox"],
                             **({"item": int(r["index"])} if args.phase1_items else {})})
            else:
                failed += 1
    out = pd.DataFrame(rows)
    name = "canonical-phase1" if args.phase1_items else "canonical"
    out.to_parquet(work_path("ae", f"{name}.parquet"))
    summary = {"model": args.model, "sampled": len(sample), "kept": len(out), "failed": failed,
               "median_words_source": float(out["text"].map(lambda t: len(corpus.words(t))).median()),
               "median_words_canonical": float(out["canonical"].map(lambda t: len(corpus.words(t))).median())}
    result_path("phase-3-own-ae", f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
