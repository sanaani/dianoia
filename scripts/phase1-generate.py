"""Phase 1 materials: per source sentence, its canonical form, bindings, intent fields and variants.

Sources are sampled by seed from mailboxes split 80/20 into train and test (PREREGISTRATION.md
§ Split). Haiku writes everything in batches; every field is then checked mechanically, and a field
that fails its rule is dropped and counted, never repaired by hand.

Rules (word Jaccard against the source):
- paraphrase <= 0.5; rename and register differ from the source;
- negation, role_swap, modality, quantity, type_change >= 0.6; predicate_change >= 0.5;
  act_change >= 0.3 (a new speech act restructures the sentence);
- canonical: every placeholder is ``[TYPE_n]``, every binding value occurs in the source, and no
  binding value survives in the canonical text.
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

SEED = 20260930
ACTS = ["request", "commit", "inform", "question", "propose", "approve", "reject", "thank", "other"]
KEEP = ["paraphrase", "rename", "register", "drop_modifiers"]
MOVE = ["act_change", "role_swap", "negation", "modality", "predicate_change", "type_change", "quantity"]
TWIN_MIN = {"negation": 0.6, "role_swap": 0.6, "modality": 0.6, "quantity": 0.6, "type_change": 0.6,
            "predicate_change": 0.5, "act_change": 0.3}
PLACEHOLDER = re.compile(r"\[([A-Z]+)_(\d+)\]")

PROMPT = """You prepare training data for an encoder of INTENT (what a sentence does and claims), not wording.
Below are {n} sentences from Enron business email. For EACH sentence return one JSON object:

"i": its number.
"usable": false if it is a fragment, a greeting, a heading, or has no clear claim or act; else true.
"domain": one of "business", "news", "sports", "personal", "other" (news = pasted press or newsletter text).
"speech_act": one of {acts}.
"canonical": the sentence with every NAME of a particular person, organization, place, product, deal \
or document replaced by a typed, numbered placeholder, numbered by first appearance: [PERSON_1], \
[PERSON_2], [ORG_1], [PLACE_1], [DEAL_1], [DOC_1], [PRODUCT_1], [DATE_1] for specific dates, \
[AMOUNT_1] for specific money amounts. The same name always gets the same placeholder. KEEP AS \
WRITTEN: first and second person (I, we, you), common nouns, and these corpus-level names: {literal}. \
Also drop modifiers that are not essential to the claim (where, how, decorative adjectives).
"bindings": object mapping each placeholder to the exact text it replaced.
"intent": {{"act": same as speech_act, "predicate": main verb phrase, "object": what it acts on, \
"agent": who acts (placeholder, I/you/we, or a common noun), "recipient": who receives or null, \
"quantities": list of material numbers/amounts/dates as written, possibly empty}}.
ALL variants below are rewrites of the ORIGINAL sentence, with its real names, never placeholders.
KEEP variants (same intent):
"paraphrase": same meaning, rebuilt from scratch: new verbs, new nouns, new structure. Reuse at most a third of the original's words, apart from names and numbers, which stay unchanged.
"rename": replace every name NOT in the corpus-level list with a different plausible name of the same \
type, consistently; null if there are none.
"register": same content, clearly more or less polite/formal ("Could you please send" <-> "Send").
"drop_modifiers": remove non-essential modifiers (place, manner, decoration), keep the core claim; \
null if there are none.
MOVE variants (different intent). Change ONLY the words needed for the change; every other word stays exactly as in the original, in the same order:
"act_change": same content, different speech act (request -> commitment, report -> question...).
"role_swap": swap who acts and who receives, or the direction; null if symmetric or impossible.
"negation": reverse the claim.
"modality": change certainty or obligation (will -> might, must -> may).
"predicate_change": change the main verb or its object so a different thing is done.
"type_change": replace one entity with an entity of a different type (a colleague -> a regulator); \
null if impossible.
"quantity": change a number, amount or date materially; null if there is none.

Hard rules: every variant is ONE complete sentence covering the WHOLE original (never truncate or summarize it); no variant contains a [PLACEHOLDER]; a MOVE variant differs from the original only in the few words its change needs.

Example. Original: "Please send Jeff the revised Transco contract by Friday."
paraphrase: "Could the updated Transco agreement reach Jeff before the week ends?"
negation: "Please do not send Jeff the revised Transco contract by Friday."
role_swap: "Please ask Jeff to send you the revised Transco contract by Friday."
modality: "You might send Jeff the revised Transco contract by Friday."
predicate_change: "Please send Jeff the revised Transco invoice by Friday."
quantity: "Please send Jeff the revised Transco contract by next month."

Return only a JSON array of {n} objects, in order.

Sentences:
{items}"""


def jaccard(a: str, b: str) -> float:
    wa, wb = {w.lower() for w in corpus.words(a)}, {w.lower() for w in corpus.words(b)}
    return len(wa & wb) / max(len(wa | wb), 1)


def split_mailboxes(mailboxes: list[str]) -> set[str]:
    boxes = sorted(set(mailboxes))
    rng = np.random.default_rng(SEED)
    return set(rng.choice(boxes, size=round(0.2 * len(boxes)), replace=False).tolist())


def check(src: str, gen: dict, drops: dict) -> dict | None:
    if not gen.get("usable") or gen.get("speech_act") not in ACTS:
        drops["unusable"] += 1
        return None
    row = {"speech_act": gen["speech_act"], "domain": gen.get("domain", "other"), "intent": json.dumps(gen.get("intent", {}), ensure_ascii=False)}
    canon = gen.get("canonical") or ""
    bindings = {k if k.startswith("[") else f"[{k}]": v for k, v in (gen.get("bindings") or {}).items()}
    ok = bool(canon) and all(PLACEHOLDER.fullmatch(k) for k in bindings) and \
        set(PLACEHOLDER.findall(canon)) == {PLACEHOLDER.fullmatch(k).groups() for k in bindings if PLACEHOLDER.fullmatch(k)} and \
        all(isinstance(v, str) and v and v in src and v not in canon for v in bindings.values())
    if not ok:
        drops["canonical_rule"] += 1
        return None
    row["canonical"], row["bindings"] = canon, json.dumps(bindings, ensure_ascii=False)
    for k in KEEP:
        t = gen.get(k) if isinstance(gen.get(k), str) else None
        good = bool(t) and corpus.normalise(t) != corpus.normalise(src)
        if k == "paraphrase":
            good = good and jaccard(src, t) <= 0.5
        if t and not good:
            drops[f"{k}_rule"] += 1
        row[k] = t if good else None
    if row["paraphrase"] is None:
        drops["no_paraphrase"] += 1
        return None
    for k in MOVE:
        t = gen.get(k) if isinstance(gen.get(k), str) else None
        good = bool(t) and corpus.normalise(t) != corpus.normalise(src) and jaccard(src, t) >= TWIN_MIN[k]
        if t and not good:
            drops[f"{k}_rule"] += 1
        row[k] = t if good else None
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    sents = pd.read_parquet(work_path("corpus", "sentences.parquet"))
    emails = pd.read_parquet(work_path("corpus", "emails.parquet"), columns=["message_id", "mailbox"])
    sents = sents.merge(emails, on="message_id")
    test_boxes = split_mailboxes(sents["mailbox"].tolist())
    literal = [e["entity"] for e in json.loads(result_path("phase-0-corpus", "literal-entities.json").read_text())["literal_entities"] if e["label"] != "PERSON"]

    pool = sents[~sents["boilerplate"]].copy()
    pool["n_words"] = pool["text"].map(lambda t: len(corpus.words(t)))
    pool = pool[pool["n_words"].between(8, 35)].drop_duplicates("text")
    pool["split"] = np.where(pool["mailbox"].isin(test_boxes), "test", "train")
    rng = np.random.default_rng(SEED)
    parts = []
    for split, n in (("test", args.test), ("train", args.train)):
        p = pool[pool["split"] == split]
        parts.append(p.iloc[np.sort(rng.choice(len(p), size=min(n, len(p)), replace=False))])
    sample = pd.concat(parts, ignore_index=True)

    prompts, groups = [], []
    for start in range(0, len(sample), args.batch):
        chunk = sample.iloc[start : start + args.batch]
        present = sorted({e for t in chunk["text"] for e in literal if e in t})
        items = "\n".join(f"{k}. {t}" for k, t in enumerate(chunk["text"]))
        prompts.append(PROMPT.format(n=len(chunk), acts=ACTS, literal=present or "none", items=items))
        groups.append(chunk)

    drops = {k: 0 for k in ["unusable", "canonical_rule", "no_paraphrase", "missing", "failed_call"]}
    drops.update({f"{k}_rule": 0 for k in KEEP + MOVE})
    rows = []

    def safe(p):
        try:
            return llm.ask_json(p)
        except Exception as exc:  # a failed batch is counted, not repaired by hand
            return {"error": str(exc)[:200]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool_ex:
        answers = list(pool_ex.map(safe, prompts))
    for chunk, answer in zip(groups, answers):
        if isinstance(answer, dict) or len(answer) != len(chunk):
            drops["failed_call" if isinstance(answer, dict) else "missing"] += len(chunk)
            continue
        for (_, src), gen in zip(chunk.iterrows(), answer):
            row = check(src["text"], gen, drops)
            if row:
                rows.append({"sid": src["sid"], "pid": src["pid"], "mailbox": src["mailbox"], "split": src["split"], "text": src["text"], **row})

    items = pd.DataFrame(rows)
    items.to_parquet(work_path("phase1", "items.parquet"))
    summary = {"sampled": {"train": int((sample["split"] == "train").sum()), "test": int((sample["split"] == "test").sum())},
               "kept": {"train": int((items["split"] == "train").sum()), "test": int((items["split"] == "test").sum())},
               "test_mailboxes": sorted(test_boxes), "drops": drops,
               "variant_counts": {k: int(items[k].notna().sum()) for k in KEEP + MOVE},
               "speech_acts": items["speech_act"].value_counts().to_dict(),
               "domains": items["domain"].value_counts().to_dict()}
    result_path("phase-1-intent", "generation.json").write_text(json.dumps(summary, indent=2) + "\n")
    items[items["split"] == "test"].to_json(result_path("phase-1-intent", "test-items.jsonl"), orient="records", lines=True, force_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "test_mailboxes"}, indent=2))


if __name__ == "__main__":
    main()
