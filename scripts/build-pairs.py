"""Assemble the idea-encoder training and test pairs (pre-registered 10/02/2026). No LLM calls.

Every row is ``{"a", "b", "rel": "same" | "diff", "source", "kind"}``. "same" pairs are pulled together; "diff"
pairs are hard negatives for ``a``. Sources:
- MultiNLI (``pairs/mnli-scored.jsonl`` from ``remote-nli.py``): entailments are the ablation variable. Arm L takes
  a random sample of all entailments; arm B takes the both-ways pool (P >= 0.9 each way); both arms take the same
  count. Contradictions with P >= 0.9 are negatives in both arms. 2% of premises are held out (grouped by premise,
  since one premise carries several hypotheses).
- MRPC: human-labeled paraphrases (train + validation), no NLI filter. Its test split is held out.
- Phase 1 (Sonnet, already paid for): train-split variants. Same: paraphrase, register. Different: negation,
  role swap, modality, quantity, act change, type change. A variant judged against its intended label is dropped.
- Literal ideas (Sonnet, from ``split/train.jsonl``): each idea is paired with its stripped and disfluent forms
  (same) and with rule edits that change its meaning (different): negation, role swap, modality, number.

Placeholders: person names (spaCy) and identifiers (emails, URLs, phone numbers, letter-digit codes) become
numbered placeholders, consistently within a pair. Plain numbers stay literal.

Test material (``pairs/test.jsonl``): the check units' literal ideas (``split/eval.jsonl`` originals) with the
same rule edits as siblings, the phase-1 test variants, held-out MultiNLI and MRPC test. The 80 geometry ideas
stay in ``geometry/`` and are scored from there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter

import pandas as pd
import spacy
from datasets import load_dataset

from dianoia.paths import work_path

FILL = ["um", "uh", "like", "you know", "i mean", "so"]
AUX = r"is|are|was|were|will|would|can|could|should|must|has|have|had|does|do|did|may|might|shall"
MODAL = {"will": "might", "would": "might", "must": "might", "should": "might", "shall": "might", "can": "might",
         "might": "will", "may": "must", "could": "must"}
ID_PATTERNS = [
    ("Email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("Link", re.compile(r"\b(?:https?://|www\.)\S+?(?=[.,;:!?)\]]*(?:\s|$))")),
    ("Phone", re.compile(r"(?<!\w)(?:\+?1[ .-])?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b")),
    ("Code", re.compile(r"\b[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*\b")),
]


TITLED = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.? [A-Z][a-z]+(?: [A-Z][a-z]+)?")


def is_code(s: str) -> bool:
    """An identifier, not a word, number, unit ("90-day") or product name ("mp3", "x86", "gnome2").

    Either one run mixes letters with at least two digits ("A1B2", "PX77"), or capital letters are followed by a
    digit run ("ENW-1234"); and the whole thing is at least 4 characters.
    """
    if re.fullmatch(r"\d+[A-Za-z]{1,3}", s):  # "1980s", "4th", "10am", "5k"
        return False
    runs = re.split(r"[-_/]", s)
    mixed = any(re.search(r"[A-Za-z]", r) and len(re.findall(r"\d", r)) >= 2 for r in runs)
    prefixed = len(runs) > 1 and re.fullmatch(r"[A-Z]{2,}", runs[0]) and any(r.isdigit() for r in runs[1:])
    return len(s) >= 4 and bool(mixed or prefixed)


class Placeholders:
    """Numbered placeholders for names and identifiers, shared across the texts of one pair."""

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        ents = pd.read_parquet(work_path("corpus", "entities.parquet"))
        self.first = set(ents[(ents.label == "PERSON") & (ents.messages >= 20)].entity.str.lower()) - set(
            ents[ents.label != "PERSON"].entity.str.lower())

    def apply(self, texts: list[str]) -> list[str]:
        seen: dict[str, str] = {}
        count: Counter = Counter()

        def tag(kind: str, surface: str) -> str:
            key = surface.lower()
            if key not in seen:
                count[kind] += 1
                seen[key] = f"{'Person' if kind == 'PERSON' else kind} {count[kind]}"
            return seen[key]

        out = []
        for t in texts:
            for kind, pat in ID_PATTERNS:
                t = pat.sub(lambda m: tag(kind, m.group(0)) if kind != "Code" or is_code(m.group(0)) else m.group(0), t)
            t = TITLED.sub(lambda m: tag("PERSON", m.group(0)), t)
            # A person is two or more capitalized words, or one word known as an Enron person and never as anything else.
            ents = [e for e in self.nlp(t).ents if e.label_ == "PERSON" and all(w.text[:1].isupper() for w in e)
                    and (len(e) >= 2 or e.text.lower() in self.first)
                    and not re.fullmatch(r"[A-Z]|Person \d+|(Email|Link|Phone|Code) \d+", e.text)]
            names = [tag("PERSON", e.text) for e in ents]
            for e, name in sorted(zip(ents, names), key=lambda x: -x[0].start_char):
                t = t[: e.start_char] + name + t[e.end_char :]
            out.append(t)
        # A surface found in one text is replaced in every text of the pair, whatever its case or punctuation
        # ("Kathryn Shemin" -> also "kathryn shemin"; "Mrs. Wagantall" -> also "mrs wagantall").
        for key, name in sorted(seen.items(), key=lambda x: -len(x[0])):
            for form in {key, " ".join(re.sub(r"[^\w\s]", " ", key).split())}:
                if form:
                    pat = re.compile(rf"(?<!\w){re.escape(form)}(?!\w)", re.I)
                    out = [pat.sub(name, t) for t in out]
        return out


def stripped(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def disfluent(t: str, rng: random.Random) -> str:
    ws, new = t.split(), []
    if len(ws) >= 6 and rng.random() < 0.4:
        new += ws[:2] + ["--"]
    for w in ws:
        if rng.random() < 0.15:
            new.append(rng.choice(FILL) + ",")
        new.append(w)
    return " ".join(new)


CONTRACT = {"can't": "can", "cannot": "can", "won't": "will", "shan't": "shall"}


def negate(t: str, nlp) -> str | None:
    """Toggle negation in the clause after the stance marker ("that", "whether", "if"), using spaCy's parse.

    An existing "not" / "n't" / "cannot" is removed; otherwise "not" goes after the first auxiliary, or do-support
    is added to the main verb ("sends" -> "does not send"). Returns None when no verb is found.
    """
    doc = nlp(t)
    marks = [w.i for w in doc if w.lower_ in ("that", "whether", "if") and w.dep_ in ("mark", "dobj", "nsubj", "det") ]
    start = marks[0] + 1 if marks else 0
    toks = list(doc)[start:]
    for w in toks:
        if w.lower_ in CONTRACT:
            return t[: w.idx] + (CONTRACT[w.lower_].capitalize() if w.text[0].isupper() else CONTRACT[w.lower_]) + t[w.idx + len(w.text) :]
        if w.dep_ == "neg":
            if re.search(r"\b(either|any|anyone|anything|anymore|yet|at all)\b", t[w.idx :], re.I):
                return None  # removing "not" would strand a negative-polarity word
            if w.text.lower() == "n't":
                return t[: w.idx] + t[w.idx + len(w.text) :]
            return (t[: w.idx].rstrip() + " " + t[w.idx + len(w.text) :].lstrip()).replace("  ", " ")
    for w in toks:
        if w.dep_ in ("aux", "auxpass") or (w.lemma_ == "be" and w.pos_ == "AUX"):
            if w.lower_ == "can":
                return t[: w.idx + len(w.text)] + "not" + t[w.idx + len(w.text) :]
            return t[: w.idx + len(w.text)] + " not" + t[w.idx + len(w.text) :]
        if w.pos_ == "VERB" and w.tag_ in ("VBZ", "VBD", "VBP"):
            do = {"VBZ": "does", "VBD": "did", "VBP": "do"}[w.tag_]
            return t[: w.idx] + f"{do} not {w.lemma_}" + t[w.idx + len(w.text) :]
    return None


def swap_roles(t: str) -> str | None:
    """Swap the first two participants named by letter (A, B, C, ...) everywhere in the idea."""
    letters = list(dict.fromkeys(re.findall(r"\b[A-Z]\b(?!\.)", t)))
    letters = [x for x in letters if x not in ("I",)]
    if len(letters) < 2:
        return None
    x, y = letters[:2]
    return re.sub(rf"\b[{x}{y}]\b(?!\.)", lambda m: y if m.group(0) == x else x, t)


def change_modality(t: str) -> str | None:
    m = re.search(rf"\b({'|'.join(MODAL)})\b", t)
    if not m:
        return None
    w = MODAL[m.group(1)]
    return t[: m.start()] + w + t[m.end() :]


def change_number(t: str, rng: random.Random) -> str | None:
    # A placeholder's index ("Person 1") is not a quantity (fixed 10/03/2026; earlier data has some of these,
    # which change the person instead and are still meaning changes).
    nums = [m for m in re.finditer(r"(?<![\w.-])\d+(?![\w.-])", t)
            if not re.search(r"\b(Person|Email|Link|Phone|Code) $", t[: m.start()])]
    if not nums:
        return None
    m = rng.choice(nums)
    n = int(m.group(0))
    new = n + rng.choice([1, 2, 3]) if n < 10 else (n * 2 if n < 1900 or n > 2100 else n + rng.choice([1, 2]))
    return t[: m.start()] + str(new) + t[m.end() :]


def held_out(key: str, pct: int = 2) -> bool:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < pct


def literal_rows(ideas: list[str], rng: random.Random, source: str, ph: Placeholders) -> list[dict]:
    """Variants are derived from the idea after its placeholders are in, so every version carries the same ones."""
    nlp = ph.nlp
    rows = []
    for i in ideas:
        i = ph.apply([i])[0]
        rows += [{"a": i, "b": stripped(i), "rel": "same", "source": source, "kind": "stripped"},
                 {"a": i, "b": disfluent(i, rng), "rel": "same", "source": source, "kind": "disfluent"}]
        for kind, f in (("negation", lambda x: negate(x, nlp)), ("role_swap", swap_roles), ("modality", change_modality),
                        ("number", lambda x: change_number(x, rng))):
            v = f(i)
            if v and v != i:
                rows.append({"a": i, "b": v, "rel": "diff", "source": source, "kind": kind})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", type=float, default=0.9)
    args = ap.parse_args()
    rng = random.Random(7)
    ph = Placeholders()
    shared, test = [], []

    # MultiNLI
    mnli = [json.loads(l) for l in open(work_path("pairs", "mnli-scored.jsonl"), encoding="utf-8")]
    for r in mnli:
        r["held"] = held_out(r["p"])
    ent = [r for r in mnli if r["label"] == 0 and not r["held"] and r["fwd"] >= args.bar]
    both = [r for r in ent if r["bwd"] >= args.bar]
    n = len(both)
    arm = {"L": rng.sample(ent, n), "B": both}
    con = [r for r in mnli if r["label"] == 2 and not r["held"] and r["contra"] >= args.bar]
    con = rng.sample(con, min(n, len(con)))  # as many contradictions as entailments per arm
    shared += [{"a": r["p"], "b": r["h"], "rel": "diff", "source": "mnli", "kind": "contradiction"} for r in con]
    test += [{"a": r["p"], "b": r["h"], "rel": "same" if r["label"] == 0 else "diff", "source": "mnli",
              "kind": "both_ways" if r["label"] == 0 and r["bwd"] >= args.bar else ("lossy" if r["label"] == 0 else "contradiction")}
             for r in mnli if r["held"] and (r["label"] == 0 and r["fwd"] >= args.bar or r["label"] == 2 and r["contra"] >= args.bar)]

    # MRPC
    def detok(t: str) -> str:
        t = re.sub(r" ([,.;:!?%)])", r"\1", t)
        t = re.sub(r"([($]) ", r"\1", t)
        return re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", t).replace("`` ", '"').replace(" ''", '"')

    for split, dest in (("train", shared), ("validation", shared), ("test", test)):
        for r in load_dataset("nyu-mll/glue", "mrpc", split=split):
            if r["label"] == 1:
                dest.append({"a": detok(r["sentence1"]), "b": detok(r["sentence2"]), "rel": "same", "source": "mrpc", "kind": "paraphrase"})

    # Phase 1 (Sonnet variants, already paid for)
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    judged = pd.read_parquet(work_path("phase1", "judged.parquet"))
    verdict = {(int(r.item), r.field): r.judged for r in judged.itertuples()}
    fields = {"paraphrase": "same", "register": "same", "negation": "diff", "role_swap": "diff", "modality": "diff",
              "quantity": "diff", "act_change": "diff", "type_change": "diff"}
    for k, r in enumerate(items.itertuples()):
        dest = test if r.split == "test" else shared
        for f, rel in fields.items():
            v = getattr(r, f)
            if not isinstance(v, str) or not v.strip() or re.search(r"\[[A-Za-z_]+[ _]?\d*\]", v + r.text):
                continue  # some Sonnet variants carry their own bracketed placeholders; the original does not
            j = verdict.get((k, f))
            if j and (j == "same") != (rel == "same"):
                continue
            dest.append({"a": r.text, "b": v, "rel": rel, "source": "phase1", "kind": f})

    # Literal ideas (Sonnet, already paid for): training units, and the check units' originals as test
    train_ideas = sorted({i for l in open(work_path("split", "train.jsonl"), encoding="utf-8") for i in json.loads(l)["ideas"]})
    shared += literal_rows(train_ideas, rng, "literal", ph)
    test_ideas = sorted({i for l in open(work_path("split", "eval.jsonl"), encoding="utf-8")
                         for r in [json.loads(l)] if r.get("kind", "original") == "original" for i in r["ideas"]})
    test += literal_rows(test_ideas, rng, "literal", ph)
    leak = set(test_ideas) & set(train_ideas)

    def finish(rows, banned=frozenset()):
        out = []
        for r in rows:
            r["a"], r["b"] = ph.apply([r["a"], r["b"]])
            if r["a"] in banned or r["b"] in banned:
                continue
            if r["a"].strip().lower() != r["b"].strip().lower() or r["kind"] in ("stripped",):
                out.append(r)
        return out

    test = finish(test)
    banned = frozenset({r["a"] for r in test} | {r["b"] for r in test})  # no test text is ever trained on
    shared = finish(shared, banned)
    work_path("pairs").mkdir(parents=True, exist_ok=True)
    for name, rows in (("test", test),):
        with open(work_path("pairs", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"shared": Counter(f"{r['source']}/{r['kind']}" for r in shared), "test": Counter(f"{r['source']}/{r['kind']}" for r in test),
             "mnli_entail_train": len(ent), "mnli_both_ways": n, "leaked_test_ideas": len(leak)}
    for a, picked in arm.items():
        rows = finish([{"a": r["p"], "b": r["h"], "rel": "same", "source": "mnli", "kind": "entail"} for r in picked], banned) + shared
        rng.shuffle(rows)
        with open(work_path("pairs", f"arm-{a}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"arm_{a}"] = len(rows)
    print(json.dumps(stats, indent=1, default=dict))
    (work_path("pairs", "stats.json")).write_text(json.dumps(stats, indent=1, default=dict), encoding="utf-8")


if __name__ == "__main__":
    main()
