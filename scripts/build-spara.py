"""Encoder pairs from the splitter's own output on paraphrased email (``spara/aligned.jsonl``, ``remote-spara.py``).

The end-to-end check showed the encoder sees ideas the splitter wrote from paraphrased text, which word and split
things differently from Sonnet's clean literal paraphrases. Per item, one placeholder mapping shared by every text:
- each original idea: same as its NLI-aligned idea from each paraphrase ("paraphrase"); different from
  rule edits of itself (negation, role swap, modality, number); the slanted versions' ideas are not used;
- each aligned paraphrase idea is an anchor too, with the original idea as its positive and the same negatives.
Test items (``contrast/items.jsonl`` split) go to ``pairs/test-spara.jsonl``; no test text is trained on.
``pairs/arm-S.jsonl`` is ``arm-P.jsonl`` plus ``spara-train``, shuffled.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

from dianoia.paths import work_path

spec = importlib.util.spec_from_file_location("build_pairs", Path(__file__).with_name("build-pairs.py"))
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)

SRC = "spara-email"


def rows(rec: dict, ph, rng: random.Random) -> list[dict]:
    texts = list(dict.fromkeys([*rec["orig"], *(p[1] for p in rec["pos"]), *(n[1] for n in rec["neg"])]))
    m = dict(zip(texts, ph.apply(texts)))
    out = []
    for o in rec["orig"]:
        pos = [m[p[1]] for p in rec["pos"] if p[0] == o]
        if not pos:
            continue
        a = m[o]
        # The slanted versions' matched ideas are not used: on inspection many were the same fact reworded or with a
        # detail dropped, which would push true paraphrases apart. Negatives are rule edits (and the batch).
        neg = []
        for kind, f in (("negation", lambda x: bp.negate(x, ph.nlp)), ("role_swap", bp.swap_roles),
                        ("modality", bp.change_modality), ("number", lambda x: bp.change_number(x, rng))):
            v = f(a)
            if v and v != a:
                neg.append((v, kind))
        for anchor, positive in [(a, p) for p in pos] + [(p, a) for p in pos]:
            out.append({"a": anchor, "b": positive, "rel": "same", "source": SRC, "kind": "paraphrase"})
        for anchor in [a, *pos]:
            out += [{"a": anchor, "b": n, "rel": "diff", "source": SRC, "kind": k} for n, k in neg]
    return [r for r in out if r["a"].strip().lower() != r["b"].strip().lower()]


def main() -> None:
    ph = bp.Placeholders()
    rng = random.Random(13)
    split = {f"c{o['id']}": o["split"] for o in map(json.loads, open(work_path("contrast", "items.jsonl"), encoding="utf-8"))}
    recs = [json.loads(l) for l in open(work_path("spara", "aligned.jsonl"), encoding="utf-8")]
    train, test = [], []
    for rec in recs:
        (test if split[rec["uid"]] == "test" else train).extend(rows(rec, ph, rng))
    banned = {r[k] for r in test for k in ("a", "b")}
    for name in ("test.jsonl", "test-contrast.jsonl", "test-lpara.jsonl"):
        banned |= {json.loads(l)[k] for l in open(work_path("pairs", name), encoding="utf-8") for k in ("a", "b")}
    train = [r for r in train if r["a"] not in banned and r["b"] not in banned]
    for name, rs in (("spara-train", train), ("test-spara", test)):
        with open(work_path("pairs", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    arm = list(open(work_path("pairs", "arm-P.jsonl"), encoding="utf-8"))
    arm += [json.dumps(r, ensure_ascii=False) + "\n" for r in train]
    random.Random(14).shuffle(arm)
    with open(work_path("pairs", "arm-S.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(arm)
    print(json.dumps({"items": len(recs), "train_rows": len(train), "test_rows": len(test),
                      "train_anchors": len({r["a"] for r in train}), "arm_S": len(arm)}))


if __name__ == "__main__":
    main()
