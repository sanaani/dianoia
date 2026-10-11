"""Turn Sonnet's literal-idea paraphrases and minimal edits (``lpara/items.jsonl``) into encoder pairs.

Per item, one placeholder mapping shared by all six texts:
- original: same as each paraphrase; different ("edit") from each minimal edit;
- paraphrase[0]: same as paraphrases 1 and 2; different ("edit") from each minimal edit.
Train items go to ``pairs/lpara-train.jsonl``, test items to ``pairs/test-lpara.jsonl``; no test text is trained on.
``pairs/arm-P.jsonl`` is ``arm-C.jsonl`` (arm L2 plus the attitude-contrast pairs) plus ``lpara-train``, shuffled.
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


def rows(o: dict, ph) -> list[dict]:
    t = ph.apply([o["text"], *o["paraphrases"], *o["edits"]])
    orig, para, edits = t[0], t[1:4], t[4:6]
    src = "lpara-literal"
    out = [{"a": orig, "b": p, "rel": "same", "source": src, "kind": "paraphrase"} for p in para]
    out += [{"a": orig, "b": e, "rel": "diff", "source": src, "kind": "edit"} for e in edits]
    out += [{"a": para[0], "b": p, "rel": "same", "source": src, "kind": "paraphrase"} for p in para[1:]]
    out += [{"a": para[0], "b": e, "rel": "diff", "source": src, "kind": "edit"} for e in edits]
    return [r for r in out if r["a"].strip().lower() != r["b"].strip().lower()]


def main() -> None:
    ph = bp.Placeholders()
    items = [json.loads(l) for l in open(work_path("lpara", "items.jsonl"), encoding="utf-8")]
    train, test = [], []
    for o in items:
        (test if o["split"] == "test" else train).extend(rows(o, ph))
    banned = {r[k] for r in test for k in ("a", "b")}
    banned |= {json.loads(l)[k] for l in open(work_path("pairs", "test.jsonl"), encoding="utf-8") for k in ("a", "b")}
    train = [r for r in train if r["a"] not in banned and r["b"] not in banned]
    for name, rs in (("lpara-train", train), ("test-lpara", test)):
        with open(work_path("pairs", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    arm = [l for l in open(work_path("pairs", "arm-C.jsonl"), encoding="utf-8")]
    arm += [json.dumps(r, ensure_ascii=False) + "\n" for r in train]
    random.Random(12).shuffle(arm)
    with open(work_path("pairs", "arm-P.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(arm)
    print(json.dumps({"items": len(items), "train_rows": len(train), "test_rows": len(test), "arm_P": len(arm)}))


if __name__ == "__main__":
    main()
