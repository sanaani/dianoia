"""Turn Sonnet's merged ideas (``merge/items.jsonl``, ``merge-data.py``) into encoder pairs.

Per item (ideas X and Y, merges M1 and M2, edits E1 and E2 of M1), one placeholder mapping shared by every text:
- X and Y: same as M1 and M2; different ("edit") from the edit that changed their own part only;
- M1 and M2: same as X, Y and each other; different ("edit") from both edits.
Sonnet does not reliably keep the edits in the order asked, so each edit's part is found from the words it
replaced: in X's words only, in Y's only, or neither (then it is a negative for the merges only).
Test items go to ``pairs/test-merge.jsonl``; no test text is trained on.
``pairs/arm-M.jsonl`` is ``arm-S.jsonl`` plus ``merge-train``, shuffled.
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import random
import re
from pathlib import Path

from dianoia.paths import work_path

spec = importlib.util.spec_from_file_location("build_pairs", Path(__file__).with_name("build-pairs.py"))
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)

SRC = "merge-literal"


def words(t: str) -> list[str]:
    return re.findall(r"[\w']+", t.lower())


def part(merge: str, edit: str, x: str, y: str) -> str | None:
    """Which idea's part an edit of ``merge`` changed: "x", "y" or None when unclear."""
    a, b = words(merge), words(edit)
    gone = {w for op, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b).get_opcodes() if op in ("replace", "delete") for w in a[i1:i2]}
    if not gone:
        return None
    wx, wy = set(words(x)), set(words(y))
    in_x, in_y = gone <= wx, gone <= wy
    return "x" if in_x and not in_y else "y" if in_y and not in_x else None


def rows(o: dict, ph) -> list[dict]:
    t = ph.apply([o["x"], o["y"], *o["merges"], *o["edits"]])
    x, y, m, e = t[0], t[1], t[2:4], t[4:6]
    which = [part(m[0], k, x, y) for k in e]
    out = []
    for anchor, own in ((x, "x"), (y, "y")):
        out += [{"a": anchor, "b": mm, "rel": "same", "source": SRC, "kind": "paraphrase"} for mm in m]
        out += [{"a": anchor, "b": k, "rel": "diff", "source": SRC, "kind": "edit"} for k, w in zip(e, which) if w == own]
    for i, anchor in enumerate(m):
        out += [{"a": anchor, "b": b, "rel": "same", "source": SRC, "kind": "paraphrase"} for b in (x, y, m[1 - i])]
        out += [{"a": anchor, "b": k, "rel": "diff", "source": SRC, "kind": "edit"} for k in e]
    return [r for r in out if r["a"].strip().lower() != r["b"].strip().lower()]


def main() -> None:
    ph = bp.Placeholders()
    items = [json.loads(l) for l in open(work_path("merge", "items.jsonl"), encoding="utf-8")]
    train, test = [], []
    for o in items:
        (test if o["split"] == "test" else train).extend(rows(o, ph))
    banned = {r[k] for r in test for k in ("a", "b")}
    for name in ("test.jsonl", "test-contrast.jsonl", "test-lpara.jsonl", "test-spara.jsonl"):
        banned |= {json.loads(l)[k] for l in open(work_path("pairs", name), encoding="utf-8") for k in ("a", "b")}
    train = [r for r in train if r["a"] not in banned and r["b"] not in banned]
    for name, rs in (("merge-train", train), ("test-merge", test)):
        with open(work_path("pairs", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    arm = list(open(work_path("pairs", "arm-S.jsonl"), encoding="utf-8"))
    arm += [json.dumps(r, ensure_ascii=False) + "\n" for r in train]
    random.Random(16).shuffle(arm)
    with open(work_path("pairs", "arm-M.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(arm)
    parts = [part(o["merges"][0], k, o["x"], o["y"]) for o in items for k in o["edits"]]
    print(json.dumps({"items": len(items), "edit_parts": {str(p): parts.count(p) for p in ("x", "y", None)}, "train_rows": len(train), "test_rows": len(test),
                      "train_anchors": len({r["a"] for r in train}), "arm_M": len(arm)}))


if __name__ == "__main__":
    main()
