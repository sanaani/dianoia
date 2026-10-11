"""Turn Sonnet's paraphrase and attitude-contrast items (``contrast/items.jsonl``) into encoder pairs.

Per item, with one placeholder mapping shared by all seven texts:
- original: same as each paraphrase; different ("slant") from each approving and disapproving version;
- approving[0]: same as approving[1]; different ("contrast") from each disapproving version;
- disapproving[0]: same as disapproving[1]; different ("contrast") from each approving version.
Train items go to ``pairs/contrast-train.jsonl`` and test items to ``pairs/test-contrast.jsonl``. No text from a
test item appears in training.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from dianoia.paths import work_path

spec = importlib.util.spec_from_file_location("build_pairs", Path(__file__).with_name("build-pairs.py"))
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)


def rows(o: dict, ph) -> list[dict]:
    texts = [o["text"], *o["paraphrases"], *o["approving"], *o["disapproving"]]
    t = ph.apply(texts)
    orig, para, appr, dis = t[0], t[1:3], t[3:5], t[5:7]
    src = f"contrast-{o['source']}"
    out = [{"a": orig, "b": p, "rel": "same", "source": src, "kind": "paraphrase"} for p in para]
    out += [{"a": orig, "b": s, "rel": "diff", "source": src, "kind": "slant"} for s in appr + dis]
    out += [{"a": appr[0], "b": appr[1], "rel": "same", "source": src, "kind": "paraphrase"}]
    out += [{"a": appr[0], "b": d, "rel": "diff", "source": src, "kind": "contrast"} for d in dis]
    out += [{"a": dis[0], "b": dis[1], "rel": "same", "source": src, "kind": "paraphrase"}]
    out += [{"a": dis[0], "b": a, "rel": "diff", "source": src, "kind": "contrast"} for a in appr]
    return [r for r in out if r["a"].strip().lower() != r["b"].strip().lower()]


def main() -> None:
    ph = bp.Placeholders()
    items = [json.loads(l) for l in open(work_path("contrast", "items.jsonl"), encoding="utf-8")]
    train, test = [], []
    for o in items:
        (test if o["split"] == "test" else train).extend(rows(o, ph))
    banned = {r[k] for r in test for k in ("a", "b")}
    train = [r for r in train if r["a"] not in banned and r["b"] not in banned]
    for name, rs in (("contrast-train", train), ("test-contrast", test)):
        with open(work_path("pairs", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(items), "train_rows": len(train), "test_rows": len(test)}))


if __name__ == "__main__":
    main()
