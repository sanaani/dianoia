"""Idea sequences for the first JEPA test (10/04/2026): ``jepa/units.jsonl``.

Each splitter training unit (``split/train.jsonl``; never an evaluation unit) becomes its reference ideas in order,
with one placeholder mapping per unit, as the encoder's training pairs use. Every 10th unit (after a seeded shuffle)
is held out. Each training unit's one-for-one paraphrase (``split/para.jsonl``, § 5) is added as a training unit
of its own (``kind`` "paraphrase"); held-out units get none. ``remote-jepa.py`` encodes them and trains the
next-idea predictor.
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


def main() -> None:
    units = [json.loads(l) for l in open(work_path("split", "train.jsonl"), encoding="utf-8")]
    units = [u for u in units if len(u.get("ideas") or []) >= 2]
    random.Random(17).shuffle(units)
    para = {}
    for l in open(work_path("split", "para.jsonl"), encoding="utf-8"):
        o = json.loads(l)
        para[o["uid"]] = o
    ph = bp.Placeholders()
    path = work_path("jepa", "units.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, u in enumerate(units):
            split = "test" if k % 10 == 0 else "train"
            versions = [("original", u)] + ([("paraphrase", para[u["uid"]])] if split == "train" and u["uid"] in para else [])
            for kind, v in versions:
                ideas = [i.rstrip(".") + "." for i in ph.apply([str(i) for i in v["ideas"]])]
                f.write(json.dumps({"uid": u["uid"], "kind": kind, "medium": u["medium"], "split": split,
                                    "ideas": ideas}, ensure_ascii=False) + "\n")
    print(json.dumps({"units": len(units), "paraphrased": sum(u["uid"] in para for u in units)}))


if __name__ == "__main__":
    main()
