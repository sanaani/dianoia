"""List every phase-1 text once, for the GPU encoder: sources, canonical forms and all variants.

Writes ``work/phase1/texts.jsonl`` (one text per line, the order the vectors come back in) and
``work/phase1/index.parquet`` (item row, field, line number).
"""

from __future__ import annotations

import json

import pandas as pd

from dianoia.paths import work_path

FIELDS = ["text", "canonical", "paraphrase", "rename", "register", "drop_modifiers",
          "act_change", "role_swap", "negation", "modality", "predicate_change", "type_change", "quantity"]


def main() -> None:
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    rows, texts = [], []
    for i, item in items.iterrows():
        for f in FIELDS:
            t = item.get(f)
            if isinstance(t, str) and t.strip():
                rows.append({"item": i, "field": f, "line": len(texts)})
                texts.append(t)
    pd.DataFrame(rows).to_parquet(work_path("phase1", "index.parquet"))
    with open(work_path("phase1", "texts.jsonl"), "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(len(items), "items,", len(texts), "texts")


if __name__ == "__main__":
    main()
