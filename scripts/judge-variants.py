"""A second, independent pass over the variants: does each one keep or change the source's intent?

Haiku sees a source sentence and its variants shuffled and unlabeled (it is not told which were meant
to keep or move), and labels each one ``same`` or ``different`` intent. A variant whose judged label
disagrees with its intended role (keep -> same, move -> different) is marked ``disputed``. The judge
never sees an encoder or a score, so the filter cannot favor any model. Tier 1 is then reported on
the original materials and on the materials with disputed variants removed.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from dianoia import llm, tier1
from dianoia.paths import result_path, work_path

PROMPT = """For each SOURCE sentence from business email, judge every numbered VARIANT: does it have \
the SAME intent as the source, or a DIFFERENT intent?

Intent = what the sentence does (reports, requests, commits, asks, proposes, approves, refuses, thanks) \
plus what it claims or asks for: the action, what it acts on, who does it to whom, how much, when, how \
certain. Wording, word order, active/passive voice, politeness, names swapped consistently for others \
of the same kind, and dropping decoration do NOT change intent. A different verb that means the same \
action (find/retrieve) does not change intent; one that changes the action (show/calculate) does.

Return only a JSON array: [{{"s": <source number>, "labels": {{"<variant number>": "same" or "different", ...}}}}, ...]

{blocks}"""

ROLE = {**{k: "same" for k in tier1.KEEP}, **{m: "different" for m in tier1.MOVE}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    val_boxes = set(json.loads(result_path("phase-1-intent", "train-head.json").read_text())["val_mailboxes"])
    role = np.where(items["split"] == "test", "test", np.where(items["mailbox"].isin(val_boxes), "val", "fit"))
    todo = items.index[np.isin(role, args.splits)].to_numpy()

    rng = np.random.default_rng(20260930)
    prompts, keys = [], []
    for s in range(0, len(todo), args.batch):
        blocks, kmap = [], {}
        for j, i in enumerate(todo[s : s + args.batch]):
            fields = [f for f in ROLE if isinstance(items.at[i, f], str) and items.at[i, f]]
            fields = list(rng.permutation(fields))
            lines = [f"SOURCE {j}: {items.at[i, 'text']}"]
            for n, f in enumerate(fields):
                lines.append(f"  {n}. {items.at[i, f]}")
                kmap[(j, str(n))] = (i, f)
            blocks.append("\n".join(lines))
        prompts.append(PROMPT.format(blocks="\n\n".join(blocks)))
        keys.append(kmap)

    def safe(p):
        try:
            return llm.ask_json(p)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        answers = list(ex.map(safe, prompts))
    rows = []
    for kmap, ans in zip(keys, answers):
        for a in ans if isinstance(ans, list) else []:
            for n, lab in (a.get("labels") or {}).items():
                key = (int(a.get("s", -1)), str(n))
                if key in kmap and lab in ("same", "different"):
                    i, f = kmap[key]
                    rows.append({"item": int(i), "field": f, "judged": lab, "intended": ROLE[f], "disputed": lab != ROLE[f]})
    judged = pd.DataFrame(rows)
    path = work_path("phase1", "judged.parquet")
    if path.exists():
        old = pd.read_parquet(path)
        judged = pd.concat([old[~old["item"].isin(judged["item"])], judged], ignore_index=True)
    judged.to_parquet(path)
    summary = judged.groupby("field")["disputed"].agg(["mean", "count"]).round(3)
    result_path("phase-1-intent", "judged-summary.txt").write_text(summary.to_string() + "\n")
    print(summary.to_string())
    print("overall disputed:", round(float(judged["disputed"].mean()), 3), "of", len(judged))


if __name__ == "__main__":
    main()
