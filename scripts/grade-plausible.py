"""Is the predicted next idea a plausible next idea? Blind grading of retrieved ideas (10/05/2026).

Median rank only asks whether the one true next idea scores high; a conversation has many reasonable next ideas, so
the rank has an unknown ceiling. Here Sonnet sees a conversation's last ideas and, shuffled and unlabeled, the true next
idea, the pool idea nearest the predictor's guess, the nearest to the "last idea" and "context mean" baselines, and a
random pool idea (``remote-jepa.py --dump-retrieved``). It scores each 1-5 as a next idea; it never learns which is
which, and identical texts are shown once. Input is public-corpus test data only (chat, meeting, wiki).

    python scripts/grade-plausible.py jepa/J9A-P/retrieved.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dianoia import llm
from dianoia.paths import work_path

GUESSES = ("true", "predictor", "last", "context_mean", "random")
PROMPT = """Each block is the end of a conversation (chat, meeting or wiki talk page), written as one short "idea" per \
turn; speakers are letters. For every numbered CANDIDATE, score how plausible it is as the NEXT idea in that \
conversation: 5 = a natural next idea, 4 = plausible, 3 = possible but a stretch, 2 = unlikely, 1 = does not fit. \
Judge fit with the conversation, not whether it matches some hidden answer. Several candidates may be good.

Return only a JSON array: [{{"b": <block number>, "scores": {{"<candidate number>": <1-5>, ...}}}}, ...]

{blocks}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("retrieved")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="sonnet")
    # T1 (10/05/2026): remote-textnext.py adds the text model's candidates, "text_gen" and "text".
    ap.add_argument("--guesses", default=",".join(GUESSES), help="comma-separated candidate fields to grade")
    args = ap.parse_args()
    guesses = tuple(args.guesses.split(","))
    rows = [json.loads(l) for l in open(work_path(args.retrieved), encoding="utf-8")]
    rng = random.Random(20261005)
    prompts, keys = [], []
    for s in range(0, len(rows), args.batch):
        blocks, kmap = [], {}
        for b, r in enumerate(rows[s : s + args.batch]):
            texts = list(dict.fromkeys(r[g] for g in guesses))
            rng.shuffle(texts)
            lines = [f"BLOCK {b}:"] + [f"  - {c}" for c in r["context"]] + ["  CANDIDATES:"]
            for n, t in enumerate(texts):
                lines.append(f"  {n}. {t}")
                kmap[(b, str(n))] = (s + b, t)
            blocks.append("\n".join(lines))
        prompts.append(PROMPT.format(blocks="\n\n".join(blocks)))
        keys.append(kmap)

    def safe(p):
        try:
            return llm.ask_json(p, model=args.model)
        except Exception:
            return []  # an unparsed batch leaves its rows ungraded; "graded" reports how many survived

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        answers = list(ex.map(safe, prompts))
    score = {}
    for kmap, ans in zip(keys, answers):
        for a in ans if isinstance(ans, list) else []:
            for n, v in (a.get("scores") or {}).items():
                key = (int(a.get("b", -1)), str(n))
                if key in kmap and isinstance(v, (int, float)):
                    score[kmap[key]] = float(v)
    per, by_med, kept = defaultdict(list), defaultdict(list), []
    for i, r in enumerate(rows):
        if not all((i, r[g]) in score for g in guesses):
            continue
        kept.append({"uid": r["uid"], "medium": r["medium"], **{g: score[(i, r[g])] for g in guesses}})
        for g in guesses:
            per[g].append(score[(i, r[g])])
            by_med[(r["medium"], g)].append(score[(i, r[g])])
    summ = lambda v: {"mean": round(st.fmean(v), 2), "plausible": round(st.fmean(x >= 4 for x in v), 3), "n": len(v)}
    # Aggregates alone hid whether a 0.16 gap was real (10/05/2026): keep each row's scores and report the paired
    # predictor-minus-baseline difference with a 95% bootstrap interval.
    boot = random.Random(1)

    def paired(base, of="predictor"):
        d = [k[of] - k[base] for k in kept]
        means = sorted(st.fmean(boot.choices(d, k=len(d))) for _ in range(2000))
        return {"diff": round(st.fmean(d), 3), "ci95": [round(means[50], 3), round(means[1949], 3)]}

    out = {"rows": len(rows), "graded": len(kept), "model": args.model, **{g: summ(v) for g, v in per.items()},
           "predictor_minus": {b: paired(b) for b in guesses if b != "predictor"} if kept else {},
           **({f"{t}_minus": {b: paired(b, t) for b in guesses if b != t} for t in ("text_gen", "text") if t in guesses}
              if kept else {}),
           "by_medium": {f"{m}/{g}": summ(v) for (m, g), v in sorted(by_med.items())}}
    path = work_path(args.retrieved).with_name("plausible.json")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(path.with_name("plausible-rows.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(k) + "\n" for k in kept)
    print(json.dumps({k: v for k, v in out.items() if k != "by_medium"}))


if __name__ == "__main__":
    main()
