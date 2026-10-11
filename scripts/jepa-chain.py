"""Chain the splitter's ideas for consecutive windows into long idea sequences (10/04/2026).

Input: ``jepa/pred-windows.jsonl`` (splitter q3 over ``jepa/windows.jsonl``, ``jepa-units.py``). Each conversation's
windows are taken in order and their ideas joined; a missing window (filtered out, dropped near a test unit, or
cut into no ideas) ends the run, so a sequence never jumps. One placeholder mapping per conversation, as each unit
has in ``jepa-prepare.py``. Each run of ideas is cut into sequences of at most ``--max-len`` ideas.
Output ``jepa/units-w.jsonl`` (``--out``): ``jepa/units.jsonl`` (``--base``; its test units unchanged) plus these as training units
(``kind`` "window").
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path

from dianoia.paths import work_path

spec = importlib.util.spec_from_file_location("build_pairs", Path(__file__).with_name("build-pairs.py"))
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)


_PH = None


def _init() -> None:
    global _PH
    _PH = bp.Placeholders()


def chain_one(item: tuple[str, dict, int]) -> list[tuple[str, list[str]]]:
    """One conversation's windows -> its idea sequences (medium, ideas). Placeholders are per conversation, so
    conversations are independent and can be processed in any order or in parallel with the same result."""
    conv, wins, max_len = item
    runs, run, prev = [], [], None
    for k in sorted(wins):
        if prev is not None and k != prev + 1 and run:
            runs.append(run)
            run = []
        run += wins[k][1]
        prev = k
    if run:
        runs.append(run)
    flat = _PH.apply([i for r in runs for i in r])
    out, pos = [], 0
    for r in runs:
        ideas = [i.rstrip(".") + "." for i in flat[pos : pos + len(r)]]
        pos += len(r)
        for s in range(0, len(ideas), max_len):
            chunk = ideas[s : s + max_len]
            if len(chunk) >= 2:
                out.append((wins[min(wins)][0], chunk))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--preds", default="pred-windows.jsonl", help="splitter output under jepa/")
    ap.add_argument("--base", default="units.jsonl", help="units under jepa/ to keep in front ('' for none)")
    ap.add_argument("--out", default="units-w.jsonl")
    ap.add_argument("--workers", type=int, default=4, help="processes for the placeholder pass (1: in this process)")
    args = ap.parse_args()
    convs = defaultdict(dict)
    for l in open(work_path("jepa", args.preds), encoding="utf-8"):
        r = json.loads(l)
        conv, k = r["uid"].rsplit("/", 1)
        if r["ideas"]:
            convs[conv][int(k)] = (r["medium"], r["ideas"])
    items = [(c, convs[c], args.max_len) for c in sorted(convs)]
    # Parallel placeholder pass (10/04/2026): the single-process spaCy pass took 15-20 minutes per 150k ideas. Each
    # worker holds about 0.7 GB (spaCy, the entity list, build-pairs' imports); 6 workers on the 16 GB laptop with
    # other apps open ran it out of memory. So the pool is sized from free memory, keeping 2 GB in reserve.
    import psutil

    free_gb = psutil.virtual_memory().available / 2**30
    workers = max(1, min(args.workers, int((free_gb - 2.0) // 0.7)))
    print(json.dumps({"workers_requested": args.workers, "workers": workers, "free_gb": round(free_gb, 1)}), flush=True)
    if workers > 1:
        with multiprocessing.Pool(workers, initializer=_init) as pool:
            results = pool.map(chain_one, items, chunksize=32)
    else:
        _init()
        results = [chain_one(i) for i in items]
    seqs = []
    for (conv, _, _), res in zip(items, results):
        for medium, chunk in res:
            seqs.append({"uid": f"{conv}@{len(seqs)}", "kind": "window", "medium": medium, "split": "train", "ideas": chunk})
    base = list(open(work_path("jepa", args.base), encoding="utf-8")) if args.base else []
    with open(work_path("jepa", args.out), "w", encoding="utf-8") as f:
        f.writelines(base)
        for s in seqs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps({"conversations": len(convs), "sequences": len(seqs), "ideas": sum(len(s["ideas"]) for s in seqs),
                      "transitions": sum(len(s["ideas"]) - 1 for s in seqs)}))


if __name__ == "__main__":
    main()
