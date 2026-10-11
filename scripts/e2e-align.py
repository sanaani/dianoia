"""End-to-end version match that allows one idea to match two (laptop CPU; 10/04/2026).

``remote-e2e.py`` matches each idea of a version to its best single idea of the original, and back, and scores a
unit by its worst idea. When rewording merges two ideas into one, the merged idea matches either half only
moderately, so the units the splitter cuts into a different number of ideas lose (``results/phase-7-e2e`` § 4-5).
A merged idea's vector is close to the normalized sum of its parts (cosine 0.86, as close as a paraphrase; § 7), so
here each side's candidates are its ideas plus the normalized sum of every adjacent pair. Every idea still has to be
matched, by a single idea or by a pair that contains its meaning.

``para_beats_move`` (paraphrase's worst idea matches better than the meaning-changed version's) is reported both
ways, by medium and by whether the splitter cut the paraphrase into the same number of ideas as the original.
That worst-of-all-ideas rule is biased against units with many ideas: a paraphrase rewords every idea, the meaning
change touches one, and the worst of ten reworded ideas often drifts further than one edit even when the encoder
separates each pair. ``targeted`` (the primary measure from § 8) compares only the original ideas the meaning
change touched, chosen by text: does the paraphrase cover them better than the meaning-changed version?
Vectors are cached in ``e2e/vec-<name>.pt``.

    python scripts/e2e-align.py --adapter work/ablation/S1/lora.pt --name S1
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from dianoia.paths import work_path

sys.path.insert(0, str(Path(__file__).parent))
from local_encode import Encoder  # noqa: E402


def cands(V: torch.Tensor) -> torch.Tensor:
    """The ideas plus the normalized sum of each adjacent pair."""
    return torch.cat([V, F.normalize(V[:-1] + V[1:], dim=-1)]) if len(V) > 1 else V


def worst(V: torch.Tensor, O: torch.Tensor, align: bool) -> float:
    if align:
        return float(min((V @ cands(O).T).max(1).values.min(), (O @ cands(V).T).max(1).values.min()))
    return float(min((V @ O.T).max(1).values.min(), (O @ V.T).max(1).values.min()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--name", required=True)
    ap.add_argument("--units", default=str(work_path("e2e", "units.jsonl")))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = [json.loads(l) for l in open(args.units, encoding="utf-8")]
    cache = work_path("e2e", f"vec-{args.name}.pt")
    ideas = sorted({i for r in rows for i in r["ideas"]})
    vec = torch.load(cache) if cache.exists() else {}
    todo = [i for i in ideas if i not in vec]
    if todo:
        # Saved every chunk, so a run killed for memory resumes where it stopped.
        enc = Encoder(args.adapter)
        for s in range(0, len(todo), args.chunk):
            part = todo[s : s + args.chunk]
            vec.update(zip(part, enc.encode(part, batch=args.batch)))
            torch.save(vec, cache)
            print(json.dumps({"encoded": len(vec), "of": len(ideas)}), flush=True)
        del enc
    by = defaultdict(dict)
    for r in rows:
        by[r["uid"]][r["kind"]] = r
    res = defaultdict(lambda: defaultdict(list))
    for u, k in by.items():
        if not all(k.get(x, {}).get("ideas") for x in ("original", "paraphrase", "move")):
            continue
        O, P, M = (torch.stack([vec[i] for i in k[x]["ideas"]]) for x in ("original", "paraphrase", "move"))
        bucket = "same" if len(P) == len(O) else "diff"
        keys = (k["original"]["medium"], f"{k['original']['medium']}-{bucket}", "all", f"all-{bucket}")
        for align in (False, True):
            win = worst(P, O, align) > worst(M, O, align)
            for key in keys:
                res[key]["align" if align else "one"].append(win)
        # Targeted: only the original ideas the meaning change touched, chosen by text (absent word for word from the
        # move's ideas), so the choice does not favor either version. Does the paraphrase cover them better?
        mi = set(k["move"]["ideas"])
        hit = [j for j, o in enumerate(k["original"]["ideas"]) if o not in mi]
        if hit:
            win = float((O @ P.T).max(1).values[hit].min()) > float((O @ M.T).max(1).values[hit].min())
            for key in keys:
                res[key]["targeted"].append(win)
    out = {key: {"n": len(v["one"]), **{m: round(st.fmean(w), 3) for m, w in v.items() if w}} for key, v in sorted(res.items())}
    for key, v in out.items():
        print(key, v)
    work_path("e2e", f"align-{args.name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
