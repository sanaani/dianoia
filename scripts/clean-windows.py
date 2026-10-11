"""Clean a source's windows with ``dianoia.clean`` (10/04/2026), before or after the splitter.

    python scripts/clean-windows.py pre  <windows.jsonl> <out.jsonl>             copies and templates, before splitting
    python scripts/clean-windows.py post <preds.jsonl> <out.jsonl>               near-copies and template ideas, after

Paths are under work/jepa/. Prints the counts; the post step also drops the near-copies from ``<windows>`` if given
as a fourth argument (so the audit reads the same conversations the units come from).
"""

from __future__ import annotations

import json
import sys

from dianoia.clean import conv_of, near_copies, prefilter, template_ideas, cross_posts
from dianoia.paths import work_path


def main() -> None:
    step, src, dst = sys.argv[1:4]
    rows = [json.loads(l) for l in open(work_path("jepa", src), encoding="utf-8")]
    if step == "pre":
        kept, stats = prefilter(rows)
    else:
        dropped, stats = near_copies(rows)
        kept = [r for r in rows if conv_of(r["uid"]) not in dropped]
        tdrop, tstats = template_ideas(kept)
        kept = [r for r in kept if r["uid"] not in tdrop]
        stats.update(tstats)
        xdrop, xstats = cross_posts(kept)
        kept = [r for r in kept if conv_of(r["uid"]) not in xdrop]
        dropped |= xdrop
        stats.update(xstats)
        if len(sys.argv) > 4:
            wins = [json.loads(l) for l in open(work_path("jepa", sys.argv[4]), encoding="utf-8")]
            with open(work_path("jepa", sys.argv[4].replace(".jsonl", "-post.jsonl")), "w", encoding="utf-8") as f:
                for w in wins:
                    if conv_of(w["uid"]) not in dropped and w["uid"] not in tdrop:
                        f.write(json.dumps(w, ensure_ascii=False) + "\n")
    with open(work_path("jepa", dst), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"step": step, "rows_in": len(rows), "rows_out": len(kept), **stats}))


if __name__ == "__main__":
    main()
