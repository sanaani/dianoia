"""The shape of an idea: material for the geometry figures.

From the splitter's ideas (``pred-<tag>.jsonl``, originals):
- the **simplest** unit (one short idea) and the **most complex** (the most ideas, with nested stance);
- 60 more ideas sampled across media, for the variance statistics.

For every chosen idea, Sonnet writes 12 rewordings that keep it hyper-literal and exactly equivalent, its
negation, and its bare proposition (speaker and stance removed). ``remote-geometry.py`` then encodes all of it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from dianoia import llm
from dianoia.paths import work_path

PROMPT = """Each item is one IDEA: a hyper-literal, self-contained statement. For each, write:
- "rewordings": 12 sentences that say EXACTLY the same thing, equally literal and self-contained (same speakers \
as letters, same stance, same details), with different words and sentence structure wherever possible;
- "negation": the same idea with its central claim negated (still literal, same speaker and stance);
- "bare": the proposition alone, with the speaker and the act or stance removed ("A asks B to send the report" -> \
"B sends the report"; "A is not sure whether A has clothes" -> "A has clothes").

Return only a JSON array: [{{"i": <number>, "rewordings": [...], "negation": "...", "bare": "..."}}, ...]

{items}"""

NEST = re.compile(r"\b(says|asks|thinks|believes|is not sure|assumes|promises|agrees|wonders|suggests|hopes|wants|knows|guesses)\b")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.preds, encoding="utf-8")]
    rows = [r for r in rows if r["kind"] == "original" and r["ideas"]]
    simple = min((r for r in rows if r["medium"] == "email" and len(r["ideas"]) == 1),
                 key=lambda r: len(r["ideas"][0].split()))
    complex_ = max(rows, key=lambda r: (len(r["ideas"]), sum(len(NEST.findall(i)) for i in r["ideas"])))
    rng = random.Random(5)
    pool = [(r["uid"], i) for r in rows if r["uid"] not in (simple["uid"], complex_["uid"]) for i in r["ideas"]]
    sample = rng.sample(pool, 60)
    chosen = [("simple", simple["uid"], i) for i in simple["ideas"]] + [("complex", complex_["uid"], i) for i in complex_["ideas"]] \
        + [("sample", u, i) for u, i in sample]
    prompts = [PROMPT.format(items="\n".join(f"{k}. {c[2]}" for k, c in enumerate(chosen[s : s + 8]))) for s in range(0, len(chosen), 8)]

    def safe(p):
        try:
            return llm.ask_json(p, model="sonnet")
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=5) as ex:
        answers = list(ex.map(safe, prompts))
    out = []
    for b, ans in enumerate(answers):
        got = {int(a["i"]): a for a in ans if isinstance(a, dict) and str(a.get("i", "")).isdigit()} if isinstance(ans, list) else {}
        for k, c in enumerate(chosen[b * 8 : b * 8 + 8]):
            a = got.get(k, {})
            out.append({"role": c[0], "uid": c[1], "idea": c[2], "rewordings": [str(x) for x in a.get("rewordings", [])][:12],
                        "negation": a.get("negation"), "bare": a.get("bare")})
    meta = {"simple": {"uid": simple["uid"], "turns": [], "ideas": simple["ideas"]},
            "complex": {"uid": complex_["uid"], "ideas": complex_["ideas"]}}
    versions = {json.loads(l)["uid"]: json.loads(l) for l in open(work_path("ideas", "versions.jsonl"), encoding="utf-8")
                if json.loads(l)["kind"] == "original"}
    for k in meta:
        meta[k]["turns"] = versions[meta[k]["uid"]]["turns"]
    work_path("geometry", "ideas.json").write_text(json.dumps({"meta": meta, "ideas": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"ideas": len(out), "with_rewordings": sum(len(o["rewordings"]) >= 8 for o in out),
                      "complex_ideas": len(complex_["ideas"]), "simple": simple["ideas"]}))


if __name__ == "__main__":
    main()
