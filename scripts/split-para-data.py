"""Paraphrased splitter training units with one-for-one ideas (Sonnet, headless, cached; 10/04/2026).

The end-to-end check (``results/phase-7-e2e`` § 4) found that the splitter cuts a paraphrase into a different
number of ideas than its original about a quarter of the time on email and most of the time on meetings, and that no
encoder training helps those units. For each splitter training unit (``split/train.jsonl``, disjoint from every
check unit), Sonnet writes a paraphrase of every turn (same speakers, same turn order) and then rewrites the unit's
reference ideas one for one in the paraphrase's wording: same number, same order, still hyper-literal. Training the
splitter on (paraphrased turns -> these ideas) teaches it to cut by meaning, not by wording.
Output ``split/para.jsonl``: the unit with ``turns`` paraphrased and ``ideas`` rewritten, ``kind`` "paraphrase".
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from dianoia import llm
from dianoia.paths import work_path

PROMPT = """Each item is a unit of communication (one turn per line as "<speaker letter>: <text>") followed by its \
IDEAS: hyper-literal, self-contained statements of everything the unit says, one per claim, question, request or \
other act, with every pronoun resolved and the speaker and stance stated ("A says that ...", "A asks B to ...").

For each item:
1. "turns": paraphrase every turn the way a different fluent person would say the same thing: clearly different \
words and structure, same meaning, same speaker letter, same number and order of turns. Keep every fact, name, \
number, date and identifier. Keep the register (a chat stays chatty, a meeting stays spoken).
2. "ideas": rewrite each idea, ONE FOR ONE, so that it reads as an idea taken from YOUR paraphrase: use the \
paraphrase's wording for the content, keep the literal form ("A says that ...", every pronoun resolved, speaker and \
stance stated) and keep exactly what the idea claims. Same number of ideas, same order. Never merge, split, add or \
drop an idea.

Return only a JSON array: [{{"i": <number>, "turns": ["<text of turn 1>", ...], "ideas": ["...", ...]}}, ...]

{items}"""


def render(u: dict) -> str:
    turns = "\n".join(f"{s}: {t}" for s, t in u["turns"])
    ideas = "\n".join(f"- {i}" for i in u["ideas"])
    return f"{turns}\nIDEAS:\n{ideas}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="first n units only (0: all)")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    units = [json.loads(l) for l in open(work_path("split", "train.jsonl"), encoding="utf-8") if json.loads(l).get("ideas")]
    units = units[: args.n] if args.n else units
    path = work_path("split", "para.jsonl")
    have = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
    got = {o["uid"] for o in have}
    todo = [u for u in units if u["uid"] not in got]
    batches = [todo[s : s + args.batch] for s in range(0, len(todo), args.batch)]
    print(json.dumps({"units": len(units), "have": len(have), "batches": len(batches)}), flush=True)

    def run(b):
        prompt = PROMPT.format(items="\n\n".join(f"Item {k}:\n{render(u)}" for k, u in enumerate(b)))
        try:
            ans = llm.ask_json(prompt, model="sonnet")
        except Exception as e:  # noqa: BLE001 - one bad batch must not stop the run; rerun retries it
            print(json.dumps({"failed": b[0]["uid"], "error": str(e)[:200]}), flush=True)
            return []
        got = {int(a["i"]): a for a in ans if isinstance(a, dict) and str(a.get("i", "")).isdigit()} if isinstance(ans, list) else {}
        out = []
        for k, u in enumerate(b):
            a = got.get(k)
            # Only exact one-for-one answers: same turn count, same idea count.
            if (a and isinstance(a.get("turns"), list) and isinstance(a.get("ideas"), list)
                    and len(a["turns"]) == len(u["turns"]) and len(a["ideas"]) == len(u["ideas"])):
                # Sonnet often echoes the "A: " speaker prefix into the turn text; strip it.
                turns = [[s, re.sub(rf"^{re.escape(s)}:\s+", "", str(t))] for (s, _), t in zip(u["turns"], a["turns"])]
                if turns != u["turns"]:
                    out.append({**u, "kind": "paraphrase", "turns": turns, "ideas": [str(i) for i in a["ideas"]]})
        return out

    done = list(have)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, res in enumerate(ex.map(run, batches), 1):
            done += res
            if k % 10 == 0:
                print(json.dumps({"batches": k, "of": len(batches), "units": len(done)}), flush=True)
    with open(path, "w", encoding="utf-8") as f:
        for o in done:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(json.dumps({"complete": len(done), "of": len(units)}))


if __name__ == "__main__":
    main()
