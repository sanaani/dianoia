"""Merged ideas for the idea encoder (Sonnet, headless, cached; 10/04/2026).

Splitter q4 (``results/phase-7-e2e`` § 5) showed that rewording legitimately merges or splits ideas: even Sonnet cuts
a paraphrase into a different number of ideas most of the time on meetings and chats, and the end-to-end measure
loses those units. So the encoder should put one idea that states two things near each of the two. For a pair of
adjacent ideas from the same speaker in a splitter training unit (``split/train.jsonl``, never an evaluation unit),
Sonnet writes:
- 2 **merges**: one literal idea that states both, as a splitter would cut it if the speaker had said both in one
  breath, in different words from each other;
- 2 **edits** of the first merge: nearly the same words, one changing the first idea's part, one the second's.
Every 10th pair is held out for testing. Output ``merge/items.jsonl``; reruns read the cache.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from dianoia import llm
from dianoia.paths import work_path

PROMPT = """Each item is a pair of IDEAS: literal, self-contained sentences stating who puts something forward and \
how ("A says that ...", "A asks B to ...", "B agrees that ..."). Both come from the same speaker. For each item write:

- "merges": 2 sentences, each ONE literal idea that states everything both ideas state, the way it would be \
written if the speaker had said both things in one breath: same literal form (speaker letters, act, stance), every \
fact, name, number, date, placeholder (such as "Person 1") and negation kept, every pronoun resolved \nexactly as in the ideas (write "A and C", never "them"), nothing added. Join them the way a \
person would ("A says that X, and that Y", "A says that X because Y", "A asks B to X so that Y" ...), not by \
gluing the two sentences. The two merges should use clearly different words from each other.
- "edits": 2 sentences that keep almost every word of the FIRST merge but change one or two words so the meaning \
changes: the first edit changes something from the first idea's part only, the second edit changes something from \
the second idea's part only (a different action, object, amount, time, place, person, direction, condition or \
judgment). Do not simply add or remove "not".

Write natural, fluent English. Return only a JSON array: \
[{{"i": <number>, "merges": ["...", "..."], "edits": ["...", "..."]}}, ...]

{items}"""


def speaker(idea: str) -> str:
    return idea.split(" ", 1)[0]


def sources(rng: random.Random, n: int) -> list[dict]:
    used = {json.loads(l)["text"] for l in open(work_path("contrast", "items.jsonl"), encoding="utf-8")}
    units = [json.loads(l) for l in open(work_path("split", "train.jsonl"), encoding="utf-8")]
    rng.shuffle(units)
    ok = lambda i: 6 <= len(i.split()) <= 35 and i not in used
    out = []
    for u in units:  # one pair per unit, so the pairs spread over every unit
        ideas = u.get("ideas") or []
        cand = [(x, y) for x, y in zip(ideas, ideas[1:]) if speaker(x) == speaker(y) and len(speaker(x)) == 1 and ok(x) and ok(y)]
        if cand:
            x, y = rng.choice(cand)
            out.append({"uid": u["uid"], "x": x, "y": y})
        if len(out) == n:
            break
    for k, o in enumerate(out):
        o["id"] = k
        o["split"] = "test" if k % 10 == 0 else "train"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()
    items = sources(random.Random(15), args.n)
    path = work_path("merge", "items.jsonl")
    have = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
    got = {o["id"] for o in have}
    todo = [o for o in items if o["id"] not in got]
    batches = [todo[s : s + args.batch] for s in range(0, len(todo), args.batch)]
    print(json.dumps({"items": len(items), "have": len(have), "batches": len(batches)}), flush=True)

    def run(b):
        prompt = PROMPT.format(items="\n\n".join(f"Item {k}:\n1. {o['x']}\n2. {o['y']}" for k, o in enumerate(b)))
        try:
            ans = llm.ask_json(prompt, model="sonnet")
        except Exception as e:  # noqa: BLE001 - one bad batch must not stop the run; rerun retries it
            print(json.dumps({"failed": b[0]["id"], "error": str(e)[:200]}), flush=True)
            return []
        got = {int(a["i"]): a for a in ans if isinstance(a, dict) and str(a.get("i", "")).isdigit()} if isinstance(ans, list) else {}
        out = []
        for k, o in enumerate(b):
            a = got.get(k)
            if (a and isinstance(a.get("merges"), list) and len(a["merges"]) == 2
                    and isinstance(a.get("edits"), list) and len(a["edits"]) == 2):
                out.append({**o, "merges": [str(x) for x in a["merges"]], "edits": [str(x) for x in a["edits"]]})
        return out

    done = list(have)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, res in enumerate(ex.map(run, batches), 1):
            done += res
            if k % 10 == 0:
                print(json.dumps({"batches": k, "of": len(batches), "items": len(done)}), flush=True)
    with open(path, "w", encoding="utf-8") as f:
        for o in sorted(done, key=lambda o: o["id"]):
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(json.dumps({"complete": len(done), "test": sum(o["split"] == "test" for o in done)}))


if __name__ == "__main__":
    main()
