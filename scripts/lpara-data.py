"""Paraphrases and minimal edits of literal ideas for the idea encoder (Sonnet, headless, cached; 10/03/2026).

The end-to-end check (``results/phase-7-e2e``) found that the splitter keeps its input's wording, so a paraphrased
input gives ideas in other words, and arm R put those about as far from the original's ideas as a meaning-changed
version. For each literal idea from the splitter's training labels (``split/train.jsonl``, never an evaluation
unit, never an item already in ``contrast/items.jsonl``), Sonnet writes:
- 3 **paraphrases**: the same idea in the same literal form (same speaker, same act), in clearly different words;
- 2 **minimal edits**: nearly the same words, a different idea (one or two words changed so the meaning changes).
Every 10th item is held out for testing. Output ``lpara/items.jsonl``; reruns read the cache.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from dianoia import llm
from dianoia.paths import work_path

PROMPT = """Each item is an IDEA: one literal, self-contained sentence stating who puts something forward and how \
("A says that ...", "A asks B to ...", "B agrees that ..."). For each item write:

- "paraphrases": 3 sentences that state exactly the same idea in clearly different words, the way different \
people would describe the same thing. Keep the same form: the same speaker letters, the same act (says, asks, \
promises, agrees ...) and the same stance (is not sure, thinks probably ...). Inside that, change the wording as \
much as you naturally can: synonyms, different sentence structure, more formal or more casual vocabulary, \
active or passive. Keep every fact, name, number, date, placeholder (such as "Person 1") and negation. Each of \
the 3 should differ from the others too.
- "edits": 2 sentences that keep almost every word of the item but change one or two words so the idea becomes \
a different one that a careful reader would treat as a different claim: a different action, object, amount, \
time, place, person, direction, condition or judgment. Do not simply add or remove "not". The two edits should \
change different things.

Write natural, fluent English. Return only a JSON array: \
[{{"i": <number>, "paraphrases": [...], "edits": [...]}}, ...]

{items}"""


def sources(rng: random.Random, n: int) -> list[dict]:
    used = {json.loads(l)["text"] for l in open(work_path("contrast", "items.jsonl"), encoding="utf-8")}
    ideas = sorted({i for l in open(work_path("split", "train.jsonl"), encoding="utf-8") for i in json.loads(l)["ideas"]
                    if 8 <= len(i.split()) <= 40} - used)
    out = [{"text": t} for t in rng.sample(ideas, n)]
    for k, o in enumerate(out):
        o["id"] = k
        o["split"] = "test" if k % 10 == 0 else "train"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()
    items = sources(random.Random(11), args.n)
    path = work_path("lpara", "items.jsonl")
    have = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
    got = {o["id"] for o in have}
    todo = [o for o in items if o["id"] not in got]
    batches = [todo[s : s + args.batch] for s in range(0, len(todo), args.batch)]
    print(json.dumps({"items": len(items), "have": len(have), "batches": len(batches)}), flush=True)

    def run(b):
        prompt = PROMPT.format(items="\n".join(f"{k}. {o['text']}" for k, o in enumerate(b)))
        try:
            ans = llm.ask_json(prompt, model="sonnet")
        except Exception as e:  # noqa: BLE001 - one bad batch must not stop the run; rerun retries it
            print(json.dumps({"failed": b[0]["id"], "error": str(e)[:200]}), flush=True)
            return []
        got = {int(a["i"]): a for a in ans if isinstance(a, dict) and str(a.get("i", "")).isdigit()} if isinstance(ans, list) else {}
        out = []
        for k, o in enumerate(b):
            a = got.get(k)
            if a and isinstance(a.get("paraphrases"), list) and len(a["paraphrases"]) == 3 and isinstance(a.get("edits"), list) and len(a["edits"]) == 2:
                out.append({**o, "paraphrases": [str(x) for x in a["paraphrases"]], "edits": [str(x) for x in a["edits"]]})
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
