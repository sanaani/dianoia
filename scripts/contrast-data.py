"""Paraphrase and attitude-contrast data for the idea encoder (Sonnet, headless, cached; 10/03/2026).

For each source sentence Sonnet writes:
- 2 **paraphrases**: same facts, same attitude, different words (same idea as the source);
- 2 **approving** versions and 2 **disapproving** versions of the same facts ("committed to her values" vs
  "stubborn"). The two approving versions are the same idea; an approving and a disapproving version are
  different ideas that share their facts: close neighbors that must stay apart.

Sources, half each: raw Enron sentences from phase-1 training mailboxes (not boilerplate, 8-35 words, never a
phase-1 item), and literal ideas from the splitter's training labels (``split/train.jsonl``). 10% of items are
held out for testing, by item. Output ``contrast/items.jsonl``. ``--n`` items in total; reruns read the cache.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from dianoia import llm
from dianoia.paths import work_path

PROMPT = """Each item is a sentence. For each one, write three things, keeping every fact, name, number and \
speaker letter exactly as given:

- "paraphrases": 2 sentences that say exactly the same thing with the same attitude, in clearly different words \
and structure, the way a fluent person would naturally rephrase it.
- "approving": 2 differently worded sentences stating the same facts so that they reflect well on the people or \
things involved (e.g. "she stands firm in her decisions, no matter the pressure").
- "disapproving": 2 differently worded sentences stating the same facts so that they reflect badly on them \
(e.g. "she is stubborn and refuses to compromise, regardless of the circumstances").

Approving and disapproving versions must describe the same events and facts; only the judgment changes. Never \
negate or alter the facts. If the item is written in the form "A says that ...", keep that form and keep the \
speaker: change only how the reported content is judged. Write natural, fluent English.

Return only a JSON array: [{{"i": <number>, "paraphrases": [...], "approving": [...], "disapproving": [...]}}, ...]

{items}"""


def sources(rng: random.Random, n: int) -> list[dict]:
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    train_boxes = set(items[items.split == "train"].mailbox)
    used = set(items.text)
    emails = pd.read_parquet(work_path("corpus", "emails.parquet"))[["message_id", "mailbox"]]
    sents = pd.read_parquet(work_path("corpus", "sentences.parquet"))
    sents = sents[~sents.boilerplate].merge(emails, on="message_id")
    sents = sents[sents.mailbox.isin(train_boxes) & ~sents.text.isin(used)]
    sents = sents[sents.text.str.split().str.len().between(8, 35)].drop_duplicates("text")
    email = [{"source": "email", "text": t} for t in rng.sample(sorted(sents.text), n // 2)]
    ideas = sorted({i for l in open(work_path("split", "train.jsonl"), encoding="utf-8") for i in json.loads(l)["ideas"]
                    if 8 <= len(i.split()) <= 35})
    literal = [{"source": "literal", "text": t} for t in rng.sample(ideas, n - n // 2)]
    out = email + literal
    rng.shuffle(out)
    for k, o in enumerate(out):
        o["id"] = k
        o["split"] = "test" if k % 10 == 0 else "train"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--fill", action="store_true", help="only ask for items missing from items.jsonl, then merge")
    args = ap.parse_args()
    rng = random.Random(10)
    items = sources(rng, args.n)
    have = []
    if args.fill:
        have = [json.loads(l) for l in open(work_path("contrast", "items.jsonl"), encoding="utf-8")]
        got = {o["id"] for o in have}
        items = [o for o in items if o["id"] not in got]
    # Larger batches spend the instructions once per many items (fewer input tokens); about 100 items is the most
    # one response can hold (6 sentences each).
    batches = [items[s : s + args.batch] for s in range(0, len(items), args.batch)]

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
            if a and all(isinstance(a.get(f), list) and len(a[f]) == 2 for f in ("paraphrases", "approving", "disapproving")):
                out.append({**o, **{f: [str(x) for x in a[f]] for f in ("paraphrases", "approving", "disapproving")}})
        return out

    done = list(have)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, res in enumerate(ex.map(run, batches), 1):
            done += res
            if k % 20 == 0:
                print(json.dumps({"batches": k, "of": len(batches), "items": len(done)}), flush=True)
    work_path("contrast").mkdir(parents=True, exist_ok=True)
    with open(work_path("contrast", "items.jsonl"), "w", encoding="utf-8") as f:
        for o in done:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(json.dumps({"asked": len(items) + len(have), "complete": len(done), "test": sum(o["split"] == "test" for o in done)}))


if __name__ == "__main__":
    main()
