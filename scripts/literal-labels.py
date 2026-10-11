"""Reference idea splits: Sonnet rewrites each unit as hyper-literal, self-contained ideas.

An idea is one plain sentence, roughly a clause, that can be understood with no context: every pronoun and
vague reference is replaced by what it refers to, the speaker and their stance are stated ("A is not sure
whether ...", "A asks B to ..."), and every material detail (negation, modality, numbers, dates, names,
manner, comparisons, conditions) is kept. This is abstractive proposition segmentation (Chen et al. 2023;
Hosseini et al. 2024) with the speaker made explicit, so dialogue works too.

Two sets, written to ``work/split/``:
- ``eval.jsonl``: the check's units (``work/ideas/versions.jsonl``), original plus stripped and disfluent
  versions, so Sonnet's own consistency is the ceiling the splitter is measured against (``--kinds`` adds the
  paraphrase, re-split and meaning-changed versions, 10/03/2026);
- ``train.jsonl``: training units disjoint from them (``idea-targets.py``'s generator), for distillation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dianoia import llm
from dianoia.paths import work_path

PROMPT = """Each item below is a unit of communication: an email sentence or paragraph, or an excerpt of a meeting \
transcript or a chat, one turn per line as "<speaker letter>: <text>". The text may lack punctuation and capitals, \
and contain fillers, false starts and repetitions.

Rewrite each unit as a list of IDEAS: hyper-literal, self-contained statements, like a contract that cares only \
about stating intent exactly, never about style.
- One idea per claim, question, request, offer, promise, agreement or other act: roughly one clause. Split \
coordinated claims ("A sent the report and B approved the report" -> two ideas). A condition stays inside the idea \
it conditions ("A asks B to call A if B has questions about the comments").
- Every idea is one complete plain sentence that can be understood alone. Replace every pronoun and vague \
reference (he, she, it, they, this, that, there, the former, the thing) with what it refers to, in full. Speakers \
are their letters: "I" -> the speaker's letter; "you" -> the addressee's letter if clear, else "the addressee". No idea may contain he, she, it, its, they, them, their, his, her, this, that, these, those or there as a reference; if the text never says what one refers to, write "an unspecified thing" (or person, place).
- State who puts each idea forward and how: "A says that ...", "A asks B whether ...", "A asks B to ...", \
"A promises to ...", "B agrees that ...", "A thanks B for ...". Keep stance exactly: "A is not sure whether ...", \
"A thinks that ... probably ...", "A assumes that ...".
- Keep every material detail: negation, modality (can, must, might), numbers, dates, times, names, identifiers, \
manner, comparisons ("rather than ..."), place.
- Drop fillers, false starts and pure backchannels ("mm-hmm", "yeah" that only signal listening). Keep an \
acknowledgement that agrees or confirms something, as an idea ("B agrees that ...").
- Never add anything the text does not say.

Return only a JSON array: [{{"i": <number>, "ideas": ["...", "..."]}}, ...]

{items}"""


def render(turns) -> str:
    return "\n".join(f"{s}: {t}" for s, t in turns)


def label(units: list[dict], batch: int, workers: int) -> list[list[str] | None]:
    prompts = [PROMPT.format(items="\n\n".join(f"Item {k}:\n{render(u['turns'])}" for k, u in enumerate(units[s : s + batch])))
               for s in range(0, len(units), batch)]

    def safe(p):
        try:
            return llm.ask_json(p, model="sonnet")
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        answers = list(ex.map(safe, prompts))
    out: list[list[str] | None] = [None] * len(units)
    for b, ans in enumerate(answers):
        for a in ans if isinstance(ans, list) else []:
            if isinstance(a, dict) and str(a.get("i", "")).isdigit() and isinstance(a.get("ideas"), list):
                k = b * batch + int(a["i"])
                if k < len(out) and (b * batch <= k < (b + 1) * batch):
                    out[k] = [str(x).strip() for x in a["ideas"] if str(x).strip()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["eval", "train"], required=True)
    ap.add_argument("--email", type=int, default=3000)
    ap.add_argument("--meeting", type=int, default=2000)
    ap.add_argument("--chat", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--kinds", default="original,stripped,disfluent", help="eval versions to label")
    args = ap.parse_args()
    if args.set == "eval":
        units = [json.loads(line) for line in open(work_path("ideas", "versions.jsonl"), encoding="utf-8")]
        units = [u for u in units if u["kind"] in args.kinds.split(",")]
        random.Random(20261002).shuffle(units)  # versions of one unit are never labelled side by side
    else:
        spec = importlib.util.spec_from_file_location("idea_targets", Path(__file__).with_name("idea-targets.py"))
        it = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(it)
        rng = random.Random(it.SEED)
        units = it.email(args.email, rng) + it.meetings(args.meeting, rng) + it.chats(args.chat, rng)
        rng.shuffle(units)
    path = work_path("split", f"{args.set}.jsonl")
    done = {(json.loads(line)["uid"], json.loads(line).get("kind")) for line in open(path, encoding="utf-8")} if path.exists() else set()
    todo = [u for u in units if (u["uid"], u.get("kind")) not in done]
    print(json.dumps({"units": len(units), "todo": len(todo)}), flush=True)
    for s in range(0, len(todo), args.chunk):
        ch = todo[s : s + args.chunk]
        ideas = label(ch, args.batch, args.workers)
        with open(path, "a", encoding="utf-8") as fh:
            for u, i in zip(ch, ideas):
                if i:
                    fh.write(json.dumps({**u, "ideas": i}, ensure_ascii=False) + "\n")
        print(json.dumps({"done": s + len(ch), "ok": sum(i is not None for i in ideas)}), flush=True)


if __name__ == "__main__":
    main()
