"""Units for the idea-object check: email sentences, meeting excerpts and chat excerpts, each in seven versions.

Sources (licenses checked 10/01/2026):
- email: phase-1 test items (Enron), with their blind-judged paraphrase and negation variants;
- meetings: AMI and ICSI transcripts as distributed by QMSum (code MIT; AMI and ICSI transcripts CC BY 4.0);
- chat: Ubuntu IRC from Kummerfeld et al. 2019 (data CC BY 4.0), whose reply-to annotations give real threads.

A unit is a list of turns ``[speaker letter, text]``. Versions:
- original, and **copy** (identical text, extracted separately: the extractor's own noise);
- **paraphrase**: every turn reworded (email: the phase-1 variant; meetings and chat: Sonnet);
- **stripped**: lowercase, every punctuation mark removed (mechanical);
- **disfluent**: fillers, repetitions and a false start inserted (mechanical, seeded);
- **split**: the longest turn broken into two consecutive turns by the same speaker (mechanical);
- **move**: one turn's meaning changed (email: the phase-1 negation; meetings and chat: Sonnet).

Every version carries ``turn_map`` (its turn index -> the original's) so structures can be aligned.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from dianoia import llm
from dianoia.paths import DATA, result_path, work_path

SEED = 20261001
EXT = DATA / "ext"
N = {"email": 150, "meeting": 120, "chat": 120}
FILLERS = ["um", "uh", "like", "you know", "i mean", "so", "kind of"]

REWRITE = """Each item below is an excerpt from a meeting transcript or a chat, one turn per line as "<speaker>: <text>". \
For each, write two rewrites with exactly the same speakers and the same number of turns, in the same order:

- "reworded": every turn says exactly the same thing in different words: different vocabulary and sentence structure \
wherever possible. Do not add, drop or change any content; keep names and numbers; keep each turn's tone. A turn that is \
only a backchannel ("mm-hmm", "yeah") becomes an equivalent backchannel.
- "changed": identical to the original except ONE turn with real content, whose meaning changes materially: negate \
its claim, or change what is done or what it is done to. Give that turn's number (1-based) as "changed_turn".

Return only a JSON array: [{{"i": <number>, "reworded": ["turn 1 text", ...], "changed": ["turn 1 text", ...], "changed_turn": <k>}}, ...]

{items}"""


def clean_meeting(text: str) -> str:
    text = re.sub(r"\{[a-z]+\}", " ", text)
    text = re.sub(r"(\w) - (\w)", r"\1-\2", text)
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def words(turns) -> int:
    return sum(len(t.split()) for _, t in turns)


def letters(speakers: list[str]) -> dict:
    out = {}
    for s in speakers:
        out.setdefault(s, "ABCDEFGHIJ"[len(out)])
    return out


def email_units(rng: random.Random) -> list[dict]:
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    judged = pd.read_parquet(work_path("phase1", "judged.parquet"))
    bad = set(zip(judged.loc[judged["disputed"], "item"], judged.loc[judged["disputed"], "field"]))
    test = [(i, r) for i, r in items[items["split"] == "test"].iterrows()
            if isinstance(r["paraphrase"], str) and isinstance(r["negation"], str)
            and (i, "paraphrase") not in bad and (i, "negation") not in bad and len(r["text"].split()) >= 6]
    rng.shuffle(test)
    return [{"uid": f"email-{i}", "medium": "email", "turns": [["A", r["text"]]],
             "paraphrase": [["A", r["paraphrase"]]], "move": [["A", r["negation"]]], "changed_turn": 0}
            for i, r in test[: N["email"]]]


def meeting_units(rng: random.Random, root: Path = EXT / "qmsum", n: int = N["meeting"]) -> list[dict]:
    cands = []
    for f in sorted(root.glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))["meeting_transcripts"]
        turns = [(x["speaker"], clean_meeting(x["content"])) for x in t]
        for s in range(0, len(turns) - 6, 6):
            win = [x for x in turns[s : s + 6] if x[1]]
            if len(win) >= 4 and len({sp for sp, _ in win}) >= 2 and 40 <= words(win) <= 160:
                cands.append((f.stem, win))
    by = defaultdict(list)
    for m, w in cands:
        by[m].append(w)
    picked = []
    for m in sorted(by):
        rng.shuffle(by[m])
    while len(picked) < n and any(by.values()):
        for m in sorted(by):
            if by[m] and len(picked) < n:
                picked.append((m, by[m].pop()))
    out = []
    for k, (m, win) in enumerate(picked):
        lt = letters([sp for sp, _ in win])
        out.append({"uid": f"meeting-{m}-{k}", "medium": "meeting", "turns": [[lt[sp], t] for sp, t in win]})
    return out


def chat_units(rng: random.Random, root: Path = EXT / "irc", n: int = N["chat"]) -> list[dict]:
    line = re.compile(r"^\[\d\d:\d\d\] <([^>]+)> (.*)$")
    cands = []
    for ann in sorted(root.glob("*.annotation.txt")):
        raw = (root / ann.name.replace("annotation", "raw")).read_text(encoding="utf-8", errors="replace").splitlines()
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x

        nodes = set()
        for a in ann.read_text().splitlines():
            p = a.split()
            if len(p) >= 2:
                i, j = int(p[0]), int(p[1])
                nodes |= {i, j}
                parent.setdefault(i, i)
                parent.setdefault(j, j)
                parent[find(i)] = find(j)
        threads = defaultdict(list)
        for node in sorted(nodes):
            threads[find(node)].append(node)
        for th in threads.values():
            msgs = [line.match(raw[n]) for n in th if n < len(raw)]
            msgs = [(m.group(1), m.group(2).strip()) for m in msgs if m]
            if len(msgs) < 4:
                continue
            win = msgs[:8]
            if len({s for s, _ in win}) < 2 or not 25 <= words(win) <= 160:
                continue
            lt = letters([s for s, _ in win])
            turns = []
            for s, t in win:
                for nick, letter in lt.items():
                    t = re.sub(rf"(?<![\w-]){re.escape(nick)}(?![\w-])", letter, t)
                turns.append([lt[s], t])
            cands.append({"uid": f"chat-{ann.name.split('.')[0]}-{th[0]}", "medium": "chat", "turns": turns})
    rng.shuffle(cands)
    return cands[:n]


def stripped(turns):
    return [[s, re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip() or "mm"] for s, t in turns]


def disfluent(turns, rng: random.Random):
    out = []
    for s, t in turns:
        ws, new = t.split(), []
        if len(ws) >= 6 and rng.random() < 0.4:  # false start: begin, abandon, restart
            new += ws[: 2] + ["--"]
        for w in ws:
            if rng.random() < 0.15:
                new.append(rng.choice(FILLERS) + ",")
            new.append(w)
            if rng.random() < 0.05:
                new.append(w)
        out.append([s, " ".join(new)])
    return out


def split(turns):
    k = max(range(len(turns)), key=lambda i: len(turns[i][1].split()))
    ws = turns[k][1].split()
    if len(ws) < 4:
        return None, None
    h = len(ws) // 2
    out = turns[:k] + [[turns[k][0], " ".join(ws[:h])], [turns[k][0], " ".join(ws[h:])]] + turns[k + 1 :]
    return out, list(range(k + 1)) + list(range(k, len(turns)))


def rewrite(units: list[dict], batch: int = 4) -> None:
    todo = [u for u in units if "paraphrase" not in u]
    prompts = []
    for s in range(0, len(todo), batch):
        ch = todo[s : s + batch]
        prompts.append((ch, REWRITE.format(items="\n\n".join(
            f"Item {k}:\n" + "\n".join(f"{sp}: {t}" for sp, t in u["turns"]) for k, u in enumerate(ch)))))

    def safe(p):
        try:
            return llm.ask_json(p, model="sonnet")
        except Exception:
            return []

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as ex:
        answers = list(ex.map(safe, [p for _, p in prompts]))
    for (ch, _), ans in zip(prompts, answers):
        got = {int(a["i"]): a for a in ans if isinstance(a, dict) and str(a.get("i", "")).isdigit()} if isinstance(ans, list) else {}
        for k, u in enumerate(ch):
            a = got.get(k, {})
            n = len(u["turns"])
            rw, chg, ct = a.get("reworded"), a.get("changed"), a.get("changed_turn")
            if isinstance(rw, list) and len(rw) == n and isinstance(chg, list) and len(chg) == n and isinstance(ct, int) and 1 <= ct <= n:
                u["paraphrase"] = [[sp, str(t)] for (sp, _), t in zip(u["turns"], rw)]
                u["move"] = [[sp, str(t)] for (sp, _), t in zip(u["turns"], chg)]
                u["changed_turn"] = ct - 1


def main() -> None:
    rng = random.Random(SEED)
    units = email_units(rng) + meeting_units(rng) + chat_units(rng)
    rewrite(units)
    rewrite(units, batch=1)  # one more try, alone, for any the batch got wrong
    units = [u for u in units if "paraphrase" in u]
    versions = []
    for u in units:
        n = len(u["turns"])
        ident = list(range(n))
        sp, sp_map = split(u["turns"])
        vs = {"original": (u["turns"], ident), "copy": (u["turns"], ident), "paraphrase": (u["paraphrase"], ident),
              "stripped": (stripped(u["turns"]), ident), "disfluent": (disfluent(u["turns"], random.Random(u["uid"])), ident),
              "move": (u["move"], ident)}
        if sp:
            vs["split"] = (sp, sp_map)
        for kind, (turns, tm) in vs.items():
            versions.append({"uid": u["uid"], "medium": u["medium"], "kind": kind, "turns": turns, "turn_map": tm,
                             "changed_turn": u["changed_turn"] if kind == "move" else None})
    path = work_path("ideas", "versions.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for v in versions:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    summary = {"units": {m: sum(u["medium"] == m for u in units) for m in N}, "versions": len(versions)}
    result_path("phase-4-idea-objects", "units.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
