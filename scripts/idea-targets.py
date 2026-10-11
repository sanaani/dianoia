"""Stage-1 training targets: Sonnet's idea object for each training unit, written as it goes.

Training units never overlap the check's: email from fit mailboxes only (1-3 consecutive sentences of one
paragraph), meetings from QMSum's train and validation meetings, chat from the IRC train files. Each unit
is extracted once, from its original text; its stripped, disfluent and split versions (made at training
time) share the target, which is how invariance is taught.

Resumable: units already in ``work/ideas/targets.jsonl`` are skipped, so it can be stopped at any time and
whatever exists can be trained on.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import pandas as pd

from dianoia import ideas
from dianoia.paths import DATA, result_path, work_path

SEED = 20261002
EXT = DATA / "ext"
_spec = importlib.util.spec_from_file_location("idea_data", Path(__file__).with_name("idea-data.py"))
idea_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(idea_data)


def email(n: int, rng: random.Random) -> list[dict]:
    pool = pd.read_parquet(work_path("pool", "pool.parquet"))
    gen = json.loads(result_path("phase-1-intent", "generation.json").read_text())
    val = set(json.loads(result_path("phase-1-intent", "train-head.json").read_text())["val_mailboxes"])
    fit = pool[~pool["mailbox"].isin(set(gen["test_mailboxes"]) | val)]
    paras = [g["text"].tolist() for _, g in fit.groupby("pid", sort=False)]
    rng.shuffle(paras)
    out = []
    for p in paras:
        k = rng.choice([1, 1, 2, 3])
        s = rng.randrange(max(1, len(p) - k + 1))
        text = " ".join(p[s : s + k])
        if 5 <= len(text.split()) <= 80:
            out.append({"uid": f"email-{len(out)}", "medium": "email", "turns": [["A", text]]})
        if len(out) >= n:
            break
    return out


def meetings(n: int, rng: random.Random) -> list[dict]:
    return [{**u, "uid": "train-" + u["uid"]} for u in idea_data.meeting_units(rng, EXT / "qmsum-train", n)]


def chats(n: int, rng: random.Random) -> list[dict]:
    return [{**u, "uid": "train-" + u["uid"]} for u in idea_data.chat_units(rng, EXT / "irc-train", n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", type=int, default=3000)
    ap.add_argument("--meeting", type=int, default=2000)
    ap.add_argument("--chat", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(SEED)
    units = email(args.email, rng) + meetings(args.meeting, rng) + chats(args.chat, rng)
    rng.shuffle(units)  # media interleaved, so a partial run covers all three
    print(json.dumps({"units": len(units), "email": sum(u["medium"] == "email" for u in units),
                      "meeting": sum(u["medium"] == "meeting" for u in units), "chat": sum(u["medium"] == "chat" for u in units)}), flush=True)
    path = work_path("ideas", "targets.jsonl")
    done = {json.loads(line)["uid"] for line in open(path, encoding="utf-8")} if path.exists() else set()
    todo = [u for u in units if u["uid"] not in done]
    for s in range(0, len(todo), args.chunk):
        ch = todo[s : s + args.chunk]
        objs = ideas.extract([u["turns"] for u in ch], workers=args.workers)
        with open(path, "a", encoding="utf-8") as fh:
            for u, o in zip(ch, objs):
                if o is not None:
                    fh.write(json.dumps({**u, "object": o}, ensure_ascii=False) + "\n")
        print(json.dumps({"done": len(done) + s + len(ch), "of": len(units), "ok": sum(o is not None for o in objs)}), flush=True)


if __name__ == "__main__":
    main()
