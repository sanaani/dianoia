"""More conversation for the next-idea predictor: consecutive windows of whole training meetings and IRC threads
(10/04/2026). Output ``jepa/windows2.jsonl``, splitter input for ``remote-split.py predict-qwen``.

J3 (``results/phase-8-jepa``) was data-limited. The splitter's training units are short excerpts sampled from the
training sources (``ext/qmsum-train``, ``ext/irc-train``; never the check's sources). Here every meeting is cut into
consecutive 6-turn windows and every annotated IRC thread into consecutive 8-message windows, with one speaker-letter
mapping per meeting or thread, so the windows of one conversation can be chained into a long idea sequence.
``uid`` is ``<conversation>/<window index>``. The IRC channel bots' messages are left out: canned factoids that
repeat word for word across days.

Splits are by whole conversation (``dianoia.splits``). The first version (``jepa/windows.jsonl``, J4-J6) instead
dropped windows near the old excerpt test units and kept the bots.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

from dianoia.paths import DATA, work_path

spec = importlib.util.spec_from_file_location("idea_data", Path(__file__).with_name("idea-data.py"))
idd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(idd)

EXT = DATA / "ext"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BOTS = {"ubottu", "ubotu", "ubot2", "ubot3", "ubot4", "ubot5"}


def letter_map(speakers) -> dict:
    out = {}
    for s in speakers:
        out.setdefault(s, LETTERS[min(len(out), len(LETTERS) - 1)])
    return out


def meetings() -> list[tuple[str, list[list[str]]]]:
    convs = []
    for f in sorted((EXT / "qmsum-train").glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))["meeting_transcripts"]
        turns = [(x["speaker"], idd.clean_meeting(x["content"])) for x in t]
        turns = [x for x in turns if x[1]]
        lt = letter_map(sp for sp, _ in turns)
        wins = []
        for s in range(0, len(turns), 6):
            win = turns[s : s + 6]
            wins.append([[lt[sp], txt] for sp, txt in win] if len(win) >= 3 and 20 <= idd.words(win) <= 200 else None)
        convs.append((f"meeting-{f.stem}", wins))
    return convs


def chats() -> list[tuple[str, list[list[str]]]]:
    root = EXT / "irc-train"
    line = re.compile(r"^\[\d\d:\d\d\] <([^>]+)> (.*)$")
    convs = []
    for ann in sorted(root.glob("*.annotation.txt")):
        raw = (root / ann.name.replace("annotation", "raw")).read_text(encoding="utf-8", errors="replace").splitlines()
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x

        for a in ann.read_text().splitlines():
            p = a.split()
            if len(p) >= 2:
                i, j = int(p[0]), int(p[1])
                parent.setdefault(i, i)
                parent.setdefault(j, j)
                parent[find(i)] = find(j)
        threads = defaultdict(list)
        for node in sorted(parent):
            threads[find(node)].append(node)
        for th in threads.values():
            msgs = [line.match(raw[n]) for n in th if n < len(raw)]
            # The channel bots' canned factoid replies are not conversation and repeat word for word across days.
            msgs = [(m.group(1), m.group(2).strip()) for m in msgs if m and m.group(2).strip() and m.group(1).lower() not in BOTS]
            if len(msgs) < 4:
                continue
            lt = letter_map(s for s, _ in msgs)
            turns = []
            for s, t in msgs:
                for nick, letter in lt.items():
                    t = re.sub(rf"(?<![\w-]){re.escape(nick)}(?![\w-])", letter, t)
                turns.append([lt[s], t])
            wins = []
            for s in range(0, len(turns), 8):
                win = turns[s : s + 8]
                wins.append(win if len(win) >= 3 and len({x[0] for x in win}) >= 2 and 15 <= idd.words(win) <= 200 else None)
            convs.append((f"chat-{ann.name.split('.')[0]}-{th[0]}", wins))
    return convs


def main() -> None:
    # Splits are by whole conversation (``dianoia.splits``, set in ``jepa-build.py``), so no window is dropped here.
    out = []
    for conv, wins in meetings() + chats():
        for k, w in enumerate(wins):
            if w is not None:
                out.append({"uid": f"{conv}/{k}", "kind": "original", "medium": conv.split("-")[0], "turns": w})
    path = work_path("jepa", "windows2.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"windows": len(out), "meeting": sum(r["medium"] == "meeting" for r in out),
                      "chat": sum(r["medium"] == "chat" for r in out),
                      "conversations": len({r["uid"].split("/")[0] for r in out})}))


if __name__ == "__main__":
    main()
