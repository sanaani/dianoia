"""G1 data: gh-collect.py pairs in the files remote-cmv.py reads (10/05/2026).

Each pull request becomes one text: its title and description by its author (speaker A), then the conversation before
the merge, reviewers lettered B, C, ... in order of first comment. Code blocks become "[code]" and HTML comments
(PR templates) are dropped: the splitter and both models read discussion, not code. The text model and the idea
models see exactly the same text.

``win`` is the clean control, ``lose`` the reverted PR, so "picks the winner" means "picks the PR that was not
reverted". There is no shared original post: each pair gets its own one-line context ("A pull request to <repo>."),
which keeps remote-cmv.py's per-post validation split a per-pair split. Split by time: in each repository the newest
20% of pairs (by the reverted PR's merge time) are held out, older ones train, so the test is "predict future reverts".

    python scripts/gh-build.py --gh ~/dianoia-data/work/gh --out ~/dianoia-data/work/gh/g1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CODE = re.compile(r"```.*?(```|$)", re.S)
HTML_COMMENT = re.compile(r"<!--.*?(-->|$)", re.S)
MAX_TURN_WORDS = 400


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean(t: str) -> str:
    t = HTML_COMMENT.sub("", CODE.sub(" [code] ", t or ""))
    t = re.sub(r"`([^`\n]{40,})`", "[code]", t)
    # Names become placeholders: email addresses first, then @mentions.
    t = re.sub(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", "someone@example.com", t)
    t = re.sub(r"(?<![\w.])@[A-Za-z0-9][\w-]*", "@someone", t)
    words = " ".join(t.split()).split(" ")
    return " ".join(words[:MAX_TURN_WORDS]).strip()


def turns_of(pr: dict) -> list[list[str]]:
    letters = {pr["author"]: "A"}
    out = [["A", clean(pr["title"] + ". " + pr["body"])]]
    for c in pr["conversation"]:
        who = c["who"] or "?"
        if who not in letters:
            letters[who] = chr(ord("A") + min(len(letters), 25))
        t = clean(c["text"])
        if t:
            out.append([letters[who], t])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gh", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gh, out = Path(args.gh).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cmv = load("cmv_build", "cmv-build.py")
    pairs = [json.loads(l) for l in open(gh / "pairs.jsonl", encoding="utf-8")]
    seen, uniq = set(), []
    for p in pairs:
        if (p["repo"], p["reverted"]) not in seen:
            seen.add((p["repo"], p["reverted"]))
            uniq.append(p)
    by_repo = defaultdict(list)
    for p in uniq:
        by_repo[p["repo"]].append(p)
    rows, texts, windows = [], {}, []
    for repo, ps in by_repo.items():
        ps.sort(key=lambda p: p["bad"]["merged"])
        cut = int(len(ps) * 0.8)
        for k, p in enumerate(ps):
            key = f"{repo}#{p['reverted']}"
            ctx, good, bad = f"ctx:{key}", f"pr:{repo}#{p['control']}", f"pr:{key}"
            rows.append({"pair": key, "repo": repo, "split": "train" if k < cut else "test", "op": ctx, "win": good, "lose": bad,
                         "size": [p["good"]["additions"] + p["good"]["deletions"], p["bad"]["additions"] + p["bad"]["deletions"]]})
            texts[ctx] = [["A", f"A pull request to {repo}."]]
            texts[good], texts[bad] = turns_of(p["good"]), turns_of(p["bad"])
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(out / "texts.jsonl", "w", encoding="utf-8") as f:
        for t, turns in texts.items():
            f.write(json.dumps({"id": t, "text": "\n\n".join(f"{sp}: {x}" for sp, x in turns)}, ensure_ascii=False) + "\n")
    # Splitter windows: consecutive turns up to about cmv.WORDS words (a longer turn is cut into its own windows).
    for t, turns in texts.items():
        k, cur, n = 0, [], 0
        pieces = []
        for sp, x in turns:
            for w in cmv.windows(x) if len(x.split()) > cmv.WORDS else [x]:
                pieces.append([sp, w])
        for sp, w in pieces:
            if cur and n + len(w.split()) > cmv.WORDS:
                windows.append({"uid": f"{t}#{k}", "kind": "original", "medium": "github", "turns": cur})
                k, cur, n = k + 1, [], 0
            cur.append([sp, w])
            n += len(w.split())
        if cur:
            windows.append({"uid": f"{t}#{k}", "kind": "original", "medium": "github", "turns": cur})
    with open(out / "windows.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(w, ensure_ascii=False) + "\n" for w in windows)
    print(json.dumps({"pairs": len(rows), "by_split": Counter(r["split"] for r in rows), "by_repo": Counter(r["repo"] for r in rows),
                      "texts": len(texts), "windows": len(windows),
                      "bigger_is_reverted": round(sum(r["size"][1] > r["size"][0] for r in rows) / max(len(rows), 1), 3)}))


if __name__ == "__main__":
    main()
