"""G1 data: merged pull requests that were later reverted, each matched with a clean neighbor (10/05/2026).

The second outcome test (phase 9). The outcome is a revert: a merged change that had to be taken back. Public
GitHub, permissively licensed repositories only (MIT, Apache-2.0, BSD-3-Clause; checked 10/05/2026); Purpose
Factory's own pull requests are never collected here (they are only a private local check, later).

1. Revert PRs: GitHub search, merged PRs with "Revert" in the title, one query per repository and half-year (search
   returns at most 1,000 rows a query).
2. The reverted PR: "Reverts owner/repo#N" (GitHub's revert button) or its URL form, else a "#N" in the revert's title.
3. A control for each: PR numbers are issued in time order, so the merged PRs numbered within +-6 of the reverted one
   were written at about the same time. Of those that are not reverted, not bots and not reverts or relands, the one
   closest in size (lines added + deleted, log scale) is the control. Pairs share a repository and a moment, and
   roughly a size, so neither side wins on those.
4. Full text of both, only what existed before the merge: title, description (lines mentioning a revert removed: a
   description edited afterwards can name the outcome), and conversation and review comments made before mergedAt.
   Labels are not kept (some repositories label reverted PRs).

Every step appends to a file in --out and is skipped for rows already there, so a stopped run resumes.

    python scripts/gh-collect.py --out ~/dianoia-data/work/gh
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
import urllib.request
from pathlib import Path

REPOS = {  # spdx license, checked with the GitHub API on 10/05/2026
    "flutter/flutter": "BSD-3-Clause", "kubernetes/kubernetes": "Apache-2.0", "microsoft/vscode": "MIT",
    "dotnet/runtime": "MIT", "rust-lang/rust": "Apache-2.0", "home-assistant/core": "Apache-2.0",
    "vercel/next.js": "MIT", "apache/airflow": "Apache-2.0", "electron/electron": "MIT", "denoland/deno": "MIT",
    "supabase/supabase": "Apache-2.0", "microsoft/TypeScript": "Apache-2.0", "huggingface/transformers": "Apache-2.0",
}
YEARS = range(2015, 2027)
NEIGHBORS = 6
SKIP_TITLE = re.compile(r"^\s*(\[[^\]]*\]\s*)*(revert|reland|re-land|roll |bump |\[automated\])", re.I)
REVERT_LINE = re.compile(r"revert", re.I)
TOKEN = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    for attempt in range(8):
        req = urllib.request.Request("https://api.github.com/graphql", data=body,
                                     headers={"Authorization": f"bearer {TOKEN}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
        except Exception as e:  # 502s and timeouts on large queries: back off and retry
            print(json.dumps({"retry": attempt, "error": str(e)[:200]}), flush=True)
            time.sleep(10 * (attempt + 1))
            continue
        rl = (out.get("data") or {}).get("rateLimit")
        if rl and rl["remaining"] < 200:
            print(json.dumps({"rate_limited_until": rl["resetAt"]}), flush=True)
            time.sleep(max(60, (time.mktime(time.strptime(rl["resetAt"], "%Y-%m-%dT%H:%M:%SZ")) - time.mktime(time.gmtime())) + 30))
        if out.get("errors") and not out.get("data"):
            if any(e.get("type") == "RATE_LIMITED" for e in out["errors"]):
                time.sleep(300)
                continue
            raise RuntimeError(json.dumps(out["errors"])[:500])
        return out
    raise RuntimeError("graphql: retries exhausted")


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []


def search_reverts(out: Path) -> None:
    path = out / "reverts.jsonl"
    done = {(r["repo"], r["window"]) for r in read(out / "reverts-done.jsonl")}
    q = """query($q: String!, $after: String) { rateLimit { remaining resetAt }
      search(query: $q, type: ISSUE, first: 100, after: $after) { issueCount pageInfo { hasNextPage endCursor }
        nodes { ... on PullRequest { number title body mergedAt } } } }"""
    for repo in REPOS:
        for y in YEARS:
            for h, (a, b) in enumerate((("01-01", "06-30"), ("07-01", "12-31"))):
                window = f"{y}-{h}"
                if (repo, window) in done:
                    continue
                after, n = None, 0
                with open(path, "a", encoding="utf-8") as f:
                    while True:
                        res = gql(q, {"q": f"repo:{repo} is:pr is:merged in:title Revert created:{y}-{a}..{y}-{b}", "after": after})
                        s = res["data"]["search"]
                        for node in s["nodes"]:
                            if node:
                                f.write(json.dumps({"repo": repo, **node}, ensure_ascii=False) + "\n")
                                n += 1
                        if not s["pageInfo"]["hasNextPage"]:
                            break
                        after = s["pageInfo"]["endCursor"]
                with open(out / "reverts-done.jsonl", "a") as f:
                    f.write(json.dumps({"repo": repo, "window": window, "n": n, "total": s["issueCount"]}) + "\n")
                if n:
                    print(json.dumps({"repo": repo, "window": window, "reverts": n}), flush=True)


def reverted_numbers(out: Path) -> dict[str, set[int]]:
    """repo -> numbers of PRs that a merged revert PR names."""
    by: dict[str, set[int]] = {}
    for r in read(out / "reverts.jsonl"):
        # 'Revert "..."', '[3.12] Revert ...' and conventional 'chore(lsp): revert ...' titles.
        if not re.match(r"^\s*(\[[^\]]*\]\s*)*(\w+(\([^)]*\))?!?:\s*)?revert", r["title"], re.I):
            continue
        body, rp = r["body"] or "", re.escape(r["repo"])
        m = re.search(r"Reverts\s+" + rp + r"#(\d+)", body, re.I) or re.search(r"Reverts\s+#(\d+)", body, re.I) \
            or re.search(r"Reverts\s+https://github\.com/" + rp + r"/pull/(\d+)", body, re.I) or re.search(r"#(\d+)", r["title"])
        if m and int(m.group(1)) != r["number"]:
            by.setdefault(r["repo"], set()).add(int(m.group(1)))
    return by


LIGHT = "number title mergedAt additions deletions author { login __typename }"
FULL = LIGHT + """ body createdAt changedFiles
  comments(first: 30) { nodes { body createdAt author { login __typename } } }
  reviews(first: 20) { nodes { body state submittedAt author { login __typename }
    comments(first: 10) { nodes { body createdAt } } } }"""


def fetch(repo: str, numbers: list[int], fields: str, per: int) -> dict[int, dict]:
    owner, name = repo.split("/")
    got = {}
    for s in range(0, len(numbers), per):
        b = numbers[s : s + per]
        alias = " ".join(f"p{n}: pullRequest(number: {n}) {{ {fields} }}" for n in b)
        res = gql(f'query {{ rateLimit {{ remaining resetAt }} repository(owner: "{owner}", name: "{name}") {{ {alias} }} }}')
        for n in b:
            node = ((res.get("data") or {}).get("repository") or {}).get(f"p{n}")
            if node:  # an issue number, or a PR that is gone: no node
                got[n] = node
    return got


def is_bot(pr: dict) -> bool:
    a = pr.get("author") or {}
    login = (a.get("login") or "").lower()
    return a.get("__typename") == "Bot" or login.endswith("[bot]") or "bot" in login or "roller" in login


def eligible(pr: dict | None) -> bool:
    return bool(pr and pr.get("mergedAt") and not is_bot(pr) and not SKIP_TITLE.match(pr["title"]))


def size(pr: dict) -> float:
    return math.log1p(pr["additions"] + pr["deletions"])


def pre_merge_text(pr: dict) -> dict:
    merged = pr["mergedAt"]
    body = "\n".join(l for l in (pr.get("body") or "").split("\n") if not REVERT_LINE.search(l)).strip()
    convo = []
    for c in pr["comments"]["nodes"]:
        if c["createdAt"] < merged and not is_bot(c):
            convo.append({"at": c["createdAt"], "who": (c.get("author") or {}).get("login"), "text": c["body"]})
    for r in pr["reviews"]["nodes"]:
        if (r.get("submittedAt") or merged) < merged and not is_bot(r):
            if r["body"].strip():
                convo.append({"at": r["submittedAt"], "who": (r.get("author") or {}).get("login"), "text": r["body"], "review": r["state"]})
            for c in r["comments"]["nodes"]:
                if c["createdAt"] < merged:
                    convo.append({"at": c["createdAt"], "who": (r.get("author") or {}).get("login"), "text": c["body"], "review": "line"})
    convo = [c for c in sorted(convo, key=lambda c: c["at"]) if c["text"].strip() and not REVERT_LINE.search(c["text"])]
    return {"title": pr["title"], "body": body, "author": (pr.get("author") or {}).get("login"), "created": pr["createdAt"],
            "merged": merged, "additions": pr["additions"], "deletions": pr["deletions"], "files": pr["changedFiles"],
            "conversation": convo}


def build_pairs(out: Path, per_repo: int) -> None:
    path = out / "pairs.jsonl"
    done = {(p["repo"], p["reverted"]) for p in read(path)} | {(p["repo"], p["reverted"]) for p in read(out / "pairs-skipped.jsonl")}
    by = reverted_numbers(out)
    print(json.dumps({"reverted_named": {r: len(v) for r, v in by.items()}}), flush=True)
    for repo, nums in by.items():
        todo = sorted(n for n in nums if (repo, n) not in done)[-per_repo:]
        for s in range(0, len(todo), 10):
            batch = todo[s : s + 10]
            # A low number (a misparsed "#1" in a revert title) once asked for PR -1, which GraphQL rejects.
            near = sorted(m for m in {m for n in batch for m in range(n - NEIGHBORS, n + NEIGHBORS + 1)} - nums | set(batch) if m > 0)
            light = fetch(repo, near, LIGHT, 50)
            chosen, skipped = {}, []
            for n in batch:
                orig = light.get(n)
                cands = [light[m] for m in range(n - NEIGHBORS, n + NEIGHBORS + 1)
                         if m != n and m not in nums and eligible(light.get(m))]
                if not eligible(orig) or not cands:
                    skipped.append({"repo": repo, "reverted": n, "why": "original" if not eligible(orig) else "no control"})
                    continue
                chosen[n] = min(cands, key=lambda c: abs(size(c) - size(orig)))["number"]
            full = fetch(repo, sorted(set(chosen) | set(chosen.values())), FULL, 10)
            with open(path, "a", encoding="utf-8") as f:
                for n, c in chosen.items():
                    if n in full and c in full:
                        f.write(json.dumps({"repo": repo, "license": REPOS[repo], "reverted": n, "control": c,
                                            "bad": pre_merge_text(full[n]), "good": pre_merge_text(full[c])}, ensure_ascii=False) + "\n")
            with open(out / "pairs-skipped.jsonl", "a", encoding="utf-8") as f:
                f.writelines(json.dumps(x) + "\n" for x in skipped)
        print(json.dumps({"repo": repo, "pairs": sum(1 for p in read(path) if p["repo"] == repo)}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-repo", type=int, default=1000, help="most recent reverted PRs kept per repository")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    search_reverts(out)
    build_pairs(out, args.per_repo)


if __name__ == "__main__":
    main()
