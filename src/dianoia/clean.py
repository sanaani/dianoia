"""Source-agnostic cleaning for conversation windows (10/04/2026), from what the audits caught on wiki and IRC data.

Every source adapter writes windows ``{"uid": "<source>-<conversation>/<k>", "turns": [[speaker, text], ...], ...}``
and passes them through here before the splitter (``prefilter``) and its ideas after (``near_copies``):

- **copies**: conversations sharing at least 2 long lines (30+ characters, normalized) are one cluster; only the
  cluster's first conversation is kept (wiki talk pages hold the same exchange under several ids);
- **templates**: a long line found in ``min_convs`` or more conversations is a template (bot factoids, deletion
  notices, welcome messages), and a window holding one is dropped: boilerplate a predictor learns by heart;
- **near-copies**: after splitting, conversations sharing at least 3 long ideas (8+ words) are one cluster too
  (a copy edited by a word shares ideas, not lines);
- **template ideas**: after splitting, a long idea (speaker letter ignored) found in ``min_convs`` or more
  conversations is a template, and a window holding 2 or more is dropped. Wikipedia's user-warning messages are
  reworded per use ("Please stop your disruptive editing", "use the sandbox"), so they pass the line filter but split
  into identical ideas: the J8 audit failed on 999 such units (10/04/2026). Two are required so a window keeping one
  generic line ("thanks for the quick response") stays;
- **cross-posts**: a held-out conversation sharing 3+ long ideas, and 30%+ of its own, with one training group (the
  audit's rule) is dropped: an editor posting the same barnstar on two user talk pages links two pages that the
  near-copy check cannot, because one of the shared ideas is too common to count as a link there. ``jepa-build.py``
  applies the same rule again to built units, after placeholders: deletion notices about two different articles
  become identical once both names are "Person 1" (6 units in J8, 10/04/2026).

A line or idea shared by more than ``cap`` conversations is boilerplate, not evidence of a copy, so it never links
two conversations (it is a template instead).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def conv_of(uid: str) -> str:
    return uid.split("/")[0].split("@")[0]


def _order(c: str):
    tail = c.rsplit("-", 1)[-1]
    return (0, int(tail), c) if tail.isdigit() else (1, 0, c)


def clusters(keys_of: dict[str, set[str]], min_shared: int, cap: int = 50) -> dict[str, str]:
    """Map each conversation to its cluster's representative (the first by id)."""
    by_key = defaultdict(set)
    for c, ks in keys_of.items():
        for k in ks:
            by_key[k].add(c)
    pairs = Counter()
    for cs in by_key.values():
        if 1 < len(cs) <= cap:
            cs = sorted(cs, key=_order)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    pairs[(cs[i], cs[j])] += 1
    parent = {c: c for c in keys_of}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for (a, b), k in pairs.items():
        if k >= min_shared:
            ra, rb = find(a), find(b)
            if ra != rb:
                first, second = sorted((ra, rb), key=_order)
                parent[second] = first
    return {c: find(c) for c in keys_of}


def prefilter(windows: list[dict], min_convs: int = 5) -> tuple[list[dict], dict]:
    """Drop copied conversations and windows holding a template line. Returns (kept windows, stats)."""
    lines_of = defaultdict(set)
    for w in windows:
        lines_of[conv_of(w["uid"])] |= {n for _, t in w["turns"] if len(n := norm(t)) >= 30}
    rep = clusters(lines_of, 2)
    convs_per_line = Counter(n for ls in lines_of.values() for n in ls)
    templates = {n for n, k in convs_per_line.items() if k >= min_convs}
    kept = [w for w in windows if rep[conv_of(w["uid"])] == conv_of(w["uid"])
            and not any(norm(t) in templates for _, t in w["turns"])]
    stats = {"conversations": len(lines_of), "copies_dropped": sum(rep[c] != c for c in rep),
             "template_lines": len(templates), "windows_in": len(windows), "windows_out": len(kept)}
    return kept, stats


def near_copies(preds: list[dict]) -> tuple[set[str], dict]:
    """Conversations to drop because they share 3+ long ideas with an earlier one. Returns (dropped, stats)."""
    ideas_of = defaultdict(set)
    for r in preds:
        ideas_of[conv_of(r["uid"])] |= {norm(i) for i in r["ideas"] if len(i.split()) >= 8}
    rep = clusters(ideas_of, 3)
    dropped = {c for c, r in rep.items() if r != c}
    return dropped, {"conversations": len(ideas_of), "near_copies_dropped": len(dropped)}


def _idea_key(i: str) -> str:
    return re.sub(r"^[a-z] ", "", norm(i))


def template_ideas(preds: list[dict], min_convs: int = 5, per_window: int = 2) -> tuple[set[str], dict]:
    """uids of windows holding ``per_window`` or more template ideas. Returns (dropped uids, stats)."""
    convs = defaultdict(set)
    for r in preds:
        for i in r["ideas"]:
            if len(i.split()) >= 8:
                convs[_idea_key(i)].add(conv_of(r["uid"]))
    templates = {k for k, cs in convs.items() if len(cs) >= min_convs}
    dropped = {r["uid"] for r in preds
               if sum(_idea_key(i) in templates for i in r["ideas"] if len(i.split()) >= 8) >= per_window}
    return dropped, {"template_ideas": len(templates), "template_windows_dropped": len(dropped)}



def heldout_repeats(rows: list[dict], key=conv_of) -> set[str]:
    """Held-out items (by ``key``: conversation for windows, uid for built units) sharing 3+ long ideas, and 30%+ of
    their own, with one training group: the audit's duplicated-unit rule. A row's split is its "split" field if set."""
    from dianoia.splits import group_of, unit_split

    in_train, held = defaultdict(set), defaultdict(set)
    for r in rows:
        ks = {norm(i) for i in r["ideas"] if len(i.split()) >= 8}
        if (r.get("split") or unit_split(r["uid"])) == "train":
            for k in ks:
                in_train[k].add(group_of(r["uid"]))
        else:
            held[key(r["uid"])] |= ks
    out = set()
    for c, ks in held.items():
        best = Counter(g for k in ks for g in in_train.get(k, ())).most_common(1)
        if best and best[0][1] >= 3 and best[0][1] / len(ks) >= 0.3:
            out.add(c)
    return out


def cross_posts(preds: list[dict]) -> tuple[set[str], dict]:
    """Held-out conversations that repeat one training group, before placeholders. Returns (dropped, stats)."""
    dropped = heldout_repeats(preds)
    return dropped, {"heldout_cross_posts_dropped": len(dropped)}
