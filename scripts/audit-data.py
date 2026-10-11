"""Gate before any training run: audit a dataset for leaks and known data problems (10/04/2026).

Writes ``<units>.audit.json`` holding the units file's sha256, the manifest's, every check's numbers and ``pass``.
``remote-jepa.py --require-audit`` refuses to train on a file without a passing audit of exactly that content.

Failing checks (each one broke a result on 10/04/2026):
- **split consistency**: every unit's split equals ``dianoia.splits`` for its group, and no group is in two splits;
- **duplicated conversations**: no val or test window has at least 2, and at least 30%, of its long turns (30+
  characters, normalized) in one training conversation (``--turns``: the files holding the units' source turns). Independent
  conversations share common lines ("sudo apt-get update", channel macros, "yeah, that's true"); those are reported
  as a warning, since a leak is a conversation or much of one in two splits. Checked by planting duplicates;
- **duplicated units**: no val or test unit has at least 3, and at least 30%, of its 8+-word ideas in one training
  group (the raw idea-overlap rate is reported as a warning above 0.5%);
- **doubled speaker prefixes**: at most 0.1% of turns begin with their own speaker letter ("A: A: ...", the Sonnet
  echo of 10/04; real IRC has a few: a user addressing a nickname that maps to their own letter);
- **broken ideas**: at most 2% of ideas are under 3 words or repeat a 3-word phrase 3 times.
Warnings (reported, not failing): person names that escaped placeholders (spaCy PERSON on a sample of 3,000 ideas; a
known gap: single first names), and units with explicit terms (to filter before anything is published).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict

from dianoia.paths import work_path
from dianoia.splits import group_of, split_of, unit_split

EXPLICIT = re.compile(r"\b(fuck\w*|shit\w*|cunt|porn\w*|vagina\w*|penis\w*|sex|sexual\w*|nigg\w*|fag\w*|rape\w*)\b", re.I)
PLACEHOLDER = re.compile(r"[A-Z]|(?:Person|Link|Email|Phone|Code) \d+|[A-Z](?: and [A-Z])+")


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def broken(i: str) -> bool:
    w = i.lower().split()
    if len(w) < 3:
        return True
    grams = Counter(tuple(w[k : k + 3]) for k in range(len(w) - 2))
    return bool(grams) and max(grams.values()) >= 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("units", help="unit file under jepa/")
    ap.add_argument("--turns", nargs="*", default=[], help="files with uid + turns (paths under work/)")
    args = ap.parse_args()
    path = work_path("jepa", args.units)
    raw = path.read_bytes()
    units = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    res, fails, warns = {}, [], []

    # Split consistency.
    bad = [u["uid"] for u in units if u["split"] != unit_split(u["uid"])]
    where = defaultdict(set)
    for u in units:
        where[group_of(u["uid"])].add(u["split"])
    multi = [":".join(g) for g, s in where.items() if len(s) > 1]
    res["split_mismatch"], res["groups_in_two_splits"] = len(bad), len(multi)
    if bad or multi:
        fails.append("split consistency")

    # Turn overlap across splits and doubled prefixes, from the source turns.
    turn_split, turn_groups, held_windows = defaultdict(set), defaultdict(set), []
    doubled = 0
    for name in args.turns:
        for l in open(work_path(*name.split("/")), encoding="utf-8"):
            r = json.loads(l)
            s, g = unit_split(r["uid"]), group_of(r["uid"])
            long_turns = []
            for sp, t in r["turns"]:
                if re.match(rf"^{re.escape(sp)}:\s", t):
                    doubled += 1
                n = norm(t)
                if len(n) >= 30:
                    turn_split[n].add(s)
                    turn_groups[n].add(g)
                    long_turns.append(n)
            if s != "train" and long_turns:
                held_windows.append(sorted(set(long_turns)))  # distinct lines: a line repeated in one window counts once
    # A leak is a conversation (or much of one) in two splits, not a common line: a held-out window counts as
    # duplicated when at least 2 and at least 30% of its long turns occur in ONE training conversation (generic lines
    # match many different conversations; a copied conversation matches one).
    train_turn_groups = {n: {g for g in gs if split_of(*g) == "train"} for n, gs in turn_groups.items()}

    def best_group(items, index):
        c = Counter(g for n in items for g in index.get(n, ()))
        return max(c.values(), default=0)

    dup = sum(1 for w in held_windows if (k := best_group(w, train_turn_groups)) >= 2 and k / len(w) >= 0.3)
    boiler = sum(1 for n, s in turn_split.items() if "train" in s and len(s) > 1)
    res.update({"turns_checked": len(turn_split), "heldout_windows": len(held_windows), "heldout_windows_duplicated": dup,
                "common_lines_shared_across_splits": boiler,
                "common_line_examples": [n[:80] for n, s in turn_split.items() if "train" in s and len(s) > 1][:8],
                "doubled_prefix_turns": doubled})
    if dup:
        fails.append("duplicated conversations across splits")
    if boiler:
        warns.append("common lines shared across splits (boilerplate, not a leak)")
    n_turns = sum(1 for _ in turn_split) or 1
    res["doubled_prefix_rate"] = round(doubled / n_turns, 5)
    if doubled / n_turns > 0.001:
        fails.append("doubled speaker prefixes")

    # Idea overlap, broken ideas, names, explicit content.
    train_groups = defaultdict(set)
    for u in units:
        if u["split"] == "train":
            for i in u["ideas"]:
                train_groups[norm(i)].add(u.get("group") or ":".join(group_of(u["uid"])))
    held = [norm(i) for u in units if u["split"] != "train" for i in u["ideas"] if len(i.split()) >= 8]
    overlap = sum(0 < len(train_groups.get(i, ())) <= 2 for i in held) / max(len(held), 1)
    # Same rule for ideas: a held-out unit is duplicated when at least 3 and at least 30% of its 8+-word ideas occur
    # in one training group.
    train_groups = {i: {tuple(g.split(":", 1)) for g in gs} for i, gs in train_groups.items()}
    dup_units = 0
    for u in units:
        if u["split"] != "train":
            long_i = [norm(i) for i in u["ideas"] if len(i.split()) >= 8]
            k = best_group(long_i, train_groups)
            dup_units += bool(long_i) and k >= 3 and k / len(long_i) >= 0.3
    res["heldout_units_duplicated"] = dup_units
    ideas = [i for u in units for i in u["ideas"]]
    br = sum(broken(i) for i in ideas) / max(len(ideas), 1)
    import random

    import spacy

    nlp = spacy.load("en_core_web_sm")
    sample = random.Random(0).sample(ideas, min(3000, len(ideas)))
    persons = [[e.text for e in d.ents if e.label_ == "PERSON" and not PLACEHOLDER.fullmatch(e.text)] for d in nlp.pipe(sample)]
    names = Counter(n for p in persons for n in p)
    named = sum(bool(p) for p in persons) / max(len(sample), 1)
    explicit = [u["uid"] for u in units if any(EXPLICIT.search(i) for i in u["ideas"])]
    res.update({"heldout_ideas_8plus": len(held), "idea_overlap_rate": round(overlap, 4), "broken_idea_rate": round(br, 4),
                "ideas_with_name_like_word": round(named, 4), "top_name_like_words": names.most_common(15),
                "units_with_explicit_terms": len(explicit), "explicit_examples": explicit[:10]})
    if dup_units:
        fails.append("duplicated units across splits")
    if overlap > 0.005:
        warns.append("idea overlap above 0.5% (generic ideas; duplicated units are the failing check)")
    if br > 0.02:
        fails.append("broken ideas")
    if named > 0.02:
        warns.append("name-like words escaped placeholders")
    if explicit:
        warns.append("explicit content (filter before publishing)")

    out = {"units_file": args.units, "units_sha256": hashlib.sha256(raw).hexdigest(),
           "manifest_sha256": hashlib.sha256(work_path("jepa", args.units + ".manifest.json").read_bytes()).hexdigest()
           if work_path("jepa", args.units + ".manifest.json").exists() else None,
           "pass": not fails, "fails": fails, "warnings": warns, **res}
    work_path("jepa", args.units + ".audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pass", "fails", "warnings")}))
    print(json.dumps({k: v for k, v in res.items() if k not in ("top_name_like_words", "explicit_examples")}))


if __name__ == "__main__":
    main()
