"""Score a stage-1 idea-object checkpoint (``remote-idea.py``'s ``decoded-<tag>.jsonl``).

On the check's held-out units, never trained on:

- **fidelity**: the object decoded from the original vs Sonnet's object of the original;
- **invariance**: the object decoded from each keep version (paraphrase, stripped, disfluent, split) vs the one
  decoded from the original, beside Sonnet's own consistency on the same pairs (the ceiling);
- **sensitivity**: the move version's decode differs at the edited turn;
- **words** (``--words``): Sonnet writes the unit back out from the decoded object alone, and the field-by-field
  intent check (``CHECK``, ``score-tier2.CHECK`` made medium-neutral) compares it with the original text. This is the round trip: words ->
  vector -> idea object -> words.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dianoia import ideas, llm
from dianoia.paths import result_path, work_path


KEEP = ["paraphrase", "stripped", "disfluent", "split"]
LAYERS = ["event_f1", "act_coarse", "neg", "mod", "pred", "content_cos", "center"]
CHECK = """Compare each ORIGINAL unit of communication (an email sentence or paragraph, a meeting excerpt, or a chat excerpt, as "<speaker>: <text>" turns) with a RECONSTRUCTION. Wording, punctuation, fillers and turn boundaries do not matter. Judge four things over the whole unit:
- "act": the same speech acts (statement, question, request, offer, agreement, thanks...) in the same order?
- "action": the same actions and things acted on, with nothing material missing or added?
- "roles": the same parties in the same roles (who does what to whom)?
- "quantities": the same material numbers, amounts, dates (true if neither has any)?

Return only a JSON array: [{{"p": <pair number>, "act": true/false, "action": true/false, "roles": true/false, "quantities": true/false}}, ...]

{pairs}"""
REALIZE = """Each item is an IDEA OBJECT: the structure of a short unit of communication (an email sentence or \
paragraph, a meeting excerpt, or a chat excerpt). Each event is "t<turn> [<speaker letter>] <dialogue act> <verb> [<agent> ; <patient \
or theme> ; <recipient or end state>] <yes|not> <modality> <time>"; ">> a b label" says event a attaches to event b \
by that discourse relation; "* k" marks the main point. Speakers are letters.

Write the unit it describes, as plain natural text, one turn per line as "<speaker letter>: <text>". Say exactly \
what the object says, no more.

Return only a JSON array: [{{"i": <number>, "text": "A: ...\\nB: ..."}}, ...]

{items}"""


def realize(objs: list[str]) -> list[str | None]:
    prompts = [REALIZE.format(items="\n\n".join(f"Item {k}: {o}" for k, o in enumerate(objs[s : s + 10])))
               for s in range(0, len(objs), 10)]

    def safe(p):
        try:
            return llm.ask_json(p, model="sonnet")
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=5) as ex:
        answers = list(ex.map(safe, prompts))
    out = [None] * len(objs)
    for b, ans in enumerate(answers):
        for a in ans if isinstance(ans, list) else []:
            if isinstance(a, dict) and str(a.get("i", "")).isdigit() and isinstance(a.get("text"), str):
                k = b * 10 + int(a["i"])
                if k < len(out):
                    out[k] = a["text"]
    return out


def check(pairs: list[tuple[str, str]]) -> list[bool | None]:
    prompts = [CHECK.format(pairs="\n\n".join(f"PAIR {k}\nORIGINAL: {a}\nRECONSTRUCTION: {b}"
                                                          for k, (a, b) in enumerate(pairs[s : s + 12])))
               for s in range(0, len(pairs), 12)]

    def safe(p):
        try:
            return llm.ask_json(p, model="sonnet")
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=5) as ex:
        answers = list(ex.map(safe, prompts))
    out = [None] * len(pairs)
    for b, ans in enumerate(answers):
        for a in ans if isinstance(ans, list) else []:
            if isinstance(a, dict) and str(a.get("p", "")).isdigit() and b * 12 + int(a["p"]) < len(out):
                out[b * 12 + int(a["p"])] = all(bool(a.get(f)) for f in ["act", "action", "roles", "quantities"])
    return out


def mean(d):
    return {k: round(float(np.mean(v)), 3) for k, v in d.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("decoded")
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", action="store_true")
    args = ap.parse_args()
    sonnet = {(r["uid"], r["kind"]): {**r, "object": ideas.reduced(r["object"])}
              for r in map(json.loads, open(work_path("ideas", "objects.jsonl"), encoding="utf-8"))}
    dec = {(r["uid"], r["kind"]): r for r in map(json.loads, open(args.decoded, encoding="utf-8"))}
    parsed = {k: ideas.parse(r["decoded"]) for k, r in dec.items()}
    al = ideas.Aligner()
    fid, inv, ceil, sens = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    uids = sorted({u for u, k in dec if k == "original"})
    for u in uids:
        so, do = sonnet.get((u, "original")), parsed.get((u, "original"))
        medium = sonnet[(u, "original")]["medium"]
        if so is None or so["object"] is None:
            continue
        fid[medium]["parsed"].append(do is not None)
        if do is None:
            continue
        for k, x in ideas.compare(so["object"], do, al.align(so["object"], do, list(range(len(so["turns"]))))).items():
            fid[medium][k].append(float(x))
        for kind in KEEP + ["move"]:
            v, dv = sonnet.get((u, kind)), parsed.get((u, kind))
            if v is None or dv is None:
                continue
            pairs = al.align(do, dv, v["turn_map"])
            if kind == "move":
                sens[medium].append(ideas.changed_at(do, dv, pairs, v["changed_turn"], v["turn_map"]))
                continue
            for k, x in ideas.compare(do, dv, pairs).items():
                inv[f"{medium}/{kind}"][k].append(float(x))
            if v["object"] is not None:
                for k, x in ideas.compare(so["object"], v["object"], al.align(so["object"], v["object"], v["turn_map"])).items():
                    ceil[f"{medium}/{kind}"][k].append(float(x))
    report = {"decoded": args.decoded, "fidelity": {m: mean(d) for m, d in fid.items()},
              "invariance": {c: mean(d) for c, d in inv.items()}, "sonnet_ceiling": {c: mean(d) for c, d in ceil.items()},
              "sensitivity": {m: round(float(np.mean(x)), 3) for m, x in sens.items()}}

    if args.words:
        rows = [(u, sonnet[(u, "original")]) for u in uids if dec.get((u, "original"))]
        texts = realize([dec[(u, "original")]["decoded"] for u, _ in rows])
        src = ["\n".join(f"{s}: {t}" for s, t in r["turns"]) for _, r in rows]
        ok = check([(a, b or "") for a, b in zip(src, texts)])
        by = defaultdict(list)
        samples = []
        for (u, r), t, o in zip(rows, texts, ok):
            if o is not None:
                by[r["medium"]].append(o)
            if len(samples) < 30:
                samples.append({"uid": u, "original": r["turns"], "decoded": dec[(u, "original")]["decoded"], "words": t, "intent_ok": o})
        report["words_intent"] = {m: {"n": len(x), "pass": round(float(np.mean(x)), 3)} for m, x in by.items()}
        report["words_samples"] = samples
    result_path("phase-4-idea-objects", f"{args.out}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for sec in ("fidelity", "invariance", "sensitivity", "words_intent"):
        print(sec, json.dumps(report.get(sec), indent=None))


if __name__ == "__main__":
    main()
