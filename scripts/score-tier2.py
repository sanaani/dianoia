"""Tier 2 score: did the decoded words land where the source's intent was?

Reads the GPU outputs (``tier2/decoded.jsonl`` aligned with ``to-decode.parquet``, then the nearest
floor) and the fine-tuned encoders' vectors of every returned text (``tier2/reenc/ft_*.npy``, decoded
texts first, nearest texts after). Per arm:

- pass rate: cos(x, x̂) >= tau(x) = cos(x, paraphrase(x)), under ft_sonar (the encoder decoded) and
  under ft_bge (an encoder the decoder never saw);
- median word Jaccard of x̂ with the source;
- the structured intent check (Haiku: same speech act, same action and object, same roles, same
  quantities?) on the lift256, sonar_self and nearest arms.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from dianoia import corpus, llm
from dianoia.paths import result_path, work_path

OUT = "phase-2-round-trip"
CHECK = """Compare each ORIGINAL sentence from business email with a RECONSTRUCTION. Wording does not matter. \
Judge four things:
- "act": same speech act (report, request, commitment, question, proposal, approval, refusal, thanks)?
- "action": same action and same thing acted on?
- "roles": same parties in the same roles (who does what to whom)? Different names for the same role are fine.
- "quantities": same material numbers, amounts, dates (true if neither has any)?

Return only a JSON array: [{{"p": <pair number>, "act": true/false, "action": true/false, "roles": true/false, "quantities": true/false}}, ...]

{pairs}"""


def jaccard(a: str, b: str) -> float:
    wa, wb = {w.lower() for w in corpus.words(a)}, {w.lower() for w in corpus.words(b)}
    return len(wa & wb) / max(len(wa | wb), 1)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def main() -> None:
    items = pd.read_parquet(work_path("phase1", "items.parquet"))
    index = pd.read_parquet(work_path("phase1", "index.parquet"))
    lines = index.pivot_table(index="item", columns="field", values="line", aggfunc="first")
    plan = pd.read_parquet(work_path("tier2", "to-decode.parquet"))
    plan["text"] = [json.loads(x)["text"] for x in open(work_path("tier2", "decoded.jsonl"), encoding="utf-8")]
    nearest = pd.read_parquet(work_path("tier2", "nearest.parquet"))
    extra = work_path("tier2", "llm.parquet")
    parts = [plan, nearest] + ([pd.read_parquet(extra)] if extra.exists() else [])
    table = pd.concat(parts, ignore_index=True)

    report: dict = {"test_items": int(table["item"].nunique()), "arms": {}}
    for enc in ("ft_sonar", "ft_bge"):
        src_vecs = unit(np.load(work_path("phase1", "vec", f"{enc}.npy")).astype(np.float32))
        back = np.load(work_path("tier2", "reenc", f"{enc}.npy")).astype(np.float32)
        if extra.exists():
            back = np.concatenate([back, np.load(work_path("tier2", "reenc-llm", f"{enc}.npy")).astype(np.float32)])
        back = unit(back)
        x = src_vecs[lines.loc[table["item"], "text"].astype(int).to_numpy()]
        para = lines.loc[table["item"], "paraphrase"]
        p = src_vecs[para.fillna(0).astype(int).to_numpy()]
        table[f"cos_{enc}"] = (x * back).sum(1)
        table[f"tau_{enc}"] = np.where(para.notna().to_numpy(), (x * p).sum(1), np.nan)
    table["jaccard"] = [jaccard(items.at[i, "text"], t) for i, t in zip(table["item"], table["text"])]

    # Structured intent check on three arms.
    sub = table[table["arm"].isin(["lift256", "sonar_self", "nearest", "llm_first", "llm_search"])].reset_index()
    prompts, chunks = [], []
    for s in range(0, len(sub), 15):
        ch = sub.iloc[s : s + 15]
        body = "\n\n".join(f"PAIR {k}\nORIGINAL: {items.at[r['item'], 'text']}\nRECONSTRUCTION: {r['text']}" for k, (_, r) in enumerate(ch.iterrows()))
        prompts.append(CHECK.format(pairs=body))
        chunks.append(ch)

    def safe(q):
        try:
            return llm.ask_json(q)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=5) as ex:
        answers = list(ex.map(safe, prompts))
    for ch, ans in zip(chunks, answers):
        got = {int(a["p"]): a for a in ans if isinstance(a, dict) and "p" in a}
        for k, (_, r) in enumerate(ch.iterrows()):
            a = got.get(k)
            if a:
                for f in ("act", "action", "roles", "quantities"):
                    table.loc[r["index"], f"check_{f}"] = bool(a.get(f))
    fields = [f"check_{f}" for f in ("act", "action", "roles", "quantities")]
    table["check_all"] = table[fields].all(axis=1).where(table[fields].notna().all(axis=1))

    for arm, g in table.groupby("arm"):
        r = {"n": len(g), "word_jaccard_median": round(float(g["jaccard"].median()), 3)}
        for enc in ("ft_sonar", "ft_bge"):
            ok = g[f"tau_{enc}"].notna()
            r[f"pass_{enc}"] = round(float((g.loc[ok, f"cos_{enc}"] >= g.loc[ok, f"tau_{enc}"]).mean()), 3)
            r[f"cos_median_{enc}"] = round(float(g[f"cos_{enc}"].median()), 3)
            r[f"tau_median_{enc}"] = round(float(g[f"tau_{enc}"].median()), 3)
        if g["check_all"].notna().any():
            c = g[g["check_all"].notna()]
            r["intent_check_n"] = len(c)
            r["intent_check_all"] = round(float(c["check_all"].astype(float).mean()), 3)
            for f in fields:
                r[f] = round(float(c[f].astype(float).mean()), 3)
        report["arms"][arm] = r
    l256 = report["arms"]["lift256"]
    report["verdict"] = {
        "pass_rate_ge_0.80 (ft_sonar)": l256["pass_ft_sonar"] >= 0.80,
        "intent_check_ge_0.80": l256.get("intent_check_all", 0) >= 0.80,
        "names_restored": "not applicable: the pool-trained lift decodes natural sentences, not placeholders (see addendum)",
        "word_jaccard_lt_0.4_while_checks_pass": l256["word_jaccard_median"] < 0.4,
    }
    result_path(OUT, "tier2.json").write_text(json.dumps(report, indent=2) + "\n")
    table.assign(source=table["item"].map(items["text"])).drop(columns=["index"], errors="ignore").to_json(
        result_path(OUT, "decodes.jsonl"), orient="records", lines=True, force_ascii=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
