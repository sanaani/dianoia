"""Phase 10 GPU stage: role replacement, encoding, historical recovery, domain leakage, H3 (10/05/2026).

    python remote-p10.py --work p10/gpu --config p10/config.used.yaml --out p10/out [--smoke 20]

Inputs in --work (from pipelines/phase10): cands.jsonl (header: generality thresholds, lexicon of every noun and verb
with its WordNet hypernym candidates, substitutes' frequency and spread; then one line per abstract with tokens,
coarse tags and lemmas), queries.jsonl, h3_words.jsonl.

Stages, each written to --out and skipped when its file exists (a spot reclaim costs one stage):
  A scores.pt     Qwen3 base LM, one forward pass per abstract: at each noun and verb, the log-probability of each
                  hypernym candidate's first token given the words before it (in context, left side only).
  B variants      for each (noun level, verb level), every abstract rewritten: a word whose generality is below the
                  level's threshold becomes its most probable candidate at or above it (kept if none qualifies). Plus
                  nouns-only and verbs-only lemma strings.
  C emb/<v>.pt    Qwen3-Embedding: queries with a retrieval instruction, candidates without; last-token pooling.
  D metrics       per variant: top-10 hit and reciprocal rank per query; leakage = 5-fold accuracy of a linear
                  classifier guessing a candidate's CPC section from its vector (chance 1/8).
  E h3            Qwen3 hidden states for domain words and hypernyms; ridge map word -> hypernym fit on 5 sections,
                  scored on 3 held-out ones against frequency-only and identity baselines.
Writes gpu.json. No verdicts here: pipelines/phase10 judges, from the per-query arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

SECTIONS = "ABCDEFGH"
# bfloat16 on the GPU; float32 on the laptop CPU, where bfloat16 ran 4 abstracts a minute in the dry run (10/05/2026).
DTYPE = {"cuda": torch.bfloat16, "cpu": torch.float32}  # passed as torch_dtype: the instance image has an older transformers
INSTRUCT = "Instruct: Given a patent abstract, retrieve an earlier patent from a different field whose idea it borrows\nQuery: "


def log(**kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), **kw}), flush=True)


def load_lm(name, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=DTYPE[dev]).to(dev).eval()
    return tok, m


def text_of(tokens):
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t) + 1
    return " ".join(tokens), starts


# ---- A: substitute scores ----------------------------------------------------------------------------------------

def stage_scores(docs, lex, cfg, dev, out, bs):
    f = out / "scores.pt"
    if f.exists():
        return torch.load(f)
    tok, m = load_lm(cfg["models"]["substitute_lm"], dev)
    first = {}

    def fid(w):
        if w not in first:
            first[w] = tok(" " + w, add_special_tokens=False)["input_ids"][0]
        return first[w]

    scores = {}
    ids = sorted(docs)
    t0 = time.time()
    for b in range(0, len(ids), bs):
        batch = ids[b:b + bs]
        texts, starts = zip(*(text_of(docs[d]["tokens"]) for d in batch))
        enc = tok(list(texts), return_offsets_mapping=True, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            hid = m.model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev)).last_hidden_state
        for j, d in enumerate(batch):
            doc, offs = docs[d], enc["offset_mapping"][j].tolist()
            want = []  # (word index, token position predicting it, candidate token ids)
            for i, (p, l) in enumerate(zip(doc["pos"], doc["lemmas"])):
                e = lex.get(f"{l}|{p}")
                if p == "o" or not e or not e["cands"]:
                    continue
                cs = starts[j][i]
                t = next((k for k, (a, z) in enumerate(offs) if z > cs and z > a), None)
                if t is None or t == 0:
                    continue
                want.append((i, t - 1, [fid(c) for c, _, _ in e["cands"]]))
            if not want:
                scores[d] = {}
                continue
            rows = hid[j, [w[1] for w in want]]
            lp = torch.log_softmax(m.lm_head(rows).float(), -1)
            scores[d] = {w[0]: lp[k, w[2]].tolist() for k, w in enumerate(want)}
        if (b // bs) % 50 == 0:
            log(stage="scores", done=b + len(batch), of=len(ids), minutes=round((time.time() - t0) / 60, 1))
    torch.save(scores, f)
    del m
    torch.cuda.empty_cache() if dev == "cuda" else None
    return scores


# ---- B: variants -------------------------------------------------------------------------------------------------

def rewrite(doc, sc, lex, subs, thr, ln, lv, tally):
    out = list(doc["tokens"])
    for i, (p, l) in enumerate(zip(doc["pos"], doc["lemmas"])):
        if p == "o":
            continue
        lvl = ln if p == "n" else lv
        if lvl == 0:
            continue
        e = lex.get(f"{l}|{p}")
        t = thr[p][lvl - 1]
        g = e["generality"] if e and e["generality"] is not None else float("-inf")
        if g >= t:
            continue
        ok = [(sc[i][k], c, depth) for k, (c, depth, gc) in enumerate(e["cands"]) if gc >= t] if e and i in sc else []
        if not ok:
            tally[p]["kept_no_candidate"] += 1
            continue
        _, c, depth = max(ok)
        out[i] = c
        tt = tally[p]
        tt["replaced"] += 1
        tt["depth"] += depth
        if e["zipf"] is not None:
            z, s = subs[f"{c}|{p}"]
            tt["d_zipf"] += z - e["zipf"]
            tt["d_spread"] += s - e["spread"]
            tt["n_delta"] += 1
    for p in ("n", "v"):
        tally[p]["words"] += sum(x == p for x in doc["pos"])
    return " ".join(out)


def variants(docs, scores, header, cfg, smoke):
    lex, subs, thr = header["lexicon"], header["substitutes"], header["thresholds"]
    nl, vl = cfg["dials"]["noun_levels"], cfg["dials"]["verb_levels"]
    if smoke:
        nl, vl = [0, nl[-1]], [0, vl[-1]]
    out, stats = {}, {}
    for ln in nl:
        for lv in vl:
            name = f"n{ln}v{lv}"
            tally = {p: dict(words=0, replaced=0, kept_no_candidate=0, depth=0, d_zipf=0.0, d_spread=0.0, n_delta=0) for p in ("n", "v")}
            out[name] = {d: rewrite(docs[d], scores[d], lex, subs, thr, ln, lv, tally) for d in docs}
            stats[name] = {p: {"words": t["words"], "replaced_share": round(t["replaced"] / max(t["words"], 1), 4),
                               "mean_depth": round(t["depth"] / max(t["replaced"], 1), 3),
                               "mean_d_zipf": round(t["d_zipf"] / max(t["n_delta"], 1), 3),
                               "mean_d_spread": round(t["d_spread"] / max(t["n_delta"], 1), 3),
                               "kept_no_candidate": t["kept_no_candidate"]} for p, t in tally.items()}
    for name, keep in (("nouns_only", "n"), ("verbs_only", "v")):
        out[name] = {d: " ".join(l for p, l in zip(docs[d]["pos"], docs[d]["lemmas"]) if p == keep) or "none" for d in docs}
    return out, stats


# ---- C: encoding -------------------------------------------------------------------------------------------------

class Embedder:
    def __init__(self, name, dev, bs):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.m = AutoModel.from_pretrained(name, torch_dtype=DTYPE[dev]).to(dev).eval()
        self.dev, self.bs, self.cache = dev, bs, {}

    def __call__(self, texts):
        key = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        todo = sorted({k: t for k, t in zip(key, texts) if k not in self.cache}.items(), key=lambda kv: len(kv[1]))
        for b in range(0, len(todo), self.bs):
            chunk = todo[b:b + self.bs]
            enc = self.tok([t for _, t in chunk], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
            with torch.no_grad():
                h = self.m(**enc).last_hidden_state[:, -1]
            for (k, _), v in zip(chunk, F.normalize(h.float(), dim=-1).half().cpu()):
                self.cache[k] = v
        return torch.stack([self.cache[k] for k in key])


# ---- D: recovery and leakage -------------------------------------------------------------------------------------

def recovery(queries, qv, dv, k):
    hits, rr = [], []
    for r in queries:
        q = qv[r["query"]].float()
        true = float(q @ dv[r["cited"]].float())
        c = torch.stack([dv[x] for x in r["candidates"]]).float() @ q
        rank = 1 + int((c >= true).sum())
        hits.append(float(rank <= k))
        rr.append(1.0 / rank)
    return hits, rr


def leakage(X, y, seed=0):
    """5-fold accuracy of a multinomial logistic classifier (section from vector)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    X, y = X[perm].float(), y[perm]
    acc = []
    for f in range(5):
        te = torch.arange(len(y)) % 5 == f
        W = torch.zeros(X.shape[1], 8, requires_grad=True)
        b = torch.zeros(8, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=100)
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(X[~te] @ W + b, y[~te]) + 1e-3 * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        acc.append(float(((X[te] @ W + b).argmax(1) == y[te]).float().mean()))
    return round(sum(acc) / 5, 4)


# ---- E: H3 -------------------------------------------------------------------------------------------------------

def stage_h3(work, cfg, dev, out, smoke):
    f = out / "h3.json"
    if f.exists():
        return json.loads(f.read_text())
    lines = [json.loads(l) for l in open(work / "h3_words.jsonl", encoding="utf-8")]
    held, pairs = lines[0]["held_out_sections"], lines[1:]
    if smoke:
        pairs = pairs[:200]
    tok, m = load_lm(cfg["models"]["substitute_lm"], dev)
    layer = min(cfg["models"]["h3_layer"], m.config.num_hidden_layers)
    words = sorted({p["word"] for p in pairs} | {p["hypernym"] for p in pairs})
    vec = {}
    for b in range(0, len(words), 64):
        ws = words[b:b + 64]
        pre = "The word: "
        enc = tok([pre + w for w in ws], padding=True, return_tensors="pt").to(dev)
        n_pre = len(tok(pre)["input_ids"])
        with torch.no_grad():
            hs = m(**enc, output_hidden_states=True).hidden_states[layer].float()
        for j, w in enumerate(ws):
            mask = enc["attention_mask"][j].bool()
            v = hs[j][mask][n_pre:]  # right padding: the word's tokens follow the prefix
            vec[w] = v.mean(0).cpu()
    del m
    tr = [p for p in pairs if p["split"] == "train"]
    te = [p for p in pairs if p["split"] == "test"]
    res = {"held_out_sections": held, "pairs": {"train": len(tr), "test": len(te)}}
    if len(tr) < 10 or len(te) < 10:
        res["skipped"] = "too few pairs"
        f.write_text(json.dumps(res))
        return res
    X = torch.stack([vec[p["word"]] for p in tr])
    Y = torch.stack([vec[p["hypernym"]] for p in tr])
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    lam = 1.0 * Xc.pow(2).sum() / X.shape[1]
    W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(X.shape[1]), Xc.T @ Yc)
    zipf = {p["hypernym"]: p["zipf_hypernym"] for p in pairs}
    pool = sorted({p["hypernym"] for p in te})
    rng = random.Random(cfg["seed"])
    hit_map, hit_id, hit_freq = [], [], []
    for p in te:
        others = [h for h in pool if h != p["hypernym"]]
        cands = [p["hypernym"]] + rng.sample(others, min(99, len(others)))
        C = F.normalize(torch.stack([vec[c] for c in cands]), dim=-1)
        pred = F.normalize((vec[p["word"]] - mx) @ W + my, dim=-1)
        ident = F.normalize(vec[p["word"]], dim=-1)
        for arr, s in ((hit_map, C @ pred), (hit_id, C @ ident)):
            arr.append(float(int((s >= s[0]).sum()) <= 10))
        by_f = sorted(cands, key=lambda c: -zipf[c])
        hit_freq.append(float(p["hypernym"] in by_f[:10]))
    res.update({"candidates": min(100, len(pool)), "map_hits": hit_map, "identity_hits": hit_id, "frequency_hits": hit_freq})
    f.write_text(json.dumps(res))
    return res


def trim_cands(src, need, dst):
    """cands.jsonl cut to the documents in <need>, the lexicon entries they use and the substitutes those name."""
    need = set(need)
    with open(src, encoding="utf-8") as fh, open(dst, "w", encoding="utf-8", newline="\n") as out:
        header = json.loads(fh.readline())
        lines = [line for line in fh if json.loads(line)["id"] in need]
        used = set()
        for line in lines:
            d = json.loads(line)
            used |= {f"{l}|{p}" for p, l in zip(d["pos"], d["lemmas"])}
        lex = {k: v for k, v in header["lexicon"].items() if k in used}
        named = {f"{c}|{k.rsplit('|', 1)[1]}" for k, e in lex.items() if e for c, _, _ in e.get("cands") or []}
        subs = {k: v for k, v in header["substitutes"].items() if k in named}
        out.write(json.dumps({**header, "lexicon": lex, "substitutes": subs}) + "\n")
        out.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    work, out = Path(a.work), Path(a.out)
    (out / "emb").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke and dev == "cpu":  # the laptop dry run: smallest base model, same code
        cfg["models"]["substitute_lm"] = "Qwen/Qwen3-0.6B-Base"
    queries = [json.loads(l) for l in open(work / "queries.jsonl", encoding="utf-8")]
    if a.smoke:
        queries = [q for s in ("val", "test") for q in [x for x in queries if x["split"] == s][:a.smoke]]
        for q in queries:
            q["candidates"] = q["candidates"][:15]  # code paths only; 30 took an hour on the laptop CPU (10/05/2026)
    need = {q["query"] for q in queries} | {q["cited"] for q in queries} | {c for q in queries for c in q["candidates"]}
    # The full lexicon is loaded and trimmed in a child process, which exits and returns its memory: held in this
    # process, it pushed the laptop dry run past its 3.6 GB cap four times, and freeing it later did not lower the
    # process's committed memory on Windows (10/07/2026). Lookups use only these keys, so results are unchanged.
    import multiprocessing as mp
    trimmed = out / "cands-trimmed.jsonl"
    p = mp.get_context("spawn").Process(target=trim_cands, args=(str(work / "cands.jsonl"), sorted(need), str(trimmed)))
    p.start(); p.join()
    assert p.exitcode == 0, f"trim_cands failed ({p.exitcode})"
    with open(trimmed, encoding="utf-8") as fh:
        header = json.loads(fh.readline())
        docs = {}
        for line in fh:
            d = json.loads(line)
            docs[d["id"]] = d
    missing = need - set(docs)
    assert not missing, f"{len(missing)} abstracts missing from cands.jsonl"
    log(stage="start", device=dev, docs=len(docs), queries=len(queries), smoke=a.smoke)

    scores = stage_scores(docs, header["lexicon"], cfg, dev, out, a.batch)
    texts, rstats = variants(docs, scores, header, cfg, a.smoke)
    log(stage="variants", n=len(texts))
    # Free the lexicon, scores and token lists before the embedder loads: on real inputs the laptop dry run passed
    # its 3.6 GB cap here, holding them all (10/07/2026). Only the ids and thresholds are used from here on.
    header = {"thresholds": header["thresholds"]}
    docs = dict.fromkeys(docs)
    del scores
    import gc
    gc.collect()
    k = cfg["recovery"]["top_k"]
    sec = {c: SECTIONS.index(q["cited_section"]) for q in queries for c in q["candidates"]}
    pool_ids = sorted(sec)
    emb = Embedder(cfg["models"]["embedder"], dev, a.batch * 4)
    qids = sorted({q["query"] for q in queries})
    metrics = {"variants": {}, "rewrite_stats": rstats, "thresholds": header["thresholds"],
               "queries": {s: [q["query"] for q in queries if q["split"] == s] for s in ("val", "test")}}
    t0 = time.time()
    for name, tx in texts.items():
        f = out / "emb" / f"{name}.pt"
        if f.exists():
            qv, dv = torch.load(f)
        else:
            dids = sorted(docs)
            dv = dict(zip(dids, emb([tx[d] for d in dids])))
            qv = dict(zip(qids, emb([INSTRUCT + tx[q] for q in qids])))
            torch.save((qv, dv), f)
        m = {}
        for s in ("val", "test"):
            h, rr = recovery([q for q in queries if q["split"] == s], qv, dv, k)
            m[s] = {"hits": h, "mrr": round(sum(rr) / max(len(rr), 1), 4), "top10": round(sum(h) / max(len(h), 1), 4)}
        X = torch.stack([dv[p] for p in pool_ids])
        m["leakage"] = leakage(X, torch.tensor([sec[p] for p in pool_ids]))
        metrics["variants"][name] = m
        log(stage="variant", name=name, val=m["val"]["top10"], test=m["test"]["top10"], leakage=m["leakage"],
            minutes=round((time.time() - t0) / 60, 1))
    del emb
    metrics["h3"] = stage_h3(work, cfg, dev, out, a.smoke)
    metrics["models"] = cfg["models"]
    (out / "gpu.json").write_text(json.dumps(metrics))
    log(stage="done", variants=len(metrics["variants"]))


if __name__ == "__main__":
    main()
