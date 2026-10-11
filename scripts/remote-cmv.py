"""C1: does idea space predict which argument changes a mind, and does surprise pay? (10/05/2026)

The first outcome test (phase 9). Data from cmv-build.py: 4,254 ChangeMyView pairs of first replies to one original
post, one of which won the poster's delta (3,450 train, 804 held out, the authors' split). Every model answers the
same paired question: which of the two replies won? Chance is 0.5; reply length alone gets 0.603 and a bag of words
0.64 (measured locally); with 804 held-out pairs a 95% interval is about +-0.034, so only large gaps are resolvable.

Models (all trained only on the train pairs; 10% of training posts held out to pick the epoch):
- ``idea``: the value model in idea space. The post's and the reply's ideas (splitter q3, encoder S1, frozen vectors)
  go through a small transformer (2 layers, 256 wide) that scores the reply. Five seeds, scores averaged.
- ``text``: Qwen3-0.6B-Base fine-tuned with a score head on the post and reply as written.
- ``text_ideas``: the same text model on the splitter's ideas written out one per line. Against ``text`` it shows what
  the rewrite into ideas loses or gains; against ``idea`` what the vectors lose or gain over the same ideas in words.

Surprise (Shaun's hypothesis: routine moments reward the expected move, stuck or open ones reward a surprising one):
J10, the next-idea predictor, reads the post's last ideas and then the reply's. A reply idea's surprise is 1 - cosine
between J10's prediction and the idea; a reply's surprise is the mean over its ideas. A post's openness is the mean
surprise of its own idea-to-idea steps (how unpredictable the discussion already is). Relevance (cosine of the
reply's mean idea to the post's) separates a surprising move from an off-topic one. Reported: how often the more
surprising reply wins, overall and by openness tertile, and a logistic model on the pair differences with the
interaction surprise x openness, with bootstrap intervals. Observational: a surprise that comes with winning is not
shown to cause it.

    python remote-cmv.py --pred cmv/pred-cmv.jsonl --pairs cmv/pairs.jsonl --texts cmv/texts.jsonl \
        --adapter ablation/S1/lora.pt --predictor jepa/J10/predictor.pt --out cmv/C1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

TEXT_BASE = "Qwen/Qwen3-0.6B-Base"
OP_IDEAS, REPLY_IDEAS = 24, 40


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def boot(xs, reps=2000, seed=0):
    """Mean and 95% bootstrap interval."""
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor(xs, dtype=torch.float64)
    m = x[torch.randint(len(x), (reps, len(x)), generator=g)].mean(1)
    return [round(float(x.mean()), 4), round(float(m.quantile(0.025)), 4), round(float(m.quantile(0.975)), 4)]


class ValueNet(torch.nn.Module):
    def __init__(self, d_in=1024, d=256, layers=2, dropout=0.2):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, d)
        self.seg = torch.nn.Embedding(2, d)
        self.pos = torch.nn.Embedding(OP_IDEAS + REPLY_IDEAS, d)
        layer = torch.nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.tf = torch.nn.TransformerEncoder(layer, layers)
        self.head = torch.nn.Linear(d, 1)

    def forward(self, x, seg, pad):
        n = x.shape[1]
        h = self.tf(self.proj(x) + self.seg(seg) + self.pos(torch.arange(n, device=x.device)), src_key_padding_mask=pad)
        keep = ((seg == 1) & ~pad).unsqueeze(-1).float()
        return self.head((h * keep).sum(1) / keep.sum(1).clamp(min=1)).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--reply-tokens", type=int, default=512, help="tokens of each reply the text models read")
    # A dry run on a few pairs (aws.sh runs a job's "# smoke:" lines first): G1 crashed twice on paths no small run had
    # exercised (10/05/2026).
    ap.add_argument("--smoke", type=int, default=0, help="this many pairs per split, 1 seed, 2 epochs, 1 text epoch")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    # Ideas of each post and reply: its windows' ideas in order, repeats dropped.
    win = {}
    for r in map(json.loads, open(args.pred, encoding="utf-8")):
        t, k = r["uid"].rsplit("#", 1)
        win.setdefault(t, []).append((int(k), r["ideas"]))
    ideas_of = {t: list(dict.fromkeys(i for _, ii in sorted(w) for i in ii)) for t, w in win.items()}
    pairs = [json.loads(l) for l in open(args.pairs)]
    pairs = [p for p in pairs if all(ideas_of.get(p[s]) for s in ("op", "win", "lose"))]
    if args.smoke:
        pairs = [p for sp in ("train", "test") for p in [q for q in pairs if q["split"] == sp][: args.smoke * (10 if sp == "train" else 1)]]
    texts = {r["id"]: r["text"] for r in map(json.loads, open(args.texts, encoding="utf-8"))}
    train = [p for p in pairs if p["split"] == "train"]
    test = [p for p in pairs if p["split"] == "test"]
    ops = sorted({p["op"] for p in train})
    val_ops = set(random.Random(0).sample(ops, len(ops) // 10))
    fit = [p for p in train if p["op"] not in val_ops]
    val = [p for p in train if p["op"] in val_ops]
    res["pairs"] = {"train": len(fit), "val": len(val), "test": len(test)}
    print(json.dumps(res), flush=True)

    # Encode every idea once (S1), unit length as the predictor reads them.
    enc = load("remote_encoder", "remote-encoder.py")
    model = enc.SonarAE(16).cuda()
    model.load_state_dict(torch.load(args.adapter, map_location="cuda"), strict=False)
    model.eval()
    all_ideas = sorted({i for p in pairs for s in ("op", "win", "lose") for i in ideas_of[p[s]]}, key=len)
    t0 = time.time()
    V = torch.empty(len(all_ideas), 1024, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(all_ideas), 128):
            V[s : s + 128] = F.normalize(model.encode(all_ideas[s : s + 128]).float(), dim=-1)
    ix = {t: k for k, t in enumerate(all_ideas)}
    vec = lambda t: V[torch.tensor([ix[i] for i in t], device="cuda")]
    res["ideas"] = len(all_ideas)
    res["encode_min"] = round((time.time() - t0) / 60, 1)
    del model
    torch.cuda.empty_cache()

    # Surprise under J10.
    jepa = load("remote_jepa", "remote-jepa.py")
    pred = jepa.Predictor(layers=2, max_len=32, dropout=0.3).cuda()
    pred.load_state_dict(torch.load(args.predictor, map_location="cuda"))
    pred.eval()

    @torch.no_grad()
    def step_cos(seq):
        x = vec(seq).unsqueeze(0)
        p = pred(x, torch.zeros(1, x.shape[1], dtype=torch.bool, device="cuda"))[0, :-1]
        return (p * x[0, 1:]).sum(-1), (x[0, :-1] * x[0, 1:]).sum(-1)

    feats = {}
    for t in sorted({p["op"] for p in pairs}):
        oi = ideas_of[t][:32]
        feats[t] = {"openness": float(1 - step_cos(oi)[0].mean()) if len(oi) > 1 else float("nan")}
    for p in pairs:
        ctx = ideas_of[p["op"]][-16:]
        for s in ("win", "lose"):
            ri = ideas_of[p[s]][: 32 - len(ctx)]
            c, last = step_cos(ctx + ri)
            c, last = c[len(ctx) - 1 :], last[len(ctx) - 1 :]
            rel = float(F.normalize(vec(ideas_of[p[s]]).mean(0), dim=0) @ F.normalize(vec(ideas_of[p["op"]]).mean(0), dim=0))
            feats[p[s]] = {"surprise": float(1 - c.mean()), "surprise_last": float(1 - last.mean()), "relevance": rel,
                           "n_ideas": len(ideas_of[p[s]]), "words": len(texts[p[s]].split())}

    def wins(ps, key):
        return [float(feats[p["win"]][key] > feats[p["lose"]][key]) + 0.5 * (feats[p["win"]][key] == feats[p["lose"]][key]) for p in ps]

    sur = {"all_pairs": len(pairs)}
    for key in ("surprise", "surprise_last", "relevance", "n_ideas", "words"):
        sur[f"higher_{key}_wins"] = boot(wins(pairs, key))
    have = sorted((p for p in pairs if feats[p["op"]]["openness"] == feats[p["op"]]["openness"]), key=lambda p: feats[p["op"]]["openness"])
    # GitHub (G1) has no shared post (a one-idea context), so no openness: the tertiles and the interaction are skipped.
    has_open = len(have) >= 30
    third = len(have) // 3
    for name, ps in (("low", have[:third]), ("mid", have[third : 2 * third]), ("high", have[2 * third :])) if has_open else ():
        sur[f"openness_{name}"] = {"n": len(ps), "higher_surprise_wins": boot(wins(ps, "surprise")),
                                   "higher_relevance_wins": boot(wins(ps, "relevance"))}

    # Logistic model on pair differences, fit on train, scored on test; coefficients with bootstrap intervals.
    op_mu = sum(feats[p["op"]]["openness"] for p in have) / len(have) if has_open else 0.0
    op_sd = (sum((feats[p["op"]]["openness"] - op_mu) ** 2 for p in have) / len(have)) ** 0.5 if has_open else 1.0
    names = ["log_words", "surprise", "relevance", "surprise_x_openness"][: 4 if has_open else 3]

    def row(p):
        z = feats[p["op"]]["openness"]
        z = 0.0 if z != z else (z - op_mu) / op_sd
        d = lambda k: feats[p["win"]][k] - feats[p["lose"]][k]
        lw = torch.log(torch.tensor(float(feats[p["win"]]["words"]) + 1)) - torch.log(torch.tensor(float(feats[p["lose"]]["words"]) + 1))
        return [float(lw), d("surprise"), d("relevance"), d("surprise") * z][: len(names)]

    Xtr = torch.tensor([row(p) for p in train], dtype=torch.float64)
    Xte = torch.tensor([row(p) for p in test], dtype=torch.float64)
    sd = Xtr.std(0)
    Xtr, Xte = Xtr / sd, Xte / sd

    def logit_fit(X):
        # Pairs are symmetric: each difference is shown both ways (label 1 for win - lose, 0 for its negation).
        Xb, y = torch.cat([X, -X]), torch.cat([torch.ones(len(X)), torch.zeros(len(X))]).double()
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(Xb @ w, y) + 1e-3 * (w**2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return w.detach()

    w = logit_fit(Xtr)
    g = torch.Generator().manual_seed(0)
    bs = torch.stack([logit_fit(Xtr[torch.randint(len(Xtr), (len(Xtr),), generator=g)]) for _ in range(200)])
    sur["logistic"] = {"test_acc": boot([float(v > 0) for v in (Xte @ w).tolist()]),
                       "coef": {n: [round(float(w[k]), 3), round(float(bs[:, k].quantile(0.025)), 3), round(float(bs[:, k].quantile(0.975)), 3)]
                                for k, n in enumerate(names)}}
    res["surprise"] = sur
    print(json.dumps(sur), flush=True)

    # Idea value model.
    def batch(ps, side):
        rows = [(vec(ideas_of[p["op"]][:OP_IDEAS]), vec(ideas_of[p[side]][:REPLY_IDEAS])) for p in ps]
        n = max(len(a) + len(b) for a, b in rows)
        x = torch.zeros(len(rows), n, 1024, device="cuda")
        seg = torch.zeros(len(rows), n, dtype=torch.long, device="cuda")
        pad = torch.ones(len(rows), n, dtype=torch.bool, device="cuda")
        for k, (a, b) in enumerate(rows):
            x[k, : len(a)], x[k, len(a) : len(a) + len(b)] = a, b
            seg[k, len(a) : len(a) + len(b)] = 1
            pad[k, : len(a) + len(b)] = False
        return x, seg, pad

    def acc_of(scores, ps):
        return [float(scores[p["win"]] > scores[p["lose"]]) for p in ps]

    def idea_scores(net, ps):
        net.eval()
        sc = {}
        with torch.no_grad():
            for s in range(0, len(ps), 256):
                b = ps[s : s + 256]
                for side in ("win", "lose"):
                    for p, v in zip(b, net(*batch(b, side)).tolist()):
                        sc[p[side]] = v
        return sc

    t0 = time.time()
    test_sum, val_sum = {}, {}
    curves = []
    for seed in range(1 if args.smoke else 5):
        torch.manual_seed(seed)
        rng = random.Random(seed)
        net = ValueNet().cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
        best, best_test = -1.0, None
        curve = []
        for ep in range(2 if args.smoke else 30):
            net.train()
            order = fit[:]
            rng.shuffle(order)
            for s in range(0, len(order), 64):
                b = order[s : s + 64]
                loss = F.softplus(-(net(*batch(b, "win")) - net(*batch(b, "lose")))).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            va = sum(acc_of(idea_scores(net, val), val)) / len(val)
            curve.append(round(va, 4))
            if va > best:
                best, best_test, best_val = va, idea_scores(net, test), idea_scores(net, val)
        curves.append({"seed": seed, "best_val": round(best, 4), "val_curve": curve})
        mu = sum(best_test.values()) / len(best_test)
        sdv = (sum((v - mu) ** 2 for v in best_test.values()) / len(best_test)) ** 0.5
        for k, v in best_test.items():
            test_sum[k] = test_sum.get(k, 0.0) + (v - mu) / sdv
        for k, v in best_val.items():
            val_sum[k] = val_sum.get(k, 0.0) + (v - mu) / sdv
    idea_acc = acc_of(test_sum, test)
    res["idea"] = {"test_acc": boot(idea_acc), "seeds": curves, "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps({"idea": res["idea"]["test_acc"]}), flush=True)

    # Text models.
    def run_text(render, label):
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TEXT_BASE)
        lm = AutoModel.from_pretrained(TEXT_BASE, torch_dtype=torch.float32).cuda()
        lm.gradient_checkpointing_enable()
        head = torch.nn.Linear(lm.config.hidden_size, 1).cuda()
        params = list(lm.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.01)

        def enc_one(p, side):
            op, rp = render(p["op"]), render(p[side])
            a = tok("Original post: " + op, add_special_tokens=False).input_ids[:256]
            b = tok("\n\nReply: " + rp, add_special_tokens=False).input_ids[: args.reply_tokens]
            return a + b

        def score(ps, side):
            ids = [enc_one(p, side) for p in ps]
            n = max(map(len, ids))
            x = torch.tensor([i + [tok.pad_token_id] * (n - len(i)) for i in ids], device="cuda")
            m = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i in ids], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = lm(input_ids=x, attention_mask=m).last_hidden_state
            last = h[torch.arange(len(ids), device="cuda"), m.sum(1) - 1]
            return head(last.float()).squeeze(-1)

        def all_scores(ps):
            lm.eval()
            sc = {}
            with torch.no_grad():
                for s in range(0, len(ps), 8):
                    b = ps[s : s + 8]
                    for side in ("win", "lose"):
                        for p, v in zip(b, score(b, side).tolist()):
                            sc[p[side]] = v
            return sc

        rng = random.Random(0)
        torch.manual_seed(0)
        best, best_test, curve = -1.0, None, []
        t1 = time.time()
        for ep in range(1 if args.smoke else 3):
            lm.train()
            order = fit[:]
            rng.shuffle(order)
            for s in range(0, len(order), 16):
                b = order[s : s + 16]
                opt.zero_grad()
                for m0 in range(0, len(b), 4):
                    mb = b[m0 : m0 + 4]
                    loss = F.softplus(-(score(mb, "win") - score(mb, "lose"))).mean() * len(mb) / len(b)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            va = sum(acc_of(all_scores(val), val)) / len(val)
            curve.append(round(va, 4))
            print(json.dumps({label: {"epoch": ep, "val": round(va, 4), "minutes": round((time.time() - t1) / 60, 1)}}), flush=True)
            if va > best:
                best, best_test, best_val = va, all_scores(test), all_scores(val)
        del lm, head, opt
        torch.cuda.empty_cache()
        a = acc_of(best_test, test)
        text_val[label] = best_val
        text_test[label] = best_test
        return a, {"test_acc": boot(a), "val_curve": curve, "minutes": round((time.time() - t1) / 60, 1)}

    text_val, text_test = {}, {}
    if not args.skip_text:
        text_acc, res["text"] = run_text(lambda t: texts[t], "text")
        ti_acc, res["text_ideas"] = run_text(lambda t: "\n".join(ideas_of[t]), "text_ideas")
        diff = lambda a, b: boot([x - y for x, y in zip(a, b)])
        res["idea_minus_text"] = diff(idea_acc, text_acc)
        res["idea_minus_text_ideas"] = diff(idea_acc, ti_acc)
        res["text_ideas_minus_text"] = diff(ti_acc, text_acc)
    res["length_test_acc"] = boot(wins(test, "words"))
    if "size" in pairs[0]:
        # GitHub (G1): the smaller change is the clean one ("win" is the PR that was not reverted).
        res["smaller_test_acc"] = boot([float(p["size"][0] < p["size"][1]) + 0.5 * (p["size"][0] == p["size"][1]) for p in test])
    if not args.skip_text:
        # Text plus idea scores: a logistic model on the pair differences of the text model's score, the idea model's
        # score, surprise, relevance and length, fit on the validation pairs (the epochs were chosen on them too, a
        # mild optimism shared by every row), scored on the held-out pairs.
        def stack_rows(ps, ts, isc):
            d = lambda f, k: f(p["win"], k) - f(p["lose"], k)
            return torch.tensor([[ts[p["win"]] - ts[p["lose"]], isc[p["win"]] - isc[p["lose"]],
                                  feats[p["win"]]["surprise"] - feats[p["lose"]]["surprise"],
                                  feats[p["win"]]["relevance"] - feats[p["lose"]]["relevance"],
                                  float(torch.log(torch.tensor(feats[p["win"]]["words"] + 1.0)) - torch.log(torch.tensor(feats[p["lose"]]["words"] + 1.0)))]
                                 for p in ps], dtype=torch.float64)

        Sv, St = stack_rows(val, text_val["text"], val_sum), stack_rows(test, text_test["text"], test_sum)
        sd2 = Sv.std(0).clamp(min=1e-9)
        w2 = logit_fit(Sv / sd2)
        comb = [float(v > 0) for v in ((St / sd2) @ w2).tolist()]
        res["text_plus_idea"] = {"test_acc": boot(comb), "weights": dict(zip(["text", "idea", "surprise", "relevance", "log_words"],
                                                                             [round(float(x), 3) for x in w2]))}
        res["text_plus_idea_minus_text"] = boot([x - y for x, y in zip(comb, text_acc)])
    (out / "cmv.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(out / "features.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({**p, "op_f": feats[p["op"]], "win_f": feats[p["win"]], "lose_f": feats[p["lose"]],
                                "idea_score": [test_sum.get(p["win"]), test_sum.get(p["lose"])]}) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("idea", "surprise")}), flush=True)


if __name__ == "__main__":
    main()
