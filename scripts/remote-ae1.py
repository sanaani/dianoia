"""GPU worker, stage 1: the bottleneck. Starts from the stage-0 checkpoint.

Training pairs (input sentence -> target):
- every canonicalized pool sentence (fit mailboxes) -> its canonical sentence;
- every phase-1 fit item's source and keep variants (paraphrase, rename, register, dropped modifiers)
  -> that item's canonical sentence. Rewording therefore cannot change the decode.
Plus, on phase-1 fit items, the phase-1 contrastive loss (``head.loss``, weight ``--lam``) so move
variants (negation, modality, act, roles, verb/object, type, quantity) stay apart.

Evaluation on phase-1 test items: decode z(source) at 64 / 128 / 256 dims and from a random vector;
restore names from the source's bindings; re-encode each decode with the same encoder; record its
cosine to z(source) and the paraphrase tolerance tau = cos(z(source), z(paraphrase)).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from importlib import import_module
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
ae = import_module("remote-ae")
from head import loss as contrastive_loss  # noqa: E402

KEEP = ["paraphrase", "rename", "register", "drop_modifiers"]
MOVE = ["act_change", "role_swap", "negation", "modality", "predicate_change", "type_change", "quantity"]
PLACEHOLDER = re.compile(r"\[[A-Z]+_\d+\]")


def restore(text: str, bindings: dict) -> tuple[str, int]:
    """Put names back; count placeholders the decode emitted that the source never bound."""
    missing = 0

    def sub(m):
        nonlocal missing
        if m.group(0) in bindings:
            return bindings[m.group(0)]
        missing += 1
        return m.group(0)

    return PLACEHOLDER.sub(sub, text), missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="out/ae0/model.pt")
    ap.add_argument("--canonical", default="ae/canonical.jsonl")
    ap.add_argument("--items", default="ae/items.jsonl")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--out", default="out/ae1")
    args = ap.parse_args()
    random.seed(20261001)
    torch.manual_seed(20261001)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    canon = [json.loads(line) for line in open(args.canonical, encoding="utf-8")]
    items = [json.loads(line) for line in open(args.items, encoding="utf-8")]
    fit_items = [r for r in items if r["role"] == "fit" and r.get("canonical")]
    test_items = [r for r in items if r["role"] == "test" and r.get("canonical")]
    pairs = [(r["text"], r["canonical"]) for r in canon]
    for r in fit_items:
        pairs.append((r["text"], r["canonical"]))
        pairs += [(r[k], r["canonical"]) for k in KEEP if r.get(k)]
    print(json.dumps({"pairs": len(pairs), "fit_items": len(fit_items), "test_items": len(test_items)}), flush=True)

    model = ae.IdeaAE(dtype).cuda()
    state = torch.load(args.init, map_location="cuda")
    model.load_state_dict(state, strict=False)
    params = [
        {"params": model.enc.parameters(), "lr": 1e-5},
        {"params": list(model.to_z.parameters()) + list(model.to_prefix.parameters()), "lr": 2e-4},
        {"params": [p for p in model.dec.parameters() if p.requires_grad], "lr": 1e-4},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    rng = random.Random(11)
    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        model.train()
        b = rng.sample(pairs, args.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            z = model.encode([x for x, _ in b])
            d = 256 if rng.random() < 0.5 else rng.choice(ae.PREFIXES)
            l_dec = model.loss(model.cut(z, d), [y for _, y in b])
            ib = rng.sample(fit_items, 16)
            fields = ["text", *KEEP, *MOVE]
            texts, where = [], {}
            for f in fields:
                for i, r in enumerate(ib):
                    if r.get(f):
                        where[(f, i)] = len(texts)
                        texts.append(r[f])
            zz = model.encode(texts).float()
            get = {f: torch.stack([zz[where[(f, i)]] if (f, i) in where else torch.zeros(256, device="cuda") for i in range(len(ib))]) for f in fields}
            msk = {f: torch.tensor([(f, i) in where for i in range(len(ib))], device="cuda") for f in fields}
            l_con = contrastive_loss(get["text"], {k: get[k] for k in KEEP}, {k: msk[k] for k in KEEP},
                                     torch.stack([get[m] for m in MOVE], 1), torch.stack([msk[m] for m in MOVE], 1))
            l = l_dec + args.lam * l_con
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            log.append({"step": step, "dec_loss": round(float(l_dec), 4), "con_loss": round(float(l_con), 4), "minutes": round((time.time() - t0) / 60, 1)})
            print(json.dumps(log[-1]), flush=True)
    torch.save({k: v for k, v in model.state_dict().items() if "lora" in k or not k.startswith("dec.")}, out / "model.pt")

    model.eval()
    results = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        for s in range(0, len(test_items), 32):
            b = test_items[s : s + 32]
            z = model.encode([r["text"] for r in b]).float()
            para = [r.get("paraphrase") or r["text"] for r in b]
            zp = model.encode(para).float()
            arms = {f"ae{d}": model.generate(model.cut(z, d)) for d in ae.PREFIXES}
            arms["noise"] = model.generate(model.cut(torch.randn_like(z), 256))
            for k, r in enumerate(b):
                bind = json.loads(r.get("bindings") or "{}")
                row = {"source": r["text"], "canonical": r["canonical"], "bindings": bind,
                       "tau": float(F.cosine_similarity(z[k : k + 1], zp[k : k + 1])) if r.get("paraphrase") else None}
                for arm, texts in arms.items():
                    restored, missing = restore(texts[k], bind)
                    row[arm] = restored
                    row[f"{arm}_raw"] = texts[k]
                    row[f"{arm}_unbound"] = missing
                results.append(row)
        for arm in [*[f"ae{d}" for d in ae.PREFIXES], "noise"]:
            for s in range(0, len(results), 32):
                chunk = results[s : s + 32]
                zs = model.encode([r["source"] for r in chunk]).float()
                zd = model.encode([r[arm] or "." for r in chunk]).float()
                for k, r in enumerate(chunk):
                    r[f"{arm}_cos"] = float(F.cosine_similarity(zs[k : k + 1], zd[k : k + 1]))
    with open(out / "decoded.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "train.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2))
    print("done", len(results), flush=True)


if __name__ == "__main__":
    main()
