"""GPU worker: is stage 0 failing to fit, or failing to generalize?

Loads a ``remote-aex.py`` checkpoint and measures the same model on three sets of 200 sentences:
Enron sentences it trained on, Wikipedia sentences it trained on, and held-out test-mailbox
sentences (the same 200 the checkpoints are graded on). For each set:

- the teacher-forced loss at 1024 and 256 dims (directly comparable across sets);
- greedy decodes at 1024 and 256 dims, for the Sonnet intent check on the laptop.

If training sentences rebuild well and held-out ones do not, the model memorizes; if neither does,
training has not fit yet.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from importlib import import_module
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
aex = import_module("remote-aex")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/ae0x/checkpoint.pt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="out/ae0x/diag")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cuda")
    a = ck["args"]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = aex.IdeaAE(a["enc"], a["dec"], a["dim"], a["k"], dtype).cuda()
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    rng = random.Random(99)
    enron = [r["text"] for r in aex.read_jsonl(a["enron"])]
    wiki = [r["text"] for r in aex.read_jsonl(a["general"])]
    held = [r for r in aex.read_jsonl(a["heldout"]) if r["role"] == "test"]
    random.Random(1).shuffle(held)
    sets = {"train_enron": rng.sample(enron, args.n), "train_wiki": rng.sample(wiki, args.n),
            "heldout": [r["text"] for r in held[: args.n]]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"step": ck["step"]}
    for name, texts in sets.items():
        losses = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            for d in (model.dim, 256):
                vals = []
                for s in range(0, len(texts), 32):
                    b = texts[s : s + 32]
                    vals.append(float(model.loss(model.cut(model.encode(b), d), b)) * len(b))
                losses[f"loss_{d}"] = round(sum(vals) / len(texts), 4)
        rows = aex.evaluate(model, [{"text": t} for t in texts], sorted({256, model.dim}), dtype, noise=False)
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[name] = losses
        print(json.dumps({name: losses}), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("diag done", flush=True)


if __name__ == "__main__":
    main()
