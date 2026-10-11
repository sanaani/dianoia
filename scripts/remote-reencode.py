"""GPU worker: encode a JSON-lines file of texts with the fine-tuned intent encoders (256-d each).

Loads each ``out/ft/<name>/best.pt`` into ``remote-finetune.Encoder`` and writes ``<out>/<name>.npy``.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
ft = import_module("remote-finetune")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("texts")
    ap.add_argument("out")
    ap.add_argument("--models", nargs="*", default=["sonar", "bge_e16"])
    args = ap.parse_args()
    texts = [json.loads(line)["text"] for line in open(args.texts, encoding="utf-8")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        enc = ft.Encoder("sonar" if name.startswith("sonar") else "bge").cuda()
        enc.load_state_dict(torch.load(Path("out/ft") / name / "best.pt"))
        np.save(out / f"ft_{name.split('_')[0]}.npy", ft.encode_texts(enc, texts).astype(np.float32))
        del enc
        torch.cuda.empty_cache()
        print(json.dumps({"model": name, "n": len(texts)}), flush=True)


if __name__ == "__main__":
    main()
