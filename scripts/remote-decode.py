"""GPU worker: decode raw SONAR vectors to English, then re-encode what was decoded.

Input an ``.npy`` of (n, 1024) raw SONAR vectors. Output ``decoded.jsonl`` (one ``{"text": ...}``
per row) and ``redecoded-sonar.npy`` (the SONAR vectors of the decoded texts), rows in input order.
Decoder: the Hugging Face port ``raxtemur/SONAR_200_text_decoder`` at a pinned revision, beam 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

DEC = ("raxtemur/SONAR_200_text_decoder", "278f6ad02ad7f737dfe120d2ccad9a5e6dfad491")


@torch.inference_mode()
def decode(vecs: np.ndarray, batch: int = 64) -> list[str]:
    tok = NllbTokenizer.from_pretrained(DEC[0], revision=DEC[1])
    model = M2M100ForConditionalGeneration.from_pretrained(DEC[0], revision=DEC[1]).cuda().half().eval()
    bos = tok.convert_tokens_to_ids("eng_Latn")
    out: list[str] = []
    for s in range(0, len(vecs), batch):
        v = torch.from_numpy(vecs[s : s + batch].astype(np.float32)).cuda().half()
        g = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=v.unsqueeze(1)), forced_bos_token_id=bos, max_length=128, num_beams=5)
        out += tok.batch_decode(g, skip_special_tokens=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vectors")
    ap.add_argument("out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = decode(np.load(args.vectors))
    with open(out / "decoded.jsonl", "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    enc = import_module("remote-encode")
    np.save(out / "redecoded-sonar.npy", enc.sonar(texts))
    print(json.dumps({"n": len(texts)}), flush=True)


if __name__ == "__main__":
    main()
