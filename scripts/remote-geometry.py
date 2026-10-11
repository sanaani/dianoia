"""GPU worker: encode the geometry material with SONAR's text encoder (as released, no adapters).

Input ``geometry/ideas.json`` (from ``idea-geometry.py``); output ``geometry/vectors.npz`` with one 1024-d vector
per text and a parallel JSON index of (idea number, role of the text: idea / rewording / negation / bare).
"""

from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

ENC = ("cointegrated/SONAR_200_text_encoder", "29f2eac73636d1548c2cb227dbfb2772b31ebc4f")


def main() -> None:
    data = json.load(open("geometry/ideas.json", encoding="utf-8"))
    texts, index = [], []
    for n, o in enumerate(data["ideas"]):
        for kind, ts in (("idea", [o["idea"]]), ("rewording", o["rewordings"]), ("negation", [o["negation"]] if o["negation"] else []),
                         ("bare", [o["bare"]] if o["bare"] else [])):
            for t in ts:
                texts.append(t)
                index.append({"n": n, "kind": kind})
    for name, ladder in data.get("ladders", {}).items():
        for k, t in enumerate(ladder):
            texts.append(t)
            index.append({"n": -1, "kind": f"ladder-{name}-{k}"})
    tok = AutoTokenizer.from_pretrained(ENC[0], revision=ENC[1])
    tok.src_lang = "eng_Latn"
    enc = M2M100Encoder.from_pretrained(ENC[0], revision=ENC[1]).cuda().eval()
    vecs = []
    with torch.no_grad():
        for s in range(0, len(texts), 64):
            b = tok(texts[s : s + 64], return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            vecs.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    np.savez_compressed("geometry/vectors.npz", v=np.concatenate(vecs))
    json.dump({"texts": texts, "index": index}, open("geometry/index.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps({"encoded": len(texts)}))


if __name__ == "__main__":
    main()
