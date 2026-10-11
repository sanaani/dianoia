"""GPU worker: encode a JSON-lines file of texts with SONAR and the off-the-shelf encoders.

Runs on the AWS instance, not the laptop. Input ``texts.jsonl`` (one ``{"text": ...}`` per line);
output one float16 ``.npy`` per encoder, rows in input order. SONAR vectors are the raw mean-pooled
encoder output (not normalised), which is what SONAR's decoder expects; the others are L2-normalised.

SONAR is the Hugging Face port (encoder ``cointegrated/SONAR_200_text_encoder`` at a pinned
revision, reported by its porter as cosine 1.000000 to the fairseq2 original).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

SONAR_ENC = ("cointegrated/SONAR_200_text_encoder", "29f2eac73636d1548c2cb227dbfb2772b31ebc4f")
ST = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge": ("BAAI/bge-base-en-v1.5", ""),
    "e5": ("intfloat/e5-base-v2", "query: "),
    "gtr": ("sentence-transformers/gtr-t5-base", ""),
    "qwen3": ("Qwen/Qwen3-Embedding-0.6B", ""),
}


@torch.inference_mode()
def sonar(texts: list[str], batch: int = 128) -> np.ndarray:
    from transformers import AutoTokenizer
    from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

    tok = AutoTokenizer.from_pretrained(SONAR_ENC[0], revision=SONAR_ENC[1])
    tok.src_lang = "eng_Latn"
    enc = M2M100Encoder.from_pretrained(SONAR_ENC[0], revision=SONAR_ENC[1]).cuda().half().eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 1024), dtype=np.float16)
    for s in range(0, len(texts), batch):
        idx = order[s : s + batch]
        b = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        h = enc(**b).last_hidden_state.float()
        m = b["attention_mask"].unsqueeze(-1).float()
        out[idx] = ((h * m).sum(1) / m.sum(1)).cpu().numpy().astype(np.float16)
    return out


def st(name: str, texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model_id, prefix = ST[name]
    model = SentenceTransformer(model_id, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    v = model.encode([prefix + t for t in texts], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("texts")
    ap.add_argument("out")
    ap.add_argument("--encoders", nargs="*", default=["sonar", *ST])
    args = ap.parse_args()
    texts = [json.loads(line)["text"] for line in open(args.texts, encoding="utf-8")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.encoders:
        t = time.time()
        v = sonar(texts) if name == "sonar" else st(name, texts)
        np.save(out / f"{name}.npy", v)
        print(json.dumps({"encoder": name, "n": len(texts), "seconds": round(time.time() - t, 1)}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
