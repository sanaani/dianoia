"""Every encoder behind one call: ``encode(name, texts) -> float32 array, rows L2-normalised``.

Four open sentence encoders of different families, plus two controls that can only see words:

- ``minilm``  all-MiniLM-L6-v2, 384-d, distilled BERT, contrastive on 1B pairs. The cheap baseline.
- ``bge``     BAAI/bge-base-en-v1.5, 768-d, BERT, contrastive with hard negatives.
- ``e5``      intfloat/e5-base-v2, 768-d, BERT, weakly supervised contrastive; ``query:`` prefix.
- ``gtr``     sentence-transformers/gtr-t5-base, 768-d, T5 encoder. The one vec2text can invert.
- ``qwen3``   Qwen/Qwen3-Embedding-0.6B, 1024-d, decoder LLM turned encoder, last-token pooling.

Controls:

- ``wordavg`` the mean of bge's *input* token embeddings: every word looked up alone, no context,
  then averaged. It is the word-level centroid, the analogue of Noema's state cloud read as a bag.
- ``lsa``     TF-IDF over the corpus reduced to 256 dimensions by SVD. Pure co-occurrence.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge": ("BAAI/bge-base-en-v1.5", ""),
    "e5": ("intfloat/e5-base-v2", "query: "),
    "gtr": ("sentence-transformers/gtr-t5-base", ""),
    "qwen3": ("Qwen/Qwen3-Embedding-0.6B", ""),
}
ENCODERS = [*MODELS, "wordavg", "lsa"]


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


@lru_cache(maxsize=None)
def _st(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODELS[name][0], device="cpu")


@lru_cache(maxsize=None)
def _wordavg():
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS["bge"][0])
    table = AutoModel.from_pretrained(MODELS["bge"][0]).get_input_embeddings().weight.detach().numpy()
    return tok, table


_LSA = None


def fit_lsa(texts: list[str]) -> None:
    """Fit the LSA control once, on the corpus pool, before it encodes anything."""
    global _LSA
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    _LSA = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, ngram_range=(1, 2)),
        TruncatedSVD(n_components=256, random_state=0),
    ).fit(texts)


def encode(name: str, texts: list[str], batch_size: int = 64) -> np.ndarray:
    if name in MODELS:
        prefix = MODELS[name][1]
        model = _st(name)
        vecs = model.encode([prefix + t for t in texts], batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        return _normalise(vecs)
    if name == "wordavg":
        tok, table = _wordavg()
        out = np.zeros((len(texts), table.shape[1]), dtype=np.float32)
        for i, t in enumerate(texts):
            ids = tok(t, add_special_tokens=False)["input_ids"]
            if ids:
                out[i] = table[ids].mean(axis=0)
        return _normalise(out)
    if name == "lsa":
        if _LSA is None:
            raise RuntimeError("fit_lsa() must run on the pool first")
        return _normalise(_LSA.transform(texts))
    raise KeyError(name)
