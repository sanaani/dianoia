"""GPU worker: 1M general English sentences for stage 0, from sentence-transformers/wikipedia-en-sentences.

Pinned revision; sentences of 5-60 words, sampled by seed. Writes ``ae/wiki.jsonl.gz``.
"""

from __future__ import annotations

import gzip
import json
import random
import re

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO, REV = "sentence-transformers/wikipedia-en-sentences", "4a0972dcb781b5b5d27799798f032606421dd422"
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

texts = []
for name in ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"):
    path = hf_hub_download(REPO, name, repo_type="dataset", revision=REV)
    col = pq.read_table(path).column(0).to_pylist()
    texts += [t for t in col if isinstance(t, str) and 5 <= len(WORD.findall(t)) <= 60]
random.Random(20261001).shuffle(texts)
with gzip.open("ae/wiki.jsonl.gz", "wt", encoding="utf-8") as fh:
    for t in texts[:1_000_000]:
        fh.write(json.dumps({"text": t, "role": "fit"}, ensure_ascii=False) + "\n")
print(json.dumps({"eligible": len(texts), "kept": min(len(texts), 1_000_000)}))
