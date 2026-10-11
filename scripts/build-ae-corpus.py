"""Stage 0 at scale: every kept Enron sentence from fit mailboxes, for the autoencoder.

Same rules as ``build-corpus.py`` (authored text only, exact-duplicate bodies dropped, template
paragraphs dropped, boilerplate sentences dropped, prose filter) but no sampling: every eligible
message whose mailbox is neither a test nor a validation mailbox. Streams the parquet in three passes.
Writes ``work/ae/enron-fit.jsonl.gz`` (one ``{"text", "role": "fit"}`` per unique sentence).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter

import pyarrow.parquet as pq
import pysbd

from dianoia import corpus
from dianoia.paths import ENRON, result_path, work_path

_ROUGH_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def h(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(corpus.normalise(text).encode(), digest_size=8).digest(), "little")


def batches(columns):
    for f in sorted(ENRON.glob("train-*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=20000, columns=columns):
            yield b.to_pandas()


def main() -> None:
    gen = json.loads(result_path("phase-1-intent", "generation.json").read_text())
    val = set(json.loads(result_path("phase-1-intent", "train-head.json").read_text())["val_mailboxes"])
    excluded = set(gen["test_mailboxes"]) | val

    first: dict[int, tuple] = {}
    para_df: Counter[int] = Counter()
    sent_df: Counter[int] = Counter()
    for df in batches(["message_id", "date", "body"]):
        for mid, date, body in zip(df["message_id"], df["date"], df["body"].fillna("")):
            text = corpus.authored(body)
            if not text:
                continue
            key = h(text)
            if key in first:
                if (date, mid) < first[key]:
                    first[key] = (date, mid)
                continue
            first[key] = (date, mid)
            ps = corpus.paragraphs(text)
            para_df.update({h(p) for p in ps})
            sent_df.update({h(s) for p in ps for s in _ROUGH_SENT.split(p)})
    keep_mid = {mid for _, mid in first.values()}
    del first

    seg = pysbd.Segmenter(language="en", clean=False)
    seen: set[int] = set()
    n_msgs = n_sents = 0
    with gzip.open(work_path("ae", "enron-fit.jsonl.gz"), "wt", encoding="utf-8") as out:
        for df in batches(["message_id", "body", "file_name"]):
            for mid, body, fname in zip(df["message_id"], df["body"].fillna(""), df["file_name"]):
                if mid not in keep_mid or fname.split("/")[0] in excluded:
                    continue
                keep_mid.discard(mid)
                used = False
                for p in corpus.paragraphs(corpus.authored(body)):
                    if not corpus.is_prose(p, min_words=20, max_words=200) or para_df[h(p)] >= 3:
                        continue
                    for s in (x.strip() for x in seg.segment(p)):
                        if not s or not corpus.is_prose(s, min_words=5, max_words=60) or sent_df[h(s)] >= 3:
                            continue
                        k = h(s)
                        if k in seen:
                            continue
                        seen.add(k)
                        out.write(json.dumps({"text": s, "role": "fit"}, ensure_ascii=False) + "\n")
                        n_sents += 1
                        used = True
                n_msgs += used
    summary = {"messages": n_msgs, "sentences": n_sents, "excluded_mailboxes": len(excluded)}
    result_path("phase-3-own-ae", "enron-fit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
