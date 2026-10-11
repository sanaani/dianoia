"""Build the phase-0 corpus: authored, de-duplicated Enron paragraphs and their sentences.

Rules fixed before looking at any encoder output:

1. Keep only the authored part of each body (``corpus.authored``) and drop exact duplicate bodies,
   keeping the earliest message. Enron files one message in several folders.
2. A paragraph is kept if it is prose of 20-200 words and appears in fewer than 3 distinct messages
   (anything more is a disclaimer, signature or template).
3. Messages with at least one kept paragraph are sampled by a fixed seed, never by hand.
4. Sentences come from pysbd. A sentence is kept at 5-60 words, and marked boilerplate if its
   normalised text appears in 3 or more distinct messages across the whole de-duplicated corpus.
5. A paragraph enters the paragraph set if it has 2-8 kept, non-boilerplate sentences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pysbd

from dianoia import corpus
from dianoia.paths import ENRON, result_path, work_path

SEED = 20260930
_ROUGH_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


COLS = ["message_id", "subject", "from", "to", "cc", "date", "body", "file_name"]


def batches(columns: list[str]):
    """Stream the Enron parquet 20,000 rows at a time; the whole corpus never sits in memory."""
    for f in sorted(ENRON.glob("train-*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=20000, columns=columns):
            yield batch.to_pandas()


def h(text: str) -> int:
    """An 8-byte digest of normalised text, small enough to count millions of."""
    return int.from_bytes(hashlib.blake2b(corpus.normalise(text).encode(), digest_size=8).digest(), "little")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emails", type=int, default=30000, help="messages to sample")
    args = ap.parse_args()
    stats: dict[str, int | float] = {"raw_messages": 0}

    # Pass 1: the earliest message per authored body, and document frequencies over unique bodies.
    first: dict[int, tuple] = {}
    para_df: Counter[int] = Counter()
    sent_df: Counter[int] = Counter()
    for df in batches(["message_id", "date", "body"]):
        stats["raw_messages"] += len(df)
        for mid, date, body in zip(df["message_id"], df["date"], df["body"].fillna("")):
            text = corpus.authored(body)
            if not text:
                continue
            key = h(text)
            stamp = (date, mid)
            if key in first:
                if stamp < first[key]:
                    first[key] = stamp
                continue
            first[key] = stamp
            ps = corpus.paragraphs(text)
            para_df.update({h(p) for p in ps})
            sent_df.update({h(s) for p in ps for s in _ROUGH_SENT.split(p)})
    keep_mid = {mid for _, mid in first.values()}
    stats["unique_authored_bodies"] = len(keep_mid)
    del first

    def keep_para(p: str) -> bool:
        return corpus.is_prose(p, min_words=20, max_words=200) and para_df[h(p)] < 3

    # Pass 2: which kept messages have a kept paragraph; metadata only.
    meta = []
    for df in batches(COLS):
        df = df[df["message_id"].isin(keep_mid)]
        for row in df.itertuples(index=False):
            text = corpus.authored(row.body or "")
            if any(keep_para(p) for p in corpus.paragraphs(text)):
                meta.append({"message_id": row.message_id, "subject": row.subject, "from": getattr(row, "_2"),
                             "to": list(row.to) if row.to is not None else [], "cc": list(row.cc) if row.cc is not None else [],
                             "date": row.date, "file_name": row.file_name})
    eligible = pd.DataFrame(meta).sort_values(["date", "message_id"]).reset_index(drop=True)
    stats["messages_with_a_kept_paragraph"] = len(eligible)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(eligible), size=min(args.emails, len(eligible)), replace=False)
    sample = eligible.iloc[np.sort(pick)].copy()
    sample["folder"] = sample["file_name"].str.split("/").str[1]
    sample["mailbox"] = sample["file_name"].str.split("/").str[0]
    sample["thread"] = sample["subject"].map(corpus.thread_key)
    stats["messages_sampled"] = len(sample)
    chosen = set(sample["message_id"])

    # Pass 3: paragraphs and sentences of the sampled messages.
    seg = pysbd.Segmenter(language="en", clean=False)
    p_rows, s_rows, authored_of = [], [], {}
    boiler = 0
    for df in batches(["message_id", "body"]):
        df = df[df["message_id"].isin(chosen)]
        for mid, body in zip(df["message_id"], df["body"].fillna("")):
            if mid in authored_of:
                continue
            text = corpus.authored(body)
            authored_of[mid] = text
            for idx, p in enumerate(corpus.paragraphs(text)):
                if not keep_para(p):
                    continue
                pid = f"{mid}#{idx}"
                kept = 0
                for j, s in enumerate(x.strip() for x in seg.segment(p)):
                    if not s or not corpus.is_prose(s, min_words=5, max_words=60):
                        continue
                    is_boiler = sent_df[h(s)] >= 3
                    boiler += is_boiler
                    kept += not is_boiler
                    s_rows.append({"sid": f"{pid}.{j}", "pid": pid, "message_id": mid, "idx": j, "text": s, "boilerplate": bool(is_boiler)})
                p_rows.append({"pid": pid, "message_id": mid, "idx": idx, "text": p, "n_sentences": kept})
    sample["authored"] = sample["message_id"].map(authored_of)

    paras = pd.DataFrame(p_rows)
    sents = pd.DataFrame(s_rows)
    paras["in_paragraph_set"] = paras["n_sentences"].between(2, 8)
    stats["paragraphs"] = len(paras)
    stats["paragraphs_in_paragraph_set"] = int(paras["in_paragraph_set"].sum())
    stats["sentences"] = len(sents)
    stats["sentences_boilerplate"] = int(boiler)
    stats["unique_sentence_texts"] = int(sents.loc[~sents["boilerplate"], "text"].map(corpus.normalise).nunique())
    stats["mailboxes"] = int(sample["mailbox"].nunique())
    stats["seed"] = SEED

    cols = ["message_id", "subject", "thread", "from", "to", "cc", "date", "mailbox", "folder", "authored"]
    sample[cols].to_parquet(work_path("corpus", "emails.parquet"))
    paras.to_parquet(work_path("corpus", "paragraphs.parquet"))
    sents.to_parquet(work_path("corpus", "sentences.parquet"))
    out = result_path("phase-0-corpus", "corpus.json")
    out.write_text(json.dumps(stats, indent=2, default=str) + "\n")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
