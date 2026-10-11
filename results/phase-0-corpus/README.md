# Phase 0: the corpus

Enron email, `corbt/enron-emails` at revision `cfc06c75`, 517,401 messages (parquet SHA-256 in
`~/dianoia-data/enron/SHA256SUMS`). Built by `scripts/build-corpus.py`; the counts are `corpus.json`.

| step | count |
|---|---|
| raw messages | 517,401 |
| unique authored bodies (quoted and forwarded text cut, exact duplicates dropped) | 213,021 |
| messages with at least one kept paragraph | 107,827 |
| sampled by seed 20260930 | 30,000 (150 mailboxes) |
| paragraphs kept | 63,683 |
| paragraphs of 2 to 8 kept sentences (the paragraph set) | 45,193 |
| sentences | 166,988, of which 5,541 boilerplate |
| unique non-boilerplate sentence texts | 159,106 |

## Caveats

- **Not all of it is office prose.** A hand read of 15 random sentences found news articles pasted
  into mail, newsletters and fantasy football beside ordinary business mail. It is real Enron mail and
  is kept; phase results that depend on register say so.
- Boilerplate is judged by a rough regex sentence split over the whole de-duplicated corpus
  (3 or more distinct messages), and the kept sentences by pysbd. The two splitters disagree at the
  edges, so a little boilerplate survives.
- A paragraph repeated in 3 or more messages is dropped as a template; one sent in 2 is kept.
