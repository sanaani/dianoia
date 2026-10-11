# Phase 11: how each hypothesis is measured and judged

Written 10/07/2026, before any phase 11 result. The hypotheses are
`results/phase-11-analogy-training/HYPOTHESES.md`; `scripts/p11-verdicts.py` applies the rules below and writes
`results/phase-11-analogy-training/REPORT.md`.

## Data

- **Training pairs:** `p10/label/train-p11.jsonl` (5,862 pairs; citing years 2013-2020; stages 1-2, Sonnet v3 borrow
  ≥ 4, Sonnet v4 C or G; no patent shared with a phase 10 val or test pair). Shuffled once (seed 0); a size-k model
  trains on the first k, so the sets are nested.
- **Evaluation:** phase 10's own queries (`p10/queries.parquet`): the same citing patents, true cited patents and 999
  same-section candidates per query, val (2018-2019) and test (2021-2024).
- **Audit gate** (`p11-data.py audit`): any training pair sharing a patent with an evaluation pair, a training year
  outside 2013-2020, duplicates, empty texts, or a missing candidate abstract stops the pipeline.

## Method

- **Model:** Qwen3-1.7B-Base with LoRA (r 16, alpha 32, all linear layers), learning rate 2e-4, batch 16, abstracts
  truncated to 320 tokens. Loss on the target only.
- **Task:** the prompt is the citing abstract followed by "An invention from a different field that supplies this
  invention's key mechanism:"; the training target is the cited abstract.
- **Descriptions:** greedy, at most 160 new tokens, cut at the first blank line.
- **Retrieval:** Qwen3-Embedding-0.6B. A description is encoded with the instruction "Given a description of an
  invention, retrieve the patent abstract that matches it"; the raw-abstract baseline with phase 10's instruction;
  candidates with none (as in phase 10). A query's score is whether the true cited patent ranks in the top 10 of
  1,000 (chance 1%); MRR is also reported.
- **Models:** untrained (the same prompt to the base model); 3k pairs at 1 and 2 epochs; 1k and N (all) pairs at the
  epoch count val chose at 3k (top-10, then MRR). The raw citing abstract as the query is the retrieval baseline.

## Verdict rules (test split; 95% paired bootstrap over queries, 2,000 resamples)

| | Comparison | Supported if | Falsified if |
|---|---|---|---|
| **H1** | N-pair model vs untrained | lower bound of the top-10 difference > 0 | the interval includes 0 or is negative |
| **H2** | N-pair model vs raw abstract | lower bound > 0 | the interval includes 0 or is negative |
| **H3** | N-pair model vs 3k model | lower bound > 0 | the interval includes 0 or is negative |

Selection (epochs) happens on val only; every verdict is on test. MRR differences and test top-10 by word-overlap
quartile are reported, not tested.

## Known limits

- N = 5,862, under the 8,000 first planned; one seed per size, so the curve's shape is indicative.
- Training pairs are an estimated 82% borrowings, 12% same product, 6% unrelated (hand check, n = 50).
- Val years 2018-2019 contribute 526 training pairs (none sharing a patent with val pairs), so val is not fully
  unseen in time; test is.
