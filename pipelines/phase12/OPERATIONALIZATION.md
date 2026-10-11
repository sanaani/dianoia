# Phase 12: how each hypothesis is measured and judged

Written 10/07/2026, before the far set existed and before any phase 12 result. The hypotheses are
`results/phase-12-far-analogy/HYPOTHESES.md`; `scripts/p12-verdicts.py` applies the rules below and writes
`results/phase-12-far-analogy/REPORT.md`.

## Data

- **Pool:** phase 10's kept pairs (`p10/label/kept-final.parquet`, texts from `p10/abstracts.parquet`) and phase 11's
  training pairs (`p10/label/train-p11.jsonl`). Every abstract is cut to its first 180 words, as the pairs' own texts
  were (`p10-label.py` WORDS).
- **Far and near** (`p12-data.py measure`): word overlap is `p10-label.py overlap_scorer` with IDF from every abstract
  in the pool (phase 10's formula; the numbers differ slightly from kept-final's column, whose IDF came from fewer
  abstracts). Cosine is BAAI/bge-small-en-v1.5, normalized, no instruction. Far = strictly below the pool median on
  both; near = at or above it on both; the rest are unused.
- **Evaluation** (`p12-data.py build`): phase 10's queries with their own 999 candidates. Far dev = val queries whose
  pair is far; far test = test queries whose pair is far; near test = test queries whose pair is near. Fewer than 200
  far test queries stops the build.
- **Training:** every far pair, and the near control, that shares no patent with **any** phase 10 query, true source
  or candidate (all 1,909 queries, not only the far ones: stricter than HYPOTHESES.md asks). The near set is sampled
  (seed 0) to the far set's size.
- **Hard negatives:** 7 per pair, drawn at random (seed 0) from patents with an abstract on disk in the cited patent's
  first section, granted before the citing patent, sharing no section with it, not cited by it in any category (the
  full citation table), never a phase 10 evaluation patent or one end of a training pair. Cut to 180 words.
- **Hand check gate** (`p12-data.py handcheck`): `p10-label.py review` draws 50 unread far test pairs and, separately,
  50 far training pairs, each shuffled with 10 pairs from `rubric-cascade50` for rater agreement. Claude labels them
  blind with RUBRIC.md. The gate passes if the test reading counts (agreement ≥ 9/10) and at least 70% of its 50
  pairs are C or G. The training reading is reported.
- **Audit gate** (`p12-data.py audit`): any training pair or negative that is a phase 10 evaluation patent, a
  negative breaking the candidate rule, a wrong negative count, duplicates, empty texts, a far or near rule broken,
  unequal set sizes, an evaluation pair with the wrong label, or a missing candidate abstract stops the pipeline.

## Method

- **Model:** Qwen3-Embedding-0.6B with LoRA (r 16, alpha 32, dropout 0.05, all linear layers), last-token pooling,
  phase 10's query instruction on the citing side and none on documents.
- **Loss:** InfoNCE at temperature 0.05, batch 32 pairs. Each query's negatives are every other document in the
  batch: its own 7 hard negatives, the other pairs' cited abstracts and their hard negatives (HYPOTHESES.md names the
  cited abstracts; their negatives are added, as is standard). A copy of the query's own cited patent elsewhere in the
  batch is masked. Exact large batches by gradient caching. AdamW, learning rate 1e-4, warmup 5% then linear decay,
  gradient norm clipped at 1, training texts truncated at 320 tokens; evaluation at 512, as phase 10.
- **Runs:** far and near at seed 0 for 3 epochs, scored after each. Each one's epoch count is chosen on far dev
  top-10, then MRR. Far is retrained at seeds 1 and 2 for its chosen count; the near control is seed 0 only.
- **Score:** for each query, whether the true cited patent ranks in the top 10 of its 1,000 (ties count against it;
  chance 1%), and the reciprocal rank.

## Verdict rules (95% paired bootstrap over test queries, 2,000 resamples)

| | Comparison (seed 0, chosen epochs) | Supported if | Falsified if |
|---|---|---|---|
| **H1** | far-trained vs off the shelf, far test top-10 | lower bound > 0 | the interval includes 0 or is negative |
| **H2** | far-trained vs near-trained, far test top-10 | lower bound > 0 | the interval includes 0 or is negative |
| **H3** | far-trained vs off the shelf, near test top-10 | lower bound > -0.03 | lower bound ≤ -0.03 |

Selection (epochs) happens on far dev only; every verdict is on test. Reported, not tested: MRR differences; far test
top-10 by quartile of each far measure; the far model's spread over seeds 0-2; both hand-check readings; the near
model on near test.

## Known limits

The limits in HYPOTHESES.md hold. Also: far test queries are a subset of phase 10's test queries chosen by their
pair's measures, so the off-the-shelf score on them is expected below phase 10's 68%; that is the point of the split,
not a defect of the baseline.
