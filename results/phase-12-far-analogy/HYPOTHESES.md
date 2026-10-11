# Phase 12: can an encoder learn to match far analogies? (written 10/07/2026, before data)

Shaun (10/07/2026): we are looking for **far** analogies, where the source shares no direct semantic relationship
with the problem (README § Critical finding). "The encoder is trying to throw away everything that is not a far
analogy and only match things that have functional relationships in the abstract sense." Reuse the patent data
already collected and sorted; train on the far borrowings as written, with no rewriting. Fixed before the far set
exists; reported as written, including what fails. Nothing runs until Shaun approves this file.

Replaces the first draft of this file (commit 1ec5451), which had Sonnet restate each problem before searching.
Dropped before any data: restating only the question leaves the library in its own field's words, and rewriting the
library is the expensive part. A trained encoder treats the question and the library the same way, with no LLM in
the search path.

## Experiment card

- **Question:** does fine-tuning the search encoder on far patent borrowings (citing abstract and cited abstract,
  as written) make it find far sources it misses today?
- **Cheapest decisive test:** a small trial on the far pairs we already have (about 2,000 to train on, about 300-400
  to test, estimated). Training takes minutes to an hour on one GPU (under $2). The test set and its decoys already
  exist (phase 10). Nothing cheaper answers it: the question is whether training changes what the encoder matches.
- **Baseline:** the same encoder off the shelf (Qwen3-Embedding-0.6B, phase 10's setup); control: the same training
  on an equal number of **near** pairs.
- **Deciding metric:** test top-10 recovery on far pairs (true source in the top 10 of 1,000; chance 1%).
- **What would change the next step:** if far training helps, scale the far pairs with the labeling cascade over the
  5.1 million cross-field citations, and fetch the abstracts that are missing. If it helps no more than near training,
  the gain is ordinary patent matching, not far analogy. If nothing helps, about 2,000 real far pairs are not enough
  to teach it, and more data is the only remaining lever to test.

## Data

- **Pool:** every confirmed borrowing we have: phase 10's kept pairs (`p10/label/kept-final.parquet`, 2,199) and
  phase 11's training pairs (`p10/label/train-p11.jsonl`, 5,862), about 8,061. All are applicant citations (the
  inventor's, not the examiner's) to a patent sharing no CPC section, and passed the phase 10 labeling cascade
  (estimated 80-82% borrowings, 12-20% same product).
- **Far and near:** a pair is far when it is below the pool median on **both** (a) IDF-weighted word overlap (`p10-label.py
  overlap`, the measure in kept-final) and (b) cosine similarity of the two abstracts under a second encoder,
  BAAI/bge-small-en-v1.5. Neither is the encoder being trained and scored, so the off-the-shelf baseline is not set
  up to fail. A pair above the median on both is near. Pairs that are mixed (below on one, above on the other) are
  used for nothing.
- **Splits, by phase 10's split:**
  - Test: phase 10's test queries (citing 2021-2024) whose pair is far. Their 1,000 candidates already exist.
  - Dev: phase 10's val queries (2018-2019) whose pair is far. Used only to choose the number of epochs.
  - Train: every other far pair. A training pair is dropped if it shares a patent (either end) with any dev or test
    query, true source or candidate.
  - If fewer than 200 far test queries exist, stop and ask Shaun before changing the rule.
- **Near control set:** near pairs from the training years, sampled with seed 0 to the same count as the far training
  set, with the same exclusions.
- **Hard negatives:** for each training pair, 7 patents from the cited patent's first CPC section, drawn by phase
  10's candidate rule (`p10-data.py`: granted before the citing patent, sharing no section with it, not cited by it)
  from patents whose abstracts are on disk. Because they come from the true source's own field, the encoder cannot
  win by recognizing the field; it has to find what connects the pair.
- **Hand check before any training:** 50 far test pairs, read blind (`p10-label.py review` format): is it a real
  borrowing, with a shared function? Gate: at least 70% real. Also 50 far training pairs, reported. Distant citations
  are more often noise, which is why the check comes first.
  **Done 10/07/2026** (`p12/review-test*`, `p12/review-train*`, seed 12, rejudge from `rubric-cascade50`). Far test
  pairs: rater agreement 9/10; of 50, 45 real (C 19, G 26), 0 same product, 5 unrelated: **90% real, gate passed**.
  Far training pairs: rater agreement 8/10, under the 9/10 rule, so that reading is **not counted** (as read: C 22,
  G 24, unrelated 4 of 50); it was reported only and gates nothing.
- **Found by the hand check (10/07/2026), before any training:** two test pairs had the same text under different
  patent numbers, and 50 training pairs had an abstract identical to an evaluation query's or true source's
  (continuations reuse an abstract). The patent-number rule missed them. The build now also drops any training pair
  or hard negative whose text (first 60 words) matches an evaluation query, true source or candidate, and the audit
  checks it.
- **Audit (exits 1 on any violation; training does not start without a pass of the exact files):** no dev or test
  patent anywhere in training (as a pair or as a negative); no duplicate pairs; no empty abstracts; the far and near
  rules hold for every pair; the near set has the far set's size.

## Method

- **Model:** Qwen3-Embedding-0.6B with LoRA (r 16), phase 10's query instruction on the citing side, no instruction
  on the cited side, as in the phase 10 search.
- **Loss:** contrastive (InfoNCE, temperature 0.05): pull each citing abstract toward its cited abstract and away from
  its 7 hard negatives and every other cited abstract in the batch. Batch 32, learning rate 1e-4, warmup 5%, abstracts
  truncated at 320 tokens.
- **Epochs:** 1, 2 or 3, chosen on dev far top-10 (then MRR), separately for the far-trained and near-trained models.
- **Seeds:** verdicts use seed 0. Seeds 1 and 2 are trained and their spread is reported.
- **Scoring (real outcome, no LLM grading):** each model ranks the same 1,000 candidates per query; share of queries
  with the true source in the top 10, and MRR.

## Hypotheses

Paired bootstrap over test queries (2,000 resamples), 95% interval on the difference in top-10 recovery.

**H1. Far training helps on far pairs.** The far-trained encoder beats the off-the-shelf encoder on far test pairs
(lower bound > 0). Falsified if the interval includes 0 or is negative.

**H2. It is the far pairs that help.** The far-trained encoder beats the near-trained encoder on far test pairs (lower
bound > 0). Falsified as for H1. If H1 passes and H2 fails, the gain is from training on patent citations at all, not
from far analogy.

**H3. It costs nothing on near pairs.** On near test pairs, the far-trained encoder minus the off-the-shelf encoder has
a lower bound above -3 points. Falsified if the lower bound is -3 points or below. The product answers near questions
too, and a model that only trades one for the other is not an improvement.

**Reported, not tested:** MRR; recovery by far-measure quartile; the seed spread; the hand check's share of real
borrowings; the near-trained encoder on near test pairs.

## Known limits

- About 2,000 far training pairs is small for teaching an encoder. A failure here may mean too little data, not a
  wrong idea; the scale-up in the experiment card tests that.
- Some far pairs are noise (an inventor citing something barely related). The hand check measures how many, and every
  result is read in that light.
- The encoder may learn finer field habits (which subclasses tend to cite which) rather than functional relationships.
  Decoys from the source's own section block the coarse version of that shortcut, not every version.
- The citing abstract describes the finished invention, so it may name the borrowed mechanism. The far filter reduces
  that but cannot remove it. The product's input is a "how do I" question with no solution in it.
- Patents are technical. A pass shows matching across engineering fields, not that the encoder connects a business
  problem to entomology; that needs research papers (OpenAlex) and is a later test.
- Budget: about $9.60 of AWS left after phase 11.
