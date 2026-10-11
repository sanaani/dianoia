# Phase 11: does training on analogy pairs teach a model to find far-field sources? (written 10/06/2026, before data)

Shaun (10/06/2026): test the LLM training idea with the phase 10 pairs, at a size well beyond the minimum, after
measuring how much data the effect needs. Fixed before the training set exists; reported as written, including what
fails.

## Data

- **Pairs:** citing patent → earlier patent from a distant CPC section that it borrowed from, kept by the exclusion
  cascade (Fastino stages 1-3), Sonnet v3 borrow ≥ 4 and Sonnet v4 C/G (`results/phase-10-analogy/RUBRIC.md`).
  Estimated 80% borrowings, 20% same product (DATA-REVIEW.md).
- **Train:** citing years 2013-2017 and 2020, from the scale-up corpus. **Test:** the phase 10 test set (citing
  2021-2024, 1,276 pairs); **val:** phase 10 val (2018-2019, 923). Any training pair that shares a patent (either end)
  with a val or test pair is dropped.
- **Amended 10/06/2026, before any training (Shaun's decision):** the fixed train years yielded 5,878 pairs after
  Sonnet v4, short of the 8,000 floor below. Training also takes corpus pairs from the val years (2018-2019), with the
  same shared-patent rule. Built (`p10-label.py train`, `p10/label/train-p11.jsonl`): **5,862 pairs, 4,439 citing
  patents**; the shared-patent rule dropped 824, about a third of 2018-2019's, so those years added only 526. Consequences, stated with every result: N is still under 8,000;
  hyperparameters are chosen on val years the model has trained near (not on val pairs or their patents); the test
  years (2021-2024) stay unseen. The learning curve is 1k, 3k and N.
- **Before any training:** a hand check of 50 training pairs with `p10-label.py review` (rater agreement ≥ 9/10).
  **Done 10/06/2026** (`p10/review-train-p11*`, seed 23, rejudge from `rubric-cascade50`): rater agreement 9/10;
  of 50 training pairs, 41 kept (C 15, G 26), 6 same product, 3 unrelated: **82% borrowings, 12% same product, 6%
  unrelated** (n = 50), in line with the test set's estimated 80/20. The phase 10 gate's 10% same-product cap is
  missed by one pair; training proceeds with this purity stated, as for the test set.

## Method

- **Model:** Qwen3-1.7B (base), LoRA. **Task:** given the citing abstract, write a short description of an invention
  from a different field that would supply its key mechanism. **Training target:** the cited abstract.
- **Scoring (real outcome, no LLM grading):** the written description is encoded with Qwen3-Embedding-0.6B and ranks
  the 1,000 phase 10 candidates (the true cited patent plus 999 from its own section, none sharing a section with the
  citing patent). Score: share of queries with the true source in the top 10 (chance 1%), and MRR.
- **Baselines:** (a) the raw citing abstract as the query (phase 10's level 0); (b) the same prompt to untrained
  Qwen3-1.7B.
- **Learning curve:** train at 1k, 3k and N pairs (N = all training pairs, at least 8k), one seed each; the same
  hyperparameters, chosen on val at 3k.

## Hypotheses

**H1. Training helps.** At N, the trained model beats untrained Qwen3-1.7B on test top-10 (lower bound of the paired
bootstrap difference > 0). Falsified if the difference's interval includes 0 or is negative.

**H2. It beats plain retrieval.** At N, the trained model beats the raw-abstract query on test top-10 (lower bound > 0).
Falsified as for H1. (H1 can pass while H2 fails: training helps the generator but not enough to beat the abstract.)

**H3. More data still pays at N.** The gain from 3k to N is positive (lower bound > 0). If so, the 5.08M-borrowing
expansion is worth building; if the curve is flat from 3k, it is not, whatever H1 says.

**Reported, not tested:** results by overlap quartile (`overlap` column), so any gain can be seen to come from far
(low overlap) or near (high overlap) pairs; the 20% same-product share is stated with every number.

## Known limits

- The citing abstract describes the solution after the borrowing was made, so it may name the borrowed mechanism. The
  overlap split shows how much that matters.
- One seed per size; the curve's shape is indicative.
- Budget: about $10 of AWS at the start of phase 11; the corpus run (~$3) comes first. Actual: the corpus run, an
  upload retry and the speed probe used about $5.50, leaving about $4.50 for training and scoring.
