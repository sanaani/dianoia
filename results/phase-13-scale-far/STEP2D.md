# Phase 13 step 2d: the scoring filter

**F1: FAIL.** The top-scored 20% of 1497 held-out rejects hold 32% of their 47 strict positives (95% interval 18% to 45%; bar 70%).

| Top share read | Recall | Strict positives per pair read |
|---|---|---|
| 5% | 11% | 0.067 |
| 10% | 17% | 0.053 |
| 20% | 32% | 0.050 |
| 30% | 47% | 0.049 |
| 50% | 70% | 0.044 |

Reading at random finds 0.031 strict positives per pair.

Training (BAAI/bge-reranker-base, 3 epochs, chosen epoch 2 by dev): dev recall in the top 20% was 35%, 41%, 40%.
The filter ranks only about 1.5x better than reading at random, so the strict rule's judgment is not learnable from
the two abstracts by this model at this data size. F2 is not run (it runs only if F1 passes).
