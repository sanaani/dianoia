# Phase 15 step 2: can Fastino throw away the worst half before Sonnet reads?

Fastino GLiNER2.5-Decide fine-tuned on 3,816 examples (816 positive, positives repeated 4 times), scored on 4,519 held-out phase 14 far pairs from articles it never saw, 92 of them strict positives.

**S1: FAIL.** Discarding the lowest-scored 50.0% keeps 73 of 92 strict positives (79.3%; exact 95% lower bound 69.6%; bar 90.0%).

| Discarded | Strict positives kept | Lower bound | v3p gate kept | At random |
|---|---|---|---|---|
| 10.0% | 97.8% (90/92) | 92.4% | 96.5% | 90.0% |
| 20.0% | 91.3% (84/92) | 83.6% | 89.6% | 80.0% |
| 30.0% | 85.9% (79/92) | 77.0% | 83.5% | 70.0% |
| 40.0% | 80.4% (74/92) | 70.9% | 78.3% | 60.0% |
| 50.0% | 79.3% (73/92) | 69.6% | 76.5% | 50.0% |
| 60.0% | 76.1% (70/92) | 66.1% | 72.2% | 40.0% |
| 70.0% | 72.8% (67/92) | 62.5% | 68.7% | 30.0% |
| 80.0% | 70.7% (65/92) | 60.2% | 63.5% | 20.0% |
| 90.0% | 54.4% (50/92) | 43.6% | 49.6% | 10.0% |

- Largest discard whose lower bound stays at or above 90.0%: 12.0%.
- Area under the ROC curve: 0.777 (0.5 is chance).
- GPU: training 15.0 min, scoring 2.7 min.
