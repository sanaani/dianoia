# Phase 15 step 2 follow-up: is Fastino a fair ranker? (descriptive)

Asked on 10/09/2026 after S1 failed; nothing here was pre-registered, so there is no verdict. 4,519 held-out test pairs; Sonnet reads the top 20% (903) by each ranker. Nearness is phase 14's word overlap and off-the-shelf cosine between the two abstracts (higher = nearer).

## Strict positives (92)

| Ranker | Positives in its top 20% | Share |
|---|---|---|
| Fastino | 65 | 70.7% |
| word overlap (high first) | 27 | 29.3% |
| cosine (high first) | 29 | 31.5% |
| word overlap (low first) | 14 | 15.2% |
| cosine (low first) | 2 | 2.2% |

| | Kept by Fastino (65) | Missed (27) | Kept minus missed (95% interval) | All test pairs |
|---|---|---|---|---|
| overlap | 0.0524 | 0.0462 | +0.0063 (-0.0040 to +0.0166) | 0.0430 |
| cosine | 0.6336 | 0.6398 | -0.0061 (-0.0185 to +0.0073) | 0.6083 |

Cited domain, kept: {'Physical Sciences': 60, 'Life Sciences': 5}; missed: {'Physical Sciences': 20, 'Health Sciences': 5, 'Life Sciences': 2}.

## V3p gate positives (115)

| Ranker | Positives in its top 20% | Share |
|---|---|---|
| Fastino | 74 | 64.3% |
| word overlap (high first) | 32 | 27.8% |
| cosine (high first) | 38 | 33.0% |
| word overlap (low first) | 17 | 14.8% |
| cosine (low first) | 4 | 3.5% |

| | Kept by Fastino (74) | Missed (41) | Kept minus missed (95% interval) | All test pairs |
|---|---|---|---|---|
| overlap | 0.0532 | 0.0465 | +0.0067 (-0.0016 to +0.0153) | 0.0430 |
| cosine | 0.6321 | 0.6366 | -0.0045 (-0.0160 to +0.0077) | 0.6083 |

Cited domain, kept: {'Physical Sciences': 68, 'Health Sciences': 1, 'Life Sciences': 5}; missed: {'Physical Sciences': 29, 'Health Sciences': 9, 'Life Sciences': 3}.

Rank correlation of Fastino's score with nearness over all test pairs: overlap +0.057, cosine +0.052.
