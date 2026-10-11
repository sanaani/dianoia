# Phase 15 step 1: is the far search still improving with more data?

Top-10 (true source among the 10 best of 1,000) for the encoder trained on nested shares of phase 12's 1,676 far patent pairs. 0 is the encoder off the shelf; 1,676 is phase 12's far-s0-e2.

| Training pairs | Pooled (546) | Patent test (291) | Paper test (255) |
|---|---|---|---|
| 0 | 37.4% | 52.6% | 20.0% |
| 419 | 55.7% | 75.6% | 32.9% |
| 838 | 54.0% | 74.9% | 30.2% |
| 1,676 | 58.4% | 77.7% | 36.5% |

**L1: STILL CLIMBING.** Full minus half (pooled): +4.4 points (95% interval +2.4 to +6.6, 546 queries). Rule: still climbing if the lower bound is above 0, flat if the upper bound is below +3 points.

- Patent test only: +2.8 points (95% interval +0.3 to +5.5, 291 queries).
- Paper test only: +6.3 points (95% interval +3.1 to +9.8, 255 queries).
- Half minus quarter (pooled, reported): -1.7 points (95% interval -3.5 to +0.2, 546 queries).
- Phase 12's three seeds at full size, patent test: 77.7%, 76.6%, 78.7% (the subsets use one seed).
- Epochs chosen on patent dev: {'419': 3, '838': 2}.
