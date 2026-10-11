# Phase 13 step 2: how much hand filtering does scale need? (10/08/2026)

**Decision (rule fixed 10/07/2026): the Sonnet stage stays.** Neither cheaper source comes close to 70% real borrowings.

Phase 12's far rule (word overlap < 0.09 and bge-small cosine < 0.6746, both phase 12's medians), unchanged, on every
cross-field applicant citation of 2013-2024 that does not touch a test question:

| Stage | Pairs |
|---|---|
| Corpus pairs | 207,959 (5,703 + 2,093 dropped as touching test questions) |
| Below the overlap median | 150,003 |
| Far (below both) | 118,959 (57%; among Sonnet-kept pairs it was about 1 in 4) |
| Far and kept by Fastino stages 1-2 | 10,131 |

Blind reading by Claude against `results/phase-10-analogy/RUBRIC.md`: 50 new far pairs per source (25 from
combinations weighted 5x or more over the whole far set, 25 others) plus 10 earlier gold pairs, shuffled.

| Source | Real (C or G) | Rare-combination half | Other half | Gold agreement | Counted |
|---|---|---|---|---|---|
| Unfiltered far pairs | 8% (C 0, G 4, A 2, B 44) | 4% | 12% | 8/10 | no |
| Fastino-only far pairs | 44% (C 1, G 21, A 10, B 18) | 44% | 44% | 7/10 | no |

- Both readings fall under the 9/10 gold agreement bar, so neither is counted, and per the rule they are not relabeled.
  An uncounted reading cannot clear the 70% bar, so the rule falls through to the Sonnet stage either way. Taken at
  face value the two shares are also far below 70%.
- The risk named in the hypotheses is confirmed: on unfiltered citations, "far" mostly means "unrelated". 44 of 50
  unfiltered far pairs were B (no borrowing).
- Weighting would amplify the rare-combination half, which is the noisiest among unfiltered pairs (4% real).

Files: `step2.json` (the decision script's output), `review-*-labels.txt` (the blind labels). The pairs and keys are
in `work/p13`.

## Step 2b: what Fastino throws away (10/08/2026)

50 unread far pairs that Fastino stages 1-2 rejected, plus 10 gold pairs, read blind. **Counted: 9/10 gold agreement.**

| | Pool | Real (C or G) in the reading | Estimated real pairs |
|---|---|---|---|
| Fastino kept | 10,131 | 44% (reading not counted, 7/10) | about 4,500 |
| Fastino rejected | 108,828 | 8% (C 1, G 3, A 0, B 46; rare half 12%, other half 4%) | about 8,700 |

- Estimated Fastino recall: **about 34%**. Roughly two of every three real far borrowings are among its rejects.
- The rejects are still 92% junk, so finding the missing ones means reading about 12 rejects per real pair.
- Uncertain: 8% is 4 pairs of 50 (95% interval about 2% to 19%, so 2,400 to 21,000 real rejects), and the kept share
  comes from an uncounted reading.

Files: `step2b.json`, `review-rejected-labels.txt`.

## Step 2c: stage 2's rejects (10/08/2026)

50 unread far pairs that Fastino stage 1 accepted and stage 2 rejected, plus 10 gold pairs, read blind. **Counted: 9/10
gold agreement.** Real (C or G): **12%** (C 2, G 4, A 9, B 35; rare-combination half 16%, other half 8%), so about
2,100 of the 17,192. That is under the 20% bar set beforehand, so Sonnet does not read them all: about 8 reads per real
pair.

Where the real far borrowings sit, all estimates rough:

| Fastino outcome | Far pairs | Real | Estimated real pairs |
|---|---|---|---|
| Kept by both stages | 10,131 | 44% (not counted) | ~4,500 |
| Stage 1 yes, stage 2 no | 17,192 | 12% | ~2,100 |
| Stage 1 no | 91,636 | ~7% (3 of 45, from 2b) | ~6,100 |

Fastino keeps roughly a third of the real far pairs. The rejects, though, are 88-93% junk in both stages, so reading
them is costly; recovering them needs a filter that ranks pairs rather than Fastino's yes/no.

Files: `step2c.json`, `review-s2reject-labels.txt`.
