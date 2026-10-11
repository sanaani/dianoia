# Phase 14 step 3: does the patent-trained far search carry over to management papers?

255 strict far pairs (one per citing article), each citing abstract asked against its true source and 999 other works from the same field. Top-10 = the true source is among the 10 best of 1,000.

| Encoder | Top-10 | Mean reciprocal rank |
|---|---|---|
| off the shelf | 20.0% | 0.089 |
| far-trained | 36.5% | 0.191 |
| near-trained | 33.3% | 0.152 |

**O1 (far-trained beats off the shelf): PASS.** Far-trained minus off the shelf: +16.5 points (95% interval +10.2 to +22.4, 255 pairs).

- Near-trained minus off the shelf (reported): +13.3 points (95% interval +7.4 to +18.8, 255 pairs).

Reported, not tested: top-10 by Sonnet v4p's label and by cited field.

| Group | n | off the shelf | far-trained | near-trained |
|---|---|---|---|---|
| label: G | 169 | 18.9% | 38.5% | 37.3% |
| label: C | 86 | 22.1% | 32.6% | 25.6% |
| field: Computer Science | 116 | 19.8% | 32.8% | 31.9% |
| field: Mathematics | 26 | 19.2% | 46.2% | 42.3% |
| field: Engineering | 23 | 13.0% | 34.8% | 26.1% |
| field: Environmental Science | 21 | 14.3% | 52.4% | 38.1% |
| field: Physics and Astronomy | 20 | 10.0% | 20.0% | 5.0% |
| field: Neuroscience | 19 | 15.8% | 21.1% | 26.3% |
| field: Biochemistry, Genetics and Molecular Biology | 8 | 25.0% | 37.5% | 50.0% |
| field: Medicine | 8 | 62.5% | 62.5% | 62.5% |
| field: Agricultural and Biological Sciences | 7 | 28.6% | 71.4% | 57.1% |
| field: Health Professions | 4 | 50.0% | 50.0% | 50.0% |
| field: Immunology and Microbiology | 1 | 0.0% | 0.0% | 0.0% |
| field: Pharmacology, Toxicology and Pharmaceutics | 1 | 100.0% | 100.0% | 100.0% |
| field: Materials Science | 1 | 0.0% | 0.0% | 100.0% |

## What it means (Claude, 10/09/2026)

- **O1 passes: patent training carries over to management papers.** Phase 12's far-trained encoder, trained only on
  patents, nearly doubles top-10 on business papers' distant sources (20.0% to 36.5%). That is the same size of gain
  phase 12 found on patents.
- **On real borrowings (Sonnet's C, 86 pairs) the far-trained encoder is the one that helps:** 32.6% against 22.1%
  off the shelf and 25.6% near-trained. On method citations (G, 169 pairs) far and near do about equally well (38.5%
  and 37.3%). These splits are reported, not tested, and 86 pairs is small.
- **The test held 255 pairs, not 300:** 315 strict pairs came from only 255 citing articles, and the rule fixed before
  the test allowed one pair per article.
- Two GPU runs this phase (step 1's cosine pass and this test), each under an hour on one g5.xlarge, instances
  terminated and every download checksummed.

**Next, as HYPOTHESES.md set it:** scale on patents and papers together. That is a new phase with its own hypotheses,
written and approved before it runs.
