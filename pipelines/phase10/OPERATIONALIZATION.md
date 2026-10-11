# Phase 10: how each hypothesis is measured and judged

Written 10/05/2026, before any patent result existed. The hypotheses are
`results/phase-10-analogy/HYPOTHESES.md`. `scripts/p10-verdicts.py` applies the rules below to the metrics and writes
`results/phase-10-analogy/REPORT.md`. Every 95% interval is a bootstrap over queries (or over specific words, for H1).

The one number seen before this was written: the frequency-only part of H1 on all of WordNet (nouns 0.70, verbs 0.64)
appeared in a smoke run. It uses no patent data, and H1's verdict uses frequency and spread jointly.

## Data

- **Borrowing:** an applicant citation (not examiner) from a utility patent to an earlier one that shares none of its
  CPC sections A-H (Y, a cross-sectional tag, is ignored).
- **Queries:** 1,000 citing patents granted 2018-2019 (val) and 1,000 granted 2021-2024 (test), each with one random
  borrowing. The cited patent was granted before 2021 in both splits.
- **Candidates:** the cited patent plus 999 patents from its own first CPC section, all granted before the query and
  before 2021, none sharing a section with the query, none cited by it in any category. Same-section candidates mean
  that knowing which field gets borrowed from does not help.
- **Spread sample:** 25,000 abstracts per section, disjoint from queries, cited patents and candidates.
- **Audit gate** (`p10-data.py audit`): any violation stops the pipeline.

## Measures

- **Recovery:** share of queries whose cited patent ranks in the top 10 of 1,000 (chance 1%); MRR also reported.
- **Generality** of a (lemma, part of speech): z(Zipf frequency in general English) + z(entropy of its use across
  the 8 sections), per part of speech, over words with at least 20 patent occurrences.
- **Role replacement, level k** (k = 1-4, thresholds at the 40th/60th/80th/90th generality percentile): a noun (or
  verb) below the threshold becomes the WordNet hypernym (up to 4 steps, any sense) at or above it that Qwen3-1.7B-Base
  finds most probable at that point in the text. If none qualifies, the word stays. Level 0 = as written. Patent
  boilerplate ("said", "comprise", "plurality", ...) is never counted or replaced (`STOP` in `p10-words.py`).
- **Encoder:** Qwen3-Embedding-0.6B. Queries carry a retrieval instruction and candidates none.
- **Leakage:** 5-fold accuracy of a linear classifier guessing a candidate's section from its vector (chance 12.5%).

## Verdict rules

| | Supported if | Falsified if | Otherwise |
|---|---|---|---|
| **H1** | for nouns and verbs both, the lower bound of the share of WordNet hypernym steps where the general word is higher in frequency **and** spread is > 0.70 | either part of speech has a share ≤ 0.60 | inconclusive. Under 200 steps with spread for a part of speech: insufficient data |
| **H2** | test, TF-IDF lemma bags: verbs-only minus nouns-only recovery, lower bound > 0 | the difference is ≤ 0 | inconclusive |
| **H3** | held-out sections: map hits minus frequency-baseline hits, lower bound > 0 | the difference is ≤ 0 | inconclusive |
| **H4** | the best cell other than (0,0), chosen on val by top-10 then MRR, beats (0,0) on test (lower bound of the paired difference > 0) **and** leaks less | the interval of the gain includes 0, or the gain is negative | (the leakage part is reported separately) |
| **H5** | at the val-best verb level, the val-best noun level is below the maximum **and** on test it beats the maximum noun level (lower bound > 0) | on test, the maximum noun level is at least as good as every lower level | inconclusive |
| **H6** | (a) at the val-best noun level, the val-best verb level > 0 beats verb level 0 on test (lower bound > 0), **and** (b) in the best cell, verbs were moved less than nouns: fewer hypernym steps and smaller rises in frequency and spread | (a) the gain is ≤ 0, or (b) the val-best verb level ≥ the val-best noun level | inconclusive. (c), which verbs need it most, is described, not tested: a per-verb test was not designed in advance |

Selection always happens on val and judgment on test, so no setting is chosen on the numbers it is judged by.

Added 10/05/2026, still before any real result: with fewer than 500 test queries, H2 and H4-H6 get "insufficient data" rather than a verdict (the dry run "falsified" them on 2 queries).

## Known limits (in addition to HYPOTHESES.md)

- The substitute is scored from the words before it only (one forward pass per abstract), and only its first token.
- Replacements use base forms ("rotates" becomes "move", not "moves").
- The test split has a single seed and a single encoder; a second encoder is a follow-up, not part of this verdict.
