# Phase 16: do more paper borrowings, labeled by Sonnet, improve the far search on papers? (DRAFT, not approved)

**Drafted 10/10/2026. Nothing below runs until Shaun approves it; every number is fixed at approval.**

Phase 15 found two things:
- **More far data still helps** (step 1, L1 still climbing). Doubling the patent training pairs gained +4.4 points
  pooled: +6.3 on papers and +2.8 on patents.
- **Fastino can't screen** (step 2, S1 failed). A follow-up check (`../phase-15-curve-screen/BIAS.md`) found it ranks
  fairly on nearness. But it misses 29% of real borrowings, including all 5 from health science.

Sonnet, run headless through the subscription, labeled 10,480 pairs in about 30 minutes in phase 14 with no per-pair
bill. So this phase drops the screen and has Sonnet read every far pair.

Training so far is patents only, while papers are the weak side: 36.5% top-10 on papers vs 77.7% on patents. This
phase adds paper pairs, the kind of data the paper test measures.

## Step 1: more candidates from OpenAlex (laptop, $0)

- **Source:** phase 14's filter, unchanged. English management articles (field 14), 2015 to 2024, with an abstract and
  more than 5 references: 504,172 articles on 10/08/2026.
- **Sample:** 17 new seeds (17 to 33) at 10,000 articles each, about 170,000. The sample drops duplicates and every
  article phase 14 drew.
- **Fetching:** cited works outside the Social Sciences are fetched with the OpenAlex key, exactly as in phase 14.
- **Expected yield:** phase 14 got 15,536 far pairs from 29,376 articles (0.53 per article), so about 90,000 far
  pairs.
- **Pairs and the far rule:** phase 14's code and phase 12's medians, unchanged: overlap below 0.09 and cosine below
  0.6746, with bge-small-en-v1.5. The cosine pass runs on the laptop CPU (an estimated 2 to 3 hours, under memcap).
  Phase 14 ran it on a GPU only because its budget allowed.
- **Leakage guard:** a new pair is dropped if its citing article, or its cited work, is any citing article or true
  source in phase 14's 255 test pairs. The 999 candidates per test query may still appear as cited works; that is
  reported.

## Step 2: Sonnet labels every far pair (subscription, about 5 hours)

- **Labels:** phase 14's v3p prompt on every new far pair, 10 pairs per call, 4 workers, resumable. Pairs that pass
  v3p (C with borrow ≥ 4) then go through v4p. **Strict** means C or G under v4p, the rule phase 11 set and phase 14
  used.
- **Expected yield:** 2.2% strict, so about 2,000 strict pairs, against phase 14's 315.
- **Quality gate Q1 (must pass before any training):**
  - A blind hand read of 50 new strict pairs (seed 16), with phase 14's 10 gold pairs shuffled in.
  - Gold agreement must be at least 9/10, and at least 70% of the 50 must be real (C or G), as in phase 14 step 2.
  - Fail: stop. The labels have drifted, and nothing trains on them.
- **Risk:** Claude subscription rate limits may spread the run over a day or more. That costs time, not money.

## Step 3: train and test (one AWS GPU, about $3 to $4)

- **Arms:**
  - **P (existing):** phase 12's far-s0-e2, trained on the 1,676 patent far pairs. Its scores exist and are not
    re-run.
  - **P+O (new):** the same recipe (Qwen3-Embedding-0.6B with LoRA, phase 12's settings, seed 0). It trains on the
    1,676 patent pairs plus every new strict paper pair and phase 14's strict pairs outside its test set.
  - **Hard negatives for the paper pairs:** other cited works from the same field, drawn as phase 14 drew candidates.
- **Epoch (1, 2 or 3):** chosen on pooled dev (phase 12's patent dev queries plus a paper dev set). The paper dev set
  is 10% of new citing articles, held out by article (seed 16).
- **Test:** phase 14's 255 strict paper test pairs and phase 12's 291 far patent test queries, unchanged. Each query
  ranks its true source among 1,000 candidates.
- **O2 (deciding):** P+O minus P on top-10 over the 255 paper test pairs, using a paired bootstrap with 2,000
  resamples and a 95% interval.
  - **Pass** if the lower bound is above 0. More Sonnet-labeled papers improve the far search on papers.
  - **Fail** otherwise.
- **G1 (guard):** P+O minus P on the 291 patent queries. Its lower bound must be above −3 points, meaning the
  patent side did not degrade. If O2 passes but G1 fails, the gain came at the patents' expense, and it is reported
  as such.
- **Expected effect:** +3 to +8 points on papers. With 255 queries the interval is about ±6 points, so the test
  reliably detects about +6 points; a true effect of +3 may well not show.
- **Reported, not tested:**
  - top-10 on C pairs vs G pairs;
  - top-10 by cited field, including health science;
  - the number of training pairs that are methods borrowings (G);
  - how many test candidates appear in training.
- **Known weakness:** phase 14 found that most strict pairs are methods borrowings (33 of 43 real ones were G:
  statistics, regression, clustering). Training on them may teach "find the method" rather than conceptual far
  analogies. The C/G split shows whether the gain is only on G.

## Not in this phase

- Retraining Fastino. Sonnet reads everything.
- A papers-only arm.
- Multiple seeds. One seed per arm, with phase 12's 2-point spread across seeds stated beside the result.

## How it runs

- **Pipeline:** `pipelines/phase16` (Snakemake via run.sh, memcap on every laptop step).
  - Steps 1 and 2 reuse phase 14's fetch, pairs, far and Sonnet code, with phase 16's seeds and the leakage guard.
  - Step 3 reuses `remote-p15.py`'s curve path, extended for paper hard negatives and a pooled dev set.
  - A laptop smoke run happens first; then one AWS g5.xlarge, on demand.
- **AWS:**
  - An estimated 3 to 3.5 hours: about 2.5 hours training on about 3,700 pairs over 3 epochs (from phase 15's 838
    pairs, 3 epochs in 39 minutes), plus dev and test scoring.
  - Self-terminate cap 4 hours, at most $4.05. Every download is checksummed before termination.
- **Budget:** the $95 AWS budget is used up after phase 15 (about $3.00 spent there). **Approval needs new AWS budget
  of at least $4.05.**
