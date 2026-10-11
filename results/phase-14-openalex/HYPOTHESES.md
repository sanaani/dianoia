# Phase 14: does far search find borrowings for management papers? (approved 10/08/2026)

**Approved by Shaun on 10/08/2026, before anything below ran.** Phase 13 stopped after step 2 (Shaun, 10/08/2026: "Move on"). Its
step 2d filter failed, so patent data stays at about 1,800 strict far pairs, about 10% more than phase 12's 1,676.
Phase 13's step 3 is set aside, and this phase asks the question that matters for business questions.

Source: OpenAlex (CC0). Probe on 10/08/2026, before any test:
- 616,623 journal articles from 2015 to 2024 whose main field is Business, Management and Accounting, with an
  abstract and more than 5 references.
- In a sample of 60 of them (2,180 references), 6% of the references point outside the Social Sciences, and 4.5%
  point outside it and have an abstract. The largest groups were Engineering, Mathematics, Medicine, Biology,
  Computer Science and Environmental Science.
- At that rate the full set holds about 1 million cross-domain citations. Many will be borrowed methods (statistics,
  optimization), which are this data's version of "generic technology".

## Step 1: build the pairs (free, laptop)

- Draw 30,000 of those articles at random (seed 14), English only. Each pair is one of the article's references whose
  main topic is outside the Social Sciences and that has an abstract.
- Apply phase 12's far rule (the same medians, overlap < 0.09 and bge-small cosine < 0.6746). Report how many pairs
  pass.
- The build saves progress per page and resumes (API calls, 10 per second at most).

## Step 2: how many are real borrowings?

- **Rubric:** RUBRIC.md with its labels reworded for papers, decided before any reading. **C** means the cited field's
  way of handling its own problem is carried into a management problem. **G** means a general method used as is
  (a statistical test, a standard algorithm). **A** means the same topic. **B** means unrelated or boilerplate.
- **Gold pairs:** there are none for papers yet. Claude labels 10 far pairs first, then re-reads them blind a day
  later. The step 2 reading counts only at 9/10 or better, as before.
- **Reading:** a blind hand read of 50 far pairs, plus Sonnet (v3 then v4, the prompts reworded the same way) on
  5,000 far pairs. Then a blind hand read of 50 pairs that pass the strict rule, which counts at 70% real (C or G) or
  better.
- Reported, not a decision rule: the share that is C and not G. On patents, telling C from G was not reliable.

## Step 3: does the patent-trained search carry over? (about $1 of GPU)

- **Test set:** 300 strict far pairs, with each management abstract as the query and 1,000 candidate abstracts from
  the same distant fields, as in phase 12.
- **Hypothesis O1:** phase 12's far-trained encoder beats off the shelf on top-10 (paired bootstrap, 2,000 resamples,
  lower bound > 0).
- Reported, not tested: phase 12's near-trained encoder; top-10 split by the cited field.
- **What changes next:**
  - If O1 passes, patent training carries over to business text. Scale on patents and papers together.
  - If it fails, train on OpenAlex pairs (step 4, written and approved before it runs).

## Known limits

- An abstract is not a short "how" question. Step 1 of phase 13, which would have tested that, was skipped.
- A reference outside the Social Sciences is not the same as a far one. The far rule and the hand read decide that.
- OpenAlex assigns topics by a model, so some of them are wrong. Pairs with a wrong topic are dropped in the reading.

## How it runs (fixed 10/08/2026, before any result)

Everything is `pipelines/phase14` (Snakemake, through `run.sh`, memcap on every step), so the sample, the far rule, the
blinding and every pass/fail rule are code, not judgment. The only judgment left is the hand reading and Sonnet's
labels, and the 10 gold pairs check the hand reading. Numbers live in `pipelines/phase14/config.yaml`.

- **OpenAlex's budget:** without a key, 1,000 requests a day. Step 1 needs about 1,500, so it saves after every
  request and stops cleanly when the day's budget runs out; the next run picks up. A free key
  (`OPENALEX_API_KEY`) is used when set.
- **The sample:** OpenAlex draws at most 10,000 per seeded sample, so the 30,000 are three draws (seeds 14, 15, 16)
  from the English articles (504,172 on 10/08/2026), duplicates dropped.
- **The references:** asked in batches of 100 articles ("cited by any of these", outside the Social Sciences, with an
  abstract), then matched back to each article's reference list.
- **Text:** each abstract rebuilt from OpenAlex's word index, cut to 180 words as on patents. Pairs where either
  abstract has fewer than 30 words are dropped (OpenAlex holds some boilerplate abstracts).
- **The far rule** uses phase 12's patent pool as its word-weight basis, unchanged, so its medians mean the same thing.
- **Gold pairs:** 10 far pairs drawn by seed. The hand-reading sheets refuse to build until Claude's gold labels are
  at least 20 hours old.
- **Strict pairs for the hand read:** drawn at random from Sonnet's strict pairs, never from the 50 already read.
- Step 3's GPU step is added once step 2 passes.

## Step 3 details (fixed 10/09/2026, after step 2 passed and before any step 3 result)

- **Strict pairs:** step 2 found 109 strict pairs in 4,979 labeled. Sonnet v3p, then v4p, labels the other 10,547
  far pairs with the same prompts and rule; step 2's files are left as they were.
- **Test set:** at most one strict pair per citing article (drawn at random, seed 14), then 300 drawn at random
  (seed 14); if fewer than 300 exist, all of them, and the count is reported.
- **Candidates:** for each test pair, 999 other cited works from the same OpenAlex field as the true source, drawn at
  random (seed 14) from every cited work step 1 fetched (outside the Social Sciences, English, 30 words or more),
  never one the same citing article cites. A field with fewer than 999 such works is filled from the same domain. All
  texts, query and candidates alike, are cut to 180 words, so length tells nothing.
- **Encoders:** Qwen3-Embedding-0.6B off the shelf, and phase 12's far-trained and near-trained adapters, all with
  phase 12's query instruction unchanged, since the adapters were trained with it.
- **O1** as written: far-trained minus off the shelf in top-10, paired bootstrap over the test pairs (2,000
  resamples); passes if the 95% lower bound is above 0.
- **Reported, not tested:** near-trained; top-10 by cited field; and top-10 separately for pairs Sonnet v4p calls C
  and G, since step 2's reading found three in four strict pairs are general methods.
