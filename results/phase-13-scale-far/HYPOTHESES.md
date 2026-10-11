# Phase 13: does the far-trained search hold up on short questions, and does more varied data help? (10/07/2026)

**Approved by Shaun on 10/07/2026.** Fixed before any phase 13 test runs. It builds on phase 12
(`results/phase-12-far-analogy/REPORT.md`): training Qwen3-Embedding-0.6B on 1,676 far patent borrowings raised far
test top-10 from 52.6% to 77.7%. About 4.5 points of that are specific to far pairs; the rest came from training on
cross-field citations at all.

Shaun (10/07/2026):
- In production the query is a short question ("How do I motivate somebody?", maybe two sentences), not an abstract.
- Training should also encode the breadth of relationships, not only move one average.
- Pairs from rare field combinations should count more in training ("award more points"), so that hundreds of
  thousands of common-combination pairs, such as medical devices citing electronics, do not teach the model the
  field instead of the type of relationship.

Three steps, in order. Each step decides whether the next one is worth running.

## Step 1: short questions (SKIPPED, Shaun, 10/08/2026)

**Not run.** The laptop could not run the encoder, so step 1 would have cost about $1 of GPU time. A Background
opening states the problem, while a far source is linked by the solution (the pill dispenser's Background talks about
missed doses, not coins). So every encoder was expected to score low, and abstract training was unlikely to carry over.
Shaun chose to skip it and go straight to the branch it would most likely have picked: **step 3 trains on Background
→ cited-abstract pairs** and is scored on Background queries, with off the shelf as the baseline. The query extraction
and the cut rule below are kept and used there; the rule was fixed on 10/07/2026, before any short query was scored.
What is lost: no clean answer to "does abstract training carry over to short queries?".

The original step, for the record:

- **Question:** does the far-trained search still find far sources when the query is short and states only the
  problem, as a product query will?
- **Queries, all written by the inventors and not rewritten:** for each phase 12 far test question,
  - (a) the patent title;
  - (b) the opening of the Background section, at most 2 sentences. It comes from PatentsView's description text
    (Zenodo record 15062212, CC BY 4.0). Text after the first sentence that names the invention's own solution is cut
    by a fixed rule written before scoring.
- **Same everything else:** the 1,000 candidates and the true sources are phase 12's. The encoders are phase 12's
  saved adapters (off the shelf, far-trained, near-trained). Scored on the laptop CPU, which is slow but costs nothing.
- **Hypothesis S1:** on Background queries, the far-trained search beats off the shelf on far test top-10 (paired
  bootstrap, lower bound > 0).
- **Reported, not tested:** the same comparison on titles; how much each encoder loses going from abstract to
  Background to title.
- **What changes next:**
  - If far training holds or helps on Background queries, step 3 keeps abstract pairs.
  - If it falls apart, step 3 trains on Background → cited abstract pairs instead. That is real text, closer to
    production.

## Step 2: how much hand filtering does scale need? (free, laptop)

The phase 12 pairs passed a long filter. Measured on the 2013-2020 citing years:

| Stage | Pairs |
|---|---|
| Cross-field applicant citations (corpus) | 172,621 |
| Survivors of Fastino stages 1-2 | 26,020 (15%) |
| Kept by Sonnet v3 + v4 | 5,862 after the shared-patent rule |
| Far | about 1 in 4 of those |

- **Risk:** the far filter may favor junk. An unrelated citation also shares few words, so on unfiltered citations
  "far" could mean noise.
- **Check:**
  1. Apply phase 12's far rule (the same medians) to the unfiltered corpus.
  2. Hand-check 50 far pairs blind (RUBRIC.md, 10 earlier pairs re-judged, agreement ≥ 9/10).
  3. Hand-check 50 far pairs that passed Fastino stages 1-2 but not Sonnet.
- **Decision rule, fixed now:**
  - If unfiltered far pairs (no Fastino, no Sonnet) are at least 70% real borrowings (C or G), scale on them with no
    labeling.
  - Else, if Fastino-only far pairs (Fastino stages 1-2, no Sonnet) are at least 70% real, scale on those. Fastino needs a GPU run for the years it
    has not scored yet; that is priced before it runs.
  - Else, the Sonnet stage stays. It runs through the Claude subscription, at no AWS cost, but slowly.

### Step 2b: what Fastino throws away (added by Shaun, 10/08/2026, after step 2, before this reading)

Step 2 read pairs Fastino kept, and unfiltered pairs, but not the far pairs Fastino rejected. Taken at face value the
two readings imply that about half of all real far pairs are among the rejects, which matters because real far pairs
are scarce. Checked the same way: 50 unread far pairs that Fastino stages 1-2 rejected (25 from combinations weighted
5x or more over the whole far set, 25 others) plus 10 earlier gold pairs, read blind against RUBRIC.md; the reading
counts only with at least 9/10 gold agreement. Reported, not a decision rule: the share real among the rejects, and
Fastino's estimated recall, i.e. real kept ÷ (real kept + real rejected), each share scaled by its pool's size. It
informs Shaun's choice of step 3's size.

### Step 2c: stage 2's rejects, and Sonnet on the rest of Fastino's keeps (Shaun, 10/08/2026, before this reading)

Fastino's two stages each score every pair, and a pair survives only if both say yes; every far pair stage 2 accepts,
stage 1 accepted too. Of the far pairs, stage 1 rejects 91,636 and stage 2 rejects 17,192 more that stage 1 accepted.
Step 2b read only 5 of the latter (1 real). Checked the same way as 2b: 50 unread far pairs that stage 1 accepted and
stage 2 rejected (25 rare-combination, 25 other) plus 10 gold pairs, blind, counted only at 9/10 or better. Reported,
not a decision rule: their share real. If it is about 20% or more, Sonnet reads them next.

Sonnet already labeled 8,569 of the 10,131 far pairs Fastino kept (phase 10, prompt v3). The other 1,562 go through
the same v3 prompt, then the v4 prompt (RUBRIC.md) for v3 C with borrow ≥ 4, phase 11's rule, unchanged. Step 2's
Fastino reading, crossed with the v3 labels (that reading was not counted, 7/10): 9 of 11 real at v3 C and borrow ≥ 4,
3 of 8 at v3 C below 4, 5 of 15 at v3 B, 0 of 6 at v3 A.

### Step 2d: a scoring filter to recover real far pairs from Fastino's rejects (Shaun, 10/08/2026, before any run)

After 2c, the strict rule (Sonnet v3 C with borrow ≥ 4, then v4 C or G) leaves 1,805 far pairs from Fastino's
10,131 keeps, while about 8,000 real far pairs sit among its 108,828 rejects at 7-12%. Fastino answers yes or no,
so it cannot be loosened.

- **Labels:** Sonnet (v3, then v4 on v3 C with borrow ≥ 4, as in phase 11) on a uniform random sample (seed 13) of
  3,000 stage 1 rejects and 2,000 stage 2 rejects, excluding every pair already read. Positive = passes the strict
  rule. Training also uses the 10,120 Sonnet-labeled keeps.
- **Model:** BAAI/bge-reranker-base (MIT), a cross-encoder reading both abstracts, fine-tuned on one AWS GPU to give
  a 0-1 score. Holdout, never trained on: 30% of the reject sample (1,500), split at random with seed 13. Epochs
  chosen on a 10% dev split of the training part.
- **F1 (pass/fail):** on the 1,500 held-out rejects, the top-scored 20% hold at least 70% of the strict positives
  among them. Reported with a bootstrap interval; the positives may number only about 50.
- **F2 (pass/fail, only if F1 passes):** Sonnet reads the top-scored unread rejects (the number set from F1's curve,
  and its run time stated, before it runs). A blind hand read of 50 that pass the strict rule (plus 10 gold pairs,
  counted only at 9/10 or better) is at least 70% real (C or G).
- **Cost:** about $0.75 of GPU (training plus scoring all 118,959 far pairs). Sonnet runs on the subscription:
  about 3 hours for the sample, more for F2's reading.
- **Expected:** about 2,000-3,000 more strict pairs, for roughly 4,000 in all.

## Step 3: more varied data, with rare combinations weighted (about $2-4 of AWS)

- **Question:** does more data, spread across field combinations and with rare combinations counting more, improve
  far search, and across how many field combinations?
- **Field combination:** the CPC class pair, citing class to cited class (for example G01 → A61). Revised by Shaun on
  10/08/2026, after step 2's far filter and before any test: at the subclass level (G01N → A61K) the 118,959 far pairs
  split into 24,638 combinations, the typical combination held 2 pairs, and no pair reached 5×, so weighting did
  nothing and the 5× half of the step 2 checks could not be drawn.
- **No cap per combination** (Shaun, 10/07/2026: a fixed cap such as 50 is arbitrary). Every eligible pair is kept;
  weighting alone handles the imbalance. When a size is smaller than the eligible set, the sample is uniform at random
  (seed 0).
- **Weighting:** each pair's share of the training loss is multiplied by √(median combination count ÷ its
  combination's count), clipped to between 1× and 10×. The median is taken over pairs, i.e. the size of the
  combination the typical pair sits in (130 pairs; revised 10/08/2026 with the class level). This gives 59,356 pairs
  above 1×, 6,942 at 5× or more and 1,315 at 10×. Rare combinations pull harder; no few pairs dominate.
  Hand-check depth goes where the weight is: the step 2 checks are drawn half from combinations weighted 5× or more.
- **Sizes:** 5k and 20k far pairs, nested, plus phase 12's 1,676 as the start of the learning curve. The query side
  is the citing patent's Background opening (Shaun, 10/08/2026, in place of step 1), cut by step 1's rule; a pair
  whose citing patent has no Background opening is dropped. Which pairs are far is still decided on the two
  abstracts, by phase 12's rule. The document side stays the cited abstract. Hard negatives, batch, learning rate and
  epoch choice are as in phase 12, with epochs chosen on dev Background queries. Seed 0 for verdicts.
- **No held-out field combinations** (Shaun, 10/07/2026: "I'd rather just train on everything because I don't want to
  pay for training twice"). Every combination is trained on, so the model trained here is the one meant for use. Only
  the test questions themselves (and any patent sharing their text) stay out of training, permanently, so this and
  later models can be scored on them. The cost: this step cannot show whether the search finds sources in field pairs
  it never saw. The per-section-pair scores below show only where it is strong and weak.
- **Hypotheses (paired bootstrap over test queries, 2,000 resamples, 95% intervals). All are asked with Background
  queries, over the test questions that have a Background opening (255 of the 291 far; revised 10/08/2026 with the
  query side, before any test ran):**
  - **B1.** The 20k model beats off the shelf on far test top-10 (lower bound > 0).
  - **B3 (guardrail).** On near test questions, the 20k model minus off the shelf has a lower bound above -3 points.
  - Reported, not tested: phase 12's far model (trained on abstracts) on the same Background queries, and every
    model on abstract queries. (B2, the held-out test, was dropped with the holdout.)
- **Scoring for breadth, reported with every result:**
  - Top-10 per section pair.
  - A headline in which every section pair counts equally (macro average) next to the plain average, so a gain
    confined to busy fields cannot pass as general.
- **Cost:** phase 12's measured speed is about 6,900 pairs per GPU-hour with 7 negatives.
  - 20k pairs for 2 epochs is about 6 hours, roughly $6 on demand. That is over the $5.40 left.
  - So: 3 negatives instead of 7 (about half the cost); spot capacity with a checkpoint after every epoch; and scoring
    only the chosen model on test. Estimate $2-4. The budget gate in the pipeline refuses anything over $4.50.

## Later (not part of this approval)

- **OpenAlex** (CC0): management and organizational papers that cite distant fields (biology, physics, military
  history), with the same far rule and hand check. This is the test that matters for business questions.
- **Patent-to-paper links:** the Reliance on Science dataset is **CC BY-NC 4.0, non-commercial**. It can inform
  research but not a commercial product's training data without a license. OpenAlex's own citations do not have that
  restriction.
- **The full patent set:** about 1 million far pairs, about $65 on spot. Only if step 3's learning curve is still
  rising at 20k.

## Known limits

- Background sections are often written after the invention and can hint at the solution. The fixed cut rule
  reduces this; it cannot remove it.
- Weighting amplifies the noise of the pairs it weights; step 2's checks are aimed there.
- The data is still patents, still engineering. A pass says nothing yet about business questions.
