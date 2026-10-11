# Phase 15: is more far data worth it, and can Fastino screen for Sonnet? (approved 10/09/2026)

**Approved by Shaun on 10/09/2026 ("Execute steps 1 and 2"), before anything below ran.** Phase 14 passed: phase 12's
patent-trained far encoder carries over to management papers. Scaling on all 504,172 OpenAlex management articles
would cost about 270,000 Sonnet reads for roughly 5,000 strict pairs. Two cheap questions come first.

## Step 1: is the far search still improving with more data? (about $1.30 of GPU)

- **Question:** phase 12 trained on 1,676 far pairs. Would more of the same kind of data still help?
- **Test:** train the same encoder on nested random subsets of phase 12's 1,676 far training pairs: 419 (a quarter)
  and 838 (a half), drawn with seed 15. Each subset is trained exactly as phase 12 trained far-s0: Qwen3-Embedding-0.6B
  with LoRA, the same hard negatives, batch, learning rate and seed 0. The run lasts 1, 2 or 3 epochs, chosen on phase
  12's patent dev queries (top-10, then MRR).
- **Full size:** phase 12's far-s0-e2 itself, trained on all 1,676 pairs. Its scores already exist and are not re-run.
- **Test queries, pooled:** phase 12's 291 far patent test queries and phase 14's 255 strict paper pairs, for 546 in
  all. Each query ranks its true source among 1,000 candidates, scored exactly as in those phases. No subset model
  trains on any of them.
- **L1 (deciding):** full minus half on top-10 over the 546 pooled queries. Paired bootstrap, 2,000 resamples, 95%
  interval.
  - **Still climbing** if the lower bound is above 0. More data still helps, so pricing a scale-up is worthwhile.
  - **Flat** if the upper bound is below +3 points. Doubling the data buys under 3 points, so more of the same data is
    not worth paying for, and the next lever is the method.
  - **Unclear** otherwise. The curve is reported and Shaun decides.
- **Reported, not tested:** quarter vs half; each source on its own; the shape of the curve with the off-the-shelf
  encoder as its zero point.
- **Known weakness:** the subsets use one seed. Phase 12's three seeds at full size spread 2 points on the patent test
  (76.6% to 78.7%), and that spread is stated beside the result.
- **Not in this step:** training on papers. Holding out half of phase 14's 315 strict pairs would leave about 150
  papers to add to 1,676 patents, too few to show anything at this budget.

## Step 2: can Fastino throw away the worst half before Sonnet reads? (about $0.70 of GPU)

- **Question:** in a scaled-up phase, could Fastino cut Sonnet's reading in half without losing real borrowings?
- **Data:** every phase 14 far pair that Sonnet v3p labeled (15,459). A pair is **positive** if it passes the strict
  rule (v3p C with borrow ≥ 4, then v4p C or G): 315 pairs, about 2%. Nothing is relabeled.
- **Split by citing article (seed 15):** 30% of articles are held out for the test and never trained on. Of the rest,
  10% are used for validation (best checkpoint) and 90% for fitting.
- **Model:** fastino/GLiNER2.5-Decide (Apache-2.0), fine-tuned as in phase 9's D1: 3 epochs, batch 8, encoder learning
  rate 1e-5, task learning rate 5e-4, best checkpoint by validation loss.
  - **Input:** "Citing paper:" with the citing abstract's first 150 words, then "Cited work:" with the cited
    abstract's first 150 words.
  - **Labels:** "real borrowing" and "no real borrowing".
  - **Balance, to keep the GPU time down:** every fitting positive is used 4 times, along with 3,000 fitting negatives
    drawn at random (seed 15).
- **Score:** the model's probability of "real borrowing" on every held-out pair.
- **S1 (deciding):** discard the lowest-scored 50% of held-out pairs and measure the share of held-out strict positives
  that survive. The exact (Clopper-Pearson) 95% lower bound on that share must be at least 90%.
  - **Pass:** a scaled-up phase can screen with Fastino and then send only what survives to Sonnet.
  - **Fail:** Sonnet has to read everything, or the next phase needs another screen.
- **Reported, not tested:** the share of positives kept when discarding 10% to 90%. Also: the largest discard whose
  lower bound is still at least 90%; the same curve for the looser v3p gate (C with borrow ≥ 4); and the curve from
  discarding at random.
- **Earlier evidence, against:** phase 13's cross-encoder screen kept only 70% of positives when it discarded the
  bottom half of patent pairs (step 2d). Fastino's yes/no filter kept about 34% of real patent borrowings.
- **Known weakness:** there are only about 95 held-out positives, so even 97% observed recall gives a lower bound
  near 91%. The test can fail from too few positives. If so, that is stated in the result.

## How it runs

- `pipelines/phase15` (Snakemake, run.sh, memcap on every laptop step): `scripts/p15-data.py` builds the subsets and
  the screen data, `scripts/remote-p15.py` runs both steps on one AWS g5.xlarge (laptop smoke run first), and
  `scripts/p15-verdicts.py` writes STEP1.md, STEP2.md and step1.json/step2.json.
- **AWS:** one instance, on demand, capped at 2.5 hours ($2.52). Budget left is estimated at $3 to $3.50 of the $95.
  Every download is checksummed before the instance is terminated.
