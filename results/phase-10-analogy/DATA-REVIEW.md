# Phase 10 data review: are patent "borrowings" analogies? (10/05/2026)

Shaun asked for a subjective read of 20-50 real pairs before spending resources. I (Claude) read 40 random
borrowings, citing years 2013-2017, outside the val and test years. Each is an applicant citation to a patent sharing
no CPC section. The pairs are in `work/p10/review-40.txt`, from `borrowings.parquet` with seed 123.

## Shape

- 146 million citation rows. 5.08 million distant applicant borrowings from 612,000 citing patents. No missing
  abstracts.
- PatentsView marks "cited by applicant" only from **2013** on. Val (2018-19) and test (2021-24) are inside that, but
  there is no earlier history.
- **G-H (physics and electricity) is 27% of all borrowings**, and G or H is in 7 of the top 10 section pairs.
- Borrowings per citing patent: median 2, 90th percentile 15, **maximum 1,917** (citation dumps).
- 6,137 duplicate pairs. Abstracts contain markup (`##STR1##`, `R.sup.1`, `&#8220;`).

## Judgment, pair by pair

**C: a real cross-domain borrowing of a mechanism (8 of 40, 20%)**

| # | Pair |
|---|---|
| 2 | drilling screw → bone implant |
| 8 | spring-arm screw-and-nut for headlamps → screwless electrical cover plate |
| 17 | in-situ soil heating to remove contaminants → heating an oil formation in place |
| 22 | telescoping tensioner for vehicle covers → adjustable air-duct seal |
| 24 | molding filled-mineral sheets → baked-goods mold with venting |
| 26 | filter pump with fluid bag → gluten-test food processor |
| 31 | splayed-fiber heat sink → thermally sealed storage container |
| 35 | ventilated vehicle seat → climate-controlled bed |

**A: same technology, split by classification (22 of 40, 55%)**

1, 3, 4, 5, 6, 7, 9, 12, 14, 15, 16, 18, 19, 23, 27, 29, 30, 32, 33, 36, 37, 38. Examples: LED lamp → LED signal;
casino server → bingo game; fan speed control → fan motor drive; tetracycline formulation → tetracyclic compounds.
Mostly G-H software and electronics, plus drug/chemistry and lighting splits.

**B: no visible connection (10 of 40, 25%)**

10, 11, 13, 20, 21, 25, 28, 34, 39, 40. Examples: sinus surgery → nicotinic acid synthesis; interactive paper →
ball-throwing machine; surgical stapler cartridge → LED semiconductor. Several come from known citation-dump
families.

## What it means

As built, about **1 pair in 5 is the kind of analogy we want**. The recovery test would mostly measure **topic
matching across a classification boundary**. That works against H4-H6 by design: on same-topic pairs, raw text wins.
Training on it would teach "find related prior art", not analogical reasoning.

The two big failure groups have mechanical fixes:

1. **Citation dumps (B):** drop citing patents with more than about 20 applicant citations in total.
2. **Same technology, different section (A):** a section is too coarse a distance. Measure distance at CPC subclass
   level (e.g. "the cited patent's subclasses almost never co-occur with the citing patent's"), and treat G-H, A61-C07
   and similar routine splits as near.

Estimate, not measured: with both filters, the real-analogy share rises to perhaps 40-60%. That share must be
**measured on a fresh 40 pairs** before the pipeline continues.

## Iterating on filters without reading the text (10/05/2026)

Threshold (set by Claude when Shaun asked for "enough clean data", suggesting 99%): **at least 80% real cross-domain
borrowings, and at most 10% same-technology pairs, in a fresh random 50.** 99% cannot be shown with a feasible
review (about 300 pairs with at most one or two bad), and citation data is noisier than that. The two error types
differ: same-technology pairs bias the test toward raw-text matching, while unrelated pairs only add noise.

The filters used only the citation network and CPC codes, never the abstracts. A text-based filter would build a test
that raw text loses by construction. Each round drew 50 fresh pairs (one per citing patent, citing years 2013-2017).

| Round | Filter | Real (C) | Same tech (A) | Unrelated (B) | Citing patents kept, val / test |
|---|---|---|---|---|---|
| 0 | none | 8/40 (20%) | 22 (55%) | 10 (25%) | all |
| 1 | first-subclass lift ≤ 0.5; ≤ 50 applicant citations | 14/50 (28%) | 21 (42%) | 15 (30%) | 28,824 / 46,790 |
| 2 | lift ≤ 0.5 over every subclass pair; ≥ 2 other citing-field patents cite it; ≤ 50 citations | 13/50 (26%) | 18 (36%) | 19 (38%) | 8,937 / 13,662 |

- Round 1, C: 1, 3, 7, 12, 17, 18, 22, 24, 25, 29, 41, 44, 49, 50. A: 2, 4, 6, 10, 13, 16, 20, 21, 23, 26, 27, 31, 33, 35,
  36, 37, 38, 45, 46, 47, 48. B: 5, 8, 9, 11, 14, 15, 19, 28, 30, 32, 34, 39, 40, 42, 43.
- Round 2, C: 1, 6, 9, 14, 15, 22, 27, 28, 33, 36, 39, 42, 47. A: 3, 4, 7, 10, 12, 16, 18, 19, 20, 21, 26, 30, 34, 35, 37,
  41, 48, 50. B: the other 19.

**Finding:** text-free filters plateau at about a quarter real. Rarer field pairings cut same-technology pairs but let in
random citations at the same rate, and the "bridge" signal did not separate real borrowings from noise. Real
borrowings, same-technology pairs and stray citations look alike in classification statistics; only the text tells
them apart. Reaching 80% needs a reader of the text.

## Existing labeled datasets (10/05/2026)

Shaun preferred an already-labeled dataset to LLM labeling. A survey found no public, human-labeled set of 1,000+
far-domain solution-transfer pairs under a usable license:

- **Human-labeled sets are small** or unavailable: SOLVENT (259 pairs), ParallelPARC gold (310, MIT), the BARcode
  expert pairs (1,005; the data link is dead) and the Quirky product analogies (not released).
- **Large sets are LLM-built** (StoryAnalogy, BioSpark) or citation-based (DAPFAM: patent citations, the same noise as
  ours).
- **Word and story analogy sets** (E-KAR, SCAN, AnalogyKB) are not solution transfer.

The best candidate, **CHIMERA** (Sternlicht & Hope, ACL 2026; arXiv CS papers' "inspiration" edges, model-extracted),
was downloaded (`raw/chimera/`, sha256 recorded) and reviewed the same way:

- Only 768 of 9,578 inspiration edges have a non-CS source.
- In 50 random ones, about **6 are real far-field transfers**: heat conduction → vision backbone (vHeat); the National
  Popular Vote compact → climate commitment mechanism; ethology → robot signaling; human visual adaptation →
  denoising; NLP translation → molecule design; soccer pitch control → Ultimate.
- The rest are a CS paper recasting its own problem as a standard ML task ("diagnosis as sequential decision-making").

**About 12% real: fails the bar.**

Usable only as small, clean secondary evaluations: ParallelPARC gold (310) and "Same Problem, Different Field"
(Kulikowski 2026; 5,812 pairs from 109 papers, CC BY 4.0).

## Labeling with Sonnet, then Fastino's GLiNER2.5-Decide (10/05/2026)

Shaun's plan: Sonnet (his subscription, no cost) labels a seed set; Decide is fine-tuned on it and labels the pool at
scale, keeping only its confident "real" pairs.

**Calibrating Sonnet** (`scripts/p10-label.py`) on Claude's 140 hand labels, blind, 10 pairs per call:

| Prompt | Agreement | Precision of "real" | Recall of "real" |
|---|---|---|---|
| v1 (strict) | 72% | 79% (15/19) | 43% |
| v2 (decide by purpose) | 76% | 71% | 63% |
| v3 (v2 plus a 1-5 "borrow" score) | 79% | 77% (27/35) | 77% |

- The prompt examples are invented. The first draft used hand-labeled pairs, which would have shown Sonnet the answers.
- Rounds 0-1 tuned the prompt; round 2 is held out.
- **Adjudication:** the 8 pairs v3 called real and Claude did not were re-read against the written rule. 2 hand labels
  became "real" (luminescent dyes → solar concentrator; collagen membrane process → cell-culture hydrogel). 4 stayed
  wrong for Sonnet, and 2 were ambiguous and kept.
- **After adjudication:** v3's "real" precision is 83% overall (90% on tuning rounds, 73% on the held-out round, n=15),
  with 5 of 35 same-technology.
- Requiring two votes (v2 and v3) or borrow ≥ 3 did not raise precision.

**Gate decision:** the 85% Sonnet gate was a margin for the student model, and Sonnet reaches 83%. The binding gate is
unchanged: a fresh 50 of the final kept pairs must be **at least 80% real and at most 10% same-technology**, read by
hand. Sonnet v3 proceeds as the teacher.

**Seed labeling:** Sonnet v3 labeled 2,989 seed pairs (2013-2017) in about 15 minutes; 22% "real". Batch size was
checked on the gold set: 70 pairs per call gave 57% precision on real borrowings and 30 per call gave 65%, against 77%
at 10. Labeling stays at 10 per call, with more calls in parallel.

**Decide failed its gate.** Fine-tuned on 2,392 Sonnet labels (4 epochs, positives rebalanced; job P10D, on demand,
about 40 minutes and $0.70):

- Its "real" calls were 50% precise against held-out Sonnet labels (597 pairs) and 48% against the hand labels.
- Its confidence was effectively binary: the same 101 pairs pass at every threshold from 0.5 to 0.95.

The job was stopped before scoring the pool, and the log kept (`work/logs/P10D-clean.log`).

**Fallback, which Shaun had already allowed:** Sonnet v3 labels 6,000 random pool pairs per split (val and test years)
directly. Only Sonnet's "real" pairs are kept, and a fresh 50 of the val-year kept pairs is hand-checked against the
gate. Test-year pairs are not read by hand.

**Sonnet on the pool:** 11,960 pairs labeled (6,000 per split; 4 refused batches cost 40). 1,213 val and 1,137 test
pairs called "real".

**Hand check, 50 random val-year "real" pairs (`work/p10/review-final-val.txt`): 24 real (48%), 8 same technology
(16%), 18 unrelated. Fails the gate.**

- C: 1, 3, 5, 7, 9, 12, 15, 16, 17, 19, 21, 22, 23, 24, 25, 29, 30, 32, 33, 34, 35, 38, 40, 42.
- A: 4, 13, 20, 31, 41, 43, 46, 48.
- B: the other 18.

By Sonnet's borrow score:

| Borrow | Pairs | Real | Same tech | Unrelated |
|---|---|---|---|---|
| 2 | 9 | 0 | 0 | 9 |
| 3 | 27 | 12 (44%) | 7 | 8 |
| 4-5 | 14 | 12 (86%) | 1 (7%) | 1 |

The dial is therefore **"real" with borrow ≥ 4**. It was chosen on this sample, so it must pass a **fresh** 50 before it
is accepted. Borrow ≥ 4 keeps about 7% of labeled pairs, so about 9,000 more pairs per split are labeled to reach
1,000 queries per split.

## Second pool round and the fresh check of the borrow ≥ 4 dial (10/06/2026)

- **Second pool round:** Sonnet labeled 17,930 more pool pairs.
- **Kept set** (real, borrow ≥ 4, one per citing patent; `work/p10/label/kept.parquet`): 1,132 val and 953 test pairs.

**Fresh hand check:** 50 val-year kept pairs never read before (`work/p10/review-final-val2.txt`; Sonnet's scores
hidden). **41 real (82%), 6 same technology (12%), 3 unrelated.**

- A: 4, 11, 29, 34, 35, 36 (e.g. vaccine drying ← pharmaceutical spray drying; touch-panel passivation ← chip
  interconnect capping).
- B: 18, 23, 37.
- C: the other 41, e.g. inflatable data center ← inflated hog-barn dome; UV disinfection robot ← rotating beacon
  reflector; spinal growing rod ← worm gear; car sound-direction cue ← doll that turns toward sounds; aircraft waste
  compactor ← surgical fluid collection.

**Verdict against the gate:** the 80% real bar passes. The 10% same-technology cap misses by one pair (6/50).
Pooled with the first check's borrow ≥ 4 pairs: 53/64 real (83%), 7/64 same technology (11%). Borderline; not
approved yet.

## Fastino round 2: trained on Sonnet's borrow scores (10/06/2026)

Shaun asked to prioritize training GLiNER2.5-Decide on Sonnet's scores. Positives: Sonnet "real" with borrow ≥ 4;
negatives: Sonnet same-technology or unrelated; Sonnet's weaker "real" calls left out. 10,162 training pairs (11%
positive), 2013-2017 citing years only. 36 minutes on one A10G (`scripts/run-P10D2.sh`); then both Sonnet-labeled pool
subsets were scored with the saved model (`scripts/run-P10D3.sh`, 30,000 pairs at 30 per second).

**Its output is still yes/no.** Over 12,000 pool pairs, 11,164 scored 0.00 and 745 scored 1.00; only 6 distinct
scores on the 190 hand labels. So it cannot be dialed; it is one vote.

**On pairs Sonnet called real, it is about as good as Sonnet's borrow score, not better.** On the 50 hand-labeled
pairs of the first pool check (`gold2.jsonl`, never trained on). Correction, same day: these 50 are a sample of pairs
Sonnet labeled real (any borrow score), not raw pool pairs, so "Fastino yes" here is Fastino on top of Sonnet's real
call. Fastino alone on raw pairs is measured on the 140 hand labels of `gold.jsonl` instead.

| Rule | Kept | Real | Same tech | Unrelated |
|---|---|---|---|---|
| All 50 | 50 | 24 | 8 | 18 |
| Fastino yes | 13 | 11 (85%) | 2 | 0 |
| Sonnet real, borrow ≥ 4 | 14 | 12 (86%) | 1 | 1 |
| Both | 8 | 7 (88%) | 1 | 0 |

On all 190 hand labels: AUC 0.63 (untrained 0.49); the top 10% by score were 84% real (16/19).

**As a second vote it clears the gate on the fresh check.** Of the 50 fresh-check pairs (all Sonnet borrow ≥ 4; 41
real, 6 same tech, 3 unrelated), Fastino said yes to 18: **16 real (89%), 1 same technology (6%), 1 unrelated.** It
said no to 5 of the 6 same-technology pairs, and to 25 of the 41 real ones.

**Cost in volume:** requiring both keeps 477 of the 1,132 val pairs and 377 of the 953 test pairs, below the 500 test
queries the verdicts need. More pairs need more labeling: Fastino can screen the remaining ~45,000 pool pairs in
about 25 GPU minutes, and only its yes pairs then go to Sonnet.

**Not yet approved:** the 18 pairs are the only both-agree pairs read so far. The gate needs a fresh 50 of the
both-agree set, read with both models' outputs hidden.

## Exclusion cascade: two Fastino stages before Sonnet (10/06/2026)

Shaun: prove exclusion through a series of steps, removing easy cases before any LLM looks. Two GLiNER2.5-Decide
stages, each trained on one distinction from Sonnet's seed labels (2013-2017, `scripts/run-P10D5.sh`):

- **Stage 1, unrelated:** Sonnet unrelated vs everything with a link (6,000 pairs). AUC 0.91 on the 190 hand labels.
  Scores are graded (198 distinct on the holdout), unlike the include model. Keeps 30,357 of 75,053 pool pairs.
- **Stage 2, same technology:** Sonnet same technology vs Sonnet real with borrow ≥ 4 (4,137 pairs). AUC 0.74.
  Scored on stage 1's survivors; 9,323 survive both.

Pair by pair on the hand labels (`scripts/p10-cascade-check.py`; hand pairs scored per stage by `run-P10D6.sh`):

| Set | Rule | Kept | Real kept | Same tech kept | Unrelated kept | Share real / same tech |
|---|---|---|---|---|---|---|
| 140 raw (gold.jsonl) | none | 140 | 37/37 | 59/59 | 44/44 | 26% / 42% |
| | stage 1 | 71 | 25 | 45 | 1 | 35% / 63% |
| | stage 2 | 37 | 18 | 6 | 13 | 49% / 16% |
| | both | 12 | 9 | 3 | 0 | 75% / 25% |
| 50 Sonnet-real (gold2) | both | 20 | 13/24 | 4/8 | 3/18 | 65% / 20% |
| fresh 50 (Sonnet borrow ≥ 4) | stage 1 | 39 | 34/41 | 3/6 | 2/3 | 87% / 8% |
| | both | 22 | 19/41 | 1/6 | 2/3 | 86% / 5% |

- Each stage removes its target well: stage 1 removed 43 of 44 unrelated raw pairs, stage 2 removed 53 of 59 same
  technology.
- Neither is "definitive": on raw pairs stage 1 also removes 12 of 37 real borrowings and stage 2 removes 19; together
  they keep 9 of 37. The cascade alone is 75% real with 25% same technology (12 pairs), so it does not pass the gate
  without Sonnet.
- Cascade, then Sonnet (borrow ≥ 4): 86% real, 5% same technology on the fresh check (22 pairs). On the 30,000
  pairs Sonnet had already labeled this keeps 596 val and 503 test.

**Plan (Shaun's order):** stages 1 and 2 remove pairs first; Sonnet labels only the survivors. The 5,525 survivors
Sonnet had not labeled are being labeled (`label/pool-cascade.jsonl`). Then a fresh 50 of the final set is read by
hand against the gate. Known bias to report: the kept pairs are those both models find clear, so the retrieval test
will lean toward more legible borrowings. Models kept: `work/p10/stage1/`, `stage2/` (`model-final.tgz`).

## Final set (cascade, then Sonnet) and a rater-consistency failure (10/06/2026)

Sonnet labeled the 5,521 cascade survivors it had not seen. Kept (stage 1 and stage 2 keep, Sonnet real with borrow
≥ 4, one pair per citing patent; `label/kept-cascade.parquet`): **1,100 val, 1,533 test.**

**Fresh check** (`work/p10/review-cascade-val.txt`, 50 random val pairs never read before, scores hidden): **30 real
(60%), 15 same technology (30%), 5 unrelated.** Fails the gate. Same technology: 4, 5, 6, 7, 9, 22, 23, 24, 29, 30,
32, 36, 37, 38, 47. Unrelated: 20, 25, 28, 34, 46.

**But the rater moved.** The 28 of these pairs from the earlier subsets came out 57% real, against 86% for the same
population that morning, so Claude re-judged the morning's fresh 50 (`review-final-val2.txt`) shuffled and
renumbered, without the old labels (`work/p10/rejudge-val2.txt`, key in `rejudge-val2-key.txt`):

| Morning label → evening label | Pairs |
|---|---|
| real → real | 24 |
| real → same technology | **17** |
| same technology → same technology | 4 |
| same technology → other | 2 |
| unrelated → unrelated | 3 |

Agreement 31/50. Judged the evening way, the morning's set is **50% real, 42% same technology**, so the cascade set
(60% / 30%) is better than Sonnet borrow ≥ 4 alone, but neither passes. The disagreement is one boundary: **a known
technology moved into a new product** (UV sterilizing from gaming machines to aircraft cabins; bearing wheels from carts
to dishwasher racks; peelable cohesive-failure film from chip tape to can lids). The morning counted these as real
borrowings; the evening as same technology, because the two abstracts share the technology's own words, so retrieval
needs no abstraction to find the source.

Every gate result so far rests on one rater's unwritten line, and that line moved within a day. Before any gate is
read again: (1) the A/C rule is written down with examples and fixed (Shaun's decision: which side these belong on),
and (2) each hand check re-judges 10 earlier pairs blind, so drift is measured rather than assumed away.

## Rubric decided, gate read against it, final set (10/06/2026, evening)

Shaun asked Claude to decide the line, by what best trains a model to think analogically. The decision and the
evidence that revised it the same evening are in `RUBRIC.md`. In short: the boundary "generic technology reused vs
the cited field's own solution" could not be applied consistently by Claude (31/50 across one day) or matched by
Sonnet with a written rubric (about chance), so it does not gate. **Kept: borrowings between different products (C
and G). Excluded: same kind of product (A) and unrelated (B).** Analogical distance is measured instead: IDF-weighted
word overlap of the two abstracts (`p10-label.py overlap`), stored with every kept pair.

**Gate sample** (`work/p10/gate2-val.txt`, key `gate2-val-key.jsonl`, labels `label/rubric-gate2.jsonl`): 50 unread
val pairs of the cascade + Sonnet v3 set, plus 10 earlier pairs shuffled in. Re-judged 10/10 the same (weak evidence:
they had been read hours before). New 50: **34 kept (68%), 14 same product (28%), 2 unrelated. Fails.**

**Stage 3, Sonnet v4** (prompt v4, labels C/G/A/B) as a same-product remover. On the gate sample it removed 7 of 16
same-product or unrelated pairs and 5 of 42 good ones: keeping its C/G gives **37 of 46 (80%) borrowings, 9 (20%) same
product**. Claude and Sonnet v4 agree on 46/60 pairs on this boundary, so purity cannot be measured much more finely
than ±10 points with these raters; further filter rounds were judged not worth their cost.

**Overlap does not separate same product from kept** (median 0.113 vs 0.105; AUC 0.55 on 93 hand pairs): same-product
pairs are not much easier to find by their words, so they should not inflate the raw-text baseline much.

**Final set** (`label/kept-final.parquet`; stages 1-2, Sonnet v3 borrow ≥ 4, Sonnet v4 C or G, one pair per citing
patent): **923 val, 1,276 test.** Estimated 80% borrowings, 20% same product (n = 46), below the 10% same-product cap
set earlier. Claude's recommendation: proceed, with that purity stated in the report and every result also reported
by overlap quartile. Pipeline `keep:` now points at this file; `review/APPROVED` is Shaun's to create.

## Stage 3 (Fastino emulating Sonnet's screen) fails; batching gives no speed (10/06/2026)

Shaun authorized a stage 3 Fastino pass, tuned to remove, emulating Sonnet's final screen on stage 1-2 survivors
(`scripts/run-P10D7.sh`; 9,164 survivors Sonnet labeled, keep = v3 borrow ≥ 4 and v4 C/G, 22.8%; hand-checked pairs
excluded). **It learned nothing usable:** AUC 0.52 on held-out citing patents and 0.46 on the 190 hand labels. At
every cut it removes Sonnet keeps at the rate it removes pairs (cut 0.1: 9.3% removed, 9.9% of keeps lost; cut 0.2:
18.3% and 18.3%). After stages 1-2, what is left to decide is the borrowing-vs-same-product line that Claude and Sonnet
agree on only about 77% of the time; a 0.5B classifier does not learn it from 6,000 noisy labels. Stage 3 is dropped:
Sonnet labels every stage 1-2 survivor (about 12% of the corpus).

**Batched scoring** (`batch_classify_text`, batch 32) on 3,000 pool pairs: 30.0 pairs/s against 29.1 one at a time,
identical scores (max difference 0.0). The batch call does not batch on the GPU; scoring is bound elsewhere.

**Speed probe** (`scripts/run-P10SP.sh`, 1,500 pool pairs each, base Decide model, g5.xlarge): fp32 28.5 pairs/s,
bf16 30.5 pairs/s (scores within 0.033, 1,497/1,500 on the same side of 0.5), three fp32 processes at once 10.2 pairs/s
each, about 31 in total. GPU busy 59% of the time in fp32; CPU about one core of four. Neither precision nor more
processes raises throughput: about 30 pairs/s per GPU is this library's ceiling here. For the 5.08M-pair expansion
that is about 47 GPU hours for stage 1 and 19 for stage 2 (~$66 on demand, ~$25 spot), beyond the remaining budget.
The expansion would need a cheaper first remover ahead of stage 1, or a faster implementation of the same model.
