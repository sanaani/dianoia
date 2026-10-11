# Phase 8: predicting the next idea (JEPA, first tests) (10/04/2026)

Question: does the idea space carry structure that a small predictor can use to anticipate where a conversation goes
next, beyond trivial baselines? The idea encoder is frozen (S1, `results/phase-7-e2e`); the predictor sees only idea
vectors. This is the first step toward a JEPA over ideas: predict in the idea space, not in words.

**Data** (`jepa-prepare.py`): the splitter's training units (`split/train.jsonl`, never an evaluation unit), each
as its reference ideas in order, with one placeholder mapping per unit. 6,327 units with at least two ideas
(40,310 ideas); every 10th unit (seeded shuffle) is held out: 633 test units, 3,374 next-idea predictions, a pool of
3,951 distinct test ideas. From J3 on, each training unit's one-for-one paraphrase (`split/para.jsonl`) is added as
another training unit, and every 10th training conversation is kept for early stopping.

**Predictor** (`remote-jepa.py`): a small causal transformer over a unit's normalized idea vectors plus a position
embedding; at each position it outputs the next idea's vector. Loss: InfoNCE against every target in the batch
(temperature 0.05) plus a cosine term. **Measure:** the rank of the true next idea among the 3,951 test ideas by
cosine (median rank, top 10), against two baselines that need no training: the previous idea itself ("last") and the
mean of all previous ideas ("context mean").

| run | change | median rank | top 10 | cosine to the true next idea |
|---|---|---|---|---|
| last (baseline) | - | 112 | 0.347 | 0.426 |
| context mean (baseline) | - | 117 | 0.295 | 0.465 |
| J1 | 4 layers, 3,000 steps | 342 | 0.043 | 0.099 |
| J2 | + residual from the last idea, cosine term, 15,000 steps | 349 | 0.169 | 0.247 |
| **J3** | 2 layers, dropout 0.3, weight decay 0.1, + paraphrased units, early stopping (8,000 steps) | **45** | 0.347 | 0.323 |

J1 underfit (training loss barely below chance). J2 overfit: training cosine 0.56, held out 0.25, on about 35,000
transitions. J3 doubled the data with paraphrased conversations, shrank and regularized the model, and had no gap
between training and validation; its validation loss was still falling at the last step.

J3 by medium (median rank; predictor / last / context mean):

| | predictor | last | context mean |
|---|---|---|---|
| chat | **69** | 256 | 174 |
| meeting | **46** | 86 | 147 |
| email | 6 | **4** | 5 |

**First positive result.** On conversations the predictor ranks the true next idea two to four times better than
either baseline, so the idea space carries turn-to-turn structure a predictor can learn from a few thousand
conversations. On email it is level with the baselines: an email is one author continuing one thought, and the
previous idea is already close. Raw cosine is lower than the baselines' because InfoNCE drops the direction every
idea shares (the baselines' cosine is mostly that shared direction); ranking is the measure that counts.

Decoded predictions are on topic but not fluent ideas (examples in `J3-jepa.json`): "A asks how A can upgrade from
Ubuntu 12.04 to 13.04 or 13.10" -> predicted "A asks A for upgrade Ubuntu 10.04 up to 1.04 or 13.10" (true next:
"A says that A is running update-manager -d and it is offering 14.04"). The predicted vector lies between real
ideas, where the decoder was never trained.

**Next.** The validation loss was still falling, and data is the limit (J2 overfit at 35,000 transitions): the
splitter can turn far more conversations from the corpora into idea sequences, which is also how the production
pipeline would feed a predictor. A predictor that names the next idea should also be read by retrieval (nearest real
idea) as well as by decoding.

Fixed after J3: the evaluation reused the variable `best`, so J3's reported `best_val_loss` (0.6243) is wrong; the
model selection itself used the right value (the curve's minimum, 4.863 at step 8,000). All downloads checksummed.

## 2. Three times the data (J4) and a bigger predictor (J5, J6)

**Data** (`jepa-units.py`, `jepa-chain.py`): every training meeting (`ext/qmsum-train`) cut into consecutive
6-turn windows and every annotated IRC thread (`ext/irc-train`) into consecutive 8-message windows, one speaker-letter
mapping per conversation: 20,821 windows from 3,453 conversations. A window sharing a turn with a test unit was
dropped with its neighbors (1,056). Splitter q3 cut them into ideas (78 GPU minutes, none empty; `remote-split.py
predict-qwen` now writes batch by batch and resumes). Each conversation's windows were chained in order (a missing
window ends the run) and cut into sequences of at most 32 ideas: 7,508 sequences, 145,369 transitions, added as
training units to J3's. The test units are J3's, so the numbers compare directly.

| median rank of the true next idea (of 3,951) | J3 | **J4** | J5 | J6 | last | context mean |
|---|---|---|---|---|---|---|
| all | 45 | **26** | 69 | 38 | 112 | 117 |
| chat | 69 | **36** | 120 | 60 | 256 | 172 |
| meeting | 46 | **25** | 57 | - | 86 | 149 |
| email | 6 | 9 | 10 | - | 4 | 5 |
| top 1 / top 10 (all) | 1.8% / 0.347 | 3.6% / **0.391** | 4.0% / 0.312 | 4.3% / 0.353 | 0 / 0.347 | 0.3% / 0.296 |

- **J4** (J3's recipe: 2 layers, dropout 0.3, weight decay 0.1; 15,000 steps) on 3x the data: median rank 45 -> 26,
  chat 69 -> 36. Chat is the clean comparison: every chat test thread was excluded whole. Meeting windows away from a
  test excerpt stay in training (nearly every training meeting has one), so meeting numbers may gain from seeing the
  same meeting's topic; J3 had the same exposure. Email got a little worse: the new data is all conversation.
- **J5** (4 layers, 40,000 steps): lower validation loss (4.17 against 4.46) and more exact hits, but a much worse
  median rank (69). The in-batch validation loss does not track the pool rank.
- **J6** (4 layers, 20,000 steps, checkpoint chosen by the validation median rank over the whole validation pool, as
  the test is scored): validation rank 17, test 38.

**Why the validation set misled.** The validation units are excerpts from the training meetings and threads, and the
window data contains other windows of those same conversations, so validation rewards memorizing a conversation.
The bigger model memorizes more: best on validation, worse on the protected test. On the clean measure, the 2-layer
J4 is the best predictor, and more capacity does not help at this data size. Validation must be whole held-out
conversations from now on (`jepa-units.py` excludes only the test's neighborhoods).

**Where this leaves the question.** Tripling the data halved the rank, so the idea space keeps giving a predictor
more to learn as conversations are added; the public training sources are now used up at the window level, and the
next sources are the full public corpora (all of Ubuntu IRC, all of AMI). J4's predictor is kept in
`work/jepa/J4/predictor.pt`. All downloads checksummed.

## 3. Clean splits (J4b) and unseen wiki conversations (J7)

**What changed.** The validation leak above led to whole-conversation splits (`splits.py`: sha256 of source and
conversation, 80/10/10), written into a manifest that `remote-jepa.py --require-audit` refuses to train without, and an
audit (`audit-data.py`, proven by `audit-selftest.py` planting copies) that found and removed IRC bot factoids,
copied wiki discussions, templates and near-copies. Every number in this section comes from data that passed it.

| median rank of the true next idea | predictor | last idea | context mean |
|---|---|---|---|
| J4b, clean splits (pool 18,965) | **221** | 916 | 939 |
| J7, adds Wikipedia talk pages (pool 40,479) | **416** | 2,504 | 2,467 |
| J7, wiki only | **590** | 3,270 | 3,578 |
| J7, chat / meeting / email | 334 / 281 / 12 | | |

- The predictor's lead over both baselines (4x for J4b, 6x for J7) is the same size as before the leak was fixed,
  so the leak inflated nothing that matters here: the J4 ranks were smaller because their pool was smaller.
- Wikipedia talk pages are new to every component (splitter, encoder, predictor), and the predictor still ranks the
  true next idea 5.5x better than either baseline on them.
- **Same pool, two predictors (E2).** J4b's predictor (no wiki in training) scored on J7's pool: wiki median rank
  1,944 against J7's 590; chat 372 against 334; meeting 290 against 281; email 13 against 12. Adding a source helps
  that source most and costs the others nothing.

**Speedups proven equivalent (E1, E3, 10/04/2026).** Vectors are now cached by adapter checksum and training keeps all
data on the GPU with vectorized scoring.
- E1: J7's saved predictor scored by the new code: median rank 415 against 416, top-10 and cosine identical.
- E3: J4b retrained by the new code: median rank 222 against 221, last/context-mean baselines within 2 ranks.
- Encoding 405,204 ideas took 7.9 minutes once and 0 afterwards; 15,000 training steps took 5.9 minutes against 7.6
  (1.3x).

## 4. Ten times the data (J8) (10/04/2026)

**Data.** WikiConv 2018 (ConvoKit `wikiconv-2018`, English talk pages, CC BY-SA): only comments posted in 2018, so it
cannot repeat `wiki-corpus` (2011). Split group: the whole talk page. Cleaning, each step now in `dianoia.clean`:
1,024,019 bot comments dropped (names ending in "bot"); then 650 copied threads and windows holding one of 612
template lines; after splitting (vLLM, 70 minutes for 72,285 windows), 2,338 near-copies, 6,262 windows holding 2+
**template ideas** (an idea in 5+ conversations: Wikipedia's user warnings are reworded per use, so they pass the line
filter but split into identical ideas; the first J8 audit failed on 999 of them), 1 held-out cross-post, and, after
placeholders, 6 held-out units repeating a training group (deletion notices about two articles become identical once
both names are "Person 1"). WikiConv adds 648,661 transitions. The audit ran on the instance (the laptop ran out of
memory on 111 MB of units) and passed: 0 duplicated windows of 23,214 held out, 0 duplicated units.

| median rank of the true next idea, J7's test pool (40,479) | J7 | **J8** |
|---|---|---|
| all | 416 | **365** |
| wiki (unseen by every component) | 590 | **408** |
| chat | **334** | 405 |
| meeting | **281** | 313 |
| email | 12 | 12 |

On its own pool (105,548 ideas) J8 ranks the true next idea at 582 against 6,090 (last idea) and 5,910 (context
mean): a tenfold lead. More data helped the source it came from (wiki -31%) and cost the others, which are now 19% of
training units: a mix problem, tested next (J9A: source mix alpha 0.5; J9B: 4 layers, since J5's memorizing came at
a tenth of this data). Validation rank flattened at 638 from step 34,000 of 40,000.

**Parallel runs.** Two processes scoring J8 on J7's pool at once both gave exactly the solo numbers (median rank 365,
top-10 0.1652), so side-by-side jobs are safe; but two J8-sized runs (about 7 GB of host memory each while loading)
do not fit the 16 GB instance, so those run one at a time.

## 5. Source mix and capacity (J9A, J9B) (10/04/2026)

Both on J8's data, J4's recipe, 40,000 steps. J9A draws each unit with weight (its source's unit count)^(alpha - 1),
alpha 0.5 (`--mix-alpha`): wiki falls from 81% to 55% of each batch, chat and meeting rise to 16% each, email 13%.
J9B: proportional sampling, 4 layers instead of 2.

| median rank, J7's test pool (40,479) | J7 | J8 | **J9A** (mix 0.5) | J9B (4 layers) |
|---|---|---|---|---|
| all | 416 | 365 | **337** | 354 |
| chat | 334 | 405 | **285** | 388 |
| meeting | 281 | 313 | **244** | 295 |
| wiki | 590 | **408** | 461 | 409 |
| email | 12 | 12 | **10.5** | 12 |
| top 10 | 0.167 | 0.165 | **0.176** | 0.165 |

- **With a balanced mix, ten times the data helps every source:** J9A beats J7 on chat (-15%), meeting (-13%), wiki
  (-22%) and email, and has the best top-10. J8's losses on chat and meeting were the mix, not the data.
- **Capacity is not the limit:** 4 layers gain 2-3% over J8 (J8's own pool: 577 against 582). The sources did not
  compete for model size; they competed for batch share.
- On J8's own pool (81% wiki) J9A scores 664 against J8's 582, because that pool rewards wiki; J7's pool, closer to
  balanced, is the fair measure across sources. J9A is the current best predictor (`work/jepa/J9A/predictor.pt`).

**Where this leaves the question.** The predictor keeps improving as conversations are added, provided no source
swamps the rest. Chat and meeting are now the scarce sources (11,216 of 78,089 training units), so the next data
should be conversation and meeting text (full Ubuntu IRC, AMI/ICSI, FOMC, parliamentary debate), not more wiki.

## 6. More conversation and meeting text (J10) and blind plausibility (10/05/2026)

J10 adds the scarce sources to J8's data: auto-disentangled #ubuntu IRC conversations (15% of unused days) and FOMC
meetings, for 148,856 units and 1,625,113 distinct ideas; J9A's recipe (mix alpha 0.5, 2 layers, 40,000 steps). Its
own validation rank flattened at 826 (pool 161,826). Scored on J7's pool:

| median rank, J7's test pool (40,479) | J7 | J8 | J9A | **J10** |
|---|---|---|---|---|
| all | 416 | 365 | 337 | **328** |
| chat | 334 | 405 | 285 | **169** |
| meeting | 281 | 313 | **245** | 259 |
| wiki | 590 | **408** | 461 | 503 |
| email | 12 | 12 | 11 | **9.5** |
| top 10 | 0.167 | 0.165 | 0.176 | **0.180** |

(J9A's figures here are from its rerun, J9A-P, which reproduced 337 exactly; chat 284 and meeting 245 by that run.)

- **Chat data helps chat a lot (-40%) and little else.** Meeting is flat and wiki worse, as wiki's batch share fell.
  The overall gain is 3%: per-source data buys per-source rank, not a general predictor.

**Blind plausibility** (`scripts/grade-plausible.py`, `results/plausible/`). Rank asks only about the one true next
idea. Here Sonnet, blind, scores 1-5 as a next idea the true idea, the pool idea nearest each guess, and a random
idea, on 325 test positions (public corpora only). Mean score, share scored 4 or 5, and the predictor's paired
difference with a 95% bootstrap interval:

| candidate | J9A mean | plausible | J10 mean | plausible |
|---|---|---|---|---|
| true next idea | 3.47 | 56% | 3.46 | 56% |
| last idea's nearest | 2.90 | 31% | 2.91 | 34% |
| **predictor** | 2.74 | 26% | 2.85 | 27% |
| context mean | 2.63 | 19% | 2.69 | 22% |
| random | 1.31 | 2% | 1.31 | 1% |
| predictor minus last | -0.16 [-0.29, -0.03] | | -0.06 [-0.20, 0.08] | |
| predictor minus context mean | +0.11 [-0.03, 0.27] | | +0.16 [+0.02, +0.31] | |

- **A far better rank does not make a more plausible retrieval.** The predictor ranks the true idea about 7x better
  than the last-idea baseline, yet its retrieval is no more plausible than simply the idea nearest the last one. J9A
  was significantly worse; J10 closes the gap (J10 minus J9A on 301 shared rows: +0.10 [-0.01, +0.21]) but does not
  pass it.
- The grader is stable: the true idea scores 3.47 and 3.46 across the two runs.
- **The bar for "predicting in idea space is good"** is now explicit: beat the last-idea and context-mean baselines on
  both rank and blind plausibility with intervals excluding zero, and match a text model of similar size that writes
  the next idea in words, scored the same way. The text baseline is the next test.

## 7. What compact vectors cost (Q1b) (10/05/2026)

`scripts/remote-quant.py` on J7's pool: share of the full float32 top 10 neighbors each compact form returns (5,000
queries, self excluded), and the median rank of the true next idea queried from the current idea (the last-idea
baseline). PCA and int8 scales are fit on 100,000 training ideas. (Q1, the first run, also jittered cosine scores;
Q1b jitters only Hamming ties.)

| form | bytes per idea | top-10 kept | next-idea rank |
|---|---|---|---|
| float32, 1,024 | 4,096 | 100% | 2,506 |
| int8, 1,024 | 1,024 | 99.2% | 2,507 |
| float32, PCA 256 | 1,024 | 78.3% | 2,766 |
| int8, PCA 256 | 256 | 78.2% | 2,770 |
| binary, 1,024 bits | 128 | 67.2% | 2,429 |
| binary, PCA 256 bits | 32 | 29.6% | 3,025 |

- **int8 at full length is free.** Cutting dimensions costs a fifth of the neighbors; binary keeps the
  next-idea signal (its centering may reduce hub ideas) but not the exact neighbors, so it suits a first-pass filter
  with float or int8 reranking.

## 8. The text baseline (T1) (10/05/2026)

`scripts/remote-textnext.py`: Qwen3-0.6B-Base (about 33 times the predictor's size) fine-tuned on J10's training
units written one idea per line (3,000 steps, source mix 0.5), then writing the next idea greedily at every test
position of J7's units. The written idea is encoded with S1 and ranked against J7's pool exactly as J10 was. On demand
(`run-T1.sh`) after a spot reclaim cost the first try 45 minutes.

| on J7's pool (40,060 positions) | top 1 | top 10 | median rank |
|---|---|---|---|
| J10 predictor | 1.5% | **18.0%** | **328** |
| T1, text model | **1.7%** | 12.9% | 4,424 |
| last idea | 0% | 18.2% | 2,506 |

Blind plausibility (Sonnet, the same 325 rows as J10-P, seven unlabeled candidates; 1-5):

| candidate | mean | rated 4-5 |
|---|---|---|
| true next idea | 3.30 | 49% |
| **T1's written idea** | **3.13** | **40%** |
| last idea | 2.77 | 26% |
| pool idea nearest T1's | 2.70 | 26% |
| pool idea nearest J10's | 2.62 | 21% |
| context mean | 2.58 | 19% |
| random | 1.25 | 0% |

Paired differences (95% bootstrap): T1 written - J10 +0.52 [0.36, 0.68]; T1 written - true -0.16 [-0.34, -0.00];
pool nearest T1 - pool nearest J10 +0.08 [-0.08, 0.23].

- **The text model wins on plausibility by a wide margin; idea space wins on rank.** T1 commits to one specific next
  idea ("B tells A to run the command killall wine"): often natural, sometimes exactly right (top 1 is higher), and when
  wrong, far away. J10 aims at the average of the likely next ideas, which is never far and rarely right.
- **Retrieval caps both.** Picking the pool idea nearest each guess drops T1 from 3.13 to 2.70, level with J10's
  retrieved idea: the pool rarely holds a fitting next idea for this conversation, so retrieved-candidate plausibility
  measures the pool more than the predictor.
- **Against the bar in section 6, idea-space prediction fails** the "match a text model" part on plausibility. What it
  does better is the expected-average target (rank), useful for "what is normal here" and surprise measures, not for
  proposing a next move. Proposals should come from a text model; idea space scores them.
