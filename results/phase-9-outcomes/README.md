# Phase 9: does idea space predict outcomes?

Phase 8 showed the next-idea predictor (J10) is a good model of the *expected* next idea and a poor proposer of
specific ones (section 8, T1). This phase asks whether idea space can tell which moves *lead to good outcomes*, the
"surprising and valuable" half of the design agreed on 10/05/2026: a routine model (what normally comes next), a value
model (which departures pay), and a switch between them.

## 1. ChangeMyView (C1) (10/05/2026)

Data: Winning Arguments (Tan et al., WWW 2016; ConvoKit `winning-args-corpus`, license class research, not
publishable). 4,254 pairs of first replies to one original post, one of which won the poster's delta; 3,101 train, 349
validation (by post), 804 held out (the authors' split). Only first replies; `EDIT`/`UPDATE` sections and delta
mentions removed (`scripts/cmv-build.py`). Every model answers: which of the two replies won? Chance 0.5.
`scripts/remote-cmv.py`, `run-C1.sh`, on demand.

| model (804 held-out pairs) | picks the winner | 95% interval |
|---|---|---|
| **text: Qwen3-0.6B fine-tuned on the replies as written** | **67.0%** | 63.8-70.3 |
| idea: value model on S1 idea vectors (5 seeds averaged) | 61.4% | 58.1-64.8 |
| longer reply wins | 60.4% | 57.0-63.6 |
| text_ideas: the same Qwen on the splitter's ideas written out | 59.3% | 55.9-62.8 |
| (local) tf-idf logistic | about 64% | |

Paired differences: idea - text -5.6 points [-9.1, -2.0]; text_ideas - text -7.7 [-11.6, -3.7];
idea - text_ideas +2.1 [-1.5, +6.1].

- **The loss is in the rewrite, not the vectors.** Reading the splitter's literal ideas instead of the reply costs the
  text model 7.7 points; idea vectors do as well as those same ideas in words. The splitter keeps what was said and
  drops how it was said (tone, hedging, quoting the poster, sources), which the 2016 paper found predicts persuasion.
- Neither idea-based model beats reply length.

Surprise under J10 (all 4,254 pairs; surprise = 1 - cosine of J10's prediction to each reply idea, given the post's
last 16 ideas):

- The more surprising reply wins 53.7% [52.1, 55.1]. In a logistic model with length, surprise keeps a positive weight
  (+0.17 [0.09, 0.23], standardized) and relevance to the post a negative one (-0.28 [-0.35, -0.19]): with length
  fixed, replies that bring something less expected and less like the post win more often.
- **No switch effect:** by the post's openness (mean surprise of its own idea steps), surprise wins 54.1% (most
  routine third), 52.7%, 54.0% (most open third); the interaction weight is +0.05 [-0.02, 0.13].
- Observational: a surprise that comes with winning is not shown to cause it.

**What this test could not show.** One reply per side, not a trajectory; a small training set that favors the
pretrained text model; one Reddit user's delta as the outcome; a domain none of the idea components were trained on.
Next: GitHub pull requests (revert as the outcome, pairs matched by repository, time and size), racing text, idea and
text-plus-idea-scores, with Purpose Factory's own PRs as a private local check.

## 2. GitHub reverts (G1b) (10/05/2026)

Data (`scripts/gh-collect.py`, `gh-build.py`): 3,633 pairs from 13 public repositories under MIT, Apache-2.0 or
BSD-3-Clause (checked with the GitHub API). Each pair is a merged PR that a later merged PR reverted and a clean merged
PR numbered within 6 of it (same repository, about the same time), the closest in size; bots, rolls, bumps, reverts and
relands excluded. Text is only what existed before the merge: title, description and conversation, code blocks as
"[code]", @mentions and email addresses replaced. Held out: the newest 20% of each repository's pairs (732), so the test
is predicting future reverts. "Picks the winner" = picks the PR that was not reverted.

The fine-tuned text models were dropped (G1 became G1b): Shaun, 10/05/2026, a fine-tuned text model is not a system
we would run. Section 3 uses the production yardstick instead.

| on 732 held-out pairs | picks the clean PR | 95% interval |
|---|---|---|
| idea value model (S1 vectors, 5 seeds) | **60.5%** | 57.0-64.1 |
| the shorter PR text | 59.0% | 55.3-62.4 |
| the smaller change (lines) | 56.2% | 52.7-59.6 |

Surprise under J10 (all 3,633 pairs): the more surprising PR is the clean one 54.0% of the time, but surprising PRs are
shorter. With length in the model, surprise turns slightly negative: -0.10 [-0.18, -0.01] (more surprising, more
likely reverted); relevance -0.09 [-0.16, -0.00]. In ChangeMyView surprise helped (+0.17); here, in routine engineering
work, it is a mild warning sign.

## 3. Against a model we would ship: GLiNER2.5-Decide (D1) (10/05/2026)

`scripts/remote-decide.py`: Fastino's GLiNER2.5-Decide (Apache-2.0, DeBERTa-v3-large, 340M parameters, runs
locally), fine-tuned on each outcome's training pairs (3 epochs; each side one example labeled good or bad) and scored
on the same held-out pairs as the idea model. No LLM grades anything: the verdict is the delta or the revert. Decide
plus idea scores: logistic over the pair differences (Decide, idea, surprise, relevance, length), 5-fold cross-fitted
over the held-out pairs.

| held-out pairs | GitHub (732) | ChangeMyView (804) |
|---|---|---|
| Decide, untrained | 52.9% [49.2, 56.3] | 52.1% [48.6, 55.6] |
| Decide, fine-tuned | 58.5% [54.9, 62.0] | 58.6% [55.4, 62.1] |
| **idea value model** | **60.5% [57.0, 64.1]** | **61.4% [58.1, 64.8]** |
| Decide + idea scores | 60.0% [56.3, 63.5] | 61.8% [58.3, 65.2] |
| Decide - idea | -2.1 [-6.4, +2.3] | -2.9 [-7.0, +1.4] |
| (Decide + idea) - Decide | +1.5 [-2.2, +5.1] | +3.2 [-0.6, +6.8] |

- **Idea space matches the shippable classifier on both outcomes and leads it by 2-3 points**; neither gap is
  resolved at this size. Added to Decide, the idea score takes about twice Decide's weight (GitHub 0.39 vs 0.18;
  ChangeMyView 0.36 vs 0.17).
- Untrained, Decide is at chance on both: it needs the outcome data.
- Both stay near the simple baselines (60% length on ChangeMyView, 59% shorter text on GitHub): neither data set yet
  shows a model that reads value much beyond length. More pairs would narrow the intervals; a bigger change needs
  better outcome signals or trajectories (whole threads, not one post).
