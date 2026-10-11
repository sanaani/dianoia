# Phase 1: does a vector hold intent rather than words?

Pre-registered in [`PREREGISTRATION.md`](../../PREREGISTRATION.md), Tier 1. Every number below is from
the **test mailboxes** (30 of 150, by seed), which no model or choice saw until the chosen model was
fixed on validation (`model-choice.txt`). Materials: 2,388 Enron sentences with Haiku-written keep and
move variants, mechanically checked (`generation.json`); 350 are test.

## Verdict

| bar | result | verdict |
|---|---|---|
| word Jaccard below 0.5 (materials are valid) | 0.30 (judged), 0.29 (original) | **pass** |
| chosen model (`ft_sonar256`) overall ≥ 0.90 | **0.881** judged, 0.842 original | **fail**, narrowly |
| worst move kind ≥ 0.80 | 0.740 (type change) | **fail** |
| chosen minus raw SONAR ≥ +0.10, interval above 0 | **+0.596** [+0.575, +0.618] | **pass** |
| natural pairs ≥ 0.80 | **0.727** [0.670, 0.783] | **fail**, see the audit below |

"Judged" drops variants an independent, encoder-blind Haiku pass disputed (addenda of 09/30/2026).

## Off-the-shelf encoders track words

On test, judged materials, every off-the-shelf encoder puts a same-words, different-meaning twin
nearer the source than a faithful paraphrase:

| encoder | overall | negation | modality | role swap | verb/object | type | quantity |
|---|---|---|---|---|---|---|---|
| word Jaccard | 0.303 | 0.255 | 0.314 | 0.313 | 0.340 | 0.305 | 0.209 |
| SONAR (raw) | 0.286 | 0.233 | 0.208 | 0.350 | 0.279 | 0.367 | 0.248 |
| minilm | 0.365 | 0.378 | 0.236 | 0.310 | 0.411 | 0.567 | 0.223 |
| gtr | 0.335 | 0.312 | 0.226 | 0.305 | 0.380 | 0.542 | 0.206 |
| Qwen3-Embedding | 0.402 | 0.447 | 0.241 | 0.316 | 0.462 | 0.624 | 0.256 |
| e5 | 0.457 | 0.643 | 0.260 | 0.282 | 0.458 | 0.520 | 0.335 |
| bge | 0.478 | 0.722 | 0.287 | 0.297 | 0.488 | 0.506 | 0.348 |
| shuffled-vector control | 0.497 | | | | | | |

SONAR, trained on translation, is the most faithful to wording and so the worst at intent.

## Trained for intent

| model | overall | negation | modality | act | role swap | verb/object | type | quantity |
|---|---|---|---|---|---|---|---|---|
| frozen head on SONAR (pre-registered architecture) | 0.737 | 0.920 | 0.894 | 0.869 | 0.609 | 0.542 | 0.548 | 0.446 |
| bge fine-tuned end to end | 0.859 | 0.954 | 0.948 | 0.900 | 0.738 | 0.793 | 0.755 | 0.752 |
| **SONAR + LoRA fine-tuned (chosen)** | **0.881** | 0.966 | 0.957 | 0.931 | 0.822 | 0.828 | 0.740 | 0.792 |
| the same, first 64 dims only | 0.884 | 0.968 | 0.955 | 0.932 | 0.834 | 0.834 | 0.746 | 0.785 |

- **The frozen head was not enough.** On validation it reached 0.70 on SONAR alone and 0.76 on four
  encoders concatenated, gaining about 0.03 per doubling of data. Pooled vectors had already lost the
  distinctions. Fine-tuning the encoder (the fallback agreed in advance) was what worked.
- **The 64-dim core is as good as 256.** Intent fits in the first block, as designed.
- **Strong where intent turns on a function word** (not, might, a question mark); **weaker where it
  turns on which entity or how much** (type change 0.74, quantity 0.79).
- The result holds across registers: business 0.887, personal 0.871, news 0.878.

## Natural pairs: the bar fails, and the test is weak

122 anchors from test mailboxes had a real Enron sentence Haiku labeled same intent and one labeled
different. `ft_sonar256` is the best encoder tested (0.727) but its lead over raw SONAR is +0.066
[−0.004, +0.131], over Qwen3 +0.033 [−0.037, +0.103], and over word Jaccard +0.085 [0.000, +0.169]
(`natural-pairs-diffs.txt`). A hand audit of 8 pairs Haiku labeled *same intent* found **1** that is
the same intent under our definition ("We apologize if this causes you any inconvenience" / "We
sincerely apologize for any inconvenience this may have caused"); most are the same speech act with a
different object ("Here is the excerpt…" / "Here is the presentation…"). True paraphrases are rare in
natural mail, so the label mostly measures "same kind of sentence". The bar is reported as failed; the
audit says how much weight that failure can bear.

## Reading, against the readings fixed in advance

Closest to *"wins on Haiku's variants, fails natural pairs: we learned Haiku's style"*, with a
qualification. The fine-tuned encoder learned the kinds of change the variants contain (negation,
modality, speech act) very well, and those lessons are real: no off-the-shelf model gets them. But on
real pairs its advantage is small and not clearly above SONAR's. What it learned is **sensitivity to
intent-changing edits**, which is not yet the same as **an address for an intent**.

## Caveats

- Variants are Haiku's; about a fifth were disputed by a second Haiku pass, and the judge itself is
  an LLM with no human check beyond the samples read here.
- 350 test sources from 30 mailboxes; intervals are bootstraps over sources.
- Every configuration tried is in the addenda; the chosen one was fixed on validation first.
