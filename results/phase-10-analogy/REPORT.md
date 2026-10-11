# Phase 10 report: historical recovery of cross-domain borrowings in patents

Written by `scripts/p10-verdicts.py` from the pipeline's metrics, under the rules in `pipelines/phase10/OPERATIONALIZATION.md`, fixed before the data. Hypotheses: `HYPOTHESES.md`.

Data audit: passed; queries {'test': 1000, 'val': 909}. Chance recovery: 1.0% (top 10 of 1,000; the dry run trims candidates).

## Verdicts

| Hypothesis | Verdict |
|---|---|
| H1 | falsified |
| H2 | falsified |
| H3 | supported |
| H4 | falsified |
| H5 | supported |
| H6 | falsified |

## What was found

**H1, abstraction climbs frequency and spread: falsified.** Share of WordNet steps where the more general word is both more common and more spread across fields: nouns 44.4% [43.2%, 45.6%] of 18,295 steps, verbs 42.7% [40.9%, 44.6%] of 11,993 (frequency alone: nouns 70.2%, verbs 64.4%; spread alone: nouns 61.1%, verbs 64.3%).

**H2, verbs carry the analogy: falsified.** Test recovery from verbs only 10.6% [8.7%, 12.6%] against nouns only 41.2% [38.2%, 44.2%]; difference -30.6% [-34.1%, -27.0%] (chance 1.0%). With the neural encoder: difference -39.3% [-43.1%, -35.5%] (reported, not judged).

**H3, a shared abstraction direction: supported.** On held-out sections D, G, H (905 words), the map finds the hypernym in its top 10 of 100 100.0% [100.0%, 100.0%]; frequency only 12.4% [10.3%, 14.6%]; the word's own vector 100.0% [100.0%, 100.0%]. Map minus frequency 87.6% [85.4%, 89.7%].

**H4, role replacement beats raw text and leaks less: falsified.** The best setting on val, nouns at level 0 and verbs at level 1, recovers 62.6% [59.5%, 65.6%] on test against 64.7% [61.7%, 67.7%] for the abstracts as written: gain -2.1% [-3.2%, -1.1%]. Leakage 65.9% against 66.1% (chance 12.5%).

**H5, a best level of abstraction: supported.** Test recovery as nouns are abstracted further (verbs at level 1): level 0: 62.6%, level 1: 63.0%, level 2: 60.7%, level 3: 54.3%, level 4: 47.4%. Val-best level 0 minus level 4: 15.2% [12.2%, 18.0%].

**H6, verbs need abstracting, but less than nouns: falsified.** (a) With nouns at level 0, verbs at level 1 against verbs as written: -2.1% [-3.2%, -1.1%]. (b) In the best setting, replaced words moved up WordNet by 1.587 steps for verbs against 0.0 for nouns; Zipf rose 1.803 against 0.0; spread rose 0.835 against 0.0 bits; share of words replaced: verbs 7.5%, nouns 0.0%. (c) Not tested (no per-verb test was designed in advance).

## Recovery by setting (val / test, top 10 of 1,000)

| nouns \ verbs | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 64.6% / 64.7% | 64.1% / 62.6% | 61.8% / 63.6% | 60.0% / 62.6% | 59.6% / 59.9% |
| 1 | 63.7% / 64.3% | 63.6% / 63.0% | 61.2% / 63.5% | 59.7% / 61.4% | 57.8% / 58.9% |
| 2 | 61.7% / 61.7% | 61.1% / 60.7% | 60.0% / 61.4% | 57.8% / 58.7% | 56.0% / 56.6% |
| 3 | 57.1% / 55.5% | 54.8% / 54.3% | 53.9% / 54.1% | 51.7% / 52.1% | 49.4% / 48.6% |
| 4 | 48.8% / 47.9% | 47.7% / 47.4% | 46.1% / 46.3% | 45.8% / 44.7% | 43.3% / 42.8% |

Encoder Qwen/Qwen3-Embedding-0.6B; substitutes chosen by Qwen/Qwen3-1.7B-Base.