# Phase 10 hypotheses: separating structure from topic (written 10/05/2026, before any test)

Shaun's idea (10/05/2026): abstraction is a move along word frequency and domain spread. Domain words ("farmer",
"wheat", "clover", "rotate") are rarer in general English and concentrated in one field. Their abstract stand-ins
("operator", "resource", "alternate") are common and spread across fields. The verb carries most of the analogy. Qwen3's
weights should already encode the relationship between a specific word and its abstract stand-in.

These are fixed before the data is built. Results are reported against them as written, including the ones that fail.

## Measures

- **Frequency:** general-English word frequency (`wordfreq`, Zipf scale).
- **Domain spread:** entropy of a word's use across the 9 CPC sections of the patent corpus. High = used in every field.
- **Historical recovery:** given a patent's abstract, rank the earlier patent it cites from a distant CPC section among
  1,000 candidates from distant sections, published before it. Score: share with the true one in the top 10. Pairs split by
  time; only applicant citations (not examiner ones) count as borrowings.
- **Domain leakage:** accuracy of a linear classifier guessing a vector's CPC section (chance = 1/9 = 11%).
- **Abstraction ground truth:** WordNet hypernyms (e.g. wheat -> grain -> plant -> organism).

## Hypotheses

**H1. Abstraction climbs frequency and spread.** Along WordNet hypernym chains, the more general word has higher frequency
and higher domain spread in more than 70% of steps, for both nouns and verbs.
Falsified if the share is 60% or less.

**H2. Verbs carry the analogy.** In real cross-domain borrowings, the citing and cited abstracts are closer on their verbs
than on their nouns, relative to random distant-section pairs. Recovery from verbs alone beats recovery from nouns alone.
Falsified if nouns-only recovery is equal or better.

**H3. Qwen3 holds a shared abstraction direction.** In Qwen3's hidden states, the step from a specific word to its
hypernym is a consistent direction that transfers across domains. A linear map fit on hypernym pairs from some CPC
sections predicts the hypernym (top 10 among candidates) for words from held-out sections well above a frequency-only
baseline that always picks the most common candidate.
Falsified if it does no better than that baseline on held-out sections.

**H4. Role replacement (method 2) recovers borrowings better than raw text and leaks less domain.** Abstracts rewritten
by replacing domain words with in-context, higher-frequency, wider-spread substitutes (chosen with Qwen3) beat the raw
abstracts on historical recovery, and lower domain leakage.
Falsified if recovery does not improve (95% intervals overlap zero gain).

**H5. There is a best level of abstraction.** Sweeping how far words are abstracted (the frequency and spread threshold),
recovery rises and then falls: too specific matches on topic, too general ("an entity acts on a thing") keeps no
structure. Peak at an intermediate level.
Falsified if recovery rises monotonically to the most abstract setting.

**H6. Verbs need abstracting, but far less than nouns.** (Added 10/05/2026, still before any test, from Shaun: a verb
like "harvest" has a home domain and must be abstracted, but only modestly, to "gather", while the nouns are abstracted
much further.) H5's sweep becomes two dials, one for nouns and one for verbs. Predictions: (a) recovery with some verb
abstraction beats recovery with verbs left as written; (b) at the best setting, verbs are abstracted by much less than
nouns: fewer WordNet hypernym steps, and a smaller rise in frequency and spread. (c) The verbs that need it most are the
ones with low domain spread ("titrate"); verbs already spread across fields ("harvest" energy, data) need less.
Falsified if leaving verbs as written is as good as any verb setting, or if the best verb setting is as aggressive as the
best noun setting.

## Order and costs

H1 runs on the laptop (word counts, WordNet). H2, H4 and H5 need abstracts encoded (one GPU, a few dollars). H3 needs
Qwen3 hidden states for a few thousand words (minutes on a GPU). No LLM grades anything: the answer key is the citation
record and WordNet.

## Known weaknesses, stated in advance

- Citations measure influence, not whether the borrowed mechanism worked.
- Patent language is legal style ("said member"), which distorts word frequencies; frequencies are measured on general
  English and spread on patents, and the two are reported separately.
- WordNet is a 1990s general-English hierarchy; technical vocabulary is often missing from it. Coverage is reported.
