# Phase 6: public data sample (10/02/2026)

`scripts/data-sample.py`, seed 6, NLI model `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, bar 0.9,
lowercased. 500 human-labeled positive pairs per source and 200 contradictions each from SNLI and MultiNLI.
No LLM was used. Scored pairs are in `dianoia-data/work/data-sample/pairs.jsonl`.

| Source | Forward entailment | Both directions | Detail dropped (mean) | Extreme generalization | Contradiction confirmed |
|---|---|---|---|---|---|
| SNLI entailments | 87% | 4% | 69% | 39% | 80% |
| MultiNLI entailments | 93% | 23% | 62% | 31% | 98% |
| MRPC paraphrases | 30% | 14% | 34% | 0% | n/a |
| STS-B ≥ 4.5 | 70% | 51% | 32% | 3% | n/a |

Readings:
- SNLI entailments are mostly heavy generalization ("A few children are rollerblading across a track" ->
  "There are kids outside"). MultiNLI is less extreme and covers more genres, including speech.
- The NLI model is unreliable on MRPC: it rejects human-judged paraphrases that differ in attribution form
  or quote placement ("said Janet Levine" vs "Levine said"). For MRPC the human label is trusted over the model.
- The model was trained on SNLI and MultiNLI, so its forward scores on them are flattering.
- Both-direction pools are about 8k (SNLI) + 30k (MultiNLI) pairs, so arm B cannot reach 50k from these sources.
