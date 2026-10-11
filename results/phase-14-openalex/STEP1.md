# Phase 14 step 1: the pairs (10/08/2026)

Run by `pipelines/phase14` (config.yaml as committed in 50c50ab). OpenAlex, 10/08/2026.

| | count |
|---|---|
| Articles sampled (three seeded draws of 10,000, duplicates dropped) | 29,376 |
| Articles with at least one pair | 16,065 |
| Cited works outside the Social Sciences, with an abstract | 55,976 |
| Pairs (both abstracts English, 30 words or more) | 69,219 |
| Below phase 12's overlap median (0.09) | 33,206 (48%) |
| **Far: below both medians (cosine 0.6746)** | **15,536 (22%)**, from 8,377 articles |

Far pairs by cited field: Mathematics 3,759; Computer Science 3,315; Engineering 2,115; Medicine 1,370; Environmental
Science 1,167; Health Professions 878; Agricultural and Biological Sciences 661; Neuroscience 503; no field 399;
others under 320 each.

The cosine pass ran on one AWS g5.xlarge (on demand, capped at 1 hour); downloads were checksummed and the instance
terminated.

**Gold pairs (Claude, 10/08/2026, blind, before any other reading):** 8 B, 1 A, 1 G, no C. Most far pairs read as
background citations (a climate report cited by a banking paper). This is 10 pairs and decides nothing; the counted
readings are step 2's.
