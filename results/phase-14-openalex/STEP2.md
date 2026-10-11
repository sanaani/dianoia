# Phase 14 step 2: how many far pairs are real borrowings? (10/09/2026)

Decision file: `step2.json` (pipelines/phase14 rule decide). Rubric: `RUBRIC.md`. Readings by Claude, blind (numbers
only), each with the 10 gold pairs shuffled in.

| Reading | Gold agreement | Real (C or G) | C | G | A | B |
|---|---|---|---|---|---|---|
| 50 random far pairs | 10/10 | 40% | 0 | 20 | 4 | 26 |
| 50 strict pairs (Sonnet v3p C, borrow ≥ 4, then v4p C or G) | 10/10 | **86%** | 10 | 33 | 2 | 5 |

**Step 2 passes:** the strict reading counts and is above 70% real. Sonnet: of 4,979 far pairs labeled (21 replies
did not parse), v3p gave 389 C, 131 of them with borrow ≥ 4; v4p kept 109 as C or G. So 2.2% of far pairs are strict.

What the readings show:
- **Most far pairs are not borrowings.** Half of the random far pairs are background citations (a climate report
  cited by a banking paper), and the real ones are all general methods: statistics textbooks, regression tests,
  clustering and optimization algorithms.
- **The strict rule finds the real borrowings, but they are mostly methods.** 33 of the 43 real strict pairs are G.
  The 10 C pairs are genuine: dendritic neuron logic → a bankruptcy classifier; "dear enemy" recognition in
  salamanders → cooperation after conflict in innovation ecosystems; learned helplessness in dogs → innovation
  fatigue; ecological resilience → shopping centers and firms (four pairs); a chemical dative bond → the stability of
  corporate venture capital; T-cell plasticity → leadership; fish-shoal interaction rules → enterprise architecture.
- **Caveat on the gold check.** The gold pairs were labeled on 10/08 and re-read 20 hours later, as the rule requires,
  but several were recognizable from the first reading. The 10/10 agreement is weaker evidence than a fully blind
  re-read; it is reported as measured, not relabeled.

For step 3, the test set as written (300 strict pairs) will be about three-quarters method citations. Sonnet v4p's C
or G label is kept on every test pair, so top-10 can be reported separately for C and G pairs (reported, not tested).
