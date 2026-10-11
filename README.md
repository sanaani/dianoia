# Dianoia

> **Public copy.** This is a cleaned snapshot of the private research repo, published 10/11/2026 with fresh history. Row-level data files that contain or are derived from the Enron email corpus (decodes, test items, training dumps, figure data) and one internal planning-token report were removed. Write-ups, code and aggregate results are unchanged, so some READMEs name files that are not in this copy.

**Can a sentence encoder hold an idea rather than its words, and can the idea be put back into words?**

[Noema](https://github.com/sanaani/noema) asked whether the geometry of a theorem's proof states
knows which theorems are related. Dianoia asks the same kind of question of business prose, using
the Enron corpus. In Plato's divided line *noēsis* is direct understanding and *dianoia* is
discursive thought, the kind that works step by step through sentences. That is the object here.

## The analogy, and what it commits us to

| Noema (Lean) | Dianoia (Enron) | what the vector is |
|---|---|---|
| proof state | word, or token | one step; carries local vocabulary |
| lemma | sentence | one idea, encoded whole |
| theorem, as a centroid of its states | paragraph, as a centroid of its sentences | a composition of ideas |
| theorem statement, encoded alone | paragraph, encoded directly | the composition read at once |
| α-rename control (delete every name, keep the math) | **paraphrase control** (change every word, keep the idea) | the invariance that decides the reading |
| lexical baseline (identifier Jaccard) | word Jaccard; a word-centroid encoder | what "only words" predicts |
| citation graph, 2024 → 2026 | reply and thread structure | the answer key |

Two of Noema's findings are predictions for this port, stated before any encoder was run:

1. **The statement matched or beat the trajectory** (0.773 against 0.726 on phase 2's pairs, 0.812
   against 0.766 under encoder B). The analogue is that a paragraph encoded directly will match or
   beat the centroid of its sentence vectors. We expect the centroid to lose.
2. **Proximity tracked wording.** Off-the-shelf sentence encoders put a theorem nearer a
   word-matched decoy than its true relative (0.125 to 0.188). The analogue is that general
   sentence encoders will be fooled by *lexical twins*: a sentence with the same words and a
   different meaning. How badly is phase 1's question.

## Critical finding: the target is far analogies (10/07/2026)

The goal of the analogy work is **far** analogies: a source whose description shares no direct semantic relationship
with the problem, such as fluid shockwaves explaining phantom traffic jams, or ant colonies informing how a team
coordinates without a manager. Near analogies, where the problem and the source describe the same mechanism in
overlapping words, are not the target, because plain search over real documents already finds them.

What showed it (phases 10 and 11, `results/phase-10-analogy`, `results/phase-11-analogy-training`):

- The confirmed patent borrowings are mostly near. A pole-vault crossbar pin borrowed from a folding knife, and both
  abstracts say "spring-biased" and "pivot". Plain search with the problem as written finds the true source in its
  top 10 of 1,000 about 68% of the time: 85% for the quarter of pairs sharing the most words, 57% for the quarter
  sharing the fewest. That success is word and concept matching, not a leap.
- Training a small LLM on those pairs made search worse (52.5% against 70.5% for the same model before our
  training): it learned to invent one plausible source, which is usually wrong.

Consequences: a test or training set counts only if it is built from far pairs, chosen by a measure independent of
the search being tested; a result on near pairs says nothing about the goal. The product use is any "how" question,
mostly business problems, answered from other fields (military, organizational behavior, biology such as entomology),
with the model proposing several angles and reasoning over real sources that search returns.

The approach (Shaun, 10/07/2026): an encoder that "throws away everything that is not a far analogy and only matches
things that have functional relationships in the abstract sense", trained on far borrowings as written, with no
rewriting. Word-level changes cannot get there: generalizing "car" and "molecule" one word at a time meets only at
"physical object", and removing a field's average vector cannot turn "brake" into "density". What traffic and fluid
shockwaves share is how the parts act on each other, and an encoder learns that only from pairs that share it.

## Phases

Each phase directory under `results/` states its hypotheses and bars before the numbers, and its
caveats next to them. A closed phase's numbers never move; corrections are dated addenda.

- **Phase 0, corpus.** Authored, de-duplicated Enron paragraphs and sentences, sampled by seed.
- **Phase 1, ideas not words.** For each sentence, a paraphrase that changes the words and keeps
  the idea, and a lexical twin that keeps the words and changes the idea. An encoder that holds
  ideas puts the paraphrase nearer. Word Jaccard is the control and must fall below 0.5.
- **Phase 2, the round trip.** Encode a sentence, decode the vector to words without ever seeing
  the sentence, re-encode, and ask whether it lands within tolerance. The tolerance is set by the
  data, not by us: a round trip passes if it lands as close as a faithful paraphrase does.
- **Phase 3, paragraphs as theorems.** Centroid of sentences against direct encoding, on the
  reply and thread structure; the round trip one level up.

## Layout

`src/dianoia/` holds the library, `scripts/` the runnable steps, `results/phase-*/` the committed
outputs. Large artifacts (the Enron parquet, vectors, the corpus tables) live in `DIANOIA_DATA`,
by default `~/dianoia-data`, outside the repo.
