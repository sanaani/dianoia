# Phase 4: idea objects

Pre-registered in `PREREGISTRATION.md` (10/01/2026 addenda: "idea objects", "stage 1 on idea objects", "after
the idea-object check").

## 1. The check: is Sonnet's structure stable? (`ceiling.json`, `ceiling.log`)

150 Enron sentences, 117 AMI/ICSI meeting excerpts and 114 Ubuntu IRC threads, each in seven versions, all
extracted blind in one shuffled list.

- **Punctuation, fillers and split turns don't matter.** In every medium, the stripped, disfluent and split
  versions agree with the original about as well as an identical copy does.
- **The full structure failed its rule** (16 of 84 cells). Relation attachment is unreliable even on identical
  text (F1 0.67-0.78). A paraphrase shifts modality (email 0.66) and event segmentation (email event F1 0.75).
- So the stage-1 target was reduced: relations dropped, modality in 5 classes. Training was therefore exploratory.

## 2. Stage 1: SONAR (LoRA r=32) trained to decode the reduced structure (`idea1-step*.json`)

- **Data:** 3,303 training units (Sonnet targets; extraction of the planned 7,000 stopped at the usage limit).
- **Run:** 4,000 steps at an effective batch of 16 on an A10G, 3.1 h, about $3.50.
- **Held-out:** the check's units, never trained on. The table is step 4,000; the bar for each row is in the
  addendum.

| | email | meeting | chat |
|---|---|---|---|
| Fidelity: event F1 vs Sonnet's | **0.93** | 0.67 | 0.52 |
| Fidelity: coarse act / polarity | **0.96 / 0.99** | 0.81 / 0.94 | 0.69 / 0.90 |
| Invariance, stripped: event F1 | **0.98** | 0.83 | 0.68 |
| Invariance, disfluent: event F1 | **0.99** | 0.85 | 0.71 |
| Invariance, split: event F1 | **0.99** | 0.88 | 0.77 |
| Invariance, paraphrase: event F1 / act | 0.72 / 0.94 | 0.67 / 0.87 | 0.53 / 0.78 |
| Sensitivity (move changes the decode) | **0.97** | 0.67 | 0.65 |
| **Round trip to words** (intent check, bar 0.80) | **0.33** | 0.07 | 0.00 |

On single email sentences the model reads structure reliably, and ignores punctuation and fillers. One
1024-d vector does not hold a 6-8 turn excerpt of about 8 events. The round trip fails everywhere.

## 3. Why the round trip fails: the target, not the model (`control-*-words.json`)

The same round trip, starting from **Sonnet's own** structure instead of the model's decode:

| Structure given to the writer | email | meeting | chat |
|---|---|---|---|
| Reduced (what stage 1 learned) | 0.42 | 0.21 | 0.06 |
| + speaker per event | 0.49 | 0.27 | 0.17 |
| + relations and fine modality | 0.48 | 0.36 | 0.26 |
| The trained model (reduced) | 0.33 | 0.07 | 0.00 |

On email the model gets about 80% of what its target allows. The target allows under half.

Per-field failures on email: mostly **act** and **action**. The structure drops **stance**:
- "I'm not sure I have clothes" becomes "I don't have clothes";
- "I thought you'd appreciate this" becomes a request;
- "I assume", "believe … probably" lose their hedges.

It also drops **manner and comparison** ("rather than …"), **adverbial time** ("yesterday") and **vocatives**
("Pat, …"). The cause is the extraction rule that folds framing verbs (think, believe, not sure) into one
modality slot: it discards exactly the intent the check grades.

## Next

1. Redesign the event to keep these:
   - stance as its own layer: holder, attitude, certainty, after ISO 24617-2's qualifiers and FactBank's
     factuality values;
   - PropBank's ARGM modifiers (manner, extent, comparison, location, time);
   - the addressee;
   - the speaker.
2. Re-run the **control first** (Sonnet structure → words). The new target must reach ≥ 80% before anything
   is trained on it.
3. For meetings and chat, encode per turn, or per small window, rather than one vector per excerpt.
