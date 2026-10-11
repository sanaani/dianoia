# Phase 7: the splitter on reworded input, and the pipeline end to end (10/03/2026)

Exploration, not a pre-registered test. 381 check units (150 email, 117 meeting, 114 chat), each in six versions:
original, stripped, disfluent, Sonnet paraphrase, re-split into different turns, and meaning changed ("move", the
control). Sonnet's splits of every version are the ceiling (`literal-labels.py --kinds`).

## 1. Splitter on reworded input (`remote-split.py score`)

Content agreement between the ideas split from the original and from each version (NLI, both ways):

| Version | Splitter (Qwen3-1.7B + LoRA) | Sonnet |
|---|---|---|
| stripped / disfluent | 0.863 | 0.847 |
| paraphrase | **0.803** | 0.816 |
| re-split turns | 0.886 | 0.872 |
| meaning changed (must be lower) | 0.484 | 0.478 |

The splitter handles paraphrased input as well as its teacher. On single-sentence email a meaning change drops
agreement to 0.11 against 0.74 for a paraphrase; on meetings and chats only one turn of many changes, so the drop
is diluted (0.68-0.77). The splitter keeps the input's wording: a paraphrase gives ideas in the paraphrase's words.
Coverage (0.81 vs Sonnet 0.94) and literalness (0.93) are unchanged weak spots.

## 2. End to end: text -> splitter -> encoder -> decoder (`e2e-prepare.py`, `remote-e2e.py`, `score-e2e.py`)

Arm R (`ablation/R1`) against SONAR as released. Ideas carry placeholders, one mapping per unit.

| | SONAR | Arm R |
|---|---|---|
| Sonnet's reference ideas covered, splitter alone | 0.809 | 0.809 |
| ... after encode and decode | 0.812 | 0.803 |
| source text entails decoded idea | 0.938 | 0.941 |
| decoded idea entails splitter idea (RoBERTa) | 0.979 | 0.987 |
| fluency gap (per-token loss, decoded minus input) | 0.001 | 0.002 |
| paraphrase version retrieves its own unit (of 381) | 0.969 | 0.971 |
| **paraphrase's worst idea matches better than the meaning-changed version's** | 0.186 | **0.512** |
| ... email only (single sentence, the clean case) | 0.073 | 0.673 |

Reading:
- **Text out is fine.** Encode and decode lose almost nothing (0.987 round trip, no fluency cost); the detail lost
  end to end is the splitter's coverage gap, not the encoder's.
- **Identity in the space is the weak link.** A paraphrased idea sits about as far from its original (worst-idea
  best match 0.62) as a meaning-changed idea does (0.62). Arm R improves on SONAR (0.51 vs 0.19 of units ranked
  correctly; 0.67 on email) but is far from reliable. Retrieval of the right unit is easy (0.97) because units
  differ in topic, so it says little.
- Multi-turn units understate the test: a paraphrase rewords every turn while the control changes one, so the
  worst-idea comparison is biased against the paraphrase there. Email is the clean number.
- An earlier run normalized vectors before decoding, which breaks SONAR's decoder; it was fixed and rerun.

Files: `splitter-versions.json`, `sonnet-versions.json`, `e2e.json`.

## 3. Arm P: literal-idea paraphrases with minimal-edit negatives (`lpara-data.py`, `build-lpara.py`)

Sonnet wrote, for 10,000 literal ideas from the splitter's training labels, 3 paraphrases (same speaker, act and
stance; different words) and 2 minimal edits (nearly the same words, a different idea). 1,000 items held out.
Arm P = arm C data + 80,939 new pairs; batch 20 (8 natural, 4 attitude-contrast, 5 literal-paraphrase, 3
uniform), 3,000 steps, seed 1 (`scripts/run-P.sh`). Compared with arm R (batch 16, 2,000 steps), so more than one
thing changed.

| | Arm R | Arm P |
|---|---|---|
| held-out literal paraphrase beats minimal edits | 0.804 | **0.917** |
| attitude contrast test | 0.871 | 0.887 |
| main test, cross-wording vs siblings (all) | 0.929 | 0.928 |
| ... email (phase 1) | 0.679 [0.636, 0.725] | 0.646 [0.600, 0.693] |
| ... MultiNLI | 0.885 | 0.908 |
| negation separation (all / email) | 0.916 / 0.627 | 0.906 / 0.574 |
| geometry round trip beats negation | 0.988 | 0.989 |
| **end to end: paraphrase beats meaning change (all / email)** | 0.512 / 0.673 | **0.577 / 0.767** |
| end to end: meeting / chat | 0.444 / 0.368 | 0.470 / 0.439 |
| end to end: coverage, idea round trip, fluency gap | 0.803, 0.987, 0.002 | 0.802, 0.987, 0.005 |

The targeted skill improved a lot on literal ideas and carried part of the way into the pipeline (email 0.67 ->
0.77). Email on the main test and email negation separation slipped, inside the intervals. Own-idea-among-all on
the literal-paraphrase test is about 0.53 for both arms by construction: the original and its first paraphrase are
both anchors there, and they are the same idea.

Data integrity: the first download of arm P's end-to-end file was corrupted (61 of 381 records unreadable), and
the contrast eval file had one stray fragment after its 1,800 complete records, which was removed. The
end-to-end step and the literal-paraphrase test were rerun on a fresh instance with sha256 checks on both ends; the
rerun reproduced the test (0.917) and its end-to-end file is the one scored. Downloads are now checksummed.

## 4. Arm S: the splitter's own paraphrase pairs (`remote-spara.py`, `build-spara.py`)

The splitter split 2,998 contrast-set emails (two duplicates of check units removed) and each one's Sonnet
paraphrases (20,986 texts). DeBERTa NLI aligned each original idea with its mutual-best paraphrase idea
(min of both directions >= 0.5): 6,072 pairs from 5,179 original ideas. Negatives are rule edits of the original
idea. The slanted versions' ideas were dropped as negatives after inspection: many were the same fact reworded or
with a detail dropped. Arm S continues arm P for 1,500 steps (lr 1e-4, batch 20: 6 natural, 3 contrast, 4
literal-paraphrase, 5 splitter-paraphrase, 2 uniform) on arm P's data plus 22,708 new pairs; 300 items held out
(`test-spara`). Fixed on the way: rule number edits could change a placeholder's index ("Person 1" -> "Person 4");
data built before 10/03 has some of these (still meaning changes).

| | Arm R | Arm P | Arm S |
|---|---|---|---|
| splitter-paraphrase test (held out) | - | 0.902 | **0.924** |
| literal paraphrase vs minimal edit | 0.804 | 0.917 | 0.917 |
| attitude contrast | 0.871 | 0.887 | 0.884 |
| main test, all / email / MultiNLI | 0.929 / 0.679 / 0.885 | 0.928 / 0.646 / 0.908 | 0.931 / 0.666 / 0.911 |
| negation separation, all / email | 0.916 / 0.627 | 0.906 / 0.574 | **0.923 / 0.650** |
| geometry round trip beats negation | 0.988 | 0.989 | 0.991 |
| **end to end: paraphrase beats meaning change, all / email** | 0.512 / 0.673 | 0.577 / 0.767 | **0.591 / 0.800** |
| end to end: meeting / chat | 0.444 / 0.368 | 0.470 / 0.439 | 0.462 / 0.447 |

Small gains everywhere it was aimed, nothing lost. Email end to end has n = 150, so 0.77 -> 0.80 is inside noise
by itself; the direction agrees with the held-out test and email negation separation.

**Where the rest of the gap is.** Split by whether the splitter cut the paraphrase into the same number of ideas
as the original:

| end to end, paraphrase beats meaning change | same count: R / P / S | different count: R / P / S |
|---|---|---|
| email (116 / 34 units) | 0.698 / 0.802 / **0.845** | 0.588 / 0.647 / 0.647 |
| meeting (47 / 70) | 0.511 / 0.617 / 0.596 | 0.400 / 0.371 / 0.371 |
| chat (61 / 53) | 0.410 / 0.557 / 0.541 | 0.321 / 0.302 / 0.340 |

Encoder training moves the units the splitter cuts the same way and does nothing for the units it cuts
differently. Most meeting units (70 of 117) are cut differently. The next limit is the splitter's segmentation
consistency, plus the measure's own bias on multi-turn units (one changed turn against every turn reworded).
All downloads checksummed (11 of 11 verified).

## 5. Splitter q4: trained on one-for-one paraphrases (`split-para-data.py`, `run-Q.sh`) (10/04/2026)

Sonnet reworded every splitter training unit (same speakers, same turns) and rewrote its reference ideas one for
one in the new wording: same number, same order. 6,789 of 6,975 units passed the exact-count check (the rest were
unparseable, miscounted or unchanged). q4 is q3's recipe trained on the originals plus these (13,764 units, 2 epochs).

Same idea count on the paraphrase as on the original (`count_paraphrase`), and the other splitter measures:

| | q3 | q4 | Sonnet |
|---|---|---|---|
| same count, email / meeting / chat | **0.773 / 0.402 / 0.535** | 0.747 / 0.385 / 0.474 | 0.760 / 0.359 / 0.404 |
| rewording invariance, email / meeting / chat | 0.736 / 0.821 / 0.873 | 0.760 / 0.799 / 0.872 | 0.744 / 0.834 / 0.891 |
| coverage, email / meeting / chat | 0.837 / 0.746 / 0.840 | 0.851 / 0.757 / 0.833 | 0.985 / 0.923 / 0.907 |
| end to end with S1: paraphrase beats meaning change | **0.591** | 0.588 | - |

No gain: q4 is a hair worse on segmentation and level end to end. The reason is in the Sonnet column. The
reference labeler itself cuts a paraphrase into the same number of ideas only 36-40% of the time on meetings and
chats, and q3 is already more consistent than it. Sonnet's count drifts by about one idea in seven to ten (mean
relative difference 8-17%): rewording legitimately merges or splits clauses. Exact count is too strict a target,
and the "different count" bucket in § 4 is mostly the units whose paraphrase restructured the most, not a
splitter defect training can fix. The remaining multi-turn gap is better attacked through the measure (match ideas
by alignment, allowing one-to-many) and through the encoder's handling of merged and split ideas, not through
segmentation. q3 stays the splitter. All downloads checksummed (6 of 6 verified).

## 6. Arm M: the encoder on merged ideas (`merge-data.py`, `build-merge.py`, `run-M.sh`) (10/04/2026)

§ 5 says rewording legitimately merges and splits ideas, so the encoder should place one idea that states two
things near each of the two. For one adjacent same-speaker idea pair per splitter training unit, Sonnet wrote 2
merges (one literal idea stating both, in different words) and 2 one-word edits of the first merge. 4,851 items;
every 10th held out (486). Each edit is attributed to the idea whose words it replaced (difflib; x 3,625, y 4,449,
unclear 1,628). Rows: X and Y are "same" as both merges and "different" from the edit of their own part; each merge
is "same" as X, Y and the other merge and "different" from both edits. 66,968 training rows, 7,627 test rows, no
test text trained on. M1 continues S1 for 400 steps (lr 1e-4, batch 24, 8 merge groups per batch; 12.9 GPU minutes).

Held-out merge test (`test-merge`), share of comparisons won:

| | S1 | M1 |
|---|---|---|
| X or Y as anchor: merge beats the edited merge (619 anchors) | 0.945 | 0.947 |
| merge as anchor: the other merge beats a one-word edit (750) | 0.649 | **0.720** |
| merge as anchor: X or Y beats a one-word edit (750) | 0.406 | 0.447 |
| cross siblings (score-encoder), mean | 0.472 | 0.498 |

The first row was already solved. The third is not a fair target: an edit that changes one word of a merge keeps
most of what the merge says, while X carries half. The second is the real one, a long paraphrase against a
near-identical meaning change, and M1 moves it 7 points; it is the long-paraphrase weakness of § 3 again. Training
loss rose over the last 100 steps (InfoNCE 0.13 -> 0.49), so the lr may be too high for this mix. The run was
stopped before its end-to-end step: the end-to-end measure aligns ideas one to one, so it cannot credit a merge;
the measure comes first. All downloads checksummed (7 of 7 verified).

## 7. Is a merged idea the sum of its parts? (`merge-additive.py`, laptop CPU) (10/04/2026)

On the 972 merge anchors of `test-merge`, encoded with S1 (cosine to the merge):

| compared with the merge | cosine |
|---|---|
| normalized sum of its two parts, X + Y | 0.859 |
| the other merge (a paraphrase) | 0.858 |
| the closer part alone | 0.819 |
| the closer one-word edit | 0.871 |

The sum of the parts is as close to the merge as a paraphrase of it, and the sum is closer to the true merge than
to either edit 89% of the time. So a one-to-many match can be done in vector space: let each idea also match the
normalized sum of two adjacent ideas on the other side (`e2e-align.py`, which caches the vectors). Its first run on
the 7,320 end-to-end ideas was stopped by laptop memory pressure before it finished; not yet rerun.

## 8. A fairer end-to-end measure (`e2e-align.py`, laptop CPU, cached vectors) (10/04/2026)

`e2e-align.py` reran the S1 + q3 end to end on the laptop and reproduced the GPU number (0.591 one to one; the
laptop encodes in float32, the GPU in bf16). Letting an idea also match the sum of two adjacent ideas (§ 7) did
not help: 0.583 overall, +0.013 on the different-count units, -0.022 on the same-count ones. Merges are not
what loses the multi-turn units.

What does: the worst-idea rule. Median worst match, paraphrase / meaning change: email 0.69 / **0.49**, chat
0.56 / 0.61, meeting 0.56 / 0.56. S1 separates a literal paraphrase from a minimal edit 91.7% of the time
idea by idea (§ 3), chat and meeting ideas included, but a multi-turn paraphrase rewords ten ideas and a meaning
change touches one, so the worst of the ten often drifts further than the one edit. The rule penalizes units for
having many ideas.

**Targeted measure.** Take the original ideas the meaning change touched, chosen by text (absent word for word
from the meaning-changed version's ideas; median 1 on email, 3 on chat and meetings, because the splitter
re-cuts the neighboring ideas too), and ask whether the paraphrase covers them better than the meaning-changed
version does. Choosing by text keeps the selection from favoring either version; the set also holds ideas the
splitter merely re-cut, so this is a floor.

| paraphrase beats meaning change | worst idea (old) | targeted (new) |
|---|---|---|
| all | 0.591 | **0.734** |
| email | 0.800 | 0.800 |
| meeting | 0.462 | 0.713 |
| chat | 0.447 | 0.663 |
| same / different idea count | 0.710 / 0.420 | 0.777 / 0.671 |

Most of the multi-turn gap was the measure. Ideas are treated as clouds from here on (the splitter need not cut a
paraphrase the same way): the targeted measure is primary, the worst-idea one is kept for continuity. The
different-count units are no longer a separate failure (0.67 against 0.78). Chat is the weakest (0.66).

Bug noted, not yet fixed: in 170 paraphrases and 182 meaning-changed versions the turn text repeats the speaker
prefix ("A: A: ..."), the same Sonnet echo fixed in § 5's data. Units with and without it score alike, and it hits
both versions, so it does not bias the comparison; it gets fixed whenever the versions are re-split.
