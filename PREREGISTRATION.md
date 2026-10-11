# Pre-registration

Written 09/30/2026, before any intent head was trained or any variant generated. Nothing in this file
changes after the first result is read; a change is a dated addendum at the bottom that says what
moved and why.

## What is encoded

**Intent**, defined by speech act theory: a force (request, commit, inform, question, propose,
approve, reject, thank, other) applied to a proposition. The proposition is a verb bound to its
object, the roles of the parties, and any material quantity. Names are not part of intent.

- **Names are variables.** A name of a particular person, organization or thing is replaced by a typed,
  numbered placeholder, numbered by first appearance within the unit (`[PERSON_1] talked to
  [PERSON_2]. [PERSON_2] agreed.`). This is α-equivalence, Noema's rename control applied to prose.
- **Bindings live beside the vector, never in it** (`PERSON_1 = David`). Decoding restores them.
- **First and second person stay as they are.** "I" and "you" carry the speech act.
- **A name that has become a concept stays literal.** An entity in 50 or more distinct messages of the
  30,000-message corpus is kept as itself (California ISO, FERC).
- **Common nouns and verbs are content.** "the ball", "the contract", "talked to" are the idea.

## The architecture (decided 09/30/2026)

SONAR's text encoder, frozen, gives a 1024-d content vector. A trained **intent head** maps it to a
256-d intent vector whose prefixes are nested (Matryoshka): the first 64 dimensions hold the core
(the speech act, the verb and object, typed roles), and later blocks add modifiers and detail.
Decoding maps an intent vector to the SONAR vector of its **canonical sentence** (placeholders, no
modifiers) and runs SONAR's decoder. The output is lossy by design; the bindings put the names back.

## Variants

Every source sentence gets, from Haiku, checked mechanically before use:

| keep (should stay put) | move (should move far) |
|---|---|
| paraphrase | speech act changed, same content |
| consistent rename, same type | roles swapped (where the verb is not symmetric) |
| register or politeness changed | negation |
| modifiers dropped (core dims only) | modality changed (will → might) |
| the canonical form | verb or object changed |
| | an entity's type changed (Frank → FERC) |
| | a quantity changed materially |

## Split

Mailboxes are split by seed `20260930`: 20% of mailboxes are **test**, never seen in training, never
used to tune anything. Every tier's numbers come from test mailboxes only.

## Tier 1, minimum: the vector holds intent

A **win** is a keep variant nearer the source than a move variant. Score: paired win rate over
(keep, move) pairs, with a 2,000-draw bootstrap interval over source sentences.

| check | bar |
|---|---|
| word Jaccard, as a control | **below 0.5**, or the materials are wrong and the phase stops |
| intent head, overall | **≥ 0.90** |
| intent head, worst move kind | **≥ 0.80** |
| intent head minus raw SONAR | **≥ +0.10**, interval above 0 |
| natural pairs (same intent in two real Enron emails, not Haiku's) | **≥ 0.80** |

## Tier 2, target: the round trip

Encode, decode to canonical words blind, re-encode, measure in intent space.

| check | bar |
|---|---|
| lands within tolerance (tolerance = median intent cosine of a paraphrase) | **≥ 80%** of test sentences |
| structured intent check (speech act, verb and object, roles, quantities) by an LLM, 50 audited by hand | **≥ 80%** fully match |
| names restored from bindings | **100%**; a miss is a bug |
| the loss is in the words | median word Jaccard with the source **< 0.4** while the checks above pass |
| truncation to 64 / 128 / 256 dims | reported; intent expected to survive at 64 |

The landing is also scored by an encoder the decoder never searched against.

## Tier 3, stretch: intent predicts the conversation

- **Reply prediction.** Rank an email's true reply among emails from the same mailbox and week. Intent
  vectors must beat word Jaccard, raw SONAR and subject-line match, interval above 0.
- **Speech-act probe.** A linear probe on intent vectors recovers email act labels better than on raw
  SONAR vectors.

## Readings fixed in advance

- Wins on Haiku's variants, fails natural pairs: **we learned Haiku's style**, not intent.
- Tier 1 passes, tier 2 fails: the vector holds intent; **the decoder cannot read it back**.
- Tiers 1 and 2 pass, tier 3 fails: the geometry is real but **predicts nothing words don't**.

Every number is reported beside word Jaccard, raw SONAR, the best off-the-shelf encoder and a
shuffled-vector control. Negative results are reported as prominently as positive ones.

## Addenda

**09/30/2026, before any variant was generated.** The literal-name rule counted 196 entities in 50 or
more messages, and 58 of them are PERSON entities, mostly bare first names (Jeff, Mark, John, Steve,
Mike, Bill). A recurring first name is many different people, not one name that became a concept, so
keeping it literal would defeat the rule's own purpose. Changed: PERSON entities are never literal;
the 50-message rule applies to organizations, places, products, laws and events only. Nothing had been
generated or scored when this changed. The cost: public figures who did work as concepts (Gray Davis,
Ken Lay, Bush) become variables too. `results/phase-0-corpus/literal-entities.json` keeps the
original count; the list the generator uses is filtered from it by label.

**09/30/2026, after training on train mailboxes only; test mailboxes still unscored.** The frozen
head missed badly on validation (train-mailbox items held out by mailbox), so the fallback agreed
under decision 2 was taken: fine-tune an encoder end to end. The evidence, all on validation:

| model | val win rate |
|---|---|
| raw SONAR / minilm / gtr / Qwen3 / e5 / bge | 0.32 / 0.40 / 0.38 / 0.43 / 0.46 / 0.48 |
| head on frozen SONAR (as pre-registered) | 0.704 |
| head on SONAR, more regularisation | 0.698 |
| head on SONAR + bge + Qwen3 + e5 | 0.758 |
| head on all six | 0.752 |
| learning curve of the four-encoder head at 25 / 50 / 100% of fit data | 0.693 / 0.727 / 0.758 |

About +0.03 per doubling of data puts 0.90 near 30x the data, so the frozen vectors are the limit,
not the head. Two fine-tuned encoders are trained with the same objective: bge-base (all weights) and
SONAR's encoder with LoRA. The decoder path is unchanged (the lift still targets frozen SONAR). Every
configuration tried is listed here; the one scored on test is the one with the best validation rate,
chosen before the test mailboxes are scored. The tier bars do not move.

**09/30/2026, before any test score was read.** Reading fine-tuned bge's validation losses showed that a
large share of Haiku's "move" variants do not change intent: a synonym verb (find → retrieve), a
passive voice labeled a role swap, "hourly" → "every hour on the hour" labeled a quantity change. The
encoder ranks those near the source, correctly, and is scored as wrong. So a second, independent
Haiku pass (`scripts/judge-variants.py`) labels every validation and test variant same or different
intent, seeing the source and its variants shuffled, unlabeled, and with no encoder or score. A
variant whose judged label disagrees with its intended role is **disputed**. Tier 1 is reported twice,
on the original materials and with disputed variants removed, and the verdict table says which is
which. The judge applies to every model equally, so the filter cannot favor one. The bars do not move.

**09/30/2026, after the judge ran and before any test score was read.** The judge disputed 17.9% of
4,995 validation and test variants (`results/phase-1-intent/judged-summary.txt`). For two keep kinds
its disagreement is with the definition, not the variant: it called 76.9% of **renames** a different
intent (another company is another fact) and 53.3% of **dropped modifiers** a different intent (it judged
the full intent, not the core). The pre-registration fixes both as keeps: names are variables, and a
dropped modifier keeps the core. Overriding that through a filter would change the definition after
the fact, so those two kinds are exempt from the judged filter and keep their pre-registered role; every
other kind drops its disputed variants. Both dispute rates are reported beside the results.

**10/01/2026, after phases 1 and 2 were read; before any of what follows was trained.** Phases 1 and 2
showed that training for discrimination gives a vector that detects intent-changing edits but cannot
be read back (round trip 17%, intent check 12%). The goal was always the round trip, so the plan is
reordered. Nothing below changes a closed phase's numbers.

1. **Tier 2 becomes the primary objective.** Its bars are unchanged: ≥ 80% land within paraphrase
   tolerance, ≥ 80% pass the field-by-field intent check, median word Jaccard < 0.4 while they pass.
   Tier 1 becomes a constraint: negation and modality separation may not fall below 0.90.
2. **Our own encoder-decoder from permissively licensed parts replaces SONAR**: a trainable bge-base
   encoder to a 256-d vector with nested 64/128 prefixes, and Qwen3-0.6B (Apache-2.0) with LoRA reading
   the vector as soft prefix tokens. Trained end to end with a token-level loss through the decoder.
3. **Stages, each gated:**
   - *Stage 0, plumbing:* a plain autoencoder on pool sentences (train mailboxes only) rebuilding the
     original sentence. **Bar to continue: ≥ 80% of held-out test-mailbox sentences pass the
     field-by-field intent check.** Failing it means the architecture, not the idea, is wrong.
   - *Stage 1, the bottleneck:* the target becomes the canonical sentence; keep variants must decode
     to the same canonical form; move variants stay separated. Scored on the Tier 2 bars.
   - *Stage 2, truncation:* decode from 64 / 128 / 256 dims; reported, no bar.
4. **Data:** the training set is judged as validation and test were; Sonnet replaces Haiku for the
   judgment-heavy kinds (verb/object, roles, quantities); natural pairs become exploratory; a human
   label check of 50-100 items is requested from the project owner.
5. **New controls:** a random-vector noise floor beside the retrieval floor and the content ceiling.

**10/01/2026, after stage 0 failed its gate (25% on 169 checked decodes) and before anything below ran.**
Stage 0 is rerun at scale, every lever at once: ~550k Enron sentences (all eligible messages from fit
mailboxes) plus ~1M general English sentences from Wikipedia; bge-large encoder; Qwen3-1.7B decoder with
LoRA; 32 soft tokens; a 1024-d vector nested at 64 / 128 / 256 / 512 / 1024; ~40,000 steps on an L40S.
A pilot reports the intent check at 2k, 5k and 10k steps; the full run continues only if that curve is
rising toward the bar, and stops as soon as the gate passes on held-out test-mailbox sentences. The
gate and every Tier 2 bar are unchanged; the 256-d prefix stays the target dimension, and the larger
ones are reported to locate where reconstruction falls off. From here all grading and judging run on
headless Sonnet instead of Haiku. Budget raised by the project owner to $30 total.

**10/01/2026, stage 0 at step 7,500 of the scaled run.** Sonnet-graded held-out intent match was 13%,
16.5% and 17% at steps 2,500, 5,000 and 7,500: flattening, so the run was stopped under the pilot rule.
A diagnostic on the step-7,500 checkpoint (`results/phase-3-own-ae/diag-step7500-*`) found training
sentences rebuilt no better than held-out ones (18.5% Enron trained, 17% held out; loss 1.58 vs 1.64):
the model under-fits, it does not memorise. The likely cause is the encoder: bge is trained for
retrieval, to discard surface detail, and at its learning rate barely changes. The next pilot replaces it
with the decoder's own model (Qwen3-1.7B with a separate LoRA, summarising at the final token; the
in-context-autoencoder design), still one 1024-d vector per sentence. It is compared to the bge run at
the same checkpoint (step 2,500: 13%) and continues only if clearly better. Sonnet regraded the first
stage-0 run at 16.6% (Haiku had said 25%); Sonnet is stricter, and the bars do not move.

**10/01/2026, SONAR stage 1, before any corrected checkpoint was read.** Two defects found at step 1,500
of the first SONAR stage-1 run (`results/phase-3-own-ae/sonar1-step1500.*`, intent 85.7%, word Jaccard
0.778), which was then stopped:

1. The phase-1 canonical forms (Haiku, phase 1) were near-copies of their sources (median Jaccard 1.0
   with names restored), so the model was being taught and graded to copy. All 2,388 phase-1 items were
   re-canonicalized by Sonnet with the stage-1 prompt (`canonical-phase1.json`, 2,381 kept).
2. Even so, a sound canonical form keeps the content words: the new targets share a median 0.845 of
   their words with the source. The bar "median word Jaccard < 0.4" is therefore unreachable by design
   and cannot show that the loss is in the words. It stays in the verdict table, reported as such.
   What does show it is the **invariance** test the stage-1 plan already named (keep variants decode to
   the same canonical form): a paraphrase shares a median 0.217 of its words with the source, so a
   paraphrase decoding to the same idea as its source is the loss being in the words. It is now scored
   by the same Sonnet intent check, between the source's decode and each variant's decode.

**10/01/2026, idea objects, before any structure was extracted.** SONAR stage 1 (sentence targets) was
stopped at step 6,000: intent 75-81% against the 80% bar, and paraphrase invariance flat at about half,
because a canonical sentence keeps its source's wording. A fixed frame failed as a target too: Sonnet
filled its open slots (verb, object) inconsistently across paraphrases (25% and 13%) while its closed
slots held (act 93%, negation 95%). And a single frame cannot hold several events. The target becomes
an **idea object** built from established linguistics (`src/dianoia/ideas.py`): a variable number of events,
each with an ISO 24617-2 dialogue act, a PropBank-style predicate and arguments, polarity, modality and
time; SDRT relations between events (STAC labels, coarsened to the PDTB level-1 senses plus Interaction);
a center (the RST nucleus of the whole); and a one-sentence gist. Events are found by meaning, not
punctuation, and the medium is not assumed to be email.

Before training on it, a check (`scripts/idea-data.py`, `scripts/idea-ceiling.py`). Units: 150 Enron test
sentences, 120 AMI/ICSI meeting excerpts (QMSum release), 120 Ubuntu IRC chat threads (Kummerfeld et al.
2019), each in seven versions: original, copy, paraphrase, stripped (no punctuation or capitals), disfluent
(fillers, repeats, false starts), split (the longest turn broken across two turns) and move (one turn's
meaning changed). All versions are extracted by Sonnet in one shuffled list. Events are aligned by content
similarity (bge-base cosine ≥ 0.7, Hungarian, turn-aware). **Bars, for each of paraphrase, stripped,
disfluent and split, in each medium (12 cells), against the original:** event F1 ≥ 0.85; coarse act ≥ 0.85;
polarity ≥ 0.90; modality ≥ 0.80; relation attachment F1 ≥ 0.70; coarse relation sense ≥ 0.75; center ≥ 0.70.
**Sensitivity:** in each medium, the move version differs at the edited turn in ≥ 80% of units. Predicate
and argument agreement are reported, not barred: the open content is expected to be a continuum, carried
by the vector, as the frame check showed. The copy version is the extractor's own noise. **Decision:** the
structure becomes the stage-1 target if the closed layers (act, polarity, modality) pass in all 12 cells,
the other layers in at least 9, and sensitivity in all three media. Otherwise the failing layers are dropped
or redesigned before anything is trained.

**10/01/2026, stage 1 on idea objects, before any checkpoint exists.** If the check above passes, SONAR's
encoder and decoder (LoRA r=32) are trained to decode each training unit's idea object, serialized, from
any of its versions: original, stripped, disfluent, split, or combinations, all sharing the one target
(`scripts/remote-idea.py`). Targets are Sonnet's objects for about 7,000 training units disjoint from the
check's: Enron fit mailboxes (1-3 sentences), QMSum's train and validation meetings, IRC train threads
(`scripts/idea-targets.py`). Paraphrases are never trained on, so paraphrase invariance is a true held-out
test. Scored on the check's units (`scripts/score-idea.py`), per medium:
- **fidelity**, the object decoded from the original vs Sonnet's: event F1 ≥ 0.75, coarse act ≥ 0.80,
  polarity ≥ 0.90, modality ≥ 0.75;
- **invariance**, the decode from each keep version vs the decode from the original: event F1 ≥ 0.80, and
  each closed layer no more than 0.05 below Sonnet's own consistency on the same pairs;
- **sensitivity**: the move version's decode differs at the edited turn in ≥ 70% of units;
- **round trip to words, the goal**: Sonnet writes the unit back out from the decoded object alone, and the
  intent check (`score-tier2.CHECK` made medium-neutral: same acts in order, same actions and objects with
  nothing material missing or added, same roles, same quantities) passes on ≥ 80% of units.
If the check passes only in part, the failing layers are dropped from the serialization before training,
and the corresponding bars with them.

**10/01/2026, after the idea-object check (`results/phase-4-idea-objects/ceiling.json`), before stage 1
trained.** By its pre-registered rule the full structure **did not qualify**: 16 of 84 barred cells
failed. Modality failed two paraphrase cells (email 0.66, chat 0.78), so the closed layers did not pass
in all 12. Relation attachment failed 8 cells and event F1 failed 5. Three findings decide what happens next:
1. **Punctuation, fillers and turn splits do not matter.** Stripped, disfluent and split versions agree
   with the original about as well as an identical copy does, in every medium and on every layer.
   Example: meetings, coarse act, copy 0.93 vs stripped 0.92, disfluent 0.92, split 0.93. So the
   structure is found by meaning, not punctuation.
2. **Much of the failing is the extractor's own noise.** On identical text, Sonnet's relation attachment
   F1 is 0.67-0.78 and meeting event F1 is 0.85. No target can be more consistent than its labeler.
3. **Paraphrase moves modality and splits events differently.** Email paraphrase event F1 is 0.75 and
   modality 0.66. Coarsening modality to five classes (none, uncertain, must, can, will) reaches only
   0.71 on email paraphrases. That is post hoc, and recorded as such.

As the rule requires, the failing layers are **dropped or redesigned** before training. **Relations are
dropped** from the stage-1 target. **Modality is coarsened** to the five classes (it still fails the
email paraphrase cell, and is kept because negation-like distinctions such as must vs can carry intent).
**Events, acts, polarity, arguments, time and the center stay.** Stage 1 therefore trains on this reduced
target and is **exploratory** with respect to the check. Its bars (stage-1 addendum above) are unchanged
for every layer that remains. The round trip to words remains the deciding bar.

**10/02/2026, the idea splitter, before any splitter output was read.** The project owner set the unit:
one idea is roughly a sentence or long phrase, rewritten hyper-literally (pronouns replaced by what they
refer to, the speaker and their stance stated, every material detail kept) before it is encoded. Ideas are
tracked as units, not as connected structures. In production this must run without an LLM, so the
splitter is a small seq2seq model. Candidates: Propositionizer (Chen et al. 2023, Flan-T5-large,
Apache-2.0) zero-shot, then fine-tuned on Sonnet's splits of training units disjoint from the test units
(`scripts/literal-labels.py`), with stripped and disfluent inputs mapped to the original's ideas.
Gemma-APS (Hosseini et al. 2024) is gated and is not used. **Accuracy function** (`scripts/remote-split.py`),
on the 381 check units, with a DeBERTa-v3-large NLI model:
- faithfulness: each idea is entailed by the source;
- coverage: each of Sonnet's reference ideas is entailed by the splitter's ideas;
- invariance: BiNLI F1 between the splits of the original and of its stripped and disfluent versions;
- **accuracy is the harmonic mean of the three.**
Literalness (no unresolved pronoun) must reach ≥ 0.95. **Bar:** accuracy ≥ 0.85 overall, and within 0.05
of Sonnet's own on invariance. Agreement with Sonnet's references (reference-based BiNLI F1) is reported,
not barred.

**10/02/2026, amendment to the splitter's accuracy function, after the zero-shot and Sonnet scores were
read and before any fine-tuned splitter existed.** Inspection showed the NLI model scoring a case
difference ("ENW" vs "enw") as non-entailment. A stripped input yields lowercase ideas, so case alone was
counted as a change of meaning. It also showed the one-to-one match penalizing the same content split at a
different granularity. From now on:
- all text is lowercased before NLI;
- invariance is **content** agreement: each version's ideas, taken together, must entail every idea of the
  other, in both directions;
- the one-to-one BiNLI F1 is still reported, as unit invariance.

The bar is unchanged: accuracy ≥ 0.85, and invariance within 0.05 of Sonnet's. Zero-shot and Sonnet are
rescored under the amended function.

**10/02/2026, second amendment, after the first fine-tuned splitter was scored (accuracy 0.83).** The
literalness constraint counted time determiners ("this morning", "these days") and dummy "it" ("it rains",
"is it ok to ...") as unresolved pronouns. They refer to nothing, so they are now excluded. Round 2 of the
splitter is fine-tuned on all 6,975 Sonnet labels for 4 epochs; the accuracy function and bar are otherwise
unchanged.

**10/02/2026, idea encoder: training data and the lossy-positive ablation, written before any public data was
scored or any encoder trained.** Positives are human-made, with no translation and no per-sentence LLM calls:
- SNLI + MultiNLI entailments, taken as lossy positives (losing detail is accepted compression);
- MRPC paraphrases, and STS-B pairs scored ≥ 4.5 / 5;
- the Sonnet pairs already paid for (phase-1 training variants, geometry rewordings);
- the splitter's literal ideas over Enron, QMSum and IRC training material, each paired with its stripped,
  disfluent and split versions.
Hard negatives are SNLI + MultiNLI contradictions, plus rule edits of literal ideas (negation, role swap,
number or date change, modality change). Quora, COCO captions, PAWS and ParaNMT are excluded.

Ablation at 50k pairs: **arm L** uses all entailments; **arm B** uses only entailments that hold in both
directions (P ≥ 0.9 each way). Both arms have the same pair count, steps and remaining mix, and two seeds each
if the budget holds. Both start from SONAR as released, with fresh adapters.

Measures. None uses the trained encoder's own distances directly, because collapse would shrink them:
- **own-idea accuracy**: the round trip (encode, decode, re-encode) lands nearer its own idea than any other.
  Reported on random test ideas and on **siblings**: rule edits that keep the frame and change the central
  claim, which is where collapse would show;
- **detail retention**: decoded text entails the original, judged by a second NLI model unrelated to the
  filter;
- **meaning kept**: the original entails the decoded text, judged by the same second model;
- negation separation; fluency by a separate small LM.
All metrics carry bootstrap 95% intervals. Test material is ours and never trained on: phase-1 test items,
the 381 check units, the 80 geometry ideas, and a 2% slice of each public set.

**Decision:** keep arm L (all entailments) unless sibling own-idea accuracy or detail retention is lower by
more than 2 points, with non-overlapping intervals. Anything smaller is no meaningful difference, and the
larger dataset wins.

**10/02/2026, amendment to the idea-encoder data, after the public sample was scored (`results/phase-6-data`) and
before any training data was built.**
- **SNLI is removed.** Its sentences are photo captions, and 39% of its entailments are extreme
  generalizations ("children are rollerblading across a track" -> "there are kids outside"). That is not the
  material we encode. STS-B is removed too: it is small and partly the same caption genre.
- **MultiNLI** is the only NLI source. Its entailments are the ablation variable; its contradictions with
  P(contradiction) ≥ 0.9 are hard negatives in both arms.
- **MRPC** positives use the human label with no NLI filter, because the model rejects human-judged
  paraphrases over attribution and quote form. They are identical in both arms.
- **Arm size:** both arms take the same number of MultiNLI pairs, equal to the size of the both-ways pool
  after filtering (about 30k). Everything else in the mix is identical.
- The 80 geometry ideas and their rewordings are test only, never training.
- **Placeholders:** person names and identifiers (codes, emails, URLs, phone numbers) become numbered
  placeholders. Plain numbers stay literal, because changing a number changes the idea and is one of the
  hard negatives.

**10/02/2026, amendment to the encoder ablation's decision metrics, after the baseline (SONAR as released) was
scored and before either trained arm was scored.** The baseline showed SONAR nearly copying its input: the self
round trip came back to its own idea 99.9% of the time, and its decoded text kept 96% of the detail. That ceiling
cannot tell the arms apart, and it does not test the agreed success criterion: a different wording of an idea must
come back to that idea. The decision metrics are now round trips from **another wording**: a stripped, disfluent,
paraphrased or register-shifted version, an MRPC paraphrase, or a MultiNLI both-ways entailment (lossy hypotheses
excluded). Each is encoded, decoded and re-encoded.
- **cross_own_siblings:** it comes back nearer the original idea than every meaning-changed sibling;
- **cross_retention:** its decoded text entails the original (second NLI model).
The rule is unchanged: keep arm L unless either metric is lower by more than 0.02 with non-overlapping intervals.
The geometry ideas add the same test from each of their 12 rewordings against the negation. The self round-trip
metrics are still reported. Baseline negation separation was 0.005: SONAR puts an idea's negation nearer to it
than its own stripped and disfluent forms.

**10/02/2026, attribution run (arm N), written before it was trained.** Arm L kept, arm B tied. To find whether
MultiNLI is doing the work, **arm N** is arm L with every MultiNLI row removed (entailments and contradictions).
MRPC, phase-1 and the literal-idea rows are unchanged, with the same steps (2,000 × 16), settings and seed 1.
MultiNLI's own held-out pairs are expected to drop and are not used to decide. **Decision:** MultiNLI earns its
place if arm N is lower than arm L by more than 0.02, with non-overlapping intervals, on either:
- cross_own_siblings for the phase-1 email sentences (natural paraphrase);
- the geometry round trip from rewordings against negation (`roundtrip_beats_negation`).
Otherwise our own data is doing the work.

**10/03/2026, weighted-batch run (arm W), written before it was trained.** Budget raised to $85. Arm W is arm L
with two changes:
- 1% of rows are removed, where a Sonnet variant carried its own bracketed placeholder (a filter bug);
- half of every batch of 16 comes from ideas with a natural paraphrase, used as the positive: 4 from
  Sonnet/MRPC paraphrases and register shifts, 4 from MultiNLI entailments. The other 8 are drawn as before.
Same steps (2,000), settings and seed 1. The test set is unchanged.

**Success:** phase-1 email cross_own_siblings rises above arm L's 0.610 by more than 0.02 with non-overlapping
intervals. **Guards:** no more than 0.02 lower than arm L on overall cross_own_all, literal negation separation
and the geometry round trip from rewordings against negation.

**10/03/2026, paraphrase and attitude-contrast data (arm C), written before the data was finished or any model
saw it.** Sonnet rewrites 6,000 source sentences (`scripts/contrast-data.py`): half raw Enron sentences from
phase-1 training mailboxes, half literal ideas. Each gets 2 paraphrases, 2 approving and 2 disapproving versions of
the same facts. The user's rule: the same facts under an opposite judgment are **different ideas** that are close
neighbors. Pairs come from `scripts/build-contrast.py`:
- the original is the same idea as each paraphrase, and different ("slant") from each judged version;
- the two approving versions are the same idea, as are the two disapproving versions;
- an approving and a disapproving version are different ideas ("contrast").
10% of items, held out by item, form `test-contrast.jsonl`.

**Arm C** is arm W (L without the bracket rows, half of each batch from natural paraphrases) plus the contrast
training pairs. Their paraphrases count as natural. Same steps (2,000), settings and seed 1. Arms W and C are both
scored on the main test set and on `test-contrast`.

**Success:** on `test-contrast`, arm C beats arm W by more than 0.02 with non-overlapping intervals on
cross_own_siblings: the round trip from a paraphrase or same-judgment version comes back nearer its original than
to every slanted or opposite-judgment version. **Guards:** on the main test set, arm C is no more than 0.02 below
arm W on phase-1 cross_own_siblings, overall cross_own_all, literal negation separation and the geometry round
trip against negation.

**10/03/2026, rebalanced contrast run (arm R), written before it was trained.** Arm C passed its contrast test
(0.634 to 0.868) but failed the phase-1 email guard (0.703 to 0.634). The contrast groups had joined the paraphrase
pool, cutting the phase-1/MRPC paraphrase share to about 22% of what it was in arm W. Arm R is arm C's data with
each batch of 16 split as:
- 4 phase-1/MRPC paraphrase groups (as in arm W);
- 4 MultiNLI entailment groups;
- 4 contrast groups;
- 4 uniform.
Same steps, settings and seed 1.

**Reading:** if phase-1 email cross_own_siblings is ≥ 0.683 (within 0.02 of arm W) while the contrast test stays
≥ 0.848 (within 0.02 of arm C), the loss was **dilution** and arm R is kept. If the email score stays down with
the contrast gain intact, the model has become **over-sensitive** to tone, and how much tone change still counts
as the same idea goes to the user. The other guards are unchanged.
