# Phase 10 pair rubric (fixed 10/06/2026)

Decided by Claude at Shaun's request (10/06/2026): "Your rubric is what data is likely to best translate into a
dataset that helps an eventual LLM train on that data to think analogically."

**Why this line.** A pair teaches analogical thinking only if the source field contributes the idea: the model has to
see that a problem in one field is like a problem in another. When the shared thing is a generic, named technology,
the source field is incidental. The inventor could have taken it from any field, and the pair teaches "reuse a known
component", which language models already do. Those pairs also share the technology's own words, so a word search
finds the source without any abstraction, and the phase 10 test would not measure what it is meant to.

## The labels

- **C, analogical borrowing.** The cited field's own way of solving its own problem is carried into a different
  problem, and seeing it required noticing that the two problems are alike. The roles map onto different objects.
- **G, generic technology reused.** The citing invention uses a technology, component or material that is generic
  and named the same way in both, doing the same job, so the cited field is incidental.
- **A, same technology.** Both are about the same kind of product or application, even under different classifications.
- **B, unrelated.** Nothing specific is reused (boilerplate, a loose topical link, or a shared generic word only).

**Revised the same evening, from evidence.** As first written, only C was kept. Then Sonnet with this rubric (prompt
v4) was compared with Claude's labels on 100 pairs. On real vs generic reuse it agreed at about chance (Claude real →
Sonnet G: 25; Claude G/A → Sonnet real: 16), and Claude's own morning and evening readings of 50 pairs agreed on 31,
almost all disagreements on this same line. A boundary nobody applies consistently cannot gate a dataset. So:

- **Gate (kept): C and G**, real cross-field borrowings between different products. **Excluded: A** (same kind of
  product) **and B** (unrelated). These two boundaries were stable across Claude's two readings.
- **The analogical value of a kept pair is measured, not judged:** IDF-weighted word overlap between the two
  abstracts. Low overlap means the source cannot be found by its words. It is reported with every phase 10 result, so
  results can be split by it, and it ranks pairs for later LLM training. The phase 10 test set is not filtered on it,
  because that would select on the raw-text baseline's failure and rig the comparison.

## The test question

> Could the citing inventor have picked up the reused thing from a supplier's catalog or a general textbook, without
> ever looking at the cited field?

Yes → **G** (or **A** if the two products are the same kind). No: the idea is the cited field's particular solution
→ **C**. For the gate, the question that matters is A: are the two the same kind of product or application?

## Examples (from the 10/06 hand checks)

| Pair | Label | Why |
|---|---|---|
| doll whose head turns toward sounds → steering-wheel cue for sound direction | C | the doll's solution (localize a sound with paired microphones, turn toward it) is mapped onto alerting a driver |
| ski-gate flagpole on a spring → self-righting riser base | C | the field's answer to "survive being hit" is mapped onto a new object |
| heat cells for pain relief (iron oxidation) → exothermic microcapsules to cure circuit boards | C | a self-heating chemistry made for the body is mapped onto curing laminates |
| surgical fluid collection → aircraft waste compactor | C | the operating-room solution for collecting and disposing of waste is mapped onto a cabin |
| inflated hog-barn dome → inflatable data center | C | controlled-environment air-supported housing for occupants is mapped from animals to racks |
| UV lamp sterilizing gaming tokens → UV sterilizing an aircraft cabin | G | a UV lamp is generic; the gaming machine contributes nothing |
| ball-bearing cart wheels → dishwasher rack wheels | G | ball-bearing wheels are a catalog part |
| peelable chip-carrier tape (cohesive failure layer) → can-lid sealing film | G | a standard peelable-seal construction, the same job in both |
| ejector vacuum pump → suction cup for handling food | G | an ejector is a generic vacuum source |
| polycrystalline diamond joint bearings → diamond drill-bit compacts | G | the same material technology; hip joints and drill bits are different products |
| pump door that will not latch on a misplaced tube → drug injector door blocked until a cartridge is in | A | both are medical fluid-delivery devices |
| microwave ice-thickness gauge → microwave thickness of dielectric parts | A | the same measurement technology |

## Consistency rule

Every hand check also re-judges 10 earlier pairs, shuffled in without their old labels. Agreement under 9 of 10
means the rubric is not being applied the same way: stop and fix the rubric before reading the gate.
