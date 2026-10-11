# Phase 2: encode, decode blind, re-encode

Pre-registered in [`PREREGISTRATION.md`](../../PREREGISTRATION.md), Tier 2. 350 test sentences. The
intent vector is `ft_sonar` (phase 1's chosen model). A round trip **passes** if the decoded text lands
at least as close to the source, in intent space, as a faithful paraphrase does (tolerance τ, median
0.881). Every landing is scored by `ft_sonar` and by `ft_bge`, an encoder no decoder searched against.

## Verdict

| bar | lift256 (pre-registered decoder) | verdict |
|---|---|---|
| lands within tolerance ≥ 80% | **17.1%** | **fail** |
| structured intent check ≥ 80% | **11.7%** | **fail** |
| names restored from bindings | not applicable (see deviation) | — |
| loss is in the words: word Jaccard < 0.4 *while the checks pass* | 0.103, but the checks do not pass | **fail** |

## All arms

| arm | pass (ft_sonar) | pass (ft_bge) | word Jaccard | intent check | act | action | roles | quantities |
|---|---|---|---|---|---|---|---|---|
| SONAR decoding its own vector (ceiling) | 0.994 | 0.974 | 0.857 | 0.920 | 0.997 | 0.951 | 0.991 | 0.966 |
| lift, first 64 dims | 0.063 | 0.103 | 0.053 | | | | | |
| lift, 128 dims | 0.109 | 0.106 | 0.079 | | | | | |
| **lift, 256 dims** | **0.171** | **0.169** | 0.103 | **0.117** | 0.677 | 0.320 | 0.406 | 0.477 |
| Haiku, first guess from neighbors (exploratory) | 0.117 | 0.123 | 0.091 | 0.131 | 0.723 | 0.343 | 0.460 | 0.577 |
| Haiku, 3 rounds of search (exploratory) | 0.354 | 0.186 | 0.091 | 0.146 | 0.746 | 0.360 | 0.477 | 0.597 |
| nearest real sentence (floor, no generation) | 0.374 | 0.174 | 0.096 | 0.171 | 0.740 | 0.286 | 0.354 | 0.637 |

What a decode looks like (lift256): a specific question about an earlier training session comes back as a
generic *"Should I do this?"*, and a sentence about changing a value to negative comes back as *"I changed it."*
(The original email sentences are omitted from this public copy.)

## Reading

- **The speech act comes back (68–75%); the action and its object do not (29–36%).** Decodes are the
  right *kind* of sentence with generic content.
- **Haiku's search games the encoder it climbs.** It doubles its pass rate under `ft_sonar` (0.117 →
  0.354) while the blind encoder barely moves (0.123 → 0.186) and the intent check stays near 0.15.
- **No decoder beats retrieval.** The nearest real sentence does as well as any generation.
- The pre-registered reading *"Tier 1 passes, Tier 2 fails: the decoder cannot read it back"* is only
  half right. The SONAR ceiling shows decoding itself works. The problem is the vector: trained to
  tell an intent from its near-copies, it is a good **discriminator** of edits but not an **address**.
  Many different requests share a neighborhood, and a decoder can only return that neighborhood's
  typical member.

## Deviation

The pre-registered lift targeted each sentence's canonical form, but 1,800 canonical examples gave a
validation cosine of 0.53 to target, too far for SONAR's fragile decoder. The lift used here is also
trained on 121,689 pool sentences from train mailboxes (validation mailboxes excluded too), mapped to their own SONAR vectors, so it
decodes natural sentences rather than placeholders; the names bar cannot be scored. Test mailboxes
were excluded from every training set.
