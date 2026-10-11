# Phase 5: the idea splitter, and the shape of an idea

The full report, with figures, is `shape-of-an-idea.html`, also published at https://claude.ai/artifact/8nssWFzS1QvR2S2DdGntrm.

## Splitter (`scripts/literal-labels.py`, `scripts/remote-split.py`)

Text in any medium becomes hyper-literal ideas: pronouns resolved, speaker and stance stated. Accuracy is the
harmonic mean of faithfulness, coverage and invariance, measured with a DeBERTa-v3-large NLI model (definition
in `remote-split.py` and PREREGISTRATION.md, including two recorded amendments).

| Splitter | Accuracy | Invariance | Literal |
|---|---|---|---|
| Propositionizer, off the shelf | 0.732 | 0.760 | 0.770 |
| Propositionizer, fine-tuned (4,631 / 6,975 examples) | 0.830 / 0.824 | 0.824 / 0.819 | 0.926 / 0.936 |
| **Qwen3-1.7B + LoRA r=64, 6,975 examples** | **0.869 (bar 0.85, pass)** | **0.863 (Sonnet 0.847, pass)** | 0.932 (constraint 0.95, short) |
| Sonnet, the teacher | 0.911 | 0.847 | 0.916 |

The adapter is saved at `~/dianoia-data/work/split/q3`. Greedy decoding, so the output is deterministic.

## Geometry (`scripts/idea-geometry.py`, `scripts/remote-geometry.py`; SONAR encoder as released)

Each of 80 ideas has 12 exact rewordings, a negation and a bare claim.
- An idea is a cloud. Median radius is 0.084 cosine and the maximum about 0.19. The cloud is a flattened
  ellipsoid with about 6.5 effective dimensions.
- Other ideas sit at a median 0.44.
- The layers (speaker, stance, modality) stack as partly shared directions: "says that" shifts have a mean
  cosine of 0.28, against 0.00 at random.
- The negation sits **inside** the cloud for all 80 ideas (median 0.06). That is what training must fix.
