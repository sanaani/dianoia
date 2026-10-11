# Phase 11: training a model on analogy pairs (10/07/2026)

<!-- Experiment card and datasheet filled in before any result. Hypotheses: HYPOTHESES.md; measures and verdict
rules: pipelines/phase11/OPERATIONALIZATION.md; the pipeline: pipelines/phase11. -->

## Experiment card

- **Question:** does fine-tuning a small open model on real cross-field borrowings make it better at naming the kind
  of far-field invention that would supply a patent's key mechanism?
- **Cheapest decisive test:** Qwen3-1.7B-Base with LoRA, trained at 1k, 3k and 5,862 pairs, writes a description for
  each phase 10 query; a fixed embedder ranks the true source among 1,000 same-field candidates. One GPU job (about
  1.5 hours, at most $3.02 under the self-terminate cap). Anything cheaper (prompting only) is the untrained baseline
  inside the same test.
- **Baseline:** the untrained base model with the same prompt (H1), and the raw citing abstract as the query (H2,
  phase 10's level 0).
- **Deciding metric:** test top-10 recovery of the true source (chance 1%); epochs selected on val by the same metric.
- **What would change the next step:** H1 and H3 supported → more data pays, so the 5.08M-borrowing expansion (and a
  faster first remover to afford it) is next. H1 supported, H3 falsified → the effect is there but flat from 3k: work
  on the task or the model, not more data. H1 falsified → this training signal does not teach the skill at this
  scale; rethink the target (for example, describe the mechanism instead of reproducing the cited abstract).

## Datasheet

- **Sources and licenses:** US patents from PatentsView (public domain text; PatentsView data CC BY 4.0): citations,
  CPC sections, abstracts. Labels from Sonnet (input labeling only) and Claude's hand checks. No client, Threadhouse
  or Purpose Factory data.
- **Splits:** by citing grant year, not `dianoia.splits` (patents, not conversations): training 2013-2020; val
  2018-2019 and test 2021-2024 are phase 10's queries. Any training pair sharing a patent with a val or test pair is
  dropped (824 dropped). Training file `p10/label/train-p11.jsonl`: 5,862 pairs, 4,439 citing patents.
- **Filters:** applicant citations between patents sharing no CPC section A-H; lift ≤ 0.5 and citing applicant
  citations ≤ 50; Fastino stages 1 (unrelated) and 2 (same technology); Sonnet v3 "real" with borrow ≥ 4; Sonnet v4
  C or G (RUBRIC.md). See results/phase-10-analogy/DATA-REVIEW.md.
- **Audit:** `p11-data.py audit` (rule audit in the pipeline); training refuses files whose hashes differ from the
  passing audit's.
- **Known problems:** about 18% of training pairs are not real cross-field borrowings (12% same product, 6%
  unrelated); N is under the 8,000 first planned; one seed per size; val years contribute 526 training pairs.

## Results

## What it means and what is next
