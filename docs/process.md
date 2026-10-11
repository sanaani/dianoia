# How the research runs (adopted 10/04/2026)

The goal is a self-improving system: every recurring mistake or manual step becomes software, so the work is
repeatable and does not depend on anyone remembering a rule.

## The cycle

1. **Review and encode.** Look over the work since the last review: what went wrong, what was done by hand, what
   only care prevented. Turn each recurring risk into a check, guard, script, template or test.
2. **Next question.** Decide what we most need to learn next.
3. **Cheapest decisive test.** Prefer, in order: analysis of data on disk, a small Sonnet probe, a laptop-light
   check, a short GPU run. Stop a run early when its targeted test shows no gain.
4. **Record and repeat.** Write the result (negative results too) and commit.

## What is already software

| Risk | Guard |
|---|---|
| Leaks between train, validation and test | `src/dianoia/splits.py`: one split by whole conversation for every source |
| Training on unchecked data | `scripts/audit-data.py`; `remote-jepa.py --require-audit` refuses data without a passing audit of the exact file |
| A leak check that silently stops catching leaks | `scripts/audit-selftest.py`: plants duplicated conversations and requires the audit to catch them, before every real audit |
| Losing outputs when an instance is terminated | `scripts/aws-finish.sh`: downloads, verifies checksums, terminates only if all pass |
| Running an experiment without a clear question or matching metric | `results/TEMPLATE.md`: experiment card and datasheet before results |
| IRC bot boilerplate in conversation data | bot filter in `scripts/jepa-units.py` and `scripts/jepa-build.py` |
| Losing a long GPU job to an interruption | `remote-split.py predict-qwen` writes batch by batch and resumes |
| Hand-run GPU steps (an upload lost to a backgrounded pipe; a process check matching its own ssh command; the launch script living in a session scratchpad) | `scripts/aws.sh` start / upload (verified) / run (confirmed running) / wait / finish |
| Imaging an instance before its results are downloaded | `aws.sh image` refuses while any remote `logs/sums-*.txt` is not verified locally (it caught J7's and J4b's predictors on 10/04/2026) and removes `~/work`, so data never travels inside an image |
| A rerun reported as failed by an earlier attempt's done file | `aws.sh run` deletes the `logs/done-*` files the job script names before starting it |
| Reinstalling packages and models on every instance | prebuilt image (`aws.sh image`, id in `work/aws-ami`, used by `start`): launch to ready in about 3 minutes, smoke-tested 10/04/2026; `AWS_STATE` gives a second instance its own record |
| The reaper killing a laptop pipeline mid-step | on Windows its Python children survive (J10 chain, 10/05/2026): find them by command line, wait on their PID, then rerun the pipeline with `SKIP_SPLIT=1 SKIP_CHAIN=1`; restart lighter only if the step died. Extend the instance cap to cover the delay |
| Editing `aws.sh` under a running watcher | the script is one `{ ... }` block, parsed before it runs |
| One long prompt killing a vLLM run | `remote-split-vllm.py`: context 8,192 tokens; prompts that still do not fit are skipped and counted |
| Speedups that change results | each is proven against the old path before use: vector cache and GPU scoring (E1, E3), placeholder workers (identical output), vLLM splitter (V3, below) |

## Standing rules

- Names become placeholders; numbers stay literal; no translation data. Self-hosted open-weight LLMs may be
  part of the product (Shaun, 10/05/2026, replacing "no LLMs in production"); hosted models such as Sonnet only
  generate data or grade, never at runtime.
- Client and Purpose Factory data never go into training data, Sonnet prompts or anything published.
- Heavy compute runs on AWS, every instance with a self-shutdown cap; every transfer is checksummed.

## The splitter for data builds: vLLM (10/04/2026)

`remote-split-vllm.py` runs splitter q3 at about 1,035 windows a minute on one A10G, against 100-260 for HF
`generate`. On 200 windows split both ways, 91% of ideas are identical, 6% differ by a word or two and about 2.5% are
worded differently but carry the same content (bf16 drift under greedy decoding). That is within tolerance: related
ideas need to be close, not identical. **Rule: one splitter path per build.** Every split of a build (train,
validation, test) is cut by the same path, so HF and vLLM output are never mixed between training and evaluation.
