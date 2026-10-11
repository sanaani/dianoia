#!/bin/bash
# T1: the text baseline. Qwen3-0.6B-Base fine-tuned on J10's training units (one idea per line, source mix 0.5), then
# writes the next idea at every position of J7's test units; scored with S1 against J7's pool as J10-on-J7 was, and
# its candidates added to J10-P's 325 plausibility rows (remote-textnext.py).
# Micro-batch 2, 768 tokens: micro 8 at 1,024 tokens ran the GPU out of memory on the 151k-word logits (10/05/2026).
# Run on demand (aws.sh start 4 od): the first run lost 45 minutes of training to a spot reclaim, and training keeps no
# checkpoint (10/05/2026).
# Job contract: logs/done-t1 holds ALLDONE only if every step succeeded.
# needs: remote-textnext.py remote-encoder.py ablation/S1/lora.pt jepa/units-j10.jsonl jepa/units-j7.jsonl jepa/J10-P/retrieved.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-t1; exit 1; }
$PY remote-textnext.py train --units jepa/units-j10.jsonl --out jepa/T1/model --micro 2 --max-tokens 768 > logs/T1-train.log 2>&1 || fail train
export PATH=/home/ubuntu/vllm-env/bin:$PATH  # vLLM JIT-compiles a kernel with ninja
/home/ubuntu/vllm-env/bin/python remote-textnext.py generate --units jepa/units-j7.jsonl --model jepa/T1/model --out jepa/T1 > logs/T1-gen.log 2>&1 || fail generate
$PY remote-textnext.py score --units jepa/units-j7.jsonl --adapter ablation/S1/lora.pt --retrieved jepa/J10-P/retrieved.jsonl --out jepa/T1 > logs/T1-score.log 2>&1 || fail score
sha256sum jepa/T1/textnext.json jepa/T1/generated.jsonl jepa/T1/retrieved.jsonl logs/T1-train.log logs/T1-gen.log logs/T1-score.log > logs/sums-t1.txt || fail sums
echo ALLDONE > logs/done-t1
