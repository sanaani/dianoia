#!/bin/bash
# C1: the first outcome test (ChangeMyView). Splitter q3 over the posts and first replies (cmv-build.py windows,
# resumable), then remote-cmv.py: surprise under J10, the idea value model and the two text models.
# Runs on T1's on-demand instance after T1 (10/05/2026): the text models keep no checkpoint.
# Job contract: logs/done-c1 holds ALLDONE only if every step succeeded.
# needs: remote-split-vllm.py remote-split.py split/q3 remote-cmv.py remote-jepa.py remote-encoder.py ablation/S1/lora.pt jepa/J10/predictor.pt cmv/windows.jsonl cmv/pairs.jsonl cmv/texts.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-c1; exit 1; }
# Started while T1 still trains: wait for it (the GPU holds one of them), whatever its outcome.
while [ ! -f logs/done-t1 ]; do sleep 60; done
export PATH=/home/ubuntu/vllm-env/bin:$PATH  # vLLM JIT-compiles a kernel with ninja
/home/ubuntu/vllm-env/bin/python remote-split-vllm.py --eval cmv/windows.jsonl --out cmv/pred-cmv.jsonl > logs/C1-split.log 2>&1 || fail split
$PY remote-cmv.py --pred cmv/pred-cmv.jsonl --pairs cmv/pairs.jsonl --texts cmv/texts.jsonl --adapter ablation/S1/lora.pt \
  --predictor jepa/J10/predictor.pt --out cmv/C1 > logs/C1.log 2>&1 || fail cmv
sha256sum cmv/pred-cmv.jsonl cmv/C1/cmv.json cmv/C1/features.jsonl logs/C1-split.log logs/C1.log > logs/sums-c1.txt || fail sums
echo ALLDONE > logs/done-c1
