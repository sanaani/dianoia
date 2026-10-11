#!/bin/bash
# vLLM splitter q3 over the second batch of new sources (cleaned before splitting): #ubuntu auto-disentangled
# conversations (15% sample, days not used before) and FOMC meetings.
# Job contract: logs/done-wc2 holds ALLDONE only if every step succeeded.
# needs: remote-split-vllm.py remote-split.py split/q3 jepa/new2-clean.jsonl
cd ~/work
export PATH=/home/ubuntu/vllm-env/bin:$PATH  # vLLM JIT-compiles a kernel with ninja
/home/ubuntu/vllm-env/bin/python remote-split-vllm.py --eval jepa/new2-clean.jsonl --out jepa/pred-new2.jsonl \
  > logs/WC2.log 2>&1 || { echo FAILED > logs/done-wc2; exit 1; }
sha256sum jepa/pred-new2.jsonl logs/WC2.log > logs/sums-wc2.txt && echo ALLDONE > logs/done-wc2 || echo FAILED > logs/done-wc2
