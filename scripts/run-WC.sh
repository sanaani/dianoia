#!/bin/bash
# vLLM splitter q3 over WikiConv 2018 windows (cleaned before splitting): the first source of the 10x data build.
# Job contract: logs/done-wc holds ALLDONE only if every step succeeded.
cd ~/work
export PATH=/home/ubuntu/vllm-env/bin:$PATH  # vLLM JIT-compiles a kernel with ninja
/home/ubuntu/vllm-env/bin/python remote-split-vllm.py --eval jepa/wc2018-clean.jsonl --out jepa/pred-wc2018.jsonl \
  > logs/WC.log 2>&1 || { echo FAILED > logs/done-wc; exit 1; }
sha256sum jepa/pred-wc2018.jsonl logs/WC.log > logs/sums-wc.txt && echo ALLDONE > logs/done-wc || echo FAILED > logs/done-wc
