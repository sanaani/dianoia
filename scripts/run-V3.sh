#!/bin/bash
# vLLM splitter retry (V2 failed: a 2,049-token prompt): new wiki windows + 200 overlap, after the equivalence job E.
# Job contract: logs/done-v3 holds ALLDONE only if every step succeeded.
cd ~/work

export PATH=/home/ubuntu/vllm-env/bin:$PATH  # vLLM JIT-compiles a kernel with ninja (failed without it, 10/04)
/home/ubuntu/vllm-env/bin/python remote-split-vllm.py --eval jepa/wiki-windows-new.jsonl --out jepa/pred-wiki-new.jsonl \
  --check jepa/pred-wiki.jsonl > logs/V3.log 2>&1 || { echo FAILED > logs/done-v3; exit 1; }
sha256sum jepa/pred-wiki-new.jsonl jepa/wiki-windows-new.jsonl logs/V3.log > logs/sums-v3.txt && echo ALLDONE > logs/done-v3 || echo FAILED > logs/done-v3
