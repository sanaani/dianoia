#!/bin/bash
# J10: J8's data + #ubuntu auto-disentangled conversations (15% of unused days) + FOMC meetings, J9A's recipe (source mix
# alpha 0.5, 2 layers, 40,000 steps); then scored on J7's pool (J10-on-J7) for the comparison with J7, J8, J9A.
# J9A-P first (10/05/2026): J9A rescored on J7's pool with a 300-prediction sample of nearest pool ideas, for blind
# plausibility grading (grade-plausible.py) while J10 trains; J10-P dumps the same positions (same seed). Both are
# non-fatal: a fault in the new dump code must not cost J10.
# Job contract: logs/done-j10 holds ALLDONE only if every step succeeded.
# A rerun skips J9A-P once logs/sums-j9ap.txt exists (10/05/2026).
# needs: remote-jepa.py remote-encoder.py ablation/S1/lora.pt jepa/J9A/predictor.pt
cd ~/work
PY=/opt/pytorch/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fail() { echo "FAILED: $1" > logs/done-j10; exit 1; }
A="--adapter ablation/S1/lora.pt --layers 2 --dropout 0.3 --wd 0.1 --vec-cache jepa/vec-cache.pt --require-audit"
[ -f logs/sums-j9ap.txt ] || $PY remote-jepa.py --units jepa/units-j7.jsonl $A --out jepa/J9A-P --eval-only jepa/J9A/predictor.pt --dump-retrieved 300 > logs/J9A-P.log 2>&1   && sha256sum jepa/J9A-P/jepa.json jepa/J9A-P/retrieved.jsonl logs/J9A-P.log > logs/sums-j9ap.txt   || echo "J9A-P failed (not fatal: new code; see logs/J9A-P.log)" 
$PY remote-jepa.py --units jepa/units-j10.jsonl $A --mix-alpha 0.5 --out jepa/J10 --steps 40000 --eval-every 2000 > logs/J10.log 2>&1 || fail J10
$PY remote-jepa.py --units jepa/units-j7.jsonl $A --out jepa/J10-on-J7 --eval-only jepa/J10/predictor.pt > logs/J10-on-J7.log 2>&1 || fail J10-on-J7
sha256sum jepa/J10/jepa.json jepa/J10/predictor.pt jepa/J10-on-J7/jepa.json logs/J10.log logs/J10-on-J7.log > logs/sums-j10.txt || fail sums
$PY remote-jepa.py --units jepa/units-j7.jsonl $A --out jepa/J10-P --eval-only jepa/J10/predictor.pt --dump-retrieved 300 > logs/J10-P.log 2>&1   && sha256sum jepa/J10-P/jepa.json jepa/J10-P/retrieved.jsonl logs/J10-P.log > logs/sums-j10p.txt   || echo "J10-P failed (not fatal)"
echo ALLDONE > logs/done-j10
