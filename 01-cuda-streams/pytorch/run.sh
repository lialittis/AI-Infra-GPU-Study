#!/usr/bin/env bash
# Profile every generation mode with nsys and summarize stream usage.
# Reports go to build/ (large, gitignored); the summary to ../results/stepB-<timestamp>.txt.
# Usage: ./run.sh [extra infer.py args, e.g. --steps 64 --batch 4]
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../../.venv/bin/python}"
mkdir -p build ../results
out="../results/stepB-$(date +%Y%m%d-%H%M%S).txt"

{
  "$PY" -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name())"
  nsys --version
  for mode in eager async async-fixed graph-sync graph compile; do
    echo; echo "################ mode=$mode ################"
    nsys profile -o "build/$mode" --force-overwrite true --trace=cuda,nvtx --cuda-graph-trace=node \
      "$PY" infer.py --mode "$mode" --steps 32 "$@" 2>&1 | grep -E "^mode=|[Ee]rror" || true
    python3 ../nsys_streams.py "build/$mode.nsys-rep"
  done

  # Async copies only pay off when the host has its own per-token work and the GPU is the bottleneck.
  echo; echo "################ host work per token: sync vs async ################"
  for work in 0 3 6; do
    for mode in graph-sync graph; do
      "$PY" infer.py --mode "$mode" --steps 32 --host-work-ms "$work" "$@" 2>&1 | grep "^mode="
    done
  done
} 2>&1 | tee "$out"
echo "saved to $out"
