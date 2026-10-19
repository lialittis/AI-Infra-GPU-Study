#!/usr/bin/env bash
# Run every reproduction case and save the log to results/<timestamp>.txt.
# Usage: ./run.sh [substring filters, e.g. pinned pool]
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../.venv/bin/python}"
mkdir -p results
out="results/$(date +%Y%m%d-%H%M%S).txt"
{
  "$PY" -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name())"
  "$PY" run.py "$@"
} 2>&1 | tee "$out"
echo "saved to $out"
