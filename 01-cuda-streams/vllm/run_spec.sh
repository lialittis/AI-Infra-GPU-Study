#!/usr/bin/env bash
# Step D: profile vLLM speculative decoding configs with nsys, summarize stream usage, then compare outputs.
# Reports and token dumps go to build/ (gitignored); the summary to ../results/stepD-<timestamp>.txt.
# Usage: ./run_spec.sh [extra spec.py args]   CONFIGS="baseline ngram-k4" ./run_spec.sh   to run a subset
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../../.venv-vllm/bin/python}"
CONFIGS="${CONFIGS:-baseline ngram-k4 draft-k2 draft-k4 draft-k8}"
mkdir -p build ../results
out="../results/stepD-$(date +%Y%m%d-%H%M%S).txt"
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

{
  "$PY" -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name())"
  nsys --version
  for config in $CONFIGS; do
    echo; echo "################ config=$config ################"
    nsys profile -o "build/spec-$config" --force-overwrite true --trace=cuda,nvtx --cuda-graph-trace=node \
      --trace-fork-before-exec=true \
      "$PY" spec.py --config "$config" "$@" 2>&1 | grep -E "^(CONFIG|RESULT)|Traceback|Error" || true
    python3 ../nsys_streams.py "build/spec-$config.nsys-rep" || echo "analysis of $config failed"
  done
  echo; echo "################ output comparison ################"
  python3 compare_tokens.py build spec baseline
} 2>&1 | tee "$out"
echo "saved to $out"
