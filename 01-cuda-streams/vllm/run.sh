#!/usr/bin/env bash
# Profile vLLM under each config with nsys and summarize stream usage.
# Reports go to build/ (large, gitignored); the summary to ../results/stepC-<timestamp>.txt.
# Usage: ./run.sh [extra bench.py args, e.g. --model Qwen/Qwen2.5-0.5B-Instruct]
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../../.venv-vllm/bin/python}"
mkdir -p build ../results
out="../results/stepC-$(date +%Y%m%d-%H%M%S).txt"
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

{
  "$PY" -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name())"
  nsys --version
  for config in default no-async eager; do
    echo; echo "################ config=$config ################"
    # vLLM forks its EngineCore process without exec; nsys only traces such children with
    # --trace-fork-before-exec=true.
    nsys profile -o "build/$config" --force-overwrite true --trace=cuda,nvtx --cuda-graph-trace=node \
      --trace-fork-before-exec=true \
      "$PY" bench.py --config "$config" "$@" 2>&1 | grep -E "^RESULT|backend|Traceback|Error" || true
    python3 ../nsys_streams.py "build/$config.nsys-rep" || echo "analysis of $config failed"
  done
} 2>&1 | tee "$out"
echo "saved to $out"
