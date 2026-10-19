#!/usr/bin/env bash
# Create .venv-vllm at the repo root with vLLM (which pins its own torch, so it gets a separate venv).
# Override the version with VLLM_VERSION=x.y.z.
set -euo pipefail
cd "$(dirname "$0")/.."

VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -q --upgrade pip
.venv-vllm/bin/pip install -q "vllm==$VLLM_VERSION"
.venv-vllm/bin/python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name())"
