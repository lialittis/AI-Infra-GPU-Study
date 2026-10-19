#!/usr/bin/env bash
# Create .venv at the repo root with PyTorch matching the installed NVIDIA driver.
# Override the wheel index with TORCH_INDEX=https://download.pytorch.org/whl/cu128 etc.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${TORCH_INDEX:-}" ]]; then
  # Highest CUDA version the driver supports, e.g. "13.0".
  drv_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p')
  major=${drv_cuda%%.*} minor=${drv_cuda#*.}
  if (( major >= 13 )); then tag=cu130
  elif (( major == 12 && minor >= 8 )); then tag=cu128
  else tag=cu126
  fi
  TORCH_INDEX="https://download.pytorch.org/whl/$tag"
fi
echo "driver CUDA ${drv_cuda:-?}, installing torch from $TORCH_INDEX"

# Isolated venv (no system site-packages) so stray packages on shared hosts can't leak in.
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url "$TORCH_INDEX"
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name())"
