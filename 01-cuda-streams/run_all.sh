#!/usr/bin/env bash
# Build and run every stream experiment; output goes to results/<timestamp>.txt.
set -euo pipefail
cd "$(dirname "$0")"
make -s
mkdir -p results
out="results/$(date +%Y%m%d-%H%M%S).txt"

{
  echo "### 00 device query";       ./build/00_device_query;   echo
  echo "### 01 concurrency";        ./build/01_concurrency;    echo
  echo "### 02 false dependencies"
  for c in 1 2 4 8 16 32; do CUDA_DEVICE_MAX_CONNECTIONS=$c ./build/02_false_deps 32 4; done
  CUDA_DEVICE_MAX_CONNECTIONS=32 ./build/02_false_deps 64 4
  echo
  echo "### 03 default stream";     ./build/03_default_stream; echo
  echo "### 04 copy overlap";       ./build/04_copy_overlap;   echo
  echo "### 05 priorities";         ./build/05_priorities
} 2>&1 | tee "$out"
echo "saved to $out"
