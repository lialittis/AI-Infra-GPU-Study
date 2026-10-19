#!/usr/bin/env bash
# Optional helper for editing locally and running on a remote GPU server.
# The target is read from .remote.env at the repo root (gitignored; see .remote.env.example)
# or from the environment: REMOTE_HOST (required), REMOTE_PORT, REMOTE_DIR.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -f "$ROOT/.remote.env" ]] && source "$ROOT/.remote.env"

: "${REMOTE_HOST:?set REMOTE_HOST (e.g. user@gpu-server) in .remote.env or the environment}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-GPU-Study}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
RSH="ssh -p $REMOTE_PORT"

case "${1:-}" in
  push)
    # results/ and build/ are excluded, so --delete never removes remote outputs.
    rsync -az --delete --exclude .git --exclude .remote.env --exclude build --exclude results \
      -e "$RSH" "$ROOT/" "$REMOTE_HOST:$REMOTE_DIR/"
    ;;
  pull)
    rsync -az --prune-empty-dirs --include '*/' --include '*/results/***' --exclude '*' \
      -e "$RSH" "$REMOTE_HOST:$REMOTE_DIR/" "$ROOT/"
    ;;
  run)
    shift
    $RSH "$REMOTE_HOST" "cd $REMOTE_DIR && export PATH=$CUDA_HOME/bin:\$PATH && $*"
    ;;
  shell)
    $RSH -t "$REMOTE_HOST" "cd $REMOTE_DIR && export PATH=$CUDA_HOME/bin:\$PATH && exec bash -l"
    ;;
  *)
    echo "usage: $0 {push|pull|run <cmd>|shell}" >&2
    exit 1
    ;;
esac
