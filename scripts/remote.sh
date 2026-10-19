#!/usr/bin/env bash
# Optional helper for editing locally and running on a remote GPU server.
# The target is read from .remote.env at the repo root (gitignored; see .remote.env.example)
# or from the environment: REMOTE_HOST (required), REMOTE_PORT, REMOTE_DIR, and REMOTE_SETUP (a shell
# snippet run in the remote repo before every command, e.g. "source tools/cpuinfo-fix/env.sh").
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -f "$ROOT/.remote.env" ]] && source "$ROOT/.remote.env"

: "${REMOTE_HOST:?set REMOTE_HOST (e.g. user@gpu-server) in .remote.env or the environment}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-GPU-Study}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PRELUDE="cd $REMOTE_DIR && export PATH=$CUDA_HOME/bin:\$PATH && ${REMOTE_SETUP:-true}"
RSH="ssh -p $REMOTE_PORT"

case "${1:-}" in
  push)
    # Gitignored paths (.venv, build/, profiles) and results/ are excluded, so --delete never removes
    # remote environments or outputs.
    rsync -az --delete --exclude .git --filter=':- .gitignore' --exclude results \
      -e "$RSH" "$ROOT/" "$REMOTE_HOST:$REMOTE_DIR/"
    ;;
  pull)
    rsync -az --prune-empty-dirs --include '*/' --include '*/results/***' --exclude '*' \
      -e "$RSH" "$REMOTE_HOST:$REMOTE_DIR/" "$ROOT/"
    ;;
  run)
    shift
    $RSH "$REMOTE_HOST" "$PRELUDE && $*"
    ;;
  shell)
    $RSH -t "$REMOTE_HOST" "$PRELUDE && exec bash"
    ;;
  *)
    echo "usage: $0 {push|pull|run <cmd>|shell}" >&2
    exit 1
    ;;
esac
