# GPU-Study

Hands-on study of AI infrastructure on NVIDIA GPUs: small, self-contained experiments plus notes on what they show.

## Requirements

- An NVIDIA GPU with a recent driver
- CUDA toolkit (`nvcc`). The Makefiles default to `CUDA_HOME=/usr/local/cuda` and `-arch=native`; override with `make CUDA_HOME=... ARCH=sm_90`

## Running

Directly on a GPU machine:

```bash
git clone <this repo> && cd GPU-Study
./01-cuda-streams/run_all.sh          # builds, runs, writes 01-cuda-streams/results/<timestamp>.txt
```

Or edit locally and run on a remote GPU server with `scripts/remote.sh`. Configure it once with `cp .remote.env.example .remote.env` (gitignored), then:

```bash
scripts/remote.sh push                            # rsync repo -> remote (excludes build/, results/)
scripts/remote.sh run './01-cuda-streams/run_all.sh'
scripts/remote.sh pull                            # fetch */results/ back
scripts/remote.sh shell                           # interactive shell in the remote repo
```

## Topics

| # | Topic | Status |
|---|---|---|
| 01 | [CUDA streams](01-cuda-streams/README.md): how they work, how many real inference uses | Step A done |

Results recorded in the notes were measured on an NVIDIA GB10 (DGX Spark: Blackwell, 48 SMs, unified memory, CUDA 13.0) unless stated otherwise.
