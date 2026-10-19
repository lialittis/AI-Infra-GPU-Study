# cpuinfo-fix: PyTorch crashes on import inside CPU-restricted containers

## Symptom

```
$ python -c "import torch"
malloc(): corrupted top size
Aborted (core dumped)
```

It happens on every recent PyTorch release (2.13 – 2.14.1 tested, aarch64, cu130). It occurs before any CUDA call.

## Root cause

This happens on an ARM host inside an LXC/LXD container that is given only some of the CPUs (seen with CPUs `1-3,5-19` of 20). `lxcfs` virtualizes some CPU views but not others, so they disagree:

| Source | What it says |
|---|---|
| `/sys/devices/system/cpu/online` (lxcfs) | `1-3,5-19` |
| `/proc/cpuinfo` (lxcfs) | 18 processors, **renumbered** `0..17` |
| `/sys/devices/system/cpu/cpuN/topology/*` (host passthrough) | real host topology, e.g. `cluster_cpus_list = 0-9`, including the hidden CPUs |

PyTorch statically links [`cpuinfo`](https://github.com/pytorch/cpuinfo). Its ARM/Linux init builds the set of valid processors from `/proc/cpuinfo`, then follows the sysfs topology lists to group CPUs into clusters and packages. Those lists name CPUs outside the valid set, so `cpuinfo` writes past heap arrays. The damage is only detected at the next `malloc` (inside `torch._C`'s `initModule`), which makes the crash look unrelated.

How it was pinned down:
1. `LD_DEBUG=libs` showed that every library's static initializer finishes before the abort.
2. Loading the `torch._C` extension by itself is enough to crash.
3. `OMP_NUM_THREADS=<n>` avoids the import-time crash. Torch only calls `cpuinfo` to choose a default thread count when that variable is unset. But `torch.backends.cpu.get_cpu_capability()` still crashes later.
4. Calling `cpuinfo_initialize()` directly through ctypes, then any `malloc`, reproduces the corruption.
5. Showing the process (via a private mount namespace) a `/proc/cpuinfo` that lists **every present CPU with its real ID** makes everything work.

## Fix (workaround)

- `make_cpuinfo.py` writes a consistent `/proc/cpuinfo`: every present CPU, real sysfs IDs. CPUs hidden by the container get a copy of a visible CPU from the same cluster.
- `cpuinfo_redirect.so` is an `LD_PRELOAD` shim that redirects only opens of `/proc/cpuinfo` to that file.

```bash
source tools/cpuinfo-fix/env.sh   # then run python / nsys as usual in this shell
```

The proper fix is on the host side: give the container a consistent CPU view, or fix `cpuinfo` upstream to ignore sibling CPUs that aren't valid. Only use this workaround where the symptom appears.
