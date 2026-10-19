// Shared helpers for the stream experiments.
//
// Overlap is measured on the GPU itself: every kernel records the earliest block start and the
// latest block end using the %globaltimer register (ns). Comparing those spans across kernels
// tells us what actually ran concurrently, independent of host-side launch timing.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                                    \
  do {                                                                                      \
    cudaError_t err_ = (call);                                                              \
    if (err_ != cudaSuccess) {                                                              \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,            \
              cudaGetErrorString(err_));                                                    \
      exit(1);                                                                              \
    }                                                                                       \
  } while (0)

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// [start, end] of one kernel in GPU global-timer nanoseconds.
struct KernelSpan {
  unsigned long long start;
  unsigned long long end;
};

// Every block busy-waits for `ns` nanoseconds; thread 0 widens the kernel's span.
// Spinning (instead of real math) gives a precise, resource-light kernel whose duration we control.
__global__ void spin_kernel(uint64_t ns, KernelSpan* span) {
  uint64_t t0 = globaltimer();
  if (threadIdx.x == 0) atomicMin(&span->start, (unsigned long long)t0);
  while (globaltimer() - t0 < ns) {
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&span->end, (unsigned long long)globaltimer());
}

// Device buffer of n spans, reset to "empty" before each trial.
class SpanRecorder {
 public:
  explicit SpanRecorder(int n) : n_(n) {
    CUDA_CHECK(cudaMalloc(&d_, n * sizeof(KernelSpan)));
    reset();
  }
  ~SpanRecorder() { cudaFree(d_); }

  void reset() {
    std::vector<KernelSpan> h(n_, KernelSpan{~0ull, 0ull});
    CUDA_CHECK(cudaMemcpy(d_, h.data(), n_ * sizeof(KernelSpan), cudaMemcpyHostToDevice));
  }
  KernelSpan* at(int i) { return d_ + i; }
  std::vector<KernelSpan> fetch() const {
    std::vector<KernelSpan> h(n_);
    CUDA_CHECK(cudaMemcpy(h.data(), d_, n_ * sizeof(KernelSpan), cudaMemcpyDeviceToHost));
    return h;
  }

 private:
  int n_;
  KernelSpan* d_;
};

struct OverlapStats {
  double wall_us;      // first start -> last end
  double busy_sum_us;  // sum of individual kernel durations
  double concurrency;  // busy_sum / wall: 1.0 = fully serial, N = N kernels always overlapping
  int max_concurrent;  // peak number of kernels running at the same instant
};

inline OverlapStats analyze(const std::vector<KernelSpan>& spans) {
  unsigned long long first = ~0ull, last = 0;
  double busy = 0;
  std::vector<std::pair<unsigned long long, int>> events;  // (time, +1 start / -1 end)
  for (const auto& s : spans) {
    first = std::min(first, s.start);
    last = std::max(last, s.end);
    busy += double(s.end - s.start);
    events.push_back({s.start, +1});
    events.push_back({s.end, -1});
  }
  std::sort(events.begin(), events.end());  // at equal times, ends (-1) sort before starts
  int cur = 0, peak = 0;
  for (const auto& e : events) peak = std::max(peak, cur += e.second);
  double wall = double(last - first);
  return {wall / 1e3, busy / 1e3, busy / wall, peak};
}

inline int device_attr(cudaDeviceAttr attr) {
  int dev, v;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&v, attr, dev));
  return v;
}

inline int sm_count() { return device_attr(cudaDevAttrMultiProcessorCount); }

// How many blocks of `threads` threads of spin_kernel fit on one SM at once.
inline int spin_blocks_per_sm(int threads) {
  int n;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, spin_kernel, threads, 0));
  return n;
}

inline const char* env_or(const char* name, const char* fallback) {
  const char* v = getenv(name);
  return v ? v : fallback;
}

// Launch one throwaway kernel so module loading / context setup doesn't pollute the first trial.
inline void warmup() {
  SpanRecorder r(1);
  spin_kernel<<<1, 32>>>(1000, r.at(0));
  CUDA_CHECK(cudaDeviceSynchronize());
}
