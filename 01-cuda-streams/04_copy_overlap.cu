// 04: Copy/compute overlap, and how host memory type decides whether it happens.
//
// A compute kernel (spin, touches no memory) runs on one stream while a 256 MB memcpy runs on
// another. The kernel duration is calibrated to the copy time, so perfect overlap halves wall time.
//   overlap = (t_copy + t_kernel - t_both) / min(t_copy, t_kernel)   (1 = perfect, 0 = serialized)
// Pageable memory forces a staged copy that can block the host thread, so launch order matters.
// GB10 is an integrated GPU with unified memory; we want to see if the classic rules still apply.
#include <chrono>
#include <cstring>

#include "common.cuh"

using Clock = std::chrono::steady_clock;
static double us_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

struct Buffers {
  void* dev;
  void* pageable;
  void* pinned;
};

static void do_copy(const Buffers& b, size_t bytes, bool h2d, bool pinned, cudaStream_t s) {
  void* host = pinned ? b.pinned : b.pageable;
  if (h2d) CUDA_CHECK(cudaMemcpyAsync(b.dev, host, bytes, cudaMemcpyHostToDevice, s));
  else CUDA_CHECK(cudaMemcpyAsync(host, b.dev, bytes, cudaMemcpyDeviceToHost, s));
}

// Median of 5 runs of fn() in microseconds (fn must synchronize).
template <typename F>
static double median_us(F fn) {
  std::vector<double> t;
  for (int i = 0; i < 5; i++) {
    auto t0 = Clock::now();
    fn();
    t.push_back(us_since(t0));
  }
  std::sort(t.begin(), t.end());
  return t[2];
}

int main() {
  const size_t bytes = 256ull << 20;
  Buffers b;
  CUDA_CHECK(cudaMalloc(&b.dev, bytes));
  CUDA_CHECK(cudaMallocHost(&b.pinned, bytes));
  b.pageable = malloc(bytes);
  memset(b.pageable, 1, bytes);  // fault pages in so first-touch isn't measured
  memset(b.pinned, 1, bytes);

  cudaStream_t ks, cs, cs2;
  CUDA_CHECK(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&cs2, cudaStreamNonBlocking));
  SpanRecorder rec(1);
  warmup();
  const int blocks = sm_count() * spin_blocks_per_sm(256);  // kernel occupies the whole GPU once

  printf("256 MB copies, compute kernel = %d blocks x 256 threads\n\n", blocks);
  printf("%-34s %9s %9s %9s %9s %8s\n", "case", "copy us", "GB/s", "kernel us", "both us", "overlap");

  struct Case {
    const char* name;
    bool h2d, pinned, kernel_first;
  } cases[] = {
      {"H2D pinned,   kernel launched 1st", true, true, true},
      {"H2D pageable, kernel launched 1st", true, false, true},
      {"H2D pageable, copy launched 1st", true, false, false},
      {"D2H pinned,   kernel launched 1st", false, true, true},
      {"D2H pageable, kernel launched 1st", false, false, true},
  };

  for (const Case& c : cases) {
    double t_copy = median_us([&] {
      do_copy(b, bytes, c.h2d, c.pinned, cs);
      CUDA_CHECK(cudaDeviceSynchronize());
    });
    uint64_t ns = uint64_t(t_copy * 1e3);
    auto kernel = [&] { spin_kernel<<<blocks, 256, 0, ks>>>(ns, rec.at(0)); };
    double t_kernel = median_us([&] {
      kernel();
      CUDA_CHECK(cudaDeviceSynchronize());
    });
    double t_both = median_us([&] {
      if (c.kernel_first) {
        kernel();
        do_copy(b, bytes, c.h2d, c.pinned, cs);
      } else {
        do_copy(b, bytes, c.h2d, c.pinned, cs);
        kernel();
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    });
    double overlap = (t_copy + t_kernel - t_both) / std::min(t_copy, t_kernel);
    printf("%-34s %9.0f %9.1f %9.0f %9.0f %8.2f\n", c.name, t_copy, bytes / t_copy / 1e3, t_kernel, t_both,
           overlap);
  }

  // Bidirectional: can an H2D and a D2H copy run at the same time (separate copy engines)?
  void* dev2;
  CUDA_CHECK(cudaMalloc(&dev2, bytes));
  void* pinned2;
  CUDA_CHECK(cudaMallocHost(&pinned2, bytes));
  double t_h2d = median_us([&] {
    do_copy(b, bytes, true, true, cs);
    CUDA_CHECK(cudaDeviceSynchronize());
  });
  double t_d2h = median_us([&] {
    CUDA_CHECK(cudaMemcpyAsync(pinned2, dev2, bytes, cudaMemcpyDeviceToHost, cs2));
    CUDA_CHECK(cudaDeviceSynchronize());
  });
  double t_bi = median_us([&] {
    do_copy(b, bytes, true, true, cs);
    CUDA_CHECK(cudaMemcpyAsync(pinned2, dev2, bytes, cudaMemcpyDeviceToHost, cs2));
    CUDA_CHECK(cudaDeviceSynchronize());
  });
  printf("\nH2D || D2H (pinned): h2d=%.0f us  d2h=%.0f us  both=%.0f us  overlap=%.2f\n", t_h2d, t_d2h, t_bi,
         (t_h2d + t_d2h - t_bi) / std::min(t_h2d, t_d2h));

  CUDA_CHECK(cudaFree(dev2));
  CUDA_CHECK(cudaFreeHost(pinned2));
  CUDA_CHECK(cudaFree(b.dev));
  CUDA_CHECK(cudaFreeHost(b.pinned));
  free(b.pageable);
  return 0;
}
