// 05: Stream priorities: can a latency-critical kernel cut ahead of a big one?
//
// A big low-priority kernel fills every SM with many waves of 200 us blocks (~4 ms total). 500 us
// later the host launches a small 1-block kernel on a second stream. Priorities are honored at
// block-scheduling granularity (no preemption of running blocks), so with high priority the small
// kernel should start at the next block boundary (<= 200 us wait) instead of after all waves.
#include <chrono>

#include "common.cuh"

int main() {
  int least, greatest;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  cudaStream_t big_s, small_lo, small_hi;
  CUDA_CHECK(cudaStreamCreateWithPriority(&big_s, cudaStreamNonBlocking, least));
  CUDA_CHECK(cudaStreamCreateWithPriority(&small_lo, cudaStreamNonBlocking, least));
  CUDA_CHECK(cudaStreamCreateWithPriority(&small_hi, cudaStreamNonBlocking, greatest));

  const int threads = 256;
  const int waves = 20;
  const int cap = sm_count() * spin_blocks_per_sm(threads);
  SpanRecorder rec(2);
  warmup();
  printf("big kernel: %d blocks (%d waves x %d) x 200 us, small kernel: 1 block x 50 us, launched ~500 us later\n\n",
         cap * waves, waves, cap);

  for (auto [name, small_s] : {std::pair{"small on equal (low) priority", small_lo},
                               std::pair{"small on HIGH priority", small_hi}}) {
    std::vector<double> waits, finish_gaps;
    for (int rep = 0; rep < 5; rep++) {
      rec.reset();
      spin_kernel<<<cap * waves, threads, 0, big_s>>>(200'000, rec.at(0));
      auto t0 = std::chrono::steady_clock::now();
      while (std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(500)) {
      }
      spin_kernel<<<1, threads, 0, small_s>>>(50'000, rec.at(1));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      auto s = rec.fetch();
      waits.push_back((double(s[1].start) - double(s[0].start)) / 1e3);
      finish_gaps.push_back((double(s[0].end) - double(s[1].end)) / 1e3);
    }
    std::sort(waits.begin(), waits.end());
    std::sort(finish_gaps.begin(), finish_gaps.end());
    printf("%-30s small starts %7.0f us after big starts; small finishes %7.0f us before big ends (median of 5)\n",
           name, waits[2], finish_gaps[2]);
  }
  return 0;
}
