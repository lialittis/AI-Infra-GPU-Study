// 01: Do kernels in different streams actually overlap? Depends on how much of the GPU each fills.
//
// K identical spin kernels are launched either all into one stream or one per stream. The block
// count per kernel is swept from 1 block up to 2x the whole-GPU capacity. Expected: multi-stream
// concurrency ~= min(K, capacity / blocks); single stream is always ~1.0.
#include "common.cuh"

static OverlapStats trial(int K, int blocks, int threads, uint64_t ns, bool multi,
                          const std::vector<cudaStream_t>& streams, SpanRecorder& rec) {
  rec.reset();
  for (int i = 0; i < K; i++)
    spin_kernel<<<blocks, threads, 0, multi ? streams[i] : streams[0]>>>(ns, rec.at(i));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return analyze(rec.fetch());
}

int main(int argc, char** argv) {
  int K = argc > 1 ? atoi(argv[1]) : 8;
  const int threads = 1024;
  const uint64_t ns = 1'000'000;  // 1 ms per block

  int sms = sm_count(), per_sm = spin_blocks_per_sm(threads), cap = sms * per_sm;
  printf("K=%d kernels, %d threads/block, 1 ms spin. SMs=%d, resident blocks/SM=%d -> capacity=%d blocks\n\n",
         K, threads, sms, per_sm, cap);

  std::vector<cudaStream_t> streams(K);
  for (auto& s : streams) CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  SpanRecorder rec(K);
  warmup();

  printf("%8s %10s | %12s %12s %8s | %12s %12s %8s\n", "blocks", "%capacity", "1-stream us", "concurrency",
         "peak", "K-stream us", "concurrency", "peak");
  for (int blocks : {1, 2, cap / 8, cap / 4, cap / 2, cap, 2 * cap}) {
    if (blocks < 1) continue;
    OverlapStats one = trial(K, blocks, threads, ns, false, streams, rec);
    OverlapStats many = trial(K, blocks, threads, ns, true, streams, rec);
    printf("%8d %9.1f%% | %12.0f %12.2f %8d | %12.0f %12.2f %8d\n", blocks, 100.0 * blocks / cap, one.wall_us,
           one.concurrency, one.max_concurrent, many.wall_us, many.concurrency, many.max_concurrent);
  }

  for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}
