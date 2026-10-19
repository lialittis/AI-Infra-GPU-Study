// 02: False dependencies when there are more streams than hardware connections.
//
// S streams each get a dependent chain of L tiny kernels (1 block, 1 ms). The streams are fully
// independent, so ideally everything finishes in L ms. Streams are multiplexed onto
// CUDA_DEVICE_MAX_CONNECTIONS hardware queues (default 8, max 32); streams sharing a queue can
// block each other. Run with different CUDA_DEVICE_MAX_CONNECTIONS values (must be set before the
// CUDA context is created, i.e. in the environment).
//
// Two issue orders are compared:
//   depth-first   s0:k0 k1 .. s1:k0 k1 ..   (a queue sees a stream's whole chain before the next stream)
//   breadth-first all streams' k0, then all k1, ...
#include <cstring>

#include "common.cuh"

static OverlapStats trial(int S, int L, bool depth_first, const std::vector<cudaStream_t>& streams,
                          SpanRecorder& rec) {
  const uint64_t ns = 1'000'000;
  rec.reset();
  if (depth_first) {
    for (int s = 0; s < S; s++)
      for (int l = 0; l < L; l++) spin_kernel<<<1, 32, 0, streams[s]>>>(ns, rec.at(s * L + l));
  } else {
    for (int l = 0; l < L; l++)
      for (int s = 0; s < S; s++) spin_kernel<<<1, 32, 0, streams[s]>>>(ns, rec.at(s * L + l));
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return analyze(rec.fetch());
}

int main(int argc, char** argv) {
  int S = argc > 1 ? atoi(argv[1]) : 32;
  int L = argc > 2 ? atoi(argv[2]) : 4;
  const char* conns = env_or("CUDA_DEVICE_MAX_CONNECTIONS", "default(8)");

  std::vector<cudaStream_t> streams(S);
  for (auto& s : streams) CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  SpanRecorder rec(S * L);
  warmup();

  for (bool depth_first : {true, false}) {
    OverlapStats r = trial(S, L, depth_first, streams, rec);
    printf("connections=%-11s streams=%-3d chain=%d %-13s wall=%7.0f us (ideal %d)  concurrency=%5.2f  peak=%d\n",
           conns, S, L, depth_first ? "depth-first" : "breadth-first", r.wall_us, L * 1000, r.concurrency,
           r.max_concurrent);
  }

  for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}
