// 03: The legacy default stream as a hidden global barrier.
//
// K streams each run one small 1 ms kernel. Between launches we optionally slip a tiny kernel onto
// a "default" stream. The legacy default stream (stream 0 / cudaStreamLegacy) waits for all
// blocking streams and makes them wait for it, so it serializes the whole program. Streams created
// with cudaStreamNonBlocking, and the per-thread default stream, do not have this behavior.
#include "common.cuh"

struct Variant {
  const char* name;
  unsigned stream_flags;
  cudaStream_t interleave;  // stream for the tiny in-between kernels (nullptr would itself mean stream 0)
  bool use_interleave;
};

int main() {
  const int K = 4;
  const uint64_t ns = 1'000'000;
  SpanRecorder rec(K), tiny(1);
  warmup();

  Variant variants[] = {
      {"blocking streams, no default-stream work", cudaStreamDefault, nullptr, false},
      {"blocking streams + legacy default stream", cudaStreamDefault, cudaStreamLegacy, true},
      {"non-blocking streams + legacy default stream", cudaStreamNonBlocking, cudaStreamLegacy, true},
      {"blocking streams + per-thread default stream", cudaStreamDefault, cudaStreamPerThread, true},
  };

  for (const Variant& v : variants) {
    std::vector<cudaStream_t> streams(K);
    for (auto& s : streams) CUDA_CHECK(cudaStreamCreateWithFlags(&s, v.stream_flags));
    rec.reset();
    for (int i = 0; i < K; i++) {
      spin_kernel<<<1, 32, 0, streams[i]>>>(ns, rec.at(i));
      if (v.use_interleave) spin_kernel<<<1, 32, 0, v.interleave>>>(10'000, tiny.at(0));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    OverlapStats r = analyze(rec.fetch());
    printf("%-46s wall=%6.0f us (ideal 1000)  concurrency=%4.2f  peak=%d\n", v.name, r.wall_us, r.concurrency,
           r.max_concurrent);
    for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
  }
  return 0;
}
