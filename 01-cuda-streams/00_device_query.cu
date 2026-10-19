// 00: Print the device properties that matter for stream concurrency.
#include "common.cuh"

int main() {
  int dev = 0;
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  int drv, rt, prio_least, prio_greatest;
  CUDA_CHECK(cudaDriverGetVersion(&drv));
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_least, &prio_greatest));

  printf("Device                      : %s (cc %d.%d)\n", p.name, p.major, p.minor);
  printf("Driver / runtime            : %d / %d\n", drv, rt);
  printf("SMs                         : %d\n", p.multiProcessorCount);
  printf("Max threads / SM            : %d\n", p.maxThreadsPerMultiProcessor);
  printf("Max blocks / SM             : %d\n", p.maxBlocksPerMultiProcessor);
  printf("Registers / SM              : %d\n", p.regsPerMultiprocessor);
  printf("Shared mem / SM             : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
  printf("L2 cache                    : %d KB\n", p.l2CacheSize / 1024);
  printf("Global memory               : %.1f GB\n", p.totalGlobalMem / 1e9);
  printf("SM clock (max)              : %d MHz\n", device_attr(cudaDevAttrClockRate) / 1000);
  printf("Integrated (shares host mem): %d\n", p.integrated);
  printf("concurrentKernels           : %d\n", p.concurrentKernels);
  printf("asyncEngineCount (copy eng.): %d\n", p.asyncEngineCount);
  printf("Stream priority range       : least=%d greatest=%d\n", prio_least, prio_greatest);
  printf("unifiedAddressing           : %d\n", p.unifiedAddressing);
  printf("managedMemory               : %d\n", p.managedMemory);
  printf("concurrentManagedAccess     : %d\n", p.concurrentManagedAccess);
  printf("pageableMemoryAccess        : %d\n", p.pageableMemoryAccess);
  printf("pageableMemAccessUsesHostPT : %d\n", p.pageableMemoryAccessUsesHostPageTables);
  printf("hostNativeAtomicSupported   : %d\n", p.hostNativeAtomicSupported);
  printf("CUDA_DEVICE_MAX_CONNECTIONS : %s\n", env_or("CUDA_DEVICE_MAX_CONNECTIONS", "(unset, default 8)"));

  printf("\nspin_kernel residency (blocks/SM) by block size:\n");
  for (int t : {32, 256, 512, 1024})
    printf("  %4d threads -> %d blocks/SM\n", t, spin_blocks_per_sm(t));
  return 0;
}
