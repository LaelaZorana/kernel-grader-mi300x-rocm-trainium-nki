// beginner fix. same global atomics as naive, but a 1024 thread block, vector loads
// and a grid stride loop. this fixes memory access, which was never the bottleneck.
#include "common.h"

__global__ void beginner_kernel(const uint32_t* __restrict__ keys, uint32_t n, uint32_t* __restrict__ hist) {
  const uint32_t n4 = n / 4u;
  const uint4* k4 = reinterpret_cast<const uint4*>(keys);
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    uint4 v = k4[i];
    atomicAdd(&hist[v.x & BIN_MASK], 1u);
    atomicAdd(&hist[v.y & BIN_MASK], 1u);
    atomicAdd(&hist[v.z & BIN_MASK], 1u);
    atomicAdd(&hist[v.w & BIN_MASK], 1u);
  }
  if (blockIdx.x == 0) {
    for (uint32_t i = n4 * 4u + threadIdx.x; i < n; i += blockDim.x) atomicAdd(&hist[keys[i] & BIN_MASK], 1u);
  }
}

extern "C" void slot_beginner_fix(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist) {
  hipMemsetAsync(d_hist, 0, BINS * sizeof(uint32_t), 0);
  const uint32_t block = 1024;
  const uint32_t grid = (uint32_t)cu_count() * 8u;
  hipLaunchKernelGGL(beginner_kernel, dim3(grid), dim3(block), 0, 0, d_keys, n, d_hist);
  hipDeviceSynchronize();
}
