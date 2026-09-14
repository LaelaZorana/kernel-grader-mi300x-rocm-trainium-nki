// golden. per block private histogram in LDS, vector loads, one merge pass at the end.
// 4096 bins x 4 bytes = 16 KB of LDS per block, 4 blocks fit per CU on MI300X.
#include "common.h"

__global__ void golden_kernel(const uint32_t* __restrict__ keys, uint32_t n, uint32_t* __restrict__ hist) {
  __shared__ uint32_t lds[BINS];
  for (uint32_t b = threadIdx.x; b < BINS; b += blockDim.x) lds[b] = 0u;
  __syncthreads();

  const uint32_t n4 = n / 4u;
  const uint4* k4 = reinterpret_cast<const uint4*>(keys);
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    uint4 v = k4[i];
    atomicAdd(&lds[v.x & BIN_MASK], 1u);
    atomicAdd(&lds[v.y & BIN_MASK], 1u);
    atomicAdd(&lds[v.z & BIN_MASK], 1u);
    atomicAdd(&lds[v.w & BIN_MASK], 1u);
  }
  if (blockIdx.x == 0) {
    for (uint32_t i = n4 * 4u + threadIdx.x; i < n; i += blockDim.x) atomicAdd(&lds[keys[i] & BIN_MASK], 1u);
  }
  __syncthreads();

  for (uint32_t b = threadIdx.x; b < BINS; b += blockDim.x) {
    uint32_t c = lds[b];
    if (c) atomicAdd(&hist[b], c);
  }
}

extern "C" void slot_golden(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist) {
  hipMemsetAsync(d_hist, 0, BINS * sizeof(uint32_t), 0);
  const uint32_t block = 256;
  const uint32_t grid = (uint32_t)cu_count() * 4u;
  hipLaunchKernelGGL(golden_kernel, dim3(grid), dim3(block), 0, 0, d_keys, n, d_hist);
  hipDeviceSynchronize();
}
