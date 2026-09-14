// naive. one thread per key, one global atomic per key. hot bins serialize in L2.
#include "common.h"

__global__ void naive_kernel(const uint32_t* __restrict__ keys, uint32_t n, uint32_t* __restrict__ hist) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&hist[keys[i] & BIN_MASK], 1u);
}

extern "C" void naive_launch_nosync(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist) {
  hipMemsetAsync(d_hist, 0, BINS * sizeof(uint32_t), 0);
  const uint32_t block = 256;
  const uint32_t grid = (n + block - 1) / block;
  hipLaunchKernelGGL(naive_kernel, dim3(grid), dim3(block), 0, 0, d_keys, n, d_hist);
}

extern "C" void slot_naive(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist) {
  naive_launch_nosync(d_keys, n, d_hist);
  hipDeviceSynchronize();
}
