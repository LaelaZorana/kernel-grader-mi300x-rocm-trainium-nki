// broken slot, on purpose. launches with 2048 threads per block, over the 1024 limit,
// so the launch never runs and hipGetLastError reports invalid configuration argument.
// used by the runtime-failure-modes check. wired in as the last SLOTS entry of harness.hip,
// the last SRCS entry of the Makefile and the last name in SLOTS of audit.py (2026-09-09).
#include "common.h"

__global__ void badlaunch_kernel(const uint32_t* __restrict__ keys, uint32_t n, uint32_t* __restrict__ hist) {
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) atomicAdd(&hist[keys[i] & BIN_MASK], 1u);
}

extern "C" void slot_cheat_badlaunch(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist) {
  hipMemsetAsync(d_hist, 0, BINS * sizeof(uint32_t), 0);
  const uint32_t block = 2048;   // limit is 1024 on gfx942, this launch is rejected
  const uint32_t grid = (uint32_t)cu_count() * 8u;
  hipLaunchKernelGGL(badlaunch_kernel, dim3(grid), dim3(block), 0, 0, d_keys, n, d_hist);
  hipDeviceSynchronize();       // the error stays sticky in hipGetLastError for the harness
}
