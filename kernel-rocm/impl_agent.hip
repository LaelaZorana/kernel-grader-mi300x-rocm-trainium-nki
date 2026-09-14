// mid tier agent. wave aggregated global atomics, no LDS privatization.
// each wave groups lanes that hold the same bin and issues one atomic per distinct bin.
// the ballot loop costs one iteration per distinct bin in the wave, so wide waves
// with many distinct bins pay for it.
#include "common.h"

__global__ void agent_kernel(const uint32_t* __restrict__ keys, uint32_t n, uint32_t* __restrict__ hist) {
  const uint32_t lane = threadIdx.x & (warpSize - 1);
  const uint32_t stride = gridDim.x * blockDim.x;
  const uint32_t wave_end = ((n + stride - 1) / stride) * stride;  // keeps whole waves in the loop
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < wave_end; i += stride) {
    const bool active = i < n;
    const uint32_t key = active ? (keys[i] & BIN_MASK) : 0xFFFFFFFFu;
    unsigned long long todo = __ballot(active);
    while (todo) {
      const int leader = __ffsll((unsigned long long)todo) - 1;
      const uint32_t lead_key = (uint32_t)__shfl((int)key, leader);
      const unsigned long long same = __ballot(active && key == lead_key);
      if ((int)lane == leader) atomicAdd(&hist[lead_key], (uint32_t)__popcll(same));
      todo &= ~same;
    }
  }
}

extern "C" void slot_agent(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist) {
  hipMemsetAsync(d_hist, 0, BINS * sizeof(uint32_t), 0);
  const uint32_t block = 256;
  const uint32_t grid = (uint32_t)cu_count() * 8u;
  hipLaunchKernelGGL(agent_kernel, dim3(grid), dim3(block), 0, 0, d_keys, n, d_hist);
  hipDeviceSynchronize();
}
