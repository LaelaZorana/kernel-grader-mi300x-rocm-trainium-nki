// key generator shared by device and host so the CPU reference can rebuild the same keys.
// skewed distribution. bin = floor(4096 * u^4), u uniform in [0,1).
// bin 0 gets about 1 in 8 keys, bins 0..15 get about 1 in 4. that is the contention.
// only double multiplies are used, no fma, so device and host agree bit for bit.
#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>

static inline __host__ __device__ uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU;
  x ^= x >> 15; x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

static inline __host__ __device__ uint32_t key_at(uint64_t seed, uint32_t i) {
  uint32_t h = mix32(i ^ (uint32_t)seed);
  h = mix32(h + (uint32_t)(seed >> 32) * 0x9E3779B9U);
  double u = (double)(h >> 11) * (1.0 / 2097152.0);
  double u2 = u * u;
  double u4 = u2 * u2;
  uint32_t b = (uint32_t)(u4 * 4096.0);
  return b < 4095u ? b : 4095u;
}

// derive the per call seed from the private seed, round and slot index.
static inline uint64_t call_seed(uint64_t seed, uint32_t round, uint32_t slot) {
  uint64_t s = seed ^ ((uint64_t)mix32(round * 0x9E3779B9U + 1u) << 32) ^ (uint64_t)mix32(slot * 0x85EBCA6BU + 7u);
  return s | 1u;
}
