// shared by every slot and the harness. HIP only, no CUDA.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define BINS 4096u
#define BIN_MASK (BINS - 1u)

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
  exit(2); } } while (0)

// slot contract. zero d_hist, count keys into it, return only when the device is done.
typedef void (*slot_fn)(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist);

// compute unit count of device 0, cached.
static inline int cu_count() {
  static int cus = 0;
  if (cus == 0) {
    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, 0));
    cus = p.multiProcessorCount > 0 ? p.multiProcessorCount : 64;
  }
  return cus;
}

extern "C" {
void slot_naive(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist);
void slot_beginner_fix(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist);
void slot_agent(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist);
void slot_golden(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist);
void slot_cheat_readback(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist);
void slot_cheat_async(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist);
void slot_cheat_badlaunch(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist);
// used by cheat_async, defined in impl_naive.hip. launches without any sync.
void naive_launch_nosync(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist);
}
