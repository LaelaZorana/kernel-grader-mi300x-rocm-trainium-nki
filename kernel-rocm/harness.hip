// harness. runs every slot, times it, checks it, prints a table and writes json.
//
// unhardened (default): one input per round, one shared output buffer that is never
//   wiped, timer stops when the slot returns, output checked later after an implicit sync.
// --hardened: fresh input per call from a private seed, one buffer per slot poisoned with
//   0xFFFFFFFF before every call, hipDeviceSynchronize and error check inside the timed
//   region, pending device time after return measured, output checked against a CPU
//   reference every call, median over rounds.
//
// usage: harness [--hardened] [--rounds R] [--n N] [--json path] [--dump-ref path --dump-n M]
#include "common.h"
#include "keygen.h"
#include <chrono>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

struct Slot { const char* name; slot_fn fn; };

static const Slot SLOTS[] = {
  {"naive", slot_naive},
  {"beginner_fix", slot_beginner_fix},
  {"agent", slot_agent},
  {"golden", slot_golden},
  {"cheat_readback", slot_cheat_readback},   // runs right after golden on purpose
  {"cheat_async", slot_cheat_async},
  {"cheat_badlaunch", slot_cheat_badlaunch},   // 2048 threads per block, launch rejected, scores 0 hardened
};
static const int NSLOTS = sizeof(SLOTS) / sizeof(SLOTS[0]);

struct Result {
  std::vector<double> times_ms, pending_ms;
  bool correct = true;
  bool async_return = false;
  uint32_t max_mismatch = 0;
  double score = 0.0;
};

static double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

__global__ void keygen_kernel(uint64_t seed, uint32_t n, uint32_t* __restrict__ keys) {
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) keys[i] = key_at(seed, i);
}

static void gen_keys(uint64_t seed, uint32_t n, uint32_t* d_keys) {
  hipLaunchKernelGGL(keygen_kernel, dim3(cu_count() * 8), dim3(256), 0, 0, seed, n, d_keys);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

// CPU reference from the keys that are actually on the device.
static void cpu_reference(const uint32_t* d_keys, uint32_t n, uint32_t* h_keys, std::vector<uint32_t>& ref) {
  HIP_CHECK(hipMemcpy(h_keys, d_keys, (size_t)n * sizeof(uint32_t), hipMemcpyDeviceToHost));
  ref.assign(BINS, 0u);
  for (uint32_t i = 0; i < n; i++) ref[h_keys[i] & BIN_MASK]++;
}

static uint32_t count_mismatch(const uint32_t* d_hist, const std::vector<uint32_t>& ref, std::vector<uint32_t>& out) {
  out.resize(BINS);
  HIP_CHECK(hipMemcpy(out.data(), d_hist, BINS * sizeof(uint32_t), hipMemcpyDeviceToHost));
  uint32_t bad = 0;
  for (uint32_t b = 0; b < BINS; b++) if (out[b] != ref[b]) bad++;
  return bad;
}

static double score_of(double t, double t_naive, double t_golden, bool ok) {
  if (!ok) return 0.0;
  double gap = t_naive - t_golden;
  if (gap <= 0.0) return 0.0;
  double s = (t_naive - t) / gap;
  return s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s);
}

static std::string json_times(const std::vector<double>& v) {
  std::string s = "[";
  char buf[64];
  for (size_t i = 0; i < v.size(); i++) { snprintf(buf, sizeof buf, "%s%.4f", i ? ", " : "", v[i]); s += buf; }
  return s + "]";
}

// writes keys and their CPU histogram for an outside check with numpy.bincount.
// also checks that device keygen equals host keygen for every key.
static int dump_ref(const char* path, uint32_t n, uint64_t seed) {
  uint32_t* d_keys; HIP_CHECK(hipMalloc(&d_keys, (size_t)n * sizeof(uint32_t)));
  gen_keys(seed, n, d_keys);
  std::vector<uint32_t> h(n), ref;
  cpu_reference(d_keys, n, h.data(), ref);
  uint32_t diff = 0;
  for (uint32_t i = 0; i < n; i++) if (h[i] != key_at(seed, i)) diff++;
  FILE* f = fopen(path, "wb");
  if (!f) { perror("dump-ref"); return 2; }
  fwrite(&n, 4, 1, f); fwrite(&diff, 4, 1, f);
  fwrite(h.data(), 4, n, f); fwrite(ref.data(), 4, BINS, f);
  fclose(f);
  printf("dump-ref: n=%u device_vs_host_key_diff=%u -> %s\n", n, diff, path);
  HIP_CHECK(hipFree(d_keys));
  return diff ? 1 : 0;
}

int main(int argc, char** argv) {
  bool hardened = false;
  int rounds = -1;
  uint32_t n = 64u * 1024u * 1024u;
  const char* json_path = nullptr;
  const char* dump_path = nullptr;
  uint32_t dump_n = 1u << 20;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--hardened") hardened = true;
    else if (a == "--rounds" && i + 1 < argc) rounds = atoi(argv[++i]);
    else if (a == "--n" && i + 1 < argc) n = (uint32_t)strtoull(argv[++i], nullptr, 10);
    else if (a == "--json" && i + 1 < argc) json_path = argv[++i];
    else if (a == "--dump-ref" && i + 1 < argc) dump_path = argv[++i];
    else if (a == "--dump-n" && i + 1 < argc) dump_n = (uint32_t)strtoull(argv[++i], nullptr, 10);
    else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  if (rounds < 1) rounds = hardened ? 5 : 1;

  // private seed. env GYM_SEED for a repeatable run, otherwise random. never printed.
  uint64_t seed;
  if (const char* e = getenv("GYM_SEED")) seed = strtoull(e, nullptr, 0);
  else { std::random_device rd; seed = ((uint64_t)rd() << 32) ^ rd(); }
  seed |= 1u;

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("device: %s  arch: %s  CUs: %d  mode: %s  rounds: %d  n: %u\n",
         prop.name, prop.gcnArchName, prop.multiProcessorCount, hardened ? "hardened" : "unhardened", rounds, n);

  if (dump_path) return dump_ref(dump_path, dump_n, seed);

  uint32_t* d_keys; HIP_CHECK(hipMalloc(&d_keys, (size_t)n * sizeof(uint32_t)));
  uint32_t* h_keys; HIP_CHECK(hipHostMalloc(&h_keys, (size_t)n * sizeof(uint32_t)));
  uint32_t* d_shared; HIP_CHECK(hipMalloc(&d_shared, BINS * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(d_shared, 0, BINS * sizeof(uint32_t)));
  uint32_t* d_own[NSLOTS];
  for (int s = 0; s < NSLOTS; s++) HIP_CHECK(hipMalloc(&d_own[s], BINS * sizeof(uint32_t)));

  // warm up so the first timed slot does not pay for module load.
  gen_keys(call_seed(seed, 999, 0), n, d_keys);
  slot_golden(d_keys, n, d_shared);
  HIP_CHECK(hipMemset(d_shared, 0, BINS * sizeof(uint32_t)));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<Result> res(NSLOTS);
  std::vector<uint32_t> ref, out;

  for (int r = 0; r < rounds; r++) {
    if (!hardened) {
      // plain grader. one input per round, one reference.
      gen_keys(call_seed(seed, r, 0), n, d_keys);
      cpu_reference(d_keys, n, h_keys, ref);
    }
    for (int s = 0; s < NSLOTS; s++) {
      uint32_t* d_hist = hardened ? d_own[s] : d_shared;
      if (hardened) {
        gen_keys(call_seed(seed, r, s), n, d_keys);
        cpu_reference(d_keys, n, h_keys, ref);
        HIP_CHECK(hipMemset(d_hist, 0xFF, BINS * sizeof(uint32_t)));
        HIP_CHECK(hipDeviceSynchronize());
      }
      (void)hipGetLastError();  // clear any stale error

      double t0 = now_ms();
      SLOTS[s].fn(d_keys, n, d_hist);
      double t_ret = now_ms();
      double t_end = t_ret;
      bool ok = true;
      if (hardened) {
        hipError_t e1 = hipDeviceSynchronize();
        hipError_t e2 = hipGetLastError();
        t_end = now_ms();
        if (e1 != hipSuccess || e2 != hipSuccess) {
          fprintf(stderr, "slot %s: device error %s / %s\n", SLOTS[s].name, hipGetErrorString(e1), hipGetErrorString(e2));
          ok = false;
        }
      }
      double total = t_end - t0;
      double pending = t_end - t_ret;
      if (hardened && pending > std::max(0.05, 0.05 * total)) res[s].async_return = true;

      // hipMemcpy is ordered after null stream work, so even in the plain grader the
      // async cheat has finished by the time the output is read.
      uint32_t bad = count_mismatch(d_hist, ref, out);
      if (bad) ok = false;
      res[s].max_mismatch = std::max(res[s].max_mismatch, bad);
      res[s].correct = res[s].correct && ok;
      res[s].times_ms.push_back(total);
      res[s].pending_ms.push_back(pending);
    }
  }

  double t_naive = median(res[0].times_ms);
  double t_golden = median(res[3].times_ms);
  for (int s = 0; s < NSLOTS; s++) {
    bool ok = res[s].correct && !(hardened && res[s].async_return);
    res[s].score = score_of(median(res[s].times_ms), t_naive, t_golden, ok);
  }

  printf("\n%-16s %-8s %10s %10s %7s\n", "slot", "correct", "time_ms", "pending_ms", "score");
  for (int s = 0; s < NSLOTS; s++) {
    printf("%-16s %-8s %10.3f %10.3f %7.3f%s\n", SLOTS[s].name, res[s].correct ? "yes" : "no",
           median(res[s].times_ms), median(res[s].pending_ms), res[s].score,
           (hardened && res[s].async_return) ? "  async return, scored 0" : "");
  }

  if (json_path) {
    FILE* f = fopen(json_path, "w");
    if (!f) { perror("json"); return 2; }
    fprintf(f, "{\n  \"mode\": \"%s\",\n  \"device\": {\"name\": \"%s\", \"arch\": \"%s\", \"cus\": %d},\n",
            hardened ? "hardened" : "unhardened", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    fprintf(f, "  \"n\": %u, \"bins\": %u, \"rounds\": %d, \"timing\": \"measured\",\n  \"slots\": [\n", n, BINS, rounds);
    for (int s = 0; s < NSLOTS; s++) {
      fprintf(f, "    {\"name\": \"%s\", \"correct\": %s, \"time_ms\": %.4f, \"times_ms\": %s, \"pending_ms\": %.4f, "
                 "\"async_return\": %s, \"mismatch_bins\": %u, \"score\": %.4f}%s\n",
              SLOTS[s].name, res[s].correct ? "true" : "false", median(res[s].times_ms),
              json_times(res[s].times_ms).c_str(), median(res[s].pending_ms),
              res[s].async_return ? "true" : "false", res[s].max_mismatch, res[s].score, s + 1 < NSLOTS ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("wrote %s\n", json_path);
  }

  for (int s = 0; s < NSLOTS; s++) HIP_CHECK(hipFree(d_own[s]));
  HIP_CHECK(hipFree(d_shared));
  HIP_CHECK(hipHostFree(h_keys));
  HIP_CHECK(hipFree(d_keys));
  return 0;
}
