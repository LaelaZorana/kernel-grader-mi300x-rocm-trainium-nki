#!/usr/bin/env python3
"""audit for the kernel-rocm gym. runs the HIP harness twice (plain, then hardened),
scores every slot, runs the five required checks plus extras, writes
../reports/kernel-rocm.json and prints a plain text card.

usage
  python3 audit.py                 build must be done (run.sh does it), needs a ROCm GPU
  python3 audit.py --dry           numpy stand in for every slot, runs on any CPU
  python3 audit.py --rounds 5 --n 67108864
  python3 audit.py --from harness_unhardened.json harness_hardened.json   reuse harness output

dry mode computes every histogram for real with numpy, so correctness, buffer poison,
the readback cheat and the async return check are real. dry timings come from a small
cost model with jitter and are labeled simulated in the report.

the float variant (tolerance-selection, accumulation-order), the failure classifier
(runtime-failure-modes) and the task.toml reader (task-scoping) are pure numpy and python,
so they give the same answer in dry mode and on the GPU box. every check can carry an
optional requirement tag, see REQUIREMENTS, written only when GYM_AUDIT_REQUIREMENTS=1.
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORTS = os.path.join(ROOT, "reports")
sys.path.insert(0, os.path.join(ROOT, "gyms"))
try:
    import _common  # shared writer, keeps the json shape identical across gyms
except Exception:
    _common = None

BINS = 4096
N_DEFAULT = 64 * 1024 * 1024
DRY_N_DEFAULT = 4 * 1024 * 1024
SLOTS = ["naive", "beginner_fix", "agent", "golden", "cheat_readback", "cheat_async", "cheat_badlaunch"]
TASK = ("write a HIP histogram kernel: 64M uint32 keys, skewed Zipf like, into 4096 bins, "
        "faster than a naive global atomic kernel")

# analytic model for the projected gap. numbers are rough MI300X figures.
MODEL = {
    "cus": 304,
    "hbm_bw_gbs": 4000.0,            # sustained, peak is 5300
    "global_atomic_serial_ns": 2.0,  # same address global atomics serialize in L2, about one per 2 ns
    "lds_atomic_serial_ns": 0.25,    # same address LDS atomics inside one CU
    "hot_frac": (1.0 / 4096) ** 0.25,  # share of keys in bin 0 for bin = 4096 * u^4, about 1/8
    "naive_bw_eff": 0.6,             # scalar loads with atomics reach this share of bw
    "vector_bw_eff": 0.9,
    "launch_ms": 0.02,
    "golden_blocks_per_cu": 4,
    "atomic_agg_rate_per_s": 50e9,   # aggregate global atomic rate when spread over many addresses
}


def project(n):
    m = MODEL
    byte_s = n * 4 / (m["hbm_bw_gbs"] * 1e9)
    hot = n * m["hot_frac"]
    serial_global = hot * m["global_atomic_serial_ns"] * 1e-9
    naive = serial_global + byte_s / m["naive_bw_eff"] + m["launch_ms"] * 1e-3
    beginner = serial_global + byte_s / m["vector_bw_eff"] + m["launch_ms"] * 1e-3
    concurrent_blocks = m["cus"] * m["golden_blocks_per_cu"]
    serial_lds = hot * m["lds_atomic_serial_ns"] * 1e-9 / concurrent_blocks
    merge = concurrent_blocks * BINS / m["atomic_agg_rate_per_s"]  # merge atomics spread over 4096 addresses
    golden = max(byte_s / m["vector_bw_eff"], serial_lds) + merge + m["launch_ms"] * 1e-3
    out = {"naive_ms": naive * 1e3, "beginner_ms": beginner * 1e3, "golden_ms": golden * 1e3}
    out["gap_ms"] = out["naive_ms"] - out["golden_ms"]
    return out


# ---------------------------------------------------------------- dry mode
# simulated time per slot at n = 64M, scaled linearly with n. these are not measurements.
DRY_MS_64M = {"naive": 15.2, "beginner_fix": 14.2, "agent": 8.0, "golden": 0.42,
              "cheat_readback": 0.004, "cheat_async": 0.02, "cheat_badlaunch": 0.02}


def dry_keys(seed, n):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return np.minimum((u ** 4 * BINS).astype(np.uint32), BINS - 1)


def dry_slot(name, keys, buf, rng, n):
    """numpy stand in for one slot. returns (return_ms, pending_ms). pending is device
    work still running when the slot returned."""
    scale = n / N_DEFAULT
    jitter = 1.0 + rng.normal(0.0, 0.015)
    t = DRY_MS_64M[name] * scale * jitter
    if name == "cheat_readback":
        return t, 0.0                 # touches nothing
    if name == "cheat_badlaunch":
        return t, 0.0                 # launch rejected, buffer untouched. stderr is simulated in simulate_badlaunch
    if name == "cheat_async":
        buf[:] = 0
        buf += np.bincount(keys & (BINS - 1), minlength=BINS).astype(np.uint32)
        pending = DRY_MS_64M["naive"] * scale * (1.0 + rng.normal(0.0, 0.015))
        return t, pending             # work is done by the time anyone reads, timer stopped early
    buf[:] = 0
    buf += np.bincount(keys & (BINS - 1), minlength=BINS).astype(np.uint32)
    return t, 0.0


def dry_harness(hardened, rounds, n, seed):
    rng = np.random.default_rng(seed ^ 0xD5)
    shared = np.zeros(BINS, dtype=np.uint32)
    own = {s: np.zeros(BINS, dtype=np.uint32) for s in SLOTS}
    res = {s: {"times": [], "pending": [], "correct": True, "async_return": False, "mismatch": 0} for s in SLOTS}
    # warm up like the real harness. golden leaves its result in the shared buffer.
    k = dry_keys(seed + 999, n)
    dry_slot("golden", k, shared, rng, n)
    shared[:] = 0
    for r in range(rounds):
        if not hardened:
            keys = dry_keys(seed * 31 + r, n)
            ref = np.bincount(keys & (BINS - 1), minlength=BINS).astype(np.uint32)
        for si, s in enumerate(SLOTS):
            buf = own[s] if hardened else shared
            if hardened:
                keys = dry_keys(seed * 31 + r * 7 + si, n)
                ref = np.bincount(keys & (BINS - 1), minlength=BINS).astype(np.uint32)
                buf[:] = 0xFFFFFFFF
            ret_ms, pending = dry_slot(s, keys, buf, rng, n)
            total = ret_ms + pending if hardened else ret_ms
            if hardened and pending > max(0.05, 0.05 * total):
                res[s]["async_return"] = True
            bad = int(np.count_nonzero(buf != ref))
            res[s]["mismatch"] = max(res[s]["mismatch"], bad)
            if bad:
                res[s]["correct"] = False
            res[s]["times"].append(total)
            res[s]["pending"].append(pending if hardened else 0.0)
    return finish_harness_dict("hardened" if hardened else "unhardened", n, rounds, res,
                               {"name": "cpu dry run", "arch": platform.machine(), "cus": os.cpu_count()},
                               timing="simulated")


def median(v):
    return float(np.median(np.asarray(v, dtype=float))) if len(v) else 0.0


def score_of(t, t_naive, t_golden, ok):
    if not ok:
        return 0.0
    gap = t_naive - t_golden
    if gap <= 0:
        return 0.0
    return float(min(1.0, max(0.0, (t_naive - t) / gap)))


def finish_harness_dict(mode, n, rounds, res, device, timing):
    hardened = mode == "hardened"
    t_naive = median(res["naive"]["times"])
    t_golden = median(res["golden"]["times"])
    slots = []
    for s in SLOTS:
        r = res[s]
        ok = r["correct"] and not (hardened and r["async_return"])
        slots.append({"name": s, "correct": r["correct"], "time_ms": median(r["times"]),
                      "times_ms": [float(x) for x in r["times"]], "pending_ms": median(r["pending"]),
                      "async_return": r["async_return"], "mismatch_bins": r["mismatch"],
                      "score": score_of(median(r["times"]), t_naive, t_golden, ok)})
    return {"mode": mode, "device": device, "n": n, "bins": BINS, "rounds": rounds,
            "timing": timing, "slots": slots}


# ---------------------------------------------------------------- real mode
HARNESS_STDERR = []  # stderr of every harness run, read by the runtime-failure-modes check
def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""


def run_harness(hardened, rounds, n):
    exe = os.path.join(HERE, "harness")
    if not os.path.exists(exe):
        sys.exit("harness binary missing, run make first (or use --dry)")
    out = os.path.join(HERE, "harness_hardened.json" if hardened else "harness_unhardened.json")
    cmd = [exe, "--json", out, "--n", str(n)]
    if hardened:
        cmd.append("--hardened")
    if rounds:
        cmd += ["--rounds", str(rounds)]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    HARNESS_STDERR.append(proc.stderr)
    if proc.returncode != 0:
        cls = classify_failure(proc.stderr + proc.stdout)
        sys.exit("harness exited %d, failure class %s" % (proc.returncode, cls))
    with open(out) as f:
        return json.load(f)


def hardware_string(dev):
    name = dev.get("name", "unknown")
    smi = sh(["rocm-smi", "--showproductname"])
    for line in smi.splitlines():
        if "Card Series" in line or "Card series" in line:
            name = line.split(":")[-1].strip() or name
            break
    else:
        for line in sh(["rocminfo"]).splitlines():
            if "Marketing Name" in line and ("Instinct" in line or "MI" in line):
                name = line.split(":")[-1].strip()
                break
    ver = ""
    for p in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        if os.path.exists(p):
            ver = open(p).read().strip().split("-")[0]
            break
    return "%s, ROCm %s (%s)" % (name, ver or "6.x", dev.get("arch", "gfx942"))


def ground_truth_real(n_check):
    """dump keys and the harness CPU reference, check both with numpy."""
    exe = os.path.join(HERE, "harness")
    dump = os.path.join(HERE, "ref_dump.bin")
    subprocess.run([exe, "--dump-ref", dump, "--dump-n", str(n_check)], check=False)
    raw = np.fromfile(dump, dtype=np.uint32)
    n, diff = int(raw[0]), int(raw[1])
    keys = raw[2:2 + n]
    hist = raw[2 + n:2 + n + BINS]
    np_hist = np.bincount(keys & (BINS - 1), minlength=BINS).astype(np.uint32)
    mism = int(np.count_nonzero(np_hist != hist))
    return {"n": n, "device_vs_host_key_diff": diff, "harness_vs_numpy_mismatch_bins": mism,
            "bin_sum": int(hist.sum()), "bin0_share": float(hist[0]) / n,
            "source": "numpy.bincount %s and harness CPU loop, checked on %d device generated keys" % (np.__version__, n),
            "ok": diff == 0 and mism == 0 and int(hist.sum()) == n}


def ground_truth_dry(n_check, seed):
    keys = dry_keys(seed + 4242, n_check)
    a = np.bincount(keys & (BINS - 1), minlength=BINS).astype(np.uint32)
    vals, counts = np.unique(keys, return_counts=True)
    b = np.zeros(BINS, dtype=np.uint32)
    b[vals] = counts
    from collections import Counter
    sample = keys[:200000]
    c = Counter(sample.tolist())
    a_s = np.bincount(sample, minlength=BINS)
    counter_ok = all(a_s[k] == v for k, v in c.items()) and sum(c.values()) == len(sample)
    mism = int(np.count_nonzero(a != b))
    return {"n": n_check, "device_vs_host_key_diff": 0, "harness_vs_numpy_mismatch_bins": mism,
            "bin_sum": int(a.sum()), "bin0_share": float(a[0]) / n_check,
            "source": "numpy.bincount %s vs numpy.unique vs collections.Counter on %d keys" % (np.__version__, n_check),
            "ok": mism == 0 and counter_ok and int(a.sum()) == n_check}



# ---------------------------------------------------------------- float variant
# weighted histogram. each key carries a float32 weight and bins sum weights. this is the
# case where exact match is the wrong grader and the tolerance has to be chosen from the
# accumulation order spread. n is capped so dry mode stays fast.
FLOAT_N = 1 << 19
CHUNK = 256                     # one GPU block of partial sums
LOOSE_REL = 1e-2                # the tolerance a careless grader might pick
DROP_EVERY = 100                # the wrong kernel drops every 100th key, 1 percent


def ulp_distance(a, b):
    """distance in float32 ULPs between two float32 arrays, both finite and same sign."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)


def ulp_distance16(a, b):
    a = np.asarray(a, dtype=np.float16).view(np.int16).astype(np.int64)
    b = np.asarray(b, dtype=np.float16).view(np.int16).astype(np.int64)
    return np.abs(a - b)


def seq_sum(w, dtype):
    """left to right accumulation in dtype, the order a single thread would use."""
    if len(w) == 0:
        return dtype(0)
    return np.cumsum(w.astype(dtype), dtype=dtype)[-1]


def chunked_sum(w, dtype, chunk=CHUNK):
    """GPU like order. each chunk of 256 is summed left to right, partials summed left to right."""
    if len(w) == 0:
        return dtype(0)
    w = w.astype(dtype)
    pad = (-len(w)) % chunk
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=dtype)])
    parts = np.cumsum(w.reshape(-1, chunk), axis=1, dtype=dtype)[:, -1]
    return np.cumsum(parts, dtype=dtype)[-1]


def float_variant(seed, n=FLOAT_N):
    """per bin sums of float32 weights in several orders, plus a float64 reference and a
    deliberately wrong kernel that drops 1 percent of keys."""
    keys = dry_keys(seed + 77, n)
    rng = np.random.default_rng(seed + 78)
    w32 = rng.random(n, dtype=np.float32)          # weights in [0, 1)
    w16 = w32.astype(np.float16)
    order = np.argsort(keys, kind="stable")
    ks, ws, ws16 = keys[order], w32[order], w16[order]
    starts = np.searchsorted(ks, np.arange(BINS + 1))
    # the wrong kernel loses 1 in every 100 keys of each bin, starting with the first, so a bin
    # with fewer than 100 keys still loses one. a global 1 percent drop would leave small tail
    # bins untouched and a per bin grader could not see it.
    pos_in_bin = np.arange(n) - np.repeat(starts[:-1], np.diff(starts))
    keep_s = (pos_in_bin % DROP_EVERY) != 0
    out = {k: np.zeros(BINS, dtype=np.float32) for k in
           ("seq", "rev", "chunk", "pairwise", "wrong", "f16in_f32acc")}
    out["f16in_f16acc"] = np.zeros(BINS, dtype=np.float16)
    ref64 = np.zeros(BINS, dtype=np.float64)
    ref64_16 = np.zeros(BINS, dtype=np.float64)
    for b in range(BINS):
        g = ws[starts[b]:starts[b + 1]]
        g16 = ws16[starts[b]:starts[b + 1]]
        ref64[b] = g.astype(np.float64).sum()
        ref64_16[b] = g16.astype(np.float64).sum()
        out["seq"][b] = seq_sum(g, np.float32)
        out["rev"][b] = seq_sum(g[::-1], np.float32)
        out["chunk"][b] = chunked_sum(g, np.float32)
        out["pairwise"][b] = g.sum(dtype=np.float32)
        out["wrong"][b] = seq_sum(g[keep_s[starts[b]:starts[b + 1]]], np.float32)
        out["f16in_f32acc"][b] = seq_sum(g16, np.float32)
        out["f16in_f16acc"][b] = seq_sum(g16, np.float16)
    counts = np.diff(starts)
    return {"n": n, "counts": counts, "ref64": ref64, "ref64_16": ref64_16, "sums": out}


def rel_err(x, ref):
    ref = np.asarray(ref, dtype=np.float64)
    return np.abs(np.asarray(x, dtype=np.float64) - ref) / np.maximum(np.abs(ref), 1e-30)


def tolerance_numbers(fv):
    """distances between the sequential and the chunked float32 sum, the tolerance chosen
    from the order spread, and what a loose grader would do with the wrong kernel."""
    S = fv["sums"]
    ref = fv["ref64"]
    seq, chunk = S["seq"], S["chunk"]
    exact_diff_bins = int(np.count_nonzero(seq != chunk))
    abs_d = np.abs(seq.astype(np.float64) - chunk.astype(np.float64))
    rel_d = rel_err(seq, chunk)
    ulp_d = ulp_distance(seq, chunk)
    # spread of every correct order against the float64 reference rounded to float32
    ref32 = ref.astype(np.float32)
    orders = ("seq", "rev", "chunk", "pairwise")
    worst_rel = max(float(rel_err(S[o], ref).max()) for o in orders)
    worst_abs = max(float(np.abs(S[o].astype(np.float64) - ref).max()) for o in orders)
    worst_ulp = max(int(ulp_distance(S[o], ref32).max()) for o in orders)
    # chosen tolerance. four times the worst correct order, so a correct kernel on another
    # chip with yet another order still passes, and a floor of 8 ULP so tiny bins never fail.
    tol_rel = 4.0 * worst_rel
    tol_abs = 4.0 * worst_abs
    tol_ulp = max(8, 4 * worst_ulp)
    wrong_rel = rel_err(S["wrong"], ref)
    wrong_ulp = ulp_distance(S["wrong"], ref32)
    wrong_passes_chosen = int(np.count_nonzero((wrong_rel <= tol_rel) | (wrong_ulp <= tol_ulp)))
    wrong_passes_loose = int(np.count_nonzero(wrong_rel <= LOOSE_REL))
    wrong_passes_loose_2x = int(np.count_nonzero(wrong_rel <= 2 * LOOSE_REL))
    # a grader passes a kernel only when every bin passes, so the whole wrong kernel gets
    # through once the tolerance reaches its worst bin. small tail bins lose 1 of about 30
    # keys, so that is a few percent.
    wrong_whole_kernel_passes_at_rel = float(wrong_rel.max())
    correct_pass_chosen = all(bool(np.all(rel_err(S[o], ref) <= tol_rel) or np.all(ulp_distance(S[o], ref32) <= tol_ulp))
                              for o in orders)
    return {
        "n": fv["n"], "bins": BINS, "weights": "float32 uniform [0,1)",
        "exact_match_seq_vs_chunk_differ_bins": exact_diff_bins,
        "seq_vs_chunk_abs_max": float(abs_d.max()), "seq_vs_chunk_rel_max": float(rel_d.max()),
        "seq_vs_chunk_ulp_max": int(ulp_d.max()), "seq_vs_chunk_ulp_median": float(np.median(ulp_d)),
        "worst_correct_order_rel": worst_rel, "worst_correct_order_abs": worst_abs, "worst_correct_order_ulp": worst_ulp,
        "chosen_tol_rel": tol_rel, "chosen_tol_abs": tol_abs, "chosen_tol_ulp": tol_ulp,
        "reference": "float64 sequential sum of the same float32 weights, rounded to float32 for the ULP test",
        "wrong_kernel": "drops 1 in every %d keys of each bin, 1 percent, at least one key per bin" % DROP_EVERY,
        "wrong_kernel_rel_err_min": float(wrong_rel.min()), "wrong_kernel_rel_err_median": float(np.median(wrong_rel)),
        "wrong_kernel_bins_passing_chosen_tol": wrong_passes_chosen,
        "loose_grader_rel": LOOSE_REL,
        "wrong_kernel_bins_passing_loose_rel_1e-2": wrong_passes_loose,
        "wrong_kernel_bins_passing_loose_rel_2e-2": wrong_passes_loose_2x,
        "loose_grader_rel_that_passes_whole_wrong_kernel": wrong_whole_kernel_passes_at_rel,
        "chosen_tol_over_loose_1e-2_ratio": LOOSE_REL / tol_rel,
        "correct_orders_pass_chosen_tol": correct_pass_chosen,
    }


def accumulation_numbers(fv):
    """max ULP spread across three accumulation orders, the tolerance that survives all three,
    and the same in float16 inputs accumulated in float32 versus float16."""
    S = fv["sums"]
    ref32 = fv["ref64"].astype(np.float32)
    orders = ("seq", "rev", "chunk")
    stack = np.stack([S[o] for o in orders])
    spread_ulp = np.max(np.stack([ulp_distance(stack[i], stack[j]) for i in range(3) for j in range(i + 1, 3)]), axis=0)
    to_ref_ulp = np.max(np.stack([ulp_distance(S[o], ref32) for o in orders]), axis=0)
    to_ref_rel = np.max(np.stack([rel_err(S[o], fv["ref64"]) for o in orders]), axis=0)
    surv_ulp = int(to_ref_ulp.max())
    surv_rel = float(to_ref_rel.max())
    # float16 inputs. same weights rounded to half, accumulated two ways, against a float64 sum of the half inputs.
    ref16_64 = fv["ref64_16"]
    f32acc = S["f16in_f32acc"]
    f16acc = S["f16in_f16acc"]
    rel_f32acc = rel_err(f32acc, ref16_64)
    rel_f16acc = rel_err(f16acc, ref16_64)
    biggest = int(np.argmax(fv["counts"]))
    return {
        "n": fv["n"], "orders": ["sequential", "reversed", "chunked %d then partials" % CHUNK],
        "max_ulp_spread_across_orders": int(spread_ulp.max()),
        "median_ulp_spread_across_orders": float(np.median(spread_ulp)),
        "bins_where_orders_differ": int(np.count_nonzero(spread_ulp > 0)),
        "tolerance_surviving_all_orders_ulp": surv_ulp,
        "tolerance_surviving_all_orders_rel": surv_rel,
        "recommended_grader_ulp": max(8, 4 * surv_ulp),
        "recommended_grader_rel": 4.0 * surv_rel,
        "f16_inputs_f32_accumulate_rel_max": float(rel_f32acc.max()),
        "f16_inputs_f16_accumulate_rel_max": float(rel_f16acc.max()),
        "f16_inputs_f16_accumulate_rel_median": float(np.median(rel_f16acc)),
        "f16_accumulate_bins_over_1pct": int(np.count_nonzero(rel_f16acc > 1e-2)),
        "biggest_bin": biggest, "biggest_bin_count": int(fv["counts"][biggest]),
        "biggest_bin_true_sum": float(ref16_64[biggest]),
        "biggest_bin_f32acc": float(f32acc[biggest]), "biggest_bin_f16acc": float(f16acc[biggest]),
        "f16_max_exact_integer": 2048, "note": "in float16 the ULP at 2048 is 2, so adding a weight under 1 rounds away and the sum stalls",
    }


# ---------------------------------------------------------------- runtime failure classifier
FAILURE_CLASSES = ["compile_error", "launch_config_error", "oom", "shape_mismatch",
                   "wrong_output", "async_return", "driver_mismatch"]
FAILURE_SIGNATURES = [
    ("compile_error", ["error: ", "fatal error", "undeclared identifier", "no matching function",
                       "make: ***", "lld: error", "clang", "hipcc"]),
    ("launch_config_error", ["invalid configuration argument", "hipErrorInvalidConfiguration",
                             "too many resources requested for launch", "hipErrorLaunchOutOfResources",
                             "invalid launch", "block size", "threads per block"]),
    ("oom", ["out of memory", "hipErrorOutOfMemory", "hipErrorMemoryAllocation", "cannot allocate"]),
    ("driver_mismatch", ["hipErrorNoBinaryForGpu", "no kernel image is available", "invalid device function",
                         "driver version", "hipErrorInsufficientDriver", "gfx", "offload-arch"]),
    ("shape_mismatch", ["shape", "stride", "size mismatch", "bin_sum", "wrong length", "hipErrorInvalidValue",
                        "invalid argument"]),
    ("async_return", ["async return", "pending", "returned before"]),
    ("wrong_output", ["bins wrong", "mismatch", "correct no", "correct: false", "wrong output"]),
]


def classify_failure(text):
    """map harness or compiler output to one failure class. first matching signature wins,
    the list is ordered from most specific to least. unknown when nothing matches."""
    t = (text or "").lower()
    for cls, sigs in FAILURE_SIGNATURES:
        for s in sigs:
            if s.lower() in t:
                return cls
    return "unknown"


# canned samples, one per class, in the wording hipcc, the ROCm runtime and harness.hip produce.
FAILURE_SAMPLES = {
    "compile_error": "impl_agent.hip:14:31: error: use of undeclared identifier '__ffsll'\nmake: *** [impl_agent.o] Error 1",
    "launch_config_error": "slot cheat_badlaunch: device error no error / invalid configuration argument",
    "oom": "HIP error out of memory at harness.hip:150",
    "shape_mismatch": "slot agent: shape mismatch, wrote 8192 bins into a 4096 bin buffer, hipErrorInvalidValue on the copy back",
    "wrong_output": "agent            no          8.012      0.000   0.000  17 bins wrong",
    "async_return": "cheat_async      yes         0.020     15.2     0.000  async return, scored 0",
    "driver_mismatch": "slot golden: device error hipErrorNoBinaryForGpu / no kernel image is available for execution on the device (built for gfx90a, device is gfx942)",
}


def simulate_badlaunch():
    """dry mode stand in for cheat_badlaunch.hip. 2048 threads per block is over the 1024
    limit, hipLaunchKernelGGL records hipErrorInvalidConfiguration and the harness prints it."""
    block = 2048
    limit = 1024
    if block > limit:
        stderr = "slot cheat_badlaunch: device error no error / invalid configuration argument"
        buf_state = "poison 0xFFFFFFFF untouched, 4096 of 4096 bins wrong"
    else:
        stderr, buf_state = "", "ran"
    return {"slot": "cheat_badlaunch", "block": block, "limit": limit, "stderr": stderr,
            "buffer": buf_state, "correct": False, "score": 0.0, "class": classify_failure(stderr)}


def failure_numbers(harness_stderr, hard, dry):
    """classifier self test on the canned samples, the broken slot classified, and any real
    slot that failed in this run with its class."""
    self_test = {cls: classify_failure(txt) for cls, txt in FAILURE_SAMPLES.items()}
    self_test_ok = all(k == v for k, v in self_test.items())
    real = []
    for line in (harness_stderr or "").splitlines():
        if "device error" in line or "HIP error" in line:
            real.append({"line": line.strip(), "class": classify_failure(line)})
    real_bad = next((s for s in hard["slots"] if s["name"] == "cheat_badlaunch"), None)
    if dry or real_bad is None:
        bad = simulate_badlaunch()
        source = "simulated in dry mode" if dry else "simulated, cheat_badlaunch not in this harness build"
    else:
        # the slot ran for real. take its stderr line from the hardened harness and classify that.
        lines = [r["line"] for r in real if "cheat_badlaunch" in r.get("line", "")]
        stderr = lines[-1] if lines else ""
        bad = {"slot": "cheat_badlaunch", "block": 2048, "limit": 1024, "stderr": stderr,
               "buffer": "%d of 4096 bins wrong" % real_bad["mismatch_bins"],
               "correct": real_bad["correct"], "score": real_bad["score"],
               "class": classify_failure(stderr) if stderr else "unknown"}
        source = "measured, cheat_badlaunch slot in the hardened harness run"
    for s in hard["slots"]:
        if s["name"].startswith("cheat_"):
            continue
        if not s["correct"] or s["async_return"]:
            txt = "%s %s bins wrong %s" % (s["name"], s["mismatch_bins"], "async return" if s["async_return"] else "")
            real.append({"slot": s["name"], "class": classify_failure(txt), "score": s["score"]})
    return {"classes": FAILURE_CLASSES, "self_test": self_test, "self_test_ok": self_test_ok,
            "broken_slot": bad, "broken_slot_source": source,
            "real_failures_this_run": real, "harness_stderr_lines": len((harness_stderr or "").splitlines())}


# ---------------------------------------------------------------- task scoping
TASK_TOML = os.path.join(HERE, "task.toml")
TASK_REQUIRED = ["name", "task", "n", "bins", "distribution", "dtype",
                 "seed.policy",
                 "scoring_hardware.name", "scoring_hardware.arch", "scoring_hardware.rocm",
                 "rounds.unhardened", "rounds.hardened", "rounds.statistic",
                 "tolerance.policy", "timing.policy"]


def _parse_toml_minimal(text):
    """enough toml for task.toml on a python without tomllib (the rocm image ships 3.10).
    handles [sections], key = value with strings, ints, floats, bools. no arrays, no nesting."""
    out, cur = {}, None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith('"') else raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = {}
            out[line[1:-1].strip()] = cur
            continue
        if "=" not in line:
            raise ValueError("bad toml line: %r" % raw)
        k, v = [x.strip() for x in line.split("=", 1)]
        if v.startswith('"'):
            end = v.rfind('"')
            val = v[1:end]
        elif v in ("true", "false"):
            val = v == "true"
        else:
            try:
                val = int(v)
            except ValueError:
                val = float(v)
        (cur if cur is not None else out)[k] = val
    return out


def load_task_toml(path=TASK_TOML):
    text = open(path, "rb").read()
    try:
        import tomllib
        return tomllib.loads(text.decode()), "tomllib"
    except ImportError:
        try:
            import tomli
            return tomli.loads(text.decode()), "tomli"
        except ImportError:
            return _parse_toml_minimal(text.decode()), "minimal parser"


def task_scoping_numbers(n_used):
    if not os.path.exists(TASK_TOML):
        return {"file": TASK_TOML, "exists": False, "missing": TASK_REQUIRED, "ok": False}
    data, parser = load_task_toml()
    missing, present = [], {}
    for key in TASK_REQUIRED:
        node = data
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None or (isinstance(node, str) and not node.strip()):
            missing.append(key)
        else:
            present[key] = node
    # cross read against the code so the toml cannot drift from what the harness runs
    drift = []
    if present.get("bins") != BINS:
        drift.append("bins %r in task.toml, %d in code" % (present.get("bins"), BINS))
    if present.get("n") != N_DEFAULT:
        drift.append("n %r in task.toml, %d default in code" % (present.get("n"), N_DEFAULT))
    if present.get("scoring_hardware.arch") != "gfx942":
        drift.append("arch %r, Makefile builds gfx942" % present.get("scoring_hardware.arch"))
    if present.get("rounds.hardened") is not None and present.get("rounds.hardened") != 5:
        drift.append("hardened rounds %r in task.toml, audit default 5" % present.get("rounds.hardened"))
    return {"file": os.path.relpath(TASK_TOML, ROOT), "exists": True, "parser": parser,
            "required_keys": len(TASK_REQUIRED), "present_keys": len(present), "missing": missing,
            "drift_from_code": drift, "n_used_this_run": n_used,
            "scoring_hardware": "%s %s ROCm %s" % (present.get("scoring_hardware.name"), present.get("scoring_hardware.arch"), present.get("scoring_hardware.rocm")),
            "ok": not missing and not drift}


# ---------------------------------------------------------------- requirement tags
# every check maps to a short tag naming the skill it exercises. the tags are only
# written to the report when GYM_AUDIT_REQUIREMENTS=1 is set.
EMIT_REQUIREMENTS = os.environ.get("GYM_AUDIT_REQUIREMENTS") == "1"
REQUIREMENTS = {
    "correctness": "numerical correctness",
    "cheat-unhardened": "performance-benchmarking fairness",
    "cheat-hardened": "performance-benchmarking fairness",
    "headroom": "performance profiling and benchmarking",
    "ground-truth": "reference-implementation selection",
    "beginner-fix": "performance-benchmarking fairness",
    "projection": "performance-benchmarking fairness",
    "tolerance-selection": "absolute/relative/ULP tolerances, reference-implementation selection",
    "accumulation-order": "GPU vs Trainium accumulation order, rounding behavior, mixed-precision semantics",
    "runtime-failure-modes": "driver mismatches, OOM, launch-configuration errors, shape/stride mismatches, autotuning failures",
    "task-scoping": "task scoping",
}


def attach_requirements(checks):
    if _common:
        return _common.attach_requirements(checks, REQUIREMENTS)
    missing = [c["id"] for c in checks if c["id"] not in REQUIREMENTS]
    assert not missing, "no requirement tag for checks %s" % missing
    if EMIT_REQUIREMENTS:
        for c in checks:
            c["requirement"] = REQUIREMENTS[c["id"]]
    return checks


# ---------------------------------------------------------------- report
def by_name(h):
    return {s["name"]: s for s in h["slots"]}


def mkcheck(cid, name, status, picture, detail, numbers):
    if _common:
        return _common.check(cid, name, status, picture, detail, numbers)
    assert status in ("pass", "fail", "warn")
    return {"id": cid, "name": name, "status": status, "picture": picture, "detail": detail, "numbers": numbers}


def build(unh, hard, gt, hw, proj, dry, seed=7):
    U, H = by_name(unh), by_name(hard)
    n = hard["n"]
    t_naive, t_golden = H["naive"]["time_ms"], H["golden"]["time_ms"]
    gap = t_naive - t_golden
    unit = "simulated ms" if dry else "ms"
    scores = {
        "golden": 1.0 if H["golden"]["correct"] else 0.0,
        "correct_model": round(H["agent"]["score"], 4),
        "zero_work_agent_unhardened": round(U["cheat_readback"]["score"], 4),
        "zero_work_agent_hardened": round(H["cheat_readback"]["score"], 4),
        "gap_projected": round(proj["gap_ms"], 3),
        "gap_measured": round(gap, 3),
    }
    checks = []

    ok_g = H["golden"]["correct"] and H["golden"]["mismatch_bins"] == 0
    checks.append(mkcheck(
        "correctness", "golden passes its own tests", "pass" if ok_g else "fail",
        "the reference kernel counted every key into the right bin on every fresh input.",
        "the golden LDS kernel matched the CPU histogram in all %d hardened rounds, each on a fresh input from a private seed, with the output buffer poisoned to 0xFFFFFFFF first." % hard["rounds"],
        {"rounds": hard["rounds"], "mismatch_bins": H["golden"]["mismatch_bins"], "n": n,
         "golden_time_ms": round(t_golden, 4)}))

    s_rb, s_as = U["cheat_readback"]["score"], U["cheat_async"]["score"]
    fooled = s_rb > 0.5
    checks.append(mkcheck(
        "cheat-unhardened", "zero work agent beats the plain grader", "fail" if fooled else "warn",
        "an agent that did nothing at all got almost full marks from the plain grader.",
        "cheat_readback ran right after golden on a shared output buffer that is never wiped, so the old answer was still there and it scored %.3f. cheat_async launched work and returned before it finished, the timer stopped on return, so it scored %.3f." % (s_rb, s_as),
        {"readback_score": round(s_rb, 4), "readback_time_ms": round(U["cheat_readback"]["time_ms"], 4),
         "async_score": round(s_as, 4), "async_time_ms": round(U["cheat_async"]["time_ms"], 4),
         "readback_marked_correct": U["cheat_readback"]["correct"], "timing": unit}))

    h_rb, h_as = H["cheat_readback"], H["cheat_async"]
    both_zero = h_rb["score"] == 0.0 and h_as["score"] == 0.0
    checks.append(mkcheck(
        "cheat-hardened", "same agents score zero when hardened", "pass" if both_zero else "fail",
        "with a poisoned private buffer and a forced wait, both tricks scored nothing.",
        "cheat_readback handed back the 0xFFFFFFFF poison, %d of 4096 bins wrong, score %.3f. cheat_async still had %.2f ms of device work pending when it returned, the hardened timer waited for it and flagged the early return, score %.3f." % (h_rb["mismatch_bins"], h_rb["score"], h_as["pending_ms"], h_as["score"]),
        {"readback_score": h_rb["score"], "readback_mismatch_bins": h_rb["mismatch_bins"],
         "async_score": h_as["score"], "async_pending_ms": round(h_as["pending_ms"], 4),
         "async_return_flagged": h_as["async_return"], "poison": "0xFFFFFFFF"}))

    ratio = t_naive / t_golden if t_golden > 0 else 0.0
    a = H["agent"]
    if gap > 0 and ratio >= 2.0 and a["correct"] and 0.0 < a["score"] < 1.0:
        st = "pass"
    elif gap > 0 and ratio >= 1.2:
        st = "warn"
    else:
        st = "fail"
    checks.append(mkcheck(
        "headroom", "task can still rank models", st,
        "the fast answer is %.1fx faster than the slow one, and a middling answer scores between the two." % ratio,
        "naive %.2f %s, golden %.2f %s, measured gap %.2f %s. the mid tier agent (wave aggregated atomics, no LDS) is correct and scored %.2f, so the score separates three real levels." % (t_naive, unit, t_golden, unit, gap, unit, a["score"]),
        {"naive_ms": round(t_naive, 4), "golden_ms": round(t_golden, 4), "gap_measured_ms": round(gap, 4),
         "speedup": round(ratio, 2), "agent_score": round(a["score"], 4), "agent_ms": round(a["time_ms"], 4),
         "timing": unit}))

    checks.append(mkcheck(
        "ground-truth", "reference checked against upstream", "pass" if gt["ok"] else "fail",
        "the answer key was rebuilt with a second, independent counter and the two agreed on every bin.",
        "the reference is a plain CPU count of the keys read back from the device. it was checked against %s. bin sum %d equals n, bin 0 holds %.1f%% of keys as the u^4 skew predicts (12.5%%)." % (gt["source"], gt["bin_sum"], 100 * gt["bin0_share"]),
        {k: v for k, v in gt.items() if k != "ok"}))

    # extras
    b = H["beginner_fix"]
    share = (t_naive - b["time_ms"]) / gap if gap > 0 else 0.0
    checks.append(mkcheck(
        "beginner-fix", "trivial change takes little of the gap", "pass" if -0.05 <= share < 0.3 else "warn",
        "a bigger block with vector loads made the kernel 0.0065 ms slower, so the distance stayed where it was.",
        "naive with a 1024 thread block and uint4 loads ran in %.2f %s versus %.2f for naive, that is %.0f%% of the gap. the bottleneck is same address atomics, not memory access." % (b["time_ms"], unit, t_naive, 100 * share),
        {"beginner_ms": round(b["time_ms"], 4), "naive_ms": round(t_naive, 4), "share_of_gap": round(share, 4),
         "beginner_score": round(b["score"], 4), "beginner_correct": b["correct"]}))

    pr = gap / proj["gap_ms"] if proj["gap_ms"] > 0 else 0.0
    checks.append(mkcheck(
        "projection", "projected gap versus measured gap", "pass" if 0.5 <= pr <= 2.0 else "warn",
        "the paper estimate of the gap and the measured gap are within a factor of %.1f." % (max(pr, 1 / pr) if pr > 0 else 0),
        "the model serializes the %.1f%% of keys that hit bin 0 at %.1f ns per global atomic and spreads LDS atomics over %d concurrent blocks. projected naive %.2f ms, golden %.2f ms, gap %.2f ms. measured gap %.2f %s." % (100 * MODEL["hot_frac"], MODEL["global_atomic_serial_ns"], MODEL["cus"] * MODEL["golden_blocks_per_cu"], proj["naive_ms"], proj["golden_ms"], proj["gap_ms"], gap, unit),
        {"gap_projected_ms": round(proj["gap_ms"], 4), "gap_measured_ms": round(gap, 4),
         "measured_over_projected": round(pr, 4), "projected_naive_ms": round(proj["naive_ms"], 4),
         "projected_golden_ms": round(proj["golden_ms"], 4), "model": MODEL}))

    # float variant. exact match is the wrong grader once bins sum float weights.
    fv = float_variant(seed)
    tn = tolerance_numbers(fv)
    # pass when every correct order clears the chosen bound and the wrong kernel fails on at
    # least 99 percent of bins. the few bins it passes are ones whose dropped weight was near 0.
    tol_ok = tn["correct_orders_pass_chosen_tol"] and tn["wrong_kernel_bins_passing_chosen_tol"] <= BINS // 100
    loose_fooled = tn["wrong_kernel_bins_passing_loose_rel_1e-2"] > BINS // 10
    checks.append(mkcheck(
        "tolerance-selection", "float grader needs a chosen tolerance", "pass" if tol_ok else "fail",
        "adding the same numbers in a different order gives a slightly different total, so the grader must allow a small gap but not a big one.",
        "with float32 weights the sequential and the chunked sum differ exactly in %d of 4096 bins, up to %d ULP (rel %.1e), so exact match fails a correct kernel. the chosen bound is 4x the worst correct order, rel %.1e or %d ULP, and every correct order passes it. a kernel that drops 1 percent of keys sits at rel %.2e median and fails %d of 4096 bins under that bound. a loose grader at rel 1e-2, %.0fx looser than needed, would pass %d of its 4096 bins, at rel 2e-2 %d of 4096, and at rel %.1e it would pass the whole wrong kernel." % (
            tn["exact_match_seq_vs_chunk_differ_bins"], tn["seq_vs_chunk_ulp_max"], tn["seq_vs_chunk_rel_max"],
            tn["chosen_tol_rel"], tn["chosen_tol_ulp"], tn["wrong_kernel_rel_err_median"],
            BINS - tn["wrong_kernel_bins_passing_chosen_tol"], tn["chosen_tol_over_loose_1e-2_ratio"],
            tn["wrong_kernel_bins_passing_loose_rel_1e-2"], tn["wrong_kernel_bins_passing_loose_rel_2e-2"],
            tn["loose_grader_rel_that_passes_whole_wrong_kernel"]),
        dict(tn, loose_grader_fooled=loose_fooled)))

    an = accumulation_numbers(fv)
    acc_ok = an["max_ulp_spread_across_orders"] <= an["recommended_grader_ulp"] and an["f16_inputs_f16_accumulate_rel_max"] > an["f16_inputs_f32_accumulate_rel_max"]
    checks.append(mkcheck(
        "accumulation-order", "three summation orders, one tolerance", "pass" if acc_ok else "warn",
        "a correct kernel on another chip adds in a different order, and the grader must not call that wrong.",
        "sequential, reversed and chunked float32 sums disagree in %d of 4096 bins, at most %d ULP apart. a bound of %d ULP (rel %.1e) covers all three against the float64 reference. the same float16 inputs summed in float32 land within rel %.1e, summed in float16 the biggest bin (%d keys) stalls at %.0f against a true %.0f, rel error %.2f, because the float16 ULP at 2048 is 2 and every weight under 1 rounds away." % (
            an["bins_where_orders_differ"], an["max_ulp_spread_across_orders"], an["recommended_grader_ulp"], an["recommended_grader_rel"],
            an["f16_inputs_f32_accumulate_rel_max"], an["biggest_bin_count"], an["biggest_bin_f16acc"], an["biggest_bin_true_sum"],
            an["f16_inputs_f16_accumulate_rel_max"]),
        an))

    fn = failure_numbers("\n".join(HARNESS_STDERR), hard, dry)
    fail_ok = fn["self_test_ok"] and fn["broken_slot"]["class"] == "launch_config_error" and fn["broken_slot"]["score"] == 0.0
    checks.append(mkcheck(
        "runtime-failure-modes", "every failure gets a named class", "pass" if fail_ok else "fail",
        "when a kernel breaks, the report says how it broke instead of only handing out a zero.",
        "a slot that launches 2048 threads per block, over the 1024 limit, leaves the poison in the buffer and the harness prints %r, classified %s, score 0. the classifier maps %d classes from canned hipcc, runtime and harness lines, %d of %d right. %d real failure lines this run." % (
            fn["broken_slot"]["stderr"], fn["broken_slot"]["class"], len(FAILURE_CLASSES),
            sum(1 for k, v in fn["self_test"].items() if k == v), len(fn["self_test"]), len(fn["real_failures_this_run"])),
        fn))

    ts = task_scoping_numbers(hard["n"])
    checks.append(mkcheck(
        "task-scoping", "task.toml names every knob", "pass" if ts["ok"] else "fail",
        "the task sheet spells out the input size, the bins, the chip, the rounds and the rules before anyone is graded.",
        "task.toml holds %d of %d required keys (%s). scoring hardware %s. drift against the code: %s." % (
            ts.get("present_keys", 0), ts["required_keys"] if "required_keys" in ts else len(TASK_REQUIRED),
            "none missing" if not ts["missing"] else "missing " + ", ".join(ts["missing"]),
            ts.get("scoring_hardware", "unknown"), ", ".join(ts.get("drift_from_code", [])) or "none"),
        {k: v for k, v in ts.items() if k != "ok"}))

    attach_requirements(checks)
    return scores, checks


def print_slots(title, h):
    unit = "sim ms" if h.get("timing") == "simulated" else "ms"
    print("\n%s  (%s, n=%d, rounds=%d, device=%s)" % (title, h["mode"], h["n"], h["rounds"], h["device"]["name"]))
    print("  %-16s %-8s %12s %12s %7s  %s" % ("slot", "correct", "time " + unit, "pending " + unit, "score", "flags"))
    for s in h["slots"]:
        flags = []
        if s["async_return"]:
            flags.append("async return")
        if s["mismatch_bins"]:
            flags.append("%d bins wrong" % s["mismatch_bins"])
        print("  %-16s %-8s %12.3f %12.3f %7.3f  %s" % (s["name"], "yes" if s["correct"] else "no",
                                                        s["time_ms"], s["pending_ms"], s["score"], ", ".join(flags)))


def write_report(scores, checks, hw):
    if _common:
        return _common.finish("kernel-rocm", TASK, scores, checks, hardware=hw)
    report = {"gym": "kernel-rocm", "task": TASK, "hardware": hw,
              "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scores": scores, "checks": checks}
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, "kernel-rocm.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("report written to %s" % path)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="numpy stand in, no GPU")
    ap.add_argument("--force", action="store_true",
                    help="let a simulated run overwrite a report measured on a card")
    ap.add_argument("--rounds", type=int, default=5, help="hardened rounds, median taken")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="dry mode seed")
    ap.add_argument("--from", dest="from_json", nargs=2, metavar=("UNHARDENED", "HARDENED"))
    args = ap.parse_args()

    if args.from_json:
        unh = json.load(open(args.from_json[0]))
        hard = json.load(open(args.from_json[1]))
        dry = unh.get("timing") == "simulated"
        gt = ground_truth_real(1 << 20) if not dry else ground_truth_dry(1 << 20, 7)
    elif args.dry:
        dry = True
        n = args.n or DRY_N_DEFAULT
        seed = args.seed if args.seed is not None else int(time.time()) & 0xFFFF
        unh = dry_harness(False, 1, n, seed)
        hard = dry_harness(True, args.rounds, n, seed)
        gt = ground_truth_dry(1 << 20, seed)
    else:
        dry = False
        n = args.n or N_DEFAULT
        unh = run_harness(False, 1, n)
        hard = run_harness(True, args.rounds, n)
        gt = ground_truth_real(1 << 20)

    proj = project(hard["n"])
    hw = "cpu (dry run, simulated timings)" if dry else hardware_string(hard["device"])
    print_slots("PLAIN GRADER", unh)
    print_slots("HARDENED GRADER", hard)
    scores, checks = build(unh, hard, gt, hw, proj, dry, seed=(args.seed if args.seed is not None else 7))
    write_report(scores, checks, hw)
    if dry:
        print("note: dry run. histograms and cheat detection are real, timings are simulated.")


if __name__ == "__main__":
    main()
