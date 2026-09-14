#!/usr/bin/env python3
"""nki kernels run in the neuron cpu simulator, no NeuronDevice.

two kernels
  1. tensor add, the quickstart kernel, one 128 row tile
  2. per bin float32 sum, 4096 bins x 128 weights, 32 tiles of 128 partitions,
     nisa.tensor_reduce over the free axis. this mirrors the per bin weighted
     sums of the mi300x histogram audit (n 524288, 4096 bins).

reference is a numpy float64 sequential sum. errors reported as abs, rel and
ulp (float32 ulp of the reference rounded to float32). tolerance bound is the
one the mi300x audit chose, 432 ulp and rel 2.5843945438686927e-05.

usage  python histogram_nki.py [--seed 3] [--json out.json]
"""
import argparse, hashlib, json, platform, sys, time
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa

BINS = 4096
KEYS_PER_BIN = 128
P = nl.tile_size.pmax  # 128
TOL_ULP = 432
TOL_REL = 2.5843945438686927e-05


@nki.jit
def tensor_add_kernel(a_input, b_input):
    assert a_input.shape == b_input.shape
    assert a_input.shape[0] <= nl.tile_size.pmax
    a_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)
    b_tile = nl.ndarray(shape=b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)
    c_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)
    c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=c_output, src=c_tile)
    return c_output


@nki.jit
def bin_sum_kernel(w):
    """w is (BINS, KEYS_PER_BIN) float32 in hbm. returns (BINS, 1) float32 row sums."""
    rows, cols = w.shape
    out = nl.ndarray(shape=(rows, 1), dtype=w.dtype, buffer=nl.shared_hbm)
    ntiles = rows // nl.tile_size.pmax
    for t in nl.affine_range(ntiles):
        r0 = t * nl.tile_size.pmax
        tile = nl.ndarray(shape=(nl.tile_size.pmax, cols), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=w[r0:r0 + nl.tile_size.pmax, :])
        acc = nl.ndarray(shape=(nl.tile_size.pmax, 1), dtype=w.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=acc, data=tile, op=nl.add, axis=(1,))
        nisa.dma_copy(dst=out[r0:r0 + nl.tile_size.pmax, :], src=acc)
    return out


def ulp32(x):
    x = np.asarray(x, dtype=np.float32)
    return np.spacing(np.abs(x)).astype(np.float64)


def errors(got, ref64):
    got = np.asarray(got, dtype=np.float32).astype(np.float64).ravel()
    ref64 = np.asarray(ref64, dtype=np.float64).ravel()
    ref32 = ref64.astype(np.float32)
    absd = np.abs(got - ref64)
    rel = absd / np.maximum(np.abs(ref64), 1e-30)
    ulp = np.abs(got - ref32.astype(np.float64)) / ulp32(ref32)
    return {
        "abs_max": float(absd.max()), "abs_median": float(np.median(absd)),
        "rel_max": float(rel.max()), "rel_median": float(np.median(rel)),
        "ulp_max": float(ulp.max()), "ulp_median": float(np.median(ulp)),
        "bins_over_432_ulp": int((ulp > TOL_ULP).sum()),
        "bins_over_rel_tol": int((rel > TOL_REL).sum()),
        "exact_match_bins": int((got == ref32.astype(np.float64)).sum()),
        "count": int(got.size),
    }


def run_on(kernel, device):
    """Return the callable for this kernel.

    The jit decorator dispatches on numpy input and runs standalone on a
    NeuronCore, so the device path is the bare kernel. The simulate wrapper
    runs the same kernel on the host inside the compiler instead.
    """
    return kernel if device else nki.simulate(kernel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", action="store_true",
                    help="run on an attached NeuronCore instead of the simulator")
    ap.add_argument("--bins-out", default=None,
                    help="write the raw per bin values so two runs compare bin by bin")
    a = ap.parse_args()
    RUN = lambda k: run_on(k, a.device)
    rng = np.random.default_rng(a.seed)
    res = {"seed": a.seed, "python": platform.python_version(), "machine": platform.machine(),
           "numpy": np.__version__, "nki": getattr(nki, "__version__", "?"),
           "execution": "device" if a.device else "simulator"}
    try:
        import neuronxcc
        res["neuronx_cc"] = neuronxcc.__version__
    except Exception as ex:  # noqa
        res["neuronx_cc"] = f"import failed {ex!r}"

    # 1 tensor add
    x = rng.random((P, 512), dtype=np.float32)
    y = rng.random((P, 512), dtype=np.float32)
    t0 = time.perf_counter()
    z = RUN(tensor_add_kernel)(x, y)
    t1 = time.perf_counter()
    ref = x.astype(np.float64) + y.astype(np.float64)
    res["tensor_add"] = {"shape": list(x.shape), "sim_wall_s": t1 - t0, **errors(z, ref),
                         "bit_exact_vs_float32_numpy": bool(np.array_equal(np.asarray(z), x + y))}

    # 2 per bin float32 sum
    w = rng.random((BINS, KEYS_PER_BIN), dtype=np.float32)
    res["input_sha256"] = hashlib.sha256(w.tobytes()).hexdigest()
    t0 = time.perf_counter()
    s = RUN(bin_sum_kernel)(w)
    t1 = time.perf_counter()
    s = np.asarray(s).reshape(BINS)
    ref64 = w.astype(np.float64).sum(axis=1)
    seq32 = np.zeros(BINS, np.float32)
    for k in range(KEYS_PER_BIN):
        seq32 += w[:, k]
    rev32 = np.zeros(BINS, np.float32)
    for k in reversed(range(KEYS_PER_BIN)):
        rev32 += w[:, k]
    chunk32 = w.reshape(BINS, 8, 16).sum(axis=2, dtype=np.float32).sum(axis=1, dtype=np.float32)
    orders = {"sequential": seq32, "reversed": rev32, "chunked_16_then_8": chunk32, "nki_sim": s}
    per_order = {k: errors(v, ref64) for k, v in orders.items()}
    stack = np.stack([v.astype(np.float64) for v in orders.values()])
    spread_ulp = (stack.max(0) - stack.min(0)) / ulp32(ref64.astype(np.float32))
    res["bin_sum"] = {
        "bins": BINS, "keys_per_bin": KEYS_PER_BIN, "n": BINS * KEYS_PER_BIN,
        "tiles": BINS // P, "sim_wall_s": t1 - t0,
        "reference": "numpy float64 sequential sum of the same float32 weights",
        "tol_ulp": TOL_ULP, "tol_rel": TOL_REL,
        "orders": per_order,
        "bins_where_orders_differ": int((spread_ulp > 0).sum()),
        "max_ulp_spread_across_orders": float(spread_ulp.max()),
        "nki_sim_matches_sequential_bit_exact": bool(np.array_equal(s, seq32)),
        "nki_sim_matches_chunked_bit_exact": bool(np.array_equal(s, chunk32)),
        "all_orders_pass_chosen_tol": all(o["bins_over_432_ulp"] == 0 and o["bins_over_rel_tol"] == 0 for o in per_order.values()),
        "sum_of_all_bins_nki": float(s.astype(np.float64).sum()),
        "sum_of_all_bins_ref": float(ref64.sum()),
    }
    # 3 a wrong kernel, drops the last key of every bin, must fail the bound
    t0 = time.perf_counter()
    sd = np.asarray(RUN(bin_sum_kernel)(np.ascontiguousarray(w[:, :KEYS_PER_BIN - 1]))).reshape(BINS)
    t1 = time.perf_counter()
    ed = errors(sd, ref64)
    res["wrong_kernel_drop_one_key"] = {"sim_wall_s": t1 - t0, **ed,
                                        "bins_failing_chosen_tol": int(max(ed["bins_over_432_ulp"], ed["bins_over_rel_tol"])),
                                        "note": "a bin whose dropped weight is under about 3e-4 stays inside the bound, so a few bins pass"}
    if a.bins_out:
        np.savez(a.bins_out, kernel=s, reference_float64=ref64, sequential=seq32,
                 reversed_order=rev32, chunked_16_then_8=chunk32, wrong_kernel=sd,
                 weights=w)
        res["bins_out"] = a.bins_out
    print(json.dumps(res, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
