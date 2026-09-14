#!/usr/bin/env bash
# Three facts a grader needs that only the card can answer.
#   1. is the divergence stable across more seeds, or was it two lucky draws
#   2. is the device run to run deterministic on the same core
#   3. does the answer change when the kernel runs on the other NeuronCore
set -uo pipefail
OUT=/home/ubuntu/out
cd /home/ubuntu/kernel-nki || exit 1
VENV=/opt/aws_neuronx_venv_jax_0_10
PY="$VENV/bin/python"
export PATH="$VENV/bin:/opt/aws/neuron/bin:$PATH"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }

log "=== 8. two more seeds, both paths ==="
for S in 11 19; do
  "$PY" histogram_nki.py --seed "$S" --json "$OUT/box_sim_seed$S.json" \
        --bins-out "$OUT/box_sim_seed${S}_bins.npz" > "$OUT/box_sim_seed$S.log" 2>&1
  log "    sim seed $S exit $?"
  timeout 900 "$PY" histogram_nki.py --device --seed "$S" --json "$OUT/device_seed$S.json" \
        --bins-out "$OUT/device_seed${S}_bins.npz" > "$OUT/device_seed$S.log" 2>&1
  log "    device seed $S exit $?"
done

log "=== 9. run to run determinism, seed 3 again on the same core ==="
timeout 900 "$PY" histogram_nki.py --device --seed 3 --json "$OUT/device_seed3_repeat.json" \
      --bins-out "$OUT/device_seed3_repeat_bins.npz" > "$OUT/device_seed3_repeat.log" 2>&1
log "    exit $?"

log "=== 10. the other NeuronCore, seed 3 ==="
NEURON_RT_VISIBLE_CORES=1 timeout 900 "$PY" histogram_nki.py --device --seed 3 \
      --json "$OUT/device_seed3_core1.json" --bins-out "$OUT/device_seed3_core1_bins.npz" \
      > "$OUT/device_seed3_core1.log" 2>&1
log "    exit $?"

log "=== 11. rolling it up ==="
"$PY" - <<'PY' 2>&1 | tee "$OUT/extra.txt"
import json, os
import numpy as np
OUT = "/home/ubuntu/out"
TOL = 432
def ulp32(x): return np.spacing(np.abs(np.asarray(x, np.float32))).astype(np.float64)
def bins(p):
    d = np.load(os.path.join(OUT, p)); return d["kernel"].astype(np.float64).ravel(), d["reference_float64"].astype(np.float64).ravel()

out = {"seeds": [], "determinism": {}}
for s in (3, 7, 11, 19):
    try:
        a, ref = bins("box_sim_seed%d_bins.npz" % s)
        b, _ = bins("device_seed%d_bins.npz" % s)
        ref32 = ref.astype(np.float32).astype(np.float64)
        gap = np.abs(a - b) / ulp32(ref32)
        out["seeds"].append({
            "seed": s,
            "bins": int(a.size),
            "identical_sim_vs_device": int((a == b).sum()),
            "differing_sim_vs_device": int((a != b).sum()),
            "max_ulp_gap": float(gap.max()),
            "median_ulp_gap": float(np.median(gap)),
            "bins_over_432_ulp": int((gap > TOL).sum()),
            "device_bit_exact_vs_reference": int((b == ref32).sum()),
            "sim_bit_exact_vs_reference": int((a == ref32).sum()),
        })
    except Exception as e:
        out["seeds"].append({"seed": s, "error": repr(e)})

try:
    base, _ = bins("device_seed3_bins.npz")
    rep, _ = bins("device_seed3_repeat_bins.npz")
    out["determinism"]["same_core_run_to_run_identical_bins"] = int((base == rep).sum())
    out["determinism"]["same_core_run_to_run_differing_bins"] = int((base != rep).sum())
except Exception as e:
    out["determinism"]["repeat_error"] = repr(e)
try:
    base, _ = bins("device_seed3_bins.npz")
    c1, _ = bins("device_seed3_core1_bins.npz")
    out["determinism"]["core0_vs_core1_identical_bins"] = int((base == c1).sum())
    out["determinism"]["core0_vs_core1_differing_bins"] = int((base != c1).sum())
except Exception as e:
    out["determinism"]["core_error"] = repr(e)

print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(OUT, "extra.json"), "w"), indent=2)
PY
log "=== 12. extra checks done ==="
