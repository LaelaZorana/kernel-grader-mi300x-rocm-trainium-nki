#!/usr/bin/env bash
# One shot device session. Everything runs unattended so the billed clock
# carries the experiment rather than a person typing.
#
# Order matters. The version alignment step runs first and reproduces the
# committed simulator reference on this box. If the numbers move there, a
# later device difference would be measuring a toolchain change instead of
# the hardware, so the alignment result is recorded before anything else.
set -uo pipefail

OUT=/home/ubuntu/out
mkdir -p "$OUT"
cd /home/ubuntu/kernel-nki || exit 1

VENV=/opt/aws_neuronx_venv_jax_0_10
PY="$VENV/bin/python"
# the venv bin has to be on PATH. the compiler driver resolves neuronx-cc
# with a path lookup, and calling the venv python by full path is not enough.
export PATH="$VENV/bin:/opt/aws/neuron/bin:$PATH"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }

log "=== 0. what this box is ==="
{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "instance: $(curl -s -m 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)"
  echo "type:     $(curl -s -m 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null)"
  echo "zone:     $(curl -s -m 2 http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null)"
  echo "kernel:   $(uname -srm)"
} | tee "$OUT/box.txt"

log "=== 1. neuron devices present ==="
neuron-ls 2>&1 | tee "$OUT/neuron-ls.txt" | head -40

log "=== 2. toolchain on this box ==="
"$PY" - <<'PY' 2>&1 | tee "$OUT/versions.json"
import json, sys
o = {"python": sys.version.split()[0]}
for m in ("numpy", "nki", "neuronxcc"):
    try:
        mod = __import__(m); o[m] = getattr(mod, "__version__", "present")
    except Exception as e:
        o[m] = "import failed %r" % e
print(json.dumps(o, indent=2))
PY

log "=== 3. VERSION ALIGNMENT, reproduce the committed reference ==="
log "    default path is unchanged code, so the numbers must match sim_seed3.json"
"$PY" histogram_nki.py --seed 3 --json "$OUT/box_sim_seed3.json" \
      --bins-out "$OUT/box_sim_seed3_bins.npz" > "$OUT/box_sim_seed3.log" 2>&1
log "    exit $?"
"$PY" - <<'PY' 2>&1 | tee "$OUT/alignment.txt"
import json
SKIP = {"execution", "bins_out", "input_sha256"}
def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if "wall" not in k and k not in SKIP}
    if isinstance(o, list):
        return [strip(x) for x in o]
    return o
VER = ("python", "machine", "numpy", "nki", "neuronx_cc")
try:
    ref = json.load(open("sim_seed3.json"))
    box = json.load(open("/home/ubuntu/out/box_sim_seed3.json"))
    print("versions, reference against this box")
    for k in VER:
        same = "same" if ref.get(k) == box.get(k) else "DIFFERS"
        print("  %-12s %-40s %-40s %s" % (k, ref.get(k), box.get(k), same))
    a = {k: strip(v) for k, v in ref.items() if k not in VER and k not in SKIP}
    b = {k: strip(v) for k, v in box.items() if k not in VER and k not in SKIP}
    print("\nnumeric payload:", "MATCH" if a == b else "DIFFER")
    if a != b:
        for k in a:
            if a[k] != b.get(k):
                print("  key differs:", k)
                print("    reference:", json.dumps(a[k])[:400])
                print("    this box :", json.dumps(b.get(k))[:400])
except Exception as e:
    print("ERROR %r" % e)
PY

log "=== 4. THE HEADLINE. same kernel, same seeds, on the device ==="
for S in 3 7; do
  log "    device seed $S"
  timeout 1200 "$PY" histogram_nki.py --device --seed "$S" \
      --json "$OUT/device_seed$S.json" --bins-out "$OUT/device_seed${S}_bins.npz" \
      > "$OUT/device_seed$S.log" 2>&1
  log "    seed $S exit $?"
  tail -5 "$OUT/device_seed$S.log" | sed 's/^/        /'
done

log "=== 5. simulator seed 7 for the second comparison pair ==="
"$PY" histogram_nki.py --seed 7 --json "$OUT/box_sim_seed7.json" \
      --bins-out "$OUT/box_sim_seed7_bins.npz" > "$OUT/box_sim_seed7.log" 2>&1
log "    exit $?"

log "=== 6. bin by bin, all 4096, simulator against device ==="
"$PY" - <<'PY' 2>&1 | tee /home/ubuntu/out/comparison.txt
import json, os
import numpy as np
OUT = "/home/ubuntu/out"
TOL_ULP = 432

def ulp32(x):
    return np.spacing(np.abs(np.asarray(x, np.float32))).astype(np.float64)

def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"load_error": repr(e)}

rows = []
for s in (3, 7):
    row = {"seed": s}
    simj = load(os.path.join(OUT, "box_sim_seed%d.json" % s))
    devj = load(os.path.join(OUT, "device_seed%d.json" % s))
    row["sim_exec"] = simj.get("execution")
    row["dev_exec"] = devj.get("execution")
    row["same_input_sha256"] = (simj.get("input_sha256") == devj.get("input_sha256")
                                and simj.get("input_sha256") is not None)
    row["input_sha256"] = simj.get("input_sha256")
    try:
        sim = np.load(os.path.join(OUT, "box_sim_seed%d_bins.npz" % s))
        dev = np.load(os.path.join(OUT, "device_seed%d_bins.npz" % s))
        a = sim["kernel"].astype(np.float64).ravel()
        b = dev["kernel"].astype(np.float64).ravel()
        ref = sim["reference_float64"].astype(np.float64).ravel()
        ref32 = ref.astype(np.float32).astype(np.float64)
        row["bins_compared"] = int(a.size)
        row["bins_bit_identical_sim_vs_device"] = int((a == b).sum())
        row["bins_differing_sim_vs_device"] = int((a != b).sum())
        d_ulp = np.abs(a - b) / ulp32(ref32)
        row["max_ulp_gap_sim_vs_device"] = float(d_ulp.max())
        row["median_ulp_gap_sim_vs_device"] = float(np.median(d_ulp))
        row["bins_over_432_ulp_sim_vs_device"] = int((d_ulp > TOL_ULP).sum())
        for name, v in (("sim", a), ("device", b)):
            u = np.abs(v - ref32) / ulp32(ref32)
            row[name + "_vs_float64_reference"] = {
                "ulp_max": float(u.max()),
                "ulp_median": float(np.median(u)),
                "bins_over_432_ulp": int((u > TOL_ULP).sum()),
                "bins_bit_exact": int((v == ref32).sum()),
            }
        row["weights_identical"] = bool(np.array_equal(sim["weights"], dev["weights"]))
        wrong_a = sim["wrong_kernel"].astype(np.float64).ravel()
        wrong_b = dev["wrong_kernel"].astype(np.float64).ravel()
        row["wrong_kernel_bins_bit_identical"] = int((wrong_a == wrong_b).sum())
        uw = np.abs(wrong_b - ref32) / ulp32(ref32)
        row["wrong_kernel_on_device_bins_over_432_ulp"] = int((uw > TOL_ULP).sum())
        row["wrong_kernel_on_device_ulp_median"] = float(np.median(uw))
    except Exception as e:
        row["bins_error"] = repr(e)
    rows.append(row)

print(json.dumps(rows, indent=2))
json.dump(rows, open(os.path.join(OUT, "comparison.json"), "w"), indent=2)
PY

log "=== 7. done, results in $OUT ==="
ls -la "$OUT" | tee -a "$OUT/session.log"
