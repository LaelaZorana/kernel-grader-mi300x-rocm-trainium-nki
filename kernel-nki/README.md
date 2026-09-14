# kernel-nki

An NKI kernel that computes the per bin float32 sums of the histogram task. It has now run both ways, in the Neuron processor simulator on an x86 box and on a NeuronCore of a trn1.2xlarge. The two do not agree on most bins, and the size of the disagreement is measured below.

## The simulator and the card return different numbers

Four seeds, 4096 bins each, same kernel and same input. Both paths record a sha256 of the weight bytes, and it matches on every pair, so the inputs are identical rather than merely drawn from the same seed.

| Seed | Bins identical | Bins differing | Worst gap | Median gap | Bins over 432 ULP |
| - | - | - | - | - | - |
| 3 | 890 | 3206 | 8 ULP | 1 ULP | 0 |
| 7 | 904 | 3192 | 8 ULP | 1 ULP | 0 |
| 11 | 921 | 3175 | 10 ULP | 1 ULP | 0 |
| 19 | 875 | 3221 | 9 ULP | 1 ULP | 0 |

About 78 percent of bins differ, on every seed, and the gap never grows past 10 units in the last place.

## The cause is the accumulation order

The card matches a numpy sequential float32 sum bit for bit, on all four seeds. The simulator matches neither the sequential order nor the chunked 16 then 8 order, and it sits closer to the float64 reference than the card does.

| Path | Worst error against float64 | Median | Bins bit exact |
| - | - | - | - |
| Simulator | 2 ULP | 0 ULP | 2642 of 4096 |
| NeuronCore | 8 ULP | 1 ULP | 891 of 4096 |

So neither one is broken. They add the same 128 numbers in a different order, and float32 addition is not associative, so the results sit a few units apart.

## The bound derived on AMD silicon holds on this card

The 432 ULP bound is four times the worst correct accumulation order measured on an AMD Instinct MI300X, which came in at 108 ULP. Applied here unchanged, **0 of 4096 bins go over it, on either path, on all four seeds**.

The kernel that drops one weight per bin still fails on the card. 4088 of 4096 bins over the bound at seed 3 with a median of 86472 ULP, and 4087 of 4096 at seed 7 with a median of 88032.

So one number, derived from a measurement on one vendor's silicon, separates a correct kernel from a wrong one on a second vendor's hardware.

## Determinism

| Check | Result |
| - | - |
| Same core, same seed, run twice | 4096 of 4096 bins identical |
| Core 0 against core 1 | 4096 of 4096 bins identical |

This card is reproducible, and it does not care which of the two NeuronCores runs the work. Its only divergence is against the host simulator, and that gap is bounded.

## The version check ran before any device number was trusted

A difference between two boxes can come from the hardware or from the toolchain. So the unchanged simulator path was re-run on the trn1 box first and compared against sim_seed3.json checked in here.

The numeric payload matched, and the python, machine, nki and neuronx-cc fields were identical. The one difference was the numpy version, 2.5.3 in the reference against 2.5.2 on the Neuron image, and every number still matched bit for bit, so that patch version is not load bearing for this kernel.

Only after that did the device runs count.

## Two things that cost an hour to work out

Calling the venv python by its full path is not enough to compile for the device. The compiler driver resolves neuronx-cc with a lookup across the search path, so the venv bin directory has to be on that path. Otherwise the driver builds the command with the literal string None in place of the compiler, and the compile fails with exit code 2.

```bash
export PATH="/opt/aws_neuronx_venv_jax_0_10/bin:/opt/aws/neuron/bin:$PATH"
```

A kernel also has to live in a real module file. Defining one inside a python -c string fails with entry function not found, because the tracer looks the function up by module.

## What is here

The file histogram_nki.py holds two kernels under the NKI jit decorator plus the checks. The first is tensor_add_kernel, which is the quickstart kernel. The second is bin_sum_kernel, which takes a 4096 by 128 float32 array in device memory, walks 32 tiles of 128 partitions, and reduces the free axis with nisa.tensor_reduce. A third run feeds the same kernel 127 of the 128 weights, so the tolerance bound has something to fail against.

One source runs in both places, because the device flag swaps the simulate wrapper for the bare jit kernel, and nothing inside either kernel changes.

Inside the device directory sit the raw output of every run, the version check in alignment.txt, the bin by bin comparison and the toolchain versions of the box. Two files, sim_seed3.json and sim_seed7.json, are the simulator references that the version check reproduces.

Two scripts, run_on_device.sh and extra_device_checks.sh, carry the exact unattended sequence that produced these numbers, in order, so the whole session repeats without anyone typing while a card bills.

## How to run

In the simulator, with no device attached.

```bash
python3 -m venv venv && . venv/bin/activate
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install "neuronx-cc==2.*" nki numpy
python3 histogram_nki.py --seed 3 --json sim.json
```

On a trn1 box with the Neuron image, where the venv is already built.

```bash
export PATH="/opt/aws_neuronx_venv_jax_0_10/bin:$PATH"
python3 histogram_nki.py --device --seed 3 --json dev.json --bins-out dev.npz
```

The bins-out flag writes every one of the 4096 values, so two runs can be compared bin by bin later without the card.

Every one of those values is published at dataset/nki_bins.jsonl in the repository root, 16384 rows across the four seeds. Reading that file back reproduces each count in the first table above, and it also shows the card equal to the numpy sequential sum in 4096 of 4096 bins on every seed.
