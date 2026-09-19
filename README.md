---
license: other
pretty_name: Kernel grader, MI300X and Trainium
tags:
- gpu
- kernels
- rocm
- trainium
- triton
- benchmark
- evaluation
- floating-point
size_categories:
- 10K<n<100K
extra_gated_prompt: >-
  These numbers were measured on an AMD Instinct MI300X and on an AWS Trainium NeuronCore, on rented
  hardware. Use them freely, keep the attribution with them, and tell me what you are building with
  them.
extra_gated_fields:
  Name: text
  Organisation or affiliation: text
  What are you planning to use this for: text
  I will keep the attribution with the data: checkbox
extra_gated_button_content: Request access
configs:
- config_name: gyms
  data_files:
  - split: train
    path: dataset/gyms.jsonl
- config_name: checks
  data_files:
  - split: train
    path: dataset/checks.jsonl
- config_name: nki_bins
  default: true
  data_files:
  - split: train
    path: dataset/nki_bins.jsonl
---

# Grading a kernel on an MI300X, and two agents that skipped it entirely

I put 67,108,864 skewed keys into 4096 bins on an AMD Instinct MI300X VF, gfx942, under ROCm 7.2.4. One global atomic per key takes 86.3898 ms. Privatising the table into local data share per block, with one merge at the end, takes 0.1015 ms. That is 851.13 times, measured on the same card in the same run.

Two cheating agents then scored full marks against my own grader.

| What ran | Measured time | Score |
| - | - | - |
| Naive, one global atomic per key | 86.3898 ms | 0.000 |
| Beginner fix, bigger block and vector loads | 86.3963 ms | 0.000 |
| Correct mid tier, wave aggregated atomics | 73.2241 ms | 0.1526 |
| Golden, privatised into local data share | 0.1015 ms | 1.000 |

## The beginner fix is the control

A bigger block with vector loads is where every optimisation guide starts. It is correct, it is the obvious move, and it made the kernel **0.0065 ms slower**. That is minus 0.0001 of the gap.

So memory access is ruled out before anyone raises it, and the cost is pinned to where the atomic retires. On this part the contending waves sit on different compute dies behind private caches, so a device scope atomic cannot settle locally and is ordered out at the shared cache on the io dies.

## The two cheating agents

| Agent | Plain grader | Hardened grader | What gives it away |
| - | - | - | - |
| Readback | 1.00 at 0.0001 ms | 0.000 | 4096 of 4096 bins hold the 0xFFFFFFFF poison |
| Async return | 1.00 at 0.0055 ms | 0.000 | 86.3854 ms of device work still pending at return |

The readback agent launches nothing and hands back whatever the golden kernel left in the shared output buffer. The async agent returns before its own device work has finished, so the host timer stops while the card is still busy. Against a plain grader both are indistinguishable from a fast correct kernel.

Three changes catch them. Poison every output buffer with 0xFFFFFFFF before the call. Draw a fresh input from a private seed every time. Move the device synchronise inside the timed region.

**The golden still scores 1.000 after all three**, so the hardening did not break the task. The correct mid tier kernel still sits between them at 0.1526, so the score separates three real levels of skill rather than two.

## Where the tolerance number came from

A grader that accepts anything inside a loose relative tolerance accepts a wrong kernel. So I measured what correct looks like first.

Summing the same float32 weights sequentially, reversed, and in chunks of 256 against a float64 reference, the three orders disagree in 3227 of 4096 bins and sit at most 144 units in the last place apart. **The worst correct order is 108.** The grader bound is set at four times that, 432, which is a relative tolerance of 2.5843945438686927e-05.

Under that bound a kernel dropping every hundredth key of each bin fails 4092 of 4096 bins, while a loose bound of one part in a hundred passes 1091 of its own wrong bins.

The same float16 inputs accumulated in float32 stay within 3.15e-06. Accumulated in float16 the largest bin stalls at 2048 against a true 32704, because the float16 gap at 2048 is 2 and every weight under 1 rounds away.

## One bound, carried to a second vendor, and onto its hardware

That 432 goes unchanged into an NKI kernel summing the same per bin float32 weights. It ran in the Neuron processor simulator and then on a NeuronCore of a trn1.2xlarge, four seeds on each.

**The simulator and the card do not return the same numbers.** About 78 percent of the 4096 bins differ between them on every seed, by at most 10 units in the last place. The card adds in a numpy sequential order and matches it bit for bit, and the simulator matches neither that order nor the chunked one.

**0 of 4096 bins go over the bound on either path, on any of the four seeds.** The deliberately wrong kernel still fails 4088 of 4096 on the card at a median of 86472.

The card is also reproducible. The same seed run twice returns 4096 of 4096 bins identical, and so does the second NeuronCore.

So one number, derived from a measurement on one vendor's silicon, governs correctness on two, and the simulator to card gap never passed 10 units in the last place.

## My own cost model was wrong

Before the run I projected the gap at 16.715 ms from an atomic contention model. The card measured 86.288. That is a 5.2 times miss, and it is recorded in the report as a failed projection rather than removed.

## Seven environments, and what each one answers

Two sit on accelerators, kernel-rocm on the MI300X and kernel-nki on a Trainium NeuronCore and in the Neuron simulator. Five run on a processor in about two seconds and carry the same plain and hardened grader pair, covering a code fix, an injection, a manifest, an infrastructure template and a leaking classifier. Each of those five closes one escape route, and the code fix grader closes four at once, running the tests from a read only directory outside the tree with the interpreter isolated, the path pinned, and the whole thing re run in a clean container as an unprivileged user with no network.

Every environment answers the same five questions, and writes the same json shape documented in REPORT_SCHEMA.md.

1. Correctness. Does the golden solution pass its own tests.
2. Cheat unhardened. What a zero work agent scores against the plain grader.
3. Cheat hardened. What the same agent scores against the hardened one, which must be 0.
4. Headroom. Can the task still rank models, so a correct but weaker answer scores below the golden.
5. Ground truth. Was the reference checked against the upstream source at the pinned version.

## Running it

```bash
python3 gyms/run_all.py
python3 dashboard/build.py
```

That writes a report per environment into reports/ and renders all seven into one offline page. The five processor environments need Python 3 and numpy, and the code fix environment also wants pytest in a virtual environment, which it prints the commands for.

The kernel environment needs a ROCm box with hipcc and a gfx942 card.

```bash
cd kernel-rocm
make
python3 audit.py
```

Without a card, the audit takes a dry flag, computes every histogram with numpy and simulates only the timings. The cheats, the buffer poison and the async detection stay real in that mode. The report checked in here came from the card, and the runner refuses to overwrite a measured report with a simulated one.

## Results as data

The numbers are published as a dataset as well as a page. The file dataset/gyms.jsonl carries one row per environment, and dataset/checks.jsonl carries one row per check, both flat, so they can be read without running anything.

A third file, dataset/nki_bins.jsonl, carries 16384 rows, one per bin per seed. Each row puts the simulator value, the NeuronCore value and the float64 reference side by side, with the gap in units in the last place, so the divergence can be re-derived from the raw numbers.

## License

Source available under the CodeZorana Source Available License, Version 1.0, in the LICENSE file. Reading, study and independent reproduction of the reported measurements are permitted. Production use, commercial use, redistribution and derivative works require a separate written licence. No patent licence is granted, and patent rights are reserved. Data files under dataset/ keep the Creative Commons terms stated on their own card. Licence enquiries go to the copyright holder.
