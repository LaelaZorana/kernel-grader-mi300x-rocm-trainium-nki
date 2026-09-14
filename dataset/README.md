---
license: cc-by-4.0
pretty_name: Kernel grader measurements, MI300X and Trainium
tags:
- gpu
- kernels
- rocm
- trainium
- benchmark
- floating-point
size_categories:
- 10K<n<100K
---

# Kernel grader measurements

Three flat files. Every row carries its own source, creator and licence, so a row that is copied into
someone else's file still says where it came from, and so does any subset of the rows.

Created by **Laela Zorana**. Free to use, change and redistribute under CC BY 4.0, which asks only that
the credit stays with it.

## What is in each file

| File | Rows | What one row is |
| - | - | - |
| gyms.jsonl | 7 | one graded environment, with its golden score and both cheat scores |
| checks.jsonl | 51 | one check inside one environment, with its status and its numbers |
| nki_bins.jsonl | 16384 | one histogram bin at one seed, with the simulator value and the card value side by side |

## Where the numbers came from

The kernel numbers were measured on an **AMD Instinct MI300X VF, gfx942, under ROCm 7.2.4**. One global
atomic per key measured 86.3898 ms. The same work privatised into local data share measured 0.1015 ms.
That is a factor of 851.13, on the same card in the same run.

The tolerance bound of **432 units in the last place** is four times a measured worst correct
accumulation order of 108 on that card. It was then carried unchanged onto an **AWS Trainium
NeuronCore of a trn1.2xlarge**, where 0 of 4096 bins went over it at any of four seeds.

nki_bins.jsonl is the file that needed both. It holds 4096 bins at each of the seeds 3, 7, 11 and 19,
with the value the Neuron simulator returned and the value the card returned for the same input, proved
identical by a sha256 of the weight bytes on both paths.

## The finding the per bin file supports

The simulator and the card return different numbers. Between 3175 and 3221 of 4096 bins differ on every
seed, by at most 10 units in the last place. The card reproduces a numpy sequential sum bit for bit and
the simulator matches neither the sequential nor the chunked order. Both stay inside the 432 bound.

## Checksums

Match a file against this publication.

| File | Rows | Bytes | sha256 |
| - | - | - | - |
| checks.jsonl | 51 | 44,988 | `c4d62fffd43eb573b815ff3951e91f9e13dc0afa6141269a9b0bd1cf734590ca` |
| gyms.jsonl | 7 | 3,151 | `a4b3abdb9162e57a2bc9e13f3dfeb751f481d804f001fa88d717d5b0f0b333d2` |
| nki_bins.jsonl | 16,384 | 9,796,822 | `8828495294d0c7e270520153590393a23f5120333c6045d74b7071ef8b9727f0` |

## Citation

```bibtex
@misc{zorana2026kernelgrader,
  author       = {Laela Zorana},
  title        = {Kernel grader measurements, MI300X and Trainium},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/LaelaZorana/kernel-grader-mi300x-rocm-trainium-nki}}
}
```

## Licence

CC BY 4.0. Keep the credit with it, and say if you changed something.
