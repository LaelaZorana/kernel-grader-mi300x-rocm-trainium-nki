#!/usr/bin/env bash
# build, run unhardened then hardened, write ../reports/kernel-rocm.json and print the card.
set -euo pipefail
cd "$(dirname "$0")"
make -j
python3 audit.py "$@"
