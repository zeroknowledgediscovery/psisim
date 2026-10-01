#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 -m pip install -r applications/psisimulation/requirements.txt

cmake -S . -B build-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLSM_BUILD_PYTHON_BINDINGS=ON

cmake --build build-tests \
  --target predict_distribution \
  -j "$(nproc)"

python3 applications/psisimulation/fetch_model.py gss/gss_2018

python3 applications/psisimulation/simulate_progressive.py \
  --model gss/gss_2018 \
  --choices applications/psisimulation/examples/gss2018_choices.json \
  --sweeps 5 \
  --empirical-n 10 \
  --threads "$(nproc)" \
  --gif
