#!/usr/bin/env bash
# One-command start of the PsiSim webapp.
#
#   bash scripts/run_webapp.sh                 # http://127.0.0.1:8000/
#   bash scripts/run_webapp.sh --port 9000
#   bash scripts/run_webapp.sh --host 0.0.0.0  # reachable from other machines
#
# Installs the Python requirements, builds the native modules when they are
# missing or older than their sources, downloads the default GSS 2018 model
# once (other surveys are downloaded on demand from the app), and starts the
# server. Extra arguments are passed to webapp/server.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

echo "==> Python requirements"
"$PY" -m pip install --disable-pip-version-check -q \
  -r applications/psisimulation/requirements.txt -r webapp/requirements.txt

# A module is stale when it is missing or older than one of its own sources.
needs_build=0
for mod in predict_distribution psisim_dynamics; do
  so=$(ls bin/${mod}*.so 2>/dev/null | head -1 || true)
  if [[ -z "$so" ]] || [[ -n "$(find "bindings/${mod}_py.cpp" src include -type f -newer "$so" -print -quit)" ]]; then
    needs_build=1
  fi
done
if [[ $needs_build == 1 ]]; then
  # A compiler or system upgrade leaves the CMake cache pointing at files that
  # no longer exist (e.g. the old GCC's libgomp.so); start that build afresh.
  if [[ -f build/CMakeCache.txt ]]; then
    while IFS= read -r path; do
      if [[ ! -e "$path" ]]; then
        echo "==> CMake cache refers to a missing file ($path); reconfiguring from scratch"
        rm -rf build
        break
      fi
    done < <(sed -n 's/^[^#/]*:FILEPATH=\(\/.*\)$/\1/p' build/CMakeCache.txt)
  fi
  echo "==> Building native modules (first time takes a few minutes)"
  generator=()
  command -v ninja >/dev/null && generator=(-G Ninja)
  cmake -S . -B build "${generator[@]}" -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target predict_distribution psisim_dynamics -j "$(nproc 2>/dev/null || echo 4)"
else
  echo "==> Native modules up to date"
fi

echo "==> Default model (GSS 2018)"
"$PY" applications/psisimulation/fetch_model.py "${PSISIM_MODEL:-gss/gss_2018}"

echo "==> Starting the webapp"
exec "$PY" webapp/server.py "$@"
