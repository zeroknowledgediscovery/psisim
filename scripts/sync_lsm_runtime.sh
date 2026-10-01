#!/usr/bin/env bash
set -euo pipefail

UPSTREAM="${1:-_upstream_lsm}"

if [[ ! -d "${UPSTREAM}/.git" ]]; then
  echo "ERROR: expected an LSM checkout at ${UPSTREAM}" >&2
  exit 2
fi

branch="$(git -C "${UPSTREAM}" branch --show-current || true)"
if [[ "${branch}" != "dev-static" ]]; then
  echo "ERROR: upstream checkout must be on dev-static, found '${branch}'" >&2
  exit 2
fi

sha="$(git -C "${UPSTREAM}" rev-parse HEAD)"

rm -rf include
mkdir -p include bindings src
cp -a "${UPSTREAM}/include/." include/

for d in SourceMaps PredictDistribution Tree ShardMap QDistance; do
  rm -rf "src/${d}"
  cp -a "${UPSTREAM}/src/${d}" "src/${d}"
done

cp "${UPSTREAM}/bindings/predict_distribution_py.cpp" bindings/
cp "${UPSTREAM}/bindings/qdistance_py.cpp" bindings/
cp "${UPSTREAM}/LICENSE" LICENSE

echo "${sha}" > LSM_RUNTIME_SOURCE_COMMIT
echo "Synchronized PsiSim LSM runtime from dev-static @ ${sha}"
