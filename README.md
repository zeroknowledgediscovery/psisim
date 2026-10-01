# psisim

Standalone Psi-dynamics simulation workbench extracted from
`zeroknowledgediscovery/lsm` branch `dev-static`.

The repository contains:

- `applications/psisimulation/`: the Psi simulation application.
- `bindings/`, `include/`, and `src/`: the minimal vendored LSM runtime
  needed to build the `predict_distribution` and `qdistance` Python
  extensions.
- `CMakeLists.txt`: a standalone build for those extensions.
- `.github/workflows/sync-lsm-bindings.yml`: scheduled/manual synchronization
  of the vendored runtime from `lsm:dev-static`.

No sibling checkout of the LSM repository is required to build or run PsiSim.

## Build

```bash
python3 -m pip install -r applications/psisimulation/requirements.txt

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target predict_distribution qdistance -j "$(nproc)"
```

The extensions are written to `bin/`, which the PsiSim scripts add to
`sys.path` automatically.

## Quick start

```bash
bash applications/psisimulation/run_example.sh
```

## Upstream synchronization

The vendored runtime is synchronized from the private
`zeroknowledgediscovery/lsm` repository, branch `dev-static`.

The workflow requires a repository or organization Actions secret named
`LSM_SYNC_TOKEN` containing a fine-grained GitHub token with **read access**
to the private `zeroknowledgediscovery/lsm` repository. The normal
`GITHUB_TOKEN` is used to commit any synchronized changes back to this
repository.

See `LSM_PROVENANCE.md` for the exact source commit currently vendored.
