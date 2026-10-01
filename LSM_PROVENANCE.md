# LSM provenance

PsiSim was initialized from the private repository
`zeroknowledgediscovery/lsm`, branch `dev-static`.

- Psi application snapshot: `73afb9eccc19184324f7078e5864c1844044088d`
- Initial vendored LSM runtime snapshot: `73afb9eccc19184324f7078e5864c1844044088d`

The application is under `applications/psisimulation/`.

The vendored runtime intentionally contains only the LSM surface required by
PsiSim:

- `bindings/predict_distribution_py.cpp`
- `bindings/qdistance_py.cpp`
- `include/**`
- `src/SourceMaps/**`
- `src/PredictDistribution/**`
- `src/Tree/**`
- `src/ShardMap/**`
- `src/QDistance/**`

`PSISIM_SOURCE_COMMIT` records the pinned source commit for the application.
`LSM_RUNTIME_SOURCE_COMMIT` records the latest upstream commit whose runtime
files were synchronized.

The automated sync updates only the vendored runtime, not the Psi application,
so PsiSim-specific changes are never overwritten by a binding refresh.
