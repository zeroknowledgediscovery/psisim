> **Canonical mathematical specification.** The repository-root `README.md`
> contains the complete current mathematical and conceptual definition of the
> native LSM model, hard and soft inference semantics, centered response kernel,
> finite-empirical-event dynamics, simplex projection, clamping, sequential
> sweeps, convergence diagnostics, qdistance geometry, clustering, and
> interpretation boundaries.  The fully worked two-question GSS-2018 example is
> in `applications/psisimulation/examples/README.md`.  This application-level
> README is retained as the operational/manual reference.

# Psi simulation: intrinsic LSM dynamics

`applications/psisimulation` is a self-contained experimental workbench for
probability-valued dynamics induced by a native Large Science Model (LSM).

The simulation does **not** require an empirical data row. It can begin from the
all-missing sample, construct the canonical empty-sample probability state
`Psi0`, observe selected variables one at a time, propagate each observation
through the learned hard conditional predictors, and relax the resulting state
toward a numerical equilibrium.

The current implementation is built on the centered hard-response dynamics in
the vendored `predict_distribution` C++/pybind binding synchronized from LSM
branch `dev-static`.

## 1. State and native LSM conditionals

Let the model have coordinates `i = 1,...,d`, with categorical alphabet
`Sigma_i`. The learned native LSM contains one predictor per modeled
coordinate,

```text
phi_i(x_-i) -> distribution on Sigma_i.
```

A dynamical state is a probability-valued row

```text
Psi = (p_1, p_2, ..., p_d),
```

where `p_i` is a categorical distribution on `Sigma_i`.

For the all-missing hard row `empty`, define

```text
p_i^0 = phi_i(empty)
Psi0  = (p_1^0, ..., p_d^0).
```

`Psi0` is therefore obtained directly from the learned LSM; no survey data row
is needed.

## 2. Why the dynamics is centered

For a source coordinate `i`, a target coordinate `j != i`, and a hard
symbol `sigma in Sigma_i`, define the one-coordinate hard-response kernel

```text
K_{j<-i}(. | sigma) = phi_j(x_{i=sigma}),
```

where all coordinates other than `i` are missing.

A tempting identity is

```text
sum_sigma p_i(sigma) K_{j<-i}(. | sigma) = p_j.
```

This would be the ordinary tower identity if all of these quantities were
exact marginals/conditionals of the same explicitly represented joint law.

We tested this identity directly on the GSS-2018 native model. It is not exact
for the implemented learned trees. Missing-variable prediction inside a tree
uses path-local subtree masses, whereas `phi_i(empty)` is produced by a
different learned predictor. The dynamics therefore does **not** assume the
tower identity.

Instead it propagates only the response to a change in the source marginal.

## 3. Centered empirical-event update

Suppose the current source marginal is `p_i` and an empirical realization is
`nu_i`. Define

```text
delta_i = nu_i - p_i.
```

The source coordinate becomes

```text
p_i' = nu_i.
```

Every dependent target is updated by

```text
p_j' = p_j
       + sum_sigma delta_i(sigma) K_{j<-i}(. | sigma).
```

Equivalently,

```text
Delta p_j = K_{j<-i} (nu_i - p_i).
```

If the raw response leaves the categorical probability simplex, the
implementation projects the result back onto the simplex and reports the
projection count.

The crucial property is immediate:

```text
nu_i = p_i  =>  delta_i = 0  =>  Psi' = Psi.
```

Thus a zero-action event is exactly the identity without requiring the tower
identity.

## 4. Large-deviation interpretation

For finite empirical size `n`, an empirical type `nu_i` generated from
`p_i` has the usual multinomial/large-deviation interpretation. Its Sanov
rate is

```text
D(nu_i || p_i),
```

and the implementation reports

```text
nD = n * D(nu_i || p_i).
```

Three event modes are available in the resident C++ state:

- `zero_action`: set `nu_i = p_i`; there is exactly no motion.
- `mode`: use the most probable empirical type on the finite-`n` grid.
- `sample`: sample an empirical type from the multinomial law.

A full sweep visits the learned coordinates sequentially and applies one source
update at a time to the current state.

For `event="mode"`, repeated sweeps can approach a deterministic numerical
equilibrium. For `event="sample"`, finite-`n` stochastic fluctuations
persist and equilibrium should instead be understood as a stationary
distribution/neighborhood.

## 5. Empty state and fixed points

There are two distinct statements that should not be conflated.

### Zero-action fixed point

For every state, and therefore in particular for `Psi0`,

```text
T_zero(Psi) = Psi.
```

This is exact by construction of the centered response.

### Finite-n mode relaxation

At finite `n`, `mode` generally replaces a marginal by the nearest/highest
probability empirical type. If `nu_i != p_i`, that deviation drives a
response. Therefore `Psi0` need not be a fixed point of a finite-`n` mode
sweep.

This distinction is intentional: the empirical large-deviation realization is
the event that drives the dynamics.

## 6. Hard observations: asking a question

A survey answer is represented as a hard observation

```text
X_i = sigma.
```

The implementation sets

```text
nu_i = delta_sigma
```

and immediately applies the same centered response

```text
Delta p_j =
    K_{j<-i} (delta_sigma - p_i).
```

The Python API is

```python
state.hard_observe(
    source=i,
    value=sigma,
    response_scale=1.0,
    threads=32,
    clamp=True,
)
```

With `clamp=True`, the observed coordinate remains fixed at
`delta_sigma` during all later relaxation sweeps. It is persistent evidence,
not a temporary perturbation.

This produces the progressive-choice sequence

```text
Psi0
  -> observe X_i = sigma
  -> immediate global response
  -> relax toward a new equilibrium
  -> observe X_k = tau
  -> immediate global response
  -> relax again
  -> ...
```

This is the principal simulation implemented by
`simulate_progressive.py`.

## 7. Numerical equilibrium diagnostics

A sweep returns:

- `mean_tv`: mean coordinate-wise total variation between the state before and
  after the sweep;
- `max_tv`: maximum coordinate-wise total variation;
- `max_col`: coordinate attaining `max_tv`;
- `event_tv`: mean size of the empirical source events in the sweep;
- `sanov_exponent`: accumulated finite-`n` `nD`;
- `projected_count`: number of response updates requiring simplex projection.

For a deterministic mode simulation, a numerical fixed point is indicated by

```text
mean_tv -> 0
max_tv  -> 0.
```

During development on GSS-2018, actual-row trajectories showed a clean
contractive tail, with many distinct respondent-specific near-equilibria rather
than collapse to one universal state.

## 8. Visualization

Different LSM variables have different categorical alphabet sizes, so a literal
category-by-variable matrix is not rectangular. The default visualization uses
a ranked marginal heatmap:

- each heatmap row is one LSM variable;
- column 1 is its largest categorical probability;
- column 2 is its second largest probability;
- etc.

Each frame also shows:

- MAP probability per coordinate;
- total-variation movement from the previous frame;
- horizontal/vertical markers for coordinates already hard-observed.

The progressive simulation writes one frame for:

1. `Psi0`;
2. the immediate response to each hard answer;
3. every relaxation sweep after that answer.

Optionally it assembles the frames into an animated GIF.

## 9. Public model retrieval

No model artifacts are committed to this repository.

The downloader follows the same public DTAG model release used by the DTAG web
application:

```text
bucket:
  gs://git-zeroknowledgediscovery-dtag/models/v0.2.0/

manifest:
  https://storage.googleapis.com/
  git-zeroknowledgediscovery-dtag/models/v0.2.0/manifest.json
```

`fetch_model.py` downloads the archive over anonymous HTTPS, verifies the
manifest SHA256, performs path-safe `.tar.zst` extraction, and installs into

```text
~/.cache/dtag/models/<family>/<model>
```

by default.

List currently published GSS models:

```bash
python3 applications/psisimulation/fetch_model.py \
  --list --family gss
```

Fetch GSS-2018:

```bash
python3 applications/psisimulation/fetch_model.py \
  gss/gss_2018
```

The same scripts accept another public GSS model key, for example
`gss/gss_2022` or `gss/gss_2024`, if that key is present in the release
manifest.

## 10. Build

From the root of the standalone `psisim` repository:

```bash
python3 -m pip install -r applications/psisimulation/requirements.txt

cmake -S . -B build-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLSM_BUILD_PYTHON_BINDINGS=ON

cmake --build build-tests \
  --target predict_distribution \
  -j "$(nproc)"
```

The scripts automatically add the repository `bin/` directory to the Python
module search path.

## 11. Quick start: complete GSS-2018 progressive simulation

The shortest complete example is:

```bash
bash applications/psisimulation/run_example.sh
```

This:

1. installs Python plotting/download dependencies;
2. builds `predict_distribution`;
3. downloads `gss/gss_2018` if necessary;
4. constructs `Psi0` from the empty row;
5. observes column 0 = `"working fulltime"`;
6. relaxes for five sweeps;
7. observes column 3 = `"yes"`;
8. relaxes for five more sweeps;
9. writes PNG state frames, history, the terminal hard realization, and a GIF.

Before a longer run, the core invariants can be checked on the downloaded
model:

```bash
python3 applications/psisimulation/smoke_test.py \
  --model gss/gss_2018 \
  --column 0 \
  --value "working fulltime" \
  --threads "$(nproc)"
```

The smoke test verifies that a zero-action sweep is numerically identical to
the input state, a hard observation collapses its source coordinate, and the
observed coordinate remains clamped during a subsequent relaxation sweep.

The underlying choice file is

```text
applications/psisimulation/examples/gss2018_choices.json
```

with schema

```json
[
  {"column": 0, "value": "working fulltime", "sweeps": 5},
  {"column": 3, "value": "yes", "sweeps": 5}
]
```

## 12. Choose different questions/answers

Inspect the empty-state marginal for a coordinate:

```bash
python3 applications/psisimulation/inspect_column.py \
  --model gss/gss_2018 \
  --column 0
```

This prints the legal raw labels and their `Psi0` probabilities.

Create a new JSON choice file and run:

```bash
python3 applications/psisimulation/simulate_progressive.py \
  --model gss/gss_2018 \
  --choices my_choices.json \
  --sweeps 5 \
  --empirical-n 10 \
  --threads "$(nproc)" \
  --gif
```

Per-choice `"sweeps"` overrides the global `--sweeps` value.

## 13. Empty-row experiment without questions

To start from `Psi0` and run ten finite-`n` mode sweeps:

```bash
python3 applications/psisimulation/simulate_empty.py \
  --model gss/gss_2018 \
  --sweeps 10 \
  --empirical-n 10 \
  --event mode \
  --threads "$(nproc)"
```

To verify exact zero-action invariance instead:

```bash
python3 applications/psisimulation/simulate_empty.py \
  --model gss/gss_2018 \
  --sweeps 1 \
  --event zero_action \
  --threads "$(nproc)"
```

The expected residual is machine zero.

## 14. Outputs

Default progressive outputs:

```text
results/psisimulation/progressive/
  frames/
    000_empty.png
    001_q1_hard.png
    002_q1_sweep1.png
    ...
  history.json
  final_hard_row.csv
  psi_dynamics.gif       # when --gif is requested
```

`history.json` stores the dynamical diagnostics for every hard observation and
every relaxation sweep.

`final_hard_row.csv` is the coordinatewise MAP realization of the terminal
probability state,

```text
x_i^* = argmax_sigma p_i^*(sigma).
```

It is a convenient discrete representation of a near-equilibrium state; the
underlying dynamical state remains the full probability-valued `Psi`.

## 15. Files

```text
applications/psisimulation/
  README.md
  common.py
  fetch_model.py
  inspect_column.py
  plotting.py
  simulate_empty.py
  simulate_progressive.py
  smoke_test.py
  run_example.sh
  requirements.txt
  examples/
    README.md
    gss2018_choices.json
```

## 16. Core C++/Python API used

The application exercises these `predict_distribution` interfaces:

```python
# canonical empty-sample conditionals
psi0 = predict_distribution.row_to_psi(...)

# resident probability-valued centered-LDP state
state = predict_distribution.centered_ldp_state_from_psi(...)

# one persistent survey answer / hard collapse
state.hard_observe(source=i, value=sigma, clamp=True, ...)

# one complete finite-n relaxation sweep
state.sweep(n=10, event="mode", ...)

# full probability-valued state
state.to_python()

# coordinatewise MAP realization
state.hard_row()
```

The resident state deliberately keeps the same loaded native model and warmed
hard-response cache across successive questions and sweeps.

## 17. Reproducibility conventions

For deterministic progressive experiments:

- use `event="mode"`;
- use `random_permutation=False`;
- record `empirical_n`;
- record the model release/key;
- keep the question order and hard answer labels in a JSON choice file;
- record the number of relaxation sweeps after each observation.

For stochastic large-deviation experiments, use the lower-level resident API
with `event="sample"` and record the random seed.

## 18. Interpretation boundaries

The centered update is a dynamical construction based on the native LSM hard
response to a marginal deviation. It is intentionally distinct from asserting
that the learned empty marginals and every one-coordinate conditional satisfy
an exact tower identity.

Likewise, convergence of `mean_tv` and `max_tv` establishes a numerical
equilibrium of this update rule. Calling such an equilibrium an attractor
requires an additional basin-of-attraction experiment: perturb nearby initial
states and verify that they return to the same equilibrium.


## 19. Equilibrium clustering across observed respondents

`cluster_equilibria.py` tests whether many observed respondents relax into a
smaller number of dynamical equilibrium families.

The workflow is:

```text
sample observed GSS rows
  -> centered-LDP evolution for N sweeps
  -> coordinatewise MAP hard equilibria
  -> save equilibria to CSV
  -> reload equilibria from CSV
  -> full native qdistance matrix
  -> qdistance-space k-medoids clustering
  -> classical MDS for 2-D visualization only
  -> one actual medoid equilibrium per major cluster
```

Clustering is performed directly on the native qdistance matrix. The 2-D MDS
coordinates are used only for plotting, so cluster assignments are not an
artifact of the 2-D projection.

By default the script samples 300 GSS rows and evolves each for 10 sweeps:

```bash
export DATA=/path/to/gss_2018.csv

python3 applications/psisimulation/cluster_equilibria.py \
  --model gss/gss_2018 \
  --data "$DATA" \
  --sample-size 300 \
  --sweeps 10 \
  --empirical-n 10 \
  --threads "$(nproc)"
```

Before running, build both native Python extensions used by the workflow:

```bash
cmake --build build-tests \
  --target predict_distribution qdistance \
  -j "$(nproc)"
```

The number of clusters is selected by mean silhouette over qdistance-space
k-medoids solutions for `k=2,...,8` by default. A major cluster must contain
at least 5% of the sampled equilibria and at least 10 equilibria; both
thresholds are configurable.

A cluster representative is its **medoid**:

```text
argmin_{x in cluster} mean_{y in cluster} qdistance(x,y).
```

Thus every reported representative is an actual hard equilibrium reached by
one sampled respondent, not an averaged or synthetic categorical row.

Outputs are written under
`results/psisimulation/gss2018_equilibrium_clusters/` by default:

```text
equilibria_hard.csv
convergence.csv
qdistance_matrix.npy
qdistance_matrix.csv
embedding_2d.csv
cluster_assignments.csv
major_cluster_representatives.csv
equilibrium_clusters_mds2d.png
summary.json
```

For a larger geometry after a successful 300-row run:

```bash
python3 applications/psisimulation/cluster_equilibria.py \
  --model gss/gss_2018 \
  --data "$DATA" \
  --sample-size 500 \
  --sweeps 10 \
  --threads "$(nproc)" \
  --out results/psisimulation/gss2018_equilibrium_clusters_n500
```
