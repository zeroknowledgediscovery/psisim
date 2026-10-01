# PsiSim webapp instruction

This document specifies the simulation strategy and implementation plan for an
interactive PsiSim web application. The first target should be the **GSS 2018
Large Science Model**.

The purpose of the app is to make the central idea visually understandable:
the LSM represents a cross-linked system of conditional opinions. The state is
not one answer per survey item, but a probability distribution over every
modeled item. When one item is assigned a concrete value, that intervention
changes other distributions through the learned LSM dependency structure. The
system then relaxes under the PsiSim dynamics. A second item is queried in the
**new** state, so the sequence of observations creates a history-dependent,
reflexive trajectory.

The intended visual metaphor is **dropping a stone into water**. A local
answer creates an immediate splash and then successive ripples across the
modeled opinion system. The animation is a presentation of native simulation
snapshots; screen distance must not be interpreted as a physical propagation
time.

---

## 1. Scientific object shown by the webapp

For model variables $X_1,\ldots,X_d$, with categorical alphabets
$\Sigma_i$, the full state is

```math
\Psi=(p_1,p_2,\ldots,p_d),
\qquad
p_i\in\Delta(\Sigma_i).
```

Each $p_i$ is a categorical response distribution for one GSS variable.

The webapp should communicate three levels simultaneously:

1. **one variable**: the current distribution over its possible answers;
2. **the full state**: all distributions together, $\Psi$;
3. **the dependency system**: learned links between variables defined by the
   actual native LSM trees.

The app should describe this as a **reflexive model of linked opinions**:
the current state determines the distribution from which the next response is
generated, and that response in turn changes the state used for later
questions.

This is a model over learned survey-response dependencies. The UI should not
claim that graph edges are causal links or that the visualization is a literal
biological representation of a human brain.

---

## 2. Initial model: GSS 2018

Use the public model key

```text
gss/gss_2018
```

with the existing PsiSim model resolver.

### Model installation location

`applications/psisimulation/common.py::resolve_model()` accepts either a local
model directory or a public model key.

By default, the downloaded GSS 2018 model is installed under

```text
~/.cache/dtag/models/gss/gss_2018/
```

unless

```text
DTAG_MODEL_ROOT
```

is set.

For deployment, set `DTAG_MODEL_ROOT` to a persistent application-data
directory.

A valid native model contains at least

```text
gss_2018/
    source_maps/
        ...
    trees/
        binary/
            tree_0.bin
            tree_1.bin
            ...
```

The native binary trees are the learned conditional predictors. The source
maps provide the exact categorical labels and integer encodings accepted by
the trees.

If the model is not installed, bootstrap it with

```bash
python3 applications/psisimulation/fetch_model.py gss/gss_2018
```

The current default release in `fetch_model.py` is `v0.2.0`. The archive is
downloaded from the public DTAG model release and SHA256-verified before
installation. Model download should happen at deployment/startup, not in the
middle of an interactive browser request.

### Human-readable GSS question metadata

The native model determines the authoritative variable ordering and response
alphabets, but the webapp needs readable GSS question text.

DTAG already has the model-authoritative map

```text
maps/gss/gss_2018_map.csv
```

and builder

```text
scripts/build_gss_native_map.py
```

which combines the native model feature list with the GSS 1972--2018 codebook.

The standalone PsiSim webapp should not require a live DTAG checkout. Copy or
generate the required metadata into a PsiSim-owned asset, for example

```text
webapp/assets/gss/gss_2018_map.csv
```

and record its provenance.

Each UI variable record should retain:

- native model column index;
- GSS variable name;
- short label;
- full question text where available;
- legal response categories from the native source map.

The native column index remains the canonical simulation identifier.

---

## 3. Native bindings used

The browser must not reimplement PsiSim mathematics in JavaScript. The
authoritative simulation remains in the vendored native C++ runtime.

The main extension is

```text
bin/predict_distribution*.so
```

built from

```text
bindings/predict_distribution_py.cpp
```

The optional distance extension is

```text
bin/qdistance*.so
```

built from

```text
bindings/qdistance_py.cpp
```

### Interactive simulation binding

The webapp should use `predict_distribution` through the current helpers:

```python
from applications.psisimulation.common import (
    resolve_model,
    resident_empty_state,
    snapshot,
)

model = resolve_model("gss/gss_2018")
state = resident_empty_state(model)
psi = snapshot(state)
```

`resident_empty_state()` creates $\Psi_0$ and wraps it in a resident native
`CenteredLdpPsiState`.

The state object should remain resident for the full browser session. Do not
reload the model for every question.

### Hard observation

A concrete response for variable $i$ is applied with

```python
summary = state.hard_observe(
    source=i,
    value=raw_answer_label,
    response_scale=1.0,
    threads=threads,
    clamp=True,
)
```

This:

1. collapses $p_i$ to the selected point mass;
2. propagates the centered hard response to dependent targets;
3. clamps the observation so subsequent updates cannot erase it.

Afterward:

```python
psi_after_hard = snapshot(state)
```

This is the first post-intervention animation frame.

### Relaxation

Then execute centered mode sweeps:

```python
summary = state.sweep(
    n=10,
    event="mode",
    seed=session_seed,
    response_scale=1.0,
    threads=threads,
    random_permutation=False,
)
```

Take a snapshot after every sweep. The first webapp should use up to five
sweeps, matching the existing progressive example, or stop early when the
residual movement is sufficiently small.

### qdistance

`qdistance` is not required for every animation frame. It is useful later for
comparing endpoints, alternate question orders, and counterfactual branches.
Use the native binding rather than an ad hoc frontend distance.

---

## 4. Initialization at $\Psi_0$

The session begins from the all-missing hard row

```math
x_{\varnothing}
=
(\varnothing,\ldots,\varnothing).
```

For each learned coordinate,

```math
p_i^0
=
\phi_i(x_{\varnothing}).
```

Therefore

```math
\Psi_0
=
(p_1^0,\ldots,p_d^0).
```

This is not a uniform prior. It is the distributional state implied by the
trained GSS 2018 LSM when no survey answers have been supplied.

The initial UI should say something close to:

> No answers have been supplied. Every node is showing the response
> distribution implied by the GSS 2018 LSM from the empty state.

---

## 5. Querying a survey variable

The user should be able to search for a GSS item by variable name, short
label, question text, or answer text.

Selecting variable $i$ should open a question card containing:

- full GSS question text;
- variable name and native column index;
- current $p_i$;
- every legal response category and probability;
- whether the variable is already observed/clamped.

The distribution displayed must be from the **current** $\Psi$, not always
from $\Psi_0$.

Example:

```text
Question:
[human-readable GSS item]

Current response distribution:
    answer A       0.46
    answer B       0.31
    answer C       0.18
    other          0.05
```

---

## 6. How a response is generated

The interface should support three explicitly different modes.

### Simulate response

This should be the default demonstration mode.

Draw one concrete category from the current marginal:

```math
\sigma\sim p_i.
```

Then pass $\sigma$ to `hard_observe()`.

The UI can briefly animate the categorical probabilities before settling on
the sampled answer. Use a seeded session RNG so the demonstration can be
reproduced.

### Most likely response

Optionally choose

```math
\sigma
=
\arg\max_{s\in\Sigma_i} p_i(s).
```

Label this clearly as deterministic/MAP. Do not call it sampling.

### Choose response

Allow the user to select a category manually. This turns the app into an
intervention tool:

> What happens to the rest of the modeled worldview if this answer is forced?

Once submitted, both simulated and manually selected answers use the same
native `hard_observe(..., clamp=True)` operation.

---

## 7. What the answer does mathematically

Suppose the current queried marginal is $p_i$ and the selected answer is
$\sigma$.

The source distribution becomes

```math
\nu_i
=
\delta_{\sigma}.
```

The centered displacement is

```math
\delta_i
=
\delta_{\sigma}-p_i.
```

For each learned target $j$ whose native tree actually uses source $i$,
PsiSim uses the cached one-coordinate hard-response kernel

```math
K_{j\leftarrow i}(\cdot\mid s)
=
\phi_j\!\left(x^{(i=s)}\right).
```

The propagated response is

```math
r_{j\leftarrow i}
=
\sum_{s\in\Sigma_i}
\delta_i(s)
K_{j\leftarrow i}(\cdot\mid s).
```

With the default response scale $\alpha=1$,

```math
p_j^{+}
=
\Pi_{\Delta(\Sigma_j)}
\left[
p_j+r_{j\leftarrow i}
\right].
```

That immediate state change is the first "splash."

The queried source remains clamped. Later perturbations can reorganize the
rest of the system but cannot overwrite the supplied answer.

---

## 8. Relaxation: the later ripples

After the hard observation, execute sequential finite-$n$ mode sweeps.

For the initial implementation:

```text
event                 mode
empirical_n           10
response_scale        1.0
random_permutation    false
maximum sweeps        5
```

Each sweep updates the resident state and produces a new snapshot.

The animation sequence should therefore be:

```text
frame 0    state immediately before the answer
frame 1    immediate hard-observation response
frame 2    after relaxation sweep 1
frame 3    after relaxation sweep 2
frame 4    after relaxation sweep 3
...
```

The native diagnostics `mean_tv` and `max_tv` quantify how much the state is
still moving. The visual ripple should fade as those quantities become small.

---
