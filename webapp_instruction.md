# PsiSim webapp instruction

This document specifies the simulation strategy and implementation plan for an
interactive PsiSim web application. The first target should be the **GSS 2018
Large Science Model**.

The purpose of the app is to make the central idea visually understandable:
the LSM represents a cross-linked system of conditional opinions. The state is
not one answer per survey item, but a probability distribution over every
modeled item. When one item is assigned a concrete value, that intervention
changes other distributions through the learned LSM dependency structure. A
second item is queried in the **new** state, so the sequence of observations
creates a history-dependent, reflexive trajectory.

The visual goal is to make it obvious that **asking one question updates the
marginal distributions of many other questions**: the answered item becomes
a point mass, and the largest resulting updates elsewhere are shown one by
one (section 9). The animation is a presentation of native simulation
snapshots.

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

The default webapp dynamics live in the PsiSim-owned extension

```text
bin/psisim_dynamics*.so
```

built from

```text
bindings/psisim_dynamics_py.cpp
```

It provides `used_columns()` (the learned dependency graph) and
`PropagationPsiState`, the propagation-only centered dynamics of section 8.
It is not part of the vendored LSM runtime, so the LSM sync never overwrites
it; its wave 0 is regression-tested to equal the vendored
`CenteredLdpPsiState.hard_observe` exactly.

The optional distance extension is

```text
bin/qdistance*.so
```

built from

```text
bindings/qdistance_py.cpp
```

### Interactive simulation binding

The webapp keeps one resident propagation-only state per browser session,
initialised at $\Psi_0$:

```python
import psisim_dynamics
from applications.psisimulation.common import psi0, resolve_model

model = resolve_model("gss/gss_2018")
state = psisim_dynamics.PropagationPsiState(
    str(model / "trees" / "binary"), str(model), psi0(model)
)
psi = state.to_python()
```

The state object should remain resident for the full browser session. Do not
reload the model for every question.

### Hard observation (wave 0)

A concrete response for variable $i$ is applied with

```python
summary = state.hard_observe(i, raw_answer_label, clamp=True, threads=threads)
```

This:

1. collapses $p_i$ to the selected point mass;
2. propagates the centered hard response
   $\Delta_i^{(0)}=\delta_\sigma-p_i^{\text{before}}$ to dependent targets;
3. clamps the observation so subsequent updates cannot erase it;
4. records the change actually induced in every target as the perturbation
   of the next wave.

The result is identical to the vendored
`CenteredLdpPsiState.hard_observe(..., response_scale=1.0, clamp=True)`.
Afterward `state.to_python()` is the first post-intervention animation frame.

### One update per question (webapp default)

The webapp shows exactly one update of $\Psi$ per question: the hard
observation above. It runs no further waves and no relaxation, so what the
viewer sees is the direct effect of the answer.

### Propagation waves (optional, API only)

Further waves can be requested through the API (`max_steps > 0`):

```python
summary = state.wave(threads=threads)
```

Each wave propagates **only** the deltas newly induced by the previous wave
(section 8). With nothing pending a wave is the identity, bit for bit. Take a
snapshot after every wave. The webapp UI does not use waves.

The default webapp does **not** run `state.sweep(event="mode", n=10)` (or any
other sweep) after initialization or as passive relaxation. Finite-$n$ sweeps
inject a new empirical perturbation at every variable by quantizing $p_i$ onto
the $1/n$ grid; they move $\Psi_0$ without any answer and are therefore not
passive relaxation.

### Optional finite-$n$ empirical dynamics

`CenteredLdpPsiState.sweep(event="mode" | "sample", n=...)` remains available
as an explicitly labelled, optional finite-sample / LDP experiment (a session
type chosen when a session is created). It must never be presented as the
default relaxation process.

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

Under the default dynamics, $\Psi_0$ is stationary:

```math
\text{no external intervention} \;\Rightarrow\; \text{no motion},
\qquad
\Psi_0 \rightarrow \Psi_0
```

exactly, until an actual survey response is imposed. Everything that moves on
screen was moved by an answer.

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

Choosing a topic asks it immediately: there is no separate response-mode
choice in the UI. The response is simulated by drawing one concrete category
from the **current** marginal,

```math
\sigma\sim p_i,
```

and passing $\sigma$ to `hard_observe()`.

The UI can briefly animate the categorical probabilities before settling on
the sampled answer. Use a seeded session RNG so the demonstration can be
reproduced; show the uniform draw and the seed.

A question can be chosen from the search list or by clicking its sparkline in
the $\Psi$ grid (section 9); hovering a sparkline only shows its distribution.

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

## 8. Propagation waves (optional)

The webapp UI applies only the hard observation (wave 0) per question. If
further propagation is requested (API `max_steps > 0`), propagate only the
perturbation created by that observation, wave by wave.

Initial perturbation:

```math
\Delta_i^{(0)}
=
\delta_\sigma-p_i^{\text{before}}.
```

For each dependent, non-clamped target $j$, use the existing centered
hard-response kernel:

```math
r_{j\leftarrow i}^{(0)}
=
\sum_s
\Delta_i^{(0)}(s)
K_{j\leftarrow i}(\cdot\mid s),
\qquad
p_j^{\text{after}}
=
\Pi_{\Delta(\Sigma_j)}
\left[p_j^{\text{before}}+r_{j\leftarrow i}^{(0)}\right].
```

Then compute the actual newly induced change

```math
\Delta_j^{(1)}
=
p_j^{\text{after}}-p_j^{\text{before}}.
```

That new $\Delta_j^{(1)}$, and only that incremental change, is the
perturbation propagated during the next wave to the variables that depend on
$j$. Repeat wave by wave. When several sources of one wave reach the same
target, their responses are summed and projected once.

Rules:

- propagate only the newly generated delta from the preceding wave, never the
  accumulated change from the original state (otherwise feedback loops
  double-count earlier perturbations);
- clamped/observed variables remain fixed and are never modified as targets;
- response scale $1.0$, no damping (for now);
- invariant: $\Delta^{(0)}=0 \Rightarrow \Delta^{(1)}=\Delta^{(2)}=\cdots=0$.

The animation sequence is:

```text
frame 0    state immediately before the answer
frame 1    wave 0: immediate hard-observation response ("splash")
frame 2    after wave 1
frame 3    after wave 2
...
```

When waves are requested (API `max_steps > 0`; the UI uses none). If deltas remain
pending after the last wave, they are dropped and the remaining perturbation
(sum of TV norms) is reported, so one answer's waves never leak into the next
answer's.

The regression test `tests/test_propagation_dynamics.py` checks on the real
model that (1) $\Psi_0$ does not move without an intervention, (2) wave 0 is
non-zero and equals the vendored `hard_observe`, (3) each later wave equals an
independent reconstruction from the previous wave's newly induced deltas and
not from the accumulated change, and (4) clamped coordinates never change.

### Observed behaviour at response scale 1 (open issue)

On GSS 2018 the undamped waves do not die out. After an answer, the total
pending perturbation first shrinks for a few waves and then grows by roughly
1.3x per wave until simplex projection saturates it; it then keeps oscillating
indefinitely. Unasked items can be driven to point masses. A likely cause is
that each kernel $K_{j\leftarrow i}$ is a one-coordinate (marginal) response,
so a target that receives correlated perturbations from many sources counts
the same association several times. Damping or a different response scale is
a separate decision and is deliberately not applied yet.

---

## 9. Visualizing that asking a question changes $\Psi$

The key message is that answering one question changes the distributions of
many other questions, not only the answered one. A node-and-edge graph of
~1,000 items is not interpretable, so the webapp shows $\Psi$ directly:

- **Question panel.** The asked question's current distribution is drawn as
  a histogram. After the (animated) draw, it collapses to the point mass: the
  drawn answer becomes a red bar at 1 and the other answers drop to 0 with a
  grey outline of their previous value. A summary states how many of the
  other distributions changed, how many by more than 0.01 (TV), and lists the
  largest updates.
- **$\Psi$ grid.** Every $p_i$ is a small sparkline histogram (fixed answer
  order and scale per item), all ~1,000 on one screen in GSS column order so
  related items sit together. Colour encodes the distance of $p_i$ from
  $\Psi_0$ (one sequential hue, log scale); red = answered.
- **Update sequence.** After each question, the changed sparklines morph from
  their old to their new shape in a fast cascade, largest change first, and
  keep a gold outline. Then only the largest updates (up to 6, TV at least
  0.002) grow, one after another, into a card whose bars slide from before
  (grey outline) to after, and shrink back into their sparkline. The
  sequence can be replayed.
- **Current mind.** A donut with one spoke per question (survey order) whose
  length and colour show its distance from $\Psi_0$; answered questions are
  red. It summarises the whole state at a glance, with the answers so far.
- **Ask next.** Instead of a history list, the side panel offers clickable
  unasked questions ranked by how far the answers so far have moved them
  from $\Psi_0$.
