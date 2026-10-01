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

For model variables \(X_1,\ldots,X_d\), with categorical alphabets
\(\Sigma_i\), the full state is

```math
\Psi=(p_1,p_2,\ldots,p_d),
\qquad
p_i\in\Delta(\Sigma_i).
```

Each \(p_i\) is a categorical response distribution for one GSS variable.

The webapp should communicate three levels simultaneously:

1. **one variable**: the current distribution over its possible answers;
2. **the full state**: all distributions together, \(\Psi\);
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

`resident_empty_state()` creates \(\Psi_0\) and wraps it in a resident native
`CenteredLdpPsiState`.

The state object should remain resident for the full browser session. Do not
reload the model for every question.

### Hard observation

A concrete response for variable \(i\) is applied with

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

1. collapses \(p_i\) to the selected point mass;
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

## 4. Initialization at \(\Psi_0\)

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

Selecting variable \(i\) should open a question card containing:

- full GSS question text;
- variable name and native column index;
- current \(p_i\);
- every legal response category and probability;
- whether the variable is already observed/clamped.

The distribution displayed must be from the **current** \(\Psi\), not always
from \(\Psi_0\).

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

Then pass \(\sigma\) to `hard_observe()`.

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

Suppose the current queried marginal is \(p_i\) and the selected answer is
\(\sigma\).

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

For each learned target \(j\) whose native tree actually uses source \(i\),
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

With the default response scale \(\alpha=1\),

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

After the hard observation, execute sequential finite-\(n\) mode sweeps.

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

## 9. Main visualization: the reflexive mind map

The center of the application should be a stable network/field of opinion
variables.

A literal full display of all GSS distributions is too dense. The default
visualization should therefore show a focused subset while the backend retains
the entire \(\Psi\).

### Node meaning

One node = one GSS model variable.

Suggested encodings:

```text
label           short variable/question label
size            recent TV change
inner fill      concentration of dominant response
outer ring      uncertainty / entropy
halo            latest perturbation magnitude
pin/lock        hard-observed and clamped
```

Do not assign one global color scale to answer categories unless their
semantics are actually comparable. "Yes" for one item and "strongly agree" for
another are not a shared numeric state.

### Edge meaning

Display a directed edge \(i\to j\) when target tree \(j\) actually uses source
\(i\).

An edge means learned model dependency, not necessarily causation.

The complete graph will be visually dense. Normally show:

- edges from the currently queried/observed node;
- edges among the strongest-changing nodes;
- one- or two-hop local neighborhoods;
- previously observed nodes;
- the full graph only as an advanced option.

### Stone-in-water animation

On response:

1. the queried node visibly collapses to the chosen categorical value;
2. the node becomes pinned;
3. a bright pulse starts at that node;
4. variables with large immediate TV movement pulse strongly;
5. the immediate post-observation state appears;
6. each relaxation snapshot produces another wave;
7. the wave decays as residual movement decreases.

For node \(j\), pulse intensity should be tied to the actual per-coordinate
total variation:

```math
d_j
=
\frac{1}{2}
\sum_{s\in\Sigma_j}
\left|
p_j^{\mathrm{after}}(s)
-
p_j^{\mathrm{before}}(s)
\right|.
```

The animation should not invent a causal propagation velocity. Spatial ripple
timing is a visualization of simulation phases.

---

## 10. Stable layout

Do not let a force-directed layout rearrange the graph after each update. That
would make visual motion ambiguous.

Use a fixed layout for the GSS 2018 dependency graph:

1. precompute 2-D coordinates once;
2. optionally identify broad graph communities;
3. keep node positions fixed within and across sessions;
4. animate state changes, not node positions.

A temporary radial/ripple overlay can originate at the queried node without
moving the graph itself.

---

## 11. Exact \(\Psi\) view

The network is intuitive but cannot by itself represent every categorical
probability.

Provide a second "Psi" view: sortable table/heatmap with one row per variable.

Suggested columns:

```text
Variable
Question
Top response
P(top)
Entropy
TV change
Observed?
```

Clicking a row opens the complete current marginal distribution.

Important sorts:

- Changed most;
- Most uncertain;
- Most concentrated;
- Observed;
- variable name.

A user should be able to toggle:

```text
Mind | Psi | Timeline
```

without altering the session.

---

## 12. Showing uncertainty

For the overview, normalized entropy can summarize distributional uncertainty:

```math
h_i
=
-
\frac{
\sum_s p_i(s)\log p_i(s)
}{
\log |\Sigma_i|
}.
```

Use the complete categorical distribution in the detail panel. Do not reduce
the whole app to MAP answers, because the scientific object is \(\Psi\), not
only its coordinatewise argmax.

---

## 13. The second question is the key reflexive demonstration

After the first answer and relaxation, the user selects a second item \(k\).

The app must display its **new current distribution** \(p_k\).

This can differ from its initial distribution \(p_k^0\).

Then:

```math
\sigma_2\sim p_k
```

in simulated mode, or the user chooses a category manually.

That answer is clamped and launches another perturbation from the already
changed worldview.

This should be visually explicit:

```text
Observation 1
    -> global reorganization
    -> new state

Question 2 is asked in that new state
    -> response distribution is different
    -> observation 2
    -> another global reorganization
```

This repeated perturbation/relaxation sequence is the core demonstration of a
reflexive LSM.

---

## 14. Persistent history

Keep a visible history rail:

```text
1. [GSS item 1] = working fulltime
2. [GSS item 2] = yes
3. ...
```

For each observation store/display:

- question;
- response;
- response mode: sampled / MAP / manual;
- probability of the chosen answer immediately before intervention;
- immediate `mean_tv` and `max_tv`;
- number of relaxation sweeps;
- final residual `max_tv`.

Clicking a historical event should eventually replay its ripple.

A later version should allow "branch here" so the same pre-answer state can be
given two alternative responses and the resulting trajectories compared.

---

## 15. Backend session state

Each browser session needs its own mutable native simulation state.

Suggested server object:

```python
session_id
model_key
model_path
state
psi0
current_psi
observations
snapshots
diagnostics
rng
seed
created_at
last_accessed_at
```

The model files can be shared read-only across sessions, but the mutable
`CenteredLdpPsiState` must be session-specific.

---

## 16. Backend API plan

A FastAPI backend is a straightforward fit because the simulation layer is
already Python calling native CPython extensions.

Suggested API:

```text
POST /api/session
GET  /api/variables?q=...
GET  /api/session/{id}/variable/{column}
POST /api/session/{id}/simulate-response
POST /api/session/{id}/observe
GET  /api/session/{id}/state
POST /api/session/{id}/reset
```

### Create session

```json
{
  "model": "gss/gss_2018",
  "seed": 12345
}
```

Create a resident \(\Psi_0\) state.

### Variable lookup

Return native column, GSS metadata, current distribution, and clamped state.

### Simulate response

Sample one answer from the **current** marginal without silently substituting
the MAP answer.

### Observe

Example request:

```json
{
  "column": 17,
  "value": "yes",
  "source": "simulated",
  "sweeps": 5,
  "empirical_n": 10
}
```

The backend should:

1. apply `hard_observe`;
2. snapshot the immediate response;
3. execute one sweep at a time;
4. snapshot every sweep;
5. stream transition data to the frontend.

---

## 17. Streaming updates

Do not wait until all five sweeps finish.

Use WebSocket or Server-Sent Events to emit transitions such as:

```json
{
  "phase": "hard_observation",
  "frame": 1,
  "mean_tv": 0.021,
  "max_tv": 0.194,
  "max_col": 322,
  "changed": []
}
```

then:

```json
{
  "phase": "relaxation",
  "sweep": 1,
  "mean_tv": 0.008,
  "max_tv": 0.067,
  "changed": []
}
```

The `changed` array should include compact per-variable changes needed by the
animation, e.g.:

- native column;
- TV movement;
- previous/new dominant answer;
- previous/new dominant probability;
- previous/new entropy;
- clamp status.

The complete \(\Psi\) can remain server-side and be fetched on demand.

---

## 18. Managing hundreds of variables

Do not render the full dependency graph as an unreadable hairball by default.

After an intervention, show roughly 30--60 nodes:

1. the observed node;
2. the largest-changing nodes;
3. direct dependency neighbors;
4. all already clamped nodes;
5. a limited number of contextual nodes.

Provide advanced controls:

```text
Show 100
Show all changed
Show complete dependency graph
```

The simulation always updates the full state even when only a subset is
visible.

---

## 19. Suggested desktop layout

### Left panel: question/query

```text
GSS 2018
Search survey items...

Selected question
[full question text]

Current response distribution
[bar chart]

[Simulate response]
[Most likely]
[Choose response]
```

### Center: reflexive mind canvas

Stable graph with ripple animation.

Persistent status:

```text
State: after 2 observations
Phase: relaxation sweep 3 / 5
```

### Right panel: state and history

Tabs:

```text
Changed most
Observed
Variable details
History
```

A bottom timeline can provide replay.

---

## 20. Frontend technology

A reasonable initial stack:

```text
React or Next.js
TypeScript
Cytoscape.js or D3 for graph rendering
Canvas/WebGL overlay for ripple effects if needed
WebSocket or SSE for streamed transitions
```

Cytoscape.js is a good first choice for a stable interactive dependency graph.
Do not move model inference or update rules into frontend code.

---

## 21. Backend performance rules

The native runtime already helps by preloading trees, caching hard-response
kernels, using the actual dependency graph, and parallelizing native work.

The web service should additionally:

- keep state resident;
- precompute graph topology;
- precompute stable node positions;
- compute per-coordinate deltas server-side;
- stream one frame per sweep in the first version;
- avoid serializing every full marginal on every frame;
- cap OpenMP threads per session;
- use a bounded simulation worker pool.

Do not allow each concurrent request to invoke `nproc` threads independently.

---

## 22. Reproducibility

Every session should log:

```text
PsiSim git commit
LSM_RUNTIME_SOURCE_COMMIT
model key
model release
model path/digest
session seed
response generation mode
ordered observations
probability of each selected answer before observation
empirical_n
response_scale
source order
number of sweeps
per-sweep diagnostics
```

A later "Download session" button should export this as JSON.

---

## 23. Wording and interpretation

Good UI language:

```text
Model state
Current response distribution
Simulated response
Observed answer
Propagation
Relaxation
Largest changes
Pinned observations
```

Avoid:

```text
The model believes...
This proves X causes Y...
This is literally how a human brain changes...
```

Recommended concise description:

> This GSS-trained Large Science Model represents a coupled system of
> conditional response distributions. Choose or simulate one response and
> watch the modeled worldview reorganize.

A useful public-facing subtitle is:

> **A reflexive model of linked opinions: ask a question, supply an answer,
> and watch the modeled worldview reorganize.**

---

## 24. Milestone 1

The first usable version should include:

- GSS 2018 only;
- model bootstrap validation;
- local GSS 2018 human-readable question map;
- creation of a resident session at \(\Psi_0\);
- searchable GSS items;
- full current marginal for selected item;
- simulated response sampled from current \(p_i\);
- optional MAP and manual response modes;
- native `hard_observe(..., clamp=True)`;
- immediate response snapshot;
- five native mode sweeps with \(n=10\);
- streamed snapshots;
- per-variable TV changes;
- stable mind-map visualization;
- ripple animation;
- visible clamped nodes;
- "changed most" panel;
- history rail;
- reset/new session.

This is enough to show the scientific concept clearly.

---

## 25. Milestone 2

After Milestone 1 is stable:

- branch a prior state and compare alternate answers;
- reverse question order and compare trajectories;
- side-by-side path visualization;
- qdistance between hard endpoints;
- connect endpoints to equilibrium/cluster structure;
- add GSS 2022 and 2024;
- add additional survey families;
- stochastic relaxation experiments;
- session replay/export;
- advanced full dependency graph;
- semantic question search.

---

## 26. Reference server-side transaction

The central operation should look approximately like:

```python
model = resolve_model("gss/gss_2018")
state = resident_empty_state(model)
psi = snapshot(state)

# User queries variable i.
current_distribution = psi[i]

# Default demonstration: draw a concrete response from current p_i.
sigma = sample_from_distribution(
    current_distribution,
    rng=session_rng,
)

# Drop the "stone".
hard_summary = dict(
    state.hard_observe(
        source=i,
        value=sigma,
        response_scale=1.0,
        threads=session_threads,
        clamp=True,
    )
)

after_hard = snapshot(state)
emit_transition(psi, after_hard, hard_summary)

previous = after_hard

# Successive ripples.
for sweep in range(1, 6):
    summary = dict(
        state.sweep(
            n=10,
            event="mode",
            seed=session_seed,
            response_scale=1.0,
            threads=session_threads,
            random_permutation=False,
        )
    )

    current = snapshot(state)
    emit_transition(previous, current, summary, sweep=sweep)
    previous = current

    if summary["max_tv"] <= DISPLAY_SETTLED_TOL:
        break

psi = previous
```

The sampling helper must use a reproducible Python/NumPy RNG and log the seed
and chosen category.

---

## 27. Reference user experience

One interaction should feel like:

```text
STATE
Psi_0

QUERY
"What is your response to ...?"

CURRENT DISTRIBUTION
A  ████████████  0.48
B  ████████      0.31
C  █████         0.21

SIMULATED RESPONSE
A

INTERVENTION
the selected node collapses to A
the node becomes pinned

IMMEDIATE RESPONSE
a strong ripple appears
the "changed most" list updates

RELAXATION 1
the next wave propagates

RELAXATION 2
the system continues reorganizing

...

SETTLED
history now contains observation 1
```

The user then asks a second item. Its response distribution is obtained from
the **current** \(\Psi\), not \(\Psi_0\). That second answer launches a second
ripple while the first answer remains clamped.

That repeated query, response, perturbation, and relaxation cycle is the core
of the PsiSim webapp.
