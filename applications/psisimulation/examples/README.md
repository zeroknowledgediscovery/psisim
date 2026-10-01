# Worked example: progressive Psi dynamics on the GSS-2018 LSM

This example is intended to be read as a complete worked specification of a
PsiSim experiment, not merely as a command-line recipe.

The experiment starts with the native GSS-2018 Large Science Model, constructs
the probability-valued empty state $\Psi_0$, imposes two survey answers as
persistent hard observations, propagates each answer through the learned LSM
response structure, and performs five deterministic finite-$n$ relaxation
sweeps after each observation.

The supplied choice file is

\`\`\`json
[
  {
    "column": 0,
    "value": "working fulltime",
    "sweeps": 5
  },
  {
    "column": 3,
    "value": "yes",
    "sweeps": 5
  }
]
\`\`\`

The categorical labels above were verified against the GSS-2018 native model.
The example deliberately identifies variables by native model column because
the dynamics operates on the exact model coordinate system and source maps.

---

## 1. Command

From the repository root:

\`\`\`bash
python3 applications/psisimulation/simulate_progressive.py \
  --model gss/gss_2018 \
  --choices applications/psisimulation/examples/gss2018_choices.json \
  --empirical-n 10 \
  --threads "$(nproc)" \
  --gif
\`\`\`

The complete convenience script is

\`\`\`bash
bash applications/psisimulation/run_example.sh
\`\`\`

The example uses

$
n=10
$

for the finite empirical mode updates,

$
\alpha=1
$

for the centered response scale, and

$
\texttt{random\_permutation}=\texttt{False}
$

so every relaxation sweep uses the deterministic native tree order.

---

## 2. What model is being used?

The model key

\`\`\`text
gss/gss_2018
\`\`\`

resolves to a native LSM model directory containing

\`\`\`text
trees/binary/tree_*.bin
source_maps/
\`\`\`

Each learned target coordinate $j$ has a native tree implementing a
categorical conditional predictor

$
\phi_j(x_{-j})
=
\Pr_{\mathrm{LSM}}(X_j=\cdot\mid x_{-j}).
$

The source maps define the legal raw categorical labels for each coordinate and
their native integer encodings.

The model is not a language model.  It is a collection of learned conditional
predictors over the survey coordinate system.

If the model is not already installed, \`fetch_model.py\` downloads the public
model archive, verifies its SHA256 against the release manifest, and installs
it under

\`\`\`text
~/.cache/dtag/models/gss/gss_2018
\`\`\`

unless \`DTAG_MODEL_ROOT\` overrides the cache root.

---

## 3. State space

Let the GSS-2018 model contain learned coordinates

$
X_1,\ldots,X_d
$

with categorical alphabets

$
\Sigma_1,\ldots,\Sigma_d.
$

PsiSim does not evolve one categorical value per variable.  It evolves the
probability-valued state

$
\Psi
=
(p_1,\ldots,p_d),
$

where

$
p_i\in\Delta(\Sigma_i).
$

Thus the dynamical state lives in

$
\prod_{i=1}^d\Delta(\Sigma_i).
$

For example, if one variable has categories

$
\Sigma_i
=
\{\text{yes},\text{no},\text{don't know}\},
$

then that coordinate of Psi might be

$
p_i
=
(0.61,0.34,0.05),
$

rather than a single hard symbol.

---

## 4. Step 0: construct the empty state

The simulation first forms the all-missing hard row

$
x_\varnothing
=
(\varnothing,\ldots,\varnothing).
$

For every learned coordinate $i$,

$
p_i^0
=
\phi_i(x_\varnothing).
$

Therefore

$
\Psi_0
=
(p_1^0,\ldots,p_d^0).
$

In the code this is

\`\`\`python
state = resident_empty_state(model)
\`\`\`

which internally evaluates the native LSM and creates a resident C++
\`CenteredLdpPsiState\`.

No empirical GSS respondent is used to create this initial state.

The first figure written by the progressive simulation is

\`\`\`text
frames/000_empty.png
\`\`\`

and represents $\Psi_0$.

---

## 5. The learned hard-response kernel

Before following the two questions, define the exact response kernel used by
the implementation.

For source coordinate $i$, target coordinate $j$, and source symbol
$\sigma\in\Sigma_i$, form the hard row

$
x^{(i=\sigma)}
$

whose only observed coordinate is

$
x_i=\sigma.
$

Every other coordinate is missing.

Then

$
K_{j\leftarrow i}(\cdot\mid\sigma)
=
\phi_j(x^{(i=\sigma)}).
$

The native runtime caches this distribution using the key

$
(i,j,\sigma).
$

A perturbation of source $i$ is propagated only to target trees $j$ whose
learned tree actually uses $i$.

This response is therefore determined by the learned LSM dependency structure,
not by a manually specified interaction matrix.

---

## 6. Question 1: column 0 = \`"working fulltime"\`

Let source coordinate $i=0$.

Immediately before the first question its distribution is

$
p_0^0.
$

The answer

\`\`\`text
working fulltime
\`\`\`

is represented as the point mass

$
\nu_0
=
\delta_{\mathrm{working\ fulltime}}.
$

The centered source displacement is

$
\Delta_0
=
\nu_0-p_0^0.
$

Written componentwise,

$
\Delta_0(\sigma)
=
\mathbf 1
\{
\sigma=\text{working fulltime}
\}
-
p_0^0(\sigma).
$

The source coordinate is hard-collapsed:

$
p_0^+
=
\delta_{\mathrm{working\ fulltime}}.
$

For every dependent target $j$, the immediate response is

$
r_{j\leftarrow 0}
=
\sum_{\sigma\in\Sigma_0}
\Delta_0(\sigma)
K_{j\leftarrow 0}(\cdot\mid\sigma).
$

With the default response scale $\alpha=1$,

$
\widetilde p_j^+
=
p_j^0+r_{j\leftarrow 0}.
$

If this candidate leaves the categorical simplex, it is projected back:

$
p_j^+
=
\Pi_{\Delta(\Sigma_j)}
\left(
\widetilde p_j^+
\right).
$

All learned target coordinates whose trees do not use column 0 remain
unchanged by this immediate response.

The runtime reports

\`\`\`text
mean_tv
max_tv
max_col
event_tv
sanov_rate
projected_count
\`\`\`

for this hard observation.

For a hard answer, the source event TV is

$
\operatorname{TV}
\left(
p_0^0,
\delta_{\mathrm{working\ fulltime}}
\right).
$

The KL rate computed internally is

$
D
\left(
\delta_{\mathrm{working\ fulltime}}
\middle\|
p_0^0
\right)
=
-\log
p_0^0(\mathrm{working\ fulltime}),
$

provided the chosen answer has positive model probability.

The hard-observation operation itself is not assigned a finite empirical
sample size $n$, so the output does not interpret this as a finite-$n$
Sanov exponent.

The resulting frame is

\`\`\`text
frames/001_q1_hard.png
\`\`\`

up to numbering changes caused by modifications to the choice sequence.

---

## 7. Why column 0 is clamped

The call uses

\`\`\`python
clamp=True
\`\`\`

so the first answer becomes persistent evidence.

From this point onward,

$
p_0
=
\delta_{\mathrm{working\ fulltime}}
$

through all later relaxation sweeps and after the second question.

There are two implementation consequences.

First, when column 0 is visited as a source during a later sweep, its empirical
distribution is forced to the same point mass.  Since it is already at that
point mass, the centered displacement is normally exactly zero.

Second, if another source would propagate a response into target column 0,
that target update is skipped.

The hard answer is therefore a fixed boundary condition, not a transient
perturbation.

---

## 8. Relaxation after Question 1

Five deterministic finite-$n$ mode sweeps are now performed.

The example uses

$
n=10.
$

For each non-clamped source coordinate $i$, suppose the current distribution
at the moment that source is visited is

$
p_i.
$

PsiSim constructs the multinomial modal empirical type

$
\nu_i
=
\frac{c}{10},
$

where the integer counts satisfy

$
\sum_{\sigma\in\Sigma_i}c_\sigma=10.
$

The implementation obtains the global multinomial mode by assigning counts one
at a time to the category maximizing

$
\frac{p_i(\sigma)}{c_\sigma+1}.
$

Because $n=10$, every component of $\nu_i$ is an integer multiple of

$
0.1.
$

This means that even a smooth probability vector such as

$
(0.63,0.24,0.13)
$

would typically be replaced by a nearby finite empirical type such as

$
(0.6,0.3,0.1),
$

depending on the exact multinomial mode.

That finite empirical displacement

$
\Delta_i
=
\nu_i-p_i
$

is then propagated with the same centered response formula used for the hard
observation.

---

## 9. A relaxation sweep is sequential

Suppose the learned source tree order is

$
i_1,i_2,\ldots,i_m.
$

One sweep is

$
S
=
T_{i_m}
\circ\cdots\circ
T_{i_2}
\circ
T_{i_1}.
$

The state used by $T_{i_2}$ is already the state produced by $T_{i_1}$.

Thus a sweep is not equivalent to computing all source events from the
beginning-of-sweep state and applying them simultaneously.

This matters because generally

$
T_iT_j\neq T_jT_i.
$

The example sets

\`\`\`text
random_permutation = false
\`\`\`

so the same native tree order is used in every sweep.

This makes the deterministic example reproducible.

---

## 10. Large-deviation diagnostic during the mode sweep

For each finite empirical event,

$
\nu_i,
$

the runtime reports

$
D(\nu_i\|p_i)
=
\sum_{\sigma:\nu_i(\sigma)>0}
\nu_i(\sigma)
\log
\frac{\nu_i(\sigma)}{p_i(\sigma)}.
$

The logarithm is natural, so the quantity is in nats.

For $n=10$, the event exponent is

$
10D(\nu_i\|p_i).
$

A complete sweep accumulates the source rates and reports

$
10
\sum_{i\in\text{sweep}}
D(\nu_i\|p_i).
$

This diagnostic quantifies how atypical the finite empirical events are
relative to the current marginals.

It is not used to accept or reject an update; it is recorded as a property of
the trajectory.

---

## 11. What happens during the five sweeps?

After the first hard answer, the trajectory is

$
\Psi_{1,0}
\rightarrow
\Psi_{1,1}
\rightarrow
\Psi_{1,2}
\rightarrow
\Psi_{1,3}
\rightarrow
\Psi_{1,4}
\rightarrow
\Psi_{1,5},
$

where

$
\Psi_{1,0}
$

is the immediate post-answer state and

$
\Psi_{1,r}
=
S_{10}
(
\Psi_{1,r-1}
)
$

for $r=1,\ldots,5$.

For every sweep the script records

$
\operatorname{meanTV}
$

and

$
\operatorname{maxTV}
$

between the state before the full sweep and the state after the full sweep.

If these values decrease toward zero, the state is approaching a numerical
fixed point of this deterministic finite-$n$ update rule.

Five sweeps are an experimental choice in the supplied example.  Five sweeps
do **not** by definition prove convergence.  Convergence should be judged from
the residual diagnostics or tested by continuing the trajectory.

---

## 12. Question 2: column 3 = \`"yes"\`

After five relaxation sweeps, denote the current state by

$
\Psi_{1,5}.
$

Now source coordinate $i=3$ is hard-observed as

$
X_3=\text{yes}.
$

Let its pre-observation marginal be

$
p_3^{(1,5)}.
$

The imposed empirical distribution is

$
\nu_3
=
\delta_{\mathrm{yes}},
$

and the centered displacement is

$
\Delta_3
=
\delta_{\mathrm{yes}}
-
p_3^{(1,5)}.
$

For each dependent target $j$,

$
r_{j\leftarrow3}
=
\sum_{\sigma\in\Sigma_3}
\Delta_3(\sigma)
K_{j\leftarrow3}(\cdot\mid\sigma).
$

The target update is again

$
p_j^+
=
\Pi_{\Delta(\Sigma_j)}
\left[
p_j
+
r_{j\leftarrow3}
\right].
$

At this point both observations are clamped:

$
p_0
=
\delta_{\mathrm{working\ fulltime}},
$

and

$
p_3
=
\delta_{\mathrm{yes}}.
$

No subsequent response is permitted to alter either clamped target.

This is an important distinction from simply restarting the model with two
hard values.  The second observation is imposed on the probability state
already reached after the first answer and its five relaxation sweeps.

---

## 13. Relaxation after Question 2

The state now undergoes another five deterministic mode sweeps:

$
\Psi_{2,0}
\rightarrow
\Psi_{2,1}
\rightarrow
\cdots
\rightarrow
\Psi_{2,5}.
$

During these sweeps,

$
X_0=\text{working fulltime}
$

and

$
X_3=\text{yes}
$

remain fixed as persistent evidence.

The remaining coordinates are free to move under the finite-$n$ centered
response dynamics.

The final probability-valued state is

$
\Psi_{\mathrm{final}}
=
\Psi_{2,5}.
$

---

## 14. The final hard row is not the state

At the end the script writes

\`\`\`text
final_hard_row.csv
\`\`\`

with

$
x_i^\star
=
\arg\max_{\sigma\in\Sigma_i}
p_i^{\mathrm{final}}(\sigma).
$

This hard row is useful for

* qdistance computation;
* endpoint clustering;
* discrete inspection;
* comparison with empirical respondent rows.

But the simulation itself evolves

$
\Psi_{\mathrm{final}},
$

not merely

$
x^\star.
$

Two probability states can have the same MAP hard row while retaining
different uncertainty distributions.

---

## 15. Exact pseudo-code for this example

Conceptually, the example is

\`\`\`text
load native GSS-2018 LSM

x_empty = all missing

for each learned i:
    p_i = phi_i(x_empty)

Psi = (p_1, ..., p_d)

# Question 1
nu_0 = point_mass("working fulltime")
Delta_0 = nu_0 - p_0
p_0 = nu_0

for every learned target j that uses source 0:
    if j is not clamped:
        response_j =
            sum_sigma Delta_0[sigma] * K_{j<-0}(. | sigma)
        p_j = simplex_projection(p_j + response_j)

clamp column 0

repeat 5 times:
    for source i in deterministic learned-tree order:
        if i is clamped:
            nu_i = its clamped point mass
        else:
            nu_i = multinomial_mode_type(p_i, n=10)

        Delta_i = nu_i - p_i
        p_i = nu_i

        for every learned target j that uses source i:
            if j is not clamped:
                response_j =
                    sum_sigma Delta_i[sigma] * K_{j<-i}(. | sigma)
                p_j = simplex_projection(p_j + response_j)

# Question 2
nu_3 = point_mass("yes")
Delta_3 = nu_3 - p_3
p_3 = nu_3

propagate the same centered response from source 3
clamp column 3

repeat 5 deterministic n=10 mode sweeps

write full trajectory diagnostics
write coordinatewise MAP hard endpoint
write visualization frames
optionally assemble GIF
\`\`\`

---

## 16. Why the centered response is essential in the example

Suppose an empirical event at source $i$ exactly equals its current marginal:

$
\nu_i=p_i.
$

Then

$
\Delta_i=0.
$

Therefore

$
\sum_\sigma
\Delta_i(\sigma)
K_{j\leftarrow i}(\cdot\mid\sigma)
=
0
$

for every target $j$.

So the entire response is exactly zero.

This property is particularly useful for debugging.  It does not depend on
the learned conditionals satisfying an exact global probability identity.

The companion test

\`\`\`bash
python3 applications/psisimulation/simulate_empty.py \
  --model gss/gss_2018 \
  --sweeps 1 \
  --event zero_action
\`\`\`

should therefore report numerical zero movement.

---

## 17. Why question order can matter

The supplied sequence is

$
Q_1:
X_0=\text{working fulltime},
$

then five sweeps, then

$
Q_2:
X_3=\text{yes}.
$

This is not generally equivalent to reversing the questions.

The first answer changes many marginals.  The relaxation sweeps further change
the state.  Therefore the source distribution at column 3 when the second
question arrives is

$
p_3^{(1,5)},
$

not the original

$
p_3^0.
$

The second centered displacement is therefore

$
\delta_{\mathrm{yes}}
-
p_3^{(1,5)},
$

which can differ substantially from

$
\delta_{\mathrm{yes}}
-
p_3^0.
$

This makes question order a scientifically meaningful perturbation variable.

A useful experiment is to create

\`\`\`json
[
  {"column": 3, "value": "yes", "sweeps": 5},
  {"column": 0, "value": "working fulltime", "sweeps": 5}
]
\`\`\`

and compare the resulting endpoint and trajectory.

---

## 18. Why the number of relaxation sweeps can matter

The second question is asked after five sweeps in the supplied protocol.

If only one sweep is used, the second observation is imposed on a less-relaxed
state.

If many sweeps are used and the first-answer trajectory converges, the second
question is effectively applied near the first-answer fixed point.

Thus the integer

$
s_q
$

after each question controls the separation of timescales between external
interventions.

This can be used to study

* rapid questioning;
* quasi-equilibrated questioning;
* hysteresis;
* path dependence;
* response memory.

---

## 19. Why empirical $n$ matters

The default

$
n=10
$

sets the empirical resolution of the deterministic relaxation.

For small $n$, modal empirical distributions are coarse and can differ
substantially from the current marginals.

For larger $n$, the modal empirical type typically lies closer to the
current marginal.

Therefore the family of deterministic maps

$
S_n
$

depends on $n$.

One should not assume that an endpoint found at

$
n=10
$

is invariant under

$
n=20,\quad50,\quad100.
$

A useful sensitivity analysis is to repeat the exact same question sequence
over several $n$ values and compare

* convergence speed;
* final TV residuals;
* endpoint qdistance;
* number of simplex projections;
* cluster identity, if using population endpoints.

---

## 20. Response scale $\alpha$

The example uses

$
\alpha=1.
$

In the general update,

$
p_j^+
=
\Pi_\Delta
\left[
p_j
+
\alpha
\sum_\sigma
(\nu_i(\sigma)-p_i(\sigma))
K_{j\leftarrow i}(\cdot\mid\sigma)
\right].
$

Thus $\alpha$ scales propagation away from the source.

At

$
\alpha=0,
$

the source itself changes to its empirical distribution but no dependent
target responds.

At

$
0<\alpha<1,
$

the propagated response is damped.

At

$
\alpha>1,
$

the response is amplified and simplex projection may occur more frequently.

The supplied script fixes $\alpha=1$, which is the unscaled learned response.

---

## 21. What the generated PNG frames show

The frame sequence begins with

\`\`\`text
000_empty.png
\`\`\`

and then includes

\`\`\`text
q1_hard
q1_sweep1
...
q1_sweep5
q2_hard
q2_sweep1
...
q2_sweep5
\`\`\`

with numeric prefixes preserving temporal order.

Because different GSS variables have different alphabet sizes, the heatmap
cannot use one universal category axis.

Instead, each row corresponds to one LSM variable and the columns show ranked
probabilities:

* rank 1: largest probability;
* rank 2: second largest;
* and so on.

The frame also shows movement from the previous state and marks hard-observed
coordinates.

With

\`\`\`text
--gif
\`\`\`

the frames are assembled into

\`\`\`text
psi_dynamics.gif
\`\`\`

which is a visualization of motion through the product of simplices.

---

## 22. How to read \`history.json\`

Every hard observation and every relaxation sweep is appended to

\`\`\`text
history.json
\`\`\`

with contextual metadata and native diagnostics.

For a hard observation, a record includes fields such as

\`\`\`json
{
  "question": 1,
  "phase": "hard_observation",
  "sweep": 0,
  "column": 0,
  "value": "working fulltime",
  "mean_tv": ...,
  "max_tv": ...,
  "max_col": ...,
  "event_tv": ...,
  "sanov_rate": ...,
  "projected_count": ...
}
\`\`\`

For a relaxation sweep,

\`\`\`json
{
  "question": 1,
  "phase": "relaxation",
  "sweep": 1,
  "event_mode": "centered_sweep_mode",
  "n": 10,
  "event_tv": ...,
  "sanov_rate": ...,
  "sanov_exponent": ...,
  "mean_tv": ...,
  "max_tv": ...,
  "max_col": ...,
  "projected_count": ...
}
\`\`\`

The most important convergence quantities are

$
\operatorname{meanTV}
$

and

$
\operatorname{maxTV}.
$

If five sweeps are insufficient, continue the trajectory rather than calling
the final state an equilibrium solely because the script stopped.

---

## 23. Simplex projection as a diagnostic

A centered target response has zero total mass, but an individual probability
component can become negative before projection.

The code projects such a vector back to the simplex.

A nonzero

\`\`\`text
projected_count
\`\`\`

therefore means at least one learned linear response was large enough to leave
the feasible probability region before correction.

This is not necessarily an error, but it is scientifically relevant.

A regime with persistent heavy projection may indicate that

* the response scale is large;
* the empirical perturbations are coarse;
* the linear centered response is being applied far from a small-perturbation
  regime.

Projection frequency should therefore be reported in serious dynamical
analyses.

---

## 24. Inspecting legal answers before editing the example

Never guess raw categorical labels.

Inspect the empty-state marginal for a coordinate:

\`\`\`bash
python3 applications/psisimulation/inspect_column.py \
  --model gss/gss_2018 \
  --column 0
\`\`\`

This reports the model-supported labels and their probabilities at $\Psi_0$.

Then construct a new choice JSON using exact labels from the source maps.

For example,

\`\`\`json
[
  {
    "column": 10,
    "value": "<exact source-map label>",
    "sweeps": 8
  },
  {
    "column": 25,
    "value": "<exact source-map label>",
    "sweeps": 8
  }
]
\`\`\`

---

## 25. Recommended experimental variants

The supplied two-question sequence is only a minimal example.

### Reverse-order test

Reverse columns 0 and 3 to test noncommutativity and path dependence.

### Relaxation-time test

Run

$
s=0,1,2,5,10,20
$

sweeps between questions.

### Empirical-resolution test

Run

$
n=5,10,20,50,100.
$

### Stochastic test

Use the lower-level API with

\`\`\`text
event = sample
\`\`\`

instead of \`mode\`, repeat many seeds, and characterize the endpoint
distribution.

### Response-scale test

Vary

$
\alpha
$

and inspect convergence and projection frequency.

### Local stability test

Take a near-fixed endpoint, perturb selected marginals slightly, and test
whether the trajectory returns to the same state.

These experiments answer different scientific questions and should not be
collapsed into one notion of “equilibrium.”

---

## 26. Relation to respondent-level equilibrium clustering

The progressive example begins at the model-implied empty state and adds
chosen observations.

\`cluster_equilibria.py\` asks a different question.

It begins from many actual GSS respondent rows, constructs leave-one-coordinate
conditional Psi states around those rows, evolves them under the same centered
finite-$n$ dynamics, takes coordinatewise MAP endpoints, and compares those
endpoints using native qdistance.

Thus the progressive example studies

$
\text{controlled intervention paths from }\Psi_0,
$

whereas equilibrium clustering studies

$
\text{population endpoint geometry from empirical initial conditions}.
$

They use the same core response operator but answer different questions.

---

## 27. Native qdistance for comparing two endpoints

If two runs produce hard endpoints

$
x^\star
$

and

$
y^\star,
$

the native qdistance does not merely count unequal symbols.

For each learned target $j$, compute

$
P_j^x=\phi_j(x^\star),
\qquad
P_j^y=\phi_j(y^\star).
$

Then

$
d_j
=
\sqrt{
\operatorname{JS}_2
(
P_j^x,
P_j^y
)
}.
$

The global distance is

$
d_Q(x^\star,y^\star)
=
\frac1{|\mathcal T|}
\sum_{j\in\mathcal T}d_j.
$

This compares the two rows through their induced learned conditional
predictions across the model.

A reverse-order question experiment can therefore be quantified by the
qdistance between its two terminal hard endpoints.

---

## 28. What should count as evidence of a fixed point?

A strong numerical fixed-point claim should include more than “the script ran
for five sweeps.”

At minimum report

1. final \`mean_tv\`;
2. final \`max_tv\`;
3. additional-sweep stability;
4. \`projected_count\`;
5. empirical $n$;
6. source ordering;
7. response scale;
8. all clamps.

For an attractor claim, additionally perturb the candidate endpoint in
multiple directions and show return.

For a basin claim, sample a neighborhood or a broader set of initial states and
map which endpoint each trajectory approaches.

---

## 29. What this example does not assume

The worked example does **not** require that

$
\{\phi_i\}
$

be exact conditionals of one known explicit joint distribution.

It does not require

$
\sum_\sigma
p_i(\sigma)
K_{j\leftarrow i}(\cdot\mid\sigma)
=
p_j.
$

It does not require detailed balance.

It does not assume that five sweeps are sufficient for convergence.

It does not assume the final MAP hard row contains all information in the
terminal Psi state.

The only dynamical rule assumed is the one implemented explicitly by the
centered response operator.

---

## 30. Reproducibility record for the supplied example

A complete record should include

\`\`\`text
model:                  gss/gss_2018
initialization:         all-missing Psi0
question 1:             column 0 = "working fulltime"
question 1 clamp:       true
relaxation after Q1:    5 sweeps
question 2:             column 3 = "yes"
question 2 clamp:       true
relaxation after Q2:    5 sweeps
event during sweeps:    mode
empirical n:            10
response scale:         1.0
source order:           deterministic native tree order
random permutation:     false
seed:                   12345
runtime provenance:     LSM_RUNTIME_SOURCE_COMMIT
application provenance: PSISIM_SOURCE_COMMIT
\`\`\`

For publication-quality results, also save the PsiSim git commit and the exact
model-release identity.

---

## 31. Output directory

The default output is

\`\`\`text
results/psisimulation/progressive/
    frames/
        000_empty.png
        001_q1_hard.png
        002_q1_sweep1.png
        ...
    history.json
    final_hard_row.csv
    psi_dynamics.gif
\`\`\`

The scientific record is primarily

* the ordered intervention sequence;
* the probability-valued trajectory;
* the update diagnostics; and
* the endpoint.

The animation is a visualization of that record, not a substitute for it.

---

## 32. Minimal invariant tests after changing the runtime

If the vendored LSM bindings are synchronized from \`dev-static\`, rerun at
least the following checks before interpreting new results.

First:

\`\`\`bash
python3 applications/psisimulation/smoke_test.py \
  --model gss/gss_2018 \
  --column 0 \
  --value "working fulltime" \
  --threads "$(nproc)"
\`\`\`

Verify:

1. a zero-action sweep is numerically identical to the input state;
2. the hard observation collapses column 0 to the requested point mass;
3. column 0 remains clamped during a later mode sweep.

Then rerun this progressive example and compare the saved trajectory
diagnostics with the prior runtime revision.

A binding change that preserves imports but changes these invariants is a
scientifically material change.

---

## 33. Conceptual summary

The worked sequence can be written compactly as

$
\boxed{
\Psi_0
\xrightarrow{
X_0=\mathrm{working\ fulltime}
}
S_{10}^{5}
\xrightarrow{
X_3=\mathrm{yes}
}
S_{10}^{5}
\Psi_{\mathrm{final}}
}
$

with both observed coordinates clamped after they are introduced.

Every arrow labeled by a hard answer means

$
\text{point-mass replacement}
+
\text{centered kernel propagation}.
$

Every $S_{10}$ means one full deterministic sequential sweep in which each
unclamped source is replaced by its size-10 multinomial modal empirical type
and the centered displacement is propagated through the learned LSM
dependency graph.

That is the exact experiment implemented by
\`gss2018_choices.json\` and \`simulate_progressive.py\`.
