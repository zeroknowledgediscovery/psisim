# PsiSim webapp

An interactive view of the **GSS 2018 Large Science Model** as a reflexive
model of linked opinions, implementing [`webapp_instruction.md`](../webapp_instruction.md).

The state is the full distributional row
$\Psi=(p_1,\ldots,p_d)$, one categorical response distribution per modelled
GSS item. It starts at the empty state $\Psi_0$ and, under the default
dynamics, **does not move until a topic is answered**. Choosing a topic asks
it immediately: an answer $\sigma\sim p_i$ is drawn from its current
distribution and dropped into the system like a stone into water.

| frame | what is shown | native call |
| --- | --- | --- |
| 0 | state immediately before the answer | — |
| 1 | wave 0, the **splash**: $\Delta_i=\delta_\sigma-p_i$ propagated to the topics that depend on $i$ | `PropagationPsiState.hard_observe(i, sigma, clamp=True)` |
| 2.. | waves 1.., the **ripples**: each wave propagates only the change it newly induced | `PropagationPsiState.wave()` |

Each wave uses the centered hard-response kernels of the vendored runtime,
$K_{j\leftarrow i}(\cdot\mid s)=\phi_j(x_\varnothing, X_i=s)$, at response
scale 1 with simplex projection; clamped topics are never modified. A wave
with nothing pending is the identity, so $\Psi_0$ is exactly stationary. See
section 8 of the instructions and `tests/test_propagation_dynamics.py`.

Up to five waves run per answer (configurable up to 20). If change is still
pending after the last wave it is dropped and its size is reported, so one
answer's waves never leak into the next answer's.

The finite-$n$ sweeps of the vendored `CenteredLdpPsiState`
(`event="mode"` / `"sample"`) are still available as an **optional session
type** (New session → dynamics), labelled as a finite-sample / LDP
experiment. They inject a new perturbation at every variable on every sweep
and move $\Psi_0$ without any answer, so they are never the default.

All mathematics runs in native code; the browser only renders snapshots.

## What the screen shows

- **Live counter** — how many other topics the current answer has moved, how
  many by more than 0.01, and the mean shift (TV), updated wave by wave.
- **Field** — one dot per model column, laid out from the learned dependency
  graph (an edge $i\to j$ exists when the native tree for $j$ splits on $i$).
  Colour is each topic's change caused by the current answer (log scale, so
  small widespread shifts are visible); a toggle switches to change since
  $\Psi_0$. Amber dots are answered and clamped. During playback a halo marks
  each topic's change in that wave, links light up from the topics that
  changed in the previous wave, and topics reached for the first time are
  outlined. The layout is for readability only: screen distance is not a
  propagation time.
- **Wave strip** — Before / Splash / Wave k, with how many topics have been
  reached so far and how many were newly reached; replay.
- **Ψ change map** — one row per answer, one column per modelled topic (same
  order in every row), brightness = how far that answer moved the topic, plus
  a row for the total change since $\Psi_0$. Click a cell to inspect.
- **Largest shifts** — the topics this answer moved most, with the change in
  their most likely response.
- **Topic card** — question text, current distribution with the $\Psi_0$
  value as a tick, and a stacked bar of the topic's distribution after every
  answer, so topics that were never asked visibly drift.
- **Export** — JSON with the dynamics, seed, answers (with the uniform draw
  `u`), per-answer movement statistics and every native step summary.

### Known behaviour of the undamped waves

At response scale 1 without damping, the waves do not die out on GSS 2018:
the pending perturbation shrinks for about four waves and then grows
(~1.3x per wave) until simplex projection saturates it. Unasked topics can be
pushed to point masses (e.g. after `abany = yes`, `absingle` goes from 0.56
to 1.00). The UI reports the dropped remainder after each answer. Damping /
response scale is an open modelling decision.

## Run locally

From the repository root:

```bash
python3 -m pip install -r applications/psisimulation/requirements.txt -r webapp/requirements.txt

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target predict_distribution psisim_dynamics -j "$(nproc)"

python3 applications/psisimulation/fetch_model.py gss/gss_2018   # once
python3 webapp/server.py --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. Startup takes ~5–25 s (the first start also
computes and caches the graph layout). Creating a session builds a resident
native state (~3–5 s); one warm spare default state is kept ready so the next
session starts immediately. A wave takes 0.01–2 s on 4 cores (the first waves
of a session warm the kernel cache); frames stream as they finish.

Tests (need the bindings and the model):

```bash
python3 tests/test_propagation_dynamics.py   # dynamics invariants, ~3 min
cd webapp && python3 smoke_test.py           # engine end to end
```

## Deployment

| variable | default | meaning |
| --- | --- | --- |
| `DTAG_MODEL_ROOT` | `~/.cache/dtag/models` | where `gss/gss_2018` is installed; set to a persistent data directory |
| `PSISIM_NO_FETCH` | unset | `1` = never download at startup (fail if the model is missing) |
| `PSISIM_THREADS` | CPU count | native threads per call |
| `PSISIM_MAX_SESSIONS` | `4` | resident states kept in memory (LRU eviction of idle sessions) |
| `PSISIM_SESSION_TTL` | `1800` | idle seconds before a session is dropped |
| `PSISIM_WEBAPP_CACHE` | `~/.cache/psisim/webapp` | layout cache |
| `PSISIM_METADATA` | `webapp/assets/gss/gss_2018_map.csv` | variable metadata |
| `PSISIM_HOST`, `PORT` | `127.0.0.1`, `8000` | bind address |

Each resident session costs roughly 150 MB; budget memory as
`(PSISIM_MAX_SESSIONS + 1) × 150 MB` plus ~150 MB for the model. Run a single
worker process: sessions live in process memory. The answer endpoint streams
NDJSON, so disable response buffering in any reverse proxy (the server already
sends `X-Accel-Buffering: no` for nginx).

Download the model during deployment (`fetch_model.py`) rather than on the
first request; with `PSISIM_NO_FETCH=1` the server will refuse to start
without it.

## HTTP API

| method | path | body / result |
| --- | --- | --- |
| `GET` | `/api/model` | variables (column, name, label, categories, degrees), edges, layout, $\Psi_0$ |
| `POST` | `/api/session` | `{seed?, dynamics?: propagation\|mode\|sample}` → `{session_id, seed, dynamics, psi, observed, history}` |
| `GET` | `/api/session/{id}` | current `psi`, answers and per-answer change history |
| `POST` | `/api/session/{id}/answer` | `{column, max_steps?, empirical_n?}` → draws $\sigma\sim p_i$ and streams NDJSON `answer`, `frame`, `progress`, `done` events |
| `GET` | `/api/session/{id}/export` | reproducibility record |
| `DELETE` | `/api/session/{id}` | drop the resident state |

Distributions are sent as probability arrays aligned with each variable's
`categories`. A `frame` event carries the native step summary, per-column TV
versus the previous snapshot, and the full distributions of columns that
changed.

## Files

- `server.py` — FastAPI app and NDJSON streaming.
- `engine.py` — model loading, dependency graph, layout, resident sessions,
  propagation waves (default) and optional finite-n sweeps.
- `build_gss_metadata.py` — regenerates `assets/gss/gss_2018_map.csv`
  (see `assets/gss/PROVENANCE.md`).
- `static/` — the single-page client (no build step, no external assets).
- `../bindings/psisim_dynamics_py.cpp` — PsiSim-owned binding: the learned
  dependency graph and `PropagationPsiState`. It is not part of the vendored
  LSM runtime, so `scripts/sync_lsm_runtime.sh` never overwrites it.
- `../tests/test_propagation_dynamics.py` — regression tests of the default
  dynamics on the real model.

`qdistance` is not used yet; it is the intended tool for later comparisons of
endpoints, question orders and counterfactual branches.
