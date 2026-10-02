# PsiSim webapp

An interactive view of the DTAG **Large Science Models** of social surveys as
reflexive models of linked opinions, implementing
[`webapp_instruction.md`](../webapp_instruction.md). It opens on GSS 2018 and
can switch to any of the 252 public models (GSS by year, Afrobarometer rounds,
Eurobarometer waves, WVS7) by choosing a country and year.

The state is $\Psi=(p_1,\ldots,p_d)$: one response distribution per item of
the selected survey (1,034 for GSS 2018, 337 for Afrobarometer R7, ...). The
app shows one thing: **asking a question changes $\Psi$**, not only the
answered item.

- It starts at the empty state $\Psi_0$, which does not move until a
  question is asked.
- Choosing a question draws an answer $\sigma\sim p_i$ from its current
  distribution (seeded, reproducible) and applies **one** native update,
  `PropagationPsiState.hard_observe(i, sigma, clamp=True)`: $p_i$ becomes the
  point mass $\delta_\sigma$ and the change $\delta_\sigma-p_i$ is passed once
  through the centered kernels
  $K_{j\leftarrow i}(\cdot\mid s)=\phi_j(x_\varnothing, X_i=s)$ to every item
  whose tree uses $i$. No further relaxation is run.
- The next question is asked in this updated $\Psi$.

All mathematics runs in native code; the browser only renders snapshots.

## Choosing a survey

The survey bar under the header picks a model by **country** and **year**, the
same way DTAG's recommender does:

- GSS (United States) by survey year; Afrobarometer rounds and Eurobarometer
  waves for the countries in their own country variable, by fieldwork period;
  WVS7 (pooled, 2017–2023) for countries it covers by coordinates.
- One best model per survey family is offered, ranked by time fit (within a
  year, within three years, further), then strength of geographic coverage,
  then recency. Eurobarometer cumulative/trend files are offered last.
- **Load survey** downloads the model from the public DTAG release on GCS if
  it is not on disk (SHA256-verified, with progress), computes its own
  $\Psi_0$, and starts a new session there. No persona is applied: every
  survey starts at its own empty state.

Afrobarometer, Eurobarometer and WVS7 models pool many countries. Their
$\Psi_0$ is the pooled empty state; the country is one of the survey's own
questions, which can be asked like any other. Item labels come from DTAG's
semantic maps (see `assets/PROVENANCE.md`); items without one show their
variable name. Item counts and answer categories always come from the model
itself.

Items with many answer categories (Afrobarometer party or language lists have
hundreds) are handled by evaluating the centred response as two soft-evidence
mixtures instead of one kernel per category; this is exact (see
`bindings/psisim_dynamics_py.cpp` and `tests/test_propagation_dynamics.py`)
and keeps every answer to about a second.

## What the screen shows

- **Question panel** — the asked question and its current distribution as a
  histogram. The draw is animated, then the histogram collapses: the drawn
  answer becomes a red bar at 1, the other answers drop to 0 and keep a grey
  outline of where they were. Next to it: how many of the other distributions
  this answer changed, how many by more than 0.01 (total variation), and the
  largest updates with the change in their most likely answer.
- **Ψ grid** — every $p_i$ as a small sparkline (a histogram with a fixed
  answer order and scale per item), in GSS column order so related items sit
  together, sized to fit one screen. Colour encodes how far each
  distribution now is from $\Psi_0$ (one blue lightness ramp, log scale:
  dim = unchanged, bright = moved); red = answered. Hover to read one; click
  to ask it.
- **The update sequence** — after each answer, the changed sparklines morph
  from their old to their new shape in a fast cascade (largest change first)
  with a glow, recolouring as they move, and keep a gold outline. Then the
  largest updates (up to 6, TV ≥ 0.002) pop out one after another as
  overlapping before/after cards whose bars slide from the old to the new
  values. "Replay updates" repeats it.
- **Current mind** — a donut with one spoke per question in survey order:
  spoke length and colour = distance of that distribution from $\Psi_0$;
  answered questions are red. The centre counts how many views have shifted
  by more than 0.001, and the answers given so far are listed below it.
  Hover a spoke to identify it; click to ask it.
- **Ask next** — clickable unasked questions, ranked by how far the answers
  so far have moved their distributions from $\Psi_0$ (suggested starters
  before the first answer), each with a coloured mini histogram. The search
  box finds any other question.
- **Export** — JSON with the seed, answers (with the uniform draw `u`),
  per-answer change statistics and the native step summaries.

### Further dynamics (API only)

`POST /api/session/{id}/answer` accepts `max_steps > 0` to add propagation
waves after the hard observation (each wave passes on only the change induced
by the previous wave; see section 8 of the instructions and
`tests/test_propagation_dynamics.py`), and a session can be created with
`dynamics: "mode" | "sample"` for the optional finite-$n$ sweeps of the
vendored `CenteredLdpPsiState` (a finite-sample / LDP experiment that moves
$\Psi_0$ without any answer). The UI does not use either. At response scale
1 without damping, the propagation waves do not die out on GSS 2018 (the
pending perturbation grows ~1.3x per wave after about four waves); this is
an open modelling question.

## Run locally

From the repository root:

```bash
python3 -m pip install -r applications/psisimulation/requirements.txt -r webapp/requirements.txt

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target predict_distribution psisim_dynamics -j "$(nproc)"

python3 applications/psisimulation/fetch_model.py gss/gss_2018   # once
python3 webapp/server.py --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. Startup takes ~5 s (GSS 2018 is loaded and
its $\Psi_0$ computed). Creating a session builds a resident native state
(~3–5 s); one warm spare GSS 2018 state is kept ready so a new session starts
immediately. Loading another survey takes a few seconds plus the download the
first time (2–60 MB per model). An answer's native update takes 0.01–1 s.

Tests (need the bindings and the model):

```bash
python3 tests/test_propagation_dynamics.py   # dynamics invariants, ~3 min
cd webapp && python3 smoke_test.py           # engine end to end
```

## Deployment

| variable | default | meaning |
| --- | --- | --- |
| `DTAG_MODEL_ROOT` | `~/.cache/dtag/models` | where models are installed (downloaded on demand); set to a persistent data directory |
| `PSISIM_NO_FETCH` | unset | `1` = never download; only models already under `DTAG_MODEL_ROOT` can be used |
| `PSISIM_MODEL` | `gss/gss_2018` | model loaded at startup and offered first |
| `PSISIM_MAX_MODELS` | `3` | loaded models kept in memory (the startup model is never evicted) |
| `PSISIM_THREADS` | CPU count | native threads per call |
| `PSISIM_MAX_SESSIONS` | `4` | resident states kept in memory (LRU eviction of idle sessions) |
| `PSISIM_SESSION_TTL` | `1800` | idle seconds before a session is dropped |
| `PSISIM_HOST`, `PORT` | `127.0.0.1`, `8000` | bind address |

Each resident GSS 2018 session costs roughly 150 MB (other surveys vary with
model size); budget memory as `(PSISIM_MAX_SESSIONS + 1) × 150 MB` plus the
loaded models. Run a single
worker process: sessions live in process memory. The answer endpoint streams
NDJSON, so disable response buffering in any reverse proxy (the server already
sends `X-Accel-Buffering: no` for nginx).

Pre-download the models you expect to use during deployment
(`python3 applications/psisimulation/fetch_model.py <key>`) so the first
visitor does not wait for the download; with `PSISIM_NO_FETCH=1` the server
refuses to start without the startup model.

## HTTP API

| method | path | body / result |
| --- | --- | --- |
| `GET` | `/api/catalog` | every public model (family, label, period, countries, size, installed/loaded) and the country index |
| `POST` | `/api/models/{family}/{name}/load` | start downloading (if needed) and loading a model in the background |
| `GET` | `/api/models/{family}/{name}/status` | `remote`, `installed`, `queued`, `downloading` (with progress), `loading`, `ready` or `error` |
| `GET` | `/api/model?key=…` | variables (column, name, label, categories, degrees) and $\Psi_0$ of a loaded model |
| `POST` | `/api/session` | `{model?, seed?, dynamics?: propagation\|mode\|sample}` → `{session_id, model, seed, dynamics, psi, observed, history}` |
| `GET` | `/api/session/{id}` | current `psi`, answers and per-answer change history |
| `POST` | `/api/session/{id}/answer` | `{column, max_steps? (default 0), empirical_n?}` → draws $\sigma\sim p_i$ and streams NDJSON `answer`, `frame`, `progress`, `done` events |
| `GET` | `/api/session/{id}/export` | reproducibility record |
| `DELETE` | `/api/session/{id}` | drop the resident state |

Distributions are sent as probability arrays aligned with each variable's
`categories`. A `frame` event carries the native step summary, per-column TV
versus the previous snapshot, and the full distributions of columns that
changed.

## Files

- `server.py` — FastAPI app and NDJSON streaming.
- `engine.py` — model loading, metadata, resident sessions, the one-update
  answer path, and the optional waves / finite-n sweeps.
- `build_catalog.py` — regenerates `assets/catalog.json` (models, periods,
  countries) and `assets/metadata/` (item labels) from a DTAG checkout (see
  `assets/PROVENANCE.md`).
- `static/` — the single-page client (no build step, no external assets).
- `../bindings/psisim_dynamics_py.cpp` — PsiSim-owned binding: the learned
  dependency graph and `PropagationPsiState`. It is not part of the vendored
  LSM runtime, so `scripts/sync_lsm_runtime.sh` never overwrites it.
- `../tests/test_propagation_dynamics.py` — regression tests of the default
  dynamics on the real model.

`qdistance` is not used yet; it is the intended tool for later comparisons of
endpoints, question orders and counterfactual branches.
