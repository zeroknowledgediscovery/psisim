# PsiSim webapp

An interactive view of the **GSS 2018 Large Science Model** as a reflexive
model of linked opinions, implementing [`webapp_instruction.md`](../webapp_instruction.md).

The state is $\Psi=(p_1,\ldots,p_d)$: one response distribution per modelled
GSS item (1,034 for GSS 2018). The app shows one thing: **asking a question
changes $\Psi$**, not only the answered item.

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

## What the screen shows

- **Question panel** — the asked question and its current distribution as a
  histogram. The draw is animated, then the histogram collapses: the drawn
  answer becomes a red bar at 1, the other answers drop to 0 and keep a grey
  outline of where they were. Next to it: how many of the other distributions
  this answer changed, how many by more than 0.01 (total variation), and the
  largest updates with the change in their most likely answer.
- **Ψ grid** — every $p_i$ as a small sparkline (a histogram with a fixed
  answer order and scale per item), in GSS column order so related items sit
  together, sized to fit one screen. Blue = current distribution, red =
  answered (all mass on the drawn answer). Hover to read one; click to ask it.
- **The update sequence** — after each answer, every changed sparkline
  pulses and keeps a gold outline. Then the largest updates (up to 8, TV ≥
  0.002) pop out of the grid one at a time: each grows into a readable card
  showing before (grey outline) and after (blue) for its main answers, then
  shrinks back into its sparkline. "Replay updates" repeats the sequence.
- **Asked so far** — every answer with how many distributions it changed.
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

Open <http://127.0.0.1:8000/>. Startup takes ~5–25 s (the first start also
computes and caches a graph layout used by the API). Creating a session
builds a resident native state (~3–5 s); one warm spare is kept ready so the
next session starts immediately. An answer's native update takes ~10 ms.

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
