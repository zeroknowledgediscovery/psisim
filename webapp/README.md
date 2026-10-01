# PsiSim webapp

An interactive view of the **GSS 2018 Large Science Model** as a reflexive
model of linked opinions, implementing [`webapp_instruction.md`](../webapp_instruction.md).

The state is the full distributional row
$\Psi=(p_1,\ldots,p_d)$, one categorical response distribution per modelled
GSS item. Answering an item drops a stone into the system:

| frame | what is shown | native call |
| --- | --- | --- |
| 0 | state immediately before the answer | — |
| 1 | the **splash**: immediate centered hard response | `state.hard_observe(source, value, response_scale=1.0, clamp=True)` |
| 2.. | the **ripples**: relaxation sweeps | `state.sweep(n=10, event="mode", seed=session_seed, response_scale=1.0, random_permutation=False)` |

Up to five sweeps are run per answer (configurable up to ten), stopping early
when the native `mean_tv` falls below a tolerance (default `1e-3`). The next
question is asked in the new state, so the trajectory is history dependent.

All mathematics runs in the vendored native runtime. The browser only renders
snapshots streamed by the server; it never recomputes PsiSim dynamics.

## What the screen shows

- **Field** — one dot per model column, laid out from the learned dependency
  graph (an edge $i\to j$ exists when the native tree for $j$ splits on $i$).
  Colour is the total-variation distance of each item's current distribution
  from $\Psi_0$; amber dots are answered and clamped. During playback, a halo
  marks how much each item changed in that frame (native snapshot TV), the
  answered item's learned links light up on the splash, and decorative rings
  fade with the frame's `mean_tv`. The layout is for readability only: screen
  distance is not a propagation time.
- **Search** — by variable name, label, question text or answer text.
- **Question card** — question text, variable name, native column, current
  $p_i$ with every legal category (from the native source map), the $\Psi_0$
  value as a tick, and whether the item is clamped.
- **Response modes**
  - *Simulate response* (default): $\sigma\sim p_i$ drawn on the server with a
    per-session seeded `numpy` generator (reproducible from the session seed;
    the uniform draw `u` is shown and exported).
  - *Most likely*: deterministic MAP, $\sigma=\arg\max_s p_i(s)$.
  - *Choose response*: force an answer (intervention).
  All three use the same `hard_observe(..., clamp=True)`.
- **Timeline** — Before / Splash / Sweep k frames with `mean_tv`, replay, and
  the largest movers of each frame.
- **Export** — JSON with the seed, answers, every native step summary and the
  native `hard_row()`.

### A caveat the UI states explicitly

$\Psi_0$ is not stationary under the finite-$n$ (`n=10`) mode sweeps: sweeping
from $\Psi_0$ **without any answer** moves the state by mean TV ≈ 0.11, 0.045,
0.024 over the first three sweeps — the same magnitudes seen after a first
answer. The relaxation frames therefore mix the answer's response with the
model's intrinsic relaxation; the splash frame is the answer's direct effect.
The timeline shows a note to that effect on every relaxation frame.

## Run locally

From the repository root:

```bash
python3 -m pip install -r applications/psisimulation/requirements.txt -r webapp/requirements.txt

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target predict_distribution psisim_graph -j "$(nproc)"

python3 applications/psisimulation/fetch_model.py gss/gss_2018   # once
python3 webapp/server.py --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. Startup takes ~5–25 s (the first start also
computes and caches the graph layout). Creating a session builds a resident
`CenteredLdpPsiState` (~5 s); one warm spare is kept ready so the next session
starts immediately. A first sweep in a session takes ~9 s while the response
kernel cache warms, later sweeps ~3 s on 4 cores; frames stream as they finish.

Smoke test (needs the bindings and the model):

```bash
cd webapp && python3 smoke_test.py
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
| `POST` | `/api/session` | `{seed?}` → `{session_id, seed, psi, observed}` |
| `GET` | `/api/session/{id}` | current `psi` and answers |
| `POST` | `/api/session/{id}/answer` | `{column, mode: sample\|map\|choose, value?, max_sweeps?, tol?, empirical_n?}` → NDJSON stream of `answer`, `frame`, `progress`, `done` events |
| `GET` | `/api/session/{id}/export` | reproducibility record |
| `DELETE` | `/api/session/{id}` | drop the resident state |

Distributions are sent as probability arrays aligned with each variable's
`categories`. A `frame` event carries the native step summary, per-column TV
versus the previous snapshot, and the full distributions of columns that
changed.

## Files

- `server.py` — FastAPI app and NDJSON streaming.
- `engine.py` — model loading, dependency graph, layout, resident sessions.
- `build_gss_metadata.py` — regenerates `assets/gss/gss_2018_map.csv`
  (see `assets/gss/PROVENANCE.md`).
- `static/` — the single-page client (no build step, no external assets).
- `../bindings/psisim_graph_py.cpp` — PsiSim-owned binding exposing which
  columns each native tree uses. It is not part of the vendored LSM runtime,
  so `scripts/sync_lsm_runtime.sh` never overwrites it.

`qdistance` is not used yet; it is the intended tool for later comparisons of
endpoints, question orders and counterfactual branches.
