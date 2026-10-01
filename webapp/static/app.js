/* PsiSim webapp client.
 *
 * The browser never computes PsiSim dynamics. It renders snapshots streamed
 * by the server from the native runtime. With the default propagation
 * dynamics, frame 0 is the state before an answer, frame 1 the hard
 * observation (wave 0, "splash") and frames 2.. the later waves ("ripples"),
 * each of which propagates only the change induced by the previous wave.
 * Without an answer nothing moves.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (x, d = 3) => (x == null || !isFinite(x) ? "—" : Number(x).toFixed(d));
const fmtTv = (x) => (x == null || !isFinite(x) ? "—" : x === 0 ? "0" : x < 1e-3 ? x.toExponential(1) : x.toFixed(3));
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};

const SUGGESTED = [
  "polviews", "partyid", "abany", "cappun", "grass", "gunlaw", "god",
  "homosex", "natenvir", "fefam", "trust", "happy", "attend", "wrkstat",
];
const EPS = 1e-9;
// "Moved" threshold, shared with the server: halfway between 1e-6 and the next
// value on the 1e-6-rounded grid the probabilities are streamed on.
const MOVED = 1.25e-6;
const CAT_COLORS = ["#4cc9f0", "#f4a261", "#b59cff", "#7bd389", "#ff7a9c", "#ffd166"];
const OTHER_COLOR = "#4a5878";

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

const S = {
  model: null,
  vars: [],
  n: 0,
  learnedIdx: [],
  pos: null,          // Float32Array [x0,y0,...]
  edges: null,        // Int32Array [s0,t0,...]
  outAdj: [],
  inAdj: [],
  psi0: [],
  psi: [],            // current Psi (arrays aligned to categories)
  sid: null,
  seed: null,
  dynamics: "propagation",
  observed: [],       // [{column, variable, value, question, u}]
  answers: [],        // completed answers: {question, column, variable, value, u, tv, psi, stats, steps, stop, residual}
  q: null,            // answer being shown: {column, value, u, before, frames:[...], ...}
  viewFrame: -1,
  follow: true,
  selected: null,
  hover: null,
  busy: false,
  colorMode: "answer",
  settings: { max_steps: 5, empirical_n: 10 },
  search: "",
};

// ---------------------------------------------------------------------------
// display helpers (no dynamics here)
// ---------------------------------------------------------------------------

function tv(a, b) {
  if (a === b) return 0;
  let s = 0;
  for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
  return 0.5 * s;
}
function argmax(arr) {
  let k = 0;
  for (let j = 1; j < arr.length; j++) if (arr[j] > arr[k]) k = j;
  return k;
}
function isFiniteN() { return S.dynamics !== "propagation"; }

// Log colour scale: 1e-4 -> 0, 1e-1 -> 1; below 1e-6 = "no change".
const RAMP = [[40, 78, 132], [47, 127, 176], [76, 201, 240], [233, 251, 255]];
const BASE = "rgb(30,44,72)";
function level(x) {
  if (!(x > 1e-6)) return -1;
  return Math.max(0, Math.min(1, (Math.log10(x) + 4) / 3));
}
function rampColor(x) {
  const l = level(x);
  if (l < 0) return BASE;
  const t = l * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t));
  const f = t - i, a = RAMP[i], b = RAMP[i + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res;
}

async function loadModel() {
  const m = await (await api("/api/model")).json();
  S.model = m;
  S.vars = m.variables;
  S.n = m.width;
  S.pos = Float32Array.from(m.layout);
  S.edges = Int32Array.from(m.edges);
  S.psi0 = m.psi0;
  S.outAdj = Array.from({ length: S.n }, () => []);
  S.inAdj = Array.from({ length: S.n }, () => []);
  for (let e = 0; e < S.edges.length; e += 2) {
    S.outAdj[S.edges[e]].push(S.edges[e + 1]);
    S.inAdj[S.edges[e + 1]].push(S.edges[e]);
  }
  S.settings.max_steps = m.defaults.max_steps;
  S.settings.empirical_n = m.defaults.empirical_n;
  for (const v of S.vars) {
    v._hay = [v.variable, v.label, v.question_text].join(" ").toLowerCase();
    v._cats = v.categories.map((c) => c.toLowerCase());
  }
  // Change-map column order: learned items around the pond (angle, radius).
  S.learnedIdx = S.vars.filter((v) => v.learned).map((v) => v.column);
  const key = (i) => {
    const x = S.pos[2 * i], y = S.pos[2 * i + 1];
    return [Math.atan2(y, x), Math.hypot(x, y)];
  };
  S.learnedIdx.sort((a, b) => {
    const ka = key(a), kb = key(b);
    return Math.round(ka[0] * 12) - Math.round(kb[0] * 12) || ka[1] - kb[1];
  });
}

function applySession(view) {
  S.sid = view.session_id;
  S.seed = view.seed;
  S.dynamics = view.dynamics || "propagation";
  S.psi = view.psi;
  S.observed = view.observed || [];
  S.answers = (view.history || []).map((h) => ({
    ...h, u: (S.observed.find((o) => o.question === h.question) || {}).u,
    tv: Float32Array.from(h.tv), residual: h.residual_pending_tv,
  }));
  S.q = null;
  S.viewFrame = -1;
  store.set("psisim.sid", S.sid);
  const dyn = S.model.dynamics[S.dynamics];
  $("session-chip").textContent = `seed ${S.seed} · ${S.dynamics === "propagation" ? "propagation waves" : `finite-n ${S.dynamics} (LDP)`}`;
  $("session-chip").title = dyn ? `${dyn.label}: ${dyn.help}` : "";
}

async function newSession(seed, dynamics) {
  showLoading("Building a resident native state at Ψ₀…");
  try {
    const body = JSON.stringify({ ...(seed != null ? { seed } : {}), dynamics: dynamics || "propagation" });
    const view = await (await api("/api/session", { method: "POST", body })).json();
    applySession(view);
  } finally {
    hideLoading();
  }
  renderAll();
}

async function restoreOrCreate() {
  const sid = store.get("psisim.sid");
  if (sid) {
    try {
      const view = await (await api(`/api/session/${encodeURIComponent(sid)}`)).json();
      applySession(view);
      renderAll();
      return;
    } catch { store.del("psisim.sid"); }
  }
  await newSession();
}

// ---------------------------------------------------------------------------
// asking a topic: draw sigma ~ p_i on the server, stream the waves
// ---------------------------------------------------------------------------

function askTopic(col) {
  const v = S.vars[col];
  if (!v || !v.learned) return;
  select(col);
  if (observedRecord(col)) return;               // already answered: inspect only
  if (S.busy) { flashHeadline("Wait for the current answer's waves to finish."); return; }
  ask(col);
}

async function ask(column) {
  if (S.busy || !S.sid) return;
  S.busy = true;
  setError("");
  const before = S.psi;
  const observed0 = new Set(S.observed.map((o) => o.column));
  S.q = {
    column, value: null, u: null, question: S.observed.length + 1, before,
    frames: [{ index: 0, phase: "before", step: 0, summary: null,
               tv: new Float32Array(S.n), cum: new Float32Array(S.n),
               front: new Uint8Array(S.n), psi: before, observed: observed0,
               reached: 0, moved001: 0, meanCum: 0 }],
    pending: null, done: false, stop: null, revealed: false, residual: 0,
  };
  S.follow = true;
  S.viewFrame = 0;
  renderTimeline();
  renderCard();
  renderHeadline();

  let showChain = Promise.resolve();
  const enqueueShow = (k, dwell) => {
    showChain = showChain.then(async () => {
      if (S.follow) showFrame(k);
      await sleep(reduceMotion ? 150 : dwell);
    });
  };

  try {
    const res = await api(`/api/session/${encodeURIComponent(S.sid)}/answer`, {
      method: "POST",
      body: JSON.stringify({ column, max_steps: S.settings.max_steps, empirical_n: S.settings.empirical_n }),
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value: chunk, done } = await reader.read();
      if (done) break;
      buf += dec.decode(chunk, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "error") throw new Error(ev.detail);
        if (ev.type === "answer") {
          S.q.value = ev.value;
          S.q.u = ev.u;
          S.q.question = ev.question;
          const reveal = revealAnswer(ev);
          showChain = showChain.then(() => reveal).then(() => {
            S.q.revealed = true;
            renderTimeline();
            const st = document.querySelector("#card .status");
            if (st && S.selected === column) st.textContent = `Answered “${ev.value}” · waves spreading…`;
          });
        } else if (ev.type === "progress") {
          S.q.pending = ev;
          if (S.q.revealed) renderTimeline();
        } else if (ev.type === "frame") {
          const frame = buildFrame(ev, column);
          S.q.frames.push(frame);
          S.q.pending = null;
          S.psi = frame.psi;
          if (ev.phase === "hard_observation") {
            const rec = { column, variable: S.vars[column].variable, value: S.q.value,
                          question: S.q.question, u: S.q.u };
            // Publish the answer only after the draw animation has revealed it.
            showChain = showChain.then(() => {
              S.observed = S.observed.concat([rec]);
              renderHistory();
              renderResults();
            });
          }
          if (S.q.revealed) renderTimeline();
          enqueueShow(S.q.frames.length - 1, ev.phase === "hard_observation" ? 1600 : 1250);
        } else if (ev.type === "done") {
          S.q.done = true;
          S.q.stop = ev.stop;
          S.q.residual = ev.residual_pending_tv;
          const last = S.q.frames[S.q.frames.length - 1];
          const entry = {
            question: ev.question, column, variable: S.vars[column].variable,
            value: S.q.value, u: S.q.u, tv: last.cum, psi: last.psi,
            stats: ev.stats, steps: ev.steps, stop: ev.stop, residual: ev.residual_pending_tv,
          };
          showChain = showChain.then(() => {
            S.answers = S.answers.concat([entry]);
            renderMap();
          });
        }
      }
    }
    await showChain;
  } catch (err) {
    setError(err.message || String(err));
    if (S.q && S.q.frames.length <= 1) S.q = null;
  } finally {
    S.busy = false;
    if (S.q) { S.q.pending = null; S.q.done = true; S.q.revealed = true; }
    renderTimeline();
    renderCard();
    renderHeadline();
    renderMap();
  }
}

function buildFrame(ev, column) {
  const prev = S.q.frames[S.q.frames.length - 1];
  const psi = prev.psi.slice();
  const cum = Float32Array.from(prev.cum);
  const front = new Uint8Array(S.n);
  for (const [c, arr] of Object.entries(ev.changed)) {
    const i = +c;
    psi[i] = arr;
    const was = cum[i];
    cum[i] = tv(arr, S.q.before[i]);
    if (!(was > MOVED) && cum[i] > MOVED && i !== column) front[i] = 1;
  }
  const observed = new Set(prev.observed);
  observed.add(column);
  let reached = 0, moved001 = 0, sum = 0;
  for (const i of S.learnedIdx) {
    if (observed.has(i)) continue;
    const x = cum[i];
    if (x > MOVED) reached++;
    if (x > 0.01) moved001++;
    sum += x;
  }
  return {
    index: ev.frame, phase: ev.phase, step: ev.step, summary: ev.summary, seconds: ev.seconds,
    tv: Float32Array.from(ev.tv), cum, front, psi, observed,
    reached, moved001, meanCum: sum / Math.max(1, S.learnedIdx.length - observed.size),
    newly: front.reduce((a, b) => a + b, 0),
  };
}

// The draw animation only replays a draw the server already made.
async function revealAnswer(ev) {
  renderCard();
  const list = document.querySelector("#card .dist");
  if (!list || reduceMotion) { markWinner(ev.value); return; }
  const items = [...list.querySelectorAll("li")];
  const weights = items.map((li) => +li.dataset.p || 0);
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  let delay = 50;
  const t0 = performance.now();
  while (performance.now() - t0 < 1100) {
    let u = Math.random() * total, k = 0;
    while (k < weights.length - 1 && (u -= weights[k]) > 0) k++;
    items.forEach((li, j) => li.classList.toggle("spin", j === k));
    await sleep(delay);
    delay *= 1.13;
  }
  items.forEach((li) => li.classList.remove("spin"));
  markWinner(ev.value);
  await sleep(300);
}

function markWinner(value) {
  const list = document.querySelector("#card .dist");
  if (list) for (const li of list.querySelectorAll("li")) li.classList.toggle("winner", li.dataset.cat === value);
  const note = document.querySelector("#card .draw-note");
  if (note && S.q) {
    note.innerHTML = `Answer drawn from p<sub>i</sub>: <b>${esc(value)}</b> ` +
      `<span class="mono muted">(u = ${fmt(S.q.u, 4)}, seed ${S.seed})</span>`;
  }
}

// ---------------------------------------------------------------------------
// frame display
// ---------------------------------------------------------------------------

const anim = { flashT0: -1e9, ringT0: -1e9, amp: 0, source: null, raf: 0 };

function currentFrame() {
  if (!S.q || S.viewFrame < 0) return null;
  return S.q.frames[S.viewFrame] || null;
}
function displayedPsi() { const f = currentFrame(); return f ? f.psi : S.psi; }
function displayedObserved() {
  const f = currentFrame();
  return f ? f.observed : new Set(S.observed.map((o) => o.column));
}
// Change caused by the answer on screen (cumulative over its waves so far).
function answerChange() {
  const f = currentFrame();
  if (f) return f.cum;
  const last = S.answers[S.answers.length - 1];
  return last ? last.tv : null;
}

function showFrame(k) {
  if (!S.q || !S.q.frames[k]) return;
  S.viewFrame = k;
  const f = S.q.frames[k];
  anim.source = S.q.column;
  if (k > 0) {
    const now = performance.now();
    anim.flashT0 = now;
    anim.ringT0 = now;
    anim.amp = f.phase === "hard_observation" ? 1
      : Math.min(1, Math.sqrt((f.summary?.mean_tv || 0) / 0.004));
  } else {
    anim.amp = 0;
  }
  computeColors();
  renderTimeline();
  renderMovers();
  renderHeadline();
  renderMap();
  kick();
}

// ---------------------------------------------------------------------------
// canvas field
// ---------------------------------------------------------------------------

const canvas = $("field");
const ctx = canvas.getContext("2d");
const view = { scale: 1, tx: 0, ty: 0, w: 0, h: 0, dpr: 1, base: 1 };
let edgeLayer = null;
let edgeLayerKey = "";
let colorVal = new Float32Array(0);
let screen = new Float32Array(0);
let radius = new Float32Array(0);

function computeColors() {
  colorVal = new Float32Array(S.n);
  if (S.colorMode === "psi0") {
    const psi = displayedPsi();
    for (let i = 0; i < S.n; i++) colorVal[i] = tv(psi[i], S.psi0[i]);
  } else {
    const c = answerChange();
    if (c) colorVal.set(c);
  }
}

function resize() {
  const r = canvas.getBoundingClientRect();
  view.dpr = Math.min(window.devicePixelRatio || 1, 2);
  view.w = r.width; view.h = r.height;
  canvas.width = Math.round(r.width * view.dpr);
  canvas.height = Math.round(r.height * view.dpr);
  view.base = Math.min(view.w, view.h) / 2.45;
  edgeLayerKey = "";
  kick();
}

function project() {
  if (screen.length !== S.n * 2) screen = new Float32Array(S.n * 2);
  const s = view.base * view.scale;
  const cx = view.w / 2 + view.tx, cy = view.h / 2 + view.ty;
  for (let i = 0; i < S.n; i++) {
    screen[2 * i] = cx + S.pos[2 * i] * s;
    screen[2 * i + 1] = cy + S.pos[2 * i + 1] * s;
  }
  if (radius.length !== S.n) {
    radius = new Float32Array(S.n);
    for (let i = 0; i < S.n; i++) radius[i] = 1.7 + Math.sqrt(S.outAdj[i].length + S.inAdj[i].length) * 0.32;
  }
}

function drawEdgeLayer() {
  const key = `${view.w}x${view.h}@${view.scale.toFixed(4)},${view.tx.toFixed(1)},${view.ty.toFixed(1)}`;
  if (key === edgeLayerKey && edgeLayer) return;
  edgeLayerKey = key;
  if (!edgeLayer) edgeLayer = document.createElement("canvas");
  edgeLayer.width = canvas.width;
  edgeLayer.height = canvas.height;
  const g = edgeLayer.getContext("2d");
  g.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  g.clearRect(0, 0, view.w, view.h);
  g.strokeStyle = "rgba(120,170,230,0.04)";
  g.lineWidth = 0.6;
  g.beginPath();
  const E = S.edges;
  for (let e = 0; e < E.length; e += 2) {
    const a = E[e], b = E[e + 1];
    g.moveTo(screen[2 * a], screen[2 * a + 1]);
    g.lineTo(screen[2 * b], screen[2 * b + 1]);
  }
  g.stroke();
}

function drawLinks(g, from, list, color, alpha) {
  if (!list.length) return;
  g.strokeStyle = color;
  g.globalAlpha = alpha;
  g.lineWidth = 1;
  g.beginPath();
  for (const t of list) {
    g.moveTo(screen[2 * from], screen[2 * from + 1]);
    g.lineTo(screen[2 * t], screen[2 * t + 1]);
  }
  g.stroke();
  g.globalAlpha = 1;
}

function draw(now) {
  anim.raf = 0;
  if (!S.n) return;
  project();
  drawEdgeLayer();
  const g = ctx;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, canvas.width, canvas.height);
  g.drawImage(edgeLayer, 0, 0);
  g.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);

  const zr = nodeScale();
  const frame = currentFrame();
  const observed = displayedObserved();
  let animating = false;

  const focus = S.hover ?? S.selected;
  if (focus != null) {
    drawLinks(g, focus, S.inAdj[focus], "#b59cff", 0.35);
    drawLinks(g, focus, S.outAdj[focus], "#4cc9f0", 0.55);
  }

  const tFlash = (now - anim.flashT0) / 1700;
  // Wave sources light their outgoing links: wave 0 = the answered item;
  // later waves = items that changed in the previous wave.
  if (frame && frame.index > 0 && tFlash < 1) {
    const prev = S.q.frames[frame.index - 1];
    g.strokeStyle = "#e9fbff";
    g.lineWidth = 0.8;
    g.globalAlpha = (frame.phase === "hard_observation" ? 0.6 : 0.12) * (1 - tFlash);
    g.beginPath();
    const srcs = frame.phase === "hard_observation" ? [anim.source] : prevChanged(prev);
    for (const s of srcs) {
      for (const t of S.outAdj[s]) {
        if (frame.tv[t] <= EPS) continue;
        g.moveTo(screen[2 * s], screen[2 * s + 1]);
        g.lineTo(screen[2 * t], screen[2 * t + 1]);
      }
    }
    g.stroke();
    g.globalAlpha = 1;
    animating = true;
  }

  for (let i = 0; i < S.n; i++) {
    const x = screen[2 * i], y = screen[2 * i + 1];
    if (x < -20 || y < -20 || x > view.w + 20 || y > view.h + 20) continue;
    const r = radius[i] * zr;
    g.beginPath();
    g.arc(x, y, r, 0, 6.2832);
    if (observed.has(i)) {
      g.fillStyle = "#f4a261";
      g.fill();
      g.lineWidth = 1.5;
      g.strokeStyle = "rgba(255,255,255,0.85)";
      g.stroke();
    } else {
      g.fillStyle = S.vars[i].learned ? rampColor(colorVal[i]) : "#1a2236";
      g.fill();
    }
  }

  if (frame && frame.index > 0 && tFlash < 1) {
    const fade = Math.pow(1 - tFlash, 1.3);
    g.globalCompositeOperation = "lighter";
    for (let i = 0; i < S.n; i++) {
      const lv = level(frame.tv[i]);
      if (lv < 0) continue;
      const I = (0.25 + 0.75 * lv) * fade;
      const x = screen[2 * i], y = screen[2 * i + 1];
      const r = (radius[i] + 3 + 9 * I) * zr;
      const grd = g.createRadialGradient(x, y, 0, x, y, r);
      grd.addColorStop(0, `rgba(233,251,255,${0.8 * I})`);
      grd.addColorStop(0.4, `rgba(120,220,255,${0.35 * I})`);
      grd.addColorStop(1, "rgba(76,201,240,0)");
      g.fillStyle = grd;
      g.beginPath();
      g.arc(x, y, r, 0, 6.2832);
      g.fill();
    }
    g.globalCompositeOperation = "source-over";
    // wave front: items first reached by this answer in this wave
    g.strokeStyle = `rgba(255,214,102,${(0.9 * fade).toFixed(3)})`;
    g.lineWidth = 1.2;
    g.beginPath();
    for (let i = 0; i < S.n; i++) {
      if (!frame.front[i]) continue;
      const x = screen[2 * i], y = screen[2 * i + 1], r = radius[i] * zr + 3;
      g.moveTo(x + r, y);
      g.arc(x, y, r, 0, 6.2832);
    }
    g.stroke();
  }

  const tRing = (now - anim.ringT0) / 2100;
  if (anim.source != null && anim.amp > 0 && tRing < 1 && frame && frame.index > 0) {
    const x = screen[2 * anim.source], y = screen[2 * anim.source + 1];
    const R = Math.min(view.w, view.h) * 0.42;
    for (let k = 0; k < 3; k++) {
      const t = tRing - k * 0.13;
      if (t <= 0 || t >= 1) continue;
      g.beginPath();
      g.arc(x, y, 6 + R * (1 - Math.pow(1 - t, 2.2)), 0, 6.2832);
      g.strokeStyle = `rgba(165,236,255,${(0.5 * anim.amp * (1 - t)).toFixed(3)})`;
      g.lineWidth = 2.2 - k * 0.5;
      g.stroke();
    }
    animating = true;
  }

  if (S.selected != null) {
    const i = S.selected;
    g.beginPath();
    g.arc(screen[2 * i], screen[2 * i + 1], radius[i] * zr + 5, 0, 6.2832);
    g.strokeStyle = "#ffffff";
    g.lineWidth = 1.5;
    g.stroke();
  }

  if (animating && !reduceMotion) kick();
}

function prevChanged(prev) {
  if (prev._changed) return prev._changed;
  const out = [];
  for (let i = 0; i < S.n; i++) if (prev.tv[i] > EPS) out.push(i);
  prev._changed = out;
  return out;
}

function kick() { if (!anim.raf) anim.raf = requestAnimationFrame(draw); }

// Node size follows zoom and shrinks on small screens so the pond stays legible.
function nodeScale() {
  return Math.sqrt(view.scale) * Math.max(0.45, Math.min(1, view.base / 320));
}

function nodeAt(px, py) {
  let best = null, bd = 1e9;
  const zr = nodeScale();
  for (let i = 0; i < S.n; i++) {
    const dx = screen[2 * i] - px, dy = screen[2 * i + 1] - py;
    const d = dx * dx + dy * dy;
    const lim = Math.max(8, radius[i] * zr + 4);
    if (d < lim * lim && d < bd) { bd = d; best = i; }
  }
  return best;
}

function setupCanvas() {
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false };
  });
  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.moved) {
        canvas.classList.add("dragging");
        view.tx = drag.tx + dx; view.ty = drag.ty + dy;
        hideTip();
        kick();
        return;
      }
    }
    const i = nodeAt(px, py);
    if (i !== S.hover) { S.hover = i; kick(); }
    canvas.classList.toggle("hovering", i != null);
    if (i != null) showTip(i, px, py); else hideTip();
  });
  const end = (e) => {
    if (drag && !drag.moved) {
      const r = canvas.getBoundingClientRect();
      const i = nodeAt(e.clientX - r.left, e.clientY - r.top);
      if (i != null) select(i);
    }
    drag = null;
    canvas.classList.remove("dragging");
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", () => { drag = null; canvas.classList.remove("dragging"); });
  canvas.addEventListener("pointerleave", () => { S.hover = null; hideTip(); kick(); });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015));
  }, { passive: false });
  canvas.addEventListener("dblclick", resetView);
  $("zoom-in").onclick = () => zoomAt(view.w / 2, view.h / 2, 1.35);
  $("zoom-out").onclick = () => zoomAt(view.w / 2, view.h / 2, 1 / 1.35);
  $("zoom-reset").onclick = resetView;
  new ResizeObserver(resize).observe(canvas);
}

function zoomAt(px, py, f) {
  const ns = Math.max(0.5, Math.min(12, view.scale * f));
  f = ns / view.scale;
  const cx = view.w / 2 + view.tx, cy = view.h / 2 + view.ty;
  view.tx += (px - cx) * (1 - f);
  view.ty += (py - cy) * (1 - f);
  view.scale = ns;
  kick();
}
function resetView() { view.scale = 1; view.tx = 0; view.ty = 0; kick(); }

function showTip(i, px, py) {
  const v = S.vars[i];
  const psi = displayedPsi()[i];
  const k = argmax(psi);
  const f = currentFrame();
  const c = answerChange();
  const tip = $("tooltip");
  tip.innerHTML =
    `<div class="t-var">${esc(v.variable)} <span class="muted">· col ${i}</span></div>` +
    `<div class="t-label">${esc(v.label || "no question text in metadata")}</div>` +
    `<div class="t-row">most likely: ${esc(v.categories[k])} (${fmt(psi[k])})</div>` +
    (c ? `<div class="t-row">moved by this answer: TV ${fmtTv(c[i])}</div>` : "") +
    `<div class="t-row">moved since Ψ₀: TV ${fmtTv(tv(psi, S.psi0[i]))}</div>` +
    (f && f.index > 0 ? `<div class="t-row">change in this wave: TV ${fmtTv(f.tv[i])}</div>` : "") +
    (displayedObserved().has(i) ? `<div class="t-row" style="color:var(--observed)">answered · clamped</div>` : "") +
    `<div class="t-hint">click to inspect</div>`;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${Math.min(view.w - w - 8, px + 14)}px`;
  tip.style.top = `${Math.min(view.h - h - 8, py + 14)}px`;
}
function hideTip() { $("tooltip").hidden = true; }

// ---------------------------------------------------------------------------
// headline: how much of Psi this answer has moved
// ---------------------------------------------------------------------------

let headlineFlash = "";
function flashHeadline(msg) {
  headlineFlash = msg;
  renderHeadline();
  setTimeout(() => { headlineFlash = ""; renderHeadline(); }, 2200);
}

function renderHeadline() {
  const el = $("headline");
  const learned = S.learnedIdx.length;
  let html = "";
  if (headlineFlash) {
    html = `<div class="hl-flash">${esc(headlineFlash)}</div>`;
  } else if (!S.q && !S.answers.length) {
    html = `<div class="hl-title">No answers have been supplied.</div>
      <div class="hl-sub">Every node is showing the response distribution implied by the GSS 2018 LSM from the empty state.
      ${isFiniteN() ? "" : "Nothing moves until a topic is answered: Ψ₀ is stationary."}</div>`;
  } else {
    const f = currentFrame();
    let q, reached, m001, mean, label;
    if (f && S.q) {
      q = S.q;
      reached = f.reached; m001 = f.moved001; mean = f.meanCum;
      label = f.phase === "before" ? "before the answer"
        : f.phase === "hard_observation" ? "after the splash (wave 0)"
        : `after ${f.phase === "wave" ? "wave" : "sweep"} ${f.step}`;
    } else {
      const a = S.answers[S.answers.length - 1];
      q = a; reached = a.stats.moved; m001 = a.stats.moved_001; mean = a.stats.mean_tv; label = "after its waves";
    }
    const v = S.vars[q.column];
    const total = learned - (S.q ? displayedObserved().size : S.observed.length);
    const pct = total > 0 ? Math.round(100 * reached / total) : 0;
    html = `<div class="hl-q">Q${q.question} · <span class="mono">${esc(v.variable)}</span>` +
      (q.value != null && (!S.q || S.q.revealed || q !== S.q) ? ` = <b>${esc(q.value)}</b>` : " …") + `</div>
      <div class="hl-stats">
        <span class="hl-grp"><span class="hl-big">${reached}</span><span class="hl-of">of ${total} other topics moved (${pct}%)</span></span>
        <span class="hl-grp"><span class="hl-big small">${m001}</span><span class="hl-of">by more than 0.01</span></span>
        <span class="hl-grp"><span class="hl-big small">${fmtTv(mean)}</span><span class="hl-of">mean shift (TV)</span></span>
      </div>
      <div class="hl-sub">${label}${isFiniteN() ? " · finite-n LDP experiment: sweeps also move Ψ on their own" : ""}</div>`;
  }
  el.innerHTML = html;
}

// ---------------------------------------------------------------------------
// change map: answers x items
// ---------------------------------------------------------------------------

const mapCanvas = $("changemap");
const mapState = { rows: [], gutter: 150, rowH: 18, colW: 1, w: 0 };

function mapRows() {
  const rows = S.answers.map((a) => ({ kind: "answer", a, tv: a.tv }));
  if (S.q && S.busy && S.q.revealed) {
    const f = currentFrame() || S.q.frames[S.q.frames.length - 1];
    rows.push({ kind: "live", a: { question: S.q.question, column: S.q.column, value: S.q.value }, tv: f.cum });
  }
  if (rows.length) {
    const psi = displayedPsi();
    const cum = new Float32Array(S.n);
    for (let i = 0; i < S.n; i++) cum[i] = tv(psi[i], S.psi0[i]);
    rows.push({ kind: "total", tv: cum });
  }
  return rows;
}

function renderMap() {
  const rows = mapRows();
  mapState.rows = rows;
  $("map-empty").hidden = rows.length > 0;
  const wrap = $("map-wrap");
  const w = wrap.clientWidth;
  const h = rows.length ? rows.length * mapState.rowH + 6 : 0;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  mapCanvas.style.height = `${h}px`;
  mapCanvas.width = Math.round(w * dpr);
  mapCanvas.height = Math.round(h * dpr);
  mapState.w = w;
  if (!rows.length) return;
  const g = mapCanvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const gut = w < 520 ? 92 : mapState.gutter;
  mapState.gutterNow = gut;
  const cols = S.learnedIdx;
  const colW = (w - gut - 4) / cols.length;
  mapState.colW = colW;
  const f = currentFrame();
  rows.forEach((row, r) => {
    const y = r * mapState.rowH + 2;
    const isLive = row.kind === "live";
    g.fillStyle = row.kind === "total" ? "#9fb3d1" : isLive ? "#ffd166" : "#c9d6ea";
    g.font = "11px ui-monospace, Menlo, monospace";
    g.textBaseline = "middle";
    const text = row.kind === "total" ? "all answers vs Ψ₀"
      : `Q${row.a.question} ${S.vars[row.a.column].variable}=${row.a.value ?? "…"}`;
    g.fillText(text.length > (gut / 6.6 | 0) ? text.slice(0, (gut / 6.6 | 0) - 1) + "…" : text, 4, y + mapState.rowH / 2 - 1);
    g.fillStyle = "#0b1426";
    g.fillRect(gut, y, w - gut - 4, mapState.rowH - 3);
    for (let k = 0; k < cols.length; k++) {
      const i = cols[k];
      const x = row.tv[i];
      if (!(x > 1e-6)) continue;
      g.fillStyle = rampColor(x);
      g.fillRect(gut + k * colW, y, Math.max(colW, 0.8), mapState.rowH - 3);
    }
    if (row.a) {
      const k = cols.indexOf(row.a.column);
      if (k >= 0) {
        g.fillStyle = "#f4a261";
        g.fillRect(gut + k * colW - 1, y, Math.max(colW, 2) + 1, mapState.rowH - 3);
      }
    }
    if (row.kind === "total") {
      g.strokeStyle = "#2a3b5e";
      g.beginPath(); g.moveTo(gut, y - 1.5); g.lineTo(w - 4, y - 1.5); g.stroke();
    }
  });
  if (S.selected != null) {
    const k = cols.indexOf(S.selected);
    if (k >= 0) {
      g.strokeStyle = "#ffffff";
      g.lineWidth = 1;
      g.strokeRect(gut + k * colW - 1.5, 0.5, Math.max(colW, 1) + 3, h - 2);
    }
  }
}

function mapHit(e) {
  const r = mapCanvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const row = mapState.rows[Math.floor((y - 2) / mapState.rowH)];
  const k = Math.floor((x - (mapState.gutterNow || mapState.gutter)) / mapState.colW);
  if (!row || k < 0 || k >= S.learnedIdx.length) return null;
  return { row, col: S.learnedIdx[k], x, y };
}

function setupMap() {
  mapCanvas.addEventListener("pointermove", (e) => {
    const hit = mapHit(e);
    const tip = $("map-tip");
    if (!hit) { tip.hidden = true; return; }
    const v = S.vars[hit.col];
    tip.innerHTML = `<span class="mono">${esc(v.variable)}</span> ${esc(v.label)}<br>` +
      `${hit.row.kind === "total" ? "moved since Ψ₀" : `moved by Q${hit.row.a.question}`}: TV ${fmtTv(hit.row.tv[hit.col])}`;
    tip.hidden = false;
    const wrap = $("map-wrap").getBoundingClientRect();
    tip.style.left = `${Math.min(wrap.width - 240, Math.max(0, e.clientX - wrap.left + 10))}px`;
    tip.style.top = `${e.clientY - wrap.top + 14}px`;
  });
  mapCanvas.addEventListener("pointerleave", () => { $("map-tip").hidden = true; });
  mapCanvas.addEventListener("click", (e) => { const hit = mapHit(e); if (hit) select(hit.col); });
  new ResizeObserver(() => renderMap()).observe($("map-wrap"));
}

// ---------------------------------------------------------------------------
// panels
// ---------------------------------------------------------------------------

function select(col) {
  S.selected = col;
  setError("");
  renderCard();
  renderResults();
  renderMap();
  kick();
}

function observedRecord(col) { return S.observed.find((o) => o.column === col) || null; }

function searchVars(q) {
  q = q.trim().toLowerCase();
  if (!q) {
    return SUGGESTED
      .map((name) => S.vars.find((v) => v.variable.toLowerCase() === name))
      .filter((v) => v && v.learned)
      .map((v) => ({ v, hit: "" }));
  }
  const out = [];
  for (const v of S.vars) {
    if (!v.learned) continue;
    const name = v.variable.toLowerCase();
    let score = 0, hit = "";
    if (name === q) score = 100;
    else if (name.startsWith(q)) score = 80;
    else if (v._hay.includes(q)) score = 60 - Math.min(20, v._hay.indexOf(q) / 4);
    else if (String(v.column) === q) score = 50;
    else {
      const j = v._cats.findIndex((c) => c.includes(q));
      if (j >= 0) { score = 30; hit = `answer: “${v.categories[j]}”`; }
    }
    if (score) out.push({ v, hit, score });
  }
  out.sort((a, b) => b.score - a.score || a.v.column - b.v.column);
  return out.slice(0, 60);
}

function renderResults() {
  const list = searchVars(S.search);
  $("results-head").textContent = S.search.trim()
    ? `${list.length}${list.length === 60 ? "+" : ""} matching topics`
    : "Suggested topics";
  $("results").innerHTML = list.length
    ? list.map(({ v, hit }) => {
        const done = observedRecord(v.column);
        return `<li><button data-col="${v.column}" aria-current="${S.selected === v.column}" title="${done ? "already answered — inspect" : "ask this topic"}">` +
          `<span class="r-var">${esc(v.variable)}${done ? `<span class="done">answered: ${esc(done.value)}</span>` : `<span class="ask">ask ›</span>`}</span>` +
          `<span class="r-label">${esc(v.label || "—")}</span>` +
          (hit ? `<span class="r-hit">${esc(hit)}</span>` : "") +
          `</button></li>`;
      }).join("")
    : `<li class="empty">No modelled topic matches.</li>`;
}

function renderHistory() {
  $("answer-count").textContent = S.observed.length;
  $("history").innerHTML = S.observed.length
    ? S.observed.map((o) => {
        const a = S.answers.find((x) => x.question === o.question);
        return `<li><button data-col="${o.column}"><div class="h-row">` +
          `<span class="h-q">Q${o.question}</span><span class="h-body">` +
          `<span class="mono">${esc(o.variable)}</span> = <span class="h-val">${esc(o.value)}</span>` +
          (a ? `<span class="h-stat">moved ${a.stats.moved} other topics · ${a.stats.moved_001} by &gt;0.01</span>` : "") +
          `</span></div></button></li>`;
      }).join("")
    : `<li class="empty">No answers yet.</li>`;
}

function frameName(f) {
  if (f.phase === "before") return "Before";
  if (f.phase === "hard_observation") return "Splash";
  return `${f.phase === "wave" ? "Wave" : "Sweep"} ${f.step}`;
}

function renderTimeline() {
  const q = S.q;
  if (!q) {
    $("tl-title").textContent = S.answers.length
      ? "Ask another topic to watch its waves."
      : "Waves appear here after the first answer.";
    $("frames").innerHTML = "";
    $("tl-status").innerHTML = "";
    $("btn-replay").disabled = true;
    renderMovers();
    return;
  }
  const v = S.vars[q.column];
  $("tl-title").innerHTML = `Q${q.question} · <span class="mono">${esc(v.variable)}</span>` +
    (q.value != null && q.revealed ? ` = <b style="color:var(--observed)">${esc(q.value)}</b>` : ` <span class="muted">…</span>`);

  const shown = q.revealed ? q.frames : q.frames.slice(0, 1);
  const maxTv = Math.max(1e-12, ...shown.slice(1).map((f) => f.summary?.mean_tv || 0));
  let html = shown.map((f, k) => {
    const m = f.summary ? f.summary.mean_tv : null;
    const w = m != null ? Math.max(3, 100 * m / maxTv) : 0;
    const meta = f.phase === "before" ? "Ψ before"
      : `${f.reached} reached${f.newly ? ` · +${f.newly}` : ""}`;
    return `<button data-frame="${k}" aria-current="${S.viewFrame === k}" title="${f.summary ? `mean TV ${fmtTv(m)} · max TV ${fmt(f.summary.max_tv, 4)}${f.summary.projected_count != null ? ` · projections ${f.summary.projected_count}` : ""}` : "state before the answer"}">` +
      `<span class="f-name">${frameName(f)}</span>` +
      `<span class="f-meta">${meta}</span>` +
      `<span class="f-bar"><i style="width:${w}%"></i></span></button>`;
  }).join("");
  if (q.pending && q.revealed) {
    html += `<button class="pending" disabled><span class="f-name">${isFiniteN() ? "Sweep" : "Wave"} ${q.pending.step}</span><span class="f-meta">computing…</span><span class="f-bar"></span></button>`;
  }
  $("frames").innerHTML = html;

  let status = "";
  if (!q.revealed) status = `<span class="spinner"></span>drawing an answer from p<sub>i</sub>`;
  else if (!q.done) status = `<span class="spinner"></span>${q.pending ? `computing ${isFiniteN() ? "sweep" : "wave"} ${q.pending.step}/${q.pending.of}` : "splash"}`;
  else if (S.busy) status = `<span class="spinner"></span>playing ${frameName(q.frames[Math.max(0, S.viewFrame)]).toLowerCase()} of ${q.frames.length - 1}`;
  else if (q.stop === "settled") status = "settled: no change left to propagate";
  else if (q.stop === "max_steps" && !isFiniteN()) status = `stopped after ${q.frames.length - 2} waves · undamped remainder (TV ${fmtTv(q.residual)}) dropped`;
  else if (q.stop) status = `${q.frames.length - 2} finite-n sweeps`;
  $("tl-status").innerHTML = status;
  $("btn-replay").disabled = S.busy || q.frames.length < 2;
  document.body.dataset.busy = S.busy ? "1" : "";
}

function renderMovers() {
  const f = currentFrame();
  const c = answerChange();
  if (!c) {
    $("movers-note").textContent = "";
    $("movers").innerHTML = `<li class="empty">Answer a topic to see which other topics shift.</li>`;
    return;
  }
  const before = S.q ? S.q.before : (S.answers.length > 1 ? S.answers[S.answers.length - 2].psi : S.psi0);
  const after = f ? f.psi : S.psi;
  const col = S.q ? S.q.column : S.answers[S.answers.length - 1].column;
  const idx = [];
  for (const i of S.learnedIdx) if (i !== col && c[i] > EPS) idx.push(i);
  idx.sort((a, b) => c[b] - c[a]);
  $("movers-note").textContent = `· caused by this answer${f && f.phase !== "before" ? ", so far" : ""}`;
  $("movers").innerHTML = idx.slice(0, 12).map((i) => {
    const v = S.vars[i];
    const a = argmax(before[i]), b = argmax(after[i]);
    const shift = a !== b
      ? `most likely: ${v.categories[a]} → ${v.categories[b]}`
      : `${v.categories[b]}: ${fmt(before[i][b])} → ${fmt(after[i][b])}`;
    return `<li><button data-col="${i}"><div class="m-row"><span class="m-name"><span class="mono">${esc(v.variable)}</span> ${esc(v.label)}</span>` +
      `<span class="m-tv">${fmtTv(c[i])}</span></div><span class="m-shift">${esc(shift)}</span></button></li>`;
  }).join("") || `<li class="empty">Nothing has moved yet.</li>`;
}

// One stacked bar per state: Psi0, then after each answer.
function itemTrajectory(col) {
  const v = S.vars[col];
  const p0 = S.psi0[col];
  const top = v.categories.map((c, k) => k).sort((a, b) => p0[b] - p0[a]).slice(0, CAT_COLORS.length);
  const colorOf = (k) => { const j = top.indexOf(k); return j >= 0 ? CAT_COLORS[j] : OTHER_COLOR; };
  const states = [{ label: "Ψ₀", p: p0 }].concat(S.answers.map((a) => ({
    label: `Q${a.question}`, p: a.psi[col], own: a.column === col,
    title: `after Q${a.question} ${S.vars[a.column].variable} = ${a.value}`,
  })));
  if (S.q && S.busy && S.q.revealed) {
    const f = currentFrame() || S.q.frames[S.q.frames.length - 1];
    states.push({ label: `Q${S.q.question}…`, p: f.psi[col], own: S.q.column === col, title: "current answer, in progress" });
  }
  const rows = states.map((s) => {
    const segs = v.categories.map((c, k) => s.p[k] > 0
      ? `<i style="width:${(100 * s.p[k]).toFixed(2)}%;background:${colorOf(k)}" title="${esc(c)}: ${fmt(s.p[k])}"></i>` : "").join("");
    return `<div class="traj-row${s.own ? " own" : ""}" title="${esc(s.title || "empty state")}"><span class="traj-l">${s.label}</span>` +
      `<span class="traj-bar">${segs}</span><span class="traj-tv">${s.label === "Ψ₀" ? "" : fmtTv(tv(s.p, p0))}</span></div>`;
  }).join("");
  const legend = top.map((k, j) => `<span><i style="background:${CAT_COLORS[j]}"></i>${esc(v.categories[k])}</span>`).join("") +
    (v.categories.length > top.length ? `<span><i style="background:${OTHER_COLOR}"></i>other</span>` : "");
  return `<div class="traj">${rows}</div><div class="traj-legend">${legend}</div>`;
}

function renderCard() {
  const card = $("card");
  if (S.selected == null) {
    const chips = SUGGESTED
      .map((n) => S.vars.find((v) => v.variable.toLowerCase() === n))
      .filter((v) => v && v.learned && !observedRecord(v.column))
      .slice(0, 9)
      .map((v) => `<button data-ask="${v.column}" title="${esc(v.label)}">${esc(v.variable)}</button>`).join("");
    card.innerHTML = `<div class="card-intro">
      <p class="lead">${S.observed.length
        ? "Choose the next topic. It is asked in the <b>current</b> state, which already carries every earlier answer."
        : "No answers have been supplied. Every node is showing the response distribution implied by the GSS 2018 LSM from the empty state."}</p>
      <p>Choosing a topic draws an answer from its current distribution and drops it into the system like a stone into water.
        Watch the waves spread: the counter shows how many other topics the answer has moved.</p>
      <p class="muted">Ask one of these:</p>
      <div class="suggest">${chips}</div>
    </div>`;
    return;
  }

  const col = S.selected;
  const v = S.vars[col];
  const active = S.busy && S.q && S.q.column === col;
  const p = active ? S.q.before[col] : S.psi[col];
  const p0 = S.psi0[col];
  const rec = observedRecord(col);
  const order = v.categories.map((c, k) => k).sort((a, b) => p[b] - p[a] || a - b);
  const rows = order.map((k) => {
    const c = v.categories[k];
    return `<li data-cat="${esc(c)}" data-p="${p[k]}" class="${rec && rec.value === c ? "winner" : ""}">` +
      `<label><span class="d-label" title="${esc(c)}">${esc(c)}</span><span class="d-p">${fmt(p[k])}</span>` +
      `<span class="d-bar"><i style="width:${(100 * p[k]).toFixed(2)}%"></i><b style="left:calc(${(100 * p0[k]).toFixed(2)}% - 1px)"></b></span></label></li>`;
  }).join("");

  const status = rec
    ? `<span class="status clamped">Answered in Q${rec.question}: “${esc(rec.value)}” · clamped</span>`
    : active ? `<span class="status">${S.q.revealed ? `Answered “${esc(S.q.value)}” · waves spreading…` : "Drawing an answer from the current distribution…"}</span>`
    : `<span class="status">Not asked yet · its distribution has been shaped by ${S.observed.length} answer${S.observed.length === 1 ? "" : "s"}</span>`;
  const button = !rec && v.learned
    ? `<button class="btn go" id="btn-ask" ${S.busy ? "disabled" : ""}>${S.busy ? "Waves running…" : "Ask this topic"}</button>` : "";

  card.innerHTML = `
    <div class="q-meta"><span class="q-var">${esc(v.variable)}</span><span class="q-col">native column ${col}</span></div>
    ${v.label ? `<div class="q-text">${esc(v.question_text || v.label)}</div>` :
      `<div class="q-text missing">No GSS question text is available for this variable in the bundled metadata.</div>`}
    <div class="q-links"><b>${S.outAdj[col].length}</b> topics respond directly to this one · it responds to <b>${S.inAdj[col].length}</b></div>
    ${status}
    <div class="dist-head"><span>${active ? "Distribution it is drawn from" : "Current distribution"}</span><span>p<sub>i</sub>(s)</span></div>
    <ul class="dist">${rows}</ul>
    <div class="dist-legend"><span><i></i>${active ? "before the answer" : "current Ψ"}</span><span><b></b>empty state Ψ₀</span></div>
    ${button}
    <div class="draw-note"></div>
    <div class="error" id="card-error" hidden></div>
    <div class="sec-title traj-title">This topic across answers <span class="muted">TV vs Ψ₀</span></div>
    ${itemTrajectory(col)}
    <details class="settings">
      <summary>Simulation settings</summary>
      <div class="set-grid">
        <label for="set-steps">${isFiniteN() ? "Sweeps" : "Waves"} per answer</label>
        <input id="set-steps" type="number" min="0" max="${S.model.defaults.max_steps_limit}" value="${S.settings.max_steps}">
        ${isFiniteN() ? `<label for="set-n">Empirical n</label>
        <input id="set-n" type="number" min="1" max="1000" value="${S.settings.empirical_n}">` : ""}
        <span class="set-note">${esc(S.model.dynamics[S.dynamics].label)}. Response scale 1.0, no damping. Seed ${S.seed}.</span>
      </div>
    </details>`;
  if (rec && S.q && S.q.column === col && S.q.value != null) markWinner(S.q.value);
}

function setError(msg) {
  const el = $("card-error");
  if (el) { el.hidden = !msg; el.textContent = msg; }
  else if (msg) flashHeadline(msg);
}

function renderAll() {
  computeColors();
  renderResults();
  renderHistory();
  renderTimeline();
  renderCard();
  renderHeadline();
  renderMap();
  kick();
}

function showLoading(text) { $("loading-text").textContent = text; $("loading").hidden = false; }
function hideLoading() { $("loading").hidden = true; }

// ---------------------------------------------------------------------------
// new-session dialog
// ---------------------------------------------------------------------------

function openNewDialog() {
  if (S.busy) { flashHeadline("Wait for the current answer's waves to finish."); return; }
  const dlg = $("new-dialog");
  $("dyn-options").innerHTML = Object.entries(S.model.dynamics).map(([key, d]) =>
    `<label class="dyn-opt${d.finite_n ? " finite" : ""}"><input type="radio" name="dyn" value="${key}" ${key === "propagation" ? "checked" : ""}>` +
    `<span><b>${esc(d.label)}</b><small>${esc(d.help)}</small></span></label>`).join("");
  $("new-seed").value = "";
  if (typeof dlg.showModal === "function") dlg.showModal(); else dlg.setAttribute("open", "");
}

async function submitNewDialog(e) {
  const dlg = $("new-dialog");
  if (e.submitter && e.submitter.value === "cancel") return;
  e.preventDefault();
  const raw = $("new-seed").value.trim();
  const seed = raw === "" ? null : Number(raw);
  if (seed != null && !(Number.isInteger(seed) && seed >= 0)) { $("new-seed").focus(); return; }
  const dyn = (document.querySelector('input[name="dyn"]:checked') || {}).value || "propagation";
  dlg.close();
  const old = S.sid;
  S.selected = null;
  try {
    await newSession(seed, dyn);
    if (old) api(`/api/session/${encodeURIComponent(old)}`, { method: "DELETE" }).catch(() => {});
  } catch (err) { flashHeadline(err.message); }
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

function wire() {
  $("search").addEventListener("input", (e) => { S.search = e.target.value; renderResults(); });
  $("search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const first = $("results").querySelector("button[data-col]");
      if (first) askTopic(+first.dataset.col);
    }
  });
  $("results").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-col]");
    if (b) askTopic(+b.dataset.col);
  });
  for (const id of ["history", "movers"]) {
    $(id).addEventListener("click", (e) => {
      const b = e.target.closest("button[data-col]");
      if (b) select(+b.dataset.col);
    });
  }
  $("frames").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-frame]");
    if (!b) return;
    S.follow = false;
    showFrame(+b.dataset.frame);
  });
  $("btn-replay").addEventListener("click", async () => {
    if (!S.q) return;
    S.follow = false;
    for (let k = 0; k < S.q.frames.length; k++) {
      showFrame(k);
      await sleep(reduceMotion ? 300 : k === 1 ? 1600 : 1250);
    }
  });
  document.querySelector(".color-mode").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-color]");
    if (!b) return;
    S.colorMode = b.dataset.color;
    for (const x of document.querySelectorAll(".color-mode button")) x.setAttribute("aria-pressed", String(x === b));
    computeColors();
    kick();
  });
  $("card").addEventListener("click", (e) => {
    const sug = e.target.closest(".suggest button[data-ask]");
    if (sug) { askTopic(+sug.dataset.ask); return; }
    if (e.target.closest("#btn-ask")) askTopic(S.selected);
  });
  $("card").addEventListener("change", (e) => {
    const num = (id, lo, hi, def) => {
      const x = Number($(id).value);
      return Number.isFinite(x) ? Math.max(lo, Math.min(hi, Math.round(x))) : def;
    };
    if (e.target.id === "set-steps") S.settings.max_steps = num("set-steps", 0, S.model.defaults.max_steps_limit, 5);
    if (e.target.id === "set-n") S.settings.empirical_n = num("set-n", 1, 1000, 10);
  });
  $("btn-new").addEventListener("click", openNewDialog);
  $("new-form").addEventListener("submit", submitNewDialog);
  $("btn-export").addEventListener("click", () => {
    if (S.sid) window.location.href = `/api/session/${encodeURIComponent(S.sid)}/export`;
  });
  $("btn-about").addEventListener("click", () => {
    const about = $("about");
    about.hidden = !about.hidden;
    $("btn-about").setAttribute("aria-expanded", String(!about.hidden));
    requestAnimationFrame(resize);
  });
}

async function main() {
  const tip = document.createElement("div");
  tip.id = "map-tip";
  tip.className = "tooltip";
  tip.hidden = true;
  $("map-wrap").appendChild(tip);
  wire();
  setupCanvas();
  setupMap();
  showLoading("Loading the GSS 2018 model…");
  try {
    await loadModel();
    resize();
    await restoreOrCreate();
    hideLoading();
  } catch (err) {
    showLoading(`Could not start: ${err.message}`);
    document.querySelector("#loading .spinner").hidden = true;
  }
}

main();
