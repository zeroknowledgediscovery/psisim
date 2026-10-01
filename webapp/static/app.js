/* PsiSim webapp client.
 *
 * The browser never computes PsiSim dynamics. It renders snapshots that the
 * server produced with the native runtime: frame 0 is the state before an
 * answer, frame 1 the immediate hard-observation response ("splash"), and
 * frames 2.. the relaxation sweeps ("ripples").
 */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (x, d = 3) => (x == null || !isFinite(x) ? "—" : Number(x).toFixed(d));
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

const MODE_TEXT = {
  sample: {
    label: "Simulate response",
    help: "Draw one answer σ ~ p<sub>i</sub> from the <b>current</b> distribution, using this session's seeded random generator (reproducible from the seed).",
    go: "Simulate an answer",
  },
  map: {
    label: "Most likely",
    help: "Deterministic MAP choice σ = argmax p<sub>i</sub>(s). This is <b>not</b> sampling.",
    go: "Answer with the most likely response",
  },
  choose: {
    label: "Choose response",
    help: "Intervention: force a chosen answer and see what happens to the rest of the modelled worldview. Pick a response above.",
    go: "Force the selected answer",
  },
};

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

const S = {
  model: null,
  vars: [],
  n: 0,
  pos: null,          // Float32Array [x0,y0,x1,y1,...] in [-1.15,1.15]
  edges: null,        // Int32Array [s0,t0,s1,t1,...]
  outAdj: [],
  inAdj: [],
  psi0: [],
  psi: [],            // current Psi (arrays aligned to categories)
  sid: null,
  seed: null,
  observed: [],       // [{column, variable, value, mode, question, u?}]
  q: null,            // current question: {column, value, mode, frames:[...], pending, done}
  viewFrame: -1,
  follow: true,
  selected: null,
  hover: null,
  mode: "sample",
  chosen: null,
  busy: false,
  settings: { max_sweeps: 5, tol: 1e-3, empirical_n: 10 },
  search: "",
};

// ---------------------------------------------------------------------------
// maths helpers on the client side are display-only (colour, sorting)
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

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
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
  S.settings.max_sweeps = m.defaults.max_sweeps;
  S.settings.tol = m.defaults.tol;
  S.settings.empirical_n = m.defaults.empirical_n;
  for (const v of S.vars) {
    v._hay = [v.variable, v.label, v.question_text].join(" ").toLowerCase();
    v._cats = v.categories.map((c) => c.toLowerCase());
  }
}

function applySession(view) {
  S.sid = view.session_id;
  S.seed = view.seed;
  S.psi = view.psi;
  S.observed = view.observed || [];
  S.q = null;
  S.viewFrame = -1;
  store.set("psisim.sid", S.sid);
  $("session-chip").textContent = `seed ${S.seed}`;
}

async function newSession(seed) {
  showLoading("Building a resident native state at Ψ₀…");
  try {
    const body = seed != null ? JSON.stringify({ seed }) : "{}";
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
// answering: stream NDJSON frames from the native simulation
// ---------------------------------------------------------------------------

async function ask(column, mode, value) {
  if (S.busy || !S.sid) return;
  S.busy = true;
  setError("");
  const before = S.psi;
  S.q = {
    column, mode, value: null, u: null,
    frames: [{ index: 0, phase: "before", sweep: 0, summary: null,
               tv: new Float32Array(S.n), psi: before,
               observed: new Set(S.observed.map((o) => o.column)) }],
    pending: null, done: false, stop: null, revealed: false,
  };
  S.follow = true;
  S.viewFrame = 0;
  renderTimeline();
  renderCard();

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
      body: JSON.stringify({
        column, mode, value,
        max_sweeps: S.settings.max_sweeps,
        tol: S.settings.tol,
        empirical_n: S.settings.empirical_n,
      }),
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
          S.q.u = ev.u ?? null;
          S.q.question = ev.question;
          const reveal = revealAnswer(ev);
          showChain = showChain.then(() => reveal).then(() => {
            S.q.revealed = true;
            renderTimeline();
          });
        } else if (ev.type === "progress") {
          S.q.pending = ev;
          if (S.q.revealed) renderTimeline();
        } else if (ev.type === "frame") {
          const prev = S.q.frames[S.q.frames.length - 1];
          const psi = prev.psi.slice();
          for (const [c, arr] of Object.entries(ev.changed)) psi[+c] = arr;
          const observed = new Set(prev.observed);
          observed.add(column);
          const frame = {
            index: ev.frame, phase: ev.phase, sweep: ev.sweep,
            summary: ev.summary, seconds: ev.seconds,
            tv: Float32Array.from(ev.tv), psi, observed,
          };
          S.q.frames.push(frame);
          S.q.pending = null;
          S.psi = psi;
          if (ev.phase === "hard_observation") {
            const rec = {
              column, variable: S.vars[column].variable, value: S.q.value,
              mode, question: S.q.question, u: S.q.u,
            };
            // Publish the answer only after the draw animation has revealed it.
            showChain = showChain.then(() => {
              S.observed = S.observed.concat([rec]);
              renderHistory();
              renderResults();
              renderFieldMsg();
            });
          }
          if (S.q.revealed) renderTimeline();
          enqueueShow(S.q.frames.length - 1, ev.phase === "hard_observation" ? 1700 : 1300);
        } else if (ev.type === "done") {
          S.q.done = true;
          S.q.stop = ev.stop;
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
  }
}

// The sampling "roulette" only animates a draw that the server already made.
async function revealAnswer(ev) {
  const list = document.querySelector("#card .dist");
  if (ev.mode !== "sample" || !list || reduceMotion) {
    markWinner(ev.value);
    return;
  }
  const items = [...list.querySelectorAll("li")];
  const weights = items.map((li) => +li.dataset.p || 0);
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  let delay = 55;
  const t0 = performance.now();
  while (performance.now() - t0 < 1250) {
    let u = Math.random() * total, k = 0;
    while (k < weights.length - 1 && (u -= weights[k]) > 0) k++;
    items.forEach((li, j) => li.classList.toggle("spin", j === k));
    await sleep(delay);
    delay *= 1.12;
  }
  items.forEach((li) => li.classList.remove("spin"));
  markWinner(ev.value);
  await sleep(350);
}

function markWinner(value) {
  const list = document.querySelector("#card .dist");
  if (list) {
    for (const li of list.querySelectorAll("li")) {
      li.classList.toggle("winner", li.dataset.cat === value);
    }
  }
  const note = document.querySelector("#card .draw-note");
  if (note && S.q) {
    const how = S.q.mode === "sample"
      ? `Drawn from p<sub>i</sub>: <b>${esc(value)}</b>${S.q.u != null ? ` <span class="mono">(u = ${fmt(S.q.u, 4)}, seed ${S.seed})</span>` : ""}`
      : S.q.mode === "map" ? `Most likely response: <b>${esc(value)}</b>`
      : `Forced response: <b>${esc(value)}</b>`;
    note.innerHTML = how;
  }
}

// ---------------------------------------------------------------------------
// frame display and animation
// ---------------------------------------------------------------------------

const anim = { flashT0: -1e9, ringT0: -1e9, amp: 0, source: null, raf: 0 };

function currentFrame() {
  if (!S.q || S.viewFrame < 0) return null;
  return S.q.frames[S.viewFrame] || null;
}

function displayedPsi() {
  const f = currentFrame();
  return f ? f.psi : S.psi;
}
function displayedObserved() {
  const f = currentFrame();
  return f ? f.observed : new Set(S.observed.map((o) => o.column));
}

function showFrame(k) {
  if (!S.q || !S.q.frames[k]) return;
  S.viewFrame = k;
  const f = S.q.frames[k];
  anim.source = S.q.column;
  const now = performance.now();
  if (k > 0) {
    anim.flashT0 = now;
    anim.ringT0 = now;
    // The ripple fades as the native residual movement becomes small.
    anim.amp = f.phase === "hard_observation"
      ? 1
      : Math.min(1, Math.sqrt((f.summary?.mean_tv || 0) / 0.08));
  } else {
    anim.amp = 0;
  }
  computeDrift();
  renderTimeline();
  renderMovers();
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
let drift = new Float32Array(0);
let screen = new Float32Array(0);
let radius = new Float32Array(0);

function computeDrift() {
  const psi = displayedPsi();
  drift = new Float32Array(S.n);
  for (let i = 0; i < S.n; i++) drift[i] = tv(psi[i], S.psi0[i]);
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
    for (let i = 0; i < S.n; i++) {
      const d = S.outAdj[i].length + S.inAdj[i].length;
      radius[i] = 1.7 + Math.sqrt(d) * 0.32;
    }
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
  g.strokeStyle = "rgba(120,170,230,0.045)";
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

// sequential ramp: Psi0 (deep blue) -> moved (cyan -> near white)
const RAMP = [[34, 64, 109], [47, 127, 176], [76, 201, 240], [233, 251, 255]];
function rampColor(x) {
  const t = Math.max(0, Math.min(1, Math.sqrt(x / 0.6))) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t));
  const f = t - i, a = RAMP[i], b = RAMP[i + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
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

  // links of the selected / hovered item: outgoing = items whose tree uses it
  const focus = S.hover ?? S.selected;
  if (focus != null) {
    drawLinks(g, focus, S.inAdj[focus], "#b59cff", 0.35);
    drawLinks(g, focus, S.outAdj[focus], "#4cc9f0", 0.55);
  }

  // splash: the hard response travels along the source's learned links
  const tFlash = (now - anim.flashT0) / 1800;
  if (frame && frame.phase === "hard_observation" && tFlash < 1 && anim.source != null) {
    drawLinks(g, anim.source, S.outAdj[anim.source], "#e9fbff", 0.7 * (1 - tFlash));
    animating = true;
  }

  // nodes
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
      g.fillStyle = S.vars[i].learned ? rampColor(drift[i] || 0) : "#2a3550";
      g.fill();
    }
  }

  // halo: per-item change in the displayed frame (native TV vs previous frame)
  if (frame && frame.index > 0 && tFlash < 1) {
    const fade = Math.pow(1 - tFlash, 1.4);
    g.globalCompositeOperation = "lighter";
    const T = frame.tv;
    for (let i = 0; i < S.n; i++) {
      const v = T[i];
      if (v < 2e-3) continue;
      const I = Math.min(1, Math.sqrt(v / 0.25)) * fade;
      const x = screen[2 * i], y = screen[2 * i + 1];
      const r = (radius[i] + 4 + 10 * I) * zr;
      const grd = g.createRadialGradient(x, y, 0, x, y, r);
      grd.addColorStop(0, `rgba(233,251,255,${0.85 * I})`);
      grd.addColorStop(0.35, `rgba(120,220,255,${0.45 * I})`);
      grd.addColorStop(1, "rgba(76,201,240,0)");
      g.fillStyle = grd;
      g.beginPath();
      g.arc(x, y, r, 0, 6.2832);
      g.fill();
    }
    g.globalCompositeOperation = "source-over";
    animating = true;
  }

  // decorative water rings around the answered item (presentation only)
  const tRing = (now - anim.ringT0) / 2200;
  if (anim.source != null && anim.amp > 0 && tRing < 1 && frame && frame.index > 0) {
    const x = screen[2 * anim.source], y = screen[2 * anim.source + 1];
    const R = Math.min(view.w, view.h) * 0.42;
    for (let k = 0; k < 3; k++) {
      const t = tRing - k * 0.13;
      if (t <= 0 || t >= 1) continue;
      g.beginPath();
      g.arc(x, y, 6 + R * (1 - Math.pow(1 - t, 2.2)), 0, 6.2832);
      g.strokeStyle = `rgba(165,236,255,${(0.55 * anim.amp * (1 - t)).toFixed(3)})`;
      g.lineWidth = 2.2 - k * 0.5;
      g.stroke();
    }
    animating = true;
  }

  // selection ring
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

function kick() {
  if (!anim.raf) anim.raf = requestAnimationFrame(draw);
}

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
  const tip = $("tooltip");
  tip.innerHTML =
    `<div class="t-var">${esc(v.variable)} <span class="muted">· col ${i}</span></div>` +
    `<div class="t-label">${esc(v.label || "no question text in metadata")}</div>` +
    `<div class="t-row">most likely: ${esc(v.categories[k])} (${fmt(psi[k])})</div>` +
    `<div class="t-row">moved from Ψ₀: TV ${fmt(drift[i])}</div>` +
    (f && f.index > 0 ? `<div class="t-row">change this frame: TV ${fmt(f.tv[i], 4)}</div>` : "") +
    (displayedObserved().has(i) ? `<div class="t-row" style="color:var(--observed)">answered · clamped</div>` : "");
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${Math.min(view.w - w - 8, px + 14)}px`;
  tip.style.top = `${Math.min(view.h - h - 8, py + 14)}px`;
}
function hideTip() { $("tooltip").hidden = true; }

// ---------------------------------------------------------------------------
// panels
// ---------------------------------------------------------------------------

function select(col) {
  S.selected = col;
  S.chosen = null;
  setError("");
  renderCard();
  renderResults();
  kick();
}

function observedRecord(col) {
  return S.observed.find((o) => o.column === col) || null;
}

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
    ? `${list.length}${list.length === 60 ? "+" : ""} matching items`
    : "Suggested items";
  $("results").innerHTML = list.length
    ? list.map(({ v, hit }) => {
        const done = observedRecord(v.column);
        return `<li><button data-col="${v.column}" aria-current="${S.selected === v.column}">` +
          `<span class="r-var">${esc(v.variable)}${done ? `<span class="done">answered</span>` : ""}</span>` +
          `<span class="r-label">${esc(v.label || "—")}</span>` +
          (hit ? `<span class="r-hit">${esc(hit)}</span>` : "") +
          `</button></li>`;
      }).join("")
    : `<li class="empty">No modelled item matches.</li>`;
}

function renderHistory() {
  $("answer-count").textContent = S.observed.length;
  $("history").innerHTML = S.observed.length
    ? S.observed.map((o) =>
        `<li><button data-col="${o.column}"><div class="h-row">` +
        `<span class="h-q">Q${o.question}</span><span class="h-body">` +
        `<span class="mono">${esc(o.variable)}</span> = <span class="h-val">${esc(o.value)}</span>` +
        `<span class="badge ${o.mode}">${o.mode === "sample" ? "sampled" : o.mode === "map" ? "MAP" : "forced"}</span>` +
        `</span></div></button></li>`).join("")
    : `<li class="empty">No answers yet.</li>`;
}

function renderFieldMsg() {
  const n = S.observed.length;
  $("field-msg").innerHTML = n === 0
    ? "No answers have been supplied. Every node is showing the response distribution implied by the GSS 2018 LSM from the empty state."
    : `<b>${n} answer${n > 1 ? "s" : ""} supplied.</b> Colour shows how far each item's response distribution has moved from the empty state Ψ₀; amber items are answered and clamped.`;
}

function frameName(f) {
  if (f.phase === "before") return "Before";
  if (f.phase === "hard_observation") return "Splash";
  return `Sweep ${f.sweep}`;
}

function renderTimeline() {
  const q = S.q;
  if (!q) {
    $("tl-title").textContent = "Frames appear here after the first answer.";
    $("frames").innerHTML = "";
    $("tl-status").innerHTML = "";
    $("btn-replay").disabled = true;
    renderMovers();
    return;
  }
  const v = S.vars[q.column];
  const modeTxt = q.mode === "sample" ? "sampled" : q.mode === "map" ? "most likely (MAP)" : "forced";
  $("tl-title").innerHTML = `Q${q.question ?? S.observed.length + 1} · <span class="mono">${esc(v.variable)}</span>` +
    (q.value != null && q.revealed ? ` = <b style="color:var(--observed)">${esc(q.value)}</b> <span class="muted">(${modeTxt})</span>` : ` <span class="muted">…</span>`);

  const shown = q.revealed ? q.frames : q.frames.slice(0, 1);
  const maxTv = Math.max(1e-9, ...shown.slice(1).map((f) => f.summary?.mean_tv || 0));
  let html = shown.map((f, k) => {
    const m = f.summary ? f.summary.mean_tv : null;
    const w = m != null ? Math.max(3, 100 * m / maxTv) : 0;
    return `<button data-frame="${k}" aria-current="${S.viewFrame === k}" title="${f.summary ? `mean TV ${fmt(m, 5)} · max TV ${fmt(f.summary.max_tv, 4)}` : "state before the answer"}">` +
      `<span class="f-name">${frameName(f)}</span>` +
      `<span class="f-meta">${m != null ? `mean ${fmt(m, 4)}` : "Ψ before"}</span>` +
      `<span class="f-bar"><i style="width:${w}%"></i></span></button>`;
  }).join("");
  if (q.pending && q.revealed) {
    html += `<button class="pending" disabled><span class="f-name">Sweep ${q.pending.sweep}</span><span class="f-meta">computing…</span><span class="f-bar"></span></button>`;
  }
  $("frames").innerHTML = html;

  let status = "";
  if (!q.revealed) {
    status = `<span class="spinner"></span>${q.mode === "sample" ? "drawing a response from p<sub>i</sub>" : "applying the answer"}`;
  } else if (!q.done) {
    status = `<span class="spinner"></span>${q.pending ? `native relaxation sweep ${q.pending.sweep}/${q.pending.of}` : "hard observation"}`;
  } else if (q.stop === "tolerance") {
    status = `stopped early: mean TV &lt; ${S.settings.tol}`;
  } else if (q.stop) {
    status = `${q.frames.length - 2} relaxation sweep${q.frames.length - 2 === 1 ? "" : "s"}`;
  }
  $("tl-status").innerHTML = status;
  $("btn-replay").disabled = !q.done || q.frames.length < 2;
}

function renderMovers() {
  const f = currentFrame();
  if (!f || f.index === 0) {
    $("movers-note").textContent = "";
    $("tl-note").hidden = true;
    $("movers").innerHTML = `<li class="empty">${S.q ? "Frame 0 is the state immediately before the answer." : "Ask a question to see which items move."}</li>`;
    return;
  }
  const prev = S.q.frames[S.viewFrame - 1];
  const idx = [];
  for (let i = 0; i < S.n; i++) if (f.tv[i] > 1e-6) idx.push(i);
  idx.sort((a, b) => f.tv[b] - f.tv[a]);
  $("movers-note").textContent = `· ${idx.length} items moved · mean TV ${fmt(f.summary.mean_tv, 4)} · max TV ${fmt(f.summary.max_tv, 3)}`;
  $("tl-note").hidden = f.phase !== "relaxation";
  $("movers").innerHTML = idx.slice(0, 10).map((i) => {
    const v = S.vars[i];
    const a = argmax(prev.psi[i]), b = argmax(f.psi[i]);
    const shift = a !== b
      ? `most likely: ${v.categories[a]} → ${v.categories[b]}`
      : `${v.categories[b]}: ${fmt(prev.psi[i][b])} → ${fmt(f.psi[i][b])}`;
    return `<li><button data-col="${i}"><div class="m-row"><span class="m-name"><span class="mono">${esc(v.variable)}</span> ${esc(v.label)}</span>` +
      `<span class="m-tv">${fmt(f.tv[i], 3)}</span></div><span class="m-shift">${esc(shift)}</span></button></li>`;
  }).join("") || `<li class="empty">No item moved in this frame.</li>`;
}

function renderCard() {
  const card = $("card");
  if (S.selected == null) {
    const chips = SUGGESTED
      .map((n) => S.vars.find((v) => v.variable.toLowerCase() === n))
      .filter((v) => v && v.learned)
      .slice(0, 9)
      .map((v) => `<button data-col="${v.column}" title="${esc(v.label)}">${esc(v.variable)}</button>`).join("");
    card.innerHTML = `<div class="card-intro">
      <p class="lead">${S.observed.length ? "Pick the next question. It will be asked in the <b>current</b> state, which already reflects every earlier answer." :
        "No answers have been supplied. Every node is showing the response distribution implied by the GSS 2018 LSM from the empty state."}</p>
      <p>Search for a GSS item on the left, or click any dot in the field. The card shows that item's current response distribution; answering it drops a stone into the system.</p>
      <p class="muted">Try one of these:</p>
      <div class="suggest">${chips}</div>
    </div>`;
    return;
  }

  const col = S.selected;
  const v = S.vars[col];
  const p = S.psi[col];
  const p0 = S.psi0[col];
  const rec = observedRecord(col);
  const running = S.busy && S.q && S.q.column === col;
  const order = v.categories.map((c, k) => k).sort((a, b) => p[b] - p[a] || a - b);
  const choosable = S.mode === "choose" && !rec && !S.busy;

  const rows = order.map((k) => {
    const c = v.categories[k];
    const id = `cat-${col}-${k}`;
    const checked = S.chosen === c ? "checked" : "";
    return `<li data-cat="${esc(c)}" data-p="${p[k]}" class="${rec && rec.value === c ? "winner" : ""}">` +
      (choosable ? `<input type="radio" name="cat" id="${id}" value="${esc(c)}" ${checked}>` : "") +
      `<label ${choosable ? `for="${id}"` : ""}><span class="d-label" title="${esc(c)}">${esc(c)}</span>` +
      `<span class="d-p">${fmt(p[k])}</span>` +
      `<span class="d-bar"><i style="width:${(100 * p[k]).toFixed(2)}%"></i><b style="left:calc(${(100 * p0[k]).toFixed(2)}% - 1px)"></b></span></label></li>`;
  }).join("");

  const links = `<b>${S.outAdj[col].length}</b> item${S.outAdj[col].length === 1 ? "" : "s"} respond directly to this answer · informed by <b>${S.inAdj[col].length}</b>`;
  const status = rec
    ? `<span class="status clamped">Answered in Q${rec.question} · clamped to “${esc(rec.value)}”</span>`
    : `<span class="status">Not yet answered · distribution from the current state Ψ</span>`;

  const controls = rec || !v.learned ? "" : `
    <div class="modes" role="group" aria-label="How the response is generated">
      ${Object.entries(MODE_TEXT).map(([m, t]) =>
        `<button data-mode="${m}" aria-pressed="${S.mode === m}" ${S.busy ? "disabled" : ""}>${t.label}</button>`).join("")}
    </div>
    <p class="mode-help">${MODE_TEXT[S.mode].help}</p>
    <button class="btn go" id="btn-go" ${S.busy || (S.mode === "choose" && S.chosen == null) ? "disabled" : ""}>${running ? "Relaxing…" : MODE_TEXT[S.mode].go}</button>`;

  card.innerHTML = `
    <div class="q-meta"><span class="q-var">${esc(v.variable)}</span><span class="q-col">native column ${col}</span></div>
    ${v.label ? `<div class="q-text">${esc(v.question_text || v.label)}</div>` :
      `<div class="q-text missing">No GSS question text is available for this variable in the bundled metadata.</div>`}
    ${v.label ? `<div class="q-source">GSS codebook label${v.label_source_year ? ` (DTAG GSS ${esc(v.label_source_year)} map)` : ""}</div>` : ""}
    <div class="q-links">${links}</div>
    ${status}
    <div class="dist-head"><span>Response</span><span>p<sub>i</sub>(s)</span></div>
    <ul class="dist ${choosable ? "choosable" : ""}">${rows}</ul>
    <div class="dist-legend"><span><i></i>current Ψ</span><span><b></b>empty state Ψ₀</span></div>
    ${controls}
    <div class="draw-note"></div>
    <div class="error" id="card-error" hidden></div>
    <details class="settings">
      <summary>Simulation settings</summary>
      <div class="set-grid">
        <label for="set-sweeps">Maximum relaxation sweeps</label>
        <input id="set-sweeps" type="number" min="0" max="${S.model.defaults.max_sweeps_limit}" value="${S.settings.max_sweeps}">
        <label for="set-tol">Stop early when mean TV &lt;</label>
        <input id="set-tol" type="number" min="0" step="0.0005" value="${S.settings.tol}">
        <label for="set-n">Empirical n (mode events)</label>
        <input id="set-n" type="number" min="1" max="1000" value="${S.settings.empirical_n}">
        <span class="set-note">Event “mode”, response scale 1.0, fixed sweep order. Session seed ${S.seed}.</span>
      </div>
    </details>`;
  if (running && S.q && S.q.value != null) markWinner(S.q.value);
}

function setError(msg) {
  const el = $("card-error");
  if (el) { el.hidden = !msg; el.textContent = msg; }
  else if (msg) alert(msg);
}

function renderAll() {
  computeDrift();
  renderFieldMsg();
  renderResults();
  renderHistory();
  renderTimeline();
  renderCard();
  kick();
}

function showLoading(text) {
  $("loading-text").textContent = text;
  $("loading").hidden = false;
}
function hideLoading() { $("loading").hidden = true; }

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

function wire() {
  $("search").addEventListener("input", (e) => { S.search = e.target.value; renderResults(); });
  $("search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const first = $("results").querySelector("button[data-col]");
      if (first) select(+first.dataset.col);
    }
  });
  for (const id of ["results", "history", "movers"]) {
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
      await sleep(reduceMotion ? 300 : k === 1 ? 1700 : 1300);
    }
  });
  $("card").addEventListener("click", (e) => {
    const sug = e.target.closest(".suggest button[data-col]");
    if (sug) { select(+sug.dataset.col); return; }
    const m = e.target.closest("button[data-mode]");
    if (m) { S.mode = m.dataset.mode; S.chosen = null; renderCard(); return; }
    if (e.target.closest("#btn-go")) {
      if (S.mode === "choose" && S.chosen == null) return;
      ask(S.selected, S.mode, S.mode === "choose" ? S.chosen : null);
    }
  });
  $("card").addEventListener("change", (e) => {
    if (e.target.name === "cat") {
      S.chosen = e.target.value;
      const go = $("btn-go");
      if (go) go.disabled = S.busy;
    }
    const num = (id, lo, hi, def) => {
      const x = Number($(id).value);
      return Number.isFinite(x) ? Math.max(lo, Math.min(hi, x)) : def;
    };
    if (e.target.id === "set-sweeps") S.settings.max_sweeps = Math.round(num("set-sweeps", 0, S.model.defaults.max_sweeps_limit, 5));
    if (e.target.id === "set-tol") S.settings.tol = num("set-tol", 0, 1, 1e-3);
    if (e.target.id === "set-n") S.settings.empirical_n = Math.round(num("set-n", 1, 1000, 10));
  });
  $("btn-new").addEventListener("click", async () => {
    if (S.busy) return;
    const raw = prompt("Start a new session from Ψ₀.\nOptional RNG seed (leave blank for a random seed):", "");
    if (raw === null) return;
    const seed = raw.trim() === "" ? null : Number(raw.trim());
    if (seed != null && !(Number.isInteger(seed) && seed >= 0)) { alert("Seed must be a non-negative integer."); return; }
    const old = S.sid;
    S.selected = null;
    try {
      await newSession(seed);
      if (old) api(`/api/session/${encodeURIComponent(old)}`, { method: "DELETE" }).catch(() => {});
    } catch (err) { alert(err.message); }
  });
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
  wire();
  setupCanvas();
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
