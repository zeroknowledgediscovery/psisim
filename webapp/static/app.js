/* PsiSim webapp client.
 *
 * Shows that asking one question updates the whole state Psi. Every item's
 * response distribution p_i is a sparkline in one grid, coloured by how far it
 * is from the empty state Psi0. Asking item i draws sigma ~ p_i on the server,
 * which applies one native hard observation: p_i becomes the point mass
 * delta_sigma (red) and the centred change is passed once through the learned
 * links. Changed sparklines then morph to their new shape in a cascade, the
 * largest updates pop out as before/after cards, and the "current mind" donut
 * shows every item's distance from Psi0 at a glance.
 * The browser never computes PsiSim dynamics; it only draws server snapshots.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (x, d = 3) => (x == null || !isFinite(x) ? "—" : Number(x).toFixed(d));
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const easeOut = (t) => 1 - Math.pow(1 - Math.max(0, Math.min(1, t)), 3);

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};

const SUGGESTED = [
  "polviews", "partyid", "abany", "cappun", "grass", "gunlaw", "god",
  "homosex", "natenvir", "fefam", "trust", "happy", "attend", "wrkstat",
];
const MOVED = 1.25e-6;       // streamed probabilities are rounded to 1e-6
const SPOT_MAX = 6;          // largest updates popped out after each answer
const SPOT_MIN_TV = 0.002;
const NEXT_MAX = 14;         // rows in the "ask next" list

const S = {
  catalog: null, modelKey: null,
  model: null, vars: [], n: 0, psi0: [],
  psi: [], sid: null, seed: null,
  observed: [],              // [{column, variable, value, question, u}]
  answers: [],               // [{question, column, variable, value, u, stats, tv}]
  last: null,                // most recent answer: {column, value, u, before, tv, changed, question}
  drift: new Float32Array(0),// TV(p_i, Psi0_i) of the current state
  busy: false,
  search: "",
  order: [], scale: [],      // per item: category display order and sparkline scale
  spotToken: 0, spotCol: null, hover: null,
  morph: null,               // {t0, before, beforeDrift, delay: Map(col -> ms)}
};

// ---------------------------------------------------------------------------
// helpers (display only)
// ---------------------------------------------------------------------------

function tv(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
  return 0.5 * s;
}
function argmax(arr) {
  let k = 0;
  for (let j = 1; j < arr.length; j++) if (arr[j] > arr[k]) k = j;
  return k;
}
function observedRecord(col) { return S.observed.find((o) => o.column === col) || null; }
function label(v) { return v.label || v.variable; }
function computeDrift() {
  S.drift = new Float32Array(S.n);
  for (let i = 0; i < S.n; i++) S.drift[i] = tv(S.psi[i], S.psi0[i]);
}

// Sequential ramp for "change from Psi0" (log scale: 1e-4 .. 1e-1).
let RAMP = null;
function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
function rampT(x) { return x > 1e-4 ? Math.max(0, Math.min(1, (Math.log10(x) + 4) / 3)) : 0; }
function rampColor(x) {
  if (!RAMP) RAMP = ["--r0", "--r1", "--r2", "--r3"].map((v) => hexRgb(css(v)));
  const t = rampT(x) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i, a = RAMP[i], b = RAMP[i + 1];
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
    throw new Error(detail);
  }
  return res;
}

async function loadModel(key) {
  const m = await (await api(`/api/model?key=${encodeURIComponent(key)}`)).json();
  S.modelKey = m.model;
  S.model = m;
  S.vars = m.variables;
  S.n = m.width;
  S.psi0 = m.psi0;
  for (const v of S.vars) {
    v._hay = [v.variable, v.label, v.question_text].join(" ").toLowerCase();
    v._cats = v.categories.map((c) => c.toLowerCase());
  }
  // Each sparkline keeps one fixed category order (by Psi0) and scale, so a
  // change in shape is a change in the distribution.
  S.order = S.psi0.map((p) => p.map((_, k) => k).sort((a, b) => p[b] - p[a] || a - b));
  S.scale = S.psi0.map((p) => Math.min(1, Math.max(...p) * 1.25) || 1);
  renderSurveyHead();
}

function modelLabel() { return (S.model && S.model.info && S.model.info.label) || S.modelKey || "the survey"; }

function makeLast(column, value, u, before, tvArr, question) {
  const changed = [];
  for (let i = 0; i < S.n; i++) if (i !== column && tvArr[i] > MOVED) changed.push(i);
  changed.sort((a, b) => tvArr[b] - tvArr[a]);
  return { column, value, u, before, tv: tvArr, changed, question };
}

async function applySession(view) {
  if (view.model && view.model !== S.modelKey) await loadModel(view.model);
  S.sid = view.session_id;
  S.seed = view.seed;
  S.psi = view.psi;
  S.observed = view.observed || [];
  const hist = view.history || [];
  const uOf = (q) => (S.observed.find((o) => o.question === q) || {}).u;
  S.answers = hist.map((h) => ({
    question: h.question, column: h.column, variable: h.variable, value: h.value,
    u: uOf(h.question), stats: h.stats, tv: Float32Array.from(h.tv),
  }));
  S.last = null;
  if (hist.length) {
    const h = hist[hist.length - 1];
    const before = hist.length > 1 ? hist[hist.length - 2].psi : S.psi0;
    S.last = makeLast(h.column, h.value, uOf(h.question), before, Float32Array.from(h.tv), h.question);
  }
  S.morph = null;
  computeDrift();
  store.set("psisim.sid", S.sid);
  $("session-chip").textContent = `seed ${S.seed}`;
}

async function newSession(seed, modelKey) {
  showLoading("Starting a new session at Ψ₀…");
  try {
    const body = JSON.stringify({ ...(seed != null ? { seed } : {}), model: modelKey || S.modelKey || undefined });
    await applySession(await (await api("/api/session", { method: "POST", body })).json());
  } finally { hideLoading(); }
  cancelSpots();
  renderAll();
}

async function restoreOrCreate() {
  const sid = store.get("psisim.sid");
  if (sid) {
    try {
      await applySession(await (await api(`/api/session/${encodeURIComponent(sid)}`)).json());
      renderAll();
      return;
    } catch { store.del("psisim.sid"); }
  }
  await newSession();
}

// ---------------------------------------------------------------------------
// asking a question: one native update of Psi
// ---------------------------------------------------------------------------

async function ask(col) {
  const v = S.vars[col];
  if (!v || !v.learned || observedRecord(col) || S.busy) return;
  S.busy = true;
  cancelSpots();
  renderResults();
  if (S.catalog) renderCandidates();
  const before = S.psi;
  renderQuestion({ phase: "asking", column: col, p: before[col] });
  renderSummaryPending();

  let answer = null, frame = null, done = null;
  try {
    const res = await api(`/api/session/${encodeURIComponent(S.sid)}/answer`, {
      method: "POST",
      body: JSON.stringify({ column: col, max_steps: 0 }),
    });
    for (const line of (await res.text()).split("\n")) {
      if (!line.trim()) continue;
      const ev = JSON.parse(line);
      if (ev.type === "error") throw new Error(ev.detail);
      if (ev.type === "answer") answer = ev;
      else if (ev.type === "frame") frame = ev;
      else if (ev.type === "done") done = ev;
    }
    if (!answer || !frame || !done) throw new Error("incomplete response from the server");
  } catch (err) {
    S.busy = false;
    renderQuestion({ phase: "error", message: err.message });
    renderResults();
    if (S.catalog) renderCandidates();
    return;
  }

  // 1. draw the answer from p_i (replays the server's seeded draw)
  await drawAnimation();
  // 2. p_i collapses to the point mass delta_sigma
  collapseAnimation(answer.value, answer.u);
  await sleep(reduceMotion ? 80 : 520);

  // 3. Psi updates: changed sparklines morph in a cascade, largest first
  const after = before.slice();
  for (const [c, arr] of Object.entries(frame.changed)) after[+c] = arr;
  const beforeDrift = S.drift;
  S.psi = after;
  computeDrift();
  const rec = { column: col, variable: v.variable, value: answer.value, question: answer.question, u: answer.u };
  S.observed = S.observed.concat([rec]);
  const tvArr = Float32Array.from(frame.tv);
  S.answers = S.answers.concat([{ ...rec, stats: done.stats, tv: tvArr }]);
  S.last = makeLast(col, answer.value, answer.u, before, tvArr, answer.question);
  startMorph(before, beforeDrift);
  renderResults();
  renderSummary();
  renderGridHead();
  renderMindText();
  S.busy = false;
  if (S.catalog) renderCandidates();

  // 4. the largest updates pop out one by one
  await sleep(reduceMotion ? 0 : 450);
  runSpots();
}

async function drawAnimation() {
  const rows = [...document.querySelectorAll("#question .bars .bar-row[data-cat]")];
  if (!rows.length || reduceMotion) return;
  const weights = rows.map((r) => +r.dataset.p || 0);
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  let delay = 38;
  const t0 = performance.now();
  while (performance.now() - t0 < 720) {
    let u = Math.random() * total, k = 0;
    while (k < weights.length - 1 && (u -= weights[k]) > 0) k++;
    rows.forEach((r, j) => r.classList.toggle("spin", j === k));
    await sleep(delay);
    delay *= 1.16;
  }
  rows.forEach((r) => r.classList.remove("spin"));
}

function collapseAnimation(value, u) {
  const bars = document.querySelector("#question .bars");
  if (!bars) return;
  bars.classList.add("collapsed");
  for (const r of bars.querySelectorAll(".bar-row")) {
    const chosen = r.dataset.cat === value;
    r.classList.toggle("chosen", chosen);
    r.querySelector(".fill").style.width = chosen ? "100%" : "0%";
    r.querySelector(".val").textContent = chosen ? "1.000" : "0.000";
  }
  const note = document.querySelector("#question .q-draw");
  if (note) {
    note.innerHTML = `Answer drawn from p<sub>i</sub>: <b>${esc(value)}</b> ` +
      `<span class="mono">(u = ${fmt(u, 4)}, seed ${S.seed})</span>`;
  }
}

// ---------------------------------------------------------------------------
// question panel
// ---------------------------------------------------------------------------

// Category rows for a distribution: the largest `max` by p, rest folded.
function barRows(col, p, max = 10) {
  const v = S.vars[col];
  const idx = v.categories.map((_, k) => k).sort((a, b) => p[b] - p[a] || a - b);
  const shown = idx.slice(0, max);
  const rest = idx.slice(max);
  const row = (lab, pv, cat) =>
    `<div class="bar-row"${cat != null ? ` data-cat="${esc(cat)}"` : ""} data-p="${pv}">` +
    `<span class="lab${cat == null ? " muted" : ""}" title="${esc(lab)}">${esc(lab)}</span>` +
    `<span class="track"><span class="ghost" style="width:${(100 * pv).toFixed(2)}%"></span>` +
    `<span class="fill" style="width:${(100 * pv).toFixed(2)}%"></span></span>` +
    `<span class="val">${fmt(pv)}</span></div>`;
  const rows = shown.map((k) => row(v.categories[k], p[k], v.categories[k]));
  if (rest.length) rows.push(row(`${rest.length} other answers`, rest.reduce((a, k) => a + p[k], 0), null));
  return rows.join("");
}

function renderQuestion(st) {
  const el = $("question");
  if (!st || st.phase === "idle") {
    if (S.last) { renderQuestion({ phase: "result" }); return; }
    el.innerHTML = `<div class="q-intro">
      <div class="q-kicker">No answers yet</div>
      <p><b>No answers have been supplied. Every sparkline below is showing the response distribution implied by the ${esc(modelLabel())} LSM from the empty state.</b></p>
      <p>Choose a question from <i>Ask next</i> on the left (or click any sparkline). An answer is drawn from its current distribution, that distribution becomes a single red bar, and the change is passed to every related question. Ψ<sub>0</sub> does not move until something is asked.</p>
    </div>`;
    return;
  }
  if (st.phase === "error") {
    el.innerHTML = `<div class="q-intro"><div class="q-kicker">Could not ask</div><p>${esc(st.message)}</p></div>`;
    return;
  }
  const col = st.phase === "asking" ? st.column : S.last.column;
  const v = S.vars[col];
  const qnum = st.phase === "asking" ? S.observed.length + 1 : S.last.question;
  const p = st.phase === "asking" ? st.p : S.last.before[col];
  el.innerHTML = `
    <div>
      <div class="q-kicker">Question ${qnum} · <span class="q-var">${esc(v.variable)}</span></div>
      <div class="q-text${v.label ? "" : " missing"}">${esc(v.question_text || v.label || "No question text available for this variable")}</div>
      <div class="bars">${barRows(col, p)}</div>
      <div class="q-draw">${st.phase === "asking" ? "Drawing an answer from its current distribution…" : ""}</div>
    </div>
    <div class="summary" id="summary"></div>`;
  if (st.phase === "result") {
    for (const f of el.querySelectorAll(".fill")) f.style.transition = "none";
    collapseAnimation(S.last.value, S.last.u);
    renderSummary();
  }
}

function renderSummaryPending() {
  const el = $("summary");
  if (el) el.innerHTML = `<div class="sum-sub">Updating Ψ…</div>`;
}

function renderSummary() {
  const el = $("summary");
  if (!el || !S.last) return;
  const L = S.last;
  const others = S.n - S.observed.length;
  const n001 = L.changed.filter((i) => L.tv[i] > 0.01).length;
  el.innerHTML = `
    <div class="sum-big"><b>${L.changed.length}</b><span>of ${others} other distributions changed</span></div>
    <div class="sum-sub">${n001} by more than 0.01 (total variation)${L.changed.length ? ` · largest ${fmt(L.tv[L.changed[0]])}` : ""}.
      The answer is passed once through the learned links to every question whose model uses <span class="mono">${esc(S.vars[L.column].variable)}</span>.</div>
    <ol class="top-list">${L.changed.slice(0, 10).map((i) => {
      const v = S.vars[i];
      const a = argmax(L.before[i]), b = argmax(S.psi[i]);
      const shift = a !== b
        ? `most likely: ${v.categories[a]} → ${v.categories[b]}`
        : `${v.categories[b]}: ${fmt(L.before[i][b])} → ${fmt(S.psi[i][b])}`;
      return `<li data-col="${i}"><span class="t-name"><span class="mono">${esc(v.variable)}</span> ${esc(v.label)}</span>` +
        `<span class="t-tv">${fmt(L.tv[i])}</span><span class="t-shift">${esc(shift)}</span></li>`;
    }).join("") || `<li><span class="t-name muted">No other distribution depends on this question.</span></li>`}</ol>`;
}

// ---------------------------------------------------------------------------
// morph: changed sparklines move from their old to their new shape
// ---------------------------------------------------------------------------

const MORPH_MS = 650, MORPH_STAGGER = 14, MORPH_SPREAD = 900;

function startMorph(before, beforeDrift) {
  const delay = new Map();
  const ch = S.last ? S.last.changed : [];
  const step = ch.length ? Math.min(MORPH_STAGGER, MORPH_SPREAD / ch.length) : 0;
  ch.forEach((i, r) => delay.set(i, r * step));            // largest change first
  if (S.last) delay.set(S.last.column, 0);
  S.morph = reduceMotion ? null : { t0: performance.now(), before, beforeDrift, delay };
  kick();
}

// Progress of item i's morph (1 = settled), and the glow strength.
function morphOf(i, now) {
  const M = S.morph;
  if (!M || !M.delay.has(i)) return { e: 1, glow: 0 };
  const t = (now - M.t0 - M.delay.get(i)) / MORPH_MS;
  if (t >= 1.6) return { e: 1, glow: 0 };
  return { e: easeOut(t), glow: t < 0 ? 0 : Math.max(0, 1 - t / 1.6) };
}

function shownP(i, e) {
  if (e >= 1 || !S.morph) return S.psi[i];
  const a = S.morph.before[i], b = S.psi[i];
  return b.map((x, k) => a[k] + (x - a[k]) * e);
}
function shownDrift(i, e) {
  if (e >= 1 || !S.morph) return S.drift[i];
  return S.morph.beforeDrift[i] + (S.drift[i] - S.morph.beforeDrift[i]) * e;
}

// ---------------------------------------------------------------------------
// sparkline grid
// ---------------------------------------------------------------------------

const canvas = $("grid");
const ctx = canvas.getContext("2d");
const G = { cols: 1, rows: 1, cw: 30, ch: 18, gap: 3, w: 0, h: 0, dpr: 1 };
let raf = 0;

function layoutGrid() {
  const wrap = $("grid-wrap");
  const W = wrap.clientWidth;
  const narrow = window.matchMedia("(max-width: 900px)").matches;
  const n = S.n, gap = 3, aspect = 1.7;
  let best = null;
  if (narrow) {
    const cols = Math.max(8, Math.floor((W + gap) / (34 + gap)));
    const cw = Math.floor((W - gap * (cols - 1)) / cols);
    best = { cols, cw, ch: Math.round(cw / aspect), rows: Math.ceil(n / cols) };
    wrap.style.height = `${best.rows * (best.ch + gap)}px`;
  } else {
    wrap.style.height = "";
    const H = wrap.clientHeight;
    for (let cols = Math.max(8, Math.ceil((W + gap) / (72 + gap))); cols <= 120; cols++) {
      const cw = Math.floor((W - gap * (cols - 1)) / cols);
      if (cw < 15) break;
      const ch = Math.max(10, Math.round(cw / aspect));
      const rows = Math.ceil(n / cols);
      best = { cols, cw, ch, rows };
      if (rows * ch + (rows - 1) * gap <= H) break;
    }
  }
  Object.assign(G, best, { gap, w: W, h: wrap.clientHeight });
  G.dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(G.w * G.dpr);
  canvas.height = Math.round(G.h * G.dpr);
  kick();
}

function cellRect(i) {
  const c = i % G.cols, r = Math.floor(i / G.cols);
  return { x: c * (G.cw + G.gap), y: r * (G.ch + G.gap), w: G.cw, h: G.ch };
}

function cellAt(px, py) {
  const c = Math.floor(px / (G.cw + G.gap)), r = Math.floor(py / (G.ch + G.gap));
  if (c < 0 || c >= G.cols || r < 0) return null;
  if (px - c * (G.cw + G.gap) > G.cw || py - r * (G.ch + G.gap) > G.ch) return null;
  const i = r * G.cols + c;
  return i < S.n ? i : null;
}

function drawSpark(g, i, x, y, w, h, color, p) {
  const ord = S.order[i];
  const k = ord.length;
  const pad = 2, iw = w - 2 * pad, ih = h - 2 * pad, bw = iw / k;
  const sc = observedRecord(i) ? 1 : S.scale[i];
  g.fillStyle = color;
  for (let j = 0; j < k; j++) {
    const q = p[ord[j]];
    if (q <= 0) continue;
    const bh = Math.min(1, q / sc) * ih;
    g.fillRect(x + pad + j * bw, y + pad + ih - bh, Math.max(0.8, bw - (bw >= 3 ? 1 : 0)), Math.max(0.6, bh));
  }
}

function draw(now = performance.now()) {
  raf = 0;
  if (!S.n || !S.psi.length) return;
  const ans = css("--ans"), hl = css("--hl");
  const g = ctx;
  g.setTransform(G.dpr, 0, 0, G.dpr, 0, 0);
  g.clearRect(0, 0, G.w, G.h);
  let animating = false;
  const glows = [];
  for (let i = 0; i < S.n; i++) {
    const r = cellRect(i);
    if (r.y > G.h) break;
    const { e, glow } = morphOf(i, now);
    if (e < 1 || glow > 0) animating = true;
    const rec = observedRecord(i);
    // the answered item flips to red when its own morph starts
    const color = rec && (e > 0.05 || !S.morph || S.last.column !== i) ? ans : rampColor(shownDrift(i, e));
    g.fillStyle = "rgba(255,255,255,0.035)";
    g.fillRect(r.x, r.y, r.w, r.h);
    drawSpark(g, i, r.x, r.y, r.w, r.h, color, shownP(i, e));
    if (glow > 0) glows.push([r, glow]);
  }
  // gold outline on the cells changed by the last answer
  if (S.last) {
    g.strokeStyle = hl;
    g.lineWidth = 1.2;
    for (const i of S.last.changed) {
      const r = cellRect(i);
      const s = Math.min(1, 0.35 + 0.65 * rampT(S.last.tv[i]));
      g.globalAlpha = 0.25 * s;
      g.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
    }
    g.globalAlpha = 1;
    const r = cellRect(S.last.column);
    g.strokeStyle = ans;
    g.lineWidth = 1.5;
    g.strokeRect(r.x - 1, r.y - 1, r.w + 2, r.h + 2);
  }
  if (glows.length) {
    g.globalCompositeOperation = "lighter";
    for (const [r, a] of glows) {
      const cx = r.x + r.w / 2, cy = r.y + r.h / 2, R = r.w * (0.7 + 0.5 * (1 - a));
      const grd = g.createRadialGradient(cx, cy, 0, cx, cy, R);
      grd.addColorStop(0, `rgba(255,214,120,${0.55 * a})`);
      grd.addColorStop(1, "rgba(255,214,120,0)");
      g.fillStyle = grd;
      g.fillRect(cx - R, cy - R, 2 * R, 2 * R);
    }
    g.globalCompositeOperation = "source-over";
  }
  for (const i of [S.spotCol, S.hover]) {
    if (i == null) continue;
    const r = cellRect(i);
    g.strokeStyle = "#ffffff";
    g.lineWidth = 1.5;
    g.strokeRect(r.x - 1.5, r.y - 1.5, r.w + 3, r.h + 3);
  }
  drawMind(now);
  if (animating) kick();
  else if (S.morph) { S.morph = null; drawMind(now); }
}
function kick() { if (!raf) raf = requestAnimationFrame(draw); }

// ---------------------------------------------------------------------------
// current mind: one spoke per item, length/colour = distance from Psi0
// ---------------------------------------------------------------------------

const mind = $("mind");
const mctx = mind.getContext("2d");
const MIND = { size: 184, r0: 56, r1: 89, hover: null };

function setupMind() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  mind.width = MIND.size * dpr;
  mind.height = MIND.size * dpr;
  mctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  mind.addEventListener("pointermove", (e) => {
    const r = mind.getBoundingClientRect();
    const x = e.clientX - r.left - MIND.size / 2, y = e.clientY - r.top - MIND.size / 2;
    const d = Math.hypot(x, y);
    let i = null;
    if (d > MIND.r0 - 8 && d < MIND.r1 + 6) {
      let a = Math.atan2(y, x) + Math.PI / 2;
      if (a < 0) a += 2 * Math.PI;
      i = Math.min(S.n - 1, Math.floor(a / (2 * Math.PI) * S.n));
    }
    if (i !== MIND.hover) { MIND.hover = i; S.hover = i; renderMindText(); kick(); }
  });
  mind.addEventListener("pointerleave", () => { MIND.hover = null; S.hover = null; renderMindText(); kick(); });
  mind.addEventListener("click", () => { if (MIND.hover != null) ask(MIND.hover); });
}

function drawMind(now) {
  if (!S.n || !S.drift.length) return;
  const g = mctx, c = MIND.size / 2, { r0, r1 } = MIND;
  g.clearRect(0, 0, MIND.size, MIND.size);
  g.beginPath();
  g.arc(c, c, r0 - 1, 0, 2 * Math.PI);
  g.strokeStyle = "rgba(255,255,255,0.08)";
  g.lineWidth = 1;
  g.stroke();
  const ans = css("--ans");
  g.lineWidth = Math.max(1, (2 * Math.PI * r1) / S.n * 0.9);
  for (let i = 0; i < S.n; i++) {
    const a = (i / S.n) * 2 * Math.PI - Math.PI / 2;
    const { e } = morphOf(i, now);
    const rec = observedRecord(i);
    const d = shownDrift(i, e);
    const t = rec ? 1 : rampT(d);
    const len = 2 + t * (r1 - r0 - 2);
    g.strokeStyle = rec ? ans : rampColor(d);
    const cs = Math.cos(a), sn = Math.sin(a);
    g.beginPath();
    g.moveTo(c + r0 * cs, c + r0 * sn);
    g.lineTo(c + (r0 + len) * cs, c + (r0 + len) * sn);
    g.stroke();
  }
  if (MIND.hover != null) {
    const a = (MIND.hover / S.n) * 2 * Math.PI - Math.PI / 2;
    g.strokeStyle = "#ffffff";
    g.lineWidth = 1.5;
    g.beginPath();
    g.moveTo(c + (r0 - 6) * Math.cos(a), c + (r0 - 6) * Math.sin(a));
    g.lineTo(c + (r1 + 4) * Math.cos(a), c + (r1 + 4) * Math.sin(a));
    g.stroke();
  }
}

function renderMindText() {
  const el = $("mind-center");
  if (MIND.hover != null) {
    const v = S.vars[MIND.hover];
    const rec = observedRecord(MIND.hover);
    el.innerHTML = `<span class="mc-var">${esc(v.variable)}</span><span>${esc(label(v)).slice(0, 60)}</span>` +
      `<span>${rec ? `answered: ${esc(rec.value)}` : `${fmt(S.drift[MIND.hover])} from Ψ₀`}</span>`;
    return;
  }
  let shifted = 0;
  for (let i = 0; i < S.n; i++) if (!observedRecord(i) && S.drift[i] > 0.001) shifted++;
  el.innerHTML = S.observed.length
    ? `<b>${shifted}</b><span>views shifted by more than 0.001 after ${S.observed.length} answer${S.observed.length > 1 ? "s" : ""}</span>`
    : `<b>0</b><span>views shifted: still at Ψ₀</span>`;
  $("asked").innerHTML = S.observed.length
    ? S.observed.map((o) => `<li title="Question ${o.question}"><span class="mono">${esc(o.variable)}</span> = <b>${esc(o.value)}</b></li>`).join("")
    : `<li class="none">No answers yet</li>`;
}

// ---------------------------------------------------------------------------
// the largest updates pop out one by one
// ---------------------------------------------------------------------------

function cancelSpots() {
  S.spotToken++;
  S.spotCol = null;
  $("spot-layer").innerHTML = "";
  for (const li of document.querySelectorAll(".top-list li.on")) li.classList.remove("on");
  kick();
}

function spotHTML(i) {
  const v = S.vars[i];
  const L = S.last;
  const pb = L.before[i], pa = S.psi[i];
  const idx = v.categories.map((_, k) => k)
    .sort((a, b) => Math.max(pa[b], pb[b]) - Math.max(pa[a], pb[a]) || a - b);
  const shown = idx.slice(0, 5);
  const color = rampColor(S.drift[i]);
  const rows = shown.map((k) =>
    `<div class="bar-row"><span class="lab" title="${esc(v.categories[k])}">${esc(v.categories[k])}</span>` +
    `<span class="track"><span class="ghost" style="width:${(100 * pb[k]).toFixed(2)}%"></span>` +
    `<span class="fill" data-to="${(100 * pa[k]).toFixed(2)}" data-color="${color}" style="width:${(100 * pb[k]).toFixed(2)}%"></span></span>` +
    `<span class="val">${fmt(pb[k], 2)} → ${fmt(pa[k], 2)}</span></div>`).join("");
  return `<div class="s-head"><span class="s-var">${esc(v.variable)}</span><span class="s-tv">changed by ${fmt(L.tv[i])}</span></div>
    <div class="s-label">${esc(label(v))}</div>
    <div class="bars">${rows}</div>
    ${idx.length > shown.length ? `<div class="s-more">+${idx.length - shown.length} smaller answers</div>` : ""}`;
}

// Pops item i out of the grid; resolves when the next card may start.
function spotlight(i, token) {
  const layer = $("spot-layer");
  const card = document.createElement("div");
  card.className = "spot";
  card.innerHTML = spotHTML(i);
  card.style.visibility = "hidden";
  layer.appendChild(card);
  const cw = card.offsetWidth, ch = card.offsetHeight;
  const r = cellRect(i);
  let fx = r.x + r.w + 12, fy = r.y - ch / 2 + r.h / 2;
  if (fx + cw > G.w) fx = r.x - cw - 12;
  if (fx < 0) fx = Math.max(0, Math.min(G.w - cw, r.x - cw / 2));
  fy = Math.max(0, Math.min(G.h - ch, fy));
  card.style.left = `${fx}px`;
  card.style.top = `${fy}px`;
  const small = `translate(${r.x - fx}px, ${r.y - fy}px) scale(${r.w / cw}, ${r.h / ch})`;
  card.style.transform = small;
  card.style.opacity = "0";
  card.style.visibility = "visible";
  void card.offsetWidth;
  card.style.transform = "none";
  card.style.opacity = "1";
  // bars slide from the old to the new distribution, in the new colour
  for (const f of card.querySelectorAll(".fill")) {
    f.style.width = `${f.dataset.to}%`;
    f.style.backgroundColor = f.dataset.color;
  }
  S.spotCol = i;
  kick();
  const hold = reduceMotion ? 900 : 1150;
  setTimeout(() => {
    card.classList.add("out");
    card.style.transform = small;
    card.style.opacity = "0";
    setTimeout(() => card.remove(), 300);
    if (token === S.spotToken && S.spotCol === i) { S.spotCol = null; kick(); }
  }, hold);
  return sleep(reduceMotion ? 950 : 820);           // next card overlaps this one's exit
}

async function runSpots() {
  cancelSpots();
  const token = S.spotToken;
  if (!S.last) return;
  const L = S.last;
  const list = L.changed.filter((i) => L.tv[i] >= SPOT_MIN_TV).slice(0, SPOT_MAX);
  $("btn-replay").disabled = !list.length;
  for (const i of list) {
    if (token !== S.spotToken) return;
    for (const li of document.querySelectorAll(".top-list li")) li.classList.toggle("on", +li.dataset.col === i);
    await spotlight(i, token);
  }
  if (token !== S.spotToken) return;
  await sleep(400);
  for (const li of document.querySelectorAll(".top-list li.on")) li.classList.remove("on");
}

// ---------------------------------------------------------------------------
// hover
// ---------------------------------------------------------------------------

function showTip(i, px, py) {
  const v = S.vars[i];
  const rec = observedRecord(i);
  const changedByLast = S.last && S.last.tv[i] > MOVED && i !== S.last.column;
  const p = S.psi[i];
  const pb = changedByLast ? S.last.before[i] : p;
  const color = rec ? "var(--ans)" : rampColor(S.drift[i]);
  const idx = v.categories.map((_, k) => k).sort((a, b) => p[b] - p[a] || a - b).slice(0, 6);
  const rows = idx.map((k) =>
    `<div class="bar-row"><span class="lab">${esc(v.categories[k])}</span>` +
    `<span class="track">${changedByLast ? `<span class="ghost" style="opacity:.9;width:${(100 * pb[k]).toFixed(2)}%"></span>` : ""}` +
    `<span class="fill" style="width:${(100 * p[k]).toFixed(2)}%;background:${color}"></span></span>` +
    `<span class="val">${fmt(p[k], 2)}</span></div>`).join("");
  const tip = $("tooltip");
  tip.innerHTML = `<div class="s-var">${esc(v.variable)}</div><div class="s-label">${esc(label(v))}</div>
    <div class="bars">${rows}</div>
    <div class="t-hint">${rec ? `answered in Q${rec.question}: “${esc(rec.value)}”`
      : `${fmt(S.drift[i])} from Ψ₀${changedByLast ? ` · last answer moved it ${fmt(S.last.tv[i])}` : ""} · click to ask`}</div>`;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${px + 14 + w > G.w ? px - w - 14 : px + 14}px`;
  tip.style.top = `${Math.max(0, Math.min(G.h - h, py + 14))}px`;
}

function setupGrid() {
  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const i = cellAt(px, py);
    if (i !== S.hover) { S.hover = i; kick(); }
    if (i == null) $("tooltip").hidden = true; else showTip(i, px, py);
  });
  canvas.addEventListener("pointerleave", () => { S.hover = null; $("tooltip").hidden = true; kick(); });
  canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect();
    const i = cellAt(e.clientX - r.left, e.clientY - r.top);
    if (i != null) ask(i);
  });
  new ResizeObserver(() => layoutGrid()).observe($("grid-wrap"));
}

// ---------------------------------------------------------------------------
// left panel: search, or the questions to ask next
// ---------------------------------------------------------------------------

function searchVars(q) {
  q = q.trim().toLowerCase();
  const out = [];
  for (const v of S.vars) {
    if (!v.learned) continue;
    const name = v.variable.toLowerCase();
    let score = 0, hit = "";
    if (name === q) score = 100;
    else if (name.startsWith(q)) score = 80;
    else if (v._hay.includes(q)) score = 60 - Math.min(20, v._hay.indexOf(q) / 4);
    else {
      const j = v._cats.findIndex((c) => c.includes(q));
      if (j >= 0) { score = 30; hit = `answer: “${v.categories[j]}”`; }
    }
    if (score) out.push({ v, hit, score });
  }
  out.sort((a, b) => b.score - a.score || a.v.column - b.v.column);
  return out.slice(0, 60);
}

// Unasked questions, most shifted by the answers so far first.
function nextVars() {
  if (!S.observed.length) {
    const sug = SUGGESTED.map((name) => S.vars.find((v) => v.variable.toLowerCase() === name))
      .filter((v) => v && v.learned);
    if (sug.length >= 6) return sug.map((v) => ({ v, hit: "" }));
    // other surveys: labelled questions with a handful of answers whose
    // answer is genuinely open at Psi0 (normalised entropy) and that many
    // other items depend on; administrative items are near-certain at Psi0.
    const score = (v) => {
      const p = S.psi0[v.column];
      const h = -p.reduce((a, x) => a + (x > 0 ? x * Math.log(x) : 0), 0) / Math.log(p.length);
      return h * Math.log1p(v.out_degree);
    };
    return S.vars.filter((v) => v.learned && v.label && v.categories.length >= 2 && v.categories.length <= 10)
      .map((v) => [score(v), v]).sort((a, b) => b[0] - a[0] || a[1].column - b[1].column)
      .slice(0, NEXT_MAX).map(([, v]) => ({ v, hit: "" }));
  }
  const idx = [];
  for (let i = 0; i < S.n; i++) if (S.vars[i].learned && !observedRecord(i) && S.drift[i] > MOVED) idx.push(i);
  idx.sort((a, b) => S.drift[b] - S.drift[a]);
  return idx.slice(0, NEXT_MAX).map((i) => ({ v: S.vars[i], hit: "" }));
}

function miniBars(i) {
  const p = S.psi[i], ord = S.order[i].slice(0, 12), sc = S.scale[i];
  const color = rampColor(S.drift[i]);
  return `<span class="mini" aria-hidden="true">${ord.map((k) =>
    `<i style="height:${Math.max(1, Math.min(1, p[k] / sc) * 22).toFixed(1)}px;background:${color}"></i>`).join("")}</span>`;
}

function renderResults() {
  const searching = S.search.trim().length > 0;
  const list = searching ? searchVars(S.search) : nextVars();
  $("results-head").textContent = searching
    ? `${list.length}${list.length === 60 ? "+" : ""} matching questions` : "Ask next";
  $("results-note").textContent = searching ? ""
    : S.observed.length ? "Unasked questions whose distributions the answers so far have moved most. Click one to ask it."
      : S.vars.some((v) => SUGGESTED.includes(v.variable.toLowerCase()))
        ? "Nothing asked yet. Start with one of these."
        : "Nothing asked yet. These are open questions that many other items depend on.";
  $("results").innerHTML = list.length ? list.map(({ v, hit }) => {
    const done = observedRecord(v.column);
    const i = v.column;
    return `<li><button data-col="${i}" ${done || S.busy ? "disabled" : ""}>` +
      `<span class="r-var">${esc(v.variable)}${done ? `<span class="tag done">answered: ${esc(done.value)}</span>` : ""}</span>` +
      miniBars(i) +
      `<span class="r-label${v.label ? "" : " muted"}">${esc(v.label || "no question text")}</span>` +
      (hit ? `<span class="r-hit">${esc(hit)}</span>`
        : S.drift[i] > MOVED ? `<span class="r-shift">${fmt(S.drift[i])} from Ψ₀</span>` : "") +
      `</button></li>`;
  }).join("") : `<li class="empty">No modelled question matches.</li>`;
}

function renderGridHead() {
  $("grid-title").textContent = `Ψ: all ${S.n.toLocaleString()} marginal distributions`;
  const n = S.observed.length;
  $("grid-sub").textContent = n === 0
    ? "Each sparkline is one survey item's response histogram, currently at Ψ₀. Hover to read one; click to ask it."
    : `After ${n} answer${n > 1 ? "s" : ""}. Colour = how far each distribution now is from Ψ₀; gold outlines mark the ones the last answer changed; red = answered.`;
  $("btn-replay").disabled = !S.last || !S.last.changed.some((i) => S.last.tv[i] >= SPOT_MIN_TV);
}

function renderAll() {
  renderQuestion({ phase: "idle" });
  renderResults();
  renderGridHead();
  renderMindText();
  layoutGrid();
}

function showLoading(t) { $("loading-text").textContent = t; $("loading").hidden = false; }
function hideLoading() { $("loading").hidden = true; }

// ---------------------------------------------------------------------------
// survey picker: country + year -> public DTAG model (downloaded on demand)
// ---------------------------------------------------------------------------

// Same ranking as DTAG's recommender: time fit (within a year, within three,
// further), then strength of geographic coverage, then distance, then recency.
const GEO_STRENGTH = { gss: 2.5, afrobarometer: 3, eurobarometer: 3, wvs: 1 };
const PICK = { country: null, year: null, choice: null, polling: 0 };

function periodText(m) {
  if (m.date) return m.date;
  if (!m.period) return "dates unknown";
  return m.period[0] === m.period[1] ? `${m.period[0]}` : `${m.period[0]}–${m.period[1]}`;
}
function mb(bytes) { return bytes ? `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB` : ""; }

function candidates(ckey, year) {
  const cat = S.catalog.models;
  const keys = (S.catalog.countries[ckey] || {}).models || [];
  const cmp = (a, b) => { for (let i = 0; i < a.sk.length; i++) if (a.sk[i] !== b.sk[i]) return a.sk[i] - b.sk[i]; return 0; };
  const best = {};
  for (const key of keys) {
    const m = cat[key];
    if (!m || !m.period) continue;
    let d = 0;
    if (year != null) d = year < m.period[0] ? m.period[0] - year : year > m.period[1] ? year - m.period[1] : 0;
    if (m.family === "eurobarometer" && !m.auto_select) d += 50;   // cumulative files last
    const bucket = d <= 1 ? 0 : d <= 3 ? 1 : 2;
    const sk = [bucket, -(GEO_STRENGTH[m.family] || 0), d, -m.period[1]];
    const cand = { key, m, d, sk };
    if (!best[m.family] || cmp(cand, best[m.family]) < 0) best[m.family] = cand;
  }
  return Object.values(best).sort(cmp);
}

function yearsFor(ckey) {
  const ys = new Set();
  for (const key of (S.catalog.countries[ckey] || {}).models || []) {
    const m = S.catalog.models[key];
    if (!m || !m.period || (m.family === "eurobarometer" && !m.auto_select)) continue;
    for (let y = m.period[0]; y <= m.period[1]; y++) ys.add(y);
  }
  return [...ys].sort((a, b) => b - a);
}

function setupSurveyPicker() {
  const cs = Object.entries(S.catalog.countries).sort((a, b) => a[1].name.localeCompare(b[1].name));
  $("sv-country").innerHTML = cs.map(([k, v]) => `<option value="${esc(k)}">${esc(v.name)}</option>`).join("");
  $("sv-country").addEventListener("change", () => { PICK.country = $("sv-country").value; PICK.year = null; renderYears(); });
  $("sv-year").addEventListener("change", () => {
    PICK.year = $("sv-year").value === "" ? null : +$("sv-year").value;
    PICK.choice = null;
    renderCandidates();
  });
  $("sv-cands").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-key]");
    if (!b) return;
    PICK.choice = b.dataset.key;
    renderCandidates();
  });
  $("sv-load").addEventListener("click", () => { if (PICK.choice) loadSurvey(PICK.choice); });
}

function renderYears() {
  const ys = yearsFor(PICK.country);
  $("sv-year").innerHTML = `<option value="">latest</option>` +
    ys.map((y) => `<option value="${y}" ${y === PICK.year ? "selected" : ""}>${y}</option>`).join("");
  PICK.choice = null;
  renderCandidates();
}

function renderCandidates() {
  const list = candidates(PICK.country, PICK.year);
  if (!PICK.choice || !list.some((c) => c.key === PICK.choice)) PICK.choice = list.length ? list[0].key : null;
  $("sv-cands").innerHTML = list.map(({ key, m, d }) => {
    const pooled = m.family !== "gss" && m.countries.length > 1 ? ` · pooled, ${m.countries.length} countries` : "";
    const off = d % 50;
    const fit = PICK.year == null || off === 0 ? "" : ` · ${off} yr from ${PICK.year}`;
    const state = key === S.modelKey ? "current" : m.loaded ? "loaded" : m.installed ? "on disk" : mb(m.archive_bytes);
    return `<button class="sv-cand${key === S.modelKey ? " current" : ""}" role="radio" aria-checked="${key === PICK.choice}" data-key="${esc(key)}"
      title="${esc(m.family_label)}: ${esc(periodText(m))}${esc(pooled)}">${esc(m.label)}<small>${esc(periodText(m))}${esc(fit)}${esc(pooled)} · ${esc(state)}</small></button>`;
  }).join("") || `<span class="sv-meta">No survey model covers this country.</span>`;
  const btn = $("sv-load");
  btn.disabled = !PICK.choice || PICK.polling > 0 || S.busy;
  btn.title = S.busy ? "Wait for the current answer to finish" : "";
  btn.textContent = PICK.choice === S.modelKey ? "Restart at Ψ₀" : "Load survey";
}

function syncPickerToModel() {
  const m = S.model && S.model.info;
  if (!m) return;
  if (!PICK.country || !(m.countries || []).includes(PICK.country)) {
    PICK.country = (m.countries || [])[0] || Object.keys(S.catalog.countries)[0];
  }
  $("sv-country").value = PICK.country;
  renderYears();
  if (m.period) {
    PICK.year = m.period[1];
    $("sv-year").value = String(PICK.year);
  }
  PICK.choice = S.modelKey;
  renderCandidates();
}

function renderSurveyHead() {
  const m = (S.model && S.model.info) || {};
  const learned = S.vars.filter((v) => v.learned).length;
  $("sv-name").textContent = m.label || S.modelKey;
  const pooled = m.family && m.family !== "gss" && (m.countries || []).length > 1
    ? ` · pooled across ${m.countries.length} countries (country is one of its questions)` : "";
  $("sv-meta").textContent = `${m.family_label || ""} · ${periodText(m)} · ${learned.toLocaleString()} questions${pooled}`;
  $("title-model").textContent = `· ${m.label || S.modelKey} Large Science Model`;
  document.title = `PsiSim · ${m.label || S.modelKey}`;
}

async function loadSurvey(key) {
  if (S.busy || PICK.polling) return;
  const token = ++PICK.polling;
  const prog = $("sv-progress"), fill = $("sv-bar-fill"), text = $("sv-progress-text");
  const [fam, name] = key.split("/");
  const show = (frac, msg) => { prog.hidden = false; fill.style.width = `${Math.round(100 * frac)}%`; text.textContent = msg; };
  renderCandidates();
  try {
    let st = await (await api(`/api/models/${encodeURIComponent(fam)}/${encodeURIComponent(name)}/load`, { method: "POST" })).json();
    while (st.state !== "ready") {
      if (st.state === "error") throw new Error(st.message || "could not load the model");
      if (st.state === "downloading") {
        show(st.progress || 0, `Downloading ${S.catalog.models[key].label} from the public model release… ` +
          (st.total_bytes ? `${mb(st.done_bytes)} of ${mb(st.total_bytes)}` : ""));
      } else {
        show(1, `${st.state === "loading" ? "Computing Ψ₀ from the empty survey" : "Preparing"}…`);
      }
      await sleep(400);
      st = await (await api(`/api/models/${encodeURIComponent(fam)}/${encodeURIComponent(name)}/status`)).json();
    }
    show(1, "Starting a session at Ψ₀…");
    const old = S.sid;
    cancelSpots();
    await newSession(null, key);
    if (old) api(`/api/session/${encodeURIComponent(old)}`, { method: "DELETE" }).catch(() => {});
    S.catalog.models[key].loaded = S.catalog.models[key].installed = true;
    prog.hidden = true;
  } catch (err) {
    show(0, `Could not load: ${err.message}`);
  } finally {
    if (token === PICK.polling) PICK.polling = 0;
    renderCandidates();
  }
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

function wire() {
  $("search").addEventListener("input", (e) => { S.search = e.target.value; renderResults(); });
  $("search").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const first = $("results").querySelector("button[data-col]:not(:disabled)");
    if (first) ask(+first.dataset.col);
  });
  $("results").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-col]");
    if (b && !b.disabled) ask(+b.dataset.col);
  });
  $("question").addEventListener("pointerover", (e) => {
    const li = e.target.closest(".top-list li[data-col]");
    const i = li ? +li.dataset.col : null;
    if (i !== S.hover) { S.hover = i; kick(); }
  });
  $("btn-replay").addEventListener("click", () => {
    if (!S.last) return;
    const prevDrift = Float32Array.from(S.drift);
    for (const i of S.last.changed) prevDrift[i] = tv(S.last.before[i], S.psi0[i]);
    startMorph(S.last.before, prevDrift);
    setTimeout(runSpots, reduceMotion ? 0 : 450);
  });
  $("btn-new").addEventListener("click", async () => {
    if (S.busy) return;
    const raw = prompt("Start a new session from Ψ₀.\nRandom seed for drawing answers (blank = random):", "");
    if (raw === null) return;
    const seed = raw.trim() === "" ? null : Number(raw.trim());
    if (seed != null && !(Number.isInteger(seed) && seed >= 0)) { alert("The seed must be a non-negative integer."); return; }
    const old = S.sid;
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
  });
}

async function main() {
  wire();
  setupGrid();
  setupMind();
  showLoading("Loading the survey catalog and model…");
  try {
    S.catalog = await (await api("/api/catalog")).json();
    setupSurveyPicker();
    await loadModel(S.catalog.default_model || "gss/gss_2018");
    await restoreOrCreate();
    syncPickerToModel();
    hideLoading();
  } catch (err) {
    showLoading(`Could not start: ${err.message}`);
    document.querySelector("#loading .spinner").hidden = true;
  }
}

main();
