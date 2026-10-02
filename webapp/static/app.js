/* PsiSim webapp client.
 *
 * Shows that asking one question updates the whole state Psi. Every item's
 * response distribution p_i is a sparkline in one grid. Asking item i draws
 * sigma ~ p_i on the server, which applies one native hard observation: p_i
 * becomes the point mass delta_sigma (red) and the centred change is passed
 * once through the learned links. The largest resulting updates then pop out
 * of the grid one by one (before = grey outline, after = blue) and shrink
 * back into their sparklines. The browser never computes PsiSim dynamics.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (x, d = 3) => (x == null || !isFinite(x) ? "—" : Number(x).toFixed(d));
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
};

const SUGGESTED = [
  "polviews", "partyid", "abany", "cappun", "grass", "gunlaw", "god",
  "homosex", "natenvir", "fefam", "trust", "happy", "attend", "wrkstat",
];
// The streamed probabilities are rounded to 1e-6; TV above this counts as a change.
const MOVED = 1.25e-6;
const SPOT_MAX = 8;          // largest updates shown one by one
const SPOT_MIN_TV = 0.002;   // ... if they moved at least this much

const S = {
  model: null, vars: [], n: 0, psi0: [],
  psi: [], sid: null, seed: null,
  observed: [],              // [{column, variable, value, question, u}]
  answers: [],               // [{question, column, variable, value, u, stats, tv}]
  last: null,                // most recent answer: {column, value, u, before, tv, changed}
  busy: false,
  search: "",
  order: [], scale: [],      // per item: category display order and sparkline scale
  spotToken: 0,
  spotCol: null,             // item currently popped out
  hover: null,
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

async function loadModel() {
  const m = await (await api("/api/model")).json();
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
}

function applySession(view) {
  S.sid = view.session_id;
  S.seed = view.seed;
  S.psi = view.psi;
  S.observed = view.observed || [];
  const hist = view.history || [];
  S.answers = hist.map((h) => ({
    question: h.question, column: h.column, variable: h.variable, value: h.value,
    u: (S.observed.find((o) => o.question === h.question) || {}).u,
    stats: h.stats, tv: Float32Array.from(h.tv),
  }));
  S.last = null;
  if (hist.length) {
    const h = hist[hist.length - 1];
    const before = hist.length > 1 ? hist[hist.length - 2].psi : S.psi0;
    S.last = makeLast(h.column, h.value, (S.observed.find((o) => o.question === h.question) || {}).u,
      before, Float32Array.from(h.tv), h.question);
  }
  store.set("psisim.sid", S.sid);
  $("session-chip").textContent = `seed ${S.seed}`;
}

function makeLast(column, value, u, before, tvArr, question) {
  const changed = [];
  for (let i = 0; i < S.n; i++) if (i !== column && tvArr[i] > MOVED) changed.push(i);
  changed.sort((a, b) => tvArr[b] - tvArr[a]);
  return { column, value, u, before, tv: tvArr, changed, question };
}

async function newSession(seed) {
  showLoading("Starting a new session at Ψ₀…");
  try {
    const body = JSON.stringify(seed != null ? { seed } : {});
    applySession(await (await api("/api/session", { method: "POST", body })).json());
  } finally { hideLoading(); }
  cancelSpots();
  renderAll();
}

async function restoreOrCreate() {
  const sid = store.get("psisim.sid");
  if (sid) {
    try {
      applySession(await (await api(`/api/session/${encodeURIComponent(sid)}`)).json());
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
  if (!v || !v.learned || observedRecord(col)) return;
  if (S.busy) return;
  S.busy = true;
  cancelSpots();
  renderResults();
  const before = S.psi;
  renderQuestion({ phase: "asking", column: col, p: before[col] });
  renderSummaryPending();

  let answer = null, frame = null, done = null;
  try {
    const res = await api(`/api/session/${encodeURIComponent(S.sid)}/answer`, {
      method: "POST",
      body: JSON.stringify({ column: col, max_steps: 0 }),
    });
    const text = await res.text();
    for (const line of text.split("\n")) {
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
    return;
  }

  // 1. draw the answer from p_i (animation replays the server's seeded draw)
  await drawAnimation();
  // 2. p_i collapses to the point mass delta_sigma
  collapseAnimation(answer.value, answer.u);
  await sleep(reduceMotion ? 100 : 900);

  // 3. Psi updates everywhere
  const after = before.slice();
  for (const [c, arr] of Object.entries(frame.changed)) after[+c] = arr;
  S.psi = after;
  const rec = { column: col, variable: v.variable, value: answer.value, question: answer.question, u: answer.u };
  S.observed = S.observed.concat([rec]);
  const tvArr = Float32Array.from(frame.tv);
  S.answers = S.answers.concat([{ ...rec, stats: done.stats, tv: tvArr }]);
  S.last = makeLast(col, answer.value, answer.u, before, tvArr, answer.question);
  pulseChanged();
  renderHistory();
  renderResults();
  renderSummary();
  renderGridHead();
  S.busy = false;

  // 4. the largest updates pop out one by one
  runSpots();
}

async function drawAnimation() {
  const rows = [...document.querySelectorAll("#question .bars .bar-row[data-cat]")];
  if (!rows.length || reduceMotion) return;
  const weights = rows.map((r) => +r.dataset.p || 0);
  const total = weights.reduce((a, b) => a + b, 0) || 1;
  let delay = 50;
  const t0 = performance.now();
  while (performance.now() - t0 < 1000) {
    let u = Math.random() * total, k = 0;
    while (k < weights.length - 1 && (u -= weights[k]) > 0) k++;
    rows.forEach((r, j) => r.classList.toggle("spin", j === k));
    await sleep(delay);
    delay *= 1.14;
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
function barRows(col, p, opts = {}) {
  const v = S.vars[col];
  const max = opts.max || 10;
  const idx = v.categories.map((_, k) => k).sort((a, b) => p[b] - p[a] || a - b);
  const shown = idx.slice(0, max);
  const rest = idx.slice(max);
  const rows = shown.map((k) =>
    `<div class="bar-row" data-cat="${esc(v.categories[k])}" data-p="${p[k]}">` +
    `<span class="lab" title="${esc(v.categories[k])}">${esc(v.categories[k])}</span>` +
    `<span class="track"><span class="ghost" style="width:${(100 * p[k]).toFixed(2)}%"></span>` +
    `<span class="fill" style="width:${(100 * p[k]).toFixed(2)}%"></span></span>` +
    `<span class="val">${fmt(p[k])}</span></div>`);
  if (rest.length) {
    const r = rest.reduce((a, k) => a + p[k], 0);
    rows.push(`<div class="bar-row" data-p="${r}"><span class="lab muted">${rest.length} other answers</span>` +
      `<span class="track"><span class="ghost" style="width:${(100 * r).toFixed(2)}%"></span>` +
      `<span class="fill" style="width:${(100 * r).toFixed(2)}%"></span></span><span class="val">${fmt(r)}</span></div>`);
  }
  return rows.join("");
}

function renderQuestion(st) {
  const el = $("question");
  if (!st || st.phase === "idle") {
    if (S.last) { renderQuestion({ phase: "result" }); return; }
    el.innerHTML = `<div class="q-intro">
      <div class="q-kicker">No answers yet</div>
      <p><b>No answers have been supplied. Every sparkline below is showing the response distribution implied by the GSS 2018 LSM from the empty state.</b></p>
      <p>Choose a question on the left (or click any sparkline). An answer is drawn from its current distribution, that distribution becomes a single red bar, and the change is passed to every related question. Ψ<sub>0</sub> does not move until something is asked.</p>
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
    collapseInstant(S.last.value);
    renderSummary();
  }
}

function collapseInstant(value) {
  const bars = document.querySelector("#question .bars");
  if (!bars) return;
  for (const f of bars.querySelectorAll(".fill")) f.style.transition = "none";
  collapseAnimation(value, S.last ? S.last.u : null);
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
  const top = L.changed.slice(0, SPOT_MAX);
  el.innerHTML = `
    <div class="sum-big"><b>${L.changed.length}</b><span>of ${others} other distributions changed</span></div>
    <div class="sum-sub">${n001} by more than 0.01 (total variation)${L.changed.length ? ` · largest ${fmt(L.tv[L.changed[0]])}` : ""}.
      The answer is passed once through the learned links to every question whose model uses <span class="mono">${esc(S.vars[L.column].variable)}</span>.</div>
    <ol class="top-list">${top.map((i) => {
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
// sparkline grid
// ---------------------------------------------------------------------------

const canvas = $("grid");
const ctx = canvas.getContext("2d");
const G = { cols: 1, rows: 1, cw: 30, ch: 18, gap: 3, w: 0, h: 0, dpr: 1 };
const pulses = new Map();   // col -> start time of the update pulse
let raf = 0;

function layoutGrid() {
  const wrap = $("grid-wrap");
  const W = wrap.clientWidth;
  const narrow = window.matchMedia("(max-width: 900px)").matches;
  const H = narrow ? Infinity : wrap.clientHeight;
  const n = S.n, gap = 3, aspect = 1.7;
  let best = null;
  for (let cols = 8; cols <= 120; cols++) {
    const cw = Math.floor((W - gap * (cols - 1)) / cols);
    const ch = Math.max(10, Math.round(cw / aspect));
    const rows = Math.ceil(n / cols);
    const need = rows * ch + (rows - 1) * gap;
    if (cw < 18) break;
    if (need <= H) { best = { cols, cw, ch, rows }; break; }
    best = { cols, cw, ch, rows };
  }
  if (narrow) {
    const cols = Math.max(8, Math.floor((W + gap) / (34 + gap)));
    const cw = Math.floor((W - gap * (cols - 1)) / cols);
    best = { cols, cw, ch: Math.round(cw / aspect), rows: Math.ceil(n / cols) };
    wrap.style.height = `${best.rows * (best.ch + gap)}px`;
  } else {
    wrap.style.height = "";
  }
  Object.assign(G, best, { gap, w: W, h: wrap.clientHeight });
  G.dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(G.w * G.dpr);
  canvas.height = Math.round(G.h * G.dpr);
  draw();
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

function drawSpark(g, i, x, y, w, h, colors) {
  const p = S.psi[i];
  const ord = S.order[i];
  const k = ord.length;
  const pad = 2;
  const iw = w - 2 * pad, ih = h - 2 * pad;
  const bw = iw / k;
  const rec = observedRecord(i);
  const sc = rec ? 1 : S.scale[i];
  g.fillStyle = rec ? colors.ans : colors.cur;
  for (let j = 0; j < k; j++) {
    const q = p[ord[j]];
    if (q <= 0) continue;
    const bh = Math.min(1, q / sc) * ih;
    const bx = x + pad + j * bw;
    g.fillRect(bx, y + pad + ih - bh, Math.max(0.8, bw - (bw >= 3 ? 1 : 0)), Math.max(0.6, bh));
  }
}

function draw(now = performance.now()) {
  raf = 0;
  if (!S.n || !S.psi.length) return;
  const colors = { cur: css("--cur"), ans: css("--ans"), hl: css("--hl") };
  const g = ctx;
  g.setTransform(G.dpr, 0, 0, G.dpr, 0, 0);
  g.clearRect(0, 0, G.w, G.h);
  let animating = false;
  for (let i = 0; i < S.n; i++) {
    const { x, y, w, h } = cellRect(i);
    if (y > G.h) break;
    g.fillStyle = "rgba(255,255,255,0.035)";
    g.fillRect(x, y, w, h);
    drawSpark(g, i, x, y, w, h, colors);
  }
  // cells updated by the last answer: a pulse, then a quiet marker
  if (S.last) {
    for (const i of S.last.changed) {
      const t0 = pulses.get(i);
      const strength = Math.min(1, 0.35 + 0.65 * Math.max(0, (Math.log10(S.last.tv[i]) + 4) / 3.3));
      let a = 0.28 * strength;
      if (t0 != null) {
        const t = (now - t0) / 1800;
        if (t < 1) { a = Math.max(a, (1 - t) * strength); animating = true; }
      }
      const { x, y, w, h } = cellRect(i);
      g.strokeStyle = colors.hl;
      g.globalAlpha = a;
      g.lineWidth = 1.2;
      g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    }
    g.globalAlpha = 1;
    const { x, y, w, h } = cellRect(S.last.column);
    g.strokeStyle = colors.ans;
    g.lineWidth = 1.5;
    g.strokeRect(x - 1, y - 1, w + 2, h + 2);
  }
  for (const i of [S.spotCol, S.hover]) {
    if (i == null) continue;
    const { x, y, w, h } = cellRect(i);
    g.strokeStyle = "#ffffff";
    g.lineWidth = 1.5;
    g.strokeRect(x - 1.5, y - 1.5, w + 3, h + 3);
  }
  if (animating && !reduceMotion) kick();
}
function kick() { if (!raf) raf = requestAnimationFrame(draw); }

function pulseChanged() {
  const t = performance.now();
  pulses.clear();
  if (S.last) for (const i of S.last.changed) pulses.set(i, t);
  kick();
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
  const rows = shown.map((k) =>
    `<div class="bar-row"><span class="lab" title="${esc(v.categories[k])}">${esc(v.categories[k])}</span>` +
    `<span class="track"><span class="ghost" style="width:${(100 * pb[k]).toFixed(2)}%"></span>` +
    `<span class="fill" style="width:${(100 * pa[k]).toFixed(2)}%"></span></span>` +
    `<span class="val">${fmt(pb[k], 2)} → ${fmt(pa[k], 2)}</span></div>`).join("");
  return `<div class="s-head"><span class="s-var">${esc(v.variable)}</span><span class="s-tv">changed by ${fmt(L.tv[i])}</span></div>
    <div class="s-label">${esc(label(v))}</div>
    <div class="bars">${rows}</div>
    ${idx.length > shown.length ? `<div class="s-more">+${idx.length - shown.length} smaller answers</div>` : ""}`;
}

async function spotlight(i, token) {
  const layer = $("spot-layer");
  const card = document.createElement("div");
  card.className = "spot";
  card.innerHTML = spotHTML(i);
  card.style.visibility = "hidden";
  layer.appendChild(card);
  const cw = card.offsetWidth, ch = card.offsetHeight;
  const r = cellRect(i);
  // final position: next to the cell, kept inside the grid
  let fx = r.x + r.w + 12, fy = r.y - ch / 2 + r.h / 2;
  if (fx + cw > G.w) fx = r.x - cw - 12;
  if (fx < 0) fx = Math.max(0, Math.min(G.w - cw, r.x - cw / 2));
  fy = Math.max(0, Math.min(G.h - ch, fy));
  card.style.left = `${fx}px`;
  card.style.top = `${fy}px`;
  const small = `translate(${r.x - fx}px, ${r.y - fy}px) scale(${r.w / cw}, ${r.h / ch})`;
  card.style.transform = small;
  card.style.opacity = "0.2";
  card.style.visibility = "visible";
  void card.offsetWidth;
  card.style.transform = "none";
  card.style.opacity = "1";
  S.spotCol = i;
  kick();
  await sleep(reduceMotion ? 1200 : 2100);
  if (token !== S.spotToken) return;
  card.style.transform = small;
  card.style.opacity = "0";
  setTimeout(() => card.remove(), 500);
  await sleep(reduceMotion ? 50 : 380);
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
  S.spotCol = null;
  for (const li of document.querySelectorAll(".top-list li.on")) li.classList.remove("on");
  kick();
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
  const idx = v.categories.map((_, k) => k).sort((a, b) => p[b] - p[a] || a - b).slice(0, 6);
  const rows = idx.map((k) =>
    `<div class="bar-row"><span class="lab">${esc(v.categories[k])}</span>` +
    `<span class="track">${changedByLast ? `<span class="ghost" style="opacity:.9;width:${(100 * pb[k]).toFixed(2)}%"></span>` : ""}` +
    `<span class="fill" style="width:${(100 * p[k]).toFixed(2)}%;${rec ? "background:var(--ans)" : ""}"></span></span>` +
    `<span class="val">${fmt(p[k], 2)}</span></div>`).join("");
  const tip = $("tooltip");
  tip.innerHTML = `<div class="s-var">${esc(v.variable)}</div><div class="s-label">${esc(label(v))}</div>
    <div class="bars">${rows}</div>
    <div class="t-hint">${rec ? `answered in Q${rec.question}: “${esc(rec.value)}”`
      : changedByLast ? `changed by the last answer (TV ${fmt(S.last.tv[i])}) · click to ask`
      : "click to ask this question"}</div>`;
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
// left panel
// ---------------------------------------------------------------------------

function searchVars(q) {
  q = q.trim().toLowerCase();
  if (!q) {
    return SUGGESTED.map((name) => S.vars.find((v) => v.variable.toLowerCase() === name))
      .filter((v) => v && v.learned).map((v) => ({ v, hit: "" }));
  }
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

function renderResults() {
  const list = searchVars(S.search);
  $("results-head").textContent = S.search.trim() ? `${list.length}${list.length === 60 ? "+" : ""} matching questions` : "Suggested questions";
  $("results").innerHTML = list.length ? list.map(({ v, hit }) => {
    const done = observedRecord(v.column);
    return `<li><button data-col="${v.column}" ${done || S.busy ? "disabled" : ""}>` +
      `<span class="r-var">${esc(v.variable)}${done ? `<span class="tag done">answered: ${esc(done.value)}</span>` : `<span class="tag">ask ›</span>`}</span>` +
      `<span class="r-label">${esc(v.label || "—")}</span>${hit ? `<span class="r-hit">${esc(hit)}</span>` : ""}</button></li>`;
  }).join("") : `<li class="empty">No modelled question matches.</li>`;
}

function renderHistory() {
  $("answer-count").textContent = S.observed.length;
  $("history").innerHTML = S.observed.length ? S.observed.map((o) => {
    const a = S.answers.find((x) => x.question === o.question);
    return `<li><button data-col="${o.column}"><div class="h-row"><span class="h-q">Q${o.question}</span><span>` +
      `<span class="mono">${esc(o.variable)}</span> = <span class="h-val">${esc(o.value)}</span>` +
      (a ? `<span class="h-stat">changed ${a.stats.moved} other distributions · ${a.stats.moved_001} by &gt;0.01</span>` : "") +
      `</span></div></button></li>`;
  }).join("") : `<li class="empty">Nothing asked yet.</li>`;
}

function renderGridHead() {
  $("grid-title").textContent = `Ψ: all ${S.n.toLocaleString()} marginal distributions`;
  const n = S.observed.length;
  $("grid-sub").textContent = n === 0
    ? "Each sparkline is one survey item's response histogram, currently at Ψ₀. Hover to read one; click to ask it."
    : `After ${n} answer${n > 1 ? "s" : ""}. Gold outlines mark the distributions changed by the last answer; red sparklines are answered questions (all mass on the drawn answer).`;
  $("btn-replay").disabled = !S.last || !S.last.changed.some((i) => S.last.tv[i] >= SPOT_MIN_TV);
}

function renderAll() {
  renderQuestion({ phase: "idle" });
  renderResults();
  renderHistory();
  renderGridHead();
  layoutGrid();
}

function showLoading(t) { $("loading-text").textContent = t; $("loading").hidden = false; }
function hideLoading() { $("loading").hidden = true; }

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
  $("history").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-col]");
    if (!b) return;
    const i = +b.dataset.col;
    const r = cellRect(i);
    S.hover = i;
    showTip(i, r.x + r.w / 2, r.y + r.h / 2);
    kick();
  });
  $("question").addEventListener("pointerover", (e) => {
    const li = e.target.closest(".top-list li[data-col]");
    S.hover = li ? +li.dataset.col : null;
    kick();
  });
  $("btn-replay").addEventListener("click", () => { pulseChanged(); runSpots(); });
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
  showLoading("Loading the GSS 2018 model…");
  try {
    await loadModel();
    await restoreOrCreate();
    hideLoading();
  } catch (err) {
    showLoading(`Could not start: ${err.message}`);
    document.querySelector("#loading .spinner").hidden = true;
  }
}

main();
