import { svgEl, createStageSVG, backgroundGrid } from "../core/svg.js";
import { renderControls, renderStatus } from "../core/controls.js";
import { resetPanel, beginPanel, addSection, addSlider, addSelect, addReadout, addHint } from "../core/panel.js";
import { fmt, randRange, randn } from "../core/math.js";
import { drawLossChart, pushHistory } from "../core/chart.js";

const VIEWBOX = "0 0 900 560";
const PLOT = { left: 220, right: 680, top: 50, bottom: 510 };
const DOMAIN = [-6, 6];
const PALETTE = ["#ffb238", "#ff6b5b", "#59c9a5", "#6d84a2", "#c792ea"];

function wait(ms) { return new Promise((r) => setTimeout(r, ms)); }
function dist2(a, b) { return (a.x - b.x) ** 2 + (a.y - b.y) ** 2; }

function generatePoints() {
  const blobs = 3;
  const pts = [];
  for (let b = 0; b < blobs; b++) {
    const cx = randRange(-3.5, 3.5), cy = randRange(-3.5, 3.5);
    const n = 9 + Math.floor(Math.random() * 5);
    for (let i = 0; i < n; i++) {
      pts.push({ x: cx + randn(0, 0.8), y: cy + randn(0, 0.8), cluster: null });
    }
  }
  return pts;
}

function initCentroids(points, k) {
  const shuffled = [...points].sort(() => Math.random() - 0.5);
  return Array.from({ length: k }, (_, i) => ({ x: shuffled[i % shuffled.length].x + randn(0, 0.3), y: shuffled[i % shuffled.length].y + randn(0, 0.3) }));
}

function freshState(k = 3, points = null) {
  const pts = points || generatePoints();
  return {
    k,
    points: pts.map((p) => ({ ...p, cluster: null })),
    centroids: initCentroids(pts, k),
    iteration: 0,
    phase: "assign", // 'assign' | 'update'
    converged: false,
    inertiaHistory: [],
    busy: false,
  };
}

function computeInertia(state) {
  return state.points.reduce((sum, p) => {
    if (p.cluster === null) return sum;
    return sum + dist2(p, state.centroids[p.cluster]);
  }, 0);
}

export default {
  id: "kmeans",
  name: "K-Means Clustering",
  blurb: "Alternate between assigning points to the nearest centroid and moving centroids to the mean of their cluster.",
  status: "ready",

  mount({ stageEl, panelEl, controlsEl, statusEl }) {
    let state = freshState(3);
    let selected = null; // { type: 'point'|'centroid'|'model', idx }
    let svg;
    let ctl;
    let playTimer = null;

    resetPanel(panelEl, "Click a point or a centroid to inspect and edit it.");

    function mapX(x) { return PLOT.left + ((x - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * (PLOT.right - PLOT.left); }
    function mapY(y) { return PLOT.bottom - ((y - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * (PLOT.bottom - PLOT.top); }

    function nearestCentroidIdx(p) {
      let best = 0, bestD = Infinity;
      state.centroids.forEach((c, i) => { const d = dist2(p, c); if (d < bestD) { bestD = d; best = i; } });
      return best;
    }

    function computeNewCentroids() {
      return state.centroids.map((old, i) => {
        const members = state.points.filter((p) => p.cluster === i);
        if (members.length === 0) return { x: old.x, y: old.y };
        return {
          x: members.reduce((s, p) => s + p.x, 0) / members.length,
          y: members.reduce((s, p) => s + p.y, 0) / members.length,
        };
      });
    }

    async function doStep() {
      if (state.busy || state.converged) return;
      state.busy = true;
      updateButtonStates();

      if (state.phase === "assign") {
        const prev = state.points.map((p) => p.cluster);
        state.points.forEach((p) => { p.cluster = nearestCentroidIdx(p); });
        const changed = state.points.some((p, i) => p.cluster !== prev[i]);
        if (state.iteration > 0 && !changed) state.converged = true;
        state.phase = "update";
        pushHistory(state.inertiaHistory, computeInertia(state), 80);
        renderAll();
      } else {
        const fresh = computeNewCentroids();
        renderStage({ preview: fresh });
        await wait(550);
        state.centroids = fresh;
        state.iteration += 1;
        state.phase = "assign";
        pushHistory(state.inertiaHistory, computeInertia(state), 80);
        renderAll();
      }
      state.busy = false;
      updateButtonStates();
    }

    function doRegenerateData() {
      stopPlay();
      state = freshState(state.k);
      selected = null;
      resetPanel(panelEl, "Click a point or a centroid to inspect and edit it.");
      renderAll();
    }

    function doChangeK(k) {
      stopPlay();
      state = freshState(k, state.points);
      selected = null;
      renderAll();
      renderPanel();
    }

    function togglePlay() {
      if (playTimer) { stopPlay(); return; }
      playTimer = setInterval(() => { if (!state.busy && !state.converged) doStep(); else if (state.converged) stopPlay(); }, 800);
      ctl.setLabel("play", "Pause");
    }
    function stopPlay() {
      if (playTimer) clearInterval(playTimer);
      playTimer = null;
      if (ctl) ctl.setLabel("play", "Play");
    }

    function selectPoint(idx) { selected = { type: "point", idx }; renderPanel(); renderStage(); }
    function selectCentroid(idx) { selected = { type: "centroid", idx }; renderPanel(); renderStage(); }
    function selectModel() { selected = { type: "model" }; renderPanel(); renderStage(); }

    function renderPanel() {
      if (!selected) return;
      if (selected.type === "point") {
        const p = state.points[selected.idx];
        beginPanel(panelEl, `Point #${selected.idx}`);
        const s = addSection(panelEl, "Position");
        addSlider(s, { label: "x", min: DOMAIN[0], max: DOMAIN[1], step: 0.1, value: p.x, onInput: (v) => { p.x = v; renderStage(); } });
        addSlider(s, { label: "y", min: DOMAIN[0], max: DOMAIN[1], step: 0.1, value: p.y, onInput: (v) => { p.y = v; renderStage(); } });
        const r = addSection(panelEl, "Readout");
        const c = p.cluster === null ? null : state.centroids[p.cluster];
        addReadout(r, `cluster = ${p.cluster === null ? "unassigned" : "c" + p.cluster}\ndistance to centroid = ${c ? fmt(Math.sqrt(dist2(p, c)), 3) : "—"}`);
        return;
      }
      if (selected.type === "centroid") {
        const c = state.centroids[selected.idx];
        beginPanel(panelEl, `Centroid c${selected.idx}`);
        const s = addSection(panelEl, "Position");
        addSlider(s, { label: "x", min: DOMAIN[0], max: DOMAIN[1], step: 0.1, value: c.x, onInput: (v) => { c.x = v; renderStage(); } });
        addSlider(s, { label: "y", min: DOMAIN[0], max: DOMAIN[1], step: 0.1, value: c.y, onInput: (v) => { c.y = v; renderStage(); } });
        addHint(s, "Moving a centroid by hand is a great way to see how the assignment step reacts.");
        const members = state.points.filter((p) => p.cluster === selected.idx).length;
        const r = addSection(panelEl, "Readout");
        addReadout(r, `assigned points = ${members}`);
        return;
      }
      beginPanel(panelEl, "Model");
      const s = addSection(panelEl, "Setup");
      addSelect(s, {
        label: "k (clusters)", value: String(state.k),
        options: [2, 3, 4, 5].map((n) => ({ value: String(n), label: String(n) })),
        onChange: (v) => doChangeK(parseInt(v, 10)),
      });
      addHint(s, "Changing k keeps the data but re-seeds centroids.");
      const r = addSection(panelEl, "Readout");
      addReadout(r, `iteration = ${state.iteration}\nstatus = ${state.converged ? "converged" : "running"}`);
    }

    function renderStage({ preview } = {}) {
      svg = createStageSVG(stageEl, VIEWBOX);
      svg.appendChild(backgroundGrid(svg, 0, 0, 900, 560));
      svg.appendChild(svgEl("rect", {
        x: PLOT.left, y: PLOT.top, width: PLOT.right - PLOT.left, height: PLOT.bottom - PLOT.top,
        fill: "none", stroke: "#2c4d70",
      }));

      // points
      state.points.forEach((p, idx) => {
        const color = p.cluster === null ? "#6d84a2" : PALETTE[p.cluster % PALETTE.length];
        const isSel = selected && selected.type === "point" && selected.idx === idx;
        const c = svgEl("circle", {
          cx: mapX(p.x), cy: mapY(p.y), r: isSel ? 8 : 6.5,
          fill: color, stroke: isSel ? "#ffffff" : "none", "stroke-width": 2, "fill-opacity": 0.9,
          style: "cursor:pointer",
        });
        c.addEventListener("click", () => selectPoint(idx));
        svg.appendChild(c);
      });

      // preview arrows (old -> new centroid) during update pause
      if (preview) {
        state.centroids.forEach((old, i) => {
          const to = preview[i];
          svg.appendChild(svgEl("line", {
            x1: mapX(old.x), y1: mapY(old.y), x2: mapX(to.x), y2: mapY(to.y),
            stroke: PALETTE[i % PALETTE.length], "stroke-width": 2, "stroke-dasharray": "5 4",
          }));
          svg.appendChild(diamond(mapX(to.x), mapY(to.y), PALETTE[i % PALETTE.length], 10, 0.4));
        });
      }

      // centroids
      state.centroids.forEach((c, idx) => {
        const isSel = selected && selected.type === "centroid" && selected.idx === idx;
        const g = svgEl("g", { style: "cursor:pointer" });
        const d = diamond(mapX(c.x), mapY(c.y), PALETTE[idx % PALETTE.length], isSel ? 15 : 12, 1, isSel);
        d.addEventListener("click", () => selectCentroid(idx));
        g.appendChild(d);
        const label = svgEl("text", { class: "node-sublabel", x: mapX(c.x), y: mapY(c.y) - 18 });
        label.textContent = `c${idx}`;
        g.appendChild(label);
        svg.appendChild(g);
      });

      // model plate
      const isModelSel = selected && selected.type === "model";
      const plate = svgEl("rect", { class: "node-shell" + (isModelSel ? " selected" : ""), x: 20, y: 40, width: 150, height: 34, rx: 4, style: "cursor:pointer" });
      plate.addEventListener("click", selectModel);
      svg.appendChild(plate);
      const t = svgEl("text", { class: "node-label", x: 95, y: 62, style: "font-size:11px" });
      t.textContent = `k = ${state.k}`;
      svg.appendChild(t);

      // inertia-over-time chart, right margin — fully clear of the plot area
      drawLossChart(svg, {
        x: 700, y: 200, w: 180, h: 100,
        history: state.inertiaHistory, color: "#59c9a5",
        title: "Inertia (WCSS)",
      });
    }

    function diamond(x, y, color, size, opacity = 1, selected = false) {
      const points = `${x},${y - size} ${x + size},${y} ${x},${y + size} ${x - size},${y}`;
      return svgEl("polygon", {
        points, fill: color, "fill-opacity": opacity, stroke: selected ? "#fff" : "#0e1b2b", "stroke-width": selected ? 2 : 1.5,
      });
    }

    function renderStatusBar() {
      renderStatus(statusEl, {
        phase: state.converged
          ? { text: "converged", variant: "done" }
          : { text: state.phase === "assign" ? "next: assign points" : "next: update centroids", variant: state.phase === "assign" ? "forward" : "backward" },
        stats: [
          { id: "iter", label: "iteration", value: state.iteration },
          { id: "k", label: "k", value: state.k },
        ],
      });
    }

    function renderButtons() {
      ctl = renderControls(controlsEl, [
        { id: "step", label: "Step", variant: "primary", onClick: doStep },
        { id: "play", label: "Play", onClick: togglePlay },
        { id: "reset", label: "New Dataset", onClick: doRegenerateData },
      ]);
      updateButtonStates();
    }
    function updateButtonStates() {
      ctl.setDisabled("step", state.busy || state.converged);
      ctl.setDisabled("play", state.converged);
    }

    function renderAll() {
      renderStage();
      renderStatusBar();
      updateButtonStates();
      if (selected) renderPanel();
    }

    renderButtons();
    renderAll();

    return { unmount() { stopPlay(); } };
  },
};
