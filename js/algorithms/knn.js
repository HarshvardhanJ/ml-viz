import { svgEl, createStageSVG, backgroundGrid } from "../core/svg.js";
import { renderControls, renderStatus } from "../core/controls.js";
import { resetPanel, beginPanel, addSection, addSlider, addSelect, addReadout, addHint } from "../core/panel.js";
import { fmt, randRange, randn } from "../core/math.js";

const VIEWBOX = "0 0 900 560";
const PLOT = { left: 220, right: 680, top: 50, bottom: 510 };
const DOMAIN = [-6, 6];
const CLASS_NAMES = ["A", "B", "C"];
const PALETTE = ["#ffb238", "#ff6b5b", "#59c9a5"];
const K_OPTIONS = [1, 3, 5, 7, 9];

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function generatePoints() {
  const pts = [];
  CLASS_NAMES.forEach((_, cls) => {
    const cx = randRange(-3.2, 3.2), cy = randRange(-3.2, 3.2);
    const n = 8 + Math.floor(Math.random() * 4);
    for (let i = 0; i < n; i++) pts.push({ x: cx + randn(0, 0.9), y: cy + randn(0, 0.9), label: cls });
  });
  return pts;
}

function freshState(k = 3) {
  return {
    k,
    points: generatePoints(),
    query: { x: 0, y: 0 },
    order: null,      // point indices sorted by distance to query, computed on first step
    revealed: 0,       // how many nearest neighbors have been revealed so far
    prediction: null,  // class index once revealed === k
    busy: false,
  };
}

export default {
  id: "knn",
  name: "K-Nearest Neighbors",
  blurb: "Click the plot to drop a query point, then step through its k nearest neighbors, closest first, until the vote decides its class.",
  status: "ready",

  mount({ stageEl, panelEl, controlsEl, statusEl }) {
    let state = freshState(3);
    let selected = null; // { type: 'point'|'query'|'model', idx }
    let svg;
    let ctl;

    resetPanel(panelEl, "Click the plot to place the query point, or click a training point to inspect it.");

    function mapX(x) { return PLOT.left + ((x - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * (PLOT.right - PLOT.left); }
    function mapY(y) { return PLOT.bottom - ((y - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * (PLOT.bottom - PLOT.top); }
    function invX(px) { return DOMAIN[0] + ((px - PLOT.left) / (PLOT.right - PLOT.left)) * (DOMAIN[1] - DOMAIN[0]); }
    function invY(py) { return DOMAIN[0] + ((PLOT.bottom - py) / (PLOT.bottom - PLOT.top)) * (DOMAIN[1] - DOMAIN[0]); }

    function invalidate() {
      state.order = null;
      state.revealed = 0;
      state.prediction = null;
    }

    function doStep() {
      if (state.busy) return;
      if (state.order === null) {
        state.order = state.points
          .map((p, i) => ({ i, d: dist(p, state.query) }))
          .sort((a, b) => a.d - b.d);
      }
      if (state.revealed >= state.k) return;
      state.revealed += 1;
      if (state.revealed === state.k) {
        const votes = {};
        for (let j = 0; j < state.k; j++) {
          const label = state.points[state.order[j].i].label;
          votes[label] = (votes[label] || 0) + 1;
        }
        let best = 0, bestCount = -1;
        for (const [label, count] of Object.entries(votes)) {
          if (count > bestCount) { bestCount = count; best = Number(label); }
        }
        state.prediction = best;
      }
      renderAll();
    }

    function placeQuery(px, py) {
      state.query = { x: invX(px), y: invY(py) };
      invalidate();
      selected = { type: "query" };
      renderAll();
      renderPanel();
    }

    function doRegenerateData() {
      state = freshState(state.k);
      selected = null;
      resetPanel(panelEl, "Click the plot to place the query point, or click a training point to inspect it.");
      renderAll();
    }

    function doChangeK(k) {
      state.k = k;
      invalidate();
      renderAll();
      if (selected) renderPanel();
    }

    function selectPoint(idx) { selected = { type: "point", idx }; renderPanel(); renderStage(); }
    function selectQuery() { selected = { type: "query" }; renderPanel(); renderStage(); }
    function selectModel() { selected = { type: "model" }; renderPanel(); renderStage(); }

    function renderPanel() {
      if (!selected) return;
      if (selected.type === "point") {
        const p = state.points[selected.idx];
        beginPanel(panelEl, `Training point #${selected.idx}`);
        const s = addSection(panelEl, "Position & class");
        addSlider(s, { label: "x", min: DOMAIN[0], max: DOMAIN[1], step: 0.1, value: p.x, onInput: (v) => { p.x = v; invalidate(); renderAll(); } });
        addSlider(s, { label: "y", min: DOMAIN[0], max: DOMAIN[1], step: 0.1, value: p.y, onInput: (v) => { p.y = v; invalidate(); renderAll(); } });
        addSelect(s, {
          label: "class", value: String(p.label),
          options: CLASS_NAMES.map((name, i) => ({ value: String(i), label: name })),
          onChange: (v) => { p.label = Number(v); invalidate(); renderAll(); },
        });
        const r = addSection(panelEl, "Readout");
        addReadout(r, `distance to query = ${fmt(dist(p, state.query), 3)}`);
        return;
      }
      if (selected.type === "query") {
        beginPanel(panelEl, "Query point");
        const s = addSection(panelEl, "Position");
        addSlider(s, { label: "x", min: DOMAIN[0], max: DOMAIN[1], step: 0.1, value: state.query.x, onInput: (v) => { state.query.x = v; invalidate(); renderAll(); } });
        addSlider(s, { label: "y", min: DOMAIN[0], max: DOMAIN[1], step: 0.1, value: state.query.y, onInput: (v) => { state.query.y = v; invalidate(); renderAll(); } });
        addHint(s, "You can also just click anywhere on the plot to move the query point there.");
        const r = addSection(panelEl, "Readout");
        addReadout(r, predictionText());
        return;
      }
      beginPanel(panelEl, "Model");
      const s = addSection(panelEl, "Setup");
      addSelect(s, {
        label: "k (neighbors)", value: String(state.k),
        options: K_OPTIONS.map((n) => ({ value: String(n), label: String(n) })),
        onChange: (v) => doChangeK(parseInt(v, 10)),
      });
      addHint(s, "Odd k avoids ties between two classes.");
      const r = addSection(panelEl, "Readout");
      addReadout(r, predictionText());
    }

    function predictionText() {
      let text = `revealed = ${state.revealed} / ${state.k}`;
      if (state.prediction !== null) text += `\npredicted class = ${CLASS_NAMES[state.prediction]}`;
      return text;
    }

    function renderStage() {
      svg = createStageSVG(stageEl, VIEWBOX);
      svg.appendChild(backgroundGrid(svg, 0, 0, 900, 560));

      // neighborhood circle once fully revealed
      if (state.revealed === state.k && state.order) {
        const farthest = state.points[state.order[state.k - 1].i];
        const radiusPx = mapX(state.query.x + dist(farthest, state.query)) - mapX(state.query.x);
        svg.appendChild(svgEl("circle", {
          cx: mapX(state.query.x), cy: mapY(state.query.y), r: Math.abs(radiusPx),
          fill: PALETTE[state.prediction], "fill-opacity": 0.08,
          stroke: PALETTE[state.prediction], "stroke-width": 1, "stroke-dasharray": "4 4",
        }));
      }

      // lines to revealed neighbors
      if (state.order) {
        for (let j = 0; j < state.revealed; j++) {
          const p = state.points[state.order[j].i];
          svg.appendChild(svgEl("line", {
            x1: mapX(state.query.x), y1: mapY(state.query.y), x2: mapX(p.x), y2: mapY(p.y),
            stroke: "#ffb238", "stroke-width": 1.5, "stroke-dasharray": "3 3",
          }));
        }
      }

      // clickable plot background — placing the query point
      const bg = svgEl("rect", {
        x: PLOT.left, y: PLOT.top, width: PLOT.right - PLOT.left, height: PLOT.bottom - PLOT.top,
        fill: "transparent", stroke: "#2c4d70", style: "cursor:crosshair",
      });
      bg.addEventListener("click", (evt) => {
        try {
          const pt = svg.createSVGPoint();
          pt.x = evt.clientX; pt.y = evt.clientY;
          const ctm = svg.getScreenCTM();
          if (!ctm) return;
          const loc = pt.matrixTransform(ctm.inverse());
          placeQuery(loc.x, loc.y);
        } catch (err) {
          // Some environments don't implement SVG geometry APIs
          // (createSVGPoint/getScreenCTM); fail quietly rather than crash —
          // the query point can still be moved via the inspector sliders.
        }
      });
      svg.appendChild(bg);

      // training points
      const revealedIdxs = state.order ? new Set(state.order.slice(0, state.revealed).map((o) => o.i)) : new Set();
      state.points.forEach((p, idx) => {
        const isSel = selected && selected.type === "point" && selected.idx === idx;
        const isNeighbor = revealedIdxs.has(idx);
        const c = svgEl("circle", {
          cx: mapX(p.x), cy: mapY(p.y), r: isNeighbor ? 9 : 7,
          fill: PALETTE[p.label], "fill-opacity": 0.9,
          stroke: isSel ? "#fff" : (isNeighbor ? "#fff" : "none"),
          "stroke-width": isSel ? 2.5 : isNeighbor ? 2 : 0,
          style: "cursor:pointer",
        });
        c.addEventListener("click", (evt) => { evt.stopPropagation(); selectPoint(idx); });
        svg.appendChild(c);
      });

      // query point
      const qSel = selected && selected.type === "query";
      const qColor = state.prediction !== null ? PALETTE[state.prediction] : "#e8ecef";
      const qx = mapX(state.query.x), qy = mapY(state.query.y);
      const size = qSel ? 11 : 9;
      const diamond = svgEl("polygon", {
        points: `${qx},${qy - size} ${qx + size},${qy} ${qx},${qy + size} ${qx - size},${qy}`,
        fill: qColor, stroke: "#0e1b2b", "stroke-width": 2, style: "cursor:pointer",
      });
      diamond.addEventListener("click", (evt) => { evt.stopPropagation(); selectQuery(); });
      svg.appendChild(diamond);
      const qLabel = svgEl("text", { class: "node-sublabel", x: qx, y: qy - size - 8 });
      qLabel.textContent = state.prediction !== null ? `query → ${CLASS_NAMES[state.prediction]}` : "query (?)";
      svg.appendChild(qLabel);

      // model plate
      const isModelSel = selected && selected.type === "model";
      const plate = svgEl("rect", { class: "node-shell" + (isModelSel ? " selected" : ""), x: 20, y: 40, width: 150, height: 34, rx: 4, style: "cursor:pointer" });
      plate.addEventListener("click", selectModel);
      svg.appendChild(plate);
      const t = svgEl("text", { class: "node-label", x: 95, y: 62, style: "font-size:11px" });
      t.textContent = `k = ${state.k}`;
      svg.appendChild(t);
    }

    function renderStatusBar() {
      renderStatus(statusEl, {
        phase: state.prediction !== null
          ? { text: `predicted ${CLASS_NAMES[state.prediction]}`, variant: "done" }
          : { text: "click plot to place query, then step", variant: "forward" },
        stats: [
          { id: "revealed", label: "revealed", value: `${state.revealed}/${state.k}` },
        ],
      });
    }

    function renderButtons() {
      ctl = renderControls(controlsEl, [
        { id: "step", label: "Step: Reveal Next Neighbor", variant: "primary", onClick: doStep },
        { id: "reset", label: "New Dataset", onClick: doRegenerateData },
      ]);
      updateButtonStates();
    }
    function updateButtonStates() {
      ctl.setDisabled("step", state.busy || state.revealed >= state.k);
    }

    function renderAll() {
      renderStage();
      renderStatusBar();
      updateButtonStates();
    }

    renderButtons();
    renderAll();

    return { unmount() {} };
  },
};
