import { svgEl, createStageSVG, backgroundGrid } from "../core/svg.js";
import { renderControls, renderStatus } from "../core/controls.js";
import { resetPanel, beginPanel, addSection, addSlider, addSelect, addReadout, addButton, addHint } from "../core/panel.js";
import { fmt, randRange, randn, clamp } from "../core/math.js";
import { drawLossChart, pushHistory } from "../core/chart.js";
import { cssVar, onThemeChange } from "../core/theme.js";

const VIEWBOX = "0 0 900 560";
const PLOT = { left: 90, right: 830, top: 60, bottom: 470 };
const DOMAINS = {
  linear: { x: [-3.5, 3.5], y: [-7, 9] },
  logistic: { x: [-4.5, 4.5], y: [-0.18, 1.18] },
};

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
function wait(ms) { return new Promise((r) => setTimeout(r, ms)); }

function generatePoints(mode) {
  const n = 14;
  if (mode === "linear") {
    return Array.from({ length: n }, () => {
      const x = randRange(-3, 3);
      const y = 1.4 * x + 0.5 + randn(0, 0.9);
      return { x, y };
    });
  }
  return Array.from({ length: n }, () => {
    const x = randRange(-4, 4);
    const y = x + randn(0, 1.3) > 0 ? 1 : 0;
    return { x, y, jitter: randRange(-0.06, 0.06) };
  });
}

function freshState(mode = "linear") {
  return {
    mode,
    points: generatePoints(mode),
    w: 0, b: 0, lr: mode === "linear" ? 0.03 : 0.15,
    epoch: 0,
    lastLoss: null, lastDw: null, lastDb: null,
    lossHistory: [],
    busy: false,
  };
}

export default {
  id: "linear-regression",
  name: "Linear / Logistic Regression",
  blurb: "Watch gradient descent nudge a line (or decision boundary) toward the data, one step at a time.",
  status: "ready",

  mount({ stageEl, panelEl, controlsEl, statusEl }) {
    let state = freshState("linear");
    let selected = null; // { type: 'point'|'model', idx }
    let svg;
    let ctl, status;
    let playTimer = null;

    resetPanel(panelEl, "Click a data point or the model plate to inspect and edit it.");

    function domain() { return DOMAINS[state.mode]; }
    function mapX(x) {
      const d = domain().x;
      return PLOT.left + ((x - d[0]) / (d[1] - d[0])) * (PLOT.right - PLOT.left);
    }
    function mapY(y) {
      const d = domain().y;
      return PLOT.bottom - ((y - d[0]) / (d[1] - d[0])) * (PLOT.bottom - PLOT.top);
    }

    function predict(x) {
      const z = state.w * x + state.b;
      return state.mode === "linear" ? z : sigmoid(z);
    }

    function computeGradients() {
      const n = state.points.length;
      let dw = 0, db = 0, loss = 0;
      for (const p of state.points) {
        const pred = predict(p.x);
        const err = pred - p.y;
        dw += err * p.x;
        db += err;
        loss += state.mode === "linear"
          ? 0.5 * err * err
          : -(p.y * Math.log(clamp(pred, 1e-6, 1 - 1e-6)) + (1 - p.y) * Math.log(clamp(1 - pred, 1e-6, 1 - 1e-6)));
      }
      return { dw: dw / n, db: db / n, loss: loss / n };
    }

    async function doStep() {
      if (state.busy) return;
      state.busy = true;
      updateButtonStates();
      renderStage(true);
      await wait(550);
      const { dw, db, loss } = computeGradients();
      state.w -= state.lr * dw;
      state.b -= state.lr * db;
      state.lastDw = dw; state.lastDb = db; state.lastLoss = loss;
      pushHistory(state.lossHistory, loss, 80);
      state.epoch += 1;
      state.busy = false;
      renderAll();
    }

    function doRegenerate(mode) {
      stopPlay();
      state = freshState(mode);
      selected = null;
      resetPanel(panelEl, "Click a data point or the model plate to inspect and edit it.");
      renderAll();
    }

    function togglePlay() {
      if (playTimer) { stopPlay(); return; }
      playTimer = setInterval(() => { if (!state.busy) doStep(); }, 900);
      ctl.setLabel("play", "Pause");
    }
    function stopPlay() {
      if (playTimer) clearInterval(playTimer);
      playTimer = null;
      if (ctl) ctl.setLabel("play", "Play");
    }

    function selectPoint(idx) { selected = { type: "point", idx }; renderPanel(); renderStage(); }
    function selectModel() { selected = { type: "model" }; renderPanel(); renderStage(); }

    function renderPanel() {
      if (!selected) return;
      if (selected.type === "point") {
        const p = state.points[selected.idx];
        beginPanel(panelEl, `Data point #${selected.idx}`);
        const s = addSection(panelEl, state.mode === "linear" ? "Value" : "Label");
        if (state.mode === "linear") {
          addSlider(s, {
            label: "y", min: domain().y[0], max: domain().y[1], step: 0.1, value: p.y,
            onInput: (v) => { p.y = v; renderStage(); },
          });
        } else {
          addButton(s, `Flip label (currently ${p.y})`, () => { p.y = p.y === 1 ? 0 : 1; renderStage(); renderPanel(); });
        }
        const r = addSection(panelEl, "Readout");
        const pred = predict(p.x);
        addReadout(r, `x = ${fmt(p.x, 2)}\ny = ${fmt(p.y, 2)}\nprediction = ${fmt(pred, 3)}\nerror = ${fmt(pred - p.y, 3)}`);
        return;
      }

      beginPanel(panelEl, "Model");
      const modeSec = addSection(panelEl, "Task");
      addSelect(modeSec, {
        label: "mode", value: state.mode,
        options: [{ value: "linear", label: "Linear regression" }, { value: "logistic", label: "Logistic regression" }],
        onChange: (v) => doRegenerate(v),
      });
      addHint(modeSec, "Switching mode regenerates the dataset to fit the task.");

      const params = addSection(panelEl, "Parameters");
      addSlider(params, { label: "w (slope)", min: -4, max: 4, step: 0.02, value: state.w, onInput: (v) => { state.w = v; renderStage(); } });
      addSlider(params, { label: "b (intercept)", min: -6, max: 6, step: 0.05, value: state.b, onInput: (v) => { state.b = v; renderStage(); } });
      addSlider(params, { label: "learning rate", min: 0.005, max: 0.5, step: 0.005, value: state.lr, format: (v) => Number(v).toFixed(3), onInput: (v) => { state.lr = v; } });
      addButton(params, "Reset parameters to 0", () => { state.w = 0; state.b = 0; state.epoch = 0; renderAll(); renderPanel(); });

      const r = addSection(panelEl, "Readout");
      addReadout(r, `epoch = ${state.epoch}\nloss = ${state.lastLoss === null ? "—" : fmt(state.lastLoss, 4)}\ngrad w = ${state.lastDw === null ? "—" : fmt(state.lastDw, 4)}\ngrad b = ${state.lastDb === null ? "—" : fmt(state.lastDb, 4)}`);
    }

    function renderAxes() {
      const d = domain();
      // x axis (y=0 line) and y axis (x=0 line), where in range
      if (d.y[0] <= 0 && d.y[1] >= 0) {
        svg.appendChild(svgEl("line", { class: "axis-line", x1: PLOT.left, y1: mapY(0), x2: PLOT.right, y2: mapY(0) }));
      }
      if (d.x[0] <= 0 && d.x[1] >= 0) {
        svg.appendChild(svgEl("line", { class: "axis-line", x1: mapX(0), y1: PLOT.top, x2: mapX(0), y2: PLOT.bottom }));
      }
      svg.appendChild(svgEl("rect", {
        x: PLOT.left, y: PLOT.top, width: PLOT.right - PLOT.left, height: PLOT.bottom - PLOT.top,
        fill: "none", stroke: cssVar("--grid-line-strong", "#2c4d70"), "stroke-width": 1,
      }));
    }

    function renderStage(showResiduals = false) {
      svg = createStageSVG(stageEl, VIEWBOX);
      svg.appendChild(backgroundGrid(svg, 0, 0, 900, 560));
      renderAxes();

      // prediction curve/line
      if (state.mode === "linear") {
        const d = domain().x;
        svg.appendChild(svgEl("line", {
          x1: mapX(d[0]), y1: mapY(predict(d[0])), x2: mapX(d[1]), y2: mapY(predict(d[1])),
          stroke: "#ffb238", "stroke-width": 2.5,
        }));
      } else {
        const d = domain().x;
        const steps = 50;
        let pts = "";
        for (let i = 0; i <= steps; i++) {
          const x = d[0] + (i / steps) * (d[1] - d[0]);
          pts += `${mapX(x)},${mapY(predict(x))} `;
        }
        svg.appendChild(svgEl("polyline", { points: pts.trim(), fill: "none", stroke: "#ffb238", "stroke-width": 2.5 }));
        if (Math.abs(state.w) > 1e-3) {
          const xb = -state.b / state.w;
          if (xb >= d[0] && xb <= d[1]) {
            svg.appendChild(svgEl("line", {
              x1: mapX(xb), y1: PLOT.top, x2: mapX(xb), y2: PLOT.bottom,
              stroke: "#6d84a2", "stroke-width": 1.5, "stroke-dasharray": "4 4",
            }));
          }
        }
      }

      // residuals (only during the pre-update pause of a step)
      if (showResiduals) {
        for (const p of state.points) {
          const py = state.mode === "linear" ? p.y : p.y + (p.jitter || 0);
          const predY = predict(p.x);
          svg.appendChild(svgEl("line", {
            class: "edge-line active-backward",
            x1: mapX(p.x), y1: mapY(py), x2: mapX(p.x), y2: mapY(predY),
          }));
        }
      }

      // points
      state.points.forEach((p, idx) => {
        const py = state.mode === "linear" ? p.y : p.y + (p.jitter || 0);
        const isSel = selected && selected.type === "point" && selected.idx === idx;
        const c = svgEl("circle", {
          class: "node-shell" + (isSel ? " selected" : ""),
          cx: mapX(p.x), cy: mapY(py), r: 8,
        });
        c.addEventListener("click", () => selectPoint(idx));
        svg.appendChild(c);
      });

      // model plate
      const plateX = PLOT.left + 10, plateY = PLOT.top - 34;
      const isModelSel = selected && selected.type === "model";
      const plate = svgEl("rect", {
        class: "node-shell" + (isModelSel ? " selected" : ""),
        x: plateX, y: plateY - 20, width: 150, height: 34, rx: 4,
      });
      plate.addEventListener("click", selectModel);
      svg.appendChild(plate);
      const t = svgEl("text", { class: "node-label", x: plateX + 75, y: plateY - 2, style: "font-size:11px" });
      t.textContent = `w=${fmt(state.w, 2)}  b=${fmt(state.b, 2)}`;
      svg.appendChild(t);

      // loss-over-time chart, bottom-right — below the plot, clear of the data
      drawLossChart(svg, {
        x: PLOT.right - 220, y: PLOT.bottom + 8, w: 220, h: 78,
        history: state.lossHistory, color: "#ffb238",
        title: state.mode === "linear" ? "MSE over steps" : "Cross-entropy over steps",
      });
    }

    function renderStatusBar() {
      status = renderStatus(statusEl, {
        phase: { text: state.mode === "linear" ? "linear regression" : "logistic regression", variant: "forward" },
        stats: [
          { id: "epoch", label: "epoch", value: state.epoch },
          { id: "loss", label: "loss", value: state.lastLoss === null ? "—" : fmt(state.lastLoss, 4) },
        ],
      });
    }

    function renderButtons() {
      ctl = renderControls(controlsEl, [
        { id: "step", label: "Step (Gradient Descent)", variant: "primary", onClick: doStep },
        { id: "play", label: "Play", onClick: togglePlay },
        { id: "reset", label: "New Dataset", onClick: () => doRegenerate(state.mode) },
      ]);
    }
    function updateButtonStates() {
      ctl.setDisabled("step", state.busy);
    }

    function renderAll() {
      renderStage(false);
      renderStatusBar();
      updateButtonStates();
      if (selected) renderPanel();
    }

    renderButtons();
    renderAll();
    const unsubTheme = onThemeChange(renderAll);

    return { unmount() { stopPlay(); unsubTheme(); } };
  },
};
