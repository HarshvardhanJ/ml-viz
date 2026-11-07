import { svgEl, createStageSVG, backgroundGrid, pulseAlongLine } from "../core/svg.js";
import { renderControls, renderStatus } from "../core/controls.js";
import { resetPanel, beginPanel, addSection, addSlider, addSelect, addReadout, addHint } from "../core/panel.js";
import { activations, fmt, randRange } from "../core/math.js";
import { pushHistory } from "../core/chart.js";
import { onThemeChange, cssVar } from "../core/theme.js";
import { withPrediction } from "../core/predict.js";

const VIEWBOX = "0 0 1000 650";
const LOSSES = { mse: "Mean squared error", mae: "Mean absolute error", bce: "Binary cross-entropy" };
const OPTIMIZERS = { sgd: "SGD", momentum: "Momentum", adam: "Adam", rmsprop: "RMSProp" };
const MAX_HISTORY = 120;

function neuron(nInputs, activation = "relu") {
  return { weights: Array.from({ length: nInputs }, () => randRange(-0.8, 0.8)), bias: 0, activation, z: null, a: null, delta: null, gradW: [], gradB: null, mw: [], vw: [], mb: 0, vb: 0 };
}

function freshState(inputCount = 2, hiddenSizes = [3], outputCount = 1) {
  const widths = [inputCount, ...hiddenSizes, outputCount];
  const layers = widths.slice(1).map((width, layerIdx) => Array.from({ length: width }, () => neuron(widths[layerIdx], layerIdx === widths.length - 2 ? "sigmoid" : "relu")));
  return { inputCount, hiddenSizes, outputCount, inputs: Array.from({ length: inputCount }, (_, i) => i === 0 ? 0.6 : i === 1 ? -0.4 : 0.2), targets: Array.from({ length: outputCount }, (_, i) => i === 0 ? 1 : 0), layers, loss: "mse", optimizer: "sgd", lr: 0.15, momentum: 0.9, beta1: 0.9, beta2: 0.999, epoch: 0, step: 0, phase: "idle", queue: [], busy: false, history: [], trajectory: [], selected: null };
}

function lossValue(predictions, targets, kind) {
  const n = predictions.length || 1;
  if (kind === "mae") return predictions.reduce((s, a, i) => s + Math.abs(targets[i] - a), 0) / n;
  if (kind === "bce") return predictions.reduce((s, a, i) => {
    const p = Math.max(1e-7, Math.min(1 - 1e-7, a));
    return s - targets[i] * Math.log(p) - (1 - targets[i]) * Math.log(1 - p);
  }, 0) / n;
  return predictions.reduce((s, a, i) => s + (targets[i] - a) ** 2, 0) / n;
}

export default {
  id: "neural-network",
  name: "Neural Network Lab",
  blurb: "Build a multilayer network, inspect forward and backward passes, and compare optimizers on a live loss landscape.",
  status: "ready",

  mount({ stageEl, panelEl, controlsEl, statusEl }) {
    let state = freshState();
    let svg, ctl, status;
    let playTimer = null;
    resetPanel(panelEl, "Configure the network, data, loss, and optimizer. Select a neuron in the diagram to inspect its values and gradients.");

    const layerWidths = () => [state.inputCount, ...state.hiddenSizes, state.outputCount];
    const coords = () => {
      const widths = layerWidths(), left = 74, right = 660, top = 100, bottom = 570;
      return widths.map((width, li) => Array.from({ length: width }, (_, i) => ({ x: left + li * (right - left) / (widths.length - 1), y: (top + bottom) / 2 + (i - (width - 1) / 2) * Math.min(92, (bottom - top) / Math.max(width, 1)) })));
    };
    const allOutputs = () => state.layers.at(-1).map(n => n.a ?? 0);
    function forwardPure(inputs = state.inputs, overrides = []) {
      let values = inputs;
      for (let li = 0; li < state.layers.length; li++) {
        values = state.layers[li].map((n, ni) => {
          const sum = n.weights.reduce((s, w, wi) => {
            let weight = w;
            const override = overrides.find(o => o.layer === li && o.neuron === ni && o.weight === wi && !o.bias);
            if (override) weight = override.value;
            return s + weight * values[wi];
          }, n.bias);
          const biasOverride = overrides.find(o => o.layer === li && o.neuron === ni && o.bias);
          const z = biasOverride ? sum + biasOverride.value - n.bias : sum;
          return activations[n.activation].fn(z);
        });
      }
      return values;
    }
    function objective(overrides = []) { return lossValue(forwardPure(state.inputs, overrides), state.targets, state.loss); }
    function clearPass() {
      for (const layer of state.layers) for (const n of layer) { n.z = n.a = n.delta = n.gradB = null; n.gradW = []; }
      state.phase = "idle"; state.queue = [];
    }
    function resetNetwork() {
      const fresh = freshState(state.inputCount, [...state.hiddenSizes], state.outputCount);
      fresh.loss = state.loss; fresh.optimizer = state.optimizer; fresh.lr = state.lr; fresh.momentum = state.momentum; fresh.beta1 = state.beta1; fresh.beta2 = state.beta2;
      state = fresh;
      renderSettings(); renderAll();
    }
    function configure(inputCount, hiddenSizes, outputCount) {
      const old = state;
      state = freshState(inputCount, hiddenSizes, outputCount);
      state.loss = old.loss; state.optimizer = old.optimizer; state.lr = old.lr; state.momentum = old.momentum; state.beta1 = old.beta1; state.beta2 = old.beta2;
      renderSettings(); renderAll();
    }
    function makeForwardQueue() { return state.layers.flatMap((layer, li) => layer.map((_, ni) => ({ layer: li, neuron: ni }))); }

    async function stepForwardReal() {
      if (state.busy || !["idle", "forward"].includes(state.phase)) return;
      if (state.phase === "idle") { state.queue = makeForwardQueue(); state.phase = "forward"; }
      state.busy = true; renderAll();
      const { layer: li, neuron: ni } = state.queue.shift();
      const xy = coords(); const n = state.layers[li][ni];
      const inputs = li === 0 ? state.inputs : state.layers[li - 1].map(v => v.a ?? 0);
      const incoming = inputs.map((_, i) => ({ from: xy[li][i], to: xy[li + 1][ni], value: inputs[i] }));
      await Promise.all(incoming.map(e => pulseAlongLine(svg, { x1: e.from.x, y1: e.from.y, x2: e.to.x, y2: e.to.y, value: e.value, color: "#ffb238" })));
      n.z = n.weights.reduce((s, w, i) => s + w * inputs[i], n.bias); n.a = activations[n.activation].fn(n.z);
      if (state.queue.length === 0) {
        state.phase = "forward-done";
      } else state.phase = "forward";
      state.busy = false; renderAll();
    }

    async function stepBackwardReal() {
      if (state.busy || !["forward-done", "backward"].includes(state.phase)) return;
      if (state.phase === "forward-done") {
        state.queue = state.layers.flatMap((layer, li) => layer.map((_, ni) => ({ layer: li, neuron: ni }))).reverse();
        state.phase = "backward";
      }
      state.busy = true; renderAll();
      const { layer: li, neuron: ni } = state.queue.shift(); const n = state.layers[li][ni];
      if (li === state.layers.length - 1) {
        const a = n.a, y = state.targets[ni], dlda = state.loss === "mse" ? 2 * (a - y) / state.outputCount : state.loss === "mae" ? Math.sign(a - y) / state.outputCount : ((a - y) / Math.max(1e-7, a * (1 - a))) / state.outputCount;
        n.delta = dlda * activations[n.activation].dfn(n.z);
      } else n.delta = state.layers[li + 1].reduce((s, next) => s + next.weights[ni] * next.delta, 0) * activations[n.activation].dfn(n.z);
      const prevValues = li === 0 ? state.inputs : state.layers[li - 1].map(v => v.a ?? 0);
      n.gradW = n.weights.map((_, i) => n.delta * prevValues[i]); n.gradB = n.delta;
      const xy = coords();
      await Promise.all(n.gradW.map((g, i) => pulseAlongLine(svg, { x1: xy[li + 1][ni].x, y1: xy[li + 1][ni].y, x2: xy[li][i].x, y2: xy[li][i].y, value: g, color: "#ff6b5b" })));
      state.phase = state.queue.length ? "backward" : "backward-done";
      state.busy = false; renderAll();
    }
    function previewForwardQuestion() {
      const next = state.phase === "idle" ? makeForwardQueue()[0] : state.queue[0];
      if (!next) return null;
      const n = state.layers[next.layer][next.neuron];
      const inputs = next.layer === 0 ? state.inputs : state.layers[next.layer - 1].map(v => v.a ?? 0);
      const a = activations[n.activation].fn(n.weights.reduce((s, w, i) => s + w * inputs[i], n.bias));
      return { question: `Will neuron L${next.layer + 1}·${next.neuron + 1} output be positive or zero/negative?`, options: ["Positive", "Zero or negative"], correctIndex: a > 0 ? 0 : 1 };
    }
    function previewBackwardQuestion() {
      const next = state.phase === "forward-done" ? [...makeForwardQueue()].reverse()[0] : state.queue[0];
      if (!next) return null;
      const n = state.layers[next.layer][next.neuron];
      function deltaAt(li, ni) {
        const current = state.layers[li][ni];
        if (current.delta !== null) return current.delta;
        if (li === state.layers.length - 1) {
          const a = current.a, y = state.targets[ni];
          const dlda = state.loss === "mse" ? 2 * (a - y) / state.outputCount : state.loss === "mae" ? Math.sign(a - y) / state.outputCount : ((a - y) / Math.max(1e-7, a * (1 - a))) / state.outputCount;
          return dlda * activations[current.activation].dfn(current.z);
        }
        return state.layers[li + 1].reduce((sum, nextNeuron, downstreamIdx) => sum + nextNeuron.weights[ni] * deltaAt(li + 1, downstreamIdx), 0) * activations[current.activation].dfn(current.z);
      }
      const delta = deltaAt(next.layer, next.neuron);
      return { question: `Will this neuron's error signal δ be positive or negative?`, options: ["Positive", "Negative"], correctIndex: delta >= 0 ? 0 : 1 };
    }
    const stepForward = withPrediction(stageEl, previewForwardQuestion, stepForwardReal);
    const stepBackward = withPrediction(stageEl, previewBackwardQuestion, stepBackwardReal);

    function updateParam(n, key, gradKey, mKey, vKey, t) {
      const grad = n[gradKey], oldM = n[mKey] ?? 0, oldV = n[vKey] ?? 0;
      let update = state.lr * grad;
      if (state.optimizer === "momentum") { const m = state.momentum * oldM + grad; n[mKey] = m; update = state.lr * m; }
      else if (state.optimizer === "rmsprop") { const v = 0.9 * oldV + 0.1 * grad * grad; n[vKey] = v; update = state.lr * grad / (Math.sqrt(v) + 1e-8); }
      else if (state.optimizer === "adam") {
        const m = state.beta1 * oldM + (1 - state.beta1) * grad, v = state.beta2 * oldV + (1 - state.beta2) * grad * grad;
        n[mKey] = m; n[vKey] = v;
        update = state.lr * (m / (1 - state.beta1 ** t)) / (Math.sqrt(v / (1 - state.beta2 ** t)) + 1e-8);
      }
      n[key] -= update;
    }
    function applyGradients() {
      state.step += 1;
      const first = state.layers[0][0];
      const p0 = first.weights[0] ?? first.bias, p1 = first.weights[1] ?? first.bias;
      // Apply each parameter with its own optimizer moments.
      for (const layer of state.layers) for (const n of layer) {
        n.weights = n.weights.map((w, i) => { const proxy = { weights: w, m: n.mw[i] ?? 0, v: n.vw[i] ?? 0, g: n.gradW[i] }; updateParam(proxy, "weights", "g", "m", "v", state.step); n.mw[i] = proxy.m; n.vw[i] = proxy.v; return proxy.weights; });
        const proxy = { bias: n.bias, m: n.mb, v: n.vb, g: n.gradB }; updateParam(proxy, "bias", "g", "m", "v", state.step); n.bias = proxy.bias; n.mb = proxy.m; n.vb = proxy.v;
      }
      state.epoch += 1; state.trajectory.push([p0, p1]); if (state.trajectory.length > 100) state.trajectory.shift();
      pushHistory(state.history, objective(), MAX_HISTORY);
      clearPass(); renderAll();
    }

    function addOptimizerSetting(section) {
      if (state.optimizer === "momentum") addSlider(section, { label: "momentum β", min: 0, max: 0.99, step: 0.01, value: state.momentum, onInput: v => { state.momentum = v; renderStage(); } });
      if (state.optimizer === "adam") {
        addSlider(section, { label: "β₁", min: 0.5, max: 0.99, step: 0.01, value: state.beta1, onInput: v => { state.beta1 = v; renderStage(); } });
        addSlider(section, { label: "β₂", min: 0.9, max: 0.999, step: 0.001, value: state.beta2, onInput: v => { state.beta2 = v; renderStage(); } });
      }
    }
    function renderSettings() {
      beginPanel(panelEl, "Network builder");
      const arch = addSection(panelEl, "Architecture");
      addSelect(arch, { label: "Inputs", value: String(state.inputCount), options: Array.from({ length: 6 }, (_, i) => ({ value: String(i + 1), label: String(i + 1) })), onChange: v => configure(+v, state.hiddenSizes, state.outputCount) });
      addSelect(arch, { label: "Hidden layers", value: String(state.hiddenSizes.length), options: [1, 2, 3, 4].map(n => ({ value: String(n), label: String(n) })), onChange: v => { const h = Array.from({ length: +v }, (_, i) => state.hiddenSizes[i] ?? 3); configure(state.inputCount, h, state.outputCount); } });
      state.hiddenSizes.forEach((size, i) => addSelect(arch, { label: `Hidden ${i + 1} neurons`, value: String(size), options: Array.from({ length: 6 }, (_, j) => ({ value: String(j + 1), label: String(j + 1) })), onChange: v => { const h = [...state.hiddenSizes]; h[i] = +v; configure(state.inputCount, h, state.outputCount); } }));
      addSelect(arch, { label: "Outputs", value: String(state.outputCount), options: Array.from({ length: 4 }, (_, i) => ({ value: String(i + 1), label: String(i + 1) })), onChange: v => configure(state.inputCount, state.hiddenSizes, +v) });
      const data = addSection(panelEl, "Training example");
      state.inputs.forEach((v, i) => addSlider(data, { label: `input x${i + 1}`, min: -2, max: 2, step: 0.05, value: v, onInput: n => { state.inputs[i] = n; renderStage(); renderStatusBar(); } }));
      state.targets.forEach((v, i) => addSlider(data, { label: `target y${i + 1}`, min: state.loss === "bce" ? 0 : -2, max: 2, step: 0.05, value: v, onInput: n => { state.targets[i] = state.loss === "bce" ? Math.max(0, Math.min(1, n)) : n; renderStage(); renderStatusBar(); } }));
      const train = addSection(panelEl, "Training setup");
      addSelect(train, { label: "Loss", value: state.loss, options: Object.entries(LOSSES).map(([value, label]) => ({ value, label })), onChange: v => { state.loss = v; if (v === "bce") state.targets = state.targets.map(t => Math.max(0, Math.min(1, t))); renderSettings(); renderAll(); } });
      addSelect(train, { label: "Optimizer", value: state.optimizer, options: Object.entries(OPTIMIZERS).map(([value, label]) => ({ value, label })), onChange: v => { state.optimizer = v; renderSettings(); renderStage(); } });
      addSlider(train, { label: "learning rate", min: 0.001, max: 1, step: 0.001, value: state.lr, format: v => Number(v).toFixed(3), onInput: v => { state.lr = v; renderStage(); } });
      addOptimizerSetting(train);
      addHint(train, state.loss === "bce" ? "Binary cross-entropy is most useful with sigmoid output neurons and targets between 0 and 1." : "One example is used per update. Step through the pass, then apply the gradients.");
      if (state.optimizer === "momentum") addHint(train, "Momentum accumulates a velocity: v ← βv + ∇L. The amber state bars show the first weight's stored velocity.");
      if (state.optimizer === "adam") addHint(train, "Adam tracks a smoothed gradient (m) and squared gradient (v), then corrects their startup bias. The state bars show m and √v.");
      if (state.selected) renderSelectedInfo();
    }
    function renderSelectedInfo() {
      const { layer, neuron: idx } = state.selected; const n = state.layers[layer]?.[idx]; if (!n) return;
      const section = addSection(panelEl, `Layer ${layer + 1} · neuron ${idx + 1}`);
      const editable = state.phase === "idle";
      addSelect(section, { label: "Activation", value: n.activation, options: Object.entries(activations).map(([value, a]) => ({ value, label: a.label })), onChange: v => { n.activation = v; renderStage(); } });
      n.weights.forEach((w, i) => addSlider(section, { label: `weight w${i + 1}`, min: -3, max: 3, step: 0.05, value: w, onInput: v => { n.weights[i] = v; renderStage(); } }));
      addSlider(section, { label: "bias", min: -3, max: 3, step: 0.05, value: n.bias, onInput: v => { n.bias = v; renderStage(); } });
      addReadout(section, `z = ${fmt(n.z)}\na = f(z) = ${fmt(n.a)}\nδ = ${fmt(n.delta)}\n∇b = ${fmt(n.gradB)}`);
      if (!editable) addHint(section, "Weights and activation can be edited between passes.");
    }

    function text(x, y, content, attrs = {}) { const el = svgEl("text", { x, y, fill: cssVar("--text-on-dark-dim", "#90a0b3"), "font-family": "IBM Plex Mono, monospace", "font-size": 11, ...attrs }); el.textContent = content; return el; }
    function drawLossHistory() {
      const x = 700, y = 18, w = 280, h = 132, hist = state.history.slice(-45), fg = cssVar("--text-on-dark", "#e8ecef"), grid = cssVar("--grid-line-strong", "#2c4d70");
      svg.appendChild(svgEl("rect", { x, y, width: w, height: h, rx: 4, fill: cssVar("--ink-panel", "#14202f"), stroke: grid }));
      svg.appendChild(text(x + 10, y + 17, "LOSS BY UPDATE", { fill: fg, "font-size": 10 }));
      svg.appendChild(text(x + w - 10, y + 17, hist.length ? fmt(hist.at(-1)) : "—", { fill: "#ffb238", "text-anchor": "end" }));
      const px = x + 12, py = y + 29, pw = w - 24, ph = h - 42;
      for (let i = 0; i < 3; i++) svg.appendChild(svgEl("line", { x1: px, y1: py + i * ph / 2, x2: px + pw, y2: py + i * ph / 2, stroke: grid, "stroke-dasharray": "2 4" }));
      if (hist.length) { const lo = Math.min(...hist), hi = Math.max(...hist), range = hi - lo || Math.max(hi, 1e-5); const pts = hist.map((v, i) => `${px + i / Math.max(1, hist.length - 1) * pw},${py + ph - (v - lo) / range * ph}`).join(" "); svg.appendChild(svgEl("polyline", { points: pts, fill: "none", stroke: "#ffb238", "stroke-width": 2 })); }
      else svg.appendChild(text(x + w / 2, y + 80, "apply an update to begin", { "text-anchor": "middle" }));
    }
    function drawLandscape() {
      const x = 700, y = 176, w = 280, h = 310, plot = { x: x + 42, y: y + 38, w: w - 58, h: h - 75 }, grid = 24;
      const n = state.layers[0][0], pA = n.weights[0] ?? n.bias, pB = n.weights[1] ?? n.bias;
      const isBiasB = n.weights.length < 2;
      const spanA = Math.max(0.8, Math.abs(pA) * 0.65), spanB = Math.max(0.8, Math.abs(pB) * 0.65);
      const values = Array.from({ length: grid + 1 }, () => Array(grid + 1).fill(0));
      for (let iy = 0; iy <= grid; iy++) for (let ix = 0; ix <= grid; ix++) {
        const a = pA - spanA + 2 * spanA * ix / grid, b = pB - spanB + 2 * spanB * iy / grid;
        const ovA = n.weights.length ? { layer: 0, neuron: 0, weight: 0, value: a } : { layer: 0, neuron: 0, bias: true, value: a };
        const ovB = isBiasB ? { layer: 0, neuron: 0, bias: true, value: b } : { layer: 0, neuron: 0, weight: 1, value: b };
        values[iy][ix] = objective(isBiasB ? [ovA, ovB] : [ovA, ovB]);
      }
      const finite = values.flat().filter(Number.isFinite), min = Math.min(...finite), max = Math.max(...finite), range = max - min || 1;
      svg.appendChild(svgEl("rect", { x, y, width: w, height: h, rx: 4, fill: cssVar("--ink-panel", "#14202f"), stroke: cssVar("--grid-line-strong", "#2c4d70") }));
      svg.appendChild(text(x + 10, y + 18, "2D LOSS LANDSCAPE", { fill: cssVar("--text-on-dark", "#e8ecef"), "font-size": 10 }));
      for (let iy = 0; iy < grid; iy++) for (let ix = 0; ix < grid; ix++) {
        const v = values[iy][ix], t = (v - min) / range;
        svg.appendChild(svgEl("rect", { x: plot.x + ix * plot.w / grid, y: plot.y + iy * plot.h / grid, width: plot.w / grid + 0.5, height: plot.h / grid + 0.5, fill: `hsl(${Math.round(205 - 180 * t)} 76% ${Math.round(24 + 36 * (1 - t))}%)`, stroke: "none" }));
      }
      const levels = [0.15, 0.3, 0.5, 0.7, 0.85].map(t => min + range * t);
      for (const level of levels) for (let iy = 0; iy < grid; iy++) for (let ix = 0; ix < grid; ix++) {
        const corners = [[ix, iy], [ix + 1, iy], [ix + 1, iy + 1], [ix, iy + 1]].map(([cx, cy]) => ({ x: cx, y: cy, v: values[cy][cx] }));
        const crossings = [];
        for (let e = 0; e < 4; e++) { const a = corners[e], b = corners[(e + 1) % 4]; if ((a.v < level) !== (b.v < level)) { const t = (level - a.v) / (b.v - a.v); crossings.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }); } }
        for (let k = 0; k + 1 < crossings.length; k += 2) svg.appendChild(svgEl("line", { x1: plot.x + crossings[k].x * plot.w / grid, y1: plot.y + crossings[k].y * plot.h / grid, x2: plot.x + crossings[k + 1].x * plot.w / grid, y2: plot.y + crossings[k + 1].y * plot.h / grid, stroke: "rgba(255,255,255,.72)", "stroke-width": 0.8 }));
      }
      const cx = plot.x + (pA - (pA - spanA)) / (2 * spanA) * plot.w, cy = plot.y + (pB - (pB - spanB)) / (2 * spanB) * plot.h;
      if (state.trajectory.length > 1) {
        const points = state.trajectory.map(([a, b]) => `${plot.x + Math.max(0, Math.min(1, (a - (pA - spanA)) / (2 * spanA))) * plot.w},${plot.y + Math.max(0, Math.min(1, (b - (pB - spanB)) / (2 * spanB))) * plot.h}`).join(" ");
        svg.appendChild(svgEl("polyline", { points, fill: "none", stroke: "#ffb238", "stroke-width": 1.5, "stroke-dasharray": "3 2", opacity: 0.9 }));
      }
      svg.appendChild(svgEl("circle", { cx, cy, r: 5, fill: "#fff", stroke: "#101820", "stroke-width": 2 }));
      svg.appendChild(text(x + 10, y + h - 15, n.weights.length < 2 ? "w₁ and bias · first neuron" : "w₁ and w₂ · first neuron", { "font-size": 9 }));
      svg.appendChild(text(plot.x + plot.w / 2, y + h - 2, `loss ${fmt(objective())}`, { fill: "#ffb238", "text-anchor": "middle", "font-size": 9 }));
    }
    function drawOptimizerState() {
      const x = 700, y = 505, w = 280, h = 112, grid = cssVar("--grid-line-strong", "#2c4d70"), first = state.layers[0][0];
      svg.appendChild(svgEl("rect", { x, y, width: w, height: h, rx: 4, fill: cssVar("--ink-panel", "#14202f"), stroke: grid }));
      svg.appendChild(text(x + 10, y + 18, `${OPTIMIZERS[state.optimizer].toUpperCase()} · FIRST WEIGHT STATE`, { fill: cssVar("--text-on-dark", "#e8ecef"), "font-size": 9 }));
      const items = state.optimizer === "momentum" ? [["velocity v", first.mw[0] ?? 0]] : state.optimizer === "adam" ? [["mean m", first.mw[0] ?? 0], ["√variance", Math.sqrt(first.vw[0] ?? 0)]] : state.optimizer === "rmsprop" ? [["√running variance", Math.sqrt(first.vw[0] ?? 0)]] : [["gradient", first.gradW[0] ?? 0]];
      items.forEach(([label, value], i) => {
        const yy = y + 43 + i * 31; svg.appendChild(text(x + 10, yy + 4, label, { "font-size": 9 }));
        const bx = x + 108, by = yy - 5, bw = 118, normalized = Math.max(-1, Math.min(1, value));
        svg.appendChild(svgEl("line", { x1: bx, y1: by + 8, x2: bx + bw, y2: by + 8, stroke: grid, "stroke-width": 2 }));
        const mid = bx + bw / 2, len = normalized * bw / 2;
        svg.appendChild(svgEl("rect", { x: Math.min(mid, mid + len), y: by + 3, width: Math.max(1, Math.abs(len)), height: 10, fill: "#ffb238" }));
        svg.appendChild(text(x + w - 10, yy + 4, fmt(value, 4), { "text-anchor": "end", fill: "#ffb238", "font-size": 9 }));
      });
      if (state.optimizer === "adam") svg.appendChild(text(x + 10, y + h - 8, `bias correction at update t = ${state.step + (state.phase === "backward-done" ? 1 : 0)}`, { "font-size": 8 }));
    }
    function renderStage() {
      svg = createStageSVG(stageEl, VIEWBOX); svg.appendChild(backgroundGrid(svg, 0, 0, 1000, 650));
      const xy = coords(), widths = layerWidths();
      for (let li = 0; li < widths.length - 1; li++) {
        const fromLayer = li === 0 ? null : state.layers[li - 1], toLayer = state.layers[li];
        toLayer.forEach((n, ni) => n.weights.forEach((weight, wi) => {
          const from = xy[li][wi], to = xy[li + 1][ni];
          svg.appendChild(svgEl("line", { class: "edge-line", x1: from.x, y1: from.y, x2: to.x, y2: to.y, opacity: 0.7 }));
          if (widths[li] * widths[li + 1] < 22) { const label = text(from.x + (to.x - from.x) * 0.36, from.y + (to.y - from.y) * 0.36 - 4, fmt(weight, 2), { fill: cssVar("--text-on-dark-dim", "#90a0b3"), "font-size": 8, "text-anchor": "middle" }); svg.appendChild(label); }
        }));
      }
      state.inputs.forEach((v, i) => node(0, i, `x${i + 1}`, fmt(v, 2), true));
      state.layers.forEach((layer, li) => layer.forEach((n, ni) => node(li + 1, ni, `L${li + 1}·${ni + 1}`, n.a === null ? "—" : fmt(n.a, 2), false, n.activation)));
      svg.appendChild(text(66, 48, "INPUTS", { "font-size": 10 }));
      state.layers.forEach((_, li) => svg.appendChild(text(xy[li + 1][0].x, 48, li === state.layers.length - 1 ? "OUTPUT" : `HIDDEN ${li + 1}`, { "text-anchor": "middle", "font-size": 10 })));
      svg.appendChild(text(665, 312, `${LOSSES[state.loss]} · ${fmt(objective())}`, { "text-anchor": "middle", fill: "#ffb238", "font-size": 10 }));
      drawLossHistory(); drawLandscape(); drawOptimizerState();
      function node(layer, ni, label, val, input, activation) {
        const { x, y } = xy[layer][ni], selected = state.selected?.layer === layer - 1 && state.selected?.neuron === ni;
        const g = svgEl("g", { cursor: input ? "default" : "pointer" });
        g.appendChild(svgEl("circle", { class: "node-shell" + (input ? " input-node" : "") + (selected ? " selected" : ""), cx: x, cy: y, r: Math.min(27, 35 - widths[layer] * 1.2) }));
        g.appendChild(text(x, y + 4, val, { fill: cssVar("--text-on-dark", "#e8ecef"), "text-anchor": "middle", "font-size": 11 }));
        g.appendChild(text(x, y - 38, label, { "text-anchor": "middle", "font-size": 9 }));
        if (activation) g.appendChild(text(x, y + 43, activation, { "text-anchor": "middle", "font-size": 8 }));
        if (!input) g.addEventListener("click", () => { state.selected = { layer: layer - 1, neuron: ni }; renderSettings(); renderStage(); });
        svg.appendChild(g);
      }
    }
    function renderStatusBar() {
      const pred = state.layers.at(-1).map(n => n.a).some(a => a === null) ? null : allOutputs();
      status = renderStatus(statusEl, { phase: { text: phaseText(), variant: state.phase.startsWith("backward") ? "backward" : state.phase.startsWith("forward") ? "forward" : "" }, stats: [{ id: "shape", label: "shape", value: layerWidths().join("–") }, { id: "epoch", label: "updates", value: state.epoch }, { id: "loss", label: "loss", value: pred ? fmt(lossValue(pred, state.targets, state.loss)) : "—" }] });
    }
    function phaseText() { return ({ idle: "ready — step forward", forward: "forward pass…", "forward-done": "forward complete — backpropagate", backward: "backpropagation…", "backward-done": "gradients ready — apply update" })[state.phase]; }
    function renderButtons() {
      ctl = renderControls(controlsEl, [
        { id: "fwd", label: "Step Forward →", variant: "primary", onClick: stepForward }, { id: "bwd", label: "← Backprop", variant: "backward", onClick: stepBackward },
        { id: "apply", label: `Apply ${OPTIMIZERS[state.optimizer]}`, onClick: applyGradients }, { id: "play", label: "Play", onClick: togglePlay }, { id: "reset", label: "Reinitialize", onClick: resetNetwork },
      ]); updateButtons();
    }
    function updateButtons() { ctl.setDisabled("fwd", state.busy || !["idle", "forward"].includes(state.phase)); ctl.setDisabled("bwd", state.busy || !["forward-done", "backward"].includes(state.phase)); ctl.setDisabled("apply", state.busy || state.phase !== "backward-done"); ctl.setLabel("apply", `Apply ${OPTIMIZERS[state.optimizer]}`); }
    function renderAll() { renderStage(); renderStatusBar(); updateButtons(); }
    function togglePlay() { if (playTimer) { clearInterval(playTimer); playTimer = null; ctl.setLabel("play", "Play"); return; } playTimer = setInterval(async () => { if (state.busy) return; if (["idle", "forward"].includes(state.phase)) await stepForwardReal(); else if (["forward-done", "backward"].includes(state.phase)) await stepBackwardReal(); else if (state.phase === "backward-done") applyGradients(); }, 450); ctl.setLabel("play", "Pause"); }

    renderSettings(); renderButtons(); renderAll();
    const unsubTheme = onThemeChange(renderAll);
    return { unmount() { if (playTimer) clearInterval(playTimer); unsubTheme(); } };
  },
};
