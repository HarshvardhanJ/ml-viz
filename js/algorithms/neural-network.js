import { svgEl, createStageSVG, backgroundGrid, pulseAlongLine } from "../core/svg.js";
import { renderControls, renderStatus } from "../core/controls.js";
import { resetPanel, beginPanel, addSection, addSlider, addSelect, addReadout, addButton, addHint } from "../core/panel.js";
import { activations, fmt, randRange } from "../core/math.js";
import { drawLossChart, pushHistory } from "../core/chart.js";

const VIEWBOX = "0 0 900 560";
const LAYOUT = {
  input: [{ x: 110, y: 170 }, { x: 110, y: 390 }],
  hidden: [{ x: 430, y: 170 }, { x: 430, y: 390 }],
  output: [{ x: 730, y: 280 }],
  loss: { x: 855, y: 280 },
};
const R = 32;

function freshNeuron(nInputs, activation) {
  return {
    weights: Array.from({ length: nInputs }, () => randRange(-1, 1)),
    bias: 0,
    activation,
    z: null,
    a: null,
    delta: null,
    gradW: null,
    gradB: null,
  };
}

function freshState() {
  return {
    inputs: [0.6, -0.4],
    hidden: [freshNeuron(2, "relu"), freshNeuron(2, "relu")],
    output: freshNeuron(2, "sigmoid"),
    target: 1.0,
    lr: 0.4,
    epoch: 0,
    phase: "idle", // idle -> forward -> forward-done -> backward -> backward-done -> (apply) -> idle
    forwardQueue: [],
    backwardQueue: [],
    lossHistory: [],
    busy: false,
  };
}

export default {
  id: "neural-network",
  name: "Neural Network (2-2-1)",
  blurb: "Step through a forward pass and backpropagation, neuron by neuron, editing weights, bias, and activation as you go.",
  status: "ready",

  mount({ stageEl, panelEl, controlsEl, statusEl }) {
    let state = freshState();
    let selected = null; // { type: 'input'|'hidden'|'output'|'loss', idx }
    let svg = createStageSVG(stageEl, VIEWBOX);
    let ctl, status;
    let playTimer = null;

    resetPanel(panelEl, "Click any neuron, input, or the Loss node to inspect and edit it.");

    // ---------- math ----------

    function srcValue(type, idx) {
      if (type === "input") return state.inputs[idx];
      if (type === "hidden") return state.hidden[idx].a ?? 0;
      return 0;
    }

    function edgesInto(type, idx) {
      if (type === "hidden") return [0, 1].map((i) => ({ from: "input", fromIdx: i, weightIdx: i }));
      if (type === "output") return [0, 1].map((i) => ({ from: "hidden", fromIdx: i, weightIdx: i }));
      return [];
    }

    async function computeNeuron(type, idx) {
      const neuron = type === "hidden" ? state.hidden[idx] : state.output;
      const edges = edgesInto(type, idx);
      const dest = coordOf(type, idx);

      // pulse every incoming value simultaneously
      await Promise.all(
        edges.map((e) => {
          const from = coordOf(e.from, e.fromIdx);
          return pulseAlongLine(svg, {
            x1: from.x, y1: from.y, x2: dest.x, y2: dest.y,
            value: srcValue(e.from, e.fromIdx),
            color: "#ffb238",
          });
        })
      );

      const z = edges.reduce((sum, e, i) => sum + neuron.weights[i] * srcValue(e.from, e.fromIdx), neuron.bias);
      const a = activations[neuron.activation].fn(z);
      neuron.z = z;
      neuron.a = a;

      if (type === "output") {
        await pulseAlongLine(svg, {
          x1: dest.x, y1: dest.y, x2: LAYOUT.loss.x, y2: LAYOUT.loss.y,
          value: a, color: "#ffb238",
        });
        pushHistory(state.lossHistory, 0.5 * (state.target - a) ** 2, 60);
      }
    }

    async function backpropNeuron(type, idx) {
      const dest = coordOf(type, idx);
      if (type === "output") {
        const neuron = state.output;
        // dL/da for L = 0.5*(target-a)^2  =>  (a - target)
        const dLda = neuron.a - state.target;
        neuron.delta = dLda * activations[neuron.activation].dfn(neuron.z);
        neuron.gradW = state.hidden.map((h) => neuron.delta * h.a);
        neuron.gradB = neuron.delta;
        await pulseAlongLine(svg, {
          x1: LAYOUT.loss.x, y1: LAYOUT.loss.y, x2: dest.x, y2: dest.y,
          value: neuron.delta, color: "#ff6b5b",
        });
        await Promise.all(
          state.hidden.map((_, i) => {
            const src = coordOf("hidden", i);
            return pulseAlongLine(svg, {
              x1: dest.x, y1: dest.y, x2: src.x, y2: src.y,
              value: neuron.gradW[i], color: "#ff6b5b",
            });
          })
        );
      } else {
        const neuron = state.hidden[idx];
        const out = state.output;
        const dLda = out.delta * out.weights[idx];
        neuron.delta = dLda * activations[neuron.activation].dfn(neuron.z);
        neuron.gradW = state.inputs.map((x) => neuron.delta * x);
        neuron.gradB = neuron.delta;
        await Promise.all(
          [0, 1].map((i) => {
            const src = coordOf("input", i);
            return pulseAlongLine(svg, {
              x1: dest.x, y1: dest.y, x2: src.x, y2: src.y,
              value: neuron.gradW[i], color: "#ff6b5b",
            });
          })
        );
      }
    }

    function applyGradients() {
      for (const h of state.hidden) {
        h.weights = h.weights.map((w, i) => w - state.lr * h.gradW[i]);
        h.bias -= state.lr * h.gradB;
        h.z = h.a = h.delta = h.gradW = h.gradB = null;
      }
      state.output.weights = state.output.weights.map((w, i) => w - state.lr * state.output.gradW[i]);
      state.output.bias -= state.lr * state.output.gradB;
      state.output.z = state.output.a = state.output.delta = state.output.gradW = state.output.gradB = null;
      state.epoch += 1;
      state.phase = "idle";
    }

    // ---------- coordinates ----------

    function coordOf(type, idx) {
      if (type === "input") return LAYOUT.input[idx];
      if (type === "hidden") return LAYOUT.hidden[idx];
      if (type === "output") return LAYOUT.output[idx];
      return LAYOUT.loss;
    }

    // ---------- actions (buttons) ----------

    async function stepForward() {
      if (state.busy) return;
      if (state.phase === "idle") {
        state.forwardQueue = [{ type: "hidden", idx: 0 }, { type: "hidden", idx: 1 }, { type: "output", idx: 0 }];
        state.phase = "forward";
      }
      if (state.phase !== "forward" || state.forwardQueue.length === 0) return;
      state.busy = true;
      renderAll();
      const next = state.forwardQueue.shift();
      await computeNeuron(next.type, next.idx);
      if (state.forwardQueue.length === 0) state.phase = "forward-done";
      state.busy = false;
      renderAll();
    }

    async function stepBackward() {
      if (state.busy) return;
      if (state.phase === "forward-done") {
        state.backwardQueue = [{ type: "output", idx: 0 }, { type: "hidden", idx: 1 }, { type: "hidden", idx: 0 }];
        state.phase = "backward";
      }
      if (state.phase !== "backward" || state.backwardQueue.length === 0) return;
      state.busy = true;
      renderAll();
      const next = state.backwardQueue.shift();
      await backpropNeuron(next.type, next.idx);
      if (state.backwardQueue.length === 0) state.phase = "backward-done";
      state.busy = false;
      renderAll();
    }

    function doApply() {
      if (state.phase !== "backward-done" || state.busy) return;
      applyGradients();
      renderAll();
    }

    function doRandomize() {
      stopPlay();
      state = freshState();
      selected = null;
      resetPanel(panelEl, "Click any neuron, input, or the Loss node to inspect and edit it.");
      renderAll();
    }

    function togglePlay() {
      if (playTimer) { stopPlay(); return; }
      playTimer = setInterval(async () => {
        if (state.busy) return;
        if (state.phase === "idle" || state.phase === "forward") await stepForward();
        else if (state.phase === "forward-done" || state.phase === "backward") await stepBackward();
        else if (state.phase === "backward-done") doApply();
      }, 750);
      ctl.setLabel("play", "Pause");
      renderStatusBar();
    }
    function stopPlay() {
      if (playTimer) clearInterval(playTimer);
      playTimer = null;
      if (ctl) ctl.setLabel("play", "Play");
    }

    // ---------- selection / panel ----------

    function selectNode(type, idx) {
      selected = { type, idx };
      renderPanel();
      renderStage(); // to show selection ring
    }

    function renderPanel() {
      if (!selected) return;
      const editable = state.phase === "idle";
      const { type, idx } = selected;

      if (type === "input") {
        beginPanel(panelEl, `Input x${idx}`);
        const s = addSection(panelEl);
        addSlider(s, {
          label: "value", min: -2, max: 2, step: 0.05, value: state.inputs[idx],
          onInput: (v) => { state.inputs[idx] = v; renderStage(); renderStatusBar(); },
        });
        if (!editable) addHint(s, "Values lock while a pass is mid-flight — finish or apply it to edit freely.");
        return;
      }

      if (type === "loss") {
        beginPanel(panelEl, "Loss");
        const s = addSection(panelEl, "Target");
        addSlider(s, {
          label: "target", min: -1.5, max: 1.5, step: 0.05, value: state.target,
          onInput: (v) => { state.target = v; renderStage(); renderStatusBar(); },
        });
        const r = addSection(panelEl, "Readout");
        const out = state.output;
        const loss = out.a === null ? null : 0.5 * (state.target - out.a) ** 2;
        addReadout(r, `output a  = ${fmt(out.a)}\ntarget    = ${fmt(state.target)}\nloss (½ err²) = ${loss === null ? "—" : fmt(loss)}`);
        return;
      }

      // hidden / output neuron
      const neuron = type === "hidden" ? state.hidden[idx] : state.output;
      const title = type === "hidden" ? `Hidden neuron h${idx}` : "Output neuron";
      beginPanel(panelEl, title);

      const act = addSection(panelEl, "Activation function");
      addSelect(act, {
        label: "f(z)",
        value: neuron.activation,
        options: Object.entries(activations).map(([k, v]) => ({ value: k, label: v.label })),
        onChange: (v) => { neuron.activation = v; renderStage(); },
      });
      if (!editable) addHint(act, "Locks mid-pass — apply or randomize to reset, then edit.");

      const w = addSection(panelEl, "Weights & bias");
      const srcLabel = type === "hidden" ? "x" : "h";
      neuron.weights.forEach((wt, i) => {
        addSlider(w, {
          label: `w(${srcLabel}${i} → ${type === "hidden" ? "h" + idx : "out"})`,
          min: -3, max: 3, step: 0.05, value: wt,
          onInput: (v) => { neuron.weights[i] = v; renderStage(); },
        });
      });
      addSlider(w, {
        label: "bias",
        min: -3, max: 3, step: 0.05, value: neuron.bias,
        onInput: (v) => { neuron.bias = v; renderStage(); },
      });
      if (!editable) addHint(w, "Locks mid-pass — apply gradients or randomize to unlock.");

      const r = addSection(panelEl, "Readout");
      let text = `z = ${fmt(neuron.z)}\na = f(z) = ${fmt(neuron.a)}`;
      if (neuron.delta !== null) {
        text += `\n\ndelta (∂L/∂z) = ${fmt(neuron.delta)}`;
        text += `\ngrad bias = ${fmt(neuron.gradB)}`;
        neuron.gradW.forEach((g, i) => { text += `\ngrad w${i} = ${fmt(g)}`; });
      }
      addReadout(r, text);
    }

    // ---------- render ----------

    function renderStage() {
      svg = createStageSVG(stageEl, VIEWBOX);
      svg.appendChild(backgroundGrid(svg, 0, 0, 900, 560));

      // Weight labels sit near the source end (t=0.3) rather than the exact
      // midpoint: two edges that cross (e.g. x0->h1 and x1->h0) meet right at
      // their midpoints, so a 50% label placement makes both labels collide.
      // Near the source, edges from the same node are still separated by
      // their distinct destinations, so labels land in different spots.
      const LABEL_T = 0.3;

      // edges: input -> hidden
      state.hidden.forEach((h, hIdx) => {
        h.weights.forEach((wt, i) => {
          const from = LAYOUT.input[i], to = LAYOUT.hidden[hIdx];
          const lx = from.x + (to.x - from.x) * LABEL_T;
          const ly = from.y + (to.y - from.y) * LABEL_T;
          svg.appendChild(svgEl("line", { class: "edge-line", x1: from.x, y1: from.y, x2: to.x, y2: to.y }));
          const lbl = svgEl("text", { class: "edge-weight-label", x: lx, y: ly - 6 });
          lbl.textContent = fmt(wt, 2);
          svg.appendChild(lbl);
        });
      });

      // edges: hidden -> output
      state.output.weights.forEach((wt, i) => {
        const from = LAYOUT.hidden[i], to = LAYOUT.output[0];
        const lx = from.x + (to.x - from.x) * LABEL_T;
        const ly = from.y + (to.y - from.y) * LABEL_T;
        svg.appendChild(svgEl("line", { class: "edge-line", x1: from.x, y1: from.y, x2: to.x, y2: to.y }));
        const lbl = svgEl("text", { class: "edge-weight-label", x: lx, y: ly - 6 });
        lbl.textContent = fmt(wt, 2);
        svg.appendChild(lbl);
      });

      // edge: output -> loss
      svg.appendChild(svgEl("line", {
        class: "edge-line", x1: LAYOUT.output[0].x, y1: LAYOUT.output[0].y,
        x2: LAYOUT.loss.x, y2: LAYOUT.loss.y,
      }));

      // input nodes
      state.inputs.forEach((val, i) => drawNode("input", i, `x${i}`, fmt(val, 2)));
      // hidden nodes
      state.hidden.forEach((h, i) => drawNode("hidden", i, `h${i}`, h.a === null ? "—" : fmt(h.a, 2), activations[h.activation].label));
      // output node
      drawNode("output", 0, "out", state.output.a === null ? "—" : fmt(state.output.a, 2), activations[state.output.activation].label);
      // loss node
      drawLossNode();

      // loss-over-time chart, top-right corner — clear of all edges/nodes
      drawLossChart(svg, { x: 655, y: 12, w: 225, h: 100, history: state.lossHistory, color: "#ffb238", title: "Loss over steps" });

      function drawNode(type, idx, label, valueText, subText) {
        const { x, y } = coordOf(type, idx);
        const isSel = selected && selected.type === type && selected.idx === idx;
        const g = svgEl("g");
        const circle = svgEl("circle", {
          class: "node-shell" + (type === "input" ? " input-node" : "") + (isSel ? " selected" : ""),
          cx: x, cy: y, r: R,
        });
        circle.addEventListener("click", () => selectNode(type, idx));
        g.appendChild(circle);
        const t1 = svgEl("text", { class: "node-label", x, y: y + 4 });
        t1.textContent = valueText;
        g.appendChild(t1);
        const t0 = svgEl("text", { class: "node-sublabel", x, y: y - R - 10 });
        t0.textContent = subText ? `${label} · ${subText}` : label;
        g.appendChild(t0);
        svg.appendChild(g);
      }

      function drawLossNode() {
        const { x, y } = LAYOUT.loss;
        const isSel = selected && selected.type === "loss";
        const size = 26;
        const points = `${x},${y - size} ${x + size},${y} ${x},${y + size} ${x - size},${y}`;
        const poly = svgEl("polygon", { class: "node-shell" + (isSel ? " selected" : ""), points });
        poly.addEventListener("click", () => selectNode("loss", 0));
        svg.appendChild(poly);
        const loss = state.output.a === null ? null : 0.5 * (state.target - state.output.a) ** 2;
        const t1 = svgEl("text", { class: "node-label", x, y: y + 4 });
        t1.textContent = loss === null ? "—" : fmt(loss, 3);
        svg.appendChild(t1);
        const t0 = svgEl("text", { class: "node-sublabel", x, y: y - size - 10 });
        t0.textContent = "Loss";
        svg.appendChild(t0);
      }
    }

    function renderStatusBar() {
      status = renderStatus(statusEl, {
        phase: phaseTag(),
        stats: [
          { id: "epoch", label: "epoch", value: state.epoch },
          { id: "out", label: "output", value: state.output.a === null ? "—" : fmt(state.output.a) },
          { id: "loss", label: "loss", value: state.output.a === null ? "—" : fmt(0.5 * (state.target - state.output.a) ** 2) },
        ],
      });
    }

    function phaseTag() {
      switch (state.phase) {
        case "idle": return { text: "ready — step forward", variant: "" };
        case "forward": return { text: "forward pass…", variant: "forward" };
        case "forward-done": return { text: "forward done — step backward", variant: "forward" };
        case "backward": return { text: "backprop…", variant: "backward" };
        case "backward-done": return { text: "gradients ready — apply", variant: "backward" };
        default: return { text: "", variant: "" };
      }
    }

    function renderButtons() {
      ctl = renderControls(controlsEl, [
        { id: "fwd", label: "Step Forward →", variant: "primary", onClick: stepForward },
        { id: "bwd", label: "← Step Backward", variant: "backward", onClick: stepBackward },
        { id: "apply", label: "Apply Gradients", onClick: doApply },
        { id: "play", label: "Play", onClick: togglePlay },
        { id: "reset", label: "Randomize", onClick: doRandomize },
      ]);
      updateButtonStates();
    }

    function updateButtonStates() {
      ctl.setDisabled("fwd", state.busy || !(state.phase === "idle" || state.phase === "forward"));
      ctl.setDisabled("bwd", state.busy || !(state.phase === "forward-done" || state.phase === "backward"));
      ctl.setDisabled("apply", state.busy || state.phase !== "backward-done");
    }

    function renderAll() {
      renderStage();
      renderStatusBar();
      updateButtonStates();
      if (selected) renderPanel();
    }

    renderButtons();
    renderAll();

    return {
      unmount() {
        stopPlay();
      },
    };
  },
};
