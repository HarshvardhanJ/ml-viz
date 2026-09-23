import { svgEl, createStageSVG, backgroundGrid, pulseAlongLine } from "../core/svg.js";
import { renderControls, renderStatus } from "../core/controls.js";
import { resetPanel, beginPanel, addSection, addSlider, addSelect, addReadout, addHint } from "../core/panel.js";
import { activations, fmt, randRange } from "../core/math.js";
import { cssVar, onThemeChange } from "../core/theme.js";

const VIEWBOX = "0 0 900 560";
const IN_N = 6, K_N = 3, OUT_N = IN_N - K_N + 1;
const CELL = 42, KCELL = 46, OCELL = 58;
const INPUT_ORIGIN = { x: 55, y: 78 };
const KERNEL_ORIGIN = { x: 55, y: INPUT_ORIGIN.y + IN_N * CELL + 46 };
const OUTPUT_ORIGIN = { x: 560, y: INPUT_ORIGIN.y + (IN_N * CELL - OUT_N * OCELL) / 2 };

const PRESETS = {
  identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  edge: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
  sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  blur: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
};

function lerp(a, b, t) { return a + (b - a) * t; }
function grayscale(v, max = 9) {
  const t = Math.max(0, Math.min(1, v / max));
  // dark navy -> light blue-white; a fixed data-intensity scale, independent of theme
  const c1 = [20, 32, 47], c2 = [219, 231, 242];
  const rgb = c1.map((c, i) => Math.round(lerp(c, c2[i], t)));
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

function makeGrid(n, fill) { return Array.from({ length: n }, () => Array.from({ length: n }, fill)); }

function freshState() {
  return {
    input: makeGrid(IN_N, () => Math.floor(randRange(0, 10))),
    kernel: PRESETS.edge.map((row) => [...row]),
    preset: "edge",
    activation: "linear",
    output: makeGrid(OUT_N, () => null),
    outputZ: makeGrid(OUT_N, () => null),
    nextIndex: 0,
    busy: false,
  };
}

export default {
  id: "cnn",
  name: "Convolution (CNN)",
  blurb: "Slide a small kernel across an editable pixel grid, one position at a time, and watch the output feature map fill in.",
  status: "ready",

  mount({ stageEl, panelEl, controlsEl, statusEl }) {
    let state = freshState();
    let selected = null; // { type: 'input'|'kernel'|'output'|'model', r, c }
    let svg;
    let ctl;
    let playTimer = null;

    resetPanel(panelEl, "Click a pixel, a kernel weight, or an output cell to inspect and edit it.");

    const total = OUT_N * OUT_N;

    function resetOutput() {
      state.output = makeGrid(OUT_N, () => null);
      state.outputZ = makeGrid(OUT_N, () => null);
      state.nextIndex = 0;
    }

    function inputCoord(r, c) { return { x: INPUT_ORIGIN.x + c * CELL + CELL / 2, y: INPUT_ORIGIN.y + r * CELL + CELL / 2 }; }
    function outputCoord(r, c) { return { x: OUTPUT_ORIGIN.x + c * OCELL + OCELL / 2, y: OUTPUT_ORIGIN.y + r * OCELL + OCELL / 2 }; }

    async function doStep() {
      if (state.busy || state.nextIndex >= total) return;
      state.busy = true;
      updateButtonStates();

      const row = Math.floor(state.nextIndex / OUT_N), col = state.nextIndex % OUT_N;
      const windowCenter = {
        x: INPUT_ORIGIN.x + (col + K_N / 2) * CELL,
        y: INPUT_ORIGIN.y + (row + K_N / 2) * CELL,
      };
      renderStage({ highlightWindow: { row, col } });

      let z = 0;
      for (let kr = 0; kr < K_N; kr++) {
        for (let kc = 0; kc < K_N; kc++) {
          z += state.input[row + kr][col + kc] * state.kernel[kr][kc];
        }
      }
      const dest = outputCoord(row, col);
      await pulseAlongLine(svg, { x1: windowCenter.x, y1: windowCenter.y, x2: dest.x, y2: dest.y, value: z, color: "#ffb238" });

      state.outputZ[row][col] = z;
      state.output[row][col] = activations[state.activation].fn(z);
      state.nextIndex += 1;
      state.busy = false;
      renderAll();
    }

    function doRegenerateInput() {
      stopPlay();
      const kernel = state.kernel, preset = state.preset, activation = state.activation;
      state = freshState();
      state.kernel = kernel; state.preset = preset; state.activation = activation;
      selected = null;
      resetPanel(panelEl, "Click a pixel, a kernel weight, or an output cell to inspect and edit it.");
      renderAll();
    }

    function applyPreset(name) {
      state.preset = name;
      state.kernel = PRESETS[name].map((row) => [...row]);
      resetOutput();
      renderAll();
      if (selected) renderPanel();
    }

    function togglePlay() {
      if (playTimer) { stopPlay(); return; }
      playTimer = setInterval(() => {
        if (state.busy) return;
        if (state.nextIndex >= total) { stopPlay(); return; }
        doStep();
      }, 700);
      ctl.setLabel("play", "Pause");
    }
    function stopPlay() {
      if (playTimer) clearInterval(playTimer);
      playTimer = null;
      if (ctl) ctl.setLabel("play", "Play");
    }

    function selectCell(type, r, c) { selected = { type, r, c }; renderPanel(); renderStage(); }
    function selectModel() { selected = { type: "model" }; renderPanel(); renderStage(); }

    function renderPanel() {
      if (!selected) return;
      if (selected.type === "input") {
        const { r, c } = selected;
        beginPanel(panelEl, `Pixel (${r}, ${c})`);
        const s = addSection(panelEl, "Value");
        addSlider(s, {
          label: "intensity", min: 0, max: 9, step: 1, value: state.input[r][c], format: (v) => String(Math.round(v)),
          onInput: (v) => { state.input[r][c] = Math.round(v); resetOutput(); renderAll(); },
        });
        addHint(s, "Editing a pixel clears the output — step through again to see the effect.");
        return;
      }
      if (selected.type === "kernel") {
        const { r, c } = selected;
        beginPanel(panelEl, `Kernel weight (${r}, ${c})`);
        const s = addSection(panelEl, "Value");
        addSlider(s, {
          label: "weight", min: -2, max: 2, step: 0.25, value: state.kernel[r][c],
          onInput: (v) => { state.kernel[r][c] = v; state.preset = "custom"; resetOutput(); renderAll(); },
        });
        addHint(s, "Editing a weight switches the preset to \"custom\" and clears the output.");
        return;
      }
      if (selected.type === "output") {
        const { r, c } = selected;
        beginPanel(panelEl, `Output cell (${r}, ${c})`);
        const rSec = addSection(panelEl, "Readout");
        const z = state.outputZ[r][c], a = state.output[r][c];
        addReadout(rSec, z === null ? "not yet computed" : `z (dot product) = ${fmt(z, 3)}\na = f(z) = ${fmt(a, 3)}`);
        return;
      }
      beginPanel(panelEl, "Model");
      const s = addSection(panelEl, "Kernel preset");
      addSelect(s, {
        label: "preset", value: state.preset,
        options: [
          { value: "identity", label: "Identity" },
          { value: "edge", label: "Edge detect" },
          { value: "sharpen", label: "Sharpen" },
          { value: "blur", label: "Box blur" },
          { value: "custom", label: "Custom" },
        ],
        onChange: (v) => { if (v !== "custom") applyPreset(v); },
      });
      const a = addSection(panelEl, "Activation");
      addSelect(a, {
        label: "f(z)", value: state.activation,
        options: Object.entries(activations).map(([k, v]) => ({ value: k, label: v.label })),
        onChange: (v) => { state.activation = v; resetOutput(); renderAll(); },
      });
      addHint(a, "Applied to each dot product before it's written into the feature map.");
    }

    function renderStage({ highlightWindow } = {}) {
      svg = createStageSVG(stageEl, VIEWBOX);
      svg.appendChild(backgroundGrid(svg, 0, 0, 900, 560));
      const borderColor = cssVar("--grid-line-strong", "#2c4d70");

      // labels
      label(INPUT_ORIGIN.x, INPUT_ORIGIN.y - 14, `Input (${IN_N}×${IN_N}) — click a pixel to edit`);
      label(KERNEL_ORIGIN.x, KERNEL_ORIGIN.y - 14, `Kernel (${K_N}×${K_N}) — click a weight to edit`);
      label(OUTPUT_ORIGIN.x, OUTPUT_ORIGIN.y - 14, `Output feature map (${OUT_N}×${OUT_N})`);

      // input grid
      for (let r = 0; r < IN_N; r++) {
        for (let c = 0; c < IN_N; c++) {
          const x = INPUT_ORIGIN.x + c * CELL, y = INPUT_ORIGIN.y + r * CELL;
          const isSel = selected && selected.type === "input" && selected.r === r && selected.c === c;
          const v = state.input[r][c];
          const rect = svgEl("rect", {
            x, y, width: CELL, height: CELL, fill: grayscale(v),
            stroke: isSel ? "#ffb238" : borderColor, "stroke-width": isSel ? 2.5 : 1, style: "cursor:pointer",
          });
          rect.addEventListener("click", () => selectCell("input", r, c));
          svg.appendChild(rect);
          const t = svgEl("text", { x: x + CELL / 2, y: y + CELL / 2 + 4, "text-anchor": "middle", "font-family": "IBM Plex Mono, monospace", "font-size": 11, fill: v > 4.5 ? "#0e1b2b" : "#e8ecef", "pointer-events": "none" });
          t.textContent = v;
          svg.appendChild(t);
        }
      }
      // current sliding window highlight
      if (highlightWindow) {
        svg.appendChild(svgEl("rect", {
          x: INPUT_ORIGIN.x + highlightWindow.col * CELL, y: INPUT_ORIGIN.y + highlightWindow.row * CELL,
          width: K_N * CELL, height: K_N * CELL, fill: "none", stroke: "#ffb238", "stroke-width": 3,
        }));
      } else if (state.nextIndex < total) {
        const row = Math.floor(state.nextIndex / OUT_N), col = state.nextIndex % OUT_N;
        svg.appendChild(svgEl("rect", {
          x: INPUT_ORIGIN.x + col * CELL, y: INPUT_ORIGIN.y + row * CELL,
          width: K_N * CELL, height: K_N * CELL, fill: "none", stroke: "#ffb238", "stroke-width": 2, "stroke-dasharray": "5 4",
        }));
      }

      // kernel grid
      for (let r = 0; r < K_N; r++) {
        for (let c = 0; c < K_N; c++) {
          const x = KERNEL_ORIGIN.x + c * KCELL, y = KERNEL_ORIGIN.y + r * KCELL;
          const isSel = selected && selected.type === "kernel" && selected.r === r && selected.c === c;
          const w = state.kernel[r][c];
          const tint = w > 0 ? `rgba(255,178,56,${Math.min(0.55, Math.abs(w) * 0.3)})` : w < 0 ? `rgba(255,107,91,${Math.min(0.55, Math.abs(w) * 0.3)})` : "transparent";
          const rect = svgEl("rect", {
            class: "node-shell", x, y, width: KCELL, height: KCELL,
            stroke: isSel ? "#ffb238" : undefined, "stroke-width": isSel ? 2.5 : undefined, style: "cursor:pointer",
          });
          rect.addEventListener("click", () => selectCell("kernel", r, c));
          svg.appendChild(rect);
          if (tint !== "transparent") {
            const overlay = svgEl("rect", { x, y, width: KCELL, height: KCELL, fill: tint, "pointer-events": "none" });
            svg.appendChild(overlay);
          }
          const t = svgEl("text", { class: "node-label", x: x + KCELL / 2, y: y + KCELL / 2 + 4, "pointer-events": "none" });
          t.textContent = fmt(w, 2);
          svg.appendChild(t);
        }
      }

      // output grid
      for (let r = 0; r < OUT_N; r++) {
        for (let c = 0; c < OUT_N; c++) {
          const x = OUTPUT_ORIGIN.x + c * OCELL, y = OUTPUT_ORIGIN.y + r * OCELL;
          const isSel = selected && selected.type === "output" && selected.r === r && selected.c === c;
          const val = state.output[r][c];
          const computed = val !== null;
          const rect = svgEl("rect", {
            x, y, width: OCELL, height: OCELL,
            fill: computed ? grayscale(val, Math.max(1, ...state.output.flat().filter((v) => v !== null).map(Math.abs))) : cssVar("--ink-panel", "#14202f"),
            stroke: isSel ? "#ffb238" : borderColor, "stroke-width": isSel ? 2.5 : 1,
            "stroke-dasharray": computed ? "none" : "4 3", style: "cursor:pointer",
          });
          rect.addEventListener("click", () => selectCell("output", r, c));
          svg.appendChild(rect);
          const t = svgEl("text", {
            x: x + OCELL / 2, y: y + OCELL / 2 + 4, "text-anchor": "middle", "font-family": "IBM Plex Mono, monospace", "font-size": 11,
            fill: computed ? (val > (Math.max(...state.output.flat().filter((v) => v !== null)) || 1) * 0.55 ? "#0e1b2b" : "#e8ecef") : cssVar("--text-on-dark-dim", "#90a0b3"),
            "pointer-events": "none",
          });
          t.textContent = computed ? fmt(val, 2) : "—";
          svg.appendChild(t);
        }
      }

      // model plate
      const isModelSel = selected && selected.type === "model";
      const plate = svgEl("rect", { class: "node-shell" + (isModelSel ? " selected" : ""), x: OUTPUT_ORIGIN.x, y: KERNEL_ORIGIN.y - 6, width: 190, height: 34, rx: 4, style: "cursor:pointer" });
      plate.addEventListener("click", selectModel);
      svg.appendChild(plate);
      const pt = svgEl("text", { class: "node-label", x: OUTPUT_ORIGIN.x + 95, y: KERNEL_ORIGIN.y + 16, style: "font-size:11px" });
      pt.textContent = `${state.preset} · ${activations[state.activation].label}`;
      svg.appendChild(pt);

      function label(x, y, text) {
        const t = svgEl("text", { x, y, class: "node-sublabel", "text-anchor": "start" });
        t.textContent = text;
        svg.appendChild(t);
      }
    }

    function renderStatusBar() {
      renderStatus(statusEl, {
        phase: state.nextIndex >= total ? { text: "convolution complete", variant: "done" } : { text: "next: slide window →", variant: "forward" },
        stats: [{ id: "pos", label: "position", value: `${Math.min(state.nextIndex, total)}/${total}` }],
      });
    }

    function renderButtons() {
      ctl = renderControls(controlsEl, [
        { id: "step", label: "Step: Slide Kernel", variant: "primary", onClick: doStep },
        { id: "play", label: "Play", onClick: togglePlay },
        { id: "reset", label: "New Grid", onClick: doRegenerateInput },
      ]);
      updateButtonStates();
    }
    function updateButtonStates() {
      ctl.setDisabled("step", state.busy || state.nextIndex >= total);
      ctl.setDisabled("play", state.nextIndex >= total);
    }

    function renderAll() {
      renderStage();
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
