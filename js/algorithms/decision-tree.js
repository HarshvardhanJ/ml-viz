import { svgEl, createStageSVG, backgroundGrid } from "../core/svg.js";
import { renderControls, renderStatus } from "../core/controls.js";
import { resetPanel, beginPanel, addSection, addSlider, addSelect, addReadout, addHint } from "../core/panel.js";
import { fmt, randRange, randn } from "../core/math.js";
import { cssVar, onThemeChange } from "../core/theme.js";
import { withPrediction } from "../core/predict.js";

const VIEWBOX = "0 0 900 560";
const SCATTER = { left: 40, right: 420, top: 60, bottom: 500 };
const TREE = { left: 470, right: 870, top: 55, rowH: 92 };
const DOMAIN = [-6, 6];
const CLASS_NAMES = ["A", "B"];
const PALETTE = ["#ffb238", "#ff6b5b"];

let nodeIdCounter = 0;

// A quadrant / XOR-style dataset: no single axis-aligned split can separate
// it cleanly, so the tree needs more than one split — a good demonstration
// of why recursive partitioning is useful.
function generatePoints() {
  const quadrants = [
    { cx: 3, cy: 3, label: 0 },
    { cx: -3, cy: -3, label: 0 },
    { cx: 3, cy: -3, label: 1 },
    { cx: -3, cy: 3, label: 1 },
  ];
  const pts = [];
  for (const q of quadrants) {
    const n = 6 + Math.floor(Math.random() * 3);
    for (let i = 0; i < n; i++) {
      pts.push({ x: q.cx + randn(0, 0.85), y: q.cy + randn(0, 0.85), label: q.label });
    }
  }
  return pts;
}

function gini(indices, points) {
  if (indices.length === 0) return 0;
  const counts = {};
  indices.forEach((i) => { counts[points[i].label] = (counts[points[i].label] || 0) + 1; });
  let g = 1;
  for (const c in counts) { const p = counts[c] / indices.length; g -= p * p; }
  return g;
}

function majorityClass(indices, points) {
  const counts = {};
  indices.forEach((i) => { counts[points[i].label] = (counts[points[i].label] || 0) + 1; });
  let best = 0, bestCount = -1;
  for (const c in counts) { if (counts[c] > bestCount) { bestCount = counts[c]; best = Number(c); } }
  return best;
}

function classCounts(indices, points) {
  const counts = CLASS_NAMES.map(() => 0);
  indices.forEach((i) => { counts[points[i].label] += 1; });
  return counts;
}

function findBestSplit(indices, points, minLeafSize) {
  let best = null;
  for (const feature of ["x", "y"]) {
    const vals = [...new Set(indices.map((i) => points[i][feature]))].sort((a, b) => a - b);
    for (let k = 0; k < vals.length - 1; k++) {
      const threshold = (vals[k] + vals[k + 1]) / 2;
      const left = indices.filter((i) => points[i][feature] <= threshold);
      const right = indices.filter((i) => points[i][feature] > threshold);
      if (left.length < minLeafSize || right.length < minLeafSize) continue;
      const weighted = (left.length / indices.length) * gini(left, points) + (right.length / indices.length) * gini(right, points);
      if (!best || weighted < best.weighted) best = { feature, threshold, weighted, left, right };
    }
  }
  return best;
}

function makeNode(points, indices, depth, region) {
  return {
    id: nodeIdCounter++, indices, depth, region,
    isLeaf: false, leftId: null, rightId: null,
    splitFeature: null, splitThreshold: null, predictedClass: null,
    gini: gini(indices, points),
  };
}

function freshState(maxDepth = 3, minSamples = 3, points = null) {
  const pts = points || generatePoints();
  nodeIdCounter = 0;
  const rootIndices = pts.map((_, i) => i);
  const root = makeNode(pts, rootIndices, 0, { xmin: DOMAIN[0], xmax: DOMAIN[1], ymin: DOMAIN[0], ymax: DOMAIN[1] });
  return {
    points: pts,
    maxDepth, minSamples,
    nodes: { [root.id]: root },
    rootId: root.id,
    queue: [root.id],
    busy: false,
  };
}

export default {
  id: "decision-tree",
  name: "Decision Tree",
  blurb: "Step through recursive splits and watch the tree grow on the right while matching rectangular regions appear on the scatter plot on the left.",
  status: "ready",

  mount({ stageEl, panelEl, controlsEl, statusEl }) {
    let state = freshState();
    let selected = null; // { type: 'point'|'node'|'model', id }
    let svg;
    let ctl;
    let playTimer = null;

    resetPanel(panelEl, "Click a training point or a tree node to inspect it.");

    function mapX(x) { return SCATTER.left + ((x - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * (SCATTER.right - SCATTER.left); }
    function mapY(y) { return SCATTER.bottom - ((y - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * (SCATTER.bottom - SCATTER.top); }

    function rebuildTree() {
      state = freshState(state.maxDepth, state.minSamples, state.points);
    }

    function previewStepQuestion() {
      if (state.queue.length === 0) return null;
      const node = state.nodes[state.queue[0]];
      const tooDeep = node.depth >= state.maxDepth;
      const tooFew = node.indices.length < state.minSamples;
      const pure = node.gini === 0;
      const split = (!tooDeep && !tooFew && !pure) ? findBestSplit(node.indices, state.points, 1) : null;
      return {
        question: "Will the highlighted region split again, or become a leaf?",
        options: ["It will split again", "It becomes a leaf"],
        correctIndex: split ? 0 : 1,
      };
    }

    function doStepReal() {
      if (state.busy || state.queue.length === 0) return;
      const id = state.queue.shift();
      const node = state.nodes[id];
      const tooDeep = node.depth >= state.maxDepth;
      const tooFew = node.indices.length < state.minSamples;
      const pure = node.gini === 0;
      const split = (!tooDeep && !tooFew && !pure) ? findBestSplit(node.indices, state.points, 1) : null;

      if (!split) {
        node.isLeaf = true;
        node.predictedClass = majorityClass(node.indices, state.points);
      } else {
        node.splitFeature = split.feature;
        node.splitThreshold = split.threshold;
        const leftRegion = { ...node.region };
        const rightRegion = { ...node.region };
        if (split.feature === "x") { leftRegion.xmax = split.threshold; rightRegion.xmin = split.threshold; }
        else { leftRegion.ymax = split.threshold; rightRegion.ymin = split.threshold; }
        const leftNode = makeNode(state.points, split.left, node.depth + 1, leftRegion);
        const rightNode = makeNode(state.points, split.right, node.depth + 1, rightRegion);
        state.nodes[leftNode.id] = leftNode;
        state.nodes[rightNode.id] = rightNode;
        node.leftId = leftNode.id;
        node.rightId = rightNode.id;
        state.queue.push(leftNode.id, rightNode.id);
      }
      renderAll();
    }

    const doStep = withPrediction(stageEl, previewStepQuestion, doStepReal);

    function doRegenerateData() {
      stopPlay();
      state = freshState(state.maxDepth, state.minSamples);
      selected = null;
      resetPanel(panelEl, "Click a training point or a tree node to inspect it.");
      renderAll();
    }

    function togglePlay() {
      if (playTimer) { stopPlay(); return; }
      // Play always runs the real step directly, bypassing Predict Mode.
      playTimer = setInterval(() => {
        if (state.queue.length === 0) { stopPlay(); return; }
        doStepReal();
      }, 750);
      ctl.setLabel("play", "Pause");
    }
    function stopPlay() {
      if (playTimer) clearInterval(playTimer);
      playTimer = null;
      if (ctl) ctl.setLabel("play", "Play");
    }

    function selectPoint(idx) { selected = { type: "point", idx }; renderPanel(); renderStage(); }
    function selectNode(id) { selected = { type: "node", id }; renderPanel(); renderStage(); }
    function selectModel() { selected = { type: "model" }; renderPanel(); renderStage(); }

    function renderPanel() {
      if (!selected) return;
      if (selected.type === "point") {
        const p = state.points[selected.idx];
        beginPanel(panelEl, `Training point #${selected.idx}`);
        const s = addSection(panelEl, "Position & class");
        addSlider(s, { label: "x", min: DOMAIN[0], max: DOMAIN[1], step: 0.1, value: p.x, onInput: (v) => { p.x = v; rebuildTree(); renderAll(); } });
        addSlider(s, { label: "y", min: DOMAIN[0], max: DOMAIN[1], step: 0.1, value: p.y, onInput: (v) => { p.y = v; rebuildTree(); renderAll(); } });
        addSelect(s, {
          label: "class", value: String(p.label),
          options: CLASS_NAMES.map((name, i) => ({ value: String(i), label: name })),
          onChange: (v) => { p.label = Number(v); rebuildTree(); renderAll(); },
        });
        addHint(s, "Editing a point rebuilds the tree from scratch.");
        return;
      }
      if (selected.type === "node") {
        const node = state.nodes[selected.id];
        if (!node) { selected = null; resetPanel(panelEl); return; }
        beginPanel(panelEl, node.isLeaf ? "Leaf node" : node.leftId !== null ? "Split node" : "Pending node");
        const counts = classCounts(node.indices, state.points);
        const r = addSection(panelEl, "Readout");
        let text = `depth = ${node.depth}\nsamples = ${node.indices.length}\ngini = ${fmt(node.gini, 3)}\n`;
        text += CLASS_NAMES.map((name, i) => `class ${name} = ${counts[i]}`).join("\n");
        if (node.isLeaf) text += `\n\npredicted class = ${CLASS_NAMES[node.predictedClass]}`;
        else if (node.leftId !== null) text += `\n\nsplit: ${node.splitFeature} ≤ ${fmt(node.splitThreshold, 2)}`;
        else text += `\n\n(not yet expanded)`;
        addReadout(r, text);
        return;
      }
      beginPanel(panelEl, "Model");
      const s = addSection(panelEl, "Stopping rules");
      addSlider(s, {
        label: "max depth", min: 1, max: 5, step: 1, value: state.maxDepth, format: (v) => String(Math.round(v)),
        onInput: (v) => { state.maxDepth = Math.round(v); rebuildTree(); renderAll(); },
      });
      addSlider(s, {
        label: "min samples to split", min: 2, max: 8, step: 1, value: state.minSamples, format: (v) => String(Math.round(v)),
        onInput: (v) => { state.minSamples = Math.round(v); rebuildTree(); renderAll(); },
      });
      addHint(s, "Changing either rebuilds the tree from scratch, keeping the current data.");
      const r = addSection(panelEl, "Readout");
      const total = Object.keys(state.nodes).length;
      const leaves = Object.values(state.nodes).filter((n) => n.isLeaf).length;
      addReadout(r, `nodes so far = ${total}\nleaves so far = ${leaves}\npending = ${state.queue.length}`);
    }

    function layoutTree() {
      let slot = 0;
      function visit(id) {
        const node = state.nodes[id];
        if (node.leftId === null && node.rightId === null) { node._slot = slot++; return node._slot; }
        const lx = visit(node.leftId), rx = visit(node.rightId);
        node._slot = (lx + rx) / 2;
        return node._slot;
      }
      visit(state.rootId);
      return slot;
    }

    function nodeKind(node) {
      if (node.isLeaf) return "leaf";
      if (node.leftId !== null) return "internal";
      return "pending";
    }

    function renderStage() {
      svg = createStageSVG(stageEl, VIEWBOX);
      svg.appendChild(backgroundGrid(svg, 0, 0, 900, 560));
      const borderColor = cssVar("--grid-line-strong", "#2c4d70");
      const dimText = cssVar("--text-on-dark-dim", "#90a0b3");

      // ---- scatter half ----
      svg.appendChild(svgEl("rect", {
        x: SCATTER.left, y: SCATTER.top, width: SCATTER.right - SCATTER.left, height: SCATTER.bottom - SCATTER.top,
        fill: "none", stroke: borderColor,
      }));

      const nodesArr = Object.values(state.nodes);
      // filled regions for leaves
      for (const node of nodesArr) {
        if (!node.isLeaf) continue;
        const { xmin, xmax, ymin, ymax } = node.region;
        svg.appendChild(svgEl("rect", {
          x: mapX(xmin), y: mapY(ymax), width: mapX(xmax) - mapX(xmin), height: mapY(ymin) - mapY(ymax),
          fill: PALETTE[node.predictedClass], "fill-opacity": 0.16, stroke: "none",
        }));
      }
      // split lines for internal nodes, clipped to their own region
      for (const node of nodesArr) {
        if (node.leftId === null || node.isLeaf) continue;
        const { xmin, xmax, ymin, ymax } = node.region;
        if (node.splitFeature === "x") {
          const px = mapX(node.splitThreshold);
          svg.appendChild(svgEl("line", { x1: px, y1: mapY(ymax), x2: px, y2: mapY(ymin), stroke: dimText, "stroke-width": 1.5 }));
        } else {
          const py = mapY(node.splitThreshold);
          svg.appendChild(svgEl("line", { x1: mapX(xmin), y1: py, x2: mapX(xmax), y2: py, stroke: dimText, "stroke-width": 1.5 }));
        }
      }
      // highlight the region about to be expanded next
      if (state.queue.length > 0) {
        const next = state.nodes[state.queue[0]];
        const { xmin, xmax, ymin, ymax } = next.region;
        svg.appendChild(svgEl("rect", {
          x: mapX(xmin), y: mapY(ymax), width: mapX(xmax) - mapX(xmin), height: mapY(ymin) - mapY(ymax),
          fill: "none", stroke: "#ffb238", "stroke-width": 2, "stroke-dasharray": "6 4",
        }));
      }
      // points
      state.points.forEach((p, idx) => {
        const isSel = selected && selected.type === "point" && selected.idx === idx;
        const c = svgEl("circle", {
          cx: mapX(p.x), cy: mapY(p.y), r: isSel ? 8 : 6,
          fill: PALETTE[p.label], stroke: isSel ? "#fff" : "none", "stroke-width": 2, style: "cursor:pointer",
        });
        c.addEventListener("click", () => selectPoint(idx));
        svg.appendChild(c);
      });

      // ---- tree half ----
      const totalSlots = layoutTree();
      const slotW = totalSlots <= 1 ? 0 : (TREE.right - TREE.left) / (totalSlots - 1);
      const nodeX = (node) => (totalSlots <= 1 ? (TREE.left + TREE.right) / 2 : TREE.left + node._slot * slotW);
      const nodeY = (node) => TREE.top + node.depth * TREE.rowH;

      // connecting lines first (under the boxes)
      for (const node of nodesArr) {
        if (node.leftId === null) continue;
        const left = state.nodes[node.leftId], right = state.nodes[node.rightId];
        svg.appendChild(svgEl("line", { x1: nodeX(node), y1: nodeY(node) + 20, x2: nodeX(left), y2: nodeY(left) - 20, stroke: borderColor, "stroke-width": 1.5 }));
        svg.appendChild(svgEl("line", { x1: nodeX(node), y1: nodeY(node) + 20, x2: nodeX(right), y2: nodeY(right) - 20, stroke: borderColor, "stroke-width": 1.5 }));
      }

      const boxW = 108, boxH = 42;
      for (const node of nodesArr) {
        const kind = nodeKind(node);
        const isNext = state.queue.length > 0 && state.queue[0] === node.id;
        const isSel = selected && selected.type === "node" && selected.id === node.id;
        const x = nodeX(node), y = nodeY(node);
        const rect = svgEl("rect", {
          x: x - boxW / 2, y: y - boxH / 2, width: boxW, height: boxH, rx: 5,
          fill: kind === "leaf" ? PALETTE[node.predictedClass] : cssVar("--ink-panel", "#14202f"),
          "fill-opacity": kind === "leaf" ? 0.85 : 1,
          stroke: isSel ? "#fff" : isNext ? "#ffb238" : kind === "leaf" ? PALETTE[node.predictedClass] : borderColor,
          "stroke-width": isSel ? 2.5 : isNext ? 2.5 : 1.5,
          "stroke-dasharray": kind === "pending" ? "4 3" : "none",
          style: "cursor:pointer",
        });
        rect.addEventListener("click", () => selectNode(node.id));
        svg.appendChild(rect);

        const line1 = svgEl("text", { class: "node-label", x, y: y - 4, style: "font-size:10.5px" });
        const line2 = svgEl("text", { class: "node-sublabel", x, y: y + 12 });
        if (kind === "leaf") {
          line1.textContent = `→ class ${CLASS_NAMES[node.predictedClass]}`;
          line2.textContent = `n=${node.indices.length}`;
        } else if (kind === "internal") {
          line1.textContent = `${node.splitFeature} ≤ ${fmt(node.splitThreshold, 2)}`;
          line2.textContent = `gini=${fmt(node.gini, 2)} n=${node.indices.length}`;
        } else {
          line1.textContent = `n=${node.indices.length}`;
          line2.textContent = `gini=${fmt(node.gini, 2)}`;
        }
        svg.appendChild(line1);
        svg.appendChild(line2);
      }

      // model plate
      const isModelSel = selected && selected.type === "model";
      const plate = svgEl("rect", { class: "node-shell" + (isModelSel ? " selected" : ""), x: TREE.left, y: 10, width: 160, height: 32, rx: 4, style: "cursor:pointer" });
      plate.addEventListener("click", selectModel);
      svg.appendChild(plate);
      const t = svgEl("text", { class: "node-label", x: TREE.left + 80, y: 31, style: "font-size:11px" });
      t.textContent = `depth ≤ ${state.maxDepth}, min ${state.minSamples}`;
      svg.appendChild(t);
    }

    function renderStatusBar() {
      const total = Object.keys(state.nodes).length;
      const leaves = Object.values(state.nodes).filter((n) => n.isLeaf).length;
      renderStatus(statusEl, {
        phase: state.queue.length === 0
          ? { text: "tree complete", variant: "done" }
          : { text: "next: expand highlighted region", variant: "forward" },
        stats: [
          { id: "nodes", label: "nodes", value: total },
          { id: "leaves", label: "leaves", value: leaves },
          { id: "pending", label: "pending", value: state.queue.length },
        ],
      });
    }

    function renderButtons() {
      ctl = renderControls(controlsEl, [
        { id: "step", label: "Step: Expand Next Node", variant: "primary", onClick: doStep },
        { id: "play", label: "Play", onClick: togglePlay },
        { id: "reset", label: "New Dataset", onClick: doRegenerateData },
      ]);
      updateButtonStates();
    }
    function updateButtonStates() {
      ctl.setDisabled("step", state.busy || state.queue.length === 0);
      ctl.setDisabled("play", state.queue.length === 0);
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
