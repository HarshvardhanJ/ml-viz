import { svgEl } from "./svg.js";

// Draws a small, self-contained line chart (background plate + polyline) into
// an existing SVG, meant to sit in a corner of an algorithm's stage and show
// a loss/objective value trending down as the user steps through updates.
//
// history: array of numbers, oldest first. Call this again after every
// render with the latest history — it always redraws from scratch, so it
// stays in sync with whatever state the algorithm module is holding.
export function drawLossChart(svg, { x, y, w, h, history, color = "#ffb238", title = "Loss", maxPoints = 50 }) {
  const g = svgEl("g", { class: "loss-chart" });

  g.appendChild(svgEl("rect", {
    x, y, width: w, height: h, rx: 4,
    fill: "#14202f", stroke: "#2c4d70", "stroke-width": 1,
  }));

  const titleEl = svgEl("text", { x: x + 8, y: y + 15, fill: "#90a0b3", "font-family": "IBM Plex Mono, monospace", "font-size": 10 });
  titleEl.textContent = title;
  g.appendChild(titleEl);

  const trimmed = history.slice(-maxPoints);

  const latestEl = svgEl("text", { x: x + w - 8, y: y + 15, "text-anchor": "end", fill: color, "font-family": "IBM Plex Mono, monospace", "font-size": 10, "font-weight": "600" });
  latestEl.textContent = trimmed.length ? trimmed[trimmed.length - 1].toFixed(3) : "—";
  g.appendChild(latestEl);

  const px0 = x + 8, py0 = y + 24, pw = w - 16, ph = h - 32;

  // baseline
  g.appendChild(svgEl("line", { x1: px0, y1: py0 + ph, x2: px0 + pw, y2: py0 + ph, stroke: "#2c4d70", "stroke-width": 1 }));

  if (trimmed.length < 2) {
    const placeholder = svgEl("text", { x: px0 + pw / 2, y: py0 + ph / 2 + 4, "text-anchor": "middle", fill: "#5b6572", "font-family": "IBM Plex Mono, monospace", "font-size": 10 });
    placeholder.textContent = trimmed.length === 0 ? "step to begin" : "…";
    g.appendChild(placeholder);
    svg.appendChild(g);
    return;
  }

  const min = Math.min(...trimmed);
  const max = Math.max(...trimmed);
  const range = max - min || Math.max(Math.abs(max), 1e-6);

  const points = trimmed.map((v, i) => {
    const px = px0 + (trimmed.length === 1 ? 0 : (i / (trimmed.length - 1)) * pw);
    const py = py0 + ph - ((v - min) / range) * ph;
    return `${px},${py}`;
  });

  g.appendChild(svgEl("polyline", { points: points.join(" "), fill: "none", stroke: color, "stroke-width": 1.8 }));

  const [lastX, lastY] = points[points.length - 1].split(",");
  g.appendChild(svgEl("circle", { cx: lastX, cy: lastY, r: 2.6, fill: color }));

  svg.appendChild(g);
}

// Small helper every module can share: push a value, keep only the most
// recent `max` entries.
export function pushHistory(history, value, max = 200) {
  history.push(value);
  if (history.length > max) history.shift();
}
