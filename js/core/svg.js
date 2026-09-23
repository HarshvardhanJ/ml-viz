// Small helpers for building and animating SVG scenes without a framework.
import { cssVar } from "./theme.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export function svgEl(tag, attrs = {}, children = []) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    el.setAttribute(k, v);
  }
  for (const child of children) el.appendChild(child);
  return el;
}

// Creates the root <svg> for a stage, with a subtle blueprint grid defined
// so algorithm modules can drop it in as a background rect if they want it.
export function createStageSVG(container, viewBox = "0 0 900 560") {
  container.innerHTML = "";
  const svg = svgEl("svg", {
    viewBox,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
  });

  const defs = svgEl("defs");
  const pattern = svgEl("pattern", {
    id: "fineGrid",
    width: 24,
    height: 24,
    patternUnits: "userSpaceOnUse",
  });
  pattern.appendChild(
    svgEl("path", {
      d: "M 24 0 L 0 0 0 24",
      fill: "none",
      stroke: cssVar("--grid-line", "#1e3450"),
      "stroke-width": 1,
    })
  );
  defs.appendChild(pattern);
  svg.appendChild(defs);

  container.appendChild(svg);
  return svg;
}

export function backgroundGrid(svg, x, y, w, h) {
  return svgEl("rect", { x, y, width: w, height: h, fill: "url(#fineGrid)" });
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// Animates a small labeled dot traveling from (x1,y1) to (x2,y2) along a
// straight line, carrying a numeric value. Resolves when the trip finishes.
export function pulseAlongLine(svg, { x1, y1, x2, y2, value, color, duration = 650, radius = 5 }) {
  return new Promise((resolve) => {
    const group = svgEl("g", { class: "pulse-dot" });
    const dot = svgEl("circle", { r: radius, fill: color, cx: x1, cy: y1 });
    const label = svgEl("text", {
      x: x1,
      y: y1 - 10,
      "text-anchor": "middle",
      class: "node-sublabel",
      fill: color,
    });
    label.textContent = typeof value === "number" ? value.toFixed(2) : value;
    group.appendChild(dot);
    group.appendChild(label);
    svg.appendChild(group);

    const start = performance.now();
    function frame(now) {
      const t = clamp((now - start) / duration, 0, 1);
      const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; // ease in-out
      const cx = lerp(x1, x2, eased);
      const cy = lerp(y1, y2, eased);
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", cy);
      label.setAttribute("x", cx);
      label.setAttribute("y", cy - 10);
      if (t < 1) {
        requestAnimationFrame(frame);
      } else {
        group.remove();
        resolve();
      }
    }
    requestAnimationFrame(frame);
  });
}

// Briefly flashes stroke class on an edge element to show it is "active".
export function flashEdge(edgeEl, className, duration = 650) {
  edgeEl.classList.add(className);
  setTimeout(() => edgeEl.classList.remove(className), duration);
}
