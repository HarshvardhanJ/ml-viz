// Handles the dark/light toggle. CSS variables in base.css do most of the
// work automatically (anything styled via a CSS class just picks up the new
// values), but algorithm modules draw a lot of raw SVG with attribute colors
// for performance/simplicity, and SVG presentation attributes can't read
// CSS custom properties directly in every browser. cssVar() below reads the
// *resolved* value of a token at draw time, so a module just calls
// cssVar("--grid-line-strong", "#2c4d70") instead of hardcoding the hex, and
// it comes out theme-correct whenever that module re-renders.

const STORAGE_KEY = "mlvizlab-theme";

export function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

export function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem(STORAGE_KEY, theme); } catch { /* storage unavailable — theme just won't persist */ }
  window.dispatchEvent(new CustomEvent("themechange", { detail: { theme } }));
}

export function toggleTheme() {
  applyTheme(currentTheme() === "dark" ? "light" : "dark");
}

// Call once, on page load, before anything renders.
export function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(STORAGE_KEY); } catch { /* ignore */ }
  const preferLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  applyTheme(saved || (preferLight ? "light" : "dark"));
}

// Algorithm modules call this once in mount() and invoke the returned
// unsubscribe function in unmount(), so a theme toggle mid-session
// immediately redraws whatever's on screen.
export function onThemeChange(callback) {
  window.addEventListener("themechange", callback);
  return () => window.removeEventListener("themechange", callback);
}

// Resolved value of a CSS custom property defined on :root, for use as a
// raw SVG attribute color. Falls back gracefully if the variable is unset
// (e.g. during a smoke test with no real stylesheet loaded).
export function cssVar(name, fallback) {
  if (typeof document === "undefined" || typeof getComputedStyle !== "function") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
