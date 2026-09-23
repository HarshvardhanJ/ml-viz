import { algorithms } from "./algorithms/index.js";
import { initTheme, toggleTheme, currentTheme, onThemeChange } from "./core/theme.js";

const select = document.getElementById("algo-select");
const stageEl = document.getElementById("stage");
const panelEl = document.getElementById("panel");
const controlsEl = document.getElementById("controls-cluster");
const statusEl = document.getElementById("status-cluster");
const panelToggle = document.getElementById("panel-toggle");
const themeToggle = document.getElementById("theme-toggle");

let current = null;

function populateSelect() {
  for (const algo of algorithms) {
    const opt = document.createElement("option");
    opt.value = algo.id;
    opt.textContent = algo.name;
    select.appendChild(opt);
  }
}

function mount(id) {
  if (current && current.unmount) current.unmount();
  stageEl.innerHTML = "";
  panelEl.innerHTML = "";
  controlsEl.innerHTML = "";
  statusEl.innerHTML = "";
  panelEl.classList.remove("open");

  const algo = algorithms.find((a) => a.id === id) || algorithms[0];
  current = algo.mount({ stageEl, panelEl, controlsEl, statusEl });

  const hash = `#${algo.id}`;
  if (location.hash !== hash) history.replaceState(null, "", hash);
}

function initialAlgoId() {
  const fromHash = location.hash.replace("#", "");
  if (fromHash && algorithms.some((a) => a.id === fromHash)) return fromHash;
  return algorithms[0].id;
}

function syncThemeButton() {
  const isLight = currentTheme() === "light";
  themeToggle.textContent = isLight ? "☀️" : "🌙";
  themeToggle.setAttribute("aria-label", `Switch to ${isLight ? "dark" : "light"} theme`);
}

initTheme();
syncThemeButton();
onThemeChange(syncThemeButton);
themeToggle.addEventListener("click", toggleTheme);

populateSelect();
const initial = initialAlgoId();
select.value = initial;
mount(initial);

select.addEventListener("change", () => mount(select.value));

panelToggle.addEventListener("click", () => {
  const open = panelEl.classList.toggle("open");
  panelToggle.setAttribute("aria-expanded", String(open));
});

window.addEventListener("hashchange", () => {
  const id = location.hash.replace("#", "");
  if (id && id !== select.value && algorithms.some((a) => a.id === id)) {
    select.value = id;
    mount(id);
  }
});
