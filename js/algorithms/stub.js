// A placeholder module for algorithms not yet built out. Registering a stub
// keeps it visible in the dropdown (so the workshop plan doesn't have to
// change) and documents exactly what the real module should do, in the same
// mount({stageEl, panelEl, controlsEl, statusEl}) shape every other
// algorithm module uses — swap this file's export for a real one later
// without touching main.js or the registry.

export function makeStub({ id, name, blurb, plan }) {
  return {
    id,
    name: `${name} (coming soon)`,
    blurb,
    status: "coming-soon",
    mount({ stageEl, panelEl, controlsEl, statusEl }) {
      stageEl.innerHTML = `
        <div class="stub-wrap">
          <div class="stub-card">
            <h2>${name}</h2>
            <p>${blurb}</p>
            <p>${plan}</p>
            <span class="stub-status">not built yet — see js/algorithms/${id}.js</span>
          </div>
        </div>`;
      panelEl.innerHTML = `<div class="panel-empty"><strong>Inspector</strong>Nothing to inspect yet — this visualizer hasn't been built.</div>`;
      controlsEl.innerHTML = "";
      statusEl.innerHTML = "";
      return { unmount() {} };
    },
  };
}
