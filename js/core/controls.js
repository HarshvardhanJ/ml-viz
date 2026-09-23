// Builds the footer's button cluster and status readouts for whichever
// algorithm module is currently mounted. Each algorithm module calls this
// once per mount and gets back handles to update button state / stats.

export function renderControls(container, buttons) {
  container.innerHTML = "";
  const els = {};
  for (const b of buttons) {
    const btn = document.createElement("button");
    btn.className = "ctl-btn" + (b.variant ? ` ${b.variant}` : "");
    btn.type = "button";
    btn.textContent = b.label;
    btn.id = `ctl-${b.id}`;
    btn.disabled = !!b.disabled;
    btn.addEventListener("click", b.onClick);
    container.appendChild(btn);
    els[b.id] = btn;
  }
  return {
    setDisabled(id, disabled) {
      if (els[id]) els[id].disabled = disabled;
    },
    setLabel(id, label) {
      if (els[id]) els[id].textContent = label;
    },
  };
}

export function renderStatus(container, { phase, stats } = {}) {
  container.innerHTML = "";
  const tag = document.createElement("span");
  tag.className = "phase-tag";
  tag.id = "status-phase";
  tag.textContent = phase?.text ?? "";
  container.appendChild(tag);

  const statEls = {};
  for (const s of stats || []) {
    const span = document.createElement("span");
    span.className = "stat";
    span.innerHTML = `${s.label}: <b id="stat-${s.id}">${s.value ?? "—"}</b>`;
    container.appendChild(span);
    statEls[s.id] = span.querySelector("b");
  }

  return {
    setPhase(text, variant) {
      tag.textContent = text;
      tag.className = "phase-tag" + (variant ? ` ${variant}` : "");
    },
    setStat(id, value) {
      if (statEls[id]) statEls[id].textContent = value;
    },
  };
}

// Clears both control containers — called by main.js before mounting a
// new algorithm so stale buttons/listeners never leak between modules.
export function clearControls(controlsContainer, statusContainer) {
  controlsContainer.innerHTML = "";
  statusContainer.innerHTML = "";
}
