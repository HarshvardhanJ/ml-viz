// Builds the right-hand inspector panel contents. Algorithm modules call
// these to describe "when you click a node, show this" without hand-rolling
// DOM each time.

export function resetPanel(panel, emptyText = "Click a node on the stage to inspect and edit it here.") {
  panel.innerHTML = `<div class="panel-empty"><strong>Inspector</strong>${emptyText}</div>`;
}

export function beginPanel(panel, titleText) {
  panel.innerHTML = "";
  if (titleText) {
    const h = document.createElement("div");
    h.className = "panel-title";
    h.style.fontSize = "15px";
    h.style.marginBottom = "14px";
    h.textContent = titleText;
    panel.appendChild(h);
  }
  return panel;
}

export function addSection(panel, title) {
  const section = document.createElement("div");
  section.className = "panel-section";
  if (title) {
    const t = document.createElement("div");
    t.className = "panel-title";
    t.textContent = title;
    section.appendChild(t);
  }
  panel.appendChild(section);
  return section;
}

export function addSlider(section, { label, min, max, step = 0.01, value, format, onInput }) {
  const row = document.createElement("div");
  row.className = "panel-row";
  row.style.flexDirection = "column";
  row.style.alignItems = "stretch";

  const top = document.createElement("div");
  top.className = "panel-row";
  top.style.marginBottom = "4px";
  const lab = document.createElement("label");
  lab.textContent = label;
  const val = document.createElement("span");
  val.className = "val";
  const fmt = format || ((v) => Number(v).toFixed(2));
  val.textContent = fmt(value);
  top.appendChild(lab);
  top.appendChild(val);

  const input = document.createElement("input");
  input.type = "range";
  input.min = min;
  input.max = max;
  input.step = step;
  input.value = value;
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    val.textContent = fmt(v);
    onInput(v);
  });

  row.appendChild(top);
  row.appendChild(input);
  section.appendChild(row);
  return {
    setValue(v) {
      input.value = v;
      val.textContent = fmt(v);
    },
  };
}

export function addSelect(section, { label, options, value, onChange }) {
  const row = document.createElement("div");
  row.className = "panel-row";
  const lab = document.createElement("label");
  lab.textContent = label;
  const select = document.createElement("select");
  for (const opt of options) {
    const o = document.createElement("option");
    o.value = opt.value;
    o.textContent = opt.label;
    if (opt.value === value) o.selected = true;
    select.appendChild(o);
  }
  select.addEventListener("change", () => onChange(select.value));
  row.appendChild(lab);
  row.appendChild(select);
  section.appendChild(row);
  return select;
}

export function addReadout(section, text, id) {
  const pre = document.createElement("div");
  pre.className = "readout";
  if (id) pre.id = id;
  pre.innerHTML = text;
  section.appendChild(pre);
  return pre;
}

export function addButton(section, label, onClick) {
  const btn = document.createElement("button");
  btn.className = "btn-secondary";
  btn.type = "button";
  btn.textContent = label;
  btn.addEventListener("click", onClick);
  section.appendChild(btn);
  return btn;
}

export function addHint(section, text) {
  const p = document.createElement("div");
  p.className = "hint";
  p.textContent = text;
  section.appendChild(p);
  return p;
}
