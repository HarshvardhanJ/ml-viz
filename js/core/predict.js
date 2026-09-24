// "Predict Before You Step" — an opt-in mode that, when on, intercepts an
// algorithm's primary step button: instead of running immediately, it shows
// a quick multiple-choice question about what's about to happen, and only
// runs the real (completely unmodified) step logic after the student answers.
//
// Each algorithm module supplies a small pure "preview" function that
// computes the same result its real step function would, one step early,
// without mutating any state — so turning Predict Mode off restores exactly
// the original behavior, untouched.

const STORAGE_KEY = "mlvizlab-predict-mode";

export function isPredictModeOn() {
  try { return localStorage.getItem(STORAGE_KEY) === "on"; } catch { return false; }
}

export function setPredictMode(on) {
  try { localStorage.setItem(STORAGE_KEY, on ? "on" : "off"); } catch { /* ignore */ }
  window.dispatchEvent(new CustomEvent("predictmodechange", { detail: { on } }));
}

export function togglePredictMode() { setPredictMode(!isPredictModeOn()); }

export function onPredictModeChange(callback) {
  window.addEventListener("predictmodechange", callback);
  return () => window.removeEventListener("predictmodechange", callback);
}

// Wraps a module's real step function so that, when Predict Mode is on and
// a preview is available, it asks first. `getPreview(state)` must be pure —
// it looks at state and returns { question, options, correctIndex } or null
// (null means "nothing meaningful to predict right now", e.g. mid-animation
// or nothing queued — in that case the real step just runs immediately).
export function withPrediction(hostEl, getPreview, realStepFn) {
  return function guardedStep(...args) {
    if (isPredictModeOn()) {
      const preview = getPreview();
      if (preview) {
        showPredictPrompt(hostEl, preview, () => realStepFn(...args));
        return;
      }
    }
    return realStepFn(...args);
  };
}

// Renders a floating question card over `hostEl` (expects hostEl to be
// position:relative, which every module's #stage already is). Disables
// interaction with the stage behind it, and calls onDone() once the student
// has picked an answer and seen the correct/incorrect flash.
export function showPredictPrompt(hostEl, { question, options, correctIndex }, onDone) {
  const overlay = document.createElement("div");
  overlay.className = "predict-overlay";
  overlay.innerHTML = `
    <div class="predict-card">
      <div class="predict-kicker">🔮 Predict before you run this step</div>
      <div class="predict-question"></div>
      <div class="predict-options"></div>
    </div>`;
  overlay.querySelector(".predict-question").textContent = question;

  const optsEl = overlay.querySelector(".predict-options");
  options.forEach((label, i) => {
    const btn = document.createElement("button");
    btn.className = "predict-option";
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", () => {
      const correct = i === correctIndex;
      [...optsEl.children].forEach((b, j) => {
        b.disabled = true;
        if (j === correctIndex) b.classList.add("predict-correct");
        else if (j === i) b.classList.add("predict-wrong");
      });
      const feedback = document.createElement("div");
      feedback.className = "predict-feedback";
      feedback.textContent = correct ? "✅ Correct — nice intuition!" : "Not quite — watch what actually happens.";
      overlay.querySelector(".predict-card").appendChild(feedback);
      setTimeout(() => { overlay.remove(); onDone(); }, correct ? 700 : 1150);
    });
    optsEl.appendChild(btn);
  });

  hostEl.appendChild(overlay);
}
