# ML/DL Visual Lab

An interactive, in-browser lab for teaching how numbers actually move through
machine learning and deep learning algorithms. Built for a first-year AI/ML
workshop: pick an algorithm from the dropdown, click on a neuron / point /
centroid to change its properties, and step through the algorithm one
computation at a time.

No build step, no server, no dependencies. Everything runs as plain HTML,
CSS, and vanilla JavaScript (ES modules) directly in the browser, so it's a
straightforward fit for GitHub Pages.

## What's included

| Algorithm | Status | What you can do |
|---|---|---|
| Neural Network (2-2-1) | ✅ built | Edit each neuron's weights, bias, and activation function. Step through the forward pass neuron-by-neuron, then step through backprop, then apply the gradient update. |
| Linear / Logistic Regression | ✅ built | Drag data points, edit `w`/`b`/learning rate, and step through gradient descent one update at a time. Toggle between linear and logistic mode. |
| K-Means Clustering | ✅ built | Edit point and centroid positions, choose k, and step through the assign / update phases until convergence. |
| K-Nearest Neighbors | ✅ built | Click the plot to drop a query point, choose k, and step through revealing its nearest neighbors closest-first until the vote decides its class. |
| Decision Tree | 🚧 scaffolded, not built | Registered in the dropdown with a "coming soon" screen — see `js/algorithms/decision-tree.js` via `stub.js`. |
| Convolution (CNN) | 🚧 scaffolded, not built | Same as above, `js/algorithms/cnn.js`. |

The three "coming soon" entries are intentionally left as stubs so the
workshop's algorithm list doesn't need to change later — see
[Adding a new algorithm](#adding-a-new-algorithm) to fill them in.

## Running it locally

Because the app uses ES module imports (`import ... from "./..."`), opening
`index.html` directly with a `file://` URL will be blocked by the browser.
Serve the folder over HTTP instead — any static server works:

```bash
# Python (already on most machines)
python3 -m http.server 8000

# or Node, if you have it
npx serve .
```

Then open `http://localhost:8000`.

## Deploying to GitHub Pages

1. Push this repo to GitHub.
2. In the repo, go to **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to "Deploy from a branch".
4. Choose the `main` branch and the `/ (root)` folder, then **Save**.
5. GitHub will publish the site at `https://<your-username>.github.io/<repo-name>/`
   within a minute or two — no workflow file needed, since this is a fully
   static site.

## Project structure

```
ml-viz-lab/
├── index.html              # App shell: header, stage, inspector panel, footer
├── css/
│   ├── base.css             # Design tokens, reset, responsive layout shell
│   └── components.css       # Header, dropdown, panel, buttons, SVG node/edge styles
├── js/
│   ├── main.js               # Populates the dropdown, mounts/unmounts algorithms
│   ├── core/
│   │   ├── svg.js             # SVG element + "pulse traveling along a wire" helpers
│   │   ├── math.js            # Activation functions + derivatives, formatting, RNG
│   │   ├── controls.js        # Footer step/play/reset button bar builder
│   │   └── panel.js           # Right-hand inspector panel builder (sliders, selects…)
│   └── algorithms/
│       ├── index.js           # ⭐ The registry — add new algorithms here
│       ├── neural-network.js
│       ├── linear-regression.js
│       ├── kmeans.js
│       └── stub.js            # "Coming soon" placeholder factory
└── assets/
    └── favicon.svg
```

## How an algorithm module works

Every entry in the dropdown is a plain object with this shape:

```js
export default {
  id: "my-algorithm",       // used in the URL hash and the registry
  name: "My Algorithm",     // shown in the dropdown
  blurb: "One line about what it teaches.",
  status: "ready",          // or "coming-soon"

  // Called once when the user selects this algorithm.
  // Build your entire UI into the four containers you're given.
  mount({ stageEl, panelEl, controlsEl, statusEl }) {
    // stageEl    — main visualization area (an SVG stage is typical)
    // panelEl    — right-hand inspector, filled in on node click
    // controlsEl — footer button cluster (step / play / reset)
    // statusEl   — footer readouts (epoch, loss, phase, ...)

    // ...build your visualization...

    return {
      unmount() {
        // stop timers/animations here — main.js calls this before
        // mounting whatever the user picks next
      },
    };
  },
};
```

`js/core/svg.js`, `js/core/math.js`, `js/core/controls.js`, and
`js/core/panel.js` exist so every algorithm module can reuse the same
building blocks (a traveling "pulse" dot for showing values move along an
edge, a slider/select/readout panel builder, a step/play/reset button bar)
instead of re-implementing them.

## Adding a new algorithm

1. Duplicate `js/algorithms/kmeans.js` (or whichever existing module is
   closest to what you're building) as a starting point.
2. Implement your algorithm's state, a `renderStage()` that draws it with
   SVG, and a `renderPanel()` that shows/edit properties for whatever the
   user clicked.
3. Register it in `js/algorithms/index.js`, replacing the matching
   `makeStub(...)` entry if there is one.
4. That's it — `main.js` and the dropdown pick it up automatically.

## Design notes

The visual language is a technical blueprint / circuit-schematic: a dark
grid background, amber pulses for values flowing forward, and coral pulses
for gradients flowing backward during backprop. The intent is for students
to build a visual habit — *amber moves right, coral moves left* — that maps
onto the forward pass / backward pass distinction in every algorithm that
has one.

## License

MIT — see [LICENSE](LICENSE). Built for classroom use; feel free to adapt it
for your own workshop.
