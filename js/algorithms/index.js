import neuralNetwork from "./neural-network.js";
import linearRegression from "./linear-regression.js";
import kmeans from "./kmeans.js";
import { makeStub } from "./stub.js";

const knn = makeStub({
  id: "knn",
  name: "K-Nearest Neighbors",
  blurb: "Classify a new point by looking at its k closest labeled neighbors.",
  plan: "Planned interaction: drop a query point anywhere on the plane, drag a k slider, and watch the k-nearest training points light up along with the connecting lines and the resulting vote.",
});

const decisionTree = makeStub({
  id: "decision-tree",
  name: "Decision Tree",
  blurb: "Split a dataset one feature threshold at a time to separate classes.",
  plan: "Planned interaction: step through recursive splits on a 2D dataset, watching the tree diagram grow on one side and the matching rectangular decision regions appear on the scatter plot on the other.",
});

const cnn = makeStub({
  id: "cnn",
  name: "Convolution (CNN)",
  blurb: "Slide a small filter over an image grid to see how a convolution produces a feature map.",
  plan: "Planned interaction: an editable small pixel grid and an editable kernel, with a step button that slides the kernel one position at a time and fills in the output feature map cell by cell.",
});

// Order here is the order shown in the dropdown.
export const algorithms = [neuralNetwork, linearRegression, kmeans, knn, decisionTree, cnn];
