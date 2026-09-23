// Activation functions and their derivatives, written in terms of the
// pre-activation input z (derivatives w.r.t. z), which is what backprop needs.
export const activations = {
  sigmoid: {
    label: "Sigmoid",
    fn: (z) => 1 / (1 + Math.exp(-z)),
    dfn: (z) => {
      const s = 1 / (1 + Math.exp(-z));
      return s * (1 - s);
    },
  },
  tanh: {
    label: "Tanh",
    fn: (z) => Math.tanh(z),
    dfn: (z) => 1 - Math.tanh(z) ** 2,
  },
  relu: {
    label: "ReLU",
    fn: (z) => Math.max(0, z),
    dfn: (z) => (z > 0 ? 1 : 0),
  },
  linear: {
    label: "Linear",
    fn: (z) => z,
    dfn: () => 1,
  },
};

export function fmt(n, digits = 3) {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

export function randRange(min, max) {
  return Math.random() * (max - min) + min;
}

export function randn(mean = 0, std = 1) {
  // Box-Muller transform
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}
