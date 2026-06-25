// End-to-end check of the browser simulation loop, run under Node with
// onnxruntime-node (same .onnx file the browser loads). Verifies:
//   1. ONNX inference works from JS and the [N][3] marshalling is correct.
//   2. A long NVT rollout stays finite, bounded, and thermostatted near target T.
//
// Run:  node webapp/export/test_e2e.mjs
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import ort from "onnxruntime-node";
import {
  simulationStep,
  makeRng,
  getTemperature,
} from "../sim.js";

const here = dirname(fileURLToPath(import.meta.url));
const pub = join(here, "..", "public");
const init = JSON.parse(readFileSync(join(pub, "paracetamol_init.json"), "utf8"));
const N = init.n_atoms;

const session = await ort.InferenceSession.create(join(pub, "hfm_paracetamol.onnx"));

function flatten(a) {
  const o = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) for (let d = 0; d < 3; d++) o[i * 3 + d] = a[i][d];
  return o;
}
function unflatten(data) {
  const o = Array.from({ length: N }, () => [0, 0, 0]);
  for (let i = 0; i < N; i++) for (let d = 0; d < 3; d++) o[i][d] = data[i * 3 + d];
  return o;
}
const model = async (dt, x, p) => {
  const out = await session.run({
    t: new ort.Tensor("float32", new Float32Array([dt]), [1, 1]),
    x: new ort.Tensor("float32", flatten(x), [1, N, 3]),
    p: new ort.Tensor("float32", flatten(p), [1, N, 3]),
  });
  return { v: unflatten(out.mean_v.data), f: unflatten(out.mean_f.data) };
};

// ---- 1. single inference sanity (finite, right shapes) ----
const x0 = init.x0.map((r) => r.slice());
const dt0 = init.default_timestep_fs * init.fs_in_ase_units;
const { v, f } = await model(dt0, x0, x0.map((r) => r.map(() => 0)));
const finite = (arr) => arr.flat().every(Number.isFinite);
console.log(`single inference: v finite=${finite(v)} f finite=${finite(f)}  ` +
  `f[0]=[${f[0].map((c) => c.toFixed(3)).join(", ")}]`);

// ---- 2. long NVT rollout stability ----
const rng = makeRng(42);
const params = {
  m: init.masses,
  fsInAse: init.fs_in_ase_units,
  timeConstantFs: 100.0,
  dtFs: init.default_timestep_fs,
  temperatureK: init.default_temperature_K,
  gauss: rng.gauss,
  rand: rng.rand,
  model,
};
// initial momenta
import { maxwellBoltzmann, removeGlobalRotation3d, stationary, forceTemperature } from "../sim.js";
let p = maxwellBoltzmann(init.masses, params.temperatureK, rng.gauss, true);
p = removeGlobalRotation3d(x0, p, init.masses);
p = stationary(p, init.masses, false);
p = forceTemperature(p, init.masses, params.temperatureK, true, true);
let state = { x: x0, p };

const NSTEPS = 500;
const temps = [];
let maxCoord = 0;
let anyNaN = false;
for (let s = 0; s < NSTEPS; s++) {
  state = await simulationStep(state, params);
  if (s > 50) temps.push(getTemperature(state.p, init.masses, false, false));
  for (const row of state.x) for (const c of row) {
    if (!Number.isFinite(c)) anyNaN = true;
    maxCoord = Math.max(maxCoord, Math.abs(c));
  }
}
const mean = temps.reduce((a, b) => a + b, 0) / temps.length;
const std = Math.sqrt(temps.reduce((a, b) => a + (b - mean) ** 2, 0) / temps.length);

console.log(`rollout ${NSTEPS} steps @ ${params.dtFs} fs:`);
console.log(`  any NaN/Inf : ${anyNaN}`);
console.log(`  max |coord| : ${maxCoord.toFixed(2)} Angstrom  (start molecule ~6 A across)`);
console.log(`  temperature : ${mean.toFixed(0)} +/- ${std.toFixed(0)} K  (target ${params.temperatureK} K)`);

let ok = !anyNaN && finite(v) && finite(f) && maxCoord < 40 && mean > 250 && mean < 800;
console.log(ok ? "\nE2E PASSED" : "\nE2E FAILED");
process.exit(ok ? 0 : 1);
