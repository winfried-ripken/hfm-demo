import { loadModel } from "./model.js";
import { createRamachandran } from "./ramachandran.js";
import {
  simulationStep,
  makeRng,
  getTemperature,
  maxwellBoltzmann,
  stationary,
  removeGlobalRotation3d,
  forceTemperature,
  dihedralDeg,
} from "./sim.js";

const ELEMENTS = { 1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S" };

const els = {
  status: document.getElementById("status"),
  backend: document.getElementById("backend"),
  playBtn: document.getElementById("playBtn"),
  resetBtn: document.getElementById("resetBtn"),
  dt: document.getElementById("dt"),
  dtVal: document.getElementById("dtVal"),
  temp: document.getElementById("temp"),
  tempVal: document.getElementById("tempVal"),
  stepCount: document.getElementById("stepCount"),
  curTemp: document.getElementById("curTemp"),
  sps: document.getElementById("sps"),
};

let init = null; // init geometry json
let model = null; // async (dt,x,p)->{v,f}
let viewer = null;
let glModel = null;
let state = null; // {x, p}
let rngState = null;
let running = false;
let stepIdx = 0;
let params = null;
let rama = null;
let dihedrals = null; // { phi: [i,j,k,l], psi: [i,j,k,l] }

// compute current (phi, psi) in degrees from positions
function currentPhiPsi(x) {
  const d = (idx) => dihedralDeg(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]]);
  return { phi: d(dihedrals.phi), psi: d(dihedrals.psi) };
}

// --- rolling steps-per-second meter ---
let spsWindow = [];

function atomList(x) {
  return init.atomic_numbers.map((z, i) => {
    const bonds = [];
    const bondOrder = [];
    for (const [a, b] of init.bonds) {
      if (a === i) { bonds.push(b); bondOrder.push(1); }
      else if (b === i) { bonds.push(a); bondOrder.push(1); }
    }
    return {
      elem: ELEMENTS[z] || "C",
      x: x[i][0], y: x[i][1], z: x[i][2],
      serial: i, bonds, bondOrder,
    };
  });
}

const STYLE = {
  stick: { radius: 0.13 },
  sphere: { scale: 0.28 },
};

// Translate coordinates so the molecule's centroid sits at the origin, which is
// where the camera is aimed. This keeps the molecule centered in view even as
// its center of mass drifts during the simulation. Purely a render-time
// transform — the physics state in `state.x` is untouched.
function centerCoords(x) {
  const c = [0, 0, 0];
  for (let i = 0; i < x.length; i++)
    for (let d = 0; d < 3; d++) c[d] += x[i][d];
  for (let d = 0; d < 3; d++) c[d] /= x.length;
  return x.map((r) => [r[0] - c[0], r[1] - c[1], r[2] - c[2]]);
}

function renderMolecule(x) {
  viewer.removeAllModels();
  glModel = viewer.addModel();
  glModel.addAtoms(atomList(centerCoords(x)));
  viewer.setStyle({}, STYLE);
  viewer.render();
}

function resetState() {
  const x0 = init.x0.map((r) => r.slice());
  rngState = makeRng(42);
  // initial momenta: Maxwell-Boltzmann at T, then zero rotation + drift, then
  // re-force temperature on the remaining dof (mirrors simbench.py start setup).
  let p = maxwellBoltzmann(init.masses, params.temperatureK, rngState.gauss, true);
  p = removeGlobalRotation3d(x0, p, init.masses);
  p = stationary(p, init.masses, false);
  p = forceTemperature(p, init.masses, params.temperatureK, true, true);
  state = { x: x0, p };
  stepIdx = 0;
  spsWindow = [];
  els.stepCount.textContent = "0";
  rama.reset();
  const { phi, psi } = currentPhiPsi(state.x);
  rama.showCurrent(phi, psi); // mark the starting conformation
  renderMolecule(state.x);
  updateReadouts();
}

function updateReadouts() {
  const T = getTemperature(state.p, init.masses, false, false);
  els.curTemp.textContent = `${T.toFixed(0)} K`;
}

// Color stops matching the temperature slider's cool→hot gradient (index.html).
const TEMP_STOPS = [
  [0.0, [0x3b, 0x82, 0xf6]],
  [0.4, [0x8b, 0x5c, 0xf6]],
  [0.72, [0xf5, 0x9e, 0x0b]],
  [1.0, [0xef, 0x44, 0x44]],
];

// Map a temperature to its color on the gradient (same stops as the slider).
function tempColor(T_K) {
  const lo = parseFloat(els.temp.min), hi = parseFloat(els.temp.max);
  const t = Math.min(1, Math.max(0, (T_K - lo) / (hi - lo)));
  for (let i = 1; i < TEMP_STOPS.length; i++) {
    const [p0, c0] = TEMP_STOPS[i - 1];
    const [p1, c1] = TEMP_STOPS[i];
    if (t <= p1) {
      const f = (t - p0) / (p1 - p0);
      const c = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    }
  }
  return `rgb(${TEMP_STOPS[3][1].join(", ")})`;
}

function syncParams() {
  params.dtFs = parseFloat(els.dt.value);
  params.temperatureK = parseFloat(els.temp.value);
  els.dtVal.textContent = `${params.dtFs.toFixed(1)} fs`;
  els.tempVal.textContent = `${params.temperatureK.toFixed(0)} K`;
  els.tempVal.style.color = tempColor(params.temperatureK);
}

async function loop() {
  if (!running) return;
  const t0 = performance.now();
  params.gauss = rngState.gauss;
  params.rand = rngState.rand;
  params.model = model;
  const next = await simulationStep(state, params);
  state = { x: next.x, p: next.p };
  stepIdx++;

  const dtMs = performance.now() - t0;
  spsWindow.push({ n: 1, ms: dtMs });
  if (spsWindow.length > 30) spsWindow.shift();
  const totN = spsWindow.reduce((a, b) => a + b.n, 0);
  const totMs = spsWindow.reduce((a, b) => a + b.ms, 0);
  els.sps.textContent = `${(1000 * totN / totMs).toFixed(0)} steps/s`;

  // accumulate this state into the Ramachandran cloud + mark current position
  const { phi, psi } = currentPhiPsi(state.x);
  rama.addPoint(phi, psi);
  rama.showCurrent(phi, psi);

  els.stepCount.textContent = stepIdx.toString();
  renderMolecule(state.x);
  updateReadouts();
  requestAnimationFrame(loop);
}

function setRunning(on) {
  running = on;
  els.playBtn.textContent = on ? "⏸ Pause" : "▶ Play";
  if (on) requestAnimationFrame(loop);
}

async function boot() {
  els.status.textContent = "Loading geometry…";
  init = await (await fetch("./public/paracetamol_init.json")).json();

  params = {
    m: init.masses,
    fsInAse: init.fs_in_ase_units,
    timeConstantFs: 100.0,
    dtFs: init.default_timestep_fs,
    temperatureK: init.default_temperature_K,
  };

  // dihedral atom indices (from configs/data_module/md17_paracetamol.yaml)
  const di = init.dihedral_atom_indices || [[6, 7, 8, 17], [1, 3, 4, 5]];
  dihedrals = { phi: di[0], psi: di[1] };
  rama = createRamachandran(
    document.getElementById("ramaCloud"),
    document.getElementById("ramaDot")
  );

  // viewer
  viewer = window.$3Dmol.createViewer(document.getElementById("viewer"), {
    backgroundColor: "0x10141c",
  });

  els.dt.value = params.dtFs;
  els.temp.value = params.temperatureK;
  syncParams();

  els.status.textContent = "Loading model (≈13 MB)…";
  model = await loadModel("./public/hfm_paracetamol.onnx");
  els.backend.textContent = model.backend.toUpperCase();

  resetState();
  viewer.zoomTo();
  viewer.render();

  els.status.textContent = "Ready";

  // wire up controls
  els.dt.addEventListener("input", syncParams);
  els.temp.addEventListener("input", syncParams);
  els.playBtn.addEventListener("click", () => setRunning(!running));
  els.resetBtn.addEventListener("click", () => {
    const wasRunning = running;
    setRunning(false);
    syncParams();
    resetState();
    if (wasRunning) setRunning(true);
  });
}

boot().catch((e) => {
  console.error(e);
  els.status.textContent = "Error: " + e.message;
});
