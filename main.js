import { loadModel } from "./model.js";
import {
  simulationStep,
  makeRng,
  getTemperature,
  maxwellBoltzmann,
  stationary,
  removeGlobalRotation3d,
  forceTemperature,
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
  speed: document.getElementById("speed"),
  speedVal: document.getElementById("speedVal"),
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

function renderMolecule(x) {
  viewer.removeAllModels();
  glModel = viewer.addModel();
  glModel.addAtoms(atomList(x));
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
  renderMolecule(state.x);
  updateReadouts();
}

function updateReadouts() {
  const T = getTemperature(state.p, init.masses, false, false);
  els.curTemp.textContent = `${T.toFixed(0)} K`;
}

function syncParams() {
  params.dtFs = parseFloat(els.dt.value);
  params.temperatureK = parseFloat(els.temp.value);
  els.dtVal.textContent = `${params.dtFs.toFixed(1)} fs`;
  els.tempVal.textContent = `${params.temperatureK.toFixed(0)} K`;
  els.speedVal.textContent = `${els.speed.value}×`;
}

async function loop() {
  if (!running) return;
  const stepsPerFrame = parseInt(els.speed.value, 10);
  const t0 = performance.now();
  for (let s = 0; s < stepsPerFrame; s++) {
    params.gauss = rngState.gauss;
    params.rand = rngState.rand;
    params.model = model;
    const next = await simulationStep(state, params);
    state = { x: next.x, p: next.p };
    stepIdx++;
  }
  const dtMs = performance.now() - t0;
  spsWindow.push({ n: stepsPerFrame, ms: dtMs });
  if (spsWindow.length > 20) spsWindow.shift();
  const totN = spsWindow.reduce((a, b) => a + b.n, 0);
  const totMs = spsWindow.reduce((a, b) => a + b.ms, 0);
  els.sps.textContent = `${(1000 * totN / totMs).toFixed(0)} steps/s`;

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
  els.speed.addEventListener("input", syncParams);
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
