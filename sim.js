// Client-side molecular-dynamics step for the paracetamol HFM (mean-flow) model.
//
// This is a faithful JS port of the simulation math in hfm/ (single molecule,
// n_dim = 3). The only learned component is `model(dt, x, p) -> {v, f}`, which is
// run via ONNX Runtime in the browser. Everything here is plain linear algebra:
// the Langevin thermostat and the three integration filters (ZeroRot,
// RemoveDrift-FlashMD, RandomRotation), matching configs/sim_env/nvt_mf.yaml.
//
// Positions x and momenta p are represented as number[N][3]. Masses are
// number[N] (amu). Units follow ASE: energy eV, length Angstrom, time in ASE
// units (1 fs = FS_IN_ASE). kB in eV/K.

export const KB = 8.617333262145e-5; // eV/K
export const EPS_TEMP = 1e-8;

// ---------------------------------------------------------------------------
// small vector / matrix helpers (operate on number[N][3])
// ---------------------------------------------------------------------------
export function zeros(N) {
  return Array.from({ length: N }, () => [0, 0, 0]);
}
export function clone(a) {
  return a.map((r) => r.slice());
}
function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

// Solve a 3x3 linear system A x = b via Cramer's rule.
function solve3(A, b) {
  const det = (m) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det(A);
  const col = (m, c, v) => m.map((row, i) => row.map((x, j) => (j === c ? v[i] : x)));
  return [det(col(A, 0, b)) / D, det(col(A, 1, b)) / D, det(col(A, 2, b)) / D];
}

// Dihedral (torsion) angle in degrees between four points p1-p2-p3-p4.
// Port of hfm.simulation.utils.compute_dihedral.
export function dihedralDeg(p1, p2, p3, p4) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const norm = (v) => Math.hypot(v[0], v[1], v[2]);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const b1 = sub(p2, p1);
  const b2 = sub(p3, p2);
  const b3 = sub(p4, p3);
  let n1 = cross(b1, b2);
  let n2 = cross(b2, b3);
  const nn1 = norm(n1) || 1, nn2 = norm(n2) || 1, nb2 = norm(b2) || 1;
  n1 = n1.map((c) => c / nn1);
  n2 = n2.map((c) => c / nn2);
  const b2n = b2.map((c) => c / nb2);
  const x = dot(n1, n2);
  const y = dot(cross(n1, n2), b2n);
  return (Math.atan2(y, x) * 180) / Math.PI;
}

// ---------------------------------------------------------------------------
// thermodynamic quantities
// ---------------------------------------------------------------------------
export function kineticEnergy(p, m) {
  let e = 0;
  for (let i = 0; i < p.length; i++)
    for (let d = 0; d < 3; d++) e += (p[i][d] * p[i][d]) / m[i];
  return 0.5 * e;
}

export function getDof(N, zeroDrift, zeroRot) {
  let ndof = N * 3;
  if (N === 1) return ndof;
  if (zeroDrift) ndof -= 3;
  if (zeroRot) ndof -= 3;
  return ndof;
}

export function getTemperature(p, m, zeroDrift = false, zeroRot = false) {
  const ekin = kineticEnergy(p, m);
  const ndof = getDof(p.length, zeroDrift, zeroRot);
  return (2 * ekin) / (ndof * KB); // Kelvin
}

// Rescale momenta so kinetic energy matches temperature T (Kelvin).
export function forceTemperature(p, m, T_K, zeroDrift = false, zeroRot = false) {
  const targetEV = T_K * KB;
  const ekin = kineticEnergy(p, m);
  const ndof = getDof(p.length, zeroDrift, zeroRot);
  const currentEV = (2 * ekin) / ndof;
  let scale = T_K > EPS_TEMP ? Math.sqrt(targetEV / currentEV) : 0;
  return p.map((r) => r.map((v) => v * scale));
}

// Draw Maxwell-Boltzmann momenta. `gauss` is a function returning N(0,1) samples.
export function maxwellBoltzmann(m, T_K, gauss, forceTemp = true) {
  const tempEV = T_K * KB;
  const N = m.length;
  const p = zeros(N);
  for (let i = 0; i < N; i++)
    for (let d = 0; d < 3; d++) p[i][d] = gauss() * Math.sqrt(m[i] * tempEV);
  return forceTemp ? forceTemperature(p, m, T_K, false, false) : p;
}

// ---------------------------------------------------------------------------
// center-of-mass / rotation utilities
// ---------------------------------------------------------------------------
function comPosition(x, m) {
  const tot = m.reduce((a, b) => a + b, 0);
  const c = [0, 0, 0];
  for (let i = 0; i < x.length; i++)
    for (let d = 0; d < 3; d++) c[d] += x[i][d] * m[i];
  return c.map((v) => v / tot);
}
function comVelocity(p, m) {
  // matches _com_velocity: sum(p) / total_mass
  const tot = m.reduce((a, b) => a + b, 0);
  const c = [0, 0, 0];
  for (let i = 0; i < p.length; i++)
    for (let d = 0; d < 3; d++) c[d] += p[i][d];
  return c.map((v) => v / tot);
}
function coordMean(x) {
  const c = [0, 0, 0];
  for (let i = 0; i < x.length; i++)
    for (let d = 0; d < 3; d++) c[d] += x[i][d];
  return c.map((v) => v / x.length);
}

function inertiaTensor(r, m) {
  let I11 = 0, I22 = 0, I33 = 0, I12 = 0, I13 = 0, I23 = 0;
  for (let i = 0; i < r.length; i++) {
    const [x, y, z] = r[i];
    I11 += m[i] * (y * y + z * z);
    I22 += m[i] * (x * x + z * z);
    I33 += m[i] * (x * x + y * y);
    I12 -= m[i] * x * y;
    I13 -= m[i] * x * z;
    I23 -= m[i] * y * z;
  }
  return [
    [I11, I12, I13],
    [I12, I22, I23],
    [I13, I23, I33],
  ];
}

// Remove global angular momentum (toward targetL). Returns new momenta.
export function removeGlobalRotation3d(x, p, m, targetL = [0, 0, 0]) {
  const v = p.map((r, i) => r.map((c) => c / m[i]));
  const com = comPosition(x, m);
  const r = x.map((row) => row.map((c, d) => c - com[d]));
  const L = [0, 0, 0];
  for (let i = 0; i < r.length; i++) {
    const c = cross(r[i], [m[i] * v[i][0], m[i] * v[i][1], m[i] * v[i][2]]);
    L[0] += c[0]; L[1] += c[1]; L[2] += c[2];
  }
  const I = inertiaTensor(r, m);
  I[0][0] += 1e-8; I[1][1] += 1e-8; I[2][2] += 1e-8;
  const omega = solve3(I, [L[0] - targetL[0], L[1] - targetL[1], L[2] - targetL[2]]);
  const out = zeros(x.length);
  for (let i = 0; i < x.length; i++) {
    const w = cross(omega, r[i]);
    for (let d = 0; d < 3; d++) out[i][d] = (v[i][d] - w[d]) * m[i];
  }
  return out;
}

// Remove net linear momentum (drift). Returns new momenta.
export function stationary(p, m, forceTemp = false) {
  const tot = m.reduce((a, b) => a + b, 0);
  const totP = [0, 0, 0];
  for (let i = 0; i < p.length; i++)
    for (let d = 0; d < 3; d++) totP[d] += p[i][d];
  const v0 = totP.map((v) => v / tot);
  let out = p.map((r, i) => r.map((c, d) => c - v0[d] * m[i]));
  if (forceTemp) {
    const t0 = getTemperature(p, m);
    out = forceTemperature(out, m, t0);
  }
  return out;
}

// ---------------------------------------------------------------------------
// random SO(3) rotation (Shoemake's method) from a uniform RNG `rand` in [0,1)
// ---------------------------------------------------------------------------
export function randomRotationMatrix(rand) {
  const u1 = rand(), u2 = rand(), u3 = rand();
  const q0 = Math.sqrt(1 - u1) * Math.sin(2 * Math.PI * u2);
  const q1 = Math.sqrt(1 - u1) * Math.cos(2 * Math.PI * u2);
  const q2 = Math.sqrt(u1) * Math.sin(2 * Math.PI * u3);
  const q3 = Math.sqrt(u1) * Math.cos(2 * Math.PI * u3);
  // quaternion (q0=w, q1=x, q2=y, q3=z) -> rotation matrix
  const w = q0, xq = q1, yq = q2, zq = q3;
  return [
    [1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * w), 2 * (xq * zq + yq * w)],
    [2 * (xq * yq + zq * w), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * w)],
    [2 * (xq * zq - yq * w), 2 * (yq * zq + xq * w), 1 - 2 * (xq * xq + yq * yq)],
  ];
}
function matvec(R, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}
function transpose3(R) {
  return [
    [R[0][0], R[1][0], R[2][0]],
    [R[0][1], R[1][1], R[2][1]],
    [R[0][2], R[1][2], R[2][2]],
  ];
}

// ---------------------------------------------------------------------------
// Integration filters. Each returns the (possibly) modified {x, p} and stores
// per-step state needed by its out-call on the provided `aux` object.
// ---------------------------------------------------------------------------
export const ZeroRotFilter = {
  inCall(x, p, m) {
    return { x, p: removeGlobalRotation3d(x, p, m) };
  },
  outCall(x, p, m) {
    return { x, p: removeGlobalRotation3d(x, p, m) };
  },
};

export const RemoveDriftFlashMD = {
  inCall(x, p, m, aux) {
    aux.xComBefore = comPosition(x, m);
    aux.vComBefore = comVelocity(p, m);
    return { x, p };
  },
  outCall(x, p, m, dt, aux) {
    const xComNow = comPosition(x, m);
    const xn = x.map((row) =>
      row.map((c, d) => c - xComNow[d] + aux.xComBefore[d] + aux.vComBefore[d] * dt)
    );
    const vComNow = comVelocity(p, m);
    const pn = zeros(x.length);
    for (let i = 0; i < x.length; i++)
      for (let d = 0; d < 3; d++) {
        const v = p[i][d] / m[i] - vComNow[d] + aux.vComBefore[d];
        pn[i][d] = v * m[i];
      }
    return { x: xn, p: pn };
  },
};

export const RandomRotationFilter = {
  inCall(x, p, m, aux, rand) {
    const R = randomRotationMatrix(rand);
    aux.R = R;
    const xmean = coordMean(x);
    const xr = x.map((row) => {
      const shifted = [row[0] - xmean[0], row[1] - xmean[1], row[2] - xmean[2]];
      const rot = matvec(R, shifted);
      return [rot[0] + xmean[0], rot[1] + xmean[1], rot[2] + xmean[2]];
    });
    const pr = p.map((row) => matvec(R, row));
    return { x: xr, p: pr };
  },
  outCall(x, p, m, aux) {
    const Rinv = transpose3(aux.R);
    const xmean = coordMean(x);
    const xr = x.map((row) => {
      const shifted = [row[0] - xmean[0], row[1] - xmean[1], row[2] - xmean[2]];
      const rot = matvec(Rinv, shifted);
      return [rot[0] + xmean[0], rot[1] + xmean[1], rot[2] + xmean[2]];
    });
    const pr = p.map((row) => matvec(Rinv, row));
    return { x: xr, p: pr };
  },
};

// ---------------------------------------------------------------------------
// Langevin operator-splitting integrator (NVT) wrapping the HFM model step.
// ---------------------------------------------------------------------------
export function langevinHalfStep(p, m, dt, T_K, gamma, gauss) {
  const pRand = maxwellBoltzmann(m, T_K, gauss, true);
  const alpha = Math.exp((-gamma * dt) / 2);
  const sigma = Math.sqrt(1 - Math.exp(-gamma * dt));
  return p.map((r, i) => r.map((c, d) => alpha * c + sigma * pRand[i][d]));
}

// One full NVT step. `model(dt, x, p)` must return {v, f} (number[N][3]); it may
// be async (ONNX). Returns the updated {x, p, v, f}.
export async function simulationStep(state, params) {
  const { m } = params;
  const dt = params.dtFs * params.fsInAse;
  const gamma = 1 / (params.timeConstantFs * params.fsInAse);
  const T_K = params.temperatureK;
  const gauss = params.gauss;
  const rand = params.rand;

  let x = state.x;
  let p = state.p;

  // Langevin half-step
  p = langevinHalfStep(p, m, dt, T_K, gamma, gauss);

  // filters IN: ZeroRot, RemoveDrift, RandomRotation
  const aux = {};
  ({ x, p } = ZeroRotFilter.inCall(x, p, m));
  ({ x, p } = RemoveDriftFlashMD.inCall(x, p, m, aux));
  ({ x, p } = RandomRotationFilter.inCall(x, p, m, aux, rand));

  // HFM model integration step: x += dt*v ; p += dt*f
  const { v, f } = await model_call(params.model, dt, x, p);
  x = x.map((row, i) => row.map((c, d) => c + dt * v[i][d]));
  p = p.map((row, i) => row.map((c, d) => c + dt * f[i][d]));

  // filters OUT: RandomRotation, RemoveDrift, ZeroRot
  ({ x, p } = RandomRotationFilter.outCall(x, p, m, aux));
  ({ x, p } = RemoveDriftFlashMD.outCall(x, p, m, dt, aux));
  ({ x, p } = ZeroRotFilter.outCall(x, p, m));

  // second Langevin half-step
  p = langevinHalfStep(p, m, dt, T_K, gamma, gauss);

  return { x, p, v, f };
}

async function model_call(model, dt, x, p) {
  return await model(dt, x, p);
}

// ---------------------------------------------------------------------------
// RNG: seedable mulberry32 + Box-Muller gaussian
// ---------------------------------------------------------------------------
export function makeRng(seed = 42) {
  let a = seed >>> 0;
  const rand = () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const gauss = () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
  return { rand, gauss };
}
