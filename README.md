# HFM Demo — Interactive Molecular Dynamics in the Browser

Simulates paracetamol in motion **entirely client-side** — no server GPU. A
mean-flow Diffusion-Transformer (the "Hamiltonian Flow Map" / HFM model) is
exported to ONNX and run with ONNX Runtime Web (WebGPU, with a WASM fallback).
The integration timestep and temperature can be changed live while it runs.

## Live demo

Hosted on GitHub Pages: **https://winfried-ripken.github.io/hfm-demo/**

Press **Play**, then drag the **timestep** (0.5–20 fs) and **temperature**
(0–1000 K) sliders while the simulation runs. Because the model takes the
timestep as a direct input, changing it mid-simulation is free.

## Run locally

It's a fully static site — just serve the folder:

```bash
python3 -m http.server 8000   # then open http://localhost:8000
```

## How it works

| Piece | Where it runs | File |
|---|---|---|
| HFM model inference `(t, x, p) → (v, f)` | ONNX Runtime Web | `public/hfm_paracetamol.onnx` |
| Langevin thermostat + filters (ZeroRot, RemoveDrift-FlashMD, RandomRotation) | hand-written JS | `sim.js` |
| 3D rendering | 3Dmol.js (CDN) | `index.html`, `main.js` |
| Start geometry, masses, atomic numbers, bonds | static JSON | `public/paracetamol_init.json` |

The model is run single-threaded (`ort.env.wasm.numThreads = 1`), so the page
needs no `COOP`/`COEP` headers and works on plain static hosting like GitHub
Pages.

## Tests

A self-contained stability check runs the full simulation loop under Node using
the same `.onnx` file the browser loads:

```bash
npm install
npm run test:e2e
```

It runs a 500-step NVT rollout and confirms it stays finite, the molecule stays
intact, and the temperature tracks the target.

## Deployment

Pushing to `main` triggers `.github/workflows/pages.yml`, which publishes the
repo root to GitHub Pages (Pages is auto-enabled on the first run).
