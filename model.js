// ONNX Runtime Web wrapper for the paracetamol HFM mean-flow model.
// Exposes an async `model(dt, x, p) -> {v, f}` matching what sim.js expects.
//
// Inputs/outputs are number[N][3] on the JS side; we marshal to/from the
// flat float32 tensors the ONNX graph uses (shapes (1,1), (1,20,3)).

const N = 20;

function flatten(arr) {
  const out = new Float32Array(N * 3);
  for (let i = 0; i < N; i++)
    for (let d = 0; d < 3; d++) out[i * 3 + d] = arr[i][d];
  return out;
}
function unflatten(data) {
  const out = Array.from({ length: N }, () => [0, 0, 0]);
  for (let i = 0; i < N; i++)
    for (let d = 0; d < 3; d++) out[i][d] = data[i * 3 + d];
  return out;
}

export async function loadModel(onnxUrl, { onProgress } = {}) {
  // ort is provided globally by the onnxruntime-web script tag.
  const ort = window.ort;
  ort.env.wasm.numThreads = 1; // simple + avoids cross-origin-isolation requirement

  // Pick the fastest available backend, falling back gracefully.
  const providers = [];
  if (navigator.gpu) providers.push("webgpu");
  providers.push("wasm");

  let session = null;
  let backend = null;
  for (const ep of providers) {
    try {
      session = await ort.InferenceSession.create(onnxUrl, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      backend = ep;
      break;
    } catch (e) {
      console.warn(`Backend ${ep} failed:`, e.message);
    }
  }
  if (!session) throw new Error("Could not initialise any ONNX execution provider.");

  const model = async (dt, x, p) => {
    const feeds = {
      t: new ort.Tensor("float32", new Float32Array([dt]), [1, 1]),
      x: new ort.Tensor("float32", flatten(x), [1, N, 3]),
      p: new ort.Tensor("float32", flatten(p), [1, N, 3]),
    };
    const out = await session.run(feeds);
    return {
      v: unflatten(out.mean_v.data),
      f: unflatten(out.mean_f.data),
    };
  };
  model.backend = backend;
  return model;
}
